//! Peer bootstrap for Orion control state. Existing nonempty shard data is never paired
//! with an unrelated catalog; corruption of such a node requires a complete native snapshot.
use super::lifecycle::{self, CONTROL_DIR, bad, io};
use crate::collection::Collection;
use crate::config::AutoShardPolicy;
use crate::operations::types::CollectionResult;
use crate::shards::replica_set::replica_set_state::ReplicaState;
use common::counter::hardware_accumulator::HwMeasurementAcc;
use common::fs::atomic_save;
use common::tar_ext::BuilderExt;
use common::types::DeferredBehavior;
use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::Write;
use std::path::Path;
use std::sync::Arc;

fn state_key(state: &lifecycle::ControlState) -> (u64, u64, u64, u64, u8) {
    let phase = match state.phase {
        lifecycle::Phase::Collecting => 0,
        lifecycle::Phase::BuildingUpper => 1,
        lifecycle::Phase::BuildingLayout => 2,
        lifecycle::Phase::Importing => 3,
        lifecycle::Phase::Indexing => 4,
        lifecycle::Phase::Finalizing => 5,
        lifecycle::Phase::Ready => 6,
        _ => 0,
    };
    (
        state.active.as_ref().map(|g| g.generation).unwrap_or(0),
        state
            .active
            .as_ref()
            .map(|g| g.membership_revision)
            .unwrap_or(0),
        state.pending_generation.unwrap_or(0),
        state
            .pending_write
            .as_ref()
            .map(|p| p.revision)
            .unwrap_or(0),
        phase,
    )
}

impl Collection {
    pub fn start_orion_recovery(self: &Arc<Self>) {
        use std::sync::atomic::Ordering;
        if !self.orion_expected
            || (self.orion_router.load().is_some()
                && !self.orion_refresh_needed.load(Ordering::Acquire))
        {
            return;
        }
        if self
            .orion_recovery_running
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return;
        }
        let collection = Arc::clone(self);
        tokio::spawn(async move {
            if let Err(error) = collection.recover_orion_metadata_from_peer().await {
                log::debug!("Orion metadata recovery pending: {error}");
            }
            collection
                .orion_recovery_running
                .store(false, Ordering::Release);
        });
    }
    pub async fn export_orion_control_snapshot(
        &self,
        generation: u64,
        checksum: &str,
    ) -> CollectionResult<tempfile::TempPath> {
        if lifecycle::read_state(&self.path)?.is_none() {
            return Err(bad("Orion control state is not installed"));
        }
        let _guard = self.orion_control_lock.lock().await;
        let state = lifecycle::read_state(&self.path)?
            .ok_or_else(|| bad("Orion control state is not installed"))?;
        let matches_active = state
            .active
            .as_ref()
            .is_some_and(|g| g.generation == generation && g.artifact_sha256 == checksum);
        let matches_pending = state.pending_generation == Some(generation)
            && state.pending_artifact_sha256.as_deref() == Some(checksum);
        if !matches_active && !matches_pending {
            let original = super::migration::read_artifact(&self.path, generation)?;
            original.verify_checksum(checksum).map_err(bad)?;
            if !state
                .active
                .as_ref()
                .is_some_and(|g| g.generation > generation)
            {
                return Err(bad("peer requested a different Orion generation/checksum"));
            }
        }
        let archive = tempfile::Builder::new()
            .prefix(".orion-peer-")
            .tempfile_in(&self.path)
            .map_err(io)?;
        let tar = BuilderExt::new_seekable_owned(File::create(archive.path()).map_err(io)?);
        let mut directories = vec![
            self.path.join(CONTROL_DIR),
            self.path.join(super::ORION_ROUTING_ARTIFACT_DIR),
        ];
        while let Some(dir) = directories.pop() {
            if !dir.is_dir() {
                continue;
            }
            for entry in fs::read_dir(dir).map_err(io)? {
                let entry = entry.map_err(io)?;
                if entry.file_type().map_err(io)?.is_symlink() {
                    return Err(io("symlink in Orion peer state"));
                }
                if entry.file_type().map_err(io)?.is_dir() {
                    directories.push(entry.path());
                } else {
                    let path = entry.path();
                    // A prior bootstrap's placement is replaced by the current consensus placement below.
                    if path == self.path.join(CONTROL_DIR).join("peer-placement.json") {
                        continue;
                    }
                    tar.append_file(&path, path.strip_prefix(&self.path).map_err(io)?)
                        .await
                        .map_err(io)?;
                }
            }
        }
        let placement: BTreeMap<u32, Vec<u64>> = self
            .shards_holder
            .read()
            .await
            .get_shards()
            .map(|(id, shard)| {
                let mut peers: Vec<_> = shard.peers().keys().copied().collect();
                peers.sort_unstable();
                (id, peers)
            })
            .collect();
        tar.append_data(
            serde_json::to_vec(&placement).map_err(io)?,
            &Path::new(CONTROL_DIR).join("peer-placement.json"),
        )
        .await
        .map_err(io)?;
        tar.finish().await.map_err(io)?;
        archive.as_file().sync_all().map_err(io)?;
        Ok(archive.into_temp_path())
    }

    pub async fn recover_orion_metadata_from_peer(&self) -> CollectionResult<()> {
        use api::grpc::qdrant::OrionStateRequest;
        use api::grpc::qdrant::collections_internal_client::CollectionsInternalClient;
        let policy = self
            .collection_config
            .read()
            .await
            .auto_shard_policy
            .clone();
        let Some(AutoShardPolicy::Orion {
            generation,
            artifact_sha256,
        }) = policy
        else {
            return Ok(());
        };
        let refreshing = self
            .orion_refresh_needed
            .load(std::sync::atomic::Ordering::Acquire);
        if self.orion_router.load().is_some() && !refreshing {
            return Ok(());
        }
        let existing = lifecycle::read_state(&self.path)?;
        if existing
            .as_ref()
            .is_some_and(|state| state.active.is_none())
        {
            // Collecting/Preparing/Indexing initial generations must not trigger an
            // eager load merely because staging has made artifact files visible.
            return Ok(());
        }
        if existing.is_some() && !refreshing {
            // A missing serving file can be reconstructed from the already-bound local generation.
            let artifact = super::migration::read_artifact(&self.path, generation)?;
            self.install_recovered_router(artifact).await?;
            return Ok(());
        }
        let holder = self.shards_holder.read().await;
        for (_, shard) in holder.get_shards().filter(|_| existing.is_none()) {
            let count = shard
                .count_local(
                    Arc::new(shard::count::CountRequestInternal {
                        filter: None,
                        exact: true,
                    }),
                    None,
                    HwMeasurementAcc::disposable(),
                    DeferredBehavior::Exclude,
                )
                .await?;
            if count.is_some_and(|c| c.count > 0) {
                return Err(io(
                    "nonempty Orion replica lost its catalog; restore a complete collection snapshot",
                ));
            }
        }
        drop(holder);
        let peers: Vec<_> = self
            .channel_service
            .id_to_address
            .read()
            .iter()
            .filter(|(id, _)| **id != self.this_peer_id)
            .map(|(id, uri)| (*id, uri.clone()))
            .collect();
        let mut candidates = Vec::new();
        for (peer_id, peer) in peers {
            let response = self
                .channel_service
                .channel_pool
                .with_channel(&peer, |channel| {
                    let request = api::grpc::qdrant::GetCollectionInfoRequest {
                        collection_name: self.id.clone(),
                    };
                    async move {
                        CollectionsInternalClient::new(channel)
                            .orion_status(request)
                            .await
                    }
                })
                .await;
            if let Ok(response) = response {
                let response = response.into_inner();
                if response.peer_id != peer_id {
                    continue;
                }
                let state: Option<lifecycle::ControlState> =
                    serde_json::from_slice(&response.state_json).map_err(io)?;
                if let Some(state) = state {
                    let key = state_key(&state);
                    candidates.push((key, peer));
                }
            }
        }
        candidates.sort_by(|a, b| b.0.cmp(&a.0));
        let own = existing.clone().map(|mut s| {
            s.pending_write = super::membership::pending_status(&self.path, self.this_peer_id)
                .ok()
                .flatten();
            state_key(&s)
        });
        if own.is_some() && candidates.first().is_none_or(|(key, _)| Some(*key) <= own) {
            self.orion_refresh_needed
                .store(false, std::sync::atomic::Ordering::Release);
            if self.orion_router.load().is_none() {
                self.install_recovered_router(super::migration::read_artifact(
                    &self.path, generation,
                )?)
                .await?;
            }
            let _guard = self.orion_control_lock.lock().await;
            if !super::membership::pending_path(&self.path).exists() {
                self.repair_orion_data().await?;
            }
            return Ok(());
        }
        let mut failure = None;
        for (_, peer) in candidates {
            let response = self
                .channel_service
                .channel_pool
                .with_channel(&peer, |channel| {
                    let request = OrionStateRequest {
                        collection_name: self.id.clone(),
                        generation,
                        artifact_sha256: artifact_sha256.clone(),
                    };
                    async move {
                        CollectionsInternalClient::new(channel)
                            .orion_snapshot(request)
                            .await
                    }
                })
                .await;
            let mut stream = match response {
                Ok(response) => response.into_inner(),
                Err(error) => {
                    failure = Some(error.to_string());
                    continue;
                }
            };
            let mut archive = tempfile::Builder::new()
                .prefix(".orion-receive-")
                .tempfile_in(&self.path)
                .map_err(io)?;
            while let Some(chunk) = stream.message().await.map_err(io)? {
                archive.write_all(&chunk.bytes).map_err(io)?;
            }
            archive.as_file().sync_all().map_err(io)?;
            let stage = tempfile::Builder::new()
                .prefix(".orion-unpack-")
                .tempdir_in(&self.path)
                .map_err(io)?;
            common::tar_unpack::tar_unpack_file(archive.path(), stage.path()).map_err(io)?;
            for entry in fs::read_dir(stage.path()).map_err(io)? {
                let name = entry.map_err(io)?.file_name();
                if name != CONTROL_DIR && name != super::ORION_ROUTING_ARTIFACT_DIR {
                    return Err(io("unexpected peer snapshot root"));
                }
            }
            let state = lifecycle::read_state(stage.path())?
                .ok_or_else(|| io("peer omitted Orion state"))?;
            let original = super::migration::read_artifact(stage.path(), generation)?;
            original.verify_checksum(&artifact_sha256).map_err(io)?;
            let selected = state
                .active
                .as_ref()
                .filter(|g| g.generation > generation)
                .map(|g| g.generation)
                .unwrap_or(generation);
            let artifact = super::migration::read_artifact(stage.path(), selected)?;
            if super::bundle::graph_sha256(&original).map_err(io)?
                != super::bundle::graph_sha256(&artifact).map_err(io)?
                || super::bundle::upper_identity_sha256(&original).map_err(io)?
                    != super::bundle::upper_identity_sha256(&artifact).map_err(io)?
            {
                return Err(io("recovered generation changed the production graph"));
            }
            if state
                .active
                .as_ref()
                .is_some_and(|g| g.generation == generation)
            {
                super::membership::Catalog::load(stage.path(), generation)?;
            }
            let placement: BTreeMap<u32, Vec<u64>> = serde_json::from_reader(
                File::open(stage.path().join(CONTROL_DIR).join("peer-placement.json"))
                    .map_err(io)?,
            )
            .map_err(io)?;
            let _guard = self.orion_control_lock.lock().await;
            let current = lifecycle::read_state(&self.path)?;
            if current
                .as_ref()
                .and_then(|s| s.active.as_ref())
                .map(|g| (g.generation, g.membership_revision))
                > state
                    .active
                    .as_ref()
                    .map(|g| (g.generation, g.membership_revision))
            {
                return Ok(());
            }
            if super::membership::pending_path(&self.path).exists() {
                return Err(io("finish local write recovery before replacing metadata"));
            }
            let control = self.path.join(CONTROL_DIR);
            self.orion_serving
                .store(false, std::sync::atomic::Ordering::Release);
            self.orion_data_ready
                .store(false, std::sync::atomic::Ordering::Release);
            if control.exists() {
                fs::rename(
                    &control,
                    self.path
                        .join(format!("orion-abandoned-{}", uuid::Uuid::new_v4())),
                )
                .map_err(io)?;
            }
            super::durable::publish_directory(
                &stage.path().join(CONTROL_DIR),
                &control,
                "state.json",
            )
            .map_err(io)?;
            // Keep all immutable generations required by pending rollback and snapshot recovery.
            let routers = stage.path().join(super::ORION_ROUTING_ARTIFACT_DIR);
            if routers.is_dir() {
                fs::create_dir_all(self.path.join(super::ORION_ROUTING_ARTIFACT_DIR))
                    .map_err(io)?;
                for entry in fs::read_dir(routers).map_err(io)? {
                    let entry = entry.map_err(io)?;
                    let bytes = fs::read(entry.path()).map_err(io)?;
                    atomic_save(
                        &self
                            .path
                            .join(super::ORION_ROUTING_ARTIFACT_DIR)
                            .join(entry.file_name()),
                        |out| out.write_all(&bytes),
                    )
                    .map_err(io)?;
                }
            }
            let count = placement
                .keys()
                .max()
                .copied()
                .map(|n| n + 1)
                .ok_or_else(|| io("empty peer placement"))?;
            {
                let mut config = self.collection_config.write().await;
                config.params.shard_number = std::num::NonZeroU32::new(count).unwrap();
                if selected != generation {
                    config.auto_shard_policy = Some(AutoShardPolicy::Orion {
                        generation: selected,
                        artifact_sha256: artifact.canonical_sha256().map_err(io)?,
                    });
                }
                config.save(&self.path).map_err(io)?;
            }
            for (id, peers) in placement {
                if self.shards_holder.read().await.get_shard(id).is_some() {
                    continue;
                }
                let shard = self
                    .create_replica_set(id, None, &peers, Some(ReplicaState::Active))
                    .await?;
                self.shards_holder
                    .write()
                    .await
                    .add_shard(id, shard, None)
                    .await?;
            }
            self.orion_managed
                .store(true, std::sync::atomic::Ordering::Release);
            self.orion_refresh_needed
                .store(false, std::sync::atomic::Ordering::Release);
            *self.orion_catalog.lock().await = None;
            self.orion_visibility
                .store(Arc::new(super::membership::load_visibility(&self.path)));
            self.install_recovered_router(artifact).await?;
            if let Some(pending) = state.pending_generation {
                if matches!(
                    state.phase,
                    lifecycle::Phase::Importing
                        | lifecycle::Phase::Indexing
                        | lifecycle::Phase::Finalizing
                ) {
                    if lifecycle::generation_path(&self.path, pending)
                        .join("migration.json")
                        .is_file()
                    {
                        self.replay_orion_migration_data(pending).await?;
                    } else {
                        self.import_orion_local(pending).await?;
                    }
                    let mut state = state;
                    state.phase = lifecycle::Phase::Indexing;
                    lifecycle::save_state(&self.path, &state)?;
                }
            } else if !super::membership::pending_path(&self.path).exists() {
                self.repair_orion_data().await?;
            }
            return Ok(());
        }
        Err(io(format!(
            "Orion peer control state is unavailable: {}",
            failure.unwrap_or_else(|| "no peer".into())
        )))
    }

    async fn install_recovered_router(
        &self,
        artifact: super::OrionRoutingArtifact,
    ) -> CollectionResult<()> {
        let path = super::routing_artifact_path(&self.path, artifact.generation);
        fs::create_dir_all(path.parent().unwrap()).map_err(io)?;
        let config = self.collection_config.read().await;
        if let Some(AutoShardPolicy::Orion {
            artifact_sha256, ..
        }) = &config.auto_shard_policy
        {
            artifact.verify_checksum(artifact_sha256).map_err(io)?;
        }
        Self::validate_orion_router_artifact(
            &self.path,
            &config,
            self.shards_holder.read().await.len(),
            &artifact,
        )
        .map_err(io)?;
        drop(config);
        // Stream canonical JSON directly to the atomic file; do not keep a 32 GB
        // serialization buffer and do not reload another full graph from that file.
        atomic_save(&path, |out| {
            serde_json::to_writer(out, &artifact).map_err(std::io::Error::other)
        })
        .map_err(io)?;
        let router = Arc::new(super::OrionRouter::new(artifact).map_err(io)?);
        self.orion_router.store(Some(router));
        self.orion_data_ready
            .store(false, std::sync::atomic::Ordering::Release);
        self.orion_serving
            .store(false, std::sync::atomic::Ordering::Release);
        Ok(())
    }
}
