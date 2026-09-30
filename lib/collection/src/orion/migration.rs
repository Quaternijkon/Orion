//! Pause-write generation migration over the same immutable production graph.
//! Source point IDs (including UUIDs), payloads, tombstones and current vector values survive.
use super::lifecycle::{self, ControlState, Phase, bad, generation_path, io};
use super::membership::{Catalog, Member};
use super::{OrionRouter, OrionRoutingArtifact, bundle, layout::LayoutOptions};
use crate::collection::Collection;
use crate::operations::CollectionUpdateOperations;
use crate::operations::consistency_params::{ReadConsistency, ReadConsistencyType};
use crate::operations::point_ops::{
    PointInsertOperationsInternal, PointOperations, PointStructPersisted,
};
use crate::operations::types::{CollectionResult, PointRequestInternal};
use crate::shards::replica_set::replica_set_state::ReplicaState;
use common::counter::hardware_accumulator::HwMeasurementAcc;
use common::fs::atomic_save;
use futures::future::join_all;
use segment::types::{ExtendedPointId, WithPayload, WithPayloadInterface, WithVector};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs::{self, File};
use std::io::Write;
use std::path::Path;
use std::sync::Arc;

#[derive(Clone, Serialize, Deserialize)]
pub struct Migration {
    pub format_version: u32,
    pub source_generation: u64,
    pub source_revision: u64,
    pub source_artifact_sha256: String,
    pub options: LayoutOptions,
    pub sealed: bool,
    pub batches: u64,
    pub artifact_sha256: String,
    pub peers: Vec<u64>,
    pub batch_sha256: Vec<String>,
}

fn save(path: &Path, value: &impl Serialize) -> CollectionResult<()> {
    let bytes = serde_json::to_vec(&serde_json::to_value(value).map_err(io)?).map_err(io)?;
    atomic_save(path, |out| out.write_all(&bytes)).map_err(io)
}
fn metadata(collection: &Path, generation: u64) -> CollectionResult<Migration> {
    serde_json::from_reader(
        File::open(generation_path(collection, generation).join("migration.json")).map_err(io)?,
    )
    .map_err(io)
}
pub(crate) fn options(collection: &Path, generation: u64) -> CollectionResult<LayoutOptions> {
    let dir = generation_path(collection, generation);
    if dir.join("migration.json").is_file() {
        Ok(metadata(collection, generation)?.options)
    } else {
        Ok(lifecycle::read_bundle(&dir)?.0.options)
    }
}
pub(crate) fn read_artifact(
    collection: &Path,
    generation: u64,
) -> CollectionResult<OrionRoutingArtifact> {
    let dir = generation_path(collection, generation);
    if dir.join("migration.json").is_file() {
        let meta = metadata(collection, generation)?;
        let artifact =
            OrionRoutingArtifact::read_json(&dir.join("artifact.json"), None).map_err(io)?;
        if artifact.canonical_sha256().map_err(io)? != meta.artifact_sha256 {
            if meta.sealed {
                return Err(io("sealed migration artifact checksum mismatch"));
            }
            let source = read_artifact(collection, meta.source_generation)?;
            let mut unsealed = artifact.clone();
            unsealed.layout_sha256 = "0".repeat(64);
            unsealed.logical_point_count = source.logical_point_count;
            unsealed.physical_point_count = source.physical_point_count;
            if unsealed.canonical_sha256().map_err(io)? != meta.artifact_sha256 {
                return Err(io("migration artifact identity changed"));
            }
        }
        Ok(artifact)
    } else {
        Ok(lifecycle::read_bundle(&dir)?.2)
    }
}

pub(crate) struct Assigner {
    router: OrionRouter,
    labels: HashMap<ExtendedPointId, usize>,
    owners: Vec<u32>,
    anchors: HashMap<ExtendedPointId, u32>,
    vector_name: String,
    shard_count: u32,
}
impl Assigner {
    pub(crate) fn new(artifact: OrionRoutingArtifact, efs: usize) -> CollectionResult<Self> {
        Self::from_artifact(&artifact, efs)
    }
    pub(crate) fn from_artifact(
        artifact: &OrionRoutingArtifact,
        efs: usize,
    ) -> CollectionResult<Self> {
        let labels = artifact
            .upper_nodes
            .iter()
            .enumerate()
            .map(|(i, n)| (n.label, i))
            .collect();
        let owners = artifact.upper_nodes.iter().map(|n| n.owner_shard).collect();
        let anchors = artifact
            .upper_nodes
            .iter()
            .map(|n| (n.label, n.owner_shard))
            .collect();
        let vector_name = artifact.vector_schema.vector_name.clone();
        let shard_count = artifact.shard_count;
        Ok(Self {
            router: OrionRouter::from_artifact_with_ef(artifact, efs).map_err(bad)?,
            labels,
            owners,
            anchors,
            vector_name,
            shard_count,
        })
    }
    pub(crate) fn assign(
        &self,
        point: &PointStructPersisted,
        old: &Member,
    ) -> CollectionResult<Member> {
        self.assign_with_scratch(point, old, &mut self.new_scratch())
    }
    pub(crate) fn new_scratch(&self) -> super::router::OrionRouteScratch {
        self.router.new_route_scratch()
    }
    pub(crate) fn assign_with_scratch(
        &self,
        point: &PointStructPersisted,
        old: &Member,
        scratch: &mut super::router::OrionRouteScratch,
    ) -> CollectionResult<Member> {
        let mut shards = if old.live && old.vector_present {
            let vectors = point.get_vectors();
            let Some(segment::data_types::vectors::VectorRef::Dense(vector)) =
                vectors.get(&self.vector_name)
            else {
                return Err(bad("migration source routing vector is absent"));
            };
            let hits = self
                .router
                .search_upper_with_scratch(vector, scratch)
                .map_err(bad)?;
            let hits: Vec<_> = hits.iter().map(|h| self.labels[&h.label]).collect();
            super::layout::strict_targets(&hits, &self.owners).map_err(bad)?
        } else if let Some(&owner) = self.anchors.get(&point.id) {
            vec![owner]
        } else if old.live {
            // Payload-only points have no ANN vector; keep one deterministic numeric shard.
            let digest = Sha256::digest(serde_json::to_vec(&point.id).map_err(io)?);
            vec![
                (u64::from_le_bytes(digest[..8].try_into().unwrap()) % u64::from(self.shard_count))
                    as u32,
            ]
        } else {
            vec![]
        };
        if let Some(&owner) = self.anchors.get(&point.id)
            && !shards.contains(&owner)
        {
            shards.push(owner);
        }
        Ok(Member {
            shards,
            live: old.live,
            vector_present: old.vector_present,
        })
    }
}

impl Collection {
    pub(crate) async fn replay_orion_migration_data(
        &self,
        generation: u64,
    ) -> CollectionResult<()> {
        let meta = metadata(&self.path, generation)?;
        let dir = generation_path(&self.path, generation);
        let source = Catalog::load(&self.path, meta.source_generation)?;
        let assigner = Assigner::new(read_artifact(&self.path, generation)?, meta.options.efs)?;
        let holder = self.shards_holder.read().await;
        for sequence in 0..meta.batches {
            let path = dir
                .join("batches")
                .join(format!("batch-{sequence:020}.json"));
            if meta.batch_sha256.get(sequence as usize)
                != Some(&bundle::file_sha256(&path).map_err(io)?)
            {
                return Err(io("recovery batch checksum mismatch"));
            }
            let points: Vec<PointStructPersisted> =
                serde_json::from_reader(File::open(path).map_err(io)?).map_err(io)?;
            let mut by_shard: BTreeMap<u32, Vec<_>> = BTreeMap::new();
            for point in points {
                let old = source
                    .entries
                    .get(&point.id)
                    .ok_or_else(|| io("missing recovery source membership"))?;
                for shard in assigner.assign(&point, old)?.shards {
                    by_shard.entry(shard).or_default().push(point.clone());
                }
            }
            for (id, points) in by_shard {
                if let Some(shard) = holder.get_shard(id).filter(|_| true) {
                    if shard.has_local_shard().await {
                        let op = CollectionUpdateOperations::PointOperation(
                            PointOperations::UpsertPoints(
                                PointInsertOperationsInternal::PointsList(points),
                            ),
                        );
                        super::clock::apply(
                            shard,
                            super::clock::tagged(
                                op,
                                generation,
                                0,
                                sequence + 1,
                                &format!("epoch-{generation}"),
                            )?,
                        )
                        .await?;
                    }
                }
            }
        }
        Ok(())
    }
    pub(crate) async fn begin_orion_migration(
        &self,
        state: &mut ControlState,
        generation: u64,
        shard_count: u32,
        source_revision: u64,
        mut peers: Vec<u64>,
    ) -> CollectionResult<()> {
        let active = state
            .active
            .clone()
            .ok_or_else(|| bad("migration requires an active generation"))?;
        if active.generation == generation {
            return Ok(());
        }
        if generation <= active.generation
            || shard_count == 0
            || super::membership::pending_path(&self.path).exists()
        {
            return Err(bad("invalid or busy migration"));
        }
        if state.pending_generation.is_some() && state.pending_generation != Some(generation) {
            return Err(bad("another generation is pending"));
        }
        if state.pending_generation.is_none() && generation <= state.highest_generation {
            return Err(bad("generation IDs cannot be reused"));
        }
        if state.pending_generation.is_none()
            && (state.phase != Phase::Ready || state.previous.is_some())
        {
            return Err(bad("finish the previous generation first"));
        }
        if active.membership_revision != source_revision {
            return Err(bad("migration source revision changed"));
        }
        peers.sort_unstable();
        peers.dedup();
        if peers.is_empty() {
            return Err(bad("empty migration peer set"));
        }
        let mut opts = options(&self.path, active.generation)?;
        opts.generation = generation;
        opts.shard_count = shard_count;
        let mut artifact = read_artifact(&self.path, active.generation)?;
        let source_graph = bundle::graph_sha256(&artifact).map_err(bad)?;
        let vectors: Vec<_> = artifact
            .upper_nodes
            .iter()
            .map(|n| n.vector.clone())
            .collect();
        let initial = super::layout::cluster(
            &vectors,
            shard_count as usize,
            opts.kmeans_iterations,
            opts.kmeans_seed,
        )
        .map_err(bad)?;
        let mut search = artifact.clone();
        search.upper_k = opts.efs.min(artifact.upper_nodes.len());
        search.upper_ef_search = opts.efs;
        let labels: HashMap<_, _> = artifact
            .upper_nodes
            .iter()
            .enumerate()
            .map(|(i, n)| (n.label, i))
            .collect();
        let router = OrionRouter::new(search).map_err(bad)?;
        let hits: Vec<Vec<usize>> = vectors
            .iter()
            .map(|v| {
                router
                    .search_upper(v)
                    .map(|hits| hits.iter().map(|h| labels[&h.label]).collect())
            })
            .collect::<Result<_, _>>()
            .map_err(bad)?;
        let (owners, _) =
            super::layout::refine_owners(&hits, &initial, opts.topology_iterations).map_err(bad)?;
        artifact.generation = generation;
        artifact.shard_count = shard_count;
        artifact.layout_sha256 = "0".repeat(64);
        for (node, owner) in artifact.upper_nodes.iter_mut().zip(owners) {
            node.owner_shard = owner;
        }
        if bundle::graph_sha256(&artifact).map_err(bad)? != source_graph {
            return Err(io("migration rebuilt the upper graph"));
        }
        let dir = generation_path(&self.path, generation);
        fs::create_dir_all(dir.join("batches")).map_err(io)?;
        if !dir.join("migration.json").is_file() {
            let artifact_sha256 = artifact.canonical_sha256().map_err(bad)?;
            let bytes = artifact.canonical_json_bytes().map_err(bad)?;
            atomic_save(&dir.join("artifact.json"), |out| out.write_all(&bytes)).map_err(io)?;
            save(
                &dir.join("migration.json"),
                &Migration {
                    format_version: 1,
                    source_generation: active.generation,
                    source_revision,
                    source_artifact_sha256: active.artifact_sha256,
                    options: opts,
                    sealed: false,
                    batches: 0,
                    artifact_sha256,
                    peers: peers.clone(),
                    batch_sha256: Vec::new(),
                },
            )?;
        } else {
            let meta = metadata(&self.path, generation)?;
            if meta.source_generation != active.generation
                || meta.source_revision != source_revision
                || meta.options.shard_count != shard_count
            {
                return Err(bad("migration resume binding mismatch"));
            }
            peers = meta.peers;
        }
        state.pending_generation = Some(generation);
        state.phase = Phase::BuildingLayout;
        state.files.clear();
        state.highest_generation = state.highest_generation.max(generation);
        state.pending_artifact_sha256 = None;
        state.pending_layout_sha256 = None;
        lifecycle::save_state(&self.path, state)?;
        self.orion_serving
            .store(false, std::sync::atomic::Ordering::Release);
        let mut config = self.collection_config.write().await;
        let rf = config.params.replication_factor.get() as usize;
        if rf > peers.len() {
            return Err(bad("migration has fewer peers than replication factor"));
        }
        config.params.shard_number =
            std::num::NonZeroU32::new(config.params.shard_number.get().max(shard_count)).unwrap();
        config.save(&self.path).map_err(io)?;
        drop(config);
        for id in 0..shard_count {
            if self.shards_holder.read().await.get_shard(id).is_some() {
                continue;
            }
            let placement: Vec<_> = (0..rf)
                .map(|copy| peers[(id as usize + copy) % peers.len()])
                .collect();
            let replica = self
                .create_replica_set(id, None, &placement, Some(ReplicaState::Active))
                .await?;
            self.shards_holder
                .write()
                .await
                .add_shard(id, replica, None)
                .await?;
        }
        Ok(())
    }

    pub async fn orion_migration_ids(
        &self,
        generation: u64,
    ) -> CollectionResult<Vec<ExtendedPointId>> {
        let meta = metadata(&self.path, generation)?;
        Ok(Catalog::load(&self.path, meta.source_generation)?
            .entries
            .into_iter()
            .filter(|(_, m)| !m.shards.is_empty())
            .map(|(id, _)| id)
            .collect())
    }
    pub async fn orion_migration_points(
        &self,
        generation: u64,
        ids: &[ExtendedPointId],
    ) -> CollectionResult<Vec<PointStructPersisted>> {
        let meta = metadata(&self.path, generation)?;
        let catalog = Catalog::load(&self.path, meta.source_generation)?;
        let mut by_shard: BTreeMap<u32, Vec<_>> = BTreeMap::new();
        for id in ids {
            let member = catalog
                .entries
                .get(id)
                .ok_or_else(|| bad("unknown migration point"))?;
            by_shard.entry(member.shards[0]).or_default().push(*id);
        }
        let holder = self.shards_holder.read().await;
        let mut result = Vec::new();
        for (shard, ids) in by_shard {
            let shard = holder
                .get_shard(shard)
                .ok_or_else(|| io("source shard missing"))?;
            let records = shard
                .retrieve(
                    Arc::new(PointRequestInternal {
                        ids: ids.clone(),
                        with_payload: Some(WithPayloadInterface::Bool(true)),
                        with_vector: WithVector::Bool(true),
                    }),
                    &WithPayload::from(&WithPayloadInterface::Bool(true)),
                    &WithVector::Bool(true),
                    Some(ReadConsistency::Type(ReadConsistencyType::All)),
                    None,
                    false,
                    HwMeasurementAcc::disposable(),
                )
                .await?;
            if records.len() != ids.len() {
                return Err(io("migration source is missing physical points"));
            }
            result.extend(records.into_iter().map(|r| {
                PointStructPersisted {
                    id: r.id,
                    vector: r
                        .vector
                        .unwrap_or(segment::data_types::vectors::VectorStructInternal::Named(
                            HashMap::new(),
                        ))
                        .into(),
                    payload: r.payload,
                }
            }));
        }
        result.sort_by_key(|p| p.id);
        Ok(result)
    }

    pub(crate) async fn migrate_orion_batch(
        &self,
        state: &ControlState,
        generation: u64,
        sequence: u64,
        points: Vec<PointStructPersisted>,
    ) -> CollectionResult<()> {
        if state
            .active
            .as_ref()
            .is_some_and(|g| g.generation == generation)
        {
            return Ok(());
        }
        if state.pending_generation != Some(generation)
            || !matches!(state.phase, Phase::BuildingLayout | Phase::Importing)
            || points.len() > 512
        {
            return Err(bad("invalid migration batch phase/size"));
        }
        let mut meta = metadata(&self.path, generation)?;
        if meta.sealed {
            return Ok(());
        }
        let artifact = read_artifact(&self.path, generation)?;
        let assigner = Assigner::new(artifact, meta.options.efs)?;
        let source = Catalog::load(&self.path, meta.source_generation)?;
        if source.revision != meta.source_revision {
            return Err(io("migration source changed after write pause"));
        }
        let dir = generation_path(&self.path, generation);
        let path = dir
            .join("batches")
            .join(format!("batch-{sequence:020}.json"));
        let canonical = serde_json::to_value(&points).map_err(bad)?;
        if path.is_file() {
            let existing: serde_json::Value =
                serde_json::from_reader(File::open(&path).map_err(io)?).map_err(io)?;
            if existing != canonical {
                return Err(bad("migration batch retry differs"));
            }
            if sequence < meta.batches {
                if meta.batch_sha256.get(sequence as usize)
                    != Some(&bundle::file_sha256(&path).map_err(io)?)
                {
                    return Err(io("migration batch checksum mismatch"));
                }
                return Ok(());
            }
        }
        if sequence != meta.batches {
            return Err(bad("migration batch sequence gap"));
        }
        let mut ids = BTreeSet::new();
        let mut by_shard: BTreeMap<u32, Vec<_>> = BTreeMap::new();
        for point in &points {
            if !ids.insert(point.id) {
                return Err(bad("duplicate migration point"));
            }
            let old = source
                .entries
                .get(&point.id)
                .ok_or_else(|| bad("unknown migration source ID"))?;
            for shard in assigner.assign(point, old)?.shards {
                by_shard.entry(shard).or_default().push(point.clone());
            }
        }
        save(&path, &canonical)?;
        let holder = self.shards_holder.read().await;
        let mut tasks = Vec::new();
        for (id, points) in by_shard {
            let shard = holder
                .get_shard(id)
                .ok_or_else(|| io("migration target shard missing"))?;
            tasks.push(async move {
                if shard.has_local_shard().await {
                    let op =
                        CollectionUpdateOperations::PointOperation(PointOperations::UpsertPoints(
                            PointInsertOperationsInternal::PointsList(points),
                        ));
                    super::clock::apply(
                        shard,
                        super::clock::tagged(
                            op,
                            generation,
                            0,
                            sequence + 1,
                            &format!("epoch-{generation}"),
                        )?,
                    )
                    .await?;
                }
                CollectionResult::Ok(())
            });
        }
        for result in join_all(tasks).await {
            result.map_err(io)?;
        }
        meta.batch_sha256
            .push(bundle::file_sha256(&path).map_err(io)?);
        meta.batches += 1;
        save(&dir.join("migration.json"), &meta)
    }

    pub(crate) async fn seal_orion_migration(
        &self,
        state: &mut ControlState,
        generation: u64,
    ) -> CollectionResult<()> {
        if state
            .active
            .as_ref()
            .is_some_and(|g| g.generation == generation)
        {
            return Ok(());
        }
        if state.pending_generation != Some(generation) {
            return Err(bad("no pending migration"));
        }
        let mut meta = metadata(&self.path, generation)?;
        let dir = generation_path(&self.path, generation);
        let mut artifact = read_artifact(&self.path, generation)?;
        let assigner = Assigner::new(artifact.clone(), meta.options.efs)?;
        let mut source = Catalog::load(&self.path, meta.source_generation)?;
        let expected: BTreeSet<_> = source
            .entries
            .iter()
            .filter(|(_, m)| !m.shards.is_empty())
            .map(|(id, _)| *id)
            .collect();
        let mut covered = BTreeSet::new();
        for sequence in 0..meta.batches {
            let path = dir
                .join("batches")
                .join(format!("batch-{sequence:020}.json"));
            if meta.batch_sha256.get(sequence as usize)
                != Some(&bundle::file_sha256(&path).map_err(io)?)
            {
                return Err(io("migration batch checksum mismatch"));
            }
            let points: Vec<PointStructPersisted> =
                serde_json::from_reader(File::open(path).map_err(io)?).map_err(io)?;
            for point in points {
                if !covered.insert(point.id) {
                    return Err(bad("migration repeats a point across batches"));
                }
                let old = source
                    .entries
                    .get(&point.id)
                    .ok_or_else(|| bad("unexpected migrated ID"))?
                    .clone();
                source
                    .entries
                    .insert(point.id, assigner.assign(&point, &old)?);
            }
        }
        if covered != expected {
            return Err(bad(
                "migration has not covered the complete source membership",
            ));
        }
        let rows: Vec<_> = source.entries.iter().collect();
        artifact.layout_sha256 = format!(
            "{:x}",
            Sha256::digest(
                serde_json::to_vec(&serde_json::to_value(rows).map_err(io)?).map_err(io)?
            )
        );
        artifact.logical_point_count = source.entries.values().filter(|m| m.live).count() as u64;
        artifact.physical_point_count =
            source.entries.values().map(|m| m.shards.len() as u64).sum();
        artifact.validate().map_err(bad)?;
        super::membership::save_base(&self.path, generation, source.entries)?;
        let bytes = artifact.canonical_json_bytes().map_err(bad)?;
        atomic_save(&dir.join("artifact.json"), |out| out.write_all(&bytes)).map_err(io)?;
        meta.artifact_sha256 = artifact.canonical_sha256().map_err(bad)?;
        meta.sealed = true;
        save(&dir.join("migration.json"), &meta)?;
        state.phase = Phase::Indexing;
        state.pending_artifact_sha256 = Some(meta.artifact_sha256);
        state.pending_layout_sha256 = Some(artifact.layout_sha256);
        lifecycle::save_state(&self.path, state)
    }

    pub(crate) async fn trim_orion_shards(&self, count: u32) -> CollectionResult<()> {
        let mut holder = self.shards_holder.write().await;
        let obsolete: Vec<_> = holder
            .get_shards()
            .filter(|(id, _)| *id >= count)
            .map(|(id, _)| id)
            .collect();
        for id in obsolete {
            let archive = self
                .path
                .join("orion_retired")
                .join(format!("shard-{id}-{}", uuid::Uuid::new_v4()));
            holder.retire_orion_shard(id, &archive).await?;
        }
        drop(holder);
        let mut config = self.collection_config.write().await;
        config.params.shard_number =
            std::num::NonZeroU32::new(count).ok_or_else(|| bad("empty shard set"))?;
        config.save(&self.path).map_err(io)
    }

    pub(crate) async fn rollback_orion_unsealed(&self, generation: u64) -> CollectionResult<()> {
        let meta = metadata(&self.path, generation)?;
        let source = Catalog::load(&self.path, meta.source_generation)?;
        let assigner = Assigner::new(read_artifact(&self.path, generation)?, meta.options.efs)?;
        let mut pending = source.entries.clone();
        let dir = generation_path(&self.path, generation).join("batches");
        for entry in fs::read_dir(dir).map_err(io)? {
            let path = entry.map_err(io)?.path();
            if !path
                .file_name()
                .is_some_and(|name| name.to_string_lossy().starts_with("batch-"))
            {
                continue;
            }
            let points: Vec<PointStructPersisted> =
                serde_json::from_reader(File::open(path).map_err(io)?).map_err(io)?;
            for point in points {
                let old = source
                    .entries
                    .get(&point.id)
                    .ok_or_else(|| io("rollback source ID missing"))?;
                pending.insert(point.id, assigner.assign(&point, old)?);
            }
        }
        // A temporary base describes all copies that may have been written, including a
        // batch interrupted between two target shards. Deleting a missing copy is idempotent.
        super::membership::save_base(&self.path, generation, pending)?;
        self.remove_orion_difference(generation, meta.source_generation)
            .await
    }
}
