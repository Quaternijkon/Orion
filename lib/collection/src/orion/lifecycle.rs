//! Durable collection-owned Orion generation protocol. The consensus adapter transports
//! commands; this module owns file validation, import, readiness, activation and rollback.
use std::collections::{BTreeMap, HashMap};
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use common::counter::hardware_accumulator::HwMeasurementAcc;
use common::fs::atomic_save;
use common::types::DeferredBehavior;
use futures::{StreamExt, stream};
use segment::types::{ExtendedPointId, WithPayloadInterface, WithVector};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::{OrionRouter, OrionRoutingArtifact, bundle};
use crate::collection::Collection;
use crate::config::AutoShardPolicy;
use crate::operations::CollectionUpdateOperations;
use crate::operations::point_ops::{
    PointInsertOperationsInternal, PointOperations, PointStructPersisted, VectorStructPersisted,
};
use crate::operations::types::{
    CollectionError, CollectionResult, CollectionStatus, OptimizersStatus,
};

pub const CONTROL_DIR: &str = "orion_control";
pub const MAX_CHUNK_BYTES: usize = 64 * 1024;

pub(crate) struct IndexProof {
    pub points: usize,
    pub vectors: usize,
}

/// Inspect actual segment/index ownership, rather than trusting the approximate HNSW
/// graph-size counter (which can still include deleted offsets).
pub(crate) async fn check_index(
    holder: shard::segment_holder::locked::LockedSegmentHolder,
    vector_name: String,
    anchors: Vec<ExtendedPointId>,
    allow_appendable: bool,
) -> CollectionResult<IndexProof> {
    tokio::task::spawn_blocking(move || {
        let holder = holder.read();
        let mut points = 0;
        let mut vectors = 0;
        let mut indexed_segments = 0;
        let mut found = std::collections::HashSet::new();
        for (_, locked) in holder.iter() {
            let segment = locked.get();
            let segment = segment.read();
            let info = segment.info();
            points += info.num_points;
            let count = info
                .vector_data
                .get(&vector_name)
                .map(|v| v.num_vectors)
                .unwrap_or(0);
            vectors += count;
            if count == 0 {
                continue;
            }
            if allow_appendable
                && info.segment_type == segment::types::SegmentType::Plain
                && info.is_appendable
                && !anchors.iter().any(|id| segment.has_point(*id))
            {
                continue;
            }
            if info.segment_type != segment::types::SegmentType::Indexed
                || info.vector_data[&vector_name].num_indexed_vectors < count
            {
                return Err(bad("routing vectors are still outside a complete HNSW"));
            }
            indexed_segments += 1;
            for &anchor in &anchors {
                if segment.has_point(anchor)
                    && segment
                        .vector(
                            &vector_name,
                            anchor,
                            &common::counter::hardware_counter::HardwareCounterCell::disposable(),
                        )
                        .map_err(io)?
                        .is_some()
                {
                    found.insert(anchor);
                }
            }
        }
        if indexed_segments > 1 || found.len() != anchors.len() {
            return Err(bad(
                "Orion requires one complete local HNSW with every owner entry",
            ));
        }
        Ok(IndexProof { points, vectors })
    })
    .await
    .map_err(io)?
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FileBinding {
    pub size: u64,
    pub sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "operation", rename_all = "snake_case", deny_unknown_fields)]
pub enum ControlCommand {
    /// Reserve an empty collection for a build whose final layout checksum is not yet known.
    BeginStream {
        spec: super::streaming::StreamSpec,
    },
    /// Import an ordered, retryable batch while the remaining layout is still being computed.
    StreamBatch {
        generation: u64,
        sequence: u64,
        rows: Vec<super::streaming::StreamRow>,
    },
    CommitWrite {
        plan: super::membership::WritePlan,
    },
    RecoverWrite,
    BeginMigration {
        generation: u64,
        shard_count: u32,
        source_revision: u64,
        peers: Vec<u64>,
    },
    MigrateBatch {
        generation: u64,
        sequence: u64,
        points: Vec<PointStructPersisted>,
    },
    SealMigration {
        generation: u64,
    },
    Write {
        plan: super::membership::WritePlan,
    },
    Begin {
        generation: u64,
        files: BTreeMap<String, FileBinding>,
    },
    Chunk {
        generation: u64,
        file: String,
        offset: u64,
        bytes: Vec<u8>,
    },
    Prepare {
        generation: u64,
    },
    Import {
        generation: u64,
    },
    Validate {
        generation: u64,
    },
    Activate {
        generation: u64,
    },
    Rollback {
        generation: u64,
    },
    Cleanup {
        generation: u64,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Phase {
    Collecting,
    BuildingUpper,
    BuildingLayout,
    Importing,
    Indexing,
    Finalizing,
    Ready,
    RollbackPending,
    Failed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Generation {
    pub generation: u64,
    pub artifact_sha256: String,
    pub layout_sha256: String,
    pub shard_count: u32,
    #[serde(default)]
    pub membership_revision: u64,
    #[serde(default)]
    pub last_write_token: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ControlState {
    pub format_version: u32,
    pub phase: Phase,
    pub active: Option<Generation>,
    pub previous: Option<Generation>,
    pub pending_generation: Option<u64>,
    pub files: BTreeMap<String, FileBinding>,
    #[serde(default)]
    pub pending_artifact_sha256: Option<String>,
    #[serde(default)]
    pub pending_layout_sha256: Option<String>,
    #[serde(default)]
    pub last_error: Option<String>,
    #[serde(default)]
    pub pending_write: Option<super::membership::PendingWriteStatus>,
    #[serde(default)]
    pub highest_generation: u64,
}

impl Default for ControlState {
    fn default() -> Self {
        Self {
            format_version: 1,
            phase: Phase::Collecting,
            active: None,
            previous: None,
            pending_generation: None,
            files: BTreeMap::new(),
            pending_artifact_sha256: None,
            pending_layout_sha256: None,
            last_error: None,
            pending_write: None,
            highest_generation: 0,
        }
    }
}

pub(crate) fn bad(error: impl std::fmt::Display) -> CollectionError {
    CollectionError::bad_request(error.to_string())
}
pub(crate) fn io(error: impl std::fmt::Display) -> CollectionError {
    CollectionError::service_error(error.to_string())
}
fn state_path(collection: &Path) -> PathBuf {
    collection.join(CONTROL_DIR).join("state.json")
}
pub fn generation_path(collection: &Path, generation: u64) -> PathBuf {
    collection
        .join(CONTROL_DIR)
        .join(format!("generation-{generation}"))
}

#[derive(Serialize, Deserialize)]
struct StateEnvelope {
    sha256: String,
    state: ControlState,
}

pub fn read_state(collection: &Path) -> CollectionResult<Option<ControlState>> {
    let path = state_path(collection);
    if !path.exists() {
        return Ok(None);
    }
    let envelope: StateEnvelope =
        serde_json::from_reader(File::open(path).map_err(io)?).map_err(io)?;
    let checksum = format!(
        "{:x}",
        Sha256::digest(serde_json::to_vec(&envelope.state).map_err(io)?)
    );
    if envelope.state.format_version != 1 || checksum != envelope.sha256 {
        return Err(io("corrupt Orion lifecycle state"));
    }
    Ok(Some(envelope.state))
}

pub(crate) fn save_state(collection: &Path, state: &ControlState) -> CollectionResult<()> {
    fs::create_dir_all(collection.join(CONTROL_DIR)).map_err(io)?;
    let envelope = StateEnvelope {
        sha256: format!(
            "{:x}",
            Sha256::digest(serde_json::to_vec(state).map_err(io)?)
        ),
        state: state.clone(),
    };
    let bytes = serde_json::to_vec(&envelope).map_err(io)?;
    atomic_save(&state_path(collection), |writer| writer.write_all(&bytes)).map_err(io)
}

pub fn may_serve(collection: &Path) -> bool {
    if super::membership::pending_path(collection).exists() {
        return false;
    }
    match read_state(collection) {
        Ok(None) => true, // Explicit legacy artifacts still load under the old static contract.
        // Managed collections recheck local index readiness on their first read after restart.
        Ok(Some(_)) => false,
        Err(_) => false,
    }
}

pub(crate) fn read_bundle(
    dir: &Path,
) -> CollectionResult<(
    bundle::BuildManifest,
    bundle::ImportManifest,
    OrionRoutingArtifact,
)> {
    let build: bundle::BuildManifest = serde_json::from_reader(BufReader::with_capacity(
        1024 * 1024,
        File::open(dir.join("build-manifest.json")).map_err(io)?,
    ))
    .map_err(bad)?;
    let import: bundle::ImportManifest =
        serde_json::from_reader(File::open(dir.join(&build.import_manifest_file)).map_err(io)?)
            .map_err(bad)?;
    let artifact = OrionRoutingArtifact::read_json(
        &dir.join(&import.orion_artifact_file),
        Some(&import.orion_artifact_sha256),
    )
    .map_err(bad)?;
    Ok((build, import, artifact))
}

pub(crate) fn memberships_for_shards(
    dir: &Path,
    local: Option<&std::collections::BTreeSet<u32>>,
) -> CollectionResult<BTreeMap<ExtendedPointId, Vec<u32>>> {
    // The bundle was independently replayed at Prepare. Loading its catalog need not
    // materialize the upper graph, nor retain remote-only memberships on a worker.
    let build: bundle::BuildManifest = serde_json::from_reader(BufReader::with_capacity(
        1024 * 1024,
        File::open(dir.join("build-manifest.json")).map_err(io)?,
    ))
    .map_err(bad)?;
    bundle::validate_filename(&build.import_manifest_file).map_err(bad)?;
    let import_path = dir.join(&build.import_manifest_file);
    if bundle::file_sha256(&import_path).map_err(io)? != build.import_manifest_sha256 {
        return Err(io("membership manifest checksum mismatch"));
    }
    let import: bundle::ImportManifest =
        serde_json::from_reader(File::open(import_path).map_err(io)?).map_err(bad)?;
    bundle::validate_filename(&import.assignments_file).map_err(bad)?;
    if bundle::file_sha256(&dir.join(&import.assignments_file)).map_err(io)?
        != import.assignments_sha256
    {
        return Err(io("membership base checksum mismatch"));
    }
    let mut result = BTreeMap::new();
    for line in BufReader::new(File::open(dir.join(import.assignments_file)).map_err(io)?).lines() {
        let row: bundle::Assignment = serde_json::from_str(&line.map_err(io)?).map_err(bad)?;
        if local.is_none_or(|shards| row.shards.iter().any(|s| shards.contains(s))) {
            result.insert(ExtendedPointId::NumId(row.id), row.shards);
        }
    }
    Ok(result)
}

impl Collection {
    pub async fn orion_control_state(&self) -> CollectionResult<Option<ControlState>> {
        let _ = self.ensure_orion_worker_ready().await;
        let mut state = read_state(&self.path)?;
        if let Some(state) = state.as_mut() {
            state.pending_write = super::membership::pending_status(&self.path, self.this_peer_id)?;
        }
        if let Some(state) = state.as_mut()
            && state.phase == Phase::Ready
            && !self
                .orion_serving
                .load(std::sync::atomic::Ordering::Acquire)
        {
            state.phase = if super::membership::pending_path(&self.path).exists() {
                Phase::Finalizing
            } else {
                Phase::Indexing
            };
        }
        Ok(state)
    }

    /// Called only by the consensus operation adapter (or standalone equivalent).
    /// Failed durable operations keep a resumable phase and never publish a router.
    pub async fn apply_orion_control(&self, command: ControlCommand) -> CollectionResult<()> {
        if read_state(&self.path)?.is_none()
            && matches!(
                &command,
                ControlCommand::Write { .. }
                    | ControlCommand::CommitWrite { .. }
                    | ControlCommand::BeginMigration { .. }
                    | ControlCommand::MigrateBatch { .. }
                    | ControlCommand::SealMigration { .. }
                    | ControlCommand::Activate { .. }
            )
        {
            self.recover_orion_metadata_from_peer().await?;
        }
        let _guard = self.orion_control_lock.lock().await;
        if super::membership::pending_path(&self.path).exists()
            && !matches!(
                &command,
                ControlCommand::Write { .. }
                    | ControlCommand::CommitWrite { .. }
                    | ControlCommand::RecoverWrite
            )
        {
            return Err(bad("finish the pending write first"));
        }
        let mut state = read_state(&self.path)?.unwrap_or_default();
        match command {
            ControlCommand::BeginStream { spec } => {
                return self.begin_orion_stream(&mut state, spec).await;
            }
            ControlCommand::StreamBatch {
                generation,
                sequence,
                rows,
            } => {
                return self
                    .import_orion_stream_batch(&state, generation, sequence, rows)
                    .await;
            }
            ControlCommand::CommitWrite { plan } => return self.commit_orion_write(plan).await,
            ControlCommand::RecoverWrite => {
                if let Some(plan) = super::membership::pending_plan(&self.path)? {
                    return self.apply_orion_write(plan).await;
                }
                return Ok(());
            }
            ControlCommand::BeginMigration {
                generation,
                shard_count,
                source_revision,
                peers,
            } => {
                return self
                    .begin_orion_migration(
                        &mut state,
                        generation,
                        shard_count,
                        source_revision,
                        peers,
                    )
                    .await;
            }
            ControlCommand::MigrateBatch {
                generation,
                sequence,
                points,
            } => {
                return self
                    .migrate_orion_batch(&state, generation, sequence, points)
                    .await;
            }
            ControlCommand::SealMigration { generation } => {
                return self.seal_orion_migration(&mut state, generation).await;
            }
            ControlCommand::Write { plan } => return self.apply_orion_write(plan).await,
            ControlCommand::Begin { generation, files } => {
                if super::membership::pending_path(&self.path).exists() {
                    return Err(bad(
                        "finish the pending write before beginning a generation",
                    ));
                }
                if !matches!(
                    self.collection_config.read().await.auto_shard_policy,
                    Some(AutoShardPolicy::Orion { .. })
                ) {
                    return Err(bad(
                        "Orion installation requires the Orion collection policy",
                    ));
                }
                if generation == 0 || files.is_empty() || !files.contains_key("build-manifest.json")
                {
                    return Err(bad("invalid generation upload"));
                }
                if (state.pending_generation == Some(generation)
                    || state
                        .active
                        .as_ref()
                        .is_some_and(|g| g.generation == generation))
                    && state.files == files
                {
                    return Ok(());
                }
                let collection_path = self.path.clone();
                let mut stream_state = state.clone();
                let stream_files = files.clone();
                if let Some(stream_state) = tokio::task::spawn_blocking(move || {
                    super::streaming::bind_files(
                        &collection_path,
                        &mut stream_state,
                        generation,
                        &stream_files,
                    )
                    .map(|bound| bound.then_some(stream_state))
                })
                .await
                .map_err(io)??
                {
                    return save_state(&self.path, &stream_state);
                }
                if generation <= state.highest_generation {
                    return Err(bad("generation IDs cannot be reused after rollback"));
                }
                if state.pending_generation.is_some()
                    || state
                        .active
                        .as_ref()
                        .is_some_and(|g| generation <= g.generation)
                {
                    return Err(bad(
                        "generation must increase and only one migration may be pending",
                    ));
                }
                for (name, binding) in &files {
                    bundle::validate_filename(name).map_err(bad)?;
                    if binding.sha256.len() != 64
                        || !binding.sha256.bytes().all(|b| b.is_ascii_hexdigit())
                    {
                        return Err(bad("invalid file checksum"));
                    }
                }
                fs::create_dir_all(generation_path(&self.path, generation)).map_err(io)?;
                state.pending_generation = Some(generation);
                state.files = files;
                state.phase = Phase::Collecting;
                state.highest_generation = generation;
                self.orion_managed
                    .store(true, std::sync::atomic::Ordering::Release);
                self.orion_data_ready
                    .store(false, std::sync::atomic::Ordering::Release);
            }
            ControlCommand::Chunk {
                generation,
                file,
                offset,
                bytes,
            } => {
                if (state.pending_generation != Some(generation)
                    && !state
                        .active
                        .as_ref()
                        .is_some_and(|g| g.generation == generation))
                    || bytes.len() > MAX_CHUNK_BYTES
                {
                    return Err(bad("invalid upload phase or oversized chunk"));
                }
                let binding = state
                    .files
                    .get(&file)
                    .ok_or_else(|| bad("undeclared upload file"))?;
                let end = offset
                    .checked_add(bytes.len() as u64)
                    .ok_or_else(|| bad("chunk offset overflow"))?;
                if end > binding.size {
                    return Err(bad("chunk exceeds declared file size"));
                }
                let path = generation_path(&self.path, generation).join(file);
                let mut out = OpenOptions::new()
                    .create(true)
                    .truncate(false)
                    .read(true)
                    .write(true)
                    .open(path)
                    .map_err(io)?;
                let length = out.metadata().map_err(io)?.len();
                if offset < length {
                    if end > length {
                        return Err(bad("overlapping partial chunk"));
                    }
                    out.seek(SeekFrom::Start(offset)).map_err(io)?;
                    let mut existing = vec![0; bytes.len()];
                    out.read_exact(&mut existing).map_err(io)?;
                    if existing != bytes {
                        return Err(bad("retried chunk has different bytes"));
                    }
                    return Ok(());
                }
                if offset != length || state.phase != Phase::Collecting {
                    return Err(bad("out of order chunk or sealed upload"));
                }
                out.seek(SeekFrom::Start(offset)).map_err(io)?;
                out.write_all(&bytes).map_err(io)?;
                out.sync_all().map_err(io)?;
                return Ok(());
            }
            ControlCommand::Prepare { generation } => {
                if state
                    .active
                    .as_ref()
                    .is_some_and(|g| g.generation == generation)
                {
                    return Ok(());
                }
                if state.pending_generation != Some(generation) {
                    return Err(bad("no matching pending generation"));
                }
                if matches!(
                    state.phase,
                    Phase::Importing | Phase::Indexing | Phase::Finalizing
                ) {
                    return Ok(());
                }
                if state.phase != Phase::Collecting && state.phase != Phase::BuildingLayout {
                    return Err(bad("generation cannot be prepared in this phase"));
                }
                let dir = generation_path(&self.path, generation);
                for (name, binding) in &state.files {
                    let path = dir.join(name);
                    if fs::metadata(&path).map_err(bad)?.len() != binding.size
                        || bundle::file_sha256(&path).map_err(bad)? != binding.sha256
                    {
                        return Err(bad(format!("upload checksum/size mismatch: {name}")));
                    }
                }
                let verified = tokio::task::spawn_blocking(move || bundle::verify_integrity(&dir))
                    .await
                    .map_err(io)?
                    .map_err(bad)?;
                if verified.orion_generation != generation {
                    return Err(bad("uploaded generation mismatch"));
                }
                let (build, _, artifact) = read_bundle(&generation_path(&self.path, generation))?;
                let collection_path = self.path.clone();
                let import = verified.clone();
                let streamed = tokio::task::spawn_blocking(move || {
                    super::streaming::verify_import(&collection_path, generation, &build, &import)
                })
                .await
                .map_err(io)??;
                let config = self.collection_config.read().await;
                if state.active.is_none() {
                    if !matches!(&config.auto_shard_policy,Some(AutoShardPolicy::Orion {generation:g,artifact_sha256}) if *g==generation&&(*artifact_sha256==verified.orion_artifact_sha256 || (streamed && *artifact_sha256 == "0".repeat(64))))
                    {
                        return Err(bad("initial generation differs from the collection policy"));
                    }
                    for (_, shard) in self
                        .shards_holder
                        .read()
                        .await
                        .get_shards()
                        .filter(|_| !streamed)
                    {
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
                            return Err(bad("initial native installation requires empty shards"));
                        }
                    }
                }
                let params = config
                    .params
                    .vectors
                    .get_params(&artifact.vector_schema.vector_name)
                    .ok_or_else(|| bad("unknown routing vector"))?;
                if params.size.get() as usize != artifact.vector_schema.dimension
                    || params.distance != artifact.vector_schema.distance
                {
                    return Err(bad(
                        "collection vector schema differs from uploaded generation",
                    ));
                }
                if params.datatype.unwrap_or_default()
                    != match artifact.vector_schema.datatype {
                        super::OrionVectorDatatype::Uint8 => {
                            crate::operations::types::Datatype::Uint8
                        }
                        _ => crate::operations::types::Datatype::Float32,
                    }
                    || config.params.vectors.params_iter().count() != 1
                {
                    return Err(bad(
                        "native Orion requires exactly one matching float32/uint8 dense vector schema",
                    ));
                }
                if artifact.shard_count as usize != self.shards_holder.read().await.len() {
                    return Err(bad("prepare shard count differs from collection"));
                }
                if let Some(active) = &state.active {
                    if active.membership_revision != 0 {
                        return Err(bad(
                            "use repartition to migrate a collection with dynamic writes",
                        ));
                    }
                    let (_, source_import, source_artifact) =
                        read_bundle(&generation_path(&self.path, active.generation))?;
                    if source_import.vectors_sha256 != verified.vectors_sha256
                        || bundle::graph_sha256(&source_artifact).map_err(bad)?
                            != bundle::graph_sha256(&artifact).map_err(bad)?
                    {
                        return Err(bad(
                            "active-generation installation must preserve data and the production graph",
                        ));
                    }
                }
                state.pending_artifact_sha256 = Some(verified.orion_artifact_sha256);
                state.pending_layout_sha256 = Some(verified.assignments_sha256);
                let mut optimizers = config.optimizer_config.clone();
                optimizers.default_segment_number = 1;
                drop(config);
                self.update_optimizer_params(optimizers).await?;
                self.recreate_optimizers_blocking().await?;
                state.phase = if streamed {
                    Phase::Indexing
                } else {
                    Phase::BuildingLayout
                };
            }
            ControlCommand::Import { generation } => {
                if state
                    .active
                    .as_ref()
                    .is_some_and(|g| g.generation == generation)
                    || (state.pending_generation == Some(generation)
                        && matches!(state.phase, Phase::Indexing | Phase::Finalizing))
                {
                    return Ok(());
                }
                if state.pending_generation != Some(generation)
                    || !matches!(
                        state.phase,
                        Phase::BuildingLayout | Phase::Importing | Phase::Indexing
                    )
                {
                    return Err(bad("invalid import phase"));
                }
                if generation_path(&self.path, generation)
                    .join("stream")
                    .is_dir()
                {
                    return Err(bad(
                        "streaming imports must finish and prepare their final bundle first",
                    ));
                }
                state.phase = Phase::Importing;
                save_state(&self.path, &state)?;
                self.orion_serving
                    .store(false, std::sync::atomic::Ordering::Release);
                self.import_orion_local(generation).await?;
                state.phase = Phase::Indexing;
            }
            ControlCommand::Validate { generation } => {
                if state
                    .active
                    .as_ref()
                    .is_some_and(|g| g.generation == generation)
                    && matches!(state.phase, Phase::Ready | Phase::Finalizing)
                {
                    return Ok(());
                }
                if state.pending_generation != Some(generation)
                    || !matches!(state.phase, Phase::Indexing | Phase::Finalizing)
                {
                    return Err(bad("invalid validation phase"));
                }
                // A readiness miss must not stop Raft application. The coordinator checks
                // every peer's persisted phase before proposing Activate.
                match self.validate_orion_local(generation).await {
                    Ok(()) => {
                        state.phase = Phase::Finalizing;
                        state.last_error = None;
                    }
                    Err(error) => {
                        state.phase = Phase::Indexing;
                        state.last_error = Some(error.to_string());
                    }
                }
            }
            ControlCommand::Activate { generation } => {
                if state
                    .active
                    .as_ref()
                    .is_some_and(|g| g.generation == generation)
                    && matches!(state.phase, Phase::Ready | Phase::Finalizing)
                {
                    return Ok(());
                }
                if state.pending_generation != Some(generation) || state.phase != Phase::Finalizing
                {
                    return Err(bad("activation requires validated local replicas"));
                }
                let artifact = super::migration::read_artifact(&self.path, generation)?;
                let artifact_sha256 = artifact.canonical_sha256().map_err(io)?;
                let router = Arc::new(OrionRouter::from_artifact(&artifact).map_err(bad)?);
                let artifact_path = super::routing_artifact_path(&self.path, generation);
                fs::create_dir_all(artifact_path.parent().unwrap()).map_err(io)?;
                atomic_save(&artifact_path, |out| {
                    serde_json::to_writer(out, &artifact).map_err(std::io::Error::other)
                })
                .map_err(io)?;
                let mut config = self.collection_config.write().await;
                config.auto_shard_policy = Some(AutoShardPolicy::Orion {
                    generation,
                    artifact_sha256: artifact_sha256.clone(),
                });
                config.save(&self.path).map_err(io)?;
                state.previous = state.active.take();
                state.active = Some(Generation {
                    generation,
                    artifact_sha256,
                    layout_sha256: artifact.layout_sha256.clone(),
                    shard_count: artifact.shard_count,
                    membership_revision: 0,
                    last_write_token: None,
                });
                state.pending_generation = None;
                state.phase = Phase::Finalizing;
                save_state(&self.path, &state)?;
                *self.orion_catalog.lock().await = None;
                self.orion_visibility
                    .store(Arc::new(super::membership::load_visibility(&self.path)));
                self.orion_router.store(Some(router));
                self.orion_serving
                    .store(false, std::sync::atomic::Ordering::Release);
                return Ok(());
            }
            ControlCommand::Rollback { generation } => {
                if state.pending_generation != Some(generation) {
                    return Err(bad("rollback requires pending generation"));
                }
                state.phase = Phase::RollbackPending;
                save_state(&self.path, &state)?;
                if let Some(active) = &state.active {
                    if generation_path(&self.path, generation)
                        .join("migration.json")
                        .is_file()
                        && !super::membership::has_base(&self.path, generation)
                    {
                        self.rollback_orion_unsealed(generation).await?;
                    } else {
                        self.remove_orion_difference(generation, active.generation)
                            .await?;
                    }
                    self.trim_orion_shards(active.shard_count).await?;
                    let artifact = OrionRoutingArtifact::read_json(
                        &super::routing_artifact_path(&self.path, active.generation),
                        Some(&active.artifact_sha256),
                    )
                    .map_err(io)?;
                    let mut config = self.collection_config.write().await;
                    config.auto_shard_policy = Some(AutoShardPolicy::Orion {
                        generation: active.generation,
                        artifact_sha256: active.artifact_sha256.clone(),
                    });
                    config.save(&self.path).map_err(io)?;
                    self.orion_router
                        .store(Some(Arc::new(OrionRouter::new(artifact).map_err(io)?)));
                }
                state.pending_generation = None;
                state.phase = if state.active.is_some() {
                    Phase::Ready
                } else {
                    Phase::Failed
                };
                self.orion_serving
                    .store(false, std::sync::atomic::Ordering::Release);
            }
            ControlCommand::Cleanup { generation } => {
                if state
                    .active
                    .as_ref()
                    .is_some_and(|g| g.generation == generation)
                    && state.phase == Phase::Ready
                    && state.previous.is_none()
                {
                    return Ok(());
                }
                if state.active.as_ref().map(|g| g.generation) != Some(generation)
                    || !matches!(state.phase, Phase::Ready | Phase::Finalizing)
                {
                    return Err(bad("cleanup requires active generation"));
                }
                if let Some(previous) = &state.previous {
                    self.remove_orion_difference(previous.generation, generation)
                        .await?;
                }
                self.trim_orion_shards(state.active.as_ref().unwrap().shard_count)
                    .await?;
                self.stamp_orion_epoch(generation).await?;
                state.previous = None; // Old artifact files remain recoverable; only obsolete L0 copies are removed.
                state.phase = Phase::Ready;
                self.orion_data_ready
                    .store(true, std::sync::atomic::Ordering::Release);
                self.orion_serving
                    .store(true, std::sync::atomic::Ordering::Release);
            }
        }
        save_state(&self.path, &state)
    }

    pub(crate) async fn import_orion_local(&self, generation: u64) -> CollectionResult<()> {
        let dir = generation_path(&self.path, generation);
        let (_, import, _) = read_bundle(&dir)?;
        let holder = Arc::new(self.shards_holder.clone().read_owned().await);
        let mut queues = BTreeMap::new();
        let mut workers = tokio::task::JoinSet::new();
        for (id, shard) in holder.get_shards() {
            if !shard.has_local_shard().await {
                continue;
            }
            let (sender, mut receiver) = tokio::sync::mpsc::channel::<Vec<PointStructPersisted>>(4);
            queues.insert(id, (sender, Vec::new()));
            let holder = holder.clone();
            workers.spawn(async move {
                let shard = holder
                    .get_shard(id)
                    .ok_or_else(|| bad("import shard vanished"))?;
                let mut sequence = 0;
                while let Some(points) = receiver.recv().await {
                    sequence += 1;
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
                            sequence,
                            &format!("epoch-{generation}"),
                        )?,
                    )
                    .await?;
                }
                CollectionResult::Ok(())
            });
        }
        let mut vectors = BufReader::new(File::open(dir.join(import.vectors_file)).map_err(io)?);
        let mut row_bytes = vec![
            0;
            import
                .dimension
                .checked_mul(import.vector_encoding.element_bytes())
                .ok_or_else(|| bad("dimension overflow"))?
        ];
        for line in
            BufReader::new(File::open(dir.join(import.assignments_file)).map_err(io)?).lines()
        {
            let assignment: bundle::Assignment =
                serde_json::from_str(&line.map_err(io)?).map_err(bad)?;
            vectors.read_exact(&mut row_bytes).map_err(io)?;
            let vector = import.vector_encoding.decode(&row_bytes).map_err(bad)?;
            let vector = if import.vector_name.is_empty() {
                VectorStructPersisted::Single(vector)
            } else {
                VectorStructPersisted::Named(
                    [(
                        import.vector_name.clone(),
                        crate::operations::point_ops::VectorPersisted::Dense(vector),
                    )]
                    .into(),
                )
            };
            for shard in assignment.shards {
                if let Some((sender, batch)) = queues.get_mut(&shard) {
                    batch.push(PointStructPersisted {
                        id: ExtendedPointId::NumId(assignment.id),
                        vector: vector.clone(),
                        payload: None,
                    });
                    if batch.len() >= 512 {
                        sender.send(std::mem::take(batch)).await.map_err(io)?;
                    }
                }
            }
        }
        for (_, (sender, batch)) in queues {
            if !batch.is_empty() {
                sender.send(batch).await.map_err(io)?;
            }
        }
        while let Some(result) = workers.join_next().await {
            result.map_err(io)??;
        }
        Ok(())
    }

    async fn validate_orion_local(&self, generation: u64) -> CollectionResult<()> {
        let dir = generation_path(&self.path, generation);
        let artifact = super::migration::read_artifact(&self.path, generation)?;
        let local = self.orion_local_shard_ids().await;
        let members =
            super::membership::Catalog::load_for_shards(&self.path, generation, &local)?.entries;
        let state = read_state(&self.path)?;
        let retained = if let Some(active) = state
            .and_then(|s| s.active)
            .filter(|g| g.generation != generation)
        {
            Some(
                super::membership::Catalog::load_for_shards(&self.path, active.generation, &local)?
                    .entries,
            )
        } else {
            None
        };
        let holder = self.shards_holder.read().await;
        for (id, shard) in holder.get_shards() {
            if !shard.has_local_shard().await {
                continue;
            }
            let anchor_by_id: HashMap<_, _> = artifact
                .upper_nodes
                .iter()
                .filter(|node| node.owner_shard == id)
                .map(|node| (node.label, node))
                .collect();
            let anchors = artifact
                .upper_nodes
                .iter()
                .filter(|n| n.owner_shard == id)
                .map(|n| n.label)
                .collect();
            let _proof = check_index(
                shard
                    .orion_segment_view()
                    .await?
                    .ok_or_else(|| bad("missing local index"))?,
                artifact.vector_schema.vector_name.clone(),
                anchors,
                false,
            )
            .await?;
            let expected: BTreeMap<_, _> = members
                .iter()
                .filter(|(_, m)| m.shards.contains(&id))
                .map(|(id, m)| (*id, m))
                .collect();
            let info = shard.info(true).await?;
            let expected_indexed = expected
                .iter()
                .filter(|(point, m)| {
                    m.vector_present
                        || artifact
                            .upper_nodes
                            .iter()
                            .any(|n| n.label == **point && n.owner_shard == id)
                })
                .count();
            if info.status != CollectionStatus::Green
                || info.optimizer_status != OptimizersStatus::Ok
                || info.indexed_vectors_count.unwrap_or(0) < expected_indexed
            {
                return Err(bad(format!(
                    "shard {id} HNSW is not ready: status={:?}, optimizer={:?}, indexed={:?}, required={expected_indexed}",
                    info.status, info.optimizer_status, info.indexed_vectors_count
                )));
            }
            let mut found = std::collections::BTreeSet::new();
            let mut offset = None;
            loop {
                let mut rows = shard
                    .local_scroll_by_id(
                        offset,
                        513,
                        &WithPayloadInterface::Bool(false),
                        &WithVector::Bool(true),
                        None,
                        None,
                        None,
                        HwMeasurementAcc::disposable(),
                        DeferredBehavior::Exclude,
                    )
                    .await?;
                if rows.is_empty() {
                    break;
                }
                let next_offset = if rows.len() > 512 {
                    rows.pop().map(|row| row.id)
                } else {
                    None
                };
                for row in &rows {
                    if expected.contains_key(&row.id) {
                        found.insert(row.id);
                        if let Some(node) = anchor_by_id.get(&row.id) {
                            let Some(segment::data_types::vectors::VectorRef::Dense(vector)) =
                                row.get_vector_by_name(&artifact.vector_schema.vector_name)
                            else {
                                return Err(bad("missing L0 entry vector"));
                            };
                            let normalized = match artifact.vector_schema.datatype {
                                super::OrionVectorDatatype::Uint8 => artifact
                                    .vector_schema
                                    .distance
                                    .preprocess_vector::<u8>(node.vector.clone()),
                                _ => super::router::preprocess_vector(
                                    artifact.vector_schema.distance,
                                    node.vector.clone(),
                                ),
                            };
                            if !dir.join("migration.json").is_file()
                                && !vector
                                    .iter()
                                    .zip(&normalized)
                                    .all(|(a, b)| a.to_bits() == b.to_bits())
                            {
                                return Err(bad("L0 entry vector mismatch"));
                            }
                        }
                    } else if !retained
                        .as_ref()
                        .and_then(|m| m.get(&row.id))
                        .is_some_and(|m| m.shards.contains(&id))
                    {
                        return Err(bad(format!(
                            "unexpected physical point {} in shard {id}",
                            row.id
                        )));
                    }
                }
                offset = next_offset;
                if offset.is_none() {
                    break;
                }
            }
            if found.len() != expected.len() {
                return Err(bad(format!("shard {id} has missing physical copies")));
            }
        }
        Ok(())
    }

    pub(crate) async fn remove_orion_difference(
        &self,
        old: u64,
        retained: u64,
    ) -> CollectionResult<()> {
        let state = read_state(&self.path)?.ok_or_else(|| io("missing cleanup state"))?;
        let clock_generation = if state.pending_generation == Some(old) {
            old
        } else {
            retained
        };
        let old = super::membership::Catalog::load(&self.path, old)?.entries;
        let retained = super::membership::Catalog::load(&self.path, retained)?.entries;
        let holder = self.shards_holder.read().await;
        let tasks: Vec<_> = holder
            .get_shards()
            .map(|(id, shard)| {
                let ids: Vec<_> = old
                    .iter()
                    .filter(|(point, m)| {
                        m.shards.contains(&id)
                            && !retained.get(point).is_some_and(|m| m.shards.contains(&id))
                    })
                    .map(|(point, _)| *point)
                    .collect();
                async move {
                    if !shard.has_local_shard().await {
                        return Ok(());
                    }
                    for (sequence, batch) in ids.chunks(512).enumerate() {
                        let op = CollectionUpdateOperations::PointOperation(
                            PointOperations::DeletePoints {
                                ids: batch.to_vec(),
                            },
                        );
                        super::clock::apply(
                            shard,
                            super::clock::tagged(
                                op,
                                clock_generation,
                                0,
                                super::clock::STRIDE / 2 + sequence as u64,
                                &format!("epoch-{clock_generation}"),
                            )?,
                        )
                        .await?;
                    }
                    CollectionResult::Ok(())
                }
            })
            .collect();
        let results: Vec<_> = stream::iter(tasks).buffer_unordered(16).collect().await;
        for result in results {
            result?;
        }
        Ok(())
    }

    pub(crate) async fn stamp_orion_epoch(&self, generation: u64) -> CollectionResult<()> {
        for (_, shard) in self.shards_holder.read().await.get_shards() {
            if shard.has_local_shard().await {
                super::clock::fence(shard, generation, 0, &format!("epoch-{generation}")).await?;
            }
        }
        Ok(())
    }
}
