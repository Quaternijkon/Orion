//! Bounded initial-build ingestion. Rows may reach local HNSW optimizers before the full
//! layout exists; the independently verified final bundle remains the activation authority.
use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::Ordering;

use common::counter::hardware_accumulator::HwMeasurementAcc;
use common::fs::atomic_save;
use common::types::DeferredBehavior;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::bundle::{self, Assignment, BuildManifest, BuildRequest, ImportManifest};
use super::layout::LayoutOptions;
use super::lifecycle::{ControlState, FileBinding, Phase, bad, generation_path, io, save_state};
use crate::collection::Collection;
use crate::config::AutoShardPolicy;
use crate::operations::CollectionUpdateOperations;
use crate::operations::point_ops::{
    PointInsertOperationsInternal, PointOperations, PointStructPersisted, VectorPersisted,
    VectorStructPersisted,
};
use crate::operations::types::{CollectionResult, Datatype};
use crate::shards::replica_set::replica_set_state::ReplicaState;

pub const MAX_BATCH_BYTES: usize = 64 * 1024;
pub const MAX_BATCH_ROWS: usize = 128;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StreamSpec {
    #[serde(
        default,
        skip_serializing_if = "super::layout::VectorEncoding::is_default"
    )]
    pub vector_encoding: super::layout::VectorEncoding,
    pub options: LayoutOptions,
    pub point_count: usize,
    pub dimension: usize,
    pub raw_vectors_sha256: String,
}

impl From<&BuildRequest> for StreamSpec {
    fn from(request: &BuildRequest) -> Self {
        Self {
            vector_encoding: request.input.encoding.normalized(),
            options: request.options.clone(),
            point_count: request.input.rows,
            dimension: request.input.dimension,
            raw_vectors_sha256: request.input.sha256.clone(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StreamRow {
    pub assignment: Assignment,
    pub vector: Vec<f32>,
}

#[derive(Serialize, Deserialize)]
struct Progress {
    spec: StreamSpec,
    next_sequence: u64,
    next_row: u64,
}

fn directory(collection: &Path, generation: u64) -> PathBuf {
    generation_path(collection, generation).join("stream")
}

fn read_progress(collection: &Path, generation: u64) -> CollectionResult<Option<Progress>> {
    let path = directory(collection, generation).join("progress.json");
    match File::open(path) {
        Ok(file) => serde_json::from_reader(file).map(Some).map_err(io),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(io(error)),
    }
}

fn save_progress(collection: &Path, generation: u64, progress: &Progress) -> CollectionResult<()> {
    let path = directory(collection, generation).join("progress.json");
    let bytes = serde_json::to_vec(progress).map_err(io)?;
    atomic_save(&path, |out| out.write_all(&bytes)).map_err(io)
}

fn batch_path(collection: &Path, generation: u64, sequence: u64) -> PathBuf {
    directory(collection, generation).join(format!("batch-{sequence:020}.json"))
}

fn validate_rows(spec: &StreamSpec, start: u64, rows: &[StreamRow]) -> CollectionResult<()> {
    let end = start
        .checked_add(rows.len() as u64)
        .ok_or_else(|| bad("stream row overflow"))?;
    if rows.is_empty() || rows.len() > MAX_BATCH_ROWS || end > spec.point_count as u64 {
        return Err(bad("invalid stream batch size or row range"));
    }
    for (offset, row) in rows.iter().enumerate() {
        if row.assignment.id != start + offset as u64
            || row.vector.len() != spec.dimension
            || row.vector.iter().any(|x| !x.is_finite())
        {
            return Err(bad(
                "stream rows must have consecutive IDs and finite vectors of the declared dimension",
            ));
        }
        spec.vector_encoding.encode(&row.vector).map_err(bad)?;
        row.assignment
            .canonical_bytes(spec.options.shard_count)
            .map_err(bad)?;
    }
    if serde_json::to_vec(rows).map_err(io)?.len() > MAX_BATCH_BYTES {
        return Err(bad("stream batch exceeds byte limit"));
    }
    Ok(())
}

/// Transition from row ingestion to the existing checksummed file upload protocol.
pub(crate) fn bind_files(
    collection: &Path,
    state: &mut ControlState,
    generation: u64,
    files: &BTreeMap<String, FileBinding>,
) -> CollectionResult<bool> {
    if state.pending_generation != Some(generation) || !state.files.is_empty() {
        return Ok(false);
    }
    let Some(progress) = read_progress(collection, generation)? else {
        return Ok(false);
    };
    if state.phase != Phase::BuildingLayout || progress.next_row != progress.spec.point_count as u64
    {
        return Err(bad(
            "finish every stream batch before binding the final bundle",
        ));
    }
    for (name, binding) in files {
        bundle::validate_filename(name).map_err(bad)?;
        if binding.sha256.len() != 64 || !binding.sha256.bytes().all(|b| b.is_ascii_hexdigit()) {
            return Err(bad("invalid file checksum"));
        }
    }
    // Reconstruct the two large canonical files from acknowledged batches locally. The client
    // uploads only the remaining artifacts, avoiding a second network transfer of all vectors.
    let dir = generation_path(collection, generation);
    let mut vectors = tempfile::NamedTempFile::new_in(&dir).map_err(io)?;
    let mut assignments = tempfile::NamedTempFile::new_in(&dir).map_err(io)?;
    {
        let mut vector_writer = BufWriter::new(vectors.as_file_mut());
        let mut assignment_writer = BufWriter::new(assignments.as_file_mut());
        let mut count = 0;
        for sequence in 1..progress.next_sequence {
            let rows: Vec<StreamRow> = serde_json::from_reader(
                File::open(batch_path(collection, generation, sequence)).map_err(io)?,
            )
            .map_err(io)?;
            validate_rows(&progress.spec, count, &rows)?;
            for row in rows {
                assignment_writer
                    .write_all(
                        &row.assignment
                            .canonical_bytes(progress.spec.options.shard_count)
                            .map_err(bad)?,
                    )
                    .map_err(io)?;
                vector_writer
                    .write_all(
                        &progress
                            .spec
                            .vector_encoding
                            .encode(&row.vector)
                            .map_err(bad)?,
                    )
                    .map_err(io)?;
                count += 1;
            }
        }
        if count != progress.next_row {
            return Err(bad("incomplete acknowledged stream"));
        }
        vector_writer.flush().map_err(io)?;
        assignment_writer.flush().map_err(io)?;
    }
    for (file, name) in [
        (
            vectors,
            format!(
                "{}.{}",
                progress.spec.options.bundle_prefix,
                progress.spec.vector_encoding.extension()
            ),
        ),
        (
            assignments,
            format!("{}.assignments.jsonl", progress.spec.options.bundle_prefix),
        ),
    ] {
        let binding = files
            .get(&name)
            .ok_or_else(|| bad("final bundle omitted streamed data"))?;
        if file.as_file().metadata().map_err(io)?.len() != binding.size
            || bundle::file_sha256(file.path()).map_err(io)? != binding.sha256
        {
            return Err(bad("streamed data differs from the final file binding"));
        }
        file.as_file().sync_all().map_err(io)?;
        file.persist(dir.join(name)).map_err(io)?;
    }
    File::open(&dir).map_err(io)?.sync_all().map_err(io)?;
    state.files = files.clone();
    state.phase = Phase::Collecting;
    Ok(true)
}

/// Bind the acknowledged row stream to both final file digests. This runs only after
/// bundle::verify has independently replayed clustering and every assignment.
pub(crate) fn verify_import(
    collection: &Path,
    generation: u64,
    build: &BuildManifest,
    import: &ImportManifest,
) -> CollectionResult<bool> {
    let Some(progress) = read_progress(collection, generation)? else {
        return Ok(false);
    };
    let spec = &progress.spec;
    if progress.next_row != spec.point_count as u64
        || spec.point_count != import.point_count
        || spec.dimension != import.dimension
        || spec.vector_encoding != import.vector_encoding
        || spec.raw_vectors_sha256 != build.raw_vectors_sha256
        || serde_json::to_vec(&spec.options).map_err(io)?
            != serde_json::to_vec(&build.options).map_err(io)?
    {
        return Err(bad("final bundle differs from the streamed build"));
    }
    let mut assignments = Sha256::new();
    let mut vectors = Sha256::new();
    let mut count = 0;
    for sequence in 1..progress.next_sequence {
        let rows: Vec<StreamRow> = serde_json::from_reader(
            File::open(batch_path(collection, generation, sequence)).map_err(io)?,
        )
        .map_err(io)?;
        validate_rows(spec, count, &rows)?;
        for row in rows {
            assignments.update(
                row.assignment
                    .canonical_bytes(spec.options.shard_count)
                    .map_err(bad)?,
            );
            vectors.update(spec.vector_encoding.encode(&row.vector).map_err(bad)?);
            count += 1;
        }
    }
    if count != progress.next_row
        || format!("{:x}", assignments.finalize()) != import.assignments_sha256
        || format!("{:x}", vectors.finalize()) != import.vectors_sha256
    {
        return Err(bad(
            "streamed assignments or vectors differ from the final bundle",
        ));
    }
    Ok(true)
}

impl Collection {
    pub(crate) async fn begin_orion_stream(
        &self,
        state: &mut ControlState,
        spec: StreamSpec,
    ) -> CollectionResult<()> {
        spec.options.validate(spec.point_count).map_err(bad)?;
        if spec.vector_encoding == super::layout::VectorEncoding::U8bin
            || spec.dimension == 0
            || spec.raw_vectors_sha256.len() != 64
            || !spec
                .raw_vectors_sha256
                .bytes()
                .all(|b| b.is_ascii_hexdigit())
        {
            return Err(bad("invalid stream input schema or checksum"));
        }
        let generation = spec.options.generation;
        if state
            .active
            .as_ref()
            .is_some_and(|active| active.generation == generation)
            && let Some(progress) = read_progress(&self.path, generation)?
            && serde_json::to_vec(&progress.spec).map_err(io)?
                == serde_json::to_vec(&spec).map_err(io)?
        {
            return Ok(());
        }
        if state.active.is_some() {
            return Err(bad(
                "streaming builds require a fresh collection; use install or repartition for an active generation",
            ));
        }
        let existing = read_progress(&self.path, generation)?;
        if let Some(progress) = &existing {
            if serde_json::to_vec(&progress.spec).map_err(io)?
                != serde_json::to_vec(&spec).map_err(io)?
                || (state.pending_generation.is_some()
                    && state.pending_generation != Some(generation))
                || (state.pending_generation.is_none() && state.highest_generation >= generation)
            {
                return Err(bad(
                    "stream retry differs from its original build or was rolled back",
                ));
            }
        } else if state.pending_generation.is_some() || generation <= state.highest_generation {
            return Err(bad(
                "another generation is pending or this generation was already used",
            ));
        }
        let config = self.collection_config.read().await;
        if !matches!(config.auto_shard_policy, Some(AutoShardPolicy::Orion { generation: g, .. }) if g == generation)
        {
            return Err(bad(
                "create an Orion collection with the stream generation first",
            ));
        }
        let params = config
            .params
            .vectors
            .get_params(&spec.options.vector_name)
            .ok_or_else(|| bad("unknown stream vector"))?;
        if params.size.get() as usize != spec.dimension
            || params.distance != spec.options.distance
            || params.datatype.unwrap_or_default()
                != match spec.vector_encoding.datatype() {
                    super::OrionVectorDatatype::Uint8 => Datatype::Uint8,
                    _ => Datatype::Float32,
                }
            || params.multivector_config.is_some()
            || config.params.vectors.params_iter().count() != 1
            || config.hnsw_config.m == 0
        {
            return Err(bad(
                "streaming requires a matching single float32/uint8 dense vector and enabled HNSW",
            ));
        }
        let holder = self.shards_holder.read().await;
        if holder.len() != spec.options.shard_count as usize {
            return Err(bad("stream shard count differs from the collection"));
        }
        if existing.is_none() {
            for (_, shard) in holder.get_shards() {
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
                if count.is_some_and(|c| c.count != 0) {
                    return Err(bad("initial streaming build requires empty shards"));
                }
            }
        }
        drop(holder);
        let mut optimizers = config.optimizer_config.clone();
        optimizers.default_segment_number = 1;
        if optimizers.indexing_threshold == Some(0) {
            optimizers.indexing_threshold = Some(1);
        }
        if optimizers.max_optimization_threads == Some(0) {
            optimizers.max_optimization_threads = Some(1);
        }
        drop(config);
        if existing.is_none() {
            fs::create_dir_all(directory(&self.path, generation)).map_err(io)?;
            save_progress(
                &self.path,
                generation,
                &Progress {
                    spec,
                    next_sequence: 1,
                    next_row: 0,
                },
            )?;
        }
        if state.pending_generation.is_none() {
            state.pending_generation = Some(generation);
            state.highest_generation = generation;
            state.phase = Phase::BuildingLayout;
            state.files.clear();
        }
        self.orion_managed.store(true, Ordering::Release);
        self.orion_data_ready.store(false, Ordering::Release);
        self.orion_serving.store(false, Ordering::Release);
        save_state(&self.path, state)?;
        self.update_optimizer_params(optimizers).await?;
        self.recreate_optimizers_blocking().await?;
        Ok(())
    }

    pub(crate) async fn import_orion_stream_batch(
        &self,
        state: &ControlState,
        generation: u64,
        sequence: u64,
        rows: Vec<StreamRow>,
    ) -> CollectionResult<()> {
        if (state.pending_generation != Some(generation)
            && !state
                .active
                .as_ref()
                .is_some_and(|active| active.generation == generation))
            || sequence == 0
            || sequence >= super::clock::STRIDE
        {
            return Err(bad("invalid streaming generation or sequence"));
        }
        let mut progress =
            read_progress(&self.path, generation)?.ok_or_else(|| bad("no streaming build"))?;
        let start = rows
            .first()
            .ok_or_else(|| bad("empty stream batch"))?
            .assignment
            .id;
        validate_rows(&progress.spec, start, &rows)?;
        let path = batch_path(&self.path, generation, sequence);
        let bytes = serde_json::to_vec(&rows).map_err(io)?;
        if path.exists() && fs::read(&path).map_err(io)? != bytes {
            return Err(bad("retried stream batch has different content"));
        }
        if sequence < progress.next_sequence {
            if !path.is_file() {
                return Err(io("missing acknowledged stream batch"));
            }
            return Ok(());
        }
        if state.pending_generation != Some(generation)
            || sequence != progress.next_sequence
            || start != progress.next_row
            || state.phase != Phase::BuildingLayout
            || !state.files.is_empty()
        {
            return Err(bad("out of order stream batch or sealed stream"));
        }
        // Durable intent precedes WAL updates. A retry after any partial local apply uses the
        // same clock and bytes, including after the process has restarted.
        atomic_save(&path, |out| out.write_all(&bytes)).map_err(io)?;
        let mut batches: BTreeMap<u32, Vec<PointStructPersisted>> = BTreeMap::new();
        for row in &rows {
            let vector = if progress.spec.options.vector_name.is_empty() {
                VectorStructPersisted::Single(row.vector.clone())
            } else {
                VectorStructPersisted::Named(
                    [(
                        progress.spec.options.vector_name.clone(),
                        VectorPersisted::Dense(row.vector.clone()),
                    )]
                    .into(),
                )
            };
            for &shard in &row.assignment.shards {
                batches
                    .entry(shard)
                    .or_default()
                    .push(PointStructPersisted {
                        id: row.assignment.id.into(),
                        vector: vector.clone(),
                        payload: None,
                    });
            }
        }
        let holder = self.shards_holder.read().await;
        let mut operations = Vec::new();
        for (id, points) in batches {
            let shard = holder
                .get_shard(id)
                .ok_or_else(|| bad("stream shard disappeared"))?;
            if !shard.has_local_shard().await {
                continue;
            }
            if !matches!(
                shard.peer_state(self.this_peer_id),
                Some(ReplicaState::Active | ReplicaState::Listener)
            ) {
                return Err(bad("wait for the local stream replica to become active"));
            }
            let operation = CollectionUpdateOperations::PointOperation(
                PointOperations::UpsertPoints(PointInsertOperationsInternal::PointsList(points)),
            );
            operations.push(super::clock::apply(
                shard,
                super::clock::tagged(
                    operation,
                    generation,
                    0,
                    sequence,
                    &format!("epoch-{generation}"),
                )?,
            ));
        }
        futures::future::try_join_all(operations).await?;
        progress.next_sequence += 1;
        progress.next_row += rows.len() as u64;
        save_progress(&self.path, generation, &progress)
    }
}
