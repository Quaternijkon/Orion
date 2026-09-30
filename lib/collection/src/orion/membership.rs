//! Replicated membership transactions. Intent precedes shard WAL changes; an atomic,
//! checksummed event commits membership before obsolete copies are removed. Replay is idempotent.
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use common::counter::hardware_accumulator::HwMeasurementAcc;
use common::fs::atomic_save;
use futures::future::join_all;
use segment::data_types::vectors::VectorStructInternal;
use segment::types::{ExtendedPointId, Filter, WithPayloadInterface, WithVector};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use shard::operations::payload_ops::PayloadOps;
use shard::operations::vector_ops::VectorOperations;

use super::lifecycle::{self, CONTROL_DIR, Phase};
use crate::collection::Collection;
use crate::operations::CollectionUpdateOperations;
use crate::operations::point_ops::{
    PointInsertOperationsInternal, PointOperations, PointStructPersisted, VectorPersisted,
    VectorStructPersisted,
};
use crate::operations::shard_selector_internal::ShardSelectorInternal;
use crate::operations::types::{CollectionError, CollectionResult, PointRequestInternal};

fn bad(e: impl std::fmt::Display) -> CollectionError {
    CollectionError::bad_request(e.to_string())
}
fn io(e: impl std::fmt::Display) -> CollectionError {
    CollectionError::service_error(e.to_string())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Member {
    pub shards: Vec<u32>,
    pub live: bool,
    pub vector_present: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WritePlan {
    pub token: String,
    pub generation: u64,
    pub expected_revision: u64,
    #[serde(with = "shard_rows")]
    pub writes: BTreeMap<u32, Vec<CollectionUpdateOperations>>,
    #[serde(with = "member_rows")]
    pub changes: BTreeMap<ExtendedPointId, Member>,
    #[serde(with = "shard_rows")]
    pub obsolete: BTreeMap<u32, Vec<ExtendedPointId>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Event {
    token: String,
    generation: u64,
    revision: u64,
    #[serde(with = "member_rows")]
    changes: BTreeMap<ExtendedPointId, Member>,
    plan: WritePlan,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PendingWriteStatus {
    pub token: String,
    pub generation: u64,
    pub revision: u64,
    pub written_shards: BTreeSet<u32>,
}
#[derive(Serialize, Deserialize)]
struct PreparedData {
    peer_id: u64,
    token: String,
    written_shards: BTreeSet<u32>,
    was_serving: bool,
    was_data_ready: bool,
}
#[derive(Serialize, Deserialize)]
struct Envelope<T> {
    sha256: String,
    value: T,
}
#[derive(Serialize, Deserialize)]
struct Head {
    generation: u64,
    revision: u64,
    token: String,
}

pub struct Catalog {
    pub generation: u64,
    pub revision: u64,
    pub entries: BTreeMap<ExtendedPointId, Member>,
    pub last_token: Option<String>,
    pub shard_revisions: BTreeMap<u32, u64>,
}
#[derive(Serialize, Deserialize)]
struct Base {
    generation: u64,
    #[serde(with = "member_rows")]
    entries: BTreeMap<ExtendedPointId, Member>,
}

pub(crate) fn save_base(
    collection: &Path,
    generation: u64,
    entries: BTreeMap<ExtendedPointId, Member>,
) -> CollectionResult<()> {
    save(
        &lifecycle::generation_path(collection, generation).join("catalog-base.json"),
        &Base {
            generation,
            entries,
        },
    )
}
pub(crate) fn has_base(collection: &Path, generation: u64) -> bool {
    lifecycle::generation_path(collection, generation)
        .join("catalog-base.json")
        .is_file()
}

#[derive(Default, Clone)]
pub struct Visibility {
    pub deleted: BTreeSet<ExtendedPointId>,
    pub no_vector: BTreeSet<ExtendedPointId>,
}

fn journal(collection: &Path, generation: u64) -> PathBuf {
    collection
        .join(CONTROL_DIR)
        .join(format!("membership-{generation}"))
}
pub fn pending_path(collection: &Path) -> PathBuf {
    collection.join(CONTROL_DIR).join("pending-write.json")
}
pub(crate) fn pending_plan(collection: &Path) -> CollectionResult<Option<WritePlan>> {
    if pending_path(collection).is_file() {
        Ok(Some(load(&pending_path(collection))?))
    } else {
        Ok(None)
    }
}
fn prepared_path(collection: &Path) -> PathBuf {
    collection.join(CONTROL_DIR).join("pending-data.json")
}
pub fn saved_plan(
    collection: &Path,
    generation: u64,
    revision: u64,
) -> CollectionResult<WritePlan> {
    Ok(
        load::<Event>(&journal(collection, generation).join(format!("event-{revision:020}.json")))?
            .plan,
    )
}
pub(crate) fn pending_status(
    collection: &Path,
    peer_id: u64,
) -> CollectionResult<Option<PendingWriteStatus>> {
    let Some(plan) = pending_plan(collection)? else {
        return Ok(None);
    };
    let written_shards = if prepared_path(collection).is_file() {
        let data: PreparedData = load(&prepared_path(collection))?;
        if data.peer_id == peer_id && data.token == plan.token {
            data.written_shards
        } else {
            BTreeSet::new()
        }
    } else {
        BTreeSet::new()
    };
    Ok(Some(PendingWriteStatus {
        token: plan.token,
        generation: plan.generation,
        revision: plan.expected_revision + 1,
        written_shards,
    }))
}

fn save<T: Serialize>(path: &Path, value: &T) -> CollectionResult<()> {
    let bytes = serde_json::to_vec(&serde_json::to_value(value).map_err(io)?).map_err(io)?;
    let envelope = Envelope {
        sha256: format!("{:x}", Sha256::digest(&bytes)),
        value,
    };
    let bytes = serde_json::to_vec(&envelope).map_err(io)?;
    fs::create_dir_all(path.parent().unwrap()).map_err(io)?;
    atomic_save(path, |out| out.write_all(&bytes)).map_err(io)
}

fn load<T: for<'de> Deserialize<'de> + Serialize>(path: &Path) -> CollectionResult<T> {
    let envelope: Envelope<T> =
        serde_json::from_reader(File::open(path).map_err(io)?).map_err(io)?;
    if format!(
        "{:x}",
        Sha256::digest(
            serde_json::to_vec(&serde_json::to_value(&envelope.value).map_err(io)?).map_err(io)?
        )
    ) != envelope.sha256
    {
        return Err(io("membership WAL checksum mismatch"));
    }
    Ok(envelope.value)
}

impl Catalog {
    pub fn load(collection: &Path, generation: u64) -> CollectionResult<Self> {
        Self::load_filtered(collection, generation, None)
    }
    /// Read-only worker view. Committed WAL events still replay in full so clocks,
    /// moves into/out of local shards, tombstones and pending-tail checks are retained.
    pub(crate) fn load_for_shards(
        collection: &Path,
        generation: u64,
        shards: &BTreeSet<u32>,
    ) -> CollectionResult<Self> {
        Self::load_filtered(collection, generation, Some(shards))
    }
    fn load_filtered(
        collection: &Path,
        generation: u64,
        shards: Option<&BTreeSet<u32>>,
    ) -> CollectionResult<Self> {
        let base_path =
            lifecycle::generation_path(collection, generation).join("catalog-base.json");
        let entries = if base_path.is_file() {
            let base: Base = load(&base_path)?;
            if base.generation != generation {
                return Err(io("membership base generation mismatch"));
            }
            base.entries
        } else {
            lifecycle::memberships_for_shards(
                &lifecycle::generation_path(collection, generation),
                shards,
            )?
            .into_iter()
            .map(|(id, shards)| {
                (
                    id,
                    Member {
                        shards,
                        live: true,
                        vector_present: true,
                    },
                )
            })
            .collect()
        };
        let mut result = Self {
            generation,
            revision: 0,
            entries,
            last_token: None,
            shard_revisions: BTreeMap::new(),
        };
        let dir = journal(collection, generation);
        if dir.is_dir() {
            let mut paths: Vec<_> = fs::read_dir(&dir)
                .map_err(io)?
                .collect::<Result<Vec<_>, _>>()
                .map_err(io)?
                .into_iter()
                .map(|e| e.path())
                .filter(|p| {
                    p.file_name()
                        .is_some_and(|n| n.to_string_lossy().starts_with("event-"))
                })
                .collect();
            paths.sort();
            for path in paths {
                let event: Event = load(&path)?;
                if event.generation != generation || event.revision != result.revision + 1 {
                    return Err(io("membership WAL generation/revision gap"));
                }
                result.apply(event);
            }
            let head = dir.join("head.json");
            let revision = if head.is_file() {
                let head: Head = load(&head)?;
                if head.generation != generation || head.revision > result.revision {
                    return Err(io("membership WAL head mismatch"));
                }
                head.revision
            } else {
                0
            };
            if revision != result.revision {
                let pending: WritePlan = load(&pending_path(collection))?;
                if pending.generation != generation
                    || pending.expected_revision != revision
                    || result.revision != revision + 1
                    || result.last_token.as_deref() != Some(&pending.token)
                {
                    return Err(io("uncommitted membership WAL tail"));
                }
            }
        }
        if let Some(state) = lifecycle::read_state(collection)? {
            if let Some(active) = state
                .active
                .iter()
                .chain(state.previous.iter())
                .find(|g| g.generation == generation)
                && result.revision < active.membership_revision
            {
                return Err(io("membership WAL is incomplete"));
            }
        }
        Ok(result)
    }
    fn apply(&mut self, event: Event) {
        for (id, next) in &event.changes {
            for shard in next.shards.iter().chain(
                self.entries
                    .get(id)
                    .into_iter()
                    .flat_map(|m| m.shards.iter()),
            ) {
                self.shard_revisions.insert(*shard, event.revision);
            }
        }
        self.revision = event.revision;
        self.last_token = Some(event.token);
        self.entries.extend(event.changes);
    }
    pub fn visibility(&self) -> Visibility {
        Visibility {
            deleted: self
                .entries
                .iter()
                .filter(|(_, m)| !m.live)
                .map(|(id, _)| *id)
                .collect(),
            no_vector: self
                .entries
                .iter()
                .filter(|(_, m)| !m.vector_present)
                .map(|(id, _)| *id)
                .collect(),
        }
    }
}

pub fn load_visibility(collection: &Path) -> Visibility {
    let Ok(Some(state)) = lifecycle::read_state(collection) else {
        return Visibility::default();
    };
    state
        .active
        // Original bundle rows are all live and have vectors. Only committed events
        // (or a migrated base, which still loads in full) can add visibility masks.
        .and_then(|g| Catalog::load_for_shards(collection, g.generation, &BTreeSet::new()).ok())
        .map(|c| c.visibility())
        .unwrap_or_default()
}

/// Exact counts from a validated, activated, unchanged initial generation.
/// Dynamic/migrated generations retain the existing WAL/catalog path.
pub(crate) fn initial_shard_counts(
    collection: &Path,
    state: &lifecycle::ControlState,
) -> CollectionResult<Option<Vec<usize>>> {
    let Some(active) = &state.active else {
        return Ok(None);
    };
    let dir = lifecycle::generation_path(collection, active.generation);
    if active.membership_revision != 0
        || state.pending_generation.is_some()
        || state.previous.is_some()
        || state.phase != Phase::Ready
        || dir.join("migration.json").exists()
        || dir.join("catalog-base.json").exists()
        || pending_path(collection).exists()
    {
        return Ok(None);
    }
    let wal = journal(collection, active.generation);
    if wal.is_dir() && fs::read_dir(wal).map_err(io)?.next().is_some() {
        return Ok(None);
    }
    let Some(binding) = state.files.get("build-manifest.json") else {
        return Ok(None);
    };
    let path = dir.join("build-manifest.json");
    if fs::metadata(&path).map_err(io)?.len() != binding.size
        || super::bundle::file_sha256(&path).map_err(io)? != binding.sha256
    {
        return Err(io("initial counts build manifest checksum mismatch"));
    }
    let build: super::bundle::BuildManifest = serde_json::from_reader(
        std::io::BufReader::with_capacity(1024 * 1024, File::open(path).map_err(io)?),
    )
    .map_err(io)?;
    super::bundle::validate_filename(&build.import_manifest_file).map_err(io)?;
    let path = dir.join(&build.import_manifest_file);
    if super::bundle::file_sha256(&path).map_err(io)? != build.import_manifest_sha256 {
        return Err(io("initial counts import manifest checksum mismatch"));
    }
    let import: super::bundle::ImportManifest =
        serde_json::from_reader(File::open(path).map_err(io)?).map_err(io)?;
    if build.format_version != 2
        || build.algorithm_version != super::layout::LAYOUT_ALGORITHM_VERSION
        || build.options.generation != active.generation
        || import.orion_generation != active.generation
        || import.orion_artifact_sha256 != active.artifact_sha256
        || import.assignments_sha256 != active.layout_sha256
        || import.shard_count != active.shard_count
        || build.shard_counts.len() != active.shard_count as usize
        || build
            .shard_counts
            .iter()
            .try_fold(0_u64, |sum, &n| sum.checked_add(n))
            != Some(import.total_point_copies)
    {
        return Err(io("initial counts do not match activated generation"));
    }
    build
        .shard_counts
        .into_iter()
        .map(|n| usize::try_from(n).map_err(io))
        .collect::<CollectionResult<Vec<_>>>()
        .map(Some)
}

fn vectors(vector: VectorStructPersisted) -> CollectionResult<HashMap<String, VectorPersisted>> {
    Ok(match vector {
        VectorStructPersisted::Single(v) => [(String::new(), VectorPersisted::Dense(v))].into(),
        VectorStructPersisted::Named(v) => v,
        VectorStructPersisted::MultiDense(_) => {
            return Err(bad("Orion requires a dense routing vector"));
        }
    })
}

impl Collection {
    pub(crate) async fn orion_local_shard_ids(&self) -> BTreeSet<u32> {
        let holder = self.shards_holder.read().await;
        let mut ids = BTreeSet::new();
        for (id, shard) in holder.get_shards() {
            if shard.has_local_shard().await {
                ids.insert(id);
            }
        }
        ids
    }
    /// Repair shard data from the immutable generation and committed full WAL intents.
    /// The caller holds the collection control lock. Clock tags skip already-newer data.
    pub(crate) async fn repair_orion_data(&self) -> CollectionResult<()> {
        let state = lifecycle::read_state(&self.path)?.ok_or_else(|| io("missing repair state"))?;
        let Some(active) = state.active.clone() else {
            return Ok(());
        };
        let generation = active.generation;
        let local = self.orion_local_shard_ids().await;
        let path = self.path.clone();
        let (catalog_revision, shard_revisions) = tokio::task::spawn_blocking(move || {
            if initial_shard_counts(&path, &state)?.is_some() {
                return CollectionResult::Ok((0, BTreeMap::new()));
            }
            let catalog = Catalog::load_for_shards(&path, generation, &local)?;
            Ok((catalog.revision, catalog.shard_revisions))
        })
        .await
        .map_err(io)??;
        let holder = self.shards_holder.read().await;
        let mut missing_epoch = false;
        let mut behind = false;
        for (id, shard) in holder.get_shards() {
            if !shard.has_local_shard().await
                || shard.peer_state(self.this_peer_id)
                    != Some(crate::shards::replica_set::replica_set_state::ReplicaState::Active)
            {
                continue;
            }
            let actual = super::clock::observed(shard, active.generation).await?;
            let expected = super::clock::fence_tick(*shard_revisions.get(&id).unwrap_or(&0))?;
            if actual.is_some_and(|tick| tick > expected) {
                self.orion_refresh_needed
                    .store(true, std::sync::atomic::Ordering::Release);
                return Err(io("newer shard data requires a newer catalog"));
            }
            missing_epoch |= actual.is_none();
            behind |= actual != Some(expected);
        }
        drop(holder);
        if !behind {
            return Ok(());
        }
        self.orion_serving
            .store(false, std::sync::atomic::Ordering::Release);
        if missing_epoch {
            if lifecycle::generation_path(&self.path, active.generation)
                .join("migration.json")
                .is_file()
            {
                self.replay_orion_migration_data(active.generation).await?;
            } else {
                self.import_orion_local(active.generation).await?;
            }
            self.stamp_orion_epoch(active.generation).await?;
        }
        let holder = self.shards_holder.read().await;
        for revision in 1..=catalog_revision {
            let plan = saved_plan(&self.path, active.generation, revision)?;
            for (&id, ops) in &plan.writes {
                if let Some(shard) = holder.get_shard(id) {
                    if shard.has_local_shard().await {
                        for (sequence, op) in ops.iter().enumerate() {
                            super::clock::apply(
                                shard,
                                super::clock::tagged(
                                    op.clone(),
                                    active.generation,
                                    revision,
                                    sequence as u64 + 1,
                                    &plan.token,
                                )?,
                            )
                            .await?;
                        }
                    }
                }
            }
            for (&id, ids) in &plan.obsolete {
                if let Some(shard) = holder.get_shard(id) {
                    if shard.has_local_shard().await {
                        let op = CollectionUpdateOperations::PointOperation(
                            PointOperations::DeletePoints { ids: ids.clone() },
                        );
                        super::clock::apply(
                            shard,
                            super::clock::tagged(
                                op,
                                active.generation,
                                revision,
                                super::clock::STRIDE - 2,
                                &plan.token,
                            )?,
                        )
                        .await?;
                    }
                }
            }
            let touched: BTreeSet<_> = plan
                .writes
                .keys()
                .chain(plan.obsolete.keys())
                .copied()
                .chain(plan.changes.values().flat_map(|m| m.shards.iter().copied()))
                .collect();
            for id in touched {
                if let Some(shard) = holder.get_shard(id) {
                    if shard.has_local_shard().await {
                        super::clock::fence(shard, active.generation, revision, &plan.token)
                            .await?;
                    }
                }
            }
        }
        Ok(())
    }
    pub async fn orion_pending_plan(&self) -> CollectionResult<Option<WritePlan>> {
        pending_plan(&self.path)
    }
    pub fn orion_saved_plan(&self, generation: u64, revision: u64) -> CollectionResult<WritePlan> {
        saved_plan(&self.path, generation, revision)
    }

    pub async fn validate_orion_write_receipts(
        &self,
        plan: &WritePlan,
        states: &[Option<lifecycle::ControlState>],
    ) -> CollectionResult<()> {
        if states.iter().flatten().any(|s| {
            s.active.as_ref().is_some_and(|g| {
                g.generation == plan.generation
                    && g.membership_revision == plan.expected_revision + 1
                    && g.last_write_token.as_deref() == Some(&plan.token)
            })
        }) {
            return Ok(());
        }
        let targets: BTreeSet<_> = plan
            .writes
            .keys()
            .chain(plan.obsolete.keys())
            .copied()
            .chain(plan.changes.values().flat_map(|m| m.shards.iter().copied()))
            .collect();
        let required = self
            .collection_config
            .read()
            .await
            .params
            .write_consistency_factor
            .get() as usize;
        let holder = self.shards_holder.read().await;
        for shard_id in targets {
            let shard = holder
                .get_shard(shard_id)
                .ok_or_else(|| io("write target disappeared"))?;
            let count = states
                .iter()
                .flatten()
                .filter_map(|s| s.pending_write.as_ref())
                .filter(|p| {
                    p.token == plan.token
                        && p.generation == plan.generation
                        && p.written_shards.contains(&shard_id)
                })
                .count();
            if count < required.min(shard.peers().len()) {
                return Err(CollectionError::ShardUnavailable {
                    description: format!(
                        "Orion shard {shard_id} has insufficient durable write receipts"
                    ),
                });
            }
        }
        Ok(())
    }
    pub async fn orion_write_permit(&self) -> CollectionResult<tokio::sync::OwnedSemaphorePermit> {
        self.orion_write_budget
            .clone()
            .acquire_owned()
            .await
            .map_err(io)
    }
    pub(crate) fn check_orion_generation(
        &self,
        params: Option<&segment::types::SearchParams>,
    ) -> CollectionResult<()> {
        if let Some(expected) = params.and_then(|p| p.orion_generation) {
            if self.orion_router.load().as_ref().map(|r| r.generation()) != Some(expected) {
                return Err(io("Orion query generation mismatch"));
            }
        }
        Ok(())
    }
    pub async fn recover_orion_write(&self) -> CollectionResult<()> {
        if pending_path(&self.path).is_file() {
            let plan = load(&pending_path(&self.path))?;
            self.apply_orion_control(lifecycle::ControlCommand::Write { plan })
                .await?;
        }
        Ok(())
    }

    pub(crate) fn orion_search_filter(
        &self,
        filter: Option<Filter>,
        needs_vector: bool,
    ) -> Option<Filter> {
        let visibility = self.orion_visibility.load();
        let mut hidden = visibility.deleted.clone();
        if needs_vector {
            hidden.extend(&visibility.no_vector);
        }
        if hidden.is_empty() {
            return filter;
        }
        let exclusion = Filter::new_must_not(segment::types::Condition::HasId(
            segment::types::HasIdCondition {
                has_id: hidden.into_iter().collect(),
            },
        ));
        Some(filter.map_or_else(|| exclusion.clone(), |filter| filter.merge(&exclusion)))
    }

    pub(crate) fn orion_filter_query(
        &self,
        query: &mut shard::query::ShardQueryRequest,
    ) -> CollectionResult<()> {
        self.check_orion_generation(query.params.as_ref())?;
        let generation = if self.is_managed_orion() {
            self.orion_router.load().as_ref().map(|r| r.generation())
        } else {
            None
        };
        if let Some(generation) = generation {
            query
                .params
                .get_or_insert_with(Default::default)
                .orion_generation = Some(generation);
        }
        let needs_vector = matches!(
            query.query,
            Some(shard::query::ScoringQuery::Vector(_) | shard::query::ScoringQuery::Mmr(_))
        );
        query.filter = self.orion_search_filter(query.filter.take(), needs_vector);
        fn visit(
            collection: &Collection,
            prefetch: &mut shard::query::ShardPrefetch,
            generation: Option<u64>,
        ) -> CollectionResult<()> {
            collection.check_orion_generation(prefetch.params.as_ref())?;
            if let Some(generation) = generation {
                prefetch
                    .params
                    .get_or_insert_with(Default::default)
                    .orion_generation = Some(generation);
            }
            let needs_vector = matches!(
                prefetch.query,
                Some(shard::query::ScoringQuery::Vector(_) | shard::query::ScoringQuery::Mmr(_))
            );
            prefetch.filter = collection.orion_search_filter(prefetch.filter.take(), needs_vector);
            for child in &mut prefetch.prefetches {
                visit(collection, child, generation)?;
            }
            Ok(())
        }
        for child in &mut query.prefetches {
            visit(self, child, generation)?;
        }
        Ok(())
    }
    pub fn is_managed_orion(&self) -> bool {
        self.orion_managed
            .load(std::sync::atomic::Ordering::Acquire)
    }

    pub(crate) async fn ensure_orion_worker_ready(&self) -> CollectionResult<()> {
        self.ensure_orion_read_ready(true).await
    }
    pub(crate) async fn ensure_orion_data_ready(&self) -> CollectionResult<()> {
        self.ensure_orion_read_ready(false).await
    }
    async fn ensure_orion_read_ready(&self, indexes: bool) -> CollectionResult<()> {
        if !self.is_managed_orion()
            || (if indexes {
                self.orion_serving
                    .load(std::sync::atomic::Ordering::Acquire)
            } else {
                self.orion_data_ready
                    .load(std::sync::atomic::Ordering::Acquire)
            })
        {
            return Ok(());
        }
        let _guard = self
            .orion_control_lock
            .try_lock()
            .map_err(|_| io("Orion state transition is in progress"))?;
        if pending_path(&self.path).exists() {
            return Err(io("Orion write recovery is pending"));
        }
        let state =
            lifecycle::read_state(&self.path)?.ok_or_else(|| io("Orion control state missing"))?;
        if indexes && (state.phase != Phase::Ready || state.pending_generation.is_some()) {
            return Err(io("Orion generation is not ready"));
        }
        let generation = state
            .active
            .as_ref()
            .ok_or_else(|| io("Orion has no active generation"))?
            .generation;
        let router = self
            .orion_router
            .load_full()
            .ok_or_else(|| io("Orion router has not loaded"))?;
        if router.generation() != generation {
            return Err(io("Orion router generation mismatch"));
        }
        let local_shards = self.orion_local_shard_ids().await;
        let collection_path = self.path.clone();
        // Static generations have no member deltas. Their committed build manifest
        // binds exact per-shard counts; do not reload N memberships or the upper graph.
        let (catalog, initial_counts) = tokio::task::spawn_blocking(move || {
            let initial_counts = initial_shard_counts(&collection_path, &state)?;
            let catalog = if initial_counts.is_some() {
                None
            } else {
                Some(Catalog::load_for_shards(
                    &collection_path,
                    generation,
                    &local_shards,
                )?)
            };
            CollectionResult::Ok((catalog, initial_counts))
        })
        .await
        .map_err(io)??;
        let holder = self.shards_holder.read().await;
        for (id, shard) in holder.get_shards() {
            if !shard.has_local_shard().await {
                continue;
            }
            if shard.peer_state(self.this_peer_id)
                != Some(crate::shards::replica_set::replica_set_state::ReplicaState::Active)
            {
                continue;
            }
            let expected_clock = super::clock::fence_tick(
                catalog
                    .as_ref()
                    .and_then(|c| c.shard_revisions.get(&id))
                    .copied()
                    .unwrap_or(0),
            )?;
            let observed = super::clock::observed(shard, generation).await?;
            if observed != Some(expected_clock) {
                self.orion_refresh_needed
                    .store(true, std::sync::atomic::Ordering::Release);
                return Err(io(
                    "Orion catalog and shard WAL epochs are not synchronized",
                ));
            }
            if !indexes {
                continue;
            }
            let anchors = router.owner_entries(id);
            let proof = lifecycle::check_index(
                shard
                    .orion_segment_view()
                    .await?
                    .ok_or_else(|| io("missing local Orion index"))?,
                router.vector_name().to_owned(),
                anchors,
                false,
            )
            .await?;
            let (expected_points, expected) = if let Some(counts) = &initial_counts {
                let count = *counts
                    .get(id as usize)
                    .ok_or_else(|| io("shard absent from initial counts"))?;
                (count, count)
            } else {
                let catalog = catalog.as_ref().expect("dynamic generation has catalog");
                let points = catalog
                    .entries
                    .values()
                    .filter(|m| m.shards.contains(&id))
                    .count();
                let vectors = catalog
                    .entries
                    .iter()
                    .filter(|(point, m)| {
                        m.shards.contains(&id)
                            && (m.vector_present || router.is_owner_entry(**point, id))
                    })
                    .count();
                (points, vectors)
            };
            if proof.vectors != expected || proof.points != expected_points {
                return Err(io(
                    "Orion entry update is waiting for the complete local HNSW",
                ));
            }
        }
        self.orion_data_ready
            .store(true, std::sync::atomic::Ordering::Release);
        if indexes {
            self.orion_serving
                .store(true, std::sync::atomic::Ordering::Release);
        }
        Ok(())
    }

    async fn write_records(
        &self,
        catalog: &Catalog,
        ids: &[ExtendedPointId],
    ) -> CollectionResult<Vec<shard::retrieve::record_internal::RecordInternal>> {
        use crate::operations::consistency_params::{ReadConsistency, ReadConsistencyType};
        let mut by_shard: BTreeMap<u32, Vec<_>> = BTreeMap::new();
        for id in ids {
            if let Some(member) = catalog.entries.get(id).filter(|m| m.live) {
                by_shard.entry(member.shards[0]).or_default().push(*id);
            }
        }
        let holder = self.shards_holder.read().await;
        let mut records = Vec::new();
        for (id, ids) in by_shard {
            let shard = holder
                .get_shard(id)
                .ok_or_else(|| io("membership shard missing"))?;
            records.extend(
                shard
                    .retrieve(
                        Arc::new(PointRequestInternal {
                            ids,
                            with_payload: Some(WithPayloadInterface::Bool(true)),
                            with_vector: WithVector::Bool(true),
                        }),
                        &segment::types::WithPayload::from(&WithPayloadInterface::Bool(true)),
                        &WithVector::Bool(true),
                        Some(ReadConsistency::Type(ReadConsistencyType::All)),
                        None,
                        false,
                        HwMeasurementAcc::disposable(),
                    )
                    .await?,
            );
        }
        let router = self.orion_router.load();
        if let Some(router) = router.as_ref() {
            for record in &mut records {
                if catalog
                    .entries
                    .get(&record.id)
                    .is_some_and(|m| !m.vector_present)
                {
                    if let Some(VectorStructInternal::Named(vectors)) = &mut record.vector {
                        vectors.remove(router.vector_name());
                    } else if router.vector_name().is_empty() {
                        record.vector = Some(VectorStructInternal::Named(HashMap::new()));
                    }
                }
            }
        }
        Ok(records)
    }

    async fn selected_ids(&self, filter: Filter) -> CollectionResult<BTreeSet<ExtendedPointId>> {
        let mut ids = BTreeSet::new();
        let mut offset = None;
        loop {
            let request = shard::scroll::ScrollRequestInternal {
                offset,
                limit: Some(512),
                filter: Some(filter.clone()),
                with_payload: Some(WithPayloadInterface::Bool(false)),
                ..Default::default()
            };
            let result = self
                .scroll_by(
                    request,
                    None,
                    &ShardSelectorInternal::All,
                    None,
                    HwMeasurementAcc::disposable(),
                )
                .await?;
            ids.extend(result.points.into_iter().map(|p| p.id));
            offset = result.next_page_offset;
            if offset.is_none() {
                break;
            }
        }
        Ok(ids)
    }

    /// Produce a CAS plan from standard Qdrant operations. No client-supplied shard map is used.
    pub async fn prepare_orion_write(
        &self,
        mut operation: CollectionUpdateOperations,
    ) -> CollectionResult<WritePlan> {
        self.ensure_orion_data_ready().await?;
        let _guard = self.orion_control_lock.lock().await;
        let state = lifecycle::read_state(&self.path)?
            .ok_or_else(|| bad("Orion lifecycle is not installed"))?;
        if state.phase != Phase::Ready
            || state.pending_generation.is_some()
            || state.previous.is_some()
            || pending_path(&self.path).exists()
        {
            return Err(bad(
                "Orion writes are paused for generation migration or recovery",
            ));
        }
        let generation = state
            .active
            .ok_or_else(|| bad("no active Orion generation"))?
            .generation;
        let mut catalog_guard = self.orion_catalog.lock().await;
        if catalog_guard
            .as_ref()
            .is_none_or(|c| c.generation != generation)
        {
            *catalog_guard = Some(Catalog::load(&self.path, generation)?);
        }
        let catalog = catalog_guard.as_ref().unwrap();
        let revision = catalog.revision;
        if let CollectionUpdateOperations::PointOperation(
            PointOperations::UpsertPointsConditional(op),
        ) = operation.clone()
        {
            let matches = self.selected_ids(op.condition).await?;
            let points = op
                .points_op
                .into_point_vec()
                .into_iter()
                .filter(|point| {
                    let exists = catalog.entries.get(&point.id).is_some_and(|m| m.live);
                    match op.update_mode.unwrap_or_default() {
                        crate::operations::point_ops::UpdateMode::Upsert => {
                            !exists || matches.contains(&point.id)
                        }
                        crate::operations::point_ops::UpdateMode::InsertOnly => !exists,
                        crate::operations::point_ops::UpdateMode::UpdateOnly => {
                            exists && matches.contains(&point.id)
                        }
                    }
                })
                .collect();
            operation = CollectionUpdateOperations::PointOperation(PointOperations::UpsertPoints(
                PointInsertOperationsInternal::PointsList(points),
            ));
        }

        // Resolve filters using the engine, then freeze explicit IDs before consensus CAS.
        let filter = match &operation {
            CollectionUpdateOperations::PointOperation(PointOperations::DeletePointsByFilter(
                f,
            )) => Some(f.clone()),
            CollectionUpdateOperations::VectorOperation(
                VectorOperations::DeleteVectorsByFilter(f, _),
            ) => Some(f.clone()),
            CollectionUpdateOperations::VectorOperation(VectorOperations::UpdateVectors(op)) => {
                op.update_filter.clone()
            }
            CollectionUpdateOperations::PayloadOperation(
                PayloadOps::SetPayload(op) | PayloadOps::OverwritePayload(op),
            ) => op.filter.clone(),
            CollectionUpdateOperations::PayloadOperation(PayloadOps::DeletePayload(op)) => {
                op.filter.clone()
            }
            CollectionUpdateOperations::PayloadOperation(PayloadOps::ClearPayloadByFilter(f)) => {
                Some(f.clone())
            }
            _ => None,
        };
        let selected = if let Some(filter) = filter {
            Some(self.selected_ids(filter).await?)
        } else {
            None
        };
        if let Some(ids) = &selected {
            match &mut operation {
                CollectionUpdateOperations::PointOperation(
                    PointOperations::DeletePointsByFilter(_),
                ) => {
                    operation =
                        CollectionUpdateOperations::PointOperation(PointOperations::DeletePoints {
                            ids: ids.iter().copied().collect(),
                        })
                }
                CollectionUpdateOperations::VectorOperation(
                    VectorOperations::DeleteVectorsByFilter(_, names),
                ) => {
                    operation = CollectionUpdateOperations::VectorOperation(
                        VectorOperations::DeleteVectors(
                            ids.iter().copied().collect::<Vec<_>>().into(),
                            names.clone(),
                        ),
                    )
                }
                CollectionUpdateOperations::VectorOperation(VectorOperations::UpdateVectors(
                    op,
                )) => {
                    op.update_filter = None;
                    op.points.retain(|p| ids.contains(&p.id));
                }
                CollectionUpdateOperations::PayloadOperation(
                    PayloadOps::SetPayload(op) | PayloadOps::OverwritePayload(op),
                ) => {
                    op.points = Some(
                        op.points
                            .take()
                            .unwrap_or_else(|| ids.iter().copied().collect())
                            .into_iter()
                            .filter(|p| ids.contains(p))
                            .collect(),
                    );
                    op.filter = None;
                }
                CollectionUpdateOperations::PayloadOperation(PayloadOps::DeletePayload(op)) => {
                    op.points = Some(
                        op.points
                            .take()
                            .unwrap_or_else(|| ids.iter().copied().collect())
                            .into_iter()
                            .filter(|p| ids.contains(p))
                            .collect(),
                    );
                    op.filter = None;
                }
                CollectionUpdateOperations::PayloadOperation(PayloadOps::ClearPayloadByFilter(
                    _,
                )) => {
                    operation =
                        CollectionUpdateOperations::PayloadOperation(PayloadOps::ClearPayload {
                            points: ids.iter().copied().collect(),
                        })
                }
                _ => {}
            }
        }
        let ids = operation
            .point_ids()
            .ok_or_else(|| bad("operation requires explicit Orion point IDs"))?;
        let existing = self.write_records(catalog, &ids).await?;
        let mut records: BTreeMap<_, _> = existing.into_iter().map(|r| (r.id, r)).collect();
        let artifact = super::OrionRoutingArtifact::read_json(
            &super::routing_artifact_path(&self.path, generation),
            None,
        )
        .map_err(io)?;
        let mut search_artifact = artifact.clone();
        let options = super::migration::options(&self.path, generation)?;
        search_artifact.upper_k = options.efs.min(artifact.upper_nodes.len());
        search_artifact.upper_ef_search = options.efs;
        let router = super::OrionRouter::new(search_artifact).map_err(io)?;
        let labels: HashMap<_, _> = artifact
            .upper_nodes
            .iter()
            .enumerate()
            .map(|(i, n)| (n.label, i))
            .collect();
        let owners: Vec<_> = artifact.upper_nodes.iter().map(|n| n.owner_shard).collect();
        let anchors: HashMap<_, _> = artifact
            .upper_nodes
            .iter()
            .map(|n| (n.label, n.owner_shard))
            .collect();
        let mut plan = WritePlan {
            token: uuid::Uuid::new_v4().to_string(),
            generation,
            expected_revision: revision,
            writes: BTreeMap::new(),
            changes: BTreeMap::new(),
            obsolete: BTreeMap::new(),
        };
        let mut upserts = BTreeMap::new();
        match &operation {
            CollectionUpdateOperations::PointOperation(PointOperations::UpsertPoints(points)) => {
                for point in points.clone().into_point_vec() {
                    upserts.insert(point.id, point);
                }
            }
            CollectionUpdateOperations::VectorOperation(VectorOperations::UpdateVectors(op)) => {
                for point in &op.points {
                    let Some(old) = records.get(&point.id) else {
                        return Err(bad(format!("point {} not found", point.id)));
                    };
                    let mut all = vectors(
                        old.vector
                            .clone()
                            .unwrap_or(VectorStructInternal::Named(HashMap::new()))
                            .into(),
                    )?;
                    all.extend(vectors(point.vector.clone())?);
                    upserts.insert(
                        point.id,
                        PointStructPersisted {
                            id: point.id,
                            vector: VectorStructPersisted::Named(all),
                            payload: old.payload.clone(),
                        },
                    );
                }
            }
            _ => {}
        }
        for id in ids.into_iter().collect::<BTreeSet<_>>() {
            let previous = catalog.entries.get(&id).cloned().unwrap_or(Member {
                shards: vec![],
                live: false,
                vector_present: false,
            });
            let mut next = previous.clone();
            let mut point_op = operation.clone();
            point_op.retain_point_ids(|point| *point == id);
            if let Some(mut point) = upserts.remove(&id) {
                if point.payload.is_none() {
                    point.payload = records.remove(&id).and_then(|r| r.payload);
                }
                let all = vectors(point.vector.clone())?;
                if all.len() != 1 {
                    return Err(bad(
                        "Orion upsert requires only the configured routing vector",
                    ));
                }
                let Some(VectorPersisted::Dense(vector)) =
                    all.get(&artifact.vector_schema.vector_name)
                else {
                    return Err(bad("upsert needs the active dense routing vector"));
                };
                let hits = router.search_upper(vector).map_err(bad)?;
                let indices: Vec<_> = hits.iter().map(|h| labels[&h.label]).collect();
                let mut shards = super::layout::strict_targets(&indices, &owners).map_err(bad)?;
                if let Some(&owner) = anchors.get(&id)
                    && !shards.contains(&owner)
                {
                    shards.push(owner);
                }
                next = Member {
                    shards,
                    live: true,
                    vector_present: true,
                };
                point_op =
                    CollectionUpdateOperations::PointOperation(PointOperations::UpsertPoints(
                        PointInsertOperationsInternal::PointsList(vec![point]),
                    ));
            } else if matches!(
                &operation,
                CollectionUpdateOperations::PointOperation(PointOperations::DeletePoints { .. })
            ) {
                next = Member {
                    shards: anchors.get(&id).copied().into_iter().collect(),
                    live: false,
                    vector_present: false,
                };
                // The immutable upper label remains a physical navigation anchor, hidden from all ordinary reads.
                point_op = CollectionUpdateOperations::PayloadOperation(PayloadOps::ClearPayload {
                    points: vec![id],
                });
            } else if let CollectionUpdateOperations::VectorOperation(
                VectorOperations::DeleteVectors(_, names),
            ) = &operation
            {
                if names.contains(&artifact.vector_schema.vector_name) {
                    next.vector_present = false;
                }
            } else if !previous.live {
                continue;
            }
            for &shard in &next.shards {
                let mut op = point_op.clone();
                if anchors.get(&id) == Some(&shard)
                    && let CollectionUpdateOperations::VectorOperation(
                        VectorOperations::DeleteVectors(_, names),
                    ) = &mut op
                {
                    names.retain(|name| name != &artifact.vector_schema.vector_name);
                    if names.is_empty() {
                        continue;
                    }
                }
                plan.writes.entry(shard).or_default().push(op);
            }
            for &shard in &previous.shards {
                if !next.shards.contains(&shard) {
                    plan.obsolete.entry(shard).or_default().push(id);
                }
            }
            plan.changes.insert(id, next);
        }
        Ok(plan)
    }

    /// Invoked in consensus order. A failed local shard update leaves its durable intent
    /// and disables reads until the same command is replayed after recovery.
    pub(crate) async fn apply_orion_write(&self, plan: WritePlan) -> CollectionResult<()> {
        self.execute_orion_write(plan, false).await
    }
    pub(crate) async fn commit_orion_write(&self, plan: WritePlan) -> CollectionResult<()> {
        self.execute_orion_write(plan, true).await
    }
    async fn execute_orion_write(&self, plan: WritePlan, commit: bool) -> CollectionResult<()> {
        let mut state =
            lifecycle::read_state(&self.path)?.ok_or_else(|| bad("Orion state missing"))?;
        if state.phase != Phase::Ready
            || state.pending_generation.is_some()
            || state.active.as_ref().map(|g| g.generation) != Some(plan.generation)
        {
            return Err(bad("Orion write generation is not active"));
        }
        let mut guard = self.orion_catalog.lock().await;
        if guard
            .as_ref()
            .is_none_or(|c| c.generation != plan.generation)
        {
            *guard = Some(Catalog::load(&self.path, plan.generation)?);
        }
        let catalog = guard.as_mut().unwrap();
        let was_serving = self
            .orion_serving
            .load(std::sync::atomic::Ordering::Acquire);
        let was_data_ready = self
            .orion_data_ready
            .load(std::sync::atomic::Ordering::Acquire);
        let replay = catalog.revision > plan.expected_revision
            && saved_plan(&self.path, plan.generation, plan.expected_revision + 1)
                .is_ok_and(|saved| saved.token == plan.token);
        if catalog.revision != plan.expected_revision && !replay {
            return Err(bad("Orion membership revision conflict; retry operation"));
        }
        if replay && !commit {
            return Ok(());
        }
        if pending_path(&self.path).exists() {
            let pending: WritePlan = load(&pending_path(&self.path))?;
            if pending.token != plan.token {
                if replay {
                    return Ok(());
                }
                return Err(bad(
                    "Orion membership revision conflict: another write needs recovery",
                ));
            }
        }
        save(&pending_path(&self.path), &plan)?;
        self.orion_serving
            .store(false, std::sync::atomic::Ordering::Release);
        self.orion_data_ready
            .store(false, std::sync::atomic::Ordering::Release);
        let holder = self.shards_holder.read().await;
        let mut tasks = Vec::new();
        let revision = plan.expected_revision + 1;
        let touched: BTreeSet<_> = plan
            .changes
            .iter()
            .flat_map(|(id, m)| {
                m.shards.iter().chain(
                    catalog
                        .entries
                        .get(id)
                        .into_iter()
                        .flat_map(|m| m.shards.iter()),
                )
            })
            .copied()
            .collect();
        let required = self
            .collection_config
            .read()
            .await
            .params
            .write_consistency_factor
            .get() as usize;
        for id in &touched {
            let shard = holder
                .get_shard(*id)
                .ok_or_else(|| io("missing write target"))?;
            let peers = shard.peers();
            let active = peers
                .values()
                .filter(|state| {
                    matches!(
                        state,
                        crate::shards::replica_set::replica_set_state::ReplicaState::Active
                            | crate::shards::replica_set::replica_set_state::ReplicaState::Listener
                    )
                })
                .count();
            if active < required.min(peers.len()) {
                return Err(CollectionError::ShardUnavailable {
                    description: format!("Orion write shard {id} lacks active replicas"),
                });
            }
        }
        let mut data_ahead = false;
        for (&id, operations) in &plan.writes {
            let shard = holder
                .get_shard(id)
                .ok_or_else(|| io("write shard missing"))?;
            let token = &plan.token;
            let generation = plan.generation;
            tasks.push(async move {
                if !shard.has_local_shard().await {
                    return Ok(());
                }
                for (sequence, op) in operations.iter().enumerate() {
                    super::clock::apply(
                        shard,
                        super::clock::tagged(
                            op.clone(),
                            generation,
                            revision,
                            sequence as u64 + 1,
                            token,
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
        if !commit {
            let was_data_ready = was_data_ready
                || if prepared_path(&self.path).is_file() {
                    let data: PreparedData = load(&prepared_path(&self.path))?;
                    data.peer_id == self.this_peer_id
                        && data.token == plan.token
                        && data.was_data_ready
                } else {
                    false
                };
            let was_serving = was_serving
                || if prepared_path(&self.path).is_file() {
                    let data: PreparedData = load(&prepared_path(&self.path))?;
                    data.peer_id == self.this_peer_id
                        && data.token == plan.token
                        && data.was_serving
                } else {
                    false
                };
            let mut written_shards = BTreeSet::new();
            for id in &touched {
                let shard = holder.get_shard(*id).unwrap();
                if shard.has_local_shard().await&&matches!(shard.peer_state(self.this_peer_id),Some(crate::shards::replica_set::replica_set_state::ReplicaState::Active|crate::shards::replica_set::replica_set_state::ReplicaState::Listener)){written_shards.insert(*id);}
            }
            save(
                &prepared_path(&self.path),
                &PreparedData {
                    peer_id: self.this_peer_id,
                    token: plan.token.clone(),
                    written_shards,
                    was_serving,
                    was_data_ready,
                },
            )?;
            return Ok(());
        }
        if !replay {
            let event = Event {
                token: plan.token.clone(),
                generation: plan.generation,
                revision: plan.expected_revision + 1,
                changes: plan.changes.clone(),
                plan: plan.clone(),
            };
            save(
                &journal(&self.path, plan.generation)
                    .join(format!("event-{:020}.json", event.revision)),
                &event,
            )?;
            catalog.apply(event);
        }
        save(
            &journal(&self.path, plan.generation).join("head.json"),
            &Head {
                generation: plan.generation,
                revision: catalog.revision,
                token: catalog
                    .last_token
                    .clone()
                    .unwrap_or_else(|| plan.token.clone()),
            },
        )?;
        state.active.as_mut().unwrap().membership_revision = catalog.revision;
        state.active.as_mut().unwrap().last_write_token = catalog.last_token.clone();
        lifecycle::save_state(&self.path, &state)?;
        let changes_index =
            plan.writes.values().any(|ops| !ops.is_empty()) || !plan.obsolete.is_empty();
        for (id, ids) in plan.obsolete {
            let shard = holder
                .get_shard(id)
                .ok_or_else(|| io("obsolete shard missing"))?;
            if shard.has_local_shard().await {
                let op =
                    CollectionUpdateOperations::PointOperation(PointOperations::DeletePoints {
                        ids,
                    });
                data_ahead |= super::clock::apply(
                    shard,
                    super::clock::tagged(
                        op,
                        plan.generation,
                        revision,
                        super::clock::STRIDE - 2,
                        &plan.token,
                    )?,
                )
                .await?;
            }
        }
        for id in touched {
            let shard = holder
                .get_shard(id)
                .ok_or_else(|| io("fenced shard missing"))?;
            if shard.has_local_shard().await {
                data_ahead |=
                    super::clock::fence(shard, plan.generation, revision, &plan.token).await?;
            }
        }
        self.orion_visibility.store(Arc::new(catalog.visibility()));
        let was_serving = if prepared_path(&self.path).is_file() {
            let data: PreparedData = load(&prepared_path(&self.path))?;
            data.peer_id == self.this_peer_id && data.token == plan.token && data.was_serving
        } else {
            was_serving
        };
        let was_data_ready = if prepared_path(&self.path).is_file() {
            let data: PreparedData = load(&prepared_path(&self.path))?;
            data.peer_id == self.this_peer_id && data.token == plan.token && data.was_data_ready
        } else {
            was_data_ready
        };
        fs::remove_file(pending_path(&self.path)).map_err(io)?;
        if prepared_path(&self.path).is_file() {
            fs::remove_file(prepared_path(&self.path)).map_err(io)?;
        }
        File::open(self.path.join(CONTROL_DIR))
            .map_err(io)?
            .sync_all()
            .map_err(io)?;
        // Qdrant applies payload mutations through copy-on-write too: a payload-only
        // update can move its vector out of the immutable HNSW into an appendable segment.
        self.orion_data_ready.store(
            was_data_ready && !data_ahead,
            std::sync::atomic::Ordering::Release,
        );
        self.orion_serving.store(
            was_serving && !changes_index && !data_ahead,
            std::sync::atomic::Ordering::Release,
        );
        Ok(())
    }
}

mod member_rows {
    use super::*;
    pub fn serialize<S: serde::Serializer>(
        value: &BTreeMap<ExtendedPointId, Member>,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        value.iter().collect::<Vec<_>>().serialize(serializer)
    }
    pub fn deserialize<'de, D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> Result<BTreeMap<ExtendedPointId, Member>, D::Error> {
        let rows = Vec::<(ExtendedPointId, Member)>::deserialize(deserializer)?;
        let size = rows.len();
        let map: BTreeMap<_, _> = rows.into_iter().collect();
        if map.len() != size {
            return Err(serde::de::Error::custom("duplicate membership ID"));
        }
        Ok(map)
    }
}

mod shard_rows {
    use super::*;
    pub fn serialize<T: Serialize, S: serde::Serializer>(
        value: &BTreeMap<u32, T>,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        value.iter().collect::<Vec<_>>().serialize(serializer)
    }
    pub fn deserialize<'de, T: Deserialize<'de>, D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> Result<BTreeMap<u32, T>, D::Error> {
        let rows = Vec::<(u32, T)>::deserialize(deserializer)?;
        let len = rows.len();
        let result: BTreeMap<_, _> = rows.into_iter().collect();
        if result.len() != len {
            return Err(serde::de::Error::custom("duplicate shard ID"));
        }
        Ok(result)
    }
}
