use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::mem;
use std::ops::Range;

use ahash::{AHashMap, AHashSet};
use ordered_float::OrderedFloat;
use segment::data_types::vectors::VectorElementType;
use segment::spaces::metric::Metric;
use segment::spaces::simple::{CosineMetric, DotProductMetric, EuclidMetric, ManhattanMetric};
use segment::types::{Distance, ExtendedPointId};

use super::artifact::{OrionRoutingArtifact, OrionUpperHnswGraph};
use super::error::{OrionRoutingError, OrionRoutingResult};
use crate::shards::shard::ShardId;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OrionUpperHit {
    pub label: ExtendedPointId,
    /// A lower-is-better distance used only for upper-result ordering.
    pub distance: f32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OrionShardTarget {
    pub shard_id: ShardId,
    /// Ordered, unique external point IDs to use as lower HNSW entry-point hints.
    pub entry_points: Vec<ExtendedPointId>,
    pub ef: usize,
}

#[derive(Clone, Copy)]
enum UpperQuery<'a> {
    Float32(&'a [f32]),
    Uint8(&'a [u8]),
}

type UpperNodeCandidate = (OrderedFloat<f32>, usize);

/// Dense visited-node bitmap for one upper-HNSW routing task.
///
/// Runtime upper-node IDs are validated and compiled into the dense range `0..nodes.len()`.
/// Using that invariant avoids hashing every level-zero neighbor while preserving exactly the
/// same first-visit test and traversal order as a hash set. The bitmap is task-local, so clearing
/// it between queries requires neither synchronization nor allocation.
#[derive(Debug)]
struct UpperVisitedSet {
    words: Vec<usize>,
    touched_words: Vec<usize>,
}

impl UpperVisitedSet {
    fn new(node_count: usize) -> Self {
        Self {
            words: vec![0; node_count.div_ceil(usize::BITS as usize)],
            touched_words: Vec::new(),
        }
    }

    fn clear(&mut self) {
        for word in self.touched_words.drain(..) {
            self.words[word] = 0;
        }
    }

    /// Returns `true` exactly once for each node between calls to [`Self::clear`].
    fn insert(&mut self, node_index: usize) -> bool {
        let word_index = node_index / usize::BITS as usize;
        let bit = 1usize << (node_index % usize::BITS as usize);
        let word = &mut self.words[word_index];
        if *word == 0 {
            self.touched_words.push(word_index);
        }
        let is_new = *word & bit == 0;
        *word |= bit;
        is_new
    }
}

/// Per-routing-task storage reused across every query in one coordinator routing chunk.
///
/// The router itself remains immutable and shareable. Keeping this scratch task-local avoids
/// synchronizing upper searches while retaining the capacity of the normalized query, visited
/// set, HNSW heaps, and sorted upper-node results between consecutive queries.
#[derive(Debug)]
pub(crate) struct OrionRouteScratch {
    query: Vec<f32>,
    byte_query: Vec<u8>,
    visited: UpperVisitedSet,
    candidates: BinaryHeap<Reverse<UpperNodeCandidate>>,
    nearest: BinaryHeap<UpperNodeCandidate>,
    sorted_nearest: Vec<UpperNodeCandidate>,
}

impl OrionRouteScratch {
    fn new(dimension: usize, upper_ef_search: usize, upper_node_count: usize) -> Self {
        Self {
            query: Vec::with_capacity(dimension),
            byte_query: Vec::with_capacity(dimension),
            visited: UpperVisitedSet::new(upper_node_count),
            candidates: BinaryHeap::with_capacity(upper_ef_search),
            nearest: BinaryHeap::with_capacity(upper_ef_search),
            sorted_nearest: Vec::with_capacity(upper_ef_search),
        }
    }
}

#[derive(Debug)]
struct RuntimeUpperNode {
    label: ExtendedPointId,
    owner_shard: ShardId,
}

#[derive(Debug)]
struct RuntimeUpperGraph {
    entry_point: usize,
    max_level: usize,
    level_count: usize,
    /// Node-major, then level-major ranges into `neighbors`.
    ///
    /// The serialized graph stores one independently allocated vector per node and level. Runtime
    /// HNSW traversal is pointer-chase heavy, so compile those immutable lists into one contiguous
    /// adjacency arena while retaining every neighbor's encounter order exactly.
    neighbor_ranges_by_node_and_level: Box<[Range<usize>]>,
    neighbors: Box<[usize]>,
}

impl RuntimeUpperGraph {
    #[inline]
    fn neighbors(&self, node_index: usize, level: usize) -> &[usize] {
        debug_assert!(level < self.level_count);
        let range = &self.neighbor_ranges_by_node_and_level[node_index * self.level_count + level];
        &self.neighbors[range.clone()]
    }
}

#[derive(Debug)]
enum UpperSearchBackend {
    Hnsw(RuntimeUpperGraph),
    BruteForceTesting,
}

/// Immutable, generation-specific Orion query router.
///
/// The normal constructor refuses artifacts without a serialized graph. Exact scanning exists only
/// behind the explicitly named testing constructor, so a production caller cannot silently change
/// the algorithm when graph state is missing.
#[derive(Debug)]
pub struct OrionRouter {
    generation: u64,
    vector_name: String,
    dimension: usize,
    distance: Distance,
    shard_count: ShardId,
    upper_k: usize,
    upper_ef_search: usize,
    dynamic_ef_base: usize,
    dynamic_ef_factor: usize,
    entry_point_capacity_hint: usize,
    nodes: Vec<RuntimeUpperNode>,
    /// Preprocessed upper vectors in immutable node-index-major order.
    ///
    /// Keeping all `node_count * dimension` elements in one allocation removes one vector-pointer
    /// chase and one independently allocated `Vec` per upper node from the distance hot path. Node
    /// indices, vector element order, and per-node preprocessing remain identical to the artifact.
    upper_vectors: Box<[VectorElementType]>,
    upper_byte_vectors: Option<Box<[u8]>>,
    node_by_label: AHashMap<ExtendedPointId, usize>,
    search_backend: UpperSearchBackend,
    scratch_pool: parking_lot::Mutex<Vec<OrionRouteScratch>>,
    learned_pruner: Option<super::learned_pruning::Policy>,
    whitebox_pruner: Option<super::whitebox_pruning::Policy>,
}

/// Exclusive workspace lease. The router snapshot owns the pool, so a generation
/// swap cannot mix bitmap shapes or leak search state between concurrent requests.
pub(crate) struct OrionRouteScratchLease<'a> {
    router: &'a OrionRouter,
    scratch: Option<OrionRouteScratch>,
}
impl std::ops::Deref for OrionRouteScratchLease<'_> {
    type Target = OrionRouteScratch;
    fn deref(&self) -> &Self::Target {
        self.scratch.as_ref().unwrap()
    }
}
impl std::ops::DerefMut for OrionRouteScratchLease<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.scratch.as_mut().unwrap()
    }
}
impl Drop for OrionRouteScratchLease<'_> {
    fn drop(&mut self) {
        let mut pool = self.router.scratch_pool.lock();
        if pool.len() < *common::defaults::POOL_KEEP_LIMIT {
            pool.push(self.scratch.take().unwrap());
        }
    }
}

impl OrionRouter {
    /// Build a production router. A complete serialized upper HNSW graph is mandatory.
    pub fn new(artifact: OrionRoutingArtifact) -> OrionRoutingResult<Self> {
        Self::build(&artifact, false)
    }

    /// Build an exact upper scanner for tests and artifact diagnostics only.
    pub fn new_brute_force_testing(artifact: OrionRoutingArtifact) -> OrionRoutingResult<Self> {
        Self::build(&artifact, true)
    }

    pub(crate) fn from_artifact_with_ef(
        artifact: &OrionRoutingArtifact,
        ef: usize,
    ) -> OrionRoutingResult<Self> {
        if ef == 0 {
            return Err(OrionRoutingError::EmptyUpperK);
        }
        let mut router = Self::from_artifact(artifact)?;
        router.upper_k = ef.min(artifact.upper_nodes.len());
        router.upper_ef_search = ef;
        Ok(router)
    }

    pub(crate) fn from_artifact(artifact: &OrionRoutingArtifact) -> OrionRoutingResult<Self> {
        Self::build(artifact, false)
    }

    fn build(
        artifact: &OrionRoutingArtifact,
        brute_force_testing: bool,
    ) -> OrionRoutingResult<Self> {
        artifact.validate()?;
        let node_by_label: AHashMap<_, _> = artifact
            .upper_nodes
            .iter()
            .enumerate()
            .map(|(index, node)| (node.label, index))
            .collect();

        let search_backend = if brute_force_testing {
            UpperSearchBackend::BruteForceTesting
        } else {
            let graph = artifact
                .upper_graph
                .as_ref()
                .ok_or(OrionRoutingError::MissingSerializedUpperGraph)?;
            UpperSearchBackend::Hnsw(compile_graph(graph, &node_by_label)?)
        };
        let distance = artifact.vector_schema.distance;
        let dimension = artifact.vector_schema.dimension;
        let entry_point_capacity_hint = route_entry_point_capacity_hint(
            artifact.upper_k,
            artifact.shard_count,
            &artifact.upper_nodes,
        );
        let upper_vector_element_count =
            checked_upper_vector_element_count(artifact.upper_nodes.len(), dimension)?;
        let is_byte = artifact.vector_schema.datatype == super::OrionVectorDatatype::Uint8;
        let mut upper_byte_vectors = is_byte.then(Vec::new);
        let mut upper_vectors = Vec::new();
        upper_vectors
            .try_reserve_exact(if is_byte {
                0
            } else {
                upper_vector_element_count
            })
            .map_err(|source| OrionRoutingError::UpperVectorStorageAllocation {
                element_count: upper_vector_element_count,
                reason: source.to_string(),
            })?;
        if let Some(bytes) = &mut upper_byte_vectors {
            bytes
                .try_reserve_exact(upper_vector_element_count)
                .map_err(|source| OrionRoutingError::UpperVectorStorageAllocation {
                    element_count: upper_vector_element_count,
                    reason: source.to_string(),
                })?;
        }
        let mut nodes = Vec::with_capacity(artifact.upper_nodes.len());
        for node in &artifact.upper_nodes {
            if let Some(bytes) = &mut upper_byte_vectors {
                bytes.extend(node.vector.iter().map(|&x| x as u8));
            } else {
                let preprocessed_vector = preprocess_vector(distance, node.vector.clone());
                debug_assert_eq!(preprocessed_vector.len(), dimension);
                upper_vectors.extend(preprocessed_vector);
            }
            nodes.push(RuntimeUpperNode {
                label: node.label,
                owner_shard: node.owner_shard,
            });
        }
        debug_assert_eq!(
            upper_byte_vectors
                .as_ref()
                .map_or(upper_vectors.len(), Vec::len),
            upper_vector_element_count
        );

        let learned_pruner = super::learned_pruning::configured(artifact)
            .map_err(|reason| OrionRoutingError::LearnedPruning { reason })?;
        let whitebox_pruner = super::whitebox_pruning::configured(artifact)
            .map_err(|reason| OrionRoutingError::WhiteboxPruning { reason })?;
        if learned_pruner.is_some() && whitebox_pruner.is_some() {
            return Err(OrionRoutingError::WhiteboxPruning {
                reason:
                    "learned and whitebox selectors cannot both be enabled for the same generation"
                        .into(),
            });
        }
        Ok(Self {
            generation: artifact.generation,
            vector_name: artifact.vector_schema.vector_name.clone(),
            dimension,
            distance,
            shard_count: artifact.shard_count,
            upper_k: artifact.upper_k,
            upper_ef_search: artifact.upper_ef_search,
            dynamic_ef_base: artifact.dynamic_ef_base,
            dynamic_ef_factor: artifact.dynamic_ef_factor,
            entry_point_capacity_hint,
            nodes,
            upper_vectors: upper_vectors.into_boxed_slice(),
            upper_byte_vectors: upper_byte_vectors.map(Vec::into_boxed_slice),
            node_by_label,
            search_backend,
            scratch_pool: parking_lot::Mutex::new(Vec::new()),
            learned_pruner,
            whitebox_pruner,
        })
    }

    pub fn generation(&self) -> u64 {
        self.generation
    }

    pub fn vector_name(&self) -> &str {
        &self.vector_name
    }

    pub fn shard_count(&self) -> ShardId {
        self.shard_count
    }

    pub(crate) fn owner_entries(&self, shard: ShardId) -> Vec<ExtendedPointId> {
        self.nodes
            .iter()
            .filter(|node| node.owner_shard == shard)
            .map(|node| node.label)
            .collect()
    }

    pub(crate) fn is_owner_entry(&self, label: ExtendedPointId, shard: ShardId) -> bool {
        self.node_by_label
            .get(&label)
            .is_some_and(|&index| self.nodes[index].owner_shard == shard)
    }

    pub fn upper_k(&self) -> usize {
        self.upper_k
    }

    pub(crate) fn borrow_route_scratch(&self) -> OrionRouteScratchLease<'_> {
        let cached = self.scratch_pool.lock().pop();
        OrionRouteScratchLease {
            router: self,
            scratch: Some(cached.unwrap_or_else(|| self.new_route_scratch())),
        }
    }

    pub(crate) fn new_route_scratch(&self) -> OrionRouteScratch {
        OrionRouteScratch::new(self.dimension, self.upper_ef_search, self.nodes.len())
    }

    pub fn search_upper(&self, query: &[f32]) -> OrionRoutingResult<Vec<OrionUpperHit>> {
        self.validate_query(query)?;
        let float_query;
        let byte_query;
        let query = if self.upper_byte_vectors.is_some() {
            byte_query = query.iter().map(|&x| x as u8).collect::<Vec<_>>();
            UpperQuery::Uint8(&byte_query)
        } else {
            float_query = preprocess_vector(self.distance, query.to_vec());
            UpperQuery::Float32(&float_query)
        };
        let hits = match &self.search_backend {
            UpperSearchBackend::Hnsw(graph) => self.search_hnsw(query, graph),
            UpperSearchBackend::BruteForceTesting => self.search_brute_force(query),
        }?;
        Ok(hits)
    }

    /// Search with `upper_ef_search`, then route only the nearest `upper_k` hits.
    /// Lower budgets depend on retained hits per owner, not the upper beam size.
    /// Optional learned selection drops whole targets after their budgets are fixed.
    pub fn route_query(&self, query: &[f32]) -> OrionRoutingResult<Vec<OrionShardTarget>> {
        let mut scratch = self.borrow_route_scratch();
        self.route_query_with_scratch(query, &mut scratch)
    }

    /// Route one query while retaining query-hot allocations owned by the caller's routing task.
    pub(crate) fn route_query_with_scratch(
        &self,
        query: &[f32],
        scratch: &mut OrionRouteScratch,
    ) -> OrionRoutingResult<Vec<OrionShardTarget>> {
        self.search_upper_indices_with_scratch(query, scratch)?;
        let targets = self.route_upper_node_indices(
            scratch
                .sorted_nearest
                .iter()
                .take(self.upper_k)
                .map(|(_, node_index)| *node_index),
        )?;
        let actual_upper_hits = scratch.sorted_nearest.len().min(self.upper_k);
        self.validate_target_entry_count(&targets, actual_upper_hits)?;
        let targets = if let Some(policy) = &self.whitebox_pruner {
            policy
                .apply(
                    scratch
                        .sorted_nearest
                        .iter()
                        .map(|(distance, index)| (self.nodes[*index].owner_shard, distance.0)),
                    targets,
                )
                .map_err(|reason| OrionRoutingError::WhiteboxPruning { reason })?
        } else if let Some(policy) = &self.learned_pruner {
            policy
                .apply(
                    scratch
                        .sorted_nearest
                        .iter()
                        .map(|(distance, index)| (self.nodes[*index].owner_shard, distance.0)),
                    targets,
                )
                .map_err(|reason| OrionRoutingError::LearnedPruning { reason })?
        } else {
            targets
        };
        // Compute and validate the complete original route first. Diagnostics may
        // remove whole targets, but must never recompute retained entries or EF.
        #[cfg(feature = "orion-oracle-diagnostics")]
        let targets =
            super::oracle_diagnostics::prune(query, self.generation, self.shard_count, targets)
                .map_err(|reason| OrionRoutingError::DiagnosticPruning { reason })?;
        Ok(targets)
    }

    /// Convert ordered upper hits into sorted logical-shard targets.
    ///
    /// Every one of the first `upper_k` hits contributes exactly once to its frozen upper owner.
    /// There is no adaptive shard pruning. Entry points retain hit order.
    pub fn route_upper_hits(
        &self,
        ordered_hits: &[OrionUpperHit],
    ) -> OrionRoutingResult<Vec<OrionShardTarget>> {
        let targets = self.route_upper_labels(ordered_hits.iter().map(|hit| hit.label))?;
        self.validate_target_entry_count(&targets, ordered_hits.len().min(self.upper_k))?;
        Ok(targets)
    }

    pub fn route_upper_labels(
        &self,
        ordered_labels: impl IntoIterator<Item = ExtendedPointId>,
    ) -> OrionRoutingResult<Vec<OrionShardTarget>> {
        self.route_upper_labels_impl(ordered_labels, true)
    }

    fn route_upper_labels_impl(
        &self,
        ordered_labels: impl IntoIterator<Item = ExtendedPointId>,
        deduplicate_labels: bool,
    ) -> OrionRoutingResult<Vec<OrionShardTarget>> {
        let mut seen_labels =
            deduplicate_labels.then(|| AHashSet::with_capacity(self.upper_k.saturating_mul(2)));
        // Logical shard IDs are validated as the dense range `0..shard_count` when the artifact
        // is loaded. Indexing that range directly avoids hashing every membership and makes the
        // final shard-ID order intrinsic rather than requiring a per-query sort.
        let mut entry_points_by_shard = vec![None; self.shard_count as usize];
        for label in ordered_labels.into_iter().take(self.upper_k) {
            let Some(&node_index) = self.node_by_label.get(&label) else {
                return Err(OrionRoutingError::UnknownUpperLabel { label });
            };
            if seen_labels
                .as_mut()
                .is_some_and(|seen_labels| !seen_labels.insert(label))
            {
                continue;
            }
            let node = &self.nodes[node_index];
            entry_points_by_shard[node.owner_shard as usize]
                .get_or_insert_with(|| Vec::with_capacity(self.entry_point_capacity_hint))
                .push(label);
        }

        self.finish_targets(entry_points_by_shard)
    }

    fn route_upper_node_indices(
        &self,
        ordered_node_indices: impl IntoIterator<Item = usize>,
    ) -> OrionRoutingResult<Vec<OrionShardTarget>> {
        // HNSW and exact-testing search both return each upper node at most once. Route directly
        // from those node indices so the production query path does not hash every upper label
        // back into the same immutable node array.
        let mut entry_points_by_shard = vec![None; self.shard_count as usize];
        for node_index in ordered_node_indices.into_iter().take(self.upper_k) {
            let node = &self.nodes[node_index];
            entry_points_by_shard[node.owner_shard as usize]
                .get_or_insert_with(|| Vec::with_capacity(self.entry_point_capacity_hint))
                .push(node.label);
        }
        self.finish_targets(entry_points_by_shard)
    }

    fn finish_targets(
        &self,
        entry_points_by_shard: Vec<Option<Vec<ExtendedPointId>>>,
    ) -> OrionRoutingResult<Vec<OrionShardTarget>> {
        entry_points_by_shard
            .into_iter()
            .enumerate()
            .filter_map(|(shard_id, entry_points)| {
                entry_points.map(|entry_points| (shard_id, entry_points))
            })
            .map(|(shard_id, entry_points)| {
                let ef = self
                    .dynamic_ef_factor
                    .checked_mul(entry_points.len())
                    .and_then(|increment| self.dynamic_ef_base.checked_add(increment))
                    .ok_or(OrionRoutingError::DynamicEfOverflow)?;
                Ok(OrionShardTarget {
                    shard_id: shard_id as ShardId,
                    entry_points,
                    ef,
                })
            })
            .collect()
    }

    fn validate_target_entry_count(
        &self,
        targets: &[OrionShardTarget],
        expected: usize,
    ) -> OrionRoutingResult<()> {
        let actual = targets
            .iter()
            .map(|target| target.entry_points.len())
            .sum::<usize>();
        if actual != expected {
            return Err(OrionRoutingError::IncompleteUpperRouting { expected, actual });
        }
        Ok(())
    }

    fn search_upper_indices_with_scratch(
        &self,
        query: &[f32],
        scratch: &mut OrionRouteScratch,
    ) -> OrionRoutingResult<()> {
        self.validate_query(query)?;

        let OrionRouteScratch {
            query: normalized_query,
            byte_query,
            visited,
            candidates,
            nearest,
            sorted_nearest,
        } = scratch;

        normalized_query.clear();
        normalized_query.extend_from_slice(query);
        let owned_query = mem::take(normalized_query);
        *normalized_query = if self.upper_byte_vectors.is_some() {
            owned_query
        } else {
            preprocess_vector(self.distance, owned_query)
        };
        let prepared_query = if self.upper_byte_vectors.is_some() {
            byte_query.clear();
            byte_query.extend(normalized_query.iter().map(|&x| x as u8));
            UpperQuery::Uint8(byte_query)
        } else {
            UpperQuery::Float32(normalized_query)
        };

        visited.clear();
        candidates.clear();
        nearest.clear();
        sorted_nearest.clear();

        match &self.search_backend {
            UpperSearchBackend::Hnsw(graph) => self.search_hnsw_indices_into(
                prepared_query,
                graph,
                visited,
                candidates,
                nearest,
                sorted_nearest,
            )?,
            UpperSearchBackend::BruteForceTesting => {
                self.search_brute_force_indices_into(prepared_query, sorted_nearest)?
            }
        }

        Ok(())
    }

    pub(crate) fn search_upper_with_scratch(
        &self,
        query: &[f32],
        scratch: &mut OrionRouteScratch,
    ) -> OrionRoutingResult<Vec<OrionUpperHit>> {
        self.search_upper_indices_with_scratch(query, scratch)?;
        Ok(scratch
            .sorted_nearest
            .iter()
            .take(self.upper_k)
            .map(|(distance, node_index)| OrionUpperHit {
                label: self.nodes[*node_index].label,
                distance: distance.into_inner(),
            })
            .collect())
    }

    pub(crate) fn validate_query(&self, query: &[f32]) -> OrionRoutingResult<()> {
        if query.len() != self.dimension {
            return Err(OrionRoutingError::QueryDimensionMismatch {
                expected: self.dimension,
                actual: query.len(),
            });
        }
        if let Some(dimension) = query.iter().position(|value| !value.is_finite()) {
            return Err(OrionRoutingError::NonFiniteQueryVector { dimension });
        }
        Ok(())
    }

    fn search_brute_force(&self, query: UpperQuery<'_>) -> OrionRoutingResult<Vec<OrionUpperHit>> {
        let mut hits = self
            .nodes
            .iter()
            .enumerate()
            .map(|(index, node)| {
                Ok((
                    OrderedFloat(self.distance_to(query, index)?),
                    index,
                    node.label,
                ))
            })
            .collect::<OrionRoutingResult<Vec<_>>>()?;
        hits.sort_unstable_by_key(|(distance, index, _)| (*distance, *index));
        Ok(hits
            .into_iter()
            .take(self.upper_k)
            .map(|(distance, _, label)| OrionUpperHit {
                label,
                distance: distance.into_inner(),
            })
            .collect())
    }

    fn search_brute_force_indices_into(
        &self,
        query: UpperQuery<'_>,
        sorted_nearest: &mut Vec<UpperNodeCandidate>,
    ) -> OrionRoutingResult<()> {
        sorted_nearest.reserve(self.nodes.len());
        for node_index in 0..self.nodes.len() {
            sorted_nearest.push((
                OrderedFloat(self.distance_to(query, node_index)?),
                node_index,
            ));
        }
        sorted_nearest.sort_unstable();
        Ok(())
    }

    fn search_hnsw(
        &self,
        query: UpperQuery<'_>,
        graph: &RuntimeUpperGraph,
    ) -> OrionRoutingResult<Vec<OrionUpperHit>> {
        let mut current = graph.entry_point;
        let mut current_distance = OrderedFloat(self.distance_to(query, current)?);

        for level in (1..=graph.max_level).rev() {
            loop {
                let mut next = (current_distance, current);
                for &neighbor in graph.neighbors(current, level) {
                    let candidate = (OrderedFloat(self.distance_to(query, neighbor)?), neighbor);
                    if candidate < next {
                        next = candidate;
                    }
                }
                if next.1 == current {
                    break;
                }
                (current_distance, current) = next;
            }
        }

        let mut visited = AHashSet::with_capacity(self.upper_ef_search.saturating_mul(2));
        let mut candidates: BinaryHeap<Reverse<(OrderedFloat<f32>, usize)>> = BinaryHeap::new();
        let mut nearest: BinaryHeap<(OrderedFloat<f32>, usize)> = BinaryHeap::new();
        visited.insert(current);
        candidates.push(Reverse((current_distance, current)));
        nearest.push((current_distance, current));

        while let Some(Reverse(candidate)) = candidates.pop() {
            if nearest.len() >= self.upper_ef_search && candidate > *nearest.peek().unwrap() {
                break;
            }
            for &neighbor in graph.neighbors(candidate.1, 0) {
                if !visited.insert(neighbor) {
                    continue;
                }
                let neighbor_candidate =
                    (OrderedFloat(self.distance_to(query, neighbor)?), neighbor);
                let should_add = nearest.len() < self.upper_ef_search
                    || neighbor_candidate < *nearest.peek().unwrap();
                if should_add {
                    candidates.push(Reverse(neighbor_candidate));
                    nearest.push(neighbor_candidate);
                    if nearest.len() > self.upper_ef_search {
                        nearest.pop();
                    }
                }
            }
        }

        let mut nearest = nearest.into_vec();
        nearest.sort_unstable();
        Ok(nearest
            .into_iter()
            .take(self.upper_k)
            .map(|(distance, index)| OrionUpperHit {
                label: self.nodes[index].label,
                distance: distance.into_inner(),
            })
            .collect())
    }

    #[allow(clippy::too_many_arguments)]
    fn search_hnsw_indices_into(
        &self,
        query: UpperQuery<'_>,
        graph: &RuntimeUpperGraph,
        visited: &mut UpperVisitedSet,
        candidates: &mut BinaryHeap<Reverse<UpperNodeCandidate>>,
        nearest: &mut BinaryHeap<UpperNodeCandidate>,
        sorted_nearest: &mut Vec<UpperNodeCandidate>,
    ) -> OrionRoutingResult<()> {
        let mut current = graph.entry_point;
        let mut current_distance = OrderedFloat(self.distance_to(query, current)?);

        for level in (1..=graph.max_level).rev() {
            loop {
                let mut next = (current_distance, current);
                for &neighbor in graph.neighbors(current, level) {
                    let candidate = (OrderedFloat(self.distance_to(query, neighbor)?), neighbor);
                    if candidate < next {
                        next = candidate;
                    }
                }
                if next.1 == current {
                    break;
                }
                (current_distance, current) = next;
            }
        }

        visited.insert(current);
        candidates.push(Reverse((current_distance, current)));
        nearest.push((current_distance, current));

        while let Some(Reverse(candidate)) = candidates.pop() {
            if nearest.len() >= self.upper_ef_search && candidate > *nearest.peek().unwrap() {
                break;
            }
            for &neighbor in graph.neighbors(candidate.1, 0) {
                if !visited.insert(neighbor) {
                    continue;
                }
                let neighbor_candidate =
                    (OrderedFloat(self.distance_to(query, neighbor)?), neighbor);
                let should_add = nearest.len() < self.upper_ef_search
                    || neighbor_candidate < *nearest.peek().unwrap();
                if should_add {
                    candidates.push(Reverse(neighbor_candidate));
                    nearest.push(neighbor_candidate);
                    if nearest.len() > self.upper_ef_search {
                        nearest.pop();
                    }
                }
            }
        }

        sorted_nearest.extend(nearest.drain());
        sorted_nearest.sort_unstable();
        Ok(())
    }

    fn distance_to(&self, query: UpperQuery<'_>, node_index: usize) -> OrionRoutingResult<f32> {
        let vector_start = node_index * self.dimension;
        let vector_end = vector_start + self.dimension;

        // Qdrant metrics return a larger-is-better similarity. Negation gives the router's
        // lower-is-better ordering without changing the production SIMD scorer semantics.
        let distance = -match query {
            UpperQuery::Float32(query) => similarity(
                self.distance,
                query,
                &self.upper_vectors[vector_start..vector_end],
            ),
            UpperQuery::Uint8(query) => {
                let node = &self
                    .upper_byte_vectors
                    .as_ref()
                    .expect("byte query requires byte storage")[vector_start..vector_end];
                match self.distance {
                    Distance::Euclid => <EuclidMetric as Metric<u8>>::similarity(query, node),
                    Distance::Cosine => <CosineMetric as Metric<u8>>::similarity(query, node),
                    Distance::Dot => <DotProductMetric as Metric<u8>>::similarity(query, node),
                    Distance::Manhattan => <ManhattanMetric as Metric<u8>>::similarity(query, node),
                }
            }
        };
        if !distance.is_finite() {
            return Err(OrionRoutingError::NonFiniteDistance {
                label: self.nodes[node_index].label,
            });
        }
        Ok(distance)
    }
}

fn checked_upper_vector_element_count(
    node_count: usize,
    dimension: usize,
) -> OrionRoutingResult<usize> {
    node_count
        .checked_mul(dimension)
        .ok_or(OrionRoutingError::UpperVectorStorageSizeOverflow {
            node_count,
            dimension,
        })
}

fn route_entry_point_capacity_hint(
    upper_k: usize,
    shard_count: ShardId,
    upper_nodes: &[super::artifact::OrionUpperNode],
) -> usize {
    let denominator = upper_nodes
        .len()
        .saturating_mul(shard_count as usize)
        .max(1);
    let expected = upper_k
        .saturating_mul(upper_nodes.len())
        .div_ceil(denominator)
        .max(1);
    expected
        .checked_next_power_of_two()
        .unwrap_or(upper_k.max(1))
        .min(upper_k.max(1))
}

pub(super) fn preprocess_vector(
    distance: Distance,
    vector: Vec<VectorElementType>,
) -> Vec<VectorElementType> {
    match distance {
        Distance::Cosine => <CosineMetric as Metric<VectorElementType>>::preprocess(vector),
        Distance::Dot => <DotProductMetric as Metric<VectorElementType>>::preprocess(vector),
        Distance::Euclid => <EuclidMetric as Metric<VectorElementType>>::preprocess(vector),
        Distance::Manhattan => <ManhattanMetric as Metric<VectorElementType>>::preprocess(vector),
    }
}

fn similarity(distance: Distance, left: &[f32], right: &[f32]) -> f32 {
    match distance {
        Distance::Cosine => <CosineMetric as Metric<VectorElementType>>::similarity(left, right),
        Distance::Dot => <DotProductMetric as Metric<VectorElementType>>::similarity(left, right),
        Distance::Euclid => <EuclidMetric as Metric<VectorElementType>>::similarity(left, right),
        Distance::Manhattan => {
            <ManhattanMetric as Metric<VectorElementType>>::similarity(left, right)
        }
    }
}

fn compile_graph(
    graph: &OrionUpperHnswGraph,
    node_by_label: &AHashMap<ExtendedPointId, usize>,
) -> OrionRoutingResult<RuntimeUpperGraph> {
    let level_count = graph.max_level + 1;
    let range_count = node_by_label.len().saturating_mul(level_count);
    let neighbor_count = graph
        .nodes
        .iter()
        .flat_map(|node| &node.neighbors_by_level)
        .map(Vec::len)
        .sum();
    let mut neighbor_ranges_by_node_and_level = vec![0..0; range_count];
    let mut neighbors = Vec::with_capacity(neighbor_count);
    for graph_node in &graph.nodes {
        let node_index = node_by_label[&graph_node.label];
        for (level, neighbor_labels) in graph_node.neighbors_by_level.iter().enumerate() {
            let start = neighbors.len();
            for neighbor in neighbor_labels {
                neighbors.push(node_by_label.get(neighbor).copied().ok_or(
                    OrionRoutingError::GraphNeighborNotFound {
                        label: graph_node.label,
                        level,
                        neighbor: *neighbor,
                    },
                )?);
            }
            neighbor_ranges_by_node_and_level[node_index * level_count + level] =
                start..neighbors.len();
        }
    }
    Ok(RuntimeUpperGraph {
        entry_point: node_by_label[&graph.entry_point],
        max_level: graph.max_level,
        level_count,
        neighbor_ranges_by_node_and_level: neighbor_ranges_by_node_and_level.into_boxed_slice(),
        neighbors: neighbors.into_boxed_slice(),
    })
}

#[cfg(test)]
mod flat_upper_vector_tests {
    use super::*;
    use crate::orion::{
        ORION_ROUTING_ARTIFACT_FORMAT_VERSION, OrionUpperGraphNode, OrionUpperNode,
        OrionVectorDatatype, OrionVectorSchemaFingerprint,
    };

    fn id(value: u64) -> ExtendedPointId {
        ExtendedPointId::NumId(value)
    }

    fn cosine_artifact() -> OrionRoutingArtifact {
        OrionRoutingArtifact {
            format_version: ORION_ROUTING_ARTIFACT_FORMAT_VERSION,
            generation: 1,
            vector_schema: OrionVectorSchemaFingerprint {
                vector_name: String::new(),
                dimension: 2,
                distance: Distance::Cosine,
                datatype: OrionVectorDatatype::Float32,
            },
            shard_count: 2,
            layout_sha256: "a".repeat(64),
            logical_point_count: 3,
            physical_point_count: 3,
            upper_k: 3,
            upper_ef_search: 3,
            dynamic_ef_base: 4,
            dynamic_ef_factor: 2,
            upper_nodes: vec![
                OrionUpperNode {
                    label: id(10),
                    vector: vec![3.0, 4.0],
                    owner_shard: 1,
                },
                OrionUpperNode {
                    label: id(20),
                    vector: vec![0.0, -2.0],
                    owner_shard: 0,
                },
                OrionUpperNode {
                    label: id(30),
                    vector: vec![-5.0, 0.0],
                    owner_shard: 1,
                },
            ],
            upper_graph: Some(OrionUpperHnswGraph {
                entry_point: id(30),
                max_level: 0,
                nodes: vec![
                    OrionUpperGraphNode {
                        label: id(10),
                        neighbors_by_level: vec![vec![id(20), id(30)]],
                    },
                    OrionUpperGraphNode {
                        label: id(20),
                        neighbors_by_level: vec![vec![id(10), id(30)]],
                    },
                    OrionUpperGraphNode {
                        label: id(30),
                        neighbors_by_level: vec![vec![id(10), id(20)]],
                    },
                ],
            }),
        }
    }

    #[test]
    fn uint8_router_uses_byte_storage_and_native_byte_metrics() {
        for distance in [
            Distance::Euclid,
            Distance::Cosine,
            Distance::Dot,
            Distance::Manhattan,
        ] {
            let mut artifact = cosine_artifact();
            artifact.vector_schema.datatype = super::super::OrionVectorDatatype::Uint8;
            artifact.vector_schema.distance = distance;
            artifact.vector_schema.dimension = 128;
            artifact.upper_graph = None;
            for (i, node) in artifact.upper_nodes.iter_mut().enumerate() {
                node.vector = (0..128).map(|j| ((i * 67 + j * 13) % 256) as f32).collect();
            }
            artifact.upper_k = artifact.upper_nodes.len();
            artifact.upper_ef_search = artifact.upper_k.max(32);
            let artifact = artifact
                .build_upper_hnsw(super::super::OrionUpperGraphBuildOptions {
                    m: 4,
                    ef_construct: 32,
                    seed: 100,
                })
                .unwrap();
            let query: Vec<f32> = (0..128)
                .map(|i| [-1.5, 0.0, 127.9, 128.0, 255.0, 300.0][i % 6])
                .collect();
            let bytes: Vec<u8> = query.iter().map(|&x| x as u8).collect();
            let mut expected = artifact
                .upper_nodes
                .iter()
                .enumerate()
                .map(|(i, node)| {
                    let v: Vec<u8> = node.vector.iter().map(|&x| x as u8).collect();
                    let score = match distance {
                        Distance::Euclid => <EuclidMetric as Metric<u8>>::similarity(&bytes, &v),
                        Distance::Cosine => <CosineMetric as Metric<u8>>::similarity(&bytes, &v),
                        Distance::Dot => <DotProductMetric as Metric<u8>>::similarity(&bytes, &v),
                        Distance::Manhattan => {
                            <ManhattanMetric as Metric<u8>>::similarity(&bytes, &v)
                        }
                    };
                    (OrderedFloat(-score), i, node.label)
                })
                .collect::<Vec<_>>();
            expected.sort_unstable();
            for router in [
                OrionRouter::new(artifact.clone()).unwrap(),
                OrionRouter::new_brute_force_testing(artifact.clone()).unwrap(),
            ] {
                assert!(router.upper_vectors.is_empty());
                assert_eq!(
                    router.upper_byte_vectors.as_ref().unwrap().len(),
                    artifact.upper_nodes.len() * 128
                );
                let hits = router.search_upper(&query).unwrap();
                assert_eq!(
                    hits.iter()
                        .map(|h| (h.label, h.distance))
                        .collect::<Vec<_>>(),
                    expected
                        .iter()
                        .map(|(d, _, id)| (*id, d.0))
                        .collect::<Vec<_>>()
                );
                let mut scratch = router.new_route_scratch();
                assert_eq!(
                    router
                        .search_upper_with_scratch(&query, &mut scratch)
                        .unwrap(),
                    hits
                );
                assert_eq!(
                    router.route_query(&query).unwrap(),
                    router.route_upper_hits(&hits).unwrap()
                );
            }
            let mut corrupt = artifact;
            corrupt.upper_nodes[0].vector[0] = 256.0;
            assert!(OrionRouter::new(corrupt).is_err());
        }
    }

    #[test]
    fn scratch_pool_reuses_storage_and_isolates_concurrent_queries() {
        let router = OrionRouter::new(cosine_artifact()).unwrap();
        let queries = [vec![1.0, -0.5], vec![0.0, 1.0], vec![-1.0, 0.0]];
        let expected: Vec<_> = queries
            .iter()
            .map(|q| {
                let mut scratch = router.new_route_scratch();
                router.route_query_with_scratch(q, &mut scratch).unwrap()
            })
            .collect();
        let pointer = {
            let scratch = router.borrow_route_scratch();
            scratch.visited.words.as_ptr()
        };
        assert_eq!(
            router.borrow_route_scratch().visited.words.as_ptr(),
            pointer
        );
        assert!(router.route_query(&[f32::NAN, 1.0]).is_err());
        std::thread::scope(|scope| {
            for worker in 0..16 {
                let router = &router;
                let queries = &queries;
                let expected = &expected;
                scope.spawn(move || {
                    for round in 0..100 {
                        let i = (worker + round) % queries.len();
                        assert_eq!(router.route_query(&queries[i]).unwrap(), expected[i]);
                    }
                });
            }
        });
        assert!(!router.scratch_pool.lock().is_empty());
        assert!(router.scratch_pool.lock().len() <= *common::defaults::POOL_KEEP_LIMIT);
    }

    #[test]
    fn flat_upper_vectors_preserve_node_order_preprocessing_and_backend_scores() {
        let artifact = cosine_artifact();
        let expected_vectors = artifact
            .upper_nodes
            .iter()
            .flat_map(|node| {
                preprocess_vector(artifact.vector_schema.distance, node.vector.clone())
            })
            .collect::<Vec<_>>();
        let expected_metadata = artifact
            .upper_nodes
            .iter()
            .map(|node| (node.label, node.owner_shard))
            .collect::<Vec<_>>();

        let brute_force = OrionRouter::new_brute_force_testing(artifact.clone()).unwrap();
        let hnsw = OrionRouter::new(artifact).unwrap();

        for router in [&brute_force, &hnsw] {
            assert_eq!(router.upper_vectors.as_ref(), expected_vectors.as_slice());
            assert_eq!(
                router
                    .nodes
                    .iter()
                    .map(|node| (node.label, node.owner_shard))
                    .collect::<Vec<_>>(),
                expected_metadata,
            );
            assert_eq!(
                router.upper_vectors.len(),
                router.nodes.len() * router.dimension,
            );
        }

        let query = [1.0, -0.5];
        let brute_force_hits = brute_force.search_upper(&query).unwrap();
        let hnsw_hits = hnsw.search_upper(&query).unwrap();
        assert_eq!(
            hnsw_hits
                .iter()
                .map(|hit| (hit.label, hit.distance.to_bits()))
                .collect::<Vec<_>>(),
            brute_force_hits
                .iter()
                .map(|hit| (hit.label, hit.distance.to_bits()))
                .collect::<Vec<_>>(),
        );
    }

    #[test]
    fn flat_upper_vector_element_count_fails_closed_on_overflow() {
        assert_eq!(checked_upper_vector_element_count(37, 200).unwrap(), 7_400);
        assert!(matches!(
            checked_upper_vector_element_count(usize::MAX, 2),
            Err(OrionRoutingError::UpperVectorStorageSizeOverflow {
                node_count: usize::MAX,
                dimension: 2,
            })
        ));
    }
}

#[cfg(test)]
mod dense_visited_tests {
    use super::UpperVisitedSet;

    #[test]
    fn dense_visited_set_preserves_insert_and_clear_semantics_across_words() {
        let word_bits = usize::BITS as usize;
        let last_node = word_bits * 2;
        let mut visited = UpperVisitedSet::new(last_node + 1);

        for node in [0, word_bits - 1, word_bits, last_node] {
            assert!(visited.insert(node));
            assert!(!visited.insert(node));
        }

        visited.clear();
        for node in [last_node, word_bits, word_bits - 1, 0] {
            assert!(visited.insert(node));
            assert!(!visited.insert(node));
        }
    }
}
