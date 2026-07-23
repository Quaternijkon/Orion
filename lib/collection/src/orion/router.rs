use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::mem;

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
}

impl UpperVisitedSet {
    fn new(node_count: usize) -> Self {
        Self {
            words: vec![0; node_count.div_ceil(usize::BITS as usize)],
        }
    }

    fn clear(&mut self) {
        self.words.fill(0);
    }

    /// Returns `true` exactly once for each node between calls to [`Self::clear`].
    fn insert(&mut self, node_index: usize) -> bool {
        let word_index = node_index / usize::BITS as usize;
        let bit = 1usize << (node_index % usize::BITS as usize);
        let word = &mut self.words[word_index];
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
    visited: UpperVisitedSet,
    candidates: BinaryHeap<Reverse<UpperNodeCandidate>>,
    nearest: BinaryHeap<UpperNodeCandidate>,
    sorted_nearest: Vec<UpperNodeCandidate>,
}

impl OrionRouteScratch {
    fn new(dimension: usize, upper_ef_search: usize, upper_node_count: usize) -> Self {
        Self {
            query: Vec::with_capacity(dimension),
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
    shard_membership: Vec<ShardId>,
}

#[derive(Debug)]
struct RuntimeUpperGraph {
    entry_point: usize,
    max_level: usize,
    neighbors_by_node_and_level: Vec<Vec<Vec<usize>>>,
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
    nodes: Vec<RuntimeUpperNode>,
    /// Preprocessed upper vectors in immutable node-index-major order.
    ///
    /// Keeping all `node_count * dimension` elements in one allocation removes one vector-pointer
    /// chase and one independently allocated `Vec` per upper node from the distance hot path. Node
    /// indices, vector element order, and per-node preprocessing remain identical to the artifact.
    upper_vectors: Box<[VectorElementType]>,
    node_by_label: AHashMap<ExtendedPointId, usize>,
    search_backend: UpperSearchBackend,
}

impl OrionRouter {
    /// Build a production router. A complete serialized upper HNSW graph is mandatory.
    pub fn new(artifact: OrionRoutingArtifact) -> OrionRoutingResult<Self> {
        Self::build(artifact, false)
    }

    /// Build an exact upper scanner for tests and artifact diagnostics only.
    pub fn new_brute_force_testing(artifact: OrionRoutingArtifact) -> OrionRoutingResult<Self> {
        Self::build(artifact, true)
    }

    fn build(
        artifact: OrionRoutingArtifact,
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
        let upper_vector_element_count =
            checked_upper_vector_element_count(artifact.upper_nodes.len(), dimension)?;
        let mut upper_vectors = Vec::new();
        upper_vectors
            .try_reserve_exact(upper_vector_element_count)
            .map_err(|source| OrionRoutingError::UpperVectorStorageAllocation {
                element_count: upper_vector_element_count,
                reason: source.to_string(),
            })?;
        let mut nodes = Vec::with_capacity(artifact.upper_nodes.len());
        for node in artifact.upper_nodes {
            let preprocessed_vector = preprocess_vector(distance, node.vector);
            debug_assert_eq!(preprocessed_vector.len(), dimension);
            upper_vectors.extend(preprocessed_vector);
            nodes.push(RuntimeUpperNode {
                label: node.label,
                shard_membership: node.shard_membership,
            });
        }
        debug_assert_eq!(upper_vectors.len(), upper_vector_element_count);

        Ok(Self {
            generation: artifact.generation,
            vector_name: artifact.vector_schema.vector_name,
            dimension,
            distance,
            shard_count: artifact.shard_count,
            upper_k: artifact.upper_k,
            upper_ef_search: artifact.upper_ef_search,
            dynamic_ef_base: artifact.dynamic_ef_base,
            dynamic_ef_factor: artifact.dynamic_ef_factor,
            nodes,
            upper_vectors: upper_vectors.into_boxed_slice(),
            node_by_label,
            search_backend,
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

    pub fn upper_k(&self) -> usize {
        self.upper_k
    }

    pub(crate) fn new_route_scratch(&self) -> OrionRouteScratch {
        OrionRouteScratch::new(self.dimension, self.upper_ef_search, self.nodes.len())
    }

    pub fn search_upper(&self, query: &[f32]) -> OrionRoutingResult<Vec<OrionUpperHit>> {
        self.validate_query(query)?;
        let query = preprocess_vector(self.distance, query.to_vec());
        let hits = match &self.search_backend {
            UpperSearchBackend::Hnsw(graph) => self.search_hnsw(&query, graph),
            UpperSearchBackend::BruteForceTesting => self.search_brute_force(&query),
        }?;
        if hits.len() != self.upper_k {
            return Err(OrionRoutingError::IncompleteUpperSearch {
                expected: self.upper_k,
                actual: hits.len(),
            });
        }
        Ok(hits)
    }

    /// Search the upper tier and build the complete configured shard union.
    pub fn route_query(&self, query: &[f32]) -> OrionRoutingResult<Vec<OrionShardTarget>> {
        let mut scratch = self.new_route_scratch();
        self.route_query_with_scratch(query, &mut scratch)
    }

    /// Route one query while retaining query-hot allocations owned by the caller's routing task.
    pub(crate) fn route_query_with_scratch(
        &self,
        query: &[f32],
        scratch: &mut OrionRouteScratch,
    ) -> OrionRoutingResult<Vec<OrionShardTarget>> {
        self.search_upper_indices_with_scratch(query, scratch)?;
        self.route_upper_node_indices(
            scratch
                .sorted_nearest
                .iter()
                .take(self.upper_k)
                .map(|(_, node_index)| *node_index),
        )
    }

    /// Convert ordered upper hits into sorted logical-shard targets.
    ///
    /// Every membership of every one of the first `upper_k` hits is retained. There is no adaptive
    /// shard pruning. Entry points retain hit order and are de-duplicated independently per shard.
    pub fn route_upper_hits(
        &self,
        ordered_hits: &[OrionUpperHit],
    ) -> OrionRoutingResult<Vec<OrionShardTarget>> {
        self.route_upper_labels(ordered_hits.iter().map(|hit| hit.label))
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
        let mut entry_points_by_shard = vec![Vec::new(); self.shard_count as usize];
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
            for &shard_id in &node.shard_membership {
                // Artifact validation guarantees that one upper node cannot list the same shard
                // twice. A label-global duplicate check is therefore exactly equivalent to the
                // previous HashSet attached to every target shard.
                entry_points_by_shard[shard_id as usize].push(label);
            }
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
        let mut entry_points_by_shard = vec![Vec::new(); self.shard_count as usize];
        for node_index in ordered_node_indices.into_iter().take(self.upper_k) {
            let node = &self.nodes[node_index];
            for &shard_id in &node.shard_membership {
                entry_points_by_shard[shard_id as usize].push(node.label);
            }
        }
        self.finish_targets(entry_points_by_shard)
    }

    fn finish_targets(
        &self,
        entry_points_by_shard: Vec<Vec<ExtendedPointId>>,
    ) -> OrionRoutingResult<Vec<OrionShardTarget>> {
        entry_points_by_shard
            .into_iter()
            .enumerate()
            .filter(|(_, entry_points)| !entry_points.is_empty())
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

    fn search_upper_indices_with_scratch(
        &self,
        query: &[f32],
        scratch: &mut OrionRouteScratch,
    ) -> OrionRoutingResult<()> {
        self.validate_query(query)?;

        let OrionRouteScratch {
            query: normalized_query,
            visited,
            candidates,
            nearest,
            sorted_nearest,
        } = scratch;

        normalized_query.clear();
        normalized_query.extend_from_slice(query);
        let owned_query = mem::take(normalized_query);
        *normalized_query = preprocess_vector(self.distance, owned_query);

        visited.clear();
        candidates.clear();
        nearest.clear();
        sorted_nearest.clear();

        match &self.search_backend {
            UpperSearchBackend::Hnsw(graph) => self.search_hnsw_indices_into(
                normalized_query,
                graph,
                visited,
                candidates,
                nearest,
                sorted_nearest,
            )?,
            UpperSearchBackend::BruteForceTesting => {
                self.search_brute_force_indices_into(normalized_query, sorted_nearest)?
            }
        }

        let actual = sorted_nearest.len().min(self.upper_k);
        if actual != self.upper_k {
            return Err(OrionRoutingError::IncompleteUpperSearch {
                expected: self.upper_k,
                actual,
            });
        }
        Ok(())
    }

    #[cfg(test)]
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

    fn search_brute_force(&self, query: &[f32]) -> OrionRoutingResult<Vec<OrionUpperHit>> {
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
        query: &[f32],
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
        query: &[f32],
        graph: &RuntimeUpperGraph,
    ) -> OrionRoutingResult<Vec<OrionUpperHit>> {
        let mut current = graph.entry_point;
        let mut current_distance = OrderedFloat(self.distance_to(query, current)?);

        for level in (1..=graph.max_level).rev() {
            loop {
                let mut next = (current_distance, current);
                for &neighbor in &graph.neighbors_by_node_and_level[current][level] {
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
            for &neighbor in &graph.neighbors_by_node_and_level[candidate.1][0] {
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
        query: &[f32],
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
                for &neighbor in &graph.neighbors_by_node_and_level[current][level] {
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
            for &neighbor in &graph.neighbors_by_node_and_level[candidate.1][0] {
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

    fn distance_to(&self, query: &[f32], node_index: usize) -> OrionRoutingResult<f32> {
        let vector_start = node_index * self.dimension;
        let vector_end = vector_start + self.dimension;
        let node_vector = &self.upper_vectors[vector_start..vector_end];
        // Qdrant metrics return a larger-is-better similarity. Negation gives the router's
        // lower-is-better ordering without changing the production SIMD scorer semantics.
        let distance = -similarity(self.distance, query, node_vector);
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

fn preprocess_vector(distance: Distance, vector: Vec<VectorElementType>) -> Vec<VectorElementType> {
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
    let mut neighbors_by_node_and_level = vec![Vec::new(); node_by_label.len()];
    for graph_node in &graph.nodes {
        let node_index = node_by_label[&graph_node.label];
        neighbors_by_node_and_level[node_index] = graph_node
            .neighbors_by_level
            .iter()
            .map(|neighbors| {
                neighbors
                    .iter()
                    .map(|label| {
                        node_by_label.get(label).copied().ok_or(
                            OrionRoutingError::GraphNeighborNotFound {
                                label: graph_node.label,
                                level: 0,
                                neighbor: *label,
                            },
                        )
                    })
                    .collect()
            })
            .collect::<OrionRoutingResult<Vec<_>>>()?;
    }
    Ok(RuntimeUpperGraph {
        entry_point: node_by_label[&graph.entry_point],
        max_level: graph.max_level,
        neighbors_by_node_and_level,
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
                    shard_membership: vec![1, 0],
                },
                OrionUpperNode {
                    label: id(20),
                    vector: vec![0.0, -2.0],
                    shard_membership: vec![0],
                },
                OrionUpperNode {
                    label: id(30),
                    vector: vec![-5.0, 0.0],
                    shard_membership: vec![1],
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
            .map(|node| (node.label, node.shard_membership.clone()))
            .collect::<Vec<_>>();

        let brute_force = OrionRouter::new_brute_force_testing(artifact.clone()).unwrap();
        let hnsw = OrionRouter::new(artifact).unwrap();

        for router in [&brute_force, &hnsw] {
            assert_eq!(router.upper_vectors.as_ref(), expected_vectors.as_slice());
            assert_eq!(
                router
                    .nodes
                    .iter()
                    .map(|node| (node.label, node.shard_membership.clone()))
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
