use std::cmp::Reverse;
use std::collections::{BTreeMap, BinaryHeap, HashMap, HashSet};

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

#[derive(Debug)]
struct RuntimeUpperNode {
    label: ExtendedPointId,
    vector: Vec<f32>,
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
    node_by_label: HashMap<ExtendedPointId, usize>,
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
        let node_by_label: HashMap<_, _> = artifact
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

        Ok(Self {
            generation: artifact.generation,
            vector_name: artifact.vector_schema.vector_name,
            dimension: artifact.vector_schema.dimension,
            distance,
            shard_count: artifact.shard_count,
            upper_k: artifact.upper_k,
            upper_ef_search: artifact.upper_ef_search,
            dynamic_ef_base: artifact.dynamic_ef_base,
            dynamic_ef_factor: artifact.dynamic_ef_factor,
            nodes: artifact
                .upper_nodes
                .into_iter()
                .map(|node| RuntimeUpperNode {
                    label: node.label,
                    vector: preprocess_vector(distance, node.vector),
                    shard_membership: node.shard_membership,
                })
                .collect(),
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
        let hits = self.search_upper(query)?;
        self.route_upper_hits(&hits)
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
        #[derive(Default)]
        struct TargetBuilder {
            entry_points: Vec<ExtendedPointId>,
            seen: HashSet<ExtendedPointId>,
        }

        let mut target_builders: BTreeMap<ShardId, TargetBuilder> = BTreeMap::new();
        for label in ordered_labels.into_iter().take(self.upper_k) {
            let Some(&node_index) = self.node_by_label.get(&label) else {
                return Err(OrionRoutingError::UnknownUpperLabel { label });
            };
            let node = &self.nodes[node_index];
            for &shard_id in &node.shard_membership {
                let target = target_builders.entry(shard_id).or_default();
                if target.seen.insert(label) {
                    target.entry_points.push(label);
                }
            }
        }

        target_builders
            .into_iter()
            .map(|(shard_id, target)| {
                let ef = self
                    .dynamic_ef_factor
                    .checked_mul(target.entry_points.len())
                    .and_then(|increment| self.dynamic_ef_base.checked_add(increment))
                    .ok_or(OrionRoutingError::DynamicEfOverflow)?;
                Ok(OrionShardTarget {
                    shard_id,
                    entry_points: target.entry_points,
                    ef,
                })
            })
            .collect()
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

        let mut visited = HashSet::with_capacity(self.upper_ef_search.saturating_mul(2));
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

    fn distance_to(&self, query: &[f32], node_index: usize) -> OrionRoutingResult<f32> {
        let node = &self.nodes[node_index];
        // Qdrant metrics return a larger-is-better similarity. Negation gives the router's
        // lower-is-better ordering without changing the production SIMD scorer semantics.
        let distance = -similarity(self.distance, query, &node.vector);
        if !distance.is_finite() {
            return Err(OrionRoutingError::NonFiniteDistance { label: node.label });
        }
        Ok(distance)
    }
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
    node_by_label: &HashMap<ExtendedPointId, usize>,
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
