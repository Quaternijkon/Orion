use common::bitvec::BitVec;
use common::counter::hardware_counter::HardwareCounterCell;
use common::types::PointOffsetType;
use rand::SeedableRng;
use rand::prelude::StdRng;
use segment::data_types::vectors::{VectorElementType, VectorRef};
use segment::index::hnsw_index::HnswM;
use segment::index::hnsw_index::graph_layers_builder::GraphLayersBuilder;
use segment::index::hnsw_index::point_scorer::FilteredScorer;
use segment::vector_storage::VectorStorage;
use segment::vector_storage::dense::volatile_dense_vector_storage::new_volatile_dense_vector_storage;

use super::artifact::{OrionRoutingArtifact, OrionUpperGraphNode, OrionUpperHnswGraph};
use super::error::{OrionRoutingError, OrionRoutingResult};

pub const ORION_UPPER_HNSW_DEFAULT_M: usize = 32;
pub const ORION_UPPER_HNSW_DEFAULT_EF_CONSTRUCT: usize = 100;
pub const ORION_UPPER_HNSW_DEFAULT_SEED: u64 = 100;

/// Deterministic offline build parameters for the portable Orion upper HNSW graph.
///
/// Construction is deliberately single-threaded. Online serving uses the serialized graph and
/// never falls back to brute force, while deterministic insertion makes graph checksums suitable
/// for manifests, snapshots, and rolling deployment validation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OrionUpperGraphBuildOptions {
    pub m: usize,
    pub ef_construct: usize,
    pub seed: u64,
}

impl Default for OrionUpperGraphBuildOptions {
    fn default() -> Self {
        Self {
            m: ORION_UPPER_HNSW_DEFAULT_M,
            ef_construct: ORION_UPPER_HNSW_DEFAULT_EF_CONSTRUCT,
            seed: ORION_UPPER_HNSW_DEFAULT_SEED,
        }
    }
}

impl OrionUpperGraphBuildOptions {
    fn validate(self) -> OrionRoutingResult<Self> {
        if self.m == 0 {
            return Err(OrionRoutingError::InvalidUpperGraphBuildConfig {
                reason: "m must be greater than zero".to_string(),
            });
        }
        if self.ef_construct == 0 {
            return Err(OrionRoutingError::InvalidUpperGraphBuildConfig {
                reason: "ef_construct must be greater than zero".to_string(),
            });
        }
        Ok(self)
    }
}

impl OrionRoutingArtifact {
    /// Consume a graphless typed artifact and return the production-loadable artifact containing
    /// its deterministic upper HNSW graph.
    pub fn build_upper_hnsw(
        self,
        options: OrionUpperGraphBuildOptions,
    ) -> OrionRoutingResult<Self> {
        build_upper_hnsw_graph(self, options)
    }
}

/// Build and attach a portable upper HNSW graph to a graphless routing artifact.
///
/// The graph is built with Qdrant's production `GraphLayersBuilder` over volatile dense vector
/// storage. The volatile storage exists only during this offline operation; all graph references
/// are translated from process-local offsets back to the artifact's external upper labels before
/// returning.
pub fn build_upper_hnsw_graph(
    mut artifact: OrionRoutingArtifact,
    options: OrionUpperGraphBuildOptions,
) -> OrionRoutingResult<OrionRoutingArtifact> {
    let options = options.validate()?;
    if artifact.upper_graph.is_some() {
        return Err(OrionRoutingError::UpperGraphAlreadyPresent);
    }
    artifact.validate()?;

    let point_count = artifact.upper_nodes.len();
    let point_count_offset = PointOffsetType::try_from(point_count).map_err(|_| {
        OrionRoutingError::InvalidUpperGraphBuildConfig {
            reason: format!(
                "upper tier contains {point_count} points, exceeding the supported point-offset range"
            ),
        }
    })?;

    let dimension = artifact.vector_schema.dimension;
    let distance = artifact.vector_schema.distance;
    let mut vector_storage = new_volatile_dense_vector_storage(dimension, distance);
    let build_counter = HardwareCounterCell::disposable();
    for (point_index, node) in artifact.upper_nodes.iter().enumerate() {
        let point_id = PointOffsetType::try_from(point_index).map_err(|_| {
            OrionRoutingError::InvalidUpperGraphBuildConfig {
                reason: format!("upper point index {point_index} does not fit PointOffsetType"),
            }
        })?;
        let vector = distance.preprocess_vector::<VectorElementType>(node.vector.clone());
        vector_storage
            .insert_vector(point_id, VectorRef::from(&vector), &build_counter)
            .map_err(|source| OrionRoutingError::UpperGraphBuild {
                reason: source.to_string(),
            })?;
    }

    let mut graph_builder = GraphLayersBuilder::new(
        point_count,
        HnswM::new2(options.m),
        options.ef_construct,
        1,
        true,
    );
    let mut rng = StdRng::seed_from_u64(options.seed);
    for point_id in 0..point_count_offset {
        let level = graph_builder.get_random_layer(&mut rng);
        graph_builder.set_levels(point_id, level);
    }

    let point_deleted: BitVec = BitVec::repeat(false, point_count);
    for point_id in 0..point_count_offset {
        let scorer = FilteredScorer::new_internal(
            point_id,
            &vector_storage,
            None,
            None,
            &point_deleted,
            HardwareCounterCell::disposable(),
        )
        .map_err(|source| OrionRoutingError::UpperGraphBuild {
            reason: source.to_string(),
        })?;
        graph_builder.link_new_point(point_id, scorer);
    }

    let portable =
        graph_builder
            .export_portable()
            .map_err(|source| OrionRoutingError::UpperGraphBuild {
                reason: source.to_string(),
            })?;
    let labels = artifact
        .upper_nodes
        .iter()
        .map(|node| node.label)
        .collect::<Vec<_>>();
    let entry_point = labels[portable.entry_point as usize];
    let nodes = portable
        .neighbors_by_point_and_level
        .into_iter()
        .enumerate()
        .map(|(point_index, levels)| OrionUpperGraphNode {
            label: labels[point_index],
            neighbors_by_level: levels
                .into_iter()
                .map(|neighbors| {
                    neighbors
                        .into_iter()
                        .map(|neighbor| labels[neighbor as usize])
                        .collect()
                })
                .collect(),
        })
        .collect();

    artifact.upper_graph = Some(OrionUpperHnswGraph {
        entry_point,
        max_level: portable.max_level,
        nodes,
    });
    artifact.validate()?;
    Ok(artifact)
}

#[cfg(test)]
mod tests {
    use segment::types::{Distance, ExtendedPointId};

    use super::*;
    use crate::orion::{
        ORION_ROUTING_ARTIFACT_FORMAT_VERSION, OrionRouter, OrionUpperNode, OrionVectorDatatype,
        OrionVectorSchemaFingerprint,
    };

    fn id(value: u64) -> ExtendedPointId {
        ExtendedPointId::NumId(value)
    }

    fn graphless_artifact() -> OrionRoutingArtifact {
        OrionRoutingArtifact {
            format_version: ORION_ROUTING_ARTIFACT_FORMAT_VERSION,
            generation: 11,
            vector_schema: OrionVectorSchemaFingerprint {
                vector_name: String::new(),
                dimension: 2,
                distance: Distance::Euclid,
                datatype: OrionVectorDatatype::Float32,
            },
            shard_count: 4,
            layout_sha256: "a".repeat(64),
            logical_point_count: 16,
            physical_point_count: 16,
            upper_k: 3,
            upper_ef_search: 8,
            dynamic_ef_base: 20,
            dynamic_ef_factor: 4,
            upper_nodes: (0..16)
                .map(|index| OrionUpperNode {
                    label: id(1_000 + index),
                    vector: vec![index as f32, (index % 3) as f32],
                    shard_membership: vec![(index % 4) as u32],
                })
                .collect(),
            upper_graph: None,
        }
    }

    #[test]
    fn graph_build_is_deterministic_and_canonical() {
        let options = OrionUpperGraphBuildOptions {
            m: 4,
            ef_construct: 16,
            seed: 100,
        };
        let first = build_upper_hnsw_graph(graphless_artifact(), options).unwrap();
        let second = build_upper_hnsw_graph(graphless_artifact(), options).unwrap();

        assert_eq!(first.upper_graph, second.upper_graph);
        assert_eq!(
            first.canonical_json_bytes().unwrap(),
            second.canonical_json_bytes().unwrap()
        );
        assert_eq!(
            first.canonical_sha256().unwrap(),
            second.canonical_sha256().unwrap()
        );
    }

    #[test]
    fn built_graph_validates_and_production_router_loads_it() {
        let artifact = build_upper_hnsw_graph(
            graphless_artifact(),
            OrionUpperGraphBuildOptions {
                m: 4,
                ef_construct: 16,
                seed: 100,
            },
        )
        .unwrap();

        artifact.validate().unwrap();
        let graph = artifact.upper_graph.as_ref().unwrap();
        assert_eq!(graph.nodes.len(), artifact.upper_nodes.len());
        assert_eq!(
            graph
                .nodes
                .iter()
                .find(|node| node.label == graph.entry_point)
                .unwrap()
                .neighbors_by_level
                .len()
                - 1,
            graph.max_level,
        );

        let router = OrionRouter::new(artifact).unwrap();
        let hits = router.search_upper(&[0.25, 0.0]).unwrap();
        assert_eq!(hits.len(), 3);
        let targets = router.route_query(&[0.25, 0.0]).unwrap();
        assert!(!targets.is_empty());
        assert!(targets.iter().all(|target| target.shard_id < 4));
    }

    #[test]
    fn builder_refuses_to_replace_an_existing_graph() {
        let built = build_upper_hnsw_graph(
            graphless_artifact(),
            OrionUpperGraphBuildOptions {
                m: 4,
                ef_construct: 16,
                seed: 100,
            },
        )
        .unwrap();
        assert!(matches!(
            build_upper_hnsw_graph(built, OrionUpperGraphBuildOptions::default()),
            Err(OrionRoutingError::UpperGraphAlreadyPresent)
        ));
    }
}
