use segment::types::{Distance, ExtendedPointId};

use super::Collection;
use crate::config::{AutoShardPolicy, ShardingMethod};
use crate::orion::{
    ORION_ROUTING_ARTIFACT_FORMAT_VERSION, OrionRoutingArtifact, OrionUpperGraphNode,
    OrionUpperHnswGraph, OrionUpperNode, OrionVectorDatatype, OrionVectorSchemaFingerprint,
    routing_artifact_path,
};
use crate::tests::fixtures::create_collection_config;

fn id(value: u64) -> ExtendedPointId {
    ExtendedPointId::NumId(value)
}

fn artifact(generation: u64, shard_count: u32, dimension: usize) -> OrionRoutingArtifact {
    let last_shard = shard_count.saturating_sub(1);
    OrionRoutingArtifact {
        format_version: ORION_ROUTING_ARTIFACT_FORMAT_VERSION,
        generation,
        vector_schema: OrionVectorSchemaFingerprint {
            vector_name: "".to_string(),
            dimension,
            distance: Distance::Dot,
            datatype: OrionVectorDatatype::Float32,
        },
        shard_count,
        layout_sha256: "a".repeat(64),
        logical_point_count: 2,
        physical_point_count: 2,
        upper_k: 1,
        upper_ef_search: 2,
        dynamic_ef_base: 20,
        dynamic_ef_factor: 4,
        upper_nodes: vec![
            OrionUpperNode {
                label: id(10),
                vector: vec![0.0; dimension],
                shard_membership: vec![0],
            },
            OrionUpperNode {
                label: id(20),
                vector: vec![1.0; dimension],
                shard_membership: vec![last_shard],
            },
        ],
        upper_graph: Some(OrionUpperHnswGraph {
            entry_point: id(10),
            max_level: 0,
            nodes: vec![
                OrionUpperGraphNode {
                    label: id(10),
                    neighbors_by_level: vec![vec![id(20)]],
                },
                OrionUpperGraphNode {
                    label: id(20),
                    neighbors_by_level: vec![vec![id(10)]],
                },
            ],
        }),
    }
}

fn write_artifact(
    collection_path: &std::path::Path,
    path_generation: u64,
    artifact: &OrionRoutingArtifact,
) -> String {
    let checksum = artifact.canonical_sha256().unwrap();
    let path = routing_artifact_path(collection_path, path_generation);
    fs_err::create_dir_all(path.parent().unwrap()).unwrap();
    fs_err::write(path, serde_json::to_vec_pretty(artifact).unwrap()).unwrap();
    checksum
}

fn configure_orion(
    generation: u64,
    artifact_sha256: String,
) -> crate::config::CollectionConfigInternal {
    let mut config = create_collection_config();
    config.auto_shard_policy = Some(AutoShardPolicy::Orion {
        generation,
        artifact_sha256,
    });
    config
}

#[test]
fn loader_ignores_missing_or_hash_all_policy() {
    let collection_path = tempfile::tempdir().unwrap();
    let config = create_collection_config();
    assert!(
        Collection::try_load_orion_router(collection_path.path(), &config, 2)
            .unwrap()
            .is_none()
    );

    let mut config = create_collection_config();
    config.auto_shard_policy = Some(AutoShardPolicy::HashAll);
    assert!(
        Collection::try_load_orion_router(collection_path.path(), &config, 2)
            .unwrap()
            .is_none()
    );
}

#[test]
fn loader_accepts_matching_artifact_generation_schema_and_shard_count() {
    let collection_path = tempfile::tempdir().unwrap();
    let artifact = artifact(7, 2, 4);
    let checksum = write_artifact(collection_path.path(), 7, &artifact);
    let config = configure_orion(7, checksum);

    let router = Collection::try_load_orion_router(collection_path.path(), &config, 2)
        .unwrap()
        .unwrap();
    assert_eq!(router.generation(), 7);
    assert_eq!(router.shard_count(), 2);
    assert_eq!(router.vector_name(), "");
}

#[test]
fn loader_rejects_missing_artifact_and_checksum_mismatch() {
    let collection_path = tempfile::tempdir().unwrap();
    let config = configure_orion(7, "0".repeat(64));
    let error = Collection::try_load_orion_router(collection_path.path(), &config, 2).unwrap_err();
    assert!(
        error.contains("failed to read Orion routing artifact"),
        "{error}"
    );

    let artifact = artifact(7, 2, 4);
    write_artifact(collection_path.path(), 7, &artifact);
    let config = configure_orion(7, "0".repeat(64));
    let error = Collection::try_load_orion_router(collection_path.path(), &config, 2).unwrap_err();
    assert!(error.contains("artifact checksum mismatch"), "{error}");
}

#[test]
fn loader_rejects_generation_schema_and_shard_count_mismatches() {
    let collection_path = tempfile::tempdir().unwrap();
    let generation_mismatch = artifact(7, 2, 4);
    let checksum = write_artifact(collection_path.path(), 8, &generation_mismatch);
    let config = configure_orion(8, checksum);
    let error = Collection::try_load_orion_router(collection_path.path(), &config, 2).unwrap_err();
    assert!(error.contains("generation mismatch"), "{error}");

    let schema_mismatch = artifact(9, 2, 3);
    let checksum = write_artifact(collection_path.path(), 9, &schema_mismatch);
    let config = configure_orion(9, checksum);
    let error = Collection::try_load_orion_router(collection_path.path(), &config, 2).unwrap_err();
    assert!(error.contains("schema fingerprint mismatch"), "{error}");

    let shard_count_mismatch = artifact(10, 3, 4);
    let checksum = write_artifact(collection_path.path(), 10, &shard_count_mismatch);
    let config = configure_orion(10, checksum);
    let error = Collection::try_load_orion_router(collection_path.path(), &config, 2).unwrap_err();
    assert!(error.contains("shard-count mismatch"), "{error}");
}

#[test]
fn loader_rejects_orion_policy_on_custom_sharding() {
    let collection_path = tempfile::tempdir().unwrap();
    let artifact = artifact(12, 2, 4);
    let checksum = write_artifact(collection_path.path(), 12, &artifact);
    let mut config = configure_orion(12, checksum);
    config.params.sharding_method = Some(ShardingMethod::Custom);

    let error = Collection::try_load_orion_router(collection_path.path(), &config, 2).unwrap_err();
    assert!(
        error.contains("requires an automatically sharded collection"),
        "{error}"
    );
}
