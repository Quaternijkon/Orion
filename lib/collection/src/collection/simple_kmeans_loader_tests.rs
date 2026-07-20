use segment::types::Distance;

use super::Collection;
use crate::config::{AutoShardPolicy, ShardingMethod};
use crate::simple_kmeans::{
    SIMPLE_KMEANS_ROUTING_ARTIFACT_FORMAT_VERSION, SimpleKmeansCentroid,
    SimpleKmeansRoutingArtifact, SimpleKmeansRoutingDistance, SimpleKmeansVectorDatatype,
    SimpleKmeansVectorSchemaFingerprint, routing_artifact_path,
};
use crate::tests::fixtures::create_collection_config;

fn artifact(generation: u64, shard_count: u32, dimension: usize) -> SimpleKmeansRoutingArtifact {
    SimpleKmeansRoutingArtifact {
        format_version: SIMPLE_KMEANS_ROUTING_ARTIFACT_FORMAT_VERSION,
        generation,
        vector_schema: SimpleKmeansVectorSchemaFingerprint {
            vector_name: String::new(),
            dimension,
            distance: Distance::Dot,
            datatype: SimpleKmeansVectorDatatype::Float32,
        },
        shard_count,
        layout_sha256: "b".repeat(64),
        logical_point_count: u64::from(shard_count),
        physical_point_count: u64::from(shard_count),
        routing_distance: SimpleKmeansRoutingDistance::SquaredL2,
        nprobe: 1,
        lower_hnsw_ef: 64,
        centroids: (0..shard_count)
            .map(|shard_id| SimpleKmeansCentroid {
                shard_id,
                vector: vec![shard_id as f32; dimension],
            })
            .collect(),
    }
}

fn write_artifact(
    collection_path: &std::path::Path,
    path_generation: u64,
    artifact: &SimpleKmeansRoutingArtifact,
) -> String {
    let checksum = artifact.canonical_sha256().unwrap();
    let path = routing_artifact_path(collection_path, path_generation);
    fs_err::create_dir_all(path.parent().unwrap()).unwrap();
    fs_err::write(path, serde_json::to_vec_pretty(artifact).unwrap()).unwrap();
    checksum
}

fn configure(generation: u64, artifact_sha256: String) -> crate::config::CollectionConfigInternal {
    let mut config = create_collection_config();
    config.auto_shard_policy = Some(AutoShardPolicy::SimpleKmeans {
        generation,
        artifact_sha256,
    });
    config
}

#[test]
fn loader_ignores_other_policies_and_accepts_matching_artifact() {
    let collection_path = tempfile::tempdir().unwrap();
    let config = create_collection_config();
    assert!(
        Collection::try_load_simple_kmeans_router(collection_path.path(), &config, 2)
            .unwrap()
            .is_none()
    );

    let artifact = artifact(7, 2, 4);
    let checksum = write_artifact(collection_path.path(), 7, &artifact);
    let config = configure(7, checksum);
    let router = Collection::try_load_simple_kmeans_router(collection_path.path(), &config, 2)
        .unwrap()
        .unwrap();
    assert_eq!(router.generation(), 7);
    assert_eq!(router.shard_count(), 2);
    assert_eq!(router.vector_name(), "");
    assert_eq!(router.nprobe(), 1);
}

#[test]
fn loader_rejects_missing_checksum_generation_schema_and_shard_mismatches() {
    let collection_path = tempfile::tempdir().unwrap();
    let error = Collection::try_load_simple_kmeans_router(
        collection_path.path(),
        &configure(7, "0".repeat(64)),
        2,
    )
    .unwrap_err();
    assert!(error.contains("failed to read Simple KMeans routing artifact"));

    let value = artifact(7, 2, 4);
    write_artifact(collection_path.path(), 7, &value);
    let error = Collection::try_load_simple_kmeans_router(
        collection_path.path(),
        &configure(7, "0".repeat(64)),
        2,
    )
    .unwrap_err();
    assert!(error.contains("artifact checksum mismatch"), "{error}");

    let value = artifact(7, 2, 4);
    let checksum = write_artifact(collection_path.path(), 8, &value);
    let error = Collection::try_load_simple_kmeans_router(
        collection_path.path(),
        &configure(8, checksum),
        2,
    )
    .unwrap_err();
    assert!(error.contains("generation mismatch"), "{error}");

    let value = artifact(9, 2, 3);
    let checksum = write_artifact(collection_path.path(), 9, &value);
    let error = Collection::try_load_simple_kmeans_router(
        collection_path.path(),
        &configure(9, checksum),
        2,
    )
    .unwrap_err();
    assert!(error.contains("schema fingerprint mismatch"), "{error}");

    let value = artifact(10, 3, 4);
    let checksum = write_artifact(collection_path.path(), 10, &value);
    let error = Collection::try_load_simple_kmeans_router(
        collection_path.path(),
        &configure(10, checksum),
        2,
    )
    .unwrap_err();
    assert!(error.contains("shard-count mismatch"), "{error}");
}

#[test]
fn loader_rejects_custom_sharding() {
    let collection_path = tempfile::tempdir().unwrap();
    let value = artifact(12, 2, 4);
    let checksum = write_artifact(collection_path.path(), 12, &value);
    let mut config = configure(12, checksum);
    config.params.sharding_method = Some(ShardingMethod::Custom);

    let error =
        Collection::try_load_simple_kmeans_router(collection_path.path(), &config, 2).unwrap_err();
    assert!(
        error.contains("requires an automatically sharded collection"),
        "{error}"
    );
}
