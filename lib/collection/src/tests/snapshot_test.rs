use std::collections::HashSet;
use std::num::NonZeroU32;
use std::sync::Arc;

use ahash::AHashMap;
use common::budget::ResourceBudget;
use fs_err::File;
use segment::types::Distance;
use shard::snapshots::snapshot_data::SnapshotData;
use tempfile::Builder;

use crate::collection::{Collection, RequestShardTransfer};
use crate::config::{AutoShardPolicy, CollectionConfigInternal, CollectionParams, WalConfig};
use crate::operations::shared_storage_config::SharedStorageConfig;
use crate::operations::types::{NodeType, VectorsConfig};
use crate::operations::vector_params_builder::VectorParamsBuilder;
use crate::orion::{
    ORION_ROUTING_ARTIFACT_FORMAT_VERSION, OrionRoutingArtifact, OrionUpperGraphNode,
    OrionUpperHnswGraph, OrionUpperNode, OrionVectorDatatype, OrionVectorSchemaFingerprint,
    routing_artifact_path, routing_artifact_relative_path,
};
use crate::shards::channel_service::ChannelService;
use crate::shards::collection_shard_distribution::CollectionShardDistribution;
use crate::shards::replica_set::{AbortShardTransfer, ChangePeerFromState};
use crate::simple_kmeans::{
    SIMPLE_KMEANS_ROUTING_ARTIFACT_FORMAT_VERSION, SimpleKmeansCentroid,
    SimpleKmeansRoutingArtifact, SimpleKmeansRoutingDistance, SimpleKmeansVectorDatatype,
    SimpleKmeansVectorSchemaFingerprint, routing_artifact_path as simple_kmeans_artifact_path,
    routing_artifact_relative_path as simple_kmeans_artifact_relative_path,
};
use crate::tests::fixtures::TEST_OPTIMIZERS_CONFIG;

pub fn dummy_on_replica_failure() -> ChangePeerFromState {
    Arc::new(move |_peer_id, _shard_id, _from_state| {})
}

pub fn dummy_request_shard_transfer() -> RequestShardTransfer {
    Arc::new(move |_transfer| {})
}

pub fn dummy_abort_shard_transfer() -> AbortShardTransfer {
    Arc::new(|_transfer, _reason| {})
}

fn init_logger() {
    let _ = env_logger::builder().is_test(true).try_init();
}

const ORION_SNAPSHOT_GENERATION: u64 = 7;
const SIMPLE_KMEANS_SNAPSHOT_GENERATION: u64 = 8;

#[derive(Clone, Copy)]
enum OrionArtifactFixture {
    Valid,
    Missing,
    ChecksumMismatch,
}

fn orion_snapshot_artifact() -> OrionRoutingArtifact {
    OrionRoutingArtifact {
        format_version: ORION_ROUTING_ARTIFACT_FORMAT_VERSION,
        generation: ORION_SNAPSHOT_GENERATION,
        vector_schema: OrionVectorSchemaFingerprint {
            vector_name: String::new(),
            dimension: 4,
            distance: Distance::Dot,
            datatype: OrionVectorDatatype::Float32,
        },
        shard_count: 1,
        layout_sha256: "a".repeat(64),
        logical_point_count: 2,
        physical_point_count: 2,
        upper_k: 1,
        upper_ef_search: 2,
        dynamic_ef_base: 20,
        dynamic_ef_factor: 4,
        upper_nodes: vec![
            OrionUpperNode {
                label: 10_u64.into(),
                vector: vec![1.0, 0.0, 0.0, 0.0],
                shard_membership: vec![0],
            },
            OrionUpperNode {
                label: 20_u64.into(),
                vector: vec![0.0, 1.0, 0.0, 0.0],
                shard_membership: vec![0],
            },
        ],
        upper_graph: Some(OrionUpperHnswGraph {
            entry_point: 10_u64.into(),
            max_level: 0,
            nodes: vec![
                OrionUpperGraphNode {
                    label: 10_u64.into(),
                    neighbors_by_level: vec![vec![20_u64.into()]],
                },
                OrionUpperGraphNode {
                    label: 20_u64.into(),
                    neighbors_by_level: vec![vec![10_u64.into()]],
                },
            ],
        }),
    }
}

async fn orion_snapshot_fixture(
    artifact_fixture: OrionArtifactFixture,
) -> (Collection, tempfile::TempDir, tempfile::TempDir) {
    let collection_dir = Builder::new()
        .prefix("orion_snapshot_collection")
        .tempdir()
        .unwrap();
    let snapshots_path = Builder::new()
        .prefix("orion_snapshot_storage")
        .tempdir()
        .unwrap();
    let artifact = orion_snapshot_artifact();
    let artifact_sha256 = artifact.canonical_sha256().unwrap();

    if !matches!(artifact_fixture, OrionArtifactFixture::Missing) {
        let artifact_path = routing_artifact_path(collection_dir.path(), ORION_SNAPSHOT_GENERATION);
        fs_err::create_dir_all(artifact_path.parent().unwrap()).unwrap();
        fs_err::write(
            &artifact_path,
            serde_json::to_vec_pretty(&artifact).unwrap(),
        )
        .unwrap();
    }

    let collection_params = CollectionParams {
        vectors: VectorsConfig::Single(VectorParamsBuilder::new(4, Distance::Dot).build()),
        shard_number: NonZeroU32::new(1).unwrap(),
        replication_factor: NonZeroU32::new(1).unwrap(),
        write_consistency_factor: NonZeroU32::new(1).unwrap(),
        ..CollectionParams::empty()
    };
    let config = CollectionConfigInternal {
        params: collection_params,
        optimizer_config: TEST_OPTIMIZERS_CONFIG.clone(),
        wal_config: WalConfig {
            wal_capacity_mb: 1,
            wal_segments_ahead: 0,
            wal_retain_closed: 1,
        },
        hnsw_config: Default::default(),
        quantization_config: Default::default(),
        strict_mode_config: Default::default(),
        uuid: None,
        metadata: None,
        auto_shard_policy: Some(AutoShardPolicy::Orion {
            generation: ORION_SNAPSHOT_GENERATION,
            artifact_sha256: if matches!(artifact_fixture, OrionArtifactFixture::ChecksumMismatch) {
                "0".repeat(64)
            } else {
                artifact_sha256
            },
        }),
    };
    let collection = Collection::new(
        "orion-snapshot-test".to_string(),
        1,
        collection_dir.path(),
        snapshots_path.path(),
        &config,
        Arc::new(SharedStorageConfig::default()),
        CollectionShardDistribution {
            shards: AHashMap::from([(0, HashSet::from([1]))]),
        },
        None,
        ChannelService::default(),
        dummy_on_replica_failure(),
        dummy_request_shard_transfer(),
        dummy_abort_shard_transfer(),
        None,
        None,
        ResourceBudget::default(),
        None,
    )
    .await
    .unwrap();

    (collection, collection_dir, snapshots_path)
}

fn simple_kmeans_snapshot_artifact() -> SimpleKmeansRoutingArtifact {
    SimpleKmeansRoutingArtifact {
        format_version: SIMPLE_KMEANS_ROUTING_ARTIFACT_FORMAT_VERSION,
        generation: SIMPLE_KMEANS_SNAPSHOT_GENERATION,
        vector_schema: SimpleKmeansVectorSchemaFingerprint {
            vector_name: String::new(),
            dimension: 4,
            distance: Distance::Dot,
            datatype: SimpleKmeansVectorDatatype::Float32,
        },
        shard_count: 1,
        layout_sha256: "b".repeat(64),
        logical_point_count: 2,
        physical_point_count: 2,
        routing_distance: SimpleKmeansRoutingDistance::SquaredL2,
        nprobe: 1,
        lower_hnsw_ef: 64,
        centroids: vec![SimpleKmeansCentroid {
            shard_id: 0,
            vector: vec![1.0, 0.0, 0.0, 0.0],
        }],
    }
}

async fn simple_kmeans_snapshot_fixture(
    artifact_fixture: OrionArtifactFixture,
) -> (Collection, tempfile::TempDir, tempfile::TempDir) {
    let collection_dir = Builder::new()
        .prefix("simple_kmeans_snapshot_collection")
        .tempdir()
        .unwrap();
    let snapshots_path = Builder::new()
        .prefix("simple_kmeans_snapshot_storage")
        .tempdir()
        .unwrap();
    let artifact = simple_kmeans_snapshot_artifact();
    let artifact_sha256 = artifact.canonical_sha256().unwrap();

    if !matches!(artifact_fixture, OrionArtifactFixture::Missing) {
        let artifact_path =
            simple_kmeans_artifact_path(collection_dir.path(), SIMPLE_KMEANS_SNAPSHOT_GENERATION);
        fs_err::create_dir_all(artifact_path.parent().unwrap()).unwrap();
        fs_err::write(
            &artifact_path,
            serde_json::to_vec_pretty(&artifact).unwrap(),
        )
        .unwrap();
    }

    let collection_params = CollectionParams {
        vectors: VectorsConfig::Single(VectorParamsBuilder::new(4, Distance::Dot).build()),
        shard_number: NonZeroU32::new(1).unwrap(),
        replication_factor: NonZeroU32::new(1).unwrap(),
        write_consistency_factor: NonZeroU32::new(1).unwrap(),
        ..CollectionParams::empty()
    };
    let config = CollectionConfigInternal {
        params: collection_params,
        optimizer_config: TEST_OPTIMIZERS_CONFIG.clone(),
        wal_config: WalConfig {
            wal_capacity_mb: 1,
            wal_segments_ahead: 0,
            wal_retain_closed: 1,
        },
        hnsw_config: Default::default(),
        quantization_config: Default::default(),
        strict_mode_config: Default::default(),
        uuid: None,
        metadata: None,
        auto_shard_policy: Some(AutoShardPolicy::SimpleKmeans {
            generation: SIMPLE_KMEANS_SNAPSHOT_GENERATION,
            artifact_sha256: if matches!(artifact_fixture, OrionArtifactFixture::ChecksumMismatch) {
                "0".repeat(64)
            } else {
                artifact_sha256
            },
        }),
    };
    let collection = Collection::new(
        "simple-kmeans-snapshot-test".to_string(),
        1,
        collection_dir.path(),
        snapshots_path.path(),
        &config,
        Arc::new(SharedStorageConfig::default()),
        CollectionShardDistribution {
            shards: AHashMap::from([(0, HashSet::from([1]))]),
        },
        None,
        ChannelService::default(),
        dummy_on_replica_failure(),
        dummy_request_shard_transfer(),
        dummy_abort_shard_transfer(),
        None,
        None,
        ResourceBudget::default(),
        None,
    )
    .await
    .unwrap();

    (collection, collection_dir, snapshots_path)
}

async fn _test_snapshot_collection(node_type: NodeType) {
    let wal_config = WalConfig {
        wal_capacity_mb: 1,
        wal_segments_ahead: 0,
        wal_retain_closed: 1,
    };

    let collection_params = CollectionParams {
        vectors: VectorsConfig::Single(VectorParamsBuilder::new(4, Distance::Dot).build()),
        shard_number: NonZeroU32::new(4).unwrap(),
        replication_factor: NonZeroU32::new(3).unwrap(),
        write_consistency_factor: NonZeroU32::new(2).unwrap(),
        ..CollectionParams::empty()
    };

    let config = CollectionConfigInternal {
        params: collection_params,
        optimizer_config: TEST_OPTIMIZERS_CONFIG.clone(),
        wal_config,
        hnsw_config: Default::default(),
        quantization_config: Default::default(),
        strict_mode_config: Default::default(),
        uuid: None,
        metadata: None,
        auto_shard_policy: None,
    };

    let snapshots_path = Builder::new().prefix("test_snapshots").tempdir().unwrap();
    let collection_dir = Builder::new().prefix("test_collection").tempdir().unwrap();

    let collection_name = "test".to_string();
    let collection_name_rec = "test_rec".to_string();
    let mut shards = AHashMap::new();
    shards.insert(0, HashSet::from([1]));
    shards.insert(1, HashSet::from([1]));
    shards.insert(2, HashSet::from([10_000])); // remote shard
    shards.insert(3, HashSet::from([1, 20_000, 30_000]));

    let storage_config: SharedStorageConfig = SharedStorageConfig {
        node_type,
        ..Default::default()
    };

    let collection = Collection::new(
        collection_name,
        1,
        collection_dir.path(),
        snapshots_path.path(),
        &config,
        Arc::new(storage_config),
        CollectionShardDistribution { shards },
        None,
        ChannelService::default(),
        dummy_on_replica_failure(),
        dummy_request_shard_transfer(),
        dummy_abort_shard_transfer(),
        None,
        None,
        ResourceBudget::default(),
        None,
    )
    .await
    .unwrap();

    let snapshots_temp_dir = Builder::new().prefix("temp_dir").tempdir().unwrap();
    let snapshot_description = collection
        .create_snapshot(snapshots_temp_dir.path(), 0)
        .await
        .unwrap();

    assert_eq!(snapshot_description.checksum.unwrap().len(), 64);

    {
        let recover_dir = Builder::new()
            .prefix("test_collection_rec")
            .tempdir()
            .unwrap();
        let snapshot_data = SnapshotData::new_packed_persistent(
            snapshots_path.path().join(&snapshot_description.name),
        );

        // Do not recover in local mode if some shards are remote
        assert!(
            Collection::restore_snapshot(snapshot_data, recover_dir.path(), 0, false,).is_err(),
        );
    }

    let recover_dir = Builder::new()
        .prefix("test_collection_rec")
        .tempdir()
        .unwrap();
    let snapshot_data =
        SnapshotData::new_packed_persistent(snapshots_path.path().join(&snapshot_description.name));
    if let Err(err) = Collection::restore_snapshot(snapshot_data, recover_dir.path(), 0, true) {
        panic!("Failed to restore snapshot: {err}")
    }

    let recovered_collection = Collection::load(
        collection_name_rec,
        1,
        recover_dir.path(),
        snapshots_path.path(),
        Default::default(),
        ChannelService::default(),
        dummy_on_replica_failure(),
        dummy_request_shard_transfer(),
        dummy_abort_shard_transfer(),
        None,
        None,
        ResourceBudget::default(),
        None,
    )
    .await;

    {
        let shards_holder = &recovered_collection.shards_holder.read().await;

        let replica_ser_0 = shards_holder.get_shard(0).unwrap();
        assert!(replica_ser_0.is_local().await);
        let replica_ser_1 = shards_holder.get_shard(1).unwrap();
        assert!(replica_ser_1.is_local().await);
        let replica_ser_2 = shards_holder.get_shard(2).unwrap();
        assert!(!replica_ser_2.is_local().await);
        assert_eq!(replica_ser_2.peers().len(), 1);

        let replica_ser_3 = shards_holder.get_shard(3).unwrap();

        assert!(replica_ser_3.is_local().await);
        assert_eq!(replica_ser_3.peers().len(), 3); // 2 remotes + 1 local
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn test_snapshot_collection_normal() {
    init_logger();
    _test_snapshot_collection(NodeType::Normal).await;
}

#[tokio::test(flavor = "multi_thread")]
async fn test_snapshot_collection_listener() {
    init_logger();
    _test_snapshot_collection(NodeType::Listener).await;
}

#[tokio::test(flavor = "multi_thread")]
async fn orion_snapshot_contains_versioned_routing_artifact() {
    let (collection, _collection_dir, snapshots_path) =
        orion_snapshot_fixture(OrionArtifactFixture::Valid).await;
    let snapshot_temp_dir = Builder::new()
        .prefix("orion_snapshot_temp")
        .tempdir()
        .unwrap();
    let snapshot = collection
        .create_snapshot(snapshot_temp_dir.path(), 1)
        .await
        .unwrap();

    let mut archive =
        tar::Archive::new(File::open(snapshots_path.path().join(snapshot.name)).unwrap());
    let archived_paths = archive
        .entries()
        .unwrap()
        .map(|entry| entry.unwrap().path().unwrap().into_owned())
        .collect::<Vec<_>>();
    let expected_path = routing_artifact_relative_path(ORION_SNAPSHOT_GENERATION);

    assert!(
        archived_paths.contains(&expected_path),
        "snapshot entries {archived_paths:?} did not contain {expected_path:?}",
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn orion_snapshot_rejects_missing_routing_artifact() {
    let (collection, _collection_dir, _snapshots_path) =
        orion_snapshot_fixture(OrionArtifactFixture::Missing).await;
    let snapshot_temp_dir = Builder::new()
        .prefix("orion_snapshot_temp")
        .tempdir()
        .unwrap();
    let error = collection
        .create_snapshot(snapshot_temp_dir.path(), 1)
        .await
        .unwrap_err()
        .to_string();

    assert!(
        error.contains("cannot create a self-contained Orion snapshot")
            && error.contains("failed to read Orion routing artifact"),
        "unexpected snapshot error: {error}",
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn orion_snapshot_rejects_routing_artifact_checksum_mismatch() {
    let (collection, _collection_dir, _snapshots_path) =
        orion_snapshot_fixture(OrionArtifactFixture::ChecksumMismatch).await;
    let snapshot_temp_dir = Builder::new()
        .prefix("orion_snapshot_temp")
        .tempdir()
        .unwrap();
    let error = collection
        .create_snapshot(snapshot_temp_dir.path(), 1)
        .await
        .unwrap_err()
        .to_string();

    assert!(
        error.contains("cannot create a self-contained Orion snapshot")
            && error.contains("artifact checksum mismatch"),
        "unexpected snapshot error: {error}",
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn simple_kmeans_snapshot_contains_versioned_routing_artifact() {
    let (collection, _collection_dir, snapshots_path) =
        simple_kmeans_snapshot_fixture(OrionArtifactFixture::Valid).await;
    let snapshot_temp_dir = Builder::new()
        .prefix("simple_kmeans_snapshot_temp")
        .tempdir()
        .unwrap();
    let snapshot = collection
        .create_snapshot(snapshot_temp_dir.path(), 1)
        .await
        .unwrap();

    let mut archive =
        tar::Archive::new(File::open(snapshots_path.path().join(snapshot.name)).unwrap());
    let archived_paths = archive
        .entries()
        .unwrap()
        .map(|entry| entry.unwrap().path().unwrap().into_owned())
        .collect::<Vec<_>>();
    let expected_path = simple_kmeans_artifact_relative_path(SIMPLE_KMEANS_SNAPSHOT_GENERATION);
    assert!(
        archived_paths.contains(&expected_path),
        "snapshot entries {archived_paths:?} did not contain {expected_path:?}",
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn simple_kmeans_snapshot_rejects_missing_or_mismatched_artifact() {
    for (fixture, expected) in [
        (
            OrionArtifactFixture::Missing,
            "failed to read Simple KMeans routing artifact",
        ),
        (
            OrionArtifactFixture::ChecksumMismatch,
            "artifact checksum mismatch",
        ),
    ] {
        let (collection, _collection_dir, _snapshots_path) =
            simple_kmeans_snapshot_fixture(fixture).await;
        let snapshot_temp_dir = Builder::new()
            .prefix("simple_kmeans_snapshot_temp")
            .tempdir()
            .unwrap();
        let error = collection
            .create_snapshot(snapshot_temp_dir.path(), 1)
            .await
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("cannot create a self-contained Simple KMeans snapshot")
                && error.contains(expected),
            "unexpected snapshot error: {error}",
        );
    }
}
