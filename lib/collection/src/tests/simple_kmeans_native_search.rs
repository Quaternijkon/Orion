use std::collections::HashSet;
use std::num::NonZeroU32;
use std::sync::Arc;

use ahash::AHashMap;
use common::budget::ResourceBudget;
use common::counter::hardware_accumulator::HwMeasurementAcc;
use segment::data_types::vectors::NamedQuery;
use segment::types::{Distance, SearchParams, WithPayloadInterface, WithVector};
use shard::query::query_enum::QueryEnum;
use tempfile::{Builder, TempDir};

use crate::collection::Collection;
use crate::config::{AutoShardPolicy, CollectionConfigInternal, CollectionParams, WalConfig};
use crate::operations::point_ops::{
    PointInsertOperationsInternal, PointOperations, PointStructPersisted, VectorStructPersisted,
    WriteOrdering,
};
use crate::operations::shard_selector_internal::ShardSelectorInternal;
use crate::operations::shared_storage_config::SharedStorageConfig;
use crate::operations::universal_query::shard_query::{ScoringQuery, ShardQueryRequest};
use crate::operations::vector_params_builder::VectorParamsBuilder;
use crate::operations::{CollectionUpdateOperations, OperationWithClockTag};
use crate::optimizers_builder::OptimizersConfig;
use crate::shards::channel_service::ChannelService;
use crate::shards::collection_shard_distribution::CollectionShardDistribution;
use crate::shards::replica_set::replica_set_state::ReplicaState;
use crate::shards::shard_trait::WaitUntil;
use crate::simple_kmeans::{
    SIMPLE_KMEANS_ROUTING_ARTIFACT_FORMAT_VERSION, SimpleKmeansCentroid, SimpleKmeansRouter,
    SimpleKmeansRoutingArtifact, SimpleKmeansRoutingDistance, SimpleKmeansVectorDatatype,
    SimpleKmeansVectorSchemaFingerprint,
};
use crate::tests::snapshot_test::{
    dummy_abort_shard_transfer, dummy_on_replica_failure, dummy_request_shard_transfer,
};

const PEER_ID: u64 = 1;
const SHARD_COUNT: u32 = 2;

fn artifact() -> SimpleKmeansRoutingArtifact {
    SimpleKmeansRoutingArtifact {
        format_version: SIMPLE_KMEANS_ROUTING_ARTIFACT_FORMAT_VERSION,
        generation: 1,
        vector_schema: SimpleKmeansVectorSchemaFingerprint {
            vector_name: String::new(),
            dimension: 2,
            distance: Distance::Dot,
            datatype: SimpleKmeansVectorDatatype::Float32,
        },
        shard_count: SHARD_COUNT,
        layout_sha256: "b".repeat(64),
        logical_point_count: 2,
        physical_point_count: 2,
        routing_distance: SimpleKmeansRoutingDistance::SquaredL2,
        nprobe: 1,
        lower_hnsw_ef: 64,
        centroids: vec![
            SimpleKmeansCentroid {
                shard_id: 0,
                vector: vec![1.0, 0.0],
            },
            SimpleKmeansCentroid {
                shard_id: 1,
                vector: vec![0.0, 1.0],
            },
        ],
    }
}

fn point_operation(id: u64, vector: Vec<f32>) -> CollectionUpdateOperations {
    CollectionUpdateOperations::PointOperation(PointOperations::UpsertPoints(
        PointInsertOperationsInternal::PointsList(vec![PointStructPersisted {
            id: id.into(),
            vector: VectorStructPersisted::Single(vector),
            payload: None,
        }]),
    ))
}

async fn fixture() -> (Collection, TempDir, TempDir) {
    let collection_params = CollectionParams {
        vectors: crate::operations::types::VectorsConfig::Single(
            VectorParamsBuilder::new(2, Distance::Dot).build(),
        ),
        shard_number: NonZeroU32::new(SHARD_COUNT).unwrap(),
        replication_factor: NonZeroU32::new(1).unwrap(),
        write_consistency_factor: NonZeroU32::new(1).unwrap(),
        ..CollectionParams::empty()
    };
    let config = CollectionConfigInternal {
        params: collection_params,
        optimizer_config: OptimizersConfig::fixture(),
        wal_config: WalConfig {
            wal_capacity_mb: 1,
            wal_segments_ahead: 0,
            wal_retain_closed: 1,
        },
        hnsw_config: Default::default(),
        quantization_config: None,
        strict_mode_config: None,
        uuid: None,
        metadata: None,
        auto_shard_policy: None,
    };

    let collection_dir = Builder::new()
        .prefix("simple_kmeans_native")
        .tempdir()
        .unwrap();
    let snapshots_path = Builder::new()
        .prefix("simple_kmeans_snapshots")
        .tempdir()
        .unwrap();
    let shards = (0..SHARD_COUNT)
        .map(|shard_id| (shard_id, HashSet::from([PEER_ID])))
        .collect::<AHashMap<_, _>>();
    let mut collection = Collection::new(
        "simple-kmeans-native-test".to_string(),
        PEER_ID,
        collection_dir.path(),
        snapshots_path.path(),
        &config,
        Arc::new(SharedStorageConfig::default()),
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

    for (shard_id, shard) in collection.shards_holder.write().await.get_shards() {
        let operation = match shard_id {
            0 => point_operation(100, vec![1.0, 0.0]),
            1 => point_operation(200, vec![0.0, 1.0]),
            _ => unreachable!(),
        };
        shard
            .update_local(
                OperationWithClockTag::from(operation),
                WaitUntil::Visible,
                None,
                HwMeasurementAcc::new(),
                false,
            )
            .await
            .unwrap();
    }
    for shard_id in 0..SHARD_COUNT {
        collection
            .set_shard_replica_state(shard_id, PEER_ID, ReplicaState::Active, None)
            .await
            .unwrap();
    }

    collection.simple_kmeans_router = Some(Arc::new(SimpleKmeansRouter::new(artifact()).unwrap()));
    collection.collection_config.write().await.auto_shard_policy =
        Some(AutoShardPolicy::SimpleKmeans {
            generation: 1,
            artifact_sha256: "0".repeat(64),
        });
    (collection, collection_dir, snapshots_path)
}

fn core_request(vector: Vec<f32>, exact: bool) -> shard::search::CoreSearchRequest {
    shard::search::CoreSearchRequest {
        query: QueryEnum::Nearest(NamedQuery::default_dense(vector)),
        filter: None,
        params: Some(SearchParams {
            exact,
            ..Default::default()
        }),
        hnsw_entry_points: None,
        hnsw_entry_points_by_shard: None,
        hnsw_ef_by_shard: None,
        source_id_dedup_block_size: None,
        limit: 1,
        offset: 0,
        with_payload: None,
        with_vector: None,
        score_threshold: None,
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn native_simple_kmeans_routes_standard_search_and_simple_query() {
    let (collection, _collection_dir, _snapshots_path) = fixture().await;
    let routed = collection
        .core_search_batch(
            shard::search::CoreSearchRequestBatch {
                searches: vec![
                    core_request(vec![1.0, 0.0], false),
                    core_request(vec![0.0, 1.0], false),
                ],
            },
            None,
            ShardSelectorInternal::All,
            None,
            HwMeasurementAcc::new(),
        )
        .await
        .unwrap();
    assert_eq!(routed[0][0].id, 100_u64.into());
    assert_eq!(routed[1][0].id, 200_u64.into());

    let query_result = collection
        .query(
            ShardQueryRequest {
                prefetches: vec![],
                query: Some(ScoringQuery::Vector(QueryEnum::Nearest(
                    NamedQuery::default_dense(vec![0.0, 1.0]),
                ))),
                filter: None,
                score_threshold: None,
                limit: 1,
                offset: 0,
                params: None,
                with_vector: WithVector::Bool(false),
                with_payload: WithPayloadInterface::Bool(false),
            },
            None,
            ShardSelectorInternal::All,
            None,
            HwMeasurementAcc::new(),
        )
        .await
        .unwrap();
    assert_eq!(query_result[0].id, 200_u64.into());
}

#[tokio::test(flavor = "multi_thread")]
async fn exact_and_explicit_shard_reads_bypass_simple_kmeans() {
    let (collection, _collection_dir, _snapshots_path) = fixture().await;
    let exact = collection
        .core_search_batch(
            shard::search::CoreSearchRequestBatch {
                searches: vec![core_request(vec![0.0, 1.0], true)],
            },
            None,
            ShardSelectorInternal::All,
            None,
            HwMeasurementAcc::new(),
        )
        .await
        .unwrap();
    assert_eq!(exact[0][0].id, 200_u64.into());

    let explicit = collection
        .core_search_batch(
            shard::search::CoreSearchRequestBatch {
                searches: vec![core_request(vec![1.0, 0.0], false)],
            },
            None,
            ShardSelectorInternal::ShardId(1),
            None,
            HwMeasurementAcc::new(),
        )
        .await
        .unwrap();
    assert_eq!(explicit[0][0].id, 200_u64.into());
}

#[tokio::test(flavor = "multi_thread")]
async fn static_simple_kmeans_policy_rejects_hash_routed_client_writes() {
    let (collection, _collection_dir, _snapshots_path) = fixture().await;
    let error = collection
        .update_from_client(
            point_operation(300, vec![1.0, 1.0]),
            WaitUntil::Visible,
            None,
            WriteOrdering::Weak,
            None,
            HwMeasurementAcc::new(),
        )
        .await
        .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("Simple KMeans routing generation 1")
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn configured_simple_kmeans_policy_fails_closed_when_router_is_unavailable() {
    let (mut collection, _collection_dir, _snapshots_path) = fixture().await;
    collection.simple_kmeans_router = None;

    let error = collection
        .core_search_batch(
            shard::search::CoreSearchRequestBatch {
                searches: vec![core_request(vec![1.0, 0.0], false)],
            },
            None,
            ShardSelectorInternal::All,
            None,
            HwMeasurementAcc::new(),
        )
        .await
        .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("refusing a silent all-shards coordinator fallback")
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn eligible_simple_kmeans_route_failure_refuses_silent_all_shards_fallback() {
    let (collection, _collection_dir, _snapshots_path) = fixture().await;

    let error = collection
        .core_search_batch(
            shard::search::CoreSearchRequestBatch {
                searches: vec![core_request(vec![1.0, 0.0, 2.0], false)],
            },
            None,
            ShardSelectorInternal::All,
            None,
            HwMeasurementAcc::new(),
        )
        .await
        .unwrap_err();

    let message = error.to_string();
    assert!(message.contains("Simple KMeans routing generation 1 failed"));
    assert!(message.contains("refusing a silent all-shards fallback"));
}
