use std::collections::HashSet;
use std::num::NonZeroU32;
use std::sync::Arc;
use std::time::Duration;

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
use crate::operations::types::PointRequestInternal;
use crate::operations::types::VectorsConfig;
use crate::operations::universal_query::shard_query::{ScoringQuery, ShardQueryRequest};
use crate::operations::vector_params_builder::VectorParamsBuilder;
use crate::operations::{CollectionUpdateOperations, OperationWithClockTag};
use crate::optimizers_builder::OptimizersConfig;
use crate::orion::{
    ORION_ROUTING_ARTIFACT_FORMAT_VERSION, OrionRouter, OrionRoutingArtifact, OrionUpperGraphNode,
    OrionUpperHnswGraph, OrionUpperNode, OrionVectorDatatype, OrionVectorSchemaFingerprint,
};
use crate::shards::channel_service::ChannelService;
use crate::shards::collection_shard_distribution::CollectionShardDistribution;
use crate::shards::replica_set::replica_set_state::ReplicaState;
use crate::shards::shard_trait::WaitUntil;
use crate::tests::snapshot_test::{
    dummy_abort_shard_transfer, dummy_on_replica_failure, dummy_request_shard_transfer,
};

const PEER_ID: u64 = 1;
const SHARD_COUNT: u32 = 2;

fn artifact() -> OrionRoutingArtifact {
    OrionRoutingArtifact {
        format_version: ORION_ROUTING_ARTIFACT_FORMAT_VERSION,
        generation: 1,
        vector_schema: OrionVectorSchemaFingerprint {
            vector_name: String::new(),
            dimension: 2,
            distance: Distance::Dot,
            datatype: OrionVectorDatatype::Float32,
        },
        shard_count: SHARD_COUNT,
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
                vector: vec![0.0, 1.0],
                shard_membership: vec![0],
            },
            OrionUpperNode {
                label: 20_u64.into(),
                vector: vec![1.0, 0.0],
                shard_membership: vec![1],
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
        vectors: VectorsConfig::Single(VectorParamsBuilder::new(2, Distance::Dot).build()),
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

    let collection_dir = Builder::new().prefix("orion_native").tempdir().unwrap();
    let snapshots_path = Builder::new().prefix("orion_snapshots").tempdir().unwrap();
    let shards = (0..SHARD_COUNT)
        .map(|shard_id| (shard_id, HashSet::from([PEER_ID])))
        .collect::<AHashMap<_, _>>();
    let mut collection = Collection::new(
        "orion-native-test".to_string(),
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

    collection.orion_router = Some(Arc::new(OrionRouter::new(artifact()).unwrap()));
    (collection, collection_dir, snapshots_path)
}

fn core_request(exact: bool) -> shard::search::CoreSearchRequest {
    core_request_for_vector(vec![0.0, 1.0], exact)
}

fn core_request_for_vector(vector: Vec<f32>, exact: bool) -> shard::search::CoreSearchRequest {
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

fn compact_lower_request(vector: Vec<f32>, entry_point: u64) -> shard::search::CoreSearchRequest {
    let mut request = core_request_for_vector(vector, false);
    request.params = Some(SearchParams {
        hnsw_ef: Some(32),
        ..Default::default()
    });
    request.hnsw_entry_points = Some(vec![entry_point.into()]);
    request.with_payload = Some(WithPayloadInterface::Bool(false));
    request.with_vector = Some(WithVector::Bool(false));
    request
}

#[tokio::test(flavor = "multi_thread")]
async fn prepared_local_read_ticket_retains_recovery_and_local_guards() {
    let (collection, _collection_dir, _snapshots_path) = fixture().await;
    let shard_holder = collection.shards_holder.read().await;
    let shard = shard_holder.get_shard(0).unwrap();

    assert!(shard.local_write_lock_available_for_test());
    let ticket = shard.prepare_local_read_ticket().await.unwrap();
    fn assert_send<T: Send>(_: &T) {}
    assert_send(&ticket);

    assert!(!shard.local_write_lock_available_for_test());
    assert!(
        tokio::time::timeout(
            Duration::from_millis(20),
            shard.partial_snapshot_meta.take_search_write_lock(),
        )
        .await
        .is_err(),
        "partial snapshot recovery must wait while a prepared read ticket exists",
    );

    drop(ticket);
    assert!(shard.local_write_lock_available_for_test());
    let recovery_guard = tokio::time::timeout(
        Duration::from_secs(1),
        shard.partial_snapshot_meta.take_search_write_lock(),
    )
    .await
    .expect("dropping the ticket must release its recovery read guard");
    drop(recovery_guard);
}

#[tokio::test(flavor = "multi_thread")]
async fn explicit_local_shard_batches_match_ordinary_replica_set_searches() {
    let (collection, _collection_dir, _snapshots_path) = fixture().await;
    let shard_requests = vec![
        (
            1,
            Arc::new(shard::search::CoreSearchRequestBatch {
                searches: vec![
                    compact_lower_request(vec![0.0, 1.0], 200),
                    compact_lower_request(vec![1.0, 0.0], 200),
                ],
            }),
        ),
        (
            0,
            Arc::new(shard::search::CoreSearchRequestBatch {
                searches: vec![
                    compact_lower_request(vec![1.0, 0.0], 100),
                    compact_lower_request(vec![0.0, 1.0], 100),
                ],
            }),
        ),
    ];

    let mut expected = Vec::new();
    for (shard_id, request) in &shard_requests {
        expected.push(
            collection
                .core_search_batch(
                    request.as_ref().clone(),
                    None,
                    ShardSelectorInternal::ShardId(*shard_id),
                    None,
                    HwMeasurementAcc::new(),
                )
                .await
                .unwrap(),
        );
    }

    let fast_path_hw = HwMeasurementAcc::new();
    let actual = collection
        .core_search_batch_explicit_local_shards(&shard_requests, None, fast_path_hw.clone())
        .await
        .unwrap()
        .expect("compact lower requests must use the explicit-shard fast path");

    let actual_rows = actual
        .iter()
        .map(|(_, rows)| rows.clone())
        .collect::<Vec<_>>();
    assert_eq!(actual_rows, expected);
    assert_eq!(actual[0].0, 1);
    assert_eq!(actual[1].0, 0);
    assert_eq!(actual[0].1[0][0].id, 200_u64.into());
    assert_eq!(actual[1].1[0][0].id, 100_u64.into());
    assert!(
        fast_path_hw.get_cpu() > 0,
        "prepared local reads must retain hardware accounting",
    );

    let timed_out_hw = HwMeasurementAcc::new();
    let timed_out = collection
        .core_search_batch_explicit_local_shards(
            &shard_requests,
            Some(Duration::ZERO),
            timed_out_hw.clone(),
        )
        .await
        .unwrap_err();
    assert!(matches!(
        timed_out,
        crate::operations::types::CollectionError::Timeout { .. }
    ));
    assert_eq!(
        timed_out_hw.get_cpu(),
        0,
        "an exhausted preflight timeout must fail before shard search",
    );

    let mut metadata_request = compact_lower_request(vec![1.0, 0.0], 100);
    metadata_request.with_payload = Some(WithPayloadInterface::Bool(true));
    let fallback = collection
        .core_search_batch_explicit_local_shards(
            &[(
                0,
                Arc::new(shard::search::CoreSearchRequestBatch {
                    searches: vec![metadata_request.clone()],
                }),
            )],
            None,
            HwMeasurementAcc::new(),
        )
        .await
        .unwrap();
    assert!(fallback.is_none());
    {
        let shard_holder = collection.shards_holder.read().await;
        assert!(
            shard_holder
                .get_shard(0)
                .unwrap()
                .local_write_lock_available_for_test(),
            "ineligible fallback must release its prepared local-read ticket",
        );
    }

    let missing_target = collection
        .core_search_batch_explicit_local_shards(
            &[
                (
                    0,
                    Arc::new(shard::search::CoreSearchRequestBatch {
                        searches: vec![metadata_request],
                    }),
                ),
                (
                    99,
                    Arc::new(shard::search::CoreSearchRequestBatch {
                        searches: vec![compact_lower_request(vec![1.0, 0.0], 100)],
                    }),
                ),
            ],
            None,
            HwMeasurementAcc::new(),
        )
        .await
        .unwrap_err();
    assert!(matches!(
        missing_target,
        crate::operations::types::CollectionError::NotFound { .. }
    ));

    let missing_target_hw = HwMeasurementAcc::new();
    let eligible_missing_target = collection
        .core_search_batch_explicit_local_shards(
            &[
                (
                    0,
                    Arc::new(shard::search::CoreSearchRequestBatch {
                        searches: vec![compact_lower_request(vec![1.0, 0.0], 100)],
                    }),
                ),
                (
                    99,
                    Arc::new(shard::search::CoreSearchRequestBatch {
                        searches: vec![compact_lower_request(vec![1.0, 0.0], 100)],
                    }),
                ),
            ],
            None,
            missing_target_hw.clone(),
        )
        .await
        .unwrap_err();
    assert!(matches!(
        eligible_missing_target,
        crate::operations::types::CollectionError::NotFound { .. }
    ));
    assert_eq!(missing_target_hw.get_cpu(), 0);

    let mut enormous_limit = compact_lower_request(vec![1.0, 0.0], 100);
    enormous_limit.limit = usize::MAX;
    let overflow = collection
        .core_search_batch_explicit_local_shards(
            &[(
                0,
                Arc::new(shard::search::CoreSearchRequestBatch {
                    searches: vec![enormous_limit, compact_lower_request(vec![1.0, 0.0], 100)],
                }),
            )],
            None,
            HwMeasurementAcc::new(),
        )
        .await
        .unwrap_err();
    assert!(
        overflow
            .to_string()
            .contains("batch search limits overflow usize")
    );

    let duplicate_target = collection
        .core_search_batch_explicit_local_shards(
            &[
                (
                    0,
                    Arc::new(shard::search::CoreSearchRequestBatch {
                        searches: vec![compact_lower_request(vec![1.0, 0.0], 100)],
                    }),
                ),
                (
                    0,
                    Arc::new(shard::search::CoreSearchRequestBatch {
                        searches: vec![compact_lower_request(vec![0.0, 1.0], 100)],
                    }),
                ),
            ],
            None,
            HwMeasurementAcc::new(),
        )
        .await
        .unwrap_err();
    assert!(
        duplicate_target
            .to_string()
            .contains("duplicate explicit local shard batch for shard 0")
    );

    let recovering_shard_holder = collection.shards_holder.read().await;
    let recovery_guard = recovering_shard_holder
        .get_shard(1)
        .unwrap()
        .partial_snapshot_meta
        .take_search_write_lock()
        .await;
    let recovering_target_hw = HwMeasurementAcc::new();
    let recovering_target = collection
        .core_search_batch_explicit_local_shards(
            &[
                (
                    0,
                    Arc::new(shard::search::CoreSearchRequestBatch {
                        searches: vec![compact_lower_request(vec![1.0, 0.0], 100)],
                    }),
                ),
                (
                    1,
                    Arc::new(shard::search::CoreSearchRequestBatch {
                        searches: vec![compact_lower_request(vec![0.0, 1.0], 200)],
                    }),
                ),
            ],
            None,
            recovering_target_hw.clone(),
        )
        .await
        .unwrap_err();
    assert!(
        recovering_target
            .to_string()
            .contains("partial snapshot recovery is in progress")
    );
    assert_eq!(recovering_target_hw.get_cpu(), 0);
    drop(recovery_guard);
    drop(recovering_shard_holder);

    {
        let shard_holder = collection.shards_holder.read().await;
        shard_holder
            .get_shard(1)
            .unwrap()
            .remove_local()
            .await
            .unwrap();
    }
    let non_local_target_hw = HwMeasurementAcc::new();
    let non_local_target = collection
        .core_search_batch_explicit_local_shards(
            &[
                (
                    0,
                    Arc::new(shard::search::CoreSearchRequestBatch {
                        searches: vec![compact_lower_request(vec![1.0, 0.0], 100)],
                    }),
                ),
                (
                    1,
                    Arc::new(shard::search::CoreSearchRequestBatch {
                        searches: vec![compact_lower_request(vec![0.0, 1.0], 200)],
                    }),
                ),
            ],
            None,
            non_local_target_hw.clone(),
        )
        .await
        .unwrap_err();
    assert!(
        non_local_target
            .to_string()
            .contains("Local shard 1 not found")
    );
    assert_eq!(non_local_target_hw.get_cpu(), 0);
}

#[tokio::test(flavor = "multi_thread")]
async fn native_orion_routes_standard_search_and_query_through_numeric_replica_sets() {
    let (collection, _collection_dir, _snapshots_path) = fixture().await;

    let routed = collection
        .core_search_batch(
            shard::search::CoreSearchRequestBatch {
                searches: vec![core_request(false)],
            },
            None,
            ShardSelectorInternal::All,
            None,
            HwMeasurementAcc::new(),
        )
        .await
        .unwrap();
    assert_eq!(routed[0][0].id, 100_u64.into());

    let routed_batch = collection
        .core_search_batch(
            shard::search::CoreSearchRequestBatch {
                searches: vec![
                    core_request_for_vector(vec![0.0, 1.0], false),
                    core_request_for_vector(vec![1.0, 0.0], false),
                ],
            },
            None,
            ShardSelectorInternal::All,
            None,
            HwMeasurementAcc::new(),
        )
        .await
        .unwrap();
    assert_eq!(routed_batch.len(), 2);
    assert_eq!(routed_batch[0][0].id, 100_u64.into());
    assert_eq!(routed_batch[1][0].id, 200_u64.into());

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
    assert_eq!(query_result[0].id, 100_u64.into());
}

#[tokio::test(flavor = "multi_thread")]
async fn exact_and_explicit_shard_reads_bypass_orion() {
    let (collection, _collection_dir, _snapshots_path) = fixture().await;

    let exact = collection
        .core_search_batch(
            shard::search::CoreSearchRequestBatch {
                searches: vec![core_request(true)],
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
                searches: vec![core_request(false)],
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
async fn static_orion_policy_rejects_hash_routed_client_writes() {
    let (collection, _collection_dir, _snapshots_path) = fixture().await;
    collection.collection_config.write().await.auto_shard_policy = Some(AutoShardPolicy::Orion {
        generation: 1,
        artifact_sha256: "0".repeat(64),
    });

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
            .contains("rejected instead of silently using point-ID hash sharding")
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn configured_orion_policy_fails_closed_when_router_is_unavailable() {
    let (mut collection, _collection_dir, _snapshots_path) = fixture().await;
    collection.collection_config.write().await.auto_shard_policy = Some(AutoShardPolicy::Orion {
        generation: 1,
        artifact_sha256: "0".repeat(64),
    });
    collection.orion_router = None;

    let search_error = collection
        .core_search_batch(
            shard::search::CoreSearchRequestBatch {
                searches: vec![core_request(false)],
            },
            None,
            ShardSelectorInternal::All,
            None,
            HwMeasurementAcc::new(),
        )
        .await
        .unwrap_err();
    assert!(
        search_error
            .to_string()
            .contains("refusing a silent all-shards coordinator fallback")
    );

    let query_error = collection
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
        .unwrap_err();
    assert!(
        query_error
            .to_string()
            .contains("refusing a silent all-shards coordinator fallback")
    );

    let explicit = collection
        .core_search_batch(
            shard::search::CoreSearchRequestBatch {
                searches: vec![core_request(false)],
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
async fn eligible_orion_route_failure_refuses_silent_all_shards_fallback() {
    let (collection, _collection_dir, _snapshots_path) = fixture().await;

    let error = collection
        .core_search_batch(
            shard::search::CoreSearchRequestBatch {
                searches: vec![core_request_for_vector(vec![0.0, 1.0, 2.0], false)],
            },
            None,
            ShardSelectorInternal::All,
            None,
            HwMeasurementAcc::new(),
        )
        .await
        .unwrap_err();

    let message = error.to_string();
    assert!(message.contains("Orion routing generation 1 failed"));
    assert!(message.contains("refusing a silent all-shards fallback"));
}

#[tokio::test(flavor = "multi_thread")]
async fn controlled_numeric_shard_import_keeps_the_same_external_id_in_two_shards() {
    let (collection, _collection_dir, _snapshots_path) = fixture().await;
    collection.collection_config.write().await.auto_shard_policy = Some(AutoShardPolicy::Orion {
        generation: 1,
        artifact_sha256: "0".repeat(64),
    });

    let external_id = 300_u64;
    for shard_id in 0..SHARD_COUNT {
        collection
            .update_from_peer(
                OperationWithClockTag::from(point_operation(external_id, vec![0.5, 0.5])),
                shard_id,
                WaitUntil::Visible,
                None,
                WriteOrdering::Medium,
                HwMeasurementAcc::new(),
            )
            .await
            .unwrap();
    }

    for shard_id in 0..SHARD_COUNT {
        let records = collection
            .retrieve(
                PointRequestInternal {
                    ids: vec![external_id.into()],
                    with_payload: Some(false.into()),
                    with_vector: false.into(),
                },
                None,
                &ShardSelectorInternal::ShardId(shard_id),
                None,
                HwMeasurementAcc::new(),
            )
            .await
            .unwrap();
        assert_eq!(records.len(), 1, "shard {shard_id} must contain the copy");
        assert_eq!(records[0].id, external_id.into());
    }

    let error = collection
        .update_from_client(
            point_operation(301, vec![1.0, 1.0]),
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
            .contains("rejected instead of silently using point-ID hash sharding")
    );
}
