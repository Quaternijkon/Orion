use super::{
    bundle,
    layout::{LayoutOptions, VectorEncoding, VectorInput},
    lifecycle::{ControlCommand, FileBinding, Phase, generation_path},
    membership,
};
use crate::collection::Collection;
use crate::config::{AutoShardPolicy, CollectionConfigInternal, CollectionParams, WalConfig};
use crate::operations::CollectionUpdateOperations;
use crate::operations::point_ops::{
    PointInsertOperationsInternal, PointOperations, PointStructPersisted, VectorStructPersisted,
};
use crate::operations::shard_selector_internal::ShardSelectorInternal;
use crate::operations::shared_storage_config::SharedStorageConfig;
use crate::operations::types::{PointRequestInternal, VectorsConfig};
use crate::operations::vector_params_builder::VectorParamsBuilder;
use crate::optimizers_builder::OptimizersConfig;
use crate::shards::channel_service::ChannelService;
use crate::shards::collection_shard_distribution::CollectionShardDistribution;
use crate::shards::replica_set::replica_set_state::ReplicaState;
use crate::tests::snapshot_test::{
    dummy_abort_shard_transfer, dummy_on_replica_failure, dummy_request_shard_transfer,
};
use common::budget::ResourceBudget;
use common::counter::hardware_accumulator::HwMeasurementAcc;
use segment::types::{Distance, WithPayloadInterface, WithVector};
use std::collections::{BTreeMap, HashSet};
use std::fs::{self, File};
use std::io::Write;
use std::num::NonZeroU32;
use std::sync::Arc;
use std::time::Duration;

async fn fixture() -> (Collection, tempfile::TempDir, std::path::PathBuf) {
    fixture_mode(false).await
}

async fn fixture_mode(streaming: bool) -> (Collection, tempfile::TempDir, std::path::PathBuf) {
    fixture_encoding(streaming, VectorEncoding::F32le).await
}

async fn fixture_encoding(
    streaming: bool,
    encoding: VectorEncoding,
) -> (Collection, tempfile::TempDir, std::path::PathBuf) {
    fixture_schema(streaming, encoding, Distance::Euclid).await
}

async fn fixture_schema(
    streaming: bool,
    encoding: VectorEncoding,
    distance: Distance,
) -> (Collection, tempfile::TempDir, std::path::PathBuf) {
    let root = tempfile::tempdir().unwrap();
    let raw = root.path().join("raw.f32le");
    let mut file = File::create(&raw).unwrap();
    for i in 0..64 {
        for x in [(i % 16) as f32 / 8.0 - 1.0, (i / 16) as f32 - 1.5] {
            if encoding == VectorEncoding::U8 {
                file.write_all(&[((x + 2.0) * 32.0) as u8]).unwrap();
            } else {
                file.write_all(&x.to_le_bytes()).unwrap();
            }
        }
    }
    let output = root.path().join("bundle");
    let request = bundle::BuildRequest {
        input: VectorInput {
            format_version: 1,
            encoding,
            path: raw.clone(),
            rows: 64,
            dimension: 2,
            sha256: bundle::file_sha256(&raw).unwrap(),
        },
        options: LayoutOptions {
            generation: 1,
            shard_count: 2,
            distance,
            vector_name: String::new(),
            sample_denominator: 4,
            sample_seed: 100,
            upper_m: 4,
            upper_ef_construct: 32,
            upper_seed: 100,
            efs: 3,
            upper_k: 4,
            upper_ef_search: 4,
            dynamic_ef_base: 20,
            dynamic_ef_factor: 4,
            kmeans_iterations: 10,
            kmeans_seed: 1,
            topology_iterations: 50,
            bundle_prefix: "import".into(),
        },
    };
    bundle::build(&request, &output).unwrap();
    let import = bundle::verify(&output).unwrap();
    let mut optimizer_config = OptimizersConfig::fixture();
    optimizer_config.default_segment_number = 1;
    optimizer_config.max_optimization_threads = Some(1);
    let config = CollectionConfigInternal {
        params: CollectionParams {
            vectors: VectorsConfig::Single(
                VectorParamsBuilder::new(2, distance)
                    .with_datatype(if encoding == VectorEncoding::U8 {
                        crate::operations::types::Datatype::Uint8
                    } else {
                        crate::operations::types::Datatype::Float32
                    })
                    .build(),
            ),
            shard_number: NonZeroU32::new(2).unwrap(),
            replication_factor: NonZeroU32::new(1).unwrap(),
            write_consistency_factor: NonZeroU32::new(1).unwrap(),
            ..CollectionParams::empty()
        },
        optimizer_config,
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
        auto_shard_policy: Some(AutoShardPolicy::Orion {
            generation: 1,
            artifact_sha256: if streaming {
                "0".repeat(64)
            } else {
                import.orion_artifact_sha256
            },
        }),
    };
    let path = root.path().join("collection");
    let snapshots = root.path().join("snapshots");
    fs::create_dir_all(&path).unwrap();
    fs::create_dir_all(&snapshots).unwrap();
    let collection = Collection::new(
        "native-lifecycle".into(),
        1,
        &path,
        &snapshots,
        &config,
        Arc::new(SharedStorageConfig::default()),
        CollectionShardDistribution {
            shards: (0..2).map(|id| (id, HashSet::from([1]))).collect(),
        },
        None,
        ChannelService::default(),
        dummy_on_replica_failure(),
        dummy_request_shard_transfer(),
        dummy_abort_shard_transfer(),
        None,
        None,
        ResourceBudget::new(2, 2),
        None,
    )
    .await
    .unwrap();
    for id in 0..2 {
        collection
            .set_shard_replica_state(id, 1, ReplicaState::Active, None)
            .await
            .unwrap();
    }
    let files: BTreeMap<_, _> = fs::read_dir(&output)
        .unwrap()
        .map(|e| {
            let e = e.unwrap();
            (
                e.file_name().into_string().unwrap(),
                FileBinding {
                    size: e.metadata().unwrap().len(),
                    sha256: bundle::file_sha256(&e.path()).unwrap(),
                },
            )
        })
        .collect();
    if streaming {
        use super::streaming::{StreamRow, StreamSpec};
        let begin = ControlCommand::BeginStream {
            spec: StreamSpec::from(&request),
        };
        collection.apply_orion_control(begin.clone()).await.unwrap();
        let (sender, mut receiver) = tokio::sync::mpsc::channel::<StreamRow>(1);
        let (continue_tx, continue_rx) = std::sync::mpsc::channel();
        let streaming_output = root.path().join("streamed");
        let producer_output = streaming_output.clone();
        let producer_request = request.clone();
        let producer = tokio::task::spawn_blocking(move || {
            bundle::build_with_sink(
                &producer_request,
                &producer_output,
                &mut |assignment, vector| {
                    sender.blocking_send(StreamRow {
                        assignment: assignment.clone(),
                        vector: vector.to_vec(),
                    })?;
                    if assignment.id == 0 {
                        continue_rx.recv()?;
                    }
                    Ok(())
                },
            )
        });
        let mut sequence = 1;
        while let Some(row) = receiver.recv().await {
            let batch = ControlCommand::StreamBatch {
                generation: 1,
                sequence,
                rows: vec![row.clone()],
            };
            collection.apply_orion_control(batch.clone()).await.unwrap();
            if sequence == 1 {
                assert!(!streaming_output.exists());
                assert!(
                    !generation_path(&collection.path, 1)
                        .join("build-manifest.json")
                        .exists()
                );
                let holder = collection.shards_holder.read().await;
                for &id in &row.assignment.shards {
                    let shard = holder.get_shard(id).unwrap();
                    let count = shard
                        .count_local(
                            Arc::new(shard::count::CountRequestInternal {
                                filter: None,
                                exact: true,
                            }),
                            None,
                            HwMeasurementAcc::disposable(),
                            common::types::DeferredBehavior::Exclude,
                        )
                        .await
                        .unwrap()
                        .unwrap();
                    assert_eq!(
                        count.count, 1,
                        "the first point must reach its shard before the rest are assigned"
                    );
                }
                drop(holder);
                assert!(collection.ensure_orion_data_ready().await.is_err());
                assert!(
                    collection
                        .apply_orion_control(ControlCommand::Import { generation: 1 })
                        .await
                        .is_err()
                );
                assert!(
                    collection
                        .apply_orion_control(ControlCommand::Begin {
                            generation: 1,
                            files: files.clone()
                        })
                        .await
                        .is_err()
                );
                let mut changed = row.clone();
                changed.vector[0] += 1.0;
                assert!(
                    collection
                        .apply_orion_control(ControlCommand::StreamBatch {
                            generation: 1,
                            sequence,
                            rows: vec![changed]
                        })
                        .await
                        .is_err()
                );
                assert!(
                    collection
                        .apply_orion_control(ControlCommand::StreamBatch {
                            generation: 1,
                            sequence: 3,
                            rows: vec![row]
                        })
                        .await
                        .is_err()
                );
                // Each call reloads durable progress, exercising interrupted-client resume.
                collection.apply_orion_control(begin.clone()).await.unwrap();
                collection.apply_orion_control(batch).await.unwrap();
                continue_tx.send(()).unwrap();
            }
            sequence += 1;
        }
        producer.await.unwrap().unwrap();
        for name in files.keys() {
            assert_eq!(
                fs::read(output.join(name)).unwrap(),
                fs::read(streaming_output.join(name)).unwrap()
            );
        }
        let receipt =
            generation_path(&collection.path, 1).join("stream/batch-00000000000000000002.json");
        let original = fs::read(&receipt).unwrap();
        let mut rows: Vec<StreamRow> = serde_json::from_slice(&original).unwrap();
        rows[0].vector[0] += 0.1;
        fs::write(&receipt, serde_json::to_vec(&rows).unwrap()).unwrap();
        let build =
            serde_json::from_reader(File::open(output.join("build-manifest.json")).unwrap())
                .unwrap();
        let import = bundle::verify(&output).unwrap();
        assert!(
            super::streaming::verify_import(&collection.path, 1, &build, &import).is_err(),
            "a different streamed vector must not be accepted with a valid final bundle"
        );
        fs::write(receipt, original).unwrap();
    }
    collection
        .apply_orion_control(ControlCommand::Begin {
            generation: 1,
            files: files.clone(),
        })
        .await
        .unwrap();
    assert!(
        collection
            .apply_orion_control(ControlCommand::Activate { generation: 1 })
            .await
            .is_err()
    );
    for name in files.keys() {
        let bytes = fs::read(output.join(name)).unwrap();
        for (i, bytes) in bytes.chunks(super::lifecycle::MAX_CHUNK_BYTES).enumerate() {
            let command = ControlCommand::Chunk {
                generation: 1,
                file: name.clone(),
                offset: (i * super::lifecycle::MAX_CHUNK_BYTES) as u64,
                bytes: bytes.to_vec(),
            };
            collection
                .apply_orion_control(command.clone())
                .await
                .unwrap();
            collection.apply_orion_control(command).await.unwrap();
        }
    }
    // Files are present but the initial generation is not active. Ordinary reads
    // and background recovery must not publish or even load its large graph.
    let serving_path = super::routing_artifact_path(&collection.path, 1);
    collection.recover_orion_metadata_from_peer().await.unwrap();
    assert!(collection.orion_router.load().is_none());
    assert!(!serving_path.exists());
    fs::create_dir_all(serving_path.parent().unwrap()).unwrap();
    fs::write(
        &serving_path,
        b"stale, incomplete router from a prior process",
    )
    .unwrap();
    let config = collection.collection_config.read().await;
    assert!(
        Collection::try_load_orion_router(&collection.path, &config, 2)
            .unwrap()
            .is_none()
    );
    drop(config);
    fs::remove_file(serving_path).unwrap();
    collection
        .apply_orion_control(ControlCommand::Prepare { generation: 1 })
        .await
        .unwrap();
    if streaming {
        assert_eq!(
            collection
                .orion_control_state()
                .await
                .unwrap()
                .unwrap()
                .phase,
            Phase::Indexing,
            "verified streams must skip the ordinary full import pass"
        );
    }
    collection
        .apply_orion_control(ControlCommand::Import { generation: 1 })
        .await
        .unwrap();
    for _ in 0..200 {
        collection
            .apply_orion_control(ControlCommand::Validate { generation: 1 })
            .await
            .unwrap();
        if collection
            .orion_control_state()
            .await
            .unwrap()
            .unwrap()
            .phase
            == Phase::Finalizing
        {
            break;
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
    assert_eq!(
        collection
            .orion_control_state()
            .await
            .unwrap()
            .unwrap()
            .phase,
        Phase::Finalizing
    );
    collection
        .apply_orion_control(ControlCommand::Activate { generation: 1 })
        .await
        .unwrap();
    collection
        .apply_orion_control(ControlCommand::Cleanup { generation: 1 })
        .await
        .unwrap();
    (collection, root, output)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn streamed_build_imports_before_assignment_finishes_and_activates_only_after_validation() {
    let (collection, _root, output) = fixture_mode(true).await;
    assert_eq!(
        collection
            .orion_control_state()
            .await
            .unwrap()
            .unwrap()
            .phase,
        Phase::Ready
    );
    let import = bundle::verify(&output).unwrap();
    assert_eq!(
        membership::Catalog::load(&collection.path, 1)
            .unwrap()
            .entries
            .len(),
        import.point_count
    );
    assert_eq!(retrieve(&collection, 0.into()).await.len(), 1);
    collection.stop_gracefully().await;
}

fn upsert(id: u64, vector: Vec<f32>) -> CollectionUpdateOperations {
    CollectionUpdateOperations::PointOperation(PointOperations::UpsertPoints(
        PointInsertOperationsInternal::PointsList(vec![PointStructPersisted {
            id: id.into(),
            vector: VectorStructPersisted::Single(vector),
            payload: None,
        }]),
    ))
}
async fn write(collection: &Collection, op: CollectionUpdateOperations) {
    let plan = collection.prepare_orion_write(op).await.unwrap();
    let command = serde_json::from_slice(
        &serde_json::to_vec(&ControlCommand::Write { plan: plan.clone() }).unwrap(),
    )
    .unwrap();
    collection.apply_orion_control(command).await.unwrap();
    collection
        .validate_orion_write_receipts(&plan, &[collection.orion_control_state().await.unwrap()])
        .await
        .unwrap();
    collection
        .apply_orion_control(ControlCommand::CommitWrite { plan })
        .await
        .unwrap();
}
async fn retrieve(
    collection: &Collection,
    id: segment::types::ExtendedPointId,
) -> Vec<shard::retrieve::record_internal::RecordInternal> {
    collection
        .retrieve(
            PointRequestInternal {
                ids: vec![id],
                with_payload: Some(WithPayloadInterface::Bool(true)),
                with_vector: WithVector::Bool(true),
            },
            None,
            &ShardSelectorInternal::All,
            None,
            HwMeasurementAcc::disposable(),
        )
        .await
        .unwrap()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn install_write_anchor_delete_and_recovery_are_native() {
    let (collection, root, output) = fixture().await;
    write(&collection, upsert(1000, vec![4.0, 1.0])).await;
    assert_eq!(retrieve(&collection, 1000.into()).await.len(), 1);
    let import = bundle::verify(&output).unwrap();
    let artifact =
        super::OrionRoutingArtifact::read_json(&output.join(import.orion_artifact_file), None)
            .unwrap();
    let anchor = artifact.upper_nodes[0].label;
    write(
        &collection,
        CollectionUpdateOperations::PointOperation(PointOperations::DeletePoints {
            ids: vec![anchor],
        }),
    )
    .await;
    assert!(retrieve(&collection, anchor).await.is_empty());
    let catalog = membership::Catalog::load(&collection.path, 1).unwrap();
    assert!(!catalog.entries[&anchor].live);
    assert_eq!(
        catalog.entries[&anchor].shards,
        vec![artifact.upper_nodes[0].owner_shard]
    );
    let old_revision = catalog.revision;
    let plan = collection
        .prepare_orion_write(upsert(1001, vec![3.0, -1.0]))
        .await
        .unwrap();
    // Fail after local shard WAL writes, before the membership event can be persisted.
    let blocker = collection
        .path
        .join(super::lifecycle::CONTROL_DIR)
        .join("membership-1")
        .join(format!("event-{:020}.json", old_revision + 1));
    fs::create_dir(&blocker).unwrap();
    collection
        .apply_orion_control(ControlCommand::Write { plan: plan.clone() })
        .await
        .unwrap();
    assert_eq!(
        collection
            .orion_control_state()
            .await
            .unwrap()
            .unwrap()
            .active
            .unwrap()
            .membership_revision,
        old_revision,
        "prepare must not publish membership"
    );
    assert!(
        collection
            .apply_orion_control(ControlCommand::CommitWrite { plan: plan.clone() })
            .await
            .is_err()
    );
    assert!(membership::pending_path(&collection.path).is_file());
    fs::remove_dir(&blocker).unwrap();
    collection.recover_orion_write().await.unwrap();
    collection
        .apply_orion_control(ControlCommand::CommitWrite { plan })
        .await
        .unwrap();
    assert_eq!(retrieve(&collection, 1001.into()).await.len(), 1);
    assert_eq!(
        membership::Catalog::load(&collection.path, 1)
            .unwrap()
            .revision,
        old_revision + 1
    );
    assert!(!membership::pending_path(&collection.path).exists());
    let anchors: Vec<_> = artifact.upper_nodes.iter().map(|n| n.label).collect();
    write(
        &collection,
        CollectionUpdateOperations::PointOperation(PointOperations::DeletePoints {
            ids: anchors.clone(),
        }),
    )
    .await;
    let search = shard::search::CoreSearchRequest {
        query: shard::query::query_enum::QueryEnum::Nearest(
            segment::data_types::vectors::NamedQuery::default_dense(vec![0.0, 0.0]),
        ),
        filter: None,
        params: None,
        hnsw_entry_points: None,
        hnsw_entry_points_by_shard: None,
        hnsw_ef_by_shard: None,
        source_id_dedup_block_size: None,
        limit: 5,
        offset: 0,
        with_payload: None,
        with_vector: None,
        score_threshold: None,
    };
    let mut result = None;
    for _ in 0..400 {
        if let Ok(rows) = collection
            .core_search_batch(
                shard::search::CoreSearchRequestBatch {
                    searches: vec![search.clone()],
                },
                None,
                ShardSelectorInternal::All,
                None,
                HwMeasurementAcc::disposable(),
            )
            .await
        {
            result = Some(rows);
            break;
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
    let results = result.expect("HNSW did not become query-ready");
    assert!(
        !results[0].is_empty(),
        "deleted anchors must still seed L0 traversal"
    );
    assert!(results[0].iter().all(|point| !anchors.contains(&point.id)));
    let count = collection
        .count(
            shard::count::CountRequestInternal {
                filter: None,
                exact: true,
            },
            None,
            &ShardSelectorInternal::All,
            None,
            HwMeasurementAcc::disposable(),
        )
        .await
        .unwrap();
    assert_eq!(count.count, 50);
    let snapshot = collection.create_snapshot(root.path(), 1).await.unwrap();
    assert!(snapshot.size > 0);
    assert!(generation_path(&collection.path, 1).is_dir());
    let restored_path = root.path().join("restored");
    fs::create_dir(&restored_path).unwrap();
    let data = shard::snapshots::snapshot_data::SnapshotData::new_packed_persistent(
        root.path().join("snapshots").join(snapshot.name),
    );
    Collection::restore_snapshot(data, &restored_path, 1, true).unwrap();
    let restored = Collection::load(
        "restored".into(),
        1,
        &restored_path,
        &root.path().join("snapshots"),
        Default::default(),
        ChannelService::default(),
        dummy_on_replica_failure(),
        dummy_request_shard_transfer(),
        dummy_abort_shard_transfer(),
        None,
        None,
        ResourceBudget::new(2, 2),
        None,
    )
    .await;
    assert!(retrieve(&restored, anchor).await.is_empty());
    assert_eq!(retrieve(&restored, 1001.into()).await.len(), 1);
    assert_eq!(
        membership::Catalog::load(&restored.path, 1)
            .unwrap()
            .revision,
        membership::Catalog::load(&collection.path, 1)
            .unwrap()
            .revision
    );
    restored.stop_gracefully().await;
    collection.stop_gracefully().await;
}

async fn finish_generation(collection: &Collection, generation: u64) {
    for _ in 0..400 {
        collection
            .apply_orion_control(ControlCommand::Validate { generation })
            .await
            .unwrap();
        if collection
            .orion_control_state()
            .await
            .unwrap()
            .unwrap()
            .phase
            == Phase::Finalizing
        {
            break;
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
    let state = collection.orion_control_state().await.unwrap().unwrap();
    assert_eq!(state.phase, Phase::Finalizing, "{:?}", state.last_error);
    collection
        .apply_orion_control(ControlCommand::Activate { generation })
        .await
        .unwrap();
    collection
        .apply_orion_control(ControlCommand::Cleanup { generation })
        .await
        .unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn repartition_preserves_uuid_payload_tombstones_and_immutable_graph() {
    let (collection, _root, output) = fixture().await;
    let id = segment::types::ExtendedPointId::Uuid(uuid::Uuid::new_v4());
    let payload = segment::types::Payload(
        serde_json::json!({"kind":"uuid","version":7})
            .as_object()
            .unwrap()
            .clone(),
    );
    write(
        &collection,
        CollectionUpdateOperations::PointOperation(PointOperations::UpsertPoints(
            PointInsertOperationsInternal::PointsList(vec![PointStructPersisted {
                id,
                vector: vec![0.8_f32, 1.2].into(),
                payload: Some(payload.clone()),
            }]),
        )),
    )
    .await;
    let original =
        super::OrionRoutingArtifact::read_json(&output.join("generation-1.json"), None).unwrap();
    let anchor = original.upper_nodes[0].label;
    write(
        &collection,
        CollectionUpdateOperations::PointOperation(PointOperations::DeletePoints {
            ids: vec![anchor],
        }),
    )
    .await;
    let revision = membership::Catalog::load(&collection.path, 1)
        .unwrap()
        .revision;
    collection
        .apply_orion_control(ControlCommand::BeginMigration {
            generation: 2,
            shard_count: 3,
            source_revision: revision,
            peers: vec![1],
        })
        .await
        .unwrap();
    assert!(
        collection
            .prepare_orion_write(upsert(999, vec![0.1, 0.2]))
            .await
            .is_err()
    );
    let ids = collection.orion_migration_ids(2).await.unwrap();
    let points = collection.orion_migration_points(2, &ids).await.unwrap();
    collection
        .apply_orion_control(ControlCommand::MigrateBatch {
            generation: 2,
            sequence: 0,
            points: points.clone(),
        })
        .await
        .unwrap();
    collection
        .apply_orion_control(ControlCommand::MigrateBatch {
            generation: 2,
            sequence: 0,
            points,
        })
        .await
        .unwrap();
    collection
        .apply_orion_control(ControlCommand::SealMigration { generation: 2 })
        .await
        .unwrap();
    finish_generation(&collection, 2).await;
    assert_eq!(collection.shards_holder.read().await.len(), 3);
    let current = super::migration::read_artifact(&collection.path, 2).unwrap();
    assert_eq!(
        bundle::graph_sha256(&current).unwrap(),
        bundle::graph_sha256(&original).unwrap()
    );
    assert_eq!(
        bundle::upper_identity_sha256(&current).unwrap(),
        bundle::upper_identity_sha256(&original).unwrap()
    );
    assert_eq!(retrieve(&collection, id).await[0].payload, Some(payload));
    assert!(retrieve(&collection, anchor).await.is_empty());
    // A partial grow can be rolled back without deleting old-generation copies.
    collection
        .apply_orion_control(ControlCommand::BeginMigration {
            generation: 3,
            shard_count: 4,
            source_revision: 0,
            peers: vec![1],
        })
        .await
        .unwrap();
    let ids = collection.orion_migration_ids(3).await.unwrap();
    let points = collection
        .orion_migration_points(3, &ids[..8])
        .await
        .unwrap();
    collection
        .apply_orion_control(ControlCommand::MigrateBatch {
            generation: 3,
            sequence: 0,
            points,
        })
        .await
        .unwrap();
    collection
        .apply_orion_control(ControlCommand::Rollback { generation: 3 })
        .await
        .unwrap();
    assert_eq!(collection.shards_holder.read().await.len(), 3);
    assert_eq!(
        collection
            .orion_control_state()
            .await
            .unwrap()
            .unwrap()
            .active
            .unwrap()
            .generation,
        2
    );
    // Advance to a fresh generation and shrink, retaining the exact graph.
    collection
        .apply_orion_control(ControlCommand::BeginMigration {
            generation: 4,
            shard_count: 2,
            source_revision: 0,
            peers: vec![1],
        })
        .await
        .unwrap();
    let ids = collection.orion_migration_ids(4).await.unwrap();
    let points = collection.orion_migration_points(4, &ids).await.unwrap();
    collection
        .apply_orion_control(ControlCommand::MigrateBatch {
            generation: 4,
            sequence: 0,
            points,
        })
        .await
        .unwrap();
    collection
        .apply_orion_control(ControlCommand::SealMigration { generation: 4 })
        .await
        .unwrap();
    finish_generation(&collection, 4).await;
    assert_eq!(collection.shards_holder.read().await.len(), 2);
    assert_eq!(retrieve(&collection, id).await.len(), 1);
    assert!(retrieve(&collection, anchor).await.is_empty());
    assert_eq!(
        bundle::graph_sha256(&super::migration::read_artifact(&collection.path, 4).unwrap())
            .unwrap(),
        bundle::graph_sha256(&original).unwrap()
    );
    collection.stop_gracefully().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn uint8_native_install_and_streaming_preserve_vectors_and_schema() {
    for streaming in [false, true] {
        let (collection, root, output) = fixture_encoding(streaming, VectorEncoding::U8).await;
        assert_eq!(
            collection
                .orion_control_state()
                .await
                .unwrap()
                .unwrap()
                .phase,
            Phase::Ready
        );
        let import = bundle::verify(&output).unwrap();
        assert_eq!(import.vector_encoding, VectorEncoding::U8);
        assert_eq!(
            fs::metadata(output.join(import.vectors_file))
                .unwrap()
                .len(),
            128
        );
        let raw = fs::read(root.path().join("raw.f32le")).unwrap();
        for id in 0..64_u64 {
            let rows = retrieve(&collection, id.into()).await;
            assert_eq!(rows.len(), 1);
            let expected = segment::data_types::vectors::VectorStructInternal::Single(
                raw[id as usize * 2..id as usize * 2 + 2]
                    .iter()
                    .map(|&x| f32::from(x))
                    .collect(),
            );
            assert_eq!(rows[0].vector.as_ref(), Some(&expected));
        }
        collection.stop_gracefully().await;
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn uint8_cosine_install_keeps_bytes_without_float_normalization() {
    let (collection, _root, _output) =
        fixture_schema(false, VectorEncoding::U8, Distance::Cosine).await;
    assert_eq!(
        collection
            .orion_control_state()
            .await
            .unwrap()
            .unwrap()
            .phase,
        Phase::Ready
    );
    let record = retrieve(&collection, 0.into()).await;
    assert_eq!(
        record[0].vector,
        Some(segment::data_types::vectors::VectorStructInternal::Single(
            vec![32.0, 16.0]
        ))
    );
    collection.stop_gracefully().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn local_catalog_view_preserves_wal_moves_visibility_and_clock_revisions() {
    let (collection, _root, _) = fixture().await;
    let check = || {
        let full = membership::Catalog::load(&collection.path, 1).unwrap();
        for shard in 0..2 {
            let partial = membership::Catalog::load_for_shards(
                &collection.path,
                1,
                &std::collections::BTreeSet::from([shard]),
            )
            .unwrap();
            assert_eq!(partial.revision, full.revision);
            assert_eq!(
                partial.shard_revisions.get(&shard),
                full.shard_revisions.get(&shard)
            );
            let wanted: Vec<_> = full
                .entries
                .iter()
                .filter(|(_, m)| m.shards.contains(&shard))
                .map(|(id, m)| (*id, m.shards.clone(), m.live, m.vector_present))
                .collect();
            let got: Vec<_> = partial
                .entries
                .iter()
                .filter(|(_, m)| m.shards.contains(&shard))
                .map(|(id, m)| (*id, m.shards.clone(), m.live, m.vector_present))
                .collect();
            assert_eq!(wanted, got);
        }
        let visible = membership::load_visibility(&collection.path);
        assert_eq!(visible.deleted, full.visibility().deleted);
        assert_eq!(visible.no_vector, full.visibility().no_vector);
    };
    let empty = membership::Catalog::load_for_shards(
        &collection.path,
        1,
        &std::collections::BTreeSet::new(),
    )
    .unwrap();
    assert!(empty.entries.is_empty());
    check();
    write(&collection, upsert(0, vec![100.0, 100.0])).await;
    check();
    write(&collection, upsert(0, vec![-100.0, -100.0])).await;
    check();
    collection.stop_gracefully().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn active_router_recovery_streams_file_and_preserves_uint8_routes() {
    let (collection, _root, _) = fixture_encoding(false, VectorEncoding::U8).await;
    let before = collection
        .orion_router
        .load_full()
        .unwrap()
        .route_query(&[128.0, 192.0])
        .unwrap();
    let path = super::routing_artifact_path(&collection.path, 1);
    let previous_bytes = fs::read(&path).unwrap();
    collection.orion_router.store(None);
    fs::remove_file(&path).unwrap();
    collection.recover_orion_metadata_from_peer().await.unwrap();
    assert_eq!(fs::read(path).unwrap(), previous_bytes);
    assert_eq!(
        collection
            .orion_router
            .load_full()
            .unwrap()
            .route_query(&[128.0, 192.0])
            .unwrap(),
        before
    );
    collection.ensure_orion_worker_ready().await.unwrap();
    collection.stop_gracefully().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn static_readiness_uses_bound_counts_and_loaded_router_without_graph_reload() {
    let (collection, _root, output) = fixture_encoding(false, VectorEncoding::U8).await;
    let state = super::lifecycle::read_state(&collection.path)
        .unwrap()
        .unwrap();
    let counts = membership::initial_shard_counts(&collection.path, &state)
        .unwrap()
        .unwrap();
    let full = membership::Catalog::load(&collection.path, 1).unwrap();
    for (shard, &count) in counts.iter().enumerate() {
        assert_eq!(
            count,
            full.entries
                .values()
                .filter(|m| m.shards.contains(&(shard as u32)))
                .count()
        );
    }
    let import = bundle::verify_integrity(&output).unwrap();
    let graph = generation_path(&collection.path, 1).join(import.orion_artifact_file);
    let moved = graph.with_extension("held");
    fs::rename(&graph, &moved).unwrap();
    collection
        .orion_serving
        .store(false, std::sync::atomic::Ordering::Release);
    collection
        .orion_data_ready
        .store(false, std::sync::atomic::Ordering::Release);
    collection.ensure_orion_worker_ready().await.unwrap();
    fs::rename(moved, graph).unwrap();
    let build = generation_path(&collection.path, 1).join("build-manifest.json");
    let bytes = fs::read(&build).unwrap();
    fs::write(&build, b"{}").unwrap();
    assert!(membership::initial_shard_counts(&collection.path, &state).is_err());
    fs::write(&build, bytes).unwrap();
    write(&collection, upsert(1000, vec![127.0, 255.0])).await;
    let state = super::lifecycle::read_state(&collection.path)
        .unwrap()
        .unwrap();
    assert!(
        membership::initial_shard_counts(&collection.path, &state)
            .unwrap()
            .is_none()
    );
    collection.stop_gracefully().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn runtime_profile_activation_preserves_lower_graph_files() {
    fn graphs(path: &std::path::Path) -> BTreeMap<std::path::PathBuf, String> {
        fn visit(
            root: &std::path::Path,
            path: &std::path::Path,
            out: &mut BTreeMap<std::path::PathBuf, String>,
        ) {
            for entry in fs::read_dir(path).unwrap() {
                let path = entry.unwrap().path();
                if path.is_dir() {
                    visit(root, &path, out);
                } else if path
                    .parent()
                    .unwrap()
                    .file_name()
                    .is_some_and(|name| name == "vector_index")
                {
                    out.insert(
                        path.strip_prefix(root).unwrap().to_path_buf(),
                        bundle::file_sha256(&path).unwrap(),
                    );
                }
            }
        }
        let mut out = BTreeMap::new();
        visit(path, path, &mut out);
        out
    }
    let (collection, root, source) = fixture().await;
    let before = graphs(&collection.path);
    assert!(!before.is_empty());
    for generation in [2, 3] {
        let target = root.path().join(format!("profile-{generation}"));
        super::profile::derive(
            &source,
            &target,
            &super::profile::RuntimeProfile {
                generation,
                upper_k: generation as usize + 2,
                upper_ef_search: 16,
                dynamic_ef_base: 30,
                dynamic_ef_factor: 2,
            },
        )
        .unwrap();
        let files: BTreeMap<_, _> = fs::read_dir(&target)
            .unwrap()
            .map(|entry| {
                let entry = entry.unwrap();
                (
                    entry.file_name().into_string().unwrap(),
                    FileBinding {
                        size: entry.metadata().unwrap().len(),
                        sha256: bundle::file_sha256(&entry.path()).unwrap(),
                    },
                )
            })
            .collect();
        collection
            .apply_orion_control(ControlCommand::Begin {
                generation,
                files: files.clone(),
            })
            .await
            .unwrap();
        for name in files.keys() {
            fs::copy(
                target.join(name),
                generation_path(&collection.path, generation).join(name),
            )
            .unwrap();
        }
        collection
            .apply_orion_control(ControlCommand::Prepare { generation })
            .await
            .unwrap();
        collection
            .apply_orion_control(ControlCommand::Import { generation })
            .await
            .unwrap();
        finish_generation(&collection, generation).await;
        assert_eq!(graphs(&collection.path), before);
        assert_eq!(
            collection.orion_router.load_full().unwrap().upper_k(),
            generation as usize + 2
        );
        assert_eq!(
            collection
                .orion_control_state()
                .await
                .unwrap()
                .unwrap()
                .active
                .unwrap()
                .generation,
            generation
        );
    }
    collection.stop_gracefully().await;
}
