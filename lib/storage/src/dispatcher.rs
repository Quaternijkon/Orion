use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use api::rest::models::HardwareUsage;
use collection::common::fetch_vectors::CollectionName;
use collection::config::ShardingMethod;
use collection::operations::verification::VerificationPass;
use collection::shards::replica_set::replica_set_state::ReplicaState;
use common::counter::hardware_accumulator::HwSharedDrain;
use common::defaults::CONSENSUS_META_OP_WAIT;
use futures::StreamExt as _;
use futures::stream::FuturesUnordered;
use segment::types::ShardKey;

use crate::content_manager::collection_meta_ops::AliasOperations;
use crate::content_manager::shard_distribution::ShardDistributionProposal;
use crate::rbac::{Auth, CollectionMultipass};
use crate::{
    ClusterStatus, CollectionMetaOperations, ConsensusOperations, ConsensusStateRef, StorageError,
    TableOfContent,
};

#[derive(Clone)]
pub struct Dispatcher {
    toc: Arc<TableOfContent>,
    consensus_state: Option<ConsensusStateRef>,
    resharding_enabled: bool,
}

impl Dispatcher {
    /// Collect applied, durable peer state through the internal RPC, never client assertions.
    pub async fn orion_peer_states(
        &self,
        name: &str,
    ) -> Result<Vec<Option<collection::orion::lifecycle::ControlState>>, StorageError> {
        use api::grpc::qdrant::GetCollectionInfoRequest;
        use api::grpc::qdrant::collections_internal_client::CollectionsInternalClient;
        let collection = self
            .toc
            .get_collection(&CollectionMultipass.issue_pass(name))
            .await?;
        let mut result = vec![collection.orion_control_state().await?];
        let channels = self.toc.get_channel_service();
        let addresses: Vec<_> = channels
            .id_to_address
            .read()
            .iter()
            .filter(|(peer, _)| **peer != self.toc.this_peer_id)
            .map(|(id, uri)| (*id, uri.clone()))
            .collect();
        let requests = addresses.into_iter().map(|(peer_id, uri)| async move {
            let response = channels
                .channel_pool
                .with_channel(&uri, |channel| {
                    let name = name.to_string();
                    async move {
                        CollectionsInternalClient::new(channel)
                            .orion_status(GetCollectionInfoRequest {
                                collection_name: name,
                            })
                            .await
                    }
                })
                .await
                .map_err(|e| {
                    StorageError::service_error(format!("Orion peer state unavailable: {e}"))
                })?;
            let response = response.into_inner();
            if response.peer_id != peer_id {
                return Err(StorageError::service_error(
                    "Orion receipt peer identity mismatch",
                ));
            }
            serde_json::from_slice(&response.state_json)
                .map_err(|e| StorageError::service_error(format!("invalid Orion peer state: {e}")))
        });
        result.extend(futures::future::try_join_all(requests).await?);
        Ok(result)
    }

    pub async fn submit_orion_control(
        &self,
        name: String,
        command: collection::orion::lifecycle::ControlCommand,
        auth: Auth,
        timeout: Option<Duration>,
    ) -> Result<bool, StorageError> {
        use collection::orion::lifecycle::{ControlCommand, Phase};
        let operation = CollectionMetaOperations::OrionControl {
            collection_name: name.clone(),
            command: serde_json::to_string(&command)
                .map_err(|e| StorageError::bad_request(e.to_string()))?,
        };
        auth.check_collection_meta_operation(&operation)?;
        if matches!(command, ControlCommand::RecoverWrite) {
            self.resume_orion_writes(&name, timeout).await?;
            return Ok(true);
        }
        // Activation is proposed only after every peer durably acknowledged validation.
        if let ControlCommand::Activate { generation } = &command {
            let states = self.orion_peer_states(&name).await?;
            if !states.iter().all(|s| {
                s.as_ref().is_some_and(|s| {
                    (s.pending_generation == Some(*generation) && s.phase == Phase::Finalizing)
                        || (s
                            .active
                            .as_ref()
                            .is_some_and(|g| g.generation == *generation)
                            && matches!(s.phase, Phase::Ready | Phase::Finalizing))
                })
            }) {
                return Err(StorageError::bad_request(
                    "not all Orion peers validated this generation",
                ));
            }
            let bindings: std::collections::HashSet<_> = states
                .iter()
                .flatten()
                .map(|s| {
                    if s.pending_generation == Some(*generation) {
                        (
                            s.pending_artifact_sha256.clone(),
                            s.pending_layout_sha256.clone(),
                        )
                    } else {
                        let g = s.active.as_ref().unwrap();
                        (
                            Some(g.artifact_sha256.clone()),
                            Some(g.layout_sha256.clone()),
                        )
                    }
                })
                .collect();
            if bindings.len() != 1 || bindings.iter().any(|(a, l)| a.is_none() || l.is_none()) {
                return Err(StorageError::bad_request(
                    "Orion peers disagree on generation checksums",
                ));
            }
        }
        let result = self
            .submit_collection_meta_op(operation, auth, timeout)
            .await?;
        if matches!(command, ControlCommand::Validate { .. })
            && !self.orion_peer_states(&name).await?.iter().all(|s| {
                s.as_ref()
                    .is_some_and(|s| matches!(s.phase, Phase::Finalizing | Phase::Ready))
            })
        {
            return Err(StorageError::bad_request(
                "Orion replicas are still indexing; retry validate",
            ));
        }
        Ok(result)
    }

    pub(crate) async fn apply_orion_write_transaction(
        &self,
        name: &str,
        plan: collection::orion::membership::WritePlan,
        timeout: Option<Duration>,
    ) -> Result<(), StorageError> {
        use collection::orion::lifecycle::ControlCommand;
        let auth = Auth::new_internal(crate::rbac::Access::full("Orion transaction recovery"));
        let states = self.orion_peer_states(name).await?;
        let collection = self
            .toc
            .get_collection(&CollectionMultipass.issue_pass(name))
            .await?;
        let mut historical = collection
            .orion_saved_plan(plan.generation, plan.expected_revision + 1)
            .is_ok_and(|p| p.token == plan.token)
            || states.iter().flatten().any(|s| {
                s.active.as_ref().is_some_and(|g| {
                    g.generation == plan.generation
                        && g.membership_revision == plan.expected_revision + 1
                        && g.last_write_token.as_deref() == Some(&plan.token)
                })
            });
        if !historical
            && states.iter().flatten().any(|s| {
                s.active.as_ref().is_some_and(|g| {
                    g.generation == plan.generation
                        && g.membership_revision > plan.expected_revision + 1
                })
            })
        {
            use api::grpc::qdrant::{
                OrionWriteRequest, collections_internal_client::CollectionsInternalClient,
            };
            let channels = self.toc.get_channel_service();
            let peers: Vec<_> = channels.id_to_address.read().values().cloned().collect();
            for uri in peers {
                let result = channels
                    .channel_pool
                    .with_channel(&uri, |channel| {
                        let request = OrionWriteRequest {
                            collection_name: name.into(),
                            generation: plan.generation,
                            revision: plan.expected_revision + 1,
                            token: plan.token.clone(),
                            committed: true,
                        };
                        async move {
                            CollectionsInternalClient::new(channel)
                                .max_decoding_message_size(32 * 1024 * 1024)
                                .orion_pending(request)
                                .await
                        }
                    })
                    .await;
                if result.is_ok() {
                    historical = true;
                    break;
                }
            }
        }
        if !historical
            && states.iter().any(|s| {
                !s.as_ref().is_some_and(|s| {
                    s.active.as_ref().is_some_and(|g| {
                        g.generation == plan.generation
                            && g.membership_revision == plan.expected_revision
                    })
                })
            })
        {
            return Err(StorageError::ShardUnavailable {
                description: "Orion catalogs are recovering; retry the write".into(),
            });
        }
        let command = serde_json::to_string(&ControlCommand::Write { plan: plan.clone() })
            .map_err(|e| StorageError::bad_request(e.to_string()))?;
        if command.len() > 16 * 1024 * 1024 {
            return Err(StorageError::bad_request(
                "Orion write batch exceeds 16 MiB; split the request",
            ));
        }
        self.submit_collection_meta_op(
            CollectionMetaOperations::OrionControl {
                collection_name: name.into(),
                command,
            },
            auth.clone(),
            timeout,
        )
        .await?;
        if !historical
            && !collection
                .orion_saved_plan(plan.generation, plan.expected_revision + 1)
                .is_ok_and(|p| p.token == plan.token)
        {
            collection
                .validate_orion_write_receipts(&plan, &self.orion_peer_states(name).await?)
                .await?;
        }
        let command = serde_json::to_string(&ControlCommand::CommitWrite { plan })
            .map_err(|e| StorageError::bad_request(e.to_string()))?;
        self.submit_collection_meta_op(
            CollectionMetaOperations::OrionControl {
                collection_name: name.into(),
                command,
            },
            auth,
            timeout,
        )
        .await?;
        Ok(())
    }

    pub(crate) async fn resume_orion_writes(
        &self,
        name: &str,
        timeout: Option<Duration>,
    ) -> Result<(), StorageError> {
        use api::grpc::qdrant::{
            OrionWriteRequest, collections_internal_client::CollectionsInternalClient,
        };
        for _ in 0..16 {
            let states = self.orion_peer_states(name).await?;
            let Some(pending) = states
                .iter()
                .flatten()
                .filter_map(|s| s.pending_write.as_ref())
                .min_by_key(|p| (p.generation, p.revision))
                .cloned()
            else {
                return Ok(());
            };
            let collection = self
                .toc
                .get_collection(&CollectionMultipass.issue_pass(name))
                .await?;
            let mut plan = collection
                .orion_pending_plan()
                .await?
                .filter(|p| p.token == pending.token)
                .or_else(|| {
                    collection
                        .orion_saved_plan(pending.generation, pending.revision)
                        .ok()
                        .filter(|p| p.token == pending.token)
                });
            if plan.is_none() {
                let channels = self.toc.get_channel_service();
                let peers: Vec<_> = channels.id_to_address.read().values().cloned().collect();
                for uri in peers {
                    let result = channels
                        .channel_pool
                        .with_channel(&uri, |channel| {
                            let request = OrionWriteRequest {
                                collection_name: name.into(),
                                generation: pending.generation,
                                revision: pending.revision,
                                token: pending.token.clone(),
                                committed: false,
                            };
                            async move {
                                CollectionsInternalClient::new(channel)
                                    .max_decoding_message_size(32 * 1024 * 1024)
                                    .orion_pending(request)
                                    .await
                            }
                        })
                        .await;
                    if let Ok(result) = result {
                        let candidate: collection::orion::membership::WritePlan =
                            serde_json::from_slice(&result.into_inner().plan_json)
                                .map_err(|e| StorageError::service_error(e.to_string()))?;
                        if candidate.token == pending.token
                            && candidate.generation == pending.generation
                            && candidate.expected_revision + 1 == pending.revision
                        {
                            plan = Some(candidate);
                            break;
                        }
                    }
                }
            }
            self.apply_orion_write_transaction(
                name,
                plan.ok_or_else(|| {
                    StorageError::service_error("Orion pending intent is unavailable")
                })?,
                timeout,
            )
            .await?;
        }
        Err(StorageError::ShardUnavailable {
            description: "Orion recovery is busy; retry".into(),
        })
    }

    /// Administrative resharding under an explicit pause-write protocol. The Rust
    /// coordinator captures current records; clients provide no membership decisions.
    pub async fn repartition_orion(
        &self,
        name: String,
        generation: u64,
        shard_count: u32,
        auth: Auth,
        timeout: Option<Duration>,
    ) -> Result<bool, StorageError> {
        use collection::orion::lifecycle::{ControlCommand, Phase};
        auth.check_collection_meta_operation(&CollectionMetaOperations::OrionControl {
            collection_name: name.clone(),
            command: String::new(),
        })?;
        self.resume_orion_writes(&name, timeout).await?;
        let current = self.orion_peer_states(&name).await?;
        if current.iter().all(|s| {
            s.as_ref().is_some_and(|s| {
                s.active
                    .as_ref()
                    .is_some_and(|g| g.generation == generation)
            })
        }) {
            return self
                .submit_orion_control(name, ControlCommand::Cleanup { generation }, auth, timeout)
                .await;
        }
        let source = current
            .first()
            .and_then(|s| s.as_ref())
            .and_then(|s| s.active.as_ref())
            .ok_or_else(|| StorageError::bad_request("Orion has no active source generation"))?;
        let mut peers: Vec<_> = self
            .toc
            .get_channel_service()
            .id_to_address
            .read()
            .keys()
            .copied()
            .collect();
        peers.push(self.toc.this_peer_id);
        peers.sort_unstable();
        peers.dedup();
        self.submit_orion_control(
            name.clone(),
            ControlCommand::BeginMigration {
                generation,
                shard_count,
                source_revision: source.membership_revision,
                peers,
            },
            auth.clone(),
            timeout,
        )
        .await?;
        let collection = self
            .toc
            .get_collection(&CollectionMultipass.issue_pass(&name))
            .await?;
        let ids = collection.orion_migration_ids(generation).await?;
        for (sequence, ids) in ids.chunks(512).enumerate() {
            let points = collection.orion_migration_points(generation, ids).await?;
            self.submit_orion_control(
                name.clone(),
                ControlCommand::MigrateBatch {
                    generation,
                    sequence: sequence as u64,
                    points,
                },
                auth.clone(),
                timeout,
            )
            .await?;
        }
        self.submit_orion_control(
            name.clone(),
            ControlCommand::SealMigration { generation },
            auth.clone(),
            timeout,
        )
        .await?;
        let start = Instant::now();
        let timeout = timeout.unwrap_or(Duration::from_secs(7200));
        loop {
            self.submit_collection_meta_op(
                CollectionMetaOperations::OrionControl {
                    collection_name: name.clone(),
                    command: serde_json::to_string(&ControlCommand::Validate { generation })
                        .map_err(|e| StorageError::bad_request(e.to_string()))?,
                },
                auth.clone(),
                Some(timeout),
            )
            .await?;
            if self
                .orion_peer_states(&name)
                .await?
                .iter()
                .all(|s| s.as_ref().is_some_and(|s| s.phase == Phase::Finalizing))
            {
                break;
            }
            if start.elapsed() > timeout {
                return Err(StorageError::service_error(
                    "timed out waiting for migrated HNSW indexes",
                ));
            }
            tokio::time::sleep(Duration::from_millis(200)).await;
        }
        self.submit_orion_control(
            name.clone(),
            ControlCommand::Activate { generation },
            auth.clone(),
            Some(timeout),
        )
        .await?;
        self.submit_orion_control(
            name,
            ControlCommand::Cleanup { generation },
            auth,
            Some(timeout),
        )
        .await
    }

    pub fn new(toc: Arc<TableOfContent>) -> Self {
        Self {
            toc,
            consensus_state: None,
            resharding_enabled: false,
        }
    }

    pub fn with_consensus(self, state_ref: ConsensusStateRef, resharding_enabled: bool) -> Self {
        Self {
            consensus_state: Some(state_ref),
            resharding_enabled,
            ..self
        }
    }

    /// Get the table of content.
    /// The `_auth` and `_verification_pass` parameter are not used, but it's required to verify caller's possession
    /// of both objects.
    pub fn toc(&self, _auth: &Auth, _verification_pass: &VerificationPass) -> &Arc<TableOfContent> {
        &self.toc
    }

    pub fn consensus_state(&self) -> Option<&ConsensusStateRef> {
        self.consensus_state.as_ref()
    }

    pub fn is_resharding_enabled(&self) -> bool {
        self.resharding_enabled
    }

    /// If `wait_timeout` is not supplied - then default duration will be used.
    ///
    /// This function needs to be called from a runtime with timers enabled.
    ///
    /// ## Cancel safety
    ///
    /// This function is cancel safe.
    ///
    /// On deployments without consensus - a submitted operation is always run to completion.
    pub async fn submit_collection_meta_op(
        &self,
        operation: CollectionMetaOperations,
        auth: Auth,
        wait_timeout: Option<Duration>,
    ) -> Result<bool, StorageError> {
        auth.check_collection_meta_operation(&operation)?;
        let strict_sync = matches!(&operation, CollectionMetaOperations::OrionControl { .. });

        // if distributed deployment is enabled
        if let Some(state) = self.consensus_state.as_ref() {
            let start = Instant::now();

            // List of operations to await for collection to be operational
            let mut expect_operations: Vec<ConsensusOperations> = vec![];

            let op = match operation {
                CollectionMetaOperations::CreateCollection(mut op) => {
                    if !op.is_distribution_set() {
                        match op.create_collection.sharding_method.unwrap_or_default() {
                            ShardingMethod::Auto => {
                                // Suggest even distribution of shards across nodes
                                let number_of_peers = state.0.peer_count();

                                let collection_defaults =
                                    self.toc.storage_config.collection.as_ref();

                                let shard_distribution = self.toc.suggest_shard_distribution(
                                    &op,
                                    collection_defaults,
                                    number_of_peers,
                                );

                                // Expect all replicas to become active eventually
                                for (shard_id, peer_ids) in &shard_distribution.distribution {
                                    for peer_id in peer_ids {
                                        expect_operations.push(
                                            ConsensusOperations::initialize_replica(
                                                op.collection_name.clone(),
                                                *shard_id,
                                                *peer_id,
                                            ),
                                        );
                                    }
                                }

                                op.set_distribution(shard_distribution);
                            }
                            ShardingMethod::Custom => {
                                // If custom sharding is used - we don't create any shards in advance
                                let empty_distribution = ShardDistributionProposal::empty();
                                op.set_distribution(empty_distribution);
                            }
                        }
                    }

                    if let Some(uuid) = &op.create_collection.uuid {
                        log::warn!(
                            "Collection UUID {uuid} explicitly specified, \
                             when proposing create collection {} operation, \
                             new random UUID will be generated instead",
                            op.collection_name,
                        );
                    }

                    op.create_collection.uuid = Some(uuid::Uuid::new_v4());

                    CollectionMetaOperations::CreateCollection(op)
                }
                CollectionMetaOperations::CreateShardKey(op) => {
                    CollectionMetaOperations::CreateShardKey(op)
                }

                op => op,
            };

            let operation_awaiter =
                // If explicit timeout is set - then we need to wait for all expected operations.
                // E.g. in case of `CreateCollection` we will explicitly wait for all replicas to be activated.
                // We need to register receivers(by calling the function) before submitting the operation.
                if !expect_operations.is_empty() {
                    Some(state.await_for_multiple_operations(expect_operations, wait_timeout))
                } else {
                    None
                };

            let do_sync_nodes = match &op {
                // Sync nodes after collection or shard key creation
                CollectionMetaOperations::OrionControl { .. }
                | CollectionMetaOperations::CreateCollection(_)
                | CollectionMetaOperations::CreateShardKey(_) => true,

                // Sync nodes when creating or renaming collection aliases
                CollectionMetaOperations::ChangeAliases(changes) => {
                    changes.actions.iter().any(|change| match change {
                        AliasOperations::CreateAlias(_) | AliasOperations::RenameAlias(_) => true,
                        AliasOperations::DeleteAlias(_) => false,
                    })
                }

                // TODO(resharding): Do we need/want to synchronize `Resharding` operations?
                CollectionMetaOperations::Resharding(_, _) => false,

                // No need to sync nodes for other operations
                CollectionMetaOperations::UpdateCollection(_)
                | CollectionMetaOperations::DeleteCollection(_)
                | CollectionMetaOperations::TransferShard(_, _)
                | CollectionMetaOperations::SetShardReplicaState(_)
                | CollectionMetaOperations::DropShardKey(_)
                | CollectionMetaOperations::CreatePayloadIndex(_)
                | CollectionMetaOperations::DropPayloadIndex(_)
                | CollectionMetaOperations::Nop { .. } => false,

                #[cfg(feature = "staging")]
                CollectionMetaOperations::TestSlowDown(_) => false,
            };

            // During creation of a shard key, we must ensure that all replicas are ready to accept
            // write requests, so the client-side script can rely on the fact that the
            // shard creation request is complete.
            //
            // For this we explicitly wait for validation this, we do following checks:
            //
            // 1. Wait for consensus to accept shard create operation on current machine.
            //    ( here newly created shards should start to report state change from `Inactive` to `Active` )
            // 2. Wait for all local shards to become active.
            //    ( At this stage we are sure, that all consensus operations are created, but might not be applied everywhere )
            // 3. Wait for all remote peers to have at least the same state as the current peer.
            //    ( So we are sure, that all remote peers have also switched to `Active` state )
            let create_shard_key = match &op {
                CollectionMetaOperations::CreateShardKey(op) => {
                    let collection_name: CollectionName = op.collection_name.clone();
                    let shard_key = op.shard_key.clone();
                    let initial_state = op.initial_state;
                    Some((collection_name, shard_key, initial_state))
                }
                _ => None,
            };

            // Send operation to consensus and wait for it to be applied locally
            let res = state
                .propose_consensus_op_with_await(
                    ConsensusOperations::CollectionMeta(Box::new(op)),
                    wait_timeout,
                )
                .await?;

            if let Some(operation_awaiter) = operation_awaiter {
                // Actually await for expected operations to complete on the consensus
                match operation_awaiter.await {
                    Ok(Ok(())) => {} // all good
                    Ok(Err(err)) => {
                        log::warn!("Not all expected operations were completed: {err}")
                    }
                    Err(err) => log::warn!("Awaiting for expected operations timed out: {err}"),
                }
            }

            // Wait for shards activation
            if let Some((collection_name, shard_key, initial_state)) = create_shard_key
                && initial_state.is_none()
            {
                // Only do if initial state is not set because we only wanted to wait for Active since introducing
                // the Initial state which needs a transition to Active.
                let remaining_timeout =
                    wait_timeout.map(|timeout| timeout.saturating_sub(start.elapsed()));
                self.wait_for_shard_key_activation(collection_name, shard_key, remaining_timeout)
                    .await?;
            };

            // On some operations, synchronize all nodes to ensure all are ready for point operations
            if do_sync_nodes {
                let remaining_timeout =
                    wait_timeout.map(|timeout| timeout.saturating_sub(start.elapsed()));
                if let Err(err) = self.await_consensus_sync(remaining_timeout).await {
                    if strict_sync {
                        return Err(err);
                    }
                    log::warn!(
                        "Failed to synchronize all nodes after collection operation in time, some nodes may not be ready: {err}",
                    );
                }
            }

            Ok(res)
        } else {
            let toc = self.toc.clone();
            tokio::task::spawn(async move { toc.perform_collection_meta_op(operation).await })
                .await?
        }
    }

    pub fn cluster_status(&self) -> ClusterStatus {
        match self.consensus_state.as_ref() {
            Some(state) => state.cluster_status(),
            None => ClusterStatus::Disabled,
        }
    }

    pub async fn await_consensus_sync(
        &self,
        timeout: Option<Duration>,
    ) -> Result<(), StorageError> {
        let timeout = timeout.unwrap_or(CONSENSUS_META_OP_WAIT);

        let Some(state) = self.consensus_state.as_ref() else {
            return Ok(());
        };

        let state = state.hard_state();
        let term = state.term;
        let commit = state.commit;
        let channel_service = self.toc.get_channel_service();
        let this_peer_id = self.toc.this_peer_id;

        channel_service
            .await_commit_on_all_peers(this_peer_id, commit, term, timeout)
            .await?;

        log::debug!("Consensus is synchronized with term: {term}, commit: {commit}");

        Ok(())
    }

    /// Waits for all shards of a specific shard key to become active.
    pub async fn wait_for_shard_key_activation(
        &self,
        collection_name: CollectionName,
        shard_key: ShardKey,
        timeout: Option<Duration>,
    ) -> Result<(), StorageError> {
        let timeout = timeout.unwrap_or(CONSENSUS_META_OP_WAIT);

        let mut wait_for_active = FuturesUnordered::new();

        {
            let shard_holder = self
                .toc
                .get_collection(&CollectionMultipass.issue_pass(&collection_name))
                .await?
                .shards_holder()
                .read_owned()
                .await;

            for replica_set in shard_holder.all_shards() {
                if replica_set.shard_key() != Some(&shard_key) {
                    continue;
                }

                for (peer_id, replica_state) in replica_set.peers() {
                    if replica_state == ReplicaState::Active {
                        continue;
                    }

                    wait_for_active.push(replica_set.wait_for_state(
                        peer_id,
                        ReplicaState::Active,
                        timeout,
                    ));
                }
            }
        }

        while let Some(result) = wait_for_active.next().await {
            result?;
        }

        Ok(())
    }

    pub fn all_hw_metrics(&self) -> HashMap<String, HardwareUsage> {
        self.toc.all_hw_metrics()
    }

    #[must_use]
    pub fn get_collection_hw_metrics(&self, collection: String) -> Arc<HwSharedDrain> {
        self.toc.get_collection_hw_metrics(collection)
    }
}
