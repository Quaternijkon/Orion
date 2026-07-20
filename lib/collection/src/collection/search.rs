use std::collections::BTreeMap;
use std::mem;
use std::sync::Arc;
use std::time::Duration;

use ahash::{AHashMap, AHashSet};
use api::grpc::qdrant::{CoreSearchBatchByShardInternal, CoreSearchByShardEntry};
use common::counter::hardware_accumulator::HwMeasurementAcc;
use futures::{TryFutureExt, future};
use itertools::{Either, Itertools};
use segment::data_types::vectors::VectorInternal;
use segment::types::{
    ExtendedPointId, Filter, Order, ScoredPoint, ShardKey, WithPayloadInterface, WithVector,
};
use shard::query::query_enum::QueryEnum;
use shard::retrieve::record_internal::RecordInternal;
use shard::search::{CoreSearchRequest, CoreSearchRequestBatch};
use tokio::time::Instant;
use tokio_util::task::AbortOnDropHandle;

use super::Collection;
use crate::config::AutoShardPolicy;
use crate::events::SlowQueryEvent;
use crate::operations::consistency_params::ReadConsistency;
use crate::operations::shard_selector_internal::ShardSelectorInternal;
use crate::operations::types::*;
use crate::orion::OrionShardTarget;
use crate::shards::remote_shard::{CollectionCoreSearchRequest, RemoteShard};
use crate::shards::shard::{PeerId, ShardId};
use crate::simple_kmeans::SimpleKmeansShardTarget;

fn search_batch_requires_shard_specialization(request: &CoreSearchRequestBatch) -> bool {
    request.searches.iter().any(|search| {
        search.hnsw_entry_points_by_shard.is_some() || search.hnsw_ef_by_shard.is_some()
    })
}

fn orion_eligible_dense_query<'a>(
    request: &'a CoreSearchRequest,
    routing_vector_name: &str,
) -> Option<&'a [f32]> {
    if request.filter.is_some()
        || request.params.as_ref().is_some_and(|params| params.exact)
        || request.hnsw_entry_points.is_some()
        || request.hnsw_entry_points_by_shard.is_some()
        || request.hnsw_ef_by_shard.is_some()
        || request.source_id_dedup_block_size.is_some()
        || request.query.get_vector_name() != routing_vector_name
    {
        return None;
    }

    let QueryEnum::Nearest(named_query) = &request.query else {
        return None;
    };
    let VectorInternal::Dense(vector) = &named_query.query else {
        return None;
    };
    Some(vector)
}

fn specialize_core_search_for_orion_target(
    request: &CoreSearchRequest,
    target: &OrionShardTarget,
) -> CoreSearchRequest {
    let mut specialized = request.clone();
    specialized.limit = request.limit.saturating_add(request.offset);
    specialized.offset = 0;
    specialized.hnsw_entry_points = Some(target.entry_points.clone());
    specialized.hnsw_entry_points_by_shard = None;
    specialized.hnsw_ef_by_shard = None;
    specialized.source_id_dedup_block_size = None;

    let mut params = specialized.params.unwrap_or_default();
    params.hnsw_ef = Some(target.ef);
    specialized.params = Some(params);
    specialized
}

fn specialize_core_search_for_simple_kmeans_target(
    request: &CoreSearchRequest,
    target: &SimpleKmeansShardTarget,
) -> CoreSearchRequest {
    let mut specialized = request.clone();
    specialized.limit = request.limit.saturating_add(request.offset);
    specialized.offset = 0;
    // Simple KMeans selects shards only. Lower HNSW starts from its ordinary entry point.
    specialized.hnsw_entry_points = None;
    specialized.hnsw_entry_points_by_shard = None;
    specialized.hnsw_ef_by_shard = None;
    specialized.source_id_dedup_block_size = None;

    let mut params = specialized.params.unwrap_or_default();
    params.hnsw_ef = Some(target.ef);
    specialized.params = Some(params);
    specialized
}

fn remaining_search_timeout(
    timeout: Option<Duration>,
    elapsed: Duration,
    operation: &str,
) -> CollectionResult<Option<Duration>> {
    let Some(timeout) = timeout else {
        return Ok(None);
    };
    let Some(remaining) = timeout.checked_sub(elapsed) else {
        return Err(CollectionError::timeout(timeout, operation));
    };
    if remaining.is_zero() {
        return Err(CollectionError::timeout(timeout, operation));
    }
    Ok(Some(remaining))
}

fn routing_chunk_size(query_count: usize, search_thread_count: usize) -> usize {
    let task_count = query_count.min(search_thread_count.max(1)).max(1);
    query_count.max(1).div_ceil(task_count)
}

pub(crate) fn specialize_search_batch_for_shard(
    request: &CoreSearchRequestBatch,
    shard_key: Option<&ShardKey>,
) -> CoreSearchRequestBatch {
    let searches = request
        .searches
        .iter()
        .map(|search| {
            let mut specialized = search.clone();

            if let Some(shard_key) = shard_key {
                if let Some(entry_points) = search
                    .hnsw_entry_points_by_shard
                    .as_ref()
                    .and_then(|entry_points_by_shard| entry_points_by_shard.get(shard_key))
                {
                    specialized.hnsw_entry_points = Some(entry_points.clone());
                }

                if let Some(hnsw_ef) = search
                    .hnsw_ef_by_shard
                    .as_ref()
                    .and_then(|ef_by_shard| ef_by_shard.get(shard_key))
                    .copied()
                {
                    let mut params = specialized.params.unwrap_or_default();
                    params.hnsw_ef = Some(hnsw_ef);
                    specialized.params = Some(params);
                }
            }

            specialized.hnsw_entry_points_by_shard = None;
            specialized.hnsw_ef_by_shard = None;
            specialized
        })
        .collect();

    CoreSearchRequestBatch { searches }
}

fn specialize_core_search_for_shard_major(
    request: &CoreSearchRequest,
    shard_key: &ShardKey,
) -> CoreSearchRequest {
    let mut specialized = request.clone();

    specialized.limit = request.limit.saturating_add(request.offset);
    specialized.offset = 0;

    if let Some(entry_points) = request
        .hnsw_entry_points_by_shard
        .as_ref()
        .and_then(|entry_points_by_shard| entry_points_by_shard.get(shard_key))
    {
        specialized.hnsw_entry_points = Some(entry_points.clone());
    }

    if let Some(hnsw_ef) = request
        .hnsw_ef_by_shard
        .as_ref()
        .and_then(|ef_by_shard| ef_by_shard.get(shard_key))
        .copied()
    {
        let mut params = specialized.params.unwrap_or_default();
        params.hnsw_ef = Some(hnsw_ef);
        specialized.params = Some(params);
    }

    specialized.hnsw_entry_points_by_shard = None;
    specialized.hnsw_ef_by_shard = None;
    specialized
}

fn search_dedup_point_id(
    point_id: ExtendedPointId,
    source_id_dedup_block_size: Option<u64>,
) -> ExtendedPointId {
    let Some(block_size) = source_id_dedup_block_size.filter(|block_size| *block_size > 0) else {
        return point_id;
    };
    let ExtendedPointId::NumId(point_num_id) = point_id else {
        return point_id;
    };
    if point_num_id == 0 {
        return point_id;
    }
    ExtendedPointId::NumId((point_num_id - 1) % block_size)
}

struct PeerShardMajorGroup {
    remote: RemoteShard,
    original_indices: Vec<usize>,
    is_payload_required_by_query: Vec<bool>,
    entries: Vec<CoreSearchByShardEntry>,
}

impl PeerShardMajorGroup {
    fn new(remote: RemoteShard) -> Self {
        Self {
            remote,
            original_indices: Vec::new(),
            is_payload_required_by_query: Vec::new(),
            entries: Vec::new(),
        }
    }

    fn local_query_index(&mut self, original_index: usize, request: &CoreSearchRequest) -> usize {
        peer_local_query_index(
            &mut self.original_indices,
            &mut self.is_payload_required_by_query,
            original_index,
            request,
        )
    }
}

fn peer_local_query_index(
    original_indices: &mut Vec<usize>,
    is_payload_required_by_query: &mut Vec<bool>,
    original_index: usize,
    request: &CoreSearchRequest,
) -> usize {
    if let Some(query_index) = original_indices
        .iter()
        .position(|known_index| *known_index == original_index)
    {
        return query_index;
    }

    let query_index = original_indices.len();
    original_indices.push(original_index);
    is_payload_required_by_query.push(
        request
            .with_payload
            .as_ref()
            .is_some_and(|with_payload| with_payload.is_required()),
    );
    query_index
}

fn peer_premerge_disabled() -> bool {
    std::env::var("QDRANT_DISABLE_SHARD_MAJOR_PEER_PREMERGE").is_ok_and(|value| {
        let value = value.to_ascii_lowercase();
        matches!(value.as_str(), "1" | "true" | "yes" | "on")
    })
}

fn numeric_peer_premerge_request_is_safe(
    replication_factor: u32,
    read_consistency: Option<ReadConsistency>,
    all_queries_are_large_better: bool,
) -> bool {
    replication_factor == 1 && read_consistency.is_none() && all_queries_are_large_better
}

fn numeric_peer_premerge_replica_peer(
    this_peer_id: PeerId,
    configured_replica_count: usize,
    readable_peers: &[PeerId],
    has_local_shard: bool,
    has_remote: bool,
) -> Option<PeerId> {
    if configured_replica_count != 1
        || readable_peers.len() != 1
        || has_local_shard
        || readable_peers[0] == this_peer_id
        || !has_remote
    {
        return None;
    }
    Some(readable_peers[0])
}

fn peer_premerge_entry(
    collection_id: &str,
    local_query_index: usize,
    shard_id: ShardId,
    specialized: &CoreSearchRequest,
    original: &CoreSearchRequest,
) -> CoreSearchByShardEntry {
    let search_points = CollectionCoreSearchRequest((collection_id.to_owned(), specialized)).into();
    CoreSearchByShardEntry {
        query_index: local_query_index as u64,
        shard_id,
        search_points: Some(search_points),
        final_limit: original.limit as u64,
        final_offset: Some(original.offset as u64),
        source_id_dedup_block_size: original.source_id_dedup_block_size,
    }
}

impl Collection {
    #[cfg(feature = "testing")]
    pub async fn search(
        &self,
        request: CoreSearchRequest,
        read_consistency: Option<ReadConsistency>,
        shard_selection: &ShardSelectorInternal,
        timeout: Option<Duration>,
        hw_measurement_acc: HwMeasurementAcc,
    ) -> CollectionResult<Vec<ScoredPoint>> {
        if request.limit == 0 {
            return Ok(vec![]);
        }
        // search is a special case of search_batch with a single batch
        let request_batch = CoreSearchRequestBatch {
            searches: vec![request],
        };
        let results = self
            .do_core_search_batch(
                request_batch,
                read_consistency,
                shard_selection,
                timeout,
                hw_measurement_acc,
            )
            .await?;
        Ok(results.into_iter().next().unwrap())
    }

    pub async fn core_search_batch(
        &self,
        request: CoreSearchRequestBatch,
        read_consistency: Option<ReadConsistency>,
        shard_selection: ShardSelectorInternal,
        timeout: Option<Duration>,
        hw_measurement_acc: HwMeasurementAcc,
    ) -> CollectionResult<Vec<Vec<ScoredPoint>>> {
        let start = Instant::now();
        // shortcuts batch if all requests with limit=0
        if request.searches.iter().all(|s| s.limit == 0) {
            return Ok(vec![]);
        }

        let is_payload_required = request
            .searches
            .iter()
            .all(|s| s.with_payload.as_ref().is_some_and(|p| p.is_required()));
        let with_vectors = request
            .searches
            .iter()
            .all(|s| s.with_vector.as_ref().is_some_and(|wv| wv.is_enabled()));

        let metadata_required = is_payload_required || with_vectors;

        let sum_limits: usize = request.searches.iter().map(|s| s.limit).sum();
        let sum_offsets: usize = request.searches.iter().map(|s| s.offset).sum();

        // Number of records we need to retrieve to fill the search result.
        let require_transfers = self.shards_holder.read().await.len() * (sum_limits + sum_offsets);
        // Actually used number of records.
        let used_transfers = sum_limits;

        let is_required_transfer_large_enough = require_transfers
            > used_transfers.saturating_mul(super::query::PAYLOAD_TRANSFERS_FACTOR_THRESHOLD);

        if metadata_required && is_required_transfer_large_enough {
            // If there is a significant offset, we need to retrieve the whole result
            // set without payload first and then retrieve the payload.
            // It is required to do this because the payload might be too large to send over the
            // network.
            let mut without_payload_requests = Vec::with_capacity(request.searches.len());
            for search in &request.searches {
                let mut without_payload_request = search.clone();
                without_payload_request
                    .with_payload
                    .replace(WithPayloadInterface::Bool(false));
                without_payload_request
                    .with_vector
                    .replace(WithVector::Bool(false));
                without_payload_requests.push(without_payload_request);
            }
            let without_payload_batch = CoreSearchRequestBatch {
                searches: without_payload_requests,
            };
            let without_payload_results = self
                .do_core_search_batch(
                    without_payload_batch,
                    read_consistency,
                    &shard_selection,
                    timeout,
                    hw_measurement_acc.clone(),
                )
                .await?;
            // update timeout
            let timeout = timeout.map(|t| t.saturating_sub(start.elapsed()));
            let filled_results = without_payload_results
                .into_iter()
                .zip(request.searches.into_iter())
                .map(|(without_payload_result, req)| {
                    self.fill_search_result_with_payload(
                        without_payload_result,
                        req.with_payload.clone(),
                        req.with_vector.unwrap_or_default(),
                        read_consistency,
                        &shard_selection,
                        timeout,
                        hw_measurement_acc.clone(),
                    )
                });
            future::try_join_all(filled_results).await
        } else {
            let result = self
                .do_core_search_batch(
                    request,
                    read_consistency,
                    &shard_selection,
                    timeout,
                    hw_measurement_acc,
                )
                .await?;
            Ok(result)
        }
    }

    pub async fn core_search_batch_shard_major_peer_premerge(
        &self,
        requests: Vec<(CoreSearchRequest, ShardSelectorInternal)>,
        read_consistency: Option<ReadConsistency>,
        timeout: Option<Duration>,
        hw_measurement_acc: HwMeasurementAcc,
    ) -> CollectionResult<Option<Vec<Vec<ScoredPoint>>>> {
        if requests.is_empty() {
            return Ok(Some(Vec::new()));
        }

        if read_consistency.is_some() {
            return Ok(None);
        }

        let original_requests = requests
            .iter()
            .map(|(request, _selector)| request.clone())
            .collect::<Vec<_>>();
        let mut peer_groups: BTreeMap<PeerId, PeerShardMajorGroup> = BTreeMap::new();

        {
            let shard_holder = self.shards_holder.read().await;
            for (original_index, (request, selector)) in requests.iter().enumerate() {
                let ShardSelectorInternal::ShardKeys(shard_keys) = selector else {
                    return Ok(None);
                };

                if shard_keys.is_empty() {
                    return Ok(None);
                }

                for shard_key in shard_keys {
                    let shard_selector = ShardSelectorInternal::ShardKey(shard_key.clone());
                    let target_shards = shard_holder.select_shards(&shard_selector)?;
                    if target_shards.len() != 1 {
                        return Ok(None);
                    }

                    let replica_set = target_shards[0].0;
                    let readable_peers = replica_set.readable_shards();
                    if readable_peers.len() != 1 {
                        return Ok(None);
                    }

                    let peer_id = readable_peers[0];
                    if peer_id == self.this_peer_id {
                        return Ok(None);
                    }

                    let Some(remote) = replica_set.remote_shard_for_peer(peer_id).await else {
                        return Ok(None);
                    };

                    let group = peer_groups
                        .entry(peer_id)
                        .or_insert_with(|| PeerShardMajorGroup::new(remote));
                    let local_query_index = group.local_query_index(original_index, request);
                    let specialized = specialize_core_search_for_shard_major(request, shard_key);
                    group.entries.push(peer_premerge_entry(
                        &self.id,
                        local_query_index,
                        replica_set.shard_id,
                        &specialized,
                        request,
                    ));
                }
            }
        }

        if peer_groups.is_empty() {
            return Ok(None);
        }

        let request = Arc::new(CoreSearchRequestBatch {
            searches: original_requests,
        });
        self.execute_peer_shard_major_premerge(peer_groups, request, timeout, hw_measurement_acc)
            .await
            .map(Some)
    }

    async fn execute_peer_shard_major_premerge(
        &self,
        peer_groups: BTreeMap<PeerId, PeerShardMajorGroup>,
        original_requests: Arc<CoreSearchRequestBatch>,
        timeout: Option<Duration>,
        hw_measurement_acc: HwMeasurementAcc,
    ) -> CollectionResult<Vec<Vec<ScoredPoint>>> {
        let processed_timeout = RemoteShard::process_read_timeout(timeout, "search")?;
        let peer_searches = peer_groups.into_values().map(|group| {
            let collection_name = self.id.clone();
            let hw_measurement_acc = hw_measurement_acc.clone();
            async move {
                let original_indices = group.original_indices;
                let is_payload_required_by_query = group.is_payload_required_by_query;
                let request = CoreSearchBatchByShardInternal {
                    collection_name,
                    searches: group.entries,
                    timeout: processed_timeout.map(|timeout| timeout.as_secs()),
                };
                let rows = group
                    .remote
                    .core_search_batch_by_shard(
                        request,
                        processed_timeout,
                        is_payload_required_by_query,
                        hw_measurement_acc,
                    )
                    .await?;

                if rows.len() != original_indices.len() {
                    return Err(CollectionError::service_error(format!(
                        "Peer-local shard-major search returned {} rows for {} query slots",
                        rows.len(),
                        original_indices.len(),
                    )));
                }

                Ok::<_, CollectionError>((original_indices, rows))
            }
        });

        let peer_results = future::try_join_all(peer_searches).await?;
        let mut all_peer_rows = Vec::with_capacity(peer_results.len());
        for (original_indices, rows) in peer_results {
            let mut peer_rows = vec![Vec::new(); original_requests.searches.len()];
            for (original_index, row) in original_indices.into_iter().zip(rows) {
                peer_rows[original_index] = row;
            }
            all_peer_rows.push(peer_rows);
        }

        self.merge_from_shards(all_peer_rows, original_requests, true)
            .await
    }

    /// Batch Orion's numeric auto-shard searches by the worker peer that owns them.
    ///
    /// The fast path deliberately has a narrow safety envelope. It is used only for an RF=1
    /// collection with no explicit read consistency and a large-better distance, where every
    /// selected shard has exactly one configured/readable replica, that replica is remote, and the
    /// coordinator has no local shard object for it. The order restriction matches the existing
    /// peer-local merge RPC's unambiguous ordering contract; small-better distances keep the
    /// ordinary path. Under these conditions the worker-side `ShardId` request still enters the
    /// ordinary `ShardReplicaSet` read path, while one internal RPC can carry all numeric shards
    /// owned by that peer. Any topology, ordering, or consistency ambiguity falls back to the
    /// existing per-shard coordinator path before issuing a peer-batched request.
    async fn try_orion_numeric_peer_premerge(
        &self,
        searches_by_shard: &BTreeMap<ShardId, Vec<(usize, CoreSearchRequest)>>,
        original_requests: Arc<CoreSearchRequestBatch>,
        read_consistency: Option<ReadConsistency>,
        timeout: Option<Duration>,
        hw_measurement_acc: HwMeasurementAcc,
    ) -> CollectionResult<Option<Vec<Vec<ScoredPoint>>>> {
        if searches_by_shard.is_empty() || peer_premerge_disabled() {
            return Ok(None);
        }
        let topology_check_started = Instant::now();

        let collection_params = self.collection_config.read().await.params.clone();
        let all_queries_are_large_better = original_requests
            .searches
            .iter()
            .map(|request| {
                collection_params
                    .get_distance(request.query.get_vector_name())
                    .map(|distance| distance.distance_order() == Order::LargeBetter)
            })
            .collect::<CollectionResult<Vec<_>>>()?
            .into_iter()
            .all(|large_better| large_better);
        if !numeric_peer_premerge_request_is_safe(
            collection_params.replication_factor.get(),
            read_consistency,
            all_queries_are_large_better,
        ) {
            return Ok(None);
        }

        let mut peer_groups: BTreeMap<PeerId, PeerShardMajorGroup> = BTreeMap::new();
        {
            let shard_holder = self.shards_holder.read().await;
            for (&shard_id, searches) in searches_by_shard {
                let Some(replica_set) = shard_holder.get_shard(shard_id) else {
                    return Ok(None);
                };

                let configured_replica_count = replica_set.peers().len();
                let readable_peers = replica_set.readable_shards();
                let has_local_shard = replica_set.has_local_shard().await;
                let remote = if readable_peers.len() == 1 {
                    replica_set.remote_shard_for_peer(readable_peers[0]).await
                } else {
                    None
                };
                let Some(peer_id) = numeric_peer_premerge_replica_peer(
                    self.this_peer_id,
                    configured_replica_count,
                    &readable_peers,
                    has_local_shard,
                    remote.is_some(),
                ) else {
                    return Ok(None);
                };
                let remote = remote.expect("numeric peer-premerge safety checked remote presence");

                let group = peer_groups
                    .entry(peer_id)
                    .or_insert_with(|| PeerShardMajorGroup::new(remote));
                for (original_index, specialized) in searches {
                    let Some(original) = original_requests.searches.get(*original_index) else {
                        return Err(CollectionError::service_error(format!(
                            "Orion numeric peer-premerge query index {original_index} is outside batch size {}",
                            original_requests.searches.len(),
                        )));
                    };
                    let local_query_index = group.local_query_index(*original_index, original);
                    group.entries.push(peer_premerge_entry(
                        &self.id,
                        local_query_index,
                        shard_id,
                        specialized,
                        original,
                    ));
                }
            }
        }

        if peer_groups.is_empty() {
            return Ok(None);
        }

        let peer_timeout = remaining_search_timeout(
            timeout,
            topology_check_started.elapsed(),
            "Orion numeric peer-premerge topology check",
        )?;

        self.execute_peer_shard_major_premerge(
            peer_groups,
            original_requests,
            peer_timeout,
            hw_measurement_acc,
        )
        .await
        .map(Some)
    }

    async fn do_core_search_batch(
        &self,
        request: CoreSearchRequestBatch,
        read_consistency: Option<ReadConsistency>,
        shard_selection: &ShardSelectorInternal,
        timeout: Option<Duration>,
        hw_measurement_acc: HwMeasurementAcc,
    ) -> CollectionResult<Vec<Vec<ScoredPoint>>> {
        self.ensure_static_router_available(shard_selection).await?;
        let request = Arc::new(request);
        let instant = Instant::now();

        if let Some(result) = self
            .try_orion_core_search_batch(
                request.clone(),
                read_consistency,
                shard_selection,
                timeout,
                instant,
                hw_measurement_acc.clone(),
            )
            .await?
        {
            let filters_refs = request.searches.iter().map(|req| req.filter.as_ref());
            self.post_process_if_slow_request(instant.elapsed(), filters_refs);
            return Ok(result);
        }

        if let Some(result) = self
            .try_simple_kmeans_core_search_batch(
                request.clone(),
                read_consistency,
                shard_selection,
                timeout,
                instant,
                hw_measurement_acc.clone(),
            )
            .await?
        {
            let filters_refs = request.searches.iter().map(|req| req.filter.as_ref());
            self.post_process_if_slow_request(instant.elapsed(), filters_refs);
            return Ok(result);
        }

        let timeout = remaining_search_timeout(timeout, instant.elapsed(), "search")?;
        let requires_shard_specialization = search_batch_requires_shard_specialization(&request);

        // query all shards concurrently
        let all_searches_res = {
            let shard_holder = self.shards_holder.read().await;
            let target_shards = shard_holder.select_shards(shard_selection)?;
            let all_searches = target_shards.into_iter().map(|(shard, shard_key)| {
                let shard_key = shard_key.cloned();
                let shard_request = if requires_shard_specialization {
                    Arc::new(specialize_search_batch_for_shard(
                        &request,
                        shard_key.as_ref(),
                    ))
                } else {
                    request.clone()
                };
                shard
                    .core_search(
                        shard_request,
                        read_consistency,
                        shard_selection.is_shard_id(),
                        timeout,
                        hw_measurement_acc.clone(),
                    )
                    .and_then(move |mut records| async move {
                        if shard_key.is_none() {
                            return Ok(records);
                        }
                        for batch in &mut records {
                            for point in batch {
                                point.shard_key.clone_from(&shard_key);
                            }
                        }
                        Ok(records)
                    })
            });
            future::try_join_all(all_searches).await?
        };

        let result = self
            .merge_from_shards(
                all_searches_res,
                request.clone(),
                !shard_selection.is_shard_id(),
            )
            .await;

        let filters_refs = request.searches.iter().map(|req| req.filter.as_ref());

        self.post_process_if_slow_request(instant.elapsed(), filters_refs);

        result
    }

    /// A declared static routing policy must never silently become a different all-shards
    /// algorithm because its serving artifact failed to load. Collection creation and controlled
    /// numeric-shard import can still proceed before activation, while ordinary coordinator reads
    /// fail closed until every node has loaded the configured generation.
    pub(super) async fn ensure_static_router_available(
        &self,
        shard_selection: &ShardSelectorInternal,
    ) -> CollectionResult<()> {
        if !matches!(shard_selection, ShardSelectorInternal::All) {
            return Ok(());
        }

        let policy = self
            .collection_config
            .read()
            .await
            .auto_shard_policy
            .clone();
        let unavailable = match policy {
            Some(AutoShardPolicy::Orion { generation, .. }) if self.orion_router.is_none() => {
                Some(("Orion", generation))
            }
            Some(AutoShardPolicy::SimpleKmeans { generation, .. })
                if self.simple_kmeans_router.is_none() =>
            {
                Some(("Simple KMeans", generation))
            }
            _ => None,
        };

        if let Some((policy_name, generation)) = unavailable {
            return Err(CollectionError::service_error(format!(
                "{policy_name} routing generation {generation} is configured but unavailable; refusing a silent all-shards coordinator fallback"
            )));
        }
        Ok(())
    }

    /// Execute a native Orion route through ordinary numeric shard replica sets.
    ///
    /// Returning `Ok(None)` is an intentional, request-wide fallback to Qdrant's normal
    /// all-shards behavior. We never execute a partial Orion route.
    async fn try_orion_core_search_batch(
        &self,
        request: Arc<CoreSearchRequestBatch>,
        read_consistency: Option<ReadConsistency>,
        shard_selection: &ShardSelectorInternal,
        timeout: Option<Duration>,
        request_started: Instant,
        hw_measurement_acc: HwMeasurementAcc,
    ) -> CollectionResult<Option<Vec<Vec<ScoredPoint>>>> {
        if !matches!(shard_selection, ShardSelectorInternal::All) {
            return Ok(None);
        }
        let Some(router) = self.orion_router.as_ref() else {
            return Ok(None);
        };

        let mut routable_queries = Vec::with_capacity(request.searches.len());
        for (query_index, original) in request.searches.iter().enumerate() {
            let Some(query) = orion_eligible_dense_query(original, router.vector_name()) else {
                return Ok(None);
            };
            routable_queries.push((query_index, query.to_vec()));
        }

        // Upper routing is CPU-only and independent per query. Execute it on Qdrant's search
        // runtime so a normal search batch does not serialize every upper-HNSW traversal on the
        // async coordinator thread.
        let chunk_size = routing_chunk_size(
            routable_queries.len(),
            self.shared_storage_config.search_thread_count,
        );
        let route_tasks = routable_queries
            .chunks(chunk_size)
            .map(|chunk| {
                let chunk = chunk.to_vec();
                let router = Arc::clone(router);
                let cpu_utilization = hw_measurement_acc.cpu_utilization();
                AbortOnDropHandle::new(self.search_runtime.spawn_blocking(move || {
                    chunk
                        .into_iter()
                        .map(|(query_index, query)| {
                            (
                                query_index,
                                cpu_utilization.measure(|| router.route_query(&query)),
                            )
                        })
                        .collect::<Vec<_>>()
                }))
            })
            .collect::<Vec<_>>();
        let routed_query_chunks = match remaining_search_timeout(
            timeout,
            request_started.elapsed(),
            "Orion upper routing",
        )? {
            Some(remaining) => tokio::time::timeout(remaining, future::join_all(route_tasks))
                .await
                .map_err(|_| CollectionError::timeout(timeout.unwrap(), "Orion upper routing"))?,
            None => future::join_all(route_tasks).await,
        };

        let mut searches_by_shard: BTreeMap<ShardId, Vec<(usize, CoreSearchRequest)>> =
            BTreeMap::new();
        for routed_query_chunk in routed_query_chunks {
            let routed_query_chunk = routed_query_chunk.map_err(|err| {
                CollectionError::service_error(format!(
                    "Orion upper-routing task failed for collection {}: {err}",
                    self.id,
                ))
            })?;
            for (query_index, route_result) in routed_query_chunk {
                let targets = match route_result {
                    Ok(targets) if !targets.is_empty() => targets,
                    Ok(_) => {
                        log::warn!(
                            "Orion routing generation {} returned no shards for collection {}; falling back to all shards for the batch",
                            router.generation(),
                            self.id,
                        );
                        return Ok(None);
                    }
                    Err(err) => {
                        log::warn!(
                            "Orion routing generation {} failed for collection {}: {err}; falling back to all shards for the batch",
                            router.generation(),
                            self.id,
                        );
                        return Ok(None);
                    }
                };

                let original = &request.searches[query_index];
                for target in targets {
                    let specialized = specialize_core_search_for_orion_target(original, &target);
                    searches_by_shard
                        .entry(target.shard_id)
                        .or_default()
                        .push((query_index, specialized));
                }
            }
        }

        if searches_by_shard.is_empty() {
            return Ok(None);
        }

        let lower_timeout = remaining_search_timeout(
            timeout,
            request_started.elapsed(),
            "Orion distributed search",
        )?;

        if let Some(rows) = self
            .try_orion_numeric_peer_premerge(
                &searches_by_shard,
                request.clone(),
                read_consistency,
                lower_timeout,
                hw_measurement_acc.clone(),
            )
            .await?
        {
            return Ok(Some(rows));
        }

        let batch_size = request.searches.len();
        let all_shard_rows = {
            let shard_holder = self.shards_holder.read().await;
            if let Some(missing_shard) = searches_by_shard
                .keys()
                .copied()
                .find(|shard_id| shard_holder.get_shard(*shard_id).is_none())
            {
                log::warn!(
                    "Orion routing generation {} selected missing shard {missing_shard} for collection {}; falling back to all shards for the batch",
                    router.generation(),
                    self.id,
                );
                return Ok(None);
            }

            let shard_searches = searches_by_shard.into_iter().map(|(shard_id, searches)| {
                let shard = shard_holder
                    .get_shard(shard_id)
                    .expect("Orion target shard existence was checked");
                let original_indices = searches
                    .iter()
                    .map(|(query_index, _)| *query_index)
                    .collect::<Vec<_>>();
                let shard_request = Arc::new(CoreSearchRequestBatch {
                    searches: searches.into_iter().map(|(_, search)| search).collect(),
                });
                let hw_measurement_acc = hw_measurement_acc.clone();
                async move {
                    let rows = shard
                        .core_search(
                            shard_request,
                            read_consistency,
                            false,
                            lower_timeout,
                            hw_measurement_acc,
                        )
                        .await?;
                    if rows.len() != original_indices.len() {
                        return Err(CollectionError::service_error(format!(
                            "Orion shard {shard_id} returned {} rows for {} query slots",
                            rows.len(),
                            original_indices.len(),
                        )));
                    }

                    let mut full_batch_rows = vec![Vec::new(); batch_size];
                    for (query_index, row) in original_indices.into_iter().zip(rows) {
                        full_batch_rows[query_index] = row;
                    }
                    Ok::<_, CollectionError>(full_batch_rows)
                }
            });
            future::try_join_all(shard_searches).await?
        };

        self.merge_from_shards(all_shard_rows, request, true)
            .await
            .map(Some)
    }

    /// Execute a static Simple KMeans nprobe route through ordinary numeric shard replica sets.
    ///
    /// Eligibility and failures are batch-wide: any unsupported request or invalid route falls
    /// back to Qdrant's normal all-shards path, never to a partial centroid route.
    async fn try_simple_kmeans_core_search_batch(
        &self,
        request: Arc<CoreSearchRequestBatch>,
        read_consistency: Option<ReadConsistency>,
        shard_selection: &ShardSelectorInternal,
        timeout: Option<Duration>,
        request_started: Instant,
        hw_measurement_acc: HwMeasurementAcc,
    ) -> CollectionResult<Option<Vec<Vec<ScoredPoint>>>> {
        if !matches!(shard_selection, ShardSelectorInternal::All) {
            return Ok(None);
        }
        let Some(router) = self.simple_kmeans_router.as_ref() else {
            return Ok(None);
        };

        let mut routable_queries = Vec::with_capacity(request.searches.len());
        for (query_index, original) in request.searches.iter().enumerate() {
            let Some(query) = orion_eligible_dense_query(original, router.vector_name()) else {
                return Ok(None);
            };
            routable_queries.push((query_index, query.to_vec()));
        }

        let chunk_size = routing_chunk_size(
            routable_queries.len(),
            self.shared_storage_config.search_thread_count,
        );
        let route_tasks = routable_queries
            .chunks(chunk_size)
            .map(|chunk| {
                let chunk = chunk.to_vec();
                let router = Arc::clone(router);
                let cpu_utilization = hw_measurement_acc.cpu_utilization();
                AbortOnDropHandle::new(self.search_runtime.spawn_blocking(move || {
                    chunk
                        .into_iter()
                        .map(|(query_index, query)| {
                            (
                                query_index,
                                cpu_utilization.measure(|| router.route_query(&query)),
                            )
                        })
                        .collect::<Vec<_>>()
                }))
            })
            .collect::<Vec<_>>();
        let routed_query_chunks = match remaining_search_timeout(
            timeout,
            request_started.elapsed(),
            "Simple KMeans centroid routing",
        )? {
            Some(remaining) => tokio::time::timeout(remaining, future::join_all(route_tasks))
                .await
                .map_err(|_| {
                    CollectionError::timeout(timeout.unwrap(), "Simple KMeans centroid routing")
                })?,
            None => future::join_all(route_tasks).await,
        };

        let mut searches_by_shard: BTreeMap<ShardId, Vec<(usize, CoreSearchRequest)>> =
            BTreeMap::new();
        for routed_query_chunk in routed_query_chunks {
            let routed_query_chunk = routed_query_chunk.map_err(|err| {
                CollectionError::service_error(format!(
                    "Simple KMeans routing task failed for collection {}: {err}",
                    self.id,
                ))
            })?;
            for (query_index, route_result) in routed_query_chunk {
                let targets = match route_result {
                    Ok(targets) if !targets.is_empty() => targets,
                    Ok(_) => {
                        log::warn!(
                            "Simple KMeans routing generation {} returned no shards for collection {}; falling back to all shards for the batch",
                            router.generation(),
                            self.id,
                        );
                        return Ok(None);
                    }
                    Err(err) => {
                        log::warn!(
                            "Simple KMeans routing generation {} failed for collection {}: {err}; falling back to all shards for the batch",
                            router.generation(),
                            self.id,
                        );
                        return Ok(None);
                    }
                };

                let original = &request.searches[query_index];
                for target in targets {
                    let specialized =
                        specialize_core_search_for_simple_kmeans_target(original, &target);
                    searches_by_shard
                        .entry(target.shard_id)
                        .or_default()
                        .push((query_index, specialized));
                }
            }
        }

        if searches_by_shard.is_empty() {
            return Ok(None);
        }
        let lower_timeout = remaining_search_timeout(
            timeout,
            request_started.elapsed(),
            "Simple KMeans distributed search",
        )?;

        let batch_size = request.searches.len();
        let all_shard_rows = {
            let shard_holder = self.shards_holder.read().await;
            if let Some(missing_shard) = searches_by_shard
                .keys()
                .copied()
                .find(|shard_id| shard_holder.get_shard(*shard_id).is_none())
            {
                log::warn!(
                    "Simple KMeans routing generation {} selected missing shard {missing_shard} for collection {}; falling back to all shards for the batch",
                    router.generation(),
                    self.id,
                );
                return Ok(None);
            }

            let shard_searches = searches_by_shard.into_iter().map(|(shard_id, searches)| {
                let shard = shard_holder
                    .get_shard(shard_id)
                    .expect("Simple KMeans target shard existence was checked");
                let original_indices = searches
                    .iter()
                    .map(|(query_index, _)| *query_index)
                    .collect::<Vec<_>>();
                let shard_request = Arc::new(CoreSearchRequestBatch {
                    searches: searches.into_iter().map(|(_, search)| search).collect(),
                });
                let hw_measurement_acc = hw_measurement_acc.clone();
                async move {
                    let rows = shard
                        .core_search(
                            shard_request,
                            read_consistency,
                            false,
                            lower_timeout,
                            hw_measurement_acc,
                        )
                        .await?;
                    if rows.len() != original_indices.len() {
                        return Err(CollectionError::service_error(format!(
                            "Simple KMeans shard {shard_id} returned {} rows for {} query slots",
                            rows.len(),
                            original_indices.len(),
                        )));
                    }

                    let mut full_batch_rows = vec![Vec::new(); batch_size];
                    for (query_index, row) in original_indices.into_iter().zip(rows) {
                        full_batch_rows[query_index] = row;
                    }
                    Ok::<_, CollectionError>(full_batch_rows)
                }
            });
            future::try_join_all(shard_searches).await?
        };

        self.merge_from_shards(all_shard_rows, request, true)
            .await
            .map(Some)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) async fn fill_search_result_with_payload(
        &self,
        search_result: Vec<ScoredPoint>,
        with_payload: Option<WithPayloadInterface>,
        with_vector: WithVector,
        read_consistency: Option<ReadConsistency>,
        shard_selection: &ShardSelectorInternal,
        timeout: Option<Duration>,
        hw_measurement_acc: HwMeasurementAcc,
    ) -> CollectionResult<Vec<ScoredPoint>> {
        // short-circuit if not needed
        if let (&Some(WithPayloadInterface::Bool(false)), &WithVector::Bool(false)) =
            (&with_payload, &with_vector)
        {
            return Ok(search_result
                .into_iter()
                .map(|point| ScoredPoint {
                    payload: None,
                    vector: None,
                    ..point
                })
                .collect());
        };

        let retrieve_request = PointRequestInternal {
            ids: search_result.iter().map(|x| x.id).collect(),
            with_payload,
            with_vector,
        };
        let retrieved_records = self
            .retrieve(
                retrieve_request,
                read_consistency,
                shard_selection,
                timeout,
                hw_measurement_acc,
            )
            .await?;

        let mut records_map: AHashMap<ExtendedPointId, RecordInternal> = retrieved_records
            .into_iter()
            .map(|rec| (rec.id, rec))
            .collect();
        let enriched_result = search_result
            .into_iter()
            .filter_map(|mut scored_point| {
                // Points might get deleted between search and retrieve.
                // But it's not a problem, because we don't want to return deleted points.
                // So we just filter out them.
                records_map.remove(&scored_point.id).map(|record| {
                    scored_point.payload = record.payload;
                    scored_point.vector = record.vector;
                    scored_point
                })
            })
            .collect();
        Ok(enriched_result)
    }

    async fn merge_from_shards(
        &self,
        mut all_searches_res: Vec<Vec<Vec<ScoredPoint>>>,
        request: Arc<CoreSearchRequestBatch>,
        is_client_request: bool,
    ) -> CollectionResult<Vec<Vec<ScoredPoint>>> {
        let batch_size = request.searches.len();

        let collection_params = self.collection_config.read().await.params.clone();

        // Merge results from shards in order and deduplicate based on point ID
        let mut top_results: Vec<Vec<ScoredPoint>> = Vec::with_capacity(batch_size);
        let mut seen_ids = AHashSet::new();

        for (batch_index, request) in request.searches.iter().enumerate() {
            let order = if request.query.is_distance_scored() {
                collection_params
                    .get_distance(request.query.get_vector_name())?
                    .distance_order()
            } else {
                // Score comes from special handling of the distances in a way that it doesn't
                // directly represent distance anymore, so the order is always `LargeBetter`
                Order::LargeBetter
            };

            let results_from_shards = all_searches_res
                .iter_mut()
                .map(|res| res.get_mut(batch_index).map_or(Vec::new(), mem::take));

            let merged_iter = match order {
                Order::LargeBetter => Either::Left(results_from_shards.kmerge_by(|a, b| a > b)),
                Order::SmallBetter => Either::Right(results_from_shards.kmerge_by(|a, b| a < b)),
            }
            .filter(|point| {
                seen_ids.insert(search_dedup_point_id(
                    point.id,
                    request.source_id_dedup_block_size,
                ))
            });

            // Skip `offset` only for client requests
            // to avoid applying `offset` twice in distributed mode.
            let top_res = if is_client_request && request.offset > 0 {
                merged_iter
                    .skip(request.offset)
                    .take(request.limit)
                    .collect()
            } else {
                merged_iter.take(request.offset + request.limit).collect()
            };

            top_results.push(top_res);

            seen_ids.clear();
        }

        Ok(top_results)
    }

    pub fn post_process_if_slow_request<'a>(
        &self,
        duration: Duration,
        filters: impl IntoIterator<Item = Option<&'a Filter>>,
    ) {
        if duration > crate::problems::UnindexedField::slow_query_threshold() {
            let filters = filters.into_iter().flatten().cloned().collect_vec();

            let schema = self.payload_index_schema.read().schema.clone();

            issues::publish(SlowQueryEvent {
                collection_id: self.id.clone(),
                filters,
                schema,
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use segment::data_types::vectors::{MultiDenseVectorInternal, NamedQuery, VectorInternal};
    use segment::types::{
        Filter, PointIdType, SearchParams, ShardKey, WithPayloadInterface, WithVector,
    };
    use segment::vector_storage::query::RecoQuery;
    use shard::query::query_enum::QueryEnum;
    use shard::search::{CoreSearchRequest, CoreSearchRequestBatch};
    use sparse::common::sparse_vector::SparseVector;

    use super::*;

    fn core_search_request(query: QueryEnum) -> CoreSearchRequest {
        CoreSearchRequest {
            query,
            filter: None,
            params: None,
            hnsw_entry_points: None,
            hnsw_entry_points_by_shard: None,
            hnsw_ef_by_shard: None,
            source_id_dedup_block_size: None,
            limit: 10,
            offset: 0,
            with_payload: None,
            with_vector: None,
            score_threshold: None,
        }
    }

    fn default_dense_request() -> CoreSearchRequest {
        core_search_request(vec![1.0, 2.0, 3.0, 4.0].into())
    }

    #[test]
    fn orion_eligibility_accepts_only_matching_dense_nearest_queries() {
        let default_request = default_dense_request();
        assert_eq!(
            orion_eligible_dense_query(&default_request, ""),
            Some([1.0, 2.0, 3.0, 4.0].as_slice()),
        );
        assert!(orion_eligible_dense_query(&default_request, "other").is_none());

        let named_request = core_search_request(QueryEnum::Nearest(NamedQuery::new(
            VectorInternal::Dense(vec![4.0, 3.0, 2.0, 1.0]),
            "routing",
        )));
        assert_eq!(
            orion_eligible_dense_query(&named_request, "routing"),
            Some([4.0, 3.0, 2.0, 1.0].as_slice()),
        );
        assert!(orion_eligible_dense_query(&named_request, "").is_none());
    }

    #[test]
    fn orion_eligibility_bypasses_filter_exact_sparse_multivector_and_non_nearest() {
        let mut filtered = default_dense_request();
        filtered.filter = Some(Filter::new());
        assert!(orion_eligible_dense_query(&filtered, "").is_none());

        let mut exact = default_dense_request();
        exact.params = Some(SearchParams {
            exact: true,
            ..Default::default()
        });
        assert!(orion_eligible_dense_query(&exact, "").is_none());

        let sparse = core_search_request(QueryEnum::Nearest(NamedQuery::new(
            VectorInternal::Sparse(SparseVector {
                indices: vec![1, 7],
                values: vec![0.25, 0.75],
            }),
            "sparse",
        )));
        assert!(orion_eligible_dense_query(&sparse, "sparse").is_none());

        let multivector = core_search_request(QueryEnum::Nearest(NamedQuery::new(
            VectorInternal::MultiDense(MultiDenseVectorInternal::new(vec![1.0, 2.0, 3.0, 4.0], 2)),
            "multi",
        )));
        assert!(orion_eligible_dense_query(&multivector, "multi").is_none());

        let recommend = core_search_request(QueryEnum::RecommendBestScore(NamedQuery::new(
            RecoQuery::new(
                vec![VectorInternal::Dense(vec![1.0, 2.0, 3.0, 4.0])],
                Vec::new(),
            ),
            "routing",
        )));
        assert!(orion_eligible_dense_query(&recommend, "routing").is_none());
    }

    #[test]
    fn orion_eligibility_bypasses_all_legacy_experiment_hints() {
        let mut request = default_dense_request();
        request.hnsw_entry_points = Some(vec![PointIdType::from(11)]);
        assert!(orion_eligible_dense_query(&request, "").is_none());

        let mut request = default_dense_request();
        request.hnsw_entry_points_by_shard = Some(HashMap::from([(
            ShardKey::from("centroid_00"),
            vec![PointIdType::from(11)],
        )]));
        assert!(orion_eligible_dense_query(&request, "").is_none());

        let mut request = default_dense_request();
        request.hnsw_ef_by_shard = Some(HashMap::from([(ShardKey::from("centroid_00"), 24)]));
        assert!(orion_eligible_dense_query(&request, "").is_none());

        let mut request = default_dense_request();
        request.source_id_dedup_block_size = Some(1_000_001);
        assert!(orion_eligible_dense_query(&request, "").is_none());
    }

    #[test]
    fn orion_target_specialization_preserves_query_fields_and_applies_route_plan() {
        let mut request = core_search_request(QueryEnum::Nearest(NamedQuery::new(
            VectorInternal::Dense(vec![4.0, 3.0, 2.0, 1.0]),
            "routing",
        )));
        request.filter = Some(Filter::new());
        request.params = Some(SearchParams {
            hnsw_ef: Some(999),
            indexed_only: true,
            ..Default::default()
        });
        request.hnsw_entry_points = Some(vec![PointIdType::from(5)]);
        request.hnsw_entry_points_by_shard = Some(HashMap::from([(
            ShardKey::from("centroid_01"),
            vec![PointIdType::from(7)],
        )]));
        request.hnsw_ef_by_shard = Some(HashMap::from([(ShardKey::from("centroid_01"), 33)]));
        request.source_id_dedup_block_size = Some(1_000_001);
        request.limit = 10;
        request.offset = 7;
        request.with_payload = Some(WithPayloadInterface::Bool(true));
        request.with_vector = Some(WithVector::Bool(true));
        request.score_threshold = Some(0.42);

        let target = OrionShardTarget {
            shard_id: 2,
            entry_points: vec![
                PointIdType::from(31),
                PointIdType::from(29),
                PointIdType::from(11),
            ],
            ef: 76,
        };
        let specialized = specialize_core_search_for_orion_target(&request, &target);

        assert_eq!(specialized.query, request.query);
        assert_eq!(specialized.filter, request.filter);
        assert_eq!(specialized.with_payload, request.with_payload);
        assert_eq!(specialized.with_vector, request.with_vector);
        assert_eq!(specialized.score_threshold, request.score_threshold);
        assert_eq!(specialized.limit, 17);
        assert_eq!(specialized.offset, 0);
        assert_eq!(specialized.hnsw_entry_points, Some(target.entry_points));
        assert!(specialized.hnsw_entry_points_by_shard.is_none());
        assert!(specialized.hnsw_ef_by_shard.is_none());
        assert!(specialized.source_id_dedup_block_size.is_none());
        assert_eq!(
            specialized.params,
            Some(SearchParams {
                hnsw_ef: Some(76),
                indexed_only: true,
                ..Default::default()
            }),
        );
    }

    #[test]
    fn simple_kmeans_target_specialization_sets_ef_without_multiep() {
        let mut request = default_dense_request();
        request.params = Some(SearchParams {
            hnsw_ef: Some(999),
            indexed_only: true,
            ..Default::default()
        });
        request.limit = 10;
        request.offset = 7;
        request.hnsw_entry_points = Some(vec![PointIdType::from(5)]);

        let target = SimpleKmeansShardTarget {
            shard_id: 2,
            ef: 80,
        };
        let specialized = specialize_core_search_for_simple_kmeans_target(&request, &target);
        assert_eq!(specialized.query, request.query);
        assert_eq!(specialized.limit, 17);
        assert_eq!(specialized.offset, 0);
        assert!(specialized.hnsw_entry_points.is_none());
        assert!(specialized.hnsw_entry_points_by_shard.is_none());
        assert!(specialized.hnsw_ef_by_shard.is_none());
        assert!(specialized.source_id_dedup_block_size.is_none());
        assert_eq!(
            specialized.params,
            Some(SearchParams {
                hnsw_ef: Some(80),
                indexed_only: true,
                ..Default::default()
            })
        );
    }

    #[test]
    fn orion_routing_time_is_charged_to_the_search_timeout() {
        let timeout = Duration::from_millis(10);
        assert_eq!(
            remaining_search_timeout(
                Some(timeout),
                Duration::from_millis(4),
                "Orion upper routing",
            )
            .unwrap(),
            Some(Duration::from_millis(6)),
        );
        assert!(
            remaining_search_timeout(
                Some(timeout),
                Duration::from_millis(10),
                "Orion upper routing",
            )
            .unwrap_err()
            .to_string()
            .contains("timed out")
        );
        assert_eq!(
            remaining_search_timeout(None, Duration::from_secs(1), "search").unwrap(),
            None,
        );
    }

    #[test]
    fn specialize_search_batch_applies_per_shard_entry_points_and_ef() {
        let request = CoreSearchRequest {
            query: vec![1.0, 2.0, 3.0, 4.0].into(),
            filter: None,
            params: Some(SearchParams {
                hnsw_ef: Some(100),
                ..Default::default()
            }),
            hnsw_entry_points: None,
            hnsw_entry_points_by_shard: Some(HashMap::from([
                (ShardKey::from("centroid_00"), vec![PointIdType::from(11)]),
                (
                    ShardKey::from("centroid_01"),
                    vec![PointIdType::from(29), PointIdType::from(31)],
                ),
            ])),
            hnsw_ef_by_shard: Some(HashMap::from([
                (ShardKey::from("centroid_00"), 24),
                (ShardKey::from("centroid_01"), 28),
            ])),
            source_id_dedup_block_size: None,
            limit: 10,
            offset: 0,
            with_payload: None,
            with_vector: None,
            score_threshold: None,
        };
        let batch = CoreSearchRequestBatch {
            searches: vec![request],
        };

        let specialized =
            specialize_search_batch_for_shard(&batch, Some(&ShardKey::from("centroid_01")));

        assert_eq!(
            specialized.searches[0].hnsw_entry_points,
            Some(vec![PointIdType::from(29), PointIdType::from(31)])
        );
        assert_eq!(
            specialized.searches[0]
                .params
                .as_ref()
                .and_then(|params| params.hnsw_ef),
            Some(28)
        );
        assert!(specialized.searches[0].hnsw_entry_points_by_shard.is_none());
        assert!(specialized.searches[0].hnsw_ef_by_shard.is_none());
    }

    #[test]
    fn specialize_search_batch_clears_per_shard_maps_when_key_is_absent() {
        let request = CoreSearchRequest {
            query: vec![1.0, 2.0, 3.0, 4.0].into(),
            filter: None,
            params: None,
            hnsw_entry_points: None,
            hnsw_entry_points_by_shard: Some(HashMap::from([(
                ShardKey::from("centroid_00"),
                vec![PointIdType::from(11)],
            )])),
            hnsw_ef_by_shard: Some(HashMap::from([(ShardKey::from("centroid_00"), 24)])),
            source_id_dedup_block_size: None,
            limit: 10,
            offset: 0,
            with_payload: None,
            with_vector: None,
            score_threshold: None,
        };
        let batch = CoreSearchRequestBatch {
            searches: vec![request],
        };

        let specialized =
            specialize_search_batch_for_shard(&batch, Some(&ShardKey::from("centroid_01")));

        assert_eq!(specialized.searches[0].hnsw_entry_points, None);
        assert_eq!(specialized.searches[0].params, None);
        assert!(specialized.searches[0].hnsw_entry_points_by_shard.is_none());
        assert!(specialized.searches[0].hnsw_ef_by_shard.is_none());
    }

    #[test]
    fn search_dedup_point_id_decodes_copy_id_when_block_size_is_set() {
        assert_eq!(
            search_dedup_point_id(ExtendedPointId::NumId(2_000_045), Some(1_000_001)),
            ExtendedPointId::NumId(42)
        );
        assert_eq!(
            search_dedup_point_id(ExtendedPointId::NumId(2_000_045), None),
            ExtendedPointId::NumId(2_000_045)
        );
    }

    #[test]
    fn routing_batches_are_bounded_by_the_search_runtime_thread_count() {
        assert_eq!(routing_chunk_size(1, 8), 1);
        assert_eq!(routing_chunk_size(8, 8), 1);
        assert_eq!(routing_chunk_size(9, 8), 2);
        assert_eq!(routing_chunk_size(200, 8), 25);
        assert_eq!(routing_chunk_size(200, 0), 200);
    }

    #[test]
    fn numeric_peer_premerge_requires_rf1_default_consistency_and_one_remote_replica() {
        assert!(numeric_peer_premerge_request_is_safe(1, None, true));
        assert!(!numeric_peer_premerge_request_is_safe(2, None, true));
        assert!(!numeric_peer_premerge_request_is_safe(
            1,
            Some(ReadConsistency::Factor(1)),
            true,
        ));
        assert!(!numeric_peer_premerge_request_is_safe(1, None, false));

        assert_eq!(
            numeric_peer_premerge_replica_peer(1, 1, &[2], false, true),
            Some(2),
        );
        assert_eq!(
            numeric_peer_premerge_replica_peer(1, 2, &[2], false, true),
            None,
        );
        assert_eq!(
            numeric_peer_premerge_replica_peer(1, 1, &[2, 3], false, true),
            None,
        );
        assert_eq!(
            numeric_peer_premerge_replica_peer(1, 1, &[1], true, false),
            None,
        );
        assert_eq!(
            numeric_peer_premerge_replica_peer(1, 1, &[2], false, false),
            None,
        );
    }

    #[test]
    fn numeric_peer_premerge_wire_entry_preserves_ordered_multiep_dynamic_ef_and_top_window() {
        let mut original = default_dense_request();
        original.limit = 10;
        original.offset = 7;
        original.source_id_dedup_block_size = Some(1_000_001);
        let target = OrionShardTarget {
            shard_id: 9,
            entry_points: vec![
                PointIdType::from(31),
                PointIdType::from(29),
                PointIdType::from(11),
            ],
            ef: 76,
        };
        let specialized = specialize_core_search_for_orion_target(&original, &target);

        let entry = peer_premerge_entry("orion", 3, target.shard_id, &specialized, &original);
        assert_eq!(entry.query_index, 3);
        assert_eq!(entry.shard_id, target.shard_id);
        assert_eq!(entry.final_limit, 10);
        assert_eq!(entry.final_offset, Some(7));
        assert_eq!(entry.source_id_dedup_block_size, Some(1_000_001));

        let roundtrip = CoreSearchRequest::try_from(entry.search_points.unwrap()).unwrap();
        assert_eq!(roundtrip.limit, 17);
        assert_eq!(roundtrip.offset, 0);
        assert_eq!(roundtrip.hnsw_entry_points, Some(target.entry_points));
        assert_eq!(roundtrip.params.and_then(|params| params.hnsw_ef), Some(76),);
    }

    #[test]
    fn peer_premerge_query_slots_preserve_non_monotonic_original_indices() {
        let mut original_indices = Vec::new();
        let mut payload_required = Vec::new();
        let mut no_payload = default_dense_request();
        no_payload.with_payload = Some(WithPayloadInterface::Bool(false));
        let mut with_payload = default_dense_request();
        with_payload.with_payload = Some(WithPayloadInterface::Bool(true));

        assert_eq!(
            peer_local_query_index(&mut original_indices, &mut payload_required, 5, &no_payload),
            0,
        );
        assert_eq!(
            peer_local_query_index(
                &mut original_indices,
                &mut payload_required,
                1,
                &with_payload,
            ),
            1,
        );
        assert_eq!(
            peer_local_query_index(
                &mut original_indices,
                &mut payload_required,
                5,
                &with_payload,
            ),
            0,
        );
        assert_eq!(
            peer_local_query_index(&mut original_indices, &mut payload_required, 9, &no_payload),
            2,
        );
        assert_eq!(original_indices, vec![5, 1, 9]);
        assert_eq!(payload_required, vec![false, true, false]);
    }
}
