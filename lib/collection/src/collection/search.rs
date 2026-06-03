use std::collections::BTreeMap;
use std::mem;
use std::sync::Arc;
use std::time::Duration;

use ahash::{AHashMap, AHashSet};
use api::grpc::qdrant::{CoreSearchBatchByShardInternal, CoreSearchByShardEntry};
use common::counter::hardware_accumulator::HwMeasurementAcc;
use futures::{TryFutureExt, future};
use itertools::{Either, Itertools};
use segment::types::{
    ExtendedPointId, Filter, Order, ScoredPoint, ShardKey, WithPayloadInterface, WithVector,
};
use shard::retrieve::record_internal::RecordInternal;
use shard::search::CoreSearchRequestBatch;
use tokio::time::Instant;

use super::Collection;
use crate::events::SlowQueryEvent;
use crate::operations::consistency_params::ReadConsistency;
use crate::operations::shard_selector_internal::ShardSelectorInternal;
use crate::operations::types::*;
use crate::shards::remote_shard::{CollectionCoreSearchRequest, RemoteShard};
use crate::shards::shard::PeerId;

fn search_batch_requires_shard_specialization(request: &CoreSearchRequestBatch) -> bool {
    request.searches.iter().any(|search| {
        search.hnsw_entry_points_by_shard.is_some() || search.hnsw_ef_by_shard.is_some()
    })
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
        struct PeerShardMajorGroup {
            remote: RemoteShard,
            original_indices: Vec<usize>,
            is_payload_required_by_query: Vec<bool>,
            entries: Vec<CoreSearchByShardEntry>,
        }

        impl PeerShardMajorGroup {
            fn local_query_index(
                &mut self,
                original_index: usize,
                request: &CoreSearchRequest,
            ) -> usize {
                if let Some(query_index) = self
                    .original_indices
                    .iter()
                    .position(|known_index| *known_index == original_index)
                {
                    return query_index;
                }

                let query_index = self.original_indices.len();
                self.original_indices.push(original_index);
                self.is_payload_required_by_query.push(
                    request
                        .with_payload
                        .as_ref()
                        .is_some_and(|with_payload| with_payload.is_required()),
                );
                query_index
            }
        }

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
                        .or_insert_with(|| PeerShardMajorGroup {
                            remote,
                            original_indices: Vec::new(),
                            is_payload_required_by_query: Vec::new(),
                            entries: Vec::new(),
                        });
                    let local_query_index = group.local_query_index(original_index, request);
                    let specialized = specialize_core_search_for_shard_major(request, shard_key);
                    let search_points =
                        CollectionCoreSearchRequest((self.id.clone(), &specialized)).into();

                    group.entries.push(CoreSearchByShardEntry {
                        query_index: local_query_index as u64,
                        shard_id: replica_set.shard_id,
                        search_points: Some(search_points),
                        final_limit: request.limit as u64,
                        final_offset: Some(request.offset as u64),
                        source_id_dedup_block_size: request.source_id_dedup_block_size,
                    });
                }
            }
        }

        if peer_groups.is_empty() {
            return Ok(None);
        }

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
            let mut peer_rows = vec![Vec::new(); original_requests.len()];
            for (original_index, row) in original_indices.into_iter().zip(rows) {
                peer_rows[original_index] = row;
            }
            all_peer_rows.push(peer_rows);
        }

        let request = Arc::new(CoreSearchRequestBatch {
            searches: original_requests,
        });
        self.merge_from_shards(all_peer_rows, request, true)
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
        let request = Arc::new(request);
        let requires_shard_specialization = search_batch_requires_shard_specialization(&request);

        let instant = Instant::now();

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

    use segment::types::{PointIdType, SearchParams, ShardKey};
    use shard::search::{CoreSearchRequest, CoreSearchRequestBatch};

    use super::*;

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
}
