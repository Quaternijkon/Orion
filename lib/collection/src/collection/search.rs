use std::mem;
use std::sync::Arc;
use std::time::Duration;

use ahash::{AHashMap, AHashSet};
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
            specialized.searches[0].params.as_ref().and_then(|params| params.hnsw_ef),
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
