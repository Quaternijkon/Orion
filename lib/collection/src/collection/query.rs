use std::future::Future;
use std::sync::Arc;
use std::time::Duration;

use common::counter::hardware_accumulator::HwMeasurementAcc;
use futures::{TryFutureExt, future};
use itertools::{Either, Itertools};
use rand::RngExt;
use segment::common::reciprocal_rank_fusion::rrf_scoring;
use segment::common::score_fusion::{ScoreFusion, score_fusion};
use segment::data_types::vectors::{Named, VectorInternal, VectorStructInternal};
use segment::types::{Order, ScoredPoint, WithPayloadInterface, WithVector};
use segment::utils::scored_point_ties::ScoredPointTies;
use tokio::time::Instant;

use super::Collection;
use crate::collection::mmr::mmr_from_points_with_vector;
use crate::collection_manager::probabilistic_search_sampling::find_search_sampling_over_point_distribution;
use crate::common::batching::batch_requests;
use crate::common::fetch_vectors::{
    build_vector_resolver_queries, resolve_referenced_vectors_batch,
};
use crate::common::retrieve_request_trait::RetrieveRequest;
use crate::common::transpose_iterator::transposed_iter;
use crate::config::AutoShardPolicy;
use crate::operations::consistency_params::ReadConsistency;
use crate::operations::shard_selector_internal::ShardSelectorInternal;
use crate::operations::types::{CollectionError, CollectionResult};
use crate::operations::universal_query::collection_query::CollectionQueryRequest;
use crate::operations::universal_query::shard_query::{
    self, FusionInternal, MmrInternal, ScoringQuery, ShardQueryRequest, ShardQueryResponse,
};
use shard::query::query_enum::QueryEnum;
use shard::search::{CoreSearchRequest, CoreSearchRequestBatch};

/// A factor which determines if we need to use the 2-step search or not.
/// Should be adjusted based on usage statistics.
pub(super) const PAYLOAD_TRANSFERS_FACTOR_THRESHOLD: usize = 10;

struct IntermediateQueryInfo<'a> {
    scoring_query: Option<&'a ScoringQuery>,
    /// Limit + offset
    take: usize,
}

fn orion_core_search_batch_from_query_requests(
    requests: &[ShardQueryRequest],
    routing_vector_name: &str,
) -> Option<CoreSearchRequestBatch> {
    let searches = requests
        .iter()
        .map(|request| {
            if !request.prefetches.is_empty()
                || request.filter.is_some()
                || request.params.as_ref().is_some_and(|params| params.exact)
            {
                return None;
            }

            let Some(ScoringQuery::Vector(QueryEnum::Nearest(named_query))) = &request.query else {
                return None;
            };
            if named_query.get_name() != routing_vector_name
                || !matches!(&named_query.query, VectorInternal::Dense(_))
            {
                return None;
            }

            Some(CoreSearchRequest {
                query: QueryEnum::Nearest(named_query.clone()),
                filter: None,
                params: request.params,
                hnsw_entry_points: None,
                hnsw_entry_points_by_shard: None,
                hnsw_ef_by_shard: None,
                source_id_dedup_block_size: None,
                limit: request.limit,
                offset: request.offset,
                with_payload: Some(request.with_payload.clone()),
                with_vector: Some(request.with_vector.clone()),
                score_threshold: request.score_threshold.map(|score| score.into_inner()),
            })
        })
        .collect::<Option<Vec<_>>>()?;

    Some(CoreSearchRequestBatch { searches })
}

fn auto_shard_policy_allows_query_undersampling(
    auto_shard_policy: Option<&AutoShardPolicy>,
) -> bool {
    AutoShardPolicy::canonical_ref(auto_shard_policy).is_none()
}

fn merge_ordered_shard_results(
    shards_results: Vec<Vec<ScoredPoint>>,
    order: Order,
    take: usize,
    deduplicate_by_external_id: bool,
) -> Vec<ScoredPoint> {
    let merged = match order {
        Order::LargeBetter => Either::Left(
            shards_results
                .into_iter()
                .kmerge_by(|a, b| ScoredPointTies(a) > ScoredPointTies(b)),
        ),
        Order::SmallBetter => Either::Right(
            shards_results
                .into_iter()
                .kmerge_by(|a, b| ScoredPointTies(a) < ScoredPointTies(b)),
        ),
    };

    if deduplicate_by_external_id {
        merged.unique_by(|point| point.id).take(take).collect()
    } else {
        merged.dedup().take(take).collect()
    }
}

impl Collection {
    /// query is a special case of query_batch with a single batch
    pub async fn query(
        &self,
        request: ShardQueryRequest,
        read_consistency: Option<ReadConsistency>,
        shard_selection: ShardSelectorInternal,
        timeout: Option<Duration>,
        hw_measurement_acc: HwMeasurementAcc,
    ) -> CollectionResult<Vec<ScoredPoint>> {
        if request.limit == 0 {
            return Ok(vec![]);
        }
        let results = self
            .do_query_batch(
                vec![request],
                read_consistency,
                shard_selection,
                timeout,
                hw_measurement_acc,
            )
            .await?;
        Ok(results.into_iter().next().unwrap())
    }

    /// If the query limit above this value, it will be a subject to undersampling.
    const SHARD_QUERY_SUBSAMPLING_LIMIT: usize = 128;

    /// Give some more ensurance for undersampling,
    /// retrieve more points to prevent undersampling errors.
    /// Errors are still possible, but rare enough to be acceptable compared to
    /// errors introduced by vector index.
    const MORE_ENSURANCE_FACTOR: f64 = 1.2;

    /// Creates a copy of requests in case it is possible to apply limit modification
    /// Returns unchanged requests if limit modification is not applicable.
    ///
    /// If there are many independent shards, and we need a very high limit, we can do an optimization.
    /// Instead of querying all shards with the same limit, we can query each shard with a smaller limit
    /// and then merge the results. Since shards are independent and data is randomly distributed, we can
    /// apply probability estimation to make sure we query enough points to get the desired number of results.
    ///
    /// Same optimization we already apply on segment level, but here it seems to be even more reliable
    /// because auto-sharding guarantee random and independent distribution of data.
    ///
    /// Unlike segments, however, the cost of re-requesting the data is much higher for shards.
    /// So we "accept" the risk of not getting enough results.
    fn modify_shard_query_for_undersampling_limits(
        batch_request: Arc<Vec<ShardQueryRequest>>,
        num_shards: usize,
        is_auto_sharding: bool,
    ) -> Arc<Vec<ShardQueryRequest>> {
        if num_shards <= 1 {
            return batch_request;
        }

        // Check this parameter inside the function
        // to ensure it is not omitted in the future.
        if !is_auto_sharding {
            return batch_request;
        }

        let max_limit = batch_request
            .iter()
            .map(|req| req.limit + req.offset)
            .max()
            .unwrap_or(0);

        if max_limit < Self::SHARD_QUERY_SUBSAMPLING_LIMIT {
            return batch_request;
        }

        let mut new_requests = Vec::with_capacity(batch_request.len());

        for request in batch_request.iter() {
            let mut new_request = request.clone();
            let request_limit = new_request.limit + new_request.offset;

            let is_exact = request.params.as_ref().map(|p| p.exact).unwrap_or(false);

            if is_exact || request_limit < Self::SHARD_QUERY_SUBSAMPLING_LIMIT {
                new_requests.push(new_request);
                continue;
            }

            // Example: 1000 limit, 10 shards
            // 1.0 / 10 * 1.2 = 0.12
            // lambda = 0.12 * 1000 = 120
            // Which is equal to 171 limit per shard
            let undersample_limit = find_search_sampling_over_point_distribution(
                request_limit as f64,
                1. / num_shards as f64 * Self::MORE_ENSURANCE_FACTOR,
            );

            new_request.limit = std::cmp::min(undersample_limit, request_limit);
            new_request.offset = 0; // Offset is handled on the collection level
            new_requests.push(new_request);
        }

        Arc::new(new_requests)
    }

    /// Returns a shape of [shard_id, batch_id, intermediate_response, points]
    async fn batch_query_shards_concurrently(
        &self,
        batch_request: Arc<Vec<ShardQueryRequest>>,
        read_consistency: Option<ReadConsistency>,
        shard_selection: &ShardSelectorInternal,
        timeout: Option<Duration>,
        hw_measurement_acc: HwMeasurementAcc,
    ) -> CollectionResult<Vec<Vec<ShardQueryResponse>>> {
        // query all shards concurrently
        let shard_holder = self.shards_holder.read().await;
        let target_shards = shard_holder.select_shards(shard_selection)?;

        let num_unique_shard_keys = target_shards
            .iter()
            .map(|(_, shard_key)| shard_key)
            .unique()
            .count();
        // Auto-sharding happens when we are only querying shards with _the_same_ shard key.
        // It either might be when we are querying a specific shard key
        // OR when we are querying all shards with no shard keys specified.
        let is_auto_sharding = num_unique_shard_keys == 1;
        // Collection-level undersampling assumes point-ID hash sharding produces independent,
        // random shard samples. Orion's vector-local placement does not satisfy that assumption,
        // including when an ineligible Orion query safely falls back to all shards.
        let is_hash_all_policy = auto_shard_policy_allows_query_undersampling(
            self.collection_config
                .read()
                .await
                .auto_shard_policy
                .as_ref(),
        );

        let batch_request = Self::modify_shard_query_for_undersampling_limits(
            batch_request,
            target_shards.len(),
            is_auto_sharding && is_hash_all_policy,
        );

        let all_searches = target_shards.iter().map(|(shard, shard_key)| {
            let shard_key = shard_key.cloned();
            let request_clone = Arc::clone(&batch_request);
            shard
                .query_batch(
                    request_clone,
                    read_consistency,
                    shard_selection.is_shard_id(),
                    timeout,
                    hw_measurement_acc.clone(),
                )
                .and_then(move |mut shard_responses| async move {
                    if shard_key.is_none() {
                        return Ok(shard_responses);
                    }
                    shard_responses
                        .iter_mut()
                        .flatten()
                        .flatten()
                        .for_each(|point| point.shard_key.clone_from(&shard_key));

                    Ok(shard_responses)
                })
        });
        future::try_join_all(all_searches).await
    }

    /// This function is used to query the collection. It will return a list of scored points.
    async fn do_query_batch(
        &self,
        requests_batch: Vec<ShardQueryRequest>,
        read_consistency: Option<ReadConsistency>,
        shard_selection: ShardSelectorInternal,
        timeout: Option<Duration>,
        hw_measurement_acc: HwMeasurementAcc,
    ) -> CollectionResult<Vec<Vec<ScoredPoint>>> {
        let start = Instant::now();

        // shortcuts batch if all requests with limit=0
        if requests_batch.iter().all(|s| s.limit == 0) {
            return Ok(vec![]);
        }

        let is_payload_required = requests_batch.iter().all(|s| s.with_payload.is_required());
        let with_vectors = requests_batch.iter().all(|s| s.with_vector.is_enabled());

        let metadata_required = is_payload_required || with_vectors;

        let sum_limits: usize = requests_batch
            .iter()
            .fold(0usize, |acc, s| acc.saturating_add(s.limit));
        let sum_offsets: usize = requests_batch
            .iter()
            .fold(0usize, |acc, s| acc.saturating_add(s.offset));

        // Number of records we need to retrieve to fill the search result.
        let require_transfers = self
            .shards_holder
            .read()
            .await
            .len()
            .saturating_mul(sum_limits.saturating_add(sum_offsets));
        // Actually used number of records.
        let used_transfers = sum_limits;

        let is_required_transfer_large_enough =
            require_transfers > used_transfers.saturating_mul(PAYLOAD_TRANSFERS_FACTOR_THRESHOLD);

        if metadata_required && is_required_transfer_large_enough {
            // If there is a significant offset, we need to retrieve the whole result
            // set without payload first and then retrieve the payload.
            // It is required to do this because the payload might be too large to send over the
            // network.
            let mut without_payload_requests = Vec::with_capacity(requests_batch.len());
            for query in &requests_batch {
                let mut without_payload_request = query.clone();
                without_payload_request.with_payload = WithPayloadInterface::Bool(false);
                without_payload_request.with_vector = WithVector::Bool(false);
                without_payload_requests.push(without_payload_request);
            }
            let without_payload_batch = without_payload_requests;
            let without_payload_results = self
                .do_query_batch_impl(
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
                .zip(requests_batch.into_iter())
                .map(|(without_payload_result, req)| {
                    self.fill_search_result_with_payload(
                        without_payload_result,
                        Some(req.with_payload),
                        req.with_vector,
                        read_consistency,
                        &shard_selection,
                        timeout,
                        hw_measurement_acc.clone(),
                    )
                });
            future::try_join_all(filled_results).await
        } else {
            self.do_query_batch_impl(
                requests_batch,
                read_consistency,
                &shard_selection,
                timeout,
                hw_measurement_acc.clone(),
            )
            .await
        }
    }

    /// This function is used to query the collection. It will return a list of scored points.
    async fn do_query_batch_impl(
        &self,
        requests_batch: Vec<ShardQueryRequest>,
        read_consistency: Option<ReadConsistency>,
        shard_selection: &ShardSelectorInternal,
        timeout: Option<Duration>,
        hw_measurement_acc: HwMeasurementAcc,
    ) -> CollectionResult<Vec<Vec<ScoredPoint>>> {
        let instant = Instant::now();
        self.ensure_static_router_available(shard_selection).await?;

        if matches!(shard_selection, ShardSelectorInternal::All)
            && let Some(routing_vector_name) = self
                .orion_router
                .as_ref()
                .map(|router| router.vector_name())
                .or_else(|| {
                    self.simple_kmeans_router
                        .as_ref()
                        .map(|router| router.vector_name())
                })
            && let Some(core_batch) =
                orion_core_search_batch_from_query_requests(&requests_batch, routing_vector_name)
        {
            return self
                .core_search_batch(
                    core_batch,
                    read_consistency,
                    shard_selection.clone(),
                    timeout,
                    hw_measurement_acc,
                )
                .await;
        }

        let requests_batch = Arc::new(requests_batch);

        let all_shards_results = self
            .batch_query_shards_concurrently(
                requests_batch.clone(),
                read_consistency,
                shard_selection,
                timeout,
                hw_measurement_acc.clone(),
            )
            .await?;

        let results_f = transposed_iter(all_shards_results)
            .zip(requests_batch.iter())
            .map(|(shards_results, request)| async {
                // shards_results shape: [num_shards, num_intermediate_results, num_points]
                // merged_intermediates shape: [num_intermediate_results, num_points]
                let merged_intermediates = self
                    .merge_intermediate_results_from_shards(request, shards_results)
                    .await?;

                let result = self
                    .intermediates_to_final_list(
                        merged_intermediates,
                        request,
                        timeout.map(|timeout| timeout.saturating_sub(instant.elapsed())),
                        hw_measurement_acc.clone(),
                    )
                    .await?;

                let filter_refs = request.filter_refs();
                self.post_process_if_slow_request(instant.elapsed(), filter_refs);

                Ok::<_, CollectionError>(result)
            });
        let results = future::try_join_all(results_f).await?;

        Ok(results)
    }

    /// Resolves the final list of scored points from the intermediate results.
    ///
    /// Finalizes queries like fusion and mmr after collecting from all shards.
    /// For other kind of queries it just passes the results through.
    ///
    /// Handles offset and limit.
    async fn intermediates_to_final_list(
        &self,
        mut intermediates: Vec<Vec<ScoredPoint>>,
        request: &ShardQueryRequest,
        timeout: Option<Duration>,
        hw_measurement_acc: HwMeasurementAcc,
    ) -> CollectionResult<Vec<ScoredPoint>> {
        let ShardQueryRequest {
            prefetches: _,
            query,
            filter: _,
            score_threshold,
            limit,
            offset,
            params: _,
            with_vector,
            with_payload: _,
        } = request;

        let result = match query.as_ref() {
            Some(ScoringQuery::Fusion(fusion)) => {
                // If the root query is a Fusion, the returned results correspond to each the prefetches.
                let mut fused = match fusion {
                    FusionInternal::Rrf { k, weights } => {
                        let weights_slice = weights
                            .as_ref()
                            .map(|w| w.iter().map(|f| f.into_inner()).collect::<Vec<_>>());
                        rrf_scoring(intermediates, *k, weights_slice.as_deref())?
                    }
                    FusionInternal::Dbsf => score_fusion(intermediates, ScoreFusion::dbsf()),
                };
                if let Some(&score_threshold) = score_threshold.as_ref() {
                    fused = fused
                        .into_iter()
                        .take_while(|point| point.score >= score_threshold.0)
                        .collect();
                }
                fused
            }
            Some(ScoringQuery::Mmr(mmr)) => {
                let points_with_vector = intermediates.into_iter().flatten();

                let collection_params = self.collection_config.read().await.params.clone();
                let search_runtime_handle = &self.search_runtime;
                let timeout = timeout.unwrap_or(self.shared_storage_config.search_timeout);

                let mut mmr_result = mmr_from_points_with_vector(
                    &collection_params,
                    points_with_vector,
                    mmr.clone(),
                    *limit,
                    search_runtime_handle,
                    timeout,
                    hw_measurement_acc,
                )
                .await?;

                // strip mmr vector if necessary
                match with_vector {
                    WithVector::Bool(false) => mmr_result.iter_mut().for_each(|p| {
                        p.vector.take();
                    }),
                    WithVector::Bool(true) => {}
                    WithVector::Selector(items) => {
                        if !items.contains(&mmr.using) {
                            mmr_result.iter_mut().for_each(|p| {
                                VectorStructInternal::take_opt(&mut p.vector, &mmr.using);
                            })
                        }
                    }
                };
                mmr_result
            }
            None
            | Some(ScoringQuery::Vector(_))
            | Some(ScoringQuery::OrderBy(_))
            | Some(ScoringQuery::Formula(_))
            | Some(ScoringQuery::Sample(_)) => {
                // Otherwise, it will be a list with a single list of scored points.
                debug_assert_eq!(intermediates.len(), 1);
                intermediates.pop().ok_or_else(|| {
                    CollectionError::service_error(
                        "Query response was expected to have one list of results.",
                    )
                })?
            }
        };

        let result: Vec<ScoredPoint> = result.into_iter().skip(*offset).take(*limit).collect();

        Ok(result)
    }

    /// To be called on the user-responding instance. Resolves ids into vectors, and merges the results from local and remote shards.
    ///
    /// This function is used to query the collection. It will return a list of scored points.
    pub async fn query_batch<F, Fut>(
        &self,
        requests_batch: Vec<(CollectionQueryRequest, ShardSelectorInternal)>,
        collection_by_name: F,
        read_consistency: Option<ReadConsistency>,
        timeout: Option<Duration>,
        hw_measurement_acc: HwMeasurementAcc,
    ) -> CollectionResult<Vec<Vec<ScoredPoint>>>
    where
        F: Fn(String) -> Fut,
        Fut: Future<Output = Option<Arc<Collection>>>,
    {
        let start = Instant::now();

        // Lift nested prefetches to root queries for vector resolution
        let resolver_requests = build_vector_resolver_queries(&requests_batch);

        // Build referenced vectors
        let ids_to_vectors = resolve_referenced_vectors_batch(
            &resolver_requests,
            self,
            collection_by_name,
            read_consistency,
            timeout,
            hw_measurement_acc.clone(),
        )
        .await?;

        // update timeout
        let timeout = timeout.map(|timeout| timeout.saturating_sub(start.elapsed()));

        // Check we actually fetched all referenced vectors from the resolver requests
        for (resolver_req, _) in &resolver_requests {
            for point_id in resolver_req.get_referenced_point_ids() {
                let lookup_collection = resolver_req.get_lookup_collection();
                if ids_to_vectors.get(lookup_collection, point_id).is_none() {
                    return Err(CollectionError::PointNotFound {
                        missed_point_id: point_id,
                    });
                }
            }
        }

        let futures = batch_requests::<
            (CollectionQueryRequest, ShardSelectorInternal),
            ShardSelectorInternal,
            Vec<ShardQueryRequest>,
            Vec<_>,
        >(
            requests_batch,
            |(_req, shard)| shard,
            |(req, _), acc| {
                req.try_into_shard_request(&self.id, &ids_to_vectors)
                    .map(|shard_req| {
                        acc.push(shard_req);
                    })
            },
            |shard_selection, shard_requests, futures| {
                if shard_requests.is_empty() {
                    return Ok(());
                }

                futures.push(self.do_query_batch(
                    shard_requests,
                    read_consistency,
                    shard_selection,
                    timeout,
                    hw_measurement_acc.clone(),
                ));

                Ok(())
            },
        )?;

        let results = future::try_join_all(futures)
            .await?
            .into_iter()
            .flatten()
            .collect();

        Ok(results)
    }

    /// To be called on the remote instance. Only used for the internal service.
    ///
    /// If the root query is a Fusion, the returned results correspond to each the prefetches.
    /// Otherwise, it will be a list with a single list of scored points.
    pub async fn query_batch_internal(
        &self,
        requests: Vec<ShardQueryRequest>,
        shard_selection: &ShardSelectorInternal,
        timeout: Option<Duration>,
        hw_measurement_acc: HwMeasurementAcc,
    ) -> CollectionResult<Vec<ShardQueryResponse>> {
        let requests_arc = Arc::new(requests);

        // Results from all shards
        // Shape: [num_shards, batch_size, num_intermediate_results, num_points]
        let all_shards_results = self
            .batch_query_shards_concurrently(
                Arc::clone(&requests_arc),
                None,
                shard_selection,
                timeout,
                hw_measurement_acc,
            )
            .await?;

        let merged_f = transposed_iter(all_shards_results)
            .zip(requests_arc.iter())
            .map(|(shards_results, request)| async {
                // shards_results shape: [num_shards, num_intermediate_results, num_points]
                self.merge_intermediate_results_from_shards(request, shards_results)
                    .await
            });
        let merged = futures::future::try_join_all(merged_f).await?;
        Ok(merged)
    }

    /// Find best result across last results of all shards.
    /// Presence of the worst result in final result means that there could be other results
    /// of that shard that could be included in the final result.
    /// Used to check undersampling.
    fn get_best_last_shard_result(
        shard_results: &[Vec<ScoredPoint>],
        order: Order,
    ) -> Option<ScoredPoint> {
        shard_results
            .iter()
            .filter_map(|shard_result| shard_result.last().cloned())
            .max_by(|a, b| match order {
                Order::LargeBetter => ScoredPointTies(a).cmp(&ScoredPointTies(b)),
                Order::SmallBetter => ScoredPointTies(a).cmp(&ScoredPointTies(b)).reverse(),
            })
    }

    /// Check that worst result of the shard in not present in the final result.
    fn check_undersampling(
        &self,
        worst_merged_point: &ScoredPoint,
        best_last_result: &ScoredPoint,
        order: Order,
    ) {
        // Merged point should be better than the best last result.
        let is_properly_sampled = match order {
            Order::LargeBetter => {
                ScoredPointTies(worst_merged_point) > ScoredPointTies(best_last_result)
            }
            Order::SmallBetter => {
                ScoredPointTies(worst_merged_point) < ScoredPointTies(best_last_result)
            }
        };
        if !is_properly_sampled {
            log::debug!(
                "Undersampling detected. Collection: {}, Best last shard score: {}, Worst merged score: {}",
                self.id,
                best_last_result.score,
                worst_merged_point.score
            );
        }
    }

    /// Merges the results in each shard for each intermediate query.
    /// ```text
    /// [ [shard1_result1, shard1_result2],
    ///          ↓               ↓
    ///   [shard2_result1, shard2_result2] ]
    ///
    /// = [merged_result1, merged_result2]
    /// ```
    async fn merge_intermediate_results_from_shards(
        &self,
        request: &ShardQueryRequest,
        all_shards_results: Vec<ShardQueryResponse>,
    ) -> CollectionResult<ShardQueryResponse> {
        let query_infos = intermediate_query_infos(request);
        let results_len = query_infos.len();
        let mut results = ShardQueryResponse::with_capacity(results_len);
        debug_assert!(
            all_shards_results
                .iter()
                .all(|shard_results| shard_results.len() == results_len)
        );

        let (collection_params, deduplicate_by_external_id) = {
            let config = self.collection_config.read().await;
            (
                config.params.clone(),
                AutoShardPolicy::canonical_ref(config.auto_shard_policy.as_ref()).is_some(),
            )
        };

        // Shape: [num_internal_queries, num_shards, num_scored_points]
        let all_shards_result_by_transposed = transposed_iter(all_shards_results);

        for (query_info, shards_results) in
            query_infos.into_iter().zip(all_shards_result_by_transposed)
        {
            // `shards_results` shape: [num_shards, num_scored_points]
            let order =
                shard_query::query_result_order(query_info.scoring_query, &collection_params)?;
            let number_of_shards = shards_results.len();

            // Equivalent to:
            //
            // shards_results
            //     .into_iter()
            //     .kmerge_by(match order {
            //         Order::LargeBetter => |a, b| ScoredPointTies(a) > ScoredPointTies(b),
            //         Order::SmallBetter => |a, b| ScoredPointTies(a) < ScoredPointTies(b),
            //     })
            //
            // if the `kmerge_by` function were able to work with reference predicates.
            // Either::Left and Either::Right are used to allow type inference to work.
            //
            let intermediate_result = if let Some(order) = order {
                let best_last_result = Self::get_best_last_shard_result(&shards_results, order);

                let merged = merge_ordered_shard_results(
                    shards_results,
                    order,
                    query_info.take,
                    deduplicate_by_external_id,
                );

                // Prevents undersampling warning in case there are not enough data to merge.
                let is_enough = merged.len() == query_info.take;

                if let Some(best_last_result) = best_last_result
                    && number_of_shards > 1
                    && is_enough
                {
                    let worst_merged_point = merged.last();
                    if let Some(worst_merged_point) = worst_merged_point {
                        self.check_undersampling(worst_merged_point, &best_last_result, order);
                    }
                }

                merged
            } else {
                // If the order is not defined, it is a random query. Take from all shards randomly.
                let mut rng = rand::rng();
                shards_results
                    .into_iter()
                    .kmerge_by(|_, _| rng.random_bool(0.5))
                    .unique_by(|point| point.id)
                    .take(query_info.take)
                    .collect()
            };

            results.push(intermediate_result);
        }

        Ok(results)
    }
}

/// Returns a list of the query that corresponds to each of the results in each shard.
///
/// Example: `[info1, info2, info3]` corresponds to `[result1, result2, result3]` of each shard
fn intermediate_query_infos(request: &ShardQueryRequest) -> Vec<IntermediateQueryInfo<'_>> {
    let scoring_query = request.query.as_ref();

    match scoring_query {
        Some(ScoringQuery::Fusion(_)) => {
            // In case of Fusion, expect the propagated intermediate results
            request
                .prefetches
                .iter()
                .map(|prefetch| IntermediateQueryInfo {
                    scoring_query: prefetch.query.as_ref(),
                    take: prefetch.limit,
                })
                .collect_vec()
        }
        Some(ScoringQuery::Mmr(MmrInternal {
            vector: _,
            using: _,
            lambda: _,
            candidates_limit,
        })) => {
            // In case of MMR, expect a single list with the amount of candidates
            vec![IntermediateQueryInfo {
                scoring_query: request.query.as_ref(),
                take: *candidates_limit,
            }]
        }
        None
        | Some(ScoringQuery::Vector(_))
        | Some(ScoringQuery::OrderBy(_))
        | Some(ScoringQuery::Formula(_))
        | Some(ScoringQuery::Sample(_)) => {
            // Otherwise, we expect the root result
            vec![IntermediateQueryInfo {
                scoring_query: request.query.as_ref(),
                take: request.offset + request.limit,
            }]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scored_point(id: u64, score: f32) -> ScoredPoint {
        ScoredPoint {
            id: id.into(),
            version: 0,
            score,
            payload: None,
            vector: None,
            shard_key: None,
            order_value: None,
        }
    }

    fn high_limit_query() -> ShardQueryRequest {
        ShardQueryRequest {
            prefetches: vec![],
            query: None,
            filter: None,
            score_threshold: None,
            limit: Collection::SHARD_QUERY_SUBSAMPLING_LIMIT,
            offset: 32,
            params: None,
            with_vector: WithVector::Bool(false),
            with_payload: WithPayloadInterface::Bool(false),
        }
    }

    fn apply_policy_gated_undersampling(
        policy: Option<&AutoShardPolicy>,
    ) -> Arc<Vec<ShardQueryRequest>> {
        Collection::modify_shard_query_for_undersampling_limits(
            Arc::new(vec![high_limit_query()]),
            4,
            auto_shard_policy_allows_query_undersampling(policy),
        )
    }

    #[test]
    fn hash_all_policy_keeps_collection_level_query_undersampling() {
        for policy in [None, Some(&AutoShardPolicy::HashAll)] {
            let request = apply_policy_gated_undersampling(policy);
            assert_eq!(request[0].offset, 0);
            assert!(
                request[0].limit < high_limit_query().limit + high_limit_query().offset,
                "HashAll must retain Qdrant's collection-level shard undersampling",
            );
        }
    }

    #[test]
    fn orion_policy_disables_collection_level_query_undersampling_on_fallback() {
        let policy = AutoShardPolicy::Orion {
            generation: 7,
            artifact_sha256: "0".repeat(64),
        };
        let request = apply_policy_gated_undersampling(Some(&policy));

        assert_eq!(request[0].limit, high_limit_query().limit);
        assert_eq!(request[0].offset, high_limit_query().offset);

        let policy = AutoShardPolicy::SimpleKmeans {
            generation: 8,
            artifact_sha256: "0".repeat(64),
        };
        let request = apply_policy_gated_undersampling(Some(&policy));
        assert_eq!(request[0].limit, high_limit_query().limit);
        assert_eq!(request[0].offset, high_limit_query().offset);
    }

    #[test]
    fn orion_fallback_deduplicates_external_ids_even_when_copy_scores_differ() {
        let shards_results = vec![
            vec![scored_point(1, 0.9), scored_point(2, 0.7)],
            vec![scored_point(1, 0.8), scored_point(3, 0.6)],
        ];

        let legacy =
            merge_ordered_shard_results(shards_results.clone(), Order::LargeBetter, 2, false);
        assert_eq!(
            legacy.iter().map(|point| point.id).collect::<Vec<_>>(),
            vec![1_u64.into(), 1_u64.into()],
        );

        let orion = merge_ordered_shard_results(shards_results, Order::LargeBetter, 2, true);
        assert_eq!(
            orion.iter().map(|point| point.id).collect::<Vec<_>>(),
            vec![1_u64.into(), 2_u64.into()],
        );
        assert_eq!(orion[0].score, 0.9);
    }
}
