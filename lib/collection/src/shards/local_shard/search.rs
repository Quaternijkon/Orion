use std::ops::Range;
use std::sync::Arc;
use std::time::Duration;

use common::counter::hardware_accumulator::HwMeasurementAcc;
use segment::data_types::query_context::QueryContext;
use segment::types::ScoredPoint;
use shard::common::stopping_guard::StoppingGuard;
use shard::query::query_enum::QueryEnum;
use shard::search::CoreSearchRequestBatch;
use tokio::runtime::Handle;

use super::LocalShard;
use crate::collection_manager::segments_searcher::SegmentsSearcher;
use crate::operations::types::{CollectionError, CollectionResult};

// Chunk requests for parallelism in certain scenarios
//
// Deeper down, each segment gets its own dedicated search thread. If this shard has just
// one segment, all requests will be executed on a single thread.
//
// To prevent this from being a bottleneck if we have a lot of requests, we can chunk the
// requests into multiple searches to allow more parallelism.
//
// For simplicity, we use a fixed chunk size. Using chunks helps to ensure our 'filter
// reuse optimization' is still properly utilized.
// See: <https://github.com/qdrant/qdrant/pull/813>
// See: <https://github.com/qdrant/qdrant/pull/6326>
const CHUNK_SIZE: usize = 16;

impl LocalShard {
    pub async fn do_search(
        &self,
        core_request: Arc<CoreSearchRequestBatch>,
        search_runtime_handle: &Handle,
        timeout: Duration,
        hw_counter_acc: HwMeasurementAcc,
    ) -> CollectionResult<Vec<Vec<ScoredPoint>>> {
        if core_request.searches.is_empty() {
            return Ok(vec![]);
        }

        let skip_batching = if core_request.searches.len() <= CHUNK_SIZE {
            // Don't batch if we have few searches, prevents cloning request
            true
        } else if self.segments.read().len() > self.shared_storage_config.search_thread_count {
            // Don't batch if we have more segments than search threads
            // Not a perfect condition, but it helps to prevent consuming a lot of search threads
            // if the number of segments is large
            // Note: search threads are shared with all other search threads on this Qdrant
            // instance, and other shards also have segments. For simplicity this only considers
            // the global search thread count and local segment count.
            // See: <https://github.com/qdrant/qdrant/pull/6478>
            true
        } else {
            false
        };

        let is_stopped_guard = StoppingGuard::new();
        let start = std::time::Instant::now();
        let (query_context, collection_params) = {
            let collection_config = self.collection_config.read().await;
            let query_context_opt = SegmentsSearcher::prepare_query_context(
                self.segments.clone(),
                &core_request,
                &collection_config,
                timeout,
                search_runtime_handle,
                &is_stopped_guard,
                hw_counter_acc,
            )
            .await?;

            let Some(query_context) = query_context_opt else {
                // No segments to search
                return Ok(vec![]);
            };

            (query_context, collection_config.params.clone())
        };

        // update timeout
        let timeout = timeout.saturating_sub(start.elapsed());
        let query_context = Arc::new(query_context);

        // Retain the existing fixed-size parallelism while sharing the original
        // request allocation and immutable query context across all chunks.
        let chunk_size = if skip_batching {
            core_request.searches.len()
        } else {
            CHUNK_SIZE
        };
        let search_ranges = (0..core_request.searches.len())
            .step_by(chunk_size)
            .map(|start| start..(start + chunk_size).min(core_request.searches.len()))
            .collect::<Vec<_>>();

        let chunk_futures = search_ranges
            .into_iter()
            .map(|batch_range| {
                self.do_search_batch_range(
                    core_request.clone(),
                    batch_range,
                    search_runtime_handle,
                    timeout,
                    query_context.clone(),
                )
            })
            .collect::<Vec<_>>();

        let res = futures::future::try_join_all(chunk_futures)
            .await?
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();

        if res.len() != core_request.searches.len() {
            return Err(CollectionError::service_error(format!(
                "search batch returned {} rows for {} requests",
                res.len(),
                core_request.searches.len(),
            )));
        }

        let top_results = res
            .into_iter()
            .zip(core_request.searches.iter())
            .map(|(vector_res, req)| {
                let vector_name = req.query.get_vector_name();
                let distance = collection_params.get_distance(vector_name).unwrap();
                let processed_res = vector_res.into_iter().map(|mut scored_point| {
                    match req.query {
                        QueryEnum::Nearest(_) => {
                            scored_point.score = distance.postprocess_score(scored_point.score);
                        }
                        // Don't post-process if we are dealing with custom scoring
                        QueryEnum::RecommendBestScore(_)
                        | QueryEnum::RecommendSumScores(_)
                        | QueryEnum::Discover(_)
                        | QueryEnum::Context(_)
                        | QueryEnum::FeedbackNaive(_) => {}
                    };
                    scored_point
                });

                if let Some(threshold) = req.score_threshold {
                    processed_res
                        .take_while(|scored_point| {
                            distance.check_threshold(scored_point.score, threshold)
                        })
                        .collect()
                } else {
                    processed_res.collect()
                }
            })
            .collect();
        Ok(top_results)
    }

    async fn do_search_batch_range(
        &self,
        core_request: Arc<CoreSearchRequestBatch>,
        batch_range: Range<usize>,
        search_runtime_handle: &Handle,
        timeout: Duration,
        query_context: Arc<QueryContext>,
    ) -> CollectionResult<Vec<Vec<ScoredPoint>>> {
        let range_for_log = batch_range.clone();

        let search_request = SegmentsSearcher::search_batch_range(
            self.segments.clone(),
            core_request,
            batch_range,
            search_runtime_handle,
            true,
            query_context,
            timeout,
        );

        let res = tokio::time::timeout(timeout, search_request)
            .await
            .map_err(|_| {
                log::debug!(
                    "Search timeout reached for batch range {range_for_log:?}: {timeout:?}"
                );
                // StoppingGuard takes care of setting is_stopped to true
                CollectionError::timeout(timeout, "Search")
            })??;
        if res.len() != range_for_log.len() {
            return Err(CollectionError::service_error(format!(
                "search batch range {range_for_log:?} returned {} rows",
                res.len(),
            )));
        }
        Ok(res)
    }
}
