use std::collections::HashSet;
use std::time::Duration;

use api::rest::SearchGroupsRequestInternal;
use collection::collection::distance_matrix::*;
use collection::common::batching::batch_requests;
use collection::grouping::group_by::GroupRequest;
use collection::operations::consistency_params::ReadConsistency;
use collection::operations::shard_selector_internal::ShardSelectorInternal;
use collection::operations::types::*;
use collection::operations::universal_query::collection_query::*;
use common::counter::hardware_accumulator::HwMeasurementAcc;
use futures::future;
use segment::types::{ExtendedPointId, ScoredPoint, ShardKey};
use shard::retrieve::record_internal::RecordInternal;
use shard::scroll::ScrollRequestInternal;
use shard::search::CoreSearchRequestBatch;
use storage::content_manager::errors::StorageError;
use storage::content_manager::toc::TableOfContent;
use storage::rbac::Auth;

#[allow(clippy::too_many_arguments)]
pub async fn do_core_search_points(
    toc: &TableOfContent,
    collection_name: &str,
    request: CoreSearchRequest,
    read_consistency: Option<ReadConsistency>,
    shard_selection: ShardSelectorInternal,
    auth: Auth,
    timeout: Option<Duration>,
    hw_measurement_acc: HwMeasurementAcc,
) -> Result<Vec<ScoredPoint>, StorageError> {
    let batch_res = do_core_search_batch_points(
        toc,
        collection_name,
        CoreSearchRequestBatch {
            searches: vec![request],
        },
        read_consistency,
        shard_selection,
        auth,
        timeout,
        hw_measurement_acc,
    )
    .await?;
    batch_res
        .into_iter()
        .next()
        .ok_or_else(|| StorageError::service_error("Empty search result"))
}

fn request_uses_per_shard_hnsw(request: &CoreSearchRequest) -> bool {
    request.hnsw_entry_points_by_shard.is_some() || request.hnsw_ef_by_shard.is_some()
}

fn should_use_shard_major_search_batch(
    requests: &[(CoreSearchRequest, ShardSelectorInternal)],
) -> bool {
    !requests.is_empty()
        && requests.iter().all(|(request, selector)| {
            request_uses_per_shard_hnsw(request)
                && matches!(selector, ShardSelectorInternal::ShardKeys(keys) if !keys.is_empty())
        })
}

fn specialize_core_search_for_shard(
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

fn infer_large_better(candidate_groups: &[Vec<ScoredPoint>]) -> bool {
    for group in candidate_groups {
        for pair in group.windows(2) {
            if pair[0].score != pair[1].score {
                return pair[0].score > pair[1].score;
            }
        }
    }
    true
}

fn merge_shard_major_candidates(
    candidate_groups: Vec<Vec<ScoredPoint>>,
    limit: usize,
    offset: usize,
    source_id_dedup_block_size: Option<u64>,
) -> Vec<ScoredPoint> {
    let large_better = infer_large_better(&candidate_groups);
    let mut candidates = candidate_groups.into_iter().flatten().collect::<Vec<_>>();
    if large_better {
        candidates.sort_by(|a, b| b.cmp(a));
    } else {
        candidates.sort_by(|a, b| a.cmp(b));
    }

    let mut seen_ids = HashSet::new();
    candidates
        .into_iter()
        .filter(|point| {
            seen_ids.insert(search_dedup_point_id(
                point.id,
                source_id_dedup_block_size,
            ))
        })
        .skip(offset)
        .take(limit)
        .collect()
}

#[allow(clippy::too_many_arguments)]
async fn do_search_batch_points_shard_major(
    toc: &TableOfContent,
    collection_name: &str,
    requests: Vec<(CoreSearchRequest, ShardSelectorInternal)>,
    read_consistency: Option<ReadConsistency>,
    auth: Auth,
    timeout: Option<Duration>,
    hw_measurement_acc: HwMeasurementAcc,
) -> Result<Vec<Vec<ScoredPoint>>, StorageError> {
    let original_requests = requests
        .iter()
        .map(|(request, _selector)| request.clone())
        .collect::<Vec<_>>();
    let mut shard_groups: Vec<(ShardKey, Vec<(usize, CoreSearchRequest)>)> = Vec::new();

    for (request_idx, (request, selector)) in requests.into_iter().enumerate() {
        let ShardSelectorInternal::ShardKeys(shard_keys) = selector else {
            unreachable!("shard-major search batch is prevalidated")
        };

        for shard_key in shard_keys {
            let specialized = specialize_core_search_for_shard(&request, &shard_key);
            if let Some((_key, items)) = shard_groups
                .iter_mut()
                .find(|(known_key, _items)| *known_key == shard_key)
            {
                items.push((request_idx, specialized));
            } else {
                shard_groups.push((shard_key, vec![(request_idx, specialized)]));
            }
        }
    }

    let shard_searches = shard_groups.into_iter().map(|(shard_key, items)| {
        let original_indices = items
            .iter()
            .map(|(request_idx, _request)| *request_idx)
            .collect::<Vec<_>>();
        let searches = items
            .into_iter()
            .map(|(_request_idx, request)| request)
            .collect::<Vec<_>>();
        let request = CoreSearchRequestBatch { searches };
        let search = toc.core_search_batch(
            collection_name,
            request,
            read_consistency,
            ShardSelectorInternal::ShardKey(shard_key),
            auth.clone(),
            timeout,
            hw_measurement_acc.clone(),
        );
        async move {
            let rows = search.await?;
            Ok::<_, StorageError>((original_indices, rows))
        }
    });

    let shard_results = future::try_join_all(shard_searches).await?;
    let mut candidates_by_request = vec![Vec::new(); original_requests.len()];
    for (original_indices, rows) in shard_results {
        for (request_idx, row) in original_indices.into_iter().zip(rows.into_iter()) {
            candidates_by_request[request_idx].push(row);
        }
    }

    Ok(candidates_by_request
        .into_iter()
        .enumerate()
        .map(|(request_idx, candidate_groups)| {
            let request = &original_requests[request_idx];
            merge_shard_major_candidates(
                candidate_groups,
                request.limit,
                request.offset,
                request.source_id_dedup_block_size,
            )
        })
        .collect())
}

pub async fn do_search_batch_points(
    toc: &TableOfContent,
    collection_name: &str,
    requests: Vec<(CoreSearchRequest, ShardSelectorInternal)>,
    read_consistency: Option<ReadConsistency>,
    auth: Auth,
    timeout: Option<Duration>,
    hw_measurement_acc: HwMeasurementAcc,
) -> Result<Vec<Vec<ScoredPoint>>, StorageError> {
    if should_use_shard_major_search_batch(&requests) {
        return do_search_batch_points_shard_major(
            toc,
            collection_name,
            requests,
            read_consistency,
            auth,
            timeout,
            hw_measurement_acc,
        )
        .await;
    }

    let requests = batch_requests::<
        (CoreSearchRequest, ShardSelectorInternal),
        ShardSelectorInternal,
        Vec<CoreSearchRequest>,
        Vec<_>,
    >(
        requests,
        |(_, shard_selector)| shard_selector,
        |(request, _), core_reqs| {
            core_reqs.push(request);
            Ok(())
        },
        |shard_selector, core_requests, res| {
            if core_requests.is_empty() {
                return Ok(());
            }

            let core_batch = CoreSearchRequestBatch {
                searches: core_requests,
            };

            let req = toc.core_search_batch(
                collection_name,
                core_batch,
                read_consistency,
                shard_selector,
                auth.clone(),
                timeout,
                hw_measurement_acc.clone(),
            );
            res.push(req);
            Ok(())
        },
    )?;

    let results = futures::future::try_join_all(requests).await?;
    let flatten_results: Vec<Vec<_>> = results.into_iter().flatten().collect();
    Ok(flatten_results)
}

#[allow(clippy::too_many_arguments)]
pub async fn do_core_search_batch_points(
    toc: &TableOfContent,
    collection_name: &str,
    request: CoreSearchRequestBatch,
    read_consistency: Option<ReadConsistency>,
    shard_selection: ShardSelectorInternal,
    auth: Auth,
    timeout: Option<Duration>,
    hw_measurement_acc: HwMeasurementAcc,
) -> Result<Vec<Vec<ScoredPoint>>, StorageError> {
    toc.core_search_batch(
        collection_name,
        request,
        read_consistency,
        shard_selection,
        auth,
        timeout,
        hw_measurement_acc,
    )
    .await
}

#[allow(clippy::too_many_arguments)]
pub async fn do_search_point_groups(
    toc: &TableOfContent,
    collection_name: &str,
    request: SearchGroupsRequestInternal,
    read_consistency: Option<ReadConsistency>,
    shard_selection: ShardSelectorInternal,
    auth: Auth,
    timeout: Option<Duration>,
    hw_measurement_acc: HwMeasurementAcc,
) -> Result<GroupsResult, StorageError> {
    toc.group(
        collection_name,
        GroupRequest::from(request),
        read_consistency,
        shard_selection,
        auth,
        timeout,
        hw_measurement_acc,
    )
    .await
}

#[allow(clippy::too_many_arguments)]
pub async fn do_recommend_point_groups(
    toc: &TableOfContent,
    collection_name: &str,
    request: RecommendGroupsRequestInternal,
    read_consistency: Option<ReadConsistency>,
    shard_selection: ShardSelectorInternal,
    auth: Auth,
    timeout: Option<Duration>,
    hw_measurement_acc: HwMeasurementAcc,
) -> Result<GroupsResult, StorageError> {
    toc.group(
        collection_name,
        GroupRequest::from(request),
        read_consistency,
        shard_selection,
        auth,
        timeout,
        hw_measurement_acc,
    )
    .await
}

pub async fn do_discover_batch_points(
    toc: &TableOfContent,
    collection_name: &str,
    request: DiscoverRequestBatch,
    read_consistency: Option<ReadConsistency>,
    auth: Auth,
    timeout: Option<Duration>,
    hw_measurement_acc: HwMeasurementAcc,
) -> Result<Vec<Vec<ScoredPoint>>, StorageError> {
    let requests = request
        .searches
        .into_iter()
        .map(|req| {
            let shard_selector = match req.shard_key {
                None => ShardSelectorInternal::All,
                Some(shard_key) => ShardSelectorInternal::from(shard_key),
            };

            (req.discover_request, shard_selector)
        })
        .collect();

    toc.discover_batch(
        collection_name,
        requests,
        read_consistency,
        auth,
        timeout,
        hw_measurement_acc,
    )
    .await
}

#[allow(clippy::too_many_arguments)]
pub async fn do_count_points(
    toc: &TableOfContent,
    collection_name: &str,
    request: CountRequestInternal,
    read_consistency: Option<ReadConsistency>,
    timeout: Option<Duration>,
    shard_selection: ShardSelectorInternal,
    auth: Auth,
    hw_measurement_acc: HwMeasurementAcc,
) -> Result<CountResult, StorageError> {
    toc.count(
        collection_name,
        request,
        read_consistency,
        timeout,
        shard_selection,
        auth,
        hw_measurement_acc,
    )
    .await
}

#[allow(clippy::too_many_arguments)]
pub async fn do_get_points(
    toc: &TableOfContent,
    collection_name: &str,
    request: PointRequestInternal,
    read_consistency: Option<ReadConsistency>,
    timeout: Option<Duration>,
    shard_selection: ShardSelectorInternal,
    auth: Auth,
    hw_measurement_acc: HwMeasurementAcc,
) -> Result<Vec<RecordInternal>, StorageError> {
    toc.retrieve(
        collection_name,
        request,
        read_consistency,
        timeout,
        shard_selection,
        auth,
        hw_measurement_acc,
    )
    .await
}

#[allow(clippy::too_many_arguments)]
pub async fn do_scroll_points(
    toc: &TableOfContent,
    collection_name: &str,
    request: ScrollRequestInternal,
    read_consistency: Option<ReadConsistency>,
    timeout: Option<Duration>,
    shard_selection: ShardSelectorInternal,
    auth: Auth,
    hw_measurement_acc: HwMeasurementAcc,
) -> Result<ScrollResult, StorageError> {
    toc.scroll(
        collection_name,
        request,
        read_consistency,
        timeout,
        shard_selection,
        auth,
        hw_measurement_acc,
    )
    .await
}

#[allow(clippy::too_many_arguments)]
pub async fn do_query_points(
    toc: &TableOfContent,
    collection_name: &str,
    request: CollectionQueryRequest,
    read_consistency: Option<ReadConsistency>,
    shard_selection: ShardSelectorInternal,
    auth: Auth,
    timeout: Option<Duration>,
    hw_measurement_acc: HwMeasurementAcc,
) -> Result<Vec<ScoredPoint>, StorageError> {
    let requests = vec![(request, shard_selection)];
    let batch_res = toc
        .query_batch(
            collection_name,
            requests,
            read_consistency,
            auth,
            timeout,
            hw_measurement_acc,
        )
        .await?;
    batch_res
        .into_iter()
        .next()
        .ok_or_else(|| StorageError::service_error("Empty query result"))
}

#[allow(clippy::too_many_arguments)]
pub async fn do_query_batch_points(
    toc: &TableOfContent,
    collection_name: &str,
    requests: Vec<(CollectionQueryRequest, ShardSelectorInternal)>,
    read_consistency: Option<ReadConsistency>,
    auth: Auth,
    timeout: Option<Duration>,
    hw_measurement_acc: HwMeasurementAcc,
) -> Result<Vec<Vec<ScoredPoint>>, StorageError> {
    toc.query_batch(
        collection_name,
        requests,
        read_consistency,
        auth,
        timeout,
        hw_measurement_acc,
    )
    .await
}

#[allow(clippy::too_many_arguments)]
pub async fn do_query_point_groups(
    toc: &TableOfContent,
    collection_name: &str,
    request: CollectionQueryGroupsRequest,
    read_consistency: Option<ReadConsistency>,
    shard_selection: ShardSelectorInternal,
    auth: Auth,
    timeout: Option<Duration>,
    hw_measurement_acc: HwMeasurementAcc,
) -> Result<GroupsResult, StorageError> {
    toc.group(
        collection_name,
        GroupRequest::from(request),
        read_consistency,
        shard_selection,
        auth,
        timeout,
        hw_measurement_acc,
    )
    .await
}

#[allow(clippy::too_many_arguments)]
pub async fn do_search_points_matrix(
    toc: &TableOfContent,
    collection_name: &str,
    request: CollectionSearchMatrixRequest,
    read_consistency: Option<ReadConsistency>,
    shard_selection: ShardSelectorInternal,
    auth: Auth,
    timeout: Option<Duration>,
    hw_measurement_acc: HwMeasurementAcc,
) -> Result<CollectionSearchMatrixResponse, StorageError> {
    toc.search_points_matrix(
        collection_name,
        request,
        read_consistency,
        shard_selection,
        auth,
        timeout,
        hw_measurement_acc,
    )
    .await
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scored(id: u64, score: f32) -> ScoredPoint {
        ScoredPoint {
            id: ExtendedPointId::NumId(id),
            version: 0,
            score,
            payload: None,
            vector: None,
            shard_key: None,
            order_value: None,
        }
    }

    #[test]
    fn shard_major_specialization_applies_entry_points_and_ef_for_one_shard() {
        let shard_key = ShardKey::from("centroid_00");
        let mut entry_points_by_shard = std::collections::HashMap::new();
        entry_points_by_shard.insert(
            shard_key.clone(),
            vec![ExtendedPointId::NumId(11), ExtendedPointId::NumId(13)],
        );
        let mut ef_by_shard = std::collections::HashMap::new();
        ef_by_shard.insert(shard_key.clone(), 24);

        let request = CoreSearchRequest {
            query: vec![0.1, 0.2].into(),
            filter: None,
            params: None,
            hnsw_entry_points: None,
            hnsw_entry_points_by_shard: Some(entry_points_by_shard),
            hnsw_ef_by_shard: Some(ef_by_shard),
            source_id_dedup_block_size: Some(1001),
            limit: 10,
            offset: 3,
            with_payload: None,
            with_vector: None,
            score_threshold: None,
        };

        let specialized = specialize_core_search_for_shard(&request, &shard_key);

        assert_eq!(
            specialized.hnsw_entry_points,
            Some(vec![ExtendedPointId::NumId(11), ExtendedPointId::NumId(13)])
        );
        assert_eq!(specialized.params.unwrap().hnsw_ef, Some(24));
        assert!(specialized.hnsw_entry_points_by_shard.is_none());
        assert!(specialized.hnsw_ef_by_shard.is_none());
        assert_eq!(specialized.limit, 13);
        assert_eq!(specialized.offset, 0);
    }

    #[test]
    fn shard_major_merge_dedups_copied_point_ids_before_limit() {
        let merged = merge_shard_major_candidates(
            vec![
                vec![scored(207, 0.90), scored(42, 0.80)],
                vec![scored(7, 0.95), scored(43, 0.70)],
            ],
            2,
            0,
            Some(100),
        );

        assert_eq!(
            merged
                .into_iter()
                .map(|point| point.id)
                .collect::<Vec<_>>(),
            vec![ExtendedPointId::NumId(7), ExtendedPointId::NumId(42)]
        );
    }

    #[test]
    fn shard_major_merge_can_infer_small_better_order_from_shard_rows() {
        let merged = merge_shard_major_candidates(
            vec![
                vec![scored(1, 0.10), scored(2, 0.20)],
                vec![scored(3, 0.05), scored(4, 0.30)],
            ],
            2,
            0,
            None,
        );

        assert_eq!(
            merged
                .into_iter()
                .map(|point| point.id)
                .collect::<Vec<_>>(),
            vec![ExtendedPointId::NumId(3), ExtendedPointId::NumId(1)]
        );
    }
}
