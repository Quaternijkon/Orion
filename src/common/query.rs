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
use segment::utils::scored_point_ties::ScoredPointTies;
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

fn shard_major_peer_premerge_disabled() -> bool {
    std::env::var("QDRANT_DISABLE_SHARD_MAJOR_PEER_PREMERGE").is_ok_and(|value| {
        let value = value.to_ascii_lowercase();
        matches!(value.as_str(), "1" | "true" | "yes" | "on")
    })
}

fn specialize_core_search_for_shard(
    request: &CoreSearchRequest,
    shard_key: &ShardKey,
) -> Result<CoreSearchRequest, StorageError> {
    let mut specialized = request.clone();

    specialized.limit = request.limit.checked_add(request.offset).ok_or_else(|| {
        StorageError::bad_request("shard-major lower-search limit + offset overflows usize")
    })?;
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
    Ok(specialized)
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

pub(crate) fn merge_shard_major_candidates(
    candidate_groups: Vec<Vec<ScoredPoint>>,
    limit: usize,
    offset: usize,
    source_id_dedup_block_size: Option<u64>,
) -> Vec<ScoredPoint> {
    let large_better = infer_large_better(&candidate_groups);
    let mut candidates = candidate_groups.into_iter().flatten().collect::<Vec<_>>();
    if large_better {
        candidates.sort_by(|a, b| ScoredPointTies(b).cmp(&ScoredPointTies(a)));
    } else {
        candidates.sort_by(|a, b| ScoredPointTies(a).cmp(&ScoredPointTies(b)));
    }

    let mut seen_ids = HashSet::new();
    candidates
        .into_iter()
        .filter(|point| {
            seen_ids.insert(search_dedup_point_id(point.id, source_id_dedup_block_size))
        })
        .skip(offset)
        .take(limit)
        .collect()
}

#[cfg(test)]
fn premerge_shard_major_candidates_by_peer(
    peer_candidate_groups: Vec<Vec<Vec<ScoredPoint>>>,
    limit: usize,
    offset: usize,
    source_id_dedup_block_size: Option<u64>,
) -> Result<Vec<Vec<ScoredPoint>>, StorageError> {
    let peer_limit = limit.checked_add(offset).ok_or_else(|| {
        StorageError::bad_request("peer-local merge limit + offset overflows usize")
    })?;
    Ok(peer_candidate_groups
        .into_iter()
        .map(|candidate_groups| {
            merge_shard_major_candidates(
                candidate_groups,
                peer_limit,
                0,
                source_id_dedup_block_size,
            )
        })
        .collect())
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
    if !shard_major_peer_premerge_disabled()
        && let Some(results) = toc
            .core_search_batch_shard_major_peer_premerge(
                collection_name,
                requests.clone(),
                read_consistency,
                auth.clone(),
                timeout,
                hw_measurement_acc.clone(),
            )
            .await?
    {
        return Ok(results);
    }

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
            let specialized = specialize_core_search_for_shard(&request, &shard_key)?;
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

        let specialized = specialize_core_search_for_shard(&request, &shard_key).unwrap();

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
    fn shard_major_specialization_and_peer_premerge_reject_window_overflow() {
        let shard_key = ShardKey::from("centroid_00");
        let request = CoreSearchRequest {
            query: vec![0.1, 0.2].into(),
            filter: None,
            params: None,
            hnsw_entry_points: None,
            hnsw_entry_points_by_shard: Some(std::collections::HashMap::new()),
            hnsw_ef_by_shard: Some(std::collections::HashMap::new()),
            source_id_dedup_block_size: None,
            limit: usize::MAX,
            offset: 1,
            with_payload: None,
            with_vector: None,
            score_threshold: None,
        };

        let specialization = specialize_core_search_for_shard(&request, &shard_key).unwrap_err();
        assert!(
            specialization
                .to_string()
                .contains("limit + offset overflows")
        );

        let premerge = premerge_shard_major_candidates_by_peer(
            vec![vec![vec![scored(1, 1.0)]]],
            usize::MAX,
            1,
            None,
        )
        .unwrap_err();
        assert!(premerge.to_string().contains("limit + offset overflows"));
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
            merged.into_iter().map(|point| point.id).collect::<Vec<_>>(),
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
            merged.into_iter().map(|point| point.id).collect::<Vec<_>>(),
            vec![ExtendedPointId::NumId(3), ExtendedPointId::NumId(1)]
        );
    }

    #[test]
    fn shard_major_peer_local_premerge_preserves_global_merge_with_offset() {
        let peer_a = vec![
            vec![scored(10, 0.99), scored(20, 0.80)],
            vec![scored(30, 0.97), scored(40, 0.96)],
        ];
        let peer_b = vec![
            vec![scored(10, 0.98), scored(50, 0.95)],
            vec![scored(60, 0.94), scored(70, 0.93)],
        ];

        let baseline = merge_shard_major_candidates(
            peer_a.clone().into_iter().chain(peer_b.clone()).collect(),
            3,
            1,
            None,
        );
        let peer_partials =
            premerge_shard_major_candidates_by_peer(vec![peer_a, peer_b], 3, 1, None).unwrap();
        let two_stage = merge_shard_major_candidates(peer_partials, 3, 1, None);

        assert_eq!(
            two_stage
                .into_iter()
                .map(|point| point.id)
                .collect::<Vec<_>>(),
            baseline
                .into_iter()
                .map(|point| point.id)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn shard_major_peer_local_premerge_preserves_source_id_dedup() {
        let peer_a = vec![
            vec![scored(101, 0.96), scored(202, 0.94), scored(303, 0.92)],
            vec![scored(404, 0.90), scored(505, 0.88), scored(606, 0.86)],
        ];
        let peer_b = vec![
            vec![scored(1, 0.99), scored(2, 0.98), scored(707, 0.84)],
            vec![scored(808, 0.83), scored(909, 0.82), scored(1000, 0.81)],
        ];

        let baseline = merge_shard_major_candidates(
            peer_a.clone().into_iter().chain(peer_b.clone()).collect(),
            4,
            1,
            Some(100),
        );
        let peer_partials =
            premerge_shard_major_candidates_by_peer(vec![peer_a, peer_b], 4, 1, Some(100)).unwrap();
        let two_stage = merge_shard_major_candidates(peer_partials, 4, 1, Some(100));

        assert_eq!(
            two_stage
                .into_iter()
                .map(|point| point.id)
                .collect::<Vec<_>>(),
            baseline
                .into_iter()
                .map(|point| point.id)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn shard_major_chunked_peer_premerge_preserves_topk_offset_and_cross_chunk_dedup() {
        let shard_rows = [
            vec![scored(101, 0.99), scored(3, 0.92), scored(5, 0.75)],
            vec![scored(4, 0.96), scored(6, 0.90), scored(8, 0.70)],
            vec![scored(1, 0.98), scored(7, 0.94), scored(9, 0.80)],
            vec![scored(102, 0.97), scored(10, 0.93), scored(11, 0.85)],
        ];

        let baseline = merge_shard_major_candidates(shard_rows.clone().into(), 3, 1, Some(100));
        let chunk_partials = premerge_shard_major_candidates_by_peer(
            vec![
                vec![shard_rows[0].clone(), shard_rows[1].clone()],
                vec![shard_rows[2].clone(), shard_rows[3].clone()],
            ],
            3,
            1,
            Some(100),
        )
        .unwrap();
        let chunked = merge_shard_major_candidates(chunk_partials, 3, 1, Some(100));

        assert_eq!(chunked, baseline);
        assert_eq!(
            chunked
                .into_iter()
                .map(|point| point.id)
                .collect::<Vec<_>>(),
            vec![
                ExtendedPointId::NumId(102),
                ExtendedPointId::NumId(4),
                ExtendedPointId::NumId(7),
            ],
        );
    }

    #[test]
    fn shard_major_peer_and_chunk_reordering_is_stable_at_equal_score_boundaries() {
        let peer_a = vec![
            vec![scored(101, 0.9), scored(2, 0.9)],
            vec![scored(303, 0.9)],
        ];
        let peer_b = vec![vec![scored(1, 0.9), scored(202, 0.9)], vec![scored(3, 0.9)]];

        let baseline = merge_shard_major_candidates(
            peer_a.clone().into_iter().chain(peer_b.clone()).collect(),
            3,
            0,
            Some(100),
        );
        let reordered = merge_shard_major_candidates(
            peer_b.clone().into_iter().chain(peer_a.clone()).collect(),
            3,
            0,
            Some(100),
        );
        let peer_partials =
            premerge_shard_major_candidates_by_peer(vec![peer_b, peer_a], 3, 0, Some(100)).unwrap();
        let two_stage = merge_shard_major_candidates(peer_partials, 3, 0, Some(100));

        assert_eq!(reordered, baseline);
        assert_eq!(two_stage, baseline);
        assert_eq!(
            baseline
                .into_iter()
                .map(|point| point.id)
                .collect::<Vec<_>>(),
            vec![
                ExtendedPointId::NumId(303),
                ExtendedPointId::NumId(202),
                ExtendedPointId::NumId(101),
            ],
        );
    }
}
