use std::collections::{BTreeMap, HashSet};
use std::time::{Duration, Instant};

use api::conversions::json::json_path_from_proto;
use api::grpc::qdrant::{
    BatchResult, CoreSearchByShardCompactEntry, CoreSearchByShardEntry,
    CoreSearchByShardQueryTemplate, CoreSearchPoints, CountPoints, CountResponse,
    DiscoverBatchResponse, DiscoverPoints, DiscoverResponse, FacetCounts, FacetResponse, GetPoints,
    GetResponse, GroupsResult, QueryBatchResponse, QueryGroupsResponse, QueryPointGroups,
    QueryPoints, QueryResponse, ReadConsistency as ReadConsistencyGrpc, RecommendBatchResponse,
    RecommendGroupsResponse, RecommendPointGroups, RecommendPoints, RecommendResponse,
    ScrollPoints, ScrollResponse, SearchBatchResponse, SearchGroupsResponse, SearchMatrixPoints,
    SearchPointGroups, SearchPoints, SearchResponse,
};
use api::grpc::{InferenceUsage, Usage};
use collection::collection::distance_matrix::{
    CollectionSearchMatrixRequest, CollectionSearchMatrixResponse,
};
use collection::operations::consistency_params::ReadConsistency;
use collection::operations::conversions::try_discover_request_from_grpc;
use collection::operations::shard_selector_internal::ShardSelectorInternal;
use collection::operations::types::{CoreSearchRequest, PointRequestInternal};
use collection::shards::shard::ShardId;
use common::counter::hardware_accumulator::HwMeasurementAcc;
use futures::future;
use segment::data_types::facets::FacetParams;
use segment::data_types::order_by::{OrderBy, OrderByInterface};
use segment::data_types::vectors::{DEFAULT_VECTOR_NAME, NamedQuery, VectorInternal};
use segment::types::{PointIdType, ScoredPoint};
use shard::count::CountRequestInternal;
use shard::query::query_enum::QueryEnum;
use shard::scroll::ScrollRequestInternal;
use shard::search::CoreSearchRequestBatch;
use storage::content_manager::errors::StorageError;
use storage::content_manager::toc::TableOfContent;
use storage::content_manager::toc::request_hw_counter::RequestHwCounter;
use storage::rbac::Auth;
use tonic::{Response, Status};

use crate::common::inference::params::InferenceParams;
use crate::common::inference::query_requests_grpc::{
    convert_query_point_groups_from_grpc, convert_query_points_from_grpc,
};
use crate::common::query::*;
use crate::common::strict_mode::*;

pub(crate) fn convert_shard_selector_for_read(
    shard_id_selector: Option<ShardId>,
    shard_key_selector: Option<api::grpc::qdrant::ShardKeySelector>,
) -> Result<ShardSelectorInternal, Status> {
    let res = match (shard_id_selector, shard_key_selector) {
        (Some(shard_id), None) => ShardSelectorInternal::ShardId(shard_id),
        (None, Some(shard_key_selector)) => ShardSelectorInternal::try_from(shard_key_selector)?,
        (None, None) => ShardSelectorInternal::All,
        (Some(shard_id), Some(_)) => {
            debug_assert!(
                false,
                "Shard selection and shard key selector are mutually exclusive"
            );
            ShardSelectorInternal::ShardId(shard_id)
        }
    };
    Ok(res)
}

pub async fn search(
    toc_provider: impl CheckedTocProvider,
    search_points: SearchPoints,
    shard_selection: Option<ShardId>,
    auth: Auth,
    hw_measurement_acc: RequestHwCounter,
) -> Result<Response<SearchResponse>, Status> {
    let SearchPoints {
        collection_name,
        vector,
        filter,
        limit,
        offset,
        with_payload,
        params,
        score_threshold,
        vector_name,
        with_vectors,
        read_consistency,
        timeout,
        shard_key_selector,
        sparse_indices,
    } = search_points;

    let vector_internal =
        VectorInternal::from_vector_and_indices(vector, sparse_indices.map(|v| v.data));

    let vector_struct =
        api::grpc::conversions::into_named_vector_struct(vector_name, vector_internal)?;

    let shard_selector = convert_shard_selector_for_read(shard_selection, shard_key_selector)?;

    let search_request = CoreSearchRequest {
        query: QueryEnum::Nearest(NamedQuery::from(vector_struct)),
        filter: filter.map(|f| f.try_into()).transpose()?,
        params: params.map(|p| p.into()),
        hnsw_entry_points: None,
        hnsw_entry_points_by_shard: None,
        hnsw_ef_by_shard: None,

        source_id_dedup_block_size: None,
        limit: limit as usize,
        offset: offset.unwrap_or_default() as usize,
        with_payload: with_payload.map(|wp| wp.try_into()).transpose()?,
        with_vector: Some(
            with_vectors
                .map(|selector| selector.into())
                .unwrap_or_default(),
        ),
        score_threshold,
    };

    let toc = toc_provider
        .check_strict_mode(
            &search_request,
            &collection_name,
            timeout.map(|i| i as usize),
            &auth,
        )
        .await?;

    let read_consistency = ReadConsistency::try_from_optional(read_consistency)?;

    let timing = Instant::now();
    let scored_points = do_core_search_points(
        toc,
        &collection_name,
        search_request,
        read_consistency,
        shard_selector,
        auth,
        timeout.map(Duration::from_secs),
        hw_measurement_acc.get_counter(),
    )
    .await?;

    let response = SearchResponse {
        result: scored_points
            .into_iter()
            .map(|point| point.into())
            .collect(),
        time: timing.elapsed().as_secs_f64(),
        usage: Usage::from_hardware_usage(hw_measurement_acc.to_grpc_api()).into_non_empty(),
    };

    Ok(Response::new(response))
}

pub async fn core_search_batch(
    toc_provider: impl CheckedTocProvider,
    collection_name: &str,
    requests: Vec<(CoreSearchRequest, ShardSelectorInternal)>,
    read_consistency: Option<ReadConsistencyGrpc>,
    auth: Auth,
    timeout: Option<Duration>,
    request_hw_counter: RequestHwCounter,
) -> Result<Response<SearchBatchResponse>, Status> {
    let toc = toc_provider
        .check_strict_mode_batch(
            &requests,
            |i| &i.0,
            collection_name,
            timeout.map(|i| i.as_secs() as usize),
            &auth,
        )
        .await?;

    let read_consistency = ReadConsistency::try_from_optional(read_consistency)?;

    let timing = Instant::now();

    let scored_points = do_search_batch_points(
        toc,
        collection_name,
        requests,
        read_consistency,
        auth,
        timeout,
        request_hw_counter.get_counter(),
    )
    .await?;

    let response = SearchBatchResponse {
        result: scored_points
            .into_iter()
            .map(|points| BatchResult {
                result: points.into_iter().map(|p| p.into()).collect(),
            })
            .collect(),
        time: timing.elapsed().as_secs_f64(),
        usage: Usage::from_hardware_usage(request_hw_counter.to_grpc_api()).into_non_empty(),
    };

    Ok(Response::new(response))
}

#[allow(clippy::too_many_arguments)]
pub async fn core_search_list(
    toc: &TableOfContent,
    collection_name: String,
    search_points: Vec<CoreSearchPoints>,
    read_consistency: Option<ReadConsistencyGrpc>,
    shard_selection: Option<ShardId>,
    auth: Auth,
    timeout: Option<Duration>,
    request_hw_counter: RequestHwCounter,
) -> Result<Response<SearchBatchResponse>, Status> {
    let searches: Result<Vec<_>, Status> = search_points
        .into_iter()
        .map(CoreSearchRequest::try_from)
        .collect();

    let request = CoreSearchRequestBatch {
        searches: searches?,
    };

    let timing = Instant::now();

    // As this function is handling an internal request,
    // we can assume that shard_key is already resolved
    let shard_selection = match shard_selection {
        None => {
            debug_assert!(false, "Shard selection is expected for internal request");
            ShardSelectorInternal::All
        }
        Some(shard_id) => ShardSelectorInternal::ShardId(shard_id),
    };

    let read_consistency = ReadConsistency::try_from_optional(read_consistency)?;

    let scored_points = toc
        .core_search_batch(
            &collection_name,
            request,
            read_consistency,
            shard_selection,
            auth,
            timeout,
            request_hw_counter.get_counter(),
        )
        .await?;

    let response = SearchBatchResponse {
        result: scored_points
            .into_iter()
            .map(|points| BatchResult {
                result: points.into_iter().map(|p| p.into()).collect(),
            })
            .collect(),
        time: timing.elapsed().as_secs_f64(),
        usage: Usage::from_hardware_usage(request_hw_counter.to_grpc_api()).into_non_empty(),
    };

    Ok(Response::new(response))
}

pub(crate) struct CoreSearchByShardRow {
    pub query_index: usize,
    pub limit: usize,
    pub offset: usize,
    pub source_id_dedup_block_size: Option<u64>,
    pub points: Vec<ScoredPoint>,
}

type CoreSearchByShardMergeSpec = (usize, usize, usize, Option<u64>);

fn attach_core_search_by_shard_results(
    shard_id: ShardId,
    original_rows: Vec<CoreSearchByShardMergeSpec>,
    rows: Vec<Vec<ScoredPoint>>,
) -> Result<Vec<CoreSearchByShardRow>, Status> {
    if rows.len() != original_rows.len() {
        return Err(Status::internal(format!(
            "Shard {shard_id} returned {} rows for {} peer-batched search slots",
            rows.len(),
            original_rows.len(),
        )));
    }

    Ok(original_rows
        .into_iter()
        .zip(rows)
        .map(
            |((query_index, limit, offset, source_id_dedup_block_size), points)| {
                CoreSearchByShardRow {
                    query_index,
                    limit,
                    offset,
                    source_id_dedup_block_size,
                    points,
                }
            },
        )
        .collect())
}

pub(crate) fn premerge_core_search_by_shard_rows(
    rows: Vec<CoreSearchByShardRow>,
    query_count: usize,
) -> Result<Vec<Vec<ScoredPoint>>, Status> {
    let mut candidate_groups_by_query = vec![Vec::new(); query_count];
    let mut merge_specs = vec![None; query_count];

    for row in rows {
        if row.query_index >= query_count {
            return Err(Status::internal(format!(
                "peer-local shard row query_index {} is outside query_count {query_count}",
                row.query_index,
            )));
        }

        let merge_spec = (row.limit, row.offset, row.source_id_dedup_block_size);
        if let Some(existing_spec) = merge_specs[row.query_index] {
            if existing_spec != merge_spec {
                return Err(Status::internal(format!(
                    "peer-local shard rows disagree on merge metadata for query_index {}",
                    row.query_index,
                )));
            }
        } else {
            merge_specs[row.query_index] = Some(merge_spec);
        }

        candidate_groups_by_query[row.query_index].push(row.points);
    }

    candidate_groups_by_query
        .into_iter()
        .enumerate()
        .map(|(query_index, candidate_groups)| {
            let (limit, offset, source_id_dedup_block_size) =
                merge_specs[query_index].ok_or_else(|| {
                    Status::internal(format!(
                        "peer-local shard response has no rows for query_index {query_index}"
                    ))
                })?;
            let peer_limit = limit.checked_add(offset).ok_or_else(|| {
                Status::invalid_argument(format!(
                    "peer-local merge limit + offset overflows usize for query_index {query_index}"
                ))
            })?;

            Ok(crate::common::query::merge_shard_major_candidates(
                candidate_groups,
                peer_limit,
                0,
                source_id_dedup_block_size,
            ))
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
pub async fn core_search_batch_by_shard(
    toc: &TableOfContent,
    collection_name: String,
    searches: Vec<CoreSearchByShardEntry>,
    auth: Auth,
    timeout: Option<Duration>,
    request_hw_counter: RequestHwCounter,
) -> Result<Response<SearchBatchResponse>, Status> {
    struct SearchByShardWork {
        query_index: usize,
        limit: usize,
        offset: usize,
        source_id_dedup_block_size: Option<u64>,
        request: CoreSearchRequest,
    }

    if searches.is_empty() {
        return Err(Status::invalid_argument("searches is empty"));
    }
    let search_entry_count = searches.len();
    let mut query_count = 0usize;
    let mut by_shard: BTreeMap<ShardId, Vec<SearchByShardWork>> = BTreeMap::new();

    for entry in searches {
        let query_index = usize::try_from(entry.query_index)
            .map_err(|_| Status::invalid_argument("query_index does not fit into usize"))?;
        if query_index >= search_entry_count {
            return Err(Status::invalid_argument(format!(
                "query_index {query_index} is outside the maximum dense query-slot count {search_entry_count}"
            )));
        }
        let limit = usize::try_from(entry.final_limit)
            .map_err(|_| Status::invalid_argument("final_limit does not fit into usize"))?;
        let offset = usize::try_from(entry.final_offset.unwrap_or_default())
            .map_err(|_| Status::invalid_argument("final_offset does not fit into usize"))?;
        if limit == 0 {
            return Err(Status::invalid_argument("final_limit must be positive"));
        }
        limit.checked_add(offset).ok_or_else(|| {
            Status::invalid_argument("final_limit + final_offset overflows usize")
        })?;
        if entry.source_id_dedup_block_size == Some(0) {
            return Err(Status::invalid_argument(
                "source_id_dedup_block_size must be positive when present",
            ));
        }
        let search_points = entry
            .search_points
            .ok_or_else(|| Status::invalid_argument("search_points is missing"))?;
        let request = CoreSearchRequest::try_from(search_points)?;

        query_count = query_count.max(
            query_index
                .checked_add(1)
                .ok_or_else(|| Status::invalid_argument("query_index is too large"))?,
        );
        by_shard
            .entry(entry.shard_id)
            .or_default()
            .push(SearchByShardWork {
                query_index,
                limit,
                offset,
                source_id_dedup_block_size: entry.source_id_dedup_block_size,
                request,
            });
    }

    let timing = Instant::now();
    let shard_searches = by_shard.into_iter().map(|(shard_id, works)| {
        let original_rows = works
            .iter()
            .map(|work| {
                (
                    work.query_index,
                    work.limit,
                    work.offset,
                    work.source_id_dedup_block_size,
                )
            })
            .collect::<Vec<_>>();
        let request = CoreSearchRequestBatch {
            searches: works.into_iter().map(|work| work.request).collect(),
        };
        let collection_name = collection_name.clone();
        let auth = auth.clone();
        let request_hw_counter = request_hw_counter.clone();

        async move {
            let rows = toc
                .core_search_batch(
                    &collection_name,
                    request,
                    None,
                    ShardSelectorInternal::ShardId(shard_id),
                    auth,
                    timeout,
                    request_hw_counter.get_counter(),
                )
                .await
                .map_err(Status::from)?;
            Ok::<_, Status>((shard_id, original_rows, rows))
        }
    });

    let shard_results = future::try_join_all(shard_searches).await?;
    let mut shard_rows = Vec::new();
    for (shard_id, original_rows, rows) in shard_results {
        shard_rows.extend(attach_core_search_by_shard_results(
            shard_id,
            original_rows,
            rows,
        )?);
    }

    let premerged_rows = premerge_core_search_by_shard_rows(shard_rows, query_count)?;
    let response = SearchBatchResponse {
        result: premerged_rows
            .into_iter()
            .map(|points| BatchResult {
                result: points.into_iter().map(|point| point.into()).collect(),
            })
            .collect(),
        time: timing.elapsed().as_secs_f64(),
        usage: Usage::from_hardware_usage(request_hw_counter.to_grpc_api()).into_non_empty(),
    };

    Ok(Response::new(response))
}

#[derive(Debug)]
struct CompactSearchTemplate {
    lower_limit: usize,
    final_limit: usize,
    final_offset: usize,
    source_id_dedup_block_size: Option<u64>,
}

#[derive(Debug)]
struct DecodedCompactQueryTemplates {
    requests: Vec<CoreSearchRequest>,
    metadata: Vec<CompactSearchTemplate>,
}

impl DecodedCompactQueryTemplates {
    fn len(&self) -> usize {
        debug_assert_eq!(self.requests.len(), self.metadata.len());
        self.requests.len()
    }
}

type CompactSearchOriginalRow = (usize, usize, usize, Option<u64>);

#[derive(Debug, Default)]
struct MaterializedCompactShardSearch {
    original_rows: Vec<CompactSearchOriginalRow>,
    requests: Vec<CoreSearchRequest>,
}

fn validate_compact_orion_query_template(
    query_slot: usize,
    request: &CoreSearchRequest,
) -> Result<(), Status> {
    if request.filter.is_some() {
        return Err(Status::invalid_argument(format!(
            "query template {query_slot} must not contain a filter"
        )));
    }
    if request.params.as_ref().is_some_and(|params| params.exact) {
        return Err(Status::invalid_argument(format!(
            "query template {query_slot} must not request exact search"
        )));
    }
    if request.hnsw_entry_points.is_some()
        || request.hnsw_entry_points_by_shard.is_some()
        || request.hnsw_ef_by_shard.is_some()
    {
        return Err(Status::invalid_argument(format!(
            "query template {query_slot} must not contain shard-specific HNSW overrides"
        )));
    }
    if request.source_id_dedup_block_size.is_some() {
        return Err(Status::invalid_argument(format!(
            "query template {query_slot} must carry source-ID dedup only in the compact template envelope"
        )));
    }

    let QueryEnum::Nearest(named_query) = &request.query else {
        return Err(Status::invalid_argument(format!(
            "query template {query_slot} must be a nearest-neighbor query"
        )));
    };
    let VectorInternal::Dense(vector) = &named_query.query else {
        return Err(Status::invalid_argument(format!(
            "query template {query_slot} must contain one dense query vector"
        )));
    };
    if vector.is_empty() {
        return Err(Status::invalid_argument(format!(
            "query template {query_slot} has an empty dense query vector"
        )));
    }
    if let Some(dimension) = vector.iter().position(|value| !value.is_finite()) {
        return Err(Status::invalid_argument(format!(
            "query template {query_slot} has a non-finite dense query value at dimension {dimension}"
        )));
    }
    Ok(())
}

fn decode_compact_query_templates(
    collection_name: &str,
    query_templates: Vec<CoreSearchByShardQueryTemplate>,
) -> Result<DecodedCompactQueryTemplates, Status> {
    if query_templates.is_empty() {
        return Err(Status::invalid_argument("query_templates is empty"));
    }

    let mut requests = Vec::with_capacity(query_templates.len());
    let mut metadata = Vec::with_capacity(query_templates.len());
    for (query_slot, template) in query_templates.into_iter().enumerate() {
        let search_points = template.search_points.ok_or_else(|| {
            Status::invalid_argument(format!(
                "query template {query_slot} is missing search_points"
            ))
        })?;
        if search_points.collection_name != collection_name {
            return Err(Status::invalid_argument(format!(
                "query template {query_slot} collection_name does not match outer request"
            )));
        }

        let request = CoreSearchRequest::try_from(search_points)?;
        if request.limit == 0 {
            return Err(Status::invalid_argument(format!(
                "query template {query_slot} has zero limit"
            )));
        }
        let lower_limit = request.limit.checked_add(request.offset).ok_or_else(|| {
            Status::invalid_argument(format!(
                "query template {query_slot} limit + offset overflows usize"
            ))
        })?;
        if template.source_id_dedup_block_size == Some(0) {
            return Err(Status::invalid_argument(format!(
                "query template {query_slot} source_id_dedup_block_size must be positive when present"
            )));
        }
        validate_compact_orion_query_template(query_slot, &request)?;

        metadata.push(CompactSearchTemplate {
            lower_limit,
            final_limit: request.limit,
            final_offset: request.offset,
            source_id_dedup_block_size: template.source_id_dedup_block_size,
        });
        requests.push(request);
    }

    Ok(DecodedCompactQueryTemplates { requests, metadata })
}

fn materialize_compact_searches_by_shard(
    templates: &DecodedCompactQueryTemplates,
    searches: Vec<CoreSearchByShardCompactEntry>,
) -> Result<BTreeMap<ShardId, MaterializedCompactShardSearch>, Status> {
    if searches.is_empty() {
        return Err(Status::invalid_argument("compact searches is empty"));
    }
    if templates.requests.len() != templates.metadata.len() {
        return Err(Status::internal(
            "compact query template materialization metadata length mismatch",
        ));
    }

    let query_count = templates.len();
    let mut referenced_slots = vec![false; query_count];
    let mut seen_query_shards = HashSet::with_capacity(searches.len());
    let mut seen_entry_points = HashSet::new();
    let mut by_shard: BTreeMap<ShardId, MaterializedCompactShardSearch> = BTreeMap::new();

    for entry in searches {
        let query_index = usize::try_from(entry.query_slot)
            .map_err(|_| Status::invalid_argument("query_slot does not fit into usize"))?;
        let Some(template_request) = templates.requests.get(query_index) else {
            return Err(Status::invalid_argument(format!(
                "query_slot {query_index} is outside query_templates length {query_count}"
            )));
        };
        let template = templates.metadata.get(query_index).ok_or_else(|| {
            Status::internal("compact query template metadata is missing for a decoded request")
        })?;
        if !seen_query_shards.insert((query_index, entry.shard_id)) {
            return Err(Status::invalid_argument(format!(
                "duplicate compact search for query_slot {query_index} and shard {}",
                entry.shard_id
            )));
        }
        if entry.hnsw_entry_points.is_empty() {
            return Err(Status::invalid_argument(format!(
                "compact search for query_slot {query_index} and shard {} has no HNSW entry points",
                entry.shard_id
            )));
        }
        let hnsw_ef = usize::try_from(entry.hnsw_ef)
            .map_err(|_| Status::invalid_argument("hnsw_ef does not fit into usize"))?;
        if hnsw_ef == 0 {
            return Err(Status::invalid_argument("hnsw_ef must be positive"));
        }
        let hnsw_entry_points = entry
            .hnsw_entry_points
            .into_iter()
            .map(PointIdType::try_from)
            .collect::<Result<Vec<_>, _>>()?;
        seen_entry_points.clear();
        seen_entry_points.reserve(hnsw_entry_points.len());
        if let Some(duplicate) = hnsw_entry_points
            .iter()
            .find(|entry_point| !seen_entry_points.insert(**entry_point))
        {
            return Err(Status::invalid_argument(format!(
                "compact search for query_slot {query_index} and shard {} contains duplicate ordered HNSW entry point {duplicate}",
                entry.shard_id,
            )));
        }

        let mut request = template_request.clone();
        request.limit = template.lower_limit;
        request.offset = 0;
        request.hnsw_entry_points = Some(hnsw_entry_points);
        request.hnsw_entry_points_by_shard = None;
        request.hnsw_ef_by_shard = None;
        request.source_id_dedup_block_size = None;
        let mut params = request.params.unwrap_or_default();
        params.hnsw_ef = Some(hnsw_ef);
        request.params = Some(params);

        referenced_slots[query_index] = true;
        let shard_search = by_shard.entry(entry.shard_id).or_default();
        shard_search.original_rows.push((
            query_index,
            template.final_limit,
            template.final_offset,
            template.source_id_dedup_block_size,
        ));
        shard_search.requests.push(request);
    }

    if let Some(unreferenced_slot) = referenced_slots.iter().position(|referenced| !referenced) {
        return Err(Status::invalid_argument(format!(
            "query template slot {unreferenced_slot} has no compact shard search"
        )));
    }

    Ok(by_shard)
}

#[cfg(test)]
fn decode_compact_searches_by_shard(
    collection_name: &str,
    query_templates: Vec<CoreSearchByShardQueryTemplate>,
    searches: Vec<CoreSearchByShardCompactEntry>,
) -> Result<
    (
        usize,
        Vec<CoreSearchRequest>,
        BTreeMap<ShardId, MaterializedCompactShardSearch>,
    ),
    Status,
> {
    let templates = decode_compact_query_templates(collection_name, query_templates)?;
    let query_count = templates.len();
    let template_requests = templates.requests.clone();
    let by_shard = materialize_compact_searches_by_shard(&templates, searches)?;
    Ok((query_count, template_requests, by_shard))
}

#[allow(clippy::too_many_arguments)]
pub async fn core_search_batch_by_shard_compact(
    toc: &TableOfContent,
    collection_name: String,
    wire_version: u32,
    query_templates: Vec<CoreSearchByShardQueryTemplate>,
    searches: Vec<CoreSearchByShardCompactEntry>,
    auth: Auth,
    timeout: Option<Duration>,
    request_hw_counter: RequestHwCounter,
) -> Result<Response<SearchBatchResponse>, Status> {
    if wire_version != 1 {
        return Err(Status::invalid_argument(format!(
            "unsupported compact peer-search wire_version {wire_version}; expected 1"
        )));
    }
    let templates = decode_compact_query_templates(&collection_name, query_templates)?;
    let query_count = templates.len();
    let by_shard = materialize_compact_searches_by_shard(&templates, searches)?;
    let collection = toc
        .validate_orion_compact_peer_search(&collection_name, &templates.requests, &auth)
        .await
        .map_err(Status::from)?;

    let timing = Instant::now();
    let shard_searches = by_shard.into_iter().map(|(shard_id, shard_search)| {
        let MaterializedCompactShardSearch {
            original_rows,
            requests,
        } = shard_search;
        let request = CoreSearchRequestBatch { searches: requests };
        let collection = collection.clone();
        let request_hw_counter = request_hw_counter.clone();

        async move {
            let rows = collection
                .core_search_batch(
                    request,
                    None,
                    ShardSelectorInternal::ShardId(shard_id),
                    timeout,
                    request_hw_counter.get_counter(),
                )
                .await
                .map_err(StorageError::from)
                .map_err(Status::from)?;
            Ok::<_, Status>((shard_id, original_rows, rows))
        }
    });

    let shard_results = future::try_join_all(shard_searches).await?;
    let mut shard_rows = Vec::new();
    for (shard_id, original_rows, rows) in shard_results {
        shard_rows.extend(attach_core_search_by_shard_results(
            shard_id,
            original_rows,
            rows,
        )?);
    }

    let premerged_rows = premerge_core_search_by_shard_rows(shard_rows, query_count)?;
    let response = SearchBatchResponse {
        result: premerged_rows
            .into_iter()
            .map(|points| BatchResult {
                result: points.into_iter().map(|point| point.into()).collect(),
            })
            .collect(),
        time: timing.elapsed().as_secs_f64(),
        usage: Usage::from_hardware_usage(request_hw_counter.to_grpc_api()).into_non_empty(),
    };

    Ok(Response::new(response))
}

pub async fn search_groups(
    toc_provider: impl CheckedTocProvider,
    search_point_groups: SearchPointGroups,
    shard_selection: Option<ShardId>,
    auth: Auth,
    request_hw_counter: RequestHwCounter,
) -> Result<Response<SearchGroupsResponse>, Status> {
    let search_groups_request = search_point_groups.clone().try_into()?;

    let SearchPointGroups {
        collection_name,
        read_consistency,
        timeout,
        shard_key_selector,
        ..
    } = search_point_groups;

    let toc = toc_provider
        .check_strict_mode(
            &search_groups_request,
            &collection_name,
            timeout.map(|i| i as usize),
            &auth,
        )
        .await?;

    let read_consistency = ReadConsistency::try_from_optional(read_consistency)?;

    let shard_selector = convert_shard_selector_for_read(shard_selection, shard_key_selector)?;

    let timing = Instant::now();
    let groups_result = crate::common::query::do_search_point_groups(
        toc,
        &collection_name,
        search_groups_request,
        read_consistency,
        shard_selector,
        auth,
        timeout.map(Duration::from_secs),
        request_hw_counter.get_counter(),
    )
    .await?;

    let groups_result = GroupsResult::try_from(groups_result)
        .map_err(|e| Status::internal(format!("Failed to convert groups result: {e}")))?;

    let response = SearchGroupsResponse {
        result: Some(groups_result),
        time: timing.elapsed().as_secs_f64(),
        usage: Usage::from_hardware_usage(request_hw_counter.to_grpc_api()).into_non_empty(),
    };

    Ok(Response::new(response))
}

pub async fn recommend(
    toc_provider: impl CheckedTocProvider,
    recommend_points: RecommendPoints,
    auth: Auth,
    request_hw_counter: RequestHwCounter,
) -> Result<Response<RecommendResponse>, Status> {
    // extract a few fields from the request and convert to internal request
    let collection_name = recommend_points.collection_name.clone();
    let read_consistency = recommend_points.read_consistency.clone();
    let shard_key_selector = recommend_points.shard_key_selector.clone();
    let timeout = recommend_points.timeout;

    let request =
        collection::operations::types::RecommendRequestInternal::try_from(recommend_points)?;

    let toc = toc_provider
        .check_strict_mode(
            &request,
            &collection_name,
            timeout.map(|i| i as usize),
            &auth,
        )
        .await?;

    let read_consistency = ReadConsistency::try_from_optional(read_consistency)?;
    let shard_selector = convert_shard_selector_for_read(None, shard_key_selector)?;
    let timeout = timeout.map(Duration::from_secs);

    let timing = Instant::now();
    let recommended_points = toc
        .recommend(
            &collection_name,
            request,
            read_consistency,
            shard_selector,
            auth,
            timeout,
            request_hw_counter.get_counter(),
        )
        .await?;

    let response = RecommendResponse {
        result: recommended_points
            .into_iter()
            .map(|point| point.into())
            .collect(),
        time: timing.elapsed().as_secs_f64(),
        usage: Usage::from_hardware_usage(request_hw_counter.to_grpc_api()).into_non_empty(),
    };

    Ok(Response::new(response))
}

pub async fn recommend_batch(
    toc_provider: impl CheckedTocProvider,
    collection_name: &str,
    recommend_points: Vec<RecommendPoints>,
    read_consistency: Option<ReadConsistencyGrpc>,
    auth: Auth,
    timeout: Option<Duration>,
    request_hw_counter: RequestHwCounter,
) -> Result<Response<RecommendBatchResponse>, Status> {
    let mut requests = Vec::with_capacity(recommend_points.len());

    for mut request in recommend_points {
        let shard_selector =
            convert_shard_selector_for_read(None, request.shard_key_selector.take())?;
        let internal_request: collection::operations::types::RecommendRequestInternal =
            request.try_into()?;
        requests.push((internal_request, shard_selector));
    }

    let toc = toc_provider
        .check_strict_mode_batch(
            &requests,
            |i| &i.0,
            collection_name,
            timeout.map(|i| i.as_secs() as usize),
            &auth,
        )
        .await?;

    let read_consistency = ReadConsistency::try_from_optional(read_consistency)?;

    let timing = Instant::now();
    let scored_points = toc
        .recommend_batch(
            collection_name,
            requests,
            read_consistency,
            auth,
            timeout,
            request_hw_counter.get_counter(),
        )
        .await?;

    let response = RecommendBatchResponse {
        result: scored_points
            .into_iter()
            .map(|points| BatchResult {
                result: points.into_iter().map(|p| p.into()).collect(),
            })
            .collect(),
        time: timing.elapsed().as_secs_f64(),
        usage: Usage::from_hardware_usage(request_hw_counter.to_grpc_api()).into_non_empty(),
    };

    Ok(Response::new(response))
}

pub async fn recommend_groups(
    toc_provider: impl CheckedTocProvider,
    recommend_point_groups: RecommendPointGroups,
    auth: Auth,
    request_hw_counter: RequestHwCounter,
) -> Result<Response<RecommendGroupsResponse>, Status> {
    let recommend_groups_request = recommend_point_groups.clone().try_into()?;

    let RecommendPointGroups {
        collection_name,
        read_consistency,
        timeout,
        shard_key_selector,
        ..
    } = recommend_point_groups;

    let toc = toc_provider
        .check_strict_mode(
            &recommend_groups_request,
            &collection_name,
            timeout.map(|i| i as usize),
            &auth,
        )
        .await?;

    let read_consistency = ReadConsistency::try_from_optional(read_consistency)?;

    let shard_selector = convert_shard_selector_for_read(None, shard_key_selector)?;

    let timing = Instant::now();
    let groups_result = crate::common::query::do_recommend_point_groups(
        toc,
        &collection_name,
        recommend_groups_request,
        read_consistency,
        shard_selector,
        auth,
        timeout.map(Duration::from_secs),
        request_hw_counter.get_counter(),
    )
    .await?;

    let groups_result = GroupsResult::try_from(groups_result)
        .map_err(|e| Status::internal(format!("Failed to convert groups result: {e}")))?;

    let response = RecommendGroupsResponse {
        result: Some(groups_result),
        time: timing.elapsed().as_secs_f64(),
        usage: Usage::from_hardware_usage(request_hw_counter.to_grpc_api()).into_non_empty(),
    };

    Ok(Response::new(response))
}

pub async fn discover(
    toc_provider: impl CheckedTocProvider,
    discover_points: DiscoverPoints,
    auth: Auth,
    request_hw_counter: RequestHwCounter,
) -> Result<Response<DiscoverResponse>, Status> {
    let (request, collection_name, read_consistency, timeout, shard_key_selector) =
        try_discover_request_from_grpc(discover_points)?;

    let toc = toc_provider
        .check_strict_mode(
            &request,
            &collection_name,
            timeout.map(|i| i.as_secs() as usize),
            &auth,
        )
        .await?;

    let timing = Instant::now();

    let shard_selector = convert_shard_selector_for_read(None, shard_key_selector)?;

    let discovered_points = toc
        .discover(
            &collection_name,
            request,
            read_consistency,
            shard_selector,
            auth,
            timeout,
            request_hw_counter.get_counter(),
        )
        .await?;

    let response = DiscoverResponse {
        result: discovered_points
            .into_iter()
            .map(|point| point.into())
            .collect(),
        time: timing.elapsed().as_secs_f64(),
        usage: Usage::from_hardware_usage(request_hw_counter.to_grpc_api()).into_non_empty(),
    };

    Ok(Response::new(response))
}

pub async fn discover_batch(
    toc_provider: impl CheckedTocProvider,
    collection_name: &str,
    discover_points: Vec<DiscoverPoints>,
    read_consistency: Option<ReadConsistencyGrpc>,
    auth: Auth,
    timeout: Option<Duration>,
    request_hw_counter: RequestHwCounter,
) -> Result<Response<DiscoverBatchResponse>, Status> {
    let mut requests = Vec::with_capacity(discover_points.len());

    for discover_request in discover_points {
        let (internal_request, _collection_name, _consistency, _timeout, shard_key_selector) =
            try_discover_request_from_grpc(discover_request)?;
        let shard_selector = convert_shard_selector_for_read(None, shard_key_selector)?;
        requests.push((internal_request, shard_selector));
    }

    let read_consistency = ReadConsistency::try_from_optional(read_consistency)?;

    let toc = toc_provider
        .check_strict_mode_batch(
            &requests,
            |i| &i.0,
            collection_name,
            timeout.map(|i| i.as_secs() as usize),
            &auth,
        )
        .await?;

    let timing = Instant::now();
    let scored_points = toc
        .discover_batch(
            collection_name,
            requests,
            read_consistency,
            auth,
            timeout,
            request_hw_counter.get_counter(),
        )
        .await?;

    let response = DiscoverBatchResponse {
        result: scored_points
            .into_iter()
            .map(|points| BatchResult {
                result: points.into_iter().map(|p| p.into()).collect(),
            })
            .collect(),
        time: timing.elapsed().as_secs_f64(),
        usage: Usage::from_hardware_usage(request_hw_counter.to_grpc_api()).into_non_empty(),
    };

    Ok(Response::new(response))
}

pub async fn scroll(
    toc_provider: impl CheckedTocProvider,
    scroll_points: ScrollPoints,
    shard_selection: Option<ShardId>,
    auth: Auth,
    request_hw_counter: RequestHwCounter,
) -> Result<Response<ScrollResponse>, Status> {
    let ScrollPoints {
        collection_name,
        filter,
        offset,
        limit,
        with_payload,
        with_vectors,
        read_consistency,
        shard_key_selector,
        order_by,
        timeout,
    } = scroll_points;

    let scroll_request = ScrollRequestInternal {
        offset: offset.map(|o| o.try_into()).transpose()?,
        limit: limit.map(|l| l as usize),
        filter: filter.map(|f| f.try_into()).transpose()?,
        with_payload: with_payload.map(|wp| wp.try_into()).transpose()?,
        with_vector: with_vectors
            .map(|selector| selector.into())
            .unwrap_or_default(),
        order_by: order_by
            .map(OrderBy::try_from)
            .transpose()?
            .map(OrderByInterface::Struct),
    };

    let toc = toc_provider
        .check_strict_mode(
            &scroll_request,
            &collection_name,
            timeout.map(|i| i as usize),
            &auth,
        )
        .await?;

    let timeout = timeout.map(Duration::from_secs);
    let read_consistency = ReadConsistency::try_from_optional(read_consistency)?;

    let shard_selector = convert_shard_selector_for_read(shard_selection, shard_key_selector)?;

    let timing = Instant::now();
    let scrolled_points = do_scroll_points(
        toc,
        &collection_name,
        scroll_request,
        read_consistency,
        timeout,
        shard_selector,
        auth,
        request_hw_counter.get_counter(),
    )
    .await?;

    let points: Result<_, _> = scrolled_points
        .points
        .into_iter()
        .map(api::grpc::qdrant::RetrievedPoint::try_from)
        .collect();

    let points = points.map_err(|e| Status::internal(format!("Failed to convert points: {e}")))?;

    let response = ScrollResponse {
        next_page_offset: scrolled_points.next_page_offset.map(|n| n.into()),
        result: points,
        time: timing.elapsed().as_secs_f64(),
        usage: Usage::from_hardware_usage(request_hw_counter.to_grpc_api()).into_non_empty(),
    };

    Ok(Response::new(response))
}

pub async fn count(
    toc_provider: impl CheckedTocProvider,
    count_points: CountPoints,
    shard_selection: Option<ShardId>,
    auth: Auth,
    request_hw_counter: RequestHwCounter,
) -> Result<Response<CountResponse>, Status> {
    let CountPoints {
        collection_name,
        filter,
        exact,
        read_consistency,
        shard_key_selector,
        timeout,
    } = count_points;

    let count_request = CountRequestInternal {
        filter: filter.map(|f| f.try_into()).transpose()?,
        exact: exact.unwrap_or_else(CountRequestInternal::default_exact),
    };

    let toc = toc_provider
        .check_strict_mode(
            &count_request,
            &collection_name,
            timeout.map(|i| i as usize),
            &auth,
        )
        .await?;

    let timeout = timeout.map(Duration::from_secs);
    let read_consistency = ReadConsistency::try_from_optional(read_consistency)?;

    let shard_selector = convert_shard_selector_for_read(shard_selection, shard_key_selector)?;

    let timing = Instant::now();

    let count_result = do_count_points(
        toc,
        &collection_name,
        count_request,
        read_consistency,
        timeout,
        shard_selector,
        auth,
        request_hw_counter.get_counter(),
    )
    .await?;

    let response = CountResponse {
        result: Some(count_result.into()),
        time: timing.elapsed().as_secs_f64(),
        usage: Usage::from_hardware_usage(request_hw_counter.to_grpc_api()).into_non_empty(),
    };

    Ok(Response::new(response))
}

pub async fn get(
    toc_provider: impl CheckedTocProvider,
    get_points: GetPoints,
    shard_selection: Option<ShardId>,
    auth: Auth,
    request_hw_counter: RequestHwCounter,
) -> Result<Response<GetResponse>, Status> {
    let GetPoints {
        collection_name,
        ids,
        with_payload,
        with_vectors,
        read_consistency,
        shard_key_selector,
        timeout,
    } = get_points;

    let point_request = PointRequestInternal {
        ids: ids
            .into_iter()
            .map(|p| p.try_into())
            .collect::<Result<_, _>>()?,
        with_payload: with_payload.map(|wp| wp.try_into()).transpose()?,
        with_vector: with_vectors
            .map(|selector| selector.into())
            .unwrap_or_default(),
    };
    let read_consistency = ReadConsistency::try_from_optional(read_consistency)?;

    let shard_selector = convert_shard_selector_for_read(shard_selection, shard_key_selector)?;

    let timing = Instant::now();

    let toc = toc_provider
        .check_strict_mode(
            &point_request,
            &collection_name,
            timeout.map(|i| i as usize),
            &auth,
        )
        .await?;

    let timeout = timeout.map(Duration::from_secs);

    let records = do_get_points(
        toc,
        &collection_name,
        point_request,
        read_consistency,
        timeout,
        shard_selector,
        auth,
        request_hw_counter.get_counter(),
    )
    .await?;

    let response = GetResponse {
        result: records.into_iter().map(|point| point.into()).collect(),
        time: timing.elapsed().as_secs_f64(),
        usage: Usage::from_hardware_usage(request_hw_counter.to_grpc_api()).into_non_empty(),
    };

    Ok(Response::new(response))
}

pub async fn query(
    toc_provider: impl CheckedTocProvider,
    query_points: QueryPoints,
    shard_selection: Option<ShardId>,
    auth: Auth,
    request_hw_counter: RequestHwCounter,
    inference_params: InferenceParams,
) -> Result<Response<QueryResponse>, Status> {
    let shard_key_selector = query_points.shard_key_selector.clone();
    let shard_selector = convert_shard_selector_for_read(shard_selection, shard_key_selector)?;
    let read_consistency = query_points
        .read_consistency
        .clone()
        .map(TryFrom::try_from)
        .transpose()?;
    let collection_name = query_points.collection_name.clone();
    let timeout = query_points.timeout;
    let (request, inference_usage) =
        convert_query_points_from_grpc(query_points, inference_params).await?;

    let toc = toc_provider
        .check_strict_mode(
            &request,
            &collection_name,
            timeout.map(|i| i as usize),
            &auth,
        )
        .await?;

    let timeout = timeout.map(Duration::from_secs);

    let timing = Instant::now();
    let scored_points = do_query_points(
        toc,
        &collection_name,
        request,
        read_consistency,
        shard_selector,
        auth,
        timeout,
        request_hw_counter.get_counter(),
    )
    .await?;

    let response = QueryResponse {
        result: scored_points
            .into_iter()
            .map(|point| point.into())
            .collect(),
        time: timing.elapsed().as_secs_f64(),
        usage: Usage::new(request_hw_counter.to_grpc_api(), Some(inference_usage)).into_non_empty(),
    };

    Ok(Response::new(response))
}

#[allow(clippy::too_many_arguments)]
pub async fn query_batch(
    toc_provider: impl CheckedTocProvider,
    collection_name: &str,
    points: Vec<QueryPoints>,
    read_consistency: Option<ReadConsistencyGrpc>,
    auth: Auth,
    timeout: Option<Duration>,
    request_hw_counter: RequestHwCounter,
    inference_params: InferenceParams,
) -> Result<Response<QueryBatchResponse>, Status> {
    let read_consistency = ReadConsistency::try_from_optional(read_consistency)?;
    let mut requests = Vec::with_capacity(points.len());
    let mut total_inference_usage = InferenceUsage::default();

    for query_points in points {
        let shard_key_selector = query_points.shard_key_selector.clone();
        let shard_selector = convert_shard_selector_for_read(None, shard_key_selector)?;
        let (request, usage) =
            convert_query_points_from_grpc(query_points, inference_params.clone()).await?;
        total_inference_usage.merge(usage);
        requests.push((request, shard_selector));
    }

    let toc = toc_provider
        .check_strict_mode_batch(
            &requests,
            |i| &i.0,
            collection_name,
            timeout.map(|i| i.as_secs() as usize),
            &auth,
        )
        .await?;

    let timing = Instant::now();
    let scored_points = do_query_batch_points(
        toc,
        collection_name,
        requests,
        read_consistency,
        auth,
        timeout,
        request_hw_counter.get_counter(),
    )
    .await?;

    let response = QueryBatchResponse {
        result: scored_points
            .into_iter()
            .map(|points| BatchResult {
                result: points.into_iter().map(|p| p.into()).collect(),
            })
            .collect(),
        time: timing.elapsed().as_secs_f64(),
        usage: Usage::new(
            request_hw_counter.to_grpc_api(),
            total_inference_usage.into_non_empty(),
        )
        .into_non_empty(),
    };

    Ok(Response::new(response))
}

pub async fn query_groups(
    toc_provider: impl CheckedTocProvider,
    query_points: QueryPointGroups,
    shard_selection: Option<ShardId>,
    auth: Auth,
    request_hw_counter: RequestHwCounter,
    inference_params: InferenceParams,
) -> Result<Response<QueryGroupsResponse>, Status> {
    let shard_key_selector = query_points.shard_key_selector.clone();
    let shard_selector = convert_shard_selector_for_read(shard_selection, shard_key_selector)?;
    let read_consistency = query_points
        .read_consistency
        .clone()
        .map(TryFrom::try_from)
        .transpose()?;
    let timeout = query_points.timeout;
    let collection_name = query_points.collection_name.clone();
    let (request, inference_usage) =
        convert_query_point_groups_from_grpc(query_points, inference_params).await?;

    let toc = toc_provider
        .check_strict_mode(
            &request,
            &collection_name,
            timeout.map(|i| i as usize),
            &auth,
        )
        .await?;

    let timeout = timeout.map(Duration::from_secs);
    let timing = Instant::now();

    let groups_result = do_query_point_groups(
        toc,
        &collection_name,
        request,
        read_consistency,
        shard_selector,
        auth,
        timeout,
        request_hw_counter.get_counter(),
    )
    .await?;

    let grpc_group_result = GroupsResult::try_from(groups_result)
        .map_err(|err| Status::internal(format!("failed to convert result: {err}")))?;

    let response = QueryGroupsResponse {
        result: Some(grpc_group_result),
        time: timing.elapsed().as_secs_f64(),
        usage: Usage::new(request_hw_counter.to_grpc_api(), Some(inference_usage)).into_non_empty(),
    };

    Ok(Response::new(response))
}

pub async fn facet(
    toc_provider: impl CheckedTocProvider,
    facet_counts: FacetCounts,
    auth: Auth,
    request_hw_counter: RequestHwCounter,
) -> Result<Response<FacetResponse>, Status> {
    let FacetCounts {
        collection_name,
        key,
        filter,
        exact,
        limit,
        read_consistency,
        shard_key_selector,
        timeout,
    } = facet_counts;

    let facet_request = FacetParams {
        key: json_path_from_proto(&key)?,
        filter: filter.map(TryInto::try_into).transpose()?,
        limit: limit
            .map(usize::try_from)
            .transpose()
            .map_err(|_| Status::invalid_argument("could not parse limit param into usize"))?
            .unwrap_or(FacetParams::DEFAULT_LIMIT),
        exact: exact.unwrap_or(FacetParams::DEFAULT_EXACT),
    };

    let toc = toc_provider
        .check_strict_mode(
            &facet_request,
            &collection_name,
            timeout.map(|i| i as usize),
            &auth,
        )
        .await?;

    let timeout = timeout.map(Duration::from_secs);
    let read_consistency = ReadConsistency::try_from_optional(read_consistency)?;

    let shard_selector = convert_shard_selector_for_read(None, shard_key_selector)?;

    let timing = Instant::now();
    let facet_response = toc
        .facet(
            &collection_name,
            facet_request,
            shard_selector,
            read_consistency,
            auth,
            timeout,
            request_hw_counter.get_counter(),
        )
        .await?;

    let segment::data_types::facets::FacetResponse { hits } = facet_response;

    let response = FacetResponse {
        hits: hits.into_iter().map(From::from).collect(),
        time: timing.elapsed().as_secs_f64(),
        usage: Usage::from_hardware_usage(request_hw_counter.to_grpc_api()).into_non_empty(),
    };

    Ok(Response::new(response))
}

pub async fn search_points_matrix(
    toc_provider: impl CheckedTocProvider,
    search_matrix_points: SearchMatrixPoints,
    auth: Auth,
    hw_measurement_acc: HwMeasurementAcc,
) -> Result<CollectionSearchMatrixResponse, Status> {
    let SearchMatrixPoints {
        collection_name,
        filter,
        sample,
        limit,
        using,
        read_consistency,
        shard_key_selector,
        timeout,
    } = search_matrix_points;

    let search_matrix_request = CollectionSearchMatrixRequest {
        filter: filter.map(TryInto::try_into).transpose()?,
        sample_size: sample
            .map(usize::try_from)
            .transpose()
            .map_err(|_| Status::invalid_argument("could not parse 'sample' param into usize"))?
            .unwrap_or(CollectionSearchMatrixRequest::DEFAULT_SAMPLE),
        limit_per_sample: limit
            .map(usize::try_from)
            .transpose()
            .map_err(|_| Status::invalid_argument("could not parse 'limit' param into usize"))?
            .unwrap_or(CollectionSearchMatrixRequest::DEFAULT_LIMIT_PER_SAMPLE),
        using: using.unwrap_or_else(|| DEFAULT_VECTOR_NAME.to_owned()),
    };

    let toc = toc_provider
        .check_strict_mode(
            &search_matrix_request,
            &collection_name,
            timeout.map(|i| i as usize),
            &auth,
        )
        .await?;

    let timeout = timeout.map(Duration::from_secs);
    let read_consistency = ReadConsistency::try_from_optional(read_consistency)?;

    let shard_selector = convert_shard_selector_for_read(None, shard_key_selector)?;

    let search_matrix_response = toc
        .search_points_matrix(
            &collection_name,
            search_matrix_request,
            read_consistency,
            shard_selector,
            auth,
            timeout,
            hw_measurement_acc,
        )
        .await?;

    Ok(search_matrix_response)
}

#[cfg(test)]
mod tests {
    use prost::Message as _;
    use segment::data_types::vectors::MultiDenseVectorInternal;
    use segment::types::{ExtendedPointId, Filter, ScoredPoint, SearchParams};
    use segment::vector_storage::query::RecoQuery;

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

    fn compact_template(
        collection_name: &str,
        limit: u64,
        offset: u64,
        source_id_dedup_block_size: Option<u64>,
    ) -> CoreSearchByShardQueryTemplate {
        CoreSearchByShardQueryTemplate {
            search_points: Some(CoreSearchPoints {
                collection_name: collection_name.to_string(),
                query: Some(api::grpc::qdrant::QueryEnum {
                    query: Some(api::grpc::qdrant::query_enum::Query::NearestNeighbors(
                        VectorInternal::Dense(vec![1.0, 2.0, 3.0, 4.0]).into(),
                    )),
                }),
                filter: None,
                limit,
                with_payload: None,
                params: None,
                score_threshold: None,
                offset: Some(offset),
                vector_name: None,
                with_vectors: None,
                read_consistency: None,
                hnsw_entry_points: Vec::new(),
            }),
            source_id_dedup_block_size,
        }
    }

    fn compact_entry(
        query_slot: u64,
        shard_id: ShardId,
        entry_points: &[u64],
        hnsw_ef: u64,
    ) -> CoreSearchByShardCompactEntry {
        CoreSearchByShardCompactEntry {
            query_slot,
            shard_id,
            hnsw_entry_points: entry_points
                .iter()
                .copied()
                .map(PointIdType::from)
                .map(Into::into)
                .collect(),
            hnsw_ef,
        }
    }

    fn dense_query(request: &CoreSearchRequest) -> &[f32] {
        let QueryEnum::Nearest(named_query) = &request.query else {
            panic!("test compact request must be nearest-neighbor");
        };
        let VectorInternal::Dense(vector) = &named_query.query else {
            panic!("test compact request must carry a dense vector");
        };
        vector
    }

    fn dense_query_ptr(request: &CoreSearchRequest) -> *const f32 {
        dense_query(request).as_ptr()
    }

    #[test]
    fn compact_decoder_preserves_ordered_multiep_dynamic_ef_and_merge_window() {
        let (query_count, template_requests, by_shard) = decode_compact_searches_by_shard(
            "orion",
            vec![compact_template("orion", 10, 7, Some(1_000_001))],
            vec![
                compact_entry(0, 9, &[31, 29, 11], 76),
                compact_entry(0, 10, &[5, 3], 88),
            ],
        )
        .unwrap();

        assert_eq!(query_count, 1);
        assert_eq!(template_requests.len(), 1);
        assert_eq!(by_shard.len(), 2);
        let shard_search = &by_shard[&9];
        assert_eq!(
            shard_search.original_rows,
            vec![(0, 10, 7, Some(1_000_001))]
        );
        let request = &shard_search.requests[0];
        assert_eq!(request.limit, 17);
        assert_eq!(request.offset, 0);
        assert_eq!(
            request.hnsw_entry_points,
            Some(vec![
                PointIdType::from(31),
                PointIdType::from(29),
                PointIdType::from(11),
            ])
        );
        assert_eq!(
            request.params.as_ref().and_then(|params| params.hnsw_ef),
            Some(76)
        );
        assert!(request.source_id_dedup_block_size.is_none());
    }

    #[test]
    fn compact_materialization_clones_independent_requests_per_entry() {
        let mut first_template = compact_template("orion", 10, 7, Some(1_000_001));
        first_template
            .search_points
            .as_mut()
            .unwrap()
            .score_threshold = Some(0.42);
        let mut second_template = compact_template("orion", 8, 2, None);
        second_template.search_points.as_mut().unwrap().query =
            Some(api::grpc::qdrant::QueryEnum {
                query: Some(api::grpc::qdrant::query_enum::Query::NearestNeighbors(
                    VectorInternal::Dense(vec![5.0, 6.0, 7.0, 8.0]).into(),
                )),
            });
        let templates =
            decode_compact_query_templates("orion", vec![first_template, second_template]).unwrap();
        let original_query_ptrs = templates
            .requests
            .iter()
            .map(dense_query_ptr)
            .collect::<Vec<_>>();
        let query_count = templates.len();
        let by_shard = materialize_compact_searches_by_shard(
            &templates,
            vec![
                compact_entry(0, 10, &[5, 3], 88),
                compact_entry(1, 9, &[17], 64),
                compact_entry(0, 9, &[31, 29, 11], 76),
                compact_entry(1, 10, &[23, 19], 72),
            ],
        )
        .unwrap();

        assert_eq!(
            by_shard[&9].original_rows,
            vec![(1, 8, 2, None), (0, 10, 7, Some(1_000_001))]
        );
        assert_eq!(
            dense_query(&by_shard[&9].requests[0]),
            &[5.0, 6.0, 7.0, 8.0]
        );
        assert_eq!(by_shard[&9].requests[0].limit, 10);
        assert_eq!(by_shard[&9].requests[0].offset, 0);
        assert_eq!(
            by_shard[&9].requests[0].hnsw_entry_points,
            Some(vec![PointIdType::from(17)])
        );
        assert_eq!(
            by_shard[&9].requests[0]
                .params
                .as_ref()
                .and_then(|params| params.hnsw_ef),
            Some(64)
        );
        assert_eq!(
            dense_query(&by_shard[&9].requests[1]),
            &[1.0, 2.0, 3.0, 4.0]
        );
        assert_eq!(by_shard[&9].requests[1].limit, 17);
        assert_eq!(by_shard[&9].requests[1].offset, 0);
        assert_eq!(
            by_shard[&9].requests[1].hnsw_entry_points,
            Some(vec![
                PointIdType::from(31),
                PointIdType::from(29),
                PointIdType::from(11),
            ])
        );
        assert_eq!(
            by_shard[&9].requests[1]
                .params
                .as_ref()
                .and_then(|params| params.hnsw_ef),
            Some(76)
        );
        assert_eq!(by_shard[&9].requests[1].score_threshold, Some(0.42));

        assert_eq!(
            by_shard[&10].original_rows,
            vec![(0, 10, 7, Some(1_000_001)), (1, 8, 2, None)]
        );
        assert_eq!(
            dense_query(&by_shard[&10].requests[0]),
            &[1.0, 2.0, 3.0, 4.0]
        );
        assert_eq!(by_shard[&10].requests[0].limit, 17);
        assert_eq!(by_shard[&10].requests[0].offset, 0);
        assert_eq!(
            by_shard[&10].requests[0].hnsw_entry_points,
            Some(vec![PointIdType::from(5), PointIdType::from(3)])
        );
        assert_eq!(
            by_shard[&10].requests[0]
                .params
                .as_ref()
                .and_then(|params| params.hnsw_ef),
            Some(88)
        );
        assert_eq!(by_shard[&10].requests[0].score_threshold, Some(0.42));
        assert_eq!(
            dense_query(&by_shard[&10].requests[1]),
            &[5.0, 6.0, 7.0, 8.0]
        );
        assert_eq!(by_shard[&10].requests[1].limit, 10);
        assert_eq!(by_shard[&10].requests[1].offset, 0);
        assert_eq!(
            by_shard[&10].requests[1].hnsw_entry_points,
            Some(vec![PointIdType::from(23), PointIdType::from(19)])
        );
        assert_eq!(
            by_shard[&10].requests[1]
                .params
                .as_ref()
                .and_then(|params| params.hnsw_ef),
            Some(72)
        );

        let mut request_counts = vec![0usize; query_count];
        let mut independent_allocation_counts = vec![0usize; query_count];
        for shard_search in by_shard.values() {
            assert_eq!(
                shard_search.original_rows.len(),
                shard_search.requests.len()
            );
            for (original_row, request) in shard_search
                .original_rows
                .iter()
                .zip(&shard_search.requests)
            {
                let query_index = original_row.0;
                request_counts[query_index] += 1;
                assert!(request.source_id_dedup_block_size.is_none());
                if dense_query_ptr(request) != original_query_ptrs[query_index] {
                    independent_allocation_counts[query_index] += 1;
                }
            }
        }

        assert_eq!(request_counts, vec![2, 2]);
        assert_eq!(independent_allocation_counts, vec![2, 2]);
    }

    #[test]
    fn compact_decoder_rejects_bad_query_slot_contracts() {
        let empty_templates = decode_compact_searches_by_shard(
            "orion",
            Vec::new(),
            vec![compact_entry(0, 9, &[31], 76)],
        )
        .unwrap_err();
        assert!(
            empty_templates
                .message()
                .contains("query_templates is empty")
        );

        let empty_searches = decode_compact_searches_by_shard(
            "orion",
            vec![compact_template("orion", 10, 0, None)],
            Vec::new(),
        )
        .unwrap_err();
        assert!(
            empty_searches
                .message()
                .contains("compact searches is empty")
        );

        let out_of_range = decode_compact_searches_by_shard(
            "orion",
            vec![compact_template("orion", 10, 0, None)],
            vec![compact_entry(1, 9, &[31], 76)],
        )
        .unwrap_err();
        assert_eq!(out_of_range.code(), tonic::Code::InvalidArgument);
        assert!(out_of_range.message().contains("outside query_templates"));

        let duplicate = decode_compact_searches_by_shard(
            "orion",
            vec![compact_template("orion", 10, 0, None)],
            vec![
                compact_entry(0, 9, &[31], 76),
                compact_entry(0, 9, &[29], 88),
            ],
        )
        .unwrap_err();
        assert_eq!(duplicate.code(), tonic::Code::InvalidArgument);
        assert!(duplicate.message().contains("duplicate compact search"));

        let unreferenced = decode_compact_searches_by_shard(
            "orion",
            vec![
                compact_template("orion", 10, 0, None),
                compact_template("orion", 10, 0, None),
            ],
            vec![compact_entry(0, 9, &[31], 76)],
        )
        .unwrap_err();
        assert_eq!(unreferenced.code(), tonic::Code::InvalidArgument);
        assert!(unreferenced.message().contains("slot 1 has no compact"));
    }

    #[test]
    fn compact_decoder_rejects_missing_template_and_shard_overrides() {
        let missing_template = decode_compact_searches_by_shard(
            "orion",
            vec![CoreSearchByShardQueryTemplate {
                search_points: None,
                source_id_dedup_block_size: None,
            }],
            vec![compact_entry(0, 9, &[31], 76)],
        )
        .unwrap_err();
        assert!(missing_template.message().contains("missing search_points"));

        let missing_eps = decode_compact_searches_by_shard(
            "orion",
            vec![compact_template("orion", 10, 0, None)],
            vec![compact_entry(0, 9, &[], 76)],
        )
        .unwrap_err();
        assert!(missing_eps.message().contains("no HNSW entry points"));

        let zero_ef = decode_compact_searches_by_shard(
            "orion",
            vec![compact_template("orion", 10, 0, None)],
            vec![compact_entry(0, 9, &[31], 0)],
        )
        .unwrap_err();
        assert!(zero_ef.message().contains("hnsw_ef must be positive"));

        let collection_mismatch = decode_compact_searches_by_shard(
            "orion",
            vec![compact_template("other", 10, 0, None)],
            vec![compact_entry(0, 9, &[31], 76)],
        )
        .unwrap_err();
        assert!(
            collection_mismatch
                .message()
                .contains("collection_name does not match")
        );

        let mut template_with_entry_points = compact_template("orion", 10, 0, None);
        template_with_entry_points
            .search_points
            .as_mut()
            .unwrap()
            .hnsw_entry_points = vec![PointIdType::from(7).into()];
        let ambiguous_template = decode_compact_searches_by_shard(
            "orion",
            vec![template_with_entry_points],
            vec![compact_entry(0, 9, &[31], 76)],
        )
        .unwrap_err();
        assert!(
            ambiguous_template
                .message()
                .contains("must not contain shard-specific")
        );

        let duplicate_entry_point = decode_compact_searches_by_shard(
            "orion",
            vec![compact_template("orion", 10, 0, None)],
            vec![compact_entry(0, 9, &[31, 29, 31], 76)],
        )
        .unwrap_err();
        assert!(
            duplicate_entry_point
                .message()
                .contains("duplicate ordered HNSW entry point")
        );

        let zero_dedup = decode_compact_searches_by_shard(
            "orion",
            vec![compact_template("orion", 10, 0, Some(0))],
            vec![compact_entry(0, 9, &[31], 76)],
        )
        .unwrap_err();
        assert!(
            zero_dedup
                .message()
                .contains("source_id_dedup_block_size must be positive")
        );

        let zero_limit = decode_compact_searches_by_shard(
            "orion",
            vec![compact_template("orion", 0, 0, None)],
            vec![compact_entry(0, 9, &[31], 76)],
        )
        .unwrap_err();
        assert!(zero_limit.message().contains("zero limit"));

        let overflow = decode_compact_searches_by_shard(
            "orion",
            vec![compact_template("orion", u64::MAX, 1, None)],
            vec![compact_entry(0, 9, &[31], 76)],
        )
        .unwrap_err();
        assert!(overflow.message().contains("limit + offset overflows"));
    }

    #[test]
    fn compact_entry_point_duplicate_scratch_is_scoped_to_each_search() {
        let templates =
            decode_compact_query_templates("orion", vec![compact_template("orion", 10, 0, None)])
                .unwrap();
        let by_shard = materialize_compact_searches_by_shard(
            &templates,
            vec![
                compact_entry(0, 9, &[31, 29, 11], 76),
                compact_entry(0, 10, &[31, 17, 5], 64),
            ],
        )
        .unwrap();

        assert_eq!(by_shard.len(), 2);
        assert_eq!(
            by_shard[&9].requests[0].hnsw_entry_points.as_ref().unwrap()[0],
            PointIdType::from(31)
        );
        assert_eq!(
            by_shard[&10].requests[0]
                .hnsw_entry_points
                .as_ref()
                .unwrap()[0],
            PointIdType::from(31)
        );
    }

    #[test]
    fn compact_template_validator_rejects_non_orion_query_shapes() {
        let base = CoreSearchRequest::try_from(
            compact_template("orion", 10, 0, None)
                .search_points
                .unwrap(),
        )
        .unwrap();

        let mut filtered = base.clone();
        filtered.filter = Some(Filter::new());
        assert!(
            validate_compact_orion_query_template(0, &filtered)
                .unwrap_err()
                .message()
                .contains("must not contain a filter")
        );

        let mut exact = base.clone();
        exact.params = Some(SearchParams {
            exact: true,
            ..Default::default()
        });
        assert!(
            validate_compact_orion_query_template(0, &exact)
                .unwrap_err()
                .message()
                .contains("must not request exact")
        );

        let mut multi_dense = base.clone();
        multi_dense.query = QueryEnum::Nearest(NamedQuery::new(
            VectorInternal::MultiDense(MultiDenseVectorInternal::new(vec![1.0, 2.0], 1)),
            "multi",
        ));
        assert!(
            validate_compact_orion_query_template(0, &multi_dense)
                .unwrap_err()
                .message()
                .contains("one dense query vector")
        );

        let mut empty = base.clone();
        empty.query = QueryEnum::Nearest(NamedQuery::new(VectorInternal::Dense(Vec::new()), ""));
        assert!(
            validate_compact_orion_query_template(0, &empty)
                .unwrap_err()
                .message()
                .contains("empty dense")
        );

        let mut non_finite = base.clone();
        non_finite.query =
            QueryEnum::Nearest(NamedQuery::new(VectorInternal::Dense(vec![f32::NAN]), ""));
        assert!(
            validate_compact_orion_query_template(0, &non_finite)
                .unwrap_err()
                .message()
                .contains("non-finite")
        );

        let mut recommend = base;
        recommend.query = QueryEnum::RecommendBestScore(NamedQuery::new(
            RecoQuery::new(
                vec![VectorInternal::Dense(vec![1.0, 2.0, 3.0, 4.0])],
                Vec::new(),
            ),
            "",
        ));
        assert!(
            validate_compact_orion_query_template(0, &recommend)
                .unwrap_err()
                .message()
                .contains("nearest-neighbor")
        );
    }

    #[test]
    fn compact_wire_eliminates_repeated_200d_query_payload_per_shard() {
        let mut template = compact_template("orion", 10, 0, None);
        template.search_points.as_mut().unwrap().query = Some(api::grpc::qdrant::QueryEnum {
            query: Some(api::grpc::qdrant::query_enum::Query::NearestNeighbors(
                VectorInternal::Dense((0..200).map(|value| value as f32 / 200.0).collect()).into(),
            )),
        });

        let legacy_searches = (0..16)
            .map(|shard_id| {
                let mut search_points = template.search_points.clone().unwrap();
                search_points.hnsw_entry_points = [31, 29, 11, 7]
                    .into_iter()
                    .map(PointIdType::from)
                    .map(Into::into)
                    .collect();
                let mut params = search_points.params.unwrap_or_default();
                params.hnsw_ef = Some(76);
                search_points.params = Some(params);
                CoreSearchByShardEntry {
                    query_index: 0,
                    shard_id,
                    search_points: Some(search_points),
                    final_limit: 10,
                    final_offset: Some(0),
                    source_id_dedup_block_size: None,
                }
            })
            .collect();
        let legacy = api::grpc::qdrant::CoreSearchBatchByShardInternal {
            collection_name: "orion".to_string(),
            searches: legacy_searches,
            timeout: None,
        };
        let compact = api::grpc::qdrant::CoreSearchBatchByShardCompactInternal {
            collection_name: "orion".to_string(),
            query_templates: vec![template],
            searches: (0..16)
                .map(|shard_id| compact_entry(0, shard_id, &[31, 29, 11, 7], 76))
                .collect(),
            timeout: None,
            wire_version: 1,
        };

        assert_eq!(compact.query_templates.len(), 1);
        assert!(
            compact.encoded_len() * 4 < legacy.encoded_len(),
            "compact={} legacy={}",
            compact.encoded_len(),
            legacy.encoded_len(),
        );
    }

    #[test]
    fn core_search_by_shard_premerge_keeps_limit_plus_offset_per_peer_query() {
        let premerged = premerge_core_search_by_shard_rows(
            vec![
                CoreSearchByShardRow {
                    query_index: 0,
                    limit: 2,
                    offset: 1,
                    source_id_dedup_block_size: Some(100),
                    points: vec![scored(207, 0.99), scored(11, 0.90)],
                },
                CoreSearchByShardRow {
                    query_index: 0,
                    limit: 2,
                    offset: 1,
                    source_id_dedup_block_size: Some(100),
                    points: vec![scored(7, 0.98), scored(12, 0.89)],
                },
                CoreSearchByShardRow {
                    query_index: 1,
                    limit: 1,
                    offset: 0,
                    source_id_dedup_block_size: None,
                    points: vec![scored(30, 0.70), scored(31, 0.60)],
                },
            ],
            2,
        )
        .unwrap();

        assert_eq!(
            premerged[0]
                .iter()
                .map(|point| point.id)
                .collect::<Vec<_>>(),
            vec![
                ExtendedPointId::NumId(207),
                ExtendedPointId::NumId(11),
                ExtendedPointId::NumId(12),
            ],
        );
        assert_eq!(
            premerged[1]
                .iter()
                .map(|point| point.id)
                .collect::<Vec<_>>(),
            vec![ExtendedPointId::NumId(30)],
        );
    }

    #[test]
    fn core_search_by_shard_premerge_rejects_invalid_row_contracts() {
        let out_of_range = premerge_core_search_by_shard_rows(
            vec![CoreSearchByShardRow {
                query_index: 1,
                limit: 1,
                offset: 0,
                source_id_dedup_block_size: None,
                points: vec![scored(1, 1.0)],
            }],
            1,
        )
        .unwrap_err();
        assert!(out_of_range.message().contains("outside query_count"));

        let missing_slot = premerge_core_search_by_shard_rows(
            vec![CoreSearchByShardRow {
                query_index: 0,
                limit: 1,
                offset: 0,
                source_id_dedup_block_size: None,
                points: vec![scored(1, 1.0)],
            }],
            2,
        )
        .unwrap_err();
        assert!(missing_slot.message().contains("no rows for query_index 1"));

        let metadata_conflict = premerge_core_search_by_shard_rows(
            vec![
                CoreSearchByShardRow {
                    query_index: 0,
                    limit: 1,
                    offset: 0,
                    source_id_dedup_block_size: None,
                    points: vec![scored(1, 1.0)],
                },
                CoreSearchByShardRow {
                    query_index: 0,
                    limit: 2,
                    offset: 0,
                    source_id_dedup_block_size: None,
                    points: vec![scored(2, 0.9)],
                },
            ],
            1,
        )
        .unwrap_err();
        assert!(metadata_conflict.message().contains("merge metadata"));

        let overflow = premerge_core_search_by_shard_rows(
            vec![CoreSearchByShardRow {
                query_index: 0,
                limit: usize::MAX,
                offset: 1,
                source_id_dedup_block_size: None,
                points: vec![scored(1, 1.0)],
            }],
            1,
        )
        .unwrap_err();
        assert!(overflow.message().contains("limit + offset overflows"));
    }

    #[test]
    fn core_search_by_shard_rejects_missing_or_extra_shard_rows() {
        let specs = vec![(0, 10, 0, None), (1, 10, 0, None)];

        let missing =
            attach_core_search_by_shard_results(7, specs.clone(), vec![vec![scored(1, 1.0)]])
                .err()
                .unwrap();
        assert_eq!(missing.code(), tonic::Code::Internal);
        assert!(missing.message().contains("Shard 7 returned 1 rows for 2"));

        let extra = attach_core_search_by_shard_results(
            9,
            specs,
            vec![
                vec![scored(1, 1.0)],
                vec![scored(2, 0.9)],
                vec![scored(3, 0.8)],
            ],
        )
        .err()
        .unwrap();
        assert_eq!(extra.code(), tonic::Code::Internal);
        assert!(extra.message().contains("Shard 9 returned 3 rows for 2"));
    }
}
