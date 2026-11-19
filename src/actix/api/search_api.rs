use actix_web::{HttpResponse, Responder, post, web};
use actix_web_validator::{Json, Path, Query};
use api::rest::{SearchMatrixOffsetsResponse, SearchMatrixPairsResponse, SearchMatrixRequest};
use collection::collection::distance_matrix::CollectionSearchMatrixRequest;
use collection::operations::shard_selector_internal::ShardSelectorInternal;
use collection::operations::types::{
    CoreSearchRequest, SearchGroupsRequest, SearchRequest, SearchRequestBatch,
};
use collection::operations::verification::new_unchecked_verification_pass;
use collection::orion::coordinator_permits::CoordinatorPermit;
use itertools::Itertools;
use serde::Deserialize;
use storage::content_manager::collection_verification::check_strict_mode;
use storage::dispatcher::Dispatcher;
use storage::rbac::{AccessRequirements, Auth};
use storage::content_manager::errors::StorageError;
use validator::Validate;
use tokio::time::Instant;

use super::CollectionPath;
use super::read_params::ReadParams;
use crate::actix::auth::ActixAuth;
use crate::actix::helpers::{
    get_request_hardware_counter, process_response, process_response_error,
};
use crate::common::query::{
    do_core_search_points, do_search_batch_points, do_search_point_groups, do_search_points_matrix,
};
use crate::settings::ServiceConfig;

#[post("/collections/{collection_name}/points/search")]
async fn search_points(
    dispatcher: web::Data<Dispatcher>,
    collection: Path<CollectionPath>,
    request: Json<SearchRequest>,
    params: Query<ReadParams>,
    service_config: web::Data<ServiceConfig>,
    ActixAuth(auth): ActixAuth,
) -> HttpResponse {
    single_search(
        dispatcher,
        collection,
        request.into_inner(),
        params,
        service_config,
        auth,
        None,
    )
    .await
}

#[derive(Deserialize, Validate)]
struct PermitRequest {
    #[validate(range(max = 64))]
    count: usize,
    /// A caller currently has an unassigned query. Older callers request slots on demand.
    waiting: Option<bool>,
}

#[post("/collections/{collection_name}/orion/coordinator/permits")]
async fn coordinator_permits(
    dispatcher: web::Data<Dispatcher>,
    collection: Path<CollectionPath>,
    request: Json<PermitRequest>,
    ActixAuth(auth): ActixAuth,
) -> HttpResponse {
    let timing = Instant::now();
    let response: Result<_, StorageError> = async {
        let pass = auth.check_collection_access(
            &collection.collection_name,
            AccessRequirements::new(),
            "orion_coordinator_permits",
        )?;
        let verification = new_unchecked_verification_pass();
        let collection = dispatcher
            .toc(&auth, &verification)
            .get_collection(&pass)
            .await?;
        Ok(collection
            .coordinator_permits(request.count, request.waiting.unwrap_or(request.count > 0))
            .await?)
    }
    .await;
    process_response(response, timing, None)
}

#[derive(Deserialize, Validate)]
struct DirectSearch {
    permit: CoordinatorPermit,
    #[validate(nested)]
    search: SearchRequest,
}

#[post("/collections/{collection_name}/orion/coordinator/search")]
async fn coordinator_search(
    dispatcher: web::Data<Dispatcher>,
    collection: Path<CollectionPath>,
    request: Json<DirectSearch>,
    params: Query<ReadParams>,
    service_config: web::Data<ServiceConfig>,
    ActixAuth(auth): ActixAuth,
) -> HttpResponse {
    let DirectSearch { permit, search } = request.into_inner();
    single_search(
        dispatcher,
        collection,
        search,
        params,
        service_config,
        auth,
        Some(permit),
    )
    .await
}

async fn single_search(
    dispatcher: web::Data<Dispatcher>,
    collection: Path<CollectionPath>,
    request: SearchRequest,
    params: Query<ReadParams>,
    service_config: web::Data<ServiceConfig>,
    auth: Auth,
    permit: Option<CoordinatorPermit>,
) -> HttpResponse {
    let SearchRequest {
        search_request,
        shard_key,
    } = request;

    let pass = match check_strict_mode(
        &search_request,
        params.timeout_as_secs(),
        &collection.collection_name,
        &dispatcher,
        &auth,
    )
    .await
    {
        Ok(pass) => pass,
        Err(err) => return process_response_error(err, Instant::now(), None),
    };

    let shard_selection = match shard_key {
        None => ShardSelectorInternal::All,
        Some(shard_keys) => shard_keys.into(),
    };

    let request_hw_counter = get_request_hardware_counter(
        &dispatcher,
        collection.collection_name.clone(),
        service_config.hardware_reporting(),
        None,
    );

    let timing = Instant::now();

    let toc = dispatcher.toc(&auth, &pass);
    let result = if let Some(permit) = permit {
        if !matches!(shard_selection, ShardSelectorInternal::All) {
            return process_response_error(
                StorageError::bad_request("direct Orion search requires automatic shard selection"),
                timing,
                None,
            );
        }
        match toc
            .search_on_coordinator(
                &collection.collection_name,
                search_request.into(),
                permit,
                params.consistency,
                auth,
                params.timeout(),
                request_hw_counter.get_counter(),
            )
            .await
        {
            Ok(Some(points)) => Ok(points),
            Ok(None) => {
                return HttpResponse::PreconditionFailed()
                    .insert_header(("x-orion-admission", "not-started"))
                    .json(serde_json::json!({
                        "status": {"error": "coordinator permit unavailable before search"}
                    }));
            }
            Err(error) => Err(error),
        }
    } else {
        do_core_search_points(
            toc,
            &collection.collection_name,
            search_request.into(),
            params.consistency,
            shard_selection,
            auth,
            params.timeout(),
            request_hw_counter.get_counter(),
        )
        .await
    }
    .map(|scored_points| {
        scored_points
            .into_iter()
            .map(api::rest::ScoredPoint::from)
            .collect_vec()
    });

    process_response(result, timing, request_hw_counter.to_rest_api())
}

#[post("/collections/{collection_name}/points/search/batch")]
async fn batch_search_points(
    dispatcher: web::Data<Dispatcher>,
    collection: Path<CollectionPath>,
    request: Json<SearchRequestBatch>,
    params: Query<ReadParams>,
    service_config: web::Data<ServiceConfig>,
    ActixAuth(auth): ActixAuth,
) -> HttpResponse {
    let pass = match check_strict_mode(
        &*request,
        params.timeout_as_secs(),
        &collection.collection_name,
        &dispatcher,
        &auth,
    )
    .await
    {
        Ok(pass) => pass,
        Err(err) => return process_response_error(err, Instant::now(), None),
    };

    let requests = request
        .into_inner()
        .searches
        .into_iter()
        .map(|req| {
            let SearchRequest {
                search_request,
                shard_key,
            } = req;
            let shard_selection = match shard_key {
                None => ShardSelectorInternal::All,
                Some(shard_keys) => shard_keys.into(),
            };
            let core_request: CoreSearchRequest = search_request.into();

            (core_request, shard_selection)
        })
        .collect::<Vec<_>>();

    let request_hw_counter = get_request_hardware_counter(
        &dispatcher,
        collection.collection_name.clone(),
        service_config.hardware_reporting(),
        None,
    );

    let timing = Instant::now();

    let result = do_search_batch_points(
        dispatcher.toc(&auth, &pass),
        &collection.collection_name,
        requests,
        params.consistency,
        auth,
        params.timeout(),
        request_hw_counter.get_counter(),
    )
    .await
    .map(|batch_scored_points| {
        batch_scored_points
            .into_iter()
            .map(|scored_points| {
                scored_points
                    .into_iter()
                    .map(api::rest::ScoredPoint::from)
                    .collect_vec()
            })
            .collect_vec()
    });

    process_response(result, timing, request_hw_counter.to_rest_api())
}

#[post("/collections/{collection_name}/points/search/groups")]
async fn search_point_groups(
    dispatcher: web::Data<Dispatcher>,
    collection: Path<CollectionPath>,
    request: Json<SearchGroupsRequest>,
    params: Query<ReadParams>,
    service_config: web::Data<ServiceConfig>,
    ActixAuth(auth): ActixAuth,
) -> HttpResponse {
    let SearchGroupsRequest {
        search_group_request,
        shard_key,
    } = request.into_inner();

    let pass = match check_strict_mode(
        &search_group_request,
        params.timeout_as_secs(),
        &collection.collection_name,
        &dispatcher,
        &auth,
    )
    .await
    {
        Ok(pass) => pass,
        Err(err) => return process_response_error(err, Instant::now(), None),
    };

    let shard_selection = match shard_key {
        None => ShardSelectorInternal::All,
        Some(shard_keys) => shard_keys.into(),
    };

    let request_hw_counter = get_request_hardware_counter(
        &dispatcher,
        collection.collection_name.clone(),
        service_config.hardware_reporting(),
        None,
    );
    let timing = Instant::now();

    let result = do_search_point_groups(
        dispatcher.toc(&auth, &pass),
        &collection.collection_name,
        search_group_request,
        params.consistency,
        shard_selection,
        auth,
        params.timeout(),
        request_hw_counter.get_counter(),
    )
    .await;

    process_response(result, timing, request_hw_counter.to_rest_api())
}

#[post("/collections/{collection_name}/points/search/matrix/pairs")]
async fn search_points_matrix_pairs(
    dispatcher: web::Data<Dispatcher>,
    collection: Path<CollectionPath>,
    request: Json<SearchMatrixRequest>,
    params: Query<ReadParams>,
    service_config: web::Data<ServiceConfig>,
    ActixAuth(auth): ActixAuth,
) -> impl Responder {
    let SearchMatrixRequest {
        search_request,
        shard_key,
    } = request.into_inner();

    let pass = match check_strict_mode(
        &search_request,
        params.timeout_as_secs(),
        &collection.collection_name,
        &dispatcher,
        &auth,
    )
    .await
    {
        Ok(pass) => pass,
        Err(err) => return process_response_error(err, Instant::now(), None),
    };

    let shard_selection = match shard_key {
        None => ShardSelectorInternal::All,
        Some(shard_keys) => shard_keys.into(),
    };

    let request_hw_counter = get_request_hardware_counter(
        &dispatcher,
        collection.collection_name.clone(),
        service_config.hardware_reporting(),
        None,
    );
    let timing = Instant::now();

    let response = do_search_points_matrix(
        dispatcher.toc(&auth, &pass),
        &collection.collection_name,
        CollectionSearchMatrixRequest::from(search_request),
        params.consistency,
        shard_selection,
        auth,
        params.timeout(),
        request_hw_counter.get_counter(),
    )
    .await
    .map(SearchMatrixPairsResponse::from);

    process_response(response, timing, request_hw_counter.to_rest_api())
}

#[post("/collections/{collection_name}/points/search/matrix/offsets")]
async fn search_points_matrix_offsets(
    dispatcher: web::Data<Dispatcher>,
    collection: Path<CollectionPath>,
    request: Json<SearchMatrixRequest>,
    params: Query<ReadParams>,
    service_config: web::Data<ServiceConfig>,
    ActixAuth(auth): ActixAuth,
) -> impl Responder {
    let SearchMatrixRequest {
        search_request,
        shard_key,
    } = request.into_inner();

    let pass = match check_strict_mode(
        &search_request,
        params.timeout_as_secs(),
        &collection.collection_name,
        &dispatcher,
        &auth,
    )
    .await
    {
        Ok(pass) => pass,
        Err(err) => return process_response_error(err, Instant::now(), None),
    };

    let shard_selection = match shard_key {
        None => ShardSelectorInternal::All,
        Some(shard_keys) => shard_keys.into(),
    };

    let request_hw_counter = get_request_hardware_counter(
        &dispatcher,
        collection.collection_name.clone(),
        service_config.hardware_reporting(),
        None,
    );
    let timing = Instant::now();

    let response = do_search_points_matrix(
        dispatcher.toc(&auth, &pass),
        &collection.collection_name,
        CollectionSearchMatrixRequest::from(search_request),
        params.consistency,
        shard_selection,
        auth,
        params.timeout(),
        request_hw_counter.get_counter(),
    )
    .await
    .map(SearchMatrixOffsetsResponse::from);

    process_response(response, timing, request_hw_counter.to_rest_api())
}

// Configure services
pub fn config_search_api(cfg: &mut web::ServiceConfig) {
    cfg.service(search_points)
        .service(coordinator_permits)
        .service(coordinator_search)
        .service(batch_search_points)
        .service(search_point_groups)
        .service(search_points_matrix_pairs)
        .service(search_points_matrix_offsets);
}
