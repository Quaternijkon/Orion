//! One-hop handoff of the complete Orion coordinator. Shard ownership and the
//! upper/lower search algorithms are unchanged by coordinator placement.
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use api::grpc::qdrant::points_internal_client::PointsInternalClient;
use api::grpc::qdrant::{
    OrionCoordinateRequest, OrionCoordinateResponse, OrionNavigationLoadRequest,
    OrionNavigationLoadResponse,
};
use common::counter::hardware_accumulator::HwMeasurementAcc;
use common::search_load;
use futures::{StreamExt, stream};
use segment::types::ScoredPoint;
use shard::search::CoreSearchRequestBatch;

use super::navigation_balance::{NavigationBalance, NavigationIdentity};
use crate::collection::Collection;
use crate::config::AutoShardPolicy;
use crate::operations::consistency_params::ReadConsistency;
use crate::operations::shard_selector_internal::ShardSelectorInternal;
use crate::operations::types::{CollectionError, CollectionResult};
use crate::shards::channel_service::ChannelService;
use crate::shards::remote_shard::CollectionCoreSearchRequest;
use crate::shards::try_scored_point_from_grpc;

impl Collection {
    async fn navigation_identity(&self) -> Option<NavigationIdentity> {
        if !self.navigation_balance.enabled() || !self.orion_serving.load(Ordering::Acquire) {
            return None;
        }
        let config = self.collection_config.read().await;
        let Some(AutoShardPolicy::Orion {
            generation,
            artifact_sha256,
        }) = &config.auto_shard_policy
        else {
            return None;
        };
        let router = self.orion_router.load_full()?;
        if router.generation() != *generation {
            return None;
        }
        Some(NavigationIdentity {
            generation: *generation,
            artifact_sha256: artifact_sha256.clone(),
            policy_sha256: router.navigation_policy_fingerprint()?.to_owned(),
        })
    }

    /// Internal load probe; CPU values are occupied cores within the configured
    /// search capacity. Queue counts cover all local collections.
    pub async fn orion_navigation_load(&self) -> OrionNavigationLoadResponse {
        let identity = self.navigation_identity().await;
        if identity.is_some() {
            search_load::enable();
        }
        let balance = &self.navigation_balance;
        let load = search_load::snapshot(balance.capacity);
        let mut response = OrionNavigationLoadResponse {
            peer_id: self.this_peer_id,
            ready: identity.is_some(),
            cpu_capacity: balance.capacity as f64,
            cpu_cores: load.cpu_cores,
            worker_cores: load.worker_cores,
            navigation_cores: load.navigation_cores,
            worker_queued: load.worker_queued as u64,
            navigation_queued: load.navigation_queued as u64,
            admitted_navigation: load.admitted_navigation as u64,
            forwarding_fraction: balance.fraction(),
            local_queries: balance.local_queries.load(Ordering::Relaxed),
            forwarded_queries: balance.forwarded_queries.load(Ordering::Relaxed),
            received_queries: balance.received_queries.load(Ordering::Relaxed),
            forwarding_failures: balance.forwarding_failures.load(Ordering::Relaxed),
            ..Default::default()
        };
        if let Some(identity) = identity {
            response.generation = identity.generation;
            response.artifact_sha256 = identity.artifact_sha256;
            response.policy_sha256 = identity.policy_sha256;
        }
        response
    }

    fn refresh_navigation_peers(&self) {
        let balance = Arc::clone(&self.navigation_balance);
        let Some(lease) = balance.begin_refresh(Instant::now()) else {
            return;
        };
        let channel_service = self.channel_service.clone();
        let own_id = self.this_peer_id;
        let collection_name = self.id.clone();
        // Detached probes contain no collection reference. They are bounded,
        // and the lease releases the refresh gate even if the task is cancelled.
        tokio::spawn(async move {
            let _lease = lease;
            let peers: Vec<_> = channel_service
                .id_to_address
                .read()
                .iter()
                .filter(|(id, _)| **id != own_id)
                .map(|(&id, uri)| (id, uri.clone()))
                .collect();
            balance.retain_peers(&peers.iter().map(|(id, _)| *id).collect::<Vec<_>>());
            let probe_timeout = Duration::from_millis(balance.config.probe_timeout_ms);
            stream::iter(peers)
                .for_each_concurrent(8, |(peer_id, uri)| {
                    let channel_service = &channel_service;
                    let balance = &balance;
                    let collection_name = &collection_name;
                    async move {
                        let result = tokio::time::timeout(
                            probe_timeout,
                            channel_service.channel_pool.with_channel_timeout(
                                &uri,
                                |channel| {
                                    let mut request =
                                        tonic::Request::new(OrionNavigationLoadRequest {
                                            collection_name: collection_name.clone(),
                                        });
                                    request.set_timeout(probe_timeout);
                                    async move {
                                        PointsInternalClient::new(channel)
                                            .orion_navigation_load(request)
                                            .await
                                    }
                                },
                                Some(probe_timeout),
                                0,
                            ),
                        )
                        .await;
                        match result {
                            Ok(Ok(response)) if response.get_ref().peer_id == peer_id => {
                                balance.observe(response.into_inner(), Instant::now());
                            }
                            _ => balance.invalidate(peer_id),
                        }
                    }
                })
                .await;
        });
    }

    pub(crate) async fn try_forward_orion_coordinator(
        &self,
        request: &CoreSearchRequestBatch,
        read_consistency: Option<ReadConsistency>,
        timeout: Duration,
        hw_measurement_acc: HwMeasurementAcc,
    ) -> CollectionResult<Option<Vec<Vec<ScoredPoint>>>> {
        let started = Instant::now();
        let local = self.orion_navigation_load().await;
        if !local.ready {
            return Ok(None);
        }
        self.refresh_navigation_peers();
        let identity = NavigationIdentity {
            generation: local.generation,
            artifact_sha256: local.artifact_sha256.clone(),
            policy_sha256: local.policy_sha256.clone(),
        };
        let Some(reservation) = self.navigation_balance.choose(
            &local,
            &identity,
            request.searches.len(),
            Instant::now(),
            rand::random(),
            rand::random(),
        ) else {
            return Ok(None);
        };
        let peer_id = reservation.peer_id;
        let address = self
            .channel_service
            .id_to_address
            .read()
            .get(&peer_id)
            .cloned();
        let Some(address) = address else {
            self.navigation_balance.failed(peer_id, Instant::now());
            return Ok(None);
        };
        let mut wire_request = OrionCoordinateRequest {
            collection_name: self.id.clone(),
            search_points: request
                .searches
                .iter()
                .map(|search| CollectionCoreSearchRequest((self.id.clone(), search)).into())
                .collect(),
            read_consistency: read_consistency.map(Into::into),
            generation: identity.generation,
            artifact_sha256: identity.artifact_sha256.clone(),
            policy_sha256: identity.policy_sha256.clone(),
            timeout_millis: 0,
        };
        let remaining = remaining_budget(timeout, started.elapsed())?;
        // Round down: forwarding must never extend the original deadline.
        wire_request.timeout_millis = remaining.as_millis().min(u64::MAX as u128) as u64;
        if wire_request.timeout_millis == 0 {
            return Ok(None);
        }
        let result =
            exchange_coordinator(&self.channel_service, &address, wire_request, remaining).await;
        let response = match result {
            Ok(response) => response,
            _ => {
                // Includes overloaded, unavailable and older (Unimplemented)
                // peers. The caller falls back locally with the remaining time.
                self.navigation_balance.failed(peer_id, Instant::now());
                return Ok(None);
            }
        };
        if self.navigation_identity().await.as_ref() != Some(&identity) {
            return Ok(None);
        }
        let decoded =
            decode_coordinate_response(response, peer_id, &identity, request, &hw_measurement_acc);
        match decoded {
            Ok((points, load)) => {
                self.navigation_balance.observe(load, Instant::now());
                self.navigation_balance
                    .forwarded_queries
                    .fetch_add(request.searches.len() as u64, Ordering::Relaxed);
                Ok(Some(points))
            }
            Err(error) => {
                log::debug!("Rejected Orion coordinator response from {peer_id}: {error}");
                self.navigation_balance.failed(peer_id, Instant::now());
                Ok(None)
            }
        }
    }

    /// Only the private P2P endpoint calls this method. Calling the local search
    /// entry point explicitly makes the one-hop rule independent of load state.
    pub async fn coordinate_orion_search(
        &self,
        request: CoreSearchRequestBatch,
        read_consistency: Option<ReadConsistency>,
        identity: NavigationIdentity,
        timeout: Duration,
        hw_measurement_acc: HwMeasurementAcc,
    ) -> CollectionResult<Vec<Vec<ScoredPoint>>> {
        let started = Instant::now();
        for search in &request.searches {
            self.check_orion_generation(search.params.as_ref())?;
        }
        if timeout.is_zero()
            || request.searches.len() > self.navigation_balance.config.max_batch_queries
            || !self.orion_navigation_eligible(&request)
            || self.navigation_identity().await.as_ref() != Some(&identity)
        {
            return Err(CollectionError::bad_request(
                "Unavailable or incompatible Orion navigator",
            ));
        }
        search_load::enable();
        let load = self.orion_navigation_load().await;
        // Probe information is advisory. Admission uses current node-wide state,
        // so concurrent requests from different ingress nodes cannot overbook it.
        if self.navigation_balance.remote_overloaded(&load) {
            return Err(CollectionError::RateLimitExceeded {
                description: "Orion navigator is busy".into(),
                retry_after: None,
            });
        }
        let _admission = search_load::admit_navigation(
            NavigationBalance::units(request.searches.len()),
            Some(self.navigation_balance.admission_limit(&load)),
        )
        .ok_or_else(|| CollectionError::RateLimitExceeded {
            description: "Orion navigator admission limit reached".into(),
            retry_after: None,
        })?;
        self.navigation_balance
            .received_queries
            .fetch_add(request.searches.len() as u64, Ordering::Relaxed);
        let remaining = remaining_budget(timeout, started.elapsed())?;
        let points = tokio::time::timeout(
            remaining,
            self.do_core_search_batch_local(
                request,
                read_consistency,
                &ShardSelectorInternal::All,
                Some(remaining),
                hw_measurement_acc,
            ),
        )
        .await
        .map_err(|_| CollectionError::timeout(timeout, "Orion remote coordinator"))??;
        if self.navigation_identity().await.as_ref() != Some(&identity) {
            return Err(CollectionError::bad_request(
                "Orion generation changed during coordination",
            ));
        }
        Ok(points)
    }
}

async fn exchange_coordinator(
    channel_service: &ChannelService,
    address: &tonic::transport::Uri,
    request: OrionCoordinateRequest,
    timeout: Duration,
) -> CollectionResult<OrionCoordinateResponse> {
    let response = tokio::time::timeout(
        timeout,
        channel_service.channel_pool.with_channel_timeout(
            address,
            |channel| {
                let mut message = tonic::Request::new(request.clone());
                message.set_timeout(timeout);
                async move {
                    PointsInternalClient::new(channel)
                        .max_decoding_message_size(usize::MAX)
                        .orion_coordinate(message)
                        .await
                }
            },
            Some(timeout),
            0,
        ),
    )
    .await
    .map_err(|_| CollectionError::timeout(timeout, "Orion coordinator RPC"))??;
    Ok(response.into_inner())
}

fn remaining_budget(timeout: Duration, elapsed: Duration) -> CollectionResult<Duration> {
    timeout
        .checked_sub(elapsed)
        .filter(|remaining| !remaining.is_zero())
        .ok_or_else(|| CollectionError::timeout(timeout, "Orion coordinator forwarding"))
}

fn decode_coordinate_response(
    response: OrionCoordinateResponse,
    peer_id: u64,
    identity: &NavigationIdentity,
    request: &CoreSearchRequestBatch,
    hw: &HwMeasurementAcc,
) -> CollectionResult<(Vec<Vec<ScoredPoint>>, OrionNavigationLoadResponse)> {
    let load = response
        .load
        .filter(|load| load.ready && load.peer_id == peer_id && identity.matches(load))
        .ok_or_else(|| CollectionError::bad_request("Orion coordinator identity mismatch"))?;
    let response = response
        .response
        .filter(|response| response.result.len() == request.searches.len())
        .ok_or_else(|| {
            CollectionError::bad_request("Orion coordinator response row count mismatch")
        })?;
    if let Some(usage) = response.usage.unwrap_or_default().hardware {
        hw.accumulate_request(usage);
    }
    let points: Result<Vec<_>, tonic::Status> = response
        .result
        .into_iter()
        .zip(&request.searches)
        .map(|(batch, search)| {
            let with_payload = search
                .with_payload
                .as_ref()
                .is_some_and(|value| value.is_required());
            batch
                .result
                .into_iter()
                .map(|point| try_scored_point_from_grpc(point, with_payload))
                .collect()
        })
        .collect();
    Ok((points?, load))
}

#[cfg(test)]
mod tests {
    use super::*;
    use api::grpc::qdrant::{BatchResult, SearchBatchResponse};
    use segment::data_types::vectors::NamedQuery;
    use segment::types::{SearchParams, WithPayloadInterface, WithVector};
    use shard::query::query_enum::QueryEnum;
    use shard::search::CoreSearchRequest;

    fn request() -> CoreSearchRequestBatch {
        CoreSearchRequestBatch {
            searches: vec![CoreSearchRequest {
                query: QueryEnum::Nearest(NamedQuery::default_dense(vec![0., 1.])),
                filter: None,
                params: Some(SearchParams {
                    hnsw_ef: Some(48),
                    orion_generation: Some(7),
                    ..Default::default()
                }),
                hnsw_entry_points: None,
                hnsw_entry_points_by_shard: None,
                hnsw_ef_by_shard: None,
                source_id_dedup_block_size: None,
                limit: 3,
                offset: 2,
                with_payload: Some(WithPayloadInterface::Bool(true)),
                with_vector: Some(WithVector::Bool(true)),
                score_threshold: Some(0.3),
            }],
        }
    }

    #[test]
    fn handoff_preserves_query_options_and_read_consistency() {
        let original = request();
        let wire: api::grpc::qdrant::CoreSearchPoints =
            CollectionCoreSearchRequest(("test".into(), &original.searches[0])).into();
        let decoded = CoreSearchRequest::try_from(wire).unwrap();
        let mut expected = original.searches[0].clone();
        // The existing protobuf conversion spells the default vector name as
        // Some("") while REST uses None; both resolve to the same vector.
        let QueryEnum::Nearest(named) = &mut expected.query else {
            unreachable!()
        };
        named.using = Some(String::new());
        assert_eq!(expected, decoded);
        let consistency = ReadConsistency::Factor(3);
        assert_eq!(
            ReadConsistency::try_from_optional(Some(consistency.into())).unwrap(),
            Some(consistency)
        );
    }

    #[test]
    fn response_validation_preserves_rows_and_rejects_wrong_identity_or_count() {
        let request = request();
        let identity = NavigationIdentity {
            generation: 7,
            artifact_sha256: "graph".into(),
            policy_sha256: "policy".into(),
        };
        let response = OrionCoordinateResponse {
            response: Some(SearchBatchResponse {
                result: vec![BatchResult { result: vec![] }],
                ..Default::default()
            }),
            load: Some(OrionNavigationLoadResponse {
                peer_id: 2,
                ready: true,
                generation: 7,
                artifact_sha256: "graph".into(),
                policy_sha256: "policy".into(),
                ..Default::default()
            }),
        };
        assert_eq!(
            decode_coordinate_response(
                response.clone(),
                2,
                &identity,
                &request,
                &HwMeasurementAcc::new()
            )
            .unwrap()
            .0,
            vec![vec![]]
        );
        let mut wrong = response.clone();
        wrong.load.as_mut().unwrap().policy_sha256 = "wrong".into();
        assert!(
            decode_coordinate_response(wrong, 2, &identity, &request, &HwMeasurementAcc::new())
                .is_err()
        );
        let mut wrong = response.clone();
        wrong.response.as_mut().unwrap().result.clear();
        assert!(
            decode_coordinate_response(wrong, 2, &identity, &request, &HwMeasurementAcc::new())
                .is_err()
        );
        assert!(
            decode_coordinate_response(response, 3, &identity, &request, &HwMeasurementAcc::new())
                .is_err()
        );
    }

    #[test]
    fn failed_handoff_does_not_reset_timeout() {
        let total = Duration::from_millis(100);
        assert_eq!(
            remaining_budget(total, Duration::from_millis(40)).unwrap(),
            Duration::from_millis(60)
        );
        assert!(remaining_budget(total, total).is_err());
        assert!(remaining_budget(total, total + Duration::from_millis(1)).is_err());
    }

    // A small actual tonic server tests the production channel pool and new
    // wire method without standing up consensus or any experiment collection.
    #[derive(Clone)]
    struct MockNavigator {
        calls: Arc<std::sync::atomic::AtomicUsize>,
        error: Option<tonic::Code>,
        delay: Duration,
    }

    impl tonic::server::NamedService for MockNavigator {
        const NAME: &'static str = "qdrant.PointsInternal";
    }

    impl tonic::server::UnaryService<OrionCoordinateRequest> for MockNavigator {
        type Response = OrionCoordinateResponse;
        type Future = tonic::codegen::BoxFuture<tonic::Response<Self::Response>, tonic::Status>;
        fn call(&mut self, request: tonic::Request<OrionCoordinateRequest>) -> Self::Future {
            let this = self.clone();
            Box::pin(async move {
                this.calls.fetch_add(1, Ordering::Relaxed);
                tokio::time::sleep(this.delay).await;
                if let Some(code) = this.error {
                    return Err(tonic::Status::new(code, "mock peer rejection"));
                }
                let request = request.into_inner();
                assert_eq!(request.collection_name, "wire-test");
                assert_eq!(request.search_points[0].offset, Some(2));
                assert_eq!(
                    request.read_consistency,
                    Some(ReadConsistency::Factor(3).into())
                );
                Ok(tonic::Response::new(OrionCoordinateResponse {
                    response: Some(SearchBatchResponse {
                        result: vec![BatchResult { result: vec![] }],
                        ..Default::default()
                    }),
                    load: Some(OrionNavigationLoadResponse {
                        peer_id: 2,
                        ready: true,
                        generation: request.generation,
                        artifact_sha256: request.artifact_sha256,
                        policy_sha256: request.policy_sha256,
                        ..Default::default()
                    }),
                }))
            })
        }
    }

    impl<B> tonic::codegen::Service<tonic::codegen::http::Request<B>> for MockNavigator
    where
        B: tonic::codegen::Body + Send + 'static,
        B::Error: Into<tonic::codegen::StdError> + Send + 'static,
    {
        type Response = tonic::codegen::http::Response<tonic::body::BoxBody>;
        type Error = std::convert::Infallible;
        type Future = tonic::codegen::BoxFuture<Self::Response, Self::Error>;
        fn poll_ready(
            &mut self,
            _: &mut std::task::Context<'_>,
        ) -> std::task::Poll<Result<(), Self::Error>> {
            std::task::Poll::Ready(Ok(()))
        }
        fn call(&mut self, request: tonic::codegen::http::Request<B>) -> Self::Future {
            assert_eq!(
                request.uri().path(),
                "/qdrant.PointsInternal/OrionCoordinate"
            );
            let this = self.clone();
            Box::pin(async move {
                let codec = tonic::codec::ProstCodec::default();
                Ok(tonic::server::Grpc::new(codec).unary(this, request).await)
            })
        }
    }

    #[tokio::test]
    async fn coordinator_transport_roundtrip_rejections_and_timeout_do_not_retry() {
        for error in [
            None,
            Some(tonic::Code::Unimplemented),
            Some(tonic::Code::Unavailable),
            Some(tonic::Code::ResourceExhausted),
        ] {
            check_transport(error, Duration::ZERO).await;
        }
        check_transport(None, Duration::from_secs(1)).await;
    }

    async fn check_transport(error: Option<tonic::Code>, delay: Duration) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = format!("http://{}", listener.local_addr().unwrap())
            .parse()
            .unwrap();
        let incoming = stream::unfold(listener, |listener| async {
            let stream = listener.accept().await.map(|(stream, _)| stream);
            Some((stream, listener))
        });
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let server = tokio::spawn(
            tonic::transport::Server::builder()
                .add_service(MockNavigator {
                    calls: Arc::clone(&calls),
                    error,
                    delay,
                })
                .serve_with_incoming(incoming),
        );
        let server = tokio_util::task::AbortOnDropHandle::new(server);
        let request = OrionCoordinateRequest {
            collection_name: "wire-test".into(),
            search_points: request()
                .searches
                .iter()
                .map(|r| CollectionCoreSearchRequest(("wire-test".into(), r)).into())
                .collect(),
            read_consistency: Some(ReadConsistency::Factor(3).into()),
            generation: 7,
            artifact_sha256: "graph".into(),
            policy_sha256: "policy".into(),
            timeout_millis: 250,
        };
        let started = Instant::now();
        let result = exchange_coordinator(
            &ChannelService::default(),
            &address,
            request,
            Duration::from_millis(250),
        )
        .await;
        if error.is_none() && delay.is_zero() {
            let response = result.unwrap();
            assert_eq!(response.load.unwrap().generation, 7);
            assert_eq!(response.response.unwrap().result.len(), 1);
        } else {
            assert!(result.is_err());
        }
        assert_eq!(
            calls.load(Ordering::Relaxed),
            1,
            "handoff must never be retried"
        );
        if !delay.is_zero() {
            assert!(started.elapsed() < delay, "RPC timeout was ignored");
        }
        drop(server);
    }
}
