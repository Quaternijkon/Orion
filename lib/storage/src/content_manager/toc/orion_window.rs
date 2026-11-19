//! The elected leader apportions the direct query window in background rounds.
//! Only samples and quotas cross this control path. Each coordinator admits,
//! executes, merges and releases the query on its own machine.
mod cpu;
pub(super) use cpu::AssignedCpu;

use std::sync::Weak;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use api::grpc::qdrant::points_internal_client::PointsInternalClient;
use api::grpc::qdrant::{
    OrionQueryWindowAllocation, OrionQueryWindowReport, OrionQueryWindowRequest,
};
use collection::orion::navigation_balance::NavigationBalanceConfig;
use collection::orion::query_window::QueryWindow;
use common::search_load;
use futures::future::join_all;
use tokio::time::MissedTickBehavior;
use tonic::{Request, Status};

use super::TableOfContent;

impl TableOfContent {
    fn query_window_view(&self) -> Option<OrionQueryWindowRequest> {
        let dispatcher = self.toc_dispatcher.lock();
        let (leader_id, term, members) =
            dispatcher.as_ref()?.consensus_state().query_window_view()?;
        Some(OrionQueryWindowRequest {
            leader_id,
            term,
            members,
            ..Default::default()
        })
    }

    /// Called by the internal service, including the local leg of a control round.
    pub fn orion_query_window(
        &self,
        request: OrionQueryWindowRequest,
    ) -> Result<OrionQueryWindowReport, Status> {
        let cfg = &self.storage_config.performance.orion_navigation;
        let view = self
            .query_window_view()
            .ok_or_else(|| Status::unavailable("Orion window leader unavailable"))?;
        if !cfg.enabled
            || view.leader_id != request.leader_id
            || view.term != request.term
            || view.members != request.members
        {
            return Err(Status::failed_precondition(
                "Orion window membership or term changed",
            ));
        }
        let pool = &self.channel_service.orion_permits;
        let now = Instant::now();
        let ack = OrionQueryWindowReport {
            peer_id: self.this_peer_id,
            ..Default::default()
        };
        if request.wake {
            if self.this_peer_id != view.leader_id || request.allocation.is_some() {
                return Err(Status::failed_precondition("invalid Orion window wakeup"));
            }
            self.orion_window_wakeup.store(true, Ordering::Release);
            return Ok(ack);
        }
        if let Some(allocation) = request.allocation {
            return if pool.apply_window(&allocation, cfg.window_lease(), now) {
                Ok(OrionQueryWindowReport {
                    node_limit: allocation.node_limit,
                    occupied: pool.occupied_slots() as u64,
                    ..ack
                })
            } else {
                Err(Status::failed_precondition("stale Orion window allocation"))
            };
        }
        let threads = common::defaults::search_thread_count(
            self.storage_config.performance.max_search_threads,
        );
        let capacity = NavigationBalanceConfig::cpu_capacity(threads);
        search_load::enable();
        let mut report = pool.inspect_window(
            self.this_peer_id,
            cfg,
            capacity,
            search_load::snapshot(capacity),
            now,
        );
        if let Some(cores) = self
            .orion_window_cpu
            .lock()
            .sample(capacity, cfg.window_lease())
        {
            report.cpu_cores = cores;
        }
        Ok(report)
    }

    async fn window_exchange(
        &self,
        peer: u64,
        message: OrionQueryWindowRequest,
        timeout: Duration,
    ) -> Result<OrionQueryWindowReport, Status> {
        if peer == self.this_peer_id {
            return self.orion_query_window(message);
        }
        let uri = self
            .channel_service
            .id_to_address
            .read()
            .get(&peer)
            .cloned()
            .ok_or_else(|| Status::unavailable("Orion window member address unavailable"))?;
        let response = tokio::time::timeout(
            timeout,
            self.channel_service.channel_pool.with_channel_timeout(
                &uri,
                |channel| {
                    let mut req = Request::new(message.clone());
                    req.set_timeout(timeout);
                    async move {
                        PointsInternalClient::new(channel)
                            .orion_query_window(req)
                            .await
                    }
                },
                Some(timeout),
                0,
            ),
        )
        .await
        .map_err(|_| Status::deadline_exceeded("Orion window peer timeout"))?
        .map_err(|err| Status::unavailable(err.to_string()))?
        .into_inner();
        if response.peer_id != peer {
            return Err(Status::failed_precondition("unexpected Orion window peer"));
        }
        Ok(response)
    }

    async fn install_window_share(
        &self,
        view: &OrionQueryWindowRequest,
        peer: u64,
        grant: &OrionQueryWindowAllocation,
        timeout: Duration,
    ) -> Result<OrionQueryWindowReport, Status> {
        let mut request = view.clone();
        request.allocation = Some(grant.clone());
        self.window_exchange(peer, request, timeout).await
    }

    async fn window_round(
        &self,
        view: &OrionQueryWindowRequest,
        control: &mut QueryWindow,
        cfg: &NavigationBalanceConfig,
    ) -> Result<bool, Status> {
        let timeout = Duration::from_millis(cfg.probe_timeout_ms);
        let replies = join_all(
            view.members
                .iter()
                .map(|&id| self.window_exchange(id, view.clone(), timeout)),
        )
        .await;
        let mut reports: Vec<_> = replies.into_iter().collect::<Result<_, _>>()?;
        if reports.iter().any(|n| {
            n.fence.len() != 16
                || n.cpu_capacity == 0
                || !n.cpu_cores.is_finite()
                || !(0. ..=n.cpu_capacity as f64).contains(&n.cpu_cores)
                || !n.coordinator_room.is_finite()
                || !(0. ..=n.cpu_capacity as f64).contains(&n.coordinator_room)
        }) {
            return Err(Status::failed_precondition("invalid Orion window sample"));
        }
        if !reports.iter().any(|n| n.demand || n.occupied > 0) {
            // A follower with a live (possibly zero) share expects us to poll it.
            // Keep observing, without renewing, until all such leases expire.
            // New demand on that follower can then resume on the next round.
            control.missed_sample();
            return Ok(reports.iter().any(|n| n.cluster_limit > 0));
        }
        let mut grants = control.targets(&reports, cfg);
        let (donors, others): (Vec<_>, Vec<_>) =
            (0..reports.len()).partition(|&i| grants[i].node_limit < reports[i].node_limit);
        // Lend freed slots in this round. Until every donor acknowledges, another
        // sender could still be using the previous quota.
        let shrinks = donors
            .iter()
            .map(|&i| self.install_window_share(view, reports[i].peer_id, &grants[i], timeout));
        let replies = join_all(shrinks).await;
        for (i, reply) in donors.into_iter().zip(replies) {
            let ack = reply?;
            if ack.node_limit != grants[i].node_limit {
                return Err(Status::failed_precondition(
                    "Orion window reduction was not acknowledged",
                ));
            }
            reports[i].node_limit = ack.node_limit;
            reports[i].occupied = ack.occupied;
        }
        control.bound_growth(&reports, &mut grants);
        let updates = others
            .into_iter()
            .map(|i| self.install_window_share(view, reports[i].peer_id, &grants[i], timeout));
        // A timeout may have applied a write. The next round fences and reads
        // all members before reallocating again.
        for reply in join_all(updates).await {
            reply?;
        }
        Ok(true)
    }
}

pub(super) async fn control_loop(owner: Weak<TableOfContent>, cfg: NavigationBalanceConfig) {
    let mut timer = tokio::time::interval(Duration::from_millis(cfg.refresh_interval_ms));
    timer.set_missed_tick_behavior(MissedTickBehavior::Delay);
    let mut last_view = None;
    let mut control = QueryWindow::default();
    let mut active = false;
    loop {
        timer.tick().await;
        let Some(toc) = owner.upgrade() else { return };
        let pool = &toc.channel_service.orion_permits;
        let now = Instant::now();
        let local_demand = pool.has_demand(now, cfg.window_lease());
        let woken = toc.orion_window_wakeup.swap(false, Ordering::AcqRel);
        if !active && !local_demand && !woken {
            continue;
        }
        let Some(view) = toc.query_window_view() else {
            active = false;
            last_view = None;
            continue;
        };
        if view.leader_id != toc.this_peer_id {
            active = false;
            last_view = None;
            if local_demand && !pool.has_current_window(now) {
                let wake = OrionQueryWindowRequest { wake: true, ..view };
                let _ = toc
                    .window_exchange(
                        wake.leader_id,
                        wake,
                        Duration::from_millis(cfg.probe_timeout_ms),
                    )
                    .await;
            }
            continue;
        }
        if last_view.as_ref() != Some(&view) {
            control = QueryWindow::default();
            last_view = Some(view.clone());
        }
        match toc.window_round(&view, &mut control, &cfg).await {
            Ok(demand) => {
                active = demand;
                if !active {
                    last_view = None;
                }
            }
            Err(err) => {
                // Never grow from a partial snapshot. New admissions cease when a
                // peer's last lease expires. Local or remote demand wakes us again.
                control.missed_sample();
                active = local_demand || woken || pool.has_current_window(now);
                log::debug!("Orion direct window round deferred: {err}");
            }
        }
    }
}
