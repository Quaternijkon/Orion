//! Adaptive placement of complete Orion query coordinators on colocated nodes.
use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use api::grpc::qdrant::OrionNavigationLoadResponse;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use validator::Validate;

use super::coordinator_permits::StartBudget;
use crate::shards::shard::PeerId;

#[derive(Clone, Debug, Serialize, Deserialize, Validate)]
#[serde(default, deny_unknown_fields)]
pub struct NavigationBalanceConfig {
    pub enabled: bool,
    #[validate(range(min = 50, max = 10000))]
    pub refresh_interval_ms: u64,
    #[validate(range(min = 250, max = 60000))]
    pub stale_after_ms: u64,
    #[validate(range(min = 10, max = 5000))]
    pub probe_timeout_ms: u64,
    #[validate(range(min = 0.5, max = 1.0))]
    pub target_utilization: f64,
    #[validate(range(min = 0.01, max = 0.3))]
    pub imbalance_threshold: f64,
    #[validate(range(min = 0.01, max = 0.5))]
    pub adjustment_rate: f64,
    #[validate(range(min = 1, max = 128))]
    pub max_inflight_per_cpu: usize,
    /// Lifetime of a sender's block of unclaimed upper-search start permits.
    #[validate(range(min = 1, max = 1000))]
    pub direct_permit_ttl_ms: u64,
    #[validate(range(min = 1, max = 4096))]
    pub max_batch_queries: usize,
}
impl Default for NavigationBalanceConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            refresh_interval_ms: 250,
            stale_after_ms: 1500,
            probe_timeout_ms: 200,
            target_utilization: 0.95,
            imbalance_threshold: 0.05,
            adjustment_rate: 0.15,
            max_inflight_per_cpu: 8,
            direct_permit_ttl_ms: 50,
            max_batch_queries: 1024,
        }
    }
}

impl NavigationBalanceConfig {
    pub fn cpu_capacity(search_threads: usize) -> usize {
        search_threads
            .max(1)
            .min(common::cpu::get_num_cpus().max(1))
    }

    pub fn window_lease(&self) -> Duration {
        Duration::from_millis(
            self.stale_after_ms
                .max(self.refresh_interval_ms + 3 * self.probe_timeout_ms),
        )
    }

    pub(super) fn direct_start_budget(
        &self,
        capacity: usize,
        load: &common::search_load::SearchLoad,
    ) -> StartBudget {
        let other = (load.cpu_cores - load.worker_cores - load.navigation_cores).max(0.);
        let target = capacity as f64 * self.target_utilization;
        let weight = (target - load.worker_cores - other - load.worker_queued as f64).max(0.);
        let limit = (weight * self.max_inflight_per_cpu as f64).floor() as usize;
        StartBudget {
            weight,
            limit,
            navigation_queued: load.navigation_queued,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NavigationIdentity {
    pub generation: u64,
    pub artifact_sha256: String,
    pub policy_sha256: String,
}
impl NavigationIdentity {
    pub(crate) fn matches(&self, sample: &OrionNavigationLoadResponse) -> bool {
        self.generation == sample.generation
            && self.artifact_sha256 == sample.artifact_sha256
            && self.policy_sha256 == sample.policy_sha256
    }
}

struct PeerLoad {
    sample: OrionNavigationLoadResponse,
    received: Instant,
    cooldown_until: Instant,
    reserved: usize,
}
struct State {
    peers: BTreeMap<PeerId, PeerLoad>,
    last_refresh: Option<Instant>,
    last_adjustment: Option<Instant>,
    forwarding_fraction: f64,
}

pub struct NavigationBalance {
    pub(super) collection_instance: uuid::Uuid,
    pub config: NavigationBalanceConfig,
    pub capacity: usize,
    distributed: bool,
    state: Mutex<State>,
    refreshing: AtomicBool,
    pub local_queries: AtomicU64,
    pub forwarded_queries: AtomicU64,
    pub received_queries: AtomicU64,
    pub forwarding_failures: AtomicU64,
}

pub struct RefreshLease {
    balance: Arc<NavigationBalance>,
}
impl Drop for RefreshLease {
    fn drop(&mut self) {
        self.balance.refreshing.store(false, Ordering::Release);
    }
}

pub struct PeerReservation {
    balance: Arc<NavigationBalance>,
    pub peer_id: PeerId,
    units: usize,
}
impl Drop for PeerReservation {
    fn drop(&mut self) {
        if let Some(peer) = self.balance.state.lock().peers.get_mut(&self.peer_id) {
            peer.reserved = peer.reserved.saturating_sub(self.units);
        }
    }
}

impl NavigationBalance {
    pub fn new(config: NavigationBalanceConfig, capacity: usize, distributed: bool) -> Arc<Self> {
        Arc::new(Self {
            collection_instance: uuid::Uuid::new_v4(),
            config,
            capacity: NavigationBalanceConfig::cpu_capacity(capacity),
            distributed,
            state: Mutex::new(State {
                peers: BTreeMap::new(),
                last_refresh: None,
                last_adjustment: None,
                forwarding_fraction: 0.,
            }),
            refreshing: AtomicBool::new(false),
            local_queries: AtomicU64::new(0),
            forwarded_queries: AtomicU64::new(0),
            received_queries: AtomicU64::new(0),
            forwarding_failures: AtomicU64::new(0),
        })
    }
    pub fn enabled(&self) -> bool {
        self.distributed && self.config.enabled
    }
    pub fn fraction(&self) -> f64 {
        self.state.lock().forwarding_fraction
    }
    pub fn units(queries: usize) -> usize {
        queries.div_ceil(32).max(1)
    }

    pub fn admission_limit(&self, sample: &OrionNavigationLoadResponse) -> usize {
        let other = (sample.cpu_cores - sample.worker_cores - sample.navigation_cores).max(0.);
        // Worker/other work consumes its budget first. A busy node may have
        // zero navigator capacity; do not reserve a minimum core for navigation.
        let available =
            (sample.cpu_capacity * self.config.target_utilization - sample.worker_cores - other)
                .max(0.);
        (available * self.config.max_inflight_per_cpu as f64).floor() as usize
    }

    pub fn remote_overloaded(&self, sample: &OrionNavigationLoadResponse) -> bool {
        !valid_sample(sample)
            || !sample.ready
            || self.admission_limit(sample) == 0
            || sample.cpu_cores / sample.cpu_capacity >= self.config.target_utilization
            || sample
                .worker_queued
                .saturating_add(sample.navigation_queued)
                > self.capacity as u64
    }

    pub fn begin_refresh(self: &Arc<Self>, now: Instant) -> Option<RefreshLease> {
        let mut state = self.state.lock();
        if state.last_refresh.is_some_and(|at| {
            now.duration_since(at) < Duration::from_millis(self.config.refresh_interval_ms)
        }) || self.refreshing.swap(true, Ordering::AcqRel)
        {
            return None;
        }
        state.last_refresh = Some(now);
        Some(RefreshLease {
            balance: Arc::clone(self),
        })
    }

    pub fn observe(&self, sample: OrionNavigationLoadResponse, now: Instant) {
        if !valid_sample(&sample) {
            self.invalidate(sample.peer_id);
            return;
        }
        let mut state = self.state.lock();
        if let Some(peer) = state.peers.get_mut(&sample.peer_id) {
            peer.sample = sample;
            peer.received = now;
        } else {
            state.peers.insert(
                sample.peer_id,
                PeerLoad {
                    sample,
                    received: now,
                    cooldown_until: now,
                    reserved: 0,
                },
            );
        }
    }
    pub fn retain_peers(&self, ids: &[PeerId]) {
        self.state.lock().peers.retain(|id, _| ids.contains(id));
    }
    pub fn invalidate(&self, peer_id: PeerId) {
        if let Some(peer) = self.state.lock().peers.get_mut(&peer_id) {
            peer.sample.ready = false;
        }
    }
    pub fn failed(&self, peer_id: PeerId, now: Instant) {
        self.forwarding_failures.fetch_add(1, Ordering::Relaxed);
        if let Some(peer) = self.state.lock().peers.get_mut(&peer_id) {
            peer.cooldown_until = now + Duration::from_millis(self.config.stale_after_ms);
        }
    }

    pub fn choose(
        self: &Arc<Self>,
        local: &OrionNavigationLoadResponse,
        identity: &NavigationIdentity,
        queries: usize,
        now: Instant,
        coin: f64,
        choice: f64,
    ) -> Option<PeerReservation> {
        if !self.enabled()
            || queries == 0
            || queries > self.config.max_batch_queries
            || !local.ready
            || !valid_sample(local)
            || !coin.is_finite()
            || !choice.is_finite()
        {
            return None;
        }
        let mut state = self.state.lock();
        if state.last_adjustment.is_none_or(|at| {
            now.duration_since(at) >= Duration::from_millis(self.config.refresh_interval_ms)
        }) {
            let utilization = local.cpu_cores / local.cpu_capacity;
            // Balance below saturation as well: a worker-heavy ingress can
            // donate navigation to idle peers before its own queues fill up.
            let (cpu, capacity) = state
                .peers
                .iter()
                .filter(|(id, peer)| {
                    **id != local.peer_id
                        && peer.sample.ready
                        && identity.matches(&peer.sample)
                        && now >= peer.cooldown_until
                        && now.duration_since(peer.received)
                            <= Duration::from_millis(self.config.stale_after_ms)
                })
                .fold(
                    (local.cpu_cores, local.cpu_capacity),
                    |(cpu, capacity), (_, peer)| {
                        (
                            cpu + peer.sample.cpu_cores,
                            capacity + peer.sample.cpu_capacity,
                        )
                    },
                );
            let target = if capacity > local.cpu_capacity {
                self.config
                    .target_utilization
                    .min(cpu / capacity + self.config.imbalance_threshold / 2.)
            } else {
                // With no usable peers, an idle node should promptly reclaim
                // its ingress navigation instead of retaining stale offload bias.
                self.config.target_utilization
            };
            let mut error = utilization - target;
            if local.worker_queued.saturating_add(local.navigation_queued)
                > local.cpu_capacity as u64
            {
                error = error.max(0.1);
            }
            if error.abs() > 0.015 {
                let nav = (local.navigation_cores / local.cpu_capacity).max(0.1);
                let change = (self.config.adjustment_rate * error / nav)
                    .clamp(-self.config.adjustment_rate, self.config.adjustment_rate);
                state.forwarding_fraction = (state.forwarding_fraction + change).clamp(0., 1.);
            }
            state.last_adjustment = Some(now);
        }
        if coin >= state.forwarding_fraction {
            return None;
        }
        let units = Self::units(queries);
        let local_pressure = local.cpu_cores / local.cpu_capacity
            + local.worker_queued.saturating_add(local.navigation_queued) as f64
                / local.cpu_capacity
                / 8.;
        let mut candidates = Vec::new();
        for (&id, peer) in &state.peers {
            let sample = &peer.sample;
            if id == local.peer_id
                || !sample.ready
                || !identity.matches(sample)
                || now.duration_since(peer.received)
                    > Duration::from_millis(self.config.stale_after_ms)
                || now < peer.cooldown_until
            {
                continue;
            }
            let limit = self.admission_limit(sample);
            let inflight = usize::try_from(sample.admitted_navigation)
                .unwrap_or(usize::MAX)
                .saturating_add(peer.reserved);
            if inflight.saturating_add(units) > limit {
                continue;
            }
            let pressure = sample.cpu_cores / sample.cpu_capacity
                + sample
                    .worker_queued
                    .saturating_add(sample.navigation_queued) as f64
                    / sample.cpu_capacity
                    / 8.
                + peer.reserved as f64 / limit.max(1) as f64 * 0.1;
            if pressure + self.config.imbalance_threshold >= local_pressure
                || pressure >= self.config.target_utilization
            {
                continue;
            }
            // Remaining worker/other CPU determines how much navigation this
            // peer can absorb. Reservations damp concurrent ingress stampedes.
            let weight = (limit as f64 / (1. + inflight as f64)) * (1. - pressure).max(0.01);
            candidates.push((id, weight));
        }
        let sum: f64 = candidates.iter().map(|(_, weight)| weight).sum();
        if sum <= 0. {
            return None;
        }
        let mut ticket = choice.clamp(0., 1. - f64::EPSILON) * sum;
        let mut selected = candidates.last()?.0;
        for (id, weight) in candidates {
            if ticket < weight {
                selected = id;
                break;
            }
            ticket -= weight;
        }
        state.peers.get_mut(&selected)?.reserved += units;
        Some(PeerReservation {
            balance: Arc::clone(self),
            peer_id: selected,
            units,
        })
    }
}

fn valid_sample(sample: &OrionNavigationLoadResponse) -> bool {
    sample.cpu_capacity.is_finite()
        && sample.cpu_capacity >= 1.
        && sample.cpu_capacity <= 1_000_000.
        && [
            sample.cpu_cores,
            sample.worker_cores,
            sample.navigation_cores,
        ]
        .iter()
        .all(|v| v.is_finite() && *v >= 0. && *v <= sample.cpu_capacity * 1.1)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn sample(id: u64, cpu: f64, worker: f64) -> OrionNavigationLoadResponse {
        OrionNavigationLoadResponse {
            peer_id: id,
            generation: 1,
            artifact_sha256: "graph".into(),
            policy_sha256: "policy".into(),
            ready: true,
            cpu_capacity: 48.,
            cpu_cores: cpu,
            worker_cores: worker,
            navigation_cores: (cpu - worker).max(0.),
            ..Default::default()
        }
    }
    fn identity() -> NavigationIdentity {
        NavigationIdentity {
            generation: 1,
            artifact_sha256: "graph".into(),
            policy_sha256: "policy".into(),
        }
    }
    #[test]
    fn direct_budget_tracks_worker_pressure_and_reopens_with_outstanding_rpc_waiters() {
        use common::search_load::SearchLoad;
        let balance = NavigationBalance::new(Default::default(), 48, true);
        let capacity = balance.capacity as f64;
        let cool = SearchLoad {
            cpu_cores: capacity * 0.8,
            worker_cores: capacity * 0.25,
            navigation_cores: capacity * 0.5,
            admitted_navigation: 10_000,
            ..Default::default()
        };
        let available = balance.config.direct_start_budget(balance.capacity, &cool);
        assert!((available.weight - capacity * 0.65).abs() < 1e-9);
        let queued = SearchLoad {
            worker_queued: balance.capacity,
            ..cool
        };
        assert_eq!(
            balance
                .config
                .direct_start_budget(balance.capacity, &queued)
                .limit,
            0
        );
        let saturated = SearchLoad {
            cpu_cores: capacity,
            worker_cores: capacity,
            ..Default::default()
        };
        assert_eq!(
            balance
                .config
                .direct_start_budget(balance.capacity, &saturated)
                .limit,
            0
        );
        assert!(
            balance
                .config
                .direct_start_budget(balance.capacity, &cool)
                .limit
                > 0
        );
    }

    #[test]
    fn busy_workers_offload_navigation_then_reclaim_it_when_idle() {
        let balance = NavigationBalance::new(Default::default(), 48, true);
        let now = Instant::now();
        let busy = sample(1, 48., 43.);
        balance.observe(sample(2, 15., 10.), now);
        for i in 0..4 {
            drop(balance.choose(
                &busy,
                &identity(),
                64,
                now + Duration::from_millis(i * 250),
                0.,
                0.,
            ));
        }
        assert!(balance.fraction() > 0.1);
        assert_eq!(
            balance
                .choose(
                    &busy,
                    &identity(),
                    64,
                    now + Duration::from_millis(1000),
                    0.,
                    0.
                )
                .unwrap()
                .peer_id,
            2
        );
        for i in 5..15 {
            drop(balance.choose(
                &sample(1, 10., 5.),
                &identity(),
                64,
                now + Duration::from_millis(i * 250),
                0.,
                0.,
            ));
        }
        assert_eq!(balance.fraction(), 0.);
    }
    #[test]
    fn stale_mismatched_and_failed_peers_are_not_selected() {
        let balance = NavigationBalance::new(Default::default(), 48, true);
        let now = Instant::now();
        let busy = sample(1, 48., 43.);
        let mut wrong = sample(2, 0., 0.);
        wrong.generation = 2;
        balance.observe(wrong, now);
        assert!(
            balance
                .choose(&busy, &identity(), 64, now, 0., 0.)
                .is_none()
        );
        balance.observe(sample(2, 0., 0.), now);
        balance.failed(2, now);
        assert!(
            balance
                .choose(&busy, &identity(), 64, now, 0., 0.)
                .is_none()
        );
        assert!(
            balance
                .choose(&busy, &identity(), 64, now + Duration::from_secs(3), 0., 0.)
                .is_none()
        );
    }
    #[test]
    fn balanced_nodes_keep_queries_local_and_invalid_samples_are_ignored() {
        let balance = NavigationBalance::new(Default::default(), 48, true);
        let now = Instant::now();
        balance.observe(sample(2, 30., 20.), now);
        assert!(
            balance
                .choose(&sample(1, 30., 20.), &identity(), 64, now, 0., 0.)
                .is_none()
        );
        let mut invalid = sample(3, 0., 0.);
        invalid.cpu_capacity = f64::NAN;
        balance.observe(invalid, now);
        assert!(!balance.state.lock().peers.contains_key(&3));
    }

    #[test]
    fn worker_imbalance_is_corrected_before_saturation() {
        let balance = NavigationBalance::new(Default::default(), 48, true);
        let now = Instant::now();
        balance.observe(sample(2, 8., 3.), now);
        let chosen = balance
            .choose(&sample(1, 38., 34.), &identity(), 32, now, 0., 0.)
            .unwrap();
        assert_eq!(chosen.peer_id, 2);
        assert!(balance.fraction() > 0.);
    }

    #[test]
    fn reservations_and_admission_prevent_overbooking_and_release_on_drop() {
        let balance = NavigationBalance::new(Default::default(), 48, true);
        let now = Instant::now();
        let mut peer = sample(2, 8., 3.);
        let limit = balance.admission_limit(&peer);
        peer.admitted_navigation = (limit - 2) as u64;
        balance.observe(peer, now);
        let busy = sample(1, 48., 43.);
        let held = balance.choose(&busy, &identity(), 64, now, 0., 0.).unwrap();
        assert!(balance.choose(&busy, &identity(), 1, now, 0., 0.).is_none());
        drop(held);
        assert!(
            balance
                .choose(&busy, &identity(), 64, now, 0., 0.)
                .is_some()
        );
    }

    #[test]
    fn worker_budget_can_leave_no_capacity_for_navigation() {
        let balance = NavigationBalance::new(Default::default(), 48, true);
        let now = Instant::now();
        let budget = 48. * balance.config.target_utilization;
        // Less than one admission unit of headroom must not be rounded up to
        // an entire core reserved for navigators at a worker's expense.
        let nearly_full = sample(2, budget - 0.05, budget - 0.05);
        assert_eq!(balance.admission_limit(&nearly_full), 0);
        assert!(balance.remote_overloaded(&nearly_full));
        let full = sample(2, 48., 48.);
        assert_eq!(balance.admission_limit(&full), 0);
        assert!(balance.remote_overloaded(&full));

        balance.observe(nearly_full, now);
        balance.observe(sample(3, 10., 5.), now);
        let destination = balance
            .choose(&sample(1, 48., 40.), &identity(), 32, now, 0., 0.)
            .unwrap();
        assert_eq!(destination.peer_id, 3);
    }

    #[test]
    fn policy_mismatch_probe_failure_and_disabled_balancing_exclude_peers() {
        let balance = NavigationBalance::new(Default::default(), 48, true);
        let now = Instant::now();
        let busy = sample(1, 48., 43.);
        let mut peer = sample(2, 0., 0.);
        peer.policy_sha256 = "different policy".into();
        balance.observe(peer, now);
        assert!(
            balance
                .choose(&busy, &identity(), 32, now, 0., 0.)
                .is_none()
        );
        balance.observe(sample(2, 0., 0.), now);
        balance.invalidate(2);
        assert!(
            balance
                .choose(&busy, &identity(), 32, now, 0., 0.)
                .is_none()
        );
        let disabled = NavigationBalance::new(
            NavigationBalanceConfig {
                enabled: false,
                ..Default::default()
            },
            48,
            true,
        );
        disabled.observe(sample(2, 0., 0.), now);
        assert!(
            disabled
                .choose(&busy, &identity(), 32, now, 0., 0.)
                .is_none()
        );
        assert!(!NavigationBalance::new(Default::default(), 48, false).enabled());
    }

    #[test]
    fn queue_counts_cannot_overflow_and_large_batches_stay_local() {
        let balance = NavigationBalance::new(Default::default(), 48, true);
        let now = Instant::now();
        balance.observe(sample(2, 0., 0.), now);
        let mut queued = sample(1, 1., 1.);
        queued.worker_queued = u64::MAX;
        queued.navigation_queued = u64::MAX;
        assert!(
            balance
                .choose(&queued, &identity(), 32, now, 0., 0.)
                .is_some()
        );
        assert!(
            balance
                .choose(&queued, &identity(), 1025, now, 0., 0.)
                .is_none()
        );
        assert!(balance.remote_overloaded(&queued));
    }

    #[test]
    fn eight_node_feedback_moves_navigation_to_worker_headroom() {
        let balances: Vec<_> = (0..8)
            .map(|_| NavigationBalance::new(Default::default(), 48, true))
            .collect();
        let workers = [0.85, 0.70, 0.60, 0.55, 0.50, 0.45, 0.30, 0.20];
        let navigation = simulate_navigation(&balances, &workers, [0.20; 8], Instant::now());
        assert!(
            navigation[0] < 0.08,
            "busy worker kept too much navigation: {navigation:?}"
        );
        assert!(
            navigation[7] > 0.30,
            "idle worker did not absorb navigation: {navigation:?}"
        );
        assert!(
            workers
                .iter()
                .zip(navigation)
                .all(|(worker, nav)| worker + nav < 0.97)
        );
    }

    // Fixed ingress traffic and fixed worker demand: only the placement of
    // navigation can change. Check work conservation at every control interval.
    fn simulate_navigation(
        balances: &[Arc<NavigationBalance>],
        workers: &[f64; 8],
        mut navigation: [f64; 8],
        start: Instant,
    ) -> [f64; 8] {
        for round in 0..80 {
            let now = start + Duration::from_millis(round * 250);
            let loads: Vec<_> = (0..8)
                .map(|i| {
                    sample(
                        i as u64 + 1,
                        (workers[i] + navigation[i]).min(1.) * 48.,
                        workers[i] * 48.,
                    )
                })
                .collect();
            let mut next = [0.; 8];
            for (i, balance) in balances.iter().enumerate() {
                for load in &loads {
                    balance.observe(load.clone(), now);
                }
                // Every ingress receives the same number/cost of queries.
                for query in 0..100 {
                    let coin = ((query * 617 + i as u64 * 127 + round * 31) % 997) as f64 / 997.;
                    let choice = ((query * 431 + i as u64 * 71 + round * 97) % 991) as f64 / 991.;
                    let reservation = balance.choose(&loads[i], &identity(), 1, now, coin, choice);
                    let destination = reservation.as_ref().map_or(i, |r| r.peer_id as usize - 1);
                    next[destination] += 0.20 / 100.;
                }
            }
            navigation = next;
            assert!(
                (navigation.iter().sum::<f64>() - 1.60).abs() < 1e-9,
                "navigation placement must neither add nor discard query work"
            );
        }
        navigation
    }

    #[test]
    fn worker_bottleneck_sheds_all_navigation_and_adapts_when_hotspot_moves() {
        let balances: Vec<_> = (0..8)
            .map(|_| NavigationBalance::new(Default::default(), 48, true))
            .collect();
        let mut workers = [0.97, 0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.15];
        let start = Instant::now();
        let navigation = simulate_navigation(&balances, &workers, [0.20; 8], start);
        assert_eq!(
            navigation[0], 0.,
            "worker-only bottleneck must shed all movable navigation"
        );
        assert_eq!(balances[0].fraction(), 1.);
        assert!(navigation[7] > 0.20);
        // No amount of navigation redistribution can remove this worker floor.
        assert!(workers[0] + navigation[0] >= balances[0].config.target_utilization);

        workers.swap(0, 7);
        let navigation = simulate_navigation(
            &balances,
            &workers,
            navigation,
            start + Duration::from_secs(20),
        );
        assert_eq!(navigation[7], 0.);
        assert!(
            navigation[0] > 0.20,
            "formerly hot worker must resume navigation when idle"
        );
        assert!(balances[0].fraction() < 1.);
    }

    #[test]
    fn saturation_does_not_latch_a_global_scheduling_stop() {
        let balance = NavigationBalance::new(Default::default(), 48, true);
        let now = Instant::now();
        let busy = sample(1, 48., 42.);
        for id in 2..=8 {
            balance.observe(sample(id, 48., 48.), now);
        }
        assert!(
            balance
                .choose(&busy, &identity(), 32, now, 0., 0.)
                .is_none()
        );
        // Recovered worker headroom must immediately become usable again.
        let later = now + Duration::from_millis(250);
        balance.observe(sample(2, 12., 8.), later);
        assert_eq!(
            balance
                .choose(&busy, &identity(), 32, later, 0., 0.)
                .unwrap()
                .peer_id,
            2
        );
    }

    #[test]
    fn refresh_is_throttled_and_cancel_safe() {
        let balance = NavigationBalance::new(Default::default(), 48, true);
        let now = Instant::now();
        let lease = balance.begin_refresh(now).unwrap();
        assert!(
            balance
                .begin_refresh(now + Duration::from_secs(1))
                .is_none()
        );
        drop(lease);
        assert!(
            balance
                .begin_refresh(now + Duration::from_millis(100))
                .is_none()
        );
        assert!(
            balance
                .begin_refresh(now + Duration::from_secs(1))
                .is_some()
        );
    }
}
