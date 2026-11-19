//! Node-wide, single-use reservations for work that has yet to start upper search.
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use api::grpc::qdrant::{OrionQueryWindowAllocation, OrionQueryWindowReport};
use common::search_load::SearchLoad;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::navigation_balance::{NavigationBalanceConfig, NavigationIdentity};

pub const MAX_GRANT: usize = u64::BITS as usize;

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
pub struct CoordinatorPermit {
    pub lease: Uuid,
    pub slot: u8,
}

#[derive(Debug, Serialize)]
pub struct CoordinatorGrant {
    pub peer_id: u64,
    pub generation: u64,
    pub artifact_sha256: String,
    pub policy_sha256: String,
    pub weight: f64,
    pub limit: usize,
    pub ttl_ms: u64,
    pub lease: Option<Uuid>,
    pub count: usize,
    /// Cluster window, this node's allocation, and accepted queries still executing here.
    pub cluster_window: usize,
    pub node_window: usize,
    pub inflight: usize,
    pub issued: u64,
    pub pending: usize,
    pub claimed_waiting: usize,
    pub peak_pending: usize,
    pub started: u64,
    pub expired: u64,
    pub revoked: u64,
    pub cancelled_before_start: u64,
}

pub(super) struct StartBudget {
    pub weight: f64,
    pub limit: usize,
    pub navigation_queued: usize,
}

struct Block {
    collection: Uuid,
    identity: NavigationIdentity,
    expires: Instant,
    unused: u64,
}

#[derive(Default)]
struct Window {
    cluster_limit: usize,
    node_limit: usize,
    expires: Option<Instant>,
    fence: Option<(Uuid, Instant)>,
    last_request: Option<Instant>,
    waiting: bool,
    peak_occupied: usize,
    inflight: usize,
}

#[derive(Default)]
struct State {
    window: Window,
    blocks: HashMap<Uuid, Block>,
    issued: u64,
    pending: usize,
    claimed_waiting: usize,
    peak_pending: usize,
    started: u64,
    expired: u64,
    revoked: u64,
    cancelled: u64,
}

impl State {
    /// Unused reservations and accepted queries are disjoint. An upper task waiting
    /// on the CPU is already included in inflight.
    fn occupied(&self) -> usize {
        (self.pending - self.claimed_waiting) + self.window.inflight
    }

    fn expire_window(&mut self, now: Instant) {
        if self.window.expires.is_some_and(|deadline| now >= deadline) {
            self.window.node_limit = 0;
            self.window.cluster_limit = 0;
            self.window.expires = None;
            self.retire_unclaimed(now, true);
        }
    }

    fn retire_unclaimed(&mut self, now: Instant, revoke: bool) {
        let Self {
            blocks,
            pending,
            expired,
            revoked,
            ..
        } = self;
        blocks.retain(|_, block| {
            let counter = if block.expires <= now {
                &mut *expired
            } else if revoke {
                &mut *revoked
            } else {
                return true;
            };
            let n = block.unused.count_ones();
            *counter += u64::from(n);
            *pending -= n as usize;
            false
        });
    }
}

#[derive(Default)]
pub struct PermitPool(Mutex<State>);

impl PermitPool {
    pub fn occupied_slots(&self) -> usize {
        self.0.lock().occupied()
    }

    pub fn has_demand(&self, now: Instant, lifetime: Duration) -> bool {
        self.0
            .lock()
            .window
            .last_request
            .is_some_and(|at| now - at < lifetime)
    }

    pub fn has_current_window(&self, now: Instant) -> bool {
        self.0
            .lock()
            .window
            .expires
            .is_some_and(|until| now < until)
    }

    /// A detached cluster member must stop accepting its old allocation immediately.
    pub fn withdraw_window(&self) {
        let mut state = self.0.lock();
        state.window.fence = None;
        state.window.node_limit = 0;
        state.window.cluster_limit = 0;
        state.window.expires = None;
        state.retire_unclaimed(Instant::now(), true);
    }

    /// Reading for a control round fences delayed writes from all preceding rounds.
    /// An ordinary client observation never changes this fence.
    pub fn inspect_window(
        &self,
        peer_id: u64,
        config: &NavigationBalanceConfig,
        capacity: usize,
        load: SearchLoad,
        now: Instant,
    ) -> OrionQueryWindowReport {
        let mut state = self.0.lock();
        state.expire_window(now);
        state.retire_unclaimed(now, false);
        let occupied = state.occupied();
        let window = &mut state.window;
        let fence = Uuid::new_v4();
        window.fence = Some((fence, now + config.window_lease()));
        let start_budget = config.direct_start_budget(capacity, &load);
        OrionQueryWindowReport {
            peer_id,
            fence: fence.as_bytes().to_vec(),
            node_limit: window.node_limit as u64,
            cluster_limit: window.cluster_limit as u64,
            occupied: occupied as u64,
            peak_occupied: std::mem::replace(&mut window.peak_occupied, occupied).max(occupied)
                as u64,
            demand: window
                .last_request
                .is_some_and(|at| now - at < config.window_lease()),
            waiting: std::mem::take(&mut window.waiting),
            cpu_capacity: capacity as u64,
            cpu_cores: load.cpu_cores,
            coordinator_room: if start_budget.limit > load.navigation_queued {
                start_budget.weight
            } else {
                0.
            },
            worker_queued: load.worker_queued as u64,
            worker_queue_drains: load.worker_queue_drains,
            navigation_queued: load.navigation_queued as u64,
            navigation_queue_drains: load.navigation_queue_drains,
        }
    }

    pub fn apply_window(
        &self,
        grant: &OrionQueryWindowAllocation,
        lifetime: Duration,
        now: Instant,
    ) -> bool {
        let mut state = self.0.lock();
        state.expire_window(now);
        if grant.node_limit > grant.cluster_limit
            || !state
                .window
                .fence
                .is_some_and(|(id, until)| now < until && grant.fence.as_slice() == id.as_bytes())
        {
            return false;
        }
        let Ok(limit) = usize::try_from(grant.node_limit) else {
            return false;
        };
        let Ok(cluster) = usize::try_from(grant.cluster_limit) else {
            return false;
        };
        state.window.fence = None;
        state.window.expires = Some(now + lifetime);
        state.window.node_limit = limit;
        state.window.cluster_limit = cluster;
        // Invalidate unused tickets on a contraction. Accepted work drains on its owner.
        let revoke = state.occupied() > limit;
        state.retire_unclaimed(now, revoke);
        true
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn grant(
        &self,
        collection: Uuid,
        identity: NavigationIdentity,
        peer_id: u64,
        requested: usize,
        waiting: bool,
        ttl: Duration,
        observe: impl FnOnce() -> (Instant, StartBudget),
    ) -> CoordinatorGrant {
        let mut state = self.0.lock();
        // Serialize observation, revocation and issuance with slot redemption. An
        // older observation cannot issue new tickets after a later zero-weight update.
        let (now, budget) = observe();
        state.expire_window(now);
        if requested > 0 {
            state.window.last_request = Some(now);
        }
        state.retire_unclaimed(now, budget.limit == 0);
        // Accepted direct searches can appear in both counts while queued for upper search.
        let occupied = state.pending.max(budget.navigation_queued);
        let start_count = requested
            .min(MAX_GRANT)
            .min(budget.limit.saturating_sub(occupied));
        let room = state.window.node_limit.saturating_sub(state.occupied());
        let count = start_count.min(room);
        state.window.waiting |= waiting && start_count > count;
        let lease = (count > 0).then(|| {
            let id = Uuid::new_v4();
            state.blocks.insert(
                id,
                Block {
                    collection,
                    identity: identity.clone(),
                    expires: now + ttl,
                    unused: u64::MAX >> (MAX_GRANT - count),
                },
            );
            state.issued += count as u64;
            state.pending += count;
            state.window.peak_occupied = state.window.peak_occupied.max(state.occupied());
            state.peak_pending = state.peak_pending.max(state.pending);
            id
        });
        CoordinatorGrant {
            peer_id,
            generation: identity.generation,
            artifact_sha256: identity.artifact_sha256,
            policy_sha256: identity.policy_sha256,
            weight: if state.window.node_limit > 0 {
                budget.weight
            } else {
                0.
            },
            limit: budget.limit.min(state.window.node_limit),
            cluster_window: state.window.cluster_limit,
            node_window: state.window.node_limit,
            inflight: state.window.inflight,
            ttl_ms: ttl.as_millis() as u64,
            lease,
            count,
            issued: state.issued,
            pending: state.pending,
            claimed_waiting: state.claimed_waiting,
            peak_pending: state.peak_pending,
            started: state.started,
            expired: state.expired,
            revoked: state.revoked,
            cancelled_before_start: state.cancelled,
        }
    }

    pub(super) fn claim(
        self: &Arc<Self>,
        collection: Uuid,
        identity: &NavigationIdentity,
        permit: CoordinatorPermit,
        clock: impl FnOnce() -> Instant,
    ) -> Option<DirectAdmission> {
        let bit = 1_u64.checked_shl(permit.slot.into())?;
        let mut state = self.0.lock();
        let now = clock();
        state.expire_window(now);
        let block = state.blocks.get_mut(&permit.lease)?;
        if block.collection != collection
            || block.identity != *identity
            || block.expires <= now
            || block.unused & bit == 0
        {
            return None;
        }
        block.unused &= !bit;
        if block.unused == 0 {
            state.blocks.remove(&permit.lease);
        }
        state.claimed_waiting += 1;
        state.window.inflight += 1;
        let inflight = Arc::new(InflightQuery {
            pool: Arc::clone(self),
        });
        Some(DirectAdmission {
            pending: PendingStart {
                inflight: Arc::clone(&inflight),
                started: false,
            },
            inflight,
        })
    }
}

pub(crate) struct DirectAdmission {
    pub(crate) pending: PendingStart,
    pub(crate) inflight: Arc<InflightQuery>,
}

/// Shared by the awaiting coordinator and its upper task. Even cancellation
/// racing a queued blocking closure releases exactly one end-to-end slot.
pub(crate) struct InflightQuery {
    pool: Arc<PermitPool>,
}

impl Drop for InflightQuery {
    fn drop(&mut self) {
        self.pool.0.lock().window.inflight -= 1;
    }
}

/// Owns one slot until its blocking upper search actually starts, even past the
/// original grant expiry. Dropping a cancelled request also releases that slot.
pub(crate) struct PendingStart {
    inflight: Arc<InflightQuery>,
    started: bool,
}

impl PendingStart {
    pub(crate) fn start(mut self) {
        self.started = true;
    }
}

impl Drop for PendingStart {
    fn drop(&mut self) {
        let mut state = self.inflight.pool.0.lock();
        state.pending -= 1;
        state.claimed_waiting -= 1;
        if self.started {
            state.started += 1;
        } else {
            state.cancelled += 1;
        }
    }
}

#[cfg(test)]
impl PermitPool {
    pub(crate) fn set_test_window(&self, limit: u64, now: Instant) {
        let cfg = NavigationBalanceConfig::default();
        let report = self.inspect_window(1, &cfg, 48, SearchLoad::default(), now);
        assert!(self.apply_window(
            &OrionQueryWindowAllocation {
                fence: report.fence,
                node_limit: limit,
                cluster_limit: limit,
            },
            cfg.window_lease(),
            now
        ));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity() -> NavigationIdentity {
        NavigationIdentity {
            generation: 1,
            artifact_sha256: "graph".into(),
            policy_sha256: "policy".into(),
        }
    }

    #[test]
    fn revocation_expiry_and_redemption_share_one_node_budget() {
        let pool = Arc::new(PermitPool::default());
        let a = Uuid::new_v4();
        let b = Uuid::new_v4();
        let id = identity();
        let start = Instant::now();
        let ttl = Duration::from_millis(50);
        pool.set_test_window(64, start);
        let grant = |c, limit, n, t| {
            pool.grant(c, id.clone(), 1, n, true, ttl, || {
                (
                    t,
                    StartBudget {
                        weight: limit as f64,
                        limit,
                        navigation_queued: 0,
                    },
                )
            })
        };

        let first = grant(a, 4, 3, start);
        let second = grant(b, 4, 3, start);
        assert_eq!((first.count, second.count), (3, 1));
        let ticket = CoordinatorPermit {
            lease: first.lease.unwrap(),
            slot: 0,
        };
        assert!(pool.claim(b, &id, ticket, || start).is_none());
        let next_generation = NavigationIdentity {
            generation: 2,
            ..id.clone()
        };
        assert!(pool.claim(a, &next_generation, ticket, || start).is_none());
        let accepted = pool.claim(a, &id, ticket, || start).unwrap();
        assert!(pool.claim(a, &id, ticket, || start).is_none());
        assert!(
            pool.claim(a, &id, CoordinatorPermit { slot: 63, ..ticket }, || start)
                .is_none()
        );

        let hot = grant(a, 0, MAX_GRANT, start);
        assert_eq!(
            (hot.count, hot.revoked, hot.pending, hot.claimed_waiting),
            (0, 3, 1, 1)
        );
        assert!(
            pool.claim(a, &id, CoordinatorPermit { slot: 1, ..ticket }, || start)
                .is_none()
        );
        let recovered = grant(b, 4, MAX_GRANT, start);
        assert_eq!(recovered.count, 3);
        let next = CoordinatorPermit {
            lease: recovered.lease.unwrap(),
            slot: 0,
        };
        let cancelled = pool.claim(b, &id, next, || start).unwrap();
        let later = start + ttl;
        assert!(
            pool.claim(b, &id, CoordinatorPermit { slot: 1, ..next }, || later)
                .is_none()
        );
        let snapshot = grant(a, 4, 0, later);
        assert_eq!((snapshot.pending, snapshot.expired), (2, 2));
        accepted.pending.start();
        drop(accepted.inflight);
        drop(cancelled);
        let end = grant(a, 4, 0, later);
        assert_eq!(
            (
                end.pending,
                end.claimed_waiting,
                end.started,
                end.cancelled_before_start
            ),
            (0, 0, 1, 1)
        );
        assert_eq!(
            end.issued,
            end.started + end.expired + end.revoked + end.cancelled_before_start
        );
    }

    #[test]
    fn full_query_holds_the_cluster_slot_after_upper_starts_and_during_drain() {
        let cfg = NavigationBalanceConfig::default();
        let pool = Arc::new(PermitPool::default());
        let collection = Uuid::new_v4();
        let now = Instant::now();
        let ttl = Duration::from_millis(50);
        let ask = |n, waiting, t| {
            pool.grant(collection, identity(), 1, n, waiting, ttl, || {
                (
                    t,
                    StartBudget {
                        weight: 20.,
                        limit: 100,
                        navigation_queued: 0,
                    },
                )
            })
        };
        let inspect = |t| pool.inspect_window(1, &cfg, 20, SearchLoad::default(), t);
        let apply = |report: &OrionQueryWindowReport, limit, t| {
            pool.apply_window(
                &OrionQueryWindowAllocation {
                    fence: report.fence.clone(),
                    cluster_limit: 4,
                    node_limit: limit,
                },
                cfg.window_lease(),
                t,
            )
        };
        assert_eq!(ask(2, true, now).count, 0);
        let old = inspect(now);
        let current = inspect(now);
        assert!(!apply(&old, 4, now));
        assert!(apply(&current, 2, now));
        assert!(!apply(&current, 4, now));
        let batch = ask(8, true, now);
        assert_eq!(
            (batch.cluster_window, batch.node_window, batch.count),
            (4, 2, 2)
        );
        let permit = CoordinatorPermit {
            lease: batch.lease.unwrap(),
            slot: 0,
        };
        let accepted = pool.claim(collection, &identity(), permit, || now).unwrap();
        accepted.pending.start();
        // Slot 1 expires. The query waiting for shard replies still counts against the window.
        let later = now + ttl;
        let one = ask(8, false, later);
        assert_eq!((one.count, one.inflight, one.pending), (1, 1, 1));
        let latest = inspect(later);
        assert_eq!(latest.occupied, 2);
        assert!(apply(&latest, 0, later));
        let draining = ask(1, true, later);
        assert_eq!(
            (draining.count, draining.inflight, draining.pending),
            (0, 1, 0)
        );
        assert!(
            pool.claim(
                collection,
                &identity(),
                CoordinatorPermit {
                    lease: one.lease.unwrap(),
                    slot: 0
                },
                || later
            )
            .is_none()
        );
        drop(accepted.inflight);
        assert_eq!(inspect(later).occupied, 0);
        pool.set_test_window(2, later);
        assert_eq!(ask(1, false, later).count, 1);
        // A partitioned node cannot issue on an expired allocation or accept an old ticket.
        let expired = later + cfg.window_lease();
        assert_eq!(ask(1, true, expired).count, 0);
        assert_eq!(inspect(expired).node_limit, 0);
    }

    #[test]
    fn cancelled_upper_and_coordinator_release_one_slot_in_either_order() {
        let pool = Arc::new(PermitPool::default());
        let id = Uuid::new_v4();
        let now = Instant::now();
        pool.set_test_window(1, now);
        for cancel_upper_first in [true, false] {
            let grant = pool.grant(
                id,
                identity(),
                1,
                1,
                true,
                Duration::from_millis(50),
                || {
                    (
                        now,
                        StartBudget {
                            weight: 1.,
                            limit: 8,
                            navigation_queued: 0,
                        },
                    )
                },
            );
            let ticket = CoordinatorPermit {
                lease: grant.lease.unwrap(),
                slot: 0,
            };
            let DirectAdmission { pending, inflight } =
                pool.claim(id, &identity(), ticket, || now).unwrap();
            if cancel_upper_first {
                drop(pending);
                assert_eq!(pool.0.lock().occupied(), 1);
                drop(inflight);
            } else {
                drop(inflight);
                assert_eq!(pool.0.lock().occupied(), 1);
                drop(pending);
            }
            assert_eq!(pool.0.lock().occupied(), 0);
        }
    }

    #[test]
    fn moving_global_capacity_between_peers_never_reissues_an_active_querys_slot() {
        use crate::orion::query_window::QueryWindow;
        let cfg = NavigationBalanceConfig::default();
        let nodes = [
            Arc::new(PermitPool::default()),
            Arc::new(PermitPool::default()),
        ];
        let now = Instant::now();
        let collection = Uuid::new_v4();
        let ask = |i: usize, n| {
            nodes[i].grant(
                collection,
                identity(),
                i as u64,
                n,
                true,
                Duration::from_millis(50),
                || {
                    (
                        now,
                        StartBudget {
                            weight: 4.,
                            limit: 32,
                            navigation_queued: 0,
                        },
                    )
                },
            )
        };
        let mut controller = QueryWindow::default();
        let round = |controller: &mut QueryWindow, a_cpu| {
            let mut reports: Vec<_> = nodes
                .iter()
                .enumerate()
                .map(|(i, node)| {
                    let cpu = if i == 0 { a_cpu } else { 0. };
                    node.inspect_window(
                        i as u64,
                        &cfg,
                        4,
                        SearchLoad {
                            cpu_cores: cpu,
                            worker_cores: cpu * 0.5,
                            navigation_cores: cpu * 0.5,
                            ..Default::default()
                        },
                        now,
                    )
                })
                .collect();
            let mut grants = controller.targets(&reports, &cfg);
            let (donors, receivers): (Vec<_>, Vec<_>) =
                (0..nodes.len()).partition(|&i| grants[i].node_limit < reports[i].node_limit);
            for i in donors {
                assert!(nodes[i].apply_window(&grants[i], cfg.window_lease(), now));
                reports[i].node_limit = grants[i].node_limit;
                reports[i].occupied = nodes[i].occupied_slots() as u64;
            }
            controller.bound_growth(&reports, &mut grants);
            for i in receivers {
                assert!(nodes[i].apply_window(&grants[i], cfg.window_lease(), now));
            }
            grants
        };
        assert_eq!((ask(0, 2).count, ask(1, 2).count), (0, 0));
        let initial = round(&mut controller, 0.);
        assert_eq!((initial[0].node_limit, initial[1].node_limit), (4, 4));
        let a = ask(0, 4);
        let mut jobs = Vec::new();
        for slot in 0..2 {
            let direct = nodes[0]
                .claim(
                    collection,
                    &identity(),
                    CoordinatorPermit {
                        lease: a.lease.unwrap(),
                        slot,
                    },
                    || now,
                )
                .unwrap();
            direct.pending.start();
            jobs.push(direct.inflight);
        }
        assert_eq!(ask(1, 4).count, 4);
        let shed = round(&mut controller, 4.);
        // The two unused reservations transfer in the same round, before the two active jobs finish.
        assert_eq!((shed[0].node_limit, shed[1].node_limit), (0, 6));
        assert_eq!(ask(0, 1).count, 0);
        // Take over as a new controller while the two accepted jobs are merging.
        let mut recovered = QueryWindow::default();
        let after_revoke = round(&mut recovered, 4.);
        assert_eq!(
            (after_revoke[1].cluster_limit, after_revoke[1].node_limit),
            (8, 6)
        );
        assert_eq!(ask(1, 10).count, 2);
        assert_eq!(
            nodes.iter().map(|n| n.0.lock().occupied()).sum::<usize>(),
            8
        );
        drop(jobs);
        let after_merge = round(&mut recovered, 4.);
        assert_eq!(after_merge[1].node_limit, 8);
        assert_eq!(ask(1, 10).count, 2);
    }

    #[test]
    fn concurrent_senders_cannot_multiply_the_node_limit() {
        let pool = PermitPool::default();
        let collection = Uuid::new_v4();
        let now = Instant::now();
        pool.set_test_window(40, now);
        std::thread::scope(|scope| {
            let mut senders = Vec::new();
            for _ in 0..8 {
                senders.push(scope.spawn(|| {
                    pool.grant(
                        collection,
                        identity(),
                        1,
                        32,
                        true,
                        Duration::from_millis(50),
                        || {
                            (
                                now,
                                StartBudget {
                                    weight: 10.,
                                    limit: 80,
                                    navigation_queued: 0,
                                },
                            )
                        },
                    )
                    .count
                }));
            }
            assert_eq!(
                senders
                    .into_iter()
                    .map(|t| t.join().unwrap())
                    .sum::<usize>(),
                40
            );
        });
    }
}
