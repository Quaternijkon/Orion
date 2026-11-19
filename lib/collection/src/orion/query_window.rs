//! Control the total number of direct Orion queries from admission through merge.
use std::collections::HashMap;

use api::grpc::qdrant::{OrionQueryWindowAllocation, OrionQueryWindowReport};

use super::navigation_balance::NavigationBalanceConfig;

#[derive(Default)]
pub struct QueryWindow {
    limit: u64,
    previous_queues: HashMap<u64, QueueSample>,
}

#[derive(Default)]
struct QueueSample {
    worker: Option<u64>,
    navigation: Option<u64>,
}

impl QueueSample {
    fn from_node(n: &OrionQueryWindowReport) -> Self {
        Self {
            worker: (n.worker_queued > 0).then_some(n.worker_queue_drains),
            navigation: (n.navigation_queued > 0).then_some(n.navigation_queue_drains),
        }
    }

    fn still_waiting(&self, previous: &Self) -> bool {
        self.worker.is_some() && self.worker == previous.worker
            || self.navigation.is_some() && self.navigation == previous.navigation
    }
}

impl QueryWindow {
    pub fn missed_sample(&mut self) {
        self.previous_queues.clear();
    }

    /// Called once per control interval, only after every cluster member has replied.
    pub fn targets(
        &mut self,
        nodes: &[OrionQueryWindowReport],
        config: &NavigationBalanceConfig,
    ) -> Vec<OrionQueryWindowAllocation> {
        if self.limit == 0 {
            // On a leader change recover the observed target. Cold start at one
            // query per CPU, then let real admission pressure grow the window.
            self.limit = nodes
                .iter()
                .map(|n| n.cluster_limit)
                .filter(|&n| n > 0)
                .min()
                .unwrap_or(0);
            if self.limit == 0 {
                self.limit = nodes
                    .iter()
                    .map(|n| n.cpu_capacity)
                    .fold(0, u64::saturating_add)
                    .max(1);
            }
        }
        let utilization = |n: &OrionQueryWindowReport| n.cpu_cores / n.cpu_capacity as f64;
        let hottest = nodes.iter().map(utilization).fold(0., f64::max);
        let mut backlog = false;
        for node in nodes {
            let sample = QueueSample::from_node(node);
            let previous = self.previous_queues.entry(node.peer_id).or_default();
            backlog |= utilization(node) == hottest && sample.still_waiting(previous);
            *previous = sample;
        }
        let waiting = nodes.iter().any(|n| n.waiting);
        let occupied = nodes
            .iter()
            .map(|n| n.peak_occupied)
            .fold(0, u64::saturating_add);
        if backlog {
            let step = (self.limit as f64 * config.adjustment_rate).ceil() as u64;
            self.limit = self.limit.saturating_sub(step).max(1);
        } else if hottest < config.target_utilization && waiting && occupied >= self.limit {
            let gain = (config.target_utilization - hottest).min(config.adjustment_rate);
            self.limit = self
                .limit
                .saturating_add((self.limit as f64 * gain).ceil() as u64);
        }

        let cool_destination = nodes.iter().any(|n| {
            n.demand && n.coordinator_room > 0. && utilization(n) < config.target_utilization
        });
        let mut weights: Vec<_> = nodes
            .iter()
            .map(|n| {
                if !n.demand || (cool_destination && utilization(n) >= config.target_utilization) {
                    0.
                } else {
                    n.coordinator_room
                }
            })
            .collect();
        if weights.iter().all(|w| *w == 0.) {
            // Keep the window available for recovery. The independent local
            // worker-first start budget can still veto every new start.
            for (weight, node) in weights.iter_mut().zip(nodes) {
                if node.demand {
                    *weight = node.cpu_capacity as f64;
                }
            }
        }
        let mut desired = apportion(self.limit, &weights);
        let wait_weights: Vec<_> = nodes
            .iter()
            .zip(&weights)
            .map(|(n, &w)| if n.waiting { w } else { 0. })
            .collect();
        if wait_weights.iter().any(|&w| w > 0.) {
            // Lend unused capacity before growing the cluster window. Keep one
            // slot for a newly interested sender until its first actual search.
            let mut spare = 0;
            for (node, share) in nodes.iter().zip(&mut desired) {
                if !node.waiting {
                    let unused = share.saturating_sub(node.peak_occupied.max(1));
                    *share -= unused;
                    spare += unused;
                }
            }
            for (share, extra) in desired.iter_mut().zip(apportion(spare, &wait_weights)) {
                *share += extra;
            }
        }

        nodes
            .iter()
            .zip(desired)
            .map(|(node, node_limit)| OrionQueryWindowAllocation {
                fence: node.fence.clone(),
                node_limit,
                cluster_limit: self.limit,
            })
            .collect()
    }

    /// Bound increases by the rights actually free after donors acknowledge their reductions.
    /// A donor with more accepted queries than its new limit keeps those slots until completion.
    pub fn bound_growth(
        &self,
        nodes: &[OrionQueryWindowReport],
        grants: &mut [OrionQueryWindowAllocation],
    ) {
        let exposure = nodes
            .iter()
            .map(|n| n.node_limit.max(n.occupied))
            .fold(0, u64::saturating_add);
        let free = self.limit.saturating_sub(exposure);
        let increases: Vec<_> = nodes
            .iter()
            .zip(grants.iter())
            .map(|(node, grant)| grant.node_limit.saturating_sub(node.node_limit))
            .collect();
        if increases.iter().copied().fold(0, u64::saturating_add) <= free {
            return;
        }
        let weights: Vec<_> = increases.iter().map(|&n| n as f64).collect();
        for ((node, grant), added) in nodes.iter().zip(grants).zip(apportion(free, &weights)) {
            grant.node_limit = grant.node_limit.min(node.node_limit.saturating_add(added));
        }
    }

    #[cfg(test)]
    pub(crate) fn allocate(
        &mut self,
        nodes: &[OrionQueryWindowReport],
        cfg: &NavigationBalanceConfig,
    ) -> Vec<OrionQueryWindowAllocation> {
        let mut grants = self.targets(nodes, cfg);
        self.bound_growth(nodes, &mut grants);
        grants
    }
}

/// Round cumulative shares so integer allocations sum to the window, including
/// small windows where rounding each peer independently loses capacity.
fn apportion(total: u64, weights: &[f64]) -> Vec<u64> {
    let sum: f64 = weights.iter().sum();
    let mut cumulative = 0.;
    let mut assigned = 0;
    weights
        .iter()
        .map(|w| {
            cumulative += w;
            let end = if sum > 0. {
                (total as f64 * (cumulative / sum)).round() as u64
            } else {
                0
            };
            let next = end.min(total).saturating_sub(assigned);
            assigned += next;
            next
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(peer: u64, cpu: f64, share: u64) -> OrionQueryWindowReport {
        OrionQueryWindowReport {
            peer_id: peer,
            cpu_capacity: 20,
            cpu_cores: cpu,
            coordinator_room: 10.,
            node_limit: share,
            cluster_limit: 20,
            occupied: share,
            peak_occupied: share,
            demand: true,
            ..Default::default()
        }
    }

    #[test]
    fn grows_only_below_target_with_waiting_queries_and_a_full_window() {
        let cfg = NavigationBalanceConfig {
            target_utilization: 0.8,
            ..Default::default()
        };
        let mut ctl = QueryWindow::default();
        let mut peers = [node(1, 12., 10), node(2, 7., 10)];
        assert_eq!(ctl.allocate(&peers, &cfg)[0].cluster_limit, 20);
        peers[0].waiting = true;
        peers[1].occupied = 2;
        peers[1].peak_occupied = 2;
        let lent = ctl.allocate(&peers, &cfg);
        assert_eq!((lent[0].cluster_limit, lent[1].node_limit), (20, 2));
        peers[1].node_limit = 2;
        let transferred = ctl.allocate(&peers, &cfg);
        assert_eq!(
            (transferred[0].node_limit, transferred[1].node_limit),
            (18, 2)
        );
        peers[0].node_limit = 18;
        peers[0].peak_occupied = 18;
        peers[0].occupied = 18;
        let increased = ctl.allocate(&peers, &cfg);
        assert_eq!(increased[0].cluster_limit, 23);
        assert_eq!(increased.iter().map(|g| g.node_limit).sum::<u64>(), 23);
    }

    #[test]
    fn shift_rights_only_after_the_hot_node_acknowledges_and_drains() {
        let cfg = NavigationBalanceConfig::default();
        let mut ctl = QueryWindow::default();
        let mut peers = [node(1, 19., 10), node(2, 8., 10)];
        peers[1].waiting = true;
        let first = ctl.allocate(&peers, &cfg);
        assert_eq!(
            (
                first[0].node_limit,
                first[1].node_limit,
                first[0].cluster_limit
            ),
            (0, 10, 20)
        );
        // Simulate an unacknowledged write. The donor may still issue all ten slots.
        assert_eq!(ctl.allocate(&peers, &cfg)[1].node_limit, 10);
        peers[0].node_limit = 0;
        peers[0].occupied = 6;
        assert_eq!(ctl.allocate(&peers, &cfg)[1].node_limit, 14);
        peers[0].coordinator_room = 0.;
        peers[0].occupied = 0;
        peers[1].node_limit = 14;
        let drained = ctl.allocate(&peers, &cfg);
        assert_eq!(
            (
                drained[0].node_limit,
                drained[1].node_limit,
                drained[0].cluster_limit
            ),
            (0, 20, 20)
        );
        // The hotspot moves to the other node. New starts can move back.
        peers[0].cpu_cores = 5.;
        peers[0].coordinator_room = 12.;
        peers[1].cpu_cores = 20.;
        peers[1].node_limit = 0;
        peers[1].occupied = 0;
        assert_eq!(ctl.allocate(&peers, &cfg)[0].node_limit, 20);
    }

    #[test]
    fn only_a_persistent_cpu_queue_on_the_hottest_node_shrinks_the_window() {
        let cfg = NavigationBalanceConfig::default();
        let mut ctl = QueryWindow::default();
        let mut peers = [node(1, 19., 10), node(2, 2., 10)];
        peers[1].worker_queued = 100;
        for _ in 0..3 {
            assert_eq!(ctl.allocate(&peers, &cfg)[0].cluster_limit, 20);
        }
        peers[0].worker_queued = 3;
        assert_eq!(ctl.allocate(&peers, &cfg)[0].cluster_limit, 20);
        // Nonzero samples at both ends can hide a queue that drained between them.
        peers[0].worker_queue_drains += 1;
        assert_eq!(ctl.allocate(&peers, &cfg)[0].cluster_limit, 20);
        ctl.missed_sample();
        assert_eq!(ctl.allocate(&peers, &cfg)[0].cluster_limit, 20);
        // CPU is now below target and there is unserved demand. Queue reduction wins.
        peers[0].cpu_cores = 17.;
        peers[0].waiting = true;
        assert_eq!(ctl.allocate(&peers, &cfg)[0].cluster_limit, 17);
        peers[0].worker_queued = 0;
        peers[0].waiting = false;
        assert_eq!(ctl.allocate(&peers, &cfg)[0].cluster_limit, 17);
    }

    #[test]
    fn an_upper_cpu_backlog_counts_but_coordinators_waiting_for_rpc_do_not() {
        let mut ctl = QueryWindow::default();
        let mut peers = [node(1, 18., 10), node(2, 12., 10)];
        for n in &mut peers {
            n.occupied = 100;
            n.peak_occupied = 100;
        }
        for _ in 0..2 {
            assert_eq!(
                ctl.allocate(&peers, &Default::default())[0].cluster_limit,
                20
            );
        }
        peers[0].navigation_queued = 1;
        assert_eq!(
            ctl.allocate(&peers, &Default::default())[0].cluster_limit,
            20
        );
        assert_eq!(
            ctl.allocate(&peers, &Default::default())[0].cluster_limit,
            17
        );
        peers[0].navigation_queue_drains += 1;
        assert_eq!(
            ctl.allocate(&peers, &Default::default())[0].cluster_limit,
            17
        );
    }

    #[test]
    fn cold_start_uses_sum_of_serving_cpus_and_integer_shares_do_not_multiply() {
        let mut nodes = [node(1, 3., 0), node(2, 5., 0)];
        for n in &mut nodes {
            n.cluster_limit = 0;
        }
        let first = QueryWindow::default().allocate(&nodes, &Default::default());
        assert_eq!((first[0].node_limit, first[1].node_limit), (20, 20));
        for n in 0..64 {
            let shares = apportion(n, &[0., 0.5, 2., 0., 9.]);
            assert_eq!(shares.iter().sum::<u64>(), n);
            assert_eq!((shares[0], shares[3]), (0, 0));
        }
    }
}
