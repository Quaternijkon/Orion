//! Node-wide CPU and queue accounting for worker searches and global navigation.
//!
//! Guards live inside the actual blocking task, so waiting for an RPC is never
//! charged as CPU work. All collections share the same machine-load estimate.
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

#[derive(Clone, Copy, Debug)]
pub enum SearchRole {
    Worker,
    Navigation,
}

#[derive(Default, Debug)]
struct Counters {
    cpu_ns: AtomicU64,
    queued: AtomicUsize,
    active: AtomicUsize,
}

#[derive(Debug)]
struct LoadState {
    enabled: AtomicBool,
    worker: Counters,
    navigation: Counters,
    admitted: AtomicUsize,
    sampler: Mutex<Sampler>,
}

#[derive(Debug)]
struct Sampler {
    at: Instant,
    process_ns: u64,
    worker_ns: u64,
    navigation_ns: u64,
    cpu: f64,
    worker: f64,
    navigation: f64,
}

/// CPU values are occupied cores, not percentages or accumulated request time.
#[derive(Clone, Copy, Debug, Default)]
pub struct SearchLoad {
    pub cpu_cores: f64,
    pub worker_cores: f64,
    pub navigation_cores: f64,
    pub worker_queued: usize,
    pub navigation_queued: usize,
    pub active_tasks: usize,
    pub admitted_navigation: usize,
}

fn global() -> &'static Arc<LoadState> {
    static STATE: OnceLock<Arc<LoadState>> = OnceLock::new();
    STATE.get_or_init(|| {
        Arc::new(LoadState {
            enabled: AtomicBool::new(false),
            worker: Counters::default(),
            navigation: Counters::default(),
            admitted: AtomicUsize::new(0),
            sampler: Mutex::new(Sampler {
                at: Instant::now(),
                process_ns: process_cpu_ns(),
                worker_ns: 0,
                navigation_ns: 0,
                cpu: 0.,
                worker: 0.,
                navigation: 0.,
            }),
        })
    })
}

pub fn enable() {
    global().enabled.store(true, Ordering::Relaxed);
}

impl LoadState {
    fn counters(&self, role: SearchRole) -> &Counters {
        match role {
            SearchRole::Worker => &self.worker,
            SearchRole::Navigation => &self.navigation,
        }
    }
}

/// Dropping a cancelled queued closure releases its queue reservation as well.
pub struct QueuedSearch {
    state: Option<Arc<LoadState>>,
    role: SearchRole,
}
pub struct RunningSearch {
    state: Arc<LoadState>,
    role: SearchRole,
    started: u64,
    // A thread CPU clock must be read on the same thread at start and finish.
    _same_thread: std::marker::PhantomData<*mut ()>,
}

pub fn queued(role: SearchRole) -> QueuedSearch {
    let state = global();
    let state = state.enabled.load(Ordering::Relaxed).then(|| {
        state.counters(role).queued.fetch_add(1, Ordering::Relaxed);
        Arc::clone(state)
    });
    QueuedSearch { state, role }
}

impl QueuedSearch {
    pub fn start(mut self) -> Option<RunningSearch> {
        self.state.take().map(|state| {
            state
                .counters(self.role)
                .queued
                .fetch_sub(1, Ordering::Relaxed);
            state
                .counters(self.role)
                .active
                .fetch_add(1, Ordering::Relaxed);
            RunningSearch {
                state,
                role: self.role,
                started: super::cpu_utilization::thread_cpu_time_ns(),
                _same_thread: std::marker::PhantomData,
            }
        })
    }
}
impl Drop for QueuedSearch {
    fn drop(&mut self) {
        if let Some(state) = &self.state {
            state
                .counters(self.role)
                .queued
                .fetch_sub(1, Ordering::Relaxed);
        }
    }
}
impl Drop for RunningSearch {
    fn drop(&mut self) {
        let counters = self.state.counters(self.role);
        counters.cpu_ns.fetch_add(
            super::cpu_utilization::thread_cpu_time_ns().saturating_sub(self.started),
            Ordering::Relaxed,
        );
        counters.active.fetch_sub(1, Ordering::Relaxed);
    }
}

pub struct NavigationAdmission {
    state: Arc<LoadState>,
    units: usize,
}

/// Remote coordinators reject excess work promptly; original ingress requests
/// may always fall back to local execution without dropping a query.
pub fn admit_navigation(units: usize, limit: Option<usize>) -> Option<NavigationAdmission> {
    let state = Arc::clone(global());
    let units = units.max(1);
    let mut old = state.admitted.load(Ordering::Relaxed);
    loop {
        let new = old.checked_add(units)?;
        if limit.is_some_and(|limit| new > limit) {
            return None;
        }
        match state
            .admitted
            .compare_exchange_weak(old, new, Ordering::Relaxed, Ordering::Relaxed)
        {
            Ok(_) => return Some(NavigationAdmission { state, units }),
            Err(actual) => old = actual,
        }
    }
}
impl Drop for NavigationAdmission {
    fn drop(&mut self) {
        self.state.admitted.fetch_sub(self.units, Ordering::Relaxed);
    }
}

pub fn snapshot(capacity: usize) -> SearchLoad {
    let state = global();
    let now = Instant::now();
    let mut sampler = state.sampler.lock().unwrap_or_else(|p| p.into_inner());
    let elapsed = now.duration_since(sampler.at);
    if elapsed >= Duration::from_millis(100) {
        let process = process_cpu_ns();
        let worker = state.worker.cpu_ns.load(Ordering::Relaxed);
        let navigation = state.navigation.cpu_ns.load(Ordering::Relaxed);
        let seconds = elapsed.as_secs_f64();
        let bound = capacity.max(1) as f64;
        let cpu =
            (process.saturating_sub(sampler.process_ns) as f64 / 1e9 / seconds).clamp(0., bound);
        let mut worker_rate = worker.saturating_sub(sampler.worker_ns) as f64 / 1e9 / seconds;
        let mut navigation_rate =
            navigation.saturating_sub(sampler.navigation_ns) as f64 / 1e9 / seconds;
        let role_total = worker_rate + navigation_rate;
        if role_total > cpu {
            worker_rate *= cpu / role_total;
            navigation_rate *= cpu / role_total;
        }
        // Normalize deferred task completions and smooth short CPU bursts.
        sampler.cpu = 0.7 * sampler.cpu + 0.3 * cpu;
        sampler.worker = 0.7 * sampler.worker + 0.3 * worker_rate;
        sampler.navigation = 0.7 * sampler.navigation + 0.3 * navigation_rate;
        sampler.at = now;
        sampler.process_ns = process;
        sampler.worker_ns = worker;
        sampler.navigation_ns = navigation;
    }
    SearchLoad {
        cpu_cores: sampler.cpu,
        worker_cores: sampler.worker,
        navigation_cores: sampler.navigation,
        worker_queued: state.worker.queued.load(Ordering::Relaxed),
        navigation_queued: state.navigation.queued.load(Ordering::Relaxed),
        active_tasks: state.worker.active.load(Ordering::Relaxed)
            + state.navigation.active.load(Ordering::Relaxed),
        admitted_navigation: state.admitted.load(Ordering::Relaxed),
    }
}

#[cfg(target_os = "linux")]
fn process_cpu_ns() -> u64 {
    let mut ts = nix::libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: the pointer references a valid timespec and the process clock is read-only.
    if unsafe { nix::libc::clock_gettime(nix::libc::CLOCK_PROCESS_CPUTIME_ID, &mut ts) } == 0 {
        ts.tv_sec as u64 * 1_000_000_000 + ts.tv_nsec as u64
    } else {
        0
    }
}
#[cfg(not(target_os = "linux"))]
fn process_cpu_ns() -> u64 {
    0
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn admission_releases_and_rejects_over_capacity() {
        let start = global().admitted.load(Ordering::Relaxed);
        let held = admit_navigation(3, Some(start + 3)).unwrap();
        assert!(admit_navigation(1, Some(start + 3)).is_none());
        drop(held);
        assert_eq!(global().admitted.load(Ordering::Relaxed), start);
    }

    #[test]
    fn queued_and_running_guards_release_on_cancel_and_unwind() {
        enable();
        let state = global();
        let before = state.worker.queued.load(Ordering::Relaxed);
        let active = state.worker.active.load(Ordering::Relaxed);
        let cancelled = queued(SearchRole::Worker);
        assert_eq!(state.worker.queued.load(Ordering::Relaxed), before + 1);
        drop(cancelled);
        assert_eq!(state.worker.queued.load(Ordering::Relaxed), before);
        let result = std::panic::catch_unwind(|| {
            let running = queued(SearchRole::Worker).start().unwrap();
            assert_eq!(state.worker.queued.load(Ordering::Relaxed), before);
            assert_eq!(state.worker.active.load(Ordering::Relaxed), active + 1);
            let _guard = running;
            panic!("cancel running search");
        });
        assert!(result.is_err());
        assert_eq!(state.worker.queued.load(Ordering::Relaxed), before);
        assert_eq!(state.worker.active.load(Ordering::Relaxed), active);
    }
}
