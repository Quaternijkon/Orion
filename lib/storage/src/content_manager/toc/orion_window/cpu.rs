//! Busy time on the CPUs assigned to the service, sampled only in control rounds.
use std::time::Duration;
#[cfg(target_os = "linux")]
use std::time::Instant;

#[derive(Default)]
pub(in crate::content_manager::toc) struct AssignedCpu {
    #[cfg(target_os = "linux")]
    previous: Option<(Instant, Vec<CoreTicks>)>,
}

#[cfg(target_os = "linux")]
struct CoreTicks {
    id: usize,
    busy: u64,
    total: u64,
}

impl AssignedCpu {
    #[cfg(target_os = "linux")]
    pub(super) fn sample(&mut self, capacity: usize, freshness: Duration) -> Option<f64> {
        let status = std::fs::read_to_string("/proc/self/status").ok()?;
        let list = status
            .lines()
            .find_map(|l| l.strip_prefix("Cpus_allowed_list:"))?;
        let previous = self.previous.take();
        let allowed = cpu_ids(list.trim())?;
        // A smaller logical CPU budget has no identified physical subset.
        if allowed.len() != capacity {
            return None;
        }
        let ticks = core_ticks(&std::fs::read_to_string("/proc/stat").ok()?, &allowed)?;
        let now = Instant::now();
        let busy = previous
            .filter(|(at, _)| now - *at <= freshness)
            .and_then(|(_, old)| occupied_cores(&old, &ticks));
        self.previous = Some((now, ticks));
        busy
    }

    #[cfg(not(target_os = "linux"))]
    pub(super) fn sample(&mut self, _capacity: usize, _freshness: Duration) -> Option<f64> {
        None
    }
}

#[cfg(target_os = "linux")]
fn cpu_ids(list: &str) -> Option<Vec<usize>> {
    let mut ids = Vec::new();
    for range in list.split(',') {
        let (first, last) = range.split_once('-').unwrap_or((range, range));
        let first: usize = first.parse().ok()?;
        let last: usize = last.parse().ok()?;
        if first > last {
            return None;
        }
        ids.extend(first..=last);
    }
    Some(ids)
}

#[cfg(target_os = "linux")]
fn core_ticks(stat: &str, allowed: &[usize]) -> Option<Vec<CoreTicks>> {
    let mut cores = Vec::with_capacity(allowed.len());
    for line in stat.lines() {
        let mut fields = line.split_whitespace();
        let Some(id) = fields
            .next()
            .and_then(|s| s.strip_prefix("cpu"))
            .and_then(|s| s.parse().ok())
        else {
            continue;
        };
        if allowed.binary_search(&id).is_err() {
            continue;
        }
        let mut counts = [0u64; 8];
        for count in &mut counts {
            *count = fields.next()?.parse().ok()?;
        }
        let [user, nice, system, _idle, _iowait, irq, softirq, _steal] = counts;
        cores.push(CoreTicks {
            id,
            busy: user + nice + system + irq + softirq,
            total: counts.iter().sum(),
        });
    }
    (cores.len() == allowed.len()).then_some(cores)
}

#[cfg(target_os = "linux")]
fn occupied_cores(before: &[CoreTicks], after: &[CoreTicks]) -> Option<f64> {
    if before.len() != after.len() {
        return None;
    }
    let mut busy = 0.;
    for (old, new) in before.iter().zip(after) {
        let total = new.total.checked_sub(old.total)?;
        let active = new.busy.checked_sub(old.busy)?;
        if old.id != new.id || total == 0 || active > total {
            return None;
        }
        busy += active as f64 / total as f64;
    }
    Some(busy)
}

#[cfg(all(test, target_os = "linux"))]
mod tests {
    use super::*;

    #[test]
    fn orion_window_cpu_counts_only_assigned_cores_and_excludes_idle_iowait_and_steal() {
        let ids = cpu_ids("1,3-4").unwrap();
        assert_eq!(ids, [1, 3, 4]);
        let before = core_ticks(
            "cpu1 0 0 0 0 0 0 0 0\ncpu3 0 0 0 0 0 0 0 0\ncpu4 0 0 0 0 0 0 0 0",
            &ids,
        )
        .unwrap();
        let after = core_ticks("cpu0 200 0 0 0 0 0 0 0\ncpu1 2 0 2 3 1 1 1 0 0 0\ncpu3 0 0 0 5 0 0 0 5\ncpu4 0 0 5 0 0 0 0 0", &ids).unwrap();
        assert!((occupied_cores(&before, &after).unwrap() - 1.6).abs() < 1e-9);
        assert!(occupied_cores(&after, &after).is_none());
        assert!(occupied_cores(&before[..2], &after).is_none());
    }
}
