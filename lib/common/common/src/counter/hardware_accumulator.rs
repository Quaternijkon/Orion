use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use super::hardware_counter::HardwareCounterCell;
use super::hardware_data::HardwareData;
use crate::cpu_utilization::CpuUtilization;

/// Data structure, that routes hardware measurement counters to specific location.
/// Shared drain MUST NOT create its own counters, but only hold a reference to the existing one,
/// as it doesn't provide any checks on drop.
#[derive(Debug)]
pub struct HwSharedDrain {
    pub(crate) cpu_counter: AtomicUsize,
    pub(crate) cpu_time_us_counter: AtomicUsize,
    pub(crate) cpu_wall_time_us_counter: AtomicUsize,
    pub(crate) graph_nodes_visited_counter: AtomicUsize,
    pub(crate) payload_io_read_counter: AtomicUsize,
    pub(crate) payload_io_write_counter: AtomicUsize,
    pub(crate) payload_index_io_read_counter: AtomicUsize,
    pub(crate) payload_index_io_write_counter: AtomicUsize,
    pub(crate) vector_io_read_counter: AtomicUsize,
    pub(crate) vector_io_write_counter: AtomicUsize,
}

impl HwSharedDrain {
    pub fn get_cpu(&self) -> usize {
        self.cpu_counter.load(Ordering::Relaxed)
    }

    pub fn get_cpu_time_us(&self) -> usize {
        self.cpu_time_us_counter.load(Ordering::Relaxed)
    }

    pub fn get_cpu_wall_time_us(&self) -> usize {
        self.cpu_wall_time_us_counter.load(Ordering::Relaxed)
    }

    pub fn get_graph_nodes_visited(&self) -> usize {
        self.graph_nodes_visited_counter.load(Ordering::Relaxed)
    }

    pub fn get_payload_io_read(&self) -> usize {
        self.payload_io_read_counter.load(Ordering::Relaxed)
    }

    pub fn get_payload_io_write(&self) -> usize {
        self.payload_io_write_counter.load(Ordering::Relaxed)
    }

    pub fn get_payload_index_io_read(&self) -> usize {
        self.payload_index_io_read_counter.load(Ordering::Relaxed)
    }

    pub fn get_payload_index_io_write(&self) -> usize {
        self.payload_index_io_write_counter.load(Ordering::Relaxed)
    }

    pub fn get_vector_io_write(&self) -> usize {
        self.vector_io_write_counter.load(Ordering::Relaxed)
    }

    pub fn get_vector_io_read(&self) -> usize {
        self.vector_io_read_counter.load(Ordering::Relaxed)
    }

    /// Accumulates all values from `src` into this HwSharedDrain.
    fn accumulate_from_hw_data(&self, src: HardwareData) {
        let HwSharedDrain {
            cpu_counter,
            cpu_time_us_counter,
            cpu_wall_time_us_counter,
            graph_nodes_visited_counter,
            payload_io_read_counter,
            payload_io_write_counter,
            payload_index_io_read_counter,
            payload_index_io_write_counter,
            vector_io_read_counter,
            vector_io_write_counter,
        } = self;

        cpu_counter.fetch_add(src.cpu, Ordering::Relaxed);
        cpu_time_us_counter.fetch_add(src.cpu_time_us, Ordering::Relaxed);
        cpu_wall_time_us_counter.fetch_add(src.cpu_wall_time_us, Ordering::Relaxed);
        graph_nodes_visited_counter.fetch_add(src.graph_nodes_visited, Ordering::Relaxed);
        payload_io_read_counter.fetch_add(src.payload_io_read, Ordering::Relaxed);
        payload_io_write_counter.fetch_add(src.payload_io_write, Ordering::Relaxed);
        payload_index_io_read_counter.fetch_add(src.payload_index_io_read, Ordering::Relaxed);
        payload_index_io_write_counter.fetch_add(src.payload_index_io_write, Ordering::Relaxed);
        vector_io_read_counter.fetch_add(src.vector_io_read, Ordering::Relaxed);
        vector_io_write_counter.fetch_add(src.vector_io_write, Ordering::Relaxed);
    }
}

impl Default for HwSharedDrain {
    fn default() -> Self {
        Self {
            cpu_counter: AtomicUsize::new(0),
            cpu_time_us_counter: AtomicUsize::new(0),
            cpu_wall_time_us_counter: AtomicUsize::new(0),
            graph_nodes_visited_counter: AtomicUsize::new(0),
            payload_io_read_counter: AtomicUsize::new(0),
            payload_io_write_counter: AtomicUsize::new(0),
            payload_index_io_read_counter: AtomicUsize::new(0),
            payload_index_io_write_counter: AtomicUsize::new(0),
            vector_io_read_counter: AtomicUsize::new(0),
            vector_io_write_counter: AtomicUsize::new(0),
        }
    }
}

/// A "slow" but thread-safe accumulator for measurement results of `HardwareCounterCell` values.
/// This type is completely reference counted and clones of this type will read/write the same values as their origin structure.
#[derive(Debug)]
pub struct HwMeasurementAcc {
    request_drain: Arc<HwSharedDrain>,
    metrics_drain: Arc<HwSharedDrain>,
    /// If this is set to true, the accumulator will not accumulate any values.
    disposable: bool,
    cpu_utilization: CpuUtilization,
}

impl HwMeasurementAcc {
    #[cfg(feature = "testing")]
    pub fn new() -> Self {
        Self {
            request_drain: Arc::new(HwSharedDrain::default()),
            metrics_drain: Arc::new(HwSharedDrain::default()),
            disposable: false,
            cpu_utilization: CpuUtilization::new(),
        }
    }

    /// Create a disposable accumulator, which will not accumulate any values.
    /// WARNING: This is intended for specific internal use-cases only.
    /// DO NOT use it in tests or if you don't know what you're doing.
    /// Prefer `new` to be used in tests.
    pub fn disposable() -> Self {
        Self {
            request_drain: Arc::new(HwSharedDrain::default()),
            metrics_drain: Arc::new(HwSharedDrain::default()),
            disposable: true,
            cpu_utilization: CpuUtilization::new(),
        }
    }

    /// Same as `disposable`, but expected to be used in edge crate for better code navigation.
    pub fn disposable_edge() -> Self {
        Self::disposable()
    }

    pub fn is_disposable(&self) -> bool {
        self.disposable
    }

    /// Returns a new `HardwareCounterCell` that accumulates it's measurements to the same parent than this `HwMeasurementAcc`.
    pub fn get_counter_cell(&self) -> HardwareCounterCell {
        HardwareCounterCell::new_with_accumulator(self.clone())
    }

    pub fn new_with_metrics_drain(metrics_drain: Arc<HwSharedDrain>) -> Self {
        Self {
            request_drain: Arc::new(HwSharedDrain::default()),
            metrics_drain,
            disposable: false,
            cpu_utilization: CpuUtilization::new(),
        }
    }

    pub fn cpu_utilization(&self) -> CpuUtilization {
        self.cpu_utilization.clone()
    }

    pub fn accumulate<T: Into<HardwareData>>(&self, src: T) {
        let src = src.into();
        self.request_drain.accumulate_from_hw_data(src);
        self.metrics_drain.accumulate_from_hw_data(src);
    }

    /// Accumulate usage values for request drain only.
    /// This is useful if we want to report usage, which happened on another machine
    /// So we don't want to accumulate the same usage on the current machine second time
    pub fn accumulate_request<T: Into<HardwareData>>(&self, src: T) {
        let src = src.into();
        self.request_drain.accumulate_from_hw_data(src);
    }

    pub fn get_cpu(&self) -> usize {
        self.request_drain.get_cpu()
    }

    pub fn get_graph_nodes_visited(&self) -> usize {
        self.request_drain.get_graph_nodes_visited()
    }

    pub fn get_payload_io_read(&self) -> usize {
        self.request_drain.get_payload_io_read()
    }

    pub fn get_payload_io_write(&self) -> usize {
        self.request_drain.get_payload_io_write()
    }

    pub fn get_payload_index_io_read(&self) -> usize {
        self.request_drain.get_payload_index_io_read()
    }

    pub fn get_payload_index_io_write(&self) -> usize {
        self.request_drain.get_payload_index_io_write()
    }

    pub fn get_vector_io_read(&self) -> usize {
        self.request_drain.get_vector_io_read()
    }

    pub fn get_vector_io_write(&self) -> usize {
        self.request_drain.get_vector_io_write()
    }

    pub fn hw_data(&self) -> HardwareData {
        let HwSharedDrain {
            cpu_counter,
            cpu_time_us_counter,
            cpu_wall_time_us_counter,
            graph_nodes_visited_counter,
            payload_io_read_counter,
            payload_io_write_counter,
            payload_index_io_read_counter,
            payload_index_io_write_counter,
            vector_io_read_counter,
            vector_io_write_counter,
        } = self.request_drain.as_ref();

        HardwareData {
            cpu: cpu_counter.load(Ordering::Relaxed),
            cpu_time_us: cpu_time_us_counter.load(Ordering::Relaxed)
                + self.cpu_utilization.cpu_time_us() as usize,
            cpu_wall_time_us: cpu_wall_time_us_counter.load(Ordering::Relaxed)
                + self.cpu_utilization.wall_time_us() as usize,
            graph_nodes_visited: graph_nodes_visited_counter.load(Ordering::Relaxed),
            payload_io_read: payload_io_read_counter.load(Ordering::Relaxed),
            payload_io_write: payload_io_write_counter.load(Ordering::Relaxed),
            vector_io_read: vector_io_read_counter.load(Ordering::Relaxed),
            vector_io_write: vector_io_write_counter.load(Ordering::Relaxed),
            payload_index_io_read: payload_index_io_read_counter.load(Ordering::Relaxed),
            payload_index_io_write: payload_index_io_write_counter.load(Ordering::Relaxed),
        }
    }
}

#[cfg(feature = "testing")]
impl Default for HwMeasurementAcc {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for HwMeasurementAcc {
    fn clone(&self) -> Self {
        Self {
            request_drain: self.request_drain.clone(),
            metrics_drain: self.metrics_drain.clone(),
            disposable: self.disposable,
            cpu_utilization: self.cpu_utilization.clone(),
        }
    }
}

#[cfg(test)]
mod graph_visit_tests {
    use super::HwMeasurementAcc;

    #[test]
    fn graph_node_visits_accumulate_across_forked_cells() {
        let accumulator = HwMeasurementAcc::new();
        {
            let parent = accumulator.get_counter_cell();
            let child = parent.fork();
            parent.graph_nodes_visited_counter().incr_delta(2);
            child.graph_nodes_visited_counter().incr_delta(5);
        }
        assert_eq!(accumulator.get_graph_nodes_visited(), 7);
    }
}
