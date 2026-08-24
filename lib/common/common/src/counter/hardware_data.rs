use std::ops::Add;

/// Contains all hardware metrics. Only serves as value holding structure without any semantics.
#[derive(Copy, Clone, Default)]
pub struct HardwareData {
    pub cpu: usize,
    /// Sum of measured worker thread CPU time for this request, in microseconds.
    pub cpu_time_us: usize,
    /// Sum of wall time inside measured worker tasks, in microseconds.
    pub cpu_wall_time_us: usize,
    /// Number of HNSW graph-node expansion events performed while serving a request.
    ///
    /// This is an algorithmic search-work counter rather than a hardware resource
    /// counter. It lives here so distributed requests can aggregate it through the
    /// same per-request accounting path as dense-vector scoring work.
    pub graph_nodes_visited: usize,
    pub payload_io_read: usize,
    pub payload_io_write: usize,
    pub vector_io_read: usize,
    pub vector_io_write: usize,
    pub payload_index_io_read: usize,
    pub payload_index_io_write: usize,
}

impl Add for HardwareData {
    type Output = HardwareData;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            cpu: self.cpu + rhs.cpu,
            cpu_time_us: self.cpu_time_us + rhs.cpu_time_us,
            cpu_wall_time_us: self.cpu_wall_time_us + rhs.cpu_wall_time_us,
            graph_nodes_visited: self.graph_nodes_visited + rhs.graph_nodes_visited,
            payload_io_read: self.payload_io_read + rhs.payload_io_read,
            payload_io_write: self.payload_io_write + rhs.payload_io_write,
            vector_io_read: self.vector_io_read + rhs.vector_io_read,
            vector_io_write: self.vector_io_write + rhs.vector_io_write,
            payload_index_io_read: self.payload_index_io_read + rhs.payload_index_io_read,
            payload_index_io_write: self.payload_index_io_write + rhs.payload_index_io_write,
        }
    }
}
