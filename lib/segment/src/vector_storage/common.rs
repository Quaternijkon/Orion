use std::sync::atomic::{AtomicBool, Ordering};

#[cfg(feature = "rocksdb")]
use serde::{Deserialize, Serialize};

static ASYNC_SCORER: AtomicBool = AtomicBool::new(false);

pub fn set_async_scorer(async_scorer: bool) {
    ASYNC_SCORER.store(async_scorer, Ordering::Relaxed);
}

pub fn get_async_scorer() -> bool {
    ASYNC_SCORER.load(Ordering::Relaxed)
}

/// Storage type for RocksDB based storage
#[derive(Debug, Deserialize, Serialize, Clone)]
#[cfg(feature = "rocksdb")]
pub struct StoredRecord<T> {
    pub deleted: bool,
    pub vector: T,
}

/// Minimal number of bytes we read from disk in one go
/// WARN: this might be system dependent, so we assume 4Kb, which might be wrong
/// ToDo: read this from system
pub const PAGE_SIZE_BYTES: usize = 4096;

/// Number of vectors we read from storage in one batch
/// in case we need to score an iterator of vector ids
pub const VECTOR_READ_BATCH_SIZE: usize = 64;

/// Number of vector scores between issuing a software prefetch and consuming the vector.
///
/// HNSW neighbor batches are made of random vector IDs. A short fixed look-ahead gives the CPU
/// time to fetch the first cache line without pulling the complete batch into L1 at once.
pub const DENSE_VECTOR_PREFETCH_DISTANCE: usize = 4;

/// Prefetch the first cache line of a dense vector into the closest cache level.
///
/// This is deliberately a best-effort hint. Unsupported architectures keep the same access path,
/// and an empty slice is never passed to an architecture intrinsic.
#[inline]
pub fn prefetch_dense_vector<T>(vector: &[T]) {
    if vector.is_empty() {
        return;
    }

    #[cfg(target_arch = "x86")]
    unsafe {
        std::arch::x86::_mm_prefetch(vector.as_ptr().cast::<i8>(), std::arch::x86::_MM_HINT_T0);
    }

    #[cfg(target_arch = "x86_64")]
    unsafe {
        std::arch::x86_64::_mm_prefetch(
            vector.as_ptr().cast::<i8>(),
            std::arch::x86_64::_MM_HINT_T0,
        );
    }

    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    let _ = vector;
}

#[cfg(debug_assertions)]
pub const CHUNK_SIZE: usize = 512 * 1024;

/// Vector storage chunk size in bytes
#[cfg(not(debug_assertions))]
pub const CHUNK_SIZE: usize = 32 * 1024 * 1024;
