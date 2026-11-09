//! Offline layout authority. No Python state is trusted by this module.
//!
//! Version 2 fixes RNG and scalar f64 K-Means arithmetic independently of NumPy/BLAS.
//! Version 3 preserves tied owners and accepts the last bounded refinement round.
//! Navigation still uses the Qdrant production graph and segment distance semantics.
mod kmeans;
mod sampler;
pub(crate) mod vector_input;
mod voting;

pub use sampler::sample_indices;
pub use vector_input::{VectorEncoding, VectorInput};
pub use voting::{refine_owners, strict_targets};

use segment::types::Distance;
use serde::{Deserialize, Serialize};

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;
pub const LAYOUT_ALGORITHM_VERSION: &str = "orion-layout-v3-stable-ties-bounded";

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LayoutOptions {
    pub generation: u64,
    pub shard_count: u32,
    pub distance: Distance,
    pub vector_name: String,
    pub sample_denominator: usize,
    pub sample_seed: u64,
    pub upper_m: usize,
    pub upper_ef_construct: usize,
    pub upper_seed: u64,
    pub efs: usize,
    pub upper_k: usize,
    pub upper_ef_search: usize,
    pub dynamic_ef_base: usize,
    pub dynamic_ef_factor: usize,
    pub kmeans_iterations: usize,
    pub kmeans_seed: u64,
    pub topology_iterations: usize,
    pub bundle_prefix: String,
}

impl LayoutOptions {
    pub fn validate(&self, rows: usize) -> Result<()> {
        if [
            self.generation as u128,
            self.shard_count as u128,
            self.sample_denominator as u128,
            self.upper_m as u128,
            self.upper_ef_construct as u128,
            self.efs as u128,
            self.upper_k as u128,
            self.dynamic_ef_base as u128,
            self.dynamic_ef_factor as u128,
            self.kmeans_iterations as u128,
            self.topology_iterations as u128,
        ]
        .contains(&0)
        {
            return Err("layout counts, budgets and dynamic EF factor must be positive".into());
        }
        if self.sample_denominator != self.upper_m {
            return Err("canonical N/M sampling requires sample_denominator == upper_m".into());
        }
        let upper_count = rows / self.sample_denominator;
        if upper_count < self.shard_count as usize
            || upper_count < self.upper_k
            || self.upper_ef_search < self.upper_k
        {
            return Err(
                "sample must cover shards and search top-k; runtime EF must cover top-k".into(),
            );
        }
        super::bundle::validate_filename(&self.bundle_prefix)?;
        Ok(())
    }
}

pub(crate) use kmeans::cluster;

#[cfg(test)]
mod tests;
