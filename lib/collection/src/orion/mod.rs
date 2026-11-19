//! Upper ownership and physical data placement are separate contracts.
//!
//! Each upper node has exactly one frozen owner shard. Its vector may also exist
//! as ordinary data in other shards. An upper hit activates only the owner copy
//! as a local-search entry point; other copies neither activate their shards nor
//! become entry points there. Another shard participates only when it receives
//! an upper hit owned by that shard. Once activated, local graph search may visit
//! any ordinary data copy in that shard.
//!
//! The production base uses capacity partitioning followed by construction-input
//! access-probability refinement, with equal cost per selected shard request.
//! Online routing uses whitebox selection and contribution-aware MultiEP HNSW.
//! Self-vote and decision-tree strategy selection is disabled. See
//! `tools/orion_capacity_build.md` for construction and compatibility requirements.
mod artifact;
pub mod access;
mod builder;
mod capacity;
pub mod coordinator_permits;
pub mod bundle;
pub mod cli;
pub mod clock;
pub mod deploy;
pub mod durable;
mod error;
pub mod navigation_balance;
mod navigation_rpc;
pub mod query_window;
// No alternate/oracle online selector is part of the fixed base.
// mod oracle_diagnostics;
pub mod import;
pub mod layout;
// Retained as source history only. The canonical base has one selector:
// whitebox pruning; decision-tree policies are not compiled or selectable.
// pub mod learned_core;
// pub mod learned_pruning;
pub mod whitebox_pruning;
pub mod lifecycle;
pub mod membership;
pub mod migration;
pub mod partition;
pub mod profile;
pub mod recovery;
mod router;
pub mod streaming;
pub mod streaming_client;

use std::path::{Path, PathBuf};

pub const ORION_ROUTING_ARTIFACT_DIR: &str = "orion_router";

pub fn routing_artifact_relative_path(generation: u64) -> PathBuf {
    PathBuf::from(ORION_ROUTING_ARTIFACT_DIR).join(format!("generation-{generation}.json"))
}

pub fn routing_artifact_path(collection_path: &Path, generation: u64) -> PathBuf {
    collection_path.join(routing_artifact_relative_path(generation))
}

pub use artifact::{
    ORION_ROUTING_ARTIFACT_FORMAT_VERSION, OrionRoutingArtifact, OrionUpperGraphNode,
    OrionUpperHnswGraph, OrionUpperNode, OrionVectorDatatype, OrionVectorSchemaFingerprint,
};
pub use builder::{
    ORION_UPPER_HNSW_DEFAULT_EF_CONSTRUCT, ORION_UPPER_HNSW_DEFAULT_M,
    ORION_UPPER_HNSW_DEFAULT_SEED, OrionUpperGraphBuildOptions, build_upper_hnsw_graph,
};
pub use error::{OrionRoutingError, OrionRoutingResult};
pub use router::{OrionRouter, OrionShardTarget, OrionUpperHit};

#[cfg(test)]
mod tests;

#[cfg(test)]
mod lifecycle_tests;
