mod artifact;
mod builder;
pub mod bundle;
pub mod cli;
pub mod clock;
pub mod deploy;
pub mod durable;
mod error;
pub mod import;
pub mod layout;
pub mod lifecycle;
pub mod membership;
pub mod migration;
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
