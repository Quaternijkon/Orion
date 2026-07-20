mod artifact;
mod builder;
mod error;
mod router;

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
