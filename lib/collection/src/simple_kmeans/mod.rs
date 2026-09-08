mod artifact;
mod error;
mod router;

use std::path::{Path, PathBuf};

pub const SIMPLE_KMEANS_ROUTING_ARTIFACT_DIR: &str = "simple_kmeans_router";

pub fn routing_artifact_relative_path(generation: u64) -> PathBuf {
    PathBuf::from(SIMPLE_KMEANS_ROUTING_ARTIFACT_DIR).join(format!("generation-{generation}.json"))
}

pub fn routing_artifact_path(collection_path: &Path, generation: u64) -> PathBuf {
    collection_path.join(routing_artifact_relative_path(generation))
}

pub use artifact::{
    SIMPLE_KMEANS_ROUTING_ARTIFACT_FORMAT_VERSION, SimpleKmeansCentroid,
    SimpleKmeansRoutingArtifact, SimpleKmeansRoutingDistance, SimpleKmeansVectorDatatype,
    SimpleKmeansVectorSchemaFingerprint,
};
pub use error::{SimpleKmeansRoutingError, SimpleKmeansRoutingResult};
pub use router::{SimpleKmeansRouter, SimpleKmeansShardTarget};

#[cfg(test)]
mod tests;
