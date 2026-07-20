use std::io;
use std::path::PathBuf;

use thiserror::Error;

use crate::shards::shard::ShardId;

#[derive(Debug, Error)]
pub enum SimpleKmeansRoutingError {
    #[error("failed to read Simple KMeans routing artifact {path}: {source}")]
    ArtifactIo {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error("invalid Simple KMeans routing artifact JSON: {0}")]
    InvalidJson(#[from] serde_json::Error),
    #[error(
        "unsupported Simple KMeans routing artifact format version {actual}; supported version is {supported}"
    )]
    UnsupportedFormatVersion { actual: u32, supported: u32 },
    #[error("Simple KMeans routing artifact generation must be greater than zero")]
    EmptyGeneration,
    #[error(
        "Simple KMeans routing artifact generation mismatch: policy requires {expected}, artifact contains {actual}"
    )]
    GenerationMismatch { expected: u64, actual: u64 },
    #[error("invalid Simple KMeans vector schema fingerprint: {reason}")]
    InvalidVectorSchema { reason: String },
    #[error(
        "Simple KMeans vector schema fingerprint mismatch: expected {expected_sha256}, artifact has {actual_sha256}"
    )]
    VectorSchemaMismatch {
        expected_sha256: String,
        actual_sha256: String,
    },
    #[error("Simple KMeans shard_count must be greater than zero")]
    EmptyShardSet,
    #[error(
        "Simple KMeans routing artifact shard-count mismatch: collection has {expected}, artifact contains {actual}"
    )]
    ShardCountMismatch { expected: ShardId, actual: ShardId },
    #[error(
        "invalid Simple KMeans layout SHA-256 {checksum:?}; expected exactly 64 hexadecimal characters"
    )]
    InvalidLayoutChecksum { checksum: String },
    #[error("Simple KMeans logical_point_count must be greater than zero")]
    EmptyLogicalPointCount,
    #[error(
        "single-assignment Simple KMeans requires physical_point_count ({physical}) to equal logical_point_count ({logical})"
    )]
    PhysicalPointCountMismatch { logical: u64, physical: u64 },
    #[error("Simple KMeans nprobe must be greater than zero")]
    EmptyNprobe,
    #[error("Simple KMeans nprobe ({nprobe}) exceeds shard_count ({shard_count})")]
    NprobeExceedsShardCount { nprobe: usize, shard_count: ShardId },
    #[error("Simple KMeans lower_hnsw_ef must be greater than zero")]
    EmptyLowerHnswEf,
    #[error(
        "Simple KMeans artifact must contain one centroid per numeric shard: expected {expected}, got {actual}"
    )]
    CentroidCountMismatch { expected: usize, actual: usize },
    #[error("Simple KMeans centroid references shard {shard_id}, but shard_count is {shard_count}")]
    ShardOutOfRange {
        shard_id: ShardId,
        shard_count: ShardId,
    },
    #[error("duplicate Simple KMeans centroid for shard {shard_id}")]
    DuplicateShardCentroid { shard_id: ShardId },
    #[error(
        "Simple KMeans centroid for shard {shard_id} has vector dimension {actual}, but the schema requires {expected}"
    )]
    CentroidDimensionMismatch {
        shard_id: ShardId,
        expected: usize,
        actual: usize,
    },
    #[error(
        "Simple KMeans centroid for shard {shard_id} has a non-finite component at dimension {dimension}"
    )]
    NonFiniteCentroid { shard_id: ShardId, dimension: usize },
    #[error("invalid SHA-256 checksum {checksum:?}; expected exactly 64 hexadecimal characters")]
    InvalidChecksum { checksum: String },
    #[error("Simple KMeans artifact checksum mismatch: expected {expected}, computed {actual}")]
    ChecksumMismatch { expected: String, actual: String },
    #[error("query dimension {actual} does not match Simple KMeans centroid dimension {expected}")]
    QueryDimensionMismatch { expected: usize, actual: usize },
    #[error("query has a non-finite component at dimension {dimension}")]
    NonFiniteQueryVector { dimension: usize },
    #[error("distance computation for Simple KMeans shard {shard_id} produced a non-finite value")]
    NonFiniteDistance { shard_id: ShardId },
}

pub type SimpleKmeansRoutingResult<T> = Result<T, SimpleKmeansRoutingError>;
