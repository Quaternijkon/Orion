use std::io;
use std::path::PathBuf;

use segment::types::ExtendedPointId;
use thiserror::Error;

use crate::shards::shard::ShardId;

#[derive(Debug, Error)]
pub enum OrionRoutingError {
    #[error("failed to read Orion routing artifact {path}: {source}")]
    ArtifactIo {
        path: PathBuf,
        #[source]
        source: io::Error,
    },

    #[error("invalid Orion routing artifact JSON: {0}")]
    InvalidJson(#[from] serde_json::Error),

    #[error(
        "unsupported Orion routing artifact format version {actual}; supported version is {supported}"
    )]
    UnsupportedFormatVersion { actual: u32, supported: u32 },

    #[error("Orion routing artifact generation must be greater than zero")]
    EmptyGeneration,

    #[error(
        "Orion routing artifact generation mismatch: policy requires {expected}, artifact contains {actual}"
    )]
    GenerationMismatch { expected: u64, actual: u64 },

    #[error("invalid Orion vector schema fingerprint: {reason}")]
    InvalidVectorSchema { reason: String },

    #[error(
        "Orion vector schema fingerprint mismatch: expected {expected_sha256}, artifact has {actual_sha256}"
    )]
    VectorSchemaMismatch {
        expected_sha256: String,
        actual_sha256: String,
    },

    #[error("Orion shard_count must be greater than zero")]
    EmptyShardSet,

    #[error("Orion logical_point_count must be greater than zero")]
    EmptyLogicalPointCount,

    #[error(
        "Orion physical_point_count ({physical}) must be at least logical_point_count ({logical})"
    )]
    InvalidPhysicalPointCount { logical: u64, physical: u64 },

    #[error(
        "invalid Orion layout SHA-256 {checksum:?}; expected exactly 64 hexadecimal characters"
    )]
    InvalidLayoutChecksum { checksum: String },

    #[error(
        "Orion routing artifact shard-count mismatch: collection has {expected}, artifact contains {actual}"
    )]
    ShardCountMismatch { expected: ShardId, actual: ShardId },

    #[error("Orion upper_k must be greater than zero")]
    EmptyUpperK,

    #[error("Orion upper_ef_search ({upper_ef_search}) must be at least upper_k ({upper_k})")]
    UpperEfSearchTooSmall {
        upper_k: usize,
        upper_ef_search: usize,
    },

    #[error("Orion dynamic EF base must be greater than zero")]
    EmptyDynamicEfBase,

    #[error("Orion dynamic EF calculation overflows usize")]
    DynamicEfOverflow,

    #[error("Orion upper tier must contain at least one node")]
    EmptyUpperTier,

    #[error("duplicate Orion upper point label {label}")]
    DuplicateUpperLabel { label: ExtendedPointId },

    #[error(
        "upper point {label} has vector dimension {actual}, but the schema requires {expected}"
    )]
    UpperVectorDimensionMismatch {
        label: ExtendedPointId,
        expected: usize,
        actual: usize,
    },

    #[error("upper point {label} has a non-finite vector component at dimension {dimension}")]
    NonFiniteUpperVector {
        label: ExtendedPointId,
        dimension: usize,
    },

    #[error("upper point {label} has no logical shard membership")]
    EmptyShardMembership { label: ExtendedPointId },

    #[error("upper point {label} references shard {shard_id}, but shard_count is {shard_count}")]
    ShardOutOfRange {
        label: ExtendedPointId,
        shard_id: ShardId,
        shard_count: ShardId,
    },

    #[error("upper point {label} contains duplicate membership for shard {shard_id}")]
    DuplicateShardMembership {
        label: ExtendedPointId,
        shard_id: ShardId,
    },

    #[error("serialized Orion upper HNSW graph is required in production mode")]
    MissingSerializedUpperGraph,

    #[error(
        "Orion upper HNSW graph is already present; the offline builder requires a graphless artifact"
    )]
    UpperGraphAlreadyPresent,

    #[error("invalid Orion upper HNSW build configuration: {reason}")]
    InvalidUpperGraphBuildConfig { reason: String },

    #[error("failed to build Orion upper HNSW graph: {reason}")]
    UpperGraphBuild { reason: String },

    #[error("Orion upper HNSW entry point {label} does not exist")]
    GraphEntryPointNotFound { label: ExtendedPointId },

    #[error("duplicate Orion upper HNSW link record for point {label}")]
    DuplicateGraphNode { label: ExtendedPointId },

    #[error("Orion upper HNSW graph has no link record for point {label}")]
    MissingGraphNode { label: ExtendedPointId },

    #[error("Orion upper HNSW link record for point {label} has no level-0 list")]
    MissingGraphLevelZero { label: ExtendedPointId },

    #[error(
        "Orion upper HNSW max_level is {declared}, but the serialized links have max level {actual}"
    )]
    GraphMaxLevelMismatch { declared: usize, actual: usize },

    #[error(
        "Orion upper HNSW entry point {label} has max level {actual}, but graph max_level is {expected}"
    )]
    GraphEntryPointLevelMismatch {
        label: ExtendedPointId,
        expected: usize,
        actual: usize,
    },

    #[error(
        "Orion upper HNSW point {label} at level {level} references unknown neighbor {neighbor}"
    )]
    GraphNeighborNotFound {
        label: ExtendedPointId,
        level: usize,
        neighbor: ExtendedPointId,
    },

    #[error(
        "Orion upper HNSW point {label} at level {level} references neighbor {neighbor}, which does not participate at that level"
    )]
    GraphNeighborMissingLevel {
        label: ExtendedPointId,
        level: usize,
        neighbor: ExtendedPointId,
    },

    #[error("Orion upper HNSW point {label} references itself at level {level}")]
    SelfGraphNeighbor {
        label: ExtendedPointId,
        level: usize,
    },

    #[error(
        "Orion upper HNSW point {label} contains duplicate neighbor {neighbor} at level {level}"
    )]
    DuplicateGraphNeighbor {
        label: ExtendedPointId,
        level: usize,
        neighbor: ExtendedPointId,
    },

    #[error("invalid SHA-256 checksum {checksum:?}; expected exactly 64 hexadecimal characters")]
    InvalidChecksum { checksum: String },

    #[error("Orion artifact checksum mismatch: expected {expected}, computed {actual}")]
    ChecksumMismatch { expected: String, actual: String },

    #[error("query dimension {actual} does not match Orion upper dimension {expected}")]
    QueryDimensionMismatch { expected: usize, actual: usize },

    #[error("query has a non-finite component at dimension {dimension}")]
    NonFiniteQueryVector { dimension: usize },

    #[error(
        "Orion upper HNSW returned only {actual} hits, but the complete routing union requires {expected}"
    )]
    IncompleteUpperSearch { expected: usize, actual: usize },

    #[error("distance computation for upper point {label} produced a non-finite value")]
    NonFiniteDistance { label: ExtendedPointId },

    #[error("upper search returned label {label}, which is absent from the active artifact")]
    UnknownUpperLabel { label: ExtendedPointId },
}

pub type OrionRoutingResult<T> = Result<T, OrionRoutingError>;
