use std::collections::HashSet;
use std::path::Path;

use segment::types::Distance;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::error::{SimpleKmeansRoutingError, SimpleKmeansRoutingResult};
use crate::shards::shard::ShardId;

pub const SIMPLE_KMEANS_ROUTING_ARTIFACT_FORMAT_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SimpleKmeansVectorSchemaFingerprint {
    /// Empty string denotes Qdrant's default unnamed vector.
    pub vector_name: String,
    pub dimension: usize,
    pub distance: Distance,
    pub datatype: SimpleKmeansVectorDatatype,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SimpleKmeansVectorDatatype {
    Float32,
    Float16,
    Uint8,
}

/// Metric used only for coordinator-side centroid selection.
///
/// The established Simple KMeans baseline always ranks raw KMeans means by squared L2. For a
/// cosine collection the query is normalized first, but centroids are deliberately not
/// re-normalized after averaging.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SimpleKmeansRoutingDistance {
    SquaredL2,
}

impl SimpleKmeansVectorSchemaFingerprint {
    pub fn validate(&self) -> SimpleKmeansRoutingResult<()> {
        if self.dimension == 0 {
            return Err(SimpleKmeansRoutingError::InvalidVectorSchema {
                reason: "dimension must be greater than zero".to_string(),
            });
        }
        Ok(())
    }

    pub fn canonical_sha256(&self) -> SimpleKmeansRoutingResult<String> {
        self.validate()?;
        Ok(sha256_hex(&serde_json::to_vec(self)?))
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SimpleKmeansCentroid {
    pub shard_id: ShardId,
    pub vector: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SimpleKmeansRoutingArtifact {
    pub format_version: u32,
    pub generation: u64,
    pub vector_schema: SimpleKmeansVectorSchemaFingerprint,
    pub shard_count: ShardId,
    /// SHA-256 of the canonical full point-to-shard assignment JSONL used for import.
    pub layout_sha256: String,
    pub logical_point_count: u64,
    pub physical_point_count: u64,
    pub routing_distance: SimpleKmeansRoutingDistance,
    pub nprobe: usize,
    pub lower_hnsw_ef: usize,
    /// Exactly one centroid for every numeric logical shard.
    pub centroids: Vec<SimpleKmeansCentroid>,
}

impl SimpleKmeansRoutingArtifact {
    pub fn validate(&self) -> SimpleKmeansRoutingResult<()> {
        if self.format_version != SIMPLE_KMEANS_ROUTING_ARTIFACT_FORMAT_VERSION {
            return Err(SimpleKmeansRoutingError::UnsupportedFormatVersion {
                actual: self.format_version,
                supported: SIMPLE_KMEANS_ROUTING_ARTIFACT_FORMAT_VERSION,
            });
        }
        if self.generation == 0 {
            return Err(SimpleKmeansRoutingError::EmptyGeneration);
        }
        self.vector_schema.validate()?;
        if self.shard_count == 0 {
            return Err(SimpleKmeansRoutingError::EmptyShardSet);
        }
        if self.layout_sha256.len() != 64
            || !self
                .layout_sha256
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit())
        {
            return Err(SimpleKmeansRoutingError::InvalidLayoutChecksum {
                checksum: self.layout_sha256.clone(),
            });
        }
        if self.logical_point_count == 0 {
            return Err(SimpleKmeansRoutingError::EmptyLogicalPointCount);
        }
        if self.physical_point_count != self.logical_point_count {
            return Err(SimpleKmeansRoutingError::PhysicalPointCountMismatch {
                logical: self.logical_point_count,
                physical: self.physical_point_count,
            });
        }
        if self.nprobe == 0 {
            return Err(SimpleKmeansRoutingError::EmptyNprobe);
        }
        if self.nprobe > self.shard_count as usize {
            return Err(SimpleKmeansRoutingError::NprobeExceedsShardCount {
                nprobe: self.nprobe,
                shard_count: self.shard_count,
            });
        }
        if self.lower_hnsw_ef == 0 {
            return Err(SimpleKmeansRoutingError::EmptyLowerHnswEf);
        }
        if self.centroids.len() != self.shard_count as usize {
            return Err(SimpleKmeansRoutingError::CentroidCountMismatch {
                expected: self.shard_count as usize,
                actual: self.centroids.len(),
            });
        }

        let mut seen_shards = HashSet::with_capacity(self.centroids.len());
        for centroid in &self.centroids {
            if centroid.shard_id >= self.shard_count {
                return Err(SimpleKmeansRoutingError::ShardOutOfRange {
                    shard_id: centroid.shard_id,
                    shard_count: self.shard_count,
                });
            }
            if !seen_shards.insert(centroid.shard_id) {
                return Err(SimpleKmeansRoutingError::DuplicateShardCentroid {
                    shard_id: centroid.shard_id,
                });
            }
            if centroid.vector.len() != self.vector_schema.dimension {
                return Err(SimpleKmeansRoutingError::CentroidDimensionMismatch {
                    shard_id: centroid.shard_id,
                    expected: self.vector_schema.dimension,
                    actual: centroid.vector.len(),
                });
            }
            if let Some(dimension) = centroid.vector.iter().position(|value| !value.is_finite()) {
                return Err(SimpleKmeansRoutingError::NonFiniteCentroid {
                    shard_id: centroid.shard_id,
                    dimension,
                });
            }
        }
        Ok(())
    }

    pub fn validate_schema(
        &self,
        expected: &SimpleKmeansVectorSchemaFingerprint,
    ) -> SimpleKmeansRoutingResult<()> {
        expected.validate()?;
        if &self.vector_schema != expected {
            return Err(SimpleKmeansRoutingError::VectorSchemaMismatch {
                expected_sha256: expected.canonical_sha256()?,
                actual_sha256: self.vector_schema.canonical_sha256()?,
            });
        }
        Ok(())
    }

    pub fn canonical_json_bytes(&self) -> SimpleKmeansRoutingResult<Vec<u8>> {
        self.validate()?;
        Ok(serde_json::to_vec(self)?)
    }

    pub fn canonical_sha256(&self) -> SimpleKmeansRoutingResult<String> {
        Ok(sha256_hex(&self.canonical_json_bytes()?))
    }

    pub fn verify_checksum(&self, expected_checksum: &str) -> SimpleKmeansRoutingResult<()> {
        let expected = normalize_checksum(expected_checksum)?;
        let actual = self.canonical_sha256()?;
        if expected != actual {
            return Err(SimpleKmeansRoutingError::ChecksumMismatch { expected, actual });
        }
        Ok(())
    }

    pub fn from_json_slice(
        json: &[u8],
        expected_checksum: Option<&str>,
    ) -> SimpleKmeansRoutingResult<Self> {
        let artifact: Self = serde_json::from_slice(json)?;
        artifact.validate()?;
        if let Some(expected_checksum) = expected_checksum {
            artifact.verify_checksum(expected_checksum)?;
        }
        Ok(artifact)
    }

    pub fn read_json(
        path: impl AsRef<Path>,
        expected_checksum: Option<&str>,
    ) -> SimpleKmeansRoutingResult<Self> {
        let path = path.as_ref();
        let json = fs_err::read(path).map_err(|source| SimpleKmeansRoutingError::ArtifactIo {
            path: path.to_path_buf(),
            source,
        })?;
        Self::from_json_slice(&json, expected_checksum)
    }
}

fn normalize_checksum(checksum: &str) -> SimpleKmeansRoutingResult<String> {
    let normalized = checksum.trim().to_ascii_lowercase();
    if normalized.len() != 64 || !normalized.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(SimpleKmeansRoutingError::InvalidChecksum {
            checksum: checksum.to_string(),
        });
    }
    Ok(normalized)
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    format!("{digest:x}")
}
