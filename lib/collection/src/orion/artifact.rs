use std::collections::{HashMap, HashSet};
use std::path::Path;

use segment::types::{Distance, ExtendedPointId};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::error::{OrionRoutingError, OrionRoutingResult};
use crate::shards::shard::ShardId;

pub const ORION_ROUTING_ARTIFACT_FORMAT_VERSION: u32 = 1;

/// Dense-vector schema fields which must remain identical between artifact build and serving.
///
/// This is deliberately independent of collection configuration structs. The collection adapter
/// can construct this value from its active vector configuration and compare the fingerprints
/// before activating an artifact.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OrionVectorSchemaFingerprint {
    /// Empty string denotes Qdrant's default unnamed vector.
    pub vector_name: String,
    pub dimension: usize,
    pub distance: Distance,
    pub datatype: OrionVectorDatatype,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OrionVectorDatatype {
    Float32,
    Float16,
    Uint8,
}

impl OrionVectorSchemaFingerprint {
    pub fn validate(&self) -> OrionRoutingResult<()> {
        if self.dimension == 0 {
            return Err(OrionRoutingError::InvalidVectorSchema {
                reason: "dimension must be greater than zero".to_string(),
            });
        }
        Ok(())
    }

    /// Stable SHA-256 over this format's typed JSON representation.
    pub fn canonical_sha256(&self) -> OrionRoutingResult<String> {
        self.validate()?;
        Ok(sha256_hex(&serde_json::to_vec(self)?))
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OrionUpperNode {
    pub label: ExtendedPointId,
    pub vector: Vec<f32>,
    /// Logical shard IDs assigned by Orion voting/multi-assignment.
    pub shard_membership: Vec<ShardId>,
}

/// Per-level adjacency for one upper point. Index 0 is level 0.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OrionUpperGraphNode {
    pub label: ExtendedPointId,
    pub neighbors_by_level: Vec<Vec<ExtendedPointId>>,
}

/// Portable upper HNSW graph. All references are external point IDs, never process-local offsets.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OrionUpperHnswGraph {
    pub entry_point: ExtendedPointId,
    pub max_level: usize,
    pub nodes: Vec<OrionUpperGraphNode>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OrionRoutingArtifact {
    pub format_version: u32,
    pub generation: u64,
    pub vector_schema: OrionVectorSchemaFingerprint,
    pub shard_count: ShardId,
    /// SHA-256 of the canonical full point-to-shards assignment JSONL used by the importer.
    pub layout_sha256: String,
    /// Number of unique external point IDs in the source dataset.
    pub logical_point_count: u64,
    /// Number of physical point copies after multi-assignment.
    pub physical_point_count: u64,
    /// Number of ordered upper labels used to form the complete shard union.
    pub upper_k: usize,
    /// Upper HNSW level-0 candidate budget. Must be at least `upper_k`.
    pub upper_ef_search: usize,
    pub dynamic_ef_base: usize,
    pub dynamic_ef_factor: usize,
    pub upper_nodes: Vec<OrionUpperNode>,
    /// Required by the production router. It may be absent only for explicit test tooling.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub upper_graph: Option<OrionUpperHnswGraph>,
}

impl OrionRoutingArtifact {
    pub fn validate(&self) -> OrionRoutingResult<()> {
        if self.format_version != ORION_ROUTING_ARTIFACT_FORMAT_VERSION {
            return Err(OrionRoutingError::UnsupportedFormatVersion {
                actual: self.format_version,
                supported: ORION_ROUTING_ARTIFACT_FORMAT_VERSION,
            });
        }
        if self.generation == 0 {
            return Err(OrionRoutingError::EmptyGeneration);
        }
        self.vector_schema.validate()?;
        if self.shard_count == 0 {
            return Err(OrionRoutingError::EmptyShardSet);
        }
        if self.layout_sha256.len() != 64
            || !self
                .layout_sha256
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit())
        {
            return Err(OrionRoutingError::InvalidLayoutChecksum {
                checksum: self.layout_sha256.clone(),
            });
        }
        if self.logical_point_count == 0 {
            return Err(OrionRoutingError::EmptyLogicalPointCount);
        }
        if self.physical_point_count < self.logical_point_count {
            return Err(OrionRoutingError::InvalidPhysicalPointCount {
                logical: self.logical_point_count,
                physical: self.physical_point_count,
            });
        }
        if self.upper_k == 0 {
            return Err(OrionRoutingError::EmptyUpperK);
        }
        if self.upper_ef_search < self.upper_k {
            return Err(OrionRoutingError::UpperEfSearchTooSmall {
                upper_k: self.upper_k,
                upper_ef_search: self.upper_ef_search,
            });
        }
        if self.dynamic_ef_base == 0 {
            return Err(OrionRoutingError::EmptyDynamicEfBase);
        }
        self.dynamic_ef_factor
            .checked_mul(self.upper_k)
            .and_then(|increment| self.dynamic_ef_base.checked_add(increment))
            .ok_or(OrionRoutingError::DynamicEfOverflow)?;
        if self.upper_nodes.is_empty() {
            return Err(OrionRoutingError::EmptyUpperTier);
        }
        if self.upper_k > self.upper_nodes.len() {
            return Err(OrionRoutingError::UpperEfSearchTooSmall {
                upper_k: self.upper_k,
                upper_ef_search: self.upper_nodes.len(),
            });
        }

        let mut upper_labels = HashSet::with_capacity(self.upper_nodes.len());
        for node in &self.upper_nodes {
            if !upper_labels.insert(node.label) {
                return Err(OrionRoutingError::DuplicateUpperLabel { label: node.label });
            }
            if node.vector.len() != self.vector_schema.dimension {
                return Err(OrionRoutingError::UpperVectorDimensionMismatch {
                    label: node.label,
                    expected: self.vector_schema.dimension,
                    actual: node.vector.len(),
                });
            }
            if let Some(dimension) = node.vector.iter().position(|value| !value.is_finite()) {
                return Err(OrionRoutingError::NonFiniteUpperVector {
                    label: node.label,
                    dimension,
                });
            }
            if node.shard_membership.is_empty() {
                return Err(OrionRoutingError::EmptyShardMembership { label: node.label });
            }
            let mut memberships = HashSet::with_capacity(node.shard_membership.len());
            for &shard_id in &node.shard_membership {
                if shard_id >= self.shard_count {
                    return Err(OrionRoutingError::ShardOutOfRange {
                        label: node.label,
                        shard_id,
                        shard_count: self.shard_count,
                    });
                }
                if !memberships.insert(shard_id) {
                    return Err(OrionRoutingError::DuplicateShardMembership {
                        label: node.label,
                        shard_id,
                    });
                }
            }
        }

        if let Some(graph) = &self.upper_graph {
            validate_graph(graph, &upper_labels)?;
        }
        Ok(())
    }

    pub fn validate_schema(
        &self,
        expected: &OrionVectorSchemaFingerprint,
    ) -> OrionRoutingResult<()> {
        expected.validate()?;
        if &self.vector_schema != expected {
            return Err(OrionRoutingError::VectorSchemaMismatch {
                expected_sha256: expected.canonical_sha256()?,
                actual_sha256: self.vector_schema.canonical_sha256()?,
            });
        }
        Ok(())
    }

    /// Serialize to deterministic bytes for this artifact version.
    ///
    /// This canonicalizes JSON whitespace and object field order by deserializing into typed
    /// structs before serialization. Array order remains significant because upper insertion and
    /// neighbor order are part of the serving artifact.
    pub fn canonical_json_bytes(&self) -> OrionRoutingResult<Vec<u8>> {
        self.validate()?;
        Ok(serde_json::to_vec(self)?)
    }

    pub fn canonical_sha256(&self) -> OrionRoutingResult<String> {
        Ok(sha256_hex(&self.canonical_json_bytes()?))
    }

    pub fn verify_checksum(&self, expected_checksum: &str) -> OrionRoutingResult<()> {
        let expected = normalize_checksum(expected_checksum)?;
        let actual = self.canonical_sha256()?;
        if expected != actual {
            return Err(OrionRoutingError::ChecksumMismatch { expected, actual });
        }
        Ok(())
    }

    pub fn from_json_slice(
        json: &[u8],
        expected_checksum: Option<&str>,
    ) -> OrionRoutingResult<Self> {
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
    ) -> OrionRoutingResult<Self> {
        let path = path.as_ref();
        let json = fs_err::read(path).map_err(|source| OrionRoutingError::ArtifactIo {
            path: path.to_path_buf(),
            source,
        })?;
        Self::from_json_slice(&json, expected_checksum)
    }
}

fn validate_graph(
    graph: &OrionUpperHnswGraph,
    upper_labels: &HashSet<ExtendedPointId>,
) -> OrionRoutingResult<()> {
    if !upper_labels.contains(&graph.entry_point) {
        return Err(OrionRoutingError::GraphEntryPointNotFound {
            label: graph.entry_point,
        });
    }

    let mut graph_nodes = HashMap::with_capacity(graph.nodes.len());
    for node in &graph.nodes {
        if graph_nodes.insert(node.label, node).is_some() {
            return Err(OrionRoutingError::DuplicateGraphNode { label: node.label });
        }
        if node.neighbors_by_level.is_empty() {
            return Err(OrionRoutingError::MissingGraphLevelZero { label: node.label });
        }
    }
    for &label in upper_labels {
        if !graph_nodes.contains_key(&label) {
            return Err(OrionRoutingError::MissingGraphNode { label });
        }
    }
    for &label in graph_nodes.keys() {
        if !upper_labels.contains(&label) {
            return Err(OrionRoutingError::GraphNeighborNotFound {
                label,
                level: 0,
                neighbor: label,
            });
        }
    }

    let actual_max_level = graph
        .nodes
        .iter()
        .map(|node| node.neighbors_by_level.len() - 1)
        .max()
        .unwrap_or(0);
    if actual_max_level != graph.max_level {
        return Err(OrionRoutingError::GraphMaxLevelMismatch {
            declared: graph.max_level,
            actual: actual_max_level,
        });
    }
    let entry_level = graph_nodes[&graph.entry_point].neighbors_by_level.len() - 1;
    if entry_level != graph.max_level {
        return Err(OrionRoutingError::GraphEntryPointLevelMismatch {
            label: graph.entry_point,
            expected: graph.max_level,
            actual: entry_level,
        });
    }

    for node in &graph.nodes {
        for (level, neighbors) in node.neighbors_by_level.iter().enumerate() {
            let mut seen = HashSet::with_capacity(neighbors.len());
            for &neighbor in neighbors {
                if neighbor == node.label {
                    return Err(OrionRoutingError::SelfGraphNeighbor {
                        label: node.label,
                        level,
                    });
                }
                if !seen.insert(neighbor) {
                    return Err(OrionRoutingError::DuplicateGraphNeighbor {
                        label: node.label,
                        level,
                        neighbor,
                    });
                }
                let Some(neighbor_node) = graph_nodes.get(&neighbor) else {
                    return Err(OrionRoutingError::GraphNeighborNotFound {
                        label: node.label,
                        level,
                        neighbor,
                    });
                };
                if neighbor_node.neighbors_by_level.len() <= level {
                    return Err(OrionRoutingError::GraphNeighborMissingLevel {
                        label: node.label,
                        level,
                        neighbor,
                    });
                }
            }
        }
    }
    Ok(())
}

fn normalize_checksum(checksum: &str) -> OrionRoutingResult<String> {
    let normalized = checksum.trim().to_ascii_lowercase();
    if normalized.len() != 64 || !normalized.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(OrionRoutingError::InvalidChecksum {
            checksum: checksum.to_string(),
        });
    }
    Ok(normalized)
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    format!("{digest:x}")
}
