use ordered_float::OrderedFloat;
use segment::types::Distance;

use super::artifact::SimpleKmeansRoutingArtifact;
use super::error::{SimpleKmeansRoutingError, SimpleKmeansRoutingResult};
use crate::shards::shard::ShardId;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SimpleKmeansShardTarget {
    pub shard_id: ShardId,
    pub ef: usize,
}

#[derive(Debug)]
struct RuntimeCentroid {
    shard_id: ShardId,
    vector: Vec<f32>,
}

/// Immutable exact-centroid router for the static Simple KMeans nprobe baseline.
#[derive(Debug)]
pub struct SimpleKmeansRouter {
    generation: u64,
    vector_name: String,
    dimension: usize,
    distance: Distance,
    shard_count: ShardId,
    nprobe: usize,
    lower_hnsw_ef: usize,
    centroids: Vec<RuntimeCentroid>,
}

impl SimpleKmeansRouter {
    pub fn new(artifact: SimpleKmeansRoutingArtifact) -> SimpleKmeansRoutingResult<Self> {
        artifact.validate()?;
        Ok(Self {
            generation: artifact.generation,
            vector_name: artifact.vector_schema.vector_name,
            dimension: artifact.vector_schema.dimension,
            distance: artifact.vector_schema.distance,
            shard_count: artifact.shard_count,
            nprobe: artifact.nprobe,
            lower_hnsw_ef: artifact.lower_hnsw_ef,
            centroids: artifact
                .centroids
                .into_iter()
                .map(|centroid| RuntimeCentroid {
                    shard_id: centroid.shard_id,
                    // Preserve the raw arithmetic mean. Re-normalizing a cosine centroid changes
                    // squared-L2 nprobe selection relative to the established baseline.
                    vector: centroid.vector,
                })
                .collect(),
        })
    }

    pub fn generation(&self) -> u64 {
        self.generation
    }

    pub fn vector_name(&self) -> &str {
        &self.vector_name
    }

    pub fn shard_count(&self) -> ShardId {
        self.shard_count
    }

    pub fn nprobe(&self) -> usize {
        self.nprobe
    }

    pub fn route_query(
        &self,
        query: &[f32],
    ) -> SimpleKmeansRoutingResult<Vec<SimpleKmeansShardTarget>> {
        if query.len() != self.dimension {
            return Err(SimpleKmeansRoutingError::QueryDimensionMismatch {
                expected: self.dimension,
                actual: query.len(),
            });
        }
        if let Some(dimension) = query.iter().position(|value| !value.is_finite()) {
            return Err(SimpleKmeansRoutingError::NonFiniteQueryVector { dimension });
        }
        let query = if self.distance == Distance::Cosine {
            normalize_cosine_query(query)
        } else {
            query.to_vec()
        };
        let mut ranked = self
            .centroids
            .iter()
            .map(|centroid| {
                let distance = squared_l2(&query, &centroid.vector);
                if !distance.is_finite() {
                    return Err(SimpleKmeansRoutingError::NonFiniteDistance {
                        shard_id: centroid.shard_id,
                    });
                }
                Ok((OrderedFloat(distance), centroid.shard_id))
            })
            .collect::<SimpleKmeansRoutingResult<Vec<_>>>()?;
        ranked.sort_unstable();
        Ok(ranked
            .into_iter()
            .take(self.nprobe)
            .map(|(_, shard_id)| SimpleKmeansShardTarget {
                shard_id,
                ef: self.lower_hnsw_ef,
            })
            .collect())
    }
}

fn normalize_cosine_query(query: &[f32]) -> Vec<f32> {
    let norm = query.iter().map(|value| value * value).sum::<f32>().sqrt();
    if norm >= 1.0e-12 {
        query.iter().map(|value| value / norm).collect()
    } else {
        query.to_vec()
    }
}

fn squared_l2(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right)
        .map(|(left, right)| {
            let delta = left - right;
            delta * delta
        })
        .sum()
}
