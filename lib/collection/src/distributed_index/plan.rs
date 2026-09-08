use segment::types::ExtendedPointId;

use crate::orion::OrionShardTarget;
use crate::shards::shard::ShardId;
use crate::simple_kmeans::SimpleKmeansShardTarget;

/// The index policy that produced a selected-shard search plan.
///
/// HashAll is represented by [`LogicalShardSearchPlan::AllShards`]. Static policies use this
/// discriminator only to select policy-specific optional optimizations and diagnostics; ordinary
/// selected-shard execution is shared.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LogicalShardRoutingPolicy {
    Orion,
    SimpleKmeans,
}

impl LogicalShardRoutingPolicy {
    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::Orion => "Orion",
            Self::SimpleKmeans => "Simple KMeans",
        }
    }
}

/// Policy-independent lower-search overrides for one logical shard.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct LogicalShardSearchTarget {
    pub(crate) shard_id: ShardId,
    /// Ordered lower-HNSW entry points. `None` asks the shard to use its ordinary entry point.
    pub(crate) entry_points: Option<Vec<ExtendedPointId>>,
    pub(crate) hnsw_ef: usize,
}

impl From<OrionShardTarget> for LogicalShardSearchTarget {
    fn from(target: OrionShardTarget) -> Self {
        Self {
            shard_id: target.shard_id,
            entry_points: Some(target.entry_points),
            hnsw_ef: target.ef,
        }
    }
}

impl From<SimpleKmeansShardTarget> for LogicalShardSearchTarget {
    fn from(target: SimpleKmeansShardTarget) -> Self {
        Self {
            shard_id: target.shard_id,
            entry_points: None,
            hnsw_ef: target.ef,
        }
    }
}

/// A selected-shard plan bucketed by numeric logical shard.
///
/// Each tuple retains the original batch slot. Bucket and tuple encounter order are preserved so
/// the shared executor has the same stable row/merge ordering as the former policy-specific paths.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SelectedLogicalShardSearchPlan {
    pub(crate) policy: LogicalShardRoutingPolicy,
    pub(crate) generation: u64,
    pub(crate) targets_by_shard: Vec<Vec<(usize, LogicalShardSearchTarget)>>,
}

impl SelectedLogicalShardSearchPlan {
    pub(crate) fn new(
        policy: LogicalShardRoutingPolicy,
        generation: u64,
        shard_count: ShardId,
    ) -> Self {
        let mut targets_by_shard = Vec::with_capacity(shard_count as usize);
        targets_by_shard.resize_with(shard_count as usize, Vec::new);
        Self {
            policy,
            generation,
            targets_by_shard,
        }
    }

    pub(crate) fn push_targets(
        &mut self,
        query_index: usize,
        targets: impl IntoIterator<Item = LogicalShardSearchTarget>,
    ) -> Result<(), ShardId> {
        for target in targets {
            let shard_id = target.shard_id;
            let Some(shard_targets) = self.targets_by_shard.get_mut(shard_id as usize) else {
                return Err(shard_id);
            };
            shard_targets.push((query_index, target));
        }
        Ok(())
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.targets_by_shard.iter().all(Vec::is_empty)
    }

    pub(crate) fn shard_count(&self) -> usize {
        self.targets_by_shard.len()
    }
}

/// Complete collection-layer plan for one ordinary coordinator search batch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum LogicalShardSearchPlan {
    /// Qdrant HashAll semantics, also used for requests outside a static router's narrow contract.
    AllShards,
    /// A static index policy selected logical shards and lower-HNSW overrides.
    SelectedByShard(SelectedLogicalShardSearchPlan),
}

#[cfg(test)]
mod tests {
    use segment::types::PointIdType;

    use super::*;

    #[test]
    fn orion_and_simple_kmeans_compile_to_the_same_target_type() {
        let orion = LogicalShardSearchTarget::from(OrionShardTarget {
            shard_id: 3,
            entry_points: vec![PointIdType::from(31), PointIdType::from(29)],
            ef: 72,
        });
        let simple = LogicalShardSearchTarget::from(SimpleKmeansShardTarget {
            shard_id: 3,
            ef: 72,
        });

        assert_eq!(orion.shard_id, simple.shard_id);
        assert_eq!(orion.hnsw_ef, simple.hnsw_ef);
        assert_eq!(
            orion.entry_points,
            Some(vec![PointIdType::from(31), PointIdType::from(29)])
        );
        assert_eq!(simple.entry_points, None);
    }

    #[test]
    fn selected_plan_preserves_query_encounter_order_and_rejects_unknown_shards() {
        let mut plan = SelectedLogicalShardSearchPlan::new(LogicalShardRoutingPolicy::Orion, 7, 3);
        plan.push_targets(
            5,
            [
                LogicalShardSearchTarget {
                    shard_id: 0,
                    entry_points: Some(vec![PointIdType::from(31)]),
                    hnsw_ef: 48,
                },
                LogicalShardSearchTarget {
                    shard_id: 2,
                    entry_points: Some(vec![PointIdType::from(29)]),
                    hnsw_ef: 52,
                },
            ],
        )
        .unwrap();
        plan.push_targets(
            1,
            [LogicalShardSearchTarget {
                shard_id: 0,
                entry_points: Some(vec![PointIdType::from(17)]),
                hnsw_ef: 56,
            }],
        )
        .unwrap();

        assert_eq!(
            plan.targets_by_shard[0]
                .iter()
                .map(|(query_index, target)| (
                    *query_index,
                    target.entry_points.as_ref().unwrap()[0]
                ))
                .collect::<Vec<_>>(),
            vec![(5, PointIdType::from(31)), (1, PointIdType::from(17))],
        );
        assert_eq!(
            plan.push_targets(
                9,
                [LogicalShardSearchTarget {
                    shard_id: 3,
                    entry_points: None,
                    hnsw_ef: 64,
                }],
            ),
            Err(3),
        );
    }
}
