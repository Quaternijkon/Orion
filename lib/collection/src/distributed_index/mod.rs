//! Common distributed-index planning primitives.
//!
//! Automatic shard policies decide *which* logical shards to search and which lower-HNSW
//! overrides to apply. Qdrant's collection layer remains responsible for executing those plans
//! through ordinary [`ShardReplicaSet`](crate::shards::replica_set::ShardReplicaSet) reads and for
//! globally merging the returned rows.

mod plan;

pub(crate) use plan::{
    LogicalShardRoutingPolicy, LogicalShardSearchPlan, LogicalShardSearchTarget,
    SelectedLogicalShardSearchPlan,
};
