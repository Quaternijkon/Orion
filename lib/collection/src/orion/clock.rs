//! Reserve one shard WAL clock domain per Orion generation. It fences catalog state
//! against restored/replicated shard data and participates in Qdrant WAL-delta recovery.
use super::lifecycle::{bad, io};
use crate::operations::types::{CollectionResult, UpdateStatus};
use crate::operations::{ClockTag, CollectionUpdateOperations, OperationWithClockTag};
use crate::shards::replica_set::ShardReplicaSet;
use crate::shards::replica_set::replica_set_state::ReplicaState;
use crate::shards::shard_trait::WaitUntil;
use common::counter::hardware_accumulator::HwMeasurementAcc;
use sha2::{Digest, Sha256};

pub const CLOCK_ID: u32 = u32::MAX;
pub const STRIDE: u64 = 1 << 32;
pub fn fence_tick(revision: u64) -> CollectionResult<u64> {
    revision
        .checked_mul(STRIDE)
        .and_then(|n| n.checked_add(STRIDE - 1))
        .ok_or_else(|| bad("Orion revision clock exhausted; advance the generation"))
}
pub fn tagged(
    operation: CollectionUpdateOperations,
    generation: u64,
    revision: u64,
    sequence: u64,
    token: &str,
) -> CollectionResult<OperationWithClockTag> {
    if sequence == 0 || sequence >= STRIDE {
        return Err(bad("Orion WAL batch sequence overflow"));
    }
    let tick = revision
        .checked_mul(STRIDE)
        .and_then(|n| n.checked_add(sequence))
        .ok_or_else(|| bad("Orion revision clock overflow"))?;
    let digest = Sha256::digest(token.as_bytes());
    let token = u64::from_le_bytes(digest[..8].try_into().unwrap());
    Ok(OperationWithClockTag::new(
        operation,
        Some(ClockTag::new_with_token(generation, CLOCK_ID, tick, token)),
    ))
}
pub async fn apply(
    shard: &ShardReplicaSet,
    operation: OperationWithClockTag,
) -> CollectionResult<bool> {
    // Recovering replicas receive changes through Qdrant's snapshot/proxy protocol.
    if !matches!(
        shard.peer_state(shard.this_peer_id()),
        Some(ReplicaState::Active | ReplicaState::Listener)
    ) {
        return Ok(false);
    }
    let expected = operation
        .clock_tag
        .ok_or_else(|| io("native WAL operation has no clock"))?;
    let result = shard
        .update_local(
            operation,
            WaitUntil::Visible,
            None,
            HwMeasurementAcc::disposable(),
            false,
        )
        .await?
        .ok_or_else(|| io("active Orion replica did not acknowledge its write"))?;
    if result.status == UpdateStatus::ClockRejected {
        let actual = result
            .clock_tag
            .ok_or_else(|| io("clock rejection omitted the clock"))?;
        if actual.peer_id != expected.peer_id
            || actual.clock_id != CLOCK_ID
            || actual.clock_tick < expected.clock_tick
            || (actual.clock_tick / STRIDE == expected.clock_tick / STRIDE
                && actual.token != expected.token)
        {
            return Err(io("conflicting Orion WAL clock"));
        }
        return Ok(actual.clock_tick / STRIDE > expected.clock_tick / STRIDE);
    }
    Ok(false)
}
pub async fn fence(
    shard: &ShardReplicaSet,
    generation: u64,
    revision: u64,
    token: &str,
) -> CollectionResult<bool> {
    let op = CollectionUpdateOperations::PointOperation(
        crate::operations::point_ops::PointOperations::DeletePoints { ids: vec![] },
    );
    apply(shard, tagged(op, generation, revision, STRIDE - 1, token)?).await
}
pub async fn observed(shard: &ShardReplicaSet, generation: u64) -> CollectionResult<Option<u64>> {
    Ok(shard
        .shard_recovery_point()
        .await?
        .iter_as_clock_tags()
        .find(|tag| tag.peer_id == generation && tag.clock_id == CLOCK_ID)
        .map(|tag| tag.clock_tick))
}
