//! Ground-truth-assisted, whole-target pruning for controlled experiments only.
//! Both control and oracle generations look up the same exact query-bit key.
//! No entry point, lower EF, graph, or upper search is changed here.
use std::collections::HashMap;
use std::sync::OnceLock;

use super::OrionShardTarget;

const MAGIC: &[u8; 8] = b"ORIONM01";
static TABLE: OnceLock<Result<Option<Masks>, String>> = OnceLock::new();

struct Masks {
    dimension: usize,
    shard_count: u32,
    prune_generations: HashMap<u64, bool>,
    queries: HashMap<Vec<u32>, u64>,
}

impl Masks {
    // Header: magic, dimension/shards/generations/queries (four LE u32s).
    // Then generation records (LE u64 ID + u32 mode), and query records
    // (dimension LE f32 bit patterns + u64 keep mask). Mode0=control,1=oracle.
    fn decode(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() < 24 || &bytes[..8] != MAGIC {
            return Err("invalid mask-table header".into());
        }
        let u32_at = |i: usize| u32::from_le_bytes(bytes[i..i + 4].try_into().unwrap());
        let dimension = u32_at(8) as usize;
        let shard_count = u32_at(12);
        let generations = u32_at(16) as usize;
        let query_count = u32_at(20) as usize;
        if dimension == 0
            || !(1..=64).contains(&shard_count)
            || generations == 0
            || query_count == 0
        {
            return Err("empty or invalid mask-table dimensions".into());
        }
        let stride = dimension.checked_mul(4).and_then(|n| n.checked_add(8));
        let expected = stride
            .and_then(|s| s.checked_mul(query_count))
            .and_then(|n| generations.checked_mul(12).and_then(|g| n.checked_add(g)))
            .and_then(|n| n.checked_add(24));
        if expected != Some(bytes.len()) {
            return Err("mask-table length mismatch".into());
        }
        let u64_at = |i: usize| u64::from_le_bytes(bytes[i..i + 8].try_into().unwrap());
        let mut offset = 24;
        let mut prune_generations = HashMap::with_capacity(generations);
        for _ in 0..generations {
            let generation = u64_at(offset);
            let mode = u32_at(offset + 8);
            if generation == 0
                || mode > 1
                || prune_generations.insert(generation, mode == 1).is_some()
            {
                return Err("invalid or duplicate diagnostic generation".into());
            }
            offset += 12;
        }
        let allowed = u64::MAX >> (64 - shard_count);
        let mut queries = HashMap::with_capacity(query_count);
        for _ in 0..query_count {
            let key: Vec<_> = (0..dimension).map(|j| u32_at(offset + 4 * j)).collect();
            if key.iter().any(|&bits| !f32::from_bits(bits).is_finite()) {
                return Err("non-finite diagnostic query".into());
            }
            offset += dimension * 4;
            let mask = u64_at(offset);
            offset += 8;
            if mask == 0 || mask & !allowed != 0 || queries.insert(key, mask).is_some() {
                return Err("invalid mask or duplicate diagnostic query".into());
            }
        }
        Ok(Self {
            dimension,
            shard_count,
            prune_generations,
            queries,
        })
    }

    fn apply(
        &self,
        query: &[f32],
        generation: u64,
        shards: u32,
        mut targets: Vec<OrionShardTarget>,
    ) -> Result<Vec<OrionShardTarget>, String> {
        if query.len() != self.dimension || shards != self.shard_count {
            return Err("diagnostic table does not match query schema or shard count".into());
        }
        let prune = self
            .prune_generations
            .get(&generation)
            .ok_or_else(|| format!("unregistered diagnostic generation {generation}"))?;
        let key: Vec<_> = query.iter().map(|v| v.to_bits()).collect();
        let oracle = self
            .queries
            .get(&key)
            .ok_or("query absent from diagnostic table")?;
        let keep = if *prune { *oracle } else { u64::MAX };
        if targets.iter().any(|t| t.shard_id >= shards) {
            return Err("diagnostic target outside shard range".into());
        }
        targets.retain(|t| keep & (1u64 << t.shard_id) != 0);
        if targets.is_empty() {
            return Err("diagnostic mask removed all original targets".into());
        }
        Ok(targets)
    }
}

pub(super) fn prune(
    query: &[f32],
    generation: u64,
    shards: u32,
    targets: Vec<OrionShardTarget>,
) -> Result<Vec<OrionShardTarget>, String> {
    let table = TABLE.get_or_init(|| {
        let path = match std::env::var("ORION_DIAGNOSTIC_SHARD_MASKS") {
            Ok(path) => path,
            Err(std::env::VarError::NotPresent) => return Ok(None),
            Err(error) => return Err(error.to_string()),
        };
        let bytes = std::fs::read(&path).map_err(|e| format!("read {path}: {e}"))?;
        Masks::decode(&bytes).map(Some)
    });
    match table {
        Ok(Some(table)) => table.apply(query, generation, shards, targets),
        Ok(None) => Ok(targets),
        Err(error) => Err(error.clone()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use segment::types::ExtendedPointId;

    fn bytes(mask: u64) -> Vec<u8> {
        let mut b = MAGIC.to_vec();
        for n in [2u32, 3, 2, 1] {
            b.extend(n.to_le_bytes());
        }
        for (g, mode) in [(10u64, 0u32), (11, 1)] {
            b.extend(g.to_le_bytes());
            b.extend(mode.to_le_bytes());
        }
        for v in [0.25f32, -0.5] {
            b.extend(v.to_bits().to_le_bytes());
        }
        b.extend(mask.to_le_bytes());
        b
    }
    fn targets() -> Vec<OrionShardTarget> {
        (0..3)
            .map(|s| OrionShardTarget {
                shard_id: s,
                entry_points: vec![
                    ExtendedPointId::NumId(10 + s as u64),
                    ExtendedPointId::NumId(20 + s as u64),
                ],
                ef: 123 + s as usize,
            })
            .collect()
    }
    #[test]
    fn pruning_preserves_retained_order_entries_and_ef() {
        let table = Masks::decode(&bytes(5)).unwrap();
        let original = targets();
        assert_eq!(
            table.apply(&[0.25, -0.5], 10, 3, original.clone()).unwrap(),
            original
        );
        assert_eq!(
            table.apply(&[0.25, -0.5], 11, 3, original.clone()).unwrap(),
            vec![original[0].clone(), original[2].clone()]
        );
    }
    #[test]
    fn unknown_query_generation_and_schema_fail_closed() {
        let table = Masks::decode(&bytes(5)).unwrap();
        assert!(table.apply(&[0.5, -0.5], 10, 3, targets()).is_err());
        assert!(table.apply(&[0.25, -0.5], 12, 3, targets()).is_err());
        assert!(table.apply(&[0.25, -0.5], 10, 4, targets()).is_err());
        assert!(table.apply(&[0.25], 10, 3, targets()).is_err());
    }
    #[test]
    fn malformed_tables_and_empty_intersections_fail_closed() {
        for mask in [0, 8, u64::MAX] {
            assert!(Masks::decode(&bytes(mask)).is_err());
        }
        let mut data = bytes(1);
        data.pop();
        assert!(Masks::decode(&data).is_err());
        let table = Masks::decode(&bytes(1)).unwrap();
        assert!(
            table
                .apply(&[0.25, -0.5], 11, 3, vec![targets()[2].clone()])
                .is_err()
        );
    }
}
