//! Explainable, uniform whole-shard pruning after the original entry/EF plan.
//! Drop a candidate only when BOTH its prefix support is low AND its best upper
//! entry is farther than the configured gap from the global best entry.
use std::collections::BTreeSet;
use std::sync::OnceLock;

use segment::types::Distance;
use serde::Deserialize;

use super::{OrionRoutingArtifact, OrionShardTarget};

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub prefix_k: usize,
    pub max_prefix_entries: usize,
    /// Gap in the router's lower-is-better distance units. For cosine this is
    /// best_global_similarity - best_shard_similarity, NOT an absolute distance.
    pub max_distance_gap: f64,
    pub distance: Distance,
    /// Optional experiment switch. It changes only whether this same uniform
    /// rule is enabled for a generation; there are no per-shard thresholds.
    #[serde(default)]
    pub enabled_generations: Option<BTreeSet<u64>>,
}

#[derive(Debug)]
pub struct Policy {
    config: Config,
    upper_k: usize,
    shard_count: usize,
}

impl Config {
    pub fn from_json(bytes: &[u8]) -> Result<Self, String> {
        let config: Self = serde_json::from_slice(bytes).map_err(|e| e.to_string())?;
        config.validate()?;
        Ok(config)
    }

    fn validate(&self) -> Result<(), String> {
        if self.prefix_k == 0
            || self.max_prefix_entries >= self.prefix_k
            || !self.max_distance_gap.is_finite()
            || self.max_distance_gap < 0.0
            || self
                .enabled_generations
                .as_ref()
                .is_some_and(|g| g.contains(&0))
        {
            return Err(
                "invalid whitebox prefix, support count, distance gap or generation".into(),
            );
        }
        Ok(())
    }

    pub fn for_artifact(&self, artifact: &OrionRoutingArtifact) -> Result<Option<Policy>, String> {
        self.validate()?;
        if self
            .enabled_generations
            .as_ref()
            .is_some_and(|g| !g.contains(&artifact.generation))
        {
            return Ok(None);
        }
        if artifact.vector_schema.distance != self.distance
            || self.prefix_k > artifact.upper_k
            || artifact.upper_ef_search < artifact.upper_k
            || artifact.shard_count == 0
        {
            return Err("whitebox rule requires its configured distance and prefix_k <= upper_k <= upper_ef_search".into());
        }
        Ok(Some(Policy {
            config: self.clone(),
            upper_k: artifact.upper_k,
            shard_count: artifact.shard_count as usize,
        }))
    }
}

impl Policy {
    pub fn apply(
        &self,
        hits: impl IntoIterator<Item = (u32, f32)>,
        mut targets: Vec<OrionShardTarget>,
    ) -> Result<Vec<OrionShardTarget>, String> {
        if targets.is_empty() {
            return Err("whitebox selector received no original targets".into());
        }
        let mut best = vec![None; self.shard_count];
        let mut prefix_counts = vec![0usize; self.shard_count];
        let mut first = None;
        let mut previous = f32::NEG_INFINITY;
        let mut hit_count = 0;
        for (rank, (shard, distance)) in hits.into_iter().take(self.upper_k).enumerate() {
            let s = shard as usize;
            if s >= self.shard_count || !distance.is_finite() || distance < previous {
                return Err("invalid or unordered whitebox upper hits".into());
            }
            previous = distance;
            first.get_or_insert((shard, distance));
            best[s].get_or_insert(distance);
            if rank < self.config.prefix_k {
                prefix_counts[s] += 1;
            }
            hit_count += 1;
        }
        let (first_shard, global_best) = first.ok_or("whitebox selector received no upper hits")?;
        if hit_count < self.config.prefix_k || !targets.iter().any(|t| t.shard_id == first_shard) {
            return Err("whitebox route is missing prefix results or the best-entry target".into());
        }
        for target in &targets {
            if best
                .get(target.shard_id as usize)
                .copied()
                .flatten()
                .is_none()
            {
                return Err("whitebox target has no original upper entry".into());
            }
        }
        targets.retain(|target| {
            let s = target.shard_id as usize;
            // Match the feature extraction used to derive the paper rule: f64
            // subtraction rounded once to f32, followed by the scalar cutoff.
            let gap = (best[s].unwrap() as f64 - global_best as f64) as f32;
            !(prefix_counts[s] <= self.config.max_prefix_entries
                && gap as f64 > self.config.max_distance_gap)
        });
        // The global-best entry has gap0 and therefore always survives. We do
        // not exempt any shard ID or silently alter the retained entries/EF.
        debug_assert!(!targets.is_empty());
        Ok(targets)
    }
}

static CONFIG: OnceLock<Result<Option<Config>, String>> = OnceLock::new();
pub(super) fn configured(artifact: &OrionRoutingArtifact) -> Result<Option<Policy>, String> {
    let config = CONFIG.get_or_init(|| {
        let path = match std::env::var("ORION_WHITEBOX_PRUNER") {
            Ok(path) => path,
            Err(std::env::VarError::NotPresent) => return Ok(None),
            Err(e) => return Err(e.to_string()),
        };
        let bytes = std::fs::read(path).map_err(|e| format!("read whitebox selector: {e}"))?;
        Config::from_json(&bytes).map(Some)
    });
    match config {
        Ok(Some(c)) => c.for_artifact(artifact),
        Ok(None) => Ok(None),
        Err(e) => Err(e.clone()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use segment::types::ExtendedPointId;
    fn policy() -> Policy {
        Policy {
            config: Config {
                prefix_k: 4,
                max_prefix_entries: 1,
                max_distance_gap: 0.125,
                distance: Distance::Cosine,
                enabled_generations: None,
            },
            upper_k: 7,
            shard_count: 8,
        }
    }
    fn target(s: u32) -> OrionShardTarget {
        OrionShardTarget {
            shard_id: s,
            entry_points: vec![ExtendedPointId::NumId(s as u64 + 100)],
            ef: 48 + 4 * (s as usize + 1),
        }
    }
    #[test]
    fn both_conditions_required_and_shard_five_has_no_exception() {
        let hits = [
            (0, 0.),
            (1, 0.0625),
            (2, 0.25),
            (2, 0.375),
            (3, 0.5),
            (5, 0.625),
            (7, 0.75),
        ];
        let original: Vec<_> = [0, 1, 2, 3, 5, 7].into_iter().map(target).collect();
        let kept = policy().apply(hits, original.clone()).unwrap();
        assert_eq!(kept, original[..3]); // near sparse shard1 and distant dense shard2 both survive.
        assert!(!kept.iter().any(|x| x.shard_id == 5));
    }
    #[test]
    fn equality_at_gap_threshold_is_kept_and_order_and_ef_are_unchanged() {
        let original: Vec<_> = [2, 1, 0, 3].into_iter().map(target).collect();
        let kept = policy()
            .apply(
                [(0, 0.), (1, 0.125), (2, 0.25), (2, 0.375), (3, 0.5)],
                original.clone(),
            )
            .unwrap();
        assert_eq!(kept, original[..3]);
    }
    #[test]
    fn invalid_inputs_fail_instead_of_silently_changing_the_rule() {
        assert!(
            policy()
                .apply([(0, 0.), (1, f32::NAN)], vec![target(0)])
                .is_err()
        );
        assert!(
            policy()
                .apply([(0, 0.), (1, -1.)], vec![target(0)])
                .is_err()
        );
        assert!(
            policy()
                .apply([(0, 0.), (1, 0.1), (2, 0.2), (3, 0.3)], vec![target(5)])
                .is_err()
        );
        assert!(Config::from_json(br#"{"prefix_k":4,"max_prefix_entries":1,"max_distance_gap":-1,"distance":"Cosine"}"#).is_err());
        assert!(Config::from_json(br#"{"prefix_k":4,"max_prefix_entries":1,"max_distance_gap":0.1,"distance":"Cosine","exempt_shards":[5]}"#).is_err());
    }
}
