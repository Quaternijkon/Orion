//! Pure feature extraction and tree evaluation for the optional shard selector.
pub const FEATURES: [&str; 29] = [
    "shard_0",
    "shard_1",
    "shard_2",
    "shard_3",
    "shard_4",
    "shard_5",
    "shard_6",
    "shard_7",
    "count_8",
    "count_16",
    "count_32",
    "count_64",
    "count_128",
    "count_256",
    "first_rank",
    "best_distance",
    "best_gap",
    "mean_top4_distance",
    "mean_distance",
    "span",
    "global_distance_1",
    "global_distance_8",
    "global_distance_32",
    "global_distance_128",
    "global_distance_256",
    "owner_sample_fraction",
    "owner_data_fraction",
    "share_top16",
    "share_top64",
];

#[derive(Clone, Copy, Debug, Default)]
struct OwnerStats {
    count: usize,
    prefix: [usize; 6],
    first: usize,
    best: f32,
    last: f32,
    sum: f64,
    first_four: f64,
}

#[derive(Debug)]
pub struct FeatureContext {
    owners: [OwnerStats; 8],
    global: [f32; 5],
    samples: [f64; 8],
    data: [f64; 8],
}
impl FeatureContext {
    pub fn new(
        hits: impl IntoIterator<Item = (u32, f32)>,
        samples: [f64; 8],
        data: [f64; 8],
    ) -> Result<Self, String> {
        let mut result = Self {
            owners: [OwnerStats::default(); 8],
            global: [0.; 5],
            samples,
            data,
        };
        let mut count = 0;
        for (rank, (owner, distance)) in hits.into_iter().enumerate() {
            if rank >= 256 || owner >= 8 || !distance.is_finite() {
                return Err("invalid selector upper hits".into());
            }
            let stats = &mut result.owners[owner as usize];
            if stats.count == 0 {
                stats.first = rank;
                stats.best = distance;
            }
            stats.last = distance;
            stats.sum += distance as f64;
            if stats.count < 4 {
                stats.first_four += distance as f64;
            }
            stats.count += 1;
            for (j, k) in [8, 16, 32, 64, 128, 256].into_iter().enumerate() {
                if rank < k {
                    stats.prefix[j] += 1;
                }
            }
            for (j, k) in [0, 7, 31, 127, 255].into_iter().enumerate() {
                if rank == k {
                    result.global[j] = distance;
                }
            }
            count += 1;
        }
        if count != 256 {
            return Err("selector requires all256 upper results".into());
        }
        Ok(result)
    }
    pub fn features(&self, shard: u32) -> Result<[f32; 29], String> {
        if shard >= 8 {
            return Err("selector shard outside range".into());
        }
        let s = shard as usize;
        let stats = &self.owners[s];
        if stats.count == 0 {
            return Err("selector target has no upper hits".into());
        }
        let mut x = [0.; 29];
        x[s] = 1.;
        for j in 0..6 {
            x[8 + j] = stats.prefix[j] as f32;
        }
        x[14] = stats.first as f32;
        x[15] = stats.best;
        x[16] = (stats.best as f64 - self.global[0] as f64) as f32;
        x[17] = (stats.first_four / stats.count.min(4) as f64) as f32;
        x[18] = (stats.sum / stats.count as f64) as f32;
        x[19] = (stats.last as f64 - stats.best as f64) as f32;
        x[20..25].copy_from_slice(&self.global);
        x[25] = self.samples[s] as f32;
        x[26] = self.data[s] as f32;
        x[27] = (stats.prefix[1] as f64 / stats.count as f64) as f32;
        x[28] = (stats.prefix[3] as f64 / stats.count as f64) as f32;
        Ok(x)
    }
}

#[derive(Clone, Debug)]
pub struct Node {
    pub left: i32,
    pub right: i32,
    pub feature: usize,
    pub threshold: f64,
    pub value: f64,
}
#[derive(Debug)]
pub struct Tree {
    pub nodes: Vec<Node>,
}
impl Tree {
    pub fn validate(&self) -> Result<(), String> {
        if self.nodes.is_empty() || self.nodes.len() > 4096 {
            return Err("invalid selector tree size".into());
        }
        let mut incoming = vec![0; self.nodes.len()];
        for (i, n) in self.nodes.iter().enumerate() {
            if !n.threshold.is_finite() || !n.value.is_finite() {
                return Err("non-finite selector tree".into());
            }
            if n.left == -1 && n.right == -1 {
                continue;
            }
            if n.feature >= 29 {
                return Err("selector feature outside schema".into());
            }
            for child in [n.left, n.right] {
                if child <= i as i32 || child as usize >= self.nodes.len() {
                    return Err("invalid selector tree edge".into());
                }
                incoming[child as usize] += 1;
            }
        }
        if incoming[0] != 0 || incoming[1..].iter().any(|&n| n != 1) {
            return Err("selector tree is not rooted".into());
        }
        Ok(())
    }
    pub fn predict(&self, x: &[f32; 29]) -> f64 {
        let mut i = 0;
        loop {
            let n = &self.nodes[i];
            if n.left == -1 {
                return n.value;
            }
            i = if x[n.feature] as f64 <= n.threshold {
                n.left as usize
            } else {
                n.right as usize
            };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn features_preserve_prefix_counts_and_distance_order() {
        let c = FeatureContext::new(
            (0..256).map(|i| ((i % 8) as u32, -1. + i as f32 * 0.001)),
            [0.125; 8],
            [0.125; 8],
        )
        .unwrap();
        let x = c.features(3).unwrap();
        assert_eq!(&x[8..14], &[1., 2., 4., 8., 16., 32.]);
        assert_eq!(x[14], 3.);
        assert_eq!(x[25], 0.125);
        assert_eq!(x[27], 0.0625);
        assert_eq!(x[28], 0.25);
        assert!(FeatureContext::new([(0, 0.)], [0.125; 8], [0.125; 8]).is_err());
    }
    #[test]
    fn tree_thresholds_and_leaves_match_boosting_semantics() {
        let t = Tree {
            nodes: vec![
                Node {
                    left: 1,
                    right: 2,
                    feature: 8,
                    threshold: 0.5,
                    value: 0.,
                },
                Node {
                    left: -1,
                    right: -1,
                    feature: 0,
                    threshold: -2.,
                    value: -2.,
                },
                Node {
                    left: -1,
                    right: -1,
                    feature: 0,
                    threshold: -2.,
                    value: 3.,
                },
            ],
        };
        t.validate().unwrap();
        let mut x = [0.; 29];
        assert_eq!(t.predict(&x), -2.);
        x[8] = 1.;
        assert_eq!(t.predict(&x), 3.);
    }
    #[test]
    fn malformed_cycles_and_missing_children_are_rejected() {
        let t = Tree {
            nodes: vec![Node {
                left: 0,
                right: 1,
                feature: 0,
                threshold: 0.,
                value: 0.,
            }],
        };
        assert!(t.validate().is_err());
    }
}
