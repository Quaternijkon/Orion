use super::Result;

/// Fully specified integer RNG: SplitMix64, with rejection-sampled bounded integers.
/// This is deliberately versioned, not an alias for NumPy PCG64.
pub(super) struct Rng(u64);

impl Rng {
    pub(super) fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }

    pub(super) fn bounded(&mut self, bound: usize) -> usize {
        let bound = bound as u64;
        let threshold = bound.wrapping_neg() % bound;
        loop {
            let x = self.next();
            if x >= threshold {
                return (x % bound) as usize;
            }
        }
    }

    pub(super) fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 * (1.0 / ((1_u64 << 53) as f64))
    }
}

pub fn sample_indices(rows: usize, denominator: usize, seed: u64) -> Result<Vec<usize>> {
    if denominator == 0 || rows / denominator == 0 {
        return Err("N/M sample is empty or M is zero".into());
    }
    let mut indices: Vec<_> = (0..rows).collect();
    let mut rng = Rng::new(seed);
    // Partial forward Fisher-Yates; output order is also HNSW insertion order.
    for i in 0..rows / denominator {
        let j = i + rng.bounded(rows - i);
        indices.swap(i, j);
    }
    indices.truncate(rows / denominator);
    // Do not carry the N-row allocation into graph construction after sampling N/M.
    indices.shrink_to_fit();
    Ok(indices)
}
