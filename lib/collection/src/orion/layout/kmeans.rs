use super::Result;
use super::sampler::Rng;

fn squared_distance(a: &[f32], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(&a, &b)| {
            let d = f64::from(a) - b;
            d * d
        })
        .sum()
}

fn nearest(point: &[f32], centers: &[Vec<f64>]) -> usize {
    (0..centers.len())
        .min_by(|&a, &b| {
            squared_distance(point, &centers[a])
                .total_cmp(&squared_distance(point, &centers[b]))
                .then(a.cmp(&b))
        })
        .unwrap()
}

/// K-Means++ followed by a fixed number of Lloyd rounds. Scalar f64 sums in
/// insertion order, first-index ties, and retained empty centers are part of v2.
pub(crate) fn cluster(
    vectors: &[impl AsRef<[f32]>],
    k: usize,
    iterations: usize,
    seed: u64,
) -> Result<Vec<u32>> {
    if k == 0
        || vectors.len() < k
        || iterations == 0
        || vectors[0].as_ref().is_empty()
        || vectors.iter().any(|v| {
            v.as_ref().len() != vectors[0].as_ref().len()
                || v.as_ref().iter().any(|x| !x.is_finite())
        })
    {
        return Err("invalid K-Means input".into());
    }
    // Preserve scalar f64 arithmetic and reduction order without a full f64 copy.
    let points = vectors;
    let mut rng = Rng::new(seed);
    let mut centers: Vec<Vec<f64>> = vec![
        points[rng.bounded(points.len())]
            .as_ref()
            .iter()
            .map(|&x| f64::from(x))
            .collect(),
    ];
    let mut distances = vec![f64::MAX; points.len()];
    while centers.len() < k {
        for (point, distance) in points.iter().zip(&mut distances) {
            *distance = distance.min(squared_distance(point.as_ref(), centers.last().unwrap()));
        }
        let total: f64 = distances.iter().sum();
        let threshold = rng.unit() * total;
        let mut cumulative = 0.0;
        let index = distances
            .iter()
            .position(|&d| {
                cumulative += d;
                cumulative >= threshold
            })
            .unwrap_or(points.len() - 1);
        centers.push(
            points[index]
                .as_ref()
                .iter()
                .map(|&x| f64::from(x))
                .collect(),
        );
    }
    for _ in 0..iterations {
        let mut sums = vec![vec![0.0; points[0].as_ref().len()]; k];
        let mut counts = vec![0_usize; k];
        for point in points {
            let owner = nearest(point.as_ref(), &centers);
            counts[owner] += 1;
            for (sum, &value) in sums[owner].iter_mut().zip(point.as_ref()) {
                *sum += f64::from(value);
            }
        }
        for i in 0..k {
            if counts[i] != 0 {
                for x in &mut sums[i] {
                    *x /= counts[i] as f64;
                }
                centers[i] = std::mem::take(&mut sums[i]);
            }
        }
    }
    Ok(points
        .iter()
        .map(|p| nearest(p.as_ref(), &centers) as u32)
        .collect())
}

#[cfg(test)]
mod original {
    use super::super::Result;
    use super::super::sampler::Rng;

    fn squared_distance(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b).map(|(a, b)| (a - b) * (a - b)).sum()
    }

    fn nearest(point: &[f64], centers: &[Vec<f64>]) -> usize {
        (0..centers.len())
            .min_by(|&a, &b| {
                squared_distance(point, &centers[a])
                    .total_cmp(&squared_distance(point, &centers[b]))
                    .then(a.cmp(&b))
            })
            .unwrap()
    }

    /// K-Means++ followed by a fixed number of Lloyd rounds. Scalar f64 sums in
    /// insertion order, first-index ties, and retained empty centers are part of v2.
    pub(crate) fn cluster(
        vectors: &[Vec<f32>],
        k: usize,
        iterations: usize,
        seed: u64,
    ) -> Result<Vec<u32>> {
        if k == 0
            || vectors.len() < k
            || iterations == 0
            || vectors[0].is_empty()
            || vectors
                .iter()
                .any(|v| v.len() != vectors[0].len() || v.iter().any(|x| !x.is_finite()))
        {
            return Err("invalid K-Means input".into());
        }
        let points: Vec<Vec<f64>> = vectors
            .iter()
            .map(|v| v.iter().map(|&x| x as f64).collect())
            .collect();
        let mut rng = Rng::new(seed);
        let mut centers = vec![points[rng.bounded(points.len())].clone()];
        let mut distances = vec![f64::MAX; points.len()];
        while centers.len() < k {
            for (point, distance) in points.iter().zip(&mut distances) {
                *distance = distance.min(squared_distance(point, centers.last().unwrap()));
            }
            let total: f64 = distances.iter().sum();
            let threshold = rng.unit() * total;
            let mut cumulative = 0.0;
            let index = distances
                .iter()
                .position(|&d| {
                    cumulative += d;
                    cumulative >= threshold
                })
                .unwrap_or(points.len() - 1);
            centers.push(points[index].clone());
        }
        for _ in 0..iterations {
            let mut sums = vec![vec![0.0; points[0].len()]; k];
            let mut counts = vec![0_usize; k];
            for point in &points {
                let owner = nearest(point, &centers);
                counts[owner] += 1;
                for (sum, &value) in sums[owner].iter_mut().zip(point) {
                    *sum += value;
                }
            }
            for i in 0..k {
                if counts[i] != 0 {
                    for x in &mut sums[i] {
                        *x /= counts[i] as f64;
                    }
                    centers[i] = std::mem::take(&mut sums[i]);
                }
            }
        }
        Ok(points.iter().map(|p| nearest(p, &centers) as u32).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn bounded_kmeans_matches_original_scalar_f64_reductions() {
        for seed in [0, 1, 100, 9999] {
            let mut rng = Rng::new(seed);
            for (count, dim, k) in [(32, 2, 4), (129, 128, 15), (4, 3, 4)] {
                let points: Vec<Vec<f32>> = (0..count)
                    .map(|_| {
                        (0..dim)
                            .map(|_| ((rng.unit() - 0.5) * 512.0) as f32)
                            .collect()
                    })
                    .collect();
                assert_eq!(
                    cluster(&points, k, 10, seed).unwrap(),
                    original::cluster(&points, k, 10, seed).unwrap()
                );
            }
        }
    }
}
