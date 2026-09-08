use super::Result;
use super::sampler::Rng;

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
