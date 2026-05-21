# KMeans Custom Shard Adaptive-EF Experiment Design

## Goal

Run two Qdrant experiments on a single collection with 44 custom shards created at the application layer by KMeans partitioning:

1. A uniform-`hnsw_ef` baseline with `nprobe = 1, 2, 4, 8, 16, 32, 44`
2. An adaptive per-shard `hnsw_ef` strategy where closer shards use larger `hnsw_ef` and farther shards use smaller `hnsw_ef`

Both experiments must report `Recall@10` and `QPS`, and both must first tune parameters so that the `nprobe=44` operating point reaches at least approximately `0.88` recall.

## Constraints

- Do not modify existing experiment scripts.
- Add a new standalone script only.
- Reuse the existing Qdrant custom-sharding route rather than modifying or recompiling Qdrant.
- Keep the dataset and high-level build settings aligned with prior experiments for fair comparison.

## Dataset And Environment

- Dataset: `glove-200-angular.hdf5`
- Path: `/home/taig/dry/faiss/datasets/glove-200-angular.hdf5`
- Query ground truth source: HDF5 `neighbors`
- Target service: distributed-mode single-node Qdrant on `http://localhost:6336`
- Shard count: `44`
- Collection sharding method: `custom`

## Build Configuration

Use the same build-side settings as the earlier custom-shard experiments unless the run proves impossible:

- `m = 32`
- `ef_construct = 100`
- `top_k = 10`
- `query_count = 10000`
- `full_scan_threshold = 10`
- one custom shard per shard key

## Partitioning Strategy

The script will:

1. Load and normalize the train vectors.
2. Train `44` centroids with KMeans on a sampled subset.
3. Assign each train vector to its nearest centroid.
4. Create one Qdrant custom shard key per centroid.
5. Upsert each point into the shard key matching its assigned centroid.

This preserves the intended application-layer KMeans routing while still using Qdrant as-is.

## Search Strategy A: Uniform EF

For each query:

1. Compute centroid scores.
2. Rank shard keys from nearest to farthest.
3. Select the top `nprobe` shard keys.
4. Issue one Qdrant batch search request for those shard keys with one shared `hnsw_ef`.

Before the full sweep, run a small `hnsw_ef` tuning pass on `nprobe=44` to find a value that reaches `Recall@10 >= 0.88`.

## Search Strategy B: Adaptive EF

Qdrant does not expose per-shard `hnsw_ef` within a single multi-shard search request. Therefore the adaptive strategy must be implemented in the application layer:

1. Rank shard keys from nearest to farthest for each query.
2. Assign `hnsw_ef` by shard rank using a monotone non-increasing schedule.
3. Issue one search sub-request per selected shard key using that shard's assigned `hnsw_ef`.
4. Merge the returned candidates client-side and keep the global top `k`.

The schedule must satisfy the experimental requirement:

- nearer shard => larger or equal `hnsw_ef`
- farther shard => smaller or equal `hnsw_ef`

Before the full sweep, run a small schedule-tuning pass on `nprobe=44` to find at least one schedule that reaches `Recall@10 >= 0.88`.

## Adaptive Schedule Family

To keep the parameter search tractable, use rank-bucket schedules instead of 44 independently tuned `hnsw_ef` values. A schedule is represented as a small ordered list of buckets over shard rank, for example:

- ranks `1-4`: `ef_high`
- ranks `5-12`: `ef_mid_high`
- ranks `13-24`: `ef_mid_low`
- ranks `25-44`: `ef_low`

All schedules must be monotone non-increasing by rank.

## Metrics

For both strategies record:

- `nprobe`
- parameterization (`hnsw_ef` or schedule name/details)
- `Recall@10`
- `QPS`
- wall-clock time
- average selected shard count

For tuning runs also record the tested candidate parameter values.

## Output Artifacts

Write all outputs under a new timestamped directory, for example:

- `results/qdrant_custom_adaptive_ef/<timestamp>/uniform_ef_tuning.csv`
- `results/qdrant_custom_adaptive_ef/<timestamp>/uniform_ef_nprobe.csv`
- `results/qdrant_custom_adaptive_ef/<timestamp>/adaptive_ef_tuning.csv`
- `results/qdrant_custom_adaptive_ef/<timestamp>/adaptive_ef_nprobe.csv`
- `results/qdrant_custom_adaptive_ef/<timestamp>/builds.csv`
- `results/qdrant_custom_adaptive_ef/<timestamp>/summary.json`

## Fairness Rules

- Keep the collection build fixed across the two search strategies.
- Use the same dataset, queries, `top_k`, and ground truth.
- Report adaptive strategy QPS including client-side fan-out and merge overhead.
- Do not compare adaptive QPS to a hypothetical server-side implementation; compare only to the uniform baseline actually measured under the same client environment.

## Risks

- The adaptive strategy may need more client-side requests and therefore may lose QPS even if it saves total graph work.
- The best uniform `hnsw_ef` satisfying `0.88` recall may differ from prior experiments because the collection is rebuilt.
- KMeans imbalance across shards may influence both recall and latency.

## Success Criteria

- A new standalone script exists and runs without modifying prior scripts.
- Uniform `nprobe=44` reaches at least `0.88` recall after tuning.
- Adaptive `nprobe=44` reaches at least `0.88` recall after tuning.
- Full `nprobe` sweeps for both strategies are saved as CSV.
- Final analysis explains the QPS/recall trade-off and whether adaptive `ef` materially helps.
