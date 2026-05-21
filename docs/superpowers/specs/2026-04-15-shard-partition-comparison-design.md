# Shard Partition Comparison Design

## Goal

Determine how shards should be partitioned in a general Qdrant setting by comparing three partition schemes under controlled conditions:

1. Hash-like random partitioning
2. KMeans partitioning
3. Balanced-KMeans partitioning

Each scheme is evaluated in two search scenarios:

1. Full-search across all shards
2. Selective-search over a routed subset of shards

The comparison must use matched recall rather than fixed `hnsw_ef`, so conclusions reflect comparable quality targets rather than arbitrary working points.

## Why This Experiment Exists

Previous results in this workspace showed an apparently surprising effect:

- Qdrant default hash-based sharding achieved higher QPS than KMeans-based semantic sharding at similar recall.

This does not directly imply that semantic partitioning is useless, because the earlier comparisons mixed:

- different sharding architectures (`auto` vs `custom`)
- different routing mechanisms
- different warmup states and measurement paths

This experiment isolates partitioning effects within one common Qdrant custom-sharding framework, then uses the previously measured auto-hash baseline as an architectural reference point.

## Controlled Environment

Keep the following fixed for all new runs:

- Qdrant server on the distributed-mode single-node container at `http://localhost:6336`
- server CPU set: `0-15`
- `QDRANT__STORAGE__PERFORMANCE__MAX_SEARCH_THREADS=16`
- `QDRANT__STORAGE__PERFORMANCE__OPTIMIZER_CPU_BUDGET=16`
- client pinned with `taskset -c 16-17`
- client env:
  - `OPENBLAS_NUM_THREADS=1`
  - `OMP_NUM_THREADS=1`
  - `MKL_NUM_THREADS=1`
  - `NUMEXPR_NUM_THREADS=1`

## Dataset And Common Parameters

- Dataset: `glove-200-angular.hdf5`
- Base vectors: full `train`
- Queries: full `test`
- Ground truth: HDF5 `neighbors`
- Shard count: `44`
- `m = 32`
- `ef_construct = 100`
- `top_k = 10`
- `query_count_tuning = 2000`
- `query_count_eval = 10000`

## Common Architecture

All three new partition schemes use the same Qdrant custom-sharding architecture:

- one collection
- `sharding_method = custom`
- 44 shard keys
- one shard per shard key

This ensures that differences are due to how data is assigned to shard keys, not because one scheme uses Qdrant auto-sharding and another uses custom shard routing.

## Partition Schemes

### 1. Hash-like Partition

Assign each point to one of 44 shard keys using a deterministic stable hash of the point ID.

Purpose:

- represent a locality-agnostic partition
- isolate the effect of random/hash-style data placement inside the same custom-sharding architecture

### 2. KMeans Partition

Train 44 centroids and assign each point to its nearest centroid.

Purpose:

- maximize locality without capacity constraints
- test whether pure geometric grouping helps or hurts in Qdrant

### 3. Balanced-KMeans Partition

Train the same 44 centroids, but assign points with capacity constraints so shard sizes remain close to balanced while preserving as much locality as possible.

Purpose:

- test whether the earlier KMeans disadvantage is caused by load imbalance rather than by semantic partitioning itself

## Search Scenarios

### Full-search

Search all 44 shard keys.

Meaning:

- evaluates how the shard-local HNSW structure behaves when every shard participates
- isolates whether the partition geometry itself helps or hurts

### Selective-search

Route each query to the nearest `nprobe` shard centroids and search only those shard keys.

Meaning:

- measures whether semantic partitioning creates real routing value
- lets us compare locality-aware partitions against locality-agnostic ones under the same client-side routing framework

## Tuning Method

### Full-search tuning

For each partition scheme:

- sweep candidate `hnsw_ef` values with all 44 shards
- choose the highest-QPS point with `Recall@10 >= 0.88`

### Selective-search tuning

For each partition scheme:

- sweep `nprobe` in `{1, 2, 4, 8, 16, 32, 44}`
- sweep candidate `hnsw_ef` values
- choose the highest-QPS point with `Recall@10 >= 0.88`

Tuning uses a smaller query set to control runtime.
The chosen operating point is then re-evaluated on the full 10000-query set.

## Output Artifacts

Write results under a new timestamped directory, for example:

- `results/qdrant_partition_scheme/<timestamp>/builds.csv`
- `results/qdrant_partition_scheme/<timestamp>/partition_stats.csv`
- `results/qdrant_partition_scheme/<timestamp>/full_tuning.csv`
- `results/qdrant_partition_scheme/<timestamp>/selective_tuning.csv`
- `results/qdrant_partition_scheme/<timestamp>/best_points.csv`
- `results/qdrant_partition_scheme/<timestamp>/stability_runs.csv`
- `results/qdrant_partition_scheme/<timestamp>/summary.json`

## Primary Questions

1. In full-search, does KMeans beat hash-like partitioning at matched recall?
2. In selective-search, does KMeans beat hash-like partitioning at matched recall?
3. Does Balanced-KMeans recover any loss observed with plain KMeans?
4. Is the earlier “hash beats KMeans” result mostly due to partition quality or due to Qdrant architecture differences?

## Reference Baseline

Use the existing auto-hash 44-shard Qdrant result as an additional reference, not as part of the strict controlled matrix:

- auto-hash full-search tuned around `Recall@10 ≈ 0.88083`
- result files already exist under `results/hnsw/20260415_120959/` and `results/hash44_stability/20260415_121142/`

Comparing:

- `auto-hash full-search`
- `custom hash-like full-search`

will indicate how much of the previous discrepancy was architectural rather than geometric.

## Success Criteria

- Three partition schemes are built inside the same custom-sharding framework
- Both full-search and selective-search are tuned to matched recall
- Best points are re-measured on 10000 queries
- Stability runs are recorded for the final chosen points
- Final analysis gives a general recommendation on shard partitioning in Qdrant
