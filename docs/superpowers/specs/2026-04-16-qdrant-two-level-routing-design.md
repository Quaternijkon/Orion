# Qdrant Two-Level Routing Experiment Design

## Goal

Run a Qdrant-only experiment that mirrors the two-level search structure in
`/home/taig/dry/dHNSW/hnswlib/examples/cpp/method0andmethod3-try1.cpp`
without modifying the user's existing implementation code.

The target architecture is:

1. One Qdrant collection
2. `44` shards inside that collection
3. Data assigned to shards by KMeans
4. A user-side global HNSW built from a random `1/32` sample from each shard
5. Query-time routing driven by the user-side HNSW
6. Per-shard Qdrant `hnsw_ef` derived from the number of routed upper-layer hits

The experiment must tune only search-side parameters until `Recall@10 >= 0.88`,
then report QPS and related performance characteristics.

## Architecture

### Lower Layer: Qdrant Collection

- Backend: Qdrant distributed-mode single-node instance at `http://localhost:6336`
- Collection layout: one collection, custom sharding, `44` shard keys
- Data partitioning: KMeans assignment to shard keys
- Collection HNSW parameters:
  - `m = 32`
  - `ef_construct = 100`
  - `full_scan_threshold = 10`
  - `indexing_threshold = 10`

### Upper Layer: User-Side HNSW

For each shard:

- randomly sample `1/32` of that shard's points
- merge all sampled points from all shards into one global user-side HNSW
- store, for each sampled point, its backing Qdrant shard identity

The upper-layer HNSW is global, not one HNSW per shard. This matches the intent
of the user's C++ implementation, where an upper tier provides cross-shard
routing evidence before lower-tier shard searches begin.

## Query Flow

For each query:

1. Search the upper-layer HNSW with `ef=100`
2. Take the upper-layer result set and count how many returned sample points belong to each shard
3. For each shard:
   - if count is `0`, do not query that shard
   - otherwise set:
     - `shard_ef = base_ef + factor * routed_count`
4. Query only the selected shards in Qdrant
5. Merge all returned candidates client-side into a global top-`k`

## Initial Parameterization

The first tested routing rule is fixed by the user:

- upper-layer search `ef = 100`
- base lower-layer `ef = 20`
- per-routed-point increment `= 4`
- shards with zero routed points are skipped

This corresponds to:

- `shard_ef = 20 + 4 * routed_count`

## Tunable Parameters

To reach `Recall@10 >= 0.88`, only search-side parameters may be tuned.

The tuning order should be:

1. upper-layer result count `upper_k`
2. lower-layer `base_ef`
3. lower-layer increment factor `factor`

The following must stay fixed:

- KMeans sharding itself
- number of shards
- sample ratio `1/32`
- one global upper-layer HNSW
- no changes to the user's original implementation source

## Dataset And Evaluation

- Dataset: `glove-200-angular.hdf5`
- Path: `/home/taig/dry/faiss/datasets/glove-200-angular.hdf5`
- Ground truth: HDF5 `neighbors`
- Evaluation target:
  - `Recall@10 >= 0.88`
- Metrics to report:
  - `Recall@10`
  - `QPS`
  - wall-clock time
  - average visited shards per query
  - average upper-layer routed hits per query
  - average assigned lower-layer `ef` per visited shard

## Environment

Keep the same environment constraints used in the recent Qdrant experiments:

### Server

- Qdrant container: `qdrant-custom-16c`
- CPU set: `0-15`
- `QDRANT__STORAGE__PERFORMANCE__MAX_SEARCH_THREADS=16`
- `QDRANT__STORAGE__PERFORMANCE__OPTIMIZER_CPU_BUDGET=16`

### Client

- `taskset -c 16-17`
- `OPENBLAS_NUM_THREADS=1`
- `OMP_NUM_THREADS=1`
- `MKL_NUM_THREADS=1`
- `NUMEXPR_NUM_THREADS=1`

## Output Artifacts

Write outputs under a new timestamped directory, for example:

- `results/qdrant_two_level/<timestamp>/builds.csv`
- `results/qdrant_two_level/<timestamp>/routing_tuning.csv`
- `results/qdrant_two_level/<timestamp>/final_metrics.csv`
- `results/qdrant_two_level/<timestamp>/summary.json`

## Expected Questions This Experiment Answers

1. Can a Qdrant-based two-level routing structure reach `Recall@10 >= 0.88`?
2. How many shards are visited on average under this routing policy?
3. Does this upper-tier-guided adaptive lower-layer `ef` strategy outperform the previously tested custom-shard selective baselines?
4. Is the C++-style “upper layer for routing, lower layer for shard-local search” idea still valuable when realized on top of Qdrant?

## Success Criteria

- A new standalone experiment script is created
- No existing user implementation files are modified
- The experiment reaches at least `Recall@10 = 0.88`
- Final QPS and routing statistics are recorded and reported
