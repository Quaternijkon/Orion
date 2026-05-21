# HNSW Experiments

This repository now includes a small local experiment runner at [tools/hnsw_experiment.py](../tools/hnsw_experiment.py).
It talks directly to a running Qdrant instance over HTTP and is meant for parameter sweeps such as:

- collection-level `m`
- collection-level `ef_construct`
- query-time `hnsw_ef`

The script measures:

- collection create / insert / indexing wait time
- exact-search latency using `params.exact=true`
- approximate-search latency using `params.hnsw_ef=<value>`
- `recall@k` and `hit@1` against the exact-search baseline

It supports two dataset modes:

- synthetic data generated inside the script
- HDF5 datasets with `train` / `test` / `neighbors` style splits

## Why `full_scan_threshold=10` by default

Qdrant may prefer full scan on small collections if `full_scan_threshold` is large.
That makes HNSW parameter changes look ineffective even though the query planner is simply bypassing HNSW.

For that reason, the script defaults to:

- `--full-scan-threshold 10`
- `--indexing-threshold 10`

That keeps the experiment on the HNSW path even for small local validation datasets.

## Quick Validation Run

```bash
python3 tools/hnsw_experiment.py \
  --base-url http://localhost:6333 \
  --collection-prefix hnsw_smoke \
  --num-points 2000 \
  --num-queries 20 \
  --dim 64 \
  --m-values 8 16 \
  --ef-construct-values 64 \
  --query-ef-values 16 64 128
```

This is a good first run to verify:

- the Docker instance is reachable
- indexing completes
- recall increases as `hnsw_ef` grows
- output files are generated as expected

## HDF5 Dataset Run

The script can also read an HDF5 ANN benchmark dataset directly.
When `neighbors` is present, recall is computed against the dataset's official ground truth instead of Qdrant exact-search results.

Example using the `glove-200-angular.hdf5` layout:

```bash
python3 tools/hnsw_experiment.py \
  --dataset-source hdf5 \
  --hdf5-path /home/taig/dry/dHNSW/hnswlib/datasets/glove-200-angular.hdf5 \
  --collection-prefix hnsw_glove_ref \
  --num-points 1183514 \
  --num-queries 10000 \
  --exact-num-queries 100 \
  --distance Cosine \
  --dim 200 \
  --m-values 32 \
  --ef-construct-values 100 \
  --query-ef-values 20 60 100 \
  --top-k 10 \
  --batch-size 1024 \
  --timeout-sec 7200
```

Notes:

- for HDF5 runs, the script streams `train` in batches instead of loading the full base set into Python objects first
- when `distance=Cosine`, both base vectors and queries are normalized before upload / search to match angular-style evaluation
- `--exact-num-queries` controls only the exact-latency probe count; recall can still be computed on all HDF5 queries via `neighbors`

## Larger Sweep Example

```bash
python3 tools/hnsw_experiment.py \
  --base-url http://localhost:6333 \
  --collection-prefix hnsw_formal \
  --num-points 50000 \
  --num-queries 200 \
  --dim 128 \
  --m-values 8 16 32 \
  --ef-construct-values 64 128 256 \
  --query-ef-values 16 32 64 128 256 \
  --top-k 10 \
  --batch-size 512 \
  --timeout-sec 1200
```

## Output Files

Each run writes a timestamped directory under `results/hnsw/`:

- `summary.csv`: one row per `(m, ef_construct, query_ef)` result
- `builds.csv`: one row per collection build configuration
- `details.json`: full structured output for later analysis

The most useful columns in `summary.csv` are:

- `recall_at_k_mean`
- `hit_at_1_mean`
- `approx_latency_p50_ms`
- `approx_latency_p95_ms`
- `build_secs`
- `wait_index_secs`
- `ground_truth_source`
- `evaluated_queries`
- `exact_latency_queries`

## Reproducible Environment

If you want later experiments to stay comparable with the recent shard-routing and adaptive-`ef` studies in this workspace, keep both the Qdrant server settings and the client launch settings fixed.

### Docker Compose distributed native baseline

For validating the official native hash-sharded baseline in a single-machine simulated distributed cluster, use the dedicated compose file:

```bash
docker compose \
  -f tools/compose/docker-compose.naive-cluster.yaml \
  -p qdrant-naive \
  up -d
```

The default layout starts three Qdrant nodes on one host:

- node 1 HTTP/gRPC: `localhost:6633` / `localhost:6634`, CPU set `0-5`
- node 2 HTTP/gRPC: `localhost:6643` / `localhost:6644`, CPU set `6-10`
- node 3 HTTP/gRPC: `localhost:6653` / `localhost:6654`, CPU set `11-15`

The compose file keeps the total search and optimizer CPU budgets aligned with the earlier 16-core experiments by defaulting to `6 + 5 + 5` worker threads across the three nodes. Override the image, ports, or CPU layout with environment variables such as `QDRANT_IMAGE`, `QDRANT_NODE_1_HTTP_PORT`, or `QDRANT_NODE_2_CPUSET`.

Confirm the cluster formed before running benchmarks:

```bash
curl -sS http://localhost:6633/cluster
```

Run a fast native hash-shard smoke test with the same `tools/hnsw_experiment.py` path used by the single-node baseline:

```bash
python3 tools/hnsw_experiment.py \
  --base-url http://localhost:6633 \
  --collection-prefix hnsw_cluster_smoke \
  --num-points 2000 \
  --num-queries 20 \
  --dim 64 \
  --collection-shard-number 44 \
  --m-values 8 \
  --ef-construct-values 64 \
  --query-ef-values 16 64 \
  --timeout-sec 600 \
  --keep-collections
```

For tiny smoke runs, `indexed_vectors_count` may be lower than `points_count` because Qdrant can leave small residual shard segments on the plain full-scan path after optimizer work is done. Treat `status=green`, full `points_count`, and empty optimizer queues as the smoke completion signal.

Check that the collection has local and remote shards across the compose cluster:

```bash
curl -sS \
  http://localhost:6633/collections/hnsw_cluster_smoke_synthetic_d64_n2000_s44_m8_efc64_fs10/cluster
```


Full GloVe-200 native hash-shard validation used the same compose cluster and the official auto-sharding path:

```bash
python3 tools/hnsw_experiment.py \
  --base-url http://localhost:6633 \
  --dataset-source hdf5 \
  --hdf5-path /home/taig/dry/faiss/datasets/glove-200-angular.hdf5 \
  --collection-prefix hash44_compose_cluster \
  --num-points 1183514 \
  --num-queries 10000 \
  --exact-num-queries 100 \
  --distance Cosine \
  --dim 200 \
  --collection-shard-number 44 \
  --m-values 32 \
  --ef-construct-values 100 \
  --query-ef-values 30 \
  --top-k 10 \
  --batch-size 1024 \
  --timeout-sec 7200 \
  --keep-collections
```

The verified 2026-05-19 full run wrote:

- `results/hnsw/20260519_104756/summary.csv`
- `results/hnsw/20260519_104756/builds.csv`
- `results/hnsw/20260519_104756/details.json`

Collection check for `hash44_compose_cluster_glove_200_angular_d200_n1183514_s44_m32_efc100_fs10`:

- `status=green`
- `optimizer_status=ok`
- `shard_number=44`
- `points_count=1183514`
- `indexed_vectors_count=1183514`
- `segments_count=88`
- collection cluster view from node 1 reported `shard_count=44`, 14 local active shards, and 30 remote active shards split across the other two peers.

Full-run metrics at `hnsw_ef=30`:

- build time: `242.41s`
- index wait time: `17.32s`
- exact-search sample: 100 queries, mean `16.91ms`, p95 `18.70ms`
- approximate search over 10000 queries: recall@10 `0.88083`, hit@1 `0.9047`, mean `3.53ms`, p50 `3.51ms`, p95 `4.20ms`

Batch throughput can be checked on the already-built collection without rebuilding:

```bash
python3 tools/qdrant_batch_search_benchmark.py \
  --base-url http://localhost:6633 \
  --collection hash44_compose_cluster_glove_200_angular_d200_n1183514_s44_m32_efc100_fs10 \
  --hdf5-path /home/taig/dry/faiss/datasets/glove-200-angular.hdf5 \
  --query-count 10000 \
  --top-k 10 \
  --hnsw-ef 30 \
  --batch-sizes 50 100 200
```

The verified 2026-05-19 batch run wrote `results/qdrant_batch_bench/20260519_125547/summary.json` and measured:

- batch size 50: recall@10 `0.88083`, `852.67` QPS
- batch size 100: recall@10 `0.88083`, `854.91` QPS
- batch size 200: recall@10 `0.88083`, `1035.41` QPS

Clean up the cluster and its named volumes when done:

```bash
docker compose \
  -f tools/compose/docker-compose.naive-cluster.yaml \
  -p qdrant-naive \
  down -v
```

### Docker Compose distributed idea baseline

For validating the two-level idea implementation on the same single-machine multi-node shape, use the dedicated idea compose file:

```bash
docker compose \
  -f tools/compose/docker-compose.idea-cluster.yaml \
  -p qdrant-idea \
  up -d
```

The default layout starts three independent Qdrant nodes and storage volumes:

- node 1 HTTP/gRPC: `localhost:6733` / `localhost:6734`, CPU set `0-5`
- node 2 HTTP/gRPC: `localhost:6743` / `localhost:6744`, CPU set `6-10`
- node 3 HTTP/gRPC: `localhost:6753` / `localhost:6754`, CPU set `11-15`

Confirm the cluster formed:

```bash
curl -sS http://localhost:6733/cluster
```

The idea experiment now defaults to `--routing-mode faithful_original_rest` in `tools/qdrant_two_level_routing_experiment.py`. This mode preserves the original C++ idea construction and routing path as far as Qdrant's public REST API can express it:

- global upper-tier sample of `N / M_GLOBAL` points before partitioning
- upper HNSW over those L1 labels
- `point_to_l1s` computed for every training point with `K_OVERLAP=10`
- weighted balanced KMeans over L1 nodes
- L1 topology convergence with the capacity guard
- simulated L1-weight recalibration
- fission-based auto-sharding
- full-data vote-based multi-assignment into custom shard keys
- query routing from upper candidates through every shard in `point_to_shards[candidate]`
- dynamic per-shard `hnsw_ef = base + factor * routed_entry_count`

There is one explicit REST-level non-equivalence: Qdrant search accepts `shard_key` and `hnsw_ef`, but it does not expose hnswlib-style per-shard MultiEP entry point injection. The script records this in `summary.json` as `qdrant_rest_equivalence_gap`. Full equivalence requires a Qdrant internal/Rust change that lets lower-tier search start from routed L1 entry point IDs. The script also records `rng_equivalence_note`: the port preserves the original seed roles and algorithm order, but Python/NumPy random streams are not bit-identical to libstdc++ `std::mt19937/std::shuffle` or C `rand()`. The previous simplified implementation remains available as `--routing-mode legacy_centroid`.

Run a fast deployment smoke test:

```bash
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
python3 tools/qdrant_two_level_routing_experiment.py \
  --base-url http://localhost:6733 \
  --collection idea_faithful_original_smoke \
  --routing-mode faithful_original_rest \
  --train-limit 2000 \
  --num-shards 8 \
  --tuning-query-count 20 \
  --eval-query-count 20 \
  --upper-k-candidates 100 \
  --base-ef-candidates 20 \
  --factor-candidates 4 \
  --target-recall 0.0 \
  --stability-repeats 1 \
  --batch-size 10 \
  --upload-batch-size 256 \
  --shard-placement auto
```

The `LD_PRELOAD` prefix is only needed on local Python environments where `hnswlib` was built against a newer `libstdc++` than the one bundled with conda. The smoke run uses only a prefix of the training vectors while ground truth still references the full dataset, so its recall is not meaningful. Check shard placement after it completes:

```bash
curl -sS \
  http://localhost:6733/collections/idea_faithful_original_smoke/cluster
```

The verified 2026-05-21 faithful smoke run wrote `results/qdrant_two_level_faithful_smoke/20260521_065310/summary.json` and confirmed:

- `routing_mode=faithful_original_rest`
- `logical_points_count=2000`
- `assigned_points_count=2755`
- `expansion_ratio=1.3775`
- initial `num_shards=8`, final `num_shards=11` after fission
- `cluster_shard_count=11`
- `cluster_peer_count=3`
- `cluster_active_shards=11`

For a full GloVe-200 faithful idea benchmark on the compose cluster, remove `--train-limit` and use the original default construction/search parameters:

```bash
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
python3 tools/qdrant_two_level_routing_experiment.py \
  --base-url http://localhost:6733 \
  --collection qdrant_original_idea_cluster \
  --hdf5-path /home/taig/dry/faiss/datasets/glove-200-angular.hdf5 \
  --routing-mode faithful_original_rest \
  --num-shards 31 \
  --kmeans-iters 10 \
  --tuning-query-count 1000 \
  --eval-query-count 10000 \
  --top-k 10 \
  --hnsw-m 32 \
  --ef-construct 100 \
  --upper-m 32 \
  --upper-ef-construction 100 \
  --upper-search-ef 100 \
  --k-overlap 10 \
  --upper-k-candidates 100 \
  --base-ef-candidates 20 \
  --factor-candidates 4 \
  --target-recall 0.88 \
  --stability-repeats 3 \
  --batch-size 100 \
  --shard-placement auto
```

The older verified 2026-05-20 runs under `results/qdrant_two_level/20260520_113331/` and `results/qdrant_two_level_match_recall/20260520_115647/` used `--routing-mode legacy_centroid` semantics before this faithful migration. Treat those numbers as simplified Qdrant routing results, not as measurements of the original C++ idea algorithm.

The faithful compose path keeps the same lower-layer Qdrant HNSW build parameters as the native baseline:

- lower Qdrant HNSW `m=32`
- lower Qdrant HNSW `ef_construct=100`
- `full_scan_threshold=10`
- `indexing_threshold=10`
- custom shard keys placed across the discovered Qdrant peers

Do not compare the old `legacy_centroid` QPS against the native baseline as if it were the original idea. A matched-recall comparison must be rerun with `routing_mode=faithful_original_rest`; even then, the summary's `qdrant_rest_equivalence_gap` remains until Qdrant exposes lower-tier MultiEP search.

For real multi-machine migration, translate each compose service into a per-node Docker command, replace `qdrant_idea_1`/`qdrant_idea_2`/`qdrant_idea_3` with real node IPs or DNS names, expose node-to-node P2P port `6335`, and keep the client command unchanged except for `--base-url`.

### Server-side Qdrant settings

The runs in this repository were executed with Qdrant pinned to 16 CPU cores:

- container CPU set: `0-15`
- `QDRANT__STORAGE__PERFORMANCE__MAX_SEARCH_THREADS=16`
- `QDRANT__STORAGE__PERFORMANCE__OPTIMIZER_CPU_BUDGET=16`

Two container patterns were used:

1. Default single-node collection experiments such as hash-based `shard_number=44`
2. Distributed-mode single-node experiments for custom sharding

Reference `docker run` examples:

```bash
# Default single-node Qdrant for standard hash-sharded collection experiments
docker run -d \
  --name qdrant-pressure-16c \
  --cpuset-cpus=0-15 \
  -p 6335:6333 \
  -e QDRANT__STORAGE__PERFORMANCE__MAX_SEARCH_THREADS=16 \
  -e QDRANT__STORAGE__PERFORMANCE__OPTIMIZER_CPU_BUDGET=16 \
  qdrant/qdrant:latest

# Distributed-mode single-node Qdrant for custom shard_key experiments
docker run -d \
  --name qdrant-custom-16c \
  --cpuset-cpus=0-15 \
  -p 6336:6333 \
  -e QDRANT__STORAGE__PERFORMANCE__MAX_SEARCH_THREADS=16 \
  -e QDRANT__STORAGE__PERFORMANCE__OPTIMIZER_CPU_BUDGET=16 \
  -e QDRANT__CLUSTER__ENABLED=true \
  -e QDRANT_URI=http://localhost:6335 \
  qdrant/qdrant:latest
```

Notes:

- these runs used Docker `cpuset` pinning, not `CpuQuota` throttling
- the containers were left on the default Docker `bridge` network
- no extra memory cap was applied in Docker

### Client-side launch settings

The Python client process was not originally pinned in all historical runs, but for reproducible follow-up work it is strongly recommended to pin the client separately and force BLAS/OpenMP style thread pools to one thread.

Recommended client launch wrapper:

```bash
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

taskset -c 16-17 python3 tools/hnsw_experiment.py ...
```

This does two useful things:

- it prevents NumPy / OpenBLAS from quietly opening extra threads
- it avoids the client process competing too aggressively with the Qdrant server pinned on `0-15`

### Fair-comparison checklist

When comparing two search strategies, keep these fixed unless the experiment explicitly studies them:

- same Qdrant image / binary
- same `cpuset` and Qdrant performance env vars
- same dataset path and query count
- same collection build settings such as `m`, `ef_construct`, `full_scan_threshold`, `indexing_threshold`
- same client-side thread env vars
- same batch size for `/points/search/batch`
- same warmup policy before recording results

### Practical recommendation

For stable QPS reporting, do not rely on a single run.
Use:

1. one warmup run that is not recorded
2. at least 3 to 5 recorded runs
3. mean and standard deviation, or at least the min/max range

This is especially important when cache state and shard fan-out differ between experiments.

## Suggested Reading Order for Results

If you want a simple first-pass analysis, look at the results in this order:

1. Fix `m` and `ef_construct`, then study how `query_ef` changes recall and latency.
2. Fix a target recall, then compare which `m` gives the best latency at that recall.
3. Compare `ef_construct` mostly through build time and indexing time first, then through final recall.

In practice:

- `query_ef` mostly trades search latency for search quality.
- `m` usually increases memory/build cost and can improve recall.
- `ef_construct` mostly affects build cost and final graph quality.
