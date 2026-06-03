# 2026-06-03 Current Method4 vs Naive Same-Recall Comparison

## Goal

Compare the current method4 idea implementation against the naive all-shards
implementation under the same lower-tier HNSW build parameters and comparable
high-recall operating points.

This experiment uses the current method4 implementation after worker-local
peer pre-merge:

- commit: `c30da650d`
- image: `qdrant/qdrant:method4-peer-premerge`
- controller dispatch mode: `coordinator`

## Build And Collection Parity

Both collections are deployed in the same controller+3-worker Docker topology
and use the same lower-tier Qdrant HNSW build configuration:

| Field | Method4 idea | Naive |
|---|---:|---:|
| Collection | `qdrant_controller_idea_method4map_full_20260601` | `qdrant_controller_naive46_full_20260521` |
| Vector size / distance | 200 / Cosine | 200 / Cosine |
| Sharding | custom | custom |
| Logical custom shards | 46 | 46 |
| Segments | 92 | 92 |
| Replication factor | 1 | 1 |
| HNSW `m` | 32 | 32 |
| HNSW `ef_construct` | 100 | 100 |
| Indexed vectors | 1,400,967 | 1,183,514 |

The indexed-vector count differs because method4 multi-assignment intentionally
creates copied lower-tier points. That is part of the idea algorithm, not a
different HNSW build parameter.

## Method4 Current Point

Command:

```bash
env PYTEST_ADDOPTS= LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
python3 tools/qdrant_two_level_routing_experiment.py \
  --base-url http://localhost:6833 \
  --collection qdrant_controller_idea_method4map_full_20260601 \
  --hdf5-path /home/taig/dry/faiss/datasets/glove-200-angular.hdf5 \
  --routing-mode faithful_original_rest \
  --num-shards 31 \
  --upper-k-candidates 160 \
  --base-ef-candidates 80 \
  --factor-candidates 8 \
  --target-recall 0.955 \
  --eval-query-count 3000 \
  --tuning-query-count 500 \
  --stability-repeats 2 \
  --batch-size 200 \
  --reuse-existing \
  --recover-routing-from-collection \
  --routed-execution-mode compact_multi_ep \
  --routed-planning-mode materialized \
  --routed-result-limit-mode top_k \
  --search-dispatch-mode coordinator \
  --shard-placement map \
  --shard-placement-map results/qdrant_goal_recall_idea_095_placement_simulation/20260601_075041/placement_simulation.json \
  --shard-placement-map-name method4_aware \
  --placement-peer-uri-contains qdrant_shard_ \
  --output-dir results/qdrant_compare_current_idea_vs_naive_095_idea_b200
```

Result directory:

```text
results/qdrant_compare_current_idea_vs_naive_095_idea_b200/20260603_094930
```

Result:

| Metric | Value |
|---|---:|
| Recall@10 | 0.955267 |
| Final QPS | 396.777 |
| Stability QPS mean | 387.140 |
| Stability QPS stddev | 1.027 |
| Avg visited shards/query | 23.214 |
| Avg EF per visited shard | 141.050 |
| Search batch calls | 15 |

## Naive Same-Batch Probe

The closest-recall naive point in the `batch_size=200` probe was `ef=76`.

Result directory:

```text
results/qdrant_compare_current_idea_vs_naive_095_naive_ef76_b200/20260603_095038
```

Result:

| Metric | Value |
|---|---:|
| Recall@10 | 0.954767 |
| Final QPS | 229.678 |
| Stability QPS mean | 171.326 |
| Stability QPS stddev | 100.452 |
| Avg visited shards/query | 46.000 |
| Avg EF per visited shard | 76.000 |
| Search batch calls | 15 |

The same-batch naive run has severe tail instability at `batch_size=200`. Its
second repeat dropped to `100.295` QPS, so this point is useful as a strict same
workload stress comparison but is not the fairest stable naive operating point.

## Naive Stable Batch Points

To avoid overstating the method4 advantage from naive's batch-200 tail behavior,
naive was also run with `batch_size=100`, which is the historically more stable
batch size for all-shards search.

Closest-recall naive point:

- result dir:
  `results/qdrant_compare_current_idea_vs_naive_095_naive_ef76_b100/20260603_095227`
- `ef=76`

| Metric | Value |
|---|---:|
| Recall@10 | 0.954767 |
| Final QPS | 274.364 |
| Stability QPS mean | 272.370 |
| Stability QPS stddev | 3.843 |
| Avg visited shards/query | 46.000 |
| Avg EF per visited shard | 76.000 |
| Search batch calls | 30 |

Higher-recall naive point:

- result dir:
  `results/qdrant_compare_current_idea_vs_naive_095_naive_ef80_b100/20260603_095342`
- `ef=80`

| Metric | Value |
|---|---:|
| Recall@10 | 0.957333 |
| Final QPS | 265.691 |
| Stability QPS mean | 264.645 |
| Stability QPS stddev | 2.612 |
| Avg visited shards/query | 46.000 |
| Avg EF per visited shard | 80.000 |
| Search batch calls | 30 |

## Comparison

Using the stable QPS means:

| Comparison | Recall@10 | QPS mean | Relative QPS |
|---|---:|---:|---:|
| Method4 current, `upper_k=160, base_ef=80, factor=8, batch=200` | 0.955267 | 387.140 | baseline |
| Naive closest recall, `ef=76, batch=100` | 0.954767 | 272.370 | method4 `+42.14%` |
| Naive higher recall, `ef=80, batch=100` | 0.957333 | 264.645 | method4 `+46.29%` |
| Naive strict same batch, `ef=76, batch=200` | 0.954767 | 171.326 | method4 `+125.97%` |

The main fair comparison is the first naive stable batch row:

```text
Method4 current: Recall@10 0.955267, QPS mean 387.140
Naive ef=76:     Recall@10 0.954767, QPS mean 272.370
Delta:           +42.14% QPS for method4
```

The conservative higher-recall naive point still leaves method4 ahead:

```text
Method4 current: Recall@10 0.955267, QPS mean 387.140
Naive ef=80:     Recall@10 0.957333, QPS mean 264.645
Delta:           +46.29% QPS for method4
```

## Interpretation

At the same high-recall level, current method4 is faster because it searches
about half as many logical lower shards per query:

```text
method4 avg visited shards: 23.214
naive avg visited shards:   46.000
```

The method4 per-visited-shard EF is higher because dynamic EF scales with routed
entry-point count, but the total routed lower-tier work is still competitive:

```text
method4 estimated EF-sum/query: 23.214 * 141.050 ~= 3274
naive ef=76 EF-sum/query:       46.000 * 76     = 3496
naive ef=80 EF-sum/query:       46.000 * 80     = 3680
```

The current worker-local peer pre-merge also reduces controller fan-in for
method4 after the lower-tier searches complete. This does not change the method4
algorithm core; it makes the distributed execution shape better match method4's
many-logical-shards/few-physical-workers routing pattern.

## Decision

Keep the current method4 implementation as the stronger high-recall operating
point.

For future comparisons, use two rows:

1. Strict same workload: same `batch_size`.
2. Best stable workload: each method gets a stable batch size, while preserving
   the same HNSW build parameters and comparable Recall@10.
