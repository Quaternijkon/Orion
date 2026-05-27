# Method4 High-Recall Paired Repeats

Date: 2026-05-27

Base experiment note:

```text
ca651424a Record method4 high-recall batch sweep
```

## Purpose

Repeat the strict high-recall comparison to check whether the previously
observed 20% idea advantage is stable rather than a single-run artifact.

## Fixed Setup

- Controller endpoint: `http://localhost:6833`
- Dataset: `/home/taig/dry/faiss/datasets/glove-200-angular.hdf5`
- Idea collection: `qdrant_controller_idea_full_20260521`
- Naive collection: `qdrant_controller_naive46_full_20260521`
- Dispatch mode: `coordinator`
- Evaluation queries per run: 3000
- Tuning queries per run: 500

## Compared Points

Naive:

```text
routing_mode=naive_hash_all_shards
num_shards=46
base_ef=72
batch_size=100
```

Idea strict:

```text
routing_mode=faithful_original_rest
routed_execution_mode=compact_multi_ep
routed_planning_mode=materialized
upper_k=160
base_ef=80
factor=8
batch_size=200
```

## Final Rows

| Method | Repeat | Recall@10 | QPS | Wall s | Avg visited shards | Avg EF per visited shard |
|---|---:|---:|---:|---:|---:|---:|
| naive | 1 | 0.951767 | 278.632 | 10.767 | 46.000 | 72.000 |
| naive | 2 | 0.951767 | 281.490 | 10.658 | 46.000 | 72.000 |
| naive | 3 | 0.951767 | 271.579 | 11.047 | 46.000 | 72.000 |
| idea strict | 1 | 0.953833 | 332.981 | 9.010 | 22.663 | 142.311 |
| idea strict | 2 | 0.953633 | 334.538 | 8.968 | 22.677 | 142.272 |
| idea strict | 3 | 0.953467 | 333.726 | 8.989 | 22.660 | 142.325 |

## Summary

| Method | n | Mean QPS | QPS stddev | Mean Recall@10 | Recall stddev |
|---|---:|---:|---:|---:|---:|
| naive | 3 | 277.234 | 5.101 | 0.951767 | 0.000000 |
| idea strict | 3 | 333.748 | 0.779 | 0.953644 | 0.000184 |

Mean speedup:

```text
333.748 / 277.234 = 1.2039
speedup = 20.39%
```

## Interpretation

The strict method4 point is consistently faster than the naive point while
also reaching slightly higher recall. The repeated result supports treating
the 20% advantage as a stable lower-bound result for the current experimental
architecture, not as a one-off run.

The next architecture experiment should target the remaining system overhead:

1. Move method4 route planning into the controller runtime.
2. Replace query-major lower execution with shard-major worker batches.
3. Add method4-aware physical placement or hot-shard replica scheduling.
