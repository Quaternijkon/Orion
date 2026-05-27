# Method4 High-Recall Batch Sweep

Date: 2026-05-27

Base commit before this note:

```text
586a10839 Add routed MultiEP planning for method4 experiments
```

## Purpose

Measure whether high-recall QPS is sensitive to batch and fanout granularity
after the method4 routed MultiEP path is materialized. This is a system-level
diagnostic, not an algorithm change.

## Fixed Architecture

- Controller endpoint: `http://localhost:6833`
- Dataset: `/home/taig/dry/faiss/datasets/glove-200-angular.hdf5`
- Idea collection: `qdrant_controller_idea_full_20260521`
- Naive collection: `qdrant_controller_naive46_full_20260521`
- Qdrant HNSW build parameters: `m=32`, `ef_construct=100`
- Dispatch mode: `coordinator`
- Result limit mode for routed idea: `top_k`
- Routed execution mode: `compact_multi_ep`
- Routed planning mode: `materialized`

## Idea Light

Parameters:

```text
upper_k=160, base_ef=64, factor=8
```

| batch_size | Recall@10 | QPS | Avg visited shards | Avg EF per visited shard |
|---:|---:|---:|---:|---:|
| 50 | 0.95003 | 274.65 | 22.68 | 126.26 |
| 100 | 0.94940 | 330.30 | 22.69 | 126.24 |
| 200 | 0.94943 | 343.54 | 22.68 | 126.26 |
| 400 | 0.94967 | 343.89 | 22.67 | 126.31 |

This parameter set is near the 0.95 boundary and should not be used as the
strict high-recall comparison point.

## Idea Strict

Parameters:

```text
upper_k=160, base_ef=80, factor=8
```

| batch_size | Recall@10 | QPS | Avg visited shards | Avg EF per visited shard |
|---:|---:|---:|---:|---:|
| 100 | 0.95350 | 321.70 | 22.68 | 142.26 |
| 200 | 0.95383 | 323.14 | 22.66 | 142.31 |
| 400 | 0.95353 | 317.36 | 22.64 | 142.39 |

The best strict point in this sweep is `batch_size=200`.

## Naive

Parameters:

```text
ef=72
```

| batch_size | Recall@10 | QPS | Avg visited shards | Avg EF per visited shard |
|---:|---:|---:|---:|---:|
| 50 | 0.95177 | 217.71 | 46.00 | 72.00 |
| 100 | 0.95177 | 282.06 | 46.00 | 72.00 |
| 200 | 0.95177 | 237.65 | 46.00 | 72.00 |
| 400 | 0.95177 | 65.19 | 46.00 | 72.00 |

The naive `batch_size=400` run shows severe tail-latency or queueing effects
and should not be treated as representative without repetition.

## Paired Confirmation

| Method | Parameters | Recall@10 | QPS | Avg visited shards | Avg EF per visited shard |
|---|---|---:|---:|---:|---:|
| naive | `ef=72, batch_size=100` | 0.95177 | 272.60 | 46.00 | 72.00 |
| idea strict | `upper_k=160, base_ef=80, factor=8, batch_size=200` | 0.95357 | 328.26 | 22.69 | 142.25 |

In this paired run, idea strict is faster by about 20.4% while also reaching a
slightly higher recall.

## Interpretation

The batch sweep supports two conclusions:

1. High-recall performance is strongly affected by system execution details
   such as batch granularity, fanout, queueing, and tail latency.
2. The observed 20% idea advantage is plausible and likely conservative,
   because method4 still runs through a client-side experimental planning path
   instead of a controller-native routing and shard-major execution engine.

## Next Experiment

Run paired repeats for:

- naive: `ef=72, batch_size=100`
- idea strict: `upper_k=160, base_ef=80, factor=8, batch_size=200`

Use at least three repetitions and compare mean QPS, QPS standard deviation,
and recall stability.
