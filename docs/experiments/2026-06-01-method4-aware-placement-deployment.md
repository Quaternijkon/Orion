# Method4-Aware Placement Deployment

Date: 2026-06-01

Implementation commit:

```text
698f15805 Support method4 placement maps for shard creation
```

## Purpose

Turn the method4-aware placement simulation into a real collection deployment.
The collection uses the same method4 routing/build path, but custom shard keys
are physically placed according to the simulated `method4_aware` placement map
instead of round-robin.

## Collection

```text
qdrant_controller_idea_method4map_full_20260601
```

Placement source:

```text
results/qdrant_goal_recall_idea_095_placement_simulation/20260601_075041/placement_simulation.json
```

The deployed collection has:

| Metric | Value |
|---|---:|
| Points / indexed vectors | 1,400,967 |
| Custom shards | 46 |
| Active shards | 46 |
| Physical shard counts | 15 / 15 / 16 |

The actual peer assignment count is:

```text
2980601005324529: 15
6015626418395790: 15
6846760844865837: 16
```

## Build And First Evaluation Command

```bash
env PYTEST_ADDOPTS= LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 python3 tools/qdrant_two_level_routing_experiment.py \
  --base-url http://localhost:6833 \
  --collection qdrant_controller_idea_method4map_full_20260601 \
  --hdf5-path /home/taig/dry/faiss/datasets/glove-200-angular.hdf5 \
  --routing-mode faithful_original_rest \
  --num-shards 31 \
  --upper-k-candidates 160 \
  --base-ef-candidates 80 \
  --factor-candidates 8 \
  --target-recall 0.95 \
  --eval-query-count 3000 \
  --tuning-query-count 500 \
  --stability-repeats 0 \
  --batch-size 200 \
  --routed-execution-mode compact_multi_ep \
  --routed-planning-mode materialized \
  --routed-result-limit-mode top_k \
  --search-dispatch-mode coordinator \
  --shard-placement map \
  --shard-placement-map results/qdrant_goal_recall_idea_095_placement_simulation/20260601_075041/placement_simulation.json \
  --shard-placement-map-name method4_aware \
  --placement-peer-uri-contains qdrant_shard_ \
  --output-dir results/qdrant_goal_recall_idea_095_method4map_deploy
```

First run result:

| Recall@10 | QPS | Avg visited shards | Avg EF per visited shard |
|---:|---:|---:|---:|
| 0.955000 | 336.273 | 23.191 | 141.112 |

## Confirmation Run

The deployed collection was then evaluated with:

```text
--reuse-existing
--recover-routing-from-collection
--stability-repeats 2
```

Confirmation directory:

```text
results/qdrant_goal_recall_idea_095_method4map_confirm/20260601_095702
```

| Row | Recall@10 | QPS |
|---|---:|---:|
| final | 0.955233 | 337.048 |
| stability 1 | 0.955233 | 338.433 |
| stability 2 | 0.955233 | 332.432 |

Combined method4-aware placement rows:

| Statistic | Value |
|---|---:|
| Mean QPS | 336.046 |
| QPS stddev | 2.570 |
| Mean Recall@10 | 0.955175 |

## Comparison

Previous round-robin idea strict paired repeats:

| Statistic | Value |
|---|---:|
| Mean QPS | 333.748 |
| QPS stddev | 0.779 |
| Mean Recall@10 | 0.953644 |

Previous naive paired repeats:

| Statistic | Value |
|---|---:|
| Mean QPS | 277.234 |
| Mean Recall@10 | 0.951767 |

Relative performance:

| Comparison | QPS gain |
|---|---:|
| Method4-aware placement vs round-robin idea | +0.69% |
| Method4-aware placement vs naive | +21.21% |

## Interpretation

The real deployment confirms that the placement map is usable and preserves the
method4 behavior at high recall. It also improves QPS slightly over the previous
round-robin idea runs, while maintaining a higher recall.

However, the real gain is smaller than the placement simulator's 3-5% query-tail
load reduction. This suggests that, in the current Qdrant Docker architecture,
physical shard placement is not the only dominant bottleneck. The next
architecture work should target:

1. Controller-native method4 routing, to remove Python route-plan construction
   and JSON map overhead.
2. Shard-major lower execution, to batch many queries per selected shard inside
   each worker.
3. Worker-side scheduling using dynamic EF as a task weight, to reduce tail
   latency when routed shards have uneven EF.
