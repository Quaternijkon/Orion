# 2026-06-02 Method4 Fast-Path Negative Controls

## Goal

Evaluate whether additional system-level tweaks on top of server-native shard-major method4 improve high-recall QPS without changing method4 routing semantics:

- Keep `upper_k=160`, `base_ef=80`, `factor=8`.
- Keep `compact_multi_ep`, materialized route planning, method4-aware placement, and `top_k` lower result limit.
- Keep `Recall@10 >= 0.95`.
- Reject any candidate that does not produce a repeatable positive QPS effect.

## Collection And Baseline

- Idea collection: `qdrant_controller_idea_method4map_full_20260601`
- Naive collection: `qdrant_controller_naive46_full_20260521`
- Dataset: `/home/taig/dry/faiss/datasets/glove-200-angular.hdf5`
- Controller: `http://localhost:6833`
- Existing trusted server-native shard-major commit: `b3449382a`

## Candidates Tested

### Candidate 1: Controller Fast-Path Micro-Optimization

Implementation attempted:

- Replace linear shard-major group lookup with a shard-key hash index.
- Replace full candidate flatten+sort with an incremental per-shard-head merge.
- Use `AHashSet` for source-id de-duplication.

The code was covered by a new unit test and passed targeted shard-major tests, but A/B results were not stable enough to keep.

| Run | Binary | Batch | Recall@10 | QPS Values | Mean QPS |
|---|---|---:|---:|---|---:|
| `old_ab` | `b3449382a` | 200 | 0.955167 | 369.508, 373.154, 371.648 | 371.437 |
| `new_ab1` | fast-path candidate | 200 | 0.955267 | 373.238, 377.411, 386.422 | 379.024 |
| `new_ab2` | fast-path candidate | 200 | 0.955567 | 357.576, 372.445, 363.265 | 364.429 |

Combined candidate mean across both new runs: `371.714`.

Conclusion: the average is only marginally above old (`+0.07%`) and one repeat is clearly worse. This is not a reliable positive effect. The code change was removed.

Result directories:

- `results/qdrant_goal_recall_idea_095_server_shard_major_old_ab/20260602_133751`
- `results/qdrant_goal_recall_idea_095_server_shard_major_fastpath/20260602_133000`
- `results/qdrant_goal_recall_idea_095_server_shard_major_fastpath_ab2/20260602_133951`

### Candidate 2: Batch Size 400

| Batch | Recall@10 | Final QPS | Stability Mean QPS |
|---:|---:|---:|---:|
| 400 | 0.955767 | 342.972 | 358.215 |

Conclusion: rejected. Larger batches reduce request count but hurt the actual QPS in this topology.

Result directory:

- `results/qdrant_goal_recall_idea_095_server_shard_major_b400_probe/20260602_134147`

### Candidate 3: Batch Size 100

| Batch | Recall@10 | Final QPS | Stability Mean QPS |
|---:|---:|---:|---:|
| 100 | 0.955300 | 333.569 | 332.644 |

Conclusion: rejected. Smaller batches increase dispatch overhead and perform worse.

Result directory:

- `results/qdrant_goal_recall_idea_095_server_shard_major_b100_probe/20260602_134311`

## Same-Binary Naive Check

The naive rerun after candidate deployment produced much lower QPS than the 2026-06-01 same-binary baseline:

| Run | Recall@10 | Final QPS | Stability Mean QPS |
|---|---:|---:|---:|
| 2026-06-01 naive after server shard-major | 0.951767 | 310.550 | 307.390 |
| 2026-06-02 naive after fast-path candidate | 0.951767 | 265.155 | 267.547 |

This confirmed that historical numbers from different cluster restart states were not sufficient for judging the micro-optimization. That is why the direct old/new A/B above was used for the final decision.

## Final Decision

Do not keep any 2026-06-02 candidate change.

The correct retained implementation remains `b3449382a` server-native shard-major method4. It is still the strongest validated architecture change so far, but the attempted micro-optimization and batch-size variants did not provide a repeatable positive result.

The next implementation work should move to a larger architectural boundary with clearer expected leverage:

- worker-local pre-merge at the physical peer/RPC layer, or
- controller-native method4 routing manifest with stage-level trace instrumentation.

Small controller-side merge micro-optimizations are not the current bottleneck.
