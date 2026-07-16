# 2026-07-04 Method4 Claim G Physical Layout Evidence

## Claim

Claim G:

```text
Method4-aware physical layout reduces hot-worker load and tail latency.
```

This evidence package evaluates the physical-layout part of Method4 without
changing the upper routing, per-shard entry points, dynamic EF formula, or lower
HNSW search semantics.

## Evidence Package

| Artifact | Path |
|---|---|
| Offline placement summary | `results/method4_claim_g_evidence_20260704/offline_placement_summary.csv` |
| Offline placement deltas | `results/method4_claim_g_evidence_20260704/offline_placement_deltas.csv` |
| Matched online batch latency runs | `results/method4_claim_g_evidence_20260704/online_batch_latency_runs.csv` |
| Matched online batch latency summary | `results/method4_claim_g_evidence_20260704/online_batch_latency_summary.csv` |
| Matched online batch latency deltas | `results/method4_claim_g_evidence_20260704/online_batch_latency_deltas.csv` |
| Online concurrency=8 latency runs | `results/method4_claim_g_evidence_20260704/online_concurrency8_latency_runs.csv` |
| Online concurrency=8 latency summary | `results/method4_claim_g_evidence_20260704/online_concurrency8_latency_summary.csv` |
| Online concurrency=8 latency deltas | `results/method4_claim_g_evidence_20260704/online_concurrency8_latency_deltas.csv` |
| Source manifest | `results/method4_claim_g_evidence_20260704/source_manifest.json` |
| Offline analyzer | `tools/method4_claim_g_placement_analysis.py` |
| Matched layout deploy helper | `tools/method4_claim_g_deploy_matched_layout.py` |
| Batch latency analyzer | `tools/method4_claim_g_batch_latency.py` |
| Online latency analyzer | `tools/method4_claim_g_online_latency.py` |

## Offline Placement Matrix

The offline matrix reconstructs Method4 compact MultiEP route plans from
`qdrant_controller_idea_full_20260521` and treats each query-shard dynamic EF as
the shard cost:

```text
cost(query, shard) = hnsw_ef_by_shard[shard]
tail_load_proxy(query) = max_worker(sum(cost(query, shard_on_worker)))
```

Setup:

| Item | Value |
|---|---|
| Dataset | `/home/taig/dry/faiss/datasets/glove-200-angular.hdf5` |
| Routing source collection | `qdrant_controller_idea_full_20260521` |
| Logical shards | 46 |
| Eval queries | 3000 |
| Upper sample denominator / seed | 32 / 100 |
| Upper routing | `upper_k=160`, `upper_search_ef=160` |
| Dynamic EF | `base_ef=80`, `factor=8` |
| Placements | `round_robin`, `size_balanced`, `method4_aware` |
| Worker counts | 2, 3, 4 |

### Offline Results

| Workers | Placement | Avg max worker EF/query | P95 max worker EF/query | Worker load CV | P95 max worker shards/query | Co-routed pair cross-worker |
|---:|---|---:|---:|---:|---:|---:|
| 2 | round_robin | 1941.7 | 2448.0 | 15.42% | 19 | 51.09% |
| 2 | size_balanced | 1873.3 | 2360.0 | 3.34% | 19 | 51.39% |
| 2 | method4_aware | 1816.4 | 2296.0 | 0.25% | 19 | 51.45% |
| 3 | round_robin | 1429.6 | 1808.0 | 7.16% | 14 | 68.00% |
| 3 | size_balanced | 1504.3 | 1896.0 | 7.74% | 14 | 68.06% |
| 3 | method4_aware | 1380.0 | 1720.0 | 1.79% | 13 | 68.57% |
| 4 | round_robin | 1259.6 | 1600.0 | 20.27% | 11 | 76.63% |
| 4 | size_balanced | 1212.2 | 1520.0 | 5.68% | 11 | 76.90% |
| 4 | method4_aware | 1140.5 | 1472.0 | 2.34% | 10 | 77.07% |

Method4-aware relative to round-robin:

| Workers | Avg max worker EF improvement | P95 max worker EF improvement | Worker load CV improvement | P95 max worker shards improvement |
|---:|---:|---:|---:|---:|
| 2 | 6.45% | 6.21% | 98.39% | 0.00% |
| 3 | 3.47% | 4.87% | 75.03% | 7.14% |
| 4 | 9.46% | 8.00% | 88.46% | 9.09% |

Interpretation:

- The 9-point offline matrix supports the hot-worker part of Claim G.
- Method4-aware is the best placement on `avg_query_max_peer_load`,
  `p95_query_max_peer_load`, and worker-load CV for all worker counts.
- The 3-worker sanity point exactly reproduces the previous 2026-06-01
  placement simulation P95 values: round-robin 1808.0 and method4-aware 1720.0.
- Size-balanced placement is not sufficient: it balances storage copy count well,
  but for 3 workers it is worse than round-robin on P95 worker-load proxy.

## Matched Online Batch-Size Matrix

To isolate physical layout online, two additional collections were deployed by
recovering the full logical `point_to_shards` membership from the round-robin
source collection and changing only physical placement:

| Placement | Collection | Logical membership |
|---|---|---|
| round_robin | `qdrant_controller_idea_full_20260521` | source |
| size_balanced | `qdrant_controller_idea_matched_sizebalanced_20260704` | recovered from source |
| method4_aware | `qdrant_controller_idea_matched_method4map_20260704` | recovered from source |

Deployment artifacts:

| Placement | Deploy dir |
|---|---|
| size_balanced | `results/method4_claim_g_matched_layout_deploy_20260704/20260704_171202` |
| method4_aware | `results/method4_claim_g_matched_layout_deploy_20260704/20260704_170511` |

The batch-size matrix follows the planning-document shape:

```text
3 placements * batch_size {100,200} * 3 repeats = 18 online runs
```

Shared online settings:

| Item | Value |
|---|---|
| Routing source | `qdrant_controller_idea_full_20260521` |
| Eval queries | 3000 measured + 100 warmup per cell |
| Upper routing | `upper_k=160`, `upper_search_ef=160` |
| Dynamic EF | `base_ef=80`, `factor=8` |
| Execution | compact MultiEP batch endpoint |
| Latency metric | client-observed batch endpoint wall time per batch |

### Matched Batch Results

| Batch size | Placement | Recall@10 | QPS | P50 batch latency | P95 batch latency | P99 batch latency |
|---:|---|---:|---:|---:|---:|---:|
| 100 | round_robin | 0.953467 | 421.08 | 236.29 ms | 251.83 ms | 255.62 ms |
| 100 | size_balanced | 0.953467 | 415.72 | 240.69 ms | 252.73 ms | 258.46 ms |
| 100 | method4_aware | 0.953033 | 429.73 | 231.75 ms | 250.74 ms | 254.18 ms |
| 200 | round_robin | 0.953467 | 469.48 | 426.72 ms | 440.09 ms | 442.29 ms |
| 200 | size_balanced | 0.953467 | 467.83 | 425.49 ms | 443.71 ms | 446.20 ms |
| 200 | method4_aware | 0.953033 | 475.27 | 420.65 ms | 435.86 ms | 437.96 ms |

Method4-aware relative to round-robin:

| Batch size | Recall delta | QPS delta | P95 latency delta | P99 latency delta |
|---:|---:|---:|---:|---:|
| 100 | -0.000433 | +2.05% | -0.43% | -0.56% |
| 200 | -0.000433 | +1.23% | -0.96% | -0.98% |

Size-balanced relative to round-robin:

| Batch size | Recall delta | QPS delta | P95 latency delta | P99 latency delta |
|---:|---:|---:|---:|---:|
| 100 | +0.000000 | -1.27% | +0.36% | +1.11% |
| 200 | +0.000000 | -0.35% | +0.82% | +0.88% |

Interpretation:

- This is the cleanest online Claim G evidence: all three placements use the same
  logical routing membership and same query plans; only physical layout changes.
- Method4-aware preserves recall within the `<=0.003` same-recall window, improves
  QPS for both batch sizes, and slightly improves P95/P99 batch latency.
- Size-balanced does not explain the result: despite balancing storage copies, it
  is slightly worse than round-robin on QPS and P95/P99 batch latency.

## Online Concurrency=8 Supplement

The online supplement compares the deployed round-robin collection against the
deployed Method4-aware placement collection:

| Placement | Collection |
|---|---|
| round_robin | `qdrant_controller_idea_full_20260521` |
| method4_aware | `qdrant_controller_idea_method4map_full_20260601` |

Method:

- Build upper labels once from the same HDF5 queries and global upper sample.
- Recover each collection's own logical routing membership before building route
  plans. This is required because the two deployed collections are not
  byte-identical in logical point-to-shard membership.
- Execute one preplanned compact MultiEP query plan per request.
- Use `source_id_dedup_block_size = train_size + 1`, matching the main harness.
- Measure client-observed per-query latency with 8 concurrent client workers.
- Run 1000 measured queries plus 50 warmup queries, 3 repeats per placement.

### Online Concurrency=8 Results

| Placement | Recall@10 mean | QPS mean | P50 latency | P95 latency | P99 latency | Avg visited shards | Avg EF-sum/query |
|---|---:|---:|---:|---:|---:|---:|---:|
| round_robin | 0.9575 | 331.72 | 24.52 ms | 34.19 ms | 38.70 ms | 22.642 | 3224.832 |
| method4_aware | 0.9587 | 339.36 | 24.20 ms | 32.39 ms | 35.64 ms | 23.228 | 3278.544 |

Method4-aware relative to round-robin:

| Metric | Delta |
|---|---:|
| Recall@10 | +0.0012 |
| QPS | +2.30% |
| Mean latency | -2.28% |
| P50 latency | -1.29% |
| P95 latency | -5.27% |
| P99 latency | -7.91% |
| Max latency | -10.21% |
| Avg visited shards | +2.59% |
| Avg EF-sum/query | +1.67% |

Interpretation:

- Under concurrent single-query load, Method4-aware placement improves both QPS
  and tail latency despite visiting slightly more shards and using slightly more
  EF-sum/query in the actual deployed collection.
- This supports the online tail-latency part of Claim G for the concurrency=8
  client-observed latency setting.
- This supplement is not the primary physical-only A/B because the historical
  deployed Method4-aware collection is not a byte-identical logical clone of the
  round-robin source. The matched batch-size matrix above is the primary online
  evidence.

## Boundary Checks

Sequential low-pressure diagnostic:

```text
results/method4_claim_g_online_latency_20260704/analysis_20260704_165135
```

In that run, Method4-aware did not improve latency. This is consistent with the
claim mechanism: physical layout helps when worker contention and hot-worker
queueing matter; at sequential low pressure, placement has little room to help
and the Method4-aware deployed collection visits slightly more shards.

Invalid diagnostic:

```text
results/method4_claim_g_online_latency_20260704/analysis_20260704_164441
```

That run is not evidence. It searched the Method4-aware collection using
round-robin collection routing membership, causing recall collapse. The fixed
online script now recovers each collection's own routing membership.

## Supported Statement

Recommended statement:

```text
Method4-aware physical layout reduces hot-worker load proxies in routing-trace
simulation: across 2, 3, and 4 workers, it lowers P95 max worker EF/query by
4.9-8.0% and reduces worker-load CV by 75-98% versus round-robin. In a matched
3-worker online batch-size matrix where only physical placement changes,
Method4-aware preserves Recall@10, improves QPS by 1.2-2.1%, and reduces
P95/P99 batch latency by 0.4-1.0% versus round-robin, while size-balanced does
not improve QPS or tail latency.
```

Avoid claiming:

```text
Method4-aware layout gives large online latency reductions in every workload.
```

The matched batch-size matrix shows a consistent but modest online P95/P99
improvement. The larger effect is in the offline hot-worker/tail-load proxy and
in the concurrency=8 single-query supplement.
