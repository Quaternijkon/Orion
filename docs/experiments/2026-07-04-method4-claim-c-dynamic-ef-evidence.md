# 2026-07-04 Method4 Claim C Dynamic EF Evidence

## Claim

Claim C:

```text
Dynamic EF allocates lower search budget to shards that are more likely to
contain useful results, and is therefore more efficient than fixed EF.
```

This evidence package has three parts:

1. A same-recall performance ablation comparing Orion Dynamic EF with Orion
   fixed EF.
2. An offline routed-entry-point relevance analysis checking whether higher
   `routed_ep_count` predicts shards containing ground-truth top10 copies.
3. A client-observed batch-latency supplement for the selected Orion fixed-vs-
   Dynamic EF pair.

## Evidence Package

| Artifact | Path |
|---|---|
| Performance summary CSV | `results/method4_claim_c_evidence_20260704/dynamic_vs_fixed_performance_summary.csv` |
| Dynamic-vs-fixed deltas CSV | `results/method4_claim_c_evidence_20260704/dynamic_vs_fixed_deltas.csv` |
| Routed EP relevance summary CSV | `results/method4_claim_c_evidence_20260704/routed_ep_relevance_summary.csv` |
| Batch latency matrix CSV | `results/method4_claim_c_evidence_20260704/claim_c_dynamic_vs_fixed_latency_matrix.csv` |
| Batch latency deltas CSV | `results/method4_claim_c_evidence_20260704/claim_c_dynamic_vs_fixed_latency_deltas.csv` |
| Source manifest | `results/method4_claim_c_evidence_20260704/source_manifest.json` |
| Relevance analysis raw output | `results/method4_claim_c_dynamic_ef_relevance_20260704/analysis_20260704_141217` |
| Latency rerun raw output | `results/method4_claim_c_dynamic_vs_fixed_latency_20260704/analysis_20260704_235925` |
| Relevance analysis script | `tools/method4_claim_c_relevance_analysis.py` |
| Latency summary script | `tools/method4_claim_c_latency_summary.py` |

## Setup

The performance experiments reuse the 2026-06-25 Claim C Orion runs:

- Dataset: `/home/taig/dry/faiss/datasets/glove-200-angular.hdf5`
- Collection: `bench095_rr_orion_s31`
- Routing mode: `faithful_original_rest`
- Routed execution mode: `compact_multi_ep`
- Routed planning mode: `materialized`
- Eval queries: 3000
- Stability repeats: 3

The fixed EF control keeps the same Orion routing and entry-point semantics as
Dynamic EF, but sets `factor=0` so every visited lower shard receives the same
EF. Dynamic EF uses:

```text
EF(shard) = base_ef + factor * routed_ep_count(shard)
```

The implementation computes per-shard EF from routed EP counts in
`tools/qdrant_two_level_routing_experiment.py`.

## Performance Evidence

The strongest isolated comparison fixes `upper_k=120`, so fixed EF and Dynamic
EF visit effectively the same number of lower shards. The difference is how
lower HNSW EF is assigned across those shards.

| Comparison | Method | Params | Recall@10 mean | QPS mean | Visited shards | Avg EF/shard | EF-sum/query |
|---|---|---|---:|---:|---:|---:|---:|
| fixed upper width ~0.95 | Fixed EF | `upper_k=120; fixed_ef=240` | 0.949333 | 374.356 | 19.304 | 240.000 | 4632.9 |
| fixed upper width ~0.95 | Dynamic EF | `upper_k=120; base_ef=80; factor=10` | 0.948500 | 476.191 | 19.329 | 147.722 | 2855.4 |

Dynamic EF relative to fixed EF:

| Comparison | Recall delta | QPS delta | QPS delta % | Visited shard delta % | EF-sum delta % |
|---|---:|---:|---:|---:|---:|
| fixed upper width ~0.95 | -0.000833 | +101.834 | +27.20% | +0.13% | -38.37% |

The frontier/robust-confirm comparisons let both fixed EF and Dynamic EF tune
the upper routing width:

| Target | Method | Params | Recall@10 mean | QPS mean | Visited shards | Avg EF/shard | EF-sum/query |
|---|---|---|---:|---:|---:|---:|---:|
| ~0.95 | Fixed EF | `upper_k=80; fixed_ef=280` | 0.952000 | 376.319 | 17.095 | 280.000 | 4786.5 |
| ~0.95 | Dynamic EF | `upper_k=80; base_ef=80; factor=20` | 0.951533 | 485.081 | 17.096 | 182.213 | 3115.1 |
| ~0.97 | Fixed EF | `upper_k=120; fixed_ef=480` | 0.971767 | 235.828 | 21.094 | 480.000 | 10125.1 |
| ~0.97 | Dynamic EF | `upper_k=200; base_ef=60; factor=20` | 0.973467 | 308.749 | 26.607 | 225.294 | 5994.5 |

Dynamic EF relative to fixed EF:

| Target | Recall delta | QPS delta % | Visited shard delta % | EF-sum delta % |
|---|---:|---:|---:|---:|
| ~0.95 | -0.000467 | +28.90% | +0.01% | -34.92% |
| ~0.97 | +0.001700 | +30.92% | +26.14% | -40.80% |

Interpretation:

- At the same recall neighborhood, Dynamic EF improves QPS by 27-31%.
- It reduces estimated lower EF-sum/query by 35-41%.
- It does not always visit fewer shards. At ~0.97 it visits more shards, but
  still uses lower total EF-sum and has higher QPS.

## Batch Latency Supplement

A new 2026-07-04 latency-producing rerun uses the same live Orion collection
and compares fixed EF against Dynamic EF at the selected `upper_k=80` operating
point. This run uses the Claim D batch-latency runner with `--skip-naive`, 3000
eval queries, 3 repeats, and `batch_size=200`.

This supplement is not bit-for-bit the same measurement path as the 2026-06-25
Claim C performance grid, so use it for the P95/P99 latency clause and keep the
older grid for the broader same-recall QPS frontier. It measures client-
observed batch endpoint wall time, not server-internal lower-search trace time.

| Method | Params | Recall@10 mean | QPS mean | Visited shards | EF-sum/query | P95 batch ms | P99 batch ms |
|---|---|---:|---:|---:|---:|---:|---:|
| Fixed EF | `upper_k=80; fixed_ef=280` | 0.949300 | 408.677 | 15.990 | 4477.1 | 514.4 | 519.5 |
| Dynamic EF | `upper_k=80; base_ef=80; factor=20` | 0.949467 | 517.988 | 15.990 | 3021.0 | 406.3 | 410.5 |

Dynamic EF relative to fixed EF in this latency rerun:

| Recall delta | QPS delta % | Visited shard delta % | EF-sum delta % | P95 delta % | P99 delta % |
|---:|---:|---:|---:|---:|---:|
| +0.000167 | +26.75% | +0.00% | -32.52% | -21.00% | -20.97% |

Source:
`results/method4_claim_c_evidence_20260704/claim_c_dynamic_vs_fixed_latency_matrix.csv`;
delta table:
`results/method4_claim_c_evidence_20260704/claim_c_dynamic_vs_fixed_latency_deltas.csv`.

## Routed EP Relevance Evidence

The new 2026-07-04 offline analysis tests the first half of Claim C: whether
`routed_ep_count` is a useful proxy for shard relevance.

Method:

1. Rebuild the upper HNSW index from the same global upper sample.
2. Recover full `point_to_shards` membership from `bench095_rr_orion_s31` by
   scrolling all custom shard keys.
3. For each of 3000 eval queries, route upper labels to lower shards.
4. Mark a visited shard as GT-hit if it contains at least one ground-truth top10
   source-id copy for that query.
5. Compare routed EP counts and budget share for GT-hit vs non-hit shards.

Recovery sanity:

| Metric | Value |
|---|---:|
| Logical points | 1,183,514 |
| Recovered point copies | 1,390,579 |
| Missing logical points | 0 |
| Effective logical shards | 46 |

Summary:

| Config | upper_k | Dynamic EF formula | Avg EP count on GT-hit shards | Avg EP count on non-hit shards | Dynamic budget share on GT-hit shards | Fixed budget share on GT-hit shards |
|---|---:|---|---:|---:|---:|---:|
| fixed-width Claim C ~0.95 | 120 | `80 + 10 * routed_ep_count` | 20.34 | 3.22 | 34.93% | 17.53% |
| frontier ~0.95 | 80 | `80 + 20 * routed_ep_count` | 14.72 | 2.55 | 43.24% | 21.05% |
| frontier ~0.97 | 200 | `60 + 20 * routed_ep_count` | 30.78 | 4.54 | 42.64% | 14.22% |

Bucketed example for the `frontier ~0.95` config:

| Routed EP count bucket | GT-hit rate |
|---|---:|
| 1 | 4.52% |
| 2 | 9.95% |
| 3 | 15.32% |
| 4 | 21.65% |
| 5 | 27.01% |
| 6 | 32.71% |
| 7 | 41.04% |
| 8 | 48.55% |
| 9+ | 77.84% |

This directly supports the relevance-proxy part of Claim C: shards with more
routed upper entry points are much more likely to contain ground-truth top10
copies. Since Dynamic EF grows with routed EP count, it assigns a larger share
of lower search budget to those more promising shards than uniform fixed EF.

## Reproduce

The relevance analysis can be rerun with:

```bash
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
python3 tools/method4_claim_c_relevance_analysis.py
```

The latency supplement can be rerun with:

```bash
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
python3 tools/method4_claim_d_high_recall_latency.py \
  --base-url http://localhost:6833 \
  --method4-collection bench095_rr_orion_s31 \
  --method4-routing-source-collection bench095_rr_orion_s31 \
  --method4-config fixed_u80_ef280=80,280,0 \
  --method4-config dynamic_u80_b80_f20=80,80,20 \
  --eval-query-count 3000 \
  --warmup-query-count 100 \
  --batch-size 200 \
  --repeats 3 \
  --upper-search-ef 160 \
  --skip-naive \
  --output-root results/method4_claim_c_dynamic_vs_fixed_latency_20260704
```

The existing performance runs can be audited from these source summaries:

```text
results/method4_claim_c_orion_fixed_ef_extended_20260625/20260625_122713/summary.json
results/method4_claim_c_orion_dynamic_ef_20260625/20260625_122853/summary.json
results/method4_claim_c_orion_frontier_fixed_095_robust_confirm_20260625/20260625_132656/summary.json
results/method4_claim_c_orion_frontier_dynamic_095_robust_confirm_20260625/20260625_132811/summary.json
results/method4_claim_c_orion_frontier_fixed_097_robust_confirm_20260625/20260625_132210/summary.json
results/method4_claim_c_orion_frontier_dynamic_097_more_robust_confirm_20260625/20260625_132923/summary.json
```

## Supported Statement

Recommended statement:

```text
In Orion Claim C ablations, Dynamic EF uses routed upper-entry-point count as a
shard relevance proxy. Offline analysis confirms that GT-hit shards have far
more routed entry points than non-hit shards, and therefore receive a larger
share of Dynamic EF budget than they would under uniform fixed EF. In same-
recall online runs, Dynamic EF improves QPS by 27-31% and reduces estimated
lower EF-sum/query by 35-41% compared with fixed EF. A client-observed batch-
latency rerun at the selected 0.95 operating point also shows Dynamic EF
reducing P95/P99 batch latency by about 21% at the same visited-shard count.
```

Avoid claiming:

```text
Dynamic EF always visits fewer shards.
```

The evidence supports higher same-recall QPS and lower lower-search budget, not
consistent visited-shard reduction.

Also avoid claiming:

```text
Dynamic EF's server-internal lower-search P99 has been directly traced.
```

The latency supplement measures client-observed batch endpoint wall time. It is
positive P95/P99 evidence for the same request path, but it is not an internal
server-side lower-search latency trace.
