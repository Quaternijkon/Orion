# 2026-06-25 Method4 Claim C Orion Dynamic EF Experiment

## Goal

Validate Claim C from
`docs/experiments/2026-06-16-method4-distributed-submission-experiment-plan.md`:
Dynamic EF allocates lower-tier HNSW budget to shards with more routed entry
points, and is therefore more efficient than fixed EF.

This run only evaluates the Orion / `faithful_original_rest` path. It does not
compare against naive or KMeans partition schemes.

## Setup

| Item | Value |
|---|---|
| Date | 2026-06-25 |
| Commit | `02755de1e` |
| Working tree | Dirty before this experiment |
| Controller | `http://localhost:6833` |
| Topology | controller + 3 shard workers |
| Dataset | `/home/taig/dry/faiss/datasets/glove-200-angular.hdf5` |
| Collection | `bench095_rr_orion_s31` |
| Initial shards | 31 |
| Effective logical shards | 46 |
| Routing mode | `faithful_original_rest` |
| Execution mode | `compact_multi_ep` |
| Planning mode | `materialized` |
| Result limit | `top_k` |
| Search dispatch | `coordinator` |
| Batch size | 200 |
| Tuning queries | 500 |
| Eval queries | 3000 |
| Stability repeats | 3 |
| Target recall | `Recall@10 ~= 0.95` |

The fixed EF control uses the same Orion routing and entry-point semantics as
Dynamic EF. It is expressed by setting `factor=0`, so every visited shard gets
the same `base_ef`.

## Parameter Grids

Fixed EF, plan grid:

```text
upper_k in {120, 160, 200}
fixed_ef in {60, 80, 100, 120, 140}
factor = 0
```

The planned fixed grid did not reach the target recall. Its best tuning point
was:

```text
upper_k=200, fixed_ef=140
Recall@10=0.9352
QPS=393.372
avg_visited_shards=25.608
```

To obtain a same-recall comparison point, the fixed EF grid was extended:

```text
upper_k in {120, 160, 200}
fixed_ef in {160, 200, 240, 280, 320}
factor = 0
```

Dynamic EF grid:

```text
upper_k in {120, 160, 200}
base_ef in {40, 60, 80}
factor in {4, 6, 8, 10, 12}
```

## Result Directories

| Method | Result directory |
|---|---|
| Fixed EF extended | `results/method4_claim_c_orion_fixed_ef_extended_20260625/20260625_122713` |
| Dynamic EF | `results/method4_claim_c_orion_dynamic_ef_20260625/20260625_122853` |

## Selected Points

Selection used the harness rule: among tuning points meeting target recall,
choose the point with highest tuning QPS. The 3000-query eval and 3 stability
repeats then used that selected point.

| Method | Selected parameters | Tuning Recall@10 | Tuning QPS | Tuning avg visited shards | Tuning avg EF/shard |
|---|---:|---:|---:|---:|---:|
| Fixed EF | `upper_k=120, fixed_ef=240` | 0.9516 | 395.337 | 19.436 | 240.000 |
| Dynamic EF | `upper_k=120, base_ef=80, factor=10` | 0.9506 | 490.268 | 19.508 | 147.287 |

Both selected points use `upper_k=120`, so they route to effectively the same
number of lower logical shards. This isolates EF allocation rather than shard
selection.

## Stability Results

| Method | Params | Recall@10 mean | QPS mean +/- stdev | Avg visited shards | Avg EF/visited shard | Est. EF-sum/query |
|---|---|---:|---:|---:|---:|---:|
| Fixed EF | `upper_k=120, fixed_ef=240` | 0.949333 | 374.356 +/- 2.016 | 19.304 | 240.000 | 4632.9 |
| Dynamic EF | `upper_k=120, base_ef=80, factor=10` | 0.948500 | 476.191 +/- 3.785 | 19.329 | 147.722 | 2855.4 |

Computed deltas, Dynamic EF relative to Fixed EF:

| Metric | Delta |
|---|---:|
| Recall@10 | -0.000833 |
| QPS | +101.834 |
| QPS % | +27.20% |
| Avg visited shards | +0.026 |
| Avg visited shards % | +0.13% |
| Est. EF-sum/query % | -38.37% |

## Interpretation

This supports the efficiency part of Claim C for Orion:

- At essentially the same recall level, Dynamic EF is 27.20% faster than Fixed
  EF.
- Dynamic EF reaches the same recall neighborhood with nearly identical visited
  shard count, so the gain is not from visiting fewer shards in the selected
  comparison.
- The estimated EF-sum/query drops by 38.37%, from 4632.9 to 2855.4, while QPS
  rises from 374.356 to 476.191.
- The fixed EF range from the original plan (`60..140`) could not reach
  `Recall@10 ~= 0.95`; fixed EF needed `ef=240` at `upper_k=120` for the same
  recall neighborhood.

Allowed paper statement:

```text
On the Orion routed path, Dynamic EF improves lower-search budget efficiency:
at the same Recall@10 neighborhood (~0.949), it keeps the visited-shard count
effectively unchanged but reduces estimated EF-sum/query by 38.4% and improves
QPS by 27.2% over a fixed-EF Orion control.
```

Do not claim from this run alone that Dynamic EF reduces visited shard count.
The selected same-recall comparison used the same `upper_k` and therefore had
almost identical visited shard counts. The supported claim is that Dynamic EF
uses lower per-shard search budget more effectively for the same routed shards.

## Follow-up: Upper-Routing-Width Frontier

The first run above isolated EF allocation at a fixed `upper_k`. To test the
broader operating strategy, this follow-up lets Fixed EF widen upper routing
and therefore visit more lower shards. Dynamic EF can tune both routing width
and the linear formula:

```text
ef = b + a * routed_ep_count
```

Harness note: before this follow-up, the faithful Orion path still hard-coded
upper HNSW search EF to `100`. The harness was patched so effective upper
search EF is:

```text
max(--upper-search-ef, max(--upper-k-candidates, --nprobe-candidates))
```

The relevant test is:

```bash
pytest -q -o addopts='' tests/tools/test_qdrant_two_level_routing_experiment.py
```

which passed with:

```text
54 passed
```

All follow-up runs used:

```text
--upper-search-ef 400
```

### Frontier Tuning Grids

Fixed EF frontier:

```text
upper_k in {80, 120, 160, 200, 240, 280, 320}
fixed_ef in {80, 120, 160, 200, 240, 280, 320}
factor = 0
```

Fixed EF high-recall extension:

```text
upper_k in {80, 120, 160, 200, 240, 280, 320}
fixed_ef in {360, 400, 480}
factor = 0
```

Dynamic EF frontier:

```text
upper_k in {60, 80, 100, 120, 160, 200, 240}
base_ef in {40, 60, 80}
factor in {6, 8, 10, 12, 16, 20}
```

### Frontier Result Directories

| Run | Result directory |
|---|---|
| Fixed EF frontier | `results/method4_claim_c_orion_frontier_fixed_ef_20260625/20260625_130801` |
| Fixed EF high extension | `results/method4_claim_c_orion_frontier_fixed_ef_high_20260625/20260625_131518` |
| Dynamic EF frontier | `results/method4_claim_c_orion_frontier_dynamic_ef_20260625/20260625_131020` |

### Frontier Tuning Selection

For `Recall@10 >= 0.95`, the highest-QPS tuning points were:

| Method | Params | Recall@10 | QPS | Avg visited shards | Avg EF/shard | Est. EF-sum/query |
|---|---|---:|---:|---:|---:|---:|
| Fixed EF | `upper_k=80, fixed_ef=240` | 0.9518 | 396.102 | 17.278 | 240.000 | 4146.7 |
| Dynamic EF | `upper_k=80, base_ef=80, factor=16` | 0.9520 | 517.599 | 17.262 | 161.259 | 2783.6 |

For `Recall@10 >= 0.97`, the initial highest-QPS fixed tuning point was
`upper_k=120, fixed_ef=400`, but it was close to the target and dipped to
`0.9698` on an independent confirm tuning sample. The official comparison
therefore uses a more robust fixed point, `upper_k=120, fixed_ef=480`.
The dynamic official point also uses a robust high-recall point rather than the
most marginal frontier point.

### Official Robust Stability Results

Each official point used 3000 eval queries and 3 stability repeats.

| Target | Method | Params | Tuning Recall@10 | Recall@10 mean | QPS mean +/- stdev | Avg visited shards | Avg EF/shard | Est. EF-sum/query |
|---|---|---|---:|---:|---:|---:|---:|---:|
| ~0.95 | Fixed EF | `upper_k=80, fixed_ef=280` | 0.9570 | 0.952000 | 376.319 +/- 4.484 | 17.095 | 280.000 | 4786.5 |
| ~0.95 | Dynamic EF | `upper_k=80, base_ef=80, factor=20` | 0.9572 | 0.951533 | 485.081 +/- 1.453 | 17.096 | 182.213 | 3115.1 |
| ~0.97 | Fixed EF | `upper_k=120, fixed_ef=480` | 0.9752 | 0.971767 | 235.828 +/- 3.451 | 21.094 | 480.000 | 10125.1 |
| ~0.97 | Dynamic EF | `upper_k=200, base_ef=60, factor=20` | 0.9758 | 0.973467 | 308.749 +/- 4.145 | 26.607 | 225.294 | 5994.5 |

Dynamic EF relative to Fixed EF:

| Target | Recall delta | QPS delta | QPS delta % | Visited shard delta | Visited shard delta % | EF-sum delta % |
|---|---:|---:|---:|---:|---:|---:|
| ~0.95 | -0.000467 | +108.762 | +28.90% | +0.001 | +0.01% | -34.92% |
| ~0.97 | +0.001700 | +72.921 | +30.92% | +5.513 | +26.14% | -40.80% |

### Follow-up Interpretation

This follow-up supports the same-recall QPS form of Claim C:

- At the `~0.95` recall level, Dynamic EF has essentially identical visited
  shard count, but improves QPS by 28.90% and reduces estimated EF-sum/query by
  34.92%.
- At the `~0.97` recall level, Dynamic EF is still 30.92% faster and reduces
  estimated EF-sum/query by 40.80%, even though it visits 26.14% more shards.
- The `~0.97` result should therefore be written as "higher QPS / lower lower
  search budget at same recall", not as "fewer shards".

Allowed paper statement from the follow-up:

```text
When both fixed EF and Dynamic EF are allowed to tune upper routing width, the
Orion Dynamic EF frontier still dominates in same-recall throughput. At
Recall@10 ~= 0.952, Dynamic EF keeps the visited-shard count effectively
unchanged while improving QPS by 28.9% and reducing estimated EF-sum/query by
34.9%. At Recall@10 ~= 0.973, Dynamic EF improves QPS by 30.9% and reduces
estimated EF-sum/query by 40.8%, although it visits more routed shards.
```

Do not claim that Dynamic EF consistently visits fewer shards from these
official runs. The supported statement is stronger on QPS and estimated lower
search budget than on shard count.

## Commands

Fixed EF extended:

```bash
env PYTEST_ADDOPTS= LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
python3 tools/qdrant_two_level_routing_experiment.py \
  --base-url http://localhost:6833 \
  --collection bench095_rr_orion_s31 \
  --hdf5-path /home/taig/dry/faiss/datasets/glove-200-angular.hdf5 \
  --routing-mode faithful_original_rest \
  --num-shards 31 \
  --upper-k-candidates 120 160 200 \
  --base-ef-candidates 160 200 240 280 320 \
  --factor-candidates 0 \
  --target-recall 0.95 \
  --eval-query-count 3000 \
  --tuning-query-count 500 \
  --stability-repeats 3 \
  --batch-size 200 \
  --reuse-existing \
  --recover-routing-from-collection \
  --routed-execution-mode compact_multi_ep \
  --routed-planning-mode materialized \
  --routed-result-limit-mode top_k \
  --search-dispatch-mode coordinator \
  --output-dir results/method4_claim_c_orion_fixed_ef_extended_20260625
```

Dynamic EF:

```bash
env PYTEST_ADDOPTS= LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
python3 tools/qdrant_two_level_routing_experiment.py \
  --base-url http://localhost:6833 \
  --collection bench095_rr_orion_s31 \
  --hdf5-path /home/taig/dry/faiss/datasets/glove-200-angular.hdf5 \
  --routing-mode faithful_original_rest \
  --num-shards 31 \
  --upper-k-candidates 120 160 200 \
  --base-ef-candidates 40 60 80 \
  --factor-candidates 4 6 8 10 12 \
  --target-recall 0.95 \
  --eval-query-count 3000 \
  --tuning-query-count 500 \
  --stability-repeats 3 \
  --batch-size 200 \
  --reuse-existing \
  --recover-routing-from-collection \
  --routed-execution-mode compact_multi_ep \
  --routed-planning-mode materialized \
  --routed-result-limit-mode top_k \
  --search-dispatch-mode coordinator \
  --output-dir results/method4_claim_c_orion_dynamic_ef_20260625
```
