# C6 Experimental Protocol: Adaptive Routing and Per-Shard Search Allocation

## 1. Objective

Establish the following claim:

> **C6. Orion improves the recall–cost trade-off by adapting both shard selection and per-shard graph-search effort to each query. Compared with static routing and uniform local search budgets, Orion reaches Recall@10 >= 0.90 with fewer searched shards, lower aggregate graph-search work, and lower tail latency, while approaching an oracle allocation policy.**

C6 contains four subclaims:

### C6-a — Query difficulty is heterogeneous

The amount of distributed work required to reach Recall@10 >= 0.90 varies substantially across queries.

Required evidence:

- the minimum number of shards required per query has a broad distribution;
- the local `efSearch` required on selected shards also varies across queries and shards;
- a single fixed routing/search configuration therefore over-searches easy queries or under-searches hard queries.

### C6-b — Adaptive shard selection outperforms fixed fan-out

At the same Recall@10 >= 0.90, Orion's query-dependent shard selection requires fewer searched shards and less aggregate graph-search work than static fixed-\(P\) routing.

### C6-c — Adaptive per-shard search intensity outperforms uniform `efSearch`

For the same selected shard set, allocating local search effort according to per-shard routing evidence reduces total graph-search work and/or tail latency relative to applying one uniform `efSearch` to every selected shard.

### C6-d — The full adaptive policy approaches the oracle frontier

The combination of adaptive shard selection and adaptive per-shard search intensity should approach the best achievable recall–cost trade-off under the same routing candidates, without per-query access to ground truth.

The required causal chain is:

\[
\text{Query difficulty varies}
\rightarrow
\text{static policies misallocate work}
\rightarrow
\text{adaptive shard selection reduces unnecessary fan-out}
\rightarrow
\text{adaptive local budgets reduce unnecessary local traversal}
\rightarrow
\text{lower total work at fixed recall}.
\]

---

## 2. Scope

This experiment suite evaluates **online adaptivity only**.

Hold the following components fixed across all primary C6 comparisons:

- Orion topology-aware partition layout;
- Orion navigation graph;
- navigation-graph query search implementation;
- routing-candidate generation;
- routing-candidate-to-shard mapping;
- local HNSW implementation;
- entry-point injection policy, if already enabled in the Orion serving path;
- HNSW construction parameters;
- dataset and query set.

Do not compare Random or K-Means partitioning in the primary C6 experiment matrix.

Partition-quality conclusions belong to C2/C3.

Do not vary entry-point injection in the primary C6 matrix.

Entry-point effectiveness belongs to C5.

---

## 3. Hardware and Logical-Shard Model

Physical cluster:

```text
4 machines
```

Use logical shard counts:

\[
M \in \{4,8,16,32\}.
\]

For \(M=4\):

- use one shard per physical machine;
- permit end-to-end throughput and latency conclusions.

For \(M>4\):

- distribute logical shards evenly across the four machines;
- retain independent local HNSW instances;
- use these runs for routing, fan-out, graph-work, oracle, and per-query allocation analysis;
- do not report colocated wall-clock latency as \(M\)-machine scale-out latency;
- label all \(M>4\) results as **logical-shard simulation**.

Primary C6 conclusions must be based on algorithmic work metrics and may use all logical shard counts.

Physical end-to-end latency/throughput validation must use \(M=4\).

---

## 4. Datasets and Accuracy Target

Use exactly:

| Dataset | Metric | Primary Target |
|---|---|---|
| SIFT1M | L2 | Recall@10 >= 0.90 |
| glove-200-angular | Angular/Cosine | Recall@10 >= 0.90 |

Use official base vectors, query vectors, and ground truth where available.

Use identical query sets across all C6 configurations.

Partition queries into:

```text
tuning set: 1,000 queries
measurement set: all remaining official queries, with >= 10,000 queries when available
```

Do not include tuning queries in reported results.

---

## 5. Fixed Orion Serving Substrate

Before running C6, construct one frozen Orion configuration for every:

```text
dataset x logical_shard_count
```

Persist:

```text
partition assignment
navigation graph
navigation-node shard labels
local HNSW indexes
local entry-point metadata
```

The same frozen index artifacts must be reused by all C6 policies.

Do not rebuild the partition or local indexes per routing policy.

Record the Git commit and index artifact checksum.

---

## 6. Routing Evidence

For each query \(q\), the navigation graph returns an ordered candidate list:

\[
TopK(q)=\{u_1,u_2,\ldots,u_K\}.
\]

Each candidate \(u\) has shard label \(C(u)\).

For every shard \(i\) represented in `TopK(q)`, record:

```text
candidate_count_i = |{u in TopK(q) : C(u)=i}|
best_candidate_rank_i
minimum_candidate_distance_i
mean_candidate_distance_i
entry_point_count_i
```

Use Orion's actual routing evidence for the adaptive policy.

If the current implementation uses only `entry_point_count_i`, do not introduce a stronger learned scoring model for C6.

---

## 7. Static and Adaptive Policies

Implement the following policies over the same navigation-search output.

### P0 — Fixed-\(P\) + Uniform `efSearch`

Static baseline.

For each query:

1. rank candidate shards using the same deterministic shard-ranking rule used by Orion;
2. select exactly \(P\) shards;
3. assign the same `efSearch = E` to every selected shard.

Tune \((P,E)\) globally on the tuning set.

This is the primary static baseline.

---

### P1 — Adaptive Shards + Uniform `efSearch`

Enable Orion's adaptive shard selection.

Keep one globally fixed local `efSearch = E`.

This isolates adaptive fan-out.

---

### P2 — Fixed-\(P\) + Adaptive Per-Shard `efSearch`

Use fixed global fan-out \(P\).

Allocate local search effort using Orion's per-shard policy:

\[
efSearch_i = \alpha |EP_i(q)| + \beta.
\]

Apply implementation-defined clamping:

\[
ef_{\min} \le efSearch_i \le ef_{\max}.
\]

This isolates adaptive local search intensity.

---

### P3 — Full Orion Adaptivity

Enable both:

- adaptive shard selection;
- adaptive per-shard `efSearch`.

This is the full C6 policy.

---

### P4 — Oracle Prefix Stopping

Offline analysis only.

Use the same ranked shard order as the deployed router.

For each query, search shards incrementally and identify the minimum prefix length that reaches Recall@10 >= 0.90:

\[
P_{\mathrm{oracle-prefix}}(q).
\]

For each searched shard, use a sufficiently high fixed local search budget so shard-selection difficulty is isolated.

This oracle evaluates the best possible per-query stopping decision under Orion's existing shard ordering.

Do not implement it in the serving path.

---

### P5 — Oracle Local-Budget Allocation

Offline analysis only.

For a fixed selected shard set, search a discrete `efSearch` grid independently per shard and identify the minimum-work allocation that reaches global Recall@10 >= 0.90.

Use the grid:

```text
efSearch in {8, 16, 32, 64, 128, 256, 512}
```

Extend only if necessary to reach the target.

The optimization objective is:

\[
\min \sum_i D_i(q)
\]

subject to:

\[
Recall@10(q)\ge0.90.
\]

where \(D_i(q)\) is local distance-computation count.

If exhaustive Cartesian search is infeasible, implement a deterministic dynamic-programming or monotone-search procedure over per-shard result traces.

Do not replace the oracle with a learned predictor.

---

### P6 — Full Oracle

Offline analysis only.

Combine:

- oracle prefix stopping;
- oracle per-shard budget allocation.

Use as a practical lower bound on graph-search work under the same partition and routing candidate ordering.

---

## 8. Shard Ranking for Fixed-\(P\) Policies

Fixed-\(P\) policies require one deterministic shard order.

Use the existing Orion routing order if explicitly defined.

Otherwise define:

1. descending `candidate_count_i`;
2. tie-break by ascending `best_candidate_rank_i`;
3. tie-break by ascending shard ID.

Persist this order per query.

Use the same order for:

- P0;
- P2;
- P4;
- P6.

Do not use ground truth for deployable shard ranking.

---

## 9. Continuous Result Recording

Create:

```text
experiments/c6/
  PLAN.md
  STATUS.md
  RESULTS.md
  runs/
    manifest.jsonl
    tuning.csv
    summary.csv
    per_query/
    per_shard/
    oracle/
  figures/
  scripts/
  logs/
```

`PLAN.md` must contain the final protocol.

`STATUS.md` must contain:

```text
current stage
completed configurations
failed configurations
active anomalies
next command
```

`RESULTS.md` must be updated after every completed stage and every major configuration group.

Append records using:

```text
Timestamp:
Git commit:
Experiment:
Dataset:
Logical shards:
Policy:
Run IDs:
Recall:
Fan-out:
Aggregate distance computations:
Max per-shard distance computations:
Latency if physical:
Observed result:
C6-a status:
C6-b status:
C6-c status:
C6-d status:
Anomalies:
Required follow-up:
```

Never overwrite prior interpretations.

Append corrections with timestamps.

---

# 10. Experiment E1 — Query-Difficulty Characterization

## 10.1 Purpose

Establish C6-a.

Quantify how much work different queries intrinsically require.

---

## 10.2 Oracle Shard Difficulty

Using P4, compute for every measurement query:

\[
P_{\mathrm{oracle-prefix}}(q).
\]

Record:

```text
mean
median
p75
p90
p95
p99
CDF
```

Also normalize:

\[
F_{\mathrm{oracle}}(q)=
\frac{P_{\mathrm{oracle-prefix}}(q)}{M}.
\]

Generate results for:

\[
M\in\{4,8,16,32\}.
\]

---

## 10.3 Oracle Local-Search Difficulty

For every query and every oracle-selected shard, compute the minimum local `efSearch` needed in the P5/P6 oracle solution.

Record:

```text
minimum efSearch per query-shard
distance computations per query-shard
number of selected shards
total distance computations
maximum shard distance computations
```

Generate distributions for:

```text
per-query total work
per-query maximum shard work
per-query mean local efSearch
per-query maximum local efSearch
```

---

## 10.4 Static-Policy Waste Classification

For the globally tuned static P0 policy, classify every measurement query as:

### Under-searched

\[
Recall@10(q)<0.90.
\]

### Exactly/near sufficiently searched

\[
Recall@10(q)\ge0.90
\]

and static graph work is within 10% of the full-oracle graph work.

### Over-searched

\[
Recall@10(q)\ge0.90
\]

and:

\[
W_{\mathrm{static}}(q)
>
1.10\cdot
W_{\mathrm{oracle}}(q).
\]

Report fractions of all three classes.

---

## 10.5 E1 Outputs

Generate:

```text
c6_fig1_oracle_shard_difficulty_cdf.pdf
c6_fig2_oracle_local_budget_cdf.pdf
c6_fig3_static_waste_breakdown.pdf
```

Primary C6-a figure:

```text
x-axis: minimum shards required for 90% Recall@10
y-axis: fraction of queries
curves: M = 4, 8, 16, 32
```

---

# 11. Experiment E2 — Adaptive Shard Selection

## 11.1 Purpose

Establish C6-b.

Compare:

```text
P0 Fixed-P + Uniform efSearch
P1 Adaptive Shards + Uniform efSearch
P4 Oracle Prefix
```

Hold local search configuration fixed.

---

## 11.2 Fixed Local Search Budget

Choose one common `efSearch = E_common` per:

```text
dataset x M
```

using the tuning set.

Selection rule:

1. use a sufficiently high fixed fan-out to make routing misses negligible;
2. find the smallest `E_common` for which mean Recall@10 >= 0.90;
3. freeze `E_common` for all E2 policies.

Do not separately tune local `efSearch` for P0 and P1.

---

## 11.3 Static Fan-Out Sweep

For P0 evaluate:

\[
P\in\{1,2,4,\ldots,M\}
\]

with all valid intermediate values near the 90% recall crossing.

Record for each \(P\):

```text
mean recall
p5 per-query recall
mean shards/query
aggregate distance computations/query
max shard distance computations/query
```

Select the smallest global \(P\) that reaches mean Recall@10 >= 0.90 on the tuning set.

Freeze it for the measurement set.

---

## 11.4 Adaptive Policy

Run P1 on the same measurement queries using Orion's adaptive shard-selection rule.

Do not tune per-query thresholds on the measurement set.

Record:

```text
recall
selected shards/query
aggregate distance computations
max per-shard distance computations
```

---

## 11.5 Primary Comparisons

At Recall@10 >= 0.90 compare:

\[
\bar P_{\mathrm{adaptive}}
\quad\text{vs.}\quad
P_{\mathrm{static}}
\]

and:

\[
W_{\mathrm{adaptive}}
\quad\text{vs.}\quad
W_{\mathrm{static}}.
\]

Also compare against:

\[
P_{\mathrm{oracle-prefix}}.
\]

Compute adaptive shard-selection overhead:

\[
Gap_P =
\frac{
P_{\mathrm{adaptive}}-P_{\mathrm{oracle-prefix}}
}{
P_{\mathrm{oracle-prefix}}
}.
\]

Report mean and p95 gap.

---

## 11.6 Per-Query Paired Analysis

For every query record:

\[
\Delta P(q)
=
P_{\mathrm{static}}
-
P_{\mathrm{adaptive}}(q)
\]

and:

\[
\Delta W(q)
=
W_{\mathrm{static}}(q)
-
W_{\mathrm{adaptive}}(q).
\]

Report:

```text
fraction of queries with fewer shards
fraction with equal shards
fraction with more shards
mean work saved on easy queries
additional work spent on hard queries
```

Adaptive routing should be allowed to spend more work on hard queries when required to satisfy recall.

Do not report only average shard reduction.

---

## 11.7 E2 Outputs

Generate:

```text
c6_fig4_static_vs_adaptive_fanout.pdf
c6_fig5_adaptive_vs_oracle_fanout.pdf
c6_fig6_per_query_shard_savings_cdf.pdf
c6_fig7_recall_work_frontier_shard_adaptation.pdf
```

Primary plot:

```text
x-axis: mean shards/query
y-axis: Recall@10

curves:
  Fixed-P
  Orion adaptive
  Oracle prefix
```

A second primary plot should use:

```text
x-axis: aggregate distance computations/query
y-axis: Recall@10
```

---

# 12. Experiment E3 — Adaptive Per-Shard Search Intensity

## 12.1 Purpose

Establish C6-c.

Compare:

```text
P0 Fixed-P + Uniform efSearch
P2 Fixed-P + Adaptive efSearch
P5 Oracle Local-Budget Allocation
```

Hold the selected shard set fixed.

---

## 12.2 Fixed Shard Set

For each:

```text
dataset x M
```

use the static fan-out \(P\) selected in E2.

For every query, use the same top-\(P\) shard set for P0, P2, and P5.

Do not allow adaptive shard selection in E3.

---

## 12.3 Uniform Baseline Sweep

Evaluate:

```text
efSearch in {8, 16, 32, 64, 128, 256, 512}
```

for all selected shards.

Find the smallest global `efSearch` reaching Recall@10 >= 0.90 on the tuning set.

Freeze it for P0.

---

## 12.4 Adaptive Budget Tuning

Use:

\[
efSearch_i=\alpha |EP_i(q)|+\beta.
\]

Tune only on the tuning set.

Recommended grid:

```text
alpha in {1, 2, 4, 8, 16}
beta in {4, 8, 16, 32, 64}
```

Use implementation-defined clamping:

```text
ef_min
ef_max
```

Record all tested parameter combinations.

Select the feasible \((\alpha,\beta)\) pair with minimum aggregate distance computations at Recall@10 >= 0.90.

If several configurations differ by <2% work, select the one with lower p99 maximum-per-shard work.

Freeze parameters before measurement.

---

## 12.5 Required Metrics

Record:

```text
Recall@10
aggregate distance computations/query
mean distance computations/selected shard
maximum distance computations among selected shards
nodes visited/query
max nodes visited among selected shards
worker CPU time/query
max worker CPU time/query
```

For \(M=4\) physical runs also record:

```text
mean latency
p50 latency
p95 latency
p99 latency
```

The primary algorithmic metric is aggregate distance computations.

The primary tail metric is maximum per-shard work because scatter-gather completion is bounded by the slowest selected shard.

---

## 12.6 Routing-Evidence Calibration

For every selected query-shard pair, group samples by:

```text
|EP_i(q)| = 1
2
3-4
5-8
>8
```

Within each group report:

```text
probability shard contributes >=1 ground-truth top-10 neighbor
mean number of ground-truth neighbors contributed
oracle local efSearch
oracle local distance computations
```

This validates whether `|EP_i(q)|` is a meaningful proxy for shard relevance.

Compute:

```text
Spearman correlation:
  |EP_i(q)| vs oracle local work
  |EP_i(q)| vs ground-truth contribution
```

---

## 12.7 E3 Outputs

Generate:

```text
c6_fig8_uniform_vs_adaptive_ef.pdf
c6_fig9_local_budget_vs_oracle.pdf
c6_fig10_max_shard_work.pdf
c6_fig11_entry_point_count_calibration.pdf
c6_fig12_physical_tail_latency.pdf
```

Primary plot:

```text
x-axis: aggregate distance computations/query
y-axis: Recall@10

curves:
  Uniform efSearch
  Adaptive efSearch
  Oracle allocation
```

---

# 13. Experiment E4 — Full Adaptive Policy

## 13.1 Purpose

Establish C6-d and measure whether the two adaptive mechanisms compose.

Compare:

```text
P0 Fixed-P + Uniform efSearch
P1 Adaptive Shards + Uniform efSearch
P2 Fixed-P + Adaptive efSearch
P3 Full Orion
P6 Full Oracle
```

This is the primary C6 ablation.

---

## 13.2 Fairness

All policies must use:

```text
same partition
same navigation graph
same navigation search
same routing candidates
same local HNSW indexes
same entry-point injection
same query set
same Recall@10 target
```

Only shard-selection and local-budget policies may differ.

---

## 13.3 Tuning

Tune each deployable policy on the tuning set.

Optimization objective:

\[
\min \text{aggregate distance computations/query}
\]

subject to:

\[
Recall@10\ge0.90.
\]

Secondary tie-breaker:

```text
minimum p99 max-per-shard distance computations
```

Do not tune directly for measurement-set latency.

---

## 13.4 Required Metrics

For each policy report:

```text
Recall@10
mean shards/query
p95 shards/query
aggregate distance computations/query
aggregate nodes visited/query
max per-shard distance computations/query
p95 max per-shard distance computations
worker CPU time/query
navigation routing overhead
```

For \(M=4\) physical execution also report:

```text
QPS
mean latency
p50 latency
p95 latency
p99 latency
aggregator CPU utilization
worker CPU utilization
network bytes/query
```

---

## 13.5 Component Contribution

Normalize P0 cost to 1.0.

Report relative work:

\[
W_{P1}/W_{P0},
\quad
W_{P2}/W_{P0},
\quad
W_{P3}/W_{P0},
\quad
W_{P6}/W_{P0}.
\]

Compute:

```text
shard-adaptation savings = W_P0 - W_P1
budget-adaptation savings = W_P0 - W_P2
full savings = W_P0 - W_P3
oracle headroom = W_P3 - W_P6
```

Do not add component savings arithmetically unless the measured composition is additive.

---

## 13.6 E4 Outputs

Generate:

```text
c6_fig13_full_ablation.pdf
c6_fig14_full_recall_work_frontier.pdf
c6_fig15_oracle_gap.pdf
c6_fig16_physical_end_to_end.pdf
```

Primary ablation:

```text
x-axis:
  Static
  Adaptive Shards
  Adaptive efSearch
  Full Orion
  Oracle

y-axis:
  normalized aggregate distance computations/query

annotate:
  achieved Recall@10
```

---

# 14. Experiment E5 — Physical 4-Machine Validation

## 14.1 Purpose

Confirm that algorithmic work reductions translate into end-to-end performance on the actual four-machine cluster.

Use only:

\[
M=4.
\]

Run both datasets.

Compare:

```text
P0
P1
P2
P3
```

Do not execute oracle policies online.

---

## 14.2 Throughput Procedure

For each policy:

1. warm up with >=1,000 queries;
2. perform a closed-loop concurrency sweep;
3. identify stable saturation throughput;
4. run >=10,000 measured queries at the selected concurrency;
5. repeat >=3 times.

All runs must satisfy:

\[
Recall@10\ge0.90.
\]

---

## 14.3 Bottleneck Checks

Record:

```text
aggregator CPU
worker CPU
network utilization
routing latency
local search latency
queue depth
```

Flag any run with:

```text
aggregator CPU > 85%
network utilization > 85%
persistent queue growth
```

Do not attribute infrastructure saturation to adaptive routing benefits.

---

## 14.4 Required Physical Metrics

Report:

```text
QPS
mean latency
p50 latency
p95 latency
p99 latency
mean shards/query
aggregate distance computations/query
max shard distance computations/query
aggregator CPU
mean worker CPU
network bytes/query
```

---

## 14.5 E5 Outputs

Generate:

```text
c6_fig17_physical_qps.pdf
c6_fig18_physical_p99_latency.pdf
c6_fig19_physical_work_vs_latency.pdf
```

---

# 15. Experiment E6 — Sensitivity

## 15.1 Purpose

Determine whether C6 depends on a narrow parameter setting.

Evaluate Full Orion P3 for:

### Navigation candidate count

Use the actual Orion parameter \(K\).

Test:

```text
0.5x default
1x default
2x default
```

rounded to valid values.

### `alpha`

Test:

```text
0.5x default
1x default
2x default
```

### `beta`

Test:

```text
0.5x default
1x default
2x default
```

### Logical shard count

Use:

\[
M\in\{4,8,16,32\}.
\]

Report:

```text
Recall@10
shards/query
aggregate distance computations
max shard distance computations
```

Do not retune all other parameters independently for each sensitivity point.

---

# 16. Per-Query Trace Schema

Every measurement query must persist:

```text
query_id
dataset
logical_shards
policy
navigation_candidate_count
candidate_shard_count
ranked_candidate_shards
selected_shard_ids
selected_shard_count
entry_point_count_per_selected_shard
assigned_efSearch_per_shard
recall_at_10
distance_computations_per_shard
nodes_visited_per_shard
worker_cpu_time_per_shard
aggregate_distance_computations
max_shard_distance_computations
end_to_end_latency_us_if_physical
oracle_prefix_shards
oracle_total_work
```

---

# 17. Run Metadata

Every run must record:

```text
experiment_id
timestamp
git_commit
dataset
dataset_checksum
logical_shards
physical_hosts
logical_to_physical_mapping
policy
target_recall
achieved_recall
k
navigation_K
shard_ranking_rule
fixed_P
uniform_efSearch
alpha
beta
ef_min
ef_max
HNSW_M
HNSW_efConstruction
query_count
warmup_query_count
worker_threads
aggregator_threads
CPU_affinity
index_artifact_checksum
qps_if_physical
mean_latency_if_physical
p95_latency_if_physical
p99_latency_if_physical
mean_shards_per_query
aggregate_distance_computations_per_query
max_shard_distance_computations
worker_cpu_time_per_query
aggregator_cpu_utilization
worker_cpu_utilization
network_bytes_per_query
```

---

# 18. Statistical Treatment

Use paired per-query comparisons because every policy evaluates the same measurement queries.

Report:

```text
mean
median
p95
p99 where relevant
standard deviation
95% bootstrap confidence interval
```

For physical throughput:

- >=3 repetitions;
- use 5 repetitions if coefficient of variation >5%.

For paired work comparisons, report the bootstrap confidence interval of the mean relative reduction.

Do not rely only on aggregate means.

---

# 19. Execution Order

Execute strictly:

```text
Stage 0
  create experiments/c6/
  implement per-query and per-shard instrumentation
  freeze index artifacts
  validate recall computation
  validate policy switching without index rebuild

Stage 1
  SIFT1M, M=4
  smoke test P0/P1/P2/P3
  verify identical routing candidates across policies

Stage 2
  SIFT1M E1 query-difficulty characterization
  M=4,8,16,32
  update RESULTS.md

Stage 3
  SIFT1M E2 adaptive shard selection
  M=4,8,16,32
  update RESULTS.md with C6-b preliminary verdict

Stage 4
  SIFT1M E3 adaptive local search intensity
  M=4,8,16,32
  update RESULTS.md with C6-c preliminary verdict

Stage 5
  SIFT1M E4 full adaptive ablation
  update RESULTS.md with C6-d preliminary verdict

Stage 6
  SIFT1M E5 physical M=4 validation

Stage 7
  inspect all C6 subclaims
  resolve instrumentation anomalies before second dataset

Stage 8
  repeat E1-E5 on glove-200-angular

Stage 9
  run E6 sensitivity

Stage 10
  generate final figures and tables
  append final C6 verdict
```

Do not launch the full glove-200-angular matrix until SIFT1M produces internally consistent results.

---

# 20. Required Final Figures

Produce:

```text
c6_fig1_oracle_shard_difficulty_cdf.pdf
c6_fig2_oracle_local_budget_cdf.pdf
c6_fig3_static_waste_breakdown.pdf
c6_fig4_static_vs_adaptive_fanout.pdf
c6_fig5_adaptive_vs_oracle_fanout.pdf
c6_fig6_per_query_shard_savings_cdf.pdf
c6_fig7_recall_work_frontier_shard_adaptation.pdf
c6_fig8_uniform_vs_adaptive_ef.pdf
c6_fig9_local_budget_vs_oracle.pdf
c6_fig10_max_shard_work.pdf
c6_fig11_entry_point_count_calibration.pdf
c6_fig12_physical_tail_latency.pdf
c6_fig13_full_ablation.pdf
c6_fig14_full_recall_work_frontier.pdf
c6_fig15_oracle_gap.pdf
c6_fig16_physical_end_to_end.pdf
c6_fig17_physical_qps.pdf
c6_fig18_physical_p99_latency.pdf
c6_fig19_physical_work_vs_latency.pdf
c6_combined_adaptivity.pdf
```

Recommended combined C6 figure:

```text
(a) Per-query oracle shard requirement CDF
(b) Fixed-P vs adaptive shard selection at 90% recall
(c) Uniform vs adaptive per-shard efSearch
(d) Static / shard-adaptive / budget-adaptive / full / oracle ablation
```

---

# 21. Required Final Tables

Generate:

```text
c6_query_difficulty_summary.csv
c6_shard_adaptation_summary.csv
c6_local_budget_summary.csv
c6_full_ablation_summary.csv
c6_oracle_gap_summary.csv
c6_physical_validation_summary.csv
c6_sensitivity_summary.csv
```

Also generate one LaTeX-ready table:

```text
Dataset
Logical Shards
Policy
Recall@10
Mean Shards/Query
Aggregate Distance Computations
P95 Max-Shard Work
QPS if Physical
P99 Latency if Physical
```

---

# 22. C6 Success Criteria

## C6-a — Query difficulty heterogeneity

`SUPPORTED` if:

- oracle shard requirement has a non-trivial distribution;
- a meaningful fraction of queries require fewer shards than the static configuration;
- a meaningful fraction require different local search budgets.

Do not require a predefined numerical spread.

---

## C6-b — Adaptive shard selection

`SUPPORTED` if, at Recall@10 >= 0.90:

- P1 searches fewer mean shards than P0 and/or performs lower aggregate graph work;
- the reduction is consistent on both datasets or explicitly qualified as dataset-dependent;
- P1 remains reasonably close to the oracle-prefix frontier.

If mean fan-out decreases but aggregate work increases substantially, mark the claim `INSUFFICIENT`.

---

## C6-c — Adaptive local search intensity

`SUPPORTED` if, with identical shard sets:

- P2 reaches Recall@10 >= 0.90;
- P2 performs less aggregate graph work than P0 and/or reduces p95/p99 maximum-per-shard work;
- `|EP_i(q)|` has a measurable relationship with shard relevance or oracle local budget.

If adaptive `efSearch` only redistributes work without reducing total or tail work, mark C6-c `INSUFFICIENT`.

---

## C6-d — Full adaptive policy

`SUPPORTED` if:

- P3 reaches Recall@10 >= 0.90;
- P3 dominates P0 in aggregate graph work;
- P3 is at least as good as the better of P1/P2 within statistical variation, or provides a clear complementary benefit;
- P3 retains limited headroom to P6 relative to P0.

Do not claim near-oracle behavior unless the measured oracle gap is small enough to support that wording.

---

# 23. Contradiction Handling

If oracle shard requirement is nearly constant:

```text
C6-a:
  CONTRADICTED or weakly supported

Action:
  do not motivate adaptive shard selection using query-difficulty heterogeneity
```

If P1 reduces fan-out but loses recall:

```text
C6-b:
  CONTRADICTED

Action:
  do not compare work without enforcing Recall@10 >= 0.90
```

If P1 matches P0 work:

```text
C6-b:
  INSUFFICIENT

Inspect:
  shard ranking
  candidate K
  label diversity
  fixed-P tuning quality
```

If `|EP_i(q)|` is uncorrelated with shard relevance or oracle local work:

```text
C6-c:
  CONTRADICTED for the current allocation signal

Action:
  retain result
  do not claim topological confidence from entry-point count
```

If P2 reduces mean work but increases max-shard work or p99 latency:

```text
record both effects
do not claim tail-latency improvement
```

If P3 is worse than P1 or P2:

```text
C6-d:
  CONTRADICTED or INSUFFICIENT

Action:
  inspect interaction between shard adaptation and local-budget adaptation
  do not present component savings as additive
```

If the oracle gap remains large:

```text
remove "near-oracle" wording
retain only measured improvement over static policies
```

---

# 24. Final Verdict Format

Append to `RESULTS.md`:

```text
C6-a Query difficulty is heterogeneous:
  SUPPORTED / CONTRADICTED / INSUFFICIENT

Evidence:
  Oracle shard-requirement distribution:
  Oracle local-budget distribution:
  Static over-search fraction:
  Static under-search fraction:

C6-b Adaptive shard selection reduces redundant work:
  SUPPORTED / CONTRADICTED / INSUFFICIENT

Evidence:
  Recall:
  Mean shard reduction:
  Aggregate work reduction:
  Oracle-prefix gap:
  Dataset consistency:

C6-c Adaptive local search intensity improves allocation:
  SUPPORTED / CONTRADICTED / INSUFFICIENT

Evidence:
  Recall:
  Aggregate work reduction:
  Max-shard work reduction:
  Physical p99 impact:
  Entry-point-count calibration:

C6-d Full adaptive policy approaches the best recall-cost frontier:
  SUPPORTED / CONTRADICTED / INSUFFICIENT

Evidence:
  Full vs static:
  Full vs component ablations:
  Full vs oracle:
  Physical validation:
```

---

# 25. Paper-Ready Claim Template

Populate only after the complete experiment suite finishes:

> Query difficulty varies substantially in distributed graph search: different queries require different shard fan-out and local search effort to reach 90% Recall@10. A fixed routing configuration therefore either performs unnecessary work on easy queries or leaves insufficient search budget for hard queries. Orion adapts both decisions using navigation-graph evidence. Compared with a globally tuned fixed-\(P\), uniform-`efSearch` baseline, adaptive shard selection reduces redundant shard searches, while adaptive per-shard search intensity shifts graph traversal toward the shards most relevant to each query. Their combination reduces aggregate graph-search work at the same recall and approaches the oracle recall–cost frontier.

Remove or weaken every clause not directly supported by recorded measurements.
