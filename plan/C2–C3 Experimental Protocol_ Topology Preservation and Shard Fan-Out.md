# C2–C3 Experimental Protocol: Topology Preservation and Shard Fan-Out

## 1. Objective

Establish two distinct but causally connected claims using one integrated experiment suite.

### C2 — Topology-Oblivious Partitioning Disrupts Graph Search

> **C2. Conventional partitioning methods such as Random and K-Means divide vectors without accounting for the graph-search paths used by HNSW. Their shard boundaries cut search-relevant graph structure and degrade local graph navigability.**

Required causal chain:

\[
\text{Topology-oblivious partitioning}
\rightarrow
\text{search-relevant topology disruption}
\rightarrow
\text{lower local graph navigability}.
\]

### C3 — Topology-Aware Partitioning Reduces Required Shard Fan-Out

> **C3. Orion's topology-aware partitioning preserves search-relevant graph structure and consequently reduces the number of shards sufficient to achieve Recall@10 \(\geq 0.90\).**

Required causal chain:

\[
\text{Topology-aware partitioning}
\rightarrow
\text{better local navigability}
\rightarrow
\text{fewer shards required at fixed recall}.
\]

C2 and C3 must be evaluated jointly but reported as separate conclusions.

C2 establishes the mechanism.

C3 establishes the system-level consequence of that mechanism.

---

# 2. Scope

This experiment suite evaluates **partition quality only**.

Do not evaluate Orion's online routing mechanism here.

Do not use:

- adaptive shard selection;
- Orion online navigation routing;
- entry-point injection;
- adaptive per-shard `efSearch`;
- topology-aware runtime pruning.

Online routing must be isolated for the later routing-specific experiment suite.

Shard-selection experiments in this protocol must use oracle or controlled shard ordering so that routing accuracy does not confound partition quality.

---

# 3. Hardware Model

Physical cluster:

```text
4 machines
```

Logical shard counts:

\[
M\in\{2,4,8,16,32\}.
\]

For:

\[
M\leq4,
\]

one logical shard may execute on one physical machine.

For:

\[
M>4,
\]

logical shards must be distributed evenly across the four physical machines.

Logical-shard experiments are valid for:

- topology metrics;
- local navigability;
- shard fan-out;
- distance computations;
- aggregate graph-search work.

Do not interpret \(M>4\) wall-clock results as \(M\)-machine performance.

For local-search microbenchmarks, execute one logical shard at a time to eliminate colocated-shard resource contention.

---

# 4. Datasets

Use exactly:

| Dataset | Metric | Accuracy Target |
|---|---|---|
| SIFT1M | L2 | Recall@10 \(\geq 0.90\) |
| glove-200-angular | Angular/Cosine | Recall@10 \(\geq 0.90\) |

Use official base vectors, query vectors, and ground truth where available.

Use identical queries for every partition configuration of a dataset.

Separate:

```text
tuning queries: 1,000
measurement queries: >= 10,000 when available
```

Do not report tuning queries.

---

# 5. Partition Configurations

The primary comparison must include exactly the following configurations.

## P0 — Random

Uniform random assignment:

\[
v\rightarrow\{1,\ldots,M\}.
\]

Use deterministic seeds.

---

## P1 — K-Means

Run \(M\)-way K-Means over the full base-vector set.

Assign each vector to its nearest centroid.

This is the primary topology-oblivious spatial baseline.

---

## P2 — Orion Topology-Aware Partitioning

Use the current Orion offline partitioning pipeline:

1. sample vectors for the navigation graph;
2. build the navigation HNSW;
3. initialize shard labels;
4. apply topology-aware self-search label refinement;
5. assign unsampled vectors using navigation-graph search and refined shard labels;
6. produce \(M\) disjoint shards.

Do not use Orion's online routing.

---

# 6. Required Orion Partition Ablation

Implement the following optional-but-preferred intermediate configuration if supported by the current codebase.

## P2-A — Orion without Topology Refinement

Use:

1. navigation graph;
2. initial K-Means shard labels;
3. search-based assignment;

but disable self-search topology refinement.

This configuration isolates the effect of topology-aware label refinement.

If implementation would require major unrelated changes, record it as unavailable and proceed with P0/P1/P2.

Do not delay the primary experiments for this ablation.

---

# 7. Global Reference Graph

For each dataset, construct one full-dataset HNSW reference graph.

Use the same HNSW implementation and graph-construction parameters used by the distributed workers.

Persist:

```text
global_graph/
  node_ids
  L0_edges
  upper_layer_edges if available
```

Primary topology analysis must use L0.

Upper-layer analysis is optional.

The reference graph is used only for offline analysis and query tracing.

It is not part of Orion's serving path.

---

# 8. Query Trace Instrumentation

Execute every measurement query on the full reference HNSW.

Record the complete search trace:

```text
query_id
visited_node_ids
visited_edges
distance_computations
result_ids
ground_truth_ids
```

If exact edge traversal order is available, also record:

```text
source_node
destination_node
traversal_order
layer
```

The same global query traces must be mapped onto every partition layout.

Do not regenerate query traces separately for Random, K-Means, and Orion.

---

# 9. Continuous Result Recording

Create:

```text
experiments/c23/
  PLAN.md
  STATUS.md
  RESULTS.md
  runs/
    manifest.jsonl
    summary.csv
    per_query/
    topology/
  figures/
  scripts/
  logs/
```

After every completed experiment stage, update:

```text
STATUS.md
RESULTS.md
```

`RESULTS.md` entries must use:

```text
Timestamp:
Git commit:
Experiment:
Dataset:
Logical shards:
Partition method:
Run IDs:
Primary metrics:
Observed result:
C2 implication:
C3 implication:
Status:
  SUPPORTS / CONTRADICTS / INSUFFICIENT
Anomalies:
Required follow-up:
```

Never overwrite previous conclusions.

Append revisions with timestamps.

---

# 10. Experiment Structure

Execute four core experiments:

| Experiment | Function | Claim |
|---|---|---|
| E1 | Structural topology preservation | C2 |
| E2 | Search-path preservation | C2 |
| E3 | Local graph navigability | C2 → C3 |
| E4 | Required shard fan-out | C3 |

The required evidence chain is:

```text
E1
topology-oblivious layouts cut graph structure

        ↓

E2
they specifically cut edges used during real searches

        ↓

E3
local HNSW search becomes less effective

        ↓

E4
more shards must be searched to reach 90% global recall
```

Do not treat E1 alone as sufficient proof of C2.

---

# 11. E1 — Structural Topology Preservation

## 11.1 Purpose

Measure how each partition layout interacts with the full HNSW topology.

Establish whether topology-aware partitioning preserves more graph structure inside individual shards.

---

## 11.2 Edge-Cut Ratio

For every L0 edge:

\[
(u,v)\in E,
\]

define:

\[
Cut(u,v)=
\begin{cases}
1,&Shard(u)\neq Shard(v),\\
0,&otherwise.
\end{cases}
\]

Compute:

\[
EdgeCut(M)=
\frac{
\sum_{(u,v)\in E}Cut(u,v)
}{
|E|
}.
\]

Report separately for:

```text
Random
K-Means
Orion
Orion-NoRefinement if available
```

---

## 11.3 Retained Neighbor Degree

For every node \(v\):

\[
d_{\mathrm{intra}}(v)
=
|\{u:(v,u)\in E,\ Shard(v)=Shard(u)\}|.
\]

Report:

```text
mean intra-shard degree
median intra-shard degree
p10 intra-shard degree
fraction of nodes losing >25% neighbors
fraction of nodes losing >50% neighbors
fraction of nodes losing >75% neighbors
```

Normalize by original reference-graph degree.

---

## 11.4 Connectedness Diagnostics

Construct the subgraph induced by each shard using reference-graph edges.

Measure:

```text
number of connected components
largest connected component fraction
isolated-node fraction
```

Aggregate across shards using both:

```text
unweighted shard mean
vector-count-weighted mean
```

Connectedness is a supporting metric.

Do not use connected-component statistics as the primary C2 evidence.

---

## 11.5 E1 Output

Generate:

```text
fig_c23_edge_cut.pdf
fig_c23_retained_degree.pdf
fig_c23_connectivity.pdf
```

Primary figure:

```text
x-axis: logical shard count
y-axis: reference L0 edge-cut ratio

curves:
  Random
  K-Means
  Orion
```

---

# 12. E2 — Search-Path Preservation

## 12.1 Purpose

Distinguish generic graph edges from edges that are actually important to ANN search.

This is the primary topology metric for C2.

---

## 12.2 Traversal-Weighted Edge Cut

Using the global HNSW query traces, define:

\[
f(u,v)
\]

as the number of measurement queries whose search traverses edge \((u,v)\).

Define:

\[
TWCut(M)
=
\frac{
\sum_{(u,v)\in E}
f(u,v)\cdot Cut(u,v)
}{
\sum_{(u,v)\in E}f(u,v)
}.
\]

This metric measures the fraction of actual graph-search transitions that cross partition boundaries.

Compute independently for every partition layout.

---

## 12.3 Per-Query Search-Path Fragmentation

For each query \(q\), map every visited node to its shard.

Compute:

### Unique Shards on Global Search Path

\[
S_{\mathrm{path}}(q)
=
|\{Shard(v):v\in Trace(q)\}|.
\]

### Shard Transition Count

For ordered traversal sequence:

\[
v_1,v_2,\ldots,v_t,
\]

compute:

\[
Transitions(q)
=
\sum_{j=1}^{t-1}
\mathbb{I}
[
Shard(v_j)\neq Shard(v_{j+1})
].
\]

### Dominant-Shard Concentration

\[
Concentration(q)
=
\max_s
\frac{
|\{v\in Trace(q):Shard(v)=s\}|
}{
|Trace(q)|
}.
\]

Report mean, median, p95, and CDF.

---

## 12.4 Hot-Edge Analysis

Rank global HNSW edges by traversal frequency \(f(u,v)\).

For:

```text
top 1%
top 5%
top 10%
all edges
```

calculate edge-cut ratio.

Required diagnostic:

Determine whether K-Means disproportionately cuts frequently traversed routing edges and whether Orion retains a larger fraction.

---

## 12.5 E2 Outputs

Generate:

```text
fig_c23_traversal_weighted_cut.pdf
fig_c23_path_shard_count_cdf.pdf
fig_c23_hot_edge_retention.pdf
```

Primary C2 topology plot:

```text
x-axis: logical shard count
y-axis: traversal-weighted edge-cut ratio

curves:
  Random
  K-Means
  Orion
```

---

# 13. E3 — Local Graph Navigability

## 13.1 Purpose

Determine whether topology preservation translates into better local HNSW search.

This experiment connects C2 to C3.

---

## 13.2 Local Index Construction

For every:

```text
dataset
partition method
M ∈ {2,4,8,16,32}
```

construct a standard local HNSW on each shard.

Use identical:

```text
HNSW M
efConstruction
distance implementation
construction procedure
```

Do not retain Orion-specific local entry points.

Do not use navigation-graph candidates.

Every partition method must use the same conventional local HNSW search procedure.

---

# 14. Query-Shard Evaluation Set

For every measurement query \(q\), identify all shards containing at least one global ground-truth top-10 neighbor.

Define:

\[
G_s(q)
=
GT(q)\cap Shard_s.
\]

Evaluate local navigability only for query-shard pairs satisfying:

\[
|G_s(q)|>0.
\]

This prevents irrelevant shards from dominating local-recall statistics.

---

# 15. Local Target Recovery

For a query-shard pair \((q,s)\), define:

\[
Recall_{\mathrm{local}}(q,s)
=
\frac{
|Result_s(q)\cap G_s(q)|
}{
|G_s(q)|
}.
\]

Run local HNSW from its standard local entry mechanism.

Evaluate a fixed `efSearch` grid:

```text
efSearch ∈ {10, 20, 40, 80, 160, 320}
```

Adjust the grid only if required by the implementation.

The grid must be identical across partition methods.

---

# 16. Local Navigability Metrics

For every configuration record:

```text
local target recall
distance computations
nodes visited
local HNSW latency
local target rank
```

Primary comparisons:

### Fixed Budget

At identical `efSearch`:

```text
Orion local recall
vs.
K-Means local recall
vs.
Random local recall
```

### Fixed Local Recall

For local target-recall thresholds:

```text
0.80
0.90
0.95
```

determine required:

```text
efSearch
distance computations
nodes visited
```

The fixed-budget result is the primary evidence for topology-induced navigability differences.

---

# 17. Local Search Efficiency Curve

For each partition method plot:

```text
x-axis: mean distance computations
y-axis: mean local target recall
```

Each point corresponds to one `efSearch`.

Orion supports C2 if its curve dominates topology-oblivious layouts.

Do not compare wall-clock latency before confirming comparable CPU placement and cache behavior.

---

# 18. E3 Outputs

Generate:

```text
fig_c23_local_recall_vs_ef.pdf
fig_c23_local_recall_vs_distance_computations.pdf
fig_c23_local_work_at_fixed_recall.pdf
```

Primary plot:

```text
x-axis: distance computations/query-shard
y-axis: local target recall

curves:
  Random
  K-Means
  Orion
```

---

# 19. E4 — Required Shard Fan-Out

## 19.1 Purpose

Establish C3 while excluding online router quality.

Measure the minimum number of shards sufficient to reach global Recall@10 \(\geq0.90\).

Use two complementary fan-out definitions.

---

# 20. E4-A — Exact-Search Oracle Fan-Out

This metric isolates **data placement locality**.

For each query and every shard:

1. perform exact search within the shard;
2. obtain the exact local top-10;
3. calculate each shard's contribution to global top-10 ground truth.

Define the minimum number of shards required to recover at least 9 of the global top-10 ground-truth neighbors:

\[
P_{\mathrm{exact}}(q).
\]

Use oracle shard selection that minimizes shard count.

This is not a deployable routing result.

It is a lower bound determined solely by the partition layout.

---

# 21. E4-B — HNSW Oracle Fan-Out

This metric measures **placement + local graph navigability**.

For every query and shard:

1. execute local HNSW using a fixed common search configuration;
2. collect the shard's returned candidates;
3. calculate the actual ground-truth contribution recovered by that shard.

Use oracle shard ordering to find:

\[
P_{\mathrm{HNSW}}(q),
\]

the minimum number of local HNSW searches required to reach global Recall@10 \(\geq0.90\).

The local search configuration must be identical across partition methods.

Use the `efSearch` selected by the following rule:

```text
Choose the smallest common efSearch at which the unpartitioned
M=1 HNSW reaches Recall@10 >= 0.90 on the tuning set.
```

Do not separately tune `efSearch` for each partition method in the primary E4-B result.

A separately tuned result may be reported as secondary analysis.

---

# 22. Fan-Out Decomposition

For every query define:

\[
\Delta P(q)
=
P_{\mathrm{HNSW}}(q)
-
P_{\mathrm{exact}}(q).
\]

Interpretation:

```text
P_exact:
  fan-out caused by distribution of relevant vectors across shards

Delta P:
  additional fan-out caused by imperfect local ANN recovery
```

Compare \(\Delta P\) across partition methods.

This decomposition is required.

It directly connects:

```text
C2 local navigability
```

to:

```text
C3 required shard fan-out
```

---

# 23. Aggregate Search Work at 90% Recall

For each query, under HNSW oracle shard ordering, accumulate local search work until global Recall@10 reaches 0.90.

Record:

\[
W_{90}(q)
=
\sum_{i=1}^{P_{\mathrm{HNSW}}(q)}
D_i(q),
\]

where \(D_i(q)\) is local distance-computation count.

Report:

```text
mean W90
median W90
p95 W90
```

This is a secondary C3 metric.

The primary C3 metric remains shard fan-out.

---

# 24. E4 Outputs

Generate:

```text
fig_c23_exact_oracle_fanout.pdf
fig_c23_hnsw_oracle_fanout.pdf
fig_c23_fanout_cdf.pdf
fig_c23_fanout_decomposition.pdf
fig_c23_work_to_90_recall.pdf
```

Primary C3 figure:

```text
x-axis: logical shard count = 2, 4, 8, 16, 32
y-axis: mean minimum shards required for Recall@10 >= 0.90

panels:
  Exact local search
  Local HNSW

curves:
  Random
  K-Means
  Orion
```

---

# 25. Required C2–C3 Causal Correlation Analysis

Across all:

```text
dataset × M × partition-method
```

configurations, construct a table containing:

```text
edge_cut_ratio
traversal_weighted_cut
retained_neighbor_degree
local_target_recall
distance_computations_at_fixed_local_recall
P_exact
P_HNSW
Delta_P
W90
```

Compute correlations between:

### Topology disruption and navigability

```text
traversal_weighted_cut
vs.
local target recall
```

and:

```text
traversal_weighted_cut
vs.
distance computations at fixed local recall
```

### Navigability and fan-out

```text
local target recall
vs.
P_HNSW
```

and:

```text
Delta_P
vs.
traversal_weighted_cut
```

Use:

```text
Pearson
Spearman
```

Do not interpret correlation as formal causality.

Use the analysis only to validate consistency of the proposed mechanism.

---

# 26. Required Ablation Analysis

If P2-A is available, evaluate:

```text
K-Means
Orion-NoRefinement
Orion-Full
```

for:

```text
traversal-weighted edge cut
local target recall
P_HNSW
W90
```

Required question:

> Does explicit topology refinement improve both topology preservation and downstream fan-out beyond search-based assignment alone?

Record results regardless of whether the answer is positive.

---

# 27. Load-Balance Control

Topology-aware partitioning must not achieve lower fan-out only by creating severely imbalanced shards.

For every layout report:

```text
min shard size
max shard size
mean shard size
standard deviation
coefficient of variation
max_size / mean_size
```

Also report:

```text
per-shard query participation frequency
```

If Orion produces substantial imbalance, record it explicitly.

Generate:

```text
fig_c23_shard_size_balance.pdf
```

Do not hide imbalance behind mean fan-out.

---

# 28. Partition Construction Cost

Record for all partition methods:

```text
partition construction time
peak construction memory
final per-shard vector counts
navigation graph memory for Orion
navigation graph construction time
topology refinement time
unsampled-vector assignment time
```

These are overhead metrics.

They are not primary C2/C3 evidence.

Persist for later system-level evaluation.

---

# 29. Experimental Execution Order

Execute strictly:

```text
Stage 0
  finalize instrumentation
  build C23 directory
  validate recall and ground truth
  validate global HNSW query-trace recording

Stage 1
  SIFT1M
  build global reference graph
  generate global query traces

Stage 2
  SIFT1M E1
  M = 2, 4, 8, 16, 32

Stage 3
  SIFT1M E2
  M = 2, 4, 8, 16, 32

Stage 4
  inspect C2 structural evidence
  update RESULTS.md

Stage 5
  SIFT1M E3
  M = 2, 4, 8, 16, 32

Stage 6
  inspect C2 navigability evidence
  update RESULTS.md

Stage 7
  SIFT1M E4
  exact fan-out
  HNSW fan-out
  fan-out decomposition
  W90

Stage 8
  issue preliminary C2 and C3 verdicts
  update RESULTS.md

Stage 9
  repeat E1-E4 on glove-200-angular

Stage 10
  run P2-A ablation if available

Stage 11
  run correlation analysis
  run load-balance analysis
  compile construction overheads

Stage 12
  generate final figures
  update final C2 verdict
  update final C3 verdict
```

Do not begin the complete GloVe experiment matrix until the SIFT1M pipeline produces internally consistent E1–E4 results.

---

# 30. Required Raw Result Schema

Every configuration must record:

```text
experiment_id
timestamp
git_commit
dataset
dataset_checksum
partition_method
partition_seed
logical_shards
physical_hosts
logical_to_physical_mapping
vector_count
dimension
distance_metric
HNSW_M
HNSW_efConstruction
efSearch
navigation_sample_size
navigation_sample_rate
k_nav
shard_size_mean
shard_size_std
shard_size_min
shard_size_max
edge_cut_ratio
traversal_weighted_edge_cut
mean_retained_degree_ratio
largest_component_fraction
mean_path_shards
mean_path_transitions
mean_local_target_recall
mean_local_distance_computations
P_exact_mean
P_exact_p95
P_HNSW_mean
P_HNSW_p95
Delta_P_mean
W90_mean
W90_p95
```

Per-query data must include:

```text
query_id
partition_method
logical_shards
global_trace_length
path_shard_count
path_transition_count
ground_truth_shards
P_exact
P_HNSW
Delta_P
W90
```

Per query-shard E3 data must include:

```text
query_id
shard_id
ground_truth_count_in_shard
efSearch
local_recovered_ground_truth
local_target_recall
distance_computations
nodes_visited
local_latency
```

---

# 31. Statistical Treatment

Use all measurement queries for E1/E2 topology statistics.

For query-level comparisons between K-Means and Orion, use paired measurements because the query set is identical.

Report:

```text
mean
median
p95
standard deviation
```

For headline differences also report 95% bootstrap confidence intervals over queries.

For partition construction randomness:

```text
Random: >= 3 seeds
K-Means: >= 3 seeds if practical
Orion: >= 3 seeds if partitioning contains randomness
```

If construction variance is negligible after validation, subsequent expensive configurations may use one fixed documented seed.

Record the decision in `RESULTS.md`.

---

# 32. C2 Success Criteria

C2 is `SUPPORTED` only if the evidence chain includes both topology and local-search effects.

Required:

### C2-1

K-Means and/or Random exhibit higher search-relevant topology disruption than Orion.

Primary metric:

\[
TWCut.
\]

### C2-2

Higher topology disruption corresponds to worse local HNSW navigability.

Required evidence at fixed search budget:

```text
lower local target recall
and/or
more graph work required for the same local recall
```

### C2-3

The trend appears on both SIFT1M and glove-200-angular or is explicitly qualified as dataset-dependent.

Edge-cut reduction without an observable local-search effect is insufficient to support the full C2 claim.

---

# 33. C3 Success Criteria

C3 is `SUPPORTED` only if Orion requires fewer shards to achieve Recall@10 \(\geq0.90\).

Primary evidence:

\[
P_{\mathrm{HNSW,Orion}}
<
P_{\mathrm{HNSW,KMeans}}
\]

for a meaningful portion of the evaluated logical-shard range.

Required supporting evidence:

- \(P_{\mathrm{exact}}\) identifies how much improvement originates from placement locality.
- \(\Delta P\) identifies how much originates from better local graph recovery.
- \(W_{90}\) confirms that lower fan-out does not result from substantially higher per-shard search work.

Do not declare C3 from lower edge-cut ratio alone.

---

# 34. Contradiction Handling

If Orion reduces edge cuts but local search quality is unchanged:

```text
C2:
  INSUFFICIENT

Interpretation:
  structural topology preservation does not materially affect
  local HNSW navigability under the tested workload.
```

If Orion improves local search quality but does not reduce \(P_{\mathrm{HNSW}}\):

```text
C2:
  potentially SUPPORTED

C3:
  CONTRADICTED or INSUFFICIENT
```

If \(P_{\mathrm{exact}}\) improves but \(P_{\mathrm{HNSW}}\) does not:

```text
placement locality improved
local graph recovery eliminated the expected benefit
investigate local HNSW construction/search
```

If \(P_{\mathrm{HNSW}}\) improves but \(P_{\mathrm{exact}}\) does not:

```text
benefit originates primarily from local graph navigability
rather than ground-truth co-location
```

This is a valid and potentially stronger C2→C3 result.

If K-Means matches or outperforms Orion:

```text
retain result
record contradiction
do not retune baselines asymmetrically
inspect dataset dependence and partition-balance effects
```

---

# 35. Required Final Figures

Produce:

```text
c23_fig1_traversal_weighted_edge_cut.pdf
c23_fig2_local_navigability.pdf
c23_fig3_exact_fanout.pdf
c23_fig4_hnsw_fanout.pdf
c23_fig5_fanout_decomposition.pdf
c23_fig6_work_to_90_recall.pdf
c23_fig7_partition_balance.pdf
c23_combined_causal_chain.pdf
```

Recommended combined figure:

```text
(a) Traversal-weighted edge cut
(b) Local recall vs distance computations
(c) Exact-search minimum fan-out
(d) HNSW minimum fan-out at 90% recall
```

The four panels must present the intended chain:

\[
\text{topology}
\rightarrow
\text{navigability}
\rightarrow
\text{fan-out}.
\]

---

# 36. Required Final Tables

Generate:

```text
c23_topology_summary.csv
c23_local_navigability_summary.csv
c23_exact_fanout_summary.csv
c23_hnsw_fanout_summary.csv
c23_fanout_decomposition.csv
c23_partition_balance.csv
c23_correlation_analysis.csv
c23_construction_overhead.csv
```

---

# 37. Final Verdict Format

Append to `RESULTS.md`:

```text
C2: Topology-oblivious partitioning disrupts graph-search topology
Status:
  SUPPORTED / CONTRADICTED / INSUFFICIENT

Evidence:
  Traversal-weighted edge cut:
  Local target recall:
  Work at fixed local recall:
  Dataset consistency:
  Exceptions:

C3: Topology-aware partitioning reduces required shard fan-out
Status:
  SUPPORTED / CONTRADICTED / INSUFFICIENT

Evidence:
  Exact oracle fan-out:
  HNSW oracle fan-out:
  Fan-out reduction at 90% recall:
  Work to 90% recall:
  Dataset consistency:
  Exceptions:
```

Do not merge the final C2 and C3 verdicts.

---

# 38. Paper-Ready Causal Claim Template

Populate only after all experiments finish:

> Topology-oblivious sharding disrupts graph structure that is actively used during HNSW traversal. Across SIFT1M and glove-200-angular, Random and K-Means cut a larger fraction of traversal-weighted graph edges than Orion, resulting in lower local target recovery or higher graph-search work at the same local recall. By preserving these search-relevant neighborhoods, Orion reduces the number of local shards sufficient to achieve 90% global Recall@10. The fan-out decomposition further shows whether the reduction originates from stronger neighbor co-location, improved local graph navigability, or both.

Remove or weaken every clause that is not directly supported by recorded results.