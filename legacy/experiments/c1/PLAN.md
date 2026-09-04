# C1 Experimental Protocol: Scale-Out Bottleneck in Scatter-Gather Graph ANN Search

## 1. Objective

Establish the following claim:

> **C1. Conventional scatter-gather graph ANN search exhibits sub-linear scale-out at a fixed 90% recall target because query shard fan-out remains high while per-shard graph-search work decreases much more slowly than shard size. The resulting aggregate work per query limits throughput scaling.**

The physical cluster contains four machines. Physical scale-out measurements are therefore limited to 1, 2, and 4 physical shards. Experiments with more than four shards must use logical-shard simulation and must not be reported as measured physical-machine throughput scaling.

The complete causal chain to establish is:

\[
\text{More shards}
\rightarrow
\begin{cases}
\text{high shard fan-out}\\
\text{slow reduction in local graph-search work}
\end{cases}
\rightarrow
\text{high aggregate work/query}
\rightarrow
\text{sub-linear physical throughput scaling}.
\]

---

## 2. Required Conclusions

### C1-a: Physical throughput scaling is sub-linear

For \(M\in\{1,2,4\}\) physical shards, increasing the number of machines does not produce proportional throughput growth at fixed Recall@10 \(\geq 0.90\).

Define:

\[
S(M)=\frac{QPS(M)}{QPS(1)}
\]

and:

\[
E(M)=\frac{QPS(M)}{M\cdot QPS(1)}.
\]

Required evidence:

\[
S(M)<M
\]

and a decreasing or clearly sub-unity scaling efficiency \(E(M)\).

Only \(M\leq4\) may be used for this conclusion.

---

### C1-b: Query fan-out remains substantial as logical shard count grows

For logical shard counts:

\[
M\in\{1,2,4,8,16,32\},
\]

measure the number of shards required to maintain Recall@10 \(\geq0.90\).

Required evidence:

- Random sharding requires high fan-out.
- K-Means reduces fan-out relative to Random but does not reduce it to a consistently small constant.
- Per-query required shard count is heterogeneous.
- Aggregate shard fan-out increases sufficiently with \(M\) to create redundant distributed work.

---

### C1-c: Local graph-search work decreases slowly as shard size decreases

For local index sizes corresponding to:

\[
N,\frac{N}{2},\frac{N}{4},\frac{N}{8},\frac{N}{16},\frac{N}{32},
\]

measure graph-search work at fixed search quality.

Required evidence:

- Distance computations/query and nodes visited/query decrease substantially more slowly than \(1/M\).
- Wall-clock latency may exhibit cache effects and is secondary evidence.
- The main algorithmic conclusion must be based on distance computations and node visits.

Do not claim that measured HNSW runtime is exactly \(O(\log N)\). Use logarithmic scaling only as an explanatory reference if supported by the observed trend.

---

### C1-d: Fan-out and local work explain aggregate query cost

Define observed aggregate graph-search work:

\[
W_{\mathrm{obs}}(q,M)
=
\sum_{i\in R(q)}
D_i(q),
\]

where \(D_i(q)\) is the number of distance computations performed by logical shard \(i\).

Define the decomposition model:

\[
W_{\mathrm{model}}(M)
=
\bar P(M)
\cdot
\bar D_{\mathrm{local}}(M).
\]

Required evidence:

- \(W_{\mathrm{model}}\) tracks the trend of observed aggregate work.
- Aggregate work does not decrease proportionally with logical shard count.
- The work trend is consistent with the physical throughput scaling observed for \(M\leq4\).

---

## 3. Datasets and Accuracy Target

Use exactly two datasets:

| Dataset | Distance Metric | Primary Accuracy Target |
|---|---|---|
| SIFT1M | L2 | Recall@10 \(\geq 0.90\) |
| glove-200-angular | Angular/Cosine | Recall@10 \(\geq 0.90\) |

Use the dataset-provided base vectors, query vectors, and ground truth when available.

Use identical query sets across all configurations for a dataset.

Partition the query set into:

- tuning set: 1,000 queries;
- measurement set: at least 10,000 queries, or all remaining official queries if fewer are available;
- tuning queries must never be included in reported measurement results.

---

## 4. Baselines

Implement exactly the following baseline layouts for C1.

### Random

Assign every base vector uniformly at random to one of \(M\) shards.

Use deterministic seeds.

For end-to-end serving, Random uses broadcast routing unless an existing implementation already defines a query-aware random-shard router. Do not introduce an Orion-specific router.

Tune only local graph-search parameters to reach 90% recall.

### K-Means

Cluster the dataset into \(M\) partitions.

Assign one cluster to each logical shard.

At query time:

1. compute query-to-centroid distances;
2. rank shards by centroid distance;
3. probe the top-\(P\) shards;
4. search their local graph indexes;
5. merge local top-\(k\) results.

Tune \(P\) and local search parameters jointly to meet 90% recall.

### Prohibited C1 mechanisms

Do not use:

- topology-aware label refinement;
- navigation-graph-guided partitioning;
- Orion adaptive routing;
- Orion entry-point injection;
- per-shard adaptive `efSearch`;
- any Orion-specific metadata.

---

## 5. Physical and Logical Shard Configurations

### 5.1 Physical scale-out configurations

Use:

\[
M_{\mathrm{physical}}\in\{1,2,4\}.
\]

Each physical shard must execute on a distinct machine.

The same worker CPU allocation must be used per physical shard.

The aggregator must use the same CPU allocation in every configuration.

If the aggregator is colocated with a worker, reserve and pin a fixed set of aggregator cores on the same host in every experiment. These cores must not be available to the worker process.

Record the complete CPU affinity and process placement for every run.

---

### 5.2 Logical-shard simulation

Use:

\[
M_{\mathrm{logical}}\in\{8,16,32\}
\]

in addition to the physical configurations.

Logical shards must be distributed evenly across the four machines:

\[
\text{logical shards per host}=M_{\mathrm{logical}}/4.
\]

Use deterministic round-robin placement.

Each logical shard must remain an independent index instance with independent statistics.

For \(M>4\):

- do not report measured QPS as equivalent to an \(M\)-machine cluster;
- do not report colocated wall-clock latency as physical scale-out latency;
- use these configurations for fan-out, local work, aggregate work, and simulation-derived scaling analysis;
- label all results explicitly as **logical-shard simulation**.

---

## 6. Continuous Result Recording

Create and maintain the following experiment directory:

```text
experiments/c1/
  PLAN.md
  STATUS.md
  RESULTS.md
  runs/
    manifest.jsonl
    summary.csv
    per_query/
  figures/
  scripts/
  logs/
```

`PLAN.md` must contain the finalized experiment protocol.

`STATUS.md` must contain the current execution state and next action.

`RESULTS.md` must be updated continuously throughout execution.

Do not defer conclusions until all experiments finish.

After every completed configuration or configuration group, append a result record to `RESULTS.md` containing:

```text
Timestamp:
Git commit:
Run IDs:
Dataset:
Partition method:
Physical machines:
Logical shards:
Recall target:
Achieved recall:
Primary metrics:
Observed trend:
Preliminary conclusion:
C1 status: SUPPORTS / CONTRADICTS / INSUFFICIENT
Anomalies:
Required follow-up:
```

Update `STATUS.md` immediately after each experiment stage.

If any run contradicts the expected C1 mechanism, record the contradiction before launching additional experiments.

Never overwrite previous conclusions. Append corrections or updated interpretations with timestamps.

---

## 7. Common Instrumentation

Instrument every distributed query with the following counters:

```text
query_id
dataset
partition_method
logical_shards
physical_hosts
target_recall
achieved_recall
queried_shards
queried_shard_ids
routing_latency_us
local_search_latency_us_per_shard
distance_computations_per_shard
nodes_visited_per_shard
worker_cpu_time_us_per_shard
response_bytes_per_shard
end_to_end_latency_us
```

Compute and persist:

```text
mean_shards_per_query
median_shards_per_query
p95_shards_per_query
mean_distance_computations_per_query
mean_nodes_visited_per_query
mean_worker_cpu_time_per_query
mean_routing_latency
mean_latency
p50_latency
p95_latency
p99_latency
qps
aggregator_cpu_utilization
worker_cpu_utilization
network_bytes_per_query
```

Distance-computation count is the primary measure of graph-search work.

Nodes visited is the secondary algorithmic work metric.

CPU time is the primary implementation-level work metric.

Wall-clock local latency is secondary because logical shard size can change cache residency.

---

## 8. Recall Tuning Protocol

All reported configurations must achieve:

\[
Recall@10\geq0.90.
\]

Use the 1,000-query tuning set.

### Random

For each \(M\):

1. broadcast to all shards;
2. search a grid of local `efSearch` values;
3. select the smallest `efSearch` reaching mean Recall@10 \(\geq0.90\).

### K-Means

For each \(M\):

1. enumerate candidate shard fan-outs \(P\);
2. for every \(P\), search local `efSearch` values;
3. identify all \((P,efSearch)\) pairs reaching Recall@10 \(\geq0.90\);
4. select the feasible pair with the lowest mean aggregate distance computations on the tuning set;
5. if two configurations differ by less than 2% aggregate work, select the lower-fan-out configuration.

Evaluate the selected configuration once on the measurement set.

Do not tune parameters on measurement queries.

Record every candidate tuning result in `runs/manifest.jsonl`.

---

# Experiment E1: Physical Scale-Out

## 9. Purpose

Establish C1-a.

Measure actual end-to-end throughput on 1, 2, and 4 physical machines.

---

## 10. Matrix

Run:

```text
Datasets:
  SIFT1M
  glove-200-angular

Partition methods:
  Random
  K-Means

Physical shards:
  1
  2
  4

Repetitions:
  >= 3
```

The \(M=1\) Random and K-Means layouts are equivalent. A common unsharded baseline may be used.

---

## 11. Throughput Measurement

Use a closed-loop concurrency sweep.

For each configuration:

1. warm the index using at least 1,000 unreported queries;
2. start at concurrency 1;
3. increase concurrency geometrically;
4. continue until completed QPS improves by less than 5% across two consecutive concurrency increases or queueing causes p99 latency to exceed 5 times the minimum observed p99;
5. select the highest stable completed QPS before the saturation knee;
6. maintain the selected concurrency for a measurement interval long enough to process at least 10,000 queries;
7. repeat the complete measurement at least three times.

Record both offered and completed QPS.

No run with achieved recall below 0.90 is valid.

---

## 12. Bottleneck Validation

A physical scale-out run is valid for C1-a only if the aggregator is not the dominant bottleneck.

Record:

- aggregator CPU utilization;
- routing CPU time;
- aggregator queue depth;
- network utilization;
- worker CPU utilization.

Flag the configuration if any of the following occurs:

```text
aggregator CPU > 85%
persistent aggregator queue growth
aggregator routing time > 25% of end-to-end CPU time
network link utilization > 85%
```

If flagged, identify the limiting resource and rerun after removing the infrastructure bottleneck where possible.

Do not attribute aggregator or network saturation to the graph-search mechanism.

---

## 13. E1 Outputs

Generate:

```text
fig_c1_physical_scaleout.pdf
fig_c1_scaling_efficiency.pdf
```

Primary plot:

```text
x-axis: physical workers = 1, 2, 4
y-axis: QPS(M) / QPS(1)

curves:
  Ideal Linear
  Random
  K-Means
```

Secondary plot:

```text
x-axis: physical workers
y-axis: E(M) = QPS(M) / (M * QPS(1))
```

Report separate panels for SIFT1M and glove-200-angular.

---

# Experiment E2: Shard Fan-Out

## 14. Purpose

Establish C1-b independently of physical-machine count.

Use:

\[
M\in\{1,2,4,8,16,32\}.
\]

---

## 15. Oracle Minimum Fan-Out

For every measurement query:

1. obtain its exact global top-10 ground-truth vectors;
2. map each ground-truth vector to its logical shard;
3. count the number of ground-truth neighbors contributed by each shard;
4. sort shards by ground-truth contribution;
5. compute the minimum number of shards whose combined ground-truth neighbors satisfy Recall@10 \(\geq0.90\).

Define:

\[
P_{\mathrm{oracle}}(q,M).
\]

This is a partition-locality lower bound and must not use the actual routing algorithm.

Compute:

```text
mean
median
p95
CDF
normalized fan-out = P_oracle / M
```

for both Random and K-Means.

---

## 16. Actual K-Means Routing Fan-Out

For K-Means:

1. rank shards by centroid distance;
2. progressively increase \(P\);
3. measure global recall;
4. determine the minimum fixed \(P\) that reaches 0.90 mean recall on the tuning set;
5. apply that \(P\) to the measurement set.

Record actual per-query fan-out and achieved recall.

For Random, use \(P=M\) for actual serving fan-out unless a pre-existing non-Orion random routing baseline exists.

---

## 17. E2 Outputs

Generate:

```text
fig_c1_fanout_vs_shards.pdf
fig_c1_oracle_fanout_cdf.pdf
fig_c1_actual_fanout_cdf.pdf
```

Primary plot:

```text
x-axis: logical shard count = 1, 2, 4, 8, 16, 32
y-axis: mean shards searched/query

curves:
  Random actual
  Random oracle
  K-Means actual
  K-Means oracle
```

Also report:

\[
\frac{\bar P(M)}{M}.
\]

CDF plots should use representative configurations \(M=4,16,32\).

---

# Experiment E3: Local Graph-Search Scaling

## 18. Purpose

Establish C1-c without routing or distributed-execution effects.

---

## 19. Index Construction

For each dataset and partition method, use logical shard counts:

\[
M\in\{1,2,4,8,16,32\}.
\]

Measure every shard independently.

Do not execute competing logical shards concurrently during the primary E3 measurement.

Pin the local search process to a fixed CPU core set.

Use identical CPU allocation for every local shard.

---

## 20. Fixed-Quality Measurement

For every logical shard:

1. build the same local HNSW implementation used by E1;
2. select queries whose routed search includes that shard;
3. tune local `efSearch` using the global E1/E2 tuning protocol;
4. execute local searches in isolation;
5. record distance computations, nodes visited, CPU time, and wall-clock latency.

Aggregate across shards using the query-weighted access frequency observed in E2.

Compute:

\[
\bar D_{\mathrm{local}}(M)
\]

as the mean distance-computation count of a searched shard.

---

## 21. Cache-Regime Check

Record for every shard:

```text
vector storage size
graph index size
total local index memory
machine LLC size if available
resident-set size
```

If a shard's index transitions from larger-than-LLC to cache-resident as \(M\) increases, record the transition in `RESULTS.md`.

Do not use wall-clock latency alone to infer graph-complexity scaling across such a transition.

Distance computations and nodes visited remain the primary metrics.

---

## 22. Reference Trends

Normalize all curves to \(M=1\).

Plot measured local work against:

\[
1/M
\]

and a normalized:

\[
\log(N/M)
\]

reference curve.

The logarithmic curve is explanatory only. Do not fit the experiment to force agreement.

---

## 23. E3 Outputs

Generate:

```text
fig_c1_local_distance_computations.pdf
fig_c1_local_nodes_visited.pdf
fig_c1_local_cpu_time.pdf
```

Primary plot:

```text
x-axis: logical shard count
y-axis: normalized distance computations per searched shard

curves:
  measured Random
  measured K-Means
  ideal 1/M
  normalized log(N/M) reference
```

---

# Experiment E4: Aggregate Work Decomposition

## 24. Purpose

Establish C1-d and connect E2/E3 to E1.

---

## 25. Observed Aggregate Work

For every measurement query and every logical shard count:

\[
M\in\{1,2,4,8,16,32\},
\]

calculate:

\[
W_{\mathrm{obs}}(q,M)
=
\sum_{i\in R(q)}
D_i(q).
\]

Also compute:

\[
CPU_{\mathrm{obs}}(q,M)
=
\sum_{i\in R(q)}
CPU_i(q).
\]

Report means and p95 values.

---

## 26. Decomposition Model

From E2 and E3 calculate:

\[
W_{\mathrm{model}}(M)
=
\bar P(M)\cdot\bar D_{\mathrm{local}}(M).
\]

Use no fitted free parameter other than normalization to the \(M=1\) value.

Compute:

```text
Pearson correlation
Spearman correlation
mean relative error
```

between modeled and observed aggregate work across shard counts.

The primary criterion is trend agreement, not exact prediction.

---

## 27. Logical Scale-Out Projection

For \(M>4\), do not use colocated throughput as measured scale-out.

Optionally construct a resource-normalized projection:

\[
QPS_{\mathrm{work\ bound}}(M)
\propto
\frac{M}{W_{\mathrm{obs}}(M)}.
\]

Normalize:

\[
QPS_{\mathrm{work\ bound}}(1)=1.
\]

This curve represents an idealized compute-work bound under one physical worker per logical shard.

Label it:

```text
Projected from measured aggregate graph-search work
```

Never label it as measured cluster throughput.

Compare the projected trend for \(M=1,2,4\) against actual E1 throughput scaling. Use agreement at these physical points as a sanity check before reporting \(M>4\) projections.

---

## 28. E4 Outputs

Generate:

```text
fig_c1_aggregate_work.pdf
fig_c1_model_vs_observed.pdf
fig_c1_projected_scaling.pdf
```

Primary aggregate-work plot:

```text
x-axis: logical shard count
y-axis: normalized distance computations/query

curves:
  observed aggregate work
  fanout x local-work model
  ideal 1/M query-work reduction
```

---

# Experiment E5: Recall Sensitivity Check

## 29. Purpose

C1 is reported at 90% recall. Verify that the main mechanism is not caused by tuning instability exactly at the selected threshold.

This is a diagnostic experiment, not a second headline target.

For \(M\in\{4,16\}\), evaluate a small search-parameter neighborhood around the selected 90%-recall configuration.

Record:

```text
recall
fan-out
aggregate distance computations
local distance computations
```

Confirm that small changes around 90% recall do not qualitatively reverse the C1 trends.

Do not add 95% or 99% recall as primary experiment targets.

---

# 30. Execution Order

Execute strictly in the following order:

```text
Stage 0
  implement instrumentation
  create PLAN.md / STATUS.md / RESULTS.md
  verify deterministic seeds
  verify ground-truth recall computation

Stage 1
  smoke test SIFT1M
  M = 1, 2, 4
  Random and K-Means
  verify counters and recall

Stage 2
  run E3 local-search scaling
  SIFT1M
  M = 1, 2, 4, 8, 16, 32

Stage 3
  run E2 fan-out
  SIFT1M
  M = 1, 2, 4, 8, 16, 32

Stage 4
  run E1 physical scale-out
  SIFT1M
  M = 1, 2, 4

Stage 5
  run E4 decomposition
  SIFT1M

Stage 6
  inspect C1-a through C1-d
  update RESULTS.md with explicit support status

Stage 7
  repeat E2-E4 on glove-200-angular

Stage 8
  run physical E1 on glove-200-angular

Stage 9
  run E5 diagnostic sensitivity checks

Stage 10
  generate final figures
  generate final statistical tables
  update RESULTS.md with final C1 verdict
```

Do not launch the full GloVe matrix until the SIFT1M pipeline is validated.

---

# 31. Run Repetition and Statistical Treatment

Every E1 physical throughput configuration must have at least three independent repetitions.

Use five repetitions if coefficient of variation exceeds 5%.

For E2-E4, deterministic per-query measurements may use one index instance if deterministic construction is guaranteed. Otherwise build at least three seeds and report variation.

For all headline values report:

```text
mean
standard deviation
number of repetitions
```

Use 95% confidence intervals for physical throughput figures when practical.

---

# 32. Run Metadata

Every run must save:

```text
experiment_id
timestamp
git_commit
dataset
dataset_checksum
partition_method
partition_seed
graph_build_seed
physical_machine_count
logical_shard_count
logical_to_physical_mapping
vector_count
dimension
distance_metric
k
target_recall
achieved_recall
routing_fanout
efSearch
HNSW_M
HNSW_efConstruction
query_count
warmup_query_count
CPU_affinity
worker_threads
aggregator_threads
machine_hostname
index_memory_bytes
qps
mean_latency
p50_latency
p95_latency
p99_latency
mean_shards_per_query
p95_shards_per_query
distance_computations_per_query
nodes_visited_per_query
worker_cpu_time_per_query
routing_cpu_time_per_query
aggregator_cpu_utilization
worker_cpu_utilization
network_bytes_per_query
```

---

# 33. Failure and Contradiction Handling

Apply the following rules without exception.

If Recall@10 < 0.90:

```text
mark run INVALID_RECALL
exclude from performance comparison
retain raw data
retune parameters
```

If the aggregator is saturated:

```text
mark run AGGREGATOR_LIMITED
exclude from C1-a mechanism claim
retain raw data
```

If the network is saturated:

```text
mark run NETWORK_LIMITED
exclude from graph-work attribution
retain raw data
```

If physical throughput scales approximately linearly from 1 to 4 machines:

```text
record C1-a as CONTRADICTED or INSUFFICIENT
do not alter baseline implementation to manufacture sub-linearity
continue E2-E4 to determine whether the mechanism appears only at larger logical shard counts
```

If local work follows approximately \(1/M\):

```text
record C1-c as CONTRADICTED
remove the slowly-decreasing-local-work claim from the paper
```

If K-Means fan-out remains nearly constant and small:

```text
record C1-b for K-Means as CONTRADICTED
retain Random result separately
do not generalize the claim to spatial partitioning
```

If \(P\times D_{\mathrm{local}}\) does not track aggregate work:

```text
record C1-d as INSUFFICIENT
inspect routing imbalance, per-shard query difficulty, cache behavior, and heterogeneous shard sizes
do not present the decomposition as the root cause
```

---

# 34. Required Final Figures

Produce the following publication-ready figures:

```text
c1_fig1_physical_scaleout.pdf
c1_fig2_fanout_vs_logical_shards.pdf
c1_fig3_required_fanout_cdf.pdf
c1_fig4_local_search_work.pdf
c1_fig5_aggregate_work_decomposition.pdf
c1_fig6_projected_logical_scaling.pdf
c1_combined_motivation.pdf
```

Recommended combined motivation figure:

```text
(a) Physical throughput scaling, M = 1, 2, 4
(b) Shard fan-out, M = 1..32 logical shards
(c) Local distance computations/shard, M = 1..32
(d) Observed aggregate work vs fanout x local-work model
```

Use separate rows or panels for SIFT1M and glove-200-angular if both remain legible.

---

# 35. Required Final Tables

Generate:

```text
c1_physical_scaleout.csv
c1_fanout_summary.csv
c1_local_work_summary.csv
c1_aggregate_work_summary.csv
c1_model_accuracy.csv
```

Also generate one compact LaTeX table containing:

```text
Dataset
Partition
Physical workers
Recall
QPS
Scaling efficiency
Mean shard fan-out
Aggregate distance computations/query
```

for physical \(M=1,2,4\).

---

# 36. Final C1 Decision

At completion, `RESULTS.md` must contain one explicit status for each subclaim:

```text
C1-a Physical sub-linear scaling:
  SUPPORTED / CONTRADICTED / INSUFFICIENT

C1-b High logical shard fan-out:
  SUPPORTED / CONTRADICTED / INSUFFICIENT

C1-c Slow decrease in local graph-search work:
  SUPPORTED / CONTRADICTED / INSUFFICIENT

C1-d Fan-out x local-work decomposition:
  SUPPORTED / CONTRADICTED / INSUFFICIENT
```

The complete C1 claim may be marked `SUPPORTED` only if C1-a, C1-b, C1-c, and C1-d are all supported.

For \(M>4\), the final text must distinguish:

```text
measured physical scale-out
```

from:

```text
logical-shard mechanism measurements
```

and:

```text
work-based scale-out projection
```

No result from an \(M>4\) colocated configuration may be described as measured \(M\)-machine performance.

---

# 37. Paper-Ready Claim Template

Populate only after all experiments are complete:

> At 90% Recall@10, conventional scatter-gather HNSW exhibits sub-linear throughput scaling from one to four physical workers on both SIFT1M and glove-200-angular. Increasing the logical shard count does not proportionally reduce per-shard graph-search work, while queries continue to access multiple shards. Consequently, aggregate graph-search work per query remains high as the index is further partitioned. A decomposition based on shard fan-out and per-shard distance computations tracks the observed aggregate work, identifying their compounding effect as the source of the scale-out bottleneck.

Replace or weaken any sentence whose corresponding subclaim is not supported by the recorded measurements.
