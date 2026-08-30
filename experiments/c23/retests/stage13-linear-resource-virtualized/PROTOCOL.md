# C2-C3 Stage 13: four-host virtualized linear-resource retest

This retest is a new evidence line. It does not replace or rewrite the accepted
`experiments/c23` results. The motivating alternative explanation is that the
old E3/E4 local microbenchmark searched one logical shard at a time and did not
prove that service compute grew with logical shard count.

## Questions

1. With one independently isolated search worker per logical shard, does Full
   Orion preserve traversal-relevant topology or local HNSW navigability better
   than full-data K-Means?
2. With total server compute increasing exactly in proportion to `M`, does Full
   Orion require fewer HNSW shard searches than K-Means to reach global
   Recall@10 >= 0.90?
3. Were any changed latency or throughput observations caused by a larger
   compute pool rather than by better placement or local graph navigability?

The primary C2/C3 metrics remain structural and work based. Wall-clock latency
and throughput are secondary resource-sensitive measurements and cannot, by
themselves, establish either claim.

## Scale and virtualization contract

- Physical hosts: four AMD EPYC 7302P machines, each exposing 16 physical cores
  and 32 SMT CPUs.
- Logical shard counts: `M={1,2,4,8,16,32}`.
- Virtualization unit: one standalone Qdrant container per logical shard. Each
  container owns exactly one non-empty HNSW shard and does not share a search
  thread pool with another logical shard.
- Placement: logical shard `s` runs on host `s mod 4`; its host-local slot is
  `floor(s/4)`.
- Server CPU pool: physical cores 0-7 on every host. Slot `j` is bound to both
  SMT CPUs of physical core `j`, encoded as `j,j+16`.
- Therefore every logical shard receives exactly one physical-core equivalent,
  total server capacity is exactly `M` physical cores, and `M=32` consumes the
  complete declared 32-core server pool. The capacity ratios relative to M=32
  are exactly `1/32, 2/32, 4/32, 8/32, 16/32, 32/32`.
- The benchmark client is pinned to physical cores 8-15 (SMT CPUs
  `8-15,24-31`) on the controller and is outside the server-compute budget.
- Every shard container has the same 8 GiB hard memory and swap cap. The
  declared experiment memory pool therefore also grows linearly as `8*M GiB`.
  This is a capacity control, not a claim that the immutable dataset footprint
  itself scales with M.
- Existing long-lived Qdrant containers are stopped, not deleted, during a
  formal run and must be restored with the same identity and running state.

## Runtime gates

Every formal configuration must record and pass all of the following before
queries are accepted:

1. Exactly M experiment containers exist and are healthy.
2. Every container image ID equals the pinned image on all four hosts.
3. Every container cpuset equals the two SMT CPUs of its assigned physical
   core; cpusets are pairwise disjoint on each host.
4. Every container reports the expected runtime `Cpus_allowed_list` and a
   8 GiB `memory.max`/Docker memory cap with no additional swap allowance.
5. The placement table contains every logical shard exactly once and follows
   `s mod 4`.
6. No non-experiment container can execute on the declared server or client
   cpusets during measurement.
7. The collection contains exactly the assigned points, is green, has one
   non-empty indexed HNSW graph, and uses the pinned graph-build parameters.
8. cgroup CPU and memory counters are captured before and after every phase;
   OOM, process exit, missing hardware counters, or a resource-gate mismatch
   invalidates the configuration.

## Experimental matrix

Datasets and query split remain unchanged:

- SIFT1M / Euclidean;
- glove-200-angular / cosine;
- queries 0-999 are tuning only;
- queries 1000-9999 are the paired measurement set.

Primary partition methods are full-data K-Means and Full Orion. Random and
Orion-NoRefinement remain required controls so that the retest preserves the
original causal interpretation rather than turning into an isolated two-line
benchmark. `M=1` is one common unpartitioned control and is not duplicated by
method.

Partition assignments, dataset checksums, reference traces, HNSW M=32,
efConstruction=200, and graph seed 20260821 remain pinned. Reusing an accepted
assignment is allowed because the intervention is the execution/resource
model, not partition construction. Every reused artifact must be checksum
bound in the new manifest.

## E3 and E4 measurements

- Calibrate one common `efSearch` per dataset on the virtualized unpartitioned
  M=1 control. Choose the smallest value reaching tuning Recall@10 >= 0.90.
- Use that same `efSearch` for every partition method and M in primary E4.
- Query every logical shard through its own container and collect returned IDs,
  distance computations, graph nodes visited, worker CPU time, wall time, and
  response bytes.
- `P_exact`, `P_HNSW`, `Delta_P`, and `W90` retain the original definitions and
  failure sentinel `M+1`.
- Oracle ordering is descending recovered ground-truth contribution, then
  ascending distance computations, then shard ID.
- Record full-fan-out concurrent wall latency and closed-loop QPS as secondary
  evidence. They must not replace fan-out or work in the C3 verdict.
- Run paired 10,000-replicate query bootstrap for Orion minus K-Means at every
  M. Run three construction seeds at M=32 and retain the balance control and
  Orion-NoRefinement ablation.
- The already-constructed accepted M=32 seed artifacts may be reused because
  partition construction is outside the execution-resource intervention, but
  Stage 13 must independently re-audit all 12 artifacts per dataset, their
  checksums, seed coverage, metric-table checksums, and linkage of the primary
  seed to the exact partition used in the virtualized matrix.

## Verdict rules

- C2 requires a consistent topology -> local-navigability chain. Lower edge
  cut without better fixed-work local recovery is insufficient.
- C3 requires lower mean `P_HNSW` with a paired confidence interval supporting
  Orion, without a material `W90` regression or severe imbalance confound.
- Resource-sensitive throughput may support a systems result, but it cannot
  rescue C2 or C3 when their primary structural/work metrics are contradicted.
- Contradictory findings and exceptions must be retained. No method-specific
  retuning is allowed in the primary matrix.

## Evidence layout

- Raw, resumable output:
  `/proj/intelisys-PG0/exp/orion-distributed/c23-linear-20260824-v1/`
- Small manifests, tables, figures, logs, status, and final audit:
  `experiments/c23/retests/stage13-linear-resource-virtualized/`
- Reproducible final aggregation, raw-array audit, bootstrap, plots, and
  data-driven verdict logic:
  `experiments/c23/scripts/c23_linear_synthesis.py`
- The old accepted evidence remains under `experiments/c23/` and
  `/proj/intelisys-PG0/exp/orion-distributed/c23-20260821-v2/`.

## Reproduction commands

Formal matrix (resumable only at fully completed configuration boundaries):

```bash
taskset -c 8-15,24-31 /users/dry/orion-distributed/venv/bin/python \
  experiments/c23/scripts/c23_linear_matrix.py
```

Final audit and synthesis after the registry contains exactly 42 PASS rows:

```bash
/users/dry/orion-distributed/venv/bin/python \
  experiments/c23/scripts/c23_linear_synthesis.py \
  --variance-root \
    /proj/intelisys-PG0/exp/orion-distributed/c23-20260821-v2/analysis \
  --output-dir \
    experiments/c23/retests/stage13-linear-resource-virtualized/evidence-v1
```
