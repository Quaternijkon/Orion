# Legacy experiment evidence (conclusions retired, data preserved)

Archived: 2026-09-04

This directory holds the previous measurement program (`experiments/`, `results/`).
Its **conclusions are retired**; its **data is preserved** deliberately, per the
`agent.md` rule to keep contradictory and negative findings intact rather than
delete them.

Nothing here may be cited as a current result. It exists for two reasons: to
record which paths have already been shown not to work, and because the current
research direction was derived from these measurements.

## Why the program was retired

The original C1/C2/C3 narrative was that topology-oblivious partitioning cuts
search-relevant graph structure, degrading local navigability and forcing high
shard fan-out, which causes sub-linear scale-out. **Its own experiments
contradicted the causal chain.**

Recorded final verdicts:

| Sub-claim | Statement | Verdict |
|---|---|---|
| C1-a | Physical throughput scaling (1->4 hosts) is sub-linear | SUPPORTED |
| C1-b | Logical shard fan-out stays high | CONTRADICTED (generalized) |
| C1-c | Local graph-search work decreases slowly | CONTRADICTED |
| C1-d | fan-out x local-work explains aggregate cost | INSUFFICIENT |
| C1 | complete claim | INSUFFICIENT |
| C2 | Topology-oblivious partitioning disrupts graph-search topology | CONTRADICTED |
| C3 | Topology-aware partitioning reduces required fan-out | CONTRADICTED |

## Facts worth carrying forward

These are the load-bearing observations extracted from the archived data. They
motivate the new program and should not have to be rediscovered.

1. **The testbed could not attribute performance to the algorithm.** Across the
   physical scale-out runs, worker CPU utilization was only 20-37% while the
   aggregator sat at 67-81% of its reserved cores (SIFT Random M=4: worker
   20.87%, aggregator 81.06%). The protocol's flag threshold was 85%, so these
   runs were never flagged, yet the workers were plainly not the bottleneck.
   This is why C1-d attribution was INSUFFICIENT, and it invalidates any
   algorithmic reading of the observed sub-linearity.
   *Consequence: platform validity must be established before any performance
   claim, with an explicit worker-saturation gate.*

2. **Orion's partition is measurably worse than plain k-means**, on both
   datasets and at every M >= 4, in traversal-weighted edge cut, mean fan-out,
   and fixed-recall work. At M=32 on GloVe: k-means cut 0.456 / fan-out 3.31 /
   W90 29512 versus Orion 0.509 / 4.93 / 40303 (+37% work). Orion's
   per-query 90%-recall reachability was also consistently lower.

3. **Yet deployed Orion was faster end-to-end** at high shard counts. GloVe,
   four hosts, matched 0.90 recall: at 32 shards Orion 1871 QPS versus HashAll
   1580 and Simple KMeans 1506 (+18.4% over HashAll). At 4 real shards HashAll
   won at every recall level, and at 0.95 recall Orion's advantage shrank to
   +1.3%.
   *Consequence: whatever advantage exists does not come from partition
   quality. (2) and (3) together are the central puzzle the new program must
   resolve.*

4. **Required fan-out is highly heterogeneous.** Median required fan-out was 2
   at every shard count on both datasets, with a very wide spread (GloVe M=32
   std 8.60), and 3-18% of queries failed to reach 90% recall at the common
   efSearch regardless of how many shards were searched. Meanwhile deployed
   baselines must fix a conservative fan-out: Simple KMeans `nprobe` grew
   4 -> 6 -> 10 -> 16 as shards grew 4 -> 32, against a median need of 2.
   *Consequence: the leading hypothesis for the new program is that the win
   comes from per-query adaptivity (routing plus per-shard budget), not from
   the partitioning objective. This is still a hypothesis: the archived
   fan-out numbers were measured at a common efSearch, not with the deployed
   adaptive router.*

5. **The traversal-weighted edge cut does not predict cost.** `TWCut` (computed
   in `experiments/c23/scripts/c23_e2.py` by weighting cut edges with observed
   query traversal frequencies) correlated near zero or negatively with
   fixed-recall work, and `Delta_P` versus `TWCut` was weakly negative. Cut
   reduction via topology refinement did not translate into lower downstream
   fan-out.
   *Consequence: minimizing a navigation-weighted cut is a natural surrogate
   that has been tested and failed empirically. Do not reintroduce it as a
   design principle without new evidence.*

## Methodological lessons

- Very large protocols were committed before the phenomenon was established,
  so substantial compute went into validating a causal chain that the data
  then contradicted. Prefer short protocols and the smallest experiment that
  can discriminate between hypotheses.
- Performance claims were partly built on offline proxies (`TWCut`, `P_HNSW` at
  a common efSearch) that do not transfer to the deployed system. Performance
  claims must come from the deployed system.
- No cross-ablation separated the partitioner from the router, so the source of
  the end-to-end win was never isolated. Every claim needs an ablation capable
  of falsifying it.
