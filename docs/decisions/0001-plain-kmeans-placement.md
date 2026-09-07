# 0001 — Plain k-means, single assignment, as Orion's primary layout

Status: accepted, 2026-09-07
Supersedes: the topology-aware placement pipeline as the *default* arm

## Decision

Orion's primary layout becomes plain k-means with single assignment: one shard
per point, no replication, no upper graph, no attachments, no topology
refinement, no fission, no capacity-constrained balancing.

This configuration already exists and needs no new code. It is
`--routing-mode kmeans_simple_nprobe` with the default
`--simple-kmeans-multi-assign-alpha 1.0`, which the help text describes as
preserving "the existing single-assignment collection path". Its online side is
nprobe centroid routing, already implemented.

The navigation-derived pipeline (`faithful_original_rest`) is **retained as the
ablation arm**, not deleted. It is the comparison Orion's claims are measured
against, and `agent.md` requires negative findings be preserved.

## Why

The offline analysis in `analysis/FINDINGS.md` removed the systems argument for
capacity-constrained balancing.

**Equal shard sizes do not produce equal query load.** Throughput is set by the
busiest shard, so the quantity that matters is how many queries land on each
shard, not how many points do. Under optimal routing at P = 32:

| layout | size skew | query load skew |
|---|---|---|
| plain k-means | 1.48 (sift) / 2.05 (glove) | 6.77 / 5.72 |
| balanced k-means | 1.00 / 1.00 | 6.05 / 4.55 |
| orion | 1.01 / 1.01 | 6.11 / 5.37 |
| orion, no replication | 1.01 / 1.01 | 5.73 / 4.87 |

Driving size skew from 2.05 to 1.00 moves load skew by about 20%, nowhere near
proportionally. The load skew is a property of where queries cluster in dense
data regions, which placement does not control.

**The throughput ceiling is nearly the same for every partitioner.** The busiest
shard's load is proportional to (shards probed) x (load skew), so that product
is inverse normalized throughput. Under optimal routing at P = 32 it spans
17.2–20.3 on SIFT and 21.5–24.4 on GloVe across all four layouts — a 13%–18%
spread. Placement barely moves the ceiling, so the simplest placement wins.

**Size imbalance is cheaper on a graph index than it looks.** At fixed `ef` an
HNSW search costs roughly O(ef * M * log n) distance computations, so an
oversized shard barely raises per-query cost. It costs memory provisioning, not
query latency. An earlier draft of `FINDINGS.md` claimed a 2.7x shard "sets the
tail latency"; that reasoning holds for scan-based IVF, not for HNSW, and has
been corrected.

**Plain k-means has the best locality of all four layouts**, at every P on both
datasets, on the router-independent oracle floor.

## What this gives up

At *matched* balance and *matched* storage, Orion's layout beats balanced k-means
on the oracle floor in 7 of 8 configurations by 8%–18%. That advantage is real
and router-independent. It is being given up because it only exists once you
have already decided to pay for equal shard sizes, and the load-skew data says
equal shard sizes do not buy throughput. Plain k-means beats Orion's layout
outright on locality.

Also given up: `analysis/FINDINGS.md` Finding 1, which credited the partitioner,
no longer drives the design. It remains valid as a measurement.

## What this buys

- The entire offline placement pipeline disappears from the primary path, and
  with it the `agent.md` constraint that offline placement and online routing
  must navigate the *same* graph. With no upper graph there is nothing to keep
  identical, which removes the project's hardest correctness requirement.
- CCNB's convergence failures disappear. On SIFT at P = 16 the default 8
  balancing passes left two shards under the lower load bound and the build
  aborted; 32 passes were needed.
- Memory provisioning becomes the one honest cost of the choice: every node must
  be sized for the largest shard, which is 1.48x the mean on SIFT and 2.05x on
  GloVe at P = 32.

## Open items

- **Replication deserves re-examination as a load-spreading device, not a
  locality device.** Multi-assignment lets a hot point's queries be split across
  shards, which targets the ~6x load skew directly. That is a better
  justification than the locality one this analysis rejected, and it is untested.
  Do not treat this decision as evidence against replication for that purpose.
- The largest measured headroom is in routing, not placement: the gap between a
  fixed-budget centroid router and an ideal adaptive one is roughly 2x–6x on
  every layout, growing with P and dataset hardness. That is where the next work
  should go.
- If the memory cost of size imbalance proves unacceptable, `balanced_kmeans`
  (k-means centroids plus a `ceil(N/P)` quota pass) reaches balance 1.00 while
  keeping this decision's simplicity, at the cost of worse locality than plain
  k-means. It is implemented in `analysis/fanout_headroom.py` for analysis but
  not in the serving path.
- The layouts behind this analysis were built with the `hnswlib` upper graph,
  which `agent.md` classifies as an ablation rather than canonical Orion. A
  canonical-path layout could differ. This matters for the magnitude of the
  advantage being given up, not for the load-skew result that drove the decision.
