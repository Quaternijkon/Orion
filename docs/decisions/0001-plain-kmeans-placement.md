# 0001 — Plain k-means, single assignment, as Orion's primary layout

Status: **reopened, 2026-09-07** (accepted the same day, then reopened after a
correctness error in the "what this buys" reasoning was found; see the
Correction section)
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

## Correction (2026-09-07)

An earlier version of this record described the rollback as removing "the fancy
parts of Orion" and claimed that dropping the upper graph is a cost-free
simplification. Both are wrong and are corrected here.

**Orion's placement is not k-means with extra steps; it is a different mechanism
that k-means only seeds.** The pipeline is: (1) k-means gives an *initial*
shard label to the sampled upper-tier L1 nodes only; (2) `converge_l1_topology`
runs self-search refinement — label propagation on the navigation graph, where
each L1 node moves to the shard holding the majority of its graph neighbours;
(3) `assign_points_by_l1_vote` places each L0 point on the shard its
graph-navigated L1 nodes vote for, not its nearest centroid. Steps 2 and 3 are
graph partitioning and have no counterpart in plain k-means. So "switch to plain
k-means" replaces the whole placement mechanism, it does not trim it.

**The navigation graph is also the online adaptive-routing substrate.** `router.rs`
derives per-shard entry points and a dynamic `ef` (`dynamic_ef_base/factor`,
`entry_point_capacity_hint`) by navigating the same upper graph. `kmeans_simple_nprobe`
routes with a fixed nprobe and is not adaptive. The offline–online symmetry
constraint in `agent.md` exists *because* one graph does double duty; dropping
the graph does not "remove a hard requirement for free", it removes the
capability the requirement protects — and that capability is adaptive routing,
which Finding 3 identifies as the single largest source of headroom (2x–6x on
every layout). This cost was not counted in the first draft.

The decision's *evidence* is unaffected: the measured `orion` layouts had
self-search refinement enabled, so Finding 4's load-skew result is about the real
refined layout, not a stripped one. What changed is the cost accounting, which is
why the status is reopened rather than the decision reversed.

**Update: the reopened question has now been measured.** Finding 5 in
`analysis/FINDINGS.md` reconstructs Orion's real navigation router and compares it
against the centroid router on the same layout, at R = 0.95:

| dataset | P | centroid fanout / skew | orion fanout / skew |
|---|---|---|---|
| sift  |  8 | 2.84 / 1.58 | 2.32 / 1.58 |
| sift  | 32 | 5.38 / 1.76 | 4.33 / 1.52 |
| glove |  8 | 4.10 / 1.13 | 3.74 / 1.21 |
| glove | 32 | 9.99 / 1.89 | 9.60 / 1.71 |

The navigation graph does have routing value the centroid router lacks: it
touches 4%–27% fewer shards at equal recall, without losing balance. But the gain
is modest, largest on easy data and small P, and it does **not** unlock the big
oracle headroom on GloVe P = 32 (oracle 3.25 vs both real routers ~9.6). So the
routing cost of dropping the graph is real but bounded — single-digit to ~20% of
fan-out — not the full 2x–6x that Finding 3's abstract oracle suggested.

**Update 2: the end-to-end baseline comparison now favours the rollback on the
primary metric.** FINDINGS' Headline section measures the full system against a
plain k-means + centroid baseline on realizable fan-out (the first-order
throughput driver) at routing recall 0.95:

| config | baseline | Orion | Orion vs base |
|---|---|---|---|
| sift  P32 | 2.81 | 4.33 | +54% |
| glove P32 | 7.80 | 9.60 | +23% |

The baseline wins on fan-out in all four configs by 23%–54%. Orion's placement
raises fan-out (locality traded for balance); its navigation router recovers only
part. Orion wins end to end only once load skew is folded in, and only on GloVe,
where plain k-means is badly load-skewed (2.57) and Orion's balancing tames it
(1.71). On evenly-loaded SIFT the baseline wins outright. So the rollback is right
on the primary throughput driver for even workloads; the one thing it forfeits is
Orion's balance advantage on skewed workloads.

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

Two distinct things, one measured and one that the first draft missed:

1. **Placement quality.** At *matched* balance and *matched* storage, Orion's
   self-search-refined layout beats balanced k-means on the oracle floor in 7 of
   8 configurations by 8%–18%. That advantage is real and router-independent. It
   is being given up because it only exists once you have already decided to pay
   for equal shard sizes, and the load-skew data says equal shard sizes do not
   buy throughput. Plain k-means beats Orion's layout outright on locality.
2. **The adaptive-routing substrate.** The upper graph that self-search uses is
   the same one `router.rs` navigates to set per-shard entry points and a dynamic
   `ef`. Dropping it forces the online side onto fixed nprobe. Since Finding 3
   puts the largest headroom (2x–6x) in adaptive routing, this is potentially the
   more expensive of the two, and it is why the decision is reopened. If the
   routing headroom is to be pursued on a k-means layout, an adaptive-nprobe
   mechanism has to be built to replace what the graph provided.

Also given up: `analysis/FINDINGS.md` Finding 1, which credited the partitioner,
no longer drives the design. It remains valid as a measurement.

## What this buys

- The offline placement pipeline leaves the primary path: no upper graph, no
  attachments, no self-search refinement, no fission, no CCNB. Whether this is a
  net win depends on the routing question above — it is only free if the adaptive
  routing it also removes turns out not to be worth capturing. The first draft
  called this the removal of "the project's hardest correctness requirement" (the
  offline–online same-graph constraint); that constraint disappears only because
  the graph disappears, so it is not an independent benefit.
- CCNB's convergence failures disappear. On SIFT at P = 16 the default 8
  balancing passes left two shards under the lower load bound and the build
  aborted; 32 passes were needed.
- Memory provisioning becomes the one honest *placement* cost of the choice:
  every node must be sized for the largest shard, which is 1.48x the mean on SIFT
  and 2.05x on GloVe at P = 32.

## Open items

- **Replication deserves re-examination as a load-spreading device, not a
  locality device.** Multi-assignment lets a hot point's queries be split across
  shards, which targets the ~6x load skew directly. That is a better
  justification than the locality one this analysis rejected, and it is untested.
  Do not treat this decision as evidence against replication for that purpose.
- The largest *abstract* headroom is in routing, but Finding 5 shows neither a
  centroid nor Orion's navigation router captures most of it on hard, high-P
  workloads (both need ~9.6 shards on GloVe P = 32 where the oracle needs 3.25).
  The next routing work should target that gap with a better signal than
  nearest-L1 membership, not assume graph navigation already closes it.
- Navigation routing's measured edge over centroid routing (4%–27% fewer shards
  at equal recall, Finding 5) is the concrete thing lost by dropping the graph.
  Weigh it against the whole offline pipeline's cost; on easy data / small P it
  is worth more, on hard data / large P almost nothing.
- If the memory cost of size imbalance proves unacceptable, `balanced_kmeans`
  (k-means centroids plus a `ceil(N/P)` quota pass) reaches balance 1.00 while
  keeping this decision's simplicity, at the cost of worse locality than plain
  k-means. It is implemented in `analysis/fanout_headroom.py` for analysis but
  not in the serving path.
- The layouts behind this analysis were built with the `hnswlib` upper graph,
  which `agent.md` classifies as an ablation rather than canonical Orion. A
  canonical-path layout could differ. This matters for the magnitude of the
  advantage being given up, not for the load-skew result that drove the decision.
