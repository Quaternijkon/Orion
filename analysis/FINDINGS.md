# Offline fan-out headroom: Orion's layout versus k-means

This is an offline, cluster-free analysis. It uses ground-truth neighbours and a
materialized shard layout to compute how many shards a query *must* touch to
reach a recall target. It replaces the retired TWCut metric, which the legacy
program showed does not predict cost (see `legacy/LEGACY.md`).

## Headline: Orion vs the plain baseline, end to end

Before attributing anything to Orion's individual techniques, the first question
is whether the whole system beats a basic baseline. It does not, on the metric
that drives throughput.

The primary throughput driver is **fan-out** — the mean number of shards a query
touches at a fixed recall. With one shard per core, per-query compute scales as
`fan-out × ef × log(shard_size)`, so at fixed recall total work is essentially
proportional to fan-out. Load skew is a second, multiplicative correction (the
busiest shard caps throughput): `throughput ∝ P / (fan-out × load_skew × log(shard_size))`.
So fan-out leads and load skew modulates.

Realizable mean fan-out at routing-recall 0.95, full system vs baseline:

| config | baseline (k-means + centroid) | Orion (orion layout + navigation) | Orion vs base | naive all-shards |
|---|---|---|---|---|
| sift  P8  | **1.59** | 2.32 | +46% | 8 |
| sift  P32 | **2.81** | 4.33 | +54% | 32 |
| glove P8  | **2.90** | 3.74 | +29% | 8 |
| glove P32 | **7.80** | 9.60 | +23% | 32 |

**The plain baseline wins on fan-out in every configuration, by 23%–54%**, and
the Orion layout is even carrying 7%–18% replication that can only lower its
fan-out. So on the first-order throughput driver, the entire Orion apparatus is a
net loss.

Decomposing the fan-out gap into the two techniques (realizable adaptive fan-out):

| config | baseline | + Orion layout (keep centroid) | + navigation router | net vs base |
|---|---|---|---|---|
| sift  P32 | 2.81 | 5.38 (**+2.57**) | 4.33 (−1.05) | **+1.53 worse** |
| glove P32 | 7.80 | 9.99 (**+2.20**) | 9.60 (−0.39) | **+1.80 worse** |
| sift  P8  | 1.59 | 2.84 (+1.26) | 2.32 (−0.52) | +0.73 worse |
| glove P8  | 2.90 | 4.10 (+1.20) | 3.74 (−0.36) | +0.84 worse |

Orion's **placement raises fan-out sharply** by trading locality for balance
(+1.2 to +2.6 shards); its **navigation router recovers only part** of that
(−0.4 to −1.05) and never enough to break even. Finding 5's "navigation beats
centroid" is real but is a partial rescue of a deficit the layout created.

Only when the load-skew term is included does Orion win anywhere. Throughput
proxy `W = fan-out × load_skew` (lower is better):

| config | baseline W | Orion W | winner |
|---|---|---|---|
| sift  P32 | 4.50 | 6.56 | baseline, by 31% |
| glove P32 | 20.07 | **16.44** | Orion, by 18% |
| glove P8  | 5.51 | **4.51** | Orion, by 18% |
| sift  P8  | 3.07 | 3.67 | baseline, by 16% |

Orion only comes out ahead on GloVe, and only because plain k-means has bad load
skew there (2.57, driven by its 2.05x size imbalance) which Orion's balancing
tames to 1.71. On SIFT, where the baseline is already evenly loaded, Orion's
fan-out penalty is unmasked and it loses. **Orion's end-to-end value is confined
to skewed workloads and comes entirely from balance, not from lower work.**

This is measured at routing recall — coverage of the true neighbours by the
probed shard set, assuming perfect within-shard search — so it isolates routing
and placement from within-shard HNSW approximation. Fidelity caveat as in
Finding 5: hnswlib upper graph and the Python router, not Rust `router.rs`.

The sections below decompose the pieces in more detail; this headline is the
answer to "how does Orion compare to a basic baseline first".

Everything here is a property of the layout and the ground truth. No search, no
QPS, no HNSW approximation. Numbers are therefore reproducible and exact, and
they are *bounds* on what any deployed router can achieve, not predictions of
measured throughput.

## What is computed

For query `q` with true neighbours `NN_k(q)`, each shard `s` gets a bitmask of
which of those neighbours it holds. Replication is handled natively: a neighbour
present on several shards is counted once.

Two shard orderings are evaluated:

- **oracle** — greedy maximum coverage, i.e. repeatedly take the shard adding the
  most not-yet-covered neighbours. This is the best ordering any router could
  produce, so it is a **lower bound on the cost of every possible router**.
- **centroid** — shards ranked by distance from the query to the shard centroid.
  This models a conventional k-means router.

Two budgeting policies:

- **fixed** — the smallest per-query constant number of shards reaching mean
  recall `R`.
- **adaptive** — the smallest *mean* number of shards reaching mean recall `R`
  when the budget may vary per query. Computed as a Lagrangian lower bound, so
  it is valid even when marginal gains are not monotone.

Headroom ratios: `H_ord = centroid_fixed / oracle_fixed`,
`H_adp = centroid_fixed / centroid_adaptive`, `H_tot = centroid_fixed / oracle_adaptive`.

Four layouts are compared. `kmeans` is plain k-means, which is free to produce
unequal shards. `balanced_kmeans` is k-means centroids plus a hard `ceil(N/P)`
quota, so it is balanced by construction. `orion` is Orion's navigation-derived
layout. `orion_norep` is the same with multi-assignment disabled, which isolates
the effect of Orion's replication.

## Finding 1: Balance and locality trade off, and the comparison hinges on it

Plain k-means has the best locality of any layout tested — but only because it is
allowed to be lopsided. Its max/mean shard size reaches 1.50 on SIFT and 2.72 on
GloVe. Comparing against it alone is what made an earlier version of this
document conclude that Orion's partitioner was dominated. That conclusion was an
artifact of the unfair baseline and is **withdrawn**.

An earlier draft also argued that a shard carrying 2.7x the mean "sets the tail
latency and caps useful scale-out". That is wrong for a graph index and is
**withdrawn** too: at fixed `ef` an HNSW search costs roughly
O(ef * M * log n) distance computations, so an oversized shard barely raises
per-query cost. Size imbalance costs memory provisioning, not query latency. The
quantity that does cap throughput is query load skew, which Finding 4 measures
and which turns out not to follow size skew at all.

Once the baseline is forced to the same balance Orion achieves, the ranking
inverts. Router-independent oracle adaptive floor at R = 0.95, in shards per
query (lower is better):

| dataset | P | kmeans (bal 1.07–2.72) | balanced_kmeans (bal 1.00) | orion (bal 1.01) | orion_norep (bal 1.01) |
|---|---|---|---|---|---|
| sift  |  4 | 1.04 | 1.24 | **1.11** | 1.13 |
| sift  |  8 | 1.21 | 1.71 | **1.50** | 1.57 |
| sift  | 16 | 1.47 | 2.02 | **1.74** | 1.67 |
| sift  | 32 | 1.81 | **2.24** | 2.14 | 2.30 |
| glove |  4 | 1.54 | 1.94 | **1.62** | 1.67 |
| glove |  8 | 1.72 | 2.55 | **2.17** | 2.29 |
| glove | 16 | 2.10 | 2.67 | **2.30** | 2.45 |
| glove | 32 | 2.41 | 3.12 | **2.75** | 2.96 |

**At matched balance Orion's navigation-derived layout beats the geometric one in
all 8 configurations**, by 8% to 18% on the oracle floor. Because the oracle
ordering lower-bounds every possible router, this is a property of where the
points sit, not something routing can create or destroy.

This is the first evidence in the project that the topology-aware partitioner
earns its keep, and it is the opposite of the legacy C3 verdict — which compared
against unbalanced k-means and therefore measured the balance/locality trade-off
rather than partition quality.

## Finding 2: Part of Orion's edge is bought with replication, and it thins at large P

Orion spends 3.8%–18.1% extra storage on replicated copies, rising with P, while
`balanced_kmeans` spends none. Disabling multi-assignment isolates this. On SIFT:

| P | orion (exp) | orion_norep (exp 1.00) | balanced_kmeans (exp 1.00) |
|---|---|---|---|
|  4 | 1.11 (1.038) | 1.13 | 1.24 |
|  8 | 1.50 (1.070) | 1.57 | 1.71 |
| 16 | 1.74 (1.077) | **1.67** | 2.02 |
| 32 | 2.14 (1.124) | 2.30 | **2.24** |

At P = 4, 8 and 16 Orion wins at equal storage, so its advantage there is genuine
partitioning quality rather than paid-for redundancy. At P = 16 removing
replication even helps slightly, which suggests the extra copies are not always
placed usefully. At P = 32 the picture reverses: without replication Orion falls
marginally behind `balanced_kmeans` (2.30 versus 2.24), so at that scale Orion's
competitiveness depends on its 12.4% storage premium. Whether that trade is worth
it is a deployment question, not one this analysis settles.

On GloVe the reversal does not happen: `orion_norep` beats `balanced_kmeans` at
every P (1.67/2.29/2.45/2.96 against 1.94/2.55/2.67/3.12). So the SIFT P = 32
result is the single exception across 8 matched-storage comparisons, and it is a
0.06-shard margin that one seed could plausibly move.

## Finding 3: The largest headroom is in routing, on every layout

`H_tot` — the ratio between what a conventional fixed-budget centroid router
spends and what an ideal adaptive router would spend — runs roughly 2x to 6x, and
grows with both P and dataset hardness. At GloVe P = 32 it is 4.97x for k-means,
5.82x for Orion and 6.16x/3.12x = 5.45x for balanced k-means.

Orion showing the largest ratio is **not** an advantage: the ratio is inflated by
a worse centroid baseline, and the centroid router is not the one Orion deploys.
The substantive point is that this gap is large on *all* layouts, so the biggest
single lever available is adaptive, well-ordered routing rather than placement.
Placement decides the floor; routing decides how close you get to it.

## Finding 4: Equal shard sizes do not produce equal query load

`load_skew.py` counts how many queries touch each shard, rather than how many
shards a query touches. Throughput is set by the busiest shard, so this is the
quantity that caps scale-out. At P = 32, R = 0.95, under the oracle ordering:

| layout | size skew (sift / glove) | query load skew (sift / glove) |
|---|---|---|
| kmeans | 1.48 / 2.05 | 6.77 / 5.72 |
| balanced_kmeans | 1.00 / 1.00 | 6.05 / 4.55 |
| orion | 1.01 / 1.01 | 6.11 / 5.37 |
| orion_norep | 1.01 / 1.01 | 5.73 / 4.87 |

Driving size skew from 2.05 down to 1.00 moves load skew by roughly 20% —
nowhere near proportionally. **Orion's capacity-constrained balancing spends the
entire offline pipeline equalizing a quantity that does not set throughput.**
Load skew is a property of where queries cluster in dense data regions, which
placement barely controls.

Combining the two halves: the busiest shard's load is proportional to
(shards probed) x (load skew), so that product is inverse normalized throughput.
Under the oracle ordering at P = 32 it spans 17.2–20.3 on SIFT and 21.5–24.4 on
GloVe across all four layouts — a 13%–18% spread.

**Correction:** the "barely moves the ceiling" reading here is an artifact of
computing it under the *oracle* ordering, which is not any deployed router and
which flattens the layouts' differences. Under the actual deployed routers (the
Headline section), placement moves fan-out a lot — Orion's layout is 23%–54%
worse on fan-out than plain k-means — and fan-out, not skew, is the first-order
term. Load skew is the second-order correction that only rescues Orion on the
skewed GloVe workload. Treat the Headline as the throughput verdict, not this.

There is also a tension worth building on. Oracle routing minimizes total work
(3 shards on SIFT at P = 32) but concentrates load 6.1x; centroid routing spreads
load to 1.76x but needs 7 shards. Getting low work *and* low skew simultaneously
is an open problem, and replication is a natural instrument for it — a hot
point placed on two shards lets its queries be split. That is a better
justification for multi-assignment than the locality one, and it is untested.

This finding drove the decision recorded in
`docs/decisions/0001-plain-kmeans-placement.md`.

## Finding 5: Orion's navigation router beats a centroid router, but modestly

Findings 1–4 order shards by oracle coverage or by centroid distance, neither of
which is Orion's deployed router. `orion_router_skew.py` reconstructs the real
one: a query's nearest upper-graph L1 nodes are looked up and the shards owning
them are the shards probed, so fan-out is emergent, not a chosen prefix. All
three routers are run on the *same* Orion layout to R = 0.95. Adaptive mean
fan-out (shards touched per query) and query load skew (max/mean shard touches):

| dataset | P | oracle fanout / skew | centroid fanout / skew | orion fanout / skew |
|---|---|---|---|---|
| sift  |  8 | 1.98 / 2.36 | 2.84 / 1.58 | 2.32 / 1.58 |
| sift  | 32 | 2.64 / 6.11 | 5.38 / 1.76 | 4.33 / 1.52 |
| glove |  8 | 2.67 / 2.15 | 4.10 / 1.13 | 3.74 / 1.21 |
| glove | 32 | 3.25 / 5.37 | 9.99 / 1.89 | 9.60 / 1.71 |

Three things stand out.

- **Navigation routing is cheaper than centroid routing at equal recall, on every
  point.** It touches 4%–27% fewer shards per query (2.32 vs 2.84 and 4.33 vs
  5.38 on SIFT; 3.74 vs 4.10 and 9.60 vs 9.99 on GloVe), with the larger gains on
  the easier dataset and the smaller P. So the navigation graph does earn its keep
  for routing — this is the first mechanism-specific evidence that it does.
- **It does so without paying in balance.** Load skew is essentially tied
  (slightly better than centroid on SIFT, slightly worse on GloVe). Navigation
  gets lower fan-out *and* comparable-or-better skew, which is the tension Finding
  4 flagged as open — Orion resolves it a little better than a centroid router.
- **Neither deployable router captures the big headroom, exactly where it is
  largest.** On GloVe P = 32 the oracle reaches the target at 3.25 shards while
  both real routers need ~9.6 — a 3x gap that navigation barely narrows. The
  headroom Finding 3 identified is real but is *not* unlocked by graph navigation
  on hard, high-P workloads; it would take a better routing signal than either
  centroid distance or nearest-L1 membership.

Net: the navigation graph is worth a modest, real routing improvement over the
centroid router that a plain k-means layout would use, biggest on easy data and
small P, and it is not the key to the large oracle headroom on hard data.

Fidelity caveat: the upper graph is the hnswlib dual-graph variant and the router
is the harness's Python encoding, not the Rust `router.rs`. This is sound for the
structural comparison, not for deployable numbers.

## Full table at R = 0.95, k = 10

`bal` is max/mean shard size, `exp` is physical copies per logical point.

```
dataset   P layout              fanout  p95   bal   exp | orc.fix  orc.ad cen.fix  cen.ad |  H_ord  H_adp  H_tot
glove    4 balanced_kmeans       2.44    4  1.00 1.000 |       3    1.94       4    2.09 |  1.33x  1.91x  2.06x
glove    4 kmeans                1.99    3  1.69 1.000 |       2    1.54       3    1.60 |  1.50x  1.87x  1.95x
glove    4 orion                 2.11    4  1.00 1.072 |       3    1.62       4    1.89 |  1.33x  2.11x  2.47x
glove    4 orion_norep           2.16    4  1.00 1.000 |       3    1.67       4    1.95 |  1.33x  2.05x  2.39x
glove    8 balanced_kmeans       3.05    6  1.00 1.000 |       4    2.55       6    3.03 |  1.50x  1.98x  2.35x
glove    8 kmeans                2.22    4  2.41 1.000 |       3    1.72       4    1.91 |  1.33x  2.09x  2.32x
glove    8 orion                 2.67    5  1.00 1.124 |       3    2.17       6    2.91 |  2.00x  2.06x  2.76x
glove    8 orion_norep           2.79    5  1.01 1.000 |       4    2.29       6    3.05 |  1.50x  1.97x  2.62x
glove   16 balanced_kmeans       3.17    7  1.00 1.000 |       4    2.67      10    3.86 |  2.50x  2.59x  3.74x
glove   16 kmeans                2.60    5  2.72 1.000 |       3    2.10       7    2.68 |  2.33x  2.61x  3.34x
glove   16 orion                 2.80    6  1.01 1.143 |       4    2.30       9    3.55 |  2.25x  2.54x  3.91x
glove   16 orion_norep           2.95    6  1.01 1.000 |       4    2.45      10    3.77 |  2.50x  2.65x  4.09x
glove   32 balanced_kmeans       3.62    7  1.00 1.000 |       5    3.12      17    6.16 |  3.40x  2.76x  5.45x
glove   32 kmeans                2.91    6  2.05 1.000 |       4    2.41      12    3.88 |  3.00x  3.09x  4.97x
glove   32 orion                 3.25    7  1.01 1.181 |       4    2.75      16    5.54 |  4.00x  2.89x  5.82x
glove   32 orion_norep           3.46    7  1.01 1.000 |       5    2.96      17    6.14 |  3.40x  2.77x  5.74x
sift     4 balanced_kmeans       1.59    3  1.00 1.000 |       2    1.24       3    1.41 |  1.50x  2.12x  2.43x
sift     4 kmeans                1.28    2  1.07 1.000 |       2    1.04       2    1.05 |  1.00x  1.90x  1.92x
sift     4 orion                 1.42    2  1.01 1.038 |       2    1.11       2    1.37 |  1.00x  1.46x  1.80x
sift     4 orion_norep           1.43    2  1.01 1.000 |       2    1.13       2    1.38 |  1.00x  1.45x  1.77x
sift     8 balanced_kmeans       2.21    5  1.00 1.000 |       3    1.71       5    2.07 |  1.67x  2.41x  2.92x
sift     8 kmeans                1.55    3  1.47 1.000 |       2    1.21       2    1.23 |  1.00x  1.63x  1.66x
sift     8 orion                 1.98    4  1.01 1.070 |       2    1.50       5    2.00 |  2.50x  2.50x  3.33x
sift     8 orion_norep           2.06    4  1.01 1.000 |       3    1.57       5    2.07 |  1.67x  2.41x  3.19x
sift    16 balanced_kmeans       2.52    5  1.00 1.000 |       3    2.02       7    2.55 |  2.33x  2.74x  3.47x
sift    16 kmeans                1.91    4  1.50 1.000 |       2    1.47       3    1.51 |  1.50x  1.98x  2.04x
sift    16 orion                 2.24    4  1.01 1.077 |       3    1.74       4    2.02 |  1.33x  1.98x  2.30x
sift    16 orion_norep           2.17    4  1.01 1.000 |       3    1.67       4    1.91 |  1.33x  2.09x  2.40x
sift    32 balanced_kmeans       2.74    5  1.00 1.000 |       3    2.24       6    2.53 |  2.00x  2.37x  2.67x
sift    32 kmeans                2.31    4  1.48 1.000 |       3    1.81       4    1.95 |  1.33x  2.05x  2.21x
sift    32 orion                 2.64    5  1.01 1.124 |       3    2.14       7    2.87 |  2.33x  2.44x  3.28x
sift    32 orion_norep           2.80    5  1.01 1.000 |       3    2.30       8    3.10 |  2.67x  2.58x  3.48x
```

## Status

The fan-out table is complete for all four layouts at P = 4, 8, 16, 32 on both
datasets. Load skew is measured at P = 8 and P = 32 on SIFT and P = 32 on GloVe;
the remaining load-skew points are not yet run.

## Caveats

- **The centroid ordering is not Orion's router.** Orion routes by navigating the
  upper graph, and its shards are not geometric blobs, so centroids summarize
  them poorly. The `cen.*` columns overstate Orion's realized cost. Findings 1
  and 2 rest entirely on the router-independent oracle columns.
- **The layouts come from the legacy dual-graph path.** The upper graph here is
  built with `hnswlib`, which `agent.md` classifies as an ablation rather than
  canonical Orion; the canonical path builds the production upper graph with
  Qdrant's Rust `GraphLayersBuilder`. The comparison is still meaningful because
  the question is about the layout family, but a canonical-path layout could
  differ and Finding 1 should be re-checked against one.
- **Greedy ordering was verified against the exact optimum.** Under replication
  maximum coverage is a set-cover style problem where greedy is only a `(1-1/e)`
  approximation in theory, which would have inflated the oracle floor of the
  replicated Orion layouts and biased the comparison against them. Brute force
  over a 2000-query sample at P = 32 found greedy suboptimal on 0.00%–0.20% of
  queries with a mean coverage gap below 0.0002, so the oracle columns are tight.
  See `check_greedy_optimality.py`.
- **CCNB did not always converge.** On SIFT at P = 16 the default 8 balancing
  passes left two shards below the lower load bound and the build aborted; 32
  passes converged. Fission never triggered in these runs, so the effective shard
  count always equalled the request.
- One dataset scale (about 1M points), `k = 10`, one seed per layout. Seed
  sensitivity is unmeasured, and the P = 32 SIFT reversal in Finding 2 is a
  0.06-shard margin that a second seed could plausibly move.

## Reproducing

```sh
# Layouts (needs the system libstdc++ ahead of conda's for hnswlib).
# Export LD_PRELOAD only for this command; leaving it set in a shell session
# affects every later process.
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
  python3 build_orion_layout.py --hdf5-path <dataset>.hdf5 \
  --out layouts/orion_<name>_p32.npz --shards 32 \
  --vector-distance euclid --enable-fission --balance-max-passes 32

# Analysis
python3 fanout_headroom.py --hdf5 <dataset>.hdf5 --method layout_file \
  --layout-file layouts/orion_<name>_p32.npz --label orion \
  --recall-targets 0.90 0.95 --out out/<name>_orion_p32.json
python3 summarize.py --target 0.95 --reports out/*.json
```
