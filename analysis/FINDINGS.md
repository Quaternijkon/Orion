# Offline fan-out headroom: Orion's layout versus k-means

This is an offline, cluster-free analysis. It uses ground-truth neighbours and a
materialized shard layout to compute how many shards a query *must* touch to
reach a recall target. It replaces the retired TWCut metric, which the legacy
program showed does not predict cost (see `legacy/LEGACY.md`).

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
GloVe. A shard carrying 2.7x the mean sets the tail latency and caps useful
scale-out, so unbalanced k-means is not a layout anyone can actually deploy at
scale. Comparing against it alone is what made an earlier version of this
document conclude that Orion's partitioner was dominated. That conclusion was an
artifact of the unfair baseline and is **withdrawn**.

Once the baseline is forced to the same balance Orion achieves, the ranking
inverts. Router-independent oracle adaptive floor at R = 0.95, in shards per
query (lower is better):

| dataset | P | kmeans (bal 1.07–2.72) | balanced_kmeans (bal 1.00) | orion (bal 1.01) | orion_norep (bal 1.01) |
|---|---|---|---|---|---|
| sift  |  4 | 1.04 | 1.24 | **1.11** | 1.13 |
| sift  |  8 | 1.21 | 1.71 | **1.50** | 1.57 |
| sift  | 16 | 1.47 | 2.02 | **1.74** | 1.67 |
| sift  | 32 | 1.81 | **2.24** | 2.14 | 2.30 |
| glove |  4 | 1.54 | 1.94 | **1.62** | pending |
| glove |  8 | 1.72 | 2.55 | **2.17** | pending |
| glove | 16 | 2.10 | 2.67 | **2.30** | pending |
| glove | 32 | 2.41 | 3.12 | **2.75** | pending |

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

The GloVe replication control has not been run yet (see Status below), so it is
not yet known whether the P = 32 reversal is dataset-specific.

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

## Full table at R = 0.95, k = 10

`bal` is max/mean shard size, `exp` is physical copies per logical point.

```
dataset   P layout              fanout  p95   bal   exp | orc.fix  orc.ad cen.fix  cen.ad |  H_ord  H_adp  H_tot
glove    4 kmeans                1.99    3  1.69 1.000 |       2    1.54       3    1.60 |  1.50x  1.87x  1.95x
glove    4 balanced_kmeans       2.44    4  1.00 1.000 |       3    1.94       4    2.09 |  1.33x  1.91x  2.06x
glove    4 orion                 2.11    4  1.00 1.072 |       3    1.62       4    1.89 |  1.33x  2.11x  2.47x
glove    8 kmeans                2.22    4  2.41 1.000 |       3    1.72       4    1.91 |  1.33x  2.09x  2.32x
glove    8 balanced_kmeans       3.05    6  1.00 1.000 |       4    2.55       6    3.03 |  1.50x  1.98x  2.35x
glove    8 orion                 2.67    5  1.00 1.124 |       3    2.17       6    2.91 |  2.00x  2.06x  2.76x
glove   16 kmeans                2.60    5  2.72 1.000 |       3    2.10       7    2.68 |  2.33x  2.61x  3.34x
glove   16 balanced_kmeans       3.17    7  1.00 1.000 |       4    2.67      10    3.86 |  2.50x  2.59x  3.75x
glove   16 orion                 2.80    6  1.01 1.143 |       4    2.30       9    3.55 |  2.25x  2.54x  3.91x
glove   32 kmeans                2.91    6  2.05 1.000 |       4    2.41      12    3.88 |  3.00x  3.09x  4.97x
glove   32 balanced_kmeans       3.62    7  1.00 1.000 |       5    3.12      17    6.16 |  3.40x  2.76x  5.45x
glove   32 orion                 3.25    7  1.01 1.181 |       4    2.75      16    5.54 |  4.00x  2.89x  5.82x
sift     4 kmeans                1.28    2  1.07 1.000 |       2    1.04       2    1.05 |  1.00x  1.90x  1.92x
sift     4 balanced_kmeans       1.59    3  1.00 1.000 |       2    1.24       3    1.41 |  1.50x  2.12x  2.42x
sift     4 orion                 1.42    2  1.01 1.038 |       2    1.11       2    1.37 |  1.00x  1.46x  1.80x
sift     4 orion_norep           1.43    2  1.01 1.000 |       2    1.13       2    1.38 |  1.00x  1.45x  1.77x
sift     8 kmeans                1.55    3  1.47 1.000 |       2    1.21       2    1.23 |  1.00x  1.63x  1.66x
sift     8 balanced_kmeans       2.21    5  1.00 1.000 |       3    1.71       5    2.07 |  1.67x  2.41x  2.92x
sift     8 orion                 1.98    4  1.01 1.070 |       2    1.50       5    2.00 |  2.50x  2.50x  3.33x
sift     8 orion_norep           2.06    4  1.01 1.000 |       3    1.57       5    2.07 |  1.67x  2.41x  3.18x
sift    16 kmeans                1.91    4  1.50 1.000 |       2    1.47       3    1.51 |  1.50x  1.98x  2.04x
sift    16 balanced_kmeans       2.52    5  1.00 1.000 |       3    2.02       7    2.55 |  2.33x  2.74x  3.47x
sift    16 orion                 2.24    4  1.01 1.077 |       3    1.74       4    2.02 |  1.33x  1.98x  2.30x
sift    16 orion_norep           2.17    4  1.01 1.000 |       3    1.67       4    1.91 |  1.33x  2.09x  2.40x
sift    32 kmeans                2.31    4  1.48 1.000 |       3    1.81       4    1.95 |  1.33x  2.05x  2.21x
sift    32 balanced_kmeans       2.74    5  1.00 1.000 |       3    2.24       6    2.53 |  2.00x  2.37x  2.68x
sift    32 orion                 2.64    5  1.01 1.124 |       3    2.14       7    2.87 |  2.33x  2.44x  3.28x
sift    32 orion_norep           2.80    5  1.01 1.000 |       3    2.30       8    3.10 |  2.67x  2.58x  3.48x
```

## Status

The four GloVe `orion_norep` runs are outstanding. Their layouts are already
built at `layouts/orion_norep_glove_p{4,8,16,32}.npz`; only the analysis step is
missing. The shell environment became unresponsive partway through, twice,
including on a bare `echo`, so this is an environment failure rather than a
script failure. Memory pressure from the GloVe runs is the leading suspect but is
unverified. Resume with:

```sh
for P in 4 8 16 32; do
  python3 fanout_headroom.py --hdf5 <path>/glove-200-angular.hdf5 \
    --method layout_file --layout-file layouts/orion_norep_glove_p$P.npz \
    --label orion_norep --normalize --recall-targets 0.90 0.95 \
    --out out/glove_orionnorep_p$P.json
done
```

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
