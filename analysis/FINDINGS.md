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

## Results at R = 0.95, k = 10

`bal` is max/mean shard size, `exp` is physical copies per logical point.

```
dataset   P layout              fanout  p95   bal   exp | orc.fix  orc.ad cen.fix  cen.ad |  H_ord  H_adp  H_tot
glove    4 kmeans                1.99    3  1.69 1.000 |       2    1.54       3    1.60 |  1.50x  1.87x  1.95x
glove    4 orion                 2.11    4  1.00 1.072 |       3    1.62       4    1.89 |  1.33x  2.11x  2.47x
glove    8 kmeans                2.22    4  2.41 1.000 |       3    1.72       4    1.91 |  1.33x  2.09x  2.32x
glove    8 orion                 2.67    5  1.00 1.124 |       3    2.17       6    2.91 |  2.00x  2.06x  2.76x
glove   16 kmeans                2.60    5  2.72 1.000 |       3    2.10       7    2.68 |  2.33x  2.61x  3.34x
glove   16 orion                 2.80    6  1.01 1.143 |       4    2.30       9    3.55 |  2.25x  2.54x  3.91x
glove   32 kmeans                2.91    6  2.05 1.000 |       4    2.41      12    3.88 |  3.00x  3.09x  4.97x
glove   32 orion                 3.25    7  1.01 1.181 |       4    2.75      16    5.54 |  4.00x  2.89x  5.82x
sift     4 kmeans                1.28    2  1.07 1.000 |       2    1.04       2    1.05 |  1.00x  1.90x  1.92x
sift     4 orion                 1.42    2  1.01 1.038 |       2    1.11       2    1.37 |  1.00x  1.46x  1.80x
sift     8 kmeans                1.55    3  1.47 1.000 |       2    1.21       2    1.23 |  1.00x  1.63x  1.66x
sift     8 orion                 1.98    4  1.01 1.070 |       2    1.50       5    2.00 |  2.50x  2.50x  3.33x
sift    16 kmeans                1.91    4  1.50 1.000 |       2    1.47       3    1.51 |  1.50x  1.98x  2.04x
sift    16 orion                 2.24    4  1.01 1.077 |       3    1.74       4    2.02 |  1.33x  1.98x  2.30x
sift    32 kmeans                2.31    4  1.48 1.000 |       3    1.81       4    1.95 |  1.33x  2.05x  2.21x
sift    32 orion                 2.64    5  1.01 1.124 |       3    2.14       7    2.87 |  2.33x  2.44x  3.28x
```

## Finding 1: Orion's partitioning is dominated on locality, router-independently

Orion's required fan-out is higher than plain k-means in all 8 configurations,
and so is its oracle adaptive floor — by 5% to 26% at R = 0.95 (and 7 of 8 at
R = 0.90, with one tie). Because the oracle ordering is a lower bound over all
routers, **no router, however good, can make Orion's layout as cheap as k-means'
layout.** This is not a routing deficiency that better engineering can recover;
it is a property of where the points sit.

The result is stronger than it first looks, because Orion also spends 3.8%–18.1%
extra storage on replicated copies, and replication can only *help* coverage.
Orion loses while paying more.

This independently reproduces the legacy C3 verdict (Orion's partition quality
worse than k-means) with an exactly computable metric instead of TWCut.

## Finding 2: Orion's real, measurable win is balance

Orion holds max/mean shard size at 1.00–1.01 everywhere. k-means ranges from
1.07 to 2.72; on GloVe at P = 16 one k-means shard carries 2.72x the mean, which
in a real deployment sets the tail latency and caps useful scale-out. This is
what the capacity-constrained balancing actually delivers, and it is a genuine
contribution — just not the one about locality.

The cost of that balance is the 3.8%–18.1% storage expansion, rising with P.

## Finding 3: The headroom is in routing, and it is not Orion-specific

`H_tot` reaches 5.82x for Orion on GloVe at P = 32 versus 4.97x for k-means, so
Orion does show the larger ratio. That is **not** an advantage. The ratio is
larger only because Orion's centroid baseline is worse (16 shards versus 12),
while its ideal floor is also worse (2.75 versus 2.41). A bigger gap above a
lower ceiling is not more attainable gain.

What the numbers do show is that on *both* layouts the gap between a conventional
fixed-budget centroid router and an ideal adaptive one is large — roughly 2x to
5x, growing with P and with dataset hardness. That gap is where the performance
is, and it is available on the k-means layout too. The configuration this
analysis points at is therefore a **well-balanced geometric layout plus a
navigation-based adaptive router**, not Orion's topology-aware partitioner.

## Caveats

- **The centroid ordering is not Orion's router.** Orion routes by navigating the
  upper graph, not by centroid distance, and its shards are not geometric blobs,
  so centroids summarize them poorly. The `cen.*` columns overstate Orion's
  realized cost. Findings 1 and 2 do not depend on those columns; Finding 1 rests
  entirely on the router-independent oracle columns.
- **The layouts come from the legacy dual-graph path.** The upper graph here is
  built with `hnswlib`, which `agent.md` classifies as an ablation rather than
  canonical Orion; the canonical path builds the production upper graph with
  Qdrant's Rust `GraphLayersBuilder`. The comparison is still meaningful because
  the question is about the layout family, but a canonical-path layout could
  differ and Finding 1 should be re-checked against one.
- **Greedy ordering was verified against the exact optimum.** Under replication
  greedy maximum coverage is only a `(1-1/e)` approximation in theory, which
  would inflate the oracle floor and bias the comparison against Orion. Brute
  force over a 2000-query sample at P = 32 found greedy suboptimal on 0.00%–0.20%
  of queries with a mean coverage gap below 0.0002, so the oracle columns are
  tight. See `check_greedy_optimality.py`.
- **CCNB did not always converge.** On SIFT at P = 16 the default 8 balancing
  passes left two shards below the lower load bound and the build aborted; 32
  passes converged. That layout uses `--balance-max-passes 32`. Fission never
  triggered in these runs, so effective shard count always equalled the request.
- Single dataset scale (about 1M points), `k = 10`, one seed per layout. The
  k-means baseline is unbalanced by construction; `balanced_kmeans` is available
  in `fanout_headroom.py` and is the fairer opponent for Finding 2.

## Reproducing

```sh
# Layouts (needs the system libstdc++ ahead of conda's for hnswlib)
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
  python3 build_orion_layout.py --hdf5-path <dataset>.hdf5 \
  --out layouts/orion_<name>_p32.npz --shards 32 \
  --vector-distance euclid --enable-fission

# Analysis
python3 fanout_headroom.py --hdf5 <dataset>.hdf5 --method layout_file \
  --layout-file layouts/orion_<name>_p32.npz --label orion \
  --recall-targets 0.90 0.95 --out out/<name>_orion_p32.json
python3 summarize.py --target 0.95 --reports out/*.json
```
