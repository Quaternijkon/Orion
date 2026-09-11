# Ablation: initial seed for Orion's navigation-graph partitioning

**Question.** Orion seeds its L1 (upper-graph) partition with balanced k-means,
then refines it with capacity-constrained label propagation over the navigation
graph. Is the k-means seed the right choice, or would a locality-free seed (the
kind a default hash-sharded system starts from) do just as well after refinement?

**Method.** Only the seed is changed; every downstream stage (topology
refinement, balance repair, L0 placement) uses identical parameters. The seed is
injected through `build_original_routing_state(..., initial_l1_to_shard=...)`
(new hook) via `build_orion_layout.py --init {kmeans,random}`:

- `kmeans` — the shipped balanced-k-means seed (reproduced exactly).
- `random` — a balanced round-robin over the upper nodes: uniform, no locality.
  This is the "hash-style" cold start.

Layouts are single-copy (`--disable-multi-assign`), P=46, cosine, with balance
band `[0.7, 1.4]`. Three quantities are reported per arm, all offline:

- **required fan-out** — shards that hold a query's true k-NN, averaged over
  queries (`fanout_headroom.py`). The best case any router can achieve; a pure
  measure of how well the layout concentrates each query's neighbors.
- **centroid fan-out @ R** — fixed shard budget a realistic k-means/IVF router
  (rank shards by query-to-centroid distance) must spend for mean recall ≥ R.
- **attachment cut** — fraction of upper-graph attachment edges that cross a
  shard boundary; exactly the objective the label-propagation refinement
  minimizes. **churn** — fraction of upper nodes the refinement relabels
  relative to the seed.

## Results

GloVe-200-angular (1.18M):

| seed   | required fan-out | oracle fixed@0.95 | centroid fixed@0.95 | attach cut (seed→final) | churn |
|--------|:---:|:---:|:---:|:---:|:---:|
| kmeans | **3.60** | **5** | **22** | 0.521 → **0.474** | 0.19 |
| random | 4.98 | 7 | 38 | 0.881 → 0.567 | 0.86 |

Deep-image-96-angular (10M):

| seed   | required fan-out | oracle fixed@0.95 | centroid fixed@0.95 | attach cut (seed→final) | churn |
|--------|:---:|:---:|:---:|:---:|:---:|
| kmeans | **2.27** | **3** | **5** | 0.312 → **0.262** | 0.14 |
| random | 4.55 | 6 | 42 | 0.881 → 0.515 | 0.90 |

## Conclusion — the k-means seed is the correct design decision

1. **Refinement cannot repair a bad seed.** Label propagation is init-sensitive:
   from a random seed every upper node's neighbors carry random labels, so the
   majority vote has no coherent signal to propagate. Random churns 86–90% of
   the nodes yet still converges to a **substantially worse fixed point** than
   k-means — final attachment cut ≈2× on Deep (0.515 vs 0.262), 1.2× on GloVe.
   K-means starts inside a good basin and the refinement only polishes it
   (churn 14–19%).

2. **The layout the refinement produces is measurably worse.** Random needs
   1.4× (GloVe) to 2.0× (Deep) more oracle fan-out — i.e. ~2× the compute per
   query at matched recall on the hardest dataset even with a perfect router.

3. **Random destroys geometric routability.** A centroid router on the
   random-seeded layout is nearly forced to broadcast (fixed budget 38/46 on
   GloVe, 42/46 on Deep) because random-shard centroids are meaningless. The
   k-means seed keeps shards geometrically coherent, so the same cheap router
   pays 22 and 5. This is the practically decisive gap.

4. **The advantage widens with difficulty.** On Deep (learned embeddings, where
   partition quality matters most) the required-fan-out gap is 2.0× and the
   centroid-router gap is 8.4×, versus 1.4× / 1.7× on GloVe. The seed matters
   *more*, not less, exactly where Orion's gains are largest.

So the k-means seed is not incidental: it places the label-propagation refiner
in a basin it cannot otherwise reach and preserves the geometric structure the
serving-time router depends on.

## Follow-up: is one refinement pass enough after the k-means seed?

Same seed (k-means), varying how many label-propagation passes the refinement
runs. `converge_l1_topology_capacity_constrained` now records a per-pass move
count and (under `ORION_LOG_REFINE_CUT=1`) the per-pass attachment cut, surfaced
in `build_orion_layout` stdout / metadata.

**Per-pass attachment cut (shipped tight band `[0.7,1.4]`, run to convergence):**

| pass | GloVe cut | GloVe Δ | Deep cut | Deep Δ |
|:---:|:---:|:---:|:---:|:---:|
| 0 (seed) | 0.521 | — | 0.312 | — |
| 1 | 0.495 | −0.026 | 0.284 | −0.028 |
| 2 | 0.476 | −0.019 | 0.268 | −0.017 |
| 3 | 0.471 | −0.006 | 0.264 | −0.003 |
| converge | 0.458 (p11) | tail −0.013 | 0.261 (p16) | tail −0.003 |

Pass 1 delivers 42% (GloVe) / 62% (Deep) of the total cut reduction; passes 1–2
give 72% / 87%; passes 1–3 give 81% / 94%. The refinement self-terminates
(changed=0) at pass 11 / 16, after which additional passes are exact no-ops.
Moves are overwhelmingly `topology_gain` (GloVe 6660 vs 934 balance repair; Deep
43804 vs 678).

**Deliverable metric (required fan-out) vs passes, GloVe, balance unconstrained
to isolate topology:**

| passes | required fan-out (mean) |
|:---:|:---:|
| 1 | 3.08 |
| 2 | 3.02 |
| 3 | 2.99 |
| converge (17) | 2.94 |

For the topology objective alone, **one pass already lands within ~5% of the
converged fan-out**; three passes within ~2%.

**Conclusion.** The refinement's *topology* gain is almost entirely front-loaded
into the first 1–3 passes, so in that sense one round is nearly enough. But in
the shipped capacity-constrained mode you cannot stop that early: capping at 1–3
passes under the `[0.7,1.4]` band **fails the balance gate** (several shards below
the lower load bound), so the run must continue — the extra passes to ~11–16 are
spent satisfying balance, not improving topology, and are cheap and
self-terminating. Continuing past convergence yields exactly zero. Separately,
enforcing balance itself costs fan-out (unconstrained converged 2.94 at 7.7×
imbalance vs tight-band converged 3.60 at 1.4×). Net: the default (run to
automatic convergence) is the right setting; there is no benefit to forcing more
passes, and no safe way to force many fewer.

## Follow-up: does the balanced-k-means seed *algorithm* matter?

Orion's seed is k-means++/Lloyd centroids plus a **custom** capacity assignment
(`cpp_style_predict_balanced`: sort all point×centroid distances globally, greedily
fill under a soft `target×max_load_ratio` cap, dump leftovers on the lightest
shard). This is not a textbook balanced-k-means. Does the specific balancing
algorithm change the outcome? Three seeds, identical downstream refinement:

- `custom` — Orion's greedy capacity assignment (the shipped seed).
- `standard_balanced` — MiniBatchKMeans + exact equal-size assignment by
  (second−best) priority (ELKI-style same-size heuristic; `fanout_headroom.balanced_kmeans`).
- `plain` — plain nearest-centroid k-means, **no balancing at all**.

| dataset | seed | seed skew | seed cut | final cut | **required fan-out** | final balance |
|---|---|:---:|:---:|:---:|:---:|:---:|
| GloVe | custom | ~1.4 | 0.521 | 0.474 | **3.60** | 1.39 |
| GloVe | standard | 1.00 | 0.533 | 0.435 | **3.62** | 1.38 |
| GloVe | plain | 2.53 | 0.447 | 0.425 | **3.58** | 1.39 |
| Deep | custom | ~1.4 | 0.312 | 0.262 | **2.27** | 1.40 |
| Deep | standard | 1.00 | 0.348 | 0.240 | **2.28** | 1.40 |
| Deep | plain | 1.40 | 0.280 | 0.229 | **2.22** | 1.40 |

**The deliverable (required fan-out) is invariant to the seed's balancing
algorithm** — 3.58–3.62 on GloVe, 2.22–2.28 on Deep, all within ~2%, and all end
at the same balance (1.4×) because the refinement's capacity repair re-balances
regardless of seed. So Orion's custom balanced k-means is **"good enough" but not
better**: a standard equal-size seed, or even a fully unbalanced plain-k-means
seed, produces the same fan-out. (On the attachment-cut proxy the custom seed is
actually slightly *worse* after refinement — 0.474 vs 0.435/0.425 on GloVe — but
that does not move fan-out.)

The single thing that matters for the seed is **geometric locality**: every
k-means variant here preserves it and lands at ~3.6 / ~2.3, whereas the random
seed above (no locality) lands at 4.98 / 4.55. Since the downstream refinement
already owns balance, the seed's own balancing is largely redundant — plain
k-means would serve as well, consistent with `docs/decisions/0001-plain-kmeans-placement.md`.

## Caveat / future work

A graph-native seed (METIS or normalized-cut/spectral on the upper graph) is the
natural "can we beat k-means" arm, since it targets the same cut objective the
refinement optimizes rather than geometric variance. It was **not completed
here**: `pymetis`/`metis` are unavailable in this environment, and a spectral
seed via sklearn hit two blockers — an `LD_PRELOAD` (system libstdc++, required
by `hnswlib`) vs. `numexpr`/`bottleneck` ABI conflict in-process, and an ARPACK
eigensolver that did not converge in reasonable time on the k-NN graph (likely
disconnected components at k=15). The `--init spectral` path and the standalone
`make_spectral_seed.py` (run without `LD_PRELOAD`, fed via `--init-file`) are in
place for a follow-up with a converging solver (LOBPCG+AMG, or a connected /
mutual-kNN graph). This ablation establishes the k-means-vs-cold-start gap; it
does not rule out a better principled seed.

## Reproduce

```bash
# in experiments/harness, LD_PRELOAD set for the hnswlib upper-graph build
COMMON="--shards 46 --vector-distance cosine --disable-multi-assign \
  --balance-mode capacity_constrained --balance-min-load-ratio 0.7 \
  --balance-max-load-ratio 1.4 --balance-max-vote-loss 20 --balance-max-passes 64"
for init in kmeans random; do
  LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
  python ../../analysis/build_orion_layout.py --hdf5-path <dataset.hdf5> \
    --out layouts/ablation/orion_<tag>_$init.npz --init $init $COMMON
  python ../../analysis/fanout_headroom.py --hdf5 <dataset.hdf5> \
    --method layout_file --layout-file layouts/ablation/orion_<tag>_$init.npz \
    --normalize --top-k 10 --recall-targets 0.90 0.95 --out ablation_reports/<tag>_$init.json
done
# churn / attachment cut are printed in the layout metadata (build_orion_layout stdout)
```

# Load balancing: moving points fails, replicating hot points wins

Throughput is capped by the busiest shard, so the proxy is `W = fan-out x
load_skew` (fan-out sets total compute, load_skew sets how unevenly it lands).
Three questions were tested: whether the size-balance band is needed at all
(below), whether finer point-level load balancing helps (negative result), and
whether replication helps (positive result).

## The size-balance band is load-protective (tight vs. loose)

Before attacking `load_skew` at the point level, the prior question is whether the
size-balance band should exist at all: pure min-cut (no band) achieves a *lower*
fan-out, so why constrain it? Two P=46 single-copy GloVe layouts, identical
pipeline (kmeans seed, capacity-constrained refinement), differing **only** in the
capacity band:

| band | size skew | attach. cut | oracle fan-out | oracle skew | **centroid fan-out** | **centroid skew** | **centroid W** |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| tight `[0.7,1.4]` | 1.40x | 0.475 | 3.60 | 6.87 | 14.62 | 2.02 | **29.5** |
| loose `[0.1,8.0]` | 7.64x | 0.331 | 2.94 | 8.22 | 9.41 | 3.58 | **33.7** |

Dropping the band lets min-cut collapse co-accessed hubs into a few shards: size
skew jumps 1.4x -> 7.6x and the attachment cut falls (0.475 -> 0.331), so the
*deliverable* fan-out drops (3.60 -> 2.94 oracle; 14.62 -> 9.41 centroid). But the
concentrated hubs inflate load skew (centroid 2.02 -> 3.58), and since `W =
fan-out x load_skew` the skew blow-up dominates: **W worsens 29.5 -> 33.7 (+14%)
under a realizable centroid router even though fan-out fell**. Under an *oracle*
router the two effects cancel (W ~24.5 either way), which is exactly why size
skew alone looks harmless — the cost only surfaces once a real router must place
probes. (The independently-built `orion_glove_loose.npz` reproduces the loose arm:
size 7.70x, oracle 2.94/8.23, centroid W 9.33x3.59 = 33.5.)

So the band is not a storage convenience: it is the mechanism that prevents hub
collapse and keeps the layout spreadable by the router. Orion ships the tight
`[0.7,1.4]` band. Beyond enforcing *some* reasonable band, finer load-specific
balancing does not help (next).

**Reproduce.**
```bash
# in experiments/harness, LD_PRELOAD set (system libstdc++, for hnswlib)
H=data/glove/glove-200-angular.hdf5
# tight arm: --balance-min-load-ratio 0.7  --balance-max-load-ratio 1.4  -> glove_tight_p46.npz
# loose arm: --balance-min-load-ratio 0.1  --balance-max-load-ratio 8.0  -> glove_looseband_p46.npz
python ../../analysis/build_orion_layout.py --hdf5-path $H \
  --out layouts/ablation/glove_<tag>_p46.npz --shards 46 --vector-distance cosine \
  --disable-multi-assign --balance-mode capacity_constrained \
  --balance-min-load-ratio <lo> --balance-max-load-ratio <hi> \
  --balance-max-vote-loss 20 --balance-max-passes 64
python ../../analysis/load_skew.py --hdf5 $H --label <tag>_p46 \
  --layout-file layouts/ablation/glove_<tag>_p46.npz --normalize --top-k 10 \
  --target 0.95 --out ablation_reports/skew_glove_<tag>_p46.json
# W = adaptive_mean_fanout x load_skew_max_over_mean, centroid ordering
```

## Negative result: hubness-weighted point balancing

Rebalancing the layout so each shard holds an equal share of *estimated query
load* (per-point hubness / kNN in-degree) instead of an equal *point count* does
**not** help. On GloVe it left `W` essentially unchanged under oracle routing
(~+2%) and *worse* under realistic centroid routing (~-7% throughput), while
blowing up `size_skew` from 1.10x to 2.57x. Reason: load skew is intrinsic to
neighbor clustering — hot points sit together, so any attempt to spread them by
*moving* points either fails to separate them or breaks geometric locality and
raises fan-out. Point movement cannot beat the fan-out/skew trade-off.

## Positive result: selective replication + load-aware routing

The one lever that spreads load at *fixed* fan-out is **replication**: copy a hot
point onto several shards and let the router cover it from whichever replica is
least loaded. With single assignment a query's cover is forced, so a load-aware
router has no freedom — replication is what unlocks it. Four arms under
full-coverage routing on TEST queries (`replication_load_balance.py`), hot points
= top-`frac` by hubness, each copied onto its `replicas` nearest shard centroids:

- **A** single-assign + load-oblivious cover (baseline)
- **C** replication + load-oblivious cover
- **D** replication + load-aware cover (proposed)

(single-assign + load-aware = A, since the cover is forced.)

GloVe, target per-query recall 0.90 (P=46 shards):

| arm | storage | fan-out | load_skew | W | vs A |
|---|---|---|---|---|---|
| A single-assign / oblivious | 1.00x | 2.94 | 2.32 | 6.83 | — |
| C repl 5% R2 / oblivious | 1.05x | 2.75 | 1.89 | 5.22 | -24% |
| **D repl 5% R2 / load-aware** | 1.05x | 2.75 | 1.25 | 3.43 | **-50%** |
| C repl 10% R3 / oblivious | 1.20x | 2.56 | 1.93 | 4.94 | -28% |
| **D repl 10% R3 / load-aware** | 1.20x | 2.56 | 1.19 | 3.04 | **-55%** |
| **D repl 20% R3 / load-aware** | 1.40x | 2.35 | 1.21 | 2.84 | **-58%** |

Deep-image-96, target 0.90:

| arm | storage | fan-out | load_skew | W | vs A |
|---|---|---|---|---|---|
| A single-assign / oblivious | 1.00x | 2.15 | 1.51 | 3.25 | — |
| **D repl 5% R2 / load-aware** | 1.05x | 2.01 | 1.21 | 2.44 | **-25%** |
| **D repl 10% R3 / load-aware** | 1.20x | 1.87 | 1.14 | 2.14 | **-34%** |

Decomposition (isolating each effect): **A->C** replication lowers fan-out (more
coverage per shard) and slightly lowers skew; **C->D** load-aware routing cuts
skew at *identical* fan-out (GloVe 1.89->1.25, Deep 1.52->1.21) — the router is
the main lever and it is free in fan-out. Net `W` drops 25-58% for 1.05-1.40x
storage; even +5% storage roughly halves `W` on GloVe (≈2x throughput).

Two dependencies. (1) **Operating point**: at target 0.95 the cover is more
constrained, so the load-aware room shrinks (GloVe C->D skew 1.96->1.87) and the
win is smaller (`W` -29% at 5% R2, -38% at 10% R3), driven mostly by
replication's fan-out reduction rather than routing. (2) **Dataset hubness**: the
win scales with intrinsic load skew — GloVe (hubness skew ~468) gains -50..-58%,
Deep (hubness skew ~35) gains -25..-34%.

**Implication for Orion.** This validates Orion's multi-assign (replication)
direction, but shows the essential complement is a **load-aware router**:
oblivious routing over replicas (arm C) captures only ~half the benefit; picking
the least-loaded replica at query time (arm D) captures the rest at zero fan-out
cost. Selective (hubness-targeted) replication makes this cheap — a few percent
of storage, not blanket duplication.

## Reproduce (load balancing)

```bash
# in experiments/harness
# 1. estimate per-point hubness offline from TRAIN (LD_PRELOAD for hnswlib)
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
python ../../analysis/estimate_load.py --train-path data/<ds>/train.npy \
  --out layouts/loadbal/<ds>_hubness.npy --top-k 10 --space cosine --normalize
# 2. build a plain count-balanced base layout (no LD_PRELOAD, sklearn)
python ../../analysis/weighted_balanced_layout.py --train-path data/<ds>/train.npy \
  --shards 46 --out layouts/loadbal/<ds>_count.npz --normalize --seed 0
# 3. four-arm replication + load-aware routing probe
python ../../analysis/replication_load_balance.py --hdf5 <dataset.hdf5> \
  --train-path data/<ds>/train.npy --layout-file layouts/loadbal/<ds>_count.npz \
  --weights-path layouts/loadbal/<ds>_hubness.npy --normalize --top-k 10 \
  --target 0.90 --replicate-frac 0.05 0.10 0.20 --replicas 2 3 \
  --out ablation_reports/repl_<ds>.json
```

## Cluster reality check: the shard-level win needs shard ~= machine

The offline proxy `W = fan-out x load_skew` treats the busiest *shard* as the
bottleneck -- correct only when one shard maps to one machine. A physical
deployment usually has many shards per machine, and that changes the verdict.

Two mechanisms were carried to the containerized 4-peer cluster (46 shards, GloVe,
one shard per ~11 shards-per-peer). First an offline nav-router probe
(`nav_load_probe.py`) confirmed the win survives Orion's *real* navigation router
(not the oracle cover): replicating only the hot upper (L1) nodes onto their
nearest shards (+0.6% storage) and routing each L1 hit to its least-loaded replica
(consolidating onto already-probed shards first) cut shard `W` ~37% at matched
recall. Then it was deployed: a single-assign base collection (arm A, oblivious
nav router) and a hot-L1-replicated collection (arm D, `--load-aware` nav router),
both built from the same k-means base via copy-id encoding, swept over upper-k at
server saturation (~90% busiest peer).

Result: **arm D is slightly worse on the cluster**, QPS/A = 0.87-0.99 at matched
Recall@10 (GloVe):

| matched recall | A QPS (single, oblivious) | D QPS (repl, load-aware) | D/A |
|---|---|---|---|
| 0.67 | 5281 | 5226 | 0.99 |
| 0.70 | 5132 | 4704 | 0.92 |
| 0.72 | 4910 | 4292 | 0.87 |
| 0.74 | 4461 | 3875 | 0.87 |

Two reasons, both measured. (1) **Peer aggregation erases the skew lever.** With
46 shards over 4 peers, each peer averages ~11-12 shards, so per-*peer* CPU is
already balanced (busiest/least peer ratio ~0.87 for A) even though per-*shard*
skew offline was 2.95. Load-aware routing does flatten peers a little further
(busiest peer 91.7% -> 87.2%, spread 12.5pt -> 4.6pt at uk=16), but there was
almost no imbalance headroom to recover. (2) **Cheap L1-only replication costs
recall.** Consolidating an L1 hit onto another shard re-routes the probe away from
the home shard where that node's data neighbors actually live, so D reaches a
given recall only at a higher upper-k -- more fan-out and ef, i.e. more total work
per query. With the load-balance benefit near zero at peer granularity, only this
recall/work penalty remains, and D loses by up to ~13%.

Implications. The replication + load-aware-routing win is real but **regime-
dependent**: it needs the bottleneck to be a single hot shard, i.e. shard
granularity ~= machine granularity (few shards per peer, or many peers). On a
cluster with many shards per machine the aggregate already balances load, and the
only lever left -- reducing per-query work -- is better served by cutting fan-out
directly (multi-assign that replicates *data* neighborhoods, not just L1 entry
points) than by L1-only replication, which trades recall. Reproduce:

### Follow-up: the shard == machine regime (16 peers, one shard each)

The default deployment is one shard per machine; the 46-shard/4-peer run above
only *simulated* multi-machine and its per-peer aggregation was an artifact. So
the experiment was rebuilt faithfully: a 16-node cluster (`docker-compose.
cluster16.yaml`, one Qdrant peer per shard, 6 disjoint cores each, load generator
on 96-111), P=16 layouts, and the same A (single-assign, oblivious) vs D
(hot-L1 replication +0.6%, `--load-aware`) arms, swept over upper-k at saturation.

Now the shard skew *does* surface as machine skew -- arm A's busiest peer runs
~2x the least loaded (peer imbalance min/max ~0.50) -- and load-aware routing
visibly flattens it (imbalance ~0.50 -> ~0.68, busiest-peer utilization 89% ->
82% at matched upper-k). **But arm D still loses at matched recall**, D/A ~0.82
QPS (GloVe, Recall@10 in the overlap 0.63-0.70), i.e. ~18% worse -- no better than
the 4-peer run. Two measured reasons:

- At matched upper-k the throughputs are nearly equal (D +~1%: 11.9k vs 12.1k QPS
  at uk=8), even though D balances the peers, because both arms plateau at ~12k
  QPS with the busiest peer only 82-89% utilized. The residual ceiling is the
  scatter-gather **coordinator**, not the hottest shard, so flattening shard load
  buys almost nothing here.
- The L1-only consolidation still costs recall (D reaches 0.65 where A reaches
  0.70 at the same uk), and buying that recall back needs a higher upper-k -- more
  fan-out and ef -- which erases the ~1% balance gain and then some.

### The lever that wins: data multi-assign + a consolidating router

The right lever is replicating **data neighborhoods**, not entry points. A
boundary point (2nd-nearest shard centroid nearly as close as its home) is copied
onto that 2nd/3rd shard until a copy budget is spent (`multiassign_layout.py`,
here 1.4x storage on the same P=16 base). Now a query's true neighbors live in
more shards, so the navigation router covers them in **fewer probes at the same
recall** -- and, unlike L1-only replication, this *raises* recall per probe rather
than lowering it.

The router matters. Probing every shard that owns a routed L1 hit (oblivious, arm
E) inflates fan-out and wins nothing. The `--load-aware` router assigns each L1
hit to a single shard, consolidating onto already-probed shards -- which the dense
replicas make easy -- so fan-out drops. Arm E-LA (multi-assign + `--load-aware`)
**dominates arm A at matched Recall@10** on the 16-node cluster:

| matched recall | A QPS | E-LA QPS | E-LA/A |
|---|---|---|---|
| 0.66 | 13878 | 15240 | 1.10 |
| 0.69 | 12344 | 14235 | 1.15 |
| 0.72 | 10723 | 12783 | 1.19 |

At matched upper-k E-LA dominates outright -- higher recall *and* higher QPS
(uk=8: recall 0.730 vs 0.703, QPS 12233 vs 11625) -- because data replication cuts
fan-out (uk=8 mean fan-out 2.8 vs A's 3.1) at higher coverage. The gain grows with
the recall target (+10% at 0.66 to +19% at 0.72) and costs 1.4x storage. This is
the fan-out lever the earlier sections pointed to, now confirmed end-to-end.

Conclusion. **Hot-L1-entry-point replication + load-aware nav routing is not a
throughput win**; the load-balancing lever it targets is either absorbed by
per-machine aggregation (many shards/peer) or, when it does exist (shard ==
machine), too small next to the coordinator ceiling and the recall cost of
re-routing entry points away from where the data lives. **Data multi-assign plus a
consolidating load-aware router is the win** (+10-19% QPS at matched recall,
1.4x storage), because it cuts per-query fan-out at fixed or better recall.

Reproduce the multi-assign win (P16, 16-node cluster up):

```bash
# in experiments/harness
python multiassign_layout.py --train-path data/glove/train.npy \
  --base-layout layouts/loadbal/glove_count_p16.npz --normalize \
  --expansion 1.4 --out layouts/loadbal/glove_multi_p16.npz
python upload_pts.py --collection glove_multi_p16 --train-path data/glove/train.npy \
  --layout-file layouts/loadbal/glove_multi_p16.npz --distance Cosine
# E-LA arm: --load-aware is essential (oblivious probes all replicas -> no win)
for uk in 4 6 8 12; do
  LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 python build_routed_requests.py \
    --router navigation --load-aware --hdf5 <glove.hdf5> --collection glove_multi_p16 \
    --vector-distance cosine --layout-file layouts/loadbal/glove_multi_p16.npz \
    --upper-k $uk --out req/repl16/gELA_navk${uk}.jsonl
  python measure.py --collection glove_multi_p16 --mode routed \
    --requests-file req/repl16/gELA_navk${uk}.jsonl --output-dir runs/p16_gELA_k${uk} \
    --client-cores 96-111 --procs 8 --inflight 1024 --top-k 10 --repeat 4
  python score_recall.py --output-dir runs/p16_gELA_k${uk} \
    --ground-truth data/glove/ground_truth.npy --top-k 10
done
```

Reproduce the (negative) L1-only arms (below is the P46 form; for P16 bring up
`tools/compose/docker-compose.cluster16.yaml` and use `layouts/loadbal/
glove_{count,replkm}_p16.npz`, client cores `96-111`, `--inflight 1024+`):

```bash
# in experiments/harness (LD_PRELOAD = system libstdc++ for hnswlib)
# offline nav-router de-risk + dump the replicated layout
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 python nav_load_probe.py \
  --hdf5 <ds.hdf5> --train-path data/<ds>/train.npy \
  --layout-file layouts/loadbal/<ds>_count.npz --vector-distance cosine --normalize \
  --upper-k 8 12 16 24 --replicate-frac 0.10 --replicas 3 \
  --dump-layout layouts/loadbal/<ds>_replkm_p46.npz
# upload both arms (copy-id encoded, round-robin over peers)
python upload_pts.py --collection <ds>_countkm_p46 --train-path data/<ds>/train.npy \
  --layout-file layouts/loadbal/<ds>_count.npz --distance Cosine
python upload_pts.py --collection <ds>_replkm_p46 --train-path data/<ds>/train.npy \
  --layout-file layouts/loadbal/<ds>_replkm_p46.npz --distance Cosine
# per upper-k: A = oblivious nav router, D = --load-aware nav router; sweep, then
# measure at saturating inflight and score recall offline (see runs/ for outputs)
```
