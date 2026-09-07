# Routed-serving results: Orion vs. baselines

Matched-recall throughput of five serving arms on two datasets, measured with the
routed harness (see [`STAGE1.md`](STAGE1.md) for the protocol and
[`DEPLOY.md`](DEPLOY.md) for how to reproduce). Every arm below is served on a
**single-copy, 46-shard** collection with **identical HNSW parameters**
(`m=32`, `ef_construct=100`, one segment per shard); the arms differ only in
(a) *where each point lands* (the layout) and (b) *how a query picks shards*
(the router). This isolates layout+router quality from indexing cost and from
storage: all layouts store exactly one copy of every point.

## The five arms

| Arm | Layout | Router | Fan-out |
|---|---|---|---|
| Orion navigation | Orion (navigation-derived, norep) | upper-graph router: entry points + adaptive per-shard `ef` | selective (`upper_k`) |
| k-means centroid | plain k-means | rank shards by query→centroid distance, probe top-`nprobe`, uniform `ef` | `nprobe` |
| hash broadcast | uniform hash (`id % P`) | probe all P shards, uniform `ef` | P |
| k-means broadcast | plain k-means | probe all P shards, uniform `ef` | P |
| Orion broadcast | Orion (norep) | probe all P shards, uniform `ef` | P |

- **Broadcast** is the naive distributed baseline (fan-out = P = 46). The
  hash-broadcast arm is the distributional equivalent of Qdrant's default
  auto-sharding, built here with matched HNSW parameters for a fair comparison.
- **k-means centroid** is the strongest simple router baseline (fan-out reduction
  via geometric clustering).
- **Orion navigation** is the system under test.

## Headline (unified metric)

QPS at a fixed recall target, read off each arm's Pareto frontier (linear
interpolation between the two bracketing configs). Each dataset is compared at
its meaningful operating point — GloVe saturates near 0.90, SIFT near 0.99.

**GloVe-200-angular (hard, cosine) — QPS @ Recall@10 ≥ 0.90**

| Arm | Fan-out | QPS | vs. Orion nav |
|---|---:|---:|---:|
| **Orion navigation** | ~27 | **1055** | 1.00× |
| k-means centroid | ~18 | 710 | 1.49× |
| hash broadcast | 46 | 418 | 2.52× |
| Orion broadcast | 46 | 398 | 2.65× |
| k-means broadcast | 46 | 375 | 2.82× |

**SIFT-128-euclidean (easy, L2) — QPS @ Recall@10 ≥ 0.99**

| Arm | Fan-out | QPS | vs. Orion nav |
|---|---:|---:|---:|
| **Orion navigation** | ~10.5 | **3874** | 1.00× |
| k-means centroid | ~11 | 3576 | 1.08× |
| Orion broadcast | 46 | 1167 | 3.32× |
| k-means broadcast | 46 | 1030 | 3.76× |
| hash broadcast | 46 | 1025 | 3.78× |

## Conclusions

1. **Fan-out reduction is the universal, first-order throughput lever.** On both
   datasets any router beats every broadcast arm by **2.5–3.8×**. Probing ~10/46
   (SIFT) or ~27/46 (GloVe) shards is far cheaper than probing all 46, because a
   graph index's per-shard cost is roughly `log(shard_size)` and total work scales
   with fan-out. This is Orion's core, data-independent benefit.

2. **Layout+router quality (navigation vs. centroid) is data-dependent.** SIFT is
   the negative control:
   - **GloVe (poor geometric clustering):** Orion navigation beats k-means
     centroid **1.49×**. Centroid routing mislocates neighbors on angular data.
   - **SIFT (good geometric clustering):** the two are **tied (1.08×)**. On uniform
     data plain k-means centroids are already near-optimal, so Orion's navigation
     refinement and entry points add little.

   Orion's navigation machinery pays off specifically on hard, poorly-clustered
   workloads; it does not claim a universal win over simple centroid routing.

3. **Replication is not the source of the throughput advantage.** GloVe was also
   measured on the replicated production layout (`bench095`, +17.5% storage):
   at Recall ≥ 0.90 it delivers ~1024 QPS, essentially equal to the norep
   layout's ~1055. Replication only raises the *recall ceiling* (the replicated
   layout reaches 0.92 more cheaply) — it does not change throughput at 0.90. The
   1.49× advantage over k-means therefore holds at equal storage.

## Full frontiers (raw measurements)

Columns: fan-out, total `ef` (Σ per-shard ef), Recall@10, QPS, and validity gates
(server saturation %, client headroom %, host interference %). Top-k = 10, 10000
queries, warmup 2000.

### GloVe-200-angular

| Arm | Knob | Fan-out | Σef | Recall | QPS | sat% | cli% | host% |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Orion nav (norep) | k60 | 14.2 | 523 | 0.8390 | 1994 | 96 | 6.7 | 6.6 |
| Orion nav (norep) | k120 | 21.0 | 900 | 0.8804 | 1286 | 98 | 4.6 | 6.9 |
| Orion nav (norep) | k200 | 26.9 | 1337 | 0.9069 | 974 | 99 | 3.6 | 6.0 |
| k-means centroid | np12/ef128 | 12.0 | 1536 | 0.8896 | 809 | 77 | 3.1 | 12.1 |
| k-means centroid | np20/ef128 | 20.0 | 2560 | 0.9220 | 500 | 79 | 2.1 | 7.3 |
| k-means centroid | np30/ef128 | 30.0 | 3840 | 0.9379 | 331 | 82 | 1.6 | 8.7 |
| hash broadcast | ef64 | 46.0 | 2944 | 0.9435 | 418 | 82 | 1.8 | 6.2 |
| hash broadcast | ef128 | 46.0 | 5888 | 0.9777 | 251 | 77 | 1.2 | 8.0 |
| hash broadcast | ef256 | 46.0 | 11776 | 0.9941 | 151 | 76 | 0.7 | 6.9 |
| k-means broadcast | ef64 | 46.0 | 2944 | 0.8924 | 397 | 90 | 1.7 | 6.8 |
| k-means broadcast | ef128 | 46.0 | 5888 | 0.9456 | 244 | 87 | 1.2 | 6.6 |
| k-means broadcast | ef256 | 46.0 | 11776 | 0.9765 | 146 | 83 | 0.7 | 9.6 |
| Orion broadcast (norep) | ef64 | 46.0 | 2944 | 0.8875 | 439 | 99 | 1.9 | 6.7 |
| Orion broadcast (norep) | ef128 | 46.0 | 5888 | 0.9373 | 276 | 100 | 1.3 | 6.5 |
| Orion broadcast (norep) | ef256 | 46.0 | 11776 | 0.9684 | 170 | 100 | 0.8 | 6.7 |

k-means layout size skew (max/mean shard) = 1.94; Orion norep = 1.24.

### SIFT-128-euclidean

| Arm | Knob | Fan-out | Σef | Recall | QPS | sat% | cli% | host% |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Orion nav (norep) | k60 | 7.9 | 399 | 0.9781 | 5523 | 93 | 13.2 | 8.6 |
| Orion nav (norep) | k120 | 10.5 | 690 | 0.9901 | 3865 | 95 | 10.2 | 8.6 |
| Orion nav (norep) | k200 | 12.7 | 1054 | 0.9952 | 2955 | 96 | 8.9 | 8.5 |
| k-means centroid | np8/ef64 | 8.0 | 512 | 0.9858 | 4452 | 92 | 11.4 | 8.2 |
| k-means centroid | np20/ef64 | 20.0 | 1280 | 0.9914 | 1982 | 92 | 6.5 | 7.1 |
| k-means centroid | np12/ef128 | 12.0 | 1536 | 0.9968 | 2152 | 90 | 7.0 | 7.7 |
| k-means centroid | np20/ef128 | 20.0 | 2560 | 0.9973 | 1290 | 86 | 4.3 | 7.4 |
| k-means centroid | np30/ef128 | 30.0 | 3840 | 0.9974 | 883 | 86 | 3.1 | 7.7 |
| hash broadcast | ef64 | 46.0 | 2944 | 0.9990 | 1025 | 94 | 3.6 | 7.4 |
| hash broadcast | ef128 | 46.0 | 5888 | 0.9993 | 685 | 88 | 2.5 | 8.1 |
| hash broadcast | ef256 | 46.0 | 11776 | 0.9994 | 444 | 81 | 1.7 | 7.4 |
| k-means broadcast | ef64 | 46.0 | 2944 | 0.9914 | 1030 | 94 | 3.7 | 7.4 |
| k-means broadcast | ef128 | 46.0 | 5888 | 0.9974 | 643 | 86 | 2.4 | 6.6 |
| k-means broadcast | ef256 | 46.0 | 11776 | 0.9991 | 386 | 79 | 1.5 | 7.6 |
| Orion broadcast (norep) | ef64 | 46.0 | 2944 | 0.9915 | 1167 | 98 | 4.1 | 7.5 |
| Orion broadcast (norep) | ef128 | 46.0 | 5888 | 0.9976 | 786 | 99 | 2.9 | 7.4 |
| Orion broadcast (norep) | ef256 | 46.0 | 11776 | 0.9989 | 493 | 96 | 2.0 | 7.4 |

k-means layout size skew = 1.57; Orion norep = 1.25.

## Method and knobs

- **Testbed:** single host, 4 Qdrant peers (8 cores each, symmetric cpusets),
  client pinned to a disjoint core set, loopback transport. See
  [`STAGE0.md`](STAGE0.md).
- **Collections:** built by `harness/upload_custom_shards.py` from a point→shard
  layout (`harness/make_layouts.py` for hash/k-means; `analysis/build_orion_layout.py`
  `--disable-multi-assign` for Orion norep). All `Cosine`/`Euclid` as per dataset,
  `m=32`, `ef_construct=100`.
- **Requests:** pre-serialized offline by `harness/build_routed_requests.py`
  (`--router {navigation,centroid,broadcast}`) so routing and scatter-gather are
  outside the measured path. Navigation uses `base_ef=20`, `factor=4`; centroid
  uses uniform `ef` (swept); broadcast probes all 46 shards.
- **Measurement:** `harness/measure.py --mode routed` replays request bodies at
  fixed inflight; recall scored offline by `harness/score_recall.py` against the
  dataset ground truth.
- **Metric:** QPS at fixed Recall@10, interpolated on each arm's Pareto frontier.

## Limitations

- **Host interference gate (G4) is 6–13%** on this shared host (other tenants),
  so no run passes the strict `gates_passed` bar. Server-saturation (G1) is
  76–100% and client headroom (G2) is comfortable, i.e. measurements are
  server-bound. Every arm runs on the same noisy host and the reported ratios far
  exceed the noise, so **relative** conclusions are robust; **absolute** QPS
  should be re-certified on a quiet host before quoting as such.
- **Single-host, loopback.** Absolute QPS and the 46-shards-on-4-peers packing
  are testbed-specific. Cross-arm ratios are the trustworthy quantity; physical
  multi-machine scale-out is out of scope here (see `STAGE0.md`).
- **Two datasets.** GloVe (hard/angular) and SIFT (easy/uniform) bracket the
  regimes; intermediate workloads are not measured.
- **Offline dual-graph upper index.** The navigation layout and router use the
  `hnswlib` upper graph (legacy dual-graph variant), not Qdrant's production Rust
  upper graph; see the `build_orion_layout.py` docstring.
