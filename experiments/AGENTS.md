# Start here: reproducing the routed-serving experiments

You (an agent or a new developer) are here to reproduce, on a possibly new
machine, the throughput comparison in [`RESULTS.md`](RESULTS.md): **Orion's
routed serving vs. hash / k-means baselines**, measured at matched Recall@10.
This file is the map. Read it fully before running anything.

## What the experiment shows (so you know when you've succeeded)

Five serving arms on one **single-copy, 46-shard** collection each (identical
HNSW params; only the layout and the router differ). Headline results:

- **Routing beats broadcast ~2.5–3.8×** on both datasets (fan-out is the
  first-order throughput lever).
- **Orion navigation vs. k-means centroid is data-dependent:** ~**1.49×** on
  GloVe (hard, angular), ~**1.08× (a tie)** on SIFT (easy, uniform — the negative
  control). Orion's navigation machinery pays off only on poorly-clustered data.
- **Equal storage:** the Orion arm is single-copy, so the win is not from
  replication.

If your numbers land near the frontier tables in `RESULTS.md`, you've reproduced
it. If routing does *not* beat broadcast, something in the setup is wrong (usually
a gate failure or a stale request file) — do not report it as a result.

## Read in this order

1. [`ENVIRONMENT.md`](ENVIRONMENT.md) — bring up the 4-peer Orion cluster, install
   Python deps, download/decode datasets. **The #1 thing that blocks a fresh
   machine.**
2. [`STAGE0.md`](STAGE0.md) — the testbed and the four validity gates (G1–G4). A
   run with `gates_passed: false` is a non-result, not a slow result.
3. [`STAGE1.md`](STAGE1.md) — the routed serving path and the comparison protocol.
4. [`DEPLOY.md`](DEPLOY.md) — the exact command sequence: layouts → collections →
   request files → measure → score.
5. [`RESULTS.md`](RESULTS.md) — the expected numbers; your pass/fail check.

## The pipeline in one breath

`make_layouts.py` / `build_orion_layout.py` → `upload_custom_shards.py` →
`build_routed_requests.py` (`--router broadcast|centroid|navigation`) →
`measure.py --mode routed` → `score_recall.py`. All scripts live in
`experiments/harness/` (except `build_orion_layout.py` in `analysis/`). Layouts
and collections are rebuilt from deterministic seeds — **you do not need any data
from the origin machine.**

## Machine-specific values to check/change on a new host

The harness auto-discovers the cluster (`docker ps --filter name=qdrant-controller`),
so most of this is verification, not editing. But confirm each:

| What | Origin value | How to adapt |
|---|---|---|
| Repo commit | `da2ee17dc` on `stage1-routed-harness` | `git checkout` the same |
| Controller URL | `http://127.0.0.1:6833` | must be reachable; discovery derives it |
| Server cpusets | `0-7,8-15,16-23,24-31` (4×8) | `configure_testbed.py --cores-per-peer 8 --first-core 0` |
| Client cores | `32-63` | `measure.py --client-cores`, disjoint from peers |
| `libstdc++` path | `/usr/lib/x86_64-linux-gnu/libstdc++.so.6` | `LD_PRELOAD` prefix; `find / -name 'libstdc++.so.6*'` |
| Dataset dir | `/home/taig/dry/faiss/datasets/` | wherever you downloaded the HDF5s |
| Server image | `qdrant/qdrant:method4-peer-premerge` | copy or rebuild the fork image |

## Sanity checks before trusting a number

- `curl -s http://127.0.0.1:6833/cluster` → status `enabled`, 4 peers.
- All peers have **equal core counts** (`configure_testbed.py` enforces it).
- k-means layout size skew prints ~**1.94** (GloVe) / ~**1.57** (SIFT); Orion
  norep ~**1.24**. Wrong skew ⇒ wrong dataset or seed.
- Request `<out>.stats.json` fan-out matches `RESULTS.md` (broadcast = 46;
  navigation ~14–27 on GloVe, ~8–13 on SIFT; centroid = `nprobe`).
- `runs/<name>/summary.json`: G1 (server saturation) high, G2 (client headroom)
  comfortable. On a shared host G4 may fail (~6–13%); relative ratios still hold,
  absolute QPS does not — re-certify on a quiet host.

## Gotchas that have bitten us

- **Forgetting `LD_PRELOAD`** → hnswlib import segfault/ImportError.
- **`make_layouts.py --out-dir layouts`** overwrites GloVe's `hash_p46.npz`/
  `kmeans_p46.npz` when you then run SIFT — use a per-dataset subdir.
- **Centroid arm at too-low `--hnsw-ef`** caps recall; sweep ef, not just nprobe.
- **`--source-id-dedup-block-size`** must be `0` for the single-copy collections
  in `DEPLOY.md` (raw ids). A mismatch silently wrecks recall.
