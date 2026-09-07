# Deploying and running the throughput tests — a new developer's guide

This is the hands-on runbook. For *why* the harness is built this way, read
`STAGE0.md` (the testbed and validity gates) and `STAGE1.md` (the routed serving
path and the comparison protocol). This document gets you from a fresh checkout to
a gate-passing matched-recall QPS number, on one host or several.

The one idea to keep in mind: **a throughput number only counts if the server is
the bottleneck.** Every step below exists to keep the load generator, routing, and
recall scoring *out* of the timed path, and the four gates (G1–G4) are how you
prove it. A run with `gates_passed: false` is not a slow result — it is a
non-result.

---

## 0. What you will measure

Three ways of serving the same query, compared at matched Recall@10:

| Arm | How the query fans out | Built by | Measured by |
|---|---|---|---|
| **broadcast** | server hits every shard (fan-out = P) | `load_collection.py` or the two-level tool | `measure.py --mode broadcast` |
| **k-means + centroid router** | probe the `nprobe` nearest shards | two-level tool (`kmeans_*`) | `measure.py --mode routed` |
| **Orion + navigation router** | probe the shards the upper graph navigates to | two-level tool (`faithful_original_rest`) | `measure.py --mode routed` |

The hypothesis (from the offline analysis in `analysis/FINDINGS.md`): at the same
recall, Orion reaches it at lower fan-out, so it sustains higher QPS. Throughput
tracks `P / (fan-out · load_skew · log|shard|)`, and fan-out is the first-order term.

---

## 1. Prerequisites

- **A Qdrant cluster of the Orion fork**, one container per peer, each pinned to a
  cpuset. On the current single host there are four: names match `qdrant-controller*`
  (`docker ps`). The entry-point / per-shard-ef features that routed serving relies
  on exist only in this fork's patched Qdrant, not stock Qdrant.
- **Python deps**: `numpy`, `h5py`, `orjson`, `aiohttp`, `requests`, and `hnswlib`
  (for building the upper graph in the request builder).
- **hnswlib needs the system libstdc++ ahead of conda's.** Prefix every command
  that imports it (`build_routed_requests.py`) with:
  ```
  LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
  ```
- **Datasets** (ann-benchmarks HDF5) live in `/home/taig/dry/faiss/datasets/`:
  `glove-200-angular.hdf5`, `coco-t2i-512-angular.hdf5`, `sift-128-euclidean.hdf5`,
  `deep-image-96-angular.hdf5`. Angular datasets are served with **cosine** and must
  be unit-normalized.
- **A quiet host.** Agent/operator/co-tenant activity counts against gate G4. Close
  other heavy containers/VMs on any participating host.
- **Multi-host only**: passwordless SSH (`BatchMode`) from the client host to every
  server host, with docker usable by that SSH user (needed to read each peer's CPU).

Run everything from `experiments/harness/`.

---

## 2. One-time testbed setup

**Make every peer identical and hold total server cores constant.** Asymmetric
peers can never pass G1 (server saturation).

```bash
cd experiments/harness
python configure_testbed.py --cores-per-peer 8 --first-core 0
# writes cpuset_backup.json; restore later with:  python configure_testbed.py --restore
```

**Pick client cores disjoint from all peer cpusets.** With peers on 0–31 and 112
cores total, the load generator can use e.g. 32–63. `measure.py` refuses overlap.

**Prepare the dataset** (decode HDF5 to npy once, outside any timed path):

```bash
python prepare_dataset.py \
  --hdf5 /home/taig/dry/faiss/datasets/glove-200-angular.hdf5 \
  --out-dir data/glove --normalize        # --normalize for angular/cosine
# -> data/glove/{train,queries,ground_truth}.npy
```

---

## 3. Build the collections

Each arm needs its own collection. **Whatever build params you choose here, the
request builder in §4 must be told the same ones** (or, better, use
`--from-collection` so membership is recovered from what is actually stored).

### 3a. Broadcast baseline (hash sharding, quick)

```bash
python load_collection.py \
  --collection glove_broadcast_p8 --train-path data/glove/train.npy \
  --shards 8 --distance Cosine --recreate
```

This auto-distributes points by hash and lets the server fan out to all shards.
Good enough as the fan-out = P reference.

### 3b. Routed arms (custom shard keys, via the two-level tool)

Custom-sharded collections (one `shard_key` per point, placed by the layout) are
built by `tools/qdrant_two_level_routing_experiment.py` via `--routing-mode`:

| `--routing-mode` | Arm |
|---|---|
| `faithful_original_rest` | **Orion** (topology-aware placement + upper graph) |
| `cpp_kmeans_baseline` | balanced k-means |
| `kmeans_simple_nprobe` | plain k-means |
| `naive_hash_all_shards` | hash / broadcast |

Representative Orion build (run `--help` for the full arg list; the tool also runs
its own *client-bound* benchmark afterwards — ignore those QPS numbers, only the
collection it creates matters):

```bash
python ../../tools/qdrant_two_level_routing_experiment.py \
  --base-url http://127.0.0.1:6833 --collection orion_glove_p8 \
  --hdf5-path /home/taig/dry/faiss/datasets/glove-200-angular.hdf5 \
  --vector-distance cosine --num-shards 8 --routing-mode faithful_original_rest
# k-means baselines: same command with --routing-mode cpp_kmeans_baseline (balanced)
#                    or kmeans_simple_nprobe (plain)
```

Leave the upper-graph params at their defaults (`--sample-denominator 32`,
`--upper-sample-seed 100`, `--upper-m 32`, `--upper-ef-construction 100`) — the
request builder defaults to the same values, so routing will reproduce. The tool
stores a `source_id` payload and uses `source_id_dedup_block_size = num_points+1`
by default; the builder defaults match, so don't override unless you did here.

---

## 4. Build the routed request files (offline, outside the timer)

For each routed arm and each value of the routing knob you want on the frontier.
Use `--from-collection` for the Orion arm: it scrolls the live collection to
recover L1 shard membership, eliminating any layout/collection drift.

```bash
# Orion navigation arm — sweep --upper-k (e.g. 40 60 80 120)
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 python build_routed_requests.py \
  --hdf5 /home/taig/dry/faiss/datasets/glove-200-angular.hdf5 \
  --collection orion_glove_p8 --vector-distance cosine \
  --router navigation --from-collection --num-shards 8 --base-url http://127.0.0.1:6833 \
  --upper-k 60 --out req/orion_glove_p8_k60.jsonl

# k-means centroid arm — sweep --nprobe (needs a layout .npz for centroids)
LD_PRELOAD=… python build_routed_requests.py \
  --hdf5 /home/taig/dry/faiss/datasets/glove-200-angular.hdf5 \
  --collection kmeans_glove_p8 --vector-distance cosine \
  --router centroid --layout-file ../../analysis/layouts/kmeans_glove_p8.npz \
  --nprobe 3 --out req/kmeans_glove_p8_np3.jsonl
```

Each run also writes `<out>.stats.json` with the fan-out and total-ef distribution
— record these; they are what you regress QPS against.

> The centroid arm needs a layout `.npz` (point→shard) to compute per-shard
> centroids. Orion/kmeans layouts are produced by `analysis/build_orion_layout.py`;
> the navigation arm does not need one when using `--from-collection`.

---

## 5. Measure (timed, gated)

Same command shape for every arm; only the body source differs.

```bash
# routed arm
python measure.py --collection orion_glove_p8 --mode routed \
  --requests-file req/orion_glove_p8_k60.jsonl \
  --output-dir runs/orion_glove_p8_k60 \
  --client-cores 32-63 --procs 8 --inflight 96 --warmup 2000 --top-k 10

# broadcast baseline — sweep --hnsw-ef instead of a routing knob
python measure.py --collection glove_broadcast_p8 --mode broadcast \
  --queries-path data/glove/queries.npy --hnsw-ef 128 \
  --output-dir runs/glove_bcast_ef128 \
  --client-cores 32-63 --procs 8 --inflight 96 --warmup 2000 --top-k 10
```

`measure.py` prints a summary and writes `runs/<name>/summary.json` with `qps`,
`per_peer_cpu_pct`, and the gates. Exit code 2 and a loud message mean a gate
failed.

---

## 6. Score recall and build the frontier

Recall is scored offline from persisted ids — never in the timed path.

```bash
python score_recall.py --output-dir runs/orion_glove_p8_k60 \
  --ground-truth data/glove/ground_truth.npy --top-k 10
```

Repeat §4–6 across the swept knob for each arm. Keep only runs with
`gates_passed: true`. Plot QPS vs Recall@10 per arm; the headline is **QPS at
matched Recall@10 ≥ 0.90** (and 0.95). Confirming result: at matched recall,
Orion navigation > k-means centroid > broadcast, in the same order as fan-out.

Suggested matrix: **GloVe-200** and **coco-t2i-512** (where Orion should win),
with **SIFT-1M as a negative control** (near-uniform; Orion is not expected to
win). Hold total server cores fixed as you vary shard count P ∈ {4, 8, 16, 32}.

---

## 7. Multi-host deployment

The load path is host-agnostic — peers are reached at `http://<host>:<port>`. A
multi-machine cluster is just a longer host list.

1. **Bring up peers on each server host** (one container per peer, pinned cpusets),
   and make cpusets **symmetric across all hosts** at fixed total cores.
2. **Point discovery at every host.** Pass `--hosts` to `measure.py`:
   ```bash
   python measure.py --hosts local,user@srv2,user@srv3 --coordinator all \
     --collection orion_glove_p8 --mode routed --requests-file req/... \
     --output-dir runs/... --client-cores 0-31 --procs 8 --inflight 128 --top-k 10
   ```
   Each peer's CPU is read where it runs — locally by file, remotely over `ssh` —
   so **G1 and G3 keep their single-host fidelity**. This needs passwordless SSH and
   docker access for that user on each server host.
3. **Real network, real scale-out.** On multi-host the loopback caveat disappears
   and physical scale-out becomes measurable (the axis Stage 0 had to rule out).
   Record the interconnect (bandwidth, latency).
4. **Keep every host quiet** (G4 is computed on the client host; co-tenants on a
   server host count against that host's saturation).
5. When generating requests with `--from-collection`, set `--base-url` to a
   reachable coordinator (`http://srv1:6833`).

---

## 8. Reading the gates (from `summary.json`)

| Gate | Passes when | Meaning |
|---|---|---|
| **G1** server saturation | max peer CPU ≥ 85% | some server component is the bottleneck |
| **G2** client headroom | client CPU < 50% | the load generator is not the bottleneck |
| **G3** load-generator headroom | doubling `--client-cores` moves QPS < 5% | *decisive* — proves not client-bound |
| **G4** host interference | non-measurement host load < 5% | the host is quiet |

G3 needs two runs (e.g. `--client-cores 32-47` then `32-63`); if more client cores
raise QPS, the reading is client-bound and invalid regardless of G1/G2. `peer_imbalance_ratio`
is a diagnostic, not a gate — the coordinator peer typically saturates before data
peers.

---

## 9. Troubleshooting

- **`ImportError`/segfault importing hnswlib** → you forgot the `LD_PRELOAD` prefix.
- **`client cores overlap <peer>`** → choose `--client-cores` disjoint from peer
  cpusets (32–63 on this host).
- **`peers have unequal core counts`** → run `configure_testbed.py`, or pass
  `--allow-asymmetric` for a throwaway smoke (never for a reported number).
- **Recall far below expectation** → the builder's upper-graph params or
  `--source-id-dedup-block-size` don't match how the collection was built. Rebuild
  requests with `--from-collection` and the same seeds/M/ef used at build time.
- **A gate fails** → do not interpret the QPS. Fix the cause (quiet the host for G4,
  raise `--inflight` for G1, add client cores for G2) and re-run.
- **`--from-collection` errors "recovered no shard membership"** → wrong
  `--num-shards`, wrong collection, or the upper-graph seeds don't match, so the
  recovered L1 point ids aren't the ones stored.

---

## 10. The exact loop, condensed

```bash
cd experiments/harness
python configure_testbed.py --cores-per-peer 8                      # once
python prepare_dataset.py --hdf5 <ds>.hdf5 --out-dir data/<ds> --normalize
# build collections (§3), then per arm and per knob:
LD_PRELOAD=… python build_routed_requests.py --router <navigation|centroid> … --out req/<name>.jsonl
python measure.py --collection <c> --mode <routed|broadcast> … --output-dir runs/<name>
python score_recall.py --output-dir runs/<name> --ground-truth data/<ds>/ground_truth.npy
# keep runs where gates_passed==true; plot QPS vs recall per arm.
```
