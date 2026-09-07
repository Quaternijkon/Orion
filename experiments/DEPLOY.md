# Deploying and running the throughput tests — a new developer's guide

This is the hands-on runbook. Before it, set up the machine with
[`ENVIRONMENT.md`](ENVIRONMENT.md) (cluster bring-up, Python deps, datasets). For
*why* the harness is built this way, read `STAGE0.md` (the testbed and validity
gates) and `STAGE1.md` (the routed serving path and the comparison protocol). The
exact numbers this runbook reproduces are in [`RESULTS.md`](RESULTS.md). This
document gets you from a healthy cluster to a matched-recall QPS number, on one
host or several.

The one idea to keep in mind: **a throughput number only counts if the server is
the bottleneck.** Every step below exists to keep the load generator, routing, and
recall scoring *out* of the timed path, and the four gates (G1–G4) are how you
prove it. A run with `gates_passed: false` is not a slow result — it is a
non-result.

---

## 0. What you will measure

**Five arms**, each on its own collection, compared at matched Recall@10. Every
collection is **single-copy, P = 46 shards, identical HNSW params** (`m=32`,
`ef_construct=100`); arms differ only in *where points land* (layout) and *how a
query picks shards* (router). This isolates layout+router quality from indexing
cost and from storage.

| Arm | Layout | Router | Fan-out | Request `--router` |
|---|---|---|---|---|
| **Orion navigation** | Orion norep | upper-graph: entry points + adaptive ef | selective (`--upper-k`) | `navigation` |
| **k-means centroid** | plain k-means | top-`nprobe` nearest centroids, uniform ef | `--nprobe` | `centroid` |
| **hash broadcast** | uniform hash | all P shards, uniform ef | P | `broadcast` |
| **k-means broadcast** | plain k-means | all P shards, uniform ef | P | `broadcast` |
| **Orion broadcast** | Orion norep | all P shards, uniform ef | P | `broadcast` |

The hypothesis: at the same recall, routing reaches it at lower fan-out, so it
sustains higher QPS. Throughput tracks `P / (fan-out · load_skew · log|shard|)`,
and fan-out is the first-order term. Confirmed result (RESULTS.md): routing beats
every broadcast arm ~2.5–3.8×; Orion navigation beats k-means centroid 1.49× on
GloVe (hard) and ties it on SIFT (easy, uniform).

---

## 1. Prerequisites

Do the full machine setup in [`ENVIRONMENT.md`](ENVIRONMENT.md) first: the 4-peer
Orion cluster (healthy, reachable at `http://127.0.0.1:6833`), `pip install -r
requirements.txt`, and decoded datasets under `data/<ds>/`. Then, specific to a
measurement:

- **hnswlib needs the system libstdc++ ahead of conda's.** Prefix every command
  that builds an upper graph (`build_routed_requests.py --router navigation`,
  `analysis/build_orion_layout.py`) with `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6`.
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

## 3. Build the layouts, then the collections

The clean pipeline has two steps: (1) compute a **point→shard layout** `.npz`,
(2) upload it into a **custom-shard collection** with `upload_custom_shards.py`.
All three layouts are single-assignment (one copy per point), so the collections
are equal-storage and comparable. `P = 46` and distance per dataset (`Cosine` for
GloVe, `Euclid` for SIFT).

### 3a. Layouts

```bash
# hash (uniform id%P) and plain k-means, one command:
python make_layouts.py --train-path data/glove/train.npy --shards 46 --out-dir layouts --seed 0
# -> layouts/hash_p46.npz, layouts/kmeans_p46.npz  (prints k-means size skew)

# Orion (navigation-derived), single-assignment (norep):
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 python ../../analysis/build_orion_layout.py \
  --hdf5-path /home/taig/dry/faiss/datasets/glove-200-angular.hdf5 \
  --out layouts/orion_glove_p46.npz --shards 46 --vector-distance cosine \
  --disable-multi-assign --balance-mode capacity_constrained \
  --balance-min-load-ratio 0.8 --balance-max-load-ratio 1.25 \
  --balance-max-vote-loss 20 --balance-max-passes 64
```

> Put SIFT layouts in a separate dir (`make_layouts.py --out-dir layouts/sift`) —
> the filenames (`hash_p46.npz`, `kmeans_p46.npz`) are dataset-agnostic and will
> otherwise overwrite GloVe's.

### 3b. Collections

Same uploader for all three; only the layout and distance differ. It creates a
custom-sharded collection, one `centroid_XX` shard key per shard, and upserts
each point onto the shard its layout assigns (raw ids, no `source_id` encoding —
single-assignment needs none). HNSW params match across arms.

```bash
python upload_custom_shards.py --collection glove_hash_p46   --train-path data/glove/train.npy --layout-file layouts/hash_p46.npz        --distance Cosine --m 32 --ef-construct 100 --recreate
python upload_custom_shards.py --collection glove_kmeans_p46 --train-path data/glove/train.npy --layout-file layouts/kmeans_p46.npz      --distance Cosine --m 32 --ef-construct 100 --recreate
python upload_custom_shards.py --collection glove_orion_p46  --train-path data/glove/train.npy --layout-file layouts/orion_glove_p46.npz --distance Cosine --m 32 --ef-construct 100 --recreate
```

Each waits for the collection to go green and prints its point count.

---

## 4. Build the routed request files (offline, outside the timer)

One command per arm per knob value; it writes `<out>.jsonl` plus `<out>.stats.json`
(fan-out and total-ef — the x-axis you regress QPS against). **`--source-id-dedup-block-size 0`**
matches the raw-id collections from §3. Broadcast is layout-independent, so **one
broadcast file is reused across all three collections**.

```bash
PRE=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
HD=/home/taig/dry/faiss/datasets/glove-200-angular.hdf5

# broadcast (fan-out = 46), shared by hash/kmeans/orion collections — sweep ef
for e in 64 128 256; do
  LD_PRELOAD=$PRE python build_routed_requests.py --hdf5 $HD --collection glove_raw --vector-distance cosine \
    --router broadcast --num-shards 46 --hnsw-ef $e --source-id-dedup-block-size 0 --out req/rawbcast_ef$e.jsonl
done

# Orion navigation — sweep --upper-k (uses the Orion layout; builds an upper graph)
for k in 60 120 200; do
  LD_PRELOAD=$PRE python build_routed_requests.py --hdf5 $HD --collection glove_orion_p46 --vector-distance cosine \
    --router navigation --layout-file layouts/orion_glove_p46.npz --upper-k $k --source-id-dedup-block-size 0 --out req/gnav_k$k.jsonl
done

# k-means centroid — sweep --nprobe × --hnsw-ef (uniform ef; uses the k-means layout for centroids)
for combo in "12 128" "20 128" "30 128"; do set -- $combo
  LD_PRELOAD=$PRE python build_routed_requests.py --hdf5 $HD --collection glove_kmeans_p46 --vector-distance cosine \
    --router centroid --layout-file layouts/kmeans_p46.npz --nprobe $1 --hnsw-ef $2 --source-id-dedup-block-size 0 --out req/kmc_np${1}_ef${2}.jsonl
done
```

Navigation uses `--base-ef 20 --factor 4` (per-shard `ef = base + factor·L1_hits`);
centroid uses a uniform `--hnsw-ef`. **Sweep `--hnsw-ef` for the centroid arm too**
— too small an ef caps its recall regardless of `--nprobe`, which would unfairly
handicap the baseline.

> Upper-graph params (`--sample-denominator 32`, `--upper-sample-seed 100`,
> `--upper-m 32`, `--upper-ef-construction 100`) default to the same values
> `build_orion_layout.py` used, so the navigation router reproduces the layout's
> L1 set. Don't override on one side only.

---

## 5. Measure (timed, gated)

**Every arm — including broadcast — replays a pre-built request file with `--mode
routed`.** Only the collection and the request file change. Point each arm's
request file at the matching collection (broadcast files are shared, so run each
one against all three collections):

```bash
# Orion navigation
python measure.py --collection glove_orion_p46 --mode routed \
  --requests-file req/gnav_k120.jsonl --output-dir runs/gnav_k120 \
  --client-cores 32-63 --procs 8 --inflight 96 --warmup 2000 --top-k 10

# broadcast on each layout (same rawbcast file, different collection)
python measure.py --collection glove_hash_p46   --mode routed --requests-file req/rawbcast_ef64.jsonl --output-dir runs/hash_ef64   --client-cores 32-63 --procs 8 --inflight 96 --warmup 2000 --top-k 10
python measure.py --collection glove_kmeans_p46 --mode routed --requests-file req/rawbcast_ef64.jsonl --output-dir runs/kmbc_ef64   --client-cores 32-63 --procs 8 --inflight 96 --warmup 2000 --top-k 10

# k-means centroid
python measure.py --collection glove_kmeans_p46 --mode routed \
  --requests-file req/kmc_np20_ef128.jsonl --output-dir runs/kmc_np20_ef128 \
  --client-cores 32-63 --procs 8 --inflight 96 --warmup 2000 --top-k 10
```

`measure.py` prints a summary and writes `runs/<name>/summary.json` with `qps`,
`per_peer_cpu_pct`, and the gates. Exit code 2 and a loud message mean a gate
failed.

---

## 6. Score recall and build the frontier

Recall is scored offline from persisted ids — never in the timed path.

```bash
python score_recall.py --output-dir runs/gnav_k120 \
  --ground-truth data/glove/ground_truth.npy --top-k 10
```

Repeat §4–6 across the swept knob for each arm. Plot QPS vs Recall@10 per arm and
read **QPS at a fixed recall target** off each arm's Pareto frontier (interpolate
between the two bracketing configs). Pick the target where the arms actually
separate: **GloVe ≈ 0.90**, **SIFT ≈ 0.99** (on SIFT everything clears 0.9, so a
0.90 target is meaningless). Compare against [`RESULTS.md`](RESULTS.md): the
frontier tables and the multipliers there are your pass/fail check.

Datasets: **GloVe-200-angular** (hard, Orion wins) and **SIFT-128-euclidean**
(negative control, near-uniform, Orion ties k-means centroid). Hold total server
cores fixed. On this single host all G4 gates read ~6–13% (co-tenants), so
`gates_passed` is strict-false; the ratios are still robust because every arm sees
the same host — re-certify absolute QPS on a quiet host.

---

## 7. Multi-host deployment

The load path is host-agnostic — peers are reached at `http://<host>:<port>`. A
multi-machine cluster is just a longer host list.

1. **Bring up peers on each server host** (one container per peer, pinned cpusets),
   and make cpusets **symmetric across all hosts** at fixed total cores.
2. **Point discovery at every host.** Pass `--hosts` to `measure.py`:
   ```bash
   python measure.py --hosts local,user@srv2,user@srv3 --coordinator all \
     --collection glove_orion_p46 --mode routed --requests-file req/... \
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
5. Request files are built offline and are host-agnostic; measure against a
   reachable coordinator via discovery. (If you use `build_routed_requests.py
   --from-collection` to recover membership from a live collection instead of a
   layout file, set `--base-url` to a reachable coordinator, e.g. `http://srv1:6833`.)

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
- **Recall far below expectation** → the builder's `--source-id-dedup-block-size`
  or upper-graph params don't match how the collection was built. §3 uploads with
  raw ids, so requests need `--source-id-dedup-block-size 0`; navigation seeds must
  match `build_orion_layout.py`.
- **k-means centroid recall plateaus low** → its uniform `--hnsw-ef` is too small.
  Sweep `--hnsw-ef` (e.g. 64, 128), not just `--nprobe`.
- **`build_orion_layout.py` "physical-copy bounds not satisfied"** → single-assign
  balance is too tight; loosen `--balance-max-vote-loss` / `--balance-max-passes`
  (20 / 64 work for GloVe and SIFT norep).
- **A gate fails** → do not interpret the QPS. Fix the cause (quiet the host for G4,
  raise `--inflight` for G1, add client cores for G2) and re-run.

---

## 10. The exact loop, condensed

```bash
cd experiments/harness
python configure_testbed.py --cores-per-peer 8                      # once
python prepare_dataset.py --hdf5 <ds>.hdf5 --out-dir data/<ds> [--normalize]
# layouts (§3a) -> collections (§3b):
python make_layouts.py --train-path data/<ds>/train.npy --shards 46 --out-dir layouts
LD_PRELOAD=… python ../../analysis/build_orion_layout.py --hdf5-path <ds>.hdf5 --out layouts/orion_<ds>_p46.npz --shards 46 --vector-distance <cosine|euclid> --disable-multi-assign --balance-mode capacity_constrained --balance-max-vote-loss 20 --balance-max-passes 64
for L in hash kmeans orion_<ds>; do python upload_custom_shards.py --collection <ds>_${L%_*}_p46 --train-path data/<ds>/train.npy --layout-file layouts/$L*p46.npz --distance <Cosine|Euclid> --m 32 --ef-construct 100 --recreate; done
# requests (§4) -> measure (§5) -> score (§6), per arm and per knob:
LD_PRELOAD=… python build_routed_requests.py --router <navigation|centroid|broadcast> --source-id-dedup-block-size 0 … --out req/<name>.jsonl
python measure.py --collection <c> --mode routed --requests-file req/<name>.jsonl --output-dir runs/<name> --client-cores 32-63 --procs 8 --inflight 96 --warmup 2000 --top-k 10
python score_recall.py --output-dir runs/<name> --ground-truth data/<ds>/ground_truth.npy --top-k 10
# read QPS at fixed recall off each arm's Pareto frontier; compare to RESULTS.md.
```
