# Environment: bring up the testbed on a fresh machine

This is the machine-level setup the rest of the experiment docs assume. When you
move to a new host, reproducing the results in [`RESULTS.md`](RESULTS.md) means
recreating three things: the **Orion Qdrant cluster**, the **Python toolchain**,
and the **datasets**. Nothing here needs the old machine's data — every layout
and collection is rebuilt from deterministic seeds (see [`DEPLOY.md`](DEPLOY.md)).

Everything the harness measures assumes the cluster is a set of Docker containers
pinned to cpusets; the harness discovers them and reads their CPU from cgroup v2.

---

## 1. Code: the Orion Qdrant fork

The routed-serving features (per-shard `ef`, entry points, custom-shard upload)
exist **only in this fork's patched Qdrant**, not in stock Qdrant.

- Repo: this checkout (`/home/taig/wzhzhu/Orion` on the origin machine).
- Branch / commit the results were produced on: **`stage1-routed-harness`**,
  commit **`da2ee17dc`**. Check out the same commit on the new machine.
- The server runs as a prebuilt Docker image, tag
  **`qdrant/qdrant:method4-peer-premerge`**. Either copy that image to the new
  machine (`docker save | docker load`) or rebuild it from the fork:
  `docker build . --tag qdrant/qdrant:method4-peer-premerge`.

---

## 2. The cluster: 4-peer distributed Qdrant via docker-compose

The testbed is a **1 controller + 3 shard** distributed cluster (4 peers), one
container per peer, brought up by compose. On the origin machine the compose
file lives **outside** this repo at:

```
/home/taig/dry/qdrant/tools/compose/docker-compose.controller-cluster.yaml
```

A copy is vendored next to this doc at [`compose/docker-compose.controller-cluster.yaml`](compose/docker-compose.controller-cluster.yaml)
so the new machine has it without the external checkout.

Topology (what the compose file encodes):

| Service | Role | REST port (host→cont) | gRPC | Cluster wiring |
|---|---|---|---|---|
| `qdrant_controller` | coordinator | `6833→6333` | `6834→6334` | `--uri http://qdrant_controller:6335` |
| `qdrant_shard_1` | data peer | `6843→6333` | `6844→6334` | `--bootstrap …controller:6335` |
| `qdrant_shard_2` | data peer | `6853→6333` | `6854→6334` | `--bootstrap …controller:6335` |
| `qdrant_shard_3` | data peer | `6863→6333` | `6864→6334` | `--bootstrap …controller:6335` |

- Cluster mode is on (`QDRANT__CLUSTER__ENABLED=true`, P2P port 6335). Shards
  `--bootstrap` to the controller with staggered `sleep` so the controller is up
  first. **The harness talks to the controller at `http://127.0.0.1:6833`.**
- Each peer persists to a host volume under
  `/home/taig/dry/qdrant/qdrant_storage/controller-cluster/<peer>`. Override the
  `QDRANT_*_STORAGE_DIR` env vars to relocate on the new machine.

Bring it up:

```bash
cd <dir containing the compose file>
docker compose -f docker-compose.controller-cluster.yaml up -d
# wait for healthy:
docker ps --filter name=qdrant-controller --format '{{.Names}}\t{{.Status}}'
curl -s http://127.0.0.1:6833/cluster | python -c 'import sys,json;print(json.load(sys.stdin)["result"]["status"])'   # -> "enabled"
```

### Peer symmetry (required for the gates)

The comparison holds **total server cores fixed and every peer identical**; an
asymmetric peer can never pass the server-saturation gate (G1). Two knobs:

1. **cpusets** — 8 cores per peer, contiguous and disjoint. The running testbed
   uses controller `0-7`, shard_1 `8-15`, shard_2 `16-23`, shard_3 `24-31`
   (32 server cores total). Set them **on the live containers** with:
   ```bash
   cd experiments/harness
   python configure_testbed.py --cores-per-peer 8 --first-core 0   # writes cpuset_backup.json
   # restore original compose cpusets later with:  python configure_testbed.py --restore
   ```
   `configure_testbed.py` uses `docker update` (no restart) and only touches
   cpusets.
2. **search threads** — the compose defaults are **asymmetric**: controller
   `MAX_SEARCH_THREADS=8`, shards `=5`. The results in RESULTS.md were taken with
   these defaults. For a strictly symmetric testbed, set every peer to match its
   core count by exporting before `up`:
   ```bash
   export QDRANT_SHARD_1_SEARCH_THREADS=8 QDRANT_SHARD_2_SEARCH_THREADS=8 QDRANT_SHARD_3_SEARCH_THREADS=8
   export QDRANT_SHARD_1_OPTIMIZER_CPU_BUDGET=8 QDRANT_SHARD_2_OPTIMIZER_CPU_BUDGET=8 QDRANT_SHARD_3_OPTIMIZER_CPU_BUDGET=8
   ```
   `configure_testbed.py` does **not** change threads (only cpusets), so this must
   be set at compose time.

### How the harness finds the cluster

`harness/cluster.py` runs `docker ps --filter name=qdrant-controller`, inspects
each container's cpuset and its `6333/tcp` host-port mapping, and reads CPU from
`/sys/fs/cgroup/system.slice/docker-<id>.scope/cpu.stat`. So on the new machine
you need only: (a) containers whose names contain **`qdrant-controller`**, (b)
each exposing container port `6333`, (c) cgroup v2. No hostnames/ports are
hardcoded in the harness — it derives `base_url` from the port mapping.

---

## 3. Python toolchain

Python 3.10. Install the harness deps (pinned):

```bash
pip install -r experiments/harness/requirements.txt
```

Two machine-specific gotchas, both documented in that file:

- **`LD_PRELOAD` for hnswlib.** Any command that builds an upper graph
  (`build_routed_requests.py --router navigation`, `analysis/build_orion_layout.py`)
  must preload the **system** libstdc++, because hnswlib was built against a newer
  one than conda ships:
  ```bash
  LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 python build_routed_requests.py ...
  ```
  Find the path on the new machine: `find / -name 'libstdc++.so.6*' 2>/dev/null`.
- **NumPy 2.x import warning** from scikit-learn/pandas is harmless (see
  requirements.txt).

---

## 4. Datasets

ann-benchmarks HDF5 files. On the origin machine they live in
`/home/taig/dry/faiss/datasets/`. The two used by RESULTS.md:

| Dataset | File | Distance | Download |
|---|---|---|---|
| GloVe-200-angular | `glove-200-angular.hdf5` | cosine (normalize) | http://ann-benchmarks.com/glove-200-angular.hdf5 |
| SIFT-128-euclidean | `sift-128-euclidean.hdf5` | L2 (no normalize) | http://ann-benchmarks.com/sift-128-euclidean.hdf5 |

Decode each to `.npy` once (outside any timed path); `--normalize` for angular:

```bash
cd experiments/harness
python prepare_dataset.py --hdf5 /path/to/glove-200-angular.hdf5 --out-dir data/glove --normalize
python prepare_dataset.py --hdf5 /path/to/sift-128-euclidean.hdf5 --out-dir data/sift
# -> data/<ds>/{train,queries,ground_truth}.npy
```

---

## 5. Machine-specific values to change on a new host

Collected in one place (also surfaced in [`AGENTS.md`](AGENTS.md)):

| What | Origin value | Where it is used |
|---|---|---|
| Controller base URL | `http://127.0.0.1:6833` | auto-discovered; override `--base-url` in `build_routed_requests.py` only |
| Server cpusets | `0-7,8-15,16-23,24-31` | `configure_testbed.py --cores-per-peer 8 --first-core 0` |
| Client cores | `32-63` | `measure.py --client-cores` (must be disjoint from peers) |
| Total host cores | 112 | sizing client vs. server |
| `libstdc++` path | `/usr/lib/x86_64-linux-gnu/libstdc++.so.6` | `LD_PRELOAD` prefix |
| Dataset dir | `/home/taig/dry/faiss/datasets/` | `--hdf5` / `prepare_dataset.py` |
| Server image tag | `qdrant/qdrant:method4-peer-premerge` | compose |
| Compose file | `/home/taig/dry/qdrant/tools/compose/…` | vendored at `experiments/compose/` |

Once the cluster is healthy and datasets are decoded, follow
[`DEPLOY.md`](DEPLOY.md) to build collections, generate requests, measure, and
score — and check the output against [`RESULTS.md`](RESULTS.md).
