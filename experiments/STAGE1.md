# Stage 1 — Matched-recall throughput of routing and layout

Status: harness extended for routed serving; ready for a real multi-host cluster.
Precondition: Stage 0 gates (`STAGE0.md` §4) must hold at every operating point.

Stage 0 built a gated, server-bound harness but only for **broadcast** search
(the server fans out to every shard, fan-out = P). That measures the wrong thing
for Orion, whose entire claim is that it touches *fewer* shards per query at the
same recall. Stage 1 adds the routed serving path so the offline finding — Orion
reaches target recall at lower fan-out than a k-means router, and much lower than
broadcast — can be turned into an end-to-end matched-recall QPS number.

## 1. What "routed" means here, and why it stays server-bound

Orion routes by navigating the upper graph: a query's nearest upper-tier (L1)
nodes are found, the shards owning them become the probed set, each probed shard
is seeded with the L1 points that landed on it (HNSW entry points) and searched
at `ef_s = base_ef + factor · hits_s`. This is *one* request to a coordinator
that fans out selectively — not one request per shard from the client.

The routing decision (an upper-graph kNN) is fixed-cost: it does not grow with
fan-out or shard count. So we compute it **outside the timer**, exactly as Stage 0
pre-serializes broadcast bodies, and the timed path only writes prepared bytes and
reads responses. This isolates *server* throughput at a fixed routing decision and
keeps G1–G4 valid — the load generator does no per-query routing work. The routing
cost is reported separately by the builder (it would live on the coordinator in a
full Rust router; the entry-point/ef-by-shard honoring already lives in the Orion
fork's patched Qdrant).

Concretely, `build_routed_requests.py` emits a JSONL file — one fully-formed
`points/search` body per query, in query order — and `measure.py --mode routed`
replays it. The routing/encoding is **reused verbatim** from
`tools/qdrant_two_level_routing_experiment.py` (`route_upper_labels_to_shard_eps`,
`shard_efs_from_routed_eps`, `search_request`, `encode_entry_point_id`), so the
per-shard entry-point ids — including replicated copies — match the deployed
collection bit for bit.

## 2. Harness pieces

| File | Role |
|---|---|
| `harness/build_routed_requests.py` | offline: query → per-shard entry points + ef → serialized bodies (JSONL). Two routers: `navigation` (Orion), `centroid` (k-means baseline). |
| `harness/measure.py --mode routed` | timed replay of the JSONL, persists result ids, records G1–G4. `--mode broadcast` is the Stage-0 path unchanged. |
| `harness/score_recall.py` | offline Recall@k from persisted ids (reads `source_id` payload for replicated layouts). Unchanged. |
| `harness/cluster.py` | peer discovery + cgroup CPU accounting, now **per host** (local or SSH) for multi-machine runs. |

All three comparison arms come from the same replay path, differing only in the
request bodies:

- **broadcast** (fan-out = P upper bound): `measure.py --mode broadcast`, sweep `--hnsw-ef`.
- **k-means + centroid router**: `build_routed_requests.py --router centroid --nprobe K`, sweep K.
- **Orion + navigation router**: `build_routed_requests.py --router navigation --upper-k K`, sweep K.

## 3. End-to-end procedure

Prerequisites: symmetric peer cpusets at fixed *total* server cores
(`harness/configure_testbed.py`), a quiet host, and the datasets/ground-truth
prepared by `harness/prepare_dataset.py`. `hnswlib` needs the system libstdc++:
`LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6`.

1. **Build one collection per layout.** Use the two-level tool to create and
   upload each arm's collection (plain k-means, balanced k-means, Orion). Record
   the exact build params: `--source-id-dedup-block-size` (default num_points+1),
   the upper-graph params (`--sample-denominator`, `--upper-*-seed`, `--upper-m`,
   `--upper-ef-construction`, `--upper-query-k`) — the request builder must be
   given the *same* values or its routing will not match what is stored.

2. **Pre-build routed request files** for each arm and each knob value. Prefer
   `--from-collection` for the navigation arm: it scrolls the live collection to
   recover L1 shard membership, so there is zero drift from what is deployed.

   ```
   # Orion navigation arm, sweep upper_k
   LD_PRELOAD=… python build_routed_requests.py \
     --hdf5 glove.hdf5 --collection orion_glove_p32 --vector-distance cosine \
     --router navigation --from-collection --num-shards 32 --base-url http://<coord>:6333 \
     --upper-k 60 --out req/orion_glove_p32_k60.jsonl
   # k-means centroid arm needs a layout .npz for centroids
   LD_PRELOAD=… python build_routed_requests.py \
     --hdf5 glove.hdf5 --collection kmeans_glove_p32 --vector-distance cosine \
     --router centroid --layout-file layouts/kmeans_glove_p32.npz --source-id-dedup-block-size 0 \
     --nprobe 8 --out req/kmeans_glove_p32_np8.jsonl
   ```

3. **Measure each file** (server-bound, gated):

   ```
   python measure.py --collection orion_glove_p32 --mode routed \
     --requests-file req/orion_glove_p32_k60.jsonl \
     --output-dir runs/orion_glove_p32_k60 \
     --client-cores 32-63 --procs 8 --inflight 96 --warmup 2000 --top-k 10
   ```
   Broadcast baseline for the same collection: `--mode broadcast --queries-path … --hnsw-ef E`.

4. **Score recall offline** and assemble the frontier:

   ```
   python score_recall.py --output-dir runs/orion_glove_p32_k60 \
     --ground-truth glove_gt.npy --top-k 10
   ```
   A run counts only if `gates_passed` is true. Plot QPS vs recall per arm across
   the swept knob; the headline is **QPS at matched Recall@10 ≥ 0.90** (and 0.95).

## 4. What to compare, and the expected result

Datasets: **GloVe-200** and **coco-t2i-512** (offline, Orion needs materially
fewer shards at target recall — the navigation router matters most on cross-modal
coco), with **SIFT-1M as a negative control** (near-uniform; Orion is not expected
to win, and plain k-means only wins there by tolerating size imbalance). Hold
total server cores fixed across shard counts.

The offline model is `throughput ∝ P / (fan-out · load_skew · log|shard|)`, with
fan-out the first-order term. A confirming result is: at matched recall, Orion's
navigation arm sustains higher QPS than the centroid-routed k-means arm, which in
turn beats broadcast, tracking the offline fan-out ordering (`analysis/FINDINGS.md`).
Also log per-run `per_peer_cpu_pct` and the builder's fan-out stats so measured
QPS can be regressed against `fan-out · load_skew`.

## 5. Multi-host deployment

The load path is host-agnostic: peers are reached at `http://<host>:<port>` from
`Peer.base_url`, so a multi-machine cluster is just a longer host list.

- **Discovery + CPU accounting per host.** `measure.py --hosts local,user@h2,user@h3`
  runs the same docker inspection on each host and reads each peer's cgroup where
  it runs — locally by file, remotely over `ssh` — so **G1 (server saturation)
  and G3 (client headroom) keep their single-host fidelity**. Requires passwordless
  SSH (`BatchMode`) and docker access for that user on every server host.
- **Symmetric capacity across hosts.** Give every peer an identical cpuset and
  hold total server cores constant, as in Stage 0. `assert_symmetric` checks this
  across all hosts; `assert_disjoint` only guards peers that share the client host.
- **Real network, no loopback caveat.** On multi-host the Stage-0 loopback
  deviation disappears and physical scale-out becomes measurable — the axis Stage 0
  had to declare out of scope. State the interconnect (bandwidth, latency).
- **G4 is per-host.** Host-interference is computed on the client host; keep every
  participating host quiet and record load on each. Co-tenants on a server host
  count against that host's saturation reading.
- **Coordinator.** Any peer can coordinate a routed request (shard-key selectors
  fan out server-side). Stage 0 found the coordinator role saturates before data
  peers; with `--coordinator all` entry points spread across peers, watch
  `peer_imbalance_ratio` and, on a real cluster, pin the coordinator if it caps QPS.

## 6. Deviations and things that must match

- **Routing is precomputed**, not timed. This measures server throughput at a
  fixed routing decision; it does not include per-query routing latency (a
  fan-out-independent coordinator cost). End-to-end latency including routing is a
  separate measurement if needed.
- **Builder params must equal the collection's build params** (upper-graph seeds,
  `source_id_dedup_block_size`). `--from-collection` removes the membership half of
  this risk; the upper-index half (seeds/M/ef) still must match.
- **Entry-point + ef-by-shard honoring needs the patched Qdrant** (the Orion fork).
  The plumbing was validated locally by replaying bodies against a stock collection;
  the entry-point *semantics* are confirmed only by a real routed run's recall.
- **The dual-graph upper index is the legacy ablation path** `agent.md` flags: a
  faithful encoding of the routing rule, not a bit-for-bit match of the Rust router.

## 7. Validation done so far

- Generator (`build_routed_requests.py`) produces correct bodies for both routers
  on synthetic data: navigation carries per-shard entry points with encoded ids
  (`shard_id·block + src + 1`) and `ef = base_ef + factor·hits`; centroid probes
  exactly `nprobe` nearest shards at uniform ef with no entry points.
- Replay (`measure.py --mode routed`) runs end to end against a live collection:
  bodies load, ids persist and align by offset (fixed an offset off-by-one), G1–G4
  compute, and **G2 shows large client headroom (~10%) → the routed path is
  server-bound**, not client-bound like the legacy harness. `score_recall.py` reads
  the run unchanged.
- Full gate-passing matched-recall runs on real Orion/k-means collections are the
  next step and require the deployed collections (and, for scale-out, real hosts).
