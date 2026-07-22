# Method4/Orion Four-Node Distributed Runbook

> [!WARNING]
> **Historical architecture boundary.** The 2026-07-18 results in this runbook were
> produced by the earlier custom-shard/client-hint architecture: the benchmark side
> planned shard fan-out and transported routing hints rather than issuing a native
> numeric-auto-shard request whose routing was decided entirely inside the Qdrant
> collection coordinator. They remain valid evidence for that historical implementation,
> but **must not be cited, relabeled, or compared as results of the native
> `sharding_method=auto` Orion architecture**. Native claims require a fresh run through
> the server-side `AutoShardPolicy::Orion`, numeric `ShardId`, standard Search/Query,
> ordinary `ShardReplicaSet`, and collection global-merge path documented in
> `2026-07-20-orion-native-auto-shard-phase1.md`.

This runbook replaces the old single-host, multi-container layout for the initial
Orion versus Simple KMeans nprobe versus Naive all-shards Recall–QPS comparison.
The Qdrant binary and search semantics remain those of commit
`1a5ac4c47237b9224ae3e4ca28c2cefb2b514352`.

> **Authoritative result as of 2026-07-18.** The final main comparison is
> `main-same-recall-pipelined-preencoded-final-20260718f`, generated from
> `tools/benchmark_configs/method4_distributed_main_same_recall_confirmation_batch200.json`.
> Earlier Naive confirmation QPS values are superseded: the sequential Naive
> evaluator omitted the uniform `hnsw_ef_by_shard` transport map, bypassed native
> physical-peer grouping, and created hundreds to thousands of transient P2P
> connections. The corrected default still visits all 46 shards with the same
> fixed EF and uses `fixed_ef_shard_chunk_size=0`. Orion now uses one-batch
> look-ahead planning and pre-encodes batch N+1 while batch N executes remotely;
> routed shards, MultiEP, Dynamic EF, source-ID dedup, and global top-k are unchanged.

## Fixed topology

| Host | qdrant-lan | Role | CPU set |
|---|---:|---|---|
| `hp057.utah.cloudlab.us` | `10.10.1.1` | controller and benchmark client | Qdrant `0-7`, client `8-19` |
| `hp052.utah.cloudlab.us` | `10.10.1.2` | worker 1 | `0-19` |
| `hp065.utah.cloudlab.us` | `10.10.1.3` | worker 2 | `0-19` |
| `hp076.utah.cloudlab.us` | `10.10.1.4` | worker 3 | `0-19` |

The machine-readable source of truth is
`tools/distributed/cloudlab_orion_4node.json`. All containers use host networking.
HTTP `6333`, gRPC `6334`, P2P `6335`, and custom-shard fan-out therefore use the
private `10.10.1.0/24` network. Lower logical shards may be placed only on the
three exact worker URIs; the controller must have zero local lower shards.

## Lifecycle

Choose a run id once and use it for every command. It scopes the container names,
shared image tar, local storage, collection names, results, and manifest.

```bash
export RUN_ID=dist-$(date -u +%Y%m%d-%H%M%S)

python3 tools/method4_distributed_cluster.py --run-id "$RUN_ID" bootstrap
python3 tools/method4_distributed_cluster.py --run-id "$RUN_ID" build
python3 tools/method4_distributed_cluster.py --run-id "$RUN_ID" deploy
python3 tools/method4_distributed_cluster.py --run-id "$RUN_ID" status
```

`bootstrap` is idempotent and installs Docker where missing, creates
`/users/dry/orion-distributed/venv`, and downloads the official
`glove-200-angular.hdf5` only on the controller. It validates HDF5 readability and
records file size and SHA-256. `build` passes `GIT_COMMIT_ID`, performs a local
container version health check, saves the image under
`/proj/intelisys-PG0/exp/orion-distributed/$RUN_ID/`, and records the Docker image
ID and tar SHA-256. Existing tar reuse is allowed only when its manifest tag,
path, SHA-256, and image ID still match the current local tag. After every remote
load, `deploy` inspects the actual tag again and rejects any digest mismatch.
An existing container is reused only when its run-id, role, private IP, image ID,
CPU set, nofile limit, and controller peer-premerge fingerprint match exactly.

### Compact wire identity and image transition

This subsection documents the current native cluster lifecycle only. It does not
retroactively turn the historical custom-shard results elsewhere in this runbook
into native compact-wire results.

`tools/distributed/cloudlab_orion_4node.json` declares the requested compact wire
under the controller only:

```json
"orion_compact_wire_version": 2
```

The runtime identity is intentionally asymmetric. An implicit v1 controller has
neither `QDRANT_ORION_COMPACT_WIRE_VERSION` nor
`orion.distributed.compact_wire_version`. A v2 controller must have the env and
label exactly once with value `2`. Workers must have neither: they decode the
version carried by the internal request, while the controller alone chooses what
it produces. `status` and `manifest` report requested/current/match at
`orion_compact_wire`; controller node entries report `1` or `2`, and worker
entries report `not_applicable`. Missing, duplicated, illegal, inconsistent, or
worker-leaked metadata is a hard validation error.

The producer env/label is not accepted as binary capability evidence. Images
that implement v2 carry
`org.qdrant.orion.compact_wire.max_version=2`; the inherited image label is
re-inspected on the controller and all three workers after container start. An
older image without this label is treated as v1-only, even if someone adds the
controller v2 env/label. The active and candidate manifests also record the
image capability.

Wire v2 is not negotiated during an RPC. All four nodes must first have the same
image digest containing the v2 encoder/decoder. Do not change an existing v1 run
by invoking ordinary `deploy` against a topology requesting v2; reuse validation
must reject that configuration. Stage an immutable candidate and perform the
offline four-node transition instead:

```bash
python3 tools/method4_distributed_cluster.py \
  --topology tools/distributed/cloudlab_orion_4node.json \
  --run-id "$RUN_ID" \
  --expected-commit "$(git rev-parse HEAD)" \
  build --stage-for-transition

export CANDIDATE_MANIFEST=/proj/intelisys-PG0/exp/orion-distributed/$RUN_ID/image-candidates/<candidate>.json
export ACTIVE_IMAGE_ID=<exact-image-id-from-current-manifest>

python3 tools/method4_distributed_cluster.py \
  --topology tools/distributed/cloudlab_orion_4node.json \
  --run-id "$RUN_ID" \
  transition-image \
  --candidate-manifest "$CANDIDATE_MANIFEST" \
  --expected-current-image-id "$ACTIVE_IMAGE_ID"
```

The candidate manifest binds schema v2, run id, topology runtime identity,
source fingerprint, image ID, archive SHA-256, and requested compact wire
version. Its path also includes `wire-v1` or `wire-v2`. Only equal image ID **and**
equal wire identity is a no-op; equal image ID with a different wire identity
still recreates all four containers. The transition stops/recreates only this
run's four containers, preserves their storage, validates cluster and collection
placement after restart, and records old/candidate/final wire identities. On a
forward failure it restores both the active image and the active wire identity;
an implicit-v1 rollback restores the absence of env/label rather than writing an
explicit v1 marker.

There is no supported rolling v1-to-v2 upgrade or automatic downgrade. A v2
controller must never be measured against an old worker image. After transition,
require `status` to show four healthy same-digest peers,
`matches_requested=true`, worker wire metadata `not_applicable`, no transfers,
image max wire capability `2` on all four nodes, and the unchanged lower-shard
placement before running smoke or timed queries.
This lifecycle is a safety/capability boundary; it does not claim that v2 is
faster than v1.

Qdrant storage is always local:

```text
/users/dry/orion-distributed/<run-id>/<role>/storage
```

The shared `/proj` directory is only an image/manifest exchange point. It never
contains a Qdrant WAL or index.

To stop containers while retaining all indexes:

```bash
python3 tools/method4_distributed_cluster.py --run-id "$RUN_ID" down
```

Deletion is deliberately separate and requires an explicit acknowledgement:

```bash
python3 tools/method4_distributed_cluster.py \
  --run-id "$RUN_ID" clean --yes-delete-storage
```

The cleaner accepts only a validated run id and removes only resources named with
the `orion-dist-<run-id>-` prefix and the corresponding exact local run directory.

## Required smoke test

Run this after `status` reports four peers and before any full-data collection is
built:

```bash
python3 tools/method4_benchmark_matrix.py \
  --config tools/benchmark_configs/method4_distributed_smoke.json \
  --run-id "$RUN_ID" \
  --run
```

The smoke matrix uses a 5,000-vector prefix, exercises Orion compact MultiEP,
Simple KMeans nprobe, and Naive all-shards, and deletes only its three temporary
collections after successful result collection. Automatic deletion is accepted
only for `dist_`/`bench_` collection templates that explicitly embed the current
`{matrix_run_id}`; fixed or reused collection names are rejected before any
benchmark or DELETE request. Every harness invocation first requires exactly
these peer URIs:

```text
http://10.10.1.1:6335
http://10.10.1.2:6335
http://10.10.1.3:6335
http://10.10.1.4:6335
```

It also verifies the collection schema, HNSW parameters, point count, replication
factor, Active shard state, remote-only placement, and balanced worker shard
counts before issuing timed queries.

## Full initial scan

The single reproducible entry command is:

```bash
python3 tools/method4_benchmark_matrix.py \
  --config tools/benchmark_configs/method4_distributed_glove200_initial.json \
  --run-id "$RUN_ID" \
  --run
```

The generated commands are automatically prefixed with `taskset -c 8-19`. The
config builds the three collections sequentially and uses round-robin placement.
It contains the complete parameter grids from the experiment plan:

- Orion: `upper_k={24,60,80,120,160}`,
  `base_ef={32,40,60,80}`, `factor={6,8,10,12}`.
- Simple KMeans: `nprobe={8,16,24,32,40,46}`,
  fixed `ef={32,48,64,80,120,160,240}`.
- Naive: all 46 shards, fixed `ef={20,32,48,64,76,88,112}`.

Each candidate receives a 200-query warmup that is excluded from QPS. The first
scan writes all points rather than only the selected point. Matrix post-processing
creates:

```text
recall_qps_points.csv
pareto_frontier.csv
same_recall_selection.csv
confirmation_matrix.json
recall_qps.png
recall_p95.png
recall_p99.png
recall_visited_shards.png
recall_estimated_ef_sum.png
```

`same_recall_selection.csv` applies the fixed `0.003` recall window for targets
`0.90` and `0.95`. A point is marked `strict` when
`abs(actual_recall - target_recall) <= 0.003`; otherwise the closest point is
marked `nearest`.
Nearest points must not be presented as strict same-recall comparisons.

## Confirmation runs

The first-stage collector writes `confirmation_matrix.json` with the selected
parameters, the same collection names, reuse guards, and the required confirmation
settings. Run it with:

```bash
python3 tools/method4_benchmark_matrix.py \
  --config "results/method4_distributed/method4_distributed_glove200_initial/$RUN_ID/confirmation_matrix.json" \
  --run-id "$RUN_ID-confirm" \
  --run
```

The generated cases use:

```text
--reuse-existing
--eval-query-count 3000
--warmup-query-count 500
--stability-repeats 3
--batch-size 200
```

For Orion reuse, also use `--recover-routing-from-collection`. The reuse guard
will reject any collection whose dimension, distance, lower HNSW configuration,
point count, shard count, replication factor, or physical placement differs from
the requested run. Keep the selected `upper_k/base_ef/factor` or
`nprobe/fixed_ef` as a one-element candidate list.

## Evidence and interpretation

Each result `summary.json` includes repository commit and dirty state, image
tag/ID, dataset SHA-256 and size, process CPU affinity, complete command,
deployment manifest, `/cluster` snapshot, full collection-cluster snapshot,
worker shard/point estimates, placement validity, Recall/QPS/P50/P95/P99,
visited logical shards, estimated EF sum, physical peers per query where route
plans are available, and controller/worker log tails.

Do not claim an Orion win in advance. Report a QPS improvement only for points
within `abs(recall delta) <= 0.003`. If Orion beats Naive but not Simple KMeans,
the supported conclusion is only that routing improves on all-shards search. If
a method cannot reach `0.95`, report its maximum recall and frontier. Preserve
tail-latency results even when they disagree with aggregate QPS.

## Final 0.95/0.97 submission confirmation

With the fixed cluster run id `dist-20260717-initial`, reproduce the final six
points with one command:

```bash
/users/dry/orion-distributed/venv/bin/python \
  tools/method4_benchmark_matrix.py \
  --config tools/benchmark_configs/method4_distributed_main_same_recall_confirmation_batch200.json \
  --run-id main-same-recall-pipelined-preencoded-final-20260718f \
  --run
```

The config freezes `lower_execution_order=query_major`,
`fixed_ef_shard_chunk_size=0`, batch size 200, 500 warmup queries, 3,000 timed
queries, and three stability repeats. The two Orion cases use
`routed_planning_mode=pipelined`; the lower HTTP requests remain serial, while
the benchmark-client CPU set prepares and JSON-encodes only the next batch. The
0.97 Naive point uses fixed EF 100 because EF 96 finished outside the declared
0.003 target window on 3,000 queries.

All six final points satisfy the strict target window. Orion has the highest
mean QPS and lower mean P50/P95 at both targets, but the 0.97 P99 comparison
must remain qualified: Orion is about 0.56% higher than Naive and substantially
more variable across repeats. Use the final result document, not the historical
tables below, for the complete numbers and limitations.

Every formal case records controller FD and P2P counts before and after the run.
`--require-clean-runtime` rejects a case when transport resources exceed the
steady-state two-channel-per-worker pool plus tolerance, or when logs, consensus,
collection health, optimizer state, shard state, or transfers are not clean.

The result interpretation, stability-mean tables, compact aggregate, and figures
are in `docs/experiments/2026-07-18-method4-four-node-submission-results.md` and
`results/method4_distributed_submission/final_report/`.

## Historical first reference run

The first full execution used run id `dist-20260717-initial`. Its raw 129-point
scan, five plots, confirmation matrix, six 3,000-query confirmation cases, and
conservative interpretation are under:

```text
results/method4_distributed/method4_distributed_glove200_initial/
  dist-20260717-initial/
```

Read `initial_findings.md` there before quoting results. In particular, the
confirmation run changed the 0.90 Orion classification from tuning-time
`strict` to confirmed `nearest`, and only the 0.95 Orion point remained a strict
target match. `same_recall_confirmation.csv` records both the selected tuning
status and the status recomputed from the 3,000-query confirmation mean.

## Historical tuned confirmation (superseded; do not cite)

The table in this section is retained only to explain the debugging history. Its
Naive QPS rows used the ungrouped transport fallback described at the top of this
runbook. Do not combine any row from this table with the final 2026-07-18 matrix,
and do not cite its Orion-versus-Naive speedups as distributed algorithm results.

After the initial scan, all three methods were re-tuned with `batch_size=200`
on the same collections and hardware. Simple KMeans remained a single-assignment
centroid partition using only `nprobe` and one fixed lower HNSW EF. Its factor was
zero, and it did not use Orion voting, fission, MultiEP entry-point routing, or
Dynamic EF. Naive continued to visit all 46 logical shards with one fixed EF.

The final confirmation command is:

```bash
python3 tools/method4_benchmark_matrix.py \
  --config tools/benchmark_configs/method4_distributed_same_recall_confirmation_strict_batch200.json \
  --run-id confirm-strict-20260717a \
  --run
```

Every case used 3,000 evaluation queries, 500 warmup queries, three stability
repeats, batch size 200, client CPUs `8-19`, the same image ID, and the same
round-robin `16/15/15` worker shard placement.

| Target | Method | Parameters | Recall@10 | QPS mean ± stdev | P50 / P95 / P99 ms | Visited shards | EF-sum/query |
|---:|---|---|---:|---:|---:|---:|---:|
| 0.90 | Orion | `upper_k=36, base=48, factor=15` | 0.89983 | 1053.36 ± 13.80 | 144.35 / 147.68 / 148.72 | 10.61 | 1098.64 |
| 0.90 | Simple KMeans | `nprobe=19, fixed_ef=110` | 0.90213 | 947.58 ± 7.31 | 196.87 / 204.05 / 206.76 | 19.00 | 2090.00 |
| 0.90 | Naive | `fixed_ef=37` | 0.90130 | 325.78 ± 37.66 | 584.77 / 608.81 / 612.87 | 46.00 | 1702.00 |
| 0.95 | Orion | `upper_k=104, base=64, factor=15` | 0.95160 | 667.26 ± 8.99 | 240.91 / 254.91 / 256.36 | 19.51 | 2963.04 |
| 0.95 | Simple KMeans | `nprobe=26, fixed_ef=210` | 0.95153 | 511.04 ± 2.78 | 365.96 / 431.46 / 432.69 | 26.00 | 5460.00 |
| 0.95 | Naive | `fixed_ef=72` | 0.95157 | 289.25 ± 22.23 | 608.48 / 645.60 / 646.23 | 46.00 | 3312.00 |

At the approximately 0.90 operating point, the maximum pairwise recall gap is
0.00230. Orion QPS is 11.16% higher than Simple KMeans and 223.34% higher than
Naive. Orion is 0.00017 below the nominal 0.90 target after expanding the query
set, so its target classification is conservatively `nearest`; the three methods
are nevertheless inside the predeclared 0.003 pairwise same-recall window.

At the approximately 0.95 point, the maximum pairwise recall gap is only 0.00007.
Orion QPS is 30.57% higher than Simple KMeans and 130.69% higher than Naive. Tail
latency agrees with the QPS result at both recall levels.

The authoritative aggregate CSV is:

```text
results/method4_distributed_confirmation/
  method4_distributed_same_recall_confirmation_strict_batch200/
  confirm-strict-20260717a/matrix_summary.csv
```

The result supports both conclusions in this four-node configuration: Orion
routing is better than all-shards search, and the retained topology/voting/
fission/MultiEP/Dynamic-EF design is faster than an independently optimized
Simple KMeans `nprobe + fixed EF` baseline at matched recall. This is an initial
GloVe-200/Cosine result, not a claim about other datasets or higher recall levels.
