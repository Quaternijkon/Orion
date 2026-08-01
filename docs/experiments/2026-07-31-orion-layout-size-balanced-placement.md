# Orion layout-size-balanced physical placement

## Outcome

The 16-shard Orion implementation now supports `layout_size_balanced` as a
physical placement strategy. It keeps Orion's logical index and search
algorithm unchanged, but assigns unequal logical shards to the three worker
peers using their offline physical-vector counts.

In the four-node GloVe-200 experiment, this reduced Orion's worker load
max/mean ratio from `1.29785` to `1.03627`. Under the same Search API,
500-query warmup, 3,000-query evaluation, batch size 200, and three repeats per
formal arm, the balanced Orion arms averaged `1626.40 QPS` at Recall@10
`0.898033`. The current HashAll baseline reached `1421.97 QPS` at Recall@10
`0.897233`. The formal-arm comparison is therefore `+14.38% QPS` and
`-13.37% P95` for Orion, with a Recall spread of `0.0008`.

The six-pair interleaved Orion/Hash experiment measured `+12.58%` QPS from arm
means, Orion won all six pairs, and the exact paired sign-flip p-value was
`0.03125`. This is evidence for this GloVe-200 four-node experiment window; it
is not a universal claim across datasets, replication modes, or deployments.

## Why placement is part of the physical architecture

Orion fission and multi-assignment intentionally produce unequal shard sizes.
The previous Qdrant-compatible implementation nevertheless used numeric shard
ID round-robin placement:

| Worker peer | Round-robin physical vectors | Balanced physical vectors |
|---|---:|---:|
| `526777670352377` | 573,512 | 437,939 |
| `1155247304029114` | 343,502 | 457,923 |
| `1777065618036149` | 408,667 | 429,819 |

The slowest physical peer is on the critical path of a distributed batch, so
logical route pruning alone did not fully translate into end-to-end QPS. The
new strategy uses deterministic, count-balanced LPT bin packing over
`build-manifest.json:routing.shard_counts`. These weights are produced during
offline layout construction; no benchmark query, ground-truth neighbor, Recall,
latency, or QPS measurement is an input to placement.

The resulting Orion placement is:

| Worker peer | Logical shards |
|---|---|
| `526777670352377` | 0, 4, 7, 9, 15 |
| `1155247304029114` | 2, 6, 10, 11, 12, 13 |
| `1777065618036149` | 1, 3, 5, 8, 14 |

The implementation also permutes the already-computed bins over physical
workers to minimize live shard moves. This permutation does not change the bin
contents or loads.

## Preserved Orion semantics

The optimization changes only the mapping from a logical numeric shard to a
physical worker peer. The following remain unchanged:

- upper-tier HNSW routing and its ordered per-query routes;
- `upper_k = upper_search_ef = 40`;
- entry points selected for each shard;
- Dynamic EF with base `48` and factor `14`;
- fission and multi-assignment;
- the visited logical shards and lower HNSW work;
- standard coordinator Search requests, global merge, IDs, and scores;
- peer-local premerge remains disabled for the selected configuration.

The 3,000-query production-router trace reports an average of `6.213` visited
logical shards and an average total per-query EF of `888.7907`.

Live moves use Qdrant snapshot shard transfer. `stream_records` is deliberately
not used by `move_numeric_shards_explicit`, because it reconstructs the target
HNSW graph and can change ANN results even when the logical data is identical.
The early `screen/balanced-disabled-r3` run used `stream_records` and is
excluded from all final semantic and performance claims.

For snapshot placement A/B/A, the same 200 queries produced identical ordered
results in all three states:

| Proof | SHA-256 |
|---|---|
| Query float32 bytes | `1cea9e588c5931b19f0f598276910ca06509487ee0ee86b4aff4b2279d3a1f93` |
| Ordered IDs | `1dc66056f94ec71386b76ad4eae0fda1410b66d0af266430c2315059ab6f8268` |
| Ordered IDs and float32 scores | `206007a987e571cedc3652b6444a855963ef778f96f78b2bc405d9a05517437d` |

## Formal results

All table rows use 16 shards, RF=1, HNSW `M=32`, `ef_construct=100`, the
ordinary per-shard replica-set transport, and the standard coordinator Search
API. Latency percentiles are percentiles of 200-query HTTP batches.

| Method and arm | Placement | Recall@10 | QPS mean ± SD | P95 |
|---|---|---:|---:|---:|
| HashAll, EF 68 | round-robin | 0.897233 | 1421.97 ± 8.23 | 146.44 ms |
| Orion sandwich B | round-robin | 0.898033 | 1501.52 ± 2.32 | 136.86 ms |
| Orion sandwich A1 | layout-size-balanced | 0.898033 | 1638.21 ± 13.29 | 126.35 ms |
| Orion sandwich A2 | layout-size-balanced | 0.898033 | 1614.59 ± 1.61 | 127.38 ms |
| Simple KMeans, nprobe 14, lower EF 120 | round-robin | 0.895433 | 1106.91 ± 3.08 | 185.86 ms |
| Simple KMeans, same search, fairness control | layout-size-balanced | 0.895433 | 1140.81 ± 4.77 | 183.15 ms |

The Simple KMeans fairness control applies the same physical-placement
optimization to that routed baseline. Snapshot transfer preserved its ordered
IDs and float32 scores. Orion's two balanced-arm mean is still `+42.57% QPS`
and `-30.73% P95` relative to balanced Simple KMeans, with a Recall spread of
`0.0026`.

The Orion placement sandwich isolates the architectural contribution:

- balanced A1/A2 mean versus Orion round-robin: `+8.32% QPS`, `-7.30% P95`,
  and identical Recall;
- Orion round-robin versus current HashAll: `+5.59% QPS`;
- balanced Orion versus current HashAll: `+14.38% QPS`.

The six interleaved Orion/Hash pairs used the order
`O-H, H-O, O-H, H-O, O-H, H-O`. Their mean paired delta was
`+181.32 QPS`, with paired 95% CI `[+165.10, +197.55] QPS`. Orion-first and
Hash-first mean deltas were `+181.01` and `+181.64 QPS`, respectively, so the
observed advantage is not explained by which method ran first.

## Implementation surface

`tools/native_auto_shard_prepare.py` exposes:

```text
--placement-strategy round_robin
--placement-strategy layout_size_balanced
```

`round_robin` remains the default, so existing HashAll and prior experiment
workflows do not silently change. `layout_size_balanced` is accepted only for
routed methods and requires a positive `routing.shard_counts` list whose length
matches the shard count and whose sum matches `physical_point_count`.

Preparation writes a `placement_map.json` that can be supplied to
`tools/native_auto_shard_benchmark.py --expected-placement-map ...`. The
benchmark then validates the exact live RF=1 placement instead of assuming
round-robin.

The reusable placement and migration functions are:

- `size_balanced_numeric_shard_targets(...)`;
- `move_numeric_shards_explicit(...)`;
- `validate_numeric_shard_explicit_placement(...)`.

Relevant validation completed with `138 passed`, Python bytecode compilation,
and `git diff --check`.

## Evidence

The compact generated summary is outside Git with the raw experiment data:

```text
/proj/intelisys-PG0/exp/orion-distributed/
native-20260730-policy-parity-s16-v1/
  final_optimized_summary.json
  final_optimized_summary.csv
  final_optimized_summary.sha256
```

The JSON records source paths and SHA-256 values for the placement plan,
snapshot identity probes, formal manifests, formal summaries, Simple KMeans
fairness control, and six-pair interleaved analysis. HDF5 data, per-query rows,
route traces, Qdrant storage/WAL/index files, and container images remain
outside Git.

## Lifecycle handoff

Before shutdown, all four nodes passed five consecutive readiness rounds, the
cluster reported four peers and zero pending operations, and the three formal
collections were green with 16 Active shards, no transfers, no controller-local
lower shards, and the expected placement maps.

The run was then stopped under both the run-scoped benchmark and lifecycle
locks using `down`. `clean` was not executed. All four containers are stopped
with exit code 0, all four HTTP ports are closed, and every storage path was
preserved with an exact zero-byte size delta across shutdown.
