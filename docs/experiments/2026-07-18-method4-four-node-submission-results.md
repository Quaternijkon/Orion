# Method4/Orion Four-Node Distributed Initial Results

> [!WARNING]
> **Historical custom-routing result.** These measurements belong to the
> earlier benchmark/client-hint four-node implementation. They must not be
> relabeled as results of the native numeric-auto-shard coordinator path. The
> authoritative native-v4 comparison is
> [`2026-07-23-orion-native-v4-four-node-results.md`](2026-07-23-orion-native-v4-four-node-results.md).

## Result summary

The final fresh six-point run is:

```text
cluster run id: dist-20260717-initial
matrix run id:  main-same-recall-pipelined-preencoded-final-20260718f
```

On GloVe-200/Cosine, Orion is the highest-throughput method at both declared
same-recall targets. At approximately 0.95 recall it is 42.60% faster than Simple
KMeans and 27.49% faster than Naive all-shards. At approximately 0.97 recall it
is 37.05% faster than Simple KMeans and 3.94% faster than Naive all-shards.

All six points satisfy the predeclared target rule:

```text
abs(actual Recall@10 - target recall) <= 0.003
```

All runs also passed the distributed runtime gate: four healthy peers, zero
pending consensus operations, no message-send failures, no shard transfers, all
46 lower shards Active and remote, controller lower-local-shard count zero,
optimizer/update queues clean, and controller transport resources stable at six
P2P connections before and after each case.

## Frozen environment

| Item | Value |
|---|---|
| Controller/client | `hp057`, `10.10.1.1`; Qdrant CPUs `0-7`, client CPUs `8-19` |
| Workers | `hp052/.2`, `hp065/.3`, `hp076/.4`; Qdrant CPUs `0-19` |
| Logical lower shards | 46 custom shards, RF=1, remote-only placement `16/15/15` |
| Qdrant image | `sha256:219d98992e52e2a06d9a5692e601669aff3cf1f8ffd049a077f662c1c4df29f0` |
| Qdrant source commit | `1a5ac4c47237b9224ae3e4ca28c2cefb2b514352` |
| Container nofile | `65536:65536` on all four nodes |
| Query endpoint | Controller REST only, `http://10.10.1.1:6333` |
| Queries | 500 warmup, 3,000 timed, three stability repeats, batch size 200 |
| Dataset | `glove-200-angular.hdf5`, 1,183,514 train × 200, 10,000 test × 200 |
| Distance/top-k | Cosine/Angular, Recall@10 |
| Dataset SHA-256 | `4839085e5a8bb293434a1a66e1aa0193afc3f07c6797a85f1dbd91656172da20` |
| Dataset source | `https://ann-benchmarks.com/glove-200-angular.hdf5` |

The local dataset path used by the run was:

```text
/users/dry/orion-distributed/datasets/glove-200-angular.hdf5
```

It is 962,819,488 bytes and is intentionally excluded from Git.

## Implementation faults found before the final run

The first distributed measurements were not accepted at face value. Two concrete
implementation faults materially changed the results.

### Controller file-descriptor limit

The original controller container had a soft nofile limit of 1,024 and emitted
`Too many open files`. The orchestrator now launches all four containers with:

```text
--ulimit nofile=65536:65536
```

The value is part of the container reuse fingerprint, status output, manifest,
and tests.

### Naive fixed-EF request bypassed physical-peer grouping

The sequential Naive evaluator built its request separately from the fixed-EF
plan builder and omitted `hnsw_ef_by_shard`. Qdrant therefore did not recognize
the request as shard-major per-shard HNSW work. A 200-query × 46-shard request
could grow the controller from about 6 to more than 2,000 P2P connections and
produce highly unstable, artificially low Naive QPS.

The corrected request contains all 46 shard keys and a uniform per-shard map:

```text
hnsw_ef_by_shard[every logical shard] = the same fixed EF
```

This is transport metadata only. Naive still searches every one of the 46 shards
with exactly one fixed EF. The final main configuration explicitly keeps
`fixed_ef_shard_chunk_size=0`, so it remains one search object per query and one
HTTP batch per query batch. The optional staged chunker is retained only as a
diagnostic fallback.

After the fix, the clean 200-query EF72 diagnostic kept controller resources at
approximately 63 FDs and 6 P2P connections, while QPS recovered from polluted
runs in the low hundreds to roughly 700. All earlier Naive rows from the
ungrouped path are superseded and must not be cited as algorithm results.

## Orion distributed implementation optimization

The first clean 0.95 comparison showed Orion at 667.75 QPS and Naive at 682.59
QPS. Lower-tier batch latency showed the opposite relationship: Orion lower
searches were faster, but the old evaluator serialized all upper HNSW routing and
all Python route-plan construction before sending the first remote batch.

The final implementation uses a one-batch look-ahead pipeline:

```text
plan and encode batch 1
  -> execute distributed lower batch 1
     while planning and encoding batch 2 on client CPUs 8-19
  -> execute distributed lower batch 2
     while planning and encoding batch 3
  -> ...
```

Only one lower HTTP batch is in flight. The QPS timer starts before planning the
first batch and stops after the last lower result; routing and serialization work
is not removed from the measurement. The next batch is encoded with the exact
same `json.dumps({"searches": ...}).encode()` bytes that the non-pipelined path
would send.

Same-index tests compare materialized and pipelined execution query by query and
require identical complete request dictionaries and identical aggregate work.
The final semantic guard also passes all nine Method4 wrapper invariants.

This scheduling optimization does not change:

- the global upper HNSW sample (`N/32`) or search;
- `K_OVERLAP=10` construction semantics;
- balanced KMeans initialization, topology convergence, load recalibration, or fission;
- voting multi-assignment or `point_to_shards`;
- the routed logical shard set;
- each shard's ordered MultiEP list;
- `EF = base + factor × routed_ep_count`;
- source-ID copy deduplication;
- worker-local native peer pre-merge;
- controller final global top-k merge;
- the prohibition on adaptive shard pruning.

It is therefore a distributed execution/scheduling optimization of the original
main idea, not an alternative ANN algorithm or a relaxed recall point.

## Final same-recall results

All performance and latency values below are arithmetic means over the three
3,000-query stability repeats. The `±` term is sample standard deviation across
those repeats. P50/P95/P99 are also aggregated from `stability_runs.csv`, rather
than copied from a single representative `final_row`.

| Target | Method | Parameters | Recall@10 | QPS mean ± sd | P50 / P95 / P99 ms | Avg shards | Avg EF/shard | EF-sum/query |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 0.95 | Orion | `u104,b64,f15` | 0.95187 | 874.17 ± 16.15 | 219.07 / 237.39 / 250.35 | 19.50 | 151.92 | 2961.92 |
| 0.95 | Simple KMeans | `nprobe32,ef180` | 0.95223 | 613.00 ± 1.56 | 309.16 / 320.77 / 327.53 | 32.00 | 180.00 | 5760.00 |
| 0.95 | Naive all-shards | `ef72` | 0.95157 | 685.68 ± 12.12 | 291.01 / 304.28 / 305.51 | 46.00 | 72.00 | 3312.00 |
| 0.97 | Orion | `u144,b88,f20` | 0.96963 | 657.41 ± 8.51 | 293.93 / 316.06 / 332.13 | 22.91 | 226.46 | 5187.18 |
| 0.97 | Simple KMeans | `nprobe32,ef280` | 0.97087 | 479.70 ± 4.04 | 401.35 / 408.81 / 409.20 | 32.00 | 280.00 | 8960.00 |
| 0.97 | Naive all-shards | `ef100` | 0.96813 | 632.51 ± 5.09 | 315.67 / 325.70 / 330.30 | 46.00 | 100.00 | 4600.00 |

At target 0.95, the maximum pairwise recall gap is 0.00067. Orion provides:

- 42.60% higher QPS than Simple KMeans;
- 27.49% higher QPS than Naive;
- 25.99% and 21.98% lower mean P95 than Simple and Naive;
- 39.07% and 57.62% fewer visited shards;
- 48.58% lower EF-sum than Simple and 10.57% lower EF-sum than Naive.

At target 0.97, the maximum pairwise recall gap is 0.00273. Orion provides:

- 37.05% higher QPS than Simple KMeans;
- 3.94% higher QPS than Naive;
- 22.69% lower mean P95 than Simple and 2.96% lower mean P95 than Naive;
- 28.42% and 50.21% fewer visited shards;
- 42.11% lower EF-sum than Simple.

The 0.97 P99 result must be kept with the QPS conclusion: Orion's mean P99 is
0.56% higher than Naive's and has much larger repeat-to-repeat variation
(`43.29 ms` standard deviation versus `1.88 ms`). Thus the supported conclusion
is higher QPS and lower P50/P95 at matched recall, with effectively tied/slightly
worse and more variable P99 relative to Naive.

The QPS coefficient of variation is below 1.85% for every final case.

## Index cost and provenance

| Method | Indexed vectors | Expansion |
|---|---:|---:|
| Orion | 1,401,454 | 1.18415× |
| Simple KMeans | 1,183,514 | 1.00000× |
| Naive | 1,183,514 | 1.00000× |

The live schema, dimension, distance, HNSW parameters, point counts, shard count,
replication factor, and physical placement were validated before reuse. The new
Simple seed-1 collection additionally has verified canonical routing-build
metadata and fingerprint. Orion and Naive were built before that metadata schema
was introduced and are conservatively reported as `missing_unverified`; this is
not a detected mismatch, but full build provenance is not symmetric across the
three collections.

## Semantic and system evidence

### Claim H: wrapper semantic gate

The final 100-query semantic audit reports:

```text
9 / 9 invariants passed
0 failures
10,400 upper labels
1,972 routed shard visits
95 / 100 queries exercised multi-assignment
```

This supports sampled compact-wrapper and route-plan fidelity. It does not claim
that Qdrant's lower HNSW execution is bit-identical to the original C++ hnswlib
program.

### Claim C: Dynamic EF versus fixed EF

With partition and upper routing fixed, Dynamic EF achieved Recall@10 0.94877
and 690.89 QPS versus fixed EF Recall@10 0.95160 and 542.69 QPS. The recall gap
is 0.00283, within the declared pairwise window. Dynamic EF improved QPS by
27.31%, reduced estimated EF-sum by 41.97%, and reduced repeat-mean P95/P99 by
23.95%/22.16%. This is a near-matched-recall result; the Dynamic point is below
the nominal 0.95 target.

### Claim E: compact request versus client-expanded negative control

At batch size 50 and identical Recall@10 0.95173, compact execution achieved
592.19 QPS versus 202.38 QPS, reduced P95 by 67.24%, used one request/query
instead of 19.47, and reduced observed controller host-interface bytes/query by
50.84%. The byte value is a Linux interface counter, not precise per-RPC payload
instrumentation.

### Claim F: native worker peer pre-merge

At essentially identical recall, native peer pre-merge achieved 661.83 QPS
versus 549.50 QPS when disabled, a 20.44% increase. Mean P95/P99 decreased by
20.60%/20.47%. The route-derived execution shape is about 19.5 logical shards
but 2.95 physical peers per query; this is not a measured internal stream count.

## Reproduction and artifacts

The final command is:

```bash
/users/dry/orion-distributed/venv/bin/python \
  tools/method4_benchmark_matrix.py \
  --config tools/benchmark_configs/method4_distributed_main_same_recall_confirmation_batch200.json \
  --run-id main-same-recall-pipelined-preencoded-final-20260718f \
  --run
```

Primary raw aggregate directory:

```text
results/method4_distributed_submission/
  method4_distributed_main_same_recall_confirmation_batch200/
  main-same-recall-pipelined-preencoded-final-20260718f/
```

Compact deliverables:

```text
results/method4_distributed_submission/final_report/
  final_six_point_aggregate.csv
  final_cluster_acceptance.json
  same_recall_qps.png
  same_recall_tail_latency.png
  same_recall_work.png
  semantic_guard/
```

Large per-query CSVs, full `summary.json` files with environment log tails,
HDF5 data, Qdrant storage/indexes, and Docker image tar files are retained locally
but intentionally excluded from Git.

## Limitations

- This is one dataset (GloVe-200) and one distance (Cosine) on one fixed four-node topology.
- The controller and benchmark client share `hp057` but use disjoint CPU sets; memory and NIC contention are not isolated by separate hosts.
- Orion uses about 18.4% more indexed vectors because of voting multi-assignment.
- The final 0.97 P99 is more variable than both baselines and needs additional randomized/interleaved repeats.
- The semantic gate is a sampled wrapper-level audit, not a lower-HNSW bit-equivalence proof.
- Host-interface byte measurements are not exact Qdrant RPC attribution.
- Orion and Naive reuse validation lacks the newer routing-build metadata envelope, although live schema/count/placement checks pass.
- A second L2 dataset, 0.99 recall, placement A/B, and cluster scaling are outside this initial result.
