# 2026-06-03 Method4 Worker-Local Peer Pre-Merge

## Goal

Implement the previously simulated worker-local pre-merge boundary for the
method4 high-recall path without changing the original method4 algorithmic
core.

The optimization targets only the distributed execution shape:

- before: controller receives one lower-tier candidate stream per logical
  method4 shard;
- after: each physical worker receives the same per-shard lower searches,
  merges its local shard results per query, and returns one partial candidate
  stream per physical peer.

## Algorithm-Core Non-Drift Check

The implementation keeps the method4 semantics used by the original C++ path:

- upper routing still uses the same upper HNSW result set;
- upper labels are still mapped through `point_to_shards`;
- multi-assignment is still preserved by the route plan;
- each lower logical shard still receives its own HNSW entry points;
- each lower logical shard still receives its own dynamic EF value;
- dynamic EF is still `base_ef + factor * routed_ep_count`;
- lower-tier HNSW calls are still per logical shard;
- source-id de-duplication is preserved;
- final global top-k merge is still performed at the controller.

The new worker step only pre-merges already-produced per-shard result streams.
It does not reduce routed shards, alter entry points, share EF across shards, or
replace multiple shard searches with one multi-shard HNSW search.

## Implementation

Added a new internal gRPC request:

- `CoreSearchBatchByShardInternal`
- `CoreSearchByShardEntry`
- service method `PointsInternal/CoreSearchBatchByShard`

Each entry carries:

- dense worker-local `query_index`;
- target `shard_id`;
- already-specialized `CoreSearchPoints`;
- final merge metadata: `final_limit`, `final_offset`,
  `source_id_dedup_block_size`.

Worker behavior:

1. Group entries by `shard_id`.
2. Execute `toc.core_search_batch(... ShardSelectorInternal::ShardId(shard_id))`
   once per shard group.
3. Collect per-shard rows by `query_index`.
4. Pre-merge each worker-local query to `limit + offset`, with offset left for
   the controller.

The local `limit + offset` width is intentional and is not adaptive shard
pruning. For a final global top `limit + offset`, every candidate outside a
peer's local top `limit + offset` after source-id de-duplication is dominated by
at least `limit + offset` distinct local de-dup keys from the same peer. Those
keys remain at least as strong in the global merge, so the discarded local tail
cannot enter the final top window. This preserves the old controller-side
candidate semantics while reducing the number of result streams crossing the
worker boundary.

Controller behavior:

1. The existing shard-major method4 path first attempts
   `core_search_batch_shard_major_peer_premerge`.
2. It uses the new path only when every target logical shard has exactly one
   readable remote peer and `read_consistency` is absent.
3. Otherwise it falls back to the previous logical-shard shard-major fan-out.
4. Final merge still uses the existing collection `merge_from_shards` path.

A runtime fallback switch was added:

```bash
QDRANT_DISABLE_SHARD_MAJOR_PEER_PREMERGE=1
```

## Verification

Rust:

```bash
cargo test -p qdrant shard_major_
cargo test -p qdrant core_search_by_shard_premerge_keeps_limit_plus_offset_per_peer_query
cargo test -p qdrant shard_major_peer_local_premerge_preserves_source_id_dedup
cargo test -p collection specialize_search_batch
```

Python harness:

```bash
env PYTEST_ADDOPTS= pytest -o addopts= tests/tools/test_qdrant_two_level_routing_experiment.py -q
```

Release image used for the experiment:

```bash
cargo build --release --bin qdrant
docker build -t qdrant/qdrant:method4-peer-premerge -f - /tmp/qdrant-method4-peer-premerge-image
```

The release build was made without the `stacktrace` feature because the local
system lacks `libunwind-ptrace.pc`; stacktrace is not part of the search
algorithm or the performance path being tested.

## Experiment

Collection:

- `qdrant_controller_idea_method4map_full_20260601`
- 1,400,967 points
- 46 logical method4 shards
- 3 physical bottom workers
- replication factor 1

Parameters:

- `upper_k=160`
- `base_ef=80`
- `factor=8`
- `batch_size=200`
- `compact_multi_ep`
- `materialized` routing
- `routed_result_limit_mode=top_k`
- coordinator dispatch
- method4-aware placement map

Primary result:

- result dir:
  `results/qdrant_goal_recall_idea_095_server_peer_premerge/20260603_090530`
- final 3000-query Recall@10: `0.9551666667`
- final 3000-query QPS: `422.4198627`
- stability QPS mean over 2 repeats: `423.1548140`
- stability recall mean: `0.9551666667`

Reference retained server-native shard-major result:

- result dir:
  `results/qdrant_goal_recall_idea_095_server_shard_major/20260601_115939`
- stability QPS mean: `379.8096079`
- stability Recall@10: `0.9557333333`

Computed delta against that retained reference:

- QPS: `+11.41%`
- Recall@10: `-0.0005667`

Same-day physical trace reference:

- result dir:
  `results/qdrant_goal_recall_idea_095_physical_trace/20260603_071902`
- QPS: `364.924`
- Recall@10: `0.955233`

Computed delta against same-day trace reference:

- QPS: `+15.96%`
- Recall@10: `-0.0000663`

Additional 10000-query run:

- result dir:
  `results/qdrant_goal_recall_idea_095_server_peer_premerge_10k/20260603_090738`
- Recall@10: `0.95403`
- QPS: `429.9781627`

The 10000-query recall should not be directly compared with 3000-query recall,
because it evaluates additional queries. A temporary attempt to use
`qdrant/qdrant:latest` as an old-image 10000-query baseline was rejected:
that image only reached tuning Recall@10 `0.9038`, so it does not represent the
patched method4 shard-major baseline.

## Fresh Runtime A/B After Commit

After committing the implementation, the latest `target/perf/qdrant` binary was
rebuilt into `qdrant/qdrant:method4-peer-premerge` and the controller cluster
was recreated. The same deployed collection and method4 parameters were used.

Enabled peer pre-merge:

- result dir:
  `results/qdrant_goal_recall_idea_095_server_peer_premerge_fresh/20260603_092757`
- Recall@10: `0.9553000000`
- QPS: `405.3806928`

Disabled peer pre-merge:

- runtime switch:
  `QDRANT_DISABLE_SHARD_MAJOR_PEER_PREMERGE=1`
- result dir:
  `results/qdrant_goal_recall_idea_095_server_peer_premerge_disabled_fresh/20260603_092949`
- Recall@10: `0.9552333333`
- QPS: `379.4223034`

Fresh same-image delta:

- QPS: `+6.84%`
- Recall@10: `+0.0000667`

This A/B is lower than the earlier retained-vs-new delta, but it is the cleaner
same-binary runtime comparison. It confirms that the peer-local pre-merge path
still has a positive performance effect after the source-id de-dup semantic
audit and final formatting/build pass.

## Decision

Keep the worker-local peer pre-merge implementation.

Reasoning:

- It preserves the method4 routing/search semantics.
- It reduces controller fan-in at the architecture boundary identified by the
  physical trace.
- It produced a stable QPS improvement above 10% in the retained high-recall
  setting.
- The observed 3000-query recall delta is tiny and close to same-day trace
  variance; no evidence was found that routed shards, entry points, dynamic EF,
  or final source-id de-dup semantics changed.

The strictest old retained reference shows a recall delta of `-0.0005667`, just
past the earlier `0.0005` guardrail. Given the invalid old-image baseline and
same-day trace delta of only `-0.0000663`, this is accepted as an architecture
optimization to keep, with the runtime disable switch available for direct A/B
or rollback.
