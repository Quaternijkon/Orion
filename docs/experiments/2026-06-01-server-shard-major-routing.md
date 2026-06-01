# Server-Native Shard-Major Method4 Routing

Date: 2026-06-01

## Goal

Improve high-recall (`Recall@10 >= 0.95`) method4 QPS without changing the original method4 external semantics:

- upper HNSW routing remains unchanged;
- lower search still uses per-shard HNSW entry points and dynamic per-shard EF;
- external client request shape remains compact (`compact_multi_ep`, one search object per query);
- final result merge still de-duplicates copied source ids before applying top-k.

The intended system optimization is to move shard-major lower execution into Qdrant's controller path instead of expanding requests in Python.

## Implementation

Commits:

- `0b05ca739 Add shard-major lower execution mode`
- `37a15a556 Add server-side shard-major method4 search batching`

Two paths were tested:

1. Client-side shard-major:
   - Python expands each compact method4 query into one single-shard search per visited shard.
   - This confirms semantics but inflates REST batch payloads from `1` to about `23` search objects per query.

2. Server-native shard-major:
   - Python keeps the compact request shape.
   - Qdrant detects batch searches with per-shard HNSW maps (`hnsw_entry_points_by_shard` or `hnsw_ef_by_shard`).
   - The controller internally specializes each request per shard, executes one contiguous batch per shard key, and merges results back into original query order.
   - This keeps `avg_search_requests_per_query = 1.0` at the REST/client layer.

The running Docker cluster was updated by building `target/perf/qdrant`, copying the binary into the 4 existing containers, and restarting them. Existing storage volumes and collections were preserved.

## Results

Dataset: `glove-200-angular.hdf5`

Collection for idea: `qdrant_controller_idea_method4map_full_20260601`

Collection for naive: `qdrant_controller_naive46_full_20260521`

Common idea config:

- `upper_k=160`
- `base_ef=80`
- `factor=8`
- `batch_size=200`
- `routed_execution_mode=compact_multi_ep`
- `routed_planning_mode=materialized`
- `routed_result_limit_mode=top_k`
- `search_dispatch_mode=coordinator`

Naive same-binary config:

- `base_ef=72`
- `batch_size=100`
- `search_dispatch_mode=coordinator`

### Main Comparison

| Run | Mean QPS | Mean Recall@10 | Notes |
|---|---:|---:|---|
| naive after server change | 308.443 | 0.951767 | same new `perf` binary |
| old method4-aware placement | 336.046 | 0.955175 | previous binary, before server-native shard-major |
| compact materialized routing | 331.249 | 0.955133 | Python compact manifest only |
| client-side shard-major | 151.292 | 0.954967 | REST payload expands to about 23 searches/query |
| server-native shard-major | 378.366 | 0.955733 | compact client shape, shard-major inside controller |

Server-native shard-major gain:

- vs same-binary naive: `+22.67%` mean QPS
- vs previous method4-aware placement run: `+12.59%` mean QPS
- vs compact materialized routing: `+14.22%` mean QPS
- vs client-side shard-major: `+150.09%` mean QPS

### Server-Native Detail

Final row:

| Recall@10 | QPS | Avg visited shards | Avg upper hits | Avg EF / visited shard | Search batch calls | Avg search requests/query |
|---:|---:|---:|---:|---:|---:|---:|
| 0.955733 | 375.479 | 23.201 | 177.137 | 141.080 | 15 | 1.0 |

Stability rows:

| Run | Recall@10 | QPS |
|---:|---:|---:|
| 1 | 0.955733 | 380.760 |
| 2 | 0.955733 | 378.859 |

Stability mean: `379.810 QPS`.

### Same-Binary Naive Detail

Final row:

| Recall@10 | QPS | Avg visited shards | Avg EF / visited shard | Search batch calls | Avg search requests/query |
|---:|---:|---:|---:|---:|---:|
| 0.951767 | 310.550 | 46.000 | 72.000 | 30 | 1.0 |

Stability rows:

| Run | Recall@10 | QPS |
|---:|---:|---:|
| 1 | 0.951767 | 307.778 |
| 2 | 0.951767 | 307.002 |

Stability mean: `307.390 QPS`.

## Interpretation

The failed client-side shard-major run is the useful negative control: simply reordering lower searches is not enough if it forces the REST/client layer to materialize about 23 requests per query. That loses badly despite preserving method4 semantics.

The server-native version keeps the compact client/controller API shape and moves the split into Qdrant's internal batch path. That is the architectural point method4 needs at high recall: the algorithm routes to fewer shards than naive, but the system must avoid paying one external request object per routed shard. Once that is fixed, method4 recovers a clear high-recall QPS lead while preserving recall.

## Artifacts

- Server-native run: `results/qdrant_goal_recall_idea_095_server_shard_major/20260601_115939`
- Client-side shard-major negative run: `results/qdrant_goal_recall_idea_095_shard_major/20260601_113943`
- Same-binary naive run: `results/qdrant_goal_recall_naive_095_after_server_shard_major/20260601_120121`
- Compact materialized run: `results/qdrant_goal_recall_idea_095_compact_materialized/20260601_112623`

