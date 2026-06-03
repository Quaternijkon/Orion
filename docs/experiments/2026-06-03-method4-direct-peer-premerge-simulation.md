# 2026-06-03 Method4 Direct-Peer Pre-Merge Simulation

## Goal

Validate whether worker-local pre-merge can preserve method4 high-recall behavior while reducing the number of candidate streams that reach the final merge.

This is a semantic and architecture-shape experiment, not a production performance optimization. The simulation still sends every logical shard result back to the Python client and only then performs peer-local pre-merge. Therefore QPS from this experiment should not be used as the expected worker-local RPC speedup.

## Implementation Slice

Added a direct-peer local pre-merge simulation to `tools/qdrant_two_level_routing_experiment.py`:

- `--direct-peer-local-premerge` is only valid with `--search-dispatch-mode direct_peer`.
- Direct-peer shard-major execution still sends one lower search per logical shard.
- Results are grouped by `(query, physical_peer)`.
- Each peer group is locally merged with the same best-score-per-source-id rule as the final top-k merge.
- The final client merge then receives one candidate stream per active physical peer instead of one stream per logical shard.

The test suite now covers:

- peer-local top-k pre-merge preserves the global top-k over the current shard result sets, including cross-peer duplicate IDs;
- direct-peer shard-major execution reduces candidate streams when local pre-merge is enabled;
- the CLI flag is parsed correctly.

## Commands

Baseline direct-peer shard-major:

```bash
out="results/qdrant_goal_recall_idea_095_direct_peer_premerge_ab_baseline"
mkdir -p "$out"
env PYTEST_ADDOPTS= LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 python3 tools/qdrant_two_level_routing_experiment.py \
  --base-url http://localhost:6833 \
  --collection qdrant_controller_idea_method4map_full_20260601 \
  --hdf5-path /home/taig/dry/faiss/datasets/glove-200-angular.hdf5 \
  --routing-mode faithful_original_rest \
  --num-shards 31 \
  --upper-k-candidates 160 \
  --base-ef-candidates 80 \
  --factor-candidates 8 \
  --target-recall 0.95 \
  --eval-query-count 3000 \
  --tuning-query-count 500 \
  --stability-repeats 0 \
  --batch-size 200 \
  --reuse-existing \
  --recover-routing-from-collection \
  --routed-execution-mode compact_multi_ep \
  --routed-planning-mode materialized \
  --routed-result-limit-mode top_k \
  --search-dispatch-mode direct_peer \
  --direct-peer-http-urls \
    6015626418395790=http://localhost:6843 \
    2980601005324529=http://localhost:6853 \
    6846760844865837=http://localhost:6863 \
  --lower-execution-order shard_major \
  --output-dir "$out"
```

Pre-merge simulation:

```bash
out="results/qdrant_goal_recall_idea_095_direct_peer_premerge_ab_premerged"
mkdir -p "$out"
env PYTEST_ADDOPTS= LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 python3 tools/qdrant_two_level_routing_experiment.py \
  --base-url http://localhost:6833 \
  --collection qdrant_controller_idea_method4map_full_20260601 \
  --hdf5-path /home/taig/dry/faiss/datasets/glove-200-angular.hdf5 \
  --routing-mode faithful_original_rest \
  --num-shards 31 \
  --upper-k-candidates 160 \
  --base-ef-candidates 80 \
  --factor-candidates 8 \
  --target-recall 0.95 \
  --eval-query-count 3000 \
  --tuning-query-count 500 \
  --stability-repeats 0 \
  --batch-size 200 \
  --reuse-existing \
  --recover-routing-from-collection \
  --routed-execution-mode compact_multi_ep \
  --routed-planning-mode materialized \
  --routed-result-limit-mode top_k \
  --search-dispatch-mode direct_peer \
  --direct-peer-http-urls \
    6015626418395790=http://localhost:6843 \
    2980601005324529=http://localhost:6853 \
    6846760844865837=http://localhost:6863 \
  --lower-execution-order shard_major \
  --direct-peer-local-premerge \
  --output-dir "$out"
```

## Results

| Run | Result Dir | Recall@10 | QPS | Avg logical shard searches/query | Avg final candidate streams/query | Avg returned candidates/query |
|---|---|---:|---:|---:|---:|---:|
| Baseline direct-peer shard-major | `results/qdrant_goal_recall_idea_095_direct_peer_premerge_ab_baseline/20260603_080131` | 0.955567 | 154.302 | 23.208 | 23.208 | 232.077 |
| Pre-merge simulation | `results/qdrant_goal_recall_idea_095_direct_peer_premerge_ab_premerged/20260603_080247` | 0.955433 | 151.968 | 23.225 | 2.977 | 29.773 |

Candidate-stream reduction:

- final candidate streams/query: `23.208 -> 2.977`, about `87.17%` lower;
- returned candidates/query: `232.077 -> 29.773`, about `87.17%` lower;
- Recall@10 delta: `-0.000133`, within the planned tolerance for this diagnostic slice.

## Interpretation

The simulation confirms the key architectural property:

- peer-local top-k pre-merge preserves method4 recall over the current per-shard result sets;
- high-recall method4 routing creates about 23 logical lower shard streams per query;
- those streams collapse to about 3 physical peer streams per query;
- the expected controller merge and result deserialization pressure can therefore be reduced by roughly 87% if pre-merge happens inside the worker before the response crosses the network.

The QPS does not improve in this Python simulation because it does not remove the dominant costs:

- every logical shard search is still sent as a separate REST search object;
- every logical shard result is still serialized by the worker and deserialized by the client;
- peer-local merge is performed after the data has already crossed the process/network boundary.

So this is positive evidence for a real worker-local internal RPC, not a standalone optimization to keep for serving.

## Decision

Keep the simulation and metrics as an experiment harness.

Proceed to the real implementation boundary:

- add a peer-local internal request shape carrying `(query_index, shard_id, specialized_search)`;
- execute the same lower HNSW calls on the worker;
- merge per `(peer, query)` inside the worker using the same source-id de-dup rule;
- return only one partial top-k stream per `(peer, query)` to the controller;
- keep final controller merge unchanged except that its inputs are peer partials instead of logical shard streams.

This is now a justified architecture change. The next acceptance test should compare current server-native shard-major against worker-local RPC with identical `upper_k=160`, `base_ef=80`, `factor=8`, and keep it only if recall stays within tolerance and stable QPS or p95 latency improves.
