# 2026-06-03 Method4 Physical Execution Trace

## Goal

Quantify whether method4's high-recall routed search still has an architectural bottleneck after the retained server-native shard-major implementation.

This trace does not change search behavior. It reconstructs each materialized method4 route plan and maps the logical shard searches onto the actual physical Qdrant peers that own those shards. The goal is to estimate the upper bound for worker-local pre-merge:

- preserve `upper_k=160`, `base_ef=80`, `factor=8`;
- preserve `compact_multi_ep`, materialized planning, method4-aware placement, and `top_k` lower result limit;
- preserve per-shard dynamic EF, routed entry points, source-id de-dup, and final top-k merge semantics;
- only measure how many logical shard result streams could be collapsed into peer-local partial top-k streams.

## Collection And Baseline

- Idea collection: `qdrant_controller_idea_method4map_full_20260601`
- Dataset: `/home/taig/dry/faiss/datasets/glove-200-angular.hdf5`
- Controller: `http://localhost:6833`
- Placement map: `results/qdrant_goal_recall_idea_095_placement_simulation/20260601_075041/placement_simulation.json`
- Retained implementation checkpoint before this trace: `fcb2d7716`

The collection had 46 logical method4 shards after fission, distributed across 3 bottom workers. The controller itself did not own local shards for this collection.

## Trace Command

```bash
placement_json=$(ls -td results/qdrant_goal_recall_idea_095_placement_simulation/2026*/placement_simulation.json | head -1)
out="results/qdrant_goal_recall_idea_095_physical_trace"
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
  --search-dispatch-mode coordinator \
  --shard-placement map \
  --shard-placement-map "$placement_json" \
  --shard-placement-map-name method4_aware \
  --placement-peer-uri-contains qdrant_shard_ \
  --physical-execution-trace \
  --output-dir "$out"
```

Result directory:

- `results/qdrant_goal_recall_idea_095_physical_trace/20260603_071902`

## Search Metrics

| Metric | Value |
|---|---:|
| Recall@10 | 0.955233 |
| QPS | 364.924 |
| Eval queries | 3000 |
| Avg visited logical shards/query | 23.208 |
| Avg upper hits/query | 177.157 |
| Avg assigned EF per visited shard | 141.068 |
| Avg search requests/query | 1.0 |

These are measurement-run metrics, not a new performance claim. Stability repeats were intentionally disabled because this slice was about route/placement shape.

## Physical Execution Shape

| Trace Metric | Value |
|---|---:|
| Logical method4 shards | 46 |
| Physical bottom peers | 3 |
| Avg logical shard streams/query | 23.208 |
| P95 logical shard streams/query | 35 |
| Avg physical peers/query | 2.977 |
| P95 physical peers/query | 3 |
| Estimated controller stream reduction with worker-local pre-merge | 87.173% |
| Estimated controller candidate reduction with worker-local pre-merge | 87.173% |
| Avg assigned EF sum/query | 3273.899 |
| P95 assigned EF sum/query | 4344 |
| Avg max peer assigned EF/query | 1426.960 |
| P95 max peer assigned EF/query | 1752 |
| Peer load CV | 8.093% |

## Interpretation

At high recall, method4 intentionally visits many logical lower shards per query. In this run the average query hits about 23 logical shards, and the 95th percentile reaches 35. But those logical shards almost always collapse onto the same 3 physical bottom workers.

That makes worker-local pre-merge a stronger architectural candidate than controller micro-optimizations:

- Current controller receives and merges one result stream per logical shard.
- A worker-local pre-merge design would let each physical peer merge all of its local logical shard results for the same query and return only one partial top-k stream per peer.
- For this placement, that changes the average controller fan-in from 23.208 streams/query to 2.977 streams/query.
- With `top_k=10`, the same ratio applies to controller-side candidate volume before final merge.

The trace also shows that method4-aware placement is already reasonably balanced at the peer-load level: peer load CV is about 8.1%. The next bottleneck is less about reshuffling placement and more about removing unnecessary controller fan-in.

## Decision

Keep the trace instrumentation and move the next implementation slice to worker-local pre-merge.

Do not spend more time on small controller merge/grouping fast paths until the physical peer fan-in issue is tested. The 2026-06-02 fast-path A/B already showed no stable gain, while this trace exposes an 87% candidate-stream reduction opportunity at a larger architecture boundary.

## Acceptance Criteria For The Next Slice

Control:

- current retained server-native shard-major method4 path;
- `upper_k=160`, `base_ef=80`, `factor=8`;
- `compact_multi_ep`, materialized route planning, method4-aware placement;
- batch size 200;
- Recall@10 around 0.955.

Worker-local pre-merge candidate should be kept only if:

- Recall@10 is no lower than control by more than 0.0005;
- stable QPS improves by at least 5%, or stable p95 latency improves by at least 10%;
- external average search requests/query remains 1.0;
- method4 routing, per-shard dynamic EF, per-shard entry points, source-id de-dup, and final global top-k semantics remain unchanged.

Reject or revise the candidate if:

- QPS improvement is below 2% or unstable;
- recall drops beyond tolerance;
- preserving recall requires returning so many local candidates that the controller fan-in reduction disappears.
