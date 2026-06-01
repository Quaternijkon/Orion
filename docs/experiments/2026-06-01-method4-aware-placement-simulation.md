# Method4-Aware Placement Simulation

Date: 2026-06-01

Implementation commit:

```text
bc35767f3 Add method4-aware placement simulation
```

## Purpose

Simulate one of the proposed architecture optimizations without changing the
method4 search algorithm: place routed shards using actual method4 query traces
instead of physical round-robin shard assignment.

The simulation uses each query's routed shards and per-shard dynamic EF as a
cost proxy:

```text
cost(query, shard) = hnsw_ef_by_shard[shard]
```

For a placement, each query's tail proxy is:

```text
max_peer_load(query) = max(sum(cost(query, shard) for shard assigned to peer))
```

This approximates the slowest bottom worker for that query.

## Experiment Command

```bash
env PYTEST_ADDOPTS= LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 python3 tools/qdrant_two_level_routing_experiment.py \
  --base-url http://localhost:6833 \
  --collection qdrant_controller_idea_full_20260521 \
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
  --shard-placement round_robin \
  --placement-peer-uri-contains qdrant_shard_ \
  --placement-simulation \
  --placement-simulation-peer-count 3 \
  --output-dir results/qdrant_goal_recall_idea_095_placement_simulation
```

Result directory:

```text
results/qdrant_goal_recall_idea_095_placement_simulation/20260601_075041
```

## Search Result

| Metric | Value |
|---|---:|
| Recall@10 | 0.953667 |
| QPS | 337.271 |
| Avg visited shards | 22.654 |
| Avg EF per visited shard | 142.326 |
| Search batch calls | 15 |

This run is consistent with the previous strict high-recall method4 point.

## Placement Simulation Result

| Metric | Round-robin | Method4-aware | Improvement |
|---|---:|---:|---:|
| Avg query max peer load | 1429.408 | 1380.677 | 3.41% |
| P95 query max peer load | 1808.000 | 1720.000 | 4.87% |
| Max query max peer load | 2152.000 | 2016.000 | 6.32% |
| Max total peer load | 3548896 | 3272592 | 7.79% |
| Min total peer load | 3051288 | 3169136 | - |
| Peer load CV | 7.12% | 1.32% | - |
| Avg active peers per query | 2.974 | 2.978 | - |

The method4-aware placement changes 32 of 46 shard assignments while keeping
the shard count per peer balanced:

| Placement | Peer 0 | Peer 1 | Peer 2 |
|---|---:|---:|---:|
| Round-robin | 16 | 15 | 15 |
| Method4-aware | 15 | 15 | 16 |

## Interpretation

This simulation supports the architecture claim that method4 should not rely on
plain round-robin physical shard placement. The routing trace is not uniform:
some shards are hotter, and some hot shards are co-routed. Reassigning shards
with the method4 routing trace reduces both query-level tail load and long-run
peer imbalance without changing:

- upper-tier search,
- shard selection,
- routed entry points,
- per-shard dynamic EF,
- lower-tier HNSW search,
- final dedup/merge semantics.

The estimated 3-5% query-tail reduction is a system-level gain that stacks on
top of the already observed 20% high-recall QPS advantage. The larger 7.79%
reduction in max total peer load suggests that real deployment may also benefit
under concurrent query pressure, where peer queueing amplifies load imbalance.

## Next Step

Turn this from a simulator into a deployment strategy:

1. Use the generated method4-aware placement map when creating custom shard
   keys instead of `round_robin`.
2. Rebuild or clone the idea collection with that placement.
3. Re-run the paired high-recall comparison against naive.

If the real QPS gain is smaller than the simulation, the next bottleneck is
likely controller fanout or worker queueing rather than placement alone.
