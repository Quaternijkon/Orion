# Adaptive Orion navigation placement

Every serving node keeps its local data shards and the complete global upper
graph. No data movement or index rebuild is required. Upper navigation and
local shard searches share the configured search runtime; there are no fixed
navigator-only CPUs or worker-only CPUs.

With the same incoming queries and search parameters, the balancer redistributes
existing coordinator work. It preserves the selected worker searches and does
not generate additional queries to fill idle CPUs. Increasing the offered QPS
also increases worker demand; placement and offered load are separate controls.

For an eligible dense nearest-neighbor Orion request with automatic shard
selection, the ingress node may delegate the **entire coordinator operation**
(upper navigation, selected-shard fanout, result merging) to one peer. That peer
executes locally and cannot delegate again. Metadata filling, when needed by
the existing batch optimization, remains at the ingress. Explicit shard reads,
filtered/exact searches, and non-Orion collections retain their existing path.

## Load feedback

The scheduler measures process CPU consumption and thread CPU spent in actual
worker/navigation tasks, not time awaiting RPCs. It tracks queued blocking tasks
and in-flight coordinators across all collections on the process. Samples use
occupied CPU cores, bounded by the configured search-thread count and available
CPU budget (`QDRANT_NUM_CPUS`/process affinity). These are process measurements,
not host-wide CPU percentages. External processes should still be isolated with
CPU affinity/cgroups when sharing a machine.

CPU samples are smoothed with an EWMA (0.3 new, 0.7 previous). Every refresh
interval, the ingress adjusts its forwarding probability toward the cluster's
capacity-weighted utilization, with a ceiling of `target_utilization`. A worker
queue can trigger adjustment before the CPU average catches up. Adjustments are
bounded and use a deadband. A destination must have materially lower load and
spare capacity. Random selection is weighted by worker/other-CPU headroom and
outstanding coordinator work; reservations damp concurrent bursts.

The destination checks current node-wide load and atomically admits only a
bounded number of coordinators. Admission units are batches of up to 32 queries,
with at least one unit per request. The available navigator capacity is the
target CPU budget minus worker and other CPU usage. Its admission limit is rounded
down and may be zero; there is no minimum CPU reservation for navigation.
All reservations are released on completion,
error, or cancellation. Running blocking search tasks retain their normal
cancellation and timeout behavior.

## Configuration

`storage.performance.orion_navigation` in `config/config.yaml`:

| Setting | Default | Meaning |
| --- | ---: | --- |
| `enabled` | `true` | Enable in distributed Orion collections; `false` keeps coordination local. |
| `refresh_interval_ms` | 250 | Minimum spacing between background probes and probability adjustments. |
| `stale_after_ms` | 1500 | Maximum sample age and cooldown after a failed handoff. |
| `probe_timeout_ms` | 200 | Per-peer probe deadline; at most eight probes run concurrently. |
| `target_utilization` | 0.95 | CPU utilization ceiling when accepting delegated work. |
| `imbalance_threshold` | 0.05 | Minimum utilization difference for a handoff. |
| `adjustment_rate` | 0.15 | Feedback gain and maximum probability change per interval. |
| `max_inflight_per_cpu` | 8 | Admission units per core of estimated available navigator capacity. |
| `max_batch_queries` | 1024 | Largest batch eligible for handoff; larger batches execute locally. |

Configuration is read at process startup. Existing upper/lower HNSW budgets,
whitebox pruning, read consistency, offsets, metadata, and shard ownership are
preserved. Set `max_search_threads`/the CPU budget for the cores assigned to the
process; the balancer does not change CPU affinity or create more search CPUs.

## Failure handling and inspection

The private `PointsInternal.OrionNavigationLoad` RPC accepts
`{"collection_name":"..."}` and returns CPU/role/queue estimates, forwarding
probability, admission count, and local/forwarded/received query counters.
`forwarding_failures` counts failed handoffs, not background probe failures.
Probes start lazily with eligible ingress queries and never block those queries.

Only peers advertising the same generation, artifact SHA-256, and effective
whitebox policy fingerprint can receive a handoff. The policy fingerprint is
computed once from the small policy configuration, never from the graph on the
request path. Missing, stale, incompatible, disabled, or older peers are excluded.
The receiver validates identity both before and after execution.

An unavailable/overloaded peer, bad response, or failed handoff causes local
execution using the **remaining original deadline**. If no time remains, the
request times out. No retries to additional coordinators are made, and a routing
failure cannot silently become an all-shards search. Original ingress work is
not rejected by the remote admission limit.

This redistributes available navigation work. Filling every CPU still requires
enough concurrent queries and movable navigation cost. A single hot worker shard,
network/IO bottlenecks, or too little traffic can prevent full utilization even
after that node has delegated all its navigation. Throughput improvements must
be established with the same recall, concurrency, and CPU budget in a benchmark.

The scheduler keeps adapting even when a worker is saturated. It can offload all
eligible ingress navigation while continuing local worker service, and reclaim
navigation when worker demand falls. A worker-only bottleneck cannot be relieved
by another node taking still more navigation. Automatic client load ramping and
finding the maximum stable QPS remain outside this scheduler; it does not latch
a global stop or change the incoming query rate when such a bottleneck appears.
