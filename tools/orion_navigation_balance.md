# Adaptive Orion navigation placement

Every serving node keeps its local data shards and the complete global upper
graph. No data movement or index rebuild is required. Upper navigation and
local shard searches share the configured search runtime; there are no fixed
navigator-only CPUs or worker-only CPUs.

Senders supply the queries. On the direct search path the cluster adjusts how
many of them can run concurrently and where new queries enter. The original
points API uses ingress handoff for eligible searches.

For an eligible dense nearest-neighbor Orion request with automatic shard
selection, the ingress node may delegate the **entire coordinator operation**
(upper navigation, selected-shard fanout, result merging) to one peer. That peer
executes locally and cannot delegate again. Metadata filling, when needed by
the existing batch optimization, remains at the ingress. Explicit shard reads,
filtered/exact searches, and non-Orion collections retain their existing path.

## Worker-first dispatch from the sender

A sender with the concrete collection URLs can send each dense unfiltered query
directly to the machine that will run its upper search. The selected machine
owns fanout, shard replies, merge, and the response to the client. Reserve slots
in the background with `POST /collections/{name}/orion/coordinator/permits`.
An eligible query with a slot uses `POST /collections/{name}/orion/coordinator/search`.
The original points API continues to support ordinary ingress and peer handoff.

Every machine computes its weight in occupied CPU cores. Let `C` be the search
CPU capacity and let `T` be the configured utilization target. Estimate other CPU
as `max(0, process CPU − worker CPU − upper-search CPU)`. Treat a queued worker
as a claim on one core. The weight is
`max(0, T × C − worker CPU − other CPU − queued workers)`. Worker queue depth
is read at each observation. CPU counters refresh at intervals of at least
100 ms and use a 0.3-gain EWMA. RPC wait time consumes no measured CPU.

The node caps pending direct starts at `floor(weight × max_inflight_per_cpu)`.
It also accounts for the observed upper-search queue when issuing slots.
The cap is shared by every sender and collection on the process. With 48 cores
and defaults at most 364 upper starts can be pending on one node. A grant can
be partly consumed. Starting upper search releases its local start reservation.
The accepted query still occupies one cluster window slot while waiting for
shard results and merging them on the same coordinator.

At the first local observation yielding zero start budget, the node revokes all
unclaimed slots under the same lock used for redemption. A stale packet arriving
later receives HTTP 412 with `x-orion-admission: not-started`. An accepted query
already waiting to start still runs, and its count is exposed as `claimed_waiting`.
On a smaller positive budget, existing slots can be used until expiry. Their
count at the instant of adjustment is bounded by the previous outstanding slot
count. The default slot lifetime is 50 ms. The time from a physical change in
load to its observation also includes sampling, smoothing, and control traffic.

The reference sender uses fresh weights in smooth weighted round robin just as
an HTTP lane becomes available. It leaves waiting queries unassigned. Each
control request replenishes up to 32 slots and can serve multiple data requests.
The sender avoids using grants near expiry based on its own monotonic clock and
measured network RTT. Zero weight, an expired observation, or a failed control
request removes a destination until a new compatible grant arrives. The sender
checks collection generation and both artifact and policy fingerprints. A higher
generation invalidates old local slots. Recreate the dispatcher after an
intentional same-generation policy replacement.

### Cluster in-flight window

The elected Raft leader samples every committed member in the background at
`refresh_interval_ms` (250 ms by default). On Linux with an explicit CPU set matching the search
capacity, the measurement is mean busy time across those CPUs, including
interrupts and excluding idle, I/O wait, and steal. A logical thread/quota
budget without a matching CPU set, or an unavailable counter, uses service
process CPU divided by assigned search capacity. The highest fraction drives
the window. All members should use the same `target_utilization`, which
defaults to 0.95. Each control round uses the following rules.

1. Reduce the window when a CPU-ready worker or upper-search queue on the
   busiest machine stays nonempty between consecutive rounds. A queue drain
   counter distinguishes persistent work from two unrelated spikes.
2. Below target, grow when a query is waiting and recent peak occupancy has filled the window.
   Reassign stranded local capacity to waiting peers. Growth is proportional
   to the CPU shortfall and bounded by `adjustment_rate`. A decrease uses
   this same fraction. Both directions change at least one slot.
3. At target, hold the cluster window. Give zero new slots to peers at or
   above target while a cooler, interested peer can take coordination. If the
   hot worker still sits at target with coordination shed, completed queries
   are replaced within the current window.

The initial window is one query per assigned core across the members.
Each machine receives a reusable share. All senders and collections on it
share that allocation. An unused permit occupies one slot. Redemption converts
it into one in-flight query, charged through shard replies and final merge.
Cancellation and errors also release it. Shrinking a share revokes excess
reservations and lets accepted work drain. A slot released on one peer can be
lent to another in the same round after the donor acknowledges the lower quota
and reports the slots still occupied. The data request goes directly to its selected peer.

Control updates carry the Raft term, membership, and a one-use fence from each
peer's latest observation. A new leader reads existing quotas and occupancy
before allocating. An incomplete snapshot cannot grow the window. If updates
stop, node allocations expire after `max(stale_after_ms, refresh_interval_ms +
3 × probe_timeout_ms)`. Existing queries finish on their original coordinator.
The control path is idle until a sender requests work. A request to any serving
peer can wake the leader. After traffic subsides, the leader observes live
allocations until they expire so a returning sender on a zero-share peer can
resume in the next round. A cold sender may receive a zero grant until its
first share arrives.

A client sends `waiting: false` for prefetch and `waiting: true` when an
unassigned query needs a permit. Positive requests from clients that
omit the field are treated as demand. A zero-count observation has no effect
on demand. The response exposes `cluster_window`, `node_window`, and
`inflight`. Local occupied slots equal `inflight + pending - claimed_waiting`.
The cluster target can be temporarily lower than already accepted work while it
drains. QPS and latency can be evaluated separately from the control signals.

The C++ example uses libcurl 7.80 or later and json-c. Build from the repository root and pass node URLs
for one collection. Each input line is a normal JSON SearchRequest. The sender streams
concurrent requests as permits arrive and prints replies in input order. HTTPS and the
optional `QDRANT_API_KEY` work on both control and search requests.

```sh
g++ -std=c++17 -O2 -pthread examples/orion_direct_search.cpp -o /tmp/orion_direct_search $(pkg-config --cflags --libs libcurl json-c)
/tmp/orion_direct_search http://node1:6333/collections/vectors http://node2:6333/collections/vectors < queries.ndjson
```

A sender can reselect on the explicit unstarted response within its original
deadline. A transport failure after submission leaves ownership ambiguous. The
example uses a pre-transfer callback to prevent transparent POST replay on reused connections and returns ambiguous errors to the caller. To inspect the pool, submit
`{"count":0}` to the permit endpoint. The counters satisfy
`issued = pending + started + expired + revoked + cancelled_before_start`.

## Ingress handoff load feedback

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
| `refresh_interval_ms` | 250 | Control round period for the direct window and minimum spacing between ingress probes. |
| `stale_after_ms` | 1500 | Maximum sample age and cooldown after a failed handoff. |
| `probe_timeout_ms` | 200 | Per-peer probe deadline; at most eight probes run concurrently. |
| `target_utilization` | 0.95 | Busiest-node CPU target for direct flow control and CPU target for ingress handoff. |
| `imbalance_threshold` | 0.05 | Minimum utilization difference for a handoff. |
| `adjustment_rate` | 0.15 | Maximum fractional change of the direct window per round and feedback gain for handoff. |
| `max_inflight_per_cpu` | 8 | Admission units per core of estimated available navigator capacity. |
| `direct_permit_ttl_ms` | 50 | Lifetime of unclaimed sender-side start slots, in milliseconds. |
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
by another node taking still more navigation. Direct senders with pending work
use the cluster window described above. The legacy ingress handoff controls
coordinator placement only.
