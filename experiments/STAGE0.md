# Stage 0 — Platform validity

Status: harness rebuilt and validated (all gates pass); exit measurement pending
Precondition for: every performance claim in the new measurement program.

No performance number may be interpreted, reported, or attributed to an
algorithm until this stage passes its gates. The previous program's only
SUPPORTED claim rested on a saturated load generator; this stage exists so that
cannot recur.

## 1. What went wrong before

The archived program measured "sub-linear physical scale-out" (legacy claim
C1-a) with a harness in which the **benchmark client performed the entire
scatter-gather in Python**. Per query, the client ranked shards, issued one HTTP
request per selected shard through a thread pool, blocked on all of them, merged
the results, and computed recall against ground truth — all inside the measured
path (`legacy/experiments/c1/scripts/c1_benchmark.py:750`, `execute_query`).

The metric named `aggregator_cpu_utilization_pct_of_reserved_cores` measured
that client process, not a server-side aggregator
(`legacy/experiments/c1/scripts/c1_e1_scaleout.py`, `aggregator_process_pids`),
and offered concurrency was capped by the number of client processes.

Because client cost per query grows with fan-out, throughput fell as shards were
added:

| SIFT1M, Random | M=1 | M=2 | M=4 |
|---|---:|---:|---:|
| QPS | 17884 | 11074 | 7044 |
| client CPU (of reserved cores) | 62.6% | 64.2% | 81.1% |
| worker CPU (max) | 51.0% | 27.7% | 20.9% |

Two cross-checks confirm the reading. With SIFT K-Means at fan-out 1 there is no
scatter-gather at all, yet the client still consumed 67-71% while workers sat at
24-26%, exposing a large fixed per-query client overhead. Conversely GloVe at
M=1 was the one worker-bound point (worker 83.7%, client 30.9%), because GloVe's
per-query server work is far larger (12830 distance computations versus 864 for
SIFT) — which is also why GloVe showed some scaling while SIFT scaled
negatively.

## 2. The testbed we actually have now

The legacy data came from a four-machine CloudLab cluster (`10.10.1.1-4`,
`/users/dry`, `/proj/intelisys-PG0`). **That cluster is no longer reachable.**
The current testbed is different and must be described honestly:

- one host, 112 logical cores, 503 GiB RAM;
- four Qdrant containers in an enabled 4-peer cluster, healthy;
- **asymmetric core pinning**: controller `0-7` (8 cores), `shard_1` `8-12`,
  `shard_2` `13-17`, `shard_3` `18-22` (5 cores each) — 23 server cores total;
- no CPU quota and no memory limit on any container (cpuset pinning only);
- cores `23-111` (89 cores) unused by Qdrant;
- host otherwise near-idle (load 4.69/112), except a `qemu-system-x86` VM
  consuming roughly two cores and other idle containers.

Two consequences follow directly.

**Physical scale-out is not measurable on this testbed.** Four containers on one
host share LLC, memory bandwidth, and NUMA topology; "adding a host" has no
meaning here. Any claim of the form "throughput scales sub-linearly in the
number of machines" is out of scope until real multi-host hardware is available.

**The measurable and honest axis is shard count at fixed total server
capacity.** This testbed can answer: *at a fixed compute budget, how does
matched-recall throughput change as the index is split into more shards, and how
does the layout plus routing choice affect that?* That is also the axis on which
the legacy end-to-end results showed a difference (advantage appearing only at
high shard counts), so it is the right question to pursue.

## 3. Required harness and testbed changes

1. **Move scatter-gather off the measured client.** Prefer one client request
   per query against the native distributed collection, letting the server
   coordinate. If client-side routing is required for an Orion variant, its
   per-query client cost must not grow with fan-out in the measured path.
2. **Take recall computation out of the hot path.** Persist result IDs; score
   recall offline.
3. **Symmetric, explicit server capacity.** Give every peer an identical cpuset,
   and hold *total* server cores constant across shard counts so that changing
   the shard count does not change the resource budget.
4. **Isolate the load generator** on cores disjoint from all peer cpusets (there
   are 89 free cores; use them) and make offered concurrency independent of the
   generator's process structure.
5. **Quiesce or pin co-tenants** (notably the `qemu-system-x86` VM) away from
   both the peer and client cpusets, and record host load with every run.

## 4. Gates

A configuration is valid for performance reporting only if all four hold at the
reported operating point. Record all four with every run.

| Gate | Threshold |
|---|---|
| G1 server saturation | **max** peer CPU >= 85% of its own cpuset |
| G2 client headroom | client CPU < 50% of its reserved cores |
| G3 load-generator headroom | doubling client cores changes QPS by < 5% |
| G4 host interference | non-measurement host load < 5% of total cores |

G3 is the decisive gate and the previous program had no equivalent. **If
doubling the load generator raises QPS, the measurement is client-bound and
invalid**, regardless of G1 and G2.

G1 asks whether *some* server component is saturated, which is what makes a
throughput reading a statement about the server. It deliberately does not
require every peer to be saturated. In scatter-gather the coordinator carries
work that data peers do not, so uniform saturation is unachievable by
construction; requiring it would conflate measurement validity with load
balance. Peer imbalance is therefore reported as a diagnostic
(`peer_imbalance_ratio`, least-loaded over most-loaded) rather than gating.

Because this is a single host, there is no network gate; G4 replaces it, and
loopback transport must be stated as a known deviation from a real deployment.
Note that agent or operator activity on the host counts against G4: the host
must be quiet during a measurement.

## 5. Harness validation (done)

The replacement harness lives in `harness/` and has been validated end to end
on a 4-shard SIFT-100k collection (`stage0_smoke_sift100k_s4`), with peers
re-pinned to symmetric 8-core cpusets by `harness/configure_testbed.py`
(original assignment saved to `harness/cpuset_backup.json`, restorable with
`--restore`).

Correctness: mean Recall@10 was exactly 1.0000 when scored offline, confirming
that response ids, query offsets, and ground truth all align.

Validity, at 96 in-flight requests, `hnsw_ef` 128, 100k measured queries:

| Client cores | QPS | client CPU | max peer CPU | least-loaded peer |
|---:|---:|---:|---:|---:|
| 8 | 2678 | 18.2% | 89.8% | 73.1% |
| 16 | 2718 | 10.6% | 90.1% | 72.4% |

All four gates pass: G1 90.1% >= 85%, G2 10.6% < 50%, **G3 delta 1.50% < 5%**,
G4 4.7% < 5%. Doubling the load generator left throughput unchanged while
halving client utilization, so the reading is server-bound. For contrast, the
legacy harness ran at 62-81% client CPU with workers at 20-37%.

One property is already visible and should carry into Stage 1: the **coordinator
peer saturates before the data peers** (90% versus 72-79%, imbalance ratio
about 0.80) even though every peer has an identical cpuset and client entry
points are spread across all four peers. In this architecture the coordinator
role, not local search, is the first thing to run out of CPU.

## 6. Exit criteria

Re-measure fixed-recall (Recall@10 >= 0.90) QPS for the unsharded baseline and
for one broadcast configuration under the corrected harness, and report all four
gates per configuration. Then record one of:

- **Phenomenon confirmed** — matched-recall throughput still degrades as shard
  count grows at fixed capacity, with servers saturated. Stage 1 proceeds on the
  shard-count axis.
- **Phenomenon was an artifact** — throughput holds or improves once G1-G4 hold.
  The scale-out premise is withdrawn and the direction is re-chosen. Record this
  outcome explicitly; do not modify the baseline to recover a degradation.

Also record explicitly that the legacy C1-a claim is **withdrawn**: it was
measured on hardware we no longer have, with a harness now known to be
client-bound.

## 7. Housekeeping

27 collections from previous experiments remain on the cluster. Inventory them,
keep only what a documented artifact needs, and remove the rest before
measuring, so that memory residency and cache behavior are not polluted.

Keep this document short. Expand the matrix only after the smallest
discriminating measurement has run.
