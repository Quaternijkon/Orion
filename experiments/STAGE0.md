# Stage 0 — Testbed and validity gates

This document defines the measurement testbed and the four gates every
performance run must satisfy. A QPS number is valid only if all four gates hold
at its operating point; otherwise it is a non-result. All measurement discipline
(routing and recall scoring kept out of the timed path) exists to keep the server
the bottleneck so that a throughput reading is a statement about the server.

## 1. Testbed

- One host, 112 logical cores, 503 GiB RAM.
- Four Qdrant peers (Orion fork) in a cluster, one container each, healthy; names
  match `qdrant-controller*` (`docker ps`).
- For measurement, every peer is given an identical cpuset at a fixed *total*
  server-core budget with `harness/configure_testbed.py` (e.g. 8 cores each);
  original cpusets are saved to `harness/cpuset_backup.json` and restorable with
  `--restore`. Holding total cores constant across shard counts keeps the resource
  budget fixed.
- Containers use cpuset pinning only — no CPU quota, no memory limit.
- The load generator runs on cores disjoint from all peer cpusets (free cores
  32–111 on this host); offered concurrency is set by `--inflight`, independent of
  the generator's process count.
- The host is quiet during a run. Any operator or co-tenant activity on a
  participating host counts against gate G4.

## 2. Scope

The measurable axis is matched-recall throughput as a function of shard count `P`
at fixed total server capacity, and of the layout and routing choice. Because the
four peers share LLC, memory bandwidth, and NUMA on one host, physical
multi-machine scale-out is not measurable on this testbed; it becomes measurable
on a real multi-host cluster (`STAGE1.md` §5, `DEPLOY.md` §7). On a single host the
transport is loopback, which is a known deviation from a networked deployment;
there is no network gate and G4 stands in its place.

## 3. Gates

A configuration is valid for performance reporting only if all four hold at the
reported operating point. Record all four with every run.

| Gate | Threshold |
|---|---|
| G1 server saturation | **max** peer CPU ≥ 85% of its own cpuset |
| G2 client headroom | client CPU < 50% of its reserved cores |
| G3 load-generator headroom | doubling client cores changes QPS by < 5% |
| G4 host interference | non-measurement host load < 5% of total cores |

G3 is decisive: if doubling the load generator raises QPS, the measurement is
client-bound and invalid regardless of G1 and G2. G1 asks whether *some* server
component is saturated, which is what makes a throughput reading a statement about
the server; it does not require every peer to be saturated. In scatter-gather the
coordinator carries work the data peers do not, so uniform saturation is
unachievable by construction. Peer imbalance is therefore a diagnostic
(`peer_imbalance_ratio`, least-loaded over most-loaded), not a gate.

## 4. Validated baseline

The harness is validated end to end on a 4-shard SIFT-100k collection with peers
pinned to symmetric 8-core cpusets. Offline mean Recall@10 is exactly 1.0000,
confirming that response ids, query offsets, and ground truth align. At 96
in-flight requests, `hnsw_ef` 128, 100k measured queries:

| Client cores | QPS | client CPU | max peer CPU | least-loaded peer |
|---:|---:|---:|---:|---:|
| 8 | 2678 | 18.2% | 89.8% | 73.1% |
| 16 | 2718 | 10.6% | 90.1% | 72.4% |

All four gates pass: G1 90.1% ≥ 85%, G2 10.6% < 50%, **G3 delta 1.50% < 5%**, G4
4.7% < 5%. Doubling the load generator leaves throughput unchanged while halving
client utilization, so the reading is server-bound.

At identical cpusets and client entry points spread across all four peers, the
coordinator peer saturates before the data peers (90% versus 72–79%, imbalance
ratio ≈ 0.80): the coordinator role, not local search, is the first thing to run
out of CPU.
