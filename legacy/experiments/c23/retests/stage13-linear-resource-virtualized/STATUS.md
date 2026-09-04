# C2-C3 Stage 13 linear-resource retest status

Updated: 2026-08-24

Current stage: complete. The 42-configuration replacement matrix, independent
synthesis, and final audit are finished. Both C2 and C3 remain
`CONTRADICTED` under the linear-resource execution model.

Completed:

- Preserved the accepted C2/C3 evidence as a separate, immutable result line.
- Verified four reachable physical hosts, each with 16 physical cores, 32 SMT
  CPUs, about 128 GiB RAM, cgroup v2, Docker 29.1.3, and the same pinned Qdrant
  image ID.
- Defined one-container-per-logical-shard virtualization and an exact M-core
  server budget for `M={1,2,4,8,16,32}`.
- Defined disjoint server/client CPU pools and fixed per-shard memory caps.
- Implemented reversible four-host container lifecycle, cgroup/runtime audits,
  real index construction, E3 local-navigability curves, E4 oracle fan-out,
  compressed per-query evidence, and a resumable full-matrix registry.
- Passed empty-container resource smoke at M=1 and M=32. All 32 M=32 shard
  containers had disjoint one-physical-core cpusets, fixed 8 GiB memory caps,
  no swap, the pinned image ID, and no foreign running containers; the prior
  four-node C1 cluster was restored by original container ID.
- Passed a complete-index SIFT1M K-Means M=2 smoke including all nine E3 ef
  points, E4, positive graph/work counters, single-HNSW-per-shard gates, and
  cleanup restoration.
- Completed M=1 common-ef calibration with client/server core isolation:
  SIFT1M selects efSearch=24 at tuning Recall@10=0.9039; GloVe selects
  efSearch=320 at 0.9026.
- Preserved the first formal-matrix attempt with seven completed points under
  `invalid-attempts/stage13-matrix-missing-phase-cgroup-snapshots-20260824T1547Z`.
  These points are explicitly ineligible for final synthesis. The interrupted
  eighth point has a dedicated recovery record; all temporary containers were
  removed and the original four C1 containers were restored by original ID.
- Added cgroup CPU/memory snapshots at five boundaries: before upload, after
  upload, after indexing, after E4, and after E3. The final audit requires
  complete per-shard coverage, monotonic cumulative CPU time, memory within the
  8 GiB cap, zero swap, and no OOM at every boundary.
- Passed a new complete-index SIFT1M K-Means M=2 smoke with all five snapshots,
  the complete nine-point E3 curve, E4, raw-array output, cleanup restoration,
  and a clean post-run four-host preflight.
- Implemented the independent Stage 13 synthesis/audit program. It recomputes
  E3/E4 metrics from compressed per-query arrays, reproduces P_exact, P_HNSW,
  Delta_P, and W90, runs the required paired 10,000-replicate bootstrap,
  verifies the exact M/32 resource contract, and generates separate C2/C3
  verdicts, controls, tables, and figures after the matrix is complete.
- Completed all 42 formal configurations: two datasets, one common M=1 control
  per dataset, and four partition methods at M=2,4,8,16,32. Every manifest and
  raw-array checksum passes.
- Verified every configuration has exactly M physical-core equivalents, M/32
  of the declared M=32 server capacity, 8*M GiB memory capacity, all five
  phase snapshots, and exactly one non-empty HNSW per logical shard.
- Completed paired 9,000-query, 10,000-replicate bootstrap, three-seed M=32
  variance, refinement ablation, balance controls, eight CSV tables, and ten
  valid PDF figures.
- Passed all 15 final-audit checks and all 35 relevant implementation tests.
- Final C2 result: K-Means has lower TWCut at 8/10 dataset/M points and the
  topology-to-local-navigability chain does not consistently favor Orion.
- Final C3 result: P_HNSW significantly favors K-Means at 8/10 points, is
  inconclusive at two SIFT points, and never significantly favors Orion.
  No point passes the full C3 support rule.
- Retained severe Full Orion imbalance: M=32 max/mean is 4.930 on SIFT and
  5.887 on GloVe.

Not yet complete:

- None within the Stage 13 protocol. A balance-constrained partitioner or an
  online-router evaluation would be a separate future experiment and must not
  overwrite this result.

Primary evidence:

- `evidence-v1/final-audit.json`
- `evidence-v1/verdicts.json`
- `evidence-v1/tables/`
- `evidence-v1/figures/`
- `RESULTS_zh.md`
- Raw registry:
  `/proj/intelisys-PG0/exp/orion-distributed/c23-linear-20260824-v1/matrix/registry.json`
