# C2-C3 Execution Plan

Authority: `plan/C2–C3 Experimental Protocol_ Topology Preservation and Shard Fan-Out.md`

This directory records the execution of the complete 1,472-line protocol. Existing C1 results
are inputs only where their raw artifacts satisfy the C2-C3 contract; they are not C2 or C3
evidence by themselves.

## Fixed experimental contract

- Datasets: SIFT1M (L2) and glove-200-angular (cosine).
- Query split: official queries `[0, 1000)` for tuning and `[1000, end)` for measurement.
- Logical shards: `M = 2, 4, 8, 16, 32` on four physical machines.
- Primary layouts: Random, K-Means, and Orion topology-aware partitioning.
- Preferred ablation: Orion-NoRefinement.
- Orion layouts are disjoint: multi-assignment and fission are disabled for this protocol.
- Local indexes use conventional Qdrant HNSW only, with identical construction/search settings.
- Online Orion routing, custom entry points, adaptive shard selection, adaptive per-shard EF, and
  topology-aware runtime pruning are excluded.

## Execution checklist

- [x] Stage 0: scaffold, dataset/ground-truth audit, instrumentation, reference-trace smoke.
- [x] Stage 1: SIFT1M full reference HNSW and global measurement-query traces.
- [x] Stage 2: SIFT1M E1 for all methods and shard counts.
- [x] Stage 3: SIFT1M E2 for all methods and shard counts.
- [x] Stage 4: inspect structural C2 evidence and append a timestamped result.
- [x] Stage 5: SIFT1M E3 local navigability matrix.
- [x] Stage 6: inspect navigability evidence and append a timestamped result.
- [x] Stage 7: SIFT1M E4 exact/HNSW fan-out, decomposition, and W90.
- [x] Stage 8: preliminary separate C2 and C3 verdicts.
- [x] Stage 9: repeat E1-E4 on glove-200-angular after SIFT consistency gates pass.
- [x] Stage 10: Orion-NoRefinement ablation.
- [x] Stage 11: causal correlations, balance analysis, and construction overheads.
- [x] Stage 12: all required figures/tables, final verdicts, and strict completion audit.

## Evidence boundary

No C2/C3 claim may be marked complete from code, tests, C1 measurements, or topology metrics alone.
The final audit must verify the E1 -> E2 -> E3 -> E4 chain on both datasets and must retain
contradictory results.

## Completion

Completed at `2026-08-22T09:46:07Z`. The strict final audit is
`logs/stage12-final-audit-v1.json`; all ten synthesis gates pass. The final separate verdicts are
`C2: CONTRADICTED` and `C3: CONTRADICTED`. Logical shard counts above four remain mapped onto four
physical hosts, and local-search microbenchmarks executed one logical shard at a time.
