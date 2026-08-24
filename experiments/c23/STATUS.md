# C2-C3 Execution Status

Updated: 2026-08-22T09:46:07Z

Current stage: Stage 12 complete. The full two-dataset protocol is finished and the strict final
audit passes all ten synthesis gates.

Final verdicts:

- C2 — Topology-oblivious partitioning disrupts graph-search topology: `CONTRADICTED`.
- C3 — Topology-aware partitioning reduces required shard fan-out: `CONTRADICTED`.

Completed:

- Stage 0 instrumentation, dataset/ground-truth audits, exact-disjoint partition contract, and
  Qdrant-native reference-trace smoke.
- Full SIFT1M and glove-200-angular reference HNSW builds, independent common-EF tuning, and 9,000
  complete measurement traces per dataset.
- E1-E4 for Random, full-data K-Means, Full Orion, and Orion-NoRefinement at logical
  `M=2,4,8,16,32` on the four-host model. For M>4, logical shards map by shard ID modulo four;
  local-search microbenchmarks execute one logical shard at a time.
- Paired 10,000-replicate query bootstrap on both datasets. SIFT P_HNSW significantly favors
  K-Means at M=4,8,16,32 and is inconclusive at M=2. GloVe significantly favors K-Means at every M.
- Required three-seed M=32 variance on both datasets. Method ranges preserve the primary ordering;
  seed variance does not reverse either verdict.
- Required P2-A ablation. Refinement lowers TWCut relative to Orion-NoRefinement, but does not
  improve both topology and downstream fan-out beyond K-Means; the ablation answer is NO.
- Cross-dataset Pearson/Spearman diagnostics. The pooled signs contradict the proposed mechanism:
  TWCut versus local recall is positive, TWCut versus fixed-recall work is near zero/negative,
  local recall versus P_HNSW is positive, and Delta_P versus TWCut is weakly negative.
- Load-balance control with all 496 logical-shard rows and per-shard query participation. Full Orion
  M=32 max/mean is 4.930 on SIFT and 5.887 on GloVe and remains explicit in the conclusion.
- Construction overheads for all 40 configurations. Random/K-Means timing uses documented
  same-parameter replay because accepted primary artifacts lacked timings; tiny high-M K-Means
  floating-point assignment drift is recorded, while all E1-E4 metrics retain primary artifacts.
- All required final deliverables: eight named CSVs, eight named PDFs, a 40-row
  `runs/summary.csv`, a 40-row causal configuration table, separate final C2/C3 verdicts, and the
  append-only result log.

Primary final evidence:

- Strict audit: `logs/stage12-final-audit-v1.json`.
- Final raw manifest:
  `/proj/intelisys-PG0/exp/orion-distributed/c23-20260821-v2/analysis/final-v1/manifest.json`.
- Required tables: `tables/`.
- Required figures: `figures/c23_*.pdf`.
- Run registry: `runs/summary.csv` and `runs/manifest.jsonl`.

In progress:

- None.

Not yet complete:

- None within the C2-C3 protocol. Any balance-constrained partitioner or online-router evaluation
  is a new experiment and must not overwrite these contradictory results.
