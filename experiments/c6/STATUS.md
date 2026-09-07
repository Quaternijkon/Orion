# C6 Execution Status

Updated: 2026-09-01T10:37:39Z

Current stage: Complete. The formal completion audit passes 47/47 checks.

Completed configurations:

- Stage 0 instrumentation and policy implementation are complete, including production routing evidence, durable exact trace capture, P0-P6 evaluation, exact oracle DP, fallback ordering, physical validation, sensitivity tooling, aggregation, LaTeX output, and figure generation.
- Frozen SIFT1M and GloVe collections exist for M=4,8,16,32 and reuse one index per dataset-by-M across all deployable policies.
- SIFT1M and glove-200-angular E1-E4 are complete for M=4,8,16,32 on all 9,000 held-out official queries.
- Physical E5 is complete at M=4 for both datasets. All GloVe policies are stable; SIFT P0/P1/P3 are stable and SIFT P2 remains invalid after the required five repetitions.
- E6 is complete with 80 rows. Navigation K uses 50/100/200 for SIFT and 24/48/96 for GloVe; alpha and beta use 0.5x/1x/2x; logical shards use 4/8/16/32.
- All seven required final CSV tables, one LaTeX-ready table, and all 20 required PDF figures are generated.
- Final verdict: C6-a SUPPORTED; C6-b, C6-c, and C6-d CONTRADICTED as general claims, with qualified GloVe-only aggregate-work benefits.

Failed configurations retained as evidence:

- GloVe M8 primary P2 recall=0.89877 and rank-1 fallback recall=0.89994 are invalid; rank-2 reaches 0.90134.
- GloVe M32 primary P3 recall=0.89828 is invalid; rank-1 reaches 0.90609.
- GloVe M16 isolated E2 P1 recall=0.89909 is invalid and excluded from fixed-recall work claims.
- SIFT physical P2 remains invalid after five repetitions because QPS CV=11.7%.
- Finite-grid P6 does not reach the per-query target for every GloVe query; target-reached fractions are 0.88056, 0.92189, 0.95833, and 0.96944 for M=4,8,16,32.

Active anomalies and interpretation constraints:

- C6-b changes sign across dataset and M.
- C6-c reduces aggregate work on GloVe but increases it on SIFT and generally increases p95 max-shard work.
- C6-d reduces work on GloVe but increases it on SIFT, does not improve physical tail latency consistently, and remains 1.83x-3.45x above P6.
- E6 contains 11 recall-invalid GloVe non-default sensitivity rows; SIFT sensitivity rows are recall-valid.
- `/proj` remains full; new raw state is under `/users/dry/orion-c6-state/` and runtime state under `/users/dry/orion-c6-runtime/`.
- All Qdrant process affinities are restored to CPUs 0-19.
- Final verification passes: 32 focused Python tests, 2 Rust route-trace tests, Python compilation, and `git diff --check`.

Next command:

```bash
# No further command is required; see runs/c6_completion_audit.json.
```
