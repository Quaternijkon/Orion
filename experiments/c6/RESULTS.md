# C6 Experimental Results

This file is append-only. Stage results and corrections are added with UTC timestamps; prior interpretations are never rewritten.

## 2026-08-31T00:00:00Z — Stage 0 started

Timestamp: 2026-08-31T00:00:00Z  
Git commit: pending Stage 0 manifest  
Experiment: Stage 0 instrumentation and policy implementation  
Dataset: not applicable  
Logical shards: not applicable  
Policy: P0-P6 implementation  
Run IDs: none  
Recall: not measured  
Fan-out: not measured  
Aggregate distance computations: not measured  
Max per-shard distance computations: not measured  
Latency if physical: not measured  
Observed result: implementation in progress  
C6-a status: INSUFFICIENT  
C6-b status: INSUFFICIENT  
C6-c status: INSUFFICIENT  
C6-d status: INSUFFICIENT  
Anomalies: no live experiment evidence yet  
Required follow-up: complete Stage 0 tests, freeze artifacts, then run the SIFT1M M=4 smoke matrix.

## 2026-08-31T21:00:15Z — Stage 1 SIFT1M M=4 smoke complete

Timestamp: 2026-08-31T21:00:15Z  
Git commit: 086d02ecbdab147ee11d2572379696f13b3c5a41  
Experiment: Stage 1 P0/P1/P2/P3 smoke and policy-switch identity audit  
Dataset: SIFT1M  
Logical shards: 4 physical machines  
Policy: P0/P1/P2/P3  
Run IDs: c6-sift-m4-stage1-20260831a; stage1-smoke20  
Recall: P0=0.9600; P1=0.9950; P2=0.9650; P3=1.0000 on 20 non-reportable smoke queries  
Fan-out: P0/P2=1.0; P1/P3=1.45  
Aggregate distance computations: P0=3246.15; P1=4559.85; P2=9803.35; P3=11633.10  
Max per-shard distance computations: positive and aligned for every query/policy  
Latency if physical: smoke latency not used as C6 evidence  
Observed result: identical ranked routing candidates across policies; P0/P2 fixed sets and P1/P3 adaptive sets match; all hardware counters are positive; all policies reuse collection fingerprint `95f8c0dd67d63452cb866cca7afa27b9ecfff2168a4d310053b16bbc00e391d2` without rebuild  
C6-a status: INSUFFICIENT  
C6-b status: INSUFFICIENT  
C6-c status: INSUFFICIENT  
C6-d status: INSUFFICIENT  
Anomalies: the restored controller required a run-scoped systemd unit because ordinary local nohup children are reclaimed by the execution sandbox  
Required follow-up: tune on exactly 1,000 queries and run all 9,000 held-out SIFT1M queries.

## 2026-08-31T21:34:04Z — SIFT1M M=4 E1-E4 algorithmic results

Timestamp: 2026-08-31T21:34:04Z  
Git commit: 086d02ecbdab147ee11d2572379696f13b3c5a41  
Experiment: E1 query difficulty, isolated E2/E3, and full E4 ablation  
Dataset: SIFT1M  
Logical shards: 4 physical machines  
Policy: P0-P6  
Run IDs: sift1m-m4-tuning; measurement-9000; sift1m-m4-measurement; sift1m-m4-isolated  
Recall: isolated E2 P0=0.94536, P1=0.94863; isolated E3 P0=0.94536, P2=0.98818, P5=0.94620; full E4 P0=0.93120, P1=0.94863, P2=0.93926, P3=0.99134, P6=0.94569  
Fan-out: E2 P0=2.0, P1=1.53589, P4=1.13878; full P0/P2=1.0, P1/P3=1.53589, P6=1.16344  
Aggregate distance computations: E2 P0=1704.52, P1=1381.19; E3 P0=1704.52, P2=2705.60, P5=1195.00; full P0=1587.76, P1=1381.19, P2=2199.45, P3=2534.56, P6=877.50  
Max per-shard distance computations: E2 p95 P0=1277.0, P1=1269.05; E3 p95 P0=1277.0, P2=3236.0, P5=2057.05  
Latency if physical: physical saturation repetitions pending  
Observed result: E2 P1 saves 0.46411 mean shards and 20.22% paired work (95% CI 19.54%-20.89%) at slightly higher recall. E3 P2 increases paired work by 58.08%. `|EP|` vs ground-truth contribution Spearman=0.95437 and vs oracle local work=0.45924. Static P0 waste classes are 75.42% over-searched, 15.97% under-searched, and 8.61% near-sufficient. Full normalized work is P1/P0=0.86990, P2/P0=1.38525, P3/P0=1.59631, P6/P0=0.55266.  
C6-a status: INSUFFICIENT  
C6-b status: SUPPORTED for SIFT1M M=4 only; cross-M and second-dataset consistency pending  
C6-c status: CONTRADICTED for the current linear entry-point-count allocation on SIFT1M M=4  
C6-d status: CONTRADICTED on SIFT1M M=4; P3 is worse than P0/P1/P2 in work and remains 2.888x P6  
Anomalies: entry-point count is informative, but the configured positive linear mapping allocates far more EF than the measured oracle requires; do not equate signal calibration with a successful allocation policy  
Required follow-up: run M=8/16/32 before making C6-a/C6-b scope-wide claims; perform M=4 physical E5 repetitions; then repeat the internally consistent matrix on GloVe.

## 2026-08-31T23:36:36Z — SIFT1M E1 complete for M=4,8,16,32

Timestamp: 2026-08-31T23:36:36Z  
Git commit: 086d02ecbdab147ee11d2572379696f13b3c5a41  
Experiment: E1 query-difficulty characterization  
Dataset: SIFT1M  
Logical shards: 4, 8, 16, 32; M>4 labeled logical-shard simulation  
Policy: P4 oracle prefix and P6 full oracle  
Run IDs: sift1m-m4-e1; sift1m-m8-e1; sift1m-m16-e1; sift1m-m32-e1  
Recall: P6 target-reached fraction M4=1.0, M8=1.0, M16=0.999889, M32=1.0  
Fan-out: oracle prefix mean M4=1.13878, M8=1.34178, M16=1.64833, M32=2.07467; p99=2,3,4,7  
Aggregate distance computations: P6 mean M4=877.50, M8=885.49, M16=920.63, M32=1003.17  
Max per-shard distance computations: P6 p95 M4=2042.0, M8=1863.05, M16=1619.05, M32=1425.05  
Latency if physical: not used for E1 algorithmic conclusions  
Observed result: query shard difficulty has a non-trivial distribution at every M. The normalized oracle-prefix mean falls from 0.28469 at M4 to 0.06483 at M32, while the absolute requirement grows and remains heterogeneous. Oracle maximum local EF has median 16,16,8,8 and p95=64 for all four M, with individual queries requiring up to 512 at M4/M8/M16 and 256 at M32.  
C6-a status: SUPPORTED on SIFT1M; second-dataset confirmation pending  
C6-b status: SUPPORTED for SIFT1M M=4 only; M=8/16/32 pending  
C6-c status: CONTRADICTED for the current linear allocation on SIFT1M M=4; other M pending  
C6-d status: CONTRADICTED on SIFT1M M=4; other M pending  
Anomalies: one M16 query cannot reach 0.90 within the finite EF grid even after all shards; retained as target-reached fraction 0.999889  
Required follow-up: run isolated E2 for M=8,16,32 using common per-M EF and exact fixed-P tuning.

## 2026-09-01T01:04:21Z — SIFT1M E2-E5 and Stage 7 inspection complete

Timestamp: 2026-09-01T01:04:21Z  
Git commit: 086d02ecbdab147ee11d2572379696f13b3c5a41  
Experiment: SIFT1M E2/E3/E4 across M=4,8,16,32 and E5 physical M=4  
Dataset: SIFT1M  
Logical shards: 4,8,16,32  
Policy: P0-P6  
Run IDs: sift1m-m{4,8,16,32}-isolated; sift1m-m{4,8,16,32}-measurement; sift1m-m4-e5-persistent  
Recall: every deployable held-out configuration reaches mean Recall@10 >=0.90  
Fan-out: isolated E2 P0 vs P1 is 2.0 vs 1.536 at M4, 2.0 vs 2.434 at M8, 3.0 vs 3.884 at M16, and 6.0 vs 5.995 at M32  
Aggregate distance computations: E2 paired relative reduction is +20.22%, -21.16%, -27.34%, +2.19% for M4,8,16,32. E3 P2 relative reduction is -58.08%, -48.69%, -9.45%, -25.32%. Full P3/P0 work ratios are 1.596,1.602,1.218,1.479; P6/P0 ratios are 0.553,0.519,0.370,0.429.  
Max per-shard distance computations: P2 raises p95 max-shard work at every M; see `c6_local_budget_summary.csv`  
Latency if physical: final stable P0/P1/P3 runs use 12 worker cores per shard and 8 isolated client cores. P1 vs P0 has p99 129.09 vs 192.32 ms and QPS 1099.68 vs 1126.21. P2 remains invalid due 11.7% QPS CV.  
Observed result: C6-a is supported. C6-b is scale-dependent and contradicted as a general claim: supported at M4, worse at M8/M16, marginal at M32. C6-c is contradicted at all M despite a calibrated signal; Spearman `|EP|` vs ground-truth contribution falls from 0.954 at M4 to 0.810 at M32 and vs oracle work from 0.459 to 0.146. C6-d is contradicted at all M and the P6 headroom remains large. Physical P1 validates lower tail latency but not higher throughput; P2 physical evidence is insufficient.  
C6-a status: SUPPORTED on SIFT1M  
C6-b status: CONTRADICTED as a general cross-scale claim on SIFT1M  
C6-c status: CONTRADICTED for the current linear `|EP|` allocation  
C6-d status: CONTRADICTED for the current full policy  
Anomalies: one M16 query misses target under the finite oracle grid; E5 P2 remains unstable after five repetitions; all failed/unstable attempts are retained  
Required follow-up: repeat the internally consistent algorithmic matrix on glove-200-angular; keep SIFT physical C6-c marked insufficient.

## 2026-09-01T10:20:00Z — GloVe E1-E5 complete for M=4,8,16,32

Timestamp: 2026-09-01T10:20:00Z  
Git commit: 086d02ecbdab147ee11d2572379696f13b3c5a41  
Experiment: GloVe E1 query difficulty, isolated E2/E3, full E4, and physical E5  
Dataset: glove-200-angular  
Logical shards: 4,8,16,32; M>4 labeled logical-shard simulation  
Policy: P0-P6; physical P0-P3 at M=4  
Run IDs: glove-m{4,8,16,32}-measurement; glove-m{4,8,16,32}-isolated; glove-m{4,8,16,32}-e1; glove-m4-e5-persistent; valid fallback completions glove-m8-measurement-fallback-p2-rank2 and glove-m32-measurement-fallback-p3-rank1  
Recall: isolated E2 P0/P1 is 0.90998/0.91598, 0.91164/0.93119, 0.90214/0.89909, and 0.90326/0.90164 for M=4/8/16/32. Isolated E3 P0/P2 is 0.90998/0.90360, 0.91164/0.90321, 0.90214/0.90826, and 0.90326/0.90688. Final full P3 recall is 0.90113, 0.90041, 0.90579, and 0.90609.  
Fan-out: isolated E2 P0 vs P1 is 3.0 vs 2.913, 4.0 vs 4.522, 10.0 vs 6.218, and 14.0 vs 8.065. The M16 P1 result is invalid because recall is 0.89909. Oracle-prefix mean fan-out is 1.963, 2.670, 3.422, and 5.509.  
Aggregate distance computations: valid isolated E2 paired reductions are +4.73% at M4, -12.35% at M8, and +43.22% at M32; the apparent +39.23% M16 result is excluded by the recall gate. Isolated E3 reductions are +33.04%, +34.91%, +42.25%, and +54.63%. Final P3/P0 work ratios are 0.582, 0.502, 0.518, and 0.458.  
Max per-shard distance computations: isolated P2 increases p95 max-shard work at every M despite lowering aggregate work; see `c6_local_budget_summary.csv`.  
Latency if physical: all GloVe M4 physical policies are stable and recall-valid. P0/P1/P2/P3 QPS is 1149.34/987.36/1139.10/1023.25 and p99 latency is 103.80/111.26/104.86/127.27 ms.  
Observed result: GloVe supports substantial aggregate-work benefits from adaptive local budgets and the full policy, but not a tail-latency benefit. Adaptive shard selection is beneficial at M4/M32, adverse at M8, and recall-invalid at M16. Entry-point count remains related to ground-truth contribution, while its correlation with oracle local work weakens to -0.073 at M32.  
C6-a status: SUPPORTED on GloVe; oracle shard and local-budget requirements are broad, although the finite P6 grid reaches the per-query target for only 88.06%-96.94% of queries depending on M  
C6-b status: CONTRADICTED as a general GloVe cross-scale claim  
C6-c status: SUPPORTED for aggregate work on GloVe, but not for max-shard work or physical p99 latency  
C6-d status: SUPPORTED versus P0 and component ablations on GloVe aggregate work, but CONTRADICTED for near-oracle and tail-latency wording  
Anomalies: primary M8 P2 recall=0.89877 and rank-1 fallback recall=0.89994 are retained as invalid; rank-2 reaches 0.90134. Primary M32 P3 recall=0.89828 is retained as invalid; rank-1 reaches 0.90609. GloVe P6 has finite-grid per-query target shortfalls at every M.  
Required follow-up: combine with SIFT, run E6 using each dataset's actual navigation K, and generate the final qualified verdict.

## 2026-09-01T10:28:40Z — Stages 9-10 and final C6 verdict

Timestamp: 2026-09-01T10:28:40Z  
Git commit: 086d02ecbdab147ee11d2572379696f13b3c5a41  
Experiment: E6 sensitivity, final two-dataset aggregation, figures, tables, and claim audit  
Dataset: SIFT1M and glove-200-angular  
Logical shards: 4,8,16,32; M>4 labeled logical-shard simulation  
Policy: P0-P6 and P3 sensitivity  
Run IDs: all final measurement/isolated/E1/E5 runs above; sensitivity runs sift-m{4,8,16,32} and glove-m{4,8,16,32}  
Recall: every deployable configuration used in the final P0-P3 ablation reaches Recall@10 >=0.90 after the recorded tuning-preordered fallbacks. E6 contains 80 held-out rows; 11 GloVe non-default sensitivity rows fall below 0.90, while all SIFT sensitivity rows remain valid.  
Fan-out: oracle-prefix means range from 1.139-2.075 on SIFT and 1.963-5.509 on GloVe as M grows from 4 to 32. Static P0 under-searches 11.78%-22.98% and over-searches 75.42%-88.17% of queries across final runs.  
Aggregate distance computations: isolated E2 is beneficial in 4 of 7 recall-valid dataset-by-M rows and adverse in 3; isolated E3 reduces work by 33.04%-54.63% at every GloVe M but increases work by 9.45%-58.08% at every SIFT M. Full P3 reduces P0 work by 41.8%-54.2% on GloVe but increases it by 21.8%-60.2% on SIFT.  
Max per-shard distance computations: the current positive linear EP allocation generally raises p95 max-shard work even when it lowers aggregate work on GloVe.  
Latency if physical: GloVe P3 lowers aggregate work but has lower QPS and higher p99 than P0. SIFT P1 lowers p99 but not QPS; SIFT P2 remains physically invalid after five repetitions because QPS CV is 11.7%.  
Observed result: query difficulty heterogeneity is robust, but the claimed adaptive improvements are dataset- and scale-dependent. The current linear `efSearch_i = alpha|EP_i| + beta` policy succeeds on GloVe aggregate work and fails on SIFT aggregate work. Full P3 is not near the oracle: P3/P6 work is 1.83-3.00 on GloVe and 2.89-3.45 on SIFT. Sensitivity uses the actual K values: GloVe 24/48/96 and SIFT 50/100/200.  
C6-a status: SUPPORTED  
C6-b status: CONTRADICTED as a general claim; dataset- and scale-dependent  
C6-c status: CONTRADICTED as a general claim; supported only for GloVe aggregate work and contradicted on SIFT and for tail-work wording  
C6-d status: CONTRADICTED as a general and near-oracle claim; supported only for GloVe aggregate work versus deployable baselines  
Anomalies: all recall-invalid, finite-oracle-grid, and physically unstable results are retained. No measurement-set retuning was performed; fallback order was frozen from tuning evidence.  
Required follow-up: none for this protocol; any future policy redesign must be evaluated as a new experiment rather than substituted into C6.

C6-a Query difficulty is heterogeneous:
  SUPPORTED

Evidence:
  Oracle shard-requirement distribution: SIFT mean 1.139-2.075 with p99 2-7; GloVe mean 1.963-5.509 with p99 4-32.
  Oracle local-budget distribution: median per-query maximum EF is 8-32 and p95 is 64-512 depending on dataset/M.
  Static over-search fraction: 75.42%-88.17%.
  Static under-search fraction: 11.78%-22.98%.

C6-b Adaptive shard selection reduces redundant work:
  CONTRADICTED

Evidence:
  Recall: one isolated GloVe M16 P1 row is invalid at 0.89909 and excluded from work comparison.
  Mean shard reduction: positive at SIFT M4/M32 and GloVe M4/M16/M32, but M16 GloVe is recall-invalid; fan-out increases at SIFT M8/M16 and GloVe M8.
  Aggregate work reduction: +20.22%, -21.16%, -27.34%, +2.19% on SIFT; +4.73%, -12.35%, invalid, +43.22% on GloVe.
  Oracle-prefix gap: adaptive fan-out remains materially above P4 in most rows.
  Dataset consistency: absent; the result changes sign across M and dataset.

C6-c Adaptive local search intensity improves allocation:
  CONTRADICTED

Evidence:
  Recall: every final isolated P2 row reaches >=0.90.
  Aggregate work reduction: +33.04% to +54.63% on GloVe, but -9.45% to -58.08% on SIFT.
  Max-shard work reduction: generally negative; P2 raises p95 max-shard work across the tested scales.
  Physical p99 impact: GloVe P2 is 104.86 ms vs P0 103.80 ms; SIFT P2 is unstable and excluded.
  Entry-point-count calibration: ground-truth contribution remains measurable (Spearman 0.629-0.954), but oracle-work correlation ranges from -0.073 to 0.459.

C6-d Full adaptive policy approaches the best recall-cost frontier:
  CONTRADICTED

Evidence:
  Full vs static: P3/P0=0.458-0.582 on GloVe and 1.218-1.602 on SIFT.
  Full vs component ablations: P3 is the best deployable aggregate-work policy on GloVe, but not on SIFT.
  Full vs oracle: P3/P6=1.83-3.00 on GloVe and 2.89-3.45 on SIFT, leaving large headroom.
  Physical validation: GloVe P3 has lower QPS and higher p99 than P0; SIFT P3 has lower p99 but also lower QPS.

Paper-ready qualified claim:

> Query difficulty varies substantially in distributed graph search, but the effectiveness of Orion's current online adaptation policy is dataset- and scale-dependent. Adaptive per-shard budgets and the full policy reduce aggregate graph work on GloVe, while the same linear entry-point-count allocation increases work on SIFT and often increases maximum-shard work. Adaptive shard selection likewise changes sign across shard counts. The evaluated full policy therefore does not support a general lower-work, lower-tail-latency, or near-oracle claim.

## 2026-09-01T10:37:39Z — Completion audit passed

Timestamp: 2026-09-01T10:37:39Z  
Git commit: 086d02ecbdab147ee11d2572379696f13b3c5a41  
Experiment: requirement-by-requirement completion and verification audit  
Dataset: SIFT1M and glove-200-angular  
Logical shards: 4,8,16,32  
Policy: P0-P6 and E6 sensitivity  
Run IDs: c6_completion_audit; all final runs listed above  
Recall: audit confirms every deployable configuration used in final comparisons reaches Recall@10 >=0.90 and every excluded recall failure remains present  
Fan-out: all E1/E2 per-query and summary artifacts are present for both datasets and all M  
Aggregate distance computations: all E2/E3/E4/oracle and E6 tables are present with expected row counts  
Max per-shard distance computations: per-query and per-shard artifacts and required summary columns are present  
Latency if physical: eight dataset-by-policy M4 rows are present; GloVe P0-P3 and SIFT P0/P1/P3 are valid, SIFT P2 is retained invalid  
Observed result: `c6_completion_audit.json` passes 47/47 checks; 32 focused Python tests and 2 Rust route-trace tests pass; all Python scripts compile; `git diff --check` is clean; all worker affinities are restored to 0-19  
C6-a status: SUPPORTED  
C6-b status: CONTRADICTED as a general claim  
C6-c status: CONTRADICTED as a general claim  
C6-d status: CONTRADICTED as a general and near-oracle claim  
Anomalies: none unresolved; contradictory, invalid, finite-grid, and unstable outcomes are retained as final evidence  
Required follow-up: none
