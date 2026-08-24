# C1 Execution Status

Updated: 2026-08-21T06:34:00Z

Current stage: Stage 12 complete. The deterministic E2-E4 rerun, finalization, lifecycle cleanup,
and strict protocol audit have completed; `c1_completion_audit.py` reports 22/22 PASS.

Completed:

- Completed the authoritative deterministic E3 matrix: 24/24 SIFT1M and GloVe Random/K-Means
  configurations pass the held-out Recall@10 gate with graph seed 20260821, one indexing thread,
  one optimizer thread, fixed affinity, and verified per-group collection cleanup. The initial
  GloVe K-Means M=2 efSearch=128 miss remains preserved as `INVALID_RECALL`; the canonical point
  was retuned to efSearch=192 and passed.
- Completed deterministic E2 and E4 for both datasets, 36 normalized Section 32 metadata rows,
  final four-peer/controller-storage cleanup proof, the completed Section 31 proof, regenerated
  canonical tables/figures, and the hash-linked Stage 12 final record. All M>4 throughput
  projections remain withheld.
- Final scientific status: C1-a `SUPPORTED`; generalized C1-b `CONTRADICTED`; C1-c
  `CONTRADICTED`; C1-d physical attribution `INSUFFICIENT`; complete C1 `INSUFFICIENT`. C1-c is
  contradicted because 6 of 40 distance/node comparisons are at or below the ideal 1/M decrease,
  including GloVe Random at M=2 and M=4.
- Completed E5 with all 24 neighborhood rows valid. The diagnostic is
  `CONTRADICTED_QUALITATIVE_TREND_REVERSAL`: SIFT K-Means at the 0.89 target increases fan-out
  from 1 to 2 and retains a slow local-work ratio of 0.4749 from M=4 to M=16, but aggregate
  distance work decreases from 2020.316 to 1918.708. The raw result is retained and reported.
- Completed the strict requirement audit at 22/22 PASS. Historical Stage 10/11 records and
  artifacts remain preserved; Stage 12 supersedes their C1-c and E5 conclusions.
- Hardened the final 22-gate audit against the underlying evidence rather than summary presence:
  it now replays all 36 tuning selections and 1,716 manifest candidates, verifies physical and
  logical placement/affinity, validates all 30 E1 repetitions and raw-query schemas, recomputes
  E2/E4 from 432,000 per-query rows, checks all 252 E3 shard-resource records, revalidates the
  deterministic proof hashes, and enforces final table/status contracts. The full focused suite
  passes 42 tests and the strengthened audit remains 22/22 PASS.
- Parsed the complete 1,151-line source protocol into an execution checklist.
- Verified the four-node cluster is reachable at 10.10.1.1 through 10.10.1.4.
- Verified `glove-200-angular.hdf5` is present with 1,183,514 train rows and 10,000 queries.
- Checksum- and metadata-audited all 12 deterministic GloVe Random/K-Means partition artifacts
  for M=1,2,4,8,16,32. Every artifact covers exactly 1,183,514 assignments with no empty shard,
  the expected seed/dataset hash/dimension, matching recorded shard counts, and the expected
  centroid shape. GloVe K-Means M=1 converged in two iterations; M=2 through M=32 all reached the
  12-iteration cap without convergence and retain substantial shard imbalance.
- Completed GloVe E3 M=1 for Random and K-Means. Both selected P=1/efSearch=384, achieved
  measurement Recall@10=0.91206, and recorded identical algorithmic work of 12,837.62 distance
  computations and 397.82 graph nodes per query. All 18,000 raw searched-shard rows, tuning and
  summary hashes, affinity, counters, resource sizes, and cross-method identity passed audit; the
  temporary collection was deleted with four-peer 404 verification.
- Completed GloVe E3 M=2 for Random and K-Means. The initial Random efSearch=192 measurement
  achieved Recall@10=0.89816, was preserved and marked `INVALID_RECALL`, and triggered the
  protocol-required retune. The valid Random fallback selected efSearch=256 and achieved 0.91547;
  K-Means selected P=2/efSearch=256 and achieved 0.90440. Normalized local distance work is
  0.7066 and 0.6876 versus the M=1 baseline, above the ideal 0.5. Both valid 18,000-row evidence
  sets passed audit and both collections were deleted with four-peer 404 verification.
- Completed GloVe E3 M=4 for Random and K-Means. Random selected broadcast P=4/efSearch=128 and
  achieved held-out Recall@10=0.90142; K-Means exact ranked-prefix tuning selected
  P=3/efSearch=256 and achieved 0.90332. Normalized local distance work is 0.3946 and 0.7208
  versus the M=1 baseline, both above the ideal 0.25. All 36,000 Random and 27,000 K-Means
  searched-shard rows, centroid-ranked routes, summary statistics, counters, resources, hashes,
  and affinity passed audit. Both collections were deleted with four-peer 404 verification.
- Completed GloVe E3 M=8 for Random and K-Means. Random selected broadcast P=8/efSearch=96 and
  achieved held-out Recall@10=0.91206; K-Means exact ranked-prefix tuning selected
  P=4/efSearch=256 and achieved 0.90302. Normalized local distance work is 0.3102 and 0.7121
  versus the M=1 baseline, far above the ideal 0.125. All 72,000 Random and 36,000 K-Means rows,
  exact centroid-ranked prefixes, summary reproduction, counters, resources, hashes, and affinity
  passed audit. Both collections were deleted with four-peer 404 verification.
- Completed GloVe E3 M=16 for Random and K-Means. Random selected broadcast P=16/efSearch=64 and
  achieved held-out Recall@10=0.91609; K-Means selected P=9/efSearch=128 and achieved 0.90157.
  Normalized local distance work is 0.2212 and 0.3987 versus the M=1 baseline, far above the ideal
  0.0625. All 144,000 Random and 81,000 K-Means rows, exact per-query centroid rankings, summary
  reproduction, counters, resources, hashes, and affinity passed audit. Both collections were
  deleted with four-peer 404 verification.
- Completed GloVe E3 Random M=32. Broadcast P=32/efSearch=48 achieved tuning Recall@10=0.9244
  and held-out Recall@10=0.92309. Mean local work is 2,197.35 distance computations and 57.46
  graph nodes per searched shard; normalized distance work is 0.1712 versus M=1, 5.48 times the
  ideal 0.03125 curve. All 288,000 searched-shard rows, one-based route ranks, summary statistics,
  counters, resource sizes, hashes, placement, and affinity passed independent audit. The
  collection was deleted with four-peer 404 verification and absent controller storage.
- Completed GloVe E3 K-Means M=32, closing the 12-configuration GloVe E3 matrix. Exact
  ranked-prefix reuse evaluated 352 candidates with 352,000 shard searches and selected
  P=10/efSearch=128 at tuning Recall@10=0.9005; held-out Recall@10=0.90241 passed. Mean local work
  is 4,664.53 distance computations and 137.05 graph nodes per searched shard; normalized
  distance work is 0.3633 versus M=1, 11.63 times the ideal 0.03125 curve. All 90,000 rows and all
  9,000 exact centroid-ranked routes passed independent audit. The collection was deleted with
  four-peer 404 verification and absent controller storage.
- Parameterized E2 input stems without changing the SIFT defaults, added two focused tests, and
  passed all 26 C1 tests. Preserved the six SIFT E2 canonical CSV/PDF artifacts under explicit
  `stage3-sift1m-*` names with byte-identical hashes before canonical names were reused.
- Completed GloVe E2 for all 12 Random/K-Means configurations. Exact ground-truth-to-partition
  mapping and validated E3 events produced 108,000 per-query rows. Random actual fan-out is
  1,2,4,8,16,32; K-Means is 1,2,3,4,9,10. At M=32 their oracle means are 7.6996 and 2.4181;
  K-Means has median 2, p95 5, p99 7, and maximum 9. All source hashes, distributions, recall,
  sensitivity, PDF format/visual checks, dataset-specific preservation copies, and the single
  manifest registration passed audit.
- Added a held-out E3 recall gate that sums routed-shard recall contributions, records
  `measurement_recall`, emits `INVALID_RECALL`, and exits non-zero below target after preserving
  raw outputs. The focused suite now passes 24 tests.
- Added distributed request accounting for HNSW graph-node expansion events.
- Verified graph-node counters accumulate across forked request counters.
- Verified the full `qdrant` crate compiles with the new REST/gRPC counter field.
- Verified both official HDF5 inputs and their disjoint 1,000-query tuning / 9,000-query
  measurement splits. SIFT1M SHA-256 is
  `dd6f0a6ed6b7ebb8934680f861a33ed01ff33991eaee4fd60914d854a0ca5984`; GloVe SHA-256 is
  `4839085e5a8bb293434a1a66e1aa0193afc3f07c6797a85f1dbd91656172da20`.
- Passed all ten C1 protocol-helper unit tests.
- Deployed run `c1-20260820-v1` on all four physical nodes with image ID
  `sha256:976ab5fd1c669fa971b5679fd5cdd6cf5d7882c976544771ad7e33ec35088a86`, hardware
  reporting enabled, Qdrant pinned to cores 0-19, and benchmark cores 20-31 reserved.
- Passed the strengthened live counter smoke test: approximate HNSW search reported 1,228
  distance computations, 70 graph-node expansions, 283 us CPU time, and 279 us wall time; exact
  search reported 2,000 distance computations, zero graph-node expansions, 196 us CPU time, and
  193 us wall time.
- Implemented and passed the end-to-end four-machine Random M=4 integration test with Recall@10
  = 1.0 and positive distance, graph-node, and CPU counters on all 80 shard searches.
- Generated and checksum-audited deterministic SIFT1M Random and K-Means partition artifacts for
  M=1,2,4. K-Means produced no empty clusters.
- Refreshed `runs/deployment-manifest.json` from the authoritative shared manifest after the image
  transition.
- Built the common full SIFT1M M=1 collection with 1,000,000 indexed vectors, HNSW M=32, and
  efConstruction=200 on the intended physical worker.
- Added a benchmark-affinity gate that rejects runs unless the client is pinned exactly to cores
  20-31; the negative and positive checks passed, and all ten C1 tests still pass.
- Completed pinned 1,000-query tuning and 9,000-query measurement for Random and K-Means M=1.
  Both selected efSearch=24 and achieved measurement Recall@10=0.91428 with identical algorithmic
  work (864.57 distance computations and 35.44 graph nodes per query).
- Completed pinned SIFT1M Random M=2 tuning and measurement. It selected efSearch=24, achieved
  Recall@10=0.94469, broadcast to both physical shards, and recorded 1,667.56 distance
  computations plus 69.18 graph nodes per query.
- Completed pinned SIFT1M K-Means M=2 tuning and measurement. It selected P=1 and efSearch=24,
  achieved Recall@10=0.91387, and recorded 860.53 distance computations plus 34.92 graph nodes
  per query; mean oracle fan-out was 1.0011.
- Completed pinned SIFT1M Random M=4 tuning and measurement. It selected efSearch=16, achieved
  Recall@10=0.93843, broadcast to all four machines, and recorded 2,448.13 distance computations
  plus 107.33 graph nodes per query. All 36,000 shard responses passed counter checks.
- Completed pinned SIFT1M K-Means M=4 tuning and measurement. It selected P=1 and efSearch=48,
  achieved Recall@10=0.91100, and recorded 1,355.02 distance computations plus 58.51 graph nodes
  per query. All 9,000 routed shard responses passed counter checks, closing the Stage 1 matrix.
- Generated and audited deterministic SIFT1M Random and K-Means partition artifacts for M=8,16,32.
  All six cover 1,000,000 points with no empty shard; all three K-Means artifacts retain a
  12-iteration non-convergence anomaly.
- Added the E3 isolated searched-shard runner and expanded the focused test suite from 10 to 13
  passing tests. A live smoke and both formal M=1 baselines passed affinity, counter, row-count,
  resource-size, and checksum audits.
- Added exact ranked-prefix reuse for K-Means tuning and expanded the focused suite to 14 passing
  tests. A live M=16 comparison matched all algorithmic candidate fields and selection exactly
  while reducing shard searches by 8.5x on the validation slice.
- Completed SIFT1M E3 M=2 for Random and K-Means. Normalized local distance work is 0.9654 and
  0.9969 versus the M=1 baseline, far above the ideal 0.5 curve. Both temporary collections were
  deleted after evidence verification.
- Completed SIFT1M E3 M=4 for Random and K-Means. Normalized local distance work is 0.7050 and
  1.5728 versus ideal 0.25. The K-Means result preserves its non-convergence and severe imbalance
  anomaly; both temporary collections were deleted after evidence verification.
- Completed SIFT1M E3 M=8 logical-shard simulation for Random and K-Means. Random broadcast to all
  eight shards and normalized local distance work is 0.6571; K-Means selected P=2 and normalized
  local distance work is 1.1446, both far above ideal 0.125. Both temporary collections were
  deleted after raw-row, checksum, affinity, and four-peer 404 verification.
- Completed SIFT1M E3 M=16 logical-shard simulation for Random and K-Means. Random broadcast to
  all 16 shards and normalized local distance work is 0.5998; K-Means selected P=3 and normalized
  local distance work is 0.8954, both far above ideal 0.0625. Both temporary collections were
  deleted after full evidence verification.
- Completed the final SIFT1M E3 M=32 logical-shard simulation for Random and K-Means. Random
  broadcast to all 32 shards and normalized local distance work is 0.5424; K-Means selected P=3
  and normalized local distance work is 1.0267, both far above ideal 0.03125. Measurement recall
  is 0.98760 and 0.92004 respectively. The full 352-candidate K-Means prefix-reuse matrix, all
  315,000 primary raw rows, summary reproduction, evidence hashes, and four-peer collection
  deletion checks passed, closing the SIFT1M E3 matrix.
- Completed SIFT1M E2 for all 12 Random/K-Means logical-shard configurations without rebuilding
  collections. Exact ground-truth-to-partition mapping and validated E3 per-search events produced
  108,000 per-query rows. Random actual fan-out is 1,2,4,8,16,32; K-Means is 1,1,1,2,3,3.
  At M=32 their oracle means are 7.6999 and 1.8714, with K-Means p95=4 and maximum=7. All
  distribution, recall, source-hash, sensitivity, PDF-format, and visual audits passed; the
  focused suite now has 17 passing tests.
- Completed SIFT1M E1 physical scale-out using five unique layouts and the shared M=1 baseline.
  Mean QPS at M=1,2,4 is 1809.03, 1065.15, 589.34 for Random and 1809.03, 1786.56, 1736.64 for
  K-Means. Scaling efficiency falls to 0.0814 and 0.2400 respectively at M=4. All 15 repetitions
  meet recall, affinity, bounded-scheduler, CPU, routing, queue, and network gates; all 270,000
  raw measurement rows and evidence hashes passed audit. Temporary M=2/M=4 collections were
  deleted on all four peers, required/canonical figures were generated, the E1 manifest record
  was registered exactly once, and the focused suite now has 20 passing tests.
- Completed SIFT1M E4 across all 12 Random/K-Means logical-shard configurations. The 108,000
  per-query aggregate rows exactly reproduce E3 event sums. Fan-out multiplied by mean local work
  has Pearson/Spearman 1.0 and effectively zero relative error against observed aggregate work,
  supporting the C1-d aggregate-cost decomposition. Random and K-Means aggregate distance work at
  M=32 is 17.3568x and 3.0800x the M=1 value. The M/W physical work-bound anticorrelates with
  measured E1 at M=1,2,4, so physical attribution is insufficient and every M>4 throughput
  projection is withheld. All output hashes and five required/canonical PDFs passed audit, and
  the E4 manifest record was registered exactly once.
- Diagnosed Stage 4 E1 as client-limited: its serialized Python query path consumed approximately
  0.63, 1.07, and 1.94 ms of client CPU/query at M=1, Random M=2, and Random M=4. Replaced it with
  128 persistent query processes, per-process closed-loop batching, concurrent shard requests,
  and `/proc` coordinator/process accounting. The focused suite reached 22 passing tests before
  the corrected formal matrix.
- Completed corrected SIFT1M E1 across the same five physical layouts. Mean QPS at M=1,2,4 is
  17884.05, 11074.41, 7044.34 for Random and 17884.05, 19985.57, 17945.88 for K-Means. All 15
  repetitions meet recall, affinity, coordinator/total-client CPU, queue, worker, and network
  gates; all 270,000 raw rows and hashes passed audit. Temporary M=2/M=4 collections were deleted
  with four-peer 404 verification. Original Stage 4 canonical artifacts were preserved under
  `stage4-original-*`, and the corrected manifest record supersedes Stage 4 exactly once.
- Completed corrected SIFT1M E4 against the Stage 6 physical curve. The aggregate-work identity
  remains exact. K-Means physical projection now has positive Pearson/Spearman 0.1982/0.5, but
  Random remains anticorrelated at -0.8291/-1.0. The global all-method sanity gate therefore
  retains `INSUFFICIENT` physical attribution and withholds every M>4 projection. Original Stage 5
  canonical artifacts were preserved under `stage5-original-*`; corrected output hashes, PDF
  format/visual checks, and the single superseding manifest registration passed.
- Completed corrected GloVe E1 across the shared M=1 and four M=2/M=4 physical layouts. Mean QPS
  at M=1,2,4 is 3857.67, 4802.12, 4197.43 for Random and 3857.67, 4563.22, 4385.88 for K-Means.
  Every valid repetition contains 18,000 rows, meets Recall@10 >= 0.90, has CV below 2%, and
  passes affinity, client/worker CPU, routing, queue, and network gates. The initial Random M=4
  c=32 repetitions were retained as invalid client-limited evidence; the auditable c=16 override
  passed. The final collection was deleted with four-peer 404 and absent-controller-storage
  verification, and the E1 aggregate record is registered exactly once.
- Completed GloVe E4 across all 12 Random/K-Means logical-shard configurations. Its 108,000
  per-query rows reproduce the E3 event sums, and the fan-out-times-local-work model has
  Pearson/Spearman 1.0 with effectively zero error. Both GloVe methods pass their dataset-local
  physical trend check, but the analyzer consumes the corrected SIFT E4 record as an external
  release gate. Therefore every M>4 projection field remains withheld in the final GloVe and
  cross-dataset outputs. The Stage 9 record is registered exactly once.
- Historical Stage 10 E5 used the already-measured M=4/M=16 tuning neighborhoods at Recall targets
  0.89/0.90/0.91. All 24 points retain non-decreasing fan-out and aggregate work from M=4 to
  M=16; local-work ratios are 0.4545 to 0.8474 versus the ideal 0.25, so no qualitative C1 trend
  reversed in that pre-deterministic evidence. The authoritative Stage 12 rerun supersedes this
  conclusion with the SIFT K-Means 0.89 aggregate-work reversal recorded above.
- Completed Stage 10. The five required CSVs now combine both datasets; the compact LaTeX table,
  all six numbered figures, and `c1_combined_motivation.pdf` were generated and visually audited.
  The final matrices contain 12 E1, 24 E2, 24 E3, 24 E4, four model-accuracy, and 24 E5 rows.
  All hashes reproduce, all M>4 throughput projection cells are blank, 32 focused tests pass,
  and the Stage 10 final record is registered exactly once.
- Added `QDRANT_HNSW_GRAPH_BUILD_SEED`, fixed HNSW indexing and optimizer construction to one
  thread, built and deployed `orion-c1:20260821-deterministic-seed` identically on all four
  peers, and verified two independent serialized graph builds have identical graph content.
- Repaired every missing final-figure requirement and added a 22-requirement completion audit.
  The focused C1 suite now passes 49 tests. The audit currently passes 14 requirements; the eight
  remaining failures are the intentionally absent Stage 12 final/proof/metadata/cleanup outputs.
- Added resumable deterministic drivers for the full 24-configuration E3 matrix and downstream
  E2/E4/finalization. The downstream path preserves Stage 11 tables and PDFs before replacing
  canonical names, retains a pre-rerun Section 31 proof, and uniquely registers every new record.

In progress:

- None. The experimental protocol and completion audit are complete.

Next action:

No experimental follow-up is required for protocol completion. Any future Git delivery remains
out of scope here; keep raw run directories local and use explicit allowlist staging.

Known boundaries:

- Existing `results/native-20260818-scale4-32-r080-095-v1` data uses Hash/Simple/Orion layouts
  and is not C1 Random/K-Means evidence.
- The old native experiment containers were stopped without deleting their storage before the
  clean C1 deployment.
- SIFT1M K-Means M=4 reached its configured 12-iteration cap without meeting the centroid-movement
  tolerance; preserve this anomaly in every result that uses that artifact.
- The controller root filesystem has limited free space, so full collections must be built and
  retired in a disk-aware sequence instead of retaining the complete matrix simultaneously.
- The initial unpinned M=1 runs are retained as `INVALID_AFFINITY`; only files and manifest records
  ending in `-pinned` are valid Stage 1 M=1 evidence.
- The earlier Stage 10 `SUPPORTED` verdict and Stage 11 C1-c `SUPPORTED` correction are
  superseded by the Stage 12 deterministic final record. Current status is C1-a `SUPPORTED`,
  generalized C1-b `CONTRADICTED`, C1-c `CONTRADICTED`, C1-d physical attribution
  `INSUFFICIENT`, and complete C1 `INSUFFICIENT`.
- E4 supports the aggregate-cost identity in C1-d, but the idealized M/W projection fails the
  cross-dataset release gate because corrected SIFT Random remains anticorrelated with measured
  E1 at M=1,2,4. Physical attribution beyond the measured range is insufficient and every M>4
  throughput projection remains withheld, even though both GloVe methods pass locally.
- SIFT K-Means actual fan-out is 1,1,1,2,3,3 at M=1,2,4,8,16,32. Per Section 33 this is a
  nearly constant small fan-out and contradicts generalizing C1-b to spatial partitioning.
- GloVe K-Means actual fan-out is 1,2,3,4,9,8. The M=16 to M=32 dip is retained as a
  non-monotonic anomaly; fan-out remains substantial rather than collapsing to a small constant.
- The canonical figure set has been regenerated from Stage 12 data after preserving Stage 11
  copies. It contains the protocol-specific E3 distance/nodes/CPU views, normalized log(N/M)
  reference, cross-dataset E1 efficiency, and M=4/16/32 oracle CDF.
- Stage 4 E1 and Stage 5 E4 remain immutable historical evidence. Their canonical CSV/PDF hashes
  are recoverable from `stage4-original-*` and `stage5-original-*`; Stage 6 and Stage 7 records
  explicitly supersede their physical interpretation without deleting them.
