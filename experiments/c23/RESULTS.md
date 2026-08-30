# C2-C3 Results Log

Entries are append-only. Revisions must be added with a new timestamp.

## 2026-08-21T00:00:00Z - Stage 0 current-state audit

Timestamp: 2026-08-21T00:00:00Z  
Git commit: working tree based on `master`; exact commit will be captured by each run  
Experiment: Stage 0 current-state and reuse-boundary audit  
Dataset: SIFT1M and glove-200-angular  
Logical shards: 2, 4, 8, 16, 32 planned  
Partition method: Random, K-Means, Orion, Orion-NoRefinement planned  
Run IDs: none; audit only  
Primary metrics: artifact availability and protocol compatibility  
Observed result: C1 supplies audited official datasets, deterministic Random/K-Means assignments,
four-node topology, and HNSW work counters. Existing native Orion scale artifacts are not accepted
because they are not the required exact disjoint C23 layouts.  
C2 implication: INSUFFICIENT; no global reference traces or local-navigability results yet.  
C3 implication: INSUFFICIENT; no exact/HNSW oracle decomposition yet.  
Status: INSUFFICIENT  
Anomalies: local Docker access is unavailable from the controller user; Stage 0 tooling therefore
uses the same Qdrant Rust HNSW implementation offline and will use the four peers only where the
protocol requires physical placement.  
Required follow-up: finish trace instrumentation, pass a tiny end-to-end smoke, then execute SIFT
Stages 1-8 before beginning the complete GloVe matrix.

## 2026-08-21T20:21:46Z - Stage 0 instrumentation and reference-trace gate

Timestamp: 2026-08-21T20:21:46Z  
Git commit: `fe61fdac6760bc4d2bcd00cc512218d5fbd7e424` plus the recorded working-tree changes  
Experiment: Stage 0 dataset, partition-artifact, P2-A, and Qdrant reference-trace validation  
Dataset: SIFT1M and glove-200-angular; deterministic 2,000-point Euclidean trace smoke  
Logical shards: reusable baseline artifacts audited at 2, 4, 8, 16, 32; smoke is unpartitioned  
Partition method: Random, K-Means artifact audit; Orion/Orion-NoRefinement tooling validation  
Run IDs: `stage0-dataset-audit`, `stage0-c1-baseline-partition-audit`,
`stage0-reference-trace-smoke`  
Primary metrics: 20/20 baseline artifacts valid; 34,747 directed smoke L0 edges; selected
efSearch=10; smoke Recall@10=0.985 across 20 measurement queries; 7/7 independent trace gates pass  
Observed result: the Stage 0 gate is complete. The reference engine uses Qdrant's
`GraphLayersBuilder` and conventional single-entry HNSW search, exports node IDs and L0 edges,
and records ordered visited nodes/edges, distance computations, results, truth, and latency. A Rust
equivalence test confirms traced and ordinary HNSW results are identical.  
C2 implication: INSUFFICIENT; the mechanism can now be measured, but no formal SIFT E1-E3 result
has been collected.  
C3 implication: INSUFFICIENT; the engine and baseline artifacts are ready, but formal exact/HNSW
oracle fan-out and W90 are not measured.  
Status: INSUFFICIENT  
Anomalies: both official datasets contain only 10,000 queries, leaving 9,000 measurement queries
after the disjoint tuning split. Reusable C1 partitions contain one fixed seed; C23 construction
variance validation remains mandatory.  
Required follow-up: execute Stage 1 on full SIFT1M, then generate the exact disjoint P0/P1/P2/P2-A
layouts and run SIFT E1-E4 before starting the GloVe matrix.  
Evidence SHA-256: SIFT audit `04cbef7b249cf7814f0192a8dd992c2bbef075606027dc702e638d04c0f61278`;
GloVe audit `39bc9888394382e9c2dd963c5cc301784a3a9c4efcddc6619c779e927cb2d485`;
baseline audit `37d5920a8591625764094359e643f29c26fdbbd03e77083194f1864b0e7ae11e`;
trace audit `8b13bbf7def8ed178a18b329dd84b97bffa7fe8de56d45645abf3cb508e550b7`;
smoke manifest `00ef84812591695cbd3ebd054da7b0cc53892610f7251624352340445e72f684`.

## 2026-08-21T20:50:12Z - Stage 1 SIFT1M global reference graph and traces

Timestamp: 2026-08-21T20:50:12Z  
Git commit: `fe61fdac6760bc4d2bcd00cc512218d5fbd7e424` plus the recorded working-tree changes  
Experiment: Stage 1 Qdrant-native full-dataset reference HNSW, common-EF tuning, and complete
measurement-query tracing  
Dataset: SIFT1M, official L2 base/query/ground-truth arrays  
Logical shards: not applicable; this is the unpartitioned global reference graph  
Partition method: not applicable  
Run IDs: `stage0-reference-trace-smoke-v2`, `stage1-sift1m-reference-v2`  
Primary metrics: 1,000,000 points; 29,538,410 directed L0 edges; 500,000 HNSW entry slots;
efSearch=24; 9,000 measurement traces; Recall@10=0.9145; construction time 628.424 seconds  
Observed result: the original fixed entry-slot capacity of one was corrected before formal work.
For 128-dimensional float32 SIFT vectors and the C1 10 KB full-scan threshold, Qdrant production
semantics yield 20 vectors per threshold and `max(1, 1,000,000 / 20 * 10) = 500,000` entry slots.
The denser EF tuning grid selected 24 as the smallest candidate reaching the 0.90 target
(tuning Recall@10=0.9036); the disjoint 9,000-query measurement set reached 0.9145. Independent
replay passed file hashes, graph cardinality, production entry-point parameters, smallest-EF
selection, trace schema, distance counts, contiguous edge order, and recall reproduction.  
C2 implication: INSUFFICIENT; the accepted reference graph and traces enable E1-E3, but no
partition comparison has yet been measured.  
C3 implication: INSUFFICIENT; global reference traces are now available for exact/HNSW fan-out
decomposition, but E4 has not yet run.  
Status: INSUFFICIENT  
Anomalies: the official dataset provides 9,000 rather than 10,000 measurement queries after the
required 1,000-query disjoint tuning split. The v1 smoke is preserved as superseded evidence and
was not reused for the formal graph.  
Required follow-up: construct and audit the exact disjoint SIFT1M P0/P1/P2/P2-A partition matrix,
then execute E1-E4 before issuing preliminary C2 or C3 verdicts.  
Evidence SHA-256: corrected smoke audit
`e77df2d6fab8834b5adf8c5083a065267826a5758caad4ba63a4a8ca8f53df6d`; formal reference audit
`d6866de0d8ed937d14d3a4883194eb0d144e564e9386cf6f5037dca67b6bafe3`; formal reference manifest
`2c82d9c5e423b139e7be07fa67f6efa777348f0ecc1c1946bd7d5567b9183b5d`.

## 2026-08-21T21:27:59Z - SIFT1M E1-E2 structural checkpoint

Timestamp: 2026-08-21T21:27:59Z  
Git commit: `fe61fdac6760bc4d2bcd00cc512218d5fbd7e424` plus the recorded working-tree changes  
Experiment: SIFT1M E1 structural topology preservation and E2 search-path preservation  
Dataset: SIFT1M, accepted 29,538,410-edge global L0 graph and 9,000 measurement traces  
Logical shards: 2, 4, 8, 16, 32 on four physical hosts; M>4 remains logical colocated scale  
Partition method: Random, full-data K-Means, Orion, Orion-NoRefinement  
Run IDs: `stage2-sift1m-orion-partition-build`, `stage2-sift1m-e1-v1`,
`stage3-sift1m-e2-v1`  
Primary metrics: E1 edge cut and E2 traversal-weighted edge cut; all 20 configurations accepted  
Observed result: Random edge cut tracks the expected `1-1/M` baseline. Orion versus K-Means E1
edge cut is 0.000454 vs 0.000750 at M=2, then 0.180236 vs 0.083055 at M=4, 0.253566 vs
0.194529 at M=8, 0.341413 vs 0.293279 at M=16, and 0.439371 vs 0.377354 at M=32. The primary
E2 traversal-weighted cut shows the same unfavorable M>=4 ordering: Orion versus K-Means is
0.004692 vs 0.004912 at M=2, 0.177728 vs 0.090964 at M=4, 0.242840 vs 0.204059 at M=8,
0.323649 vs 0.299672 at M=16, and 0.419831 vs 0.383398 at M=32. Orion improves over
Orion-NoRefinement at every M>=4, and on the hottest 1% of M=32 observed edges Orion cut is
0.418117 versus K-Means 0.428487. However, Orion max-shard/mean grows to 3.120 at M=16 and 4.930
at M=32, so lower path-shard counts cannot be interpreted without the balance control.  
C2 implication: NOT YET SUPPORTED on the structural checkpoint. The required primary TWCut trend
does not favor Orion over full-data K-Means for M>=4. E3 must determine whether hot-edge retention
or search-based assignment nevertheless improves conventional local HNSW navigability.  
C3 implication: INSUFFICIENT; no accepted E4 fan-out matrix exists yet.  
Status: INSUFFICIENT  
Anomalies: Orion's severe M=16/M=32 imbalance is a material confound and must remain visible in
fan-out and query-participation results. The reusable Random/K-Means primary matrix still contains
one fixed seed; required construction-variance validation remains pending.  
Required follow-up: finish the sequential E3/E4 local matrix, issue separate preliminary C2 and C3
verdicts, and proceed to GloVe only if the complete SIFT E1-E4 chain is internally consistent.  
Evidence SHA-256: Orion partition build
`6c7c470ff9a60a0b22a3a717c42c21c028558d4375f5803850a3c9f2dd1fba07`; E1 audit
`e5973f4df409f06ad17244570a7a4fc75b489118c1a0c6b5f509e0672233841c`; E2 audit
`55e67dec0a17eff1a2114c61cb574ce8c27a396f260528b4d311f94b07f02d58`.

## 2026-08-21T23:55:31Z - SIFT1M E3-E4 and preliminary C2/C3 verdicts

Timestamp: 2026-08-21T23:55:31Z  
Git commit: `fe61fdac6760bc4d2bcd00cc512218d5fbd7e424` plus the recorded working-tree changes  
Experiment: SIFT1M E3 conventional local-HNSW navigability, E4 exact/HNSW oracle fan-out,
paired bootstrap, and three-seed M=32 construction variance  
Dataset: SIFT1M, official L2 vectors with 1,000 tuning and 9,000 measurement queries  
Logical shards: 2, 4, 8, 16, 32 on four physical hosts; M>4 is logical colocated scale and not
M-machine wall-clock evidence  
Partition method: Random, full-data K-Means, Orion, Orion-NoRefinement  
Run IDs: `stage5-7-sift1m-local-matrix-v2`, `stage5-7-sift1m-local-v1`,
`stage8-sift1m-bootstrap-v1`, `stage8-sift1m-variance-v1`  
Primary metrics: fixed-budget local target recall and distance computations; P_exact; P_HNSW;
Delta_P; W90; paired 95% query-bootstrap confidence intervals; three-seed M=32 TWCut ranges  
Observed result: all 20 local configurations and every independent audit gate pass. Orion uses
fewer common-EF distance computations than K-Means for M>=4, but its paired local-recall gain is
significant only at M=4, inconclusive at M=8 and M=16, and reverses in favor of K-Means at M=32.
Random has the highest TWCut yet the highest local recall and lowest local work, so higher topology
cut does not correspond to worse local navigability here. Orion mean P_HNSW versus K-Means is
1.438 vs 1.433 at M=2, 2.007 vs 1.911 at M=4, 2.719 vs 2.650 at M=8, 3.882 vs 3.583 at M=16,
and 5.629 vs 4.782 at M=32. The Orion-minus-K-Means P_HNSW bootstrap differences are +0.096
[0.067,+0.125], +0.069 [+0.013,+0.123], +0.299 [+0.189,+0.410], and +0.847
[+0.632,+1.068] for M=4,8,16,32. P_exact also favors K-Means for every M>=4. At M=32 Orion
W90 is 3294.4 versus 3098.7, a paired difference of +195.7 [76.0,319.2]. Explicit topology
refinement improves TWCut over Orion-NoRefinement, but downstream P_HNSW is only slightly better
at M=4 and M=8 and is worse at M=16 and M=32. Three-seed M=32 TWCut ranges are K-Means
0.3822-0.3846, Orion 0.4198-0.4521, Orion-NoRefinement 0.4885-0.4958, and Random
0.9686-0.9693, preserving the unfavorable structural ordering.  
C2 implication: preliminary SIFT1M verdict is CONTRADICTED. Full-data K-Means has lower primary
TWCut than Orion for M>=4, Random's extreme cut does not degrade local navigability, and refinement's
structural improvement does not consistently improve downstream local recovery. The required
topology-disruption-to-navigability chain is not supported on SIFT1M.  
C3 implication: preliminary SIFT1M verdict is CONTRADICTED. Orion does not require fewer shards
than K-Means at any M; M=2 is statistically inconclusive and M=4,8,16,32 significantly favor
K-Means. Placement locality (P_exact) and additional ANN fan-out (Delta_P) both contribute to the
high-M deficit, while M=32 W90 rules out a hidden work advantage.  
Status: CONTRADICTS  
Anomalies: Orion remains severely imbalanced at M=16 and M=32 (max/mean 3.120 and 4.930).
Latency is secondary because an unrelated Qdrant process remains active; distance computations and
nodes visited are the comparable work metrics. Random's strong local navigability despite poor
placement locality is a real decomposition result, not a C3 win: its P_exact and P_HNSW remain far
higher than K-Means and Orion. Orion seed variance is material, although the three-seed method
ordering does not overlap.  
Required follow-up: repeat the complete accepted E1-E4, bootstrap, balance, and variance pipeline
on glove-200-angular; then compute cross-dataset correlations and issue final separate C2/C3
verdicts without masking dataset dependence or imbalance.  
Evidence SHA-256: local audit
`91928c7682d1004cd2a7b8d2a7aee4455dd41fb2248d55a74f69f0d19c04fff5`; local manifest
`43f58758eb944d242c7ae4baad27038865b2fd8a3fe275a3360089f3a0f4793c`; bootstrap audit
`0d4f683885b4459a886b202e684f887535d15fb1517309a68ea3c047825a680c`; bootstrap manifest
`c84146803b29950d9c6c1240a0fedc02888cfe5397d36d9c91cc27fc8e2e07fe`; variance audit
`60231c50eaa8d93444c511667626f3671c854841fd231dce71e7c5ce99151248`; variance manifest
`7a2f8c7ca2dad5eebf5200384e382f5ffd8117221c43748e5c361538e5222199`.

## 2026-08-22T08:59:27Z - GloVe E1-E4 and dataset-scoped C2/C3 verdicts

Timestamp: 2026-08-22T08:59:27Z  
Git commit: `fe61fdac6760bc4d2bcd00cc512218d5fbd7e424` plus the recorded working-tree changes  
Experiment: glove-200-angular reference HNSW, E1-E4, paired bootstrap, and three-seed M=32
construction variance  
Dataset: glove-200-angular, official cosine vectors with 1,000 tuning and 9,000 measurement queries  
Logical shards: 2, 4, 8, 16, 32 on four physical hosts; M>4 is logical scale and local-search
microbenchmarks execute one logical shard at a time  
Partition method: Random, full-data K-Means, Orion, Orion-NoRefinement  
Run IDs: `stage9-glove-reference-v2`, `stage9-glove-e1-v1`, `stage9-glove-e2-v1`,
`stage9-glove-local-v1`, `stage9-glove-bootstrap-v1`, `stage9-glove-variance-v1`  
Primary metrics: traversal-weighted cut; common-ef local target recall and distance computations;
P_exact; P_HNSW; Delta_P; W90; paired 95% bootstrap intervals; three-seed M=32 TWCut ranges  
Observed result: the accepted reference contains 1,183,514 vectors, 51,034,393 directed L0 edges,
9,000 complete measurement traces, and selects common efSearch=320 independently; measurement
Recall@10 is 0.9030 and all eight reference gates pass. K-Means versus Orion TWCut is 0.159160
vs 0.151922 at M=2, then 0.240855 vs 0.384265 at M=4, 0.351029 vs 0.422920 at M=8,
0.386494 vs 0.447120 at M=16, and 0.455615 vs 0.508834 at M=32. At the common budget,
Orion local recall is higher at M=2, effectively tied at M=4, and lower at M=8,16,32 while using
fewer distance computations; this is a trade-off, not curve dominance. Orion mean P_HNSW versus
K-Means is 1.703 vs 1.670 at M=2, 2.281 vs 2.140 at M=4, 2.885 vs 2.700 at M=8,
3.481 vs 3.090 at M=16, and 4.932 vs 3.309 at M=32. The Orion-minus-K-Means paired
P_HNSW differences and 95% intervals are +0.033 [+0.020,+0.046], +0.141
[+0.119,+0.163], +0.185 [+0.143,+0.228], +0.391 [+0.309,+0.472], and +1.623
[+1.457,+1.789], so every evaluated M significantly favors K-Means. Orion W90 is also higher
at every M and exceeds K-Means by 10,791 computations/query at M=32. Full Orion max/mean shard
imbalance is 1.591, 1.534, 2.130, 2.611, and 5.887 across M=2,4,8,16,32. Three-seed M=32
TWCut ranges are K-Means 0.4475-0.4556, Orion 0.4900-0.5088,
Orion-NoRefinement 0.5582-0.5726, and Random 0.9686-0.9689; method ranges do not overlap.  
C2 implication: GloVe verdict is CONTRADICTED. K-Means has lower primary TWCut than Orion for
M=4,8,16,32, and the local-search result does not supply the missing monotonic mechanism: lower
Orion work coincides with lower recall at high M rather than navigability dominance.  
C3 implication: GloVe verdict is CONTRADICTED. Orion requires significantly more HNSW oracle
shards than K-Means at every M, P_exact is worse for M=4,8,16,32, and W90 provides no hidden
aggregate-work advantage.  
Status: CONTRADICTS  
Anomalies: M=32 Full Orion assigns 217,725 vectors to its largest shard and only 4,908 to its
smallest. Latency remains secondary; distance computations and node visits are primary. The
M=2 TWCut mean favors Orion but its paired interval is inconclusive and does not reverse the
M>=4 or fan-out results.  
Required follow-up: combine both accepted datasets, compute the prescribed causal correlations,
compile load balance and construction overheads, generate every final named table/PDF, and issue
separate final C2 and C3 verdicts.  
Evidence SHA-256: reference audit
`c1b21baa664cb3b3367b10cb53e79c2fb5c563acd1ee1c070e33e86e859775d3`; E1 audit
`ef59f2252f545384edc81d2345dadf1b9ae75466173cbaa150ff88e861dd083f`; E2 audit
`73031c4c6fd49742842b32ab38d260d4048e27339c006ebfd24c04207f459a1f`; local audit
`a525d441e63a2049fe61bf6a9dad9c432f59848e387b3a69afde825c85cf11be`; bootstrap audit
`43ce013b4f9e26f6e016d9ee7553e506d88b05ec438b82b30a1b2f890d02e656`; variance audit
`acd5b0632055187db2268bb06491983b91e4d950eac0daae43c3f38ca5023e8f`.

## 2026-08-22T09:46:07Z - Final two-dataset C2 and C3 verdicts

Timestamp: 2026-08-22T09:46:07Z  
Git commit: `fe61fdac6760bc4d2bcd00cc512218d5fbd7e424` plus the recorded working-tree changes  
Experiment: Stage 11-12 cross-dataset causal analysis, load-balance control, construction overhead,
required final figures/tables, and strict completion audit  
Dataset: SIFT1M and glove-200-angular  
Logical shards: 2, 4, 8, 16, 32 mapped by logical shard ID modulo four onto four physical hosts;
local-search microbenchmarks execute one logical shard at a time  
Partition method: Random, full-data K-Means, Orion, Orion-NoRefinement  
Run IDs: `stage11-construction-v1`, `stage12-c23-final-two-dataset-synthesis-v1`  
Primary metrics: the complete E1-E4 chain; Pearson and Spearman mechanism-consistency diagnostics;
per-shard balance/query participation; construction time and memory; final deliverable coverage  
Observed result: all 13 source audits and all 10 final synthesis gates pass. The final output has
40-row topology, local-navigability, exact-fan-out, HNSW-fan-out, decomposition, causal-chain, and
construction tables; 496 per-shard balance/participation rows; 18 correlation rows; 40 complete
run-summary rows; and all eight required PDFs. The pooled correlations run opposite to the proposed
mechanism: TWCut versus common-ef local recall has Pearson r=+0.846 and Spearman rho=+0.757
(claim expectation: negative); TWCut versus work at fixed 0.90 local recall is near zero/negative
(r=-0.053, rho=-0.141; expectation: positive); local recall versus P_HNSW is strongly positive
(r=+0.906, rho=+0.877; expectation: negative); and Delta_P versus TWCut is weakly negative
(r=-0.200, rho=-0.288; expectation: positive). These are consistency diagnostics, not formal
causal estimates. Explicit refinement lowers TWCut relative to Orion-NoRefinement but does not
improve both topology and downstream fan-out beyond K-Means, so the required ablation question is
answered NO. Construction overheads cover all 40 configurations. Because accepted Random/K-Means
artifacts did not persist timing, their times are same-parameter replays; four high-M K-Means
replays show only 0.0028%-0.0159% assignment drift from floating-point reduction order, recorded
explicitly, while all final E1-E4 metrics continue to use the accepted primary artifacts.  

C2: Topology-oblivious partitioning disrupts graph-search topology  
Status: CONTRADICTED  
Evidence:  
- Traversal-weighted edge cut: K-Means is lower than Full Orion at M=4,8,16,32 on both datasets;
  the three-seed M=32 ranges do not overlap in Orion's favor.  
- Local target recall: Orion has isolated small gains, but no consistent fixed-budget dominance;
  Random combines the highest TWCut with strong local recall on both datasets.  
- Work at fixed local recall: the prescribed pooled Pearson/Spearman trend is not positive and the
  SIFT dataset trend is strongly negative.  
- Dataset consistency: both SIFT1M and GloVe independently contradict the proposed mechanism.  
- Exceptions: Orion is slightly lower-TWCut at M=2 and refinement improves on NoRefinement, but
  neither exception restores the full topology-to-navigability evidence chain.  

C3: Topology-aware partitioning reduces required shard fan-out  
Status: CONTRADICTED  
Evidence:  
- Exact oracle fan-out: K-Means is better for every M>=4 on both datasets; GloVe M=2 alone favors
  Orion slightly.  
- HNSW oracle fan-out: Orion is never lower than K-Means in the evaluated means; paired bootstrap
  significantly favors K-Means at SIFT M=4,8,16,32 and at every GloVe M.  
- Fan-out reduction at 90% recall: no meaningful portion of the tested range favors Orion.  
- Work to 90% recall: Orion M=32 is higher by 195.7 computations/query on SIFT and by 10,791.2 on
  GloVe; it has no consistent compensating work advantage.  
- Dataset consistency: both datasets contradict C3.  
- Exceptions: SIFT M=2 P_HNSW is statistically inconclusive, but it is not an Orion reduction.  

Status: COMPLETE; both claims are retained as contradictory results rather than retuned.  
Anomalies: Full Orion imbalance remains material (SIFT max/mean 4.930 and GloVe 5.887 at M=32).
M>4 is logical shard scale on four physical hosts, not M-machine performance. Wall-clock local
latency is secondary and is not used to support either verdict.  
Required follow-up: none for this protocol. Any future work is a new experiment, such as a
balance-constrained Orion partitioner or an online-router study, and must not rewrite this result.  
Evidence SHA-256: construction audit
`6b7d2f04366d6cc5ab5329c7e5b248a7208da3eec27017828cf011d467316a51`; final audit
`3a57f2515baa8b569c22b515e781288811e203fd031561ef39c54f924bdd5df1`; final manifest
`a16c397d1dbaf0ed6f9c9905634504f918aa3fe584cdb785753dd4d7a2cfa474`.

## 2026-08-24T23:52:37Z - Stage 13 four-host virtualized linear-resource retest

Timestamp: 2026-08-24T23:52:37Z

Git commit: `7468366bdffddebf9b938ebb7217360e10944da1` plus checksum-bound Stage 13 implementation files

Experiment: independent C2/C3 retest with one isolated Qdrant container and one non-empty HNSW per
logical shard, while total server CPU and memory capacity grow exactly in proportion to M

Dataset: SIFT1M and glove-200-angular

Logical shards: M=1 common control and M=2,4,8,16,32 for Random, full-data K-Means, Full Orion,
and Orion-NoRefinement on four physical hosts

Resource contract: exactly M physical-core equivalents and 8*M GiB server-memory capacity;
M=32 consumes the complete declared 32-core/256-GiB pool, while lower M receives exactly M/32

Run IDs: `s13-sift1m-*` and `s13-glove-*`; registry has 42/42 PASS configurations

Primary metrics: TWCut, paired common-EF local navigability, P_exact, P_HNSW, Delta_P, W90,
9,000-query paired 10,000-replicate bootstrap, M=32 variance, refinement ablation, and balance

Observed result: all 15 final-audit checks pass. Every configuration has all five cgroup snapshots,
monotonic cumulative CPU, memory within its 8-GiB-per-shard cap, zero swap/OOM, client/server core
isolation, one non-empty HNSW per shard, checksum-bound manifests/raw arrays, and successful cleanup
restoration. Eight final CSV tables and ten valid PDFs are present.

C2: Topology-oblivious partitioning disrupts graph-search topology

Status: CONTRADICTED

Evidence: K-Means has lower TWCut than Full Orion at 8/10 dataset/M points: M=4,8,16,32 on both
datasets. Orion is lower only at M=2. Paired local-recall evidence is mixed rather than a consistent
Orion advantage, so the required topology-to-local-navigability chain remains absent.

C3: Topology-aware partitioning reduces required shard fan-out

Status: CONTRADICTED

Evidence: paired P_HNSW confidence intervals significantly favor K-Means at SIFT M=4,16,32 and at
all five GloVe scales; SIFT M=2 and M=8 are inconclusive. No point significantly favors Orion and
no point simultaneously satisfies the fan-out, W90, and balance support rule. At M=32, Orion minus
K-Means mean P_HNSW is +0.803 on SIFT and +1.615 on GloVe.

Status: COMPLETE. The alternative explanation that the prior result was caused by lower M using
all four-host compute is not supported: lower M is now strictly resource-limited to M/32, yet both
claim verdicts remain contradicted.

Anomalies: Full Orion remains severely imbalanced at high M. M=32 max/mean is 4.930 on SIFT and
5.887 on GloVe; GloVe Full Orion ranges from 4,908 to 217,725 vectors per shard. Refinement improves
TWCut over Orion-NoRefinement at all ten points but does not establish superiority over K-Means.

Required follow-up: none for this retest. Balance-constrained partitioning or online routing would
be new experiments and must not rewrite this independent negative evidence line.

Evidence SHA-256: final audit
`616492f420c0aef41dd446fb5c5fa8eac0f0534c3537d2bd16e713073d6c597c`; verdicts
`5e23bdbadf841d2c47c062aa80277c1f7aa981861468f1f24857c3b1c9087e6b`; raw registry
`19b4472225dbf8ca8a5374f73011eb4f1b2f816fd62a0cbf63ecbdf674808b27`.
