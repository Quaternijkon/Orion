# C1 Continuous Results

## 2026-08-20T15:59:34Z - Stage 0 instrumentation checkpoint

Timestamp: 2026-08-20T15:59:34Z  
Git commit: fe61fdac6760bc4d2bcd00cc512218d5fbd7e424 (dirty experiment worktree)  
Run IDs: stage0-counter-unit; stage0-qdrant-check  
Dataset: none (instrumentation validation)  
Partition method: none  
Physical machines: 4 reachable; no C1 collection deployed  
Logical shards: not applicable  
Recall target: 0.90  
Achieved recall: not measured  
Primary metrics: graph counter unit tests passed; `cargo check -p qdrant` passed  
Observed trend: no experimental trend yet  
Preliminary conclusion: request-level graph work can now be transported through local and remote
Qdrant accounting; live response validation remains required.  
C1 status: INSUFFICIENT  
Anomalies: the pre-existing cluster image has hardware reporting disabled and returned no usage
object during a temporary exact/ANN search probe.  
Required follow-up: deploy the instrumented image with hardware reporting enabled and verify that
distance-computation conversion and non-zero HNSW graph-node counts hold on a forced-index smoke
collection.

## 2026-08-20T16:47:01Z - Stage 0 live counter and deployment gate

Timestamp: 2026-08-20T16:47:01Z  
Git commit: fe61fdac6760bc4d2bcd00cc512218d5fbd7e424 (dirty experiment worktree)  
Run IDs: c1-20260820-v1; stage0-live-counter-smoke  
Dataset: deterministic synthetic 2,000 x 128 counter-only smoke; official SIFT1M and GloVe
inputs separately checksum-audited  
Partition method: one custom shard; not a C1 baseline result  
Physical machines: 4-node deployment validated; smoke shard placed on one node  
Logical shards: 1  
Recall target: 0.90  
Achieved recall: not reported; exact and approximate top-10 IDs matched in this smoke query  
Primary metrics: approximate = 1,227 distance computations and 70 graph-node expansions;
exact = 2,000 distance computations and 0 graph-node expansions  
Observed trend: the live REST response transports both primary and secondary graph-work counters,
and the graph-node counter distinguishes HNSW traversal from exact scan  
Preliminary conclusion: Stage 0 instrumentation and deployment gates pass; full-dataset fixed-recall
measurements are still required  
C1 status: INSUFFICIENT  
Anomalies: the first smoke attempt used the obsolete invalid `full_scan_threshold=1`; Qdrant
rejected it with HTTP 422. The script now uses the minimum valid 10 KiB threshold and passed.  
Required follow-up: execute the SIFT1M M=1,2,4 Random/K-Means Stage 1 matrix and validate recall,
per-shard work, placement, and result persistence.

## 2026-08-20T17:25:39Z - Stage 1 partition and CPU-counter gate

Timestamp: 2026-08-20T17:25:39Z  
Git commit: fe61fdac6760bc4d2bcd00cc512218d5fbd7e424 (dirty experiment worktree)  
Run IDs: stage1-sift1m-partition-artifacts-m1-m2-m4; stage0-live-counter-smoke-cpu-v2  
Dataset: SIFT1M for partition artifacts; deterministic synthetic 2,000 x 128 for the live gate  
Partition method: Random and K-Means artifacts; one custom shard for the counter-only gate  
Physical machines: four-node cluster on the instrumented image; no full SIFT1M collection yet  
Logical shards: 1, 2, and 4 artifacts  
Recall target: 0.90  
Achieved recall: not measured for a full SIFT1M configuration  
Primary metrics: six deterministic partition artifacts passed shape and population checks; the
strengthened live gate reported ANN = 1,228 distance computations, 70 graph-node expansions,
283 us worker CPU, and 279 us worker wall time; exact = 2,000 distance computations, zero graph
nodes, 196 us worker CPU, and 193 us worker wall time  
Observed trend: all required work and CPU-time fields survive the live REST path on image
`sha256:976ab5fd1c669fa971b5679fd5cdd6cf5d7882c976544771ad7e33ec35088a86`  
Preliminary conclusion: the Stage 1 loader/tuner may proceed with positive per-shard CPU-time
enforcement; C1-a through C1-d remain untested  
C1 status: INSUFFICIENT  
Anomalies: SIFT1M K-Means M=4 used all 12 configured Lloyd iterations and did not reach the
centroid-movement tolerance; the deterministic final artifact is retained and this non-convergence
must remain visible in downstream interpretation  
Required follow-up: build the common M=1 SIFT1M collection, run a limited official-query smoke,
then execute the full 1,000-query tuning protocol before measurement.

## 2026-08-20T17:37:22Z - Stage 1 SIFT1M M=1 fixed-recall checkpoint

Timestamp: 2026-08-20T17:37:22Z  
Git commit: fe61fdac6760bc4d2bcd00cc512218d5fbd7e424 (dirty experiment worktree)  
Run IDs: stage1-sift1m-random-m1-tune-q1000-pinned;
stage1-sift1m-kmeans-m1-tune-q1000-pinned;
stage1-sift1m-random-m1-measure-q9000-pinned;
stage1-sift1m-kmeans-m1-measure-q9000-pinned  
Dataset: SIFT1M, official query split (0-999 tuning; 1000-9999 measurement)  
Partition method: Random and K-Means; both reduce to the same single shard at M=1  
Physical machines: 1 worker on 10.10.1.1; benchmark fixed to cores 20-31 and Qdrant fixed to
cores 0-19  
Logical shards: 1  
Recall target: 0.90  
Achieved recall: 0.9099 on tuning and 0.91428 on the 9,000-query measurement set  
Primary metrics: selected `efSearch=24`; mean 864.57 distance computations, 35.44 graph-node
expansions, and 194.48 us worker CPU per Random measurement query (188.18 us for the separate
K-Means execution); fan-out = 1  
Observed trend: M=1 Random and K-Means have identical result IDs, per-query recall, distance work,
and graph-node work as required; K-Means adds about 24 us mean centroid-routing latency at M=1  
Preliminary conclusion: the full-dataset loader, fixed-recall tuner, worker counters, client merge,
and measurement split all pass for the M=1 reference point; no scale-out trend can yet be inferred  
C1 status: INSUFFICIENT  
Anomalies: the first full M=1 tuning/measurement executions inherited CPU affinity 0-31 rather
than the reserved 20-31; they are retained and explicitly marked `INVALID_AFFINITY`. The pinned
reruns above are the only valid Stage 1 M=1 evidence. The reported completed-QPS values are not
closed-loop E1 saturation measurements and must not be used for C1-a.  
Required follow-up: execute the pinned M=2 Random and K-Means Stage 1 configurations, then M=4.

## 2026-08-20T17:42:36Z - Stage 1 SIFT1M Random M=2 checkpoint

Timestamp: 2026-08-20T17:42:36Z  
Git commit: fe61fdac6760bc4d2bcd00cc512218d5fbd7e424 (dirty experiment worktree)  
Run IDs: stage1-sift1m-random-m2-tune-q1000-pinned;
stage1-sift1m-random-m2-measure-q9000-pinned  
Dataset: SIFT1M, official disjoint tuning and measurement sets  
Partition method: Random broadcast  
Physical machines: 2 workers, one logical shard on each of 10.10.1.1 and 10.10.1.2  
Logical shards: 2  
Recall target: 0.90  
Achieved recall: 0.9381 tuning; 0.94469 measurement  
Primary metrics: selected `efSearch=24`; fan-out = 2; 1,667.56 distance computations, 69.18
graph-node expansions, and 355.66 us worker CPU per measurement query  
Observed trend: relative to the valid M=1 Random point, shard size halved but aggregate distance
work rose from 864.57 to 1,667.56 (1.93x) because broadcast fan-out doubled; work per searched
shard fell only from 864.57 to 833.78 (3.6%)  
Preliminary conclusion: this single M=2 checkpoint is directionally consistent with C1-b/C1-c,
but Stage 1 is a pipeline gate and is not yet the ordered E2/E3 evidence  
C1 status: SUPPORTS (preliminary mechanism only; insufficient for final claim)  
Anomalies: no invalid counters or placement mismatches; completed-QPS is not an E1 saturation
measurement and is intentionally omitted from the headline summary  
Required follow-up: retire the M=2 Random collection after its evidence is persisted, then execute
the pinned M=2 K-Means configuration.

## 2026-08-20T17:48:01Z - Stage 1 SIFT1M K-Means M=2 checkpoint

Timestamp: 2026-08-20T17:48:01Z  
Git commit: fe61fdac6760bc4d2bcd00cc512218d5fbd7e424 (dirty experiment worktree)  
Run IDs: stage1-sift1m-kmeans-m2-tune-q1000-pinned;
stage1-sift1m-kmeans-m2-measure-q9000-pinned  
Dataset: SIFT1M, official disjoint tuning and measurement sets  
Partition method: K-Means centroid top-P  
Physical machines: 2 workers with deterministic one-shard-per-host placement  
Logical shards: 2  
Recall target: 0.90  
Achieved recall: 0.9048 tuning; 0.91387 measurement  
Primary metrics: selected `(P=1, efSearch=24)`; 860.53 distance computations, 34.92 graph-node
expansions, and 184.34 us worker CPU per measurement query; mean oracle fan-out = 1.0011  
Observed trend: the centroid router sent 4,583 measurement queries to shard 0 and 4,417 to shard
1 while probing exactly one shard each. Aggregate graph work remained essentially equal to M=1
because local work per searched shard decreased by only 0.5%  
Preliminary conclusion: at M=2, SIFT1M K-Means reaches the recall target with fan-out 1; this point
does not yet establish high K-Means fan-out and must be extended through M=32 before judging C1-b  
C1 status: INSUFFICIENT  
Anomalies: no counter, recall, affinity, placement, or shard-balance validity failures  
Required follow-up: persist checksums, retire the K-Means M=2 collection, and execute Random and
K-Means M=4 to close Stage 1.

## 2026-08-20T17:52:38Z - Stage 1 SIFT1M Random M=4 checkpoint

Timestamp: 2026-08-20T17:52:38Z  
Git commit: fe61fdac6760bc4d2bcd00cc512218d5fbd7e424 (dirty experiment worktree)  
Run IDs: stage1-sift1m-random-m4-tune-q1000-pinned;
stage1-sift1m-random-m4-measure-q9000-pinned  
Dataset: SIFT1M, official disjoint tuning and measurement sets  
Partition method: Random broadcast  
Physical machines: 4 workers, one logical shard per machine  
Logical shards: 4  
Recall target: 0.90  
Achieved recall: 0.9374 tuning; 0.93843 measurement  
Primary metrics: selected `efSearch=16`; fan-out = 4; 2,448.13 distance computations, 107.33
graph-node expansions, and 603.71 us worker CPU per measurement query  
Observed trend: work per searched shard was 612.03 distance computations, only 29.2% below M=1
despite a 75% shard-size reduction; broadcast raised aggregate distance work to 2.83x M=1  
Preliminary conclusion: the valid M=4 Random checkpoint strongly supports the local-work and
fan-out mechanism expected by C1-b/C1-c, pending the ordered E2/E3 experiments  
C1 status: SUPPORTS (preliminary mechanism only; insufficient for final claim)  
Anomalies: all 36,000 shard responses had positive distance, graph-node, and CPU-time counters;
no recall, placement, or affinity failure occurred  
Required follow-up: persist checksums, retire Random M=4, and execute the final Stage 1 K-Means
M=4 configuration while retaining the known 12-iteration partition anomaly.

## 2026-08-20T18:02:28Z - Stage 1 SIFT1M K-Means M=4 checkpoint

Timestamp: 2026-08-20T18:02:28Z  
Git commit: fe61fdac6760bc4d2bcd00cc512218d5fbd7e424 (dirty experiment worktree)  
Run IDs: stage1-sift1m-kmeans-m4-tune-q1000-pinned;
stage1-sift1m-kmeans-m4-measure-q9000-pinned  
Dataset: SIFT1M, official disjoint tuning and measurement sets  
Partition method: K-Means centroid top-P  
Physical machines: 4 workers, one logical shard per machine  
Logical shards: 4  
Recall target: 0.90  
Achieved recall: 0.9049 tuning; 0.91100 measurement  
Primary metrics: selected `(P=1, efSearch=48)`; 1,355.02 distance computations, 58.51
graph-node expansions, and 305.41 us worker CPU per measurement query; mean oracle fan-out =
1.1526 and p95 oracle fan-out = 2  
Observed trend: centroid routing reached the target while probing one shard per query, but
aggregate distance work was 1.57x the M=1 K-Means reference rather than decreasing with shard
count. The router sent 4,582, 1,202, 1,785, and 1,431 measurement queries to shards 0 through 3.
The selected search effort rose to efSearch=48, and the highly imbalanced largest shard retained
51.4% of the full dataset  
Preliminary conclusion: the M=4 K-Means point does not yet support persistent high fan-out because
P=1 still meets recall. It does show that partitioning alone did not reduce per-query graph work;
Stage 2 and Stage 3 must extend the mechanism check through M=32 before judging C1-b/C1-c  
C1 status: INSUFFICIENT  
Anomalies: Lloyd iteration reached the configured 12-iteration cap without convergence. Final
shard sizes were 513,838, 138,144, 197,753, and 150,265. All 9,000 routed shard responses had
positive distance, graph-node, and worker-CPU counters; query IDs were exactly 1000 through 9999  
Evidence SHA-256: partition `b4e5e9559708296b1509d8aa51b9826cee8356dc795c1a24e4cdd9072030a259`;
tuning `c856a9b7029063b0e6c8d2a5a6a2e3c14c9d3dabbc4cbaa7175af2c6a2233a68`;
summary `04419ee741f43b35b3fe8bf3b76557896f2dadd9d79bec29cebc05b5643788b8`;
per-query `50c0bead4e8bbad62a6517582286b29fe4499c5b5585ee8f686068de11670c99`  
Required follow-up: retire only the temporary K-Means M=4 collection, then begin Stage 2 E3
local-search scaling for SIFT1M in M=1,2,4,8,16,32 order.

## 2026-08-20T18:13:11Z - Stage 2 SIFT1M M=8/16/32 partition gate

Timestamp: 2026-08-20T18:13:11Z  
Dataset: SIFT1M  
Methods: deterministic Random and K-Means  
Logical shards: 8, 16, 32  
Validation: every artifact contains exactly 1,000,000 assignments, covers the complete shard-ID
range, has no empty shard, and matches its recorded shard-count metadata  
Random shard-size ranges: M=8 124,536–125,527; M=16 62,060–62,901; M=32
30,937–31,639  
K-Means shard-size ranges: M=8 75,637–223,116; M=16 40,145–92,138; M=32
11,607–51,413  
Anomalies: K-Means M=8,16,32 all reached the configured 12-iteration limit without satisfying
the 0.0001 centroid-movement tolerance. Final maximum movements were 5.2499, 9.1108, and 7.5484.
The artifacts remain valid deterministic partitions, but every dependent result must retain this
non-convergence and imbalance warning  
C1 status: INSUFFICIENT; this is an artifact gate, not C1-c evidence  
Required follow-up: implement and validate isolated per-shard E3 measurement, then execute the
SIFT1M local-work matrix in protocol order.

## 2026-08-20T18:22:29Z - Stage 2 SIFT1M E3 M=1 baseline

Timestamp: 2026-08-20T18:22:29Z  
Dataset: SIFT1M measurement queries 1000–9999  
Methods: Random and K-Means; identical one-shard layout  
Selected configuration: fan-out 1, efSearch=24, tuning Recall@10=0.9099  
Isolation: one request at a time, zero competing logical shards, benchmark cores 20–31, worker
cores 0–19  
Primary metrics: both methods measured 864.57 distance computations and 35.44 graph-node
expansions per searched shard. Isolated worker CPU was 143.03 us for Random and 145.08 us for
K-Means  
Cache facts: vector storage = 512,000,000 B; HNSW graph files = 87,245,374 B; total local index
= 599,245,374 B; machine LLC = 16,777,216 B. The M=1 index is 35.7x LLC and is not cache-resident  
Validation: each raw file contains exactly 9,000 positive-counter searched-shard events with
query IDs 1000–9999. Distance/node means exactly match the valid Stage 1 M=1 measurement  
C1 status: INSUFFICIENT; M=1 establishes the normalization reference only  
Required follow-up: rebuild the M=2 Random and K-Means collections and execute their isolated E3
measurements using the already selected Stage 1 configurations.

## 2026-08-20T18:30:37Z - Stage 2 SIFT1M E3 M=2 checkpoint

Timestamp: 2026-08-20T18:30:37Z  
Random: fan-out 2, efSearch=24, tuning Recall@10=0.9381, 18,000 isolated searched-shard events  
K-Means: fan-out 1, efSearch=24, tuning Recall@10=0.9048, 9,000 isolated searched-shard events  
Primary metrics: Random = 834.69 distance computations and 35.65 graph nodes per searched shard;
K-Means = 861.93 distance computations and 35.28 graph nodes per searched shard  
Normalized distance work versus M=1: Random = 0.9654; K-Means = 0.9969. Both are far above the
ideal 0.5 curve, so halving shard size produced only a 3.46% and 0.31% reduction in local work  
Cache facts: mean vector storage is 256,000,000 B per shard, already 15.3x the 16 MiB LLC before
adding HNSW graph bytes; both M=2 regimes remain larger than LLC  
Validation: Random contains the full 1000–9999 range once for each shard; K-Means contains the
range once overall with exact 4,583/4,417 routed counts. Every distance/node/CPU counter is
positive. Both temporary collections were checksum-audited and deleted on all four peers  
Evidence boundary: vector bytes and RSS are available from each worker telemetry. Graph-file bytes
are exact for the controller shard; remote graph-file sizes are null pending a later local-copy
resource audit and are not estimated  
C1 status: SUPPORTS C1-c (preliminary; M=4–32 still required)  
Required follow-up: execute the M=4 Random and K-Means isolated E3 points, preserving the known
K-Means M=4 non-convergence and severe imbalance anomaly.

## 2026-08-20T18:37:53Z - Stage 2 SIFT1M E3 M=4 checkpoint

Timestamp: 2026-08-20T18:37:53Z  
Random: fan-out 4, efSearch=16, tuning Recall@10=0.9374, 36,000 isolated events  
K-Means: fan-out 1, efSearch=48, tuning Recall@10=0.9049, 9,000 isolated events  
Primary metrics: Random = 609.52 distance computations and 27.33 graph nodes per searched shard;
K-Means = 1,359.79 distance computations and 57.69 graph nodes per searched shard  
Normalized distance work versus M=1: Random = 0.7050; K-Means = 1.5728, compared with ideal
0.25. Random local work is 2.82x ideal; K-Means local work increases above the M=1 baseline  
Cache facts: Random mean vector storage is 128,000,000 B (7.63x LLC) before graph bytes. Every
K-Means shard has vector storage above LLC; the largest controller shard has 263,085,056 B of
vectors plus 42,355,245 B of graph files  
Anomalies: K-Means remained unconverged at 12 iterations and retained shard sizes 513,838,
138,144, 197,753, and 150,265. The larger efSearch selected to satisfy fixed recall and severe
imbalance are part of the observed result, not post-hoc exclusions  
Validation: all raw rows, route counts, counters, affinities, checksums, and collection deletion
checks passed. Remote graph-file bytes remain pending local-copy audit and are not estimated  
C1 status: SUPPORTS C1-c through the physical M=4 range; M=8–32 logical points remain required  
Required follow-up: build and tune M=8 Random and K-Means, then execute their isolated E3 points.

## 2026-08-20T18:53:13Z - Stage 2 SIFT1M E3 M=8 logical-shard checkpoint

Timestamp: 2026-08-20T18:53:13Z  
Execution scope: logical-shard simulation on four physical machines; not physical M=8 throughput  
Random: fan-out 8, efSearch=16, tuning Recall@10=0.9599, 72,000 isolated events  
K-Means: fan-out 2, efSearch=32, tuning Recall@10=0.9244, 18,000 isolated events  
Primary metrics: Random = 568.09 distance computations and 26.23 graph nodes per searched shard;
K-Means = 989.62 distance computations and 41.63 graph nodes per searched shard  
Normalized distance work versus M=1: Random = 0.6571; K-Means = 1.1446, compared with ideal
0.125. Random local work is 5.26x ideal; K-Means local work remains above the M=1 baseline  
Fan-out consequence: Random aggregate distance work is 4,544.71 per query and K-Means aggregate
distance work is 1,979.24 per query at their selected fixed-recall configurations  
Cache facts: mean vector storage is 64,000,000 B per shard, 3.81x the 16 MiB LLC before graph
bytes. The two controller-local graph files are exact for each method; remote graph-file bytes
remain null pending the local-copy resource audit  
Anomalies: K-Means remained unconverged at 12 iterations and retained shard sizes from 75,637 to
223,116. This warning and the imbalance are preserved in the primary record  
Validation: Random contains every query exactly eight times; K-Means contains every query exactly
twice. All 90,000 raw rows have positive distance, node, and CPU counters, all summary means
reproduce exactly, and both temporary collections were deleted with 404 verification on all four
peers  
C1 status: SUPPORTS C1-c through M=8 logical simulation; M=16 and M=32 remain required  
Required follow-up: implement exact prefix-reuse tuning, validate equivalence to exhaustive P x ef
enumeration, then execute the M=16 and M=32 logical-shard E3 points.

## 2026-08-20T19:09:51Z - Stage 2 SIFT1M E3 M=16 logical-shard checkpoint

Timestamp: 2026-08-20T19:09:51Z  
Execution scope: logical-shard simulation on four physical machines; not physical M=16 throughput  
Random: fan-out 16, efSearch=16, tuning Recall@10=0.9765, 144,000 isolated events  
K-Means: fan-out 3, efSearch=24, tuning Recall@10=0.9184, 27,000 isolated events  
Primary metrics: Random = 518.55 distance computations and 25.45 graph nodes per searched shard;
K-Means = 774.15 distance computations and 33.06 graph nodes per searched shard  
Normalized distance work versus M=1: Random = 0.5998; K-Means = 0.8954, compared with ideal
0.0625. Random local work is 9.60x ideal and K-Means local work is 14.33x ideal  
Fan-out consequence: Random aggregate distance work is 8,296.81 per query and K-Means aggregate
distance work is 2,322.45 per query at the selected fixed-recall configurations  
Prefix-reuse validation: on the same 20-query ef=24 slice, all 16 exhaustive and prefix-derived
candidates matched exactly on fan-out, recall, distance work, node work, validity, and selection.
Both selected P=3/ef=24. Prefix reuse reduced shard searches from 2,720 to 320; worker CPU time
varied between the two separate timing runs and is not part of the candidate selection rule  
Anomalies: K-Means remained unconverged at 12 iterations with shard sizes 40,145–92,138. Every
dependent artifact and primary record retains this warning  
Validation: all 171,000 raw rows have positive distance, node, and CPU counters; route counts and
summary means reproduce exactly; both temporary collections were deleted with 404 verification
on all four peers. Controller graph-file bytes are exact for four local shards per method; remote
graph-file bytes remain pending local-copy audit  
C1 status: SUPPORTS C1-c through M=16 logical simulation; M=32 remains required  
Required follow-up: execute the M=32 Random and K-Means logical-shard E3 points, using the
validated prefix-reuse path for K-Means tuning.

## 2026-08-20T19:30:12Z - Stage 2 SIFT1M E3 M=32 logical-shard checkpoint

Timestamp: 2026-08-20T19:30:12Z  
Execution scope: logical-shard simulation on four physical machines; not physical M=32 throughput  
Random: fan-out 32, efSearch=16, Recall@10=0.9843 tuning and 0.98760 measurement, 288,000
isolated searched-shard events  
K-Means: fan-out 3, efSearch=32, Recall@10=0.9136 tuning and 0.92004 measurement, 27,000
isolated searched-shard events  
Primary metrics: Random = 468.94 distance computations and 24.61 graph nodes per searched shard;
K-Means = 887.61 distance computations and 40.11 graph nodes per searched shard  
Normalized distance work versus M=1: Random = 0.5424; K-Means = 1.0267, compared with ideal
0.03125. Random local work is 17.36x ideal and K-Means local work is 32.85x ideal  
Fan-out consequence: Random aggregate distance work is 15,006.16 per query, 17.36x the M=1
reference; K-Means aggregate distance work is 2,662.84 per query, 3.08x the M=1 reference  
Prefix-reuse execution: the complete 32 x 11 P-by-ef matrix contains 352 unique candidates and
used 352,000 shard searches. Every candidate has the pinned 20-31 affinity and positive worker
CPU time; the deterministic selection is P=3/ef=32  
Cache facts: all eight exact controller-local Random indexes exceed the 16 MiB LLC, with mean
total size 17,806,144 B. K-Means has a mixed controller-local cache regime because of its severe
size imbalance. Remote graph-file sizes remain null pending local-copy audit and are not estimated  
Anomalies: K-Means again reached the 12-iteration cap without convergence and retained shard sizes
from 11,607 to 51,413. This warning, exact shard counts, and routing frequencies are retained in
the primary record  
Validation: every Random query appears exactly 32 times and every K-Means query exactly three
times over query IDs 1000-9999. All counters are positive, integer summary means reproduce
exactly, latency differs only by CSV decimal rounding, and both collections were deleted with
four-peer 404 verification  
C1 status: SUPPORTS C1-c across the complete SIFT1M M=1-32 range and provides preliminary C1-b
fan-out evidence; E2 is still required for the per-query minimum-shard distribution  
Required follow-up: execute SIFT1M E2 fan-out in M=1,2,4,8,16,32 order, then E1 and E4.

## 2026-08-20T19:44:48Z - Stage 3 SIFT1M E2 fan-out checkpoint

Timestamp: 2026-08-20T19:44:48Z  
Run ID: stage3-e2-sift1m-fanout  
Dataset: SIFT1M measurement queries 1000-9999, 9,000 queries per configuration  
Configurations: Random and K-Means at M=1,2,4,8,16,32; 108,000 derived per-query rows  
Actual fixed fan-out: Random = 1,2,4,8,16,32; K-Means = 1,1,1,2,3,3  
Oracle mean fan-out: Random = 1.000, 1.978, 3.148, 4.903, 6.595, 7.700; K-Means =
1.000, 1.001, 1.153, 1.379, 1.610, 1.871  
Normalized M=32 fan-out: Random actual = 1.000 and oracle = 0.2406; K-Means actual = 0.09375
and oracle = 0.05848  
Per-query heterogeneity: at K-Means M=32 the oracle lower bound has median 2, p95 4, p99 4,
and range 1-7. Counts are 3,701 queries at one shard, 3,312 at two, 1,517 at three, and 470
at four or more. Random M=32 has median 8, p95/p99 9, and range 4-9  
Tuning boundary: the globally selected K-Means M=16 serving point is P=3/ef=24 because it minimizes
aggregate graph work under the protocol. P=2 is recall-feasible only with greater local search
effort; both selected P and minimum feasible P are recorded rather than conflated  
Sensitivity: repeating the global selection and oracle calculation at Recall@10 thresholds 0.89,
0.90, and 0.91 preserves the monotone fan-out trends and the M=32 endpoints. For top-10 recall,
0.89 and 0.90 both require nine ground-truth hits, while 0.91 requires all ten  
Validation: all 12 configurations contain exactly 9,000 unique query IDs, every actual selected
shard list matches its fixed fan-out, all measurement recalls remain above 0.90, all normalized
values reproduce, all source/evidence hashes match, and the five required/canonical PDFs pass
format and visual inspection  
C1 status: PARTIALLY SUPPORTS C1-b. Random fan-out grows maximally and its oracle lower bound
remains high. K-Means reduces fan-out substantially but its actual P rises from 1 to 3 and its
heterogeneous oracle tail reaches 7; the P=3 plateau from M=16 to M=32 makes the SIFT-only
evidence insufficient for the final cross-dataset conclusion  
Required follow-up: run SIFT1M E1 closed-loop physical scale-out at M=1,2,4 with at least three
independent repetitions per configuration, then execute E4.

## 2026-08-20T20:19:35Z - Stage 4 SIFT1M E1 physical scale-out checkpoint

Timestamp: 2026-08-20T20:19:35Z  
Git commit: `fe61fdac6760bc4d2bcd00cc512218d5fbd7e424`  
Run IDs: `stage4-e1-sift1m-common-m1`, `stage4-e1-sift1m-random-m2`,
`stage4-e1-sift1m-random-m4`, `stage4-e1-sift1m-kmeans-m2`, and
`stage4-e1-sift1m-kmeans-m4`  
Dataset: SIFT1M  
Partition methods: Random and K-Means; the M=1 unsharded layout is shared  
Physical machines and logical shards: M=1,2,4 with one physical shard per machine  
Recall target: Recall@10 >= 0.90  
Achieved recall: shared M=1 = 0.91428; Random M=2/4 = 0.94630/0.93807; K-Means
M=2/4 = 0.91257/0.91049  
Primary QPS metrics, mean +/- sample standard deviation over three repetitions: shared M=1 =
1809.03 +/- 11.99; Random M=2/4 = 1065.15 +/- 9.78 and 589.34 +/- 3.49; K-Means M=2/4 =
1786.56 +/- 5.33 and 1736.64 +/- 10.15  
Normalized QPS at M=1,2,4: Random = 1.0000, 0.5888, 0.3258; K-Means = 1.0000, 0.9876,
0.9600  
Scaling efficiency at M=1,2,4: Random = 1.0000, 0.2944, 0.0814; K-Means = 1.0000,
0.4938, 0.2400  
Observed trend: Random throughput falls sharply as broadcast fan-out grows. K-Means preserves
nearly the one-worker absolute QPS but gains no throughput from additional physical workers and
therefore remains far below ideal linear scaling  
Bottleneck validation: all 15 repetitions are unflagged. Maximum aggregator CPU is 9.56% of the
12 reserved cores, maximum routing share is 15.10%, maximum network utilization is 0.0853%, and
maximum aggregator queue depth is zero  
Preliminary conclusion: SIFT1M physical throughput scaling is strongly sub-linear for both
partition methods. The Random curve exposes the broadcast scatter cost directly; K-Means reduces
fan-out enough to preserve absolute throughput but not enough to produce physical speedup  
C1 status: SUPPORTS C1-a for SIFT1M; GloVe remains required for the final cross-dataset claim  
Anomalies: the K-Means M=4 partition again reached the configured 12-iteration cap without
convergence and retained shard sizes 513,838, 138,144, 197,753, and 150,265. The anomaly is
preserved in the aggregate record and does not trigger an infrastructure bottleneck flag  
Validation: all 270,000 measurement rows across five unique layouts match their recorded hashes;
all source-result, CSV, summary, and figure hashes reproduce; all temporary M=2/M=4 collections
were deleted with four-peer 404 verification. The focused suite passes 20 tests and the three E1
PDFs pass format and visual inspection  
Required follow-up: execute SIFT1M E4 aggregate-work decomposition and compare its M=1,2,4
projection against these measured physical points before starting the GloVe sequence.

## 2026-08-20T20:29:46Z - Stage 5 SIFT1M E4 aggregate-work checkpoint

Timestamp: 2026-08-20T20:29:46Z  
Git commit: `fe61fdac6760bc4d2bcd00cc512218d5fbd7e424`  
Run ID: `stage5-e4-sift1m-aggregate-work`  
Dataset: SIFT1M measurement queries 1000-9999, 9,000 queries per configuration  
Configurations: Random and K-Means at M=1,2,4,8,16,32; 108,000 aggregate per-query rows
derived from 666,000 validated E3 searched-shard events  
Observed aggregate distance computations/query at M=1,2,4,8,16,32: Random = 864.57,
1669.37, 2438.09, 4544.71, 8296.81, 15006.16; K-Means = 864.57, 861.93, 1359.79,
1979.24, 2322.45, 2662.84  
Normalized aggregate work at M=32: Random = 17.3568 and K-Means = 3.0800 versus the ideal
1/32 = 0.03125 query-work curve  
Observed aggregate worker CPU time/query at M=1 and M=32: Random = 143.03 us and 1849.95 us;
K-Means = 145.08 us and 300.46 us  
Decomposition accuracy: fan-out multiplied by mean local distance work reproduces observed mean
aggregate distance work exactly by construction of the fixed serving fan-out. Pearson and
Spearman correlations are 1.0 for both methods; mean relative error is zero for Random and
2.85e-17 for K-Means  
C1-d aggregate-cost status: SUPPORTS. Fan-out and slowly decreasing local search work fully
account for the observed aggregate graph-search work trend  
Physical projection sanity check: FAILED. At M=1,2,4 the work-bound projections are
1.0000, 1.0358, 1.4184 for Random and 1.0000, 2.0061, 2.5432 for K-Means, while measured E1
throughput is 1.0000, 0.5888, 0.3258 and 1.0000, 0.9876, 0.9600. Pearson correlations are
-0.8405 and -0.9252, with Spearman = -1.0 for both  
Physical-attribution status: INSUFFICIENT. The aggregate-work decomposition is valid, but the
protocol's idealized M/W work-bound does not explain the measured physical E1 curve. M>4
throughput projections are therefore withheld and are not reported as measured or validated
cluster throughput  
Anomalies: K-Means M=4,8,16,32 retain their 12-iteration non-convergence warnings. The new
cross-stage contradiction indicates an unmodeled physical/client execution limit rather than an
error in the per-query work identity  
Validation: every E2 selected shard set matches the corresponding E3 events and efSearch, all
distance/node/CPU counters are positive, all source and output hashes reproduce, the focused
suite passes 20 tests, all five E4 PDFs are one-page PDF 1.4 files, and the E4 manifest record is
registered exactly once  
Required follow-up: inspect the E1 execution path for an unmodeled aggregator/client serialization
limit before starting GloVe. Do not publish or extrapolate the M>4 work-bound curve unless a
corrected physical E1 run passes the M=1,2,4 sanity check.

## 2026-08-20T21:04:24Z - Stage 6 corrected SIFT1M E1 physical scale-out checkpoint

Timestamp: 2026-08-20T21:04:24Z  
Git commit: `fe61fdac6760bc4d2bcd00cc512218d5fbd7e424`  
Run IDs: `stage6-e1-corrected-sift1m-common-m1`,
`stage6-e1-corrected-sift1m-random-m2`, `stage6-e1-corrected-sift1m-random-m4`,
`stage6-e1-corrected-sift1m-kmeans-m2`, and
`stage6-e1-corrected-sift1m-kmeans-m4`  
Correction boundary: Stage 4 used a serialized single-process Python query path whose CPU cost
rose with fan-out, so its QPS primarily measured client capacity rather than distributed worker
capacity. Stage 4 evidence is retained under its original run IDs and checksum-identical
`stage4-original-*` canonical artifacts, but is superseded for physical E1 interpretation  
Corrected client: 128 persistent query processes, four concurrent shard-request workers per
process, closed-loop batches, `/proc` accounting for the coordinator and every query process,
and zero growing client queue by construction. Client cores remain 20-31 and Qdrant workers
remain pinned to 0-19  
Selected concurrency at the p99 knee: shared M=1 = 32; Random M=2/4 = 16/16; K-Means M=2/4 =
32/32  
Achieved recall: shared M=1 = 0.91428; Random M=2/4 = 0.94591/0.93809; K-Means M=2/4 =
0.91438/0.91078  
Primary QPS metrics, mean +/- sample standard deviation over three repetitions: shared M=1 =
17884.05 +/- 751.80; Random M=2/4 = 11074.41 +/- 190.75 and 7044.34 +/- 96.66; K-Means
M=2/4 = 19985.57 +/- 800.56 and 17945.88 +/- 575.41  
Normalized QPS at M=1,2,4: Random = 1.0000, 0.6192, 0.3939; K-Means = 1.0000, 1.1175,
1.0035  
Scaling efficiency at M=1,2,4: Random = 1.0000, 0.3096, 0.0985; K-Means = 1.0000,
0.5588, 0.2509  
Bottleneck validation: all 15 repetitions are unflagged. Maximum coordinator CPU is 39.00% of
one core, maximum total client CPU is 81.86% of the 12 reserved cores, maximum worker CPU is
50.95% of reserved worker cores, maximum network utilization is 0.9347%, and maximum client
queue depth is zero  
Conclusion: corrected Random throughput remains strongly sub-linear as broadcast fan-out grows.
K-Means gains 11.75% at M=2 but returns to the M=1 throughput level at M=4, so it also remains
far below ideal linear physical scaling  
C1 status: SUPPORTS C1-a for SIFT1M with corrected client-capacity controls; GloVe remains
required for the final cross-dataset claim  
Anomalies: K-Means M=4 retains the exact 12-iteration non-convergence artifact and shard sizes
513,838, 138,144, 197,753, and 150,265  
Validation: all 270,000 measurement rows match their recorded hashes; all five source results,
the corrected aggregate CSV/summary, and three E1 PDFs reproduce; every temporary M=2/M=4
collection was deleted with four-peer 404 verification; the corrected E1 manifest record is
registered exactly once and explicitly supersedes Stage 4  
Required follow-up: rerun E4 against the corrected physical E1 curve and retain the global M>4
release gate unless both partition methods pass the M=1,2,4 sanity check.

## 2026-08-20T21:10:52Z - Stage 7 corrected SIFT1M E4 checkpoint

Timestamp: 2026-08-20T21:10:52Z  
Git commit: `fe61fdac6760bc4d2bcd00cc512218d5fbd7e424`  
Run ID: `stage7-e4-corrected-sift1m-aggregate-work`  
Correction boundary: the Stage 5 aggregate-work identity remains valid and its original
canonical artifacts are retained under checksum-identical `stage5-original-*` names. Stage 7
supersedes only the physical E1 comparison by using the corrected Stage 6 curve  
Aggregate-work validation: the 108,000 per-query rows derived from 666,000 E3 searched-shard
events are byte-identical to Stage 5. Fan-out multiplied by mean local distance work still
matches observed aggregate distance work with Pearson/Spearman 1.0 for both methods and
negligible relative error  
Random physical sanity check at M=1,2,4: work-bound projection = 1.0000, 1.0358, 1.4184;
corrected measured QPS = 1.0000, 0.6192, 0.3939; Pearson = -0.8291, Spearman = -1.0, and mean
relative error = 1.0913  
K-Means physical sanity check at M=1,2,4: work-bound projection = 1.0000, 2.0061, 2.5432;
corrected measured QPS = 1.0000, 1.1175, 1.0035; Pearson = 0.1982, Spearman = 0.5, and mean
relative error = 0.7766  
Physical-attribution status: INSUFFICIENT. K-Means now has positive trend agreement, but Random
remains anticorrelated. The all-partition-method release gate therefore fails and every M>4
throughput projection remains withheld  
C1-d aggregate-cost status: SUPPORTS. The graph-search work identity is unchanged; it is the
idealized conversion from aggregate work to physical throughput that remains unsupported  
Validation: all corrected E4 source/output hashes reproduce; the per-query, aggregate summary,
model-accuracy, and five PDF artifacts pass row, format, and visual inspection; the corrected E4
manifest record is registered exactly once and explicitly supersedes Stage 5  
Required follow-up: begin the ordered GloVe sequence with the corrected multiprocess E1 runner,
while retaining `INSUFFICIENT` physical attribution and withholding M>4 projections unless the
cross-dataset evidence changes the release decision.

## 2026-08-20T21:39:31Z - Stage 8 GloVe dataset and partition gate

Timestamp: 2026-08-20T21:39:31Z  
Git commit: `fe61fdac6760bc4d2bcd00cc512218d5fbd7e424`  
Dataset: `glove-200-angular`, 1,183,514 train vectors x 200 dimensions and 10,000 queries;
cosine normalization enabled  
Dataset SHA-256: `4839085e5a8bb293434a1a66e1aa0193afc3f07c6797a85f1dbd91656172da20`
reproduced from the live HDF5 file and matched by every partition artifact  
Partition scope: Random and K-Means at M=1,2,4,8,16,32; seed 20260820; 12 NPZ artifacts  
Validation: every artifact hash reproduced; metadata records the expected dataset, angular/Cosine
distance, normalization flag, seed, M, point count, and dimension. Every assignment array covers
IDs 0 through M-1, totals exactly 1,183,514 points, has no empty shard, and matches its recorded
shard counts. Random artifacts have no centroids or K-Means history; K-Means artifacts contain
exactly M normalized 200-dimensional centroids and no empty cluster in any recorded iteration  
Random shard balance: M=2 counts are 592,332/591,182; at M=32 the minimum and maximum counts are
36,675 and 37,353  
K-Means convergence: M=1 converged in two iterations with final movement 0.0. M=2,4,8,16,32 all
used the 12-iteration maximum without reaching tolerance; their final maximum centroid movements
are 0.00619, 0.01886, 0.06750, 0.15173, and 0.19271  
K-Means imbalance: M=2 counts are 828,099/355,415; M=4 counts are
528,812/185,963/291,179/177,560; M=8,16,32 minimum/maximum counts are
68,251/256,096, 20,789/119,678, and 15,843/68,153  
Artifact SHA-256 values: Random M=1/2/4/8/16/32 =
`97313d67339f0ea621aa42ed5562c3701d783589b96364be1ee981a03901f13b`,
`5c154087494614be1871db5638e1eec106487dff4c6d96a0c35ec6de855ff77b`,
`57931d3f726ec2e3bd5960c47145792b036502044929061e2bbc2135e0e0c56f`,
`576c1b3df2c3e34d92bca9ab819474e609d01d3f64d1643863dee80320eb4aa8`,
`1f6140d37db3bf86495860d8494e36985aa8ba523e2095e8527b72fe655c0a4b`, and
`27eb4d507a626277b2755c430d33cc816a37e269a19518045a34e70cdce81569`; K-Means =
`c459b380050e2765667289ba2a55884ce3b4421a00ec49d0c9123bd2dc297c56`,
`675f7d857b1abbb94b4ffc08da17df710b9c639aaf9cfe73c2581b21f4dcd46b`,
`9bca00c43e83ffc07b944055c92bc6054e469e51e5db970e01996fa9db2cda7b`,
`026c59a43604885c1cd5c0f87ed907c5f10627a3587cf5b605c17fd98b41d4de`,
`e0a2d50f079def7c556e3b6973da4c1e59445ffe40f9c00ef5d9ebe96baa5f93`, and
`79786a469498c64fe6eabe821ec716bf7ac72da621f37afd66d5141783991bd4`  
C1 status: partition gate passed; no GloVe C1-a through C1-d conclusion is drawn before the
ordered E3/E2/E1/E4 measurements  
Required follow-up: verify persisted SIFT evidence and collection headroom, retire only the
residual SIFT1M M=1 collection if needed, then build and measure GloVe E3 M=1.

## 2026-08-20T22:01:14Z - Stage 8 GloVe E3 M=1 checkpoint

Timestamp: 2026-08-20T22:01:14Z  
Dataset and query split: `glove-200-angular`; queries 0-999 for tuning and 1000-9999 for
measurement  
Collection: shared one-shard `c1v1_glove_common_m1`, built with 1,183,514 normalized 200-D
vectors, Cosine distance, HNSW M=32, efConstruction=200, and exact controller placement  
Build validation: collection reached green with 1,183,514 points and indexed vectors, two final
segments, zero queued updates, and Random M=1 partition SHA-256
`97313d67339f0ea621aa42ed5562c3701d783589b96364be1ee981a03901f13b`  
Pinned tuning: both Random and K-Means evaluated the 11-value efSearch grid on cores 20-31 and
selected P=1/efSearch=384 at Recall@10=0.9148. efSearch=256 remained below target at 0.8901  
Measurement Recall@10: 0.91206 for both methods over 9,000 queries  
Local work per searched shard: Random and K-Means are algorithmically identical at 12,837.62
distance computations and 397.82 graph nodes; p95 values are 18,101.05 and 408  
Worker CPU time: Random = 2,944.92 us/query and K-Means = 2,833.19 us/query; timing differences
are retained but are not algorithmic-work differences  
Resource facts: vector storage = 946,811,200 B, graph index = 145,382,363 B, total local index =
1,092,193,563 B versus a 16,777,216 B LLC  
Evidence SHA-256: Random tuning/per-search/summary =
`81b0ead9b97cd4c57f4ca4db3ce661b6845ad8e14d9a86fd7bd39fce77163590`,
`3ef9f05e855f1ab7f98ccfe7fefb3fd7214a1394d1c5146bbb4847e030ec51f5`,
`596cdf1a4cdc92aff7527cec14e4d49c0932772f1568b7d34bdda4d79c5d0bb6`;
K-Means = `4c55bfffb291c71cfd2696f494c9940ea8056fc30b5f74f5669d985e32991009`,
`26877f56539d68cfaa46f37d75315c5d7fcc2b449732dad1c20fa75fa0eac010`, and
`1a7184f1aced7018212f590bba244a8edf7fd9592a89044925f921470bd6fd12`  
Validation: both 9,000-row files contain query IDs 1000-9999 exactly once, enforce the full
20-31 benchmark affinity, have positive distance/node/CPU/wall counters, reproduce every summary
mean, and match exactly on query IDs, result IDs, ground-truth hits, recall contribution,
distance work, and graph-node work  
Lifecycle: after evidence verification, the collection was deleted; all four peers return HTTP
404, the controller collection directory is absent, and disk headroom returned to 4.0 GB  
C1 status: valid GloVe M=1 E3 reference point; no scale trend is inferred until M=2 through M=32  
Required follow-up: execute GloVe Random M=2, then K-Means M=2, preserving the K-Means
non-convergence and shard-imbalance anomaly.

## 2026-08-20T22:25:49Z - Stage 8 GloVe E3 M=2 checkpoint

Timestamp: 2026-08-20T22:25:49Z  
Execution scope: two physical machines with one custom shard on each of 10.10.1.1 and 10.10.1.2  
Random partition: 592,332/591,182 points; full broadcast fan-out P=2  
Initial Random selection: efSearch=192 reached tuning Recall@10=0.9004 but held-out measurement
Recall@10=0.89816. Per protocol, the run is retained as `INVALID_RECALL`, excluded from the valid
E3 curve, and preserved under explicit `invalid-recall-ef192` filenames  
Random retune: the next predeclared tuning candidates 256/384/512 were evaluated only on queries
0-999. efSearch=256 was selected at tuning Recall@10=0.9179 and achieved measurement
Recall@10=0.91547 on queries 1000-9999  
Random valid local work: 9,071.05 distance computations and 267.69 graph nodes per searched shard;
normalized distance work = 0.7066 versus M=1, compared with ideal 0.5  
K-Means partition anomaly: the artifact reached the 12-iteration cap without convergence and
retained the severe 828,099/355,415 split  
K-Means selection: exact ranked-prefix tuning evaluated all 22 P-by-ef candidates with 22,000
shard searches and selected P=2/efSearch=256 at tuning Recall@10=0.9042. The held-out measurement
passed at Recall@10=0.90440, so K-Means also broadcasts to both shards at M=2  
K-Means valid local work: 8,827.00 distance computations and 268.57 graph nodes per searched
shard; normalized distance work = 0.6876 versus M=1. The 828,099-point shard averaged 9,383.36
distance computations while the 355,415-point shard averaged 8,270.65  
Fixed-recall runner correction: the E3 runner now sums per-shard recall contributions for every
measurement query, records `measurement_recall`, marks sub-target runs `INVALID_RECALL`, writes
the raw CSV/summary before returning non-zero, and rejects missing/out-of-range query evidence.
The focused suite passes 24 tests  
Valid evidence SHA-256: Random prepare/tuning/per-search/summary =
`91872ee8dc9ee5a0aaf415c2cd124900f597295780be7e9ed8583c160904cd15`,
`7fa408d647ef1a2e2a1e93d18bf75b19049c76de7affa9701e239ae8745f2a77`,
`fc891488d07bbae7ed966ecfa4cf29b32726828dd6cbc22766e719a867bf868d`, and
`1aaab12af9a5022e167caebe6dca4b72bc9cd23962fc849bbc311b4e098c365a`;
K-Means = `a020596db51653e557a5dc08f753d9d166c992a95ee039035fe50375ae4a7ffd`,
`feb1470f87aad30d1843578df776f537188afa9fdb7ccdc12ac6cd88fec5b14e`,
`fdb4bd9652589a327297de01814aa501fd4fdb484b331a2af4895ee8786ca291`, and
`7e1163e5b0bea5d3decc0f7223e2c89edb3b2db41a7768c8b8eefb16a041b800`  
Invalid Random evidence SHA-256: initial tuning =
`b483e100fdebb7aa0a0d084c06d3d2d8eb67b3d5040354aa0770d28af127f7c3`, per-search =
`ba712d04e4f397f7226ca7fc4d27a76d51668c20bb5fbdc2c9cf3926964f3d91`, corrected invalid
summary = `60322b97b8ffa56c3528f252a17f91739019fd3c879b19480e87a54224d98a19`  
Validation: both valid files contain all 9,000 query IDs exactly twice, search shard IDs 0 and 1
for every query, retain positive distance/node/CPU/wall counters, reproduce summary means and
measurement recall exactly, and have the required 20-31 affinity. Both temporary collections were
deleted with four-peer HTTP 404 verification and controller headroom returned to 4.0 GB  
C1 status: SUPPORTS C1-c at GloVe M=2; reducing shard size by half reduces valid per-shard
distance work by only 29.3% for Random and 31.2% for K-Means. K-Means provides no fan-out
reduction at this point  
Required follow-up: execute GloVe E3 Random and K-Means M=4 under the new recall gate.

## 2026-08-20T22:47:40Z - Stage 8 GloVe E3 M=4 checkpoint

Timestamp: 2026-08-20T22:47:40Z  
Git commit: `fe61fdac6760bc4d2bcd00cc512218d5fbd7e424`  
Execution scope: four physical machines with one custom shard on each of 10.10.1.1 through
10.10.1.4  
Random partition: 296,026/296,306/295,828/295,354 points; full broadcast fan-out P=4  
Random selection: efSearch=128 reached tuning Recall@10=0.9013 and held-out measurement
Recall@10=0.90142 on queries 1000-9999, so no recall retune was required  
Random local work: 5,065.21 distance computations and 139.75 graph nodes per searched shard;
normalized distance work = 0.3946 versus M=1, compared with ideal 0.25  
K-Means partition anomaly: the artifact reached the 12-iteration cap without convergence and
retained the imbalanced 528,812/185,963/291,179/177,560 split  
K-Means selection: exact ranked-prefix tuning evaluated all 44 P-by-ef candidates with 44,000
shard searches and selected P=3/efSearch=256 at tuning Recall@10=0.9042. Held-out measurement
Recall@10=0.90332 passed the fixed-quality gate, reducing serving fan-out from four to three  
K-Means local work: 9,253.34 distance computations and 267.64 graph nodes per searched shard;
normalized distance work = 0.7208 versus M=1. Shard selection counts were 8,857/6,326/8,074/3,743,
and the independently reconstructed normalized-cosine centroid ranking matched every routed prefix  
Evidence SHA-256: Random prepare/tuning/per-search/summary =
`477f34c12d0269b396dd8bdc39e710818bb11df6877ddcd21ff9116214cfeb86`,
`55a09b29f2ca737a415a459759f289956f9e87bce2b385e01f612a39cce64441`,
`1e756794b838e6e5156dbc2c9cc8041e4aaee98fdc67ee5953e3b1083557801`, and
`250b813d84eaef3a86f9fce2243f30fea0e5091dd6802d4606def7ecc2c73038`; K-Means =
`92136e04057ff56ef6411d493fa5655747da875f6b66ce1695465f18872ce15e`,
`d61af0919f7e51935089b1eda9fd2a226c13f4c2c4c3f78a7db7347b9aad58bd`,
`19e3b7084214e436902a6ab2db356b9d4705196f3312625bd303a505799949b4`, and
`0fede945dc17e4dd41f836b239c3f44492a77df45236f4045a58532cd4b3f0b8`  
Validation: all 63,000 primary rows cover queries 1000-9999 exactly at their selected fan-out,
retain positive distance/node/CPU/wall counters, reproduce summary means and measurement recall,
match intended physical placement and required 20-31 benchmark affinity, and reference the
audited partition hashes. Both temporary collections were deleted with four-peer HTTP 404
verification, their controller storage directories are absent, and disk headroom is 3.7 GB  
C1 status: SUPPORTS C1-c at GloVe M=4. Quartering the balanced Random shard size reduces local
distance work by only 60.5%, while K-Means local work falls by only 27.9% and increases relative
to its M=2 value despite reducing routing fan-out to three shards  
Required follow-up: execute GloVe E3 Random and K-Means M=8 under the same held-out recall gate.

## 2026-08-20T23:06:03Z - Stage 8 GloVe E3 M=8 checkpoint

Timestamp: 2026-08-20T23:06:03Z  
Git commit: `fe61fdac6760bc4d2bcd00cc512218d5fbd7e424`  
Execution scope: eight logical custom shards placed round-robin as two shards on each of four
physical machines  
Random partition: 147,530 to 148,470 points per shard; full broadcast fan-out P=8  
Random selection: efSearch=96 reached tuning Recall@10=0.9100 and held-out measurement
Recall@10=0.91206 on queries 1000-9999  
Random local work: 3,982.80 distance computations and 106.61 graph nodes per searched shard;
normalized distance work = 0.3102 versus M=1, compared with ideal 0.125  
K-Means partition anomaly: the artifact reached the 12-iteration cap without convergence, with
final centroid movement 0.06750 and shard sizes
176,385/100,608/176,261/161,546/256,096/130,553/68,251/113,814  
K-Means selection: exact ranked-prefix tuning evaluated all 88 P-by-ef candidates with 88,000
shard searches and selected P=4/efSearch=256 at tuning Recall@10=0.9004. Despite the narrow
tuning margin, held-out Recall@10=0.90302 passed without retuning  
K-Means local work: 9,142.22 distance computations and 266.11 graph nodes per searched shard;
normalized distance work = 0.7121 versus M=1. Shard selection counts were
5,773/2,680/4,541/6,035/7,402/5,017/1,102/3,450, and independent normalized-cosine ranking matched
every routed four-shard prefix  
Evidence SHA-256: Random prepare/tuning/per-search/summary =
`6aa174a129717005bbc425da992b129f547fd3f6a1f61c62c59b9d9ae7ad6381`,
`8121e306057a4ef12ef972ff40a3eda3a14887856edc48b208f50534d10f756b`,
`7fe1b81da57f51c7ad657ce37640388a2d61900ad9376596235096e9b632cfc8`, and
`ee8b4ea342b328c139202fbe20eb330799bc76c7c22b3f243b7a0f7d3aef7994`; K-Means =
`f04eebd34bdbdac634344bd46e120b34d2ff023d7cae3746094b31515db69643`,
`8befff73916622bff77dfca7f9e513398f9ce1a1d22108fb22fe7199dc0f0d25`,
`43bad345678e576c545031137ed6b8881fff0b476de312f31c26b73cad5e5e13`, and
`e9876b293ceb35cfd7cc1d992e4ac394c125ef16afcaa32514fb7e85c3839bb3`  
Validation: all 108,000 rows cover queries 1000-9999 exactly at their selected fan-out, retain
positive distance/node/CPU/wall counters, reproduce summary means and measurement recall, match
round-robin physical placement and 20-31 benchmark affinity, and reference the audited partition
hashes. Both collections were deleted with four-peer HTTP 404 verification, their controller
directories are absent, and disk headroom is 3.6 GB  
C1 status: SUPPORTS C1-c and the high-fan-out mechanism at GloVe M=8. Random local work remains
2.48 times the ideal normalized curve while broadcasting to every shard; K-Means halves routing
fan-out but retains 71.2% of M=1 local distance work, nearly unchanged from M=4  
Required follow-up: execute GloVe E3 Random and K-Means M=16 under the same held-out recall gate.

## 2026-08-20T23:27:54Z - Stage 8 GloVe E3 M=16 checkpoint

Timestamp: 2026-08-20T23:27:54Z  
Git commit: `fe61fdac6760bc4d2bcd00cc512218d5fbd7e424`  
Execution scope: 16 logical custom shards placed round-robin as four shards on each of four
physical machines; every E3 shard was still searched in isolation  
Random partition: 73,581 to 74,353 points per shard; full broadcast fan-out P=16  
Random selection: efSearch=64 reached tuning Recall@10=0.9099 and held-out measurement
Recall@10=0.91609 on queries 1000-9999  
Random local work: 2,839.48 distance computations and 74.09 graph nodes per searched shard;
normalized distance work = 0.2212 versus M=1, compared with ideal 0.0625  
K-Means partition anomaly: the artifact reached the 12-iteration cap without convergence, with
final centroid movement 0.15173 and shard sizes ranging from 20,789 to 119,678  
K-Means selection: exact ranked-prefix tuning evaluated all 176 P-by-ef candidates with 176,000
shard searches and selected P=9/efSearch=128 at tuning Recall@10=0.9001. Held-out
Recall@10=0.90157 passed the fixed-quality gate without retuning  
K-Means local work: 5,117.84 distance computations and 138.46 graph nodes per searched shard;
normalized distance work = 0.3987 versus M=1. Per-shard selection counts ranged from 498 to
8,139, reflecting both centroid routing and the severe partition imbalance  
Routing audit note: the CSV matches the runner's exact per-query float32 centroid ranking for all
9,000 queries. A batched matrix multiply swaps only ranks 6 and 7 for query 3173 because of
floating-point accumulation order; the selected top-nine shard set is identical, so neither
fan-out nor measured work is affected  
Evidence SHA-256: Random prepare/tuning/per-search/summary =
`54d4a98add9784dc7c4481058ab05db59bc89fd3552e8e6d21539751688f2030`,
`051eb2eca9f0611c4180cc068bfebbefff1451a3e0fd536022e933a83b38d435`,
`d8db958d9ee5a0d90841f8dab6c018a07033bb953f01ce7495e82f7ad6733d9b`, and
`587f6d500a9c8cf456cb7c066105c2a4b21aede487c5d4f7f79ddb3c6e88ce66`; K-Means =
`81803751979c3e45fae12dfd9508b477042d7738a6d4d66f4358147c4e77fd48`,
`5bd1c765a71fed57499d135fa62042a5575729f68fd4c2344c3793723986ddc6`,
`1b250b8fccd8f657458acb87c43e03e30707d454325b2453cd17c8e6caa4cf89`, and
`ade960453e8a9ffbf59ab2834e8f83691388acb5dcc4fd05ce46bcab3113f704`  
Validation: all 225,000 rows cover queries 1000-9999 exactly at their selected fan-out, retain
positive distance/node/CPU/wall counters, reproduce summary means and measurement recall, match
round-robin physical placement and 20-31 benchmark affinity, and reference the audited partition
hashes. Both collections were deleted with four-peer HTTP 404 verification, their controller
directories are absent, and disk headroom is 3.4 GB  
C1 status: SUPPORTS C1-c and high fan-out at GloVe M=16. Random local work remains 3.54 times the
ideal normalized curve while broadcasting to every shard; K-Means still routes to nine shards
and retains 39.9% of M=1 local distance work, 6.38 times the ideal curve  
Required follow-up: execute the final GloVe E3 Random and K-Means M=32 pair under the same gate.

## 2026-08-20T23:47:04Z - Stage 8 GloVe E3 Random M=32 checkpoint

Timestamp: 2026-08-20T23:47:04Z  
Git commit: `fe61fdac6760bc4d2bcd00cc512218d5fbd7e424`  
Execution scope: 32 logical custom shards placed round-robin as eight shards on each of four
physical machines; every E3 shard was searched in isolation and this is not physical M=32
throughput evidence  
Random partition: 36,675 to 37,353 points per shard; partition SHA-256
`27eb4d507a626277b2755c430d33cc816a37e269a19518045a34e70cdce81569`; full broadcast fan-out
P=32  
Random selection: the pinned 11-value grid selected efSearch=48 at tuning Recall@10=0.9244;
efSearch=32 remained below target at 0.8871. Held-out Recall@10=0.92309 passed on queries
1000-9999 without retuning  
Random local work: 2,197.35 distance computations and 57.46 graph nodes per searched shard;
normalized distance work = 0.1712 versus M=1, compared with ideal 0.03125. The observed local
distance work is therefore 5.48 times the ideal normalized curve  
Resource facts: each shard stores 29.34-29.88 MB of vectors; the controller-local shard graph
indexes were 3.64-3.70 MB, below the 16,777,216 B LLC, while remote graph-file sizes remain
unavailable in Qdrant telemetry and are not inferred  
Evidence SHA-256: prepare/tuning/per-search/summary =
`4c8b7ee67909a05239789b1f786236db1775126c2c530970d579bca78a375892`,
`fba185ec4f78c58da6d5140acab86acef11ff0d62174062ddd474022c7fccec5`,
`9844f55af36819d85dd58d5f876e0f35e243ef00b36251af8dbffad9bfc99500`, and
`d046f355ae8173064b1a35c9b17fa30140a7c26e9884cd6038763bd57a0050da`  
Validation: all 288,000 rows cover every query ID 1000-9999 exactly 32 times and every shard
exactly 9,000 times; one-based route ranks cover 1-32 for every query. All result lists contain
ten unique IDs, distance/node/CPU/wall counters are positive, placement and point counts match
the audited artifact, recall and every summary mean reproduce, the benchmark affinity is exactly
cores 20-31, and the manifest contains exactly the 11 expected tuning-candidate records  
Lifecycle: after evidence verification, only `c1v1_e3_glove_random_m32` was deleted. All four
peers return HTTP 404, the controller collection directory is absent, and disk headroom returned
to 3.8 GB  
C1 status: SUPPORTS C1-c and the Random high-fan-out mechanism through M=32. Random still
broadcasts to every shard, while 32-way partitioning reduces local distance work to only 17.1%
of M=1 rather than the ideal 3.125%  
Required follow-up: execute GloVe E3 K-Means M=32 under the same held-out recall and deletion
gates, preserving its 12-iteration non-convergence and severe shard imbalance.

## 2026-08-20T23:58:54Z - Stage 8 GloVe E3 K-Means M=32 and matrix closure

Timestamp: 2026-08-20T23:58:54Z  
Git commit: `fe61fdac6760bc4d2bcd00cc512218d5fbd7e424`  
Execution scope: 32 logical custom shards placed round-robin as eight shards on each of four
physical machines; this is logical-shard simulation and not physical M=32 throughput evidence  
Partition anomaly: K-Means reached its 12-iteration cap without convergence, with final maximum
centroid movement 0.192705. Shard sizes range from 15,843 to 68,153 points, a 4.30x imbalance;
partition SHA-256 is
`79786a469498c64fe6eabe821ec716bf7ac72da621f37afd66d5141783991bd4`  
K-Means selection: exact ranked-prefix reuse evaluated all 352 P-by-ef candidates with 352,000
shard searches and selected P=10/efSearch=128 at tuning Recall@10=0.9005. Held-out
Recall@10=0.90241 passed on queries 1000-9999 without retuning  
K-Means local work: 4,664.53 distance computations and 137.05 graph nodes per searched shard;
normalized distance work = 0.3633 versus M=1, compared with ideal 0.03125. Local distance work is
therefore 11.63 times the ideal normalized curve  
Routing distribution: every query searches exactly ten centroid-ranked shards. Per-shard
selection counts range from 451 to 6,623 of the 9,000 measurement queries, reflecting substantial
route heterogeneity together with the imbalanced partition  
Evidence SHA-256: prepare/tuning/per-search/summary =
`971d06ba18204f3bab3ce8941f124950780dd9b383eb7cecb81f84b756048219`,
`f13aedfddadb650c9834a440a74c7d668f96ecbb5028acf52173a9fb2e01df41`,
`4d08227e1851c84edc9724a70ef6045d36e459dc03e74bb5677dc7ecc8eb0505`, and
`015cee0243fbee176947d4df6b529d8773c1cd732bc2b68ae3b7d29c063a565b`  
Validation: all 90,000 rows cover every query ID 1000-9999 exactly ten times. Independent
per-query float32 normalized-cosine ranking matches every ordered top-ten prefix; all result
lists contain ten unique IDs; distance/node/CPU/wall counters are positive; placement, point
counts, resources, recall, and summary means reproduce; affinity is exactly cores 20-31; and the
manifest contains exactly 352 tuning-candidate records  
Lifecycle: after evidence verification, only `c1v1_e3_glove_kmeans_m32` was deleted. All four
peers return HTTP 404, the controller collection directory is absent, and disk headroom returned
to 3.7 GB  
GloVe E3 matrix conclusion: SUPPORTS C1-c across M=1,2,4,8,16,32. At M=32, Random retains
17.1% and K-Means 36.3% of M=1 local distance work instead of the ideal 3.125%. It also supports
the high actual-fan-out mechanism in C1-b: Random broadcasts to all 32 shards and K-Means still
requires ten. E2 remains required for the per-query oracle fan-out distribution and sensitivity
analysis  
Required follow-up: parameterize and execute GloVe E2 from the complete audited E3 matrix while
preserving corrected SIFT canonical CSV/PDF artifacts before canonical names are reused.

## 2026-08-21T00:06:32Z - Stage 8 GloVe E2 fan-out analysis

Timestamp: 2026-08-21T00:06:32Z  
Run ID: `stage3-e2-glove-200-angular-fanout`  
Input parameterization: E3 sources use `stage2-e3-{dataset}` while both small- and large-M
GloVe tuning sources use `stage1-{dataset}`. The legacy SIFT defaults remain unchanged, and the
focused suite passes 26 tests  
Actual selected fan-out by M=1,2,4,8,16,32: Random = 1,2,4,8,16,32; K-Means =
1,2,3,4,9,10  
Oracle mean fan-out: Random = 1.0000, 1.9774, 3.1453, 4.9291, 6.6109, 7.6996; K-Means =
1.0000, 1.3297, 1.5386, 1.8713, 2.1137, 2.4181  
M=32 heterogeneity: Random oracle median/p95/p99/max = 8/9/9/9 with range 4-9. K-Means oracle
median/p95/p99/max = 2/5/7/9 with range 1-9. Thus K-Means lowers the mean but retains a long
per-query tail and the globally selected fixed-quality fan-out remains ten shards  
Measurement Recall@10: Random M=1/2/4/8/16/32 =
0.91206/0.91547/0.90142/0.91206/0.91609/0.92309; K-Means =
0.91206/0.90440/0.90332/0.90302/0.90157/0.90241  
Sensitivity: at targets 0.89, 0.90, and 0.91, Random remains broadcast at every M. K-Means
selected fan-out at M=32 is 9, 10, and 12 respectively; all monotone actual/oracle trend checks
pass, so small target changes do not reverse the qualitative conclusion  
Evidence SHA-256: per-query/summary CSV/summary JSON =
`a017c4e4caf75b74dd529133c8b37cc842b2beb7b7582dfc3ee4c305c01b1418`,
`15beb8c1e59e923f14645e165ad45a02422a3a438d38a2e8abc8dfda5a23e0f5`, and
`83095ace8a53bb0a5192c85ba32316bc3c420c8bde5cb456fabcba88743ac4dd`  
Figure SHA-256: canonical/protocol fan-out =
`7019463fcd578cc020fea4e9c368ed72afa35495f86b9c9ba01c9f50b3ba3992`; canonical/protocol
oracle CDF = `977a80ab22943f122c2d9876838f375486efa1fc0cecf6a9ab08572edcdde804`;
actual CDF = `47cc6c38f55c9fac0832692c19d0e367d105bae2294f241fc0a3002690dc6278`  
Preservation: the pre-existing SIFT E2 summary and five PDFs were copied under
`stage3-sift1m-*` names and retain their original hashes; the GloVe canonical outputs also have
`stage3-glove-200-angular-*` preservation copies  
Validation: all 108,000 rows cover 9,000 measurement queries for each of 12 configurations;
oracle values independently reproduce from ground-truth shard assignments; actual fan-out,
selected routes, recall, and every source hash match E3/tuning evidence. All five PDFs are
single-page PDF 1.4 files and pass contact-sheet visual inspection. The manifest record is
registered exactly once and byte-equivalent to the audited record JSON  
C1-b status: SUPPORTS across both datasets. Random fan-out stays maximal. K-Means reduces fan-out
but does not make it a consistently small constant: on GloVe it rises from 1 at M=1 to 10 at
M=32, while its per-query oracle tail reaches nine  
Required follow-up: preserve corrected SIFT E1/E4 canonical artifacts, parameterize the remaining
SIFT-specific E1/E4 assumptions, and run corrected multiprocess GloVe E1 followed by E4.

## 2026-08-21T01:06:18Z - Stage 8 corrected GloVe E1 physical scale-out

Timestamp: 2026-08-21T01:06:18Z  
Run ID: `stage8-e1-corrected-glove-200-angular-physical-scaleout`  
Git commit: `fe61fdac6760bc4d2bcd00cc512218d5fbd7e424` (dirty experiment worktree)  
Physical scope: five unique layouts, with the shared unsharded M=1 baseline reused by Random and
K-Means  
Recall target: 0.90; all valid repetitions pass  
Random mean QPS at M=1/2/4: 3857.67 / 4802.12 / 4197.43; scaling efficiency:
1.000 / 0.622 / 0.272  
K-Means mean QPS at M=1/2/4: 3857.67 / 4563.22 / 4385.88; scaling efficiency:
1.000 / 0.591 / 0.284  
Selected concurrency: M=1 c=64, Random M=2 c=32, Random M=4 c=16, K-Means M=2/M=4
c=64. Random M=4 retains the automatic c=32 selection and the explicit override reason because
all original formal repetitions exceeded the 85% total-client CPU gate  
Validation: all 15 valid repetitions have 18,000 rows, exact cores 20-31 benchmark affinity,
no bottleneck flags, no persistent queue growth, matching raw hashes, and QPS CV below 2%  
Lifecycle: `c1v1_e1_glove_kmeans_m4` was deleted; all four peers return HTTP 404 and the
controller storage directory is absent  
Evidence SHA-256: aggregate CSV =
`73e95e2f25f878bb630df0dc7fd99fb7bc416c669fb07c25c91e8f3ee887f74b`; physical figure =
`b499ff49321c499c52287e48377f8f335034aad7ec6f5925db94532fc17f1274`; efficiency figure =
`a43eef2b3191ecfe42170db5913529dcf1c0fafb189a6648e8fa6764e2475367`  
C1-a status: SUPPORTED for GloVe. Throughput is far below ideal 2x/4x scale-out for both
partition methods, while every protocol bottleneck gate remains clear  
Required follow-up: run GloVe E4 and apply the corrected SIFT E4 result as a cross-dataset
projection-release gate.

## 2026-08-21T01:13:37Z - Stage 9 GloVe E4 and cross-dataset projection gate

Timestamp: 2026-08-21T01:13:37Z  
Run ID: `stage9-e4-glove-200-angular-aggregate-work`  
Configurations: Random and K-Means at M=1,2,4,8,16,32; 108,000 per-query rows  
Aggregate distance work normalized to M=1 at M=32: Random = 5.4773; K-Means = 3.6335  
Decomposition accuracy: Pearson/Spearman = 1.0/1.0 for both methods; maximum relative error is
0 for Random and `1.56e-16` for K-Means, supporting C1-d  
GloVe-local projection sanity: Random Pearson/Spearman = 0.1039/0.5; K-Means = 0.7468/0.5.
Both local trends pass, so GloVe alone would make the work-bound curve available  
Cross-dataset release: the corrected SIFT Stage 7 record is an explicit external gate. SIFT
Random remains anticorrelated, so the global status is
`WITHHELD_PHYSICAL_SANITY_CHECK_FAILED`; all M>4 projection values are blank and the projection
figure shows only the M=1,2,4 sanity comparison  
Evidence SHA-256: per-query =
`ffaf707586024b2946721797b3649e04c947f9726bae9cdfc8ac1344a9944b59`; aggregate CSV =
`5c6e2768c3db8fd071986c38a43cee5abca0d9ee8bf267fd11a2863f2284e548`; model CSV =
`6564cfb4a53ec1b1a890062dd427b34522dd76221116a2eccfff041a2d1605cb`; summary JSON =
`e54b5c76cc4bbc35486719b7380f4fef8ebe11c40a70b56e31a48fe4c7db0e45`  
C1-d status: SUPPORTED. Physical attribution beyond four workers remains INSUFFICIENT  
Required follow-up: complete E5 and the final cross-dataset artifact/verdict audit without
publishing any M>4 throughput projection.

## 2026-08-21T01:23:28Z - Stage 10 final cross-dataset verdict

Timestamp: 2026-08-21T01:23:28Z  
Run ID: `stage10-c1-final-cross-dataset`  
Final matrix sizes: E1 = 12 rows; E2 = 24; E3 = 24; E4 = 24; model accuracy = 4; E5 = 24  
E5: existing M=4/M=16 tuning neighborhoods were evaluated at recall targets 0.89/0.90/0.91.
Across both datasets and methods, M16/M4 local-work ratios range from 0.4545 to 0.8474 versus
the ideal 0.25; fan-out and aggregate work are non-decreasing, so the C1 mechanism is stable near
the headline threshold  
Final artifacts: all five required cross-dataset CSVs, `c1_physical_scale_table.tex`, six numbered
PDFs, and `c1_combined_motivation.pdf` pass row-count, source-hash, output-hash, PDF-format, and
contact-sheet visual audits. The focused suite passes 32 tests, and the final manifest record is
registered exactly once  
Projection boundary: M=1,2,4 are measured physical scale-out. M=8,16,32 are logical-shard
mechanism measurements. Every M>4 throughput projection is withheld because the cross-dataset
physical sanity gate fails  

C1-a Physical sub-linear scaling: SUPPORTED  
C1-b High logical shard fan-out: SUPPORTED  
C1-c Slow decrease in local graph-search work: SUPPORTED  
C1-d Fan-out x local-work decomposition: SUPPORTED  
Complete C1 claim: SUPPORTED  

Paper-ready claim: At 90% Recall@10, conventional scatter-gather HNSW exhibits sub-linear
throughput scaling from one to four physical workers on both SIFT1M and glove-200-angular.
Increasing the logical shard count does not proportionally reduce per-shard graph-search work,
while queries continue to access multiple shards. Consequently, aggregate graph-search work per
query remains high as the index is further partitioned. A decomposition based on shard fan-out
and per-shard distance computations exactly tracks the observed aggregate work.  
Evidence SHA-256: final record =
`258c43b71e680f9098630a05578fb7f9976f882e63333cbbafa590c2b885e399`; combined physical CSV =
`9d04bb0940340e00774e4a19461929951cc34ea0ba773a558d0c3e5f96ba5e3b`; combined aggregate CSV =
`90ea87e165efd23ce79d0e596829ba9064ac4bc2ec7e500c6f9ecb4fe4e0e0ea`; E5 CSV =
`8c94c6c94d4347134fd565f608fcacb8bf2b8c2053769b6edd256c1b3b25066a`; combined figure =
`8cef677741cafcf8f60d70d7a3492b2a039bd867a7d34c120dd448acf21a67a1`  
Required follow-up: none. Preserve raw results locally and use explicit allowlist staging for any
future Git delivery.

## 2026-08-21T01:34:27Z - Stage 10 completion-audit correction

Timestamp: 2026-08-21T01:34:27Z  
Git commit: `fe61fdac6760bc4d2bcd00cc512218d5fbd7e424` (dirty experiment worktree)  
Audit scope: every explicit requirement in Sections 1-37 of the authoritative protocol, rather
than only the previously generated final-record checks  
Contradiction 1: SIFT K-Means actual fan-out at M=1/2/4/8/16/32 is 1/1/1/2/3/3. This is a
nearly constant small fan-out at large M. Section 33 therefore requires C1-b for K-Means to be
`CONTRADICTED`; the Random result remains separately supported, and the claim cannot be
generalized to spatial partitioning  
Contradiction 2: corrected SIFT Random work-bound scaling has Pearson/Spearman -0.8291/-1.0
against measured E1 at M=1,2,4. Although fan-out x local work exactly reproduces observed
aggregate work, the required connection from that work trend to physical throughput is not
established. C1-d physical attribution is `INSUFFICIENT`  
Statistical gap: production segment optimization constructs HNSW with `rand::rng()` at
`lib/shard/src/optimize.rs:289`. A deterministic graph-build seed is not guaranteed, while the
current E2-E4 evidence uses one index instance per configuration. Section 31 requires either a
deterministic construction guarantee or at least three builds with variation reported  
Artifact gaps: the protocol-specific E3 distance/nodes/CPU figures do not exist; the required
distance plot lacks the normalized log(N/M) reference; the final E1 efficiency plot is GloVe-only;
the final oracle CDF shows only M=32 rather than representative M=4/16/32; and the final aggregate
plot lacks the ideal 1/M reference  
Corrected current status: C1-a `SUPPORTED`; generalized C1-b `CONTRADICTED`; C1-c `SUPPORTED`;
C1-d physical attribution `INSUFFICIENT`; complete C1 `INSUFFICIENT`  
Anomalies: the earlier Stage 10 final record and `SUPPORTED` verdict are retained as historical
evidence but are superseded by this audit correction  
Required follow-up: repair all missing figures, add a requirement-level verifier, establish
deterministic HNSW construction or repeat E2-E4 over at least three builds, then regenerate the
final decision without releasing M>4 throughput projections.

## 2026-08-21T03:00:54Z - Stage 12 deterministic construction and rerun checkpoint

Timestamp: 2026-08-21T03:00:54Z  
Git commit: `fe61fdac6760bc4d2bcd00cc512218d5fbd7e424` (dirty experiment worktree)  
Construction correction: production HNSW optimization now accepts
`QDRANT_HNSW_GRAPH_BUILD_SEED=20260821`; formal E2-E4 collections use one indexing thread and
one optimizer thread. The exact image `orion-c1:20260821-deterministic-seed` is deployed on all
four peers with image ID
`sha256:54819a6648dc706f654f3e541351c68814c0c1798312149c5e3ee099c183bfc8`  
Independent-build proof: two separately created 10,000-point HNSW indexes have identical
serialized graph content. Canonical graph-content SHA-256 is
`4ac14807fc025d64a58ba9403deff3d47db2a9c9b3c220be76de40d5f05668b8`;
`graph.bin` and `links_compressed.bin` independently match at
`a85902ea8e2ea03e36e0f761bda674083a57ddc1d68a4b75f69337990ac09955` and
`2eac270993c01c09a0cd0034b1ca130fde7949dd370a4a585829af62676dac24`  
Image preservation: the retained image tar SHA-256 is
`1ccab8006db7f1f580c9f5ec1afa54f16b95e9baf7a6ea674056dae0c3eeead7`  
Rerun progress: five of 24 authoritative E3 configurations are complete and valid: both SIFT
M=1 methods, both SIFT M=2 methods, and SIFT Random M=4. SIFT K-Means M=4 is active. Each
configuration uses the official disjoint 1,000-query tuning and 9,000-query measurement sets,
fixed benchmark affinity on cores 20-31, deterministic round-robin placement, and collection
deletion only after its configuration group validates  
Downstream contract: the resumable driver will derive two deterministic E2 records, execute
preliminary SIFT E4, GloVe E4 gated by SIFT, and final corrected SIFT E4 gated by GloVe, emit 36
normalized Section 32 rows, verify the final collection is absent on all peers and controller
storage, complete the Section 31 proof, preserve Stage 11 canonical artifacts, and create a new
hash-linked Stage 12 final record  
Validation: 49 focused tests pass. The strict verifier currently reports 14/22 PASS; all eight
remaining failures are expected absent Stage 12 proof/metadata/cleanup/final-record artifacts,
not failures in completed evidence  
Evidence SHA-256: current Section 31 proof =
`1129b31dae7e65d84dafcf7a58a8e75ca1884a199300793e5f542339693a3a03`;
E3 driver = `151459e3a595d2052bbee553fa56ab98b4a2ed5282ebfc2a525230cc81d7ae34`;
downstream driver = `d132a9cf205db88bba590f0c92f3dd0181d8672ddbd468d9009e8a22516910ca`;
completion audit = `80b768d44ae194700d3c2221acf0495b6696b384999ecfe65e1b22d00d4c0c16`;
Stage 12 finalizer = `571ac4113443882d4039d055ded4a82c541d6e613277c3146e4cde5d75b148c3`  
Current scientific boundary: C1-a remains `SUPPORTED`; generalized C1-b remains
`CONTRADICTED`; deterministic completion is still pending for C1-c; physical C1-d attribution
and complete C1 remain `INSUFFICIENT`. No M>4 throughput projection is released  
Required follow-up: finish all 24 E3 configurations, run the deterministic downstream chain,
regenerate final reports, and require the completion audit to report 22/22 PASS.

## 2026-08-21T06:10:14Z - Stage 12 deterministic completion and strict protocol audit

Timestamp: 2026-08-21T06:10:14Z  
Run ID: `stage12-c1-deterministic-final`  
Git commit: `fe61fdac6760bc4d2bcd00cc512218d5fbd7e424` (dirty experiment worktree)  
Deterministic scope: all 24 SIFT1M and GloVe Random/K-Means E3 configurations at logical
M=1,2,4,8,16,32 completed with graph seed 20260821, one indexing thread, one optimizer thread,
fixed benchmark affinity, deterministic round-robin placement, held-out Recall@10 >= 0.90, and
verified collection cleanup. The initial GloVe K-Means M=2 efSearch=128 held-out recall miss is
preserved as `INVALID_RECALL`; the canonical configuration was retuned to efSearch=192 and passed  
Section 31: two independent serialized HNSW builds have identical graph content, and the complete
deterministic E2-E4 rerun is authoritative. The matrix contains 24 E2 fan-out rows, 24 E3
local-work rows, 24 E4 aggregate-work rows, 36 normalized Section 32 metadata rows, and 24 valid
E5 sensitivity rows  
Mechanism result: C1-c is `CONTRADICTED`. Of 40 non-baseline distance/node comparisons, 34 are
above the ideal 1/M decrease and six are at or below it, including GloVe Random at M=2 and M=4.
The measured evidence therefore does not support the generalized slow-local-work subclaim  
E5 result: `CONTRADICTED_QUALITATIVE_TREND_REVERSAL`. For SIFT1M K-Means at Recall@10 target
0.89, M=4 selects fan-out 1 with local/aggregate distance work 2020.316/2020.316, while M=16
selects fan-out 2 with local/aggregate work 959.354/1918.708. Local work falls slowly relative to
the ideal ratio (observed M16/M4 ratio 0.4749 versus 0.25), but aggregate work decreases, reversing
the qualitative trend. The contradictory raw result is retained  
Final scientific status: C1-a `SUPPORTED`; generalized C1-b `CONTRADICTED`; C1-c
`CONTRADICTED`; C1-d physical attribution `INSUFFICIENT`; complete C1 `INSUFFICIENT`.
Fan-out times local work still exactly reproduces observed aggregate work, but the physical
throughput attribution does not pass the cross-dataset release gate. M=1,2,4 remain measured
physical scale-out; M=8,16,32 remain logical-shard mechanism measurements only; every M>4
throughput projection remains withheld  
Lifecycle and provenance: the final experiment collection is absent on all four peers and from
controller storage. Stage 10 and Stage 11 records, figures, raw evidence, and invalid runs remain
preserved. This Stage 12 record supersedes their C1-c `SUPPORTED` conclusion and the historical
stable-E5 wording  
Evidence SHA-256: final record =
`c2cc7f0f20299636593320327ddf81010bc71fb5645d3b7148cf375edfdaf2c0`; final summary =
`111fb56140bdf2306a9ca9d8263583a383b5888d40e0204d799c81ca109ddd69`; E3 matrix =
`7d3395c437b0c8355b495b60937024f2eeec1be0a7aefd7cd6d5672195ad5252`; Section 31 proof =
`ccea5798cd53722032773fb77ee80208b236f6af891c0743e985949cd9eabb32`; normalized metadata =
`ebe881be3bb1b72953bb7e63c8a74193528e2fe997e10c96a5931319a066c960`; final cleanup proof =
`b31f17f48edf5bede8bec278fa66fc0eceb0e112f5ea1edf3edcd99813de8495`  
Validation: the strict requirement-level completion audit reports 22/22 PASS  
Required follow-up: none for protocol completion. Preserve raw results locally; any future Git
delivery remains separate and must use explicit allowlist staging.

## 2026-08-21T06:34:00Z - Stage 12 independent completion-audit hardening

Timestamp: 2026-08-21T06:34:00Z  
Git commit: `fe61fdac6760bc4d2bcd00cc512218d5fbd7e424` (dirty experiment worktree)  
Audit scope: independently remapped all Sections 1-37 of the authoritative 1,151-line protocol
to current evidence. `experiments/c1/PLAN.md` matches the source plan modulo its final newline  
Strengthened verification: replayed the protocol selection rule for all 36 tuning artifacts and
confirmed all 1,716 candidates are retained verbatim in `manifest.jsonl`; validated deterministic
round-robin placement, distinct physical hosts, fixed CPU affinity, all 30 E1 repetitions and raw
query schemas, all 24 E3 configurations and 252 shard-resource records, and every recorded source
hash. Recomputed the 24 E2 distributions/recalls from 216,000 per-query rows and the 24 E4
distance/node/CPU means from another 216,000 rows. Revalidated the Section 31 build/file hashes,
the preserved `INVALID_RECALL` run and retuned canonical point, final figure/table contracts,
cleanup proof, final statuses, and append-only supersession chain  
Visual audit: all 18 required PDFs are single-page and legible with the required dataset panels,
logical/physical shard ranges, ideal/reference curves, decomposition comparison, and explicit
withheld-projection labels  
Validation: 42 focused tests pass; all C1 Python scripts compile; `git diff --check` passes; the
strengthened `c1_completion_audit.py` reports 22/22 PASS with zero failures. A fresh local Rust
compile is unavailable because this shell has no Cargo toolchain; executable validity is instead
established by the retained deterministic image identity, successful four-peer deployment, two
identical independent HNSW builds, and the completed live measurement matrix  
Scientific conclusion: unchanged. C1-a `SUPPORTED`; generalized C1-b `CONTRADICTED`; C1-c
`CONTRADICTED`; C1-d physical attribution `INSUFFICIENT`; complete C1 `INSUFFICIENT`; E5
`CONTRADICTED_QUALITATIVE_TREND_REVERSAL`; every M>4 throughput projection remains withheld  
Required follow-up: none for C1 protocol completion. No Git staging, push, or publication was
performed.
