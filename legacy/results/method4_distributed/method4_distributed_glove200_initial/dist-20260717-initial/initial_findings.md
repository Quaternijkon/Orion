# Orion four-node initial Recall-QPS findings

Run id: `dist-20260717-initial`  
Confirmation run id: `dist-20260717-initial-confirm`  
Qdrant commit: `1a5ac4c47237b9224ae3e4ca28c2cefb2b514352`  
Image ID: `sha256:219d98992e52e2a06d9a5692e601669aff3cf1f8ffd049a077f662c1c4df29f0`  
Dataset SHA-256: `4839085e5a8bb293434a1a66e1aa0193afc3f07c6797a85f1dbd91656172da20`

## Cluster acceptance

- The controller and three workers formed exactly four peers on
  `10.10.1.1/.2/.3/.4:6335`.
- Every formal collection had 46 Active remote shards, zero controller-local
  lower shards, replication factor 1, no shard transfers, and round-robin
  placement `16/15/15`.
- At the final health check all collections were green, fully indexed, and had
  optimizer status `ok`; Raft had zero pending operations and no peer send
  failures.
- Simple KMeans and Naive each stored 1,183,514 points. Orion stored 1,401,454
  assignments, for an expansion ratio of 1.18415.
- The benchmark process affinity was `8-19`; controller Qdrant used `0-7`.

## First-stage scan

The scan produced all 129 requested parameter points: 80 Orion, 42 Simple
KMeans, and 7 Naive. Maximum observed tuning Recall@10 was 0.9628 for Orion,
0.9726 for Simple KMeans, and 0.9778 for Naive. These maxima are not
same-recall comparisons.

The selected frontier points were:

| Method | Target | Tuning parameters | Tuning recall | Tuning QPS | Selection status |
|---|---:|---|---:|---:|---|
| Orion | 0.90 | upper_k 60, base EF 40, factor 6 | 0.9022 | 864.16 | strict |
| Simple KMeans | 0.90 | nprobe 16, fixed EF 120 | 0.8978 | 992.57 | nearest |
| Naive | 0.90 | all 46 shards, fixed EF 32 | 0.8886 | 573.52 | nearest |
| Orion | 0.95 | upper_k 120, base EF 80, factor 10 | 0.9514 | 573.10 | strict |
| Simple KMeans | 0.95 | nprobe 32, fixed EF 160 | 0.9488 | 543.47 | nearest |
| Naive | 0.95 | all 46 shards, fixed EF 64 | 0.9494 | 496.11 | nearest |

## 3,000-query confirmation

Recall and QPS are stability means across three repeated runs. Latency columns
are end-to-end HTTP batch latencies for batches of 100 queries, not individual
server-side query latencies.

| Method | Target | Parameters | Recall mean | QPS mean +/- sd | P50 / P95 / P99 ms | Logical shards | Physical peers | EF sum | Expansion | Confirmed status |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| Orion | 0.90 | upper_k 60, base 40, factor 6 | 0.89693 | 865.41 +/- 14.86 | 89.90 / 97.72 / 98.44 | 14.411 | 2.874 | 970.9 | 1.184 | nearest |
| Simple KMeans | 0.90 | nprobe 16, EF 120 | 0.89610 | 1009.77 +/- 1.38 | 92.18 / 96.38 / 98.50 | 16.000 | 2.999 | 1920.0 | 1.000 | nearest |
| Naive | 0.90 | 46 shards, EF 32 | 0.88783 | 571.36 +/- 3.40 | 175.56 / 184.28 / 187.57 | 46.000 | 3.000 | 1472.0 | 1.000 | nearest |
| Orion | 0.95 | upper_k 120, base 80, factor 10 | 0.95090 | 586.56 +/- 3.73 | 134.81 / 145.73 / 146.28 | 20.977 | 2.958 | 2998.4 | 1.184 | strict |
| Simple KMeans | 0.95 | nprobe 32, EF 160 | 0.94733 | 545.18 +/- 0.22 | 175.46 / 183.92 / 184.25 | 32.000 | 3.000 | 5120.0 | 1.000 | nearest |
| Naive | 0.95 | 46 shards, EF 64 | 0.94353 | 490.12 +/- 4.07 | 203.75 / 212.69 / 213.15 | 46.000 | 3.000 | 2944.0 | 1.000 | nearest |

The confirmed 0.90 Orion and Simple KMeans points differ in recall by only
0.00083, so they are a valid cross-method same-recall comparison even though
both missed the nominal target. Simple KMeans was 16.7% faster than Orion
(equivalently, Orion QPS was 14.3% lower). Orion visited 9.9% fewer logical
shards and used roughly half the estimated EF sum, but this did not translate
into higher QPS. P95 latency was 1.4% higher for Orion; P99 was effectively tied.

At the 0.95 target only Orion confirmed strict target recall. The Orion-Simple
recall gap was 0.00357 and the Orion-Naive gap was 0.00737, both outside the
fixed 0.003 comparison window. Therefore the observed Orion QPS advantages at
those selected parameters must not be reported as strict same-recall wins.

## Conservative conclusion

This initial run does not support a claim that Orion outperforms Simple KMeans.
At the only confirmed cross-method pair inside the 0.003 recall window, Simple
KMeans had higher QPS. Orion substantially reduced routing work relative to
Naive and had materially lower latency at the selected points, but the Naive
confirmation points were too far away in recall for a strict same-recall speedup
claim. A denser parameter sweep around 0.90 and 0.95 is needed before making a
same-recall Orion-versus-Naive conclusion or a 0.95 Orion-versus-Simple
conclusion.

During collection construction, workers logged repeated benign consensus
warnings that an already-Active replica was expected to be Initializing. All
warnings predated the timed confirmation phase; no warnings or errors appeared
after confirmation began, and final cluster/collection health was clean.
