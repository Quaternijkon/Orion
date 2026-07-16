# 2026-06-29 Method4 Multi-Assignment Expansion vs QPS

## Goal

验证在 indexed-vector ratio 不超过 2.0x 的条件下，Orion 和 Simple KMeans 通过
多分配提高索引膨胀率后，在目标 Recall@10 为 0.80、0.85、0.90、0.95 时 QPS 和
访问 shard 数如何变化。

本实验只把索引膨胀率作为自变量扫描 Orion 与 Simple KMeans：

- Orion: 调整 voting multi-assignment 策略，例如只分配最高票 shard、分配最高票
  ties、分配票数不低于 `max_vote - delta` 的 shard，并用 `max_shards` 限制上限。
- Simple KMeans: 调整 `alpha`，把点复制到所有满足
  `distance <= alpha * nearest_distance` 的 KMeans centroid shard。
- Naive all-shards: 作为标准在线对照在主计划中保留，但它不是本轮可调多分配曲线；
  naive 固定每点一个 copy，访问全部 46 logical shards。

## Setup

| Item | Value |
|---|---|
| Date | 2026-06-29 |
| Controller | `http://localhost:6833` |
| Dataset | `/home/taig/dry/faiss/datasets/glove-200-angular.hdf5` |
| top_k | 10 |
| lower HNSW | `m=32`, `ef_construct=100` |
| upper HNSW | `m=32`, `ef_construction=100`, `upper_search_ef=400` |
| tuning queries | 500 |
| final eval queries | 3000 |
| batch_size | 100 |
| repeats | `stability_repeats=0` |
| dispatch | `coordinator` |
| lower execution order | `query_major` |
| placement | `round_robin`, `qdrant_shard_` workers only |
| runtime workaround | `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6` |

Selection rule: harness tuning selects the fastest tuning point whose tuning
recall meets `target_recall`. Final 3000-query recall can drift slightly below
the target; tables below report the actual final recall.

## Variants

| Family | Strategy | Main knobs | Actual expansion |
|---|---|---|---:|
| Orion | `orion_single` | `--disable-multi-assign` | 1.000x |
| Orion | `orion_default` | top ties, `min_max_vote=2`, `vote_delta=0`, no cap | 1.185x |
| Orion | `orion_w2c2` | `vote_delta=2`, `max_shards=2` | 1.499x |
| Orion | `orion_w2c3` | `vote_delta=2`, `max_shards=3` | 1.852x |
| Simple KMeans | `simple_a1.000` | `alpha=1.000` | 1.000x |
| Simple KMeans | `simple_a1.004` | `alpha=1.004` | 1.191x |
| Simple KMeans | `simple_a1.010` | `alpha=1.010` | 1.566x |
| Simple KMeans | `simple_a1.014` | `alpha=1.014` | 1.887x |

All variants are below the 2.0x expansion limit.

## QPS vs Expansion

### Orion

| Strategy | Expansion | QPS@0.80 | QPS@0.85 | QPS@0.90 | QPS@0.95 |
|---|---:|---:|---:|---:|---:|
| `orion_single` | 1.000 | 499.8 | 372.7 | 322.5 | 219.1 |
| `orion_default` | 1.185 | 1127.4 | 943.1 | 755.7 | 409.1 |
| `orion_w2c2` | 1.499 | 1069.8 | 915.1 | 632.6 | 458.0 |
| `orion_w2c3` | 1.852 | 1001.9 | 865.4 | 591.2 | 385.2 |

### Simple KMeans

| Strategy | Expansion | QPS@0.80 | QPS@0.85 | QPS@0.90 | QPS@0.95 |
|---|---:|---:|---:|---:|---:|
| `simple_a1.000` | 1.000 | 845.6 | 617.8 | 478.6 | 223.1 |
| `simple_a1.004` | 1.191 | 700.0 | 618.8 | 429.2 | 219.5 |
| `simple_a1.010` | 1.566 | 726.1 | 567.4 | 409.9 | 199.0 |
| `simple_a1.014` | 1.887 | 706.7 | 539.1 | 400.8 | 220.0 |

## Visited Shards vs Expansion

### Orion

| Strategy | Expansion | Visited@0.80 | Visited@0.85 | Visited@0.90 | Visited@0.95 |
|---|---:|---:|---:|---:|---:|
| `orion_single` | 1.000 | 15.32 | 25.50 | 28.12 | 30.15 |
| `orion_default` | 1.185 | 6.34 | 7.33 | 11.40 | 21.14 |
| `orion_w2c2` | 1.499 | 8.11 | 8.12 | 14.39 | 21.30 |
| `orion_w2c3` | 1.852 | 7.86 | 9.43 | 16.66 | 24.48 |

### Simple KMeans

| Strategy | Expansion | Visited@0.80 | Visited@0.85 | Visited@0.90 | Visited@0.95 |
|---|---:|---:|---:|---:|---:|
| `simple_a1.000` | 1.000 | 10.00 | 16.00 | 20.00 | 24.00 |
| `simple_a1.004` | 1.191 | 8.00 | 12.00 | 20.00 | 24.00 |
| `simple_a1.010` | 1.566 | 8.00 | 12.00 | 20.00 | 24.00 |
| `simple_a1.014` | 1.887 | 8.00 | 12.00 | 20.00 | 24.00 |

## Detailed Results

| Family | Strategy | Target | Final recall | QPS | Visited | EF/shard | EF-sum | Expansion | Params |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| Orion | `orion_single` | 0.80 | 0.8012 | 499.8 | 15.32 | 203.5 | 3119 | 1.000 | `upper_k=80, base=120, factor=16` |
| Orion | `orion_single` | 0.85 | 0.8468 | 372.7 | 25.50 | 160.9 | 4104 | 1.000 | `upper_k=240, base=48, factor=12` |
| Orion | `orion_single` | 0.90 | 0.9496 | 322.5 | 28.12 | 139.0 | 3910 | 1.000 | `upper_k=320, base=48, factor=8` |
| Orion | `orion_single` | 0.95 | 0.9711 | 219.1 | 30.15 | 279.2 | 8418 | 1.000 | `upper_k=400, base=120, factor=12` |
| Orion | `orion_default` | 0.80 | 0.7963 | 1127.4 | 6.34 | 38.0 | 241 | 1.185 | `upper_k=16, base=16, factor=8` |
| Orion | `orion_default` | 0.85 | 0.8459 | 943.1 | 7.33 | 71.8 | 526 | 1.185 | `upper_k=20, base=48, factor=8` |
| Orion | `orion_default` | 0.90 | 0.8963 | 755.7 | 11.40 | 90.7 | 1034 | 1.185 | `upper_k=40, base=60, factor=8` |
| Orion | `orion_default` | 0.95 | 0.9479 | 409.1 | 21.14 | 132.5 | 2802 | 1.185 | `upper_k=120, base=70, factor=10` |
| Orion | `orion_w2c2` | 0.80 | 0.7988 | 1069.8 | 8.11 | 26.5 | 215 | 1.499 | `upper_k=16, base=16, factor=4` |
| Orion | `orion_w2c2` | 0.85 | 0.8493 | 915.1 | 8.12 | 52.9 | 430 | 1.499 | `upper_k=16, base=32, factor=8` |
| Orion | `orion_w2c2` | 0.90 | 0.9124 | 632.6 | 14.39 | 69.7 | 1004 | 1.499 | `upper_k=40, base=40, factor=8` |
| Orion | `orion_w2c2` | 0.95 | 0.9501 | 458.0 | 21.30 | 100.6 | 2142 | 1.499 | `upper_k=80, base=50, factor=10` |
| Orion | `orion_w2c3` | 0.80 | 0.8065 | 1001.9 | 7.86 | 26.2 | 206 | 1.852 | `upper_k=12, base=8, factor=8` |
| Orion | `orion_w2c3` | 0.85 | 0.8510 | 865.4 | 9.43 | 42.2 | 397 | 1.852 | `upper_k=16, base=32, factor=4` |
| Orion | `orion_w2c3` | 0.90 | 0.9282 | 591.2 | 16.66 | 69.0 | 1150 | 1.852 | `upper_k=40, base=40, factor=8` |
| Orion | `orion_w2c3` | 0.95 | 0.9618 | 385.2 | 24.48 | 100.0 | 2449 | 1.852 | `upper_k=80, base=50, factor=10` |
| Simple KMeans | `simple_a1.000` | 0.80 | 0.8164 | 845.6 | 10.00 | 64.0 | 640 | 1.000 | `nprobe=10, ef=64` |
| Simple KMeans | `simple_a1.000` | 0.85 | 0.8526 | 617.8 | 16.00 | 64.0 | 1024 | 1.000 | `nprobe=16, ef=64` |
| Simple KMeans | `simple_a1.000` | 0.90 | 0.8971 | 478.6 | 20.00 | 96.0 | 1920 | 1.000 | `nprobe=20, ef=96` |
| Simple KMeans | `simple_a1.000` | 0.95 | 0.9532 | 223.1 | 24.00 | 240.0 | 5760 | 1.000 | `nprobe=24, ef=240` |
| Simple KMeans | `simple_a1.004` | 0.80 | 0.8126 | 700.0 | 8.00 | 64.0 | 512 | 1.191 | `nprobe=8, ef=64` |
| Simple KMeans | `simple_a1.004` | 0.85 | 0.8622 | 618.8 | 12.00 | 80.0 | 960 | 1.191 | `nprobe=12, ef=80` |
| Simple KMeans | `simple_a1.004` | 0.90 | 0.8943 | 429.2 | 20.00 | 80.0 | 1600 | 1.191 | `nprobe=20, ef=80` |
| Simple KMeans | `simple_a1.004` | 0.95 | 0.9525 | 219.5 | 24.00 | 200.0 | 4800 | 1.191 | `nprobe=24, ef=200` |
| Simple KMeans | `simple_a1.010` | 0.80 | 0.8308 | 726.1 | 8.00 | 64.0 | 512 | 1.566 | `nprobe=8, ef=64` |
| Simple KMeans | `simple_a1.010` | 0.85 | 0.8597 | 567.4 | 12.00 | 64.0 | 768 | 1.566 | `nprobe=12, ef=64` |
| Simple KMeans | `simple_a1.010` | 0.90 | 0.9047 | 409.9 | 20.00 | 80.0 | 1600 | 1.566 | `nprobe=20, ef=80` |
| Simple KMeans | `simple_a1.010` | 0.95 | 0.9577 | 199.0 | 24.00 | 200.0 | 4800 | 1.566 | `nprobe=24, ef=200` |
| Simple KMeans | `simple_a1.014` | 0.80 | 0.8190 | 706.7 | 8.00 | 48.0 | 384 | 1.887 | `nprobe=8, ef=48` |
| Simple KMeans | `simple_a1.014` | 0.85 | 0.8850 | 539.1 | 12.00 | 80.0 | 960 | 1.887 | `nprobe=12, ef=80` |
| Simple KMeans | `simple_a1.014` | 0.90 | 0.9104 | 400.8 | 20.00 | 80.0 | 1600 | 1.887 | `nprobe=20, ef=80` |
| Simple KMeans | `simple_a1.014` | 0.95 | 0.9533 | 220.0 | 24.00 | 160.0 | 3840 | 1.887 | `nprobe=24, ef=160` |

## Cross-Family Comparison

At similar expansion, Orion is consistently better than Simple KMeans except
for the single-assignment Orion point, which is not the intended multi-assign
Method4 operating point.

| Expansion neighborhood | Orion strategy | Simple strategy | Main observation |
|---|---|---|---|
| ~1.19x | `orion_default` | `simple_a1.004` | Orion has higher QPS at all four targets and visits fewer shards: 6.34/7.33/11.40/21.14 vs 8/12/20/24. |
| ~1.5x | `orion_w2c2` | `simple_a1.010` | Orion QPS is 1.47x to 2.30x higher depending on target, while visiting fewer or equal shards. |
| ~1.85x | `orion_w2c3` | `simple_a1.014` | Orion QPS remains higher at all targets. At 0.95 it visits about the same number of shards, but dynamic EF uses lower EF/shard. |

## Interpretation

Orion:

- Moving from no multi-assignment to modest multi-assignment is decisive.
  `orion_default` at 1.185x is 2.26x / 2.53x / 2.34x / 1.87x faster than
  `orion_single` at the four target buckets.
- Moderate expansion is the useful region. `orion_default` is best for
  0.80, 0.85, and 0.90 buckets; `orion_w2c2` is best for 0.95.
- More aggressive expansion to 1.852x raises recall headroom, but QPS falls
  relative to 1.185x or 1.499x because larger indexes and more duplicated
  candidates start to dominate.

Simple KMeans:

- Increasing `alpha` reduces visited shards at low targets, but QPS does not
  improve monotonically.
- The best Simple KMeans QPS is usually the no-expansion point or very close to
  it. The extra copies reduce `nprobe` in some buckets, but they also enlarge
  the searched index and require source-id de-duplication.
- This suggests Simple KMeans distance-band multi-assignment is a weak
  space-for-speed tradeoff on this dataset.

Main answer:

```text
Under <=2.0x index expansion, Orion benefits strongly from moderate
multi-assignment. Simple KMeans does not show the same QPS-vs-expansion gain.
At comparable expansion ratios, Orion visits fewer useful shards and achieves
higher QPS than Simple KMeans across the tested recall levels.
```

## Result Sources

Configs:

- `tools/benchmark_configs/method4_multiassign_expansion_qps_20260629.json`
- `tools/benchmark_configs/method4_multiassign_expansion_qps_20260629_orion_low_recover.json`
- `tools/benchmark_configs/method4_multiassign_expansion_qps_20260629_orion_high_recover.json`
- `tools/benchmark_configs/method4_multiassign_expansion_qps_20260629_simple_low.json`
- `tools/benchmark_configs/method4_multiassign_expansion_qps_20260629_simple_s157_only.json`
- `tools/benchmark_configs/method4_multiassign_expansion_qps_20260629_simple_s189_only.json`

Result tables:

- `results/method4_benchmark_matrix/method4_multiassign_expansion_qps_20260629_orion_low/eval3000_20260629_v2/matrix_summary.csv`
- `results/method4_benchmark_matrix/method4_multiassign_expansion_qps_20260629_orion_low_recover/eval3000_20260629/matrix_summary.csv`
- `results/method4_benchmark_matrix/method4_multiassign_expansion_qps_20260629_orion_high_recover/eval3000_20260629/matrix_summary.csv`
- `results/method4_benchmark_matrix/method4_multiassign_expansion_qps_20260629_simple_low/eval3000_20260629/matrix_summary.csv`
- `results/method4_benchmark_matrix/method4_multiassign_expansion_qps_20260629_simple_s157_only/eval3000_20260629/matrix_summary.csv`
- `results/method4_benchmark_matrix/method4_multiassign_expansion_qps_20260629_simple_s189_only/eval3000_20260629/matrix_summary.csv`

One additional Orion default 0.80 row came from:

```text
results/method4_benchmark_matrix/method4_multiassign_expansion_qps_20260629_orion_default/eval3000_20260629/orion_o118_default_r080/20260629_101825/summary.json
```

## Caveats

- `stability_repeats=0`, so QPS is single-run final eval rather than stable mean.
- Some final recall values are slightly below their target after tuning selected
  a point that met the target on 500 tuning queries. The detailed table reports
  actual final recall.
- Orion `orion_single` 0.90 and 0.95 were recovered from an existing collection
  and over-shot the target recall; their QPS is therefore conservative for
  strict 0.90 / 0.95 comparison.
- Orion recover runs evaluate the deployed collection by recovering routing
  membership from Qdrant. This was necessary because recomputing routing while
  reusing an existing collection can produce an `expected_points` mismatch.
- During Simple KMeans runs the root filesystem filled due Qdrant WAL/storage.
  Temporary `bench_ma_*` collections were deleted after their results were
  written. The result directories above remain intact.
