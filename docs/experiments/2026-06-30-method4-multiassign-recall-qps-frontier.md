# 2026-06-30 Method4 Multi-Assignment Recall-QPS Frontier

## Question

用户问题：

> 在 recall-qps 这条 trade-off 曲线上，Orion 和 simple KMeans 的各种多分配策略
> 是否使得原有单分配的 trade-off 曲线更好了？还是只是在原来的 trade-off 曲线上移动？

本记录对 2026-06-29 的多分配实验做二级分析：不只看 0.80 / 0.85 / 0.90 / 0.95
四个 final selected points，而是把每个策略所有 `routing_tuning.csv` 中的 500-query
tuning 点合并，计算 recall-QPS Pareto frontier。

## Method

数据来源：

- `results/method4_benchmark_matrix/method4_multiassign_expansion_qps_20260629_orion_low/eval3000_20260629_v2/`
- `results/method4_benchmark_matrix/method4_multiassign_expansion_qps_20260629_orion_low_recover/eval3000_20260629/`
- `results/method4_benchmark_matrix/method4_multiassign_expansion_qps_20260629_orion_high_recover/eval3000_20260629/`
- `results/method4_benchmark_matrix/method4_multiassign_expansion_qps_20260629_simple_low/eval3000_20260629/`
- `results/method4_benchmark_matrix/method4_multiassign_expansion_qps_20260629_simple_s157_only/eval3000_20260629/`
- `results/method4_benchmark_matrix/method4_multiassign_expansion_qps_20260629_simple_s189_only/eval3000_20260629/`

每个 tuning point 包含：

- Recall@10 on 500 tuning queries；
- QPS；
- avg visited shards；
- avg EF/shard；
- 参数，例如 Orion 的 `upper_k/base/factor` 或 Simple KMeans 的 `nprobe/ef`。

判定方式：

```text
best_qps_at_recall(T) = max QPS among points with recall >= T
```

如果多分配策略在多数 recall threshold 上的 `best_qps_at_recall(T)` 高于单分配策略，
说明它把 recall-QPS frontier 抬高；如果只有少数局部点略高，或者大多数低于单分配，
说明它主要是在原曲线上移动，甚至被原曲线支配。

## Tuning Point Coverage

| Strategy | Points | Recall range | QPS range | Expansion |
|---|---:|---:|---:|---:|
| `orion_single` | 195 | 0.6148-0.9830 | 156.0-1582.9 | 1.000 |
| `orion_default` | 78 | 0.7162-0.9656 | 330.1-1441.4 | 1.185 |
| `orion_w2c2` | 78 | 0.7534-0.9798 | 281.8-1309.2 | 1.499 |
| `orion_w2c3` | 78 | 0.7690-0.9854 | 233.7-1174.9 | 1.852 |
| `simple_a1.000` | 52 | 0.7300-0.9704 | 142.7-980.5 | 1.000 |
| `simple_a1.004` | 52 | 0.7518-0.9768 | 116.8-726.0 | 1.191 |
| `simple_a1.010` | 52 | 0.7662-0.9760 | 108.1-749.4 | 1.566 |
| `simple_a1.014` | 52 | 0.7828-0.9782 | 102.3-701.9 | 1.887 |

## Orion: Best QPS at Recall Thresholds

| Recall threshold | `orion_single` | `orion_default` | `orion_w2c2` | `orion_w2c3` | Best multi / single |
|---:|---:|---:|---:|---:|---:|
| 0.75 | 1188.2 | 1441.4 | 1309.2 | 1174.9 | 1.21x |
| 0.80 | 1026.4 | 1260.9 | 1237.3 | 1174.9 | 1.23x |
| 0.82 | 495.7 | 1222.0 | 1205.0 | 1168.3 | 2.47x |
| 0.85 | 389.1 | 990.0 | 1124.8 | 980.8 | 2.89x |
| 0.88 | 340.4 | 823.2 | 882.0 | 835.6 | 2.59x |
| 0.90 | 320.6 | 751.6 | 761.9 | 752.7 | 2.38x |
| 0.92 | 320.6 | 619.0 | 641.6 | 637.8 | 2.00x |
| 0.95 | 320.6 | 428.0 | 448.7 | 438.5 | 1.40x |
| 0.97 | 254.8 | N/A | 331.5 | 335.4 | 1.32x |

Interpretation:

- Orion multi-assignment shifts the frontier upward, not merely along the
  single-assignment curve.
- `orion_default` and `orion_w2c2` dominate the single-assignment curve across
  the tested recall region where both have data.
- The biggest lift is in the middle-to-high recall band, roughly 0.82-0.92.
  This is exactly the region where single assignment has to inflate `upper_k`
  and lower EF to recover missed shards.
- The high-recall tail still benefits, but the lift shrinks because every
  strategy eventually has to visit many shards and search them deeply.

## Simple KMeans: Best QPS at Recall Thresholds

| Recall threshold | `simple_a1.000` | `simple_a1.004` | `simple_a1.010` | `simple_a1.014` | Best multi / single |
|---:|---:|---:|---:|---:|---:|
| 0.75 | 980.5 | 726.0 | 749.4 | 701.9 | 0.76x |
| 0.80 | 859.3 | 672.8 | 749.4 | 641.9 | 0.87x |
| 0.82 | 819.1 | 672.8 | 749.4 | 641.9 | 0.91x |
| 0.85 | 640.2 | 606.1 | 613.2 | 556.4 | 0.96x |
| 0.88 | 518.6 | 494.7 | 478.1 | 556.4 | 1.07x |
| 0.90 | 467.2 | 453.9 | 410.3 | 464.6 | 0.99x |
| 0.92 | 358.5 | 307.9 | 325.0 | 363.0 | 1.01x |
| 0.95 | 230.2 | 223.3 | 198.6 | 220.5 | 0.97x |
| 0.97 | 142.7 | 148.1 | 135.5 | 144.0 | 1.04x |

Interpretation:

- Simple KMeans alpha multi-assignment does not produce a broad frontier lift.
- Most alpha > 1.0 curves are below the original `alpha=1.000` curve.
- There are small local wins, especially `alpha=1.014` around 0.88 / 0.92 and
  alpha variants around 0.97, but those gains are narrow and small.
- The behavior is closer to moving along the original trade-off curve: extra
  copies can reduce `nprobe` or EF in some buckets, but the larger index and
  de-duplication cost mostly cancel the benefit.

## Representative Pareto Points

### Orion Single vs Default

| Strategy | Recall | QPS | Visited | EF/shard | Params |
|---|---:|---:|---:|---:|---|
| `orion_single` | 0.8012 | 1026.4 | 7.59 | 49.3 | `upper_k=24, base=24, factor=8` |
| `orion_default` | 0.8002 | 1260.9 | 6.33 | 38.0 | `upper_k=16, base=16, factor=8` |
| `orion_single` | 0.8540 | 389.1 | 25.58 | 160.6 | `upper_k=240, base=48, factor=12` |
| `orion_default` | 0.8502 | 990.0 | 7.32 | 71.8 | `upper_k=20, base=48, factor=8` |
| `orion_single` | 0.9540 | 320.6 | 28.17 | 138.9 | `upper_k=320, base=48, factor=8` |
| `orion_default` | 0.9526 | 428.0 | 21.19 | 132.6 | `upper_k=120, base=70, factor=10` |

The single-assignment curve has a large gap: to move from ~0.88 recall toward
~0.95, it needs very wide upper routing. Orion default fills that gap with
moderate multi-assignment and keeps the same recall neighborhood at far lower
visited-shard count.

### Simple KMeans Single vs Alpha

| Strategy | Recall | QPS | Visited | EF/shard | Params |
|---|---:|---:|---:|---:|---|
| `simple_a1.000` | 0.8124 | 859.3 | 10.00 | 64.0 | `nprobe=10, ef=64` |
| `simple_a1.010` | 0.8286 | 749.4 | 8.00 | 64.0 | `nprobe=8, ef=64` |
| `simple_a1.000` | 0.8554 | 640.2 | 16.00 | 64.0 | `nprobe=16, ef=64` |
| `simple_a1.004` | 0.8648 | 606.1 | 12.00 | 80.0 | `nprobe=12, ef=80` |
| `simple_a1.000` | 0.9004 | 467.2 | 20.00 | 96.0 | `nprobe=20, ef=96` |
| `simple_a1.014` | 0.9046 | 464.6 | 16.00 | 80.0 | `nprobe=16, ef=80` |
| `simple_a1.000` | 0.9526 | 230.2 | 24.00 | 240.0 | `nprobe=24, ef=240` |
| `simple_a1.014` | 0.9546 | 220.5 | 24.00 | 160.0 | `nprobe=24, ef=160` |

Simple KMeans alpha can reduce visited shards or EF at selected points, but QPS
usually remains equal or lower. That is the signature of moving on or near the
same trade-off curve, not creating a new dominant curve.

## Answer

For Orion:

```text
Multi-assignment improves the recall-QPS frontier.
It is not merely moving along the original single-assignment curve.
```

For Simple KMeans:

```text
Distance-alpha multi-assignment mostly moves along, or below, the original
single-assignment trade-off curve. It gives small local gains in a few recall
regions, but no broad Pareto-frontier improvement.
```

Mechanistic interpretation:

- Orion default/w2c2 copies points at topology/vote-supported shard boundaries.
  These copies directly reduce routing misses, so the same recall can be reached
  with smaller `upper_k`, fewer visited shards, and lower EF-sum.
- Simple KMeans alpha copies points based only on centroid-distance bands. The
  extra copies sometimes reduce `nprobe`, but they are less aligned with online
  routing misses, while index/search/de-duplication cost rises.

## Caveats

- This is a 500-query tuning-frontier analysis, not a 3000-query stability
  frontier. It is the right lens for curve shape, but exact QPS values should
  be confirmed with selected final eval points before using them as final paper
  numbers.
- The available single-assignment Orion grid has sparse coverage between
  recall ~0.88 and ~0.95. That sparsity is itself evidence that single
  assignment struggles to reach the middle-high recall region cheaply, but it
  can exaggerate threshold-ratio numbers around 0.90.
- Some Orion rows use recover-from-collection for already-built collections.
  They evaluate the deployed collection state, but should be documented
  separately from fully rebuilt non-recover runs.
