# 2026-06-26 Orion vs Simple KMeans: 相近平均 EF 下的访问 shard 数与 QPS

## 问题

用户问题：

> 如果在平均 EF 相近的情况下，Orion 是否比 simple KMeans 访问更少的 shard？请在 0.8, 0.85, 0.9, 0.95 召回率下实验。

本实验只比较：

- **Orion**: `routing_mode=faithful_original_rest`, collection `bench095_rr_orion_s31`, recover existing routing, lower 使用 dynamic EF。
- **Simple KMeans**: `routing_mode=kmeans_simple_nprobe`, collection `bench095_cpp_kmeans_s46`, lower 使用 fixed EF，访问 shard 数由 `nprobe` 控制。

这里的“平均 EF”采用 harness 输出的 `avg_ef_per_visited_shard`。辅助指标 `EF-sum` 定义为：

```text
EF-sum = avg_visited_shards * avg_ef_per_visited_shard
```

它近似表示每个 query 的 lower 层总搜索预算。

## 实验设置

配置文件：

```text
tools/benchmark_configs/method4_orion_vs_simple_kmeans_same_avg_ef_20260626.json
```

结果目录：

```text
results/method4_benchmark_matrix/method4_orion_vs_simple_kmeans_same_avg_ef_20260626/eval3000_20260626_preload
```

核心参数：

| 项 | 值 |
|---|---:|
| Dataset | `/home/taig/dry/faiss/datasets/glove-200-angular.hdf5` |
| top_k | 10 |
| tuning queries | 1000 |
| final eval queries | 3000 |
| batch_size | 100 |
| stability_repeats | 0 |
| upper_search_ef | 400 |
| dispatch | `coordinator` |
| lower execution order | `query_major` |

运行时环境说明：当前 `python3` 来自 conda，默认会优先加载 conda 自带的旧 `libstdc++.so.6`，导致 `hnswlib` import 缺少 `GLIBCXX_3.4.32`。本次运行使用系统 libstdc++ 预加载：

```bash
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
  python3 tools/method4_benchmark_matrix.py \
  --config tools/benchmark_configs/method4_orion_vs_simple_kmeans_same_avg_ef_20260626.json \
  --run-id eval3000_20260626_preload \
  --run
```

0.95 处曾探测 `simple_r095_n32_b160`，其 tuning recall 为 `0.9494`，低于 strict target `0.9500`，因此正式对比采用同样 fixed EF=160 但达到目标的 `nprobe=40`。

## 结果

源表：

```text
results/method4_benchmark_matrix/method4_orion_vs_simple_kmeans_same_avg_ef_20260626/eval3000_20260626_preload/matrix_summary.csv
```

| Target recall | Orion params | Simple KMeans params | Recall O / S | Avg EF O / S | Visited shards O / S | Orion shard reduction | QPS O / S | QPS speedup | EF-sum O / S |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 0.80 | `upper_k=16, base=16, factor=10` | `nprobe=12, ef=48` | 0.8040 / 0.8060 | 43.2 / 48.0 | 6.37 / 12.00 | 46.9% fewer | 1009.9 / 870.1 | 1.16x | 275 / 576 |
| 0.85 | `upper_k=24, base=32, factor=10` | `nprobe=16, ef=64` | 0.8545 / 0.8520 | 63.4 / 64.0 | 8.30 / 16.00 | 48.1% fewer | 887.8 / 617.8 | 1.44x | 526 / 1024 |
| 0.90 | `upper_k=60, base=40, factor=8` | `nprobe=32, ef=80` | 0.9063 / 0.9041 | 76.0 / 80.0 | 14.54 / 32.00 | 54.6% fewer | 702.0 / 352.4 | 1.99x | 1105 / 2560 |
| 0.95 | `upper_k=120, base=80, factor=12` | `nprobe=40, ef=160` | 0.9542 / 0.9538 | 154.7 / 160.0 | 21.12 / 40.00 | 47.2% fewer | 391.3 / 206.9 | 1.89x | 3267 / 6400 |

其中 O 表示 Orion，S 表示 simple KMeans。

## 结论

在这组相近平均 EF 的 final 3000-query 实验里，答案是 **是的**：

- 四个召回率目标下，Orion 的 `avg_ef_per_visited_shard` 与 simple KMeans 很接近，差距约为 0.6 到 9.9%。
- 在相同召回水平下，Orion 平均访问 shard 数显著更低：
  - 0.80: 少 46.9%
  - 0.85: 少 48.1%
  - 0.90: 少 54.6%
  - 0.95: 少 47.2%
- QPS 也随之更高：
  - 0.80: 1.16x
  - 0.85: 1.44x
  - 0.90: 1.99x
  - 0.95: 1.89x
- `EF-sum` 约减少一半，说明收益不只是“每个 shard 的 EF 不同”，而是 Orion routing 在相近 shard 内搜索强度下访问了更少 lower shards。

这支持如下表述：

> 在相近平均 lower EF 和相近 Recall@10 下，Orion 的 upper routing / 多分配结构比 simple KMeans nprobe 更集中地命中有用 lower shards，因此需要访问更少 shard，并获得更高 QPS。

## Caveats

- 本次是 `stability_repeats=0` 的单轮 final eval，QPS 会受当时集群状态影响。访问 shard 数和 recall 趋势更稳定，但若用于论文图表，建议对这 8 个参数点再跑 3 次 repeat。
- 0.95 的 simple KMeans 使用 `nprobe=40, ef=160`，因为 `nprobe=32, ef=160` 在 tuning 上只有 `0.9494`，没有满足 strict target。这个选择让 0.95 对比保持 recall 达标，且平均 EF 与 Orion 仍然贴近。
