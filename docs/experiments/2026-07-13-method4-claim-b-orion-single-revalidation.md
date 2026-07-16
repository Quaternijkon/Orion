# 2026-07-13 Method4 Claim B Orion single 异常行复测

## 目的

复核最终 Excel/TSV 中 `Orion / orion_single` 的 4 行是否可作为 Claim B 的目标召回 bucket 数据：

| Target recall bucket | Params | Old final recall | Old QPS |
|---:|---|---:|---:|
| 0.80 | `upper_k=80, base=120, factor=16` | 0.8012 | 499.8 |
| 0.85 | `upper_k=240, base=48, factor=12` | 0.8468 | 372.7 |
| 0.90 | `upper_k=320, base=48, factor=8` | 0.9496 | 322.5 |
| 0.95 | `upper_k=400, base=120, factor=12` | 0.9711 | 219.1 |

## 复测设置

- Dataset: GloVe, `/home/taig/dry/faiss/datasets/glove-200-angular.hdf5`
- Collection: `bench_ma_revalidate_orion_single_s31_20260713`
- Initial shards: 31
- Effective logical shards after fission: 42
- `multi_assign=False`
- `routing_mode=faithful_original_rest`
- `routed_execution_mode=compact_multi_ep`
- `routed_planning_mode=materialized`
- `tuning_query_count=500`
- `eval_query_count=3000`
- `batch_size=100`
- 临时 collection 已在复测结束后删除，仅保留结果文件。

汇总 CSV:

`results/method4_claim_b_orion_single_revalidation_20260713/orion_single_old_rows_revalidation_20260713.csv`

## 复测结果

| Target recall bucket | Params | Old final recall | Rerun final recall | Recall delta | Old QPS | Rerun QPS | Old avg visited shards | Rerun avg visited shards | Old EF-sum/query | Rerun EF-sum/query | 解释 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0.80 | `upper_k=80, base=120, factor=16` | 0.8012 | 0.92587 | +0.12467 | 499.8 | 428.03 | 15.32 | 15.27 | 3119 | 3112.44 | 不复现旧低召回，旧 bucket 行应视为过期/异常 |
| 0.85 | `upper_k=240, base=48, factor=12` | 0.8468 | 0.94853 | +0.10173 | 372.7 | 360.57 | 25.50 | 25.34 | 4104 | 4096.32 | 不复现旧低召回，且旧 final recall 低于 0.85 目标 |
| 0.90 | `upper_k=320, base=48, factor=8` | 0.9496 | 0.94840 | -0.00120 | 322.5 | 314.64 | 28.12 | 28.10 | 3910 | 3908.58 | 复现为高召回点，但明显超出 0.90 bucket |
| 0.95 | `upper_k=400, base=120, factor=12` | 0.9711 | 0.97037 | -0.00073 | 219.1 | 219.72 | 30.15 | 30.11 | 8418 | 8413.28 | 复现为高召回点，但明显超出 0.95 bucket |

## 结论

这组 `Orion / orion_single` 数据不能继续作为按 `0.80/0.85/0.90/0.95` 目标召回 bucket 排列的 Claim B frontier 数据直接使用。

具体原因：

1. `0.80` 和 `0.85` 两行的旧低召回结果在当前复测中不复现。同一参数分别达到 `0.92587` 和 `0.94853` 的 3000-query recall，说明旧表低召回值不是当前实验状态下可复现的稳定结果。
2. `0.90` 行的测量值基本可复现，但实际 recall 约 `0.948`，更接近 0.95 档，不适合代表 0.90 目标 bucket。
3. `0.95` 行的测量值基本可复现，但实际 recall 约 `0.970`，作为 0.95 目标 bucket 时有明显 overshoot；如果绘制 recall-QPS 曲线，应使用 actual recall，而不是仅按 nominal bucket 画点。

因此，最终 Excel 中对应 B-01 的 `orion_single` 行需要被标注为异常/过期，或改为从严格复测/新的 near-target formal run 中选择 replacement rows。
