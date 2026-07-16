# 2026-07-13 Method4 Claim B: B-01 `orion_single` 目标召回复测

## 目的

复测 B-01 中 `Orion / orion_single` 对照组在 `0.80 / 0.85 / 0.90 / 0.95` 四档目标召回率水平下的表现。

这次复测的对象是无多分配的 Orion baseline：

- Strategy: `orion_single`
- `multi_assign=False`
- 命令侧使用 `--disable-multi-assign`
- 不使用 GIST，仅使用 GloVe
- 只选择 3000-query formal evaluation 中实际达到对应 target recall 的行

这不同于 `2026-07-13-method4-claim-b-orion-single-revalidation.md` 中的旧异常参数复跑。旧文档只是复核几行旧参数是否可复现；本文档记录的是重新按目标召回 bucket 选择的 B-01 `orion_single` 结果。

## 实验设置

| 项目 | 值 |
|---|---|
| Dataset | `/home/taig/dry/faiss/datasets/glove-200-angular.hdf5` |
| Collection | `bench_ma_b01_orion_single_target_s31_20260713` |
| Initial shards | 31 |
| Effective logical shards after fission | 42 |
| Routing mode | `faithful_original_rest` |
| Routed execution mode | `compact_multi_ep` |
| Routed planning mode | `materialized` |
| Search dispatch mode | `coordinator` |
| Lower execution order | `query_major` |
| Shard placement | `round_robin` |
| Tuning query count | 500 |
| Eval query count | 3000 |
| Batch size | 100 |
| Multi-assignment | disabled |
| Index expansion | 1.0 |

说明：这里的 `index expansion=1.0` 按 B-01 中“是否因多分配产生数据/索引副本扩张”的定义计算。虽然 collection 从 31 个 initial shards fission 到 42 个 effective logical shards，但本组没有 multi-assignment，不产生多分配副本扩张。

## 目标召回复测结果

| 目标召回 | 方法 | Strategy | Final Recall@10 | QPS | 平均访问 shards | 平均 EF/访问 shard | EF-sum/query | Index expansion | 选择参数 |
|---:|---|---|---:|---:|---:|---:|---:|---:|---|
| 0.80 | Orion | `orion_single` | 0.81310 | 926.03 | 6.72 | 83.72 | 562.50 | 1.0 | `upper_k=20, base_ef=48, factor=12` |
| 0.85 | Orion | `orion_single` | 0.85637 | 773.51 | 10.35 | 78.39 | 811.09 | 1.0 | `upper_k=40, base_ef=32, factor=12` |
| 0.90 | Orion | `orion_single` | 0.90900 | 563.19 | 17.24 | 101.61 | 1751.62 | 1.0 | `upper_k=100, base_ef=32, factor=12` |
| 0.95 | Orion | `orion_single` | 0.95143 | 341.89 | 23.70 | 195.02 | 4621.98 | 1.0 | `upper_k=200, base_ef=60, factor=16` |

## Excel/TSV 友好表格

下面内容使用 Tab 分隔，可直接粘贴到 Excel：

```tsv
目标召回	方法	Strategy	Final Recall@10	QPS	平均访问shards	平均EF/访问shard	EF-sum/query	Index expansion	upper_k	base_ef	factor
0.80	Orion	orion_single	0.81310	926.03	6.72	83.72	562.50	1.0	20	48	12
0.85	Orion	orion_single	0.85637	773.51	10.35	78.39	811.09	1.0	40	32	12
0.90	Orion	orion_single	0.90900	563.19	17.24	101.61	1751.62	1.0	100	32	12
0.95	Orion	orion_single	0.95143	341.89	23.70	195.02	4621.98	1.0	200	60	16
```

## 被排除候选

0.80 档最初选择过一条 tuning recall 达到 0.80 的候选，但该候选在 3000-query formal evaluation 中未达到 0.80，因此不能作为 0.80 目标召回结果使用。

| 目标召回 | Tuning target | 状态 | Final Recall@10 | QPS | 平均访问 shards | EF-sum/query | 参数 |
|---:|---:|---|---:|---:|---:|---:|---|
| 0.80 | 0.800 | excluded, final below target | 0.79707 | 958.75 | 7.54 | 372.95 | `upper_k=24, base_ef=24, factor=8` |

## 源文件

汇总表：

- `results/method4_claim_b_orion_single_target_retest_20260713/orion_single_target_retest_selected_20260713.csv`
- `results/method4_claim_b_orion_single_target_retest_20260713/orion_single_target_retest_audit_20260713.csv`

Selected source summaries：

- `results/method4_claim_b_orion_single_target_retest_20260713/r080_buffer/20260713_125250/summary.json`
- `results/method4_claim_b_orion_single_target_retest_20260713/r085/20260713_125522/summary.json`
- `results/method4_claim_b_orion_single_target_retest_20260713/r090/20260713_125720/summary.json`
- `results/method4_claim_b_orion_single_target_retest_20260713/r095/20260713_130012/summary.json`

Excluded audit source summary：

- `results/method4_claim_b_orion_single_target_retest_20260713/r080/20260713_124505/summary.json`

## 清理状态

复测使用的临时 collection `bench_ma_b01_orion_single_target_s31_20260713` 已在复测结束后删除。

即时复查结果：

- Qdrant API 返回：collection 不存在
- 未发现仍在运行的 `qdrant_two_level_routing_experiment.py` 进程匹配该 collection
- `/home/taig/dry/qdrant` 所在 `/home` 分区即时可用空间约 20G
- `qdrant_storage/controller-cluster` 即时大小约 36G

## 结论

新的 B-01 `orion_single` 复测结果显示，之前可疑表格中的 `orion_single` 行不能作为四档目标召回 bucket 的最终数据直接使用。应使用本文档中的 near-target formal rows 作为 B-01 无多分配 Orion 对照组数据。

在当前复测中，`orion_single` 的 3000-query formal evaluation 指标为：

- 0.80 档：Recall@10 `0.81310`，QPS `926.03`，平均访问 shards `6.72`，EF-sum/query `562.50`，index expansion `1.0`
- 0.85 档：Recall@10 `0.85637`，QPS `773.51`，平均访问 shards `10.35`，EF-sum/query `811.09`，index expansion `1.0`
- 0.90 档：Recall@10 `0.90900`，QPS `563.19`，平均访问 shards `17.24`，EF-sum/query `1751.62`，index expansion `1.0`
- 0.95 档：Recall@10 `0.95143`，QPS `341.89`，平均访问 shards `23.70`，EF-sum/query `4621.98`，index expansion `1.0`
