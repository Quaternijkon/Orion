# C1 PDF 中文说明版

本目录包含 `131` 张 C1 PDF 的中文说明副本：顶层当前/阶段图 `53` 张，
`history/` 历史图 `78` 张。原始 PDF 未被修改。

每一页图表下方均包含：

- 用途：该图回答什么实验问题；
- 如何理解：横纵轴、参考线、CDF 或模型线应如何阅读；
- 最终结论：采用 Stage 12 确定性最终记录的科学结论；
- 证据状态：区分当前权威图、数据集快照、已被取代图和历史审计副本。

顶层无 `stage` 前缀的图是当前引用版本。`stage*` 及 `history/` 文件仅用于复现和结论演变追溯。

| 图表类型 | PDF 数量 |
|---|---:|
| `actual_fanout_cdf` | 8 |
| `aggregate_work` | 22 |
| `combined_motivation` | 2 |
| `fanout_vs_shards` | 16 |
| `local_cpu_time` | 2 |
| `local_distance_computations` | 2 |
| `local_nodes_visited` | 2 |
| `local_search_work` | 4 |
| `model_vs_observed` | 11 |
| `oracle_fanout_cdf` | 16 |
| `physical_scaleout` | 16 |
| `projected_scaling` | 22 |
| `scaling_efficiency` | 8 |

生成与检查：

```bash
/users/dry/orion-distributed/venv/bin/python experiments/c1/scripts/c1_annotate_figures_zh.py
/users/dry/orion-distributed/venv/bin/python experiments/c1/scripts/c1_annotate_figures_zh.py --check
```
