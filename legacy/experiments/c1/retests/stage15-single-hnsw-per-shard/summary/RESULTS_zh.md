# Stage 15：每个 shard 单 HNSW 图的 1～4 台真实机器重测

状态：`MEASUREMENTS_COMPLETE`。SIFT1M 固定 `efSearch=24`；GloVe 单图在有限网格中选择最小达标值并固定为 `efSearch=320`（硬上限 384）。

## 正式 QPS

| 数据集 | 方法 | ef | P(M=1/2/3/4) | M=1 | M=2 | M=3 | M=4 | M=4/M=1 |
|---|---|---:|---|---:|---:|---:|---:|---:|
| SIFT1M | Random broadcast | 24 | 1/2/3/4 | 17,491.5 | 10,880.7 | 8,261.2 | 6,965.2 | 0.398× |
| SIFT1M | K-Means routing | 24 | 1/1/2/2 | 17,491.5 | 20,134.2 | 13,786.1 | 13,774.6 | 0.788× |
| GloVe-200 | Random broadcast | 320 | 1/2/3/4 | 3,155.0 | 3,413.5 | 3,576.4 | 3,616.1 | 1.146× |
| GloVe-200 | K-Means routing | 320 | 1/2/3/3 | 3,155.0 | 3,158.2 | 3,285.3 | 3,396.6 | 1.077× |

尾部单机基线漂移：

| 数据集 | baseline-A | baseline-B | 漂移 |
|---|---:|---:|---:|
| SIFT1M | 17,491.5 | 17,323.4 | -0.96% |
| GloVe-200 | 3,155.0 | 3,120.0 | -1.11% |

## 固定 ef 的 K-Means fan-out

| 数据集 | 固定 ef | M=1 | M=2 | M=3 | M=4 |
|---|---:|---:|---:|---:|---:|
| SIFT1M | 24 | 1 | 1 | 2 | 2 |
| GloVe-200 | 320 | 1 | 2 | 3 | 3 |

GloVe 单图有限 ef 网格：192→0.8719，224→0.8815，256→0.8887，320→0.9013，384→0.9126；最终 ef=320 的独立留出 Recall=0.90269。

## 现象是否合理

合理，而且不能要求端到端 QPS 被调成对数增长。HNSW 的对数项只近似描述一张局部图的搜索工作；在 N≈100 万时，从 M=1 分到 M=4，`log(N)/log(N/4)` 只有约 1.11× 的局部收益。在负载均衡的粗略工作模型中，吞吐上界更接近 `M / (P × D_local + routing + scatter/gather + merge)`。

- Random 必须 P=M 全广播，因此 M 与 P 在粗略模型中相互抵消，只剩很小的局部图收益。SIFT 的每-shard 距离工作降至 0.90×，但聚合 worker-pool CPU 从 48.9% 升到 75.1%，scatter/gather 开销使 QPS 反而下降。
- GloVe 固定 ef=320 后，每-shard 距离工作从 M=1 到 M=4 为 1.00×，几乎不降；这说明 ef 下限主导了局部工作，因此 Random 只得到 1.15×，而不是四倍。
- K-Means 只有在 P 增长慢于 M、分区均衡且协调端不受限时才可能扩展。GloVe 在 M=2/3/4 的实测 P 接近或等于 M，不具备低 fan-out 前提。
- Stage 14 的单机集合含多张非空 HNSW 图，压低了 M=1 基线；Stage 15 将每 shard 固定为一张图后，消除了这个图数量混杂，但也暴露出真实的 fan-out/总工作瓶颈。
- GloVe M=3 的 K-Means 最大 shard 占约 56.1%，所以三台机器并未提供三份均匀算力。

结论不是“修正后符合预设曲线”，而是：局部 HNSW 工作可随 shard 变小而缓慢下降；端到端 QPS 是否上升由 P 和系统开销决定。本轮数据不支持把 HNSW 分图直接表述成 QPS 的对数扩展定律。

## PDF 图表

每张 PDF 页面下方都内嵌中文“用途、如何理解、最终结论、证据边界”。

- [stage15_fig1_physical_qps.pdf](figures/stage15_fig1_physical_qps.pdf)
- [stage15_fig2_scaling_model.pdf](figures/stage15_fig2_scaling_model.pdf)
- [stage15_fig3_fixed_ef_fanout.pdf](figures/stage15_fig3_fixed_ef_fanout.pdf)
- [stage15_fig4_local_vs_aggregate_work.pdf](figures/stage15_fig4_local_vs_aggregate_work.pdf)
- [stage15_fig5_stage14_comparison.pdf](figures/stage15_fig5_stage14_comparison.pdf)
- [stage15_fig6_glove_ef_calibration.pdf](figures/stage15_fig6_glove_ef_calibration.pdf)
- [stage15_fig7_diagnostics.pdf](figures/stage15_fig7_diagnostics.pdf)

## 完整性

- collection 清理证明：16/16 通过；四个 peer 均返回 HTTP 404。
- 绑核、物理核无重叠、实验后恢复：`PASS`；Qdrant 已恢复到 `0-19`。
- 16 个独立 E1 文件均为 `VALID_E1`；所有正式重复 Recall@10≥0.90、`bottleneck_flags=[]`。
- 16 个 prepare gate 均满足总 segment 数 `2×M`，即每 shard 一张非空 HNSW 图加一个空 appendable segment。
- Stage 14 原证据未覆盖，现重分类为生产默认多-segment 行为。
