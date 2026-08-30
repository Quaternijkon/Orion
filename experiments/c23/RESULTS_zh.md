# C2/C3 中文结果摘要

最新证据是 Stage 13 四机虚拟化线性资源复验。它是对原 Stage 12 的独立补充，不覆盖旧证据；两条证据线的最终方向一致。

| 实验 | 资源模型 | C2 | C3 |
|---|---|---|---|
| Stage 12 原协议 | 四台物理机上的逻辑分片；局部微基准逐分片执行 | `CONTRADICTED` | `CONTRADICTED` |
| Stage 13 线性资源复验 | 每逻辑分片独占一个物理核等价物和 8 GiB；总资源严格为 M/32 | `CONTRADICTED` | `CONTRADICTED` |

Stage 13 的核心数据：

- 42/42 个正式配置和 15/15 个最终审计门通过。
- C2：K-Means 在 8/10 个数据集×规模点具有更低 TWCut；Orion 仅在两个 `M=2` 点更低，局部可导航性没有一致优势。
- C3：`P_HNSW` 的配对 bootstrap 有 8/10 点显著偏向 K-Means、2/10 点不显著、0/10 点显著偏向 Orion。
- C3 完整支持点为 0/10；高 M 时 Full Orion 还存在严重失衡，`M=32` 最大分片/均值为 SIFT 4.930×、GloVe 5.887×。
- 因此，现有证据不支持“原反向结果只是四机资源不足的特殊情况”，也不支持 Orion 在 C2 或 C3 上优于 K-Means。

详细数据、置信区间、资源合同和审计入口见 [Stage 13 中文结果](retests/stage13-linear-resource-virtualized/RESULTS_zh.md)。
