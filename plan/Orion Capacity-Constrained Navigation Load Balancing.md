# Orion 固定分片数的容量约束导航负载均衡方案

## 1. 目标和边界

目标是在现有 Orion 架构内，使最底层逻辑分片存储的 physical-copy 数量尽可能均匀，同时保留导航局部性和整体高 QPS。

本方案的硬约束是：

- production `UpperNavigator` 及其 immutable upper graph 不重建、不替换；
- 只改变 L1 membership 和最终 L0 layout；
- 每次 L1 迁移和每个 L0 放置必须有该节点或点的 L1 attachment 导航证据；
- 固定逻辑分片数 P；capacity-constrained 模式不使用会改变 P 的 fission；
- multi-assignment 开启时保留每个点原策略请求的副本数，并按 physical-copy 计量分片负载；
- 在线 artifact lookup、shard selection、MultiEP 和 Dynamic-EF 路径不增加调度器、负载统计或二次路由。

因此新增开销位于离线 layout 构建阶段；在线复杂度和 artifact 读取形式不变。

## 2. 算法：Capacity-Constrained Navigation Balancing（CCNB）

### 2.1 严格容量初始化

保留现有 weighted K-Means 几何初始化，但将原来的 `1.5 × target` 宽松上限替换为可配置容量带。当前实验默认值为：

- `min_load_ratio = 0.99`
- `max_load_ratio = 1.01`
- `max_vote_loss = 3`
- `max_passes = 8`

这一步仍使用同一组 upper points、同一距离度量和同一 K-Means seed。

### 2.2 顺序式、容量约束的 L1 topology refinement

旧实现按整轮 majority vote 同步迁移，target 没有容量上限，容易发生多数投票塌缩。CCNB 改为确定性的顺序提交：

1. 对每个 L1 节点从 immutable attachments 计算 shard votes；
2. 只考虑有导航证据且票损不超过 `max_vote_loss` 的 target；
3. target 加入该节点权重后不得超过容量上界；
4. source 移出该节点权重后不得低于容量下界；
5. topology vote gain 为正时允许迁移；或在 source/target 违反容量带时，允许能减少绝对负载误差的 bounded repair；
6. 候选按 repair 优先级、vote gain/weight、负载改善和稳定 shard id 排序后顺序提交。

这避免了一轮内多个节点同时涌入同一 target。

### 2.3 最终 L0 physical-copy 分配

对每个点：

1. 从其 L1 attachments 汇总 shard votes；
2. 按原 multi-assignment 规则计算请求的副本数；
3. 候选仅限于有导航证据、且与最高票相差不超过 bounded vote loss 的 shard；
4. 先按“候选余量最小、票数置信度最高”排序点，再进行容量感知的确定性分配；
5. 用直接 evidence-preserving move 修复 over-cap 和 under-floor；
6. 如果直接 move 被迁移链卡住，按相同候选集合聚合 point groups，求解整数 capacity circulation，再确定性分解回逐点 membership。

capacity flow 的 group-to-shard 容量不超过 group 内点数，因此同一点不会在同一 shard 放置两个副本。若 attachment 候选图本身不支持指定容量带，构建结果保留 evidence-only layout，并在 manifest 中明确记录 infeasible、未饱和流量、覆盖不足 shard 和最终违例；不会静默退化到 Hash。

## 3. Artifact 和审计信息

`balance_diagnostics` 至少记录：

- L1 和 L0 的 initial/final load histogram；
- min、max、mean、std、CV、max/mean、min/mean；
- 容量上下界和是否满足；
- topology gain move、balance repair move 和拒绝原因；
- navigation vote delta 和实际最大票损；
- 每个 shard 的 navigation/eligible point coverage；
- multi-assignment 请求的总副本数和是否逐点保留；
- grouped capacity flow 的 group 数、可行性和饱和量；
- balance 阶段离线耗时；
- fixed-P 和 fission 未执行的原因。

现有 `layout_sha256` 和 production upper-graph binding 流程保持不变。不同 layout 必须产生不同 layout checksum，但不能改变 upper graph 本身。

## 4. 50k/P=32 预验证结果

以下仅是 graphless、single-assignment、50k train smoke；它证明 layout 机制工作，不证明完整 Recall/QPS 结论。

| 数据集 | 方法 | min | max | max/mean | min/mean | CV | CCNB 离线阶段 |
|---|---:|---:|---:|---:|---:|---:|---:|
| SIFT | 原 Orion，无 fission | 7 | 8,240 | 5.2736 | 0.0045 | 1.2302 | - |
| SIFT | CCNB，vote loss 3 | 1,546 | 1,579 | 1.0106 | 0.9894 | 0.0081 | 2.52 s |
| GloVe | 原 Orion，无 fission | 118 | 19,642 | 12.5709 | 0.0755 | 2.2027 | - |
| GloVe | CCNB，vote loss 3 | 1,546 | 1,579 | 1.0106 | 0.9894 | 0.0074 | 3.07 s |

两个 CCNB layout 都保持 P=32、expansion ratio=1.0，且不需要 grouped-flow fallback；所有最终放置都有 navigation evidence。

参数扫描也显示，票损不是可以忽略的自由参数：在同一 0.99/1.01 容量带下，SIFT 的 `max_vote_loss=1` 不可行，而 `max_vote_loss=3` 可行；GloVe 在 1 时已可行。这一差异必须进入后续 Recall、fan-out 和 QPS Pareto 分析，不能只报告均衡数字。

### 4.1 全量 P=32 layout 验证

同一默认配置随后在完整数据集上通过：

| 数据集 | logical points | min | max | max/mean | min/mean | CV | graphless 总构建 | CCNB 阶段 | peak RSS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SIFT1M | 1,000,000 | 30,937 | 31,563 | 1.010016 | 0.989984 | 0.009254 | 73.52 s | 48.00 s | 1.44 GiB |
| GloVe | 1,183,514 | 36,614 | 37,355 | 1.010009 | 0.989974 | 0.008024 | 121.25 s | 62.12 s | 2.16 GiB |

两次全量构建均保持 P=32、expansion ratio=1.0、无空 shard，并且 direct evidence-preserving repair 已满足容量带，不需要 grouped capacity-flow fallback。该结果验证的是全量 layout 和离线资源开销；它仍不替代 matched-recall 在线 QPS 实验。

### 4.2 Multi-assignment physical-copy 验证

50k/P=32 开启现有 multi-assignment 规则后，CCNB 保留了每个点请求的副本数，并直接按 physical copies 做容量约束：

| 数据集 | physical copies | expansion ratio | min | max | max/mean | min/mean | copy count preserved |
|---|---:|---:|---:|---:|---:|---:|---:|
| SIFT | 59,804 | 1.19608 | 1,850 | 1,888 | 1.01023 | 0.98990 | yes |
| GloVe | 69,575 | 1.39150 | 2,152 | 2,196 | 1.01002 | 0.98978 | yes |

两组结果的所有副本放置都有 navigation evidence，且同一点不会在同一 shard 重复放置。

原始 smoke artifacts 位于：

`/proj/intelisys-PG0/exp/orion-distributed/orion-logical-balance-20260825/`

## 5. 完整验收协议

正式结论必须在 SIFT1M 和 GloVe、P={4,8,16,32} 上分别比较原 Orion 与 CCNB，并固定相同 upper graph、upper query 参数、lower HNSW、physical hosts 和资源合同。

必须同时报告：

- logical shard count 和 physical host count；
- shard physical-copy min/max/std/CV/max÷mean；
- expansion ratio；
- Recall@10；
- TWCut、local navigability、visited logical shards、P_HNSW 和 fan-out；
- matched-recall QPS、p50/p95/p99 latency、CPU、网络和 coordinator/worker 利用率；
- upper graph checksum、attachments checksum、layout checksum；
- 总构建时间和 CCNB 增量构建时间。

建议主门禁：

- max/mean 不高于 1.01-1.05，且无空 shard；
- Recall@10 不低于 0.90，并与原 Orion 做 matched-recall 比较；
- QPS 不显著下降；若均衡减少最大 shard 尾延迟，应验证是否转化为稳定 QPS 增益；
- 在线 router 路径无新增状态和调度步骤；
- 所有放置有 navigation evidence，或明确将该配置判为 infeasible；
- 不覆盖 C2/C3 已接受的负结果，CCNB 是独立的新实验线。

在完整 Recall/QPS 矩阵完成前，只能得出“CCNB 显著修复了 smoke layout imbalance，且在线代码路径不变”，不能声称已经证明整体高 QPS。
