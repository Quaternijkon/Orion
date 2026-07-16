# Method4 Distributed Qdrant 投稿实验执行对照表

这份文档只保留实验执行需要的信息：**要 claim 什么、对照谁、怎么设参数、看什么指标、什么结果算支撑**。
长版论证见：

```text
docs/experiments/2026-06-16-method4-distributed-submission-experiment-plan.md
```

## 使用方式

每完成一组实验，在对应 claim 下填：

```text
result_dir:
commit/image:
dataset:
main numbers:
decision: support / weak support / reject
```

不要把“没有正收益”的实验写成强 claim。没有正收益的优化只能写成工程尝试或删去。

## 全局固定条件

除非该 claim 明确改变变量，其余条件固定。

| 项目 | 设置 |
|---|---|
| 主数据集 | `glove-200-angular.hdf5` |
| 推荐补充数据集 | 一个 L2 数据集，如 SIFT/GIST/Deep |
| top-k | 10 |
| lower HNSW `m` | 32 |
| lower HNSW `ef_construct` | 100 |
| upper sample | `N / 32` |
| lower logical shards | 当前主线 46 |
| topology | Qdrant controller + physical workers |
| repeats | 正式结果每点至少 3 repeats |
| eval queries | tuning 可用 500；正式至少 3000；最终确认建议 10000 |
| 主评价口径 | same-recall comparison |

当前已有参考点，只能作为起点，投稿最终表格需在最终 commit/image 上重跑：

| 配置 | Recall@10 | QPS mean | 说明 |
|---|---:|---:|---|
| Method4 current, `upper_k=160, base_ef=80, factor=8, batch=200` | 0.955267 | 387.140 | 当前同召回主结果 |
| Naive closest recall, `ef=76, batch=100` | 0.954767 | 272.370 | 稳定 naive 对照 |
| Worker-local peer pre-merge enabled | 0.955300 | 405.381 | same-image A/B enabled |
| Worker-local peer pre-merge disabled | 0.955233 | 379.422 | same-image A/B disabled |

## Claim 总览

| 优先级 | Claim | 最小必须实验 | 结果用途 |
|---:|---|---|---|
| 1 | 拓扑感知划分提升路由局部性 | partition oracle + KMeans/random 对照 | 证明划分方式本身有意义 |
| 2 | 同召回下 Method4 QPS / latency 优于 naive | same-recall sweep | 主性能结论 |
| 3 | 多分配降低 routing miss | single vs voting multi-assignment | 解释高召回来源 |
| 4 | dynamic EF 比 fixed EF 更高效 | fixed EF vs dynamic EF | 解释预算分配机制 |
| 5 | shard-major + peer pre-merge 契合分布式执行 | execution ablation | 证明系统优化有效 |
| 6 | 实现保持原初 Method4 外部语义 | semantic trace audit | 防止算法漂移质疑 |
| 7 | controller-native / compact request 降低调度开销 | routing overhead ablation | 系统工程贡献 |
| 8 | physical layout 降低 hot worker | placement simulation + online A/B | 有收益再强 claim |

## Claim A: 拓扑感知划分提升路由局部性

**要证明**

Method4 的划分考虑 upper graph / L1 topology 后，ground-truth 近邻更集中在被路由访问的 shards 中；
因此同等访问 shard 数下比 random / KMeans 更不容易漏掉真实近邻。

**对照组**

| 组别 | 目的 |
|---|---|
| Random balanced partition | 排除随机均衡分片偶然性 |
| Balanced KMeans only | 对照纯向量聚类 |
| KMeans + topology convergence | 隔离拓扑收敛作用 |
| KMeans + topology + load recalibration | 隔离负载校准作用 |
| Full Method4 with fission | 当前完整方法 |
| METIS partition, optional | 强图划分 baseline |
| Naive all-shards | 访问 46 shards 的召回上限参考 |

**离线 oracle 设置**

不跑 lower HNSW，只分析 query routing 与 ground truth 的 shard 关系。

```text
for each query q:
  U(q) = upper HNSW top upper_k labels
  R(q) = shards routed by point_to_shards[U(q)]
  G(q) = ground-truth top10 source ids
  S_g(q) = all lower shards containing copies of G(q)
```

**参数规模**

最低：

```text
5 partition methods * upper_k {80,120,160} = 15 oracle points
```

推荐：

```text
6 partition methods * upper_k {80,120,160,200} = 24 oracle points
```

在线验证最低：

```text
5 partition methods
* 1 selected upper_k
* base_ef {60,80}
* factor {6,8,10}
= 30 online tuning points
```

**必须记录**

| 指标 | 判读 |
|---|---|
| `avg |R(q)|` | 平均访问 shards，越低越好 |
| `P95 |R(q)|` | 访问 shards tail，越低越好 |
| `oracle_gt_coverage@10` | GT top10 至少一个 copy 在 routed shards 中，越高越好 |
| `oracle_gt_miss@10` | GT top10 全部 copies 都在未访问 shards 中，越低越好 |
| `gt_shard_entropy@10` | GT top10 shard 分布熵，越低越集中 |
| `min_shards_for_gt@10` | 覆盖 GT top10 所需最少 shard 数，越低越好 |
| `routed_waste_ratio` | routed shards 中没有 GT copy 的比例，越低越好 |
| `topology_edge_cut` | L1/topology/co-routing graph 跨 shard 边比例，越低越好 |

**支撑标准**

- Full Method4 的 `avg |R(q)|` 明显低于 naive 46；
- Full Method4 在相近 `avg |R(q)|` 下 `oracle_gt_coverage@10` 高于 random / KMeans；
- Full Method4 的 `oracle_gt_miss@10` 低于 random / KMeans；
- Full Method4 的 `gt_shard_entropy@10` 或 `topology_edge_cut` 优于 KMeans only；
- 在线同召回下 Full Method4 QPS/P95/P99 优于 KMeans only。

**实验记录**

```text
status:
result_dir:
best_supported_statement:
notes:
```

## Claim B: 多分配降低 routing miss

**要证明**

Voting multi-assignment 让边界点在多个 lower shards 中有 copy，从而减少“查询访问了较少 shards 但真实近邻全在其他 shards”的情况。

**对照组**

| 组别 | 目的 |
|---|---|
| Single assignment | 不做多分配 |
| Voting multi-assignment | 当前方法 |
| Aggressive multi-assignment, optional | 空间换召回上限 |

**参数规模**

离线 oracle 最低：

```text
2 assignment modes * upper_k {80,120,160} = 6 oracle points
```

在线最低：

```text
2 assignment modes
* 1 selected upper_k
* base_ef {60,80}
* factor {6,8,10}
= 12 online points
```

**必须记录**

| 指标 | 判读 |
|---|---|
| index expansion ratio | 多分配代价 |
| `oracle_gt_coverage@10` | 多分配是否提高 routed coverage |
| `oracle_gt_miss@10` | 多分配是否降低漏 shard 概率 |
| Recall@10 | 最终质量 |
| QPS / P95 / P99 | 代价是否可接受 |
| avg visited shards/query | 是否因为多分配导致访问爆炸 |

**支撑标准**

- Voting 显著降低 `oracle_gt_miss@10`；
- Voting 在相近 visited shards 下 Recall@10 高于 single assignment；
- index expansion ratio 明确可量化；
- QPS/P95 没有被索引膨胀完全抵消。

**实验记录**

```text
status:
result_dir:
best_supported_statement:
notes:
```

## Claim C: Dynamic EF 更高效地分配 lower search 预算

**要证明**

`routed_ep_count` 是 shard 重要性的有效信号；`base_ef + factor * routed_ep_count`
比所有 visited shards 使用固定 EF 更有效。

**对照组**

| 组别 | 目的 |
|---|---|
| Fixed EF | 每个 visited shard 同一 EF |
| Dynamic EF linear | 当前公式 |
| Dynamic EF capped, optional | 防止极端高 EF |
| Oracle EF bucket, optional | 只做上界分析，不作为可部署方法 |

**参数规模**

```text
upper_k {120,160,200}
fixed_ef {60,80,100,120,140}
= 15 fixed points

upper_k {120,160,200}
dynamic_base_ef {40,60,80}
dynamic_factor {4,6,8,10,12}
= 45 dynamic points
```

**必须记录**

| 指标 | 判读 |
|---|---|
| routed_ep_count | shard 重要性信号 |
| shard_has_gt_copy | routed_ep_count 是否关联真实候选 |
| shard_contributes_final_topk | routed_ep_count 是否关联最终结果 |
| assigned EF | dynamic EF 分布 |
| Recall@10 | 质量 |
| QPS / P95 / P99 | 性能 |
| estimated EF-sum/query | 搜索预算 |

**支撑标准**

- `routed_ep_count` 与 `shard_has_gt_copy` 或 `shard_contributes_final_topk` 正相关；
- dynamic EF 在同 Recall@10 下 QPS 高于 fixed EF；
- 或 dynamic EF 在同 QPS 下 Recall@10 高于 fixed EF；
- dynamic EF 不导致 P99 明显恶化。

**实验记录**

```text
status:
result_dir:
best_supported_statement:
notes:
```

## Claim D: 高召回同召回率下 Method4 比 naive 更快

**要证明**

Method4 通过 upper routing 访问更少 lower shards，并用 multi-EP/dynamic EF 保持召回；
因此同召回下 QPS 更高、tail latency 不差。

**对照组**

| 组别 | 目的 |
|---|---|
| Naive all-shards fixed EF | 主 baseline |
| Full Method4 routing | 主方法 |
| Method4 search-all-shards, optional | 拆分 Method4 index effect 与 routing effect |

**参数扫描**

Method4：

```text
upper_k {80,120,160,200}
base_ef {40,60,80,100}
factor {4,6,8,10,12}
batch_size {100,200}
```

Naive：

```text
ef {48,56,64,72,76,80,88,96,112,128}
batch_size {100,200}
```

正式目标点：

```text
Recall@10 ~= 0.90
Recall@10 ~= 0.95
Recall@10 ~= 0.97
Recall@10 ~= 0.99
```

正式最低：

```text
4 recall targets * 2 methods * 3 repeats = 24 official runs
```

**必须记录**

| 指标 | 判读 |
|---|---|
| Recall@10 | 必须同召回比较 |
| QPS mean/stddev | 主性能 |
| P50/P95/P99 | tail latency |
| avg visited shards/query | Method4 是否少访问 |
| avg EF per visited shard | dynamic EF 工作量 |
| estimated EF-sum/query | lower work 粗估 |
| distance computations/query, if available | 更强 lower work 证据 |

**支撑标准**

- Method4 与 naive Recall@10 差距在约 `0.003` 内，或用 Recall-QPS 曲线插值；
- Method4 avg visited shards/query 明显低于 46；
- Method4 QPS 高于 naive stable mean；
- Method4 P95/P99 不显著差于 naive，最好更低。

**实验记录**

```text
status:
result_dir:
best_supported_statement:
notes:
```

## Claim E: controller-native routing / compact request 降低调度开销

**要证明**

把 Method4 route plan 从 Python/JSON/shard-key string hot path 转到 server-side compact request 和 per-shard specialization，
可以减少调度开销，同时不改变 Method4 搜索语义。

**对照组**

| 组别 | 目的 |
|---|---|
| Python route plan / JSON hot path | 旧路径或模拟旧路径 |
| Compact multi-EP request | request 压缩 |
| Controller-native expansion | server-side per-shard specialization |

**参数规模**

固定高召回主配置，例如：

```text
upper_k = 160
base_ef = 80
factor = 8
batch_size {50,100,200,400}
```

最低：

```text
3 execution modes * 4 batch sizes * 3 repeats = 36 runs
```

**必须记录**

| 指标 | 判读 |
|---|---|
| route planning time/query | Python/JSON 是否是热点 |
| request bytes/query | compact request 是否减少 payload |
| RPC count/query | fanout 是否减少 |
| controller CPU | controller hot path |
| Recall@10 | 语义不变 |
| QPS / P95 / P99 | 性能收益 |

**支撑标准**

- Recall@10 基本不变；
- route planning time、request bytes 或 RPC count 明显下降；
- QPS/P95 有正收益；
- 如果收益很小，只写成工程优化，不写成核心贡献。

**实验记录**

```text
status:
result_dir:
best_supported_statement:
notes:
```

## Claim F: shard-major + worker-local peer pre-merge 契合分布式执行

**要证明**

Method4 的 lower search 是 many logical shards / few physical workers。
把 query-major fan-in 改为 worker-local pre-merge，可以减少 controller 收到的结果流并改善 tail latency。

**对照组**

| 组别 | 目的 |
|---|---|
| Logical-shard fan-out/fan-in | 旧执行形态 |
| Shard-major without peer pre-merge | 拆出 shard-major grouping |
| Shard-major + worker-local peer pre-merge | 当前方法 |

**参数规模**

固定高召回主配置，改变：

```text
batch_size {50,100,200,400}
```

最低：

```text
3 execution variants * 4 batch sizes * 3 repeats = 36 runs
```

可选 worker scaling：

```text
3 execution variants * workers {1,2,3} * 1 selected batch * 3 repeats = 27 runs
```

**必须记录**

| 指标 | 判读 |
|---|---|
| physical peers visited/query | peer 级 fanout |
| controller result streams/query | pre-merge 是否减少 fan-in |
| per-worker local shard searches/query | worker 内工作量 |
| worker local merge time | worker 额外成本 |
| controller final merge time | controller 是否减负 |
| Recall@10 | 语义不变 |
| QPS / P95 / P99 | 性能收益 |

**支撑标准**

- Recall@10 基本不变；
- controller input streams 从 logical-shard 级降到 physical-peer 级；
- QPS 或 P95/P99 稳定改善；
- same-image A/B 有正收益。

**实验记录**

```text
status:
result_dir:
best_supported_statement:
notes:
```

## Claim G: Method4-aware physical layout 降低 hot worker

**要证明**

Method4 routing 有 co-routed shard 集合，因此 physical placement 应考虑 routing trace，而不是只 round-robin。
如果 online 结果不明显，这个 claim 降级为部署策略。

**对照组**

| 组别 | 目的 |
|---|---|
| Round-robin placement | 简单 baseline |
| Size-balanced placement | 只按 shard size 平衡 |
| Method4-aware co-routing placement | 当前方法 |
| Hot-replica, optional | 热 shard 副本扩展 |

**参数规模**

离线模拟：

```text
workers {2,3,4}
placement {round_robin,size_balanced,method4_aware}
= 9 simulations
```

在线最低：

```text
3 placements * batch_size {100,200} * 3 repeats = 18 runs
```

**必须记录**

| 指标 | 判读 |
|---|---|
| worker load CV | worker 均衡 |
| P95 worker load/query | tail worker load |
| max worker routed shards/query | hot worker |
| co-routed shard cut | co-routing 是否被均衡放置 |
| QPS / P95 / P99 | 在线收益 |
| Recall@10 | 应保持不变 |

**支撑标准**

- method4-aware placement 降低 worker load CV 或 P95 worker load；
- 在线 P95/P99 改善或 QPS 不下降；
- 如果只有离线改善，不能强写性能 claim。

**实验记录**

```text
status:
result_dir:
best_supported_statement:
notes:
```

## Claim H: 实现保持原初 Method4 外部语义

**要证明**

HNSW 内部实现可以不同，但外部调度语义与原初 C++ method4 一致。

**抽样规模**

```text
100 queries * selected high-recall config
```

**必须检查**

| 不变量 | 检查方式 |
|---|---|
| upper routing unchanged | upper labels 与 route plan 输入一致 |
| `point_to_shards` unchanged | 同一 upper label 映射到同一 shard set |
| multi-assignment preserved | source id 多 copy 可被路由到 |
| per-shard entry points preserved | 每个 lower shard 收到自己的 EP list |
| dynamic EF formula preserved | `ef = base_ef + factor * routed_ep_count` |
| lower search remains per logical shard | trace 显示每个 logical shard 单独 HNSW search |
| no adaptive shard pruning | routed shard set 与 executed shard set 一致 |
| source-id dedup preserved | copy ids merge 回 source id |
| final global merge preserved | controller 做最终 global top-k |

**支撑标准**

所有不变量必须通过。任何破坏不变量的优化，即使 QPS 提升，也不能作为 faithful Method4 结果。

**实验记录**

```text
status:
trace_dir:
violations:
decision:
```

## 最小投稿实验包

如果时间紧，按这个顺序做：

1. **Claim A offline oracle**：先证明 Method4 partition 的 routing locality 优于 random/KMeans。
2. **Claim D same-recall main comparison**：Recall@10 约 0.95 和 0.97 至少两个点。
3. **Claim B assignment ablation**：single vs voting multi-assignment。
4. **Claim C EF ablation**：fixed EF vs dynamic EF。
5. **Claim F execution ablation**：peer pre-merge same-image A/B。
6. **Claim H semantic audit**：避免被质疑偏离原始 Method4。

Claim E 和 Claim G 有稳定正收益再写成强贡献；否则作为工程优化或部署策略。

## 每组实验统一记录模板

```text
claim:
experiment_name:
date:
commit:
docker_image:
collection:
dataset:
query_count:
repeats:
params:
baseline:
result_dir:

Recall@10:
QPS_mean:
QPS_stddev:
P50:
P95:
P99:
avg_visited_shards:
avg_ef_per_visited_shard:
estimated_ef_sum:
index_expansion_ratio:
rpc_per_query:
controller_streams_per_query:

decision: support / weak_support / reject
paper_statement_allowed:
notes:
```

## 结果写作规则

- 只有 same-recall 结果可以支撑主性能 claim。
- 只有 oracle locality 优于 KMeans/random，才能强写“拓扑感知划分更好”。
- 只有 ablation 有正收益，才能把该机制写成贡献。
- 只有 semantic audit 通过，才能说 faithful Method4 implementation。
- 对没有稳定正收益的机制，不要包装成主要创新。
