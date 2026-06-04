# 2026-06-04 当前 Method4 Idea 的 Docker 分布式架构设计

## 目标

当前 idea 实现的目标，是把原初 C++ method4 的两层图索引思想迁移到
Qdrant 的 Docker 分布式架构中，同时保持 method4 的外部调度语义不变：

- 上层 HNSW 负责路由；
- 上层候选 label 通过 `point_to_shards` 映射到 lower logical shards；
- lower tier 仍然按逻辑 shard 分别执行 HNSW 搜索；
- 每个 lower shard 使用自己的入口点和动态 `hnsw_ef`；
- 多分配、source-id 去重、最终全局 top-k 合并都保留。

HNSW 内部图构建不追求与原始 C++ bit-level 一致。只要求它是标准 HNSW，
并且对 method4 外部包装层暴露出一致的调用语义。

## Docker 拓扑

当前运行形态是单机多容器的 controller + worker 集群：

| 角色 | 容器服务 | HTTP | gRPC | 说明 |
|---|---|---:|---:|---|
| Controller | `qdrant_controller` | `6833` | `6834` | 接收客户端查询，执行上层路由后的分布式 fan-out 和最终合并 |
| Worker 1 | `qdrant_shard_1` | `6843` | `6844` | 持有一部分 method4 lower logical shards |
| Worker 2 | `qdrant_shard_2` | `6853` | `6854` | 持有一部分 method4 lower logical shards |
| Worker 3 | `qdrant_shard_3` | `6863` | `6864` | 持有一部分 method4 lower logical shards |

对应 compose 文件：

```text
tools/compose/docker-compose.controller-cluster.yaml
```

当前实验镜像：

```text
qdrant/qdrant:method4-peer-premerge
```

当前主力 idea collection：

```text
qdrant_controller_idea_method4map_full_20260601
```

该 collection 使用 Qdrant custom sharding，每个 method4 lower shard 对应一个
custom shard key，物理上由 3 个 worker 分摊承载。controller 本身不持有该
collection 的本地 lower shard。

## Lower Tier 数据布局

lower tier 不是普通的按 hash 均匀切分，而是 method4 的逻辑分片结果：

1. 从训练集抽取全局 upper sample，构建 upper HNSW。
2. 对全量点查询 upper HNSW，得到每个点的 `K_OVERLAP` 个 L1 入口候选。
3. 按原初 method4 路径执行 L1 初始划分、自然拓扑收敛、负载重校准和 fission。
4. 对全量数据执行 L1 投票多分配，得到 `point_to_shards`。
5. 按 `point_to_shards` 把点写入对应 lower logical shard。

当前部署的关键形态：

| 项目 | 值 |
|---|---:|
| 原始训练点数 | 1,183,514 |
| method4 indexed vectors | 1,400,967 |
| lower logical shards | 46 |
| segments | 92 |
| HNSW `m` | 32 |
| HNSW `ef_construct` | 100 |
| 距离 | Cosine |

`1,400,967` 大于原始训练点数，是因为 method4 多分配会为一个 source point
创建多个 lower-tier copies。这是 idea 的算法特性，不是额外改变 HNSW 构建
参数。

为了支持最终结果去重，copied point ID 使用 block 编码：

```text
copy_id = shard_id * source_id_dedup_block_size + source_id + 1
```

最终合并时通过 `source_id_dedup_block_size` 把多个 copies 映射回同一个
source-id 去重键。

## 物理放置

method4 lower logical shards 不再简单 round-robin 放置，而是使用
method4-aware placement map。该 map 来自真实 method4 routing trace，目标是让
高频 co-routed 的 logical shards 在 3 个 worker 上更均衡。

当前 placement 来源：

```text
results/qdrant_goal_recall_idea_095_placement_simulation/20260601_075041/placement_simulation.json
```

当前物理 shard 数分布：

```text
Worker peer A: 15 shards
Worker peer B: 15 shards
Worker peer C: 16 shards
```

这一步只改变 logical shard 到 physical worker 的归属，不改变 method4 的划分、
路由、入口点、动态 EF 或 lower HNSW 搜索语义。

## 查询路径

当前查询仍然从 method4 的 upper routing 开始：

```text
query vector
  -> upper HNSW search, k = upper_k
  -> upper labels
  -> point_to_shards[label]
  -> shard_to_eps: shard -> routed entry labels
  -> per-shard dynamic EF: base_ef + factor * routed_ep_count
  -> lower logical shard searches
  -> source-id dedup
  -> global top-k
```

当前高召回主力参数：

```text
upper_k = 160
base_ef = 80
factor = 8
top_k = 10
batch_size = 200
routed_execution_mode = compact_multi_ep
routed_planning_mode = materialized
routed_result_limit_mode = top_k
```

`compact_multi_ep` 的含义是：客户端/benchmark 对每个 query 发出一个 compact
search request，但 request 内部携带两个 per-shard map：

```text
hnsw_entry_points_by_shard: shard_key -> entry point labels
hnsw_ef_by_shard:           shard_key -> dynamic hnsw_ef
```

因此外部请求数量被压缩，但 method4 的 per-shard 入口点和 per-shard EF 没有
被合并成一个全局值。

## Controller Native Shard-Major 执行

Qdrant controller 收到 compact method4 request 后，不让它以普通 multi-shard
搜索的方式丢失 per-shard 语义，而是在 server 内部展开为 shard-major lower
execution：

1. 识别 request 中的 `hnsw_entry_points_by_shard` 和 `hnsw_ef_by_shard`。
2. 对每个 routed shard 生成一个已经特化的 `CoreSearchRequest`。
3. 特化时写入该 shard 的 `hnsw_entry_points` 和 `params.hnsw_ef`。
4. 清空 per-shard maps，避免 lower shard 再次解释全局 routing 信息。
5. 每个 lower logical shard 单独执行 HNSW 搜索。

核心语义是：

```text
compact external request
  -> server-side per-shard specialization
  -> per logical shard HNSW search
```

这一步消除了 Python route plan / JSON map 在 lower execution 上的部分开销，
但不会把多个 lower shards 合成一次共享 HNSW 搜索。

相关实现入口：

```text
src/common/query.rs
lib/collection/src/collection/search.rs
```

## Worker-Local Peer Pre-Merge

高召回下，method4 一个 query 平均访问约 23 个 lower logical shards，但这些
logical shards 最终通常只落在 3 个 physical workers 上。旧路径中，controller
会收到每个 logical shard 的一条结果流；当前实现把这一步改成 worker-local
pre-merge：

```text
旧路径:
  controller <- shard_00 result
  controller <- shard_01 result
  ...
  controller <- shard_N result

当前路径:
  worker A: merge its local shard results for query -> one partial result
  worker B: merge its local shard results for query -> one partial result
  worker C: merge its local shard results for query -> one partial result
  controller: merge physical peer partial results -> final top-k
```

新增 internal gRPC：

```text
PointsInternal/CoreSearchBatchByShard
CoreSearchBatchByShardInternal
CoreSearchByShardEntry
```

worker 侧处理流程：

1. 按 `shard_id` group entries。
2. 对每个 `shard_id` 调用 `toc.core_search_batch(... ShardSelectorInternal::ShardId(shard_id))`。
3. 收集每个 query 的 per-shard rows。
4. worker 本地预合并到 `limit + offset`。
5. 返回每个 query 在该 physical peer 上的一条 partial result。

controller 侧最终仍然使用原有全局 merge 逻辑。worker 本地 `limit + offset`
预合并不是 adaptive shard pruning；它只压缩已经产生的本地 per-shard 结果流。
被丢弃的本地尾部候选，在同一个 worker 内已经被至少 `limit + offset` 个更强
的去重键支配，不可能进入最终全局 top window。

运行时回退开关：

```bash
QDRANT_DISABLE_SHARD_MAJOR_PEER_PREMERGE=1
```

## 语义不变边界

当前 Docker 分布式实现刻意保持以下 method4 核心不变：

- 不改变 upper HNSW routing 的 `upper_k` 语义；
- 不改变 `point_to_shards`；
- 不改变 multi-assignment；
- 不改变每个 lower shard 的 routed entry points；
- 不改变动态 EF 公式：`base_ef + factor * routed_ep_count`；
- 不把多个 lower logical shards 合并成一个 HNSW 搜索；
- 不做 adaptive shard pruning；
- 不跳过 source-id 去重；
- 不替换 controller 的最终 global top-k merge。

允许变化的是分布式执行形态：

- compact request 减少外部请求对象；
- server-side shard-major 展开避免客户端发 N 个 shard requests；
- worker-local peer pre-merge 降低 controller fan-in；
- method4-aware placement 改善 logical shard 到 physical worker 的负载映射。

## 当前性能位置

当前同召回对比中，method4 idea 相比 naive all-shards 有稳定优势：

| 实现 | Recall@10 | QPS mean | Avg visited shards/query |
|---|---:|---:|---:|
| method4 current | 0.955267 | 387.140 | 23.214 |
| naive closest recall | 0.954767 | 272.370 | 46.000 |
| naive higher recall | 0.957333 | 264.645 | 46.000 |

主要收益来自两点：

1. method4 在高召回下仍只访问约一半 lower logical shards；
2. 当前 Docker 架构把 method4 的 many-logical-shards/few-physical-workers 特征
   映射成 worker-local pre-merge，减少 controller 端结果流和候选流压力。

## 一句话总结

当前 idea 的 Docker 分布式架构，可以理解为：

```text
原初 method4 的两层路由/分片算法
  + Qdrant custom shard lower HNSW
  + server-side shard-major per-shard specialization
  + method4-aware physical placement
  + worker-local peer pre-merge
```

其中 algorithm core 仍然是 method4；系统优化只负责让 Docker/Qdrant 的分布式
执行形态更贴合 method4 的访问模式。
