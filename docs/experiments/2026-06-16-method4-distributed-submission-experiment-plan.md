# 2026-06-16 Method4 Distributed Qdrant 投稿实验计划

## 目标

本实验计划用于支撑当前 Method4 idea 在 Docker/Qdrant 分布式架构下的投稿版本。
核心目标不是单纯证明某个参数点 QPS 更高，而是建立一条完整证据链：

```text
Method4 的 upper routing / multi-assignment / per-shard entry points / dynamic EF
  -> 减少高召回下的无效 lower logical shard search
  -> 分布式实现通过 controller-native routing、compact request、
     shard-major lower execution、worker-local peer pre-merge 把算法优势转化为
     QPS 和 tail-latency 收益
  -> 在同构建参数、同召回水平和可解释成本下优于 naive distributed Qdrant
```

所有实验必须区分三类事实：

- **算法语义事实**：是否保持原初 C++ method4 的外部包装语义；
- **系统实现事实**：Qdrant 分布式执行路径是否减少 routing、RPC、fan-in 和合并开销；
- **性能事实**：在同召回水平下是否稳定提升 QPS / latency。

## Claim 到实验的证据矩阵

投稿时不要只写“我们的方法更快”，而要把主张拆成可以被反事实实验检验的
claim。每个 claim 必须有对应 baseline、参数控制、观测量和判定标准。

### Claim A: Method4 的划分方式保留了图拓扑局部性，因此比 naive / 纯 KMeans 更适合路由

**论文表述**

Method4 的 L1 划分不是普通向量聚类，也不是 hash / range 式 naive 分片。它先用
upper HNSW 建立 L1 backbone，再通过 balanced KMeans 初始化、L1 topology
convergence、voting load recalibration 和 fission，让经常在 upper graph 或
upper routing 中共同出现的 L1 入口更倾向落在相近的 lower logical shards。因此，
在线搜索时更少访问 shard，同时 ground-truth 近邻更少散落在未访问 shard 中。

**需要对照的方法**

| 方法 | 作用 |
|---|---|
| Naive all-shards | 上限式 baseline，永远访问全部 46 shards，召回不受路由漏掉 shard 影响 |
| Random balanced partition | 排除“只是 shard 数一样”带来的偶然收益 |
| Balanced KMeans only | 检验纯向量聚类是否已经足够 |
| KMeans + L1 topology convergence | 隔离 topology convergence 的贡献 |
| KMeans + topology + voting load recalibration | 检验负载校准对热点和真实挂载的影响 |
| Full Method4 partition with fission | 当前完整 idea |
| METIS partition, if available | 可选强 baseline，检验显式图划分是否优于 Method4 启发式 |

所有对照必须使用同一份 train/test/ground-truth、同一 upper sample、同一
upper HNSW、同一 `K_OVERLAP` 和同一 lower HNSW 构建参数。对照之间只改变
`L1 -> shard` 和 `point_to_shards` 生成方式。

**离线 oracle 实验设置**

对每个 query，先不真正执行 lower HNSW，而是只分析 routing 与 ground-truth 的
shard 关系：

```text
for each query q:
  U(q) = upper HNSW top upper_k labels
  R(q) = shards routed by point_to_shards[U(q)]
  G(q) = ground-truth top-k source ids
  S_g(q) = all lower shards containing any copy of each g in G(q)
```

需要统计：

| 指标 | 含义 |
|---|---|
| `avg |R(q)|` | 平均访问 lower shards 数，越低越好 |
| `P95 |R(q)|` | 路由访问 shard 的 tail，越低越好 |
| `oracle_gt_coverage@10` | ground-truth top10 中至少一个 copy 落在 `R(q)` 的比例，越高越好 |
| `oracle_gt_miss@10` | ground-truth top10 全部 copies 都在未访问 shards 的比例，越低越好 |
| `min_shards_for_gt@10` | 覆盖 ground-truth top10 所需的最少 shard 数，越低表示近邻更集中 |
| `gt_shard_entropy@10` | ground-truth top10 在 shard 上的分布熵，越低表示分布更集中 |
| `routed_waste_ratio` | 访问 shard 中没有任何 ground-truth top10 copy 的比例，越低越好 |
| `topology_edge_cut` | upper graph / L1 co-routing graph 中跨 shard 边的加权比例，越低越好 |

**参数组数**

固定 lower logical shards 为 46，至少比较 5 组 partition：

```text
random balanced
balanced KMeans only
KMeans + topology convergence
KMeans + topology + load recalibration
full Method4 with fission
```

每组在 3 个 `upper_k` 下做 oracle routing：

```text
upper_k in {80, 120, 160}
```

如果算力允许，再加：

```text
upper_k in {200}
METIS partition
```

最低实验量：

```text
5 partitions * 3 upper_k = 15 offline oracle points
```

推荐实验量：

```text
6 partitions * 4 upper_k = 24 offline oracle points
```

**在线验证设置**

在 oracle 指标最有区分度的 `upper_k` 上，为每种 partition 构建 lower shards 并跑
真实 HNSW search。为了公平，每组使用相同：

```text
M = 32
ef_construct = 100
base_ef in {60, 80}
factor in {6, 8, 10}
top_k = 10
batch_size in {100, 200}
```

最低在线参数量：

```text
5 partitions * 1 upper_k * 2 base_ef * 3 factor = 30 points
```

如果成本过高，可以先用 tuning queries 选出每组接近 Recall@10 ~= 0.95 的一个
配置，再对每组做 3000 或 10000 queries 的正式评估。

**支撑该 claim 的结果形态**

Full Method4 不一定要在所有 oracle 指标上第一，但必须表现出：

- `avg |R(q)|` 明显低于 naive all-shards；
- 在相同或更少 `avg |R(q)|` 下，`oracle_gt_coverage@10` 高于 random / KMeans only；
- `oracle_gt_miss@10` 低于 random / KMeans only；
- `topology_edge_cut` 或 `gt_shard_entropy@10` 低于 KMeans only；
- 在线同召回下 QPS / P95 / P99 优于 KMeans only 和 naive。

如果 Full Method4 只比 naive 快，但不比 KMeans only 的 oracle locality 更好，则
不能强 claim “拓扑划分带来局部性优势”，只能 claim “routing 减少访问 shard”。

### Claim B: 多分配减少 routing miss，使高召回下少访问 shard 仍可保持召回

**论文表述**

Method4 的 voting multi-assignment 通过把边界点写入多个 lower logical shards，
降低了 ground-truth 近邻全部落在未访问 shard 中的概率。因此它能在访问较少
shard 的同时保持高召回。

**需要对照的方法**

| 方法 | 作用 |
|---|---|
| Single assignment | 每个点只进入主 shard，测试不多分配时的 routing miss |
| Full voting multi-assignment | 当前方法 |
| Aggressive multi-assignment | 可选，提高重叠阈值，观察空间换召回上限 |

**实验设置**

固定 full Method4 L1 shard map，只改变 final point assignment：

```text
assignment_mode in {single, voting_multi, aggressive_multi_optional}
upper_k in {80, 120, 160}
base_ef in {60, 80}
factor in {6, 8, 10}
```

至少记录：

- index expansion ratio；
- `oracle_gt_coverage@10`；
- `oracle_gt_miss@10`；
- Recall@10；
- QPS；
- avg visited shards/query。

**最低实验量**

```text
2 assignment modes * 3 upper_k = 6 oracle points
2 assignment modes * 1 selected upper_k * 2 base_ef * 3 factor = 12 online points
```

**支撑该 claim 的结果形态**

Voting multi-assignment 应该：

- 显著降低 `oracle_gt_miss@10`；
- 在相近 visited shards 下提高 Recall@10；
- 付出的 index expansion ratio 可被量化；
- QPS 不因索引膨胀完全抵消收益。

### Claim C: Dynamic EF 把 lower search 预算分配到更可能有结果的 shards，因此比固定 EF 更高效

**论文表述**

在 Method4 routing 中，不同 lower shard 收到的 routed entry point 数不同。
`base_ef + factor * routed_ep_count` 用 routed EP count 作为该 shard 重要性的轻量
估计，把更大搜索预算分配给更可能包含近邻的 shard。

**需要对照的方法**

| 方法 | 作用 |
|---|---|
| Fixed EF, same for all visited shards | 基础对照 |
| Dynamic EF linear formula | 当前方法 |
| Dynamic EF capped | 可选，限制极端 shard 的 EF |
| Oracle EF bucket | 可选，用 ground-truth shard hit 作为上界分析，不作为可部署方法 |

**实验设置**

固定 partition、assignment、upper routing，只改变 per-shard EF：

```text
upper_k in {120, 160, 200}
fixed_ef in {60, 80, 100, 120, 140}
dynamic_base_ef in {40, 60, 80}
dynamic_factor in {4, 6, 8, 10, 12}
```

同时统计每个 visited shard：

- `routed_ep_count`；
- shard 是否包含 ground-truth top10 copy；
- shard 返回结果中是否进入 final top10；
- dynamic EF 值；
- lower search latency。

**最低实验量**

```text
3 upper_k * 5 fixed_ef = 15 fixed points
3 upper_k * 3 base_ef * 5 factor = 45 dynamic points
```

如果成本过高，先用 500 tuning queries 筛选，再对候选点做 3000 query 正式评估。

**支撑该 claim 的结果形态**

需要证明：

- `routed_ep_count` 与“该 shard 含有 ground-truth copy / 产生 final top-k”的概率正相关；
- dynamic EF 在同 Recall@10 下 QPS 高于 fixed EF；
- 或者在同 QPS 下 Recall@10 高于 fixed EF；
- dynamic EF 没有导致 P99 明显恶化。

### Claim D: Method4 在高召回下比 naive 搜索更少 lower shards，因此同召回 QPS 更高

**论文表述**

高召回场景下 naive all-shards 需要对全部 lower logical shards 提高 EF，而 Method4
通过 upper routing 只访问一部分 lower shards，并通过 multi-EP 和 dynamic EF 保持
召回，因此同召回下 lower work 更少、QPS 更高。

**需要对照的方法**

| 方法 | 作用 |
|---|---|
| Naive all-shards, fixed EF | 主 baseline |
| Method4 full routing | 主方法 |
| Method4 search-all-shards | 可选，保留 Method4 index 但搜索全部 shards，用于拆分 index effect 和 routing effect |

**实验设置**

Method4:

```text
upper_k in {80, 120, 160, 200}
base_ef in {40, 60, 80, 100}
factor in {4, 6, 8, 10, 12}
batch_size in {100, 200}
```

Naive:

```text
ef in {48, 56, 64, 72, 76, 80, 88, 96, 112, 128}
batch_size in {100, 200}
```

目标召回：

```text
Recall@10 ~= 0.90
Recall@10 ~= 0.95
Recall@10 ~= 0.97
Recall@10 ~= 0.99
```

**最低实验量**

先用 tuning queries 扫描全部参数，再选每个 recall target 的 closest stable point。
正式评估至少：

```text
4 recall targets * 2 methods * 3 repeats = 24 official runs
```

**支撑该 claim 的结果形态**

在同 recall 邻域内：

- Method4 `avg visited shards/query` 明显低于 46；
- Method4 QPS 高于 naive；
- Method4 P95/P99 不显著差于 naive，最好更低；
- estimated EF-sum/query 或实际 distance computations/query 不高于 naive 太多。

### Claim E: controller-native routing 和 compact multi-EP request 消除了分布式调度开销，但不改变搜索语义

**论文表述**

Method4 的原始语义需要为每个 query 构造 `shard -> entry points` 和
`shard -> hnsw_ef`。如果这些映射长期停留在 Python route plan / JSON map /
shard-key string fanout 中，分布式开销会吞掉算法收益。controller-native routing
和 compact request 把同样的 per-shard 语义移入 server hot path。

**需要对照的方法**

| 方法 | 作用 |
|---|---|
| Python route plan / materialized JSON hot path | 旧路径或模拟旧路径 |
| Compact multi-EP request | 当前 request compaction |
| Controller-native expansion | 当前 server-side per-shard specialization |

**实验设置**

固定 Method4 参数：

```text
upper_k = selected high-recall value, e.g. 160
base_ef = selected value, e.g. 80
factor = selected value, e.g. 8
batch_size in {50, 100, 200, 400}
```

记录：

- route planning time/query；
- request serialization bytes/query；
- controller CPU；
- RPC count/query；
- QPS；
- P95/P99；
- Recall@10。

**最低实验量**

```text
3 execution modes * 4 batch sizes * 3 repeats = 36 runs
```

如果旧路径已经不可运行，可以做 microbenchmark + end-to-end simulation，但正文中
必须标清“模拟调度开销”。

**支撑该 claim 的结果形态**

- Recall@10 基本不变；
- request size、route planning time 或 RPC count 明显下降；
- QPS / P95 有正收益；
- 若收益很小，只能作为 engineering cleanup，不能作为主要贡献。

**Compact internal wire 的版本控制**

Claim E/F 的后续四节点实验必须把 internal compact wire 作为显式实验身份记录，不能只写
“compact enabled”。当前协议规则如下：

| 字段 | wire v1 | wire v2 |
|---|---|---|
| 版本选择 | controller 未设置 `QDRANT_ORION_COMPACT_WIRE_VERSION`，或显式为 `1` | controller 显式设置为 `2` |
| query template | nested `search_points` | controller 生产 encoded bytes；worker 接受 nested/encoded 严格二选一 |
| numeric MultiEP | 通用 `PointId` | packed `uint64`，顺序不变 |
| UUID MultiEP | 通用 `PointId` | 该 shard 整组回退通用 `PointId`，不报错、不裁剪 |

v1 携带 packed/encoded v2 字段、v2 同时携带两种 query 或 EP 表示、空/损坏/超过
16 MiB 的 encoded template，以及未知 wire version 都必须 fail closed。controller v2
对每个原始 query 只 encode 一次，并以引用计数 `Bytes` clone 跨 peer、RPC shard chunk
和 outer retry 复用；这只定义 controller 内存/序列化准备路径，不预设端到端性能结果。

该开关只存在于 controller。workers 从 RPC envelope 读取版本，容器不得携带 controller
wire env/label，但四节点 binary 必须来自同一个支持该协议的 image digest。运行 manifest
必须同时记录 requested/current wire version、image ID/digest、peer-premerge mode 和
shards-per-RPC。正式 v1/v2 或 transport A/B 还必须核对 Recall、ordered result IDs、
float32 score bytes、visited shards 和 EF-sum，不能用 RPC 数下降代替语义等价或性能证明。
支持能力由 image label `org.qdrant.orion.compact_wire.max_version=2` 绑定；controller 和
三个 workers 的 post-start inspect 都必须证明该 capability。没有 label 的旧 image 只能
作为 v1-capable，不能仅靠 controller runtime env/label 宣称为 v2。

当前 lifecycle 不支持 rolling wire upgrade，也没有 v2 到 v1 的协商降级。新的 v2 image
先生成 immutable candidate manifest，再以 offline transition 重建四个 run-scoped
containers 并保留 storage；forward 健康/placement 校验失败时，rollback 必须恢复 active
image 及其原 wire identity。只有 image ID 和 wire version 都相同才可视为 no-op；相同
image ID 但 wire identity 不同仍必须重建四节点。

### Claim F: shard-major lower execution 和 worker-local peer pre-merge 更契合 Method4 的分布式形态

**论文表述**

Method4 在线阶段是 query 路由到多个 lower logical shards，但这些 logical shards
物理上集中在少量 worker 上。相比 controller 接收每个 logical shard 的结果流，
worker-local pre-merge 先在 physical peer 内合并本地 shard streams，再由
controller 做 global merge，减少 fan-in 和 tail latency。

**需要对照的方法**

| 方法 | 作用 |
|---|---|
| Logical-shard fan-out/fan-in | 旧分布式执行形态 |
| Shard-major execution without peer pre-merge | 拆出 shard-major grouping 的作用 |
| Shard-major + worker-local peer pre-merge | 当前方法 |

**实验设置**

固定高召回主配置，并改变：

```text
batch_size in {50, 100, 200, 400}
workers in {1, 2, 3, 4 optional}
```

记录：

- controller received result streams/query；
- physical peers visited/query；
- per-worker local shard searches/query；
- worker local merge time；
- controller final merge time；
- QPS；
- P95/P99；
- Recall@10。

**最低实验量**

```text
3 execution variants * 4 batch sizes * 3 repeats = 36 runs
```

如果做 worker scaling：

```text
3 execution variants * 3 worker counts * 1 selected batch * 3 repeats = 27 runs
```

**支撑该 claim 的结果形态**

- peer pre-merge 不改变 Recall@10；
- controller input streams/query 从 logical-shard 级降低到 physical-peer 级；
- QPS 或 P95/P99 稳定改善；
- same-image A/B 有正收益。

### Claim G: Method4-aware physical layout 降低 hot worker 和 tail latency

**论文表述**

因为 Method4 routing 会产生 co-routed shard 集合，physical placement 不应只按
logical shard 数做简单 round-robin，而应该考虑 co-routing 热度，让常被同时访问
的 shard 在 worker 间更均衡，降低 hot worker 和 tail latency。

**需要对照的方法**

| 方法 | 作用 |
|---|---|
| Round-robin placement | 简单 physical baseline |
| Size-balanced placement | 只按 shard size 平衡 |
| Method4-aware co-routing placement | 当前方法 |
| Hot-replica optional | 如果实现 replica，再作为扩展 |

**实验设置**

先从 routing trace 构建 `query -> routed_shards`，离线模拟 placement：

```text
workers in {2, 3, 4}
placement in {round_robin, size_balanced, method4_aware}
```

离线指标：

- max worker routed shards/query；
- worker load CV；
- P95 worker load/query；
- co-routed shard cut across workers；
- predicted fan-in balance。

在线验证选择 3-worker 和最优/最差两个 placement：

```text
upper_k = selected
base_ef = selected
factor = selected
batch_size in {100, 200}
3 repeats
```

**最低实验量**

```text
3 placements * 3 worker counts = 9 offline simulations
3 placements * 2 batch sizes * 3 repeats = 18 online runs
```

**支撑该 claim 的结果形态**

- method4-aware placement 降低 worker load CV 或 P95 worker load；
- 在线 P95/P99 改善；
- QPS 不下降；
- 如果只有离线改善而在线无收益，则不能作为主要性能 claim，只能作为部署策略。

### Claim H: 系统优化保持原始 Method4 外部包装语义

**论文表述**

HNSW 内部实现可以不同，但 Method4 的外部调度和包装语义保持一致。当前分布式
实现不是一个新算法，而是原初 C++ method4 的分布式系统化实现。

**实验和审计设置**

建立 semantic invariant checklist，并对每个 query / shard 采样 trace：

| 不变量 | 验证方式 |
|---|---|
| upper routing unchanged | 记录 upper labels，与 route plan 输入一致 |
| `point_to_shards` unchanged | 同一 upper label 映射到同一 shard set |
| multi-assignment preserved | source id 有多个 copy 时均可被路由到 |
| per-shard entry points preserved | 每个 lower shard 收到自己的 EP list |
| dynamic EF formula preserved | `ef = base_ef + factor * routed_ep_count` |
| lower search remains per logical shard | trace 中每个 logical shard 单独 HNSW search |
| no adaptive shard pruning | routed shard set 与执行 shard set 一致 |
| source-id dedup preserved | copy ids merge 回 source id |
| final global merge preserved | controller 执行最终 top-k merge |

至少抽样：

```text
100 queries * selected high-recall config
```

若可行，补 C++ reference 对比：

- shard size distribution；
- index expansion ratio；
- avg visited shards/query；
- recall trend。

**支撑该 claim 的结果形态**

所有 semantic invariants 必须通过。任何破坏不变量的优化，即使 QPS 提升，也不能
作为 Method4 faithful implementation 的结果。

## Claim 优先级

投稿时建议按下面优先级组织贡献：

| 优先级 | Claim | 是否必须 |
|---|---|---|
| 1 | Claim A: topology-aware partition improves locality | 必须 |
| 2 | Claim D: same-recall QPS / latency improves over naive | 必须 |
| 3 | Claim B: multi-assignment reduces routing miss | 必须 |
| 4 | Claim C: dynamic EF improves budget allocation | 必须 |
| 5 | Claim F: shard-major + peer pre-merge matches distributed shape | 必须 |
| 6 | Claim H: Method4 semantic fidelity | 必须 |
| 7 | Claim E: controller-native routing / compact request reduces overhead | 强烈建议 |
| 8 | Claim G: physical layout reduces hot worker | 有收益再写成强 claim |

## 固定实验条件

除非某个实验明确改变变量，以下条件必须固定：

| 项目 | 固定值 / 要求 |
|---|---|
| 数据集 | 至少 `glove-200-angular.hdf5`，投稿版建议再加一个 L2 数据集 |
| 距离 | GloVe 使用 Cosine；L2 数据集使用 L2 |
| lower HNSW `m` | 32 |
| lower HNSW `ef_construct` | 100 |
| upper sample | `N / 32` |
| lower logical shards | 当前主线为 46；扩展性实验单独改变 |
| Qdrant topology | controller + physical workers |
| replication factor | 1，除非做 replica 实验 |
| query top-k | 10 |
| eval query count | 主结果至少 3000；最终确认建议 10000 |
| repeats | 每个关键点至少 3 次稳定性重复 |
| warmup | 每个配置正式计时前跑固定 warmup queries |
| commit / image | 同一组对比必须使用同一 commit 或明确记录 image digest |

当前已有主线结果可以作为计划起点，但投稿数据需要在最终 commit/image 上重跑：

| 配置 | Recall@10 | QPS mean | 说明 |
|---|---:|---:|---|
| Method4 current, `upper_k=160, base_ef=80, factor=8, batch=200` | 0.955267 | 387.140 | 当前 same-recall 主结果 |
| Naive closest recall, `ef=76, batch=100` | 0.954767 | 272.370 | 稳定 naive 对照 |
| Worker-local peer pre-merge enabled | 0.955300 | 405.381 | same-image A/B enabled |
| Worker-local peer pre-merge disabled | 0.955233 | 379.422 | same-image A/B disabled |

这些数值只能在文中标为当前实现结果；最终投稿表格必须使用最终实验批次。

## Baselines

### B1: Naive distributed Qdrant, all lower shards

这是最重要 baseline。它使用相同 lower-tier HNSW 构建参数和相同 46 logical
custom shards，但查询时访问全部 lower shards。

目的：

- 证明 Method4 的 upper routing 在高召回下减少无效 lower search；
- 提供同召回水平的 QPS / latency 对照。

### B2: Vanilla Qdrant / ordinary custom-shard search

用于说明当前系统不是只相对一个人为弱化的 naive baseline 有优势。该 baseline
可以是普通 Qdrant distributed search 或不带 Method4 routing 的 custom shard
配置。

目的：

- 给出工业系统默认路径的参考；
- 帮助解释 Method4 自定义 routing 的额外收益和代价。

### B3: Original C++ Method4 reference

如果原始 C++ 可以跑通，应作为 algorithm reference，而不是必须作为 QPS 公平
对照。它的价值在于验证语义同构：

- `point_to_l1s` 分布；
- `point_to_shards` 分布；
- shard size 分布；
- index expansion ratio；
- avg visited shards/query；
- recall trend。

如果不能完全同环境运行，文中必须说明限制，并保留语义不变量检查。

### B4: Method4 distributed without system optimizations

这是系统贡献的 baseline。它保留 Method4 算法语义，但关闭一个或多个系统优化，
例如关闭 worker-local peer pre-merge 或退回非 compact request 路径。

目的：

- 证明性能收益不仅来自算法 routing，也来自分布式执行适配。

## 统一在线对比要求：Orion / Simple KMeans / Naive

后续每次发起 Method4 online 对比实验时，除非实验问题明确排除某个 baseline，
默认至少要把下面三类方法放在同一张结果表中。这样避免每次重新解释“Orion、
simple KMeans、naive 应该如何设置和如何公平比较”。

### 方法定义

| 方法 | 主要作用 | Harness / preset | 默认 collection | 在线搜索语义 |
|---|---|---|---|---|
| Orion / Full Method4 | 主方法；验证 topology-aware routing、multi-assignment、per-shard entry points 和 dynamic EF 的整体效果 | `preset=orion`, `routing_mode=faithful_original_rest` | `bench095_rr_orion_s31`；初始 31 shards，经 fission 后有效 logical shards 通常为 46 | upper HNSW routing 选 entry points，经 `point_to_shards` 访问部分 lower shards；每个 visited shard 使用 `base_ef + factor * routed_ep_count` |
| Simple KMeans nprobe | 纯 KMeans centroid routing baseline；验证 Orion 是否比简单向量聚类路由更集中地命中有效 shards | `preset=kmeans_simple_nprobe`, `routing_mode=kmeans_simple_nprobe` | `bench095_cpp_kmeans_s46` | 根据 query 到 centroid 的距离选择 `nprobe` 个 lower shards；每个 visited shard 使用同一个 fixed EF |
| Naive all-shards | 分布式全量 lower shards baseline；验证 routing 减少 lower search work 是否转化为同召回 QPS/latency 收益 | `preset=naive`, `routing_mode=naive_hash_all_shards` | `bench095_rr_naive_s46` | 每个 query 固定访问全部 46 lower logical shards；每个 shard 使用同一个 fixed EF |

注意：`simple KMeans nprobe` 与 Claim A 里的 `method4_kmeans_ablation` 不是同一个
baseline。前者是简单 centroid-to-shard routing，参数是 `nprobe` 和 fixed EF；
后者仍在 Method4 framework 内，主要用于隔离 topology convergence / fission 等
离线 partition 步骤的贡献。

### 固定条件

三类方法的同组对比必须固定：

- 同一份 train/test/ground-truth；
- 同一 `top_k`，默认 `top_k=10`；
- 同一 lower HNSW 构建参数，默认 `m=32`, `ef_construct=100`；
- 同一 lower logical shard 主线规模，默认有效 shards 为 46；
- 同一 Qdrant topology、commit/image、replication factor 和 collection 构建批次；
- 同一 `eval_query_count`、`batch_size` 和 repeats，除非该实验专门研究 batch 或稳定性；
- Orion 的 `upper_search_ef` 必须不小于本轮最大的 upper candidate fanout，例如：

```text
upper_search_ef >= max(max(upper_k_candidates), max(nprobe_candidates if present))
```

否则 upper routing 的 entry point 候选会被 search EF 截断，导致不同方法的访问
shard 数和 recall 不可解释。

### 参数调节口径

Orion 调节：

```text
upper_k_candidates: 控制 upper routing 返回的 entry points 数量，进而影响 visited shards
base_ef_candidates: dynamic EF 的截距
factor_candidates: dynamic EF 的斜率，EF = base + factor * routed_ep_count
```

Simple KMeans 调节：

```text
nprobe_candidates: 控制访问多少个 KMeans centroid shards
base_ef_candidates: fixed EF，即每个 visited shard 的 lower HNSW EF
factor_candidates: 固定为 0
```

Naive 调节：

```text
base_ef_candidates: fixed EF
visited shards: 固定为全部 46 logical shards
factor_candidates: 固定为 0
```

如果实验目标是验证 “Dynamic EF 比 fixed EF 更有效”，应优先在 Orion 内部固定
partition、assignment 和 upper routing，只改变 per-shard EF 策略。若需要构造同召回
对照，也可以按用户指定的方式调节 fixed EF 的 upper fanout，例如增大 `upper_k`
或 `upper_search_ef`，让 fixed EF 访问更多 entry points / shards，再与 dynamic EF
在同 recall 附近比较 QPS、visited shards 和 EF-sum。

### 匹配规则

所有 online 对比先按目标 recall 分桶，再在桶内选择可解释成本最接近的点。
默认目标召回点：

```text
Recall@10 in {0.80, 0.85, 0.90, 0.95}
```

高召回投稿主结果可追加：

```text
Recall@10 in {0.97, 0.99}
```

同召回匹配要求：

```text
abs(method_recall - baseline_recall) <= 0.003
```

如果某个方法无法严格落入该窗口，必须明确标记为 nearest point，并用 Recall-QPS
curve 或 interpolation 解释，不能把 recall 明显不同的点直接当同召回结论。

当问题要求“平均 EF 相近”时，匹配 `avg_ef_per_visited_shard`，而不是只匹配命令行
里的 `base_ef`。报告时必须同时给出：

- `avg_ef_per_visited_shard`；
- `avg_visited_shards`；
- `estimated EF-sum/query = avg_visited_shards * avg_ef_per_visited_shard`；
- Recall@10；
- QPS；
- P95/P99 latency，如果本轮 benchmark 已采集。

解释时要区分两个事实：

- **平均 EF 相近**：每个 visited shard 上的搜索强度相近；
- **EF-sum 更低**：因为 visited shards 更少，每个 query 的 lower 总搜索预算更低。

因此，Orion vs simple KMeans 的标准问题应写成：

> 在 Recall@10 相近且 `avg_ef_per_visited_shard` 相近时，Orion 是否比
> simple KMeans 访问更少 lower shards，并获得更高 QPS？

Orion vs naive 的标准问题应写成：

> 在 Recall@10 相近时，Orion 是否显著少访问 lower shards，并因此相对
> naive all-shards 获得更高 QPS / 更低 tail latency？

### 默认 sweep 起点

如果没有额外说明，可以用下面的 sweep 起点；之后根据 tuning 结果收窄。

Orion：

```text
upper_k in {16, 24, 40, 60, 80, 120, 160, 200}
base_ef in {16, 24, 32, 40, 48, 60, 80}
factor in {4, 6, 8, 10, 12, 16, 20}
```

Simple KMeans:

```text
nprobe in {4, 6, 8, 10, 12, 16, 20, 24, 32, 40, 46}
fixed_ef in {16, 24, 32, 48, 64, 80, 96, 120, 160, 200, 240, 320}
```

Naive:

```text
fixed_ef in {8, 12, 16, 20, 24, 32, 40, 48, 56, 64, 72, 76, 80, 88, 96, 112, 128}
```

### 索引膨胀率 sweep 标准设置

如果实验问题是“在不超过 200% indexed-vector ratio，也就是不超过 2.0x 写入量的
条件下，观察同召回水平 QPS 如何随索引膨胀率变化”，必须把索引膨胀率作为第一类
自变量记录，而不是只记录 recall / QPS。

Orion multi-assignment sweep:

```text
single assignment:
  --disable-multi-assign

default top-ties:
  --orion-multi-assign-min-max-vote 2
  --orion-multi-assign-vote-delta 0
  --orion-multi-assign-max-shards 0

within-N votes, capped:
  --orion-multi-assign-vote-delta N
  --orion-multi-assign-max-shards C
```

Orion 的候选点应覆盖：

```text
expansion ~= 1.0x, 1.2x, 1.5x, 1.8x
```

其中 `max_vote == 1` 的点仍保持单分配，避免把完全不确定的边界点无意义复制到多个
shard。对于同一个已经构建好的 Orion collection，后续不同 target recall 的调参应优先
使用：

```text
--recover-routing-from-collection --reuse-existing
```

这样后续 case 评估的是已经部署的 collection 本身，避免重新计算 routing 时产生的
`expected_points` 与复用 collection 不一致。

Simple KMeans multi-assignment sweep:

```text
assign point i to every centroid c where
  distance(i, c) <= alpha * distance(i, nearest_centroid)

alpha in {1.000, 1.004, 1.010, 1.014}
```

报告时必须写清楚 `alpha` 对应的实际 indexed-vector ratio。GloVe-200/angular、
46 centroids、当前 KMeans seed 下的参考范围约为：

```text
alpha=1.000 -> 1.00x
alpha=1.004 -> 1.19x
alpha=1.010 -> 1.57x
alpha=1.014 -> 1.89x
```

Naive all-shards 在这个 sweep 中通常不是“可调膨胀率方法”：它固定每个点一个 copy，
index expansion 约为 1.0x，visited shards 固定为全部 logical shards。它应作为
all-shards fixed-EF 参照点进入最终对照表，而不是和 Orion / Simple KMeans 一起画成
多分配膨胀曲线。

索引膨胀率 sweep 的结果表至少包含：

- `index_expansion_ratio = indexed_vectors / logical_points`；
- target recall 与 final Recall@10；
- QPS；
- avg visited shards；
- avg EF/shard；
- EF-sum/query；
- 调参参数，例如 Orion 的 `upper_k/base/factor/vote_delta/max_shards` 或 Simple
  KMeans 的 `alpha/nprobe/fixed_ef`；
- 每行结果对应的 `summary.json` 或 `matrix_summary.csv` 路径。

### 标准报告表

Orion / simple KMeans / naive 的正式结果表至少采用下面格式：

| Target recall | Method | Params | Recall@10 | QPS | P95 | P99 | Avg EF/shard | Avg visited shards | EF-sum/query | Relative QPS | 备注 |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0.95 | Orion | `upper_k=..., base=..., factor=...` |  |  |  |  |  |  |  |  |  |
| 0.95 | Simple KMeans | `nprobe=..., ef=...` |  |  |  |  |  |  |  |  |  |
| 0.95 | Naive | `ef=...` |  |  |  |  |  |  |  |  | all 46 shards |

已有可复用配置示例：

- `tools/benchmark_configs/method4_recall_sweep_simple_kmeans_nprobe_20260623.json`
- `tools/benchmark_configs/method4_orion_vs_simple_kmeans_same_avg_ef_20260626.json`

## 统一指标

每个正式实验点至少记录：

| 指标 | 用途 |
|---|---|
| Recall@10 | 主质量指标 |
| QPS | 主吞吐指标 |
| P50 latency | 中位查询延迟 |
| P95 latency | tail latency |
| P99 latency | extreme tail latency |
| avg visited logical shards/query | 验证路由工作量 |
| avg routed entry points/query | 验证 multi-entry routing 负载 |
| avg EF per visited shard | 验证 dynamic EF 使用情况 |
| estimated EF-sum/query | 粗略比较 lower search work |
| RPC count/query | 验证 request compaction 和 peer fan-out |
| controller merge input streams/query | 验证 pre-merge 减少 fan-in |
| indexed vectors | 验证 multi-assignment 成本 |
| index expansion ratio | 解释空间代价 |
| memory / disk footprint | 系统成本 |
| build time | 构建成本 |
| worker CPU / controller CPU | 判断瓶颈位置 |

最终论文主表不一定全部展示，但实验记录必须保存，方便解释异常结果。

## 实验 1: 同召回率主对比

### 问题

在同 lower HNSW 构建参数下，Method4 distributed 是否在相同 Recall@10 附近
优于 naive all-shards distributed Qdrant？

### 配置

Method4 参数扫描：

```text
upper_k in {80, 120, 160, 200}
base_ef in {40, 60, 80, 100}
factor in {4, 6, 8, 10, 12}
batch_size in {100, 200}
```

Naive 参数扫描：

```text
ef in {48, 56, 64, 72, 76, 80, 88, 96, 112, 128}
batch_size in {100, 200}
```

对比目标召回点：

```text
Recall@10 ~= 0.90
Recall@10 ~= 0.95
Recall@10 ~= 0.97
Recall@10 ~= 0.99
```

### 报告方式

每个目标召回点选取 Recall@10 最接近的 Method4 和 naive 稳定配置。若 naive 在
某个 batch 下 tail instability 明显，应同时报告：

- strict same batch comparison；
- best stable batch comparison。

主表模板：

| Target recall | Method | Params | Recall@10 | QPS mean | QPS stddev | P95 | P99 | Avg visited shards | Relative QPS |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 0.95 | Method4 | `upper_k=160, base_ef=80, factor=8` |  |  |  |  |  |  |  |
| 0.95 | Naive | `ef=76` |  |  |  |  |  |  |  |

### 判定标准

Method4 的 Recall@10 与 baseline 差距应在同一目标召回邻域内，例如：

```text
abs(method4_recall - naive_recall) <= 0.003
```

若 Recall@10 不能严格匹配，则必须用 Recall-QPS curve 或 interpolation 解释。

## 实验 2: Recall-QPS / Recall-Latency 曲线

### 问题

Method4 的优势是否覆盖一段高召回区间，而不是某个孤立参数点？

### 方法

使用实验 1 的扫描结果绘制：

- Recall@10 vs QPS；
- Recall@10 vs P95 latency；
- Recall@10 vs P99 latency；
- Recall@10 vs avg visited logical shards/query；
- Recall@10 vs estimated EF-sum/query。

### 预期解释

如果 Method4 有效，曲线应体现：

- 高召回区间 Method4 在相同 recall 附近 QPS 更高；
- Method4 avg visited shards 显著低于 naive 的全量 46；
- 随 recall 增长，naive 需要整体提升 EF，而 Method4 可以通过 upper routing 和
  per-shard dynamic EF 更集中地分配 lower work。

## 实验 3: 系统优化 Ablation

### 问题

当前性能收益分别来自哪些算法机制和系统优化？哪些优化没有正收益？

### 实验组

以同一 Method4 配置为基础，逐项关闭：

| 组别 | 改动 | 要验证的机制 |
|---|---|---|
| Full Method4 distributed | 全部开启 | 当前主线 |
| No worker-local pre-merge | `QDRANT_DISABLE_SHARD_MAJOR_PEER_PREMERGE=1` | controller fan-in 优化 |
| No compact multi-EP | 退回 per-shard request 或模拟 per-shard request 开销 | request compaction |
| No controller-native routing | 使用 Python route plan / JSON hot path 或模拟其开销 | controller native routing |
| Fixed EF | 所有 visited shard 使用固定 EF | dynamic EF |
| No multi-assignment | 每点只写入主 shard | recall / index expansion tradeoff |
| No fission / auto-sharding | 保留 topology convergence 后的 shard map | tail latency / load balance |
| Round-robin placement | 不使用 method4-aware layout | hot worker / physical balance |

### 报告方式

| Variant | Recall@10 | QPS | P95 | P99 | Avg visited shards | RPC/query | Controller streams/query | Index expansion | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Full |  |  |  |  |  |  |  |  | keep |
| No pre-merge |  |  |  |  |  |  |  |  |  |

### 判定标准

- 若优化提升 QPS 或 tail latency，且 Recall@10 不下降超过 0.003，则保留；
- 若优化无稳定正收益，文中不能作为核心贡献宣称；
- 若优化提升 QPS 但显著降低 recall，必须归为质量-性能 tradeoff，而不是纯优化。

## 实验 4: 分布式扩展性

### 问题

Method4 distributed 是否真正适合分布式图索引检索，而不是只在当前 3-worker
配置上有效？

### Worker 数扩展

测试：

```text
1 worker
2 workers
3 workers
4 workers 或更多
```

记录：

- QPS scaling；
- P95 / P99 latency；
- per-worker query load；
- per-worker logical shard count；
- per-worker CPU；
- controller CPU；
- network throughput；
- avg physical peers visited/query。

### Logical shard 数扩展

测试：

```text
lower logical shards in {31, 46, 64, 96}
```

记录：

- avg visited logical shards/query；
- avg physical peers/query；
- worker-local pre-merge 收益；
- index expansion ratio；
- tail latency。

### 判定标准

如果 worker 增加但 QPS 不提升，需要判断瓶颈是否在：

- controller routing；
- final merge；
- network；
- hot worker；
- lower HNSW CPU；
- shard placement。

该实验用于支撑架构设计，而不是只证明单机容器内参数调优。

## 实验 5: Method4 语义保真验证

### 问题

当前 distributed implementation 是否仍然是原初 C++ method4 idea 的实现，而不是
变成了另一个算法？

### 检查项

对同一数据集和同一 method4 path，比较或审计：

| 项目 | 验证方式 |
|---|---|
| upper sample size | 是否为 `N / 32` |
| upper routing | 是否使用 upper HNSW top candidates |
| `point_to_l1s` | 每个点是否保留 K_OVERLAP L1 candidates |
| L1 partition path | balanced KMeans init -> topology convergence -> load recalibration -> fission |
| final assignment | full-dataset voting multi-assignment |
| `point_to_shards` | 是否用于 online routing |
| lower entry points | 是否按 shard 分别传入 |
| dynamic EF | 是否按 `base_ef + factor * routed_ep_count` |
| lower search granularity | 是否 per logical shard |
| dedup | 是否使用 source-id dedup |
| final merge | 是否 controller global top-k |

如果能运行 C++ reference，补充对比：

- shard size distribution；
- index expansion ratio；
- avg visited shards/query；
- recall at matched parameters。

### 判定标准

允许 HNSW 内部图拓扑和搜索随机性不同；不允许 external wrapper semantics 改变。

## 实验 6: 成本与资源代价

### 问题

Method4 的性能收益是否值得它的额外索引膨胀、构建时间和系统复杂度？

### 指标

| 成本项 | 说明 |
|---|---|
| indexed vectors | multi-assignment 后实际写入向量数 |
| expansion ratio | `indexed_vectors / original_vectors` |
| build time | upper build、assignment、lower build 分开记录 |
| memory footprint | controller 与 worker 分开记录 |
| disk footprint | collection 总大小与每 worker 分布 |
| network payload/query | compact request 与 result streams |
| RPC/query | controller 到 worker 请求数量 |
| CPU utilization | controller / workers |

### 报告方式

把性能收益与成本放在同一张表中：

| Method | Recall@10 | QPS | P95 | Index expansion | Build time | Memory | RPC/query |
|---|---:|---:|---:|---:|---:|---:|---:|
| Method4 |  |  |  |  |  |  |  |
| Naive |  |  |  |  |  |  |  |

## 实验 7: 多数据集泛化

### 问题

Method4 distributed 的收益是否只存在于 GloVe angular，还是对不同距离和数据分布
也有效？

### 最低要求

至少两个数据集：

- GloVe angular / Cosine；
- 一个 L2 数据集，例如 SIFT、GIST 或 Deep 系列。

### 推荐报告

每个数据集至少报告：

- same-recall main comparison；
- Recall-QPS curve；
- index expansion ratio；
- avg visited shards/query；
- 最关键 ablation：worker-local pre-merge 和 dynamic EF。

如果时间有限，多数据集可以减少 ablation，但不能只给单点 QPS。

## 实验 8: 鲁棒性与重复性

### 问题

结果是否稳定，还是来自 batch、缓存、容器状态或 query 顺序偶然性？

### 方法

- 每个关键配置至少 3 repeats；
- 固定 query set，另做一次 shuffled query order；
- 报告 mean、stddev、min、max；
- 分开报告 cold-ish run 和 warmed run；
- 对 QPS 和 P95/P99 都给误差范围；
- 所有结果目录保存 `summary.json`、命令、commit、image tag。

### 判定标准

主结论必须基于稳定 mean，而不是某次 best run。若 baseline 某个 batch size 不稳定，
必须同时给 strict same batch 和 best stable batch 两种解释。

## 推荐执行顺序

### Phase 1: 锁定最终实验环境

- 确定 final commit；
- 构建 final Docker image；
- 记录 image tag 和 digest；
- 记录 controller requested/current compact wire version，并确认三个 workers 没有
  controller-only wire env/label；
- 若从 v1 升级到 v2，使用 immutable candidate + offline four-node transition，不做
  rolling upgrade；
- 确认 collection、dataset、ground truth；
- 跑一次 smoke test 验证 recall 和 routing stats 正常。

### Phase 2: 主对比和曲线

- 跑 Method4 parameter sweep；
- 跑 naive EF sweep；
- 生成 same-recall table；
- 生成 Recall-QPS / Recall-Latency curves；
- 选择论文主配置。

### Phase 3: Ablation

- 以论文主配置为中心，逐项关闭系统和算法机制；
- 每项至少 3 repeats；
- 对没有正收益的优化，从论文贡献中降级或移除。

### Phase 4: 分布式扩展性

- worker 数变化；
- lower logical shard 数变化；
- 记录 physical load balance 和 tail latency。

### Phase 5: 成本与语义保真

- 统计 index expansion、memory、disk、build time；
- 做 Method4 semantic invariants audit；
- 如可行，运行 C++ reference 对比。

### Phase 6: 多数据集确认

- 至少补一个 L2 数据集；
- 重复主对比和核心 ablation；
- 判断哪些结论是 general，哪些只对 GloVe/Cosine 有效。

## 最低投稿实验包

如果时间有限，至少完成下面 5 组，缺一组都会让论文说服力明显下降：

1. **Same-recall main comparison**：
   Method4 distributed vs naive distributed，在 Recall@10 约 0.95 和 0.97 两个
   点比较 QPS、P95、P99。

2. **Recall-QPS / Recall-Latency curve**：
   证明优势覆盖高召回区间，而不是单点调参结果。

3. **Ablation**：
   至少包括 worker-local peer pre-merge、compact multi-EP、dynamic EF、
   multi-assignment、fission / placement。

4. **Scalability**：
   至少比较不同 worker 数或不同 logical shard 数下的 QPS 和 tail latency。

5. **Cost analysis**：
   报告 index expansion、memory、build time、RPC/query 和 network/fan-in 成本。

强烈建议补充：

6. **Method4 semantic fidelity audit**；
7. **至少一个额外数据集**。

## 论文图表建议

### Figure 1: Architecture

使用当前 Method4-on-distributed-Qdrant 架构图，强调：

- offline build path；
- online routing path；
- controller-native routing；
- shard-major lower execution；
- worker-local pre-merge；
- semantic invariants。

### Figure 2: Recall-QPS Curve

主图。展示 Method4 与 naive 在高召回区间的 QPS 差距。

### Figure 3: Recall-P95/P99 Latency Curve

证明收益不只是 throughput，也改善或至少不恶化 tail latency。

### Figure 4: Work Reduction

展示 avg visited logical shards/query、avg EF/query 或 estimated EF-sum/query。

### Figure 5: Ablation

柱状图展示每个优化关闭后的 QPS / latency 变化。

### Figure 6: Scalability

展示 worker 数或 logical shard 数变化下的 scaling。

### Table 1: Same-Recall Main Result

在目标 recall 点下汇总 Method4、naive、vanilla Qdrant。

### Table 2: Cost

展示 index expansion、memory、build time、disk、RPC/query。

## 风险与处理

### 风险 1: Method4 和 naive batch size 不同

处理：

- 报告 strict same batch comparison；
- 同时报告 best stable batch comparison；
- 主张基于 stable mean，不基于 unstable best run。

### 风险 2: Method4 多分配导致索引膨胀

处理：

- 明确报告 expansion ratio；
- 将 QPS 收益与 memory/disk/build cost 放在一起；
- 说明这是算法设计的空间换时间成本。

### 风险 3: 某个系统优化收益不稳定

处理：

- 用 same-image runtime A/B；
- 若收益不稳定，则从核心贡献降级为 engineering optimization；
- 不把无稳定正收益的点写入主结论。

### 风险 4: HNSW 内部实现与 C++ 不同

处理：

- 论文中明确 HNSW internals are transparent；
- 强调验证的是 Method4 external scheduling semantics；
- 用 semantic fidelity audit 证明 wrapper 行为一致。

### 风险 5: 单数据集不足

处理：

- 至少补一个 L2 数据集；
- 如果受资源限制，只把单数据集结论限定为 case study，并在 limitations 中说明。

## 最终判定标准

一组结果可以支撑投稿主张，需要同时满足：

- Method4 在至少两个高召回点达到与 naive 接近的 Recall@10；
- Method4 在 stable mean QPS 上有明确收益；
- P95/P99 latency 不显著恶化，最好同步改善；
- avg visited logical shards/query 明显低于 naive；
- ablation 能解释主要收益来源；
- semantic invariants 没有被破坏；
- index expansion、build time 和 memory 成本被完整报告；
- 至少一个分布式扩展性实验证明架构合理；
- 所有核心数字能追溯到 result directory、commit、image 和命令。
