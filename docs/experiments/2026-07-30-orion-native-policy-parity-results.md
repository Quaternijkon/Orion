# Orion 在 Qdrant 中的原生策略对等实现与四节点实验

## 结论

当前实现已经达到论文所需的**静态分布式 ANN 查询层策略对等**：HashAll、Orion 和
Simple KMeans 都是 Qdrant collection 的 `auto_shard_policy`，都通过标准 Search API
进入同一个 collection 查询入口，并使用 Qdrant 的 replica-set read path 和 collection-level
global merge。三种方法的区别被限制在“如何产生 logical-shard search plan”这一层。

这个结论不应扩大为“完整生命周期完全对等”。HashAll 仍支持 Qdrant 原有的在线写入和
resharding；当前 Orion 与 Simple KMeans 是由版本化 artifact 驱动的静态 routed collection，
尚未实现在线 artifact generation、CRUD 后的增量维护、原子 generation 切换和动态 resharding。

因此，对论文可以准确表述为：

> Orion 已作为 Qdrant 原生 `auto_shard_policy` 实现，并在相同标准 Search API、相同
> `ShardReplicaSet::core_search`、相同 `Collection::merge_from_shards` 和相同普通 per-shard
> replica-set transport 下，与 Qdrant HashAll 和 Simple KMeans 进行静态分布式 ANN 查询比较。

## 对外策略接口

平台层的实际选项是：

```text
auto_shard_policy:
  hash_all | orion | simple_kmeans
```

HashAll 可显式声明，也可以省略该字段以保留旧 Qdrant 行为：

```json
{
  "auto_shard_policy": {
    "type": "hash_all"
  }
}
```

Orion 和 Simple KMeans 需要绑定已安装的 immutable routing artifact，因此不是只有一个裸
字符串，还必须带 generation 和 SHA-256：

```json
{
  "auto_shard_policy": {
    "type": "orion",
    "generation": 7,
    "artifact_sha256": "3f61d32e83ecae0ca69c792b9811266ae7a51dfc393e9e9ffbb0ff6c90b32007"
  }
}
```

```json
{
  "auto_shard_policy": {
    "type": "simple_kmeans",
    "generation": 11,
    "artifact_sha256": "d82c47174a9f2188589516a5b69ff05b3dc2c3400a5ea050e2adbff35405afc7"
  }
}
```

如果外部产品界面希望显示 `hash | orion`，可以把 `hash` 作为 `hash_all` 的 UI alias；Qdrant
内部保留 `hash_all` 更准确，因为它同时表示 point-ID hash placement 和 vector query 的
all-shards read。

## 架构：对等 peers，而不是固定 controller/worker 服务

最终架构是：

```text
对等 Qdrant peers
  +
每个请求临时产生 coordinator/worker 角色
```

接收客户端 Search 请求的 peer 是该请求的 coordinator；持有所选 shard replica 的 peer 是该
请求的 worker。实验把请求固定发给 node0，是为了控制部署、CPU affinity 和测量入口，不表示
node0 是一个独立 controller 服务。正式拓扑中 node0 不持有 lower shards，46 个 RF=1 shards
全部位于 node1/2/3，但四个进程仍然是同一种 Qdrant peer。

实现中的关键抽象如下：

| 层 | HashAll | Orion | Simple KMeans | 是否共享 |
|---|---|---|---|---|
| collection policy | `hash_all`/缺省 | `orion` artifact | `simple_kmeans` artifact | 同一 `AutoShardPolicy` |
| client API | Search | Search | Search | 是 |
| request hints | 无 | 无 | 无 | 是 |
| plan | all 46 shards | upper-HNSW selected shards + per-shard lower overrides | centroid nprobe selected shards | 仅此层不同 |
| selected-shard executor | Qdrant all-shards path | common selected-shard executor | common selected-shard executor | routed policies 共享 |
| replica read | `ShardReplicaSet::core_search` | `ShardReplicaSet::core_search` | `ShardReplicaSet::core_search` | 是 |
| global merge | `Collection::merge_from_shards` | `Collection::merge_from_shards` | `Collection::merge_from_shards` | 是 |
| transport | ordinary per-shard replica-set | ordinary per-shard replica-set | ordinary per-shard replica-set | 是 |

`AutoShardPolicy` 的三个 variant 位于
[`lib/collection/src/config.rs`](../../lib/collection/src/config.rs)。查询入口首先把策略编译为
`AllShards` 或 `SelectedByShard`，见
[`lib/collection/src/collection/search.rs`](../../lib/collection/src/collection/search.rs)；Orion 和
Simple KMeans 的目标随后统一成 `LogicalShardSearchTarget`，见
[`lib/collection/src/distributed_index/plan.rs`](../../lib/collection/src/distributed_index/plan.rs)。

正式实验显式关闭了 Orion 的 optional peer-premerge fast path。因此 Orion 没有通过专用 compact
transport 获得实验优势，三种方法的 `transport_mode` 均为
`ordinary_per_shard_replica_set`。HashAll 必然搜索所有 shards，而 Orion/Simple KMeans 搜索策略
选出的 shards；这正是被比较算法的核心差异，不是平台执行路径的不公平差异。

## 正式实验身份

最终有效矩阵为：

```text
/proj/intelisys-PG0/exp/orion-distributed/
  native-20260729-policy-parity-v1/
  matrix/policy-parity-common-executor-strict-v4/
```

冻结身份：

| 项目 | 值 |
|---|---|
| Git commit | `4385b1acbc6d22ac796aa0d46d6c386fc429b47b` |
| implementation commit | `9e29ad607387dc5213f82b525870fdf778d8902e` |
| HashAll r090 tuning commit | `4385b1acbc6d22ac796aa0d46d6c386fc429b47b` |
| image tag | `orion-method4:4385b1acbc6d-source-5d0ec50ad07a` |
| image digest | `sha256:82d5bdea101f5a0f93b7f86184474bb30e4182519acffddb9583eea7ad11ed1c` |
| source fingerprint | `5d0ec50ad07a64d07f25a8a12fcfc20c94fb20abab7ed8561537ebd02a06e47f` |
| image tar SHA-256 | `1d126ee1ee6f3d6fdfa6e365e0e71abbe299e376a684f87a056eb75e8a4ba512` |
| matrix config SHA-256 | `dad3edb9d3c03e5ac5d8b6498f26072f3f3e82eac176895dfa46bbb7784ca9e5` |
| matrix manifest SHA-256 | `1ffa89250181a093b13806f444ecf55ad868b7ee81e4fd65b6d18abd2aff3672` |
| benchmark lock token SHA-256 | `8fa2046186302443604ced0fe49c1d2c92004a415d82314948be5a93cd6ea478` |
| compact wire | v2 capability present；本轮 peer-premerge disabled |

审计确认六个 case 的 benchmark/deployment commit、image digest、transport identity、config SHA、
benchmark lock token 完全一致；六个 matrix stderr 均为 0 字节。

## 共同实验参数

| 参数 | 值 |
|---|---|
| dataset | ANN-Benchmarks GloVe-200 Angular，Qdrant Cosine |
| train set | `1,183,514 × 200` |
| query evaluation | 3,000 queries，top-10 |
| warmup | 500 queries |
| HTTP batch size | 200 queries |
| repeat | 每个 case 3 次 |
| logical shards | 46，全部 worker-only |
| replication factor | 1 |
| placement | node1/node2/node3 round-robin：16/15/15 |
| HNSW | `M=32`，`ef_construct=100` |
| API | standard Qdrant Search API |
| client CPU affinity | node0 logical CPUs `8-19` |
| controller Qdrant affinity | node0 logical CPUs `0-7` |
| worker Qdrant affinity | node1/2/3 logical CPUs `0-19` |
| peer premerge | disabled |
| transport | ordinary per-shard replica-set |

每个 case 的 collection 均为 green、`optimizer_status=ok`、fully indexed、update queue 为空、
46 个 shards 全部 Active、controller local lower shards 为 0、shard transfer 为 0。

## strict same-recall 门禁

选择规则是每个方法都必须落在 `[target, target + 0.003]`，且三个方法之间的最大 pairwise
recall spread 不超过 `0.003`。

| 目标 | HashAll Recall@10 | Orion Recall@10 | Simple KMeans Recall@10 | 最大 spread | 结论 |
|---:|---:|---:|---:|---:|---|
| 0.90 | 0.9009667 | 0.9024667 | 0.9018000 | 0.0015000 | strict pass |
| 0.95 | 0.9508000 | 0.9527667 | 0.9513000 | 0.0019667 | strict pass |

完整门禁副本见
[`same_recall_confirmation.csv`](2026-07-30-orion-native-policy-parity-results/same_recall_confirmation.csv)。

## 正式性能结果

P50/P95/P99 是**一个 200-query HTTP batch 的 latency**，不是单 query latency。

| Recall 目标 | 方法 | Recall@10 | QPS mean ± sd | P50 batch ms | P95 batch ms | P99 batch ms | Visited shards | EF sum/query |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 0.90 | HashAll ef36 | 0.9009667 | 1281.10 ± 2.69 | 155.65 | 163.97 | 166.71 | 46.000 | 1656.000 |
| 0.90 | Orion | 0.9024667 | 1460.70 ± 6.51 | 136.38 | 142.75 | 143.83 | 10.825 | 1249.716 |
| 0.90 | Simple KMeans | 0.9018000 | 994.11 ± 6.32 | 201.94 | 208.00 | 210.08 | 19.000 | 2071.000 |
| 0.95 | HashAll ef71 | 0.9508000 | 961.51 ± 8.33 | 207.71 | 217.37 | 221.86 | 46.000 | 3266.000 |
| 0.95 | Orion | 0.9527667 | 952.16 ± 6.54 | 209.73 | 218.87 | 219.59 | 18.549 | 3146.368 |
| 0.95 | Simple KMeans | 0.9513000 | 591.31 ± 4.50 | 339.83 | 351.35 | 354.88 | 26.000 | 5460.000 |

逐 case 精确值见
[`case_metrics.csv`](2026-07-30-orion-native-policy-parity-results/case_metrics.csv)。

## repeat-level 描述统计

统计单位是每个 case 的 3 个 repeat run，不能把一个 repeat 内的 3,000 个 query 当成 3,000
个独立样本。本轮按 case 分块执行，顺序为 HashAll r090、HashAll r095、Orion r090、Orion
r095、Simple r090、Simple r095，并非 AB/BA 或完全 interleaved。因此下表的 Welch 结果只作为
描述性辅助证据，环境漂移可能影响小差异。

| 目标 | 比较 | QPS 差值 | 相对差异 | Welch 95% CI，QPS | 双侧 p |
|---:|---|---:|---:|---:|---:|
| 0.90 | Orion − HashAll | +179.59 | +14.02% | [165.67, 193.52] | 0.0000683 |
| 0.90 | Orion − Simple KMeans | +466.59 | +46.94% | [452.03, 481.14] | <0.000001 |
| 0.95 | Orion − HashAll | −9.35 | −0.97% | [−26.72, 8.01] | 0.2048 |
| 0.95 | Orion − Simple KMeans | +360.85 | +61.02% | [347.46, 374.23] | <0.000001 |

据此，本轮可支持的结论是：

- 在约 0.90 Recall 下，Orion 相对 HashAll 和 Simple KMeans 都有较大的正向 QPS 差异；
- 在约 0.95 Recall 下，Orion 与 HashAll 基本接近，均值低约 1%，95% CI 跨零，不能判定
  二者有稳定输赢；
- Orion 在两个 recall 目标下都明显优于当前 Simple KMeans baseline；
- 不能据此声称 Orion 在所有 recall、数据集、距离度量、shard count 或拓扑下普遍优于 HashAll。

18 个 repeat 的原始小型副本见
[`repeat_metrics.csv`](2026-07-30-orion-native-policy-parity-results/repeat_metrics.csv)，完整 Welch
计算见
[`welch_qps_comparisons.csv`](2026-07-30-orion-native-policy-parity-results/welch_qps_comparisons.csv)。

## 资源成本

HashAll 和 Simple KMeans 的 logical/physical point count 都是 `1,183,514 / 1,183,514`。
Orion 是 `1,183,514 / 1,394,406`，即 `1.17819×` physical points，额外约 `17.82%`。

因此这不是“完全相同存储 footprint 下”的比较。Orion 的 overlap/multi-assignment 是算法设计的
组成部分，论文必须同时报告该物理扩张；后续还应补充 collection bytes、RAM、build time、
network bytes 和 CPU cycles，才能形成完整的资源—质量—吞吐权衡。

## 当前尚未对等的生命周期能力

以下内容没有在本轮实现或证明：

1. Orion/Simple KMeans 对任意在线 upsert/delete 的增量 placement 与 router 维护；
2. 新 artifact generation 的在线构建、Raft 协调和原子激活；
3. 静态 routed collection 的动态 resharding；当前代码会 fail closed；
4. RF>1、read consistency、replica failover 和 shard transfer 中的完整行为；
5. 对 worker peer 作为客户端入口的独立性能复测；架构允许任意 peer 协调，但本轮测量入口固定为
   node0；
6. 其他数据集、Euclid/Dot、不同 shard counts 和更高 recall 目标；
7. 与 HashAll 相同 storage/RAM/network budget 下的受约束比较。

这意味着论文中适合使用“Qdrant-native static distributed query policy parity”或“查询路径对等”，
不适合使用“full online lifecycle parity”。

## 独立审计与复现

新增审计入口：

```bash
python3 tools/audit_native_auto_shard_policy_parity.py \
  /proj/intelisys-PG0/exp/orion-distributed/\
native-20260729-policy-parity-v1/matrix/\
policy-parity-common-executor-strict-v4 \
  --config tools/benchmark_configs/\
native_auto_shard_glove200_policy_parity_v1.json \
  --expected-commit 4385b1acbc6d22ac796aa0d46d6c386fc429b47b \
  --expected-repeats 3 \
  --output-dir /tmp/orion-policy-parity-audit
```

本次运行返回 `ok=true`、`errors=[]`，六个 case 均为 `ok=true`。冻结审计输出见
[`audit_report.json`](2026-07-30-orion-native-policy-parity-results/audit_report.json)。审计工具的
Student-t/Welch 单元测试为 3/3 passed。

正式矩阵可由以下入口复现，但必须使用新的 run-id，不能覆盖本次 raw evidence：

```bash
RUN_ROOT=/proj/intelisys-PG0/exp/orion-distributed/\
native-20260729-policy-parity-v1

/users/dry/orion-distributed/venv/bin/python \
  tools/native_auto_shard_matrix.py \
  --config tools/benchmark_configs/\
native_auto_shard_glove200_policy_parity_v1.json \
  --run-id <new-run-id> \
  --output-root "$RUN_ROOT/matrix" \
  --taskset-cpus 8-19 \
  --run
```

运行前必须重新确认四节点 image/commit、wire v2、peer-premerge disabled、46-shard placement、
collection readiness，以及 benchmark/lifecycle locks 均符合要求。

## SSH 与停止前集群复核

安装 node0 的 Ed25519 公钥后，node0 使用 `/users/dry/.ssh/id_ed25519` 已实时、无交互登录：

```text
hp052.utah.cloudlab.us -> node1 -> SSH_OK
hp065.utah.cloudlab.us -> node2 -> SSH_OK
hp076.utah.cloudlab.us -> node3 -> SSH_OK
```

四个私网 Qdrant `/readyz` 均返回 HTTP 200。停止前 `status` 再次确认四个容器运行同一 image
digest、四 peer cluster 正常、五个 collections 全部 green/fully indexed、0 transfers，并且
`validation.ok=true`。因此此前“node0 无法 SSH 到 node1/2/3”的状态已经失效；当前公钥安装已
生效。

证据冻结后只执行了 run-id-scoped `down`，没有执行 `clean`。停止后四个容器均仍存在且为
`running=false, status=exited`，四个 `/readyz` 均不再监听；node0/1/2/3 上本 run 的 storage
inode、storage 目录和五个 collection 目录全部保留。旧
`native-20260721-glove200-full-v4` storage 的 inode/mtime 在停止前后完全不变，说明旧实验存储
未被触碰。停止后 benchmark/lifecycle locks 仍为空闲。

## 小型证据包与 raw artifact 边界

Git 内只保留本文、审计工具及以下小型证据：

```text
docs/experiments/2026-07-30-orion-native-policy-parity-results/
```

大型或运行时数据继续保留在 Git 外，包括 HDF5、Docker tar、Qdrant storage/WAL/index、
per-query metrics、Orion route trace 和既有未跟踪 `results/`。文件及外部权威 source artifact
的 SHA-256 见
[`artifact_checksums.tsv`](2026-07-30-orion-native-policy-parity-results/artifact_checksums.tsv)。
