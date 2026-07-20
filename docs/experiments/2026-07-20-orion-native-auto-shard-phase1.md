# 2026-07-20 Orion Native Auto-Shard Phase 1

## 当前结论

本阶段已经把 Orion 的**读取路由**放进 Qdrant collection coordinator，并把离线
layout 以同一 external point ID 写入 Qdrant 的 numeric auto-shards；同时实现了使用同一
collection、numeric shard 和 `ShardReplicaSet` 执行模型的 native Simple KMeans
baseline。当前实现的准确名称是：

> **native static/read-only Orion and Simple KMeans routing over Qdrant numeric
> auto-shards**

它不再是 benchmark/client 侧的 custom-shard 路由器：客户端只发送标准 Search 或
Query 请求，coordinator 在服务端执行 upper HNSW、完整 shard-membership union、ordered
MultiEP 和 Dynamic EF，然后仍通过普通 `ShardReplicaSet`、本地/远程副本 RPC 和
collection global merge 完成查询。Simple KMeans 同样由 coordinator 在服务端完成
centroid routing，而不是由 benchmark client 选择 shards。

当前实现已经具备可校验的两种 routed layout 构建、generic numeric-shard 导入、四节点
artifact 安装和重启激活、标准 Search/Query benchmark 以及三方法 matrix 汇总入口，并已
完成一个 1,024-point 的真实四节点 native smoke；但还没有完整在线 CRUD、热 generation
激活或 native 四节点正式 Recall–QPS 结果。因此不能把旧
custom-shard/client-hint/synthetic-ID 实验表述为本架构的性能结果，也不能把 Phase 1
称为完整生产级的 Qdrant default-sharding replacement。

## 与 Qdrant 默认行为的对应关系

| 路径 | Qdrant 默认 `HashAll` | Native Orion Phase 1 | Native Simple KMeans baseline |
|---|---|---|---|
| collection 类型 | `sharding_method=auto`，numeric `ShardId` | 同样是 auto + numeric `ShardId` | 同样是 auto + numeric `ShardId` |
| 写入放置 | external point ID 经 hash ring 进入一个 logical shard | 离线 vector/topology voting，同一 external ID 可进入多个 numeric shards | 离线最近 centroid 分配，每个 external ID 恰好进入一个 numeric shard |
| 普通向量读取 | coordinator fan-out 到全部 logical shards | upper HNSW 的完整 membership union | coordinator 精确计算 centroid squared-L2，选择最近 `nprobe` shards |
| shard 内搜索 | 标准 HNSW entry point 和请求 EF | ordered MultiEP，`EF = base + factor * unique_EP_count` | 标准 shard HNSW entry point 和 artifact 固定 lower EF |
| replica 执行 | 普通 `ShardReplicaSet` | 同一 `ShardReplicaSet` 路径 | 同一 `ShardReplicaSet` 路径 |
| 聚合 | collection global score merge | 同一 distance-aware merge，并按 external ID 去重 copies | 同一 distance-aware merge；无复制膨胀 |
| 在线写入 | 完整支持 | 普通 client writes 被拒绝 | 普通 client writes 被拒绝 |

Orion 和 Simple KMeans 替换的是默认 auto-shard 的**数据放置与查询路由策略**，不是
另建 controller、worker 协议或旁路执行引擎。logical shard 到 physical peer 的
placement 仍由 Qdrant 管理，server-side router 只产生 numeric logical shard targets，
不直接选择物理机器。

## 真实四节点 native smoke：`native-20260720-smoke-v1`

该 smoke 使用固定四节点私网拓扑 `10.10.1.1` 至 `10.10.1.4`，从完整
GloVe-200/Cosine 数据集取前 1,024 个 train vectors，验证的是 native auto-shard 的构建、
放置、artifact 激活和标准 API 链路，不是 Recall–QPS 正式矩阵。Orion 从初始 6 shards 经
原 fission 逻辑得到 10 个 effective numeric shards；三种 collection 都使用相同的 10
shards、RF=1 和 worker-only round-robin placement。

| collection | policy | logical points | physical points | expansion | artifact SHA-256 |
|---|---|---:|---:|---:|---|
| `native_smoke_hash` | Qdrant HashAll | 1,024 | 1,024 | 1.0000x | 不适用 |
| `native_smoke_simple` | native Simple KMeans | 1,024 | 1,024 | 1.0000x | `6eafa4f324bcb887cbf5817ab7de600865f5b15333fa9bcafac1c35e39b5c519` |
| `native_smoke_orion` | native Orion generation 1 | 1,024 | 1,246 | 1.216796875x | `311e4dc336ac85ef4d2a890350e32709e6e587128b55498425dfee554fc94a62` |

三个 workers 的 local shard 数按 `qdrant_shard_1/2/3` 为 `3/4/3`；controller 的 lower
local shard 数为 0，10 个 replicas 均为 Active，没有 shard transfer。HashAll 和 Simple
KMeans 的 `indexed_vectors_count` 都为 1,024；Orion 为 1,230/1,246，这个 16-point 差额被
readiness proof 明确记录为 `stable_small_segment_full_scan_exception`。三者均为 green、
optimizer `ok`、update queue 为空。

任意 peer coordinator 验证覆盖三个 collection、四个 HTTP 地址和 Search/Query 两种标准
API，共 `3 * 4 * 2 = 24` 个请求。所有请求都只含标准 dense-nearest 字段，不含 shard
selector、EP、per-shard EF map 或 source-ID hint；每个请求均返回 10 个结果且 external IDs
唯一。这证明任意 peer 都能进入相同的 collection coordinator 和普通 local/remote
`ShardReplicaSet` 路径。

对同一 production `OrionRouter` 和 32 条 query 的计时窗口外 exact route trace 得到：平均
访问 `3.46875/10` logical shards，平均 `EF-sum/query=146.25`；其区间分别为 2–6 shards 和
104–212 EF-sum。该 trace 只证明给定 artifact/query 的精确 routing plan，不计入 QPS，
也不是 Recall 结果。

本次 smoke 的四节点 release image 是提交
`bb2a0637c4259e36d9a14ff2081fa6a3ff3dea22` 构建的
`orion-method4:bb2a0637c425`，image ID 为
`sha256:915969cf06db7610289eb52fab4e510c3d1ca977f674bbb59838579b1120391a`。
该 image 与 deployment manifest 绑定的 commit 早于下文的 attachment/runtime EF 显式
解耦、prepare/benchmark readiness 加固、remote shell 修复和 commit binding 加固。因此
这个 smoke 可以作为旧 image 的 native 链路证据，
不能作为新实现的正式性能证据；正式 full Recall–QPS 必须先提交当前修复，从该新提交构建
并四节点部署同一新 image，再重新生成 layout/artifact、重新 prepare collection 并测量，
不得直接复用该 smoke 的旧 image 或 collection。

## 已实现的静态原生链路

### 1. Collection policy 与 production artifact

collection config 新增向后兼容的 `AutoShardPolicy`：

- 字段缺失或显式 `HashAll` 等价于现有 Qdrant 行为；
- `Orion { generation, artifact_sha256 }` 激活指定静态 routing generation；
- `SimpleKmeans { generation, artifact_sha256 }` 激活指定静态 baseline generation；
- 两种 routed policy 都只允许配合 `sharding_method=auto` 和 numeric shards；
- generation 必须大于零，artifact SHA-256 必须为 64 位十六进制；
- policy 进入持久化 config、REST/gRPC 转换、兼容性检查和 telemetry。

每个节点分别使用固定 artifact 路径：

```text
<collection_path>/orion_router/generation-<generation>.json
<collection_path>/simple_kmeans_router/generation-<generation>.json
```

Orion production artifact 包含：

- format version 和 generation；
- dense single-vector schema：name、dimension、distance、datatype；
- numeric logical shard count；
- `layout_sha256`、logical point count、physical copy count；
- `upper_k`、runtime upper search EF、Dynamic EF base/factor；
- upper external labels、预处理前的 vectors 和每个 upper point 的完整 shard membership；
- portable upper HNSW entry point、levels 和逐层 adjacency。

Orion production router 不接受 graphless artifact，也不会静默改用 brute-force upper scan。
brute-force 构造函数只用于测试。

### 1.1 Native Simple KMeans 公平 baseline

Simple KMeans 不是 Orion 路由器的特殊模式。它有独立 production artifact 和独立
server-side router，artifact 记录 generation、完整 vector schema、shard/count/layout
binding、`routing_distance=squared_l2`、`nprobe`、`lower_hnsw_ef` 以及每个 numeric shard
唯一的 centroid。构建和加载时强制：

- `physical_point_count == logical_point_count`；
- 每个 point 恰好一个 shard membership；
- 每个 numeric shard 恰好一个 centroid；
- 不包含 Orion `upper_graph`、MultiEP 或 Dynamic EF 字段。

查询时 coordinator 对所有 centroids 做精确 squared-L2 排序，只 fan-out 到最近
`nprobe` 个 numeric shards。该语义严格复用原 Simple KMeans baseline：Cosine collection
只对 query 做 cosine 预处理，arithmetic-mean centroids 保持 raw 值，不把 centroid 再次
归一化。lower shard 使用普通 HNSW entry point 和 artifact 固定 EF，然后回到正常
`ShardReplicaSet` 与 collection merge。

### 2. Graphless writer、Rust builder 与 full-layout 强绑定

现有 Method4 harness 在完成原算法的 upper sampling、topology convergence、fission 和
multi-assignment 后，通过：

```python
write_orion_graphless_artifact(...)
```

把 `upper_indices`、完整 `point_to_shards` 和路由参数写入 typed graphless artifact。
writer 同时对**全量 point-to-shards layout**生成 canonical SHA-256。canonical assignment
逐行格式为：

```json
{"id":0,"shards":[1,7]}
```

每行以 `\n` 结束；shard IDs 已验证、去重并按稳定顺序编码。artifact 还记录：

```text
layout_sha256
logical_point_count
physical_point_count
```

Rust production builder 使用 Qdrant 标准 `GraphLayersBuilder` 构建 upper HNSW：

```bash
CARGO_TARGET_DIR=/proj/intelisys-PG0/exp/orion-distributed/cargo-target-native \
  tools/cargo_in_docker.sh run --release -p collection --example orion_build_artifact -- \
  graphless-orion.json generation-1.json \
  --seed 100 --m 32 --ef 100
```

builder 使用稳定 upper-node 插入顺序和固定 seed，输出 canonical JSON、
`generation-1.json.sha256`，并在发布 checksum 前重新加载 artifact、构造 production
`OrionRouter`。它原样保留 layout SHA 和两个 count，因此 upper graph、routing membership
和后续全量 numeric-shard assignments 不能被不同 run 的文件静默拼接。

Simple KMeans 使用 `write_simple_kmeans_graphless_artifact(...)` 和独立 Rust
`simple_kmeans_build_artifact` canonicalizer。它复用现有
`build_cpp_kmeans_baseline_assignments` 生成的单分配 layout，不复用 Orion 的 topology、
multi-assignment 或 upper graph。两种 layout CLI 都输出 `build-manifest.json` 和
`checksums.sha256`，后续 prepare 阶段只接受完整 `production_bundle`，并重新验证清单中的
每个文件。

### 2.1 Offline attachment EF 与 faithful runtime upper search

原始 main idea 在构建阶段明确执行 `up_tier_index->setEf(100)`，再为每个 L0 point 搜索
`K_OVERLAP` 个 L1 attachments；在线查询阶段则先执行
`up_tier_index->setEf(EF_SEARCH_UP)`，随后调用
`searchKnn(query, EF_SEARCH_UP)`。因此 native 实现需要同时保留两个边界：

- `--attachment-search-ef` 只控制 offline L0-to-L1 attachment，faithful Orion 固定为
  `100`，不会因 runtime profile 调整而改变 layout membership；
- production artifact 分别记录 runtime `upper_search_ef` 和 `upper_k`，但 faithful main
  idea 要求二者相等，因为原 C++ 用同一个 `EF_SEARCH_UP` 同时作为 HNSW beam 和
  `searchKnn` 返回数量；
- layout CLI 默认强制 `upper_search_ef == upper_k`；
  `--allow-decoupled-runtime-upper-search` 只允许生成 diagnostic artifact，不能用于正式
  Orion 对比；
- prepare 会校验 `attachment_search_ef=100`、runtime 参数与 artifact 一致，并继续拒绝
  `upper_search_ef != upper_k` 的 diagnostic layout；它还 fail-closed 锁定原 main idea 的
  initial `P=31`、`sample_denominator=32`、upper sample/HNSW seeds 与 build 参数、`K_OVERLAP=10`、
  KMeans/拓扑迭代参数、启用 fission，以及原始完整 multi-assignment 规则
  (`max_vote>=2`、只取并列最高票、membership union 不设上限)。关闭 multi-assignment、
  限制每点 shard 数、扩大 vote delta 或关闭 fission 的 artifact 均不能进入正式
  collection；faithful 参数会写入版本化 collection provenance，其中 attachment EF 显式
  记录，runtime pair 由已校验的 artifact SHA 绑定。

上述 smoke 的旧 build manifest 尚未单列 `attachment_search_ef`，但当时旧代码把
`upper_search_ef=100` 用于 attachment，因此 offline attachment 实际仍为 100；其 production
artifact 则是 `upper_k=16, upper_search_ef=100`。这组 16/100 参数不满足上面的 faithful
runtime 约束，所以该 run 只能证明 native distributed plumbing、artifact 激活和标准 API
路径，不能作为 Orion 算法性能证据；其 32-query route trace 也只能解释该 diagnostic
artifact。正式实验必须用新入口重新构建，且每个 Orion profile 都同时设置相等的
`upper_k` 与 `upper_search_ef`。

### 3. Generic numeric auto-shard 离线导入

`write_orion_numeric_shard_import_bundle(...)` 从同一份 `train` 和
`point_to_shards` 生成：

```text
<prefix>.f32le
<prefix>.assignments.jsonl
<prefix>.manifest.json
generation-<N>.json
```

production artifact 必须与 bundle 位于同一输出目录。bundle writer 会重新计算
assignments SHA、记录 production artifact 的 file SHA，并拒绝以下 layout/schema
不一致；Rust importer 随后会重新验证 manifest 中记录的所有 file SHA：

- Orion artifact 缺少 `upper_graph`，或 Simple KMeans artifact 错带 `upper_graph`、不使用
  squared-L2 routing、不是严格 single-copy layout；
- production artifact、vectors 或 assignments 的实际 file SHA 与 manifest 不一致；
- artifact `layout_sha256` 与完整 assignments SHA 不一致；
- generation、shard count、logical count 或 physical count 不一致；
- vector name、dimension 或 datatype 不一致。

importer 同时支持两种 manifest contract。legacy Orion v1 使用：

```json
{
  "format_version": 1,
  "orion_generation": 7,
  "orion_artifact_file": "generation-7.json",
  "orion_artifact_sha256": "..."
}
```

generic v2 把 policy binding 泛化为：

```json
{
  "format_version": 2,
  "routing_policy": "simple_kmeans",
  "routing_generation": 8,
  "routing_artifact_file": "generation-8.json",
  "routing_artifact_sha256": "..."
}
```

`routing_policy` 可为 `orion` 或 `simple_kmeans`。当前 Orion layout CLI 为兼容既有 bundle
仍输出 v1；Simple KMeans layout CLI 输出 generic v2。`write_numeric_shard_import_bundle_v2`
也可为任一 policy 生成 v2。Simple KMeans 的 v2 校验额外要求无 `upper_graph`、
`routing_distance=squared_l2`、每个 logical point 只有一个 physical copy；跨 policy 的
`--resume` checkpoint 会被拒绝。新 checkpoint 使用 generic v2，但仍可读取 legacy Orion
v1 checkpoint。

Rust importer：

```text
examples/orion_numeric_shard_import.rs
```

使用 Qdrant internal `PointsInternal/Upsert` 的显式 numeric `shard_id`。同一个 external
point ID 会被写入其所有 memberships；不使用 custom shard key、synthetic copy ID 或
`source_id` payload。写入从指定 coordinator 进入，但仍调用普通
`ShardReplicaSet::update_with_consistency`，并只允许 Medium 或 Strong ordering。导入器还会：

- 通过 controller `/cluster` 与 collection cluster endpoint 发现真实 shard owners；
- 对 owner 使用 internal exact count/config 检查；
- fresh import 要求所有目标 numeric shards 为空；
- `--resume` 只接受完全相同的 manifest、bundle 和 checkpoint；
- 流式读取时再次计算 vectors/assignments SHA；
- 导入后逐 shard 校验 exact count 和总 physical copy count。

这是一条受控静态构建路径，不等价于普通 client online upsert。

### 4. 四节点 artifact 安装、重启与加载证明

集群编排器提供两个 policy-specific 安装入口：

```text
install-orion-artifact
install-simple-kmeans-artifact
```

它只向指定 `run-id` 的四个 storage roots 安装 canonical builder 输出，并执行：

- generation、file SHA、layout SHA、schema、shard count、logical/physical counts 校验；
- collection policy generation/SHA 校验；
- collection `points_count == physical_point_count` 校验；
- numeric shard placement、replica state、peer health 和 transfer 状态校验；
- staging、同目录原子 rename、已有相同 SHA 复用、已有不同 SHA 拒绝；
- 只重启带本 run 精确 labels 的容器；
- workers-first 或 controller-first 顺序重启；
- 每节点 `/readyz`、四 peer membership 和 collection health 复查；
- 四节点日志均出现对应 policy 的 `Loaded ... routing generation N for collection ...`；
- 重启后出现对应 router 的 fallback/unavailable warning 即判定激活失败；
- 安装、节点 SHA、collection proof 和 log proof 写回 run manifest。

这不是 consensus hot activation。当前 generation 的生效方式仍是安装全部节点后安全
重启。

### 5. 标准 Search/Query 的 server-side route

客户端不发送 shard、EP 或 EF map：

```text
standard dense nearest Search / Query
  -> 任意收到请求的 Qdrant coordinator
  -> Orion: production upper HNSW -> 完整 membership union -> ordered MultiEP/Dynamic EF
     或 Simple KMeans: exact centroid squared-L2 -> nearest nprobe -> fixed lower EF
  -> 普通 ShardReplicaSet::core_search
  -> Qdrant 原生 distance-aware global merge
  -> external point-ID dedup、offset、limit
```

Orion upper scoring 使用 Qdrant 自身的 Cosine、Dot、Euclid 或 Manhattan metric 实现；
vectors 在 router 构造时预处理，query 每次只预处理一次。upper HNSW batch routing 按
`search_thread_count` 切成有界 chunk，在 Qdrant search runtime 的 blocking tasks 中并行
执行；它计入原始 request timeout，lower shard RPC 只能使用扣除 upper route 后的剩余
deadline。这个实现避免 batch=200 时为每条 query 各投递一个不可协作取消的 blocking
task，同时不改变每条 query 的 upper search、membership union 或 Dynamic EF。Simple
KMeans routing 则固定为与原 baseline 一致的 squared-L2 centroid ranking。

每个被选中的 numeric shard 都只需要返回该 query 的
`top-(limit + offset)` candidates：coordinator 把 shard request 改为
`limit=client_limit+client_offset, offset=0`，完成跨 shard score merge 和 external-ID
dedup 后才应用一次 client offset/limit。这个 candidate envelope 与不在 shard 端提前应用
offset 的全局 top-k 语义等价：若某个 ID 在它所在 shard 的前 `limit+offset` 之外，则该
shard 内已经至少有同样数量的不同 IDs 排在它之前，它不可能进入全局前
`limit+offset`。multi-assignment copies 在单个 shard 内仍是唯一 external IDs，所以这一
论证不因跨 shard copies 而失效。

`top-(limit+offset)` 是返回结果上限，不是 HNSW beam。Orion 的 per-shard HNSW exploration
仍严格使用 `EF = dynamic_ef_base + dynamic_ef_factor * ordered_unique_EP_count`；Simple
KMeans 使用 artifact 固定 lower EF，HashAll 使用该 case 的普通 scalar HNSW EF。三种方法
都受到相同的 per-shard result envelope 和同一个 collection global merge，不给 Orion
额外返回候选；它们的 EF 差异是被比较、记录和汇报的算法搜索预算。因而压缩 shard 返回
结果不会削弱 Dynamic EF 本身，也不会造成 Orion 独占的 candidate-count 优势。

两种 routed policy 当前共同的接管条件为：

- `ShardSelectorInternal::All`；
- dense `Nearest`，vector name 与 artifact 一致；
- 无 filter、无 exact、无 prefetch；
- 不携带旧实验用 `hnsw_entry_points`、per-shard EP/EF 或 source-ID dedup hints。

标准 Search 和简单 `/points/query` dense-nearest 请求复用同一条路径。batch 中只要有
一个请求不满足条件，整个 batch 回退 all-shards，绝不 partial takeover。回退时非
HashAll collection 禁用只对随机 hash 分片成立的 probabilistic shard undersampling；
Orion 的 Query merge 继续按 external ID 去重 multi-assignment copies。

“不满足接管条件”的算法级回退与“配置的 artifact 没有成功加载”严格区分。后者现在
fail closed：声明 Orion 或 Simple KMeans policy、但对应 router 为 unavailable 时，普通
`ShardSelectorInternal::All` coordinator Search/Query 返回明确错误，不会静默变成另一种
all-shards 算法；显式 numeric-shard internal/import 路径仍可用于受控构建和恢复。

### 6. 保留的 Qdrant 分布式语义

选中的 numeric shards 仍通过普通 `ShardReplicaSet` 执行，因此代码路径保留：

- 任意 peer 作为 coordinator；
- local-first replica selection 和正常 remote fallback；
- replication factor；
- read consistency 与受控导入的 write consistency；
- 正常 local/remote shard RPC；
- replica state、transfer 和 failure reporting；
- collection-level distance-aware merge、offset/limit 和 external-ID dedup。

Native Orion 和 native Simple KMeans 都没有使用旧实验中 RF=1 专用的 direct-worker
请求、custom shard-key fan-out 或 worker pre-merge 快捷路径。公平对比时，三种方法都
必须从标准 coordinator API 进入。

### 7. Segment、snapshot 与静态安全边界

如果 routed EP 在某个优化/新建 segment 中不存在、为空或被 filter 排除，该 segment 会
回到自己的标准 HNSW entry point；存在有效 EP 时仍使用 ordered MultiEP。这避免了
segment 生命周期造成错误空结果。

声明 Orion 或 Simple KMeans policy 的 collection snapshot 必须包含并重新校验固定路径
artifact。artifact 缺失、checksum、generation、schema 或 shard count 不匹配时 snapshot
失败。
目前已有 snapshot 内容与失败边界测试，但尚缺完整的
`snapshot -> restore -> Collection::load -> standard routed Search` 端到端恢复测试。

static artifact 不能与 hash-ring reshard 原子联动，因此两种 routed collection 当前都
明确拒绝 reshard。普通 client writes 也被拒绝，避免静默退回 point-ID hash sharding 并
破坏 offline layout。peer/internal explicit-shard update、恢复和受控 importer 仍可使用。

## 可复现的静态构建、准备与测量入口

下面的命令是当前仓库已有 CLI 的直接入口，可复现 artifact、numeric auto-shard
collection、受控导入、artifact 激活、标准 API benchmark 和 matrix 汇总。命令存在并不
代表 native 四节点正式实验已经运行；本节不声明任何尚未产生的 Recall–QPS 结果。

所有 layout、import bundle、proof 和 benchmark 输出都必须放在仓库外，不能提交 HDF5、
`.f32le`、assignments、artifact、index 或 `results/` 大文件。

### 0. 公共变量与 Rust Docker runner

```bash
export RUN_ID=native-20260720-smoke-v1
export TOPOLOGY=tools/distributed/cloudlab_orion_4node.json
export DATASET=/users/dry/orion-distributed/datasets/glove-200-angular.hdf5
export BASE_URL=http://10.10.1.1:6333
export PYTHON=/users/dry/orion-distributed/venv/bin/python
export GENERATION=1
export INITIAL_ORION_SHARDS=31
export ARTIFACT_ROOT=/users/dry/orion-distributed/artifacts/$RUN_ID
export PROOF_ROOT=/users/dry/orion-distributed/proofs/$RUN_ID
export MATRIX_ROOT=/users/dry/orion-distributed/native-matrix
export CARGO_TARGET_DIR=/proj/intelisys-PG0/exp/orion-distributed/cargo-target-native
export DEPLOYMENT_MANIFEST=/proj/intelisys-PG0/exp/orion-distributed/$RUN_ID/manifest.json
mkdir -p "$ARTIFACT_ROOT" "$PROOF_ROOT" "$MATRIX_ROOT"
```

controller 不要求宿主机预装 Cargo。`tools/cargo_in_docker.sh` 在 image 缺失时从
`tools/docker/Dockerfile.rust-tools` 构建 Rust 1.94 tool image，然后以 host network 运行，
挂载仓库、共享 `/proj/intelisys-PG0`、本地 `/users/dry/orion-distributed` 和 named Cargo
registry cache。默认 target 位于共享大盘，不占用仓库所在根分区。可先单条验证：

```bash
CARGO_TARGET_DIR="$CARGO_TARGET_DIR" tools/cargo_in_docker.sh --version
```

后续 layout CLI 使用 `--cargo tools/cargo_in_docker.sh`；也可以直接把任意 Cargo 子命令
交给这个 runner。

### 1. 四节点 lifecycle

四节点固定为 controller/client `10.10.1.1` 和 workers
`10.10.1.2/10.10.1.3/10.10.1.4`。各阶段均为单条幂等命令：

```bash
python3 tools/method4_distributed_cluster.py --topology "$TOPOLOGY" --run-id "$RUN_ID" --expected-commit "$(git rev-parse HEAD)" bootstrap
```

```bash
python3 tools/method4_distributed_cluster.py --topology "$TOPOLOGY" --run-id "$RUN_ID" --expected-commit "$(git rev-parse HEAD)" build
```

```bash
python3 tools/method4_distributed_cluster.py --topology "$TOPOLOGY" --run-id "$RUN_ID" --expected-commit "$(git rev-parse HEAD)" deploy
```

```bash
python3 tools/method4_distributed_cluster.py --topology "$TOPOLOGY" --run-id "$RUN_ID" status
```

```bash
python3 tools/method4_distributed_cluster.py --topology "$TOPOLOGY" --run-id "$RUN_ID" manifest
```

正式数据生成前必须确认四 peers、私网 URI、相同 image digest、controller/client CPU
集合和 worker 资源配置。`manifest` 写入上面的 `$DEPLOYMENT_MANIFEST`。CloudLab 节点
启用了 SMT；固定的 controller `0-7` 与 client `8-19` 虽没有重复逻辑 CPU 编号，但
`0-7` 与 `10-17` 是 sibling threads，因此不能声称物理核心完全隔离。三种方法仍使用
完全相同的 affinity，这一共享物理核心限制必须写入 QPS/P95/P99 结果说明。

远端 orchestration command 使用 non-login `bash -c`，不再使用 `bash -lc`。CloudLab 默认
`.bash_logout` 曾在远端 `set -e` 脚本主体成功后把 SSH exit status 改成 1；绕开 login
profile 可以消除这种假失败，同时仍保留 `BatchMode=yes`、连接超时和 run-id 资源边界。
本地 controller command 仍可使用本地 `bash -lc`。

### 2. Orion 与 Simple KMeans layout CLI

Orion 的单条 production build 命令会依次执行 dataset preprocessing、upper sample/HNSW、
L0-to-L1 attachment、原 topology convergence/fission/multi-assignment、graphless artifact、
Rust production upper-HNSW builder、numeric import bundle、build manifest 和 checksums：

```bash
export ORION_LAYOUT=$ARTIFACT_ROOT/orion-profile-a
"$PYTHON" tools/orion_native_layout.py --hdf5-path "$DATASET" --output-dir "$ORION_LAYOUT" --generation "$GENERATION" --p "$INITIAL_ORION_SHARDS" --vector-distance cosine --sample-denominator 32 --upper-m 32 --upper-ef-construction 100 --attachment-search-ef 100 --upper-search-ef 100 --upper-k 100 --k-overlap 10 --dynamic-ef-base 20 --dynamic-ef-factor 4 --cargo tools/cargo_in_docker.sh --cargo-target-dir "$CARGO_TARGET_DIR"
export EFFECTIVE_SHARDS="$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["routing"]["effective_num_shards"])' "$ORION_LAYOUT/build-manifest.json")"
```

该入口复用 `qdrant_two_level_routing_experiment.py` 中的原 Orion build，不能通过导出阶段
改成 single assignment、截断 membership 或 adaptive shard pruning。输出目录必须事先
不存在；需要小规模 layout smoke 时可增加 `--train-limit 5000`，但该 smoke artifact 不能
与全量 collection 混用。这里的 `31` 是 fission 前 initial P；必须从 Orion manifest 读取
`EFFECTIVE_SHARDS`，再让 HashAll 和 Simple KMeans 使用该最终数量，不能预先把历史有效值
`46` 直接作为 Orion 的 initial P。

#### Build-once / derive-profile 公平性边界

Offline attachment upper HNSW 与原 C++ 一样采用并发插入。固定 seed 只能固定随机流起点，
不能固定线程完成顺序；同一组 offline 参数独立重建时，具体 HNSW realization、L0-to-L1
attachments、fission 和 multi-assignment layout 仍可能不同。2026-07-20 的两次 full 诊断
构建虽然都得到 46 个 effective shards，但 layout SHA、shard counts 和 physical copies
不同，因此已在 prepare 前拒绝，不能作为同一 runtime sweep。

正式 Orion 曲线必须先构建一次 canonical offline bundle，再从该 bundle 派生其它 runtime
profile：

```bash
"$PYTHON" tools/orion_native_runtime_profile.py \
  --source-layout-dir "$ORION_LAYOUT" \
  --output-dir "$ARTIFACT_ROOT/orion-profile-b" \
  --generation 1 \
  --upper-k 96 \
  --dynamic-ef-base 64 \
  --dynamic-ef-factor 15 \
  --payload-mode copy \
  --cargo tools/cargo_in_docker.sh \
  --cargo-target-dir "$CARGO_TARGET_DIR"
```

派生器只允许改变 generation、`upper_k == upper_search_ef` 与 Dynamic EF base/factor；它
复制并重新校验相同 `.f32le`/assignments，使用 Rust typed writer 生成新 artifact，且要求
ordered upper nodes、完整 shard memberships、`layout_sha256`、physical count、routing summary
和 rebuilt upper graph 与 source exact equality。正式默认 `--payload-mode copy` 使用独立
inode；显式 hardlink 仅作节省空间的诊断，不能标为正式证据。

Simple KMeans 的单条 production build 命令复用现有 C++ baseline assignment 语义，并
生成独立 artifact 与 generic v2 import bundle：

```bash
export SIMPLE_LAYOUT=$ARTIFACT_ROOT/simple-kmeans-profile-a
"$PYTHON" tools/simple_kmeans_native_layout.py --hdf5-path "$DATASET" --output-dir "$SIMPLE_LAYOUT" --generation "$GENERATION" --p "$EFFECTIVE_SHARDS" --vector-distance cosine --nprobe 32 --lower-hnsw-ef 80 --kmeans-train-size 10000 --kmeans-iters 10 --kmeans-seed 1 --cargo tools/cargo_in_docker.sh --cargo-target-dir "$CARGO_TARGET_DIR"
```

两种 CLI 都打印 production artifact、artifact SHA、import manifest、logical/physical count、
layout SHA 和 checksums 路径。这里的 profile 参数只是可执行示例，不表示已达到任何目标
recall。

### 3. 统一 prepare CLI

`native_auto_shard_prepare.py` 把手工 collection 创建、placement、导入和安装合并为一个
可审计阶段。它只接受 `sharding_method=auto`、RF=1、三个 worker 的 exact round-robin
numeric placement；已有 collection 只有在 schema、HNSW、policy、count、版本化 provenance
metadata 和安全 placement 完全匹配时才复用。HashAll metadata 绑定 dataset SHA、train
shape/count、distance、vector name 和 shard count；routed metadata 绑定 method、layout SHA、
artifact SHA/generation 以及 logical/physical counts。每个 `--output-dir` 必须是仓库外的新
目录。

默认 HashAll 使用普通 public upsert 和原 external IDs，不使用 shard selector：

```bash
export HASH_COLLECTION=native_hash_all_$GENERATION
"$PYTHON" tools/native_auto_shard_prepare.py --method hash_all --topology "$TOPOLOGY" --run-id "$RUN_ID" --collection "$HASH_COLLECTION" --base-url "$BASE_URL" --output-dir "$PROOF_ROOT/prepare-hash-all" --hdf5-path "$DATASET" --p "$EFFECTIVE_SHARDS" --vector-distance cosine --hnsw-m 32 --ef-construct 100 --batch-size 512
```

Orion 使用 layout 中的 importer manifest，导入所有 multi-assignment copies，再调用
`install-orion-artifact` 以 workers-first 顺序激活：

```bash
export ORION_COLLECTION=native_orion_$GENERATION
"$PYTHON" tools/native_auto_shard_prepare.py --method orion --topology "$TOPOLOGY" --run-id "$RUN_ID" --collection "$ORION_COLLECTION" --base-url "$BASE_URL" --output-dir "$PROOF_ROOT/prepare-orion" --layout-dir "$ORION_LAYOUT" --hnsw-m 32 --ef-construct 100 --batch-size 16384 --cargo-target-dir "$CARGO_TARGET_DIR"
```

Simple KMeans 使用同一个 generic importer，再调用
`install-simple-kmeans-artifact`：

```bash
export SIMPLE_COLLECTION=native_simple_kmeans_$GENERATION
"$PYTHON" tools/native_auto_shard_prepare.py --method simple_kmeans --topology "$TOPOLOGY" --run-id "$RUN_ID" --collection "$SIMPLE_COLLECTION" --base-url "$BASE_URL" --output-dir "$PROOF_ROOT/prepare-simple-kmeans" --layout-dir "$SIMPLE_LAYOUT" --hnsw-m 32 --ef-construct 100 --batch-size 16384 --cargo-target-dir "$CARGO_TARGET_DIR"
```

routed import 中断时，只能在 exact bundle/checkpoint/collection 一致时对新的 proof 输出
目录使用 `--resume`。`16384` 只减少离线 importer 的串行 gRPC batch 数，不进入 query
计时，也不改变 point assignment、HNSW 或路由语义。prepare 在新 collection 创建后先等待初始状态稳定，在导入目标
physical count 后再次等待：points count 精确匹配、collection green、optimizer `ok`、所有
shards Active、无 transfer，并要求 indexed count 达到目标；对于低于 indexing threshold
的小 segment，允许 indexed count 连续 30 秒不再变化后进入 full-scan exception，再连续
确认稳定 5 秒。readiness 状态和 completion mode 会写入 manifest；benchmark preflight 会
再次执行同一等待并额外要求 update queue 为空，正式计时不能与 indexing/optimizer backlog
重叠。

导入和 artifact 激活完成后，prepare 还会从 controller 分别发一条不含 shard、EP、
per-shard EF 或 source-ID hint 的标准 Search 与简单 Query，要求结果非空且 external IDs
唯一。prepare 成功会写 `preparation_manifest.json`，证明 collection config、provenance
metadata、image manifest、import/install command、标准 API smoke、logical/physical count、
indexing readiness 和最终 placement；它本身不宣称已经完成 Recall–QPS 测量。

### 4. 标准 Search/Query benchmark CLI

`native_auto_shard_benchmark.py` 只通过 controller 的标准 Search 或 Query API 发请求。
它拒绝 client shard selector、EP、per-shard EF 和 source-ID hint；HashAll 必须显式给一个
正常 public scalar `--hnsw-ef`，Orion/Simple KMeans 则禁止 client EF。

下面分别给出三种方法的单 case 命令：

```bash
taskset -c 8-19 "$PYTHON" tools/native_auto_shard_benchmark.py --method hash_all --base-url "$BASE_URL" --collection "$HASH_COLLECTION" --hdf5-path "$DATASET" --topology "$TOPOLOGY" --deployment-manifest "$DEPLOYMENT_MANIFEST" --output-dir "$PROOF_ROOT/bench-hash-ef40" --warmup-query-count 500 --eval-query-count 3000 --batch-size 200 --stability-repeats 3 --hnsw-ef 40
```

```bash
taskset -c 8-19 "$PYTHON" tools/native_auto_shard_benchmark.py --method orion --base-url "$BASE_URL" --collection "$ORION_COLLECTION" --hdf5-path "$DATASET" --topology "$TOPOLOGY" --deployment-manifest "$DEPLOYMENT_MANIFEST" --artifact "$ORION_LAYOUT/generation-$GENERATION.json" --output-dir "$PROOF_ROOT/bench-orion-a" --warmup-query-count 500 --eval-query-count 3000 --batch-size 200 --stability-repeats 3 --orion-route-trace --cargo-target-dir "$CARGO_TARGET_DIR"
```

```bash
taskset -c 8-19 "$PYTHON" tools/native_auto_shard_benchmark.py --method simple_kmeans --base-url "$BASE_URL" --collection "$SIMPLE_COLLECTION" --hdf5-path "$DATASET" --topology "$TOPOLOGY" --deployment-manifest "$DEPLOYMENT_MANIFEST" --artifact "$SIMPLE_LAYOUT/generation-$GENERATION.json" --output-dir "$PROOF_ROOT/bench-simple-a" --warmup-query-count 500 --eval-query-count 3000 --batch-size 200 --stability-repeats 3
```

每个 case 输出 `run_manifest.json`、`stability_runs.csv`、`final_metrics.csv` 和
`summary.json`；增加 `--write-per-query-metrics` 时另写 `per_query_metrics.csv`。preflight
会校验四节点、dataset SHA、image/commit、auto policy、RF、worker-only exact placement、
artifact generation/SHA/schema/shards/count 和 optimizer 状态，并把实际 CPU affinity 写入
manifest；`taskset -c 8-19` 负责固定 benchmark client CPU。

`--orion-route-trace` 在正式计时窗口之外临时导出已经按 benchmark distance 预处理的 eval
queries，并调用 production `OrionRouter` example；临时 `.f32le` 会删除，经过 artifact/query
SHA、generation、layout、count 和 dimension 校验的 trace JSON 与日志保留在 case 输出中。
因此 matrix 可以绘制 Orion 的 exact offline visited-shards/EF-sum，而不会把 artifact 参数
冒充执行数据，也不会把 trace 时间计入 QPS。

### 5. Orion production router 的 exact offline route trace

`orion_route_trace` 对 production artifact 和 query rows 直接调用与 server 相同的
`OrionRouter`，精确输出每个 query 的 selected shards、ordered unique EP、per-shard EF、
EF-sum，以及 aggregate P50/P95/P99。输入必须是恰好
`query_count * dimension * 4` bytes 的 row-major little-endian f32。下面先从 HDF5 导出
3,000 条 raw query；Cosine preprocessing 由 production router 自身完成：

```bash
export QUERY_COUNT=3000
export QUERY_F32LE=$PROOF_ROOT/glove-test-$QUERY_COUNT.f32le
"$PYTHON" - "$DATASET" "$QUERY_F32LE" "$QUERY_COUNT" <<'PY'
import h5py
import numpy as np
import sys

with h5py.File(sys.argv[1], "r") as handle:
    np.asarray(handle["test"][: int(sys.argv[3])], dtype="<f4").tofile(sys.argv[2])
PY
```

然后在计时窗口外执行单条 trace 命令：

```bash
CARGO_TARGET_DIR="$CARGO_TARGET_DIR" tools/cargo_in_docker.sh run --release -p collection --example orion_route_trace -- "$ORION_LAYOUT/generation-$GENERATION.json" "$QUERY_F32LE" "$QUERY_COUNT" 200 "$PROOF_ROOT/orion-route-trace.json" --per-query
```

输出文件已存在时命令拒绝覆盖。该 trace 是给定 artifact/query 的**精确 offline routing
plan**，适合补充 Orion 的 visited-shards、EP 和 EF 成本分析；它不接触 Qdrant，不证明
某次 live request 实际执行了相同 plan，也不能并入 QPS 计时。当前 benchmark CLI 因没有
server-side per-query trace，仍把 Orion 的 `visited_shards` 和 `ef_sum_per_query` 记录为
`null/unknown_without_server_trace`，不会拿 artifact 参数冒充实际观测。显式启用
`--orion-route-trace` 时，benchmark 会在正式计时前用同一批预处理 query 调用 production
`OrionRouter` example，并把来源标为 `exact_offline_production_router_trace`；该 replay 的
耗时不计入 QPS。

### 6. 三方法 native matrix CLI

仓库提供通过测试的配置模板：

```text
tools/benchmark_configs/native_auto_shard_glove200_initial.example.json
```

把模板复制到仓库外，替换 dataset/deployment/artifact 路径和已 prepare 的 collection
名称。config 必须至少包含 HashAll、Orion、Simple KMeans 三种 method；HashAll case 必须
有 `hnsw_ef`，routed case 必须有 artifact 且不得有 client EF。执行整个矩阵的单条命令：

```bash
cp tools/benchmark_configs/native_auto_shard_glove200_initial.example.json "$PROOF_ROOT/native-matrix.json"
```

完成上述替换后，执行整个矩阵：

```bash
"$PYTHON" tools/native_auto_shard_matrix.py --config "$PROOF_ROOT/native-matrix.json" --run-id native-glove-initial-v1 --output-root "$MATRIX_ROOT" --taskset-cpus 8-19 --run
```

若 case 原始目录已经存在，仅重新校验 provenance 并汇总：

```bash
"$PYTHON" tools/native_auto_shard_matrix.py --config "$PROOF_ROOT/native-matrix.json" --run-id native-glove-initial-v1 --output-root "$MATRIX_ROOT" --taskset-cpus 8-19 --collect-only
```

matrix 保留每个 case 的原始输出和 stdout/stderr，并生成
`recall_qps_points.csv`、`pareto_frontier.csv`、`same_recall_selection.csv`、
`same_recall_confirmation.csv`、共享 provenance manifest，以及 Recall–QPS、
Recall–latency、Recall–visited-shards、Recall–EF-sum 图。它默认选择 0.90/0.95、窗口
0.003 的点，并单独报告三方法 pairwise recall spread。screen 配置可保持
`require_strict_same_recall=false` 以保留 nearest 诊断；正式 confirmation 必须设为 `true`，
此时任何 nearest selection 或 pairwise spread 超过 `same_recall_pairwise_window` 都会使
汇总失败。matrix 还强制三方法 commit、image、numeric shard count、RF/WCF、HNSW、
optimizer 参数和 exact placement 一致，并拒绝 tracked-dirty、active transfer、非空 update
queue、未稳定 collection 或 `fully_indexed != true`。同一 routed 方法的多个 runtime
profiles 还必须具有相同 offline-layout fingerprint：dataset、offline 参数、routing/fission
summary、vectors/assignments SHA、ordered routing structure、logical/physical counts 和 shard
counts 均一致。小数据 smoke 可以记录
`stable_small_segment_full_scan_exception` 作为功能证据，但该状态不能进入正式 Recall-QPS
matrix。原始 null routing costs 保持 null，不会被强填。上述命令和模板是可执行入口，
不是已经运行完成的正式结果。

## 正式实验的公平性与验收标准

三种方法的主对比必须满足：

1. 相同 Qdrant commit、image digest、四节点拓扑、numeric shard count、RF、read/write
   consistency、HNSW build 参数、数据集、query 集和计时窗口。
2. 所有主请求从标准 Search/Query API 进入正常 coordinator；客户端不得发送 shard、
   EP、per-shard EF map，也不得直接请求 worker 测量主结果。
3. 相同 round-robin physical placement。Orion-aware placement 只能作为独立 ablation。
4. Orion 必须保留完整 membership union、multi-assignment、ordered MultiEP、Dynamic EF
   和 external-ID dedup；禁止 adaptive shard pruning。faithful profile 还必须保持
   `attachment_search_ef=100` 且 runtime `upper_search_ef == upper_k`；若调整
   `EF_SEARCH_UP`，二者必须同步调整；prepare 必须拒绝关闭 fission/multi-assignment、
   截断 membership union 或改变原始投票阈值的 artifact。
5. controller/client 逻辑 CPU 集合、worker CPU、optimizer/indexing 状态和网络条件一致；
   固定 cpuset 存在 SMT sibling overlap，必须作为所有方法共同的实验限制报告。
6. 正式计时期间四 peers healthy、collection `fully_indexed=true`、无 optimizer backlog、
   update queue 为空且无 transfer。
7. 报告 Recall@10、QPS、P50/P95/P99、visited logical shards、per-shard EF、EF-sum、
   physical peers、network bytes、CPU、logical/physical counts 和 expansion ratio。
8. 同召回结论必须给出 raw rows 和重复波动；不能因 Orion 某次较慢就隐藏或丢弃结果。
   应先诊断 artifact/layout binding、upper routing CPU、MultiEP fallback、candidate budget、
   RPC 调度、timeout accounting、placement 和资源隔离；任何修复都只能改善分布式实现，
   不能改变 main idea 或给 Orion 独占资源。

## 尚未完成的系统边界

以下能力仍未完成：

- online Orion-aware upsert 与 voting/multi-assignment，以及 online Simple KMeans centroid
  assignment；
- replicated `external ID -> numeric shard copies` directory；
- vector/payload update 后的原子 copy relocation；
- delete、overwrite 和 batch failure 时删除/回滚全部 copies；
- multi-assignment 下标准 count、facet、scroll 等 API 的 logical-point 语义；
- consensus-controlled generation activation、hot reload 和原子 rollback；
- artifact 自动下发到新 peer；
- routed-policy-aware online reshard、rebalance 与 generation metadata 联动；
- filter、hybrid、prefetch、sparse/multivector 的 routed takeover；
- snapshot restore-to-routed-search 端到端验证；
- RF>1、read consistency、replica failure 和网络分区的 live 验证；
- native numeric-shard 架构上的四节点 Orion/Naive/Simple KMeans Recall–QPS 正式重跑。

因此当前可以声称：

> Orion 已经拥有 Qdrant collection 内的 server-side upper HNSW 读取路径；Orion 与
> Simple KMeans 都已拥有 numeric auto-shard 静态数据布局、完整 layout 强绑定、受控导入、
> 四节点可验证 artifact 激活和标准 API benchmark/matrix 入口。

当前不能声称：

> Orion 或 Simple KMeans 已经完整替代 Qdrant 默认分布式 CRUD 生命周期，或旧实验已经
> 证明了新 native 架构的 Recall–QPS 优势。
