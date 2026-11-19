# Orion / KMeans 分片局部性实验

下次完整构建优先使用 [加速构建入口与手册](orion_fast_build.md)，不要重新走完整
串行关联记录 pass。`orion_fast_build.py` 自动执行原生一致性检查和 8 机迁移。

八机 100M 构建与端到端 Recall–QPS 的统一规范见仓库根目录的 [AGENTS.md](../AGENTS.md)。
本分片局部性专项使用 **Orion / KMeans 两组，SIFT100M，8 台机器共 8 个分片，
每机 1 个分片，统一 M=32、m0=64、ef_construct=200**。只测 Oracle / Real
召回率，不指定目标召回率，不做 QPS 或吞吐校准，不增加 Orion 参数扫描。

`config/shard_locality_sift100m.json` 已固定这组参数和 8 台机器。正式入口为
`shard_locality_archives.py`，默认只检查元数据。执行前检查集合配置、每个物理
HNSW 的参数、Orion 上层图构建参数和 8 机映射，任何不匹配直接拒绝。
原单机入口 `shard_locality.py --execute` 不能执行这份 8 机配置。

按最新实验周转加速要求，新任务的关联记录和下层构图使用各机全部可用物理核，
8 机并发，不再限定 NUMA 0；JSONL 分片物化按控制机物理核及可用内存选择进程数，
按源 ID 顺序拼接，逐行校验覆盖和副本数。当前 8 机均检测到 96 个可用物理核。
搜索阶段的资源配置和所有实验参数保持原设定。详见加速构建手册中的资源策略。
传输、归档以及按物理分片的独立搜索也并发执行，最终合并仍按固定 shard-ID 顺序。
METIS 全局划分保留原实现，不通过替换算法或重跑当前规划来追求 CPU 使用率。

构建线程数是资源调度变更，可能改变 HNSW 的并发插入次序，不承诺得到同一张图。
历史报告记录 Orion 构建线程数 48、复用 KMeans 索引历史构建线程数 8；新报告记录
各台机器实际构建线程数。两者 M=32、
ef_construct=200 等质量参数保持一致。正式查询参数不随构建并行度调整。

此前单机、Orion ef_construct=100 的运行已停止并标记 `SUPERSEDED.json`，
不作为正式对比。新的 32/200 Orion 布局、8 机索引和两组正式召回率测量现已完成，
原始路由和索引证据位于 `/mnt/rocksdb_nvme/shard-locality-matched8-20261117/results/`。
用户修订后的主指标为 `gt-coverage-paired-queries-v2`，新结果位于
`/mnt/rocksdb_nvme/shard-locality-matched8-20261117/results-gt-paired/`。
旧目录中的 ANN Real 与全集 Oracle 曲线保留作历史诊断，不是当前主指标。
仓库示例配置中的旧 Orion manifest 仅用于标识原输入；重测必须替换为新 run 的
`layout/import.manifest.json`，不能绕过参数检查。

## 构建期关联记录并行化

冻结的上层图无需重建。`orion_attachment_parallel` 用同一个原生上层导航函数，
按可用核数比例将数据行拆分到 8 台机器，每台使用其可用物理核数量的线程；
保留已完成的完整记录前缀，按原始行序
合并剩余分段。每台机器都重放同一份 8192 行样本，与原串行输出逐字节比较，
并检查图、输入分段和结果文件 SHA256。最终 `PreparedCapacity` 由 Rust 原生
`read_prepared` 验证后，原构建器以 `--reuse-prepared` 继续访问概率细化和全局划分。
这是计算调度变化，不改变 M、ef_construct、efs、上层图、分片算法或评估口径。
全局划分及容量约束仍使用原算法；不能将“关联行可独立搜索”推广为这些步骤也
能直接拆成八份。迁移和逐机校验证据保存在实验根目录 `parallel-attachments/`。

## 指标与横轴

固定数据集、查询集、GT Top-k 和布局的总分片数 S，横轴是一次查询访问的
**不同物理数据分片数 n**，不是改变布局的 S，也不是上层命中数或请求批次数。
默认 Recall@10；修改 `top_k` 时 GT 必须至少有相应列数。

* **Oracle**：对每条查询，知道全库精确 GT 后，选择至多 n 个分片，使其并集
  覆盖最多的不同 GT ID，再对 **Real[n] 的同一批 query** 取平均。
  无重叠的 KMeans 等价于将每个分片的 GT 个数降序累加。
  Orion 有多副本，不能直接相加；脚本在 GT 的 bitmask 上做精确动态规划，
  不是贪心，也不是把独立命中数最多的 n 个分片相加。后者在有重叠时未必
  达到上帝视角最优值。限制 k <= 20，默认 k=10，不限制 S 为 8。
* **Real**：使用原生路由选择分片，统计这些分片并集覆盖的不同 GT ID 数 / k。
  **假设选中分片内部的全部 GT 均能找回**，不计局部 HNSW 漏检。
  KMeans 扫描 `nprobe=1..S`，使用原生的平方 L2 质心排序，Cosine 查询先归一化，
  质心保留算术均值。Orion 调用 `OrionRouter::route_query`，保留原生上层
  HNSW 和 whitebox 筛选，保留真实上层路由限制；MultiEP 和局部 EF 仍在路由记录中，
  但不影响新的 GT 覆盖率评分。

按用户指定，主图横轴是整数 **n=1..S**，Real[n] 是实际访问恰好 n 个分片的
query 的召回率平均数，即 `mean(recall(q) for q if fanout(q) == n)`，不再以
平均访问数作横轴。每个点同时报告 query 数。无样本的组保留行，recall 为
空值/N/A；不填 0、不插值、不将参数值当成实际访问数。

Orion 不支持直接指定 `nprobe=n`。**不截断 targets、不按 GT 改写路由、不按
分片 ID 造访问顺序**。`real_profile` 指定主图采用的固定 Orion 配置；其他
配置单独输出，不把同一 query 在多个配置下的运行混为一组。KMeans 每个 n
采用 nprobe=n。`real_by_fanout.csv` 列出每个配置的全部整数分组和样本数。
分组内 query 的难度分布可能不同，曲线不要求单调；不能把这些条件均值解释为
同一组 query 在逐渐加大硬预算时的召回率。

当前 `Real` 就是之前的 `routed_coverage`。对某种方法定义
`Q_n = {q : 实际路由选择了 n 个不同分片}`，主图中 Oracle 和 Real 都对 Q_n 平均。
因此每个 n 上的 `Oracle - Real` 是同一组查询的路由选择损失，必须非负：

```text
Real(q) = unique_GT_coverage(actual_route(q)) / k
Oracle(q,n) = max_{|S| <= n} unique_GT_coverage(S) / k
Real[n] = mean_{q in Q_n} Real(q)
Oracle[n] = mean_{q in Q_n} Oracle(q,n)
Real[n] <= Oracle[n]
```

空组时 Oracle 和 Real **同时为 N/A**。全查询 Oracle 另存为补充数据，不混入
配对主图。Orion 的 Q_n 与 KMeans nprobe=n 的全查询集仍不同，因此还导出
`common_orion_query_groups.csv`，把双方都限制到 Orion 的同一批 Q_n，供跨方法
比较；使用的仍是原路由，没有新增配置或搜索。

默认 `collect_ann_diagnostic=false`：原生工作进程只读取实际 GT memberships，
不执行分片内 ANN 搜索。若显式打开 ANN 诊断，结果单独记为 `ann_recall`，
永远不代替 Real。上层路由的不完备性仍然保留，这正是需要与 Oracle 比较的差距。

## 数据和公平性

输入为原生 Orion `import.manifest.json` 及查询 NPZ（默认键 `queries`、`truth`）。
支持 manifest 的 f32le/u8 基向量及 float32/uint8 索引；uint8 当前仅支持 Euclid。
要求原始 ID 是连续的 `0..N-1`、assignment 按输入行排列，GT ID 来自同一全量数据集。
不支持把全量 GT 直接套到缩小的训练库；`query_limit` 仅缩小测试查询集。

Orion 直接复用已定型布局的全部 memberships，包括副本和 owner-entry completion。
当前正式入口复用已经核验的 KMeans 索引，不重训；下述训练方法仅属于原单机诊断入口。
诊断入口的 KMeans 在**同一个基向量文件**上构建互斥分片：从基向量均匀无放回采样，使用
scikit-learn 的 Lloyd KMeans / k-means++，配置固定 train_size、iterations、seed、
n_init，然后对全量基向量做最近质心分配。归一化策略与 Cosine baseline 相同，
最终分配按 f32 平方 L2、维度顺序累加及 shard-ID 打破平局。训练不使用测试查询
或 GT；GT 只用于后续评分。它是明确记录训练配方的新 KMeans 对照，不声称与
历史 C++/FAISS KMeans 布局逐位相同。

原单机诊断入口的两种方法下层图共用 M、ef_construct、seed、源 ID 升序插入，
逐分片确定性单线程构建。使用生产代码 `GraphLayersBuilder` / `GraphLayers::search`，每个
分片一个未量化 HNSW。KMeans 使用常规 HNSW 入口；Orion 仅从合法 MultiEP
进入 level 0，缺失 owner entry 直接报错。它测的是单机原生算法路径，不包括
线上服务的多 segment、调度、网络和优化器状态；新建下层图也不等同于历史线上图。

新的 Real 不受局部 EF 漏检影响；示例中 KMeans EF=64、Orion EF=32+4*entry_count
仅作为原始路由/可选 ANN 诊断参数记录。不存在相同计算量的 QPS 对比。
输出每个分片大小、总副本数和 replication_factor，避免把额外副本带来的增益
误称为相同存储预算下的收益。测试集上不挑选最优参数包络；所有预设点全部保留。

## 使用

### 无需重建或重搜的 GT 配对重统计

已有完整路由、GT 分片归属和逐查询 Oracle 时，优先使用：

```sh
python3 tools/shard_locality_gt.py \
  --source /absolute/path/completed-results \
  --output /absolute/path/new-gt-paired-results
```

该入口重新从 `truth.npy`、`gt-memberships.json` 和 `route-*.jsonl` 计算覆盖率与
精确 Oracle，使用独立的集合算法复核 bitmask 结果；不读取 ANN 返回结果来评分。
输出包括配对主图、逐 query 记录、每组样本数、双方同 Q_n 的对照、全集 Oracle
补充表和审计记录。源目录不被覆盖。

### 8 机正式入口与已有索引核验

`tools/shard_locality_archives.py` 支持从控制机的归档恢复到 8 台机器，保留
原有图字节。当前已核验的归档：

* Orion（32/100，不合格）：`/home/wangyiduo/orion-access-base-sift100m-20261116/archives/`
* KMeans（32/200，可复用）：`/home/wangyiduo/ann-native-orion-aligned-20261114/archives/sift100m/kmeans/`

每个归档只含一个本地分片和两个 segments（一个 HNSW、一个空 plain）；脚本
检查全部 0..7 分片、集合 schema、路由 artifact 绑定和每个被提取文件的 SHA256。
逐个分片提取到独立临时目录，在配置指定的物理机器上调用原生
`load_segment` / `search_batch`，使用完删除临时副本；原始归档不会被加载器
迁移或改写。主机列表的第 n 项固定处理 shard n，查询仅发给路由选择的分片。
控制机做原生路由及最终合并；这是只统计召回率的分布式批量重放，不测时延、
QPS，也不需要压测服务器。执行期间持有 8 台机器的共享实验锁，防止与其他
实验混跑；每台机器的 staging 使用其独立 `/dev/shm/orion-shard-locality/`。
Oracle 的 GT memberships 从
实际索引的 ID tracker 中读取，避免用重算的标签替代归档中的真实分片分配。

目前这对归档不会通过正式前置检查。已准备唯一匹配的 Orion 构建请求
`config/shard_locality_sift100m_build.json`（上层 M=32、ef_construct=200），
后续下层索引也必须使用 32/200；不能只修改 JSON 声称旧图符合新参数。
准备好后，将主配置的 `orion_manifest` 和下方路径指向新 bundle/归档。

```sh
python3 tools/shard_locality_archives.py \
  --config config/shard_locality_sift100m.json \
  --orion-archives /absolute/path/orion-M32-efc200-8shards/archives \
  --kmeans-archives /home/wangyiduo/ann-native-orion-aligned-20261114/archives/sift100m/kmeans \
  --native-binary /absolute/path/shard_locality \
  --output /absolute/path/new-locality-results \
  --execute
```

去掉 `--execute` 只检查小型元数据；归档需为可 seek 的未压缩 tar 及配套
`.manifest.json`。该入口只运行 `real_profile` 指定的 Orion 配置以及 KMeans
的 nprobe=1..8。旧 32/100 索引的历史路由记录中 6..8 为空组；新的 32/200
索引必须重新统计分组，不能沿用旧结果。如果新实验也出现空组，则保留行并标
N/A。调整配置必须在报表中注明，不能为填满图形改写路由。

### 原单机诊断入口（不是当前正式实验）

已提供当前本机可定位的 **SIFT100M / 8 分片**配置：
`config/shard_locality_sift100m.json`，使用 v5 access-probability-capacity 布局。
它是首个可运行的实例，不意味着实验只能使用 SIFT；其他数据集复制配置并替换
manifest、NPZ、distance/datatype 和 whitebox 参数即可。每个 S 使用独立配置和输出。

可以安全查看配置计划（不读大图、不扫描基向量、不搜索）：

```sh
python3 tools/shard_locality.py --config config/shard_locality_sift100m.json
```

在另一项实验结束后编译原生工具；正式运行使用上面的 8 机入口：

```sh
# Python 环境需要 numpy、scikit-learn、threadpoolctl、matplotlib。
# 本机已有 Rust 环境时可先 source：
# /home/wangyiduo/qdrant-hash-sift1m-20261022/system-opt/env.sh
# 独立 target，避免与其他实验共用构建目录。
export CARGO_TARGET_DIR=/absolute/path/locality-target
cargo build --release --locked -p collection --example shard_locality -j 2

# 单机诊断入口拒绝正式 eight_hosts 配置，不能用它替代 8 机实验。
```

默认一个下层分片驻留内存；上层路由阶段仍需容纳完整原生上层图，Python 还会
保留各配置的查询路由。ID 文件约 `8*(Orion physical copies + N)` 字节；临时
下层图搜索后删除。不会复制全量基向量，但逐分片读取会重复访问源文件。
100M 的图构建和全量 assignment 扫描仍然是重任务，不能在其他实验期间试跑。
输出目录必须是新目录；失败时保留日志与 `.partial.jsonl`，不会将部分结果标为完成。
当前不自动续跑；失败修复后使用新输出目录，保留旧目录供检查。

## 产物及检查

* `locality.png` / `locality.pdf`：四条召回率系列。
* `curves.csv`：主图每个原始点，含配置名和查询数。
* `per_query.csv` / `real_by_fanout.csv`：逐查询分片数、召回率、覆盖上界及整数分组。
* `orion/summary.json`、`kmeans/summary.json`：副本开销、分片大小、完整参数及访问数范围。
* 各方法 `oracle.npy`、`route-*.jsonl`、`lower-*.jsonl`：逐查询 Oracle、原生
  shard/entry/EF 路由及局部搜索结果；可独立复核分片数、合并去重和召回率。
* `provenance.json`、`config.json`：输入/脚本/二进制哈希、依赖版本。
  KMeans 另存训练样本 ID、质心和训练记录。
* 只有所有步骤成功后才写 `COMPLETED.json`。

轻量评分测试（只用合成小数组，不运行正式实验）：

```sh
python3 tests/tools/test_shard_locality.py
```
