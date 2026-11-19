# Orion-rust 的 100M 构建与搜索实验规范

这份文件供在新机器上工作的实验人员和 Codex 使用。适用对象是原生 Hash、KMeans、Orion 的 100M 分布式实验。具体任务新确认的要求优先，将覆盖的参数和原因写入该次运行记录。已约定的 Hash/KMeans 曲线覆盖 SIFT100M 和 DEEP100M，Text2Image100M 和 LAION100M 已约定的任务是构建三类索引。**只构建索引**、**测 Recall–QPS 曲线**、**验收 Orion 调度器**各有完成条件。

## 新机器与并行运行

GitHub 仓库默认分支是 `master`，本 Rust 项目使用 **`main`**。新机器显式选择分支，两个控制端用完整提交号核对版本。以下命令跳过与 100M 工作负载无关的 Git LFS 样例下载。

```sh
GIT_LFS_SKIP_SMUDGE=1 git clone --single-branch --branch main https://github.com/Quaternijkon/Orion.git Orion-rust
cd Orion-rust
git branch --show-current
git rev-parse HEAD
```

- 开始时在独立运行目录保存协议，记录任务范围、Git SHA、实际二进制和配置的 SHA-256、主机与分片映射、CPU 绑定、数据和索引版本。用 release 二进制做正式测量。数据、索引归档和正式窗口的原始结果在 Git 之外传递，在新机器上按清单核验。已有环境使用 `~/ann100m` 作为数据入口，新机器可映射到自己的数据盘。
- 数据目录约定 `<数据根>/<dataset>/raw/` 放原始输入、`evidence/` 放清单、`indexes/<方法>/<版本>/` 放冻结索引。已登记的 SIFT/DEEP Hash 和 KMeans 基线版本为 `m32-efc200-s8-native-20261114`，Text2Image/LAION 对应版本为 `m32-efc200-s8-native-20261119`，四库的 Orion 1/256 版本为 `m32-efc200-s8-d256-v5-20261119`。新机按需要取得对应原始数据、GT、八份分片和路由文件，核验清单后复用。SSH 身份凭据由机器环境提供。
- 两组不相交的八台物理机可以并行跑。共用测试主机时，满资源构建与正式计时窗口错开，控制端可以同时做本地准备和离线分析。每次使用独立的运行 ID、端口、暂存目录和日志，冻结输入及归档只读。
- 使用现有八机工具时，按实际环境设置 `ORION_BUILD_HOSTS` 和 `ORION_BUILD_CONTROLLER_IP`。主机列表第一台是运行控制器的本机，八个地址须对应八个不同的 boot ID。多个控制端使用同一逐机租约 `/tmp/orion-eight-host-benchmark.lock`，获得整组资源后开始占用。停止、恢复和清理按本次 PID、启动时间和路径确认归属。[资源与租约实现](tools/orion_build_admission.py)

## 索引构建要求

| 数据集 | 原始基向量 | 正式搜索距离 |
|---|---|---|
| SIFT100M | 128 维 uint8 | Euclid |
| DEEP100M | 96 维 float32 | Euclid |
| Text2Image100M | 200 维 float32 | Dot / inner product |
| LAION100M | 512 维 float32，已登记的 100M cosine 图像向量版本 | Cosine |

1. 每套索引是 **100,000,000 个逻辑点、8 台物理机、8 个逻辑数据分片、每机 1 个本地分片**，集合副本因子为 1。所有方法的下层 HNSW 用 **`m=32`、`ef_construct=200`**，核验物理图的 `m0=64`。清单分别记录原始编码和实际索引编码，连同行序、向量 ID、距离、归一化、基向量、查询及真值的身份一起核对。
2. Orion 上层导航图按 **`sample_denominator=256`（1/256）** 构建，100M 对应 **390,625 个上层点**，上层也用 `upper_m=32`、`upper_ef_construct=200`。冻结上层图，八机分担全量关联记录，完成容量划分与 owner 细化，再按最终布局物化和并行构建每个分片的下层 HNSW。Orion 可把同一逻辑点放入多个数据分片，归档分别记录逻辑点数和物理点数。[原生构建流程](tools/orion_capacity_build.md)
3. 构建阶段充分利用八机当时可用的物理核、内存和 I/O。按实际 affinity、cgroup 配额和可用内存选线程及并行进程数，记录各机实际值。全局上层图只构建一次，可分发的关联计算和下层构图由八机并行完成。**Text2Image100M 和 LAION100M 的方法顺序是 Hash → KMeans → Orion。** [并行构建手册](tools/orion_fast_build.md)
4. 复用索引时核对输入 SHA-256、距离、分片映射、布局/路由、HNSW 实际参数，Orion 还要核对上层采样数、图和策略指纹。建好后从归档在独立目录冷恢复，确认八个成员、八个本地分片各自 Active、点数与物理清单一致、下层 HNSW 已建完，Orion 路由 generation 与 artifact SHA 一致且可服务。用真实查询验证可返回 Top-10，把逐机校验和恢复结果写入 `INDEX_READY.json` 或等价记录。只构建任务到此完成。

**注意示例配置的适用范围。** `config/shard_locality_sift100m_build.json` 是早期 SIFT 分片局部性模板，其中的采样分母是 32。`tools/orion_fast_build.py` 的完整入口针对 128 维 f32le / Euclid 的 SIFT 物化输入，还串接了局部性测量。执行本次构建前核对生效的分母 256、输入编码和执行阶段，其他数据集按各自的清单生成请求。[分片局部性专项](tools/shard_locality.md)

SIFT1M 另有一项导航图规模对照。其自变量是采样分母 32、64、128、256，使用 `ef_construct=100`，按目标召回 0.95 校准，每组正式测量 3 次。该专项的并行、恢复和分析流程可以参考，100M 工作单的参数以本文件和本次任务协议为准。[SIFT1M 专项配置](config/orion_navigation_size_sift1m.json)

## 原生搜索路径与参数

| 方法 | 一条 query 的搜索路径 | 采点参数 |
|---|---|---|
| Hash | 原生 HashAll 访问全部 8 个分片，归并全局 Top-10 | 下层搜索 EF |
| KMeans | 原生 SimpleKmeans 按质心选 `nprobe` 个分片，归并全局 Top-10 | 先选 `nprobe`，固定后调下层 EF |
| Orion | 原生上层 HNSW 导航、whitebox 选片、按分片 MultiEP 入口搜索，按逻辑 ID 去重归并 Top-10 | 记录上层 `upper_k`、`upper_ef_search` 及下层 EF 规则 |

- Orion 的动态下层 EF 为 `dynamic_ef_base + dynamic_ef_factor × 该分片入口点数`。SIFT/DEEP 已验证的拟合策略对照用过 构建关联宽度 `efs=10`、上层 `upper_k=128`、`upper_ef_search=256`、下层 `32 + 4 × 入口点数`。复现该对照时沿用这些值，其他扫描显式记录所改变的参数。构建的 `ef_construct` 与搜索 EF 分开记录。
- 拟合 `prefix_k`、`max_prefix_entries`、`max_distance_gap` 时，使用该版索引构建已得到的入口排序、距离及分片归属。要求拟合独立于在线 upper/lower EF，额外向量距离计算为零，统计和 I/O 开销单独记录。按数据集与图版本冻结策略及 SHA-256，在所有节点通过 `ORION_WHITEBOX_PRUNER` 加载同一份策略。公开测试 query 和 GT 留给冻结后的验证，其他数据集重新从各自构建观测学习阈值。

## Recall–QPS 曲线

1. **逐点决定下一组参数。** Hash 直接扫 EF。KMeans 先用短校准比较，当前 `nprobe` 需要显著提高 EF 才能达到召回要求时，令 `nprobe += 1` 再评估。选定后同一条正式曲线保持该值。若改变 `nprobe`，作为另一组参数单列。
2. **在 EF 轴上等步长增加。** 每完成一个有效点就看实测 Recall、QPS 和 CPU，再决定沿用步长、调步长、补区间或结束。召回通常随 EF 增大而增长变慢，高召回段自然更密。各方法独立采点，用实际 Recall 作横轴，可直接画各自的曲线，无须命中同一组召回刻度。
3. **每个数据集和方法分别检查覆盖。** 推荐覆盖约 `0.80～1.00`，取得多组不同 EF 的合格点，其中至少一组 **Recall@10 严格大于 0.99**。若高点超过 0.995，检查是否另有一组落在 **`0.99 < Recall@10 ≤ 0.995`**，缺少时补测一个不同的 EF。用未四舍五入的数值判断。满足覆盖后即可结束高召回段。
4. **预热后只取一个正式窗口。** SIFT100M / DEEP100M 的既有对照采用每个配置预热 **15 秒**，随后正式测 **60 秒、1 次**。短校准用于选 `nprobe`、EF 步长和发压并发。失败或未达门槛的窗口保留原因，修复后只补缺失的有效点。
5. **运行实验的 agent 全程跟进。** 持续检查进程、心跳、逐点结果和待测组合，按结果调整下一点。中断先定位原因，保留已验证的点，从检查点续跑，直到所有指定的数据集、方法及高召回补点交付完成。遇到需要用户改变范围的阻塞时明确说明。

## 正式窗口与交付

- **查询与真值。** 同一数据集各方法共用同一版查询、拆分和精确 GT。SIFT/DEEP 的 10,000 条公开查询按 `np.random.default_rng(20261109).permutation(10000)` 排列，前 2,000 条校准、后 8,000 条正式测量。后 8,000 个原始 query ID 按顺序编码成小端 uint32 的 SHA-256 是 `4c7904d99b51ea7ee54eac4e0e5337fde2bdbe4bd86eab3860c954bcb25479c2`。Text2Image/LAION 的搜索按各自版本的查询与 GT 另行冻结拆分。
- **Recall@10 与 QPS。** 八机使用对齐的单调时钟窗口，统计该窗口内成功完成的单向量 HTTP 搜索。服务端完成路由、分片搜索及全局归并。集群 QPS 为八机成功完成数之和除以窗口秒数。保存每条返回的 Top-10 ID，与精确 GT 取交集后除以 10。同一 query 在时间窗内多次出现时先在该 query 内求平均，再对全部正式 query 等权平均。复现该基线时，正式窗口须覆盖全部 8,000 条测试 query，搜索错误数为零，并核对实际分片 HNSW 调用。报告端到端 P99。Oracle 分片覆盖率另列为路由诊断。
- **CPU 合格线。** 常规 Recall–QPS 正式点要求 **最忙机器的正式窗平均 CPU 利用率 > 95%**。先对每台机器分配给服务的 CPU 核计算整个窗口的平均忙时占比，再对八台取最大值。八机都要留存服务绑核 CPU、Qdrant 进程 CPU 和客户端 CPU，排查发压端或外部负载瓶颈。SIFT/DEEP 同口径历史对照每机 48 个服务物理核、16 个独立发压核，原机核号为 `48–95` 和 `0–15`。新硬件按物理核、NUMA 和网卡拓扑映射，记录差异。复现原对照另检查最忙机服务进程归因 ≥90%、发压端最高 <90%、未归因负载 ≤1.5 核。
- **结果。** 保存每点的数据集、方法、有效 EF / `nprobe` / Orion 参数、并发、预热与正式时长、请求数、错误数、Recall@10、QPS、P99 和逐机 CPU。区分尝试记录与合格点表，仅用合格点绘制每个数据集的 Recall–QPS 曲线。提交汇总时列出索引/策略版本、原始请求记录位置、补点情况和任何缺项。

## Orion 在途窗口调度验收

此次交接的已验证调度源码提交为 `c418ad050bdf58218522e0ea5075a93b73595e6b`。用 `cargo build --release --locked -p qdrant --bin qdrant` 编译服务，以配套的 `examples/orion_direct_search.cpp` 为直达发送端。发送前按余量选择 B，通过 `/collections/{name}/orion/coordinator/permits` 取名额、通过 `/collections/{name}/orion/coordinator/search` 发查询。分片结果回 B 归并。各发送端共享全局在途窗口，已接纳的查询占用名额直到归并完成。端到端延迟包括发送端准入等待至收到最终结果。原入口 `/collections/{name}/points/search` 可作单列对照。[发送端构建与接口](tools/orion_navigation_balance.md)

默认 `storage.performance.orion_navigation.target_utilization=0.95`，控制周期 250 ms。最忙节点的 worker 或上层 CPU 队列跨相邻采样持续积压时优先减窗。有查询等候、窗口已用满且最忙节点低于目标时增窗。达到目标时保持总窗，把新协调任务分给有余量的节点。热点机主要只做 worker 仍达目标时保持窗口。QPS 和 P99 用于验收。记录全局/逐机额度和在途数（`cluster_window`、`node_window`、`inflight`）、worker/coordinator CPU、排队、QPS 和端到端 P99。

**当前交接状态（2026-11-19）。** 该实现已完成本地功能验证。用户将另行给出八机正式性能验收要求，收到后按其负载、对照和通过条件执行。调度器的 95% 是运行目标，上节的严格 `>95%` 是既有曲线的窗口合格线，两者在运行协议中分别填写。
