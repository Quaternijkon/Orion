# Orion 负载均衡替换设计约束

## 原始设计直觉

更接近原生 HNSW 的构建方式是：先取约 `1/M` 的点构造上层导航图
（这里 `M=32` 指 HNSW 非底层最大边数，不代表物理机器数），先将 L1
导航点划分为固定数量的逻辑分片；随后构建 L0 时，每个点依据其在上层
导航图中的归属进入对应分片，并在这一步应用 Orion 原有的多分配策略。

负载均衡只负责决定哪些相邻 L1 导航点放入同一分类，使后续自然产生的
L0 分配更均衡。它不得通过全量模拟实际 L0 结果，再反过来调整 L1 分类。
L1 负载均衡与 L0 多分配属于两个正交阶段。

## 本次替换的冻结语义

1. 当前 Orion 的 L0-informed weighting、topology refinement 和
   P24-to-P32 fission 负载均衡全部移出新设计；不得与新候选叠加。
2. `N_native` 是无负载均衡的 Orion 基线：固定 `P=32`，仅对冻结的 L1
   向量执行原生 KMeans 分类；无权重、容量、配额、修补、fission 或 L0
   反馈。
3. `C_CNBR` 是唯一替换候选，只能读取冻结的 upper/L1 图、L1 向量和
   upper self-navigation 统计，并从 `N_native` owner 做轻量的 L1 边界修补。
   Phase-A 候选生成器唯一允许读取的 artifact 必须是 neutral upper-only
   projection：它逐字节保留 upper 图、L1 label 顺序和向量位模式，但移除
   历史 `shard_membership`、layout 及旧均衡参数。原历史 production artifact
   只能由独立 projection 步骤读取，不能直接进入 `N_native`/`C_CNBR`。
4. 在 L1 owner 校验和冻结之前，候选生成器不得读取或接收完整 L0
   attachments、实际分片负载、L0 vote stream、查询、ground truth，或任何
   可间接暴露这些数据的回调；不得执行额外的 `O(N)` L0 pass、后验修补、
   load-guided retry 或 fission。
5. owner 冻结并退出候选生成进程后，独立构建/评估过程才可执行一次 Orion
   原本就需要的 L0 attachment。`N_native` 与 `C_CNBR` 必须使用完全相同的
   多分配代码、参数与逐点复制语义；负载均衡不得取消、缩减或重调
   multi-assignment。
6. 历史方案 `H` 只作为部署性能锚点，不是候选、拓扑 veto 或 fallback。
   `C_CNBR` 任一硬门禁失败时回退到 `N_native`。

## 替换实施边界

“去除当前负载均衡”首先是算法与实验输入上的彻底隔离：`N_native` 和
`C_CNBR` 的构建、筛选、物化及在线主比较均不得调用旧 weighting、topology
refinement、fission 或 L0 capacity repair。旧实现源码在最终 A/B 决策前仅为
复现历史 `H` 和安全回滚而保留，不能作为隐藏默认、候选或失败回退。

全部门禁通过后，生产通用入口应把胜者设为新的 canonical 路径，并把旧路径
降为必须显式选择的 `legacy`；待历史复现和回滚窗口结束后，再决定是否删除旧
源码。若 `C_CNBR` 未通过，则 canonical 路径是无负载均衡的 `N_native`，而不
是恢复旧 `H`。多分配不属于被移除的旧负载均衡，始终保留在 owner 冻结后的
正常 L0 attachment 阶段。

## 采用门禁

- 拓扑优先：upper 图的边切割、保留度、孤立点和分片内连通性任一门禁失败，
  即使负载更均衡也不得采用。
- 轻量构建：除共享的 `N_native` KMeans 外，upper self-navigation、为验证
  upper-only 输入而必须执行的校验与 proxy-mass replay、以及 CNBR 修补三项
  wall time 之和，不得超过同一构建正常 full-attachment wall time 的 `5%`，
  且不得新增全量 L0 pass。正式公式为：

  ```text
  (external self-navigation
   + required input validation and proxy-mass replay
   + formal CNBR)
  / normal full attachment <= 0.05
  ```

  这里的“required input validation”从共享 upper 状态已经由正常上层图构建过程
  产生并可在内存中交给 `N_native`/`C_CNBR` 之后开始；计入 projected-artifact
  checksum、neutral projection provenance/redaction、upper labels/vectors 逐字节
  一致性、self-navigation 行校验以及 proxy-mass replay。共享的 upper artifact
  反序列化与图 normalization、`N_native` KMeans、离线拓扑验收和证据写出分别
  报告，但不伪装成负载均衡算法的增量成本。neutral projection 只用于本次
  隔离历史旧均衡的实验链；正式替换后的生产构建应直接把刚构造好的 upper/L1
  状态交给新分类器，不应再执行“从历史 Orion artifact 投影”的步骤。
- 最终性能：在相同四台物理主机、32 个逻辑分片、相同 upper graph、HNSW
  参数、资源和查询集下，以 `Recall@10 >= 0.90` 的 matched-recall QPS 为最终
  采用依据。`C_CNBR` 必须先通过所有离线、拓扑和构建门禁，再证明相对
  `N_native` 的 QPS 改善；历史 `H` 只单独报告性能差距。

开发阶段可以在 owner 冻结后，用 SIFT/GloVe 的真实 L0 构建结果评估并一次性
冻结候选参数；这只是算法研究与验证。正式 Orion 构建路径不得按每个新数据集
的 L0 实际结果重新调参或重分 L1。
