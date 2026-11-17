# Orion 8 机加速构建与召回率实验手册

推荐入口是 `tools/orion_fast_build.py`。默认只打印计划；加 `--execute` 才启动。
不要再将低层 `prepare-capacity` 的完整串行 pass 当作大数据构建的默认方式。
Python 容量构建器现已拒绝 **超过 100 万条、未提供复用数据、未显式允许串行**
的调用。分布式阶段失败时，快速入口停止本次任务，不静默退回串行。

本版本的完整自动入口限定 SIFT100M / 128 维 f32le / Euclid /
8 机 8 分片 / M=32 / ef_construct=200。它会拒绝不兼容的输入配置。
其他数据集可参考各阶段工具，但不能直接套用此入口中的行数、维度和路径。

## 固定的执行计划

| 阶段 | 默认执行方式 | 必须保持的约束 |
|---|---|---|
| 上层 HNSW 构建 | 原生确定性构建一次并冻结 | M、ef_construct、采样种子不变 |
| 全量关联记录 | 8 机，每机使用全部可用物理核；按核数比例分配行范围 | 同一图、efs=10、保持行序与命中顺序 |
| 全局分片与访问平衡 | 原来的容量规划 / METIS 流程 | 不替换分片算法，不消耗测试查询或 GT |
| 物化分片向量和 ID | 控制机按可用物理核与内存决定进程数，按源 ID 顺序合并 | 无缺行、重行、漏副本 |
| 下层 HNSW | 8 机并发，每机使用全部可用物理核，跨 NUMA 节点 | M=32、ef_construct=200 |
| 归档与校验 | 各机器并发归档，独立文件并行计算 SHA256 | 源文件和归档内容 SHA256 一致 |
| 局部性测量 | 原生路由 + 实际 GT memberships 覆盖统计 | 精确分片内假设；Oracle/Real 使用同一 query 子集 |

这里加速的是**做完实验、得到数据的过程**。搜索算法、搜索并发和资源配置、
搜索预算、查询子集及评分口径继续使用实验配置，不由构建资源策略调整。

过去的“8 个线程”只是 8 个工作线程，并不代表绑定了 8 个物理核。新实现读取
每机 CPU 拓扑、进程亲和性和 cgroup 配额，每个物理核选择一个允许使用的逻辑
CPU，并把构建进程限制在这些 CPU 上；不再限定 CPU0..47 或一个 NUMA 节点。
本次只读探测的 8 台机器均为 96 个可用物理核、192 个逻辑 CPU；在资源状态
不变时，关联记录和下层构图各为每机 96 线程，控制机物化为 96 进程。
实际数值在取得实验锁后解析；物化进程数还受可用内存约束。

控制机同时验证多个归档时共享其 CPU 预算，避免每个归档各启动一个满核线程池。
线程数 1 的原生一致性基线、确定性的上层构图、METIS 和访问概率细化中有依赖的
步骤保留原顺序。传输、文件拼接受网络/磁盘带宽限制，线程增加不保证线性提速。
NumPy/BLAS 内部维持单线程以避免物化进程再嵌套线程池。

提高构建线程数可能改变 HNSW 的并发插入次序，因此不承诺新图与旧图逐位相同。
结果必须记录每台机器实际构建线程数。历史 Orion 构图为 48，复用 KMeans 控制
索引的历史值为 8；新构图按实测资源记录，不能沿用 48 的标签。
两者 M=32、m0=64、ef_construct=200 一致。正式搜索预算不随构建线程数调整。
主指标已改为 GT 覆盖率，不包含局部 ANN 漏检；默认不运行下层 ANN 搜索。
已有完整实验可用 `shard_locality_gt.py` 重统计，无需重新构建或搜索，详见
[指标定义](shard_locality.md)。

## 第一次运行前

需要：Linux `/proc`、8 台机器的免交互 SSH、rsync、taskset，以及同架构原生程序。
控制机 Python 需要 NumPy、PyMetis、Matplotlib；单机诊断的重训功能才需要 sklearn。
主机列表第一项必须是当前控制机；执行时核验 8 个不同机器实例及每机允许使用
的物理 CPU。控制机需有足够 NVMe 空间存放工作文件和归档；下层构建脚本要求每台机器的
`/dev/shm` 至少剩余 60 GiB。原始向量、GT、历史索引不应被改写。

从同一个已固定的代码版本构建三个程序，避免把不同版本的路由和索引器混用。
本次修改后必须重编译 `shard_locality`：旧程序仅接受 8 或 48 个构图线程。
启动器在计划模式就检查程序能力，旧二进制会被立即拒绝，避免构建中途才失败。
下面 `-j` 使用当前进程的物理核/配额预算；编译也应安排在不干扰测量的时段：

```sh
export CARGO_TARGET_DIR=/absolute/path/build-cache
cargo build --release --locked -p collection \
  --bin orion_build_layout \
  --example orion_attachment_parallel \
  --example shard_locality \
  -j "$(python3 -c 'from tools.orion_build_resources import detect; print(detect()["worker_count"])')"
```

核验 `config/shard_locality_sift100m_build.json` 的原始向量路径及 SHA256，以及
`config/shard_locality_sift100m.json` 的查询 NPZ、8 台主机和唯一 Orion 搜索配置。
启动器会从唯一搜索配置生成 whitebox 文件，不需要复制旧实验目录中的配置。

```sh
python3 tools/orion_fast_build.py \
  --root /absolute/path/new-unique-run \
  --build-request config/shard_locality_sift100m_build.json \
  --experiment-config config/shard_locality_sift100m.json \
  --layout-binary "$CARGO_TARGET_DIR/release/orion_build_layout" \
  --attachment-binary "$CARGO_TARGET_DIR/release/examples/orion_attachment_parallel" \
  --search-binary "$CARGO_TARGET_DIR/release/examples/shard_locality" \
  --kmeans-archives /absolute/path/kmeans-M32-efc200-8shards/archives
```

上面只检查参数、依赖和小型归档元数据并打印计划，不创建任务目录、不搜索数据。
确认计划后，对同一命令加 `--execute`。目录必须是新的，名称也应唯一，避免
与远程临时目录重名。若依赖安装在独立目录，增加 `--python-deps /path/to/deps`；
需要指定管理网卡源地址时增加 `--controller-ip <控制机管理IP>`。

## 为什么新任务不会再次跑完整串行 pass

1. 启动器将代码、配置和二进制复制到本次任务目录，记录 SHA256；运行中的快照
   不会被仓库里同时进行的开发改动影响。
2. 准备控制器等到 8 台机器空闲并持有全部共享实验锁，构建并冻结上层图。
3. 自动监视器发现关联文件已有至少 8192 条完整记录后，抽取样本。
4. 新原生导出器分别以 1 线程和控制机实际工作线程数运行样本，与原生串行输出逐字节比较。
5. 只暂停本任务的串行关联生产者，保留所有完整前缀记录，将余下范围按可用核数
   分到 8 台机器。每台机器使用自己的实际线程数重放同一份样本，独立检查图、
   向量分段、输出 SHA256。
6. 按原始行序合并，检查行数和字节数，再由 Rust `read_prepared` 验证完整输入绑定。
7. 验证成功后才替换旧生产者，使用 `--reuse-prepared` 续接同一个容量构建算法。
   后续自动资源物化、构图、归档和两组召回率测量自动执行。搜索资源不随之调整。

短串行前缀只提供原生一致性基线，并被保留复用；它不会跑完整个 100M 数据集。
自动并行失败会写明原因并停止本任务。旧的手动迁移工具可以恢复串行生产者用于
保全进度，但快速启动器会在其失败后设置 `STOP`，不把这种回退当作正常生产路径。

## 优先复用已验证的中间结果

复用就是继续使用前一次已经计算并校验的文件，跳过重复步骤。例如，上层图与
1 亿条向量的关联记录已经生成时，不必再搜索这 1 亿条向量。

| 已有结果 | 可省掉的工作 | 入口与条件 |
|---|---|---|
| 完整 prepared 数据 | 上层图构建、向量预处理和关联记录生成 | `--reuse-prepared`；当前实现要求全部构建 options 和原始输入 SHA256 一致，并重验文件哈希 |
| 匹配的完整索引归档 | 布局和下层索引重建 | `shard_locality_archives.py`；核验参数、输入身份和索引内容，再按实验配置测量 |
| 完整路由及 GT 分片归属 | 重建和重新搜索 | `shard_locality_gt.py`；在同一批保存的数据上重新统计覆盖率和图表 |

如果改了会影响这些结果的输入或参数，就需要重算对应阶段。复用不允许把旧参数
结果当作新参数结果；当前 `--reuse-prepared` 甚至不接受仅部分 options 相同的请求。

当完整构建请求和原始输入 SHA256 与已有准备数据完全一致时，对启动命令增加：

```sh
--reuse-prepared /absolute/path/previous-run/prepared-parallel
```

这样跳过上层图及关联记录生成。入口检查 `capacity-input.json` 的全部 options、
输入身份和 `validated.json`；后续原生构建器仍会重新核验文件哈希。仅通过一个
只读目录引用复用数据，不修改它。不能把 ef_construct=100 的图当作 200 的图复用。

若只需重新测量已有的两套匹配索引，直接使用 `shard_locality_archives.py`，不必
再启动构建。具体指标和搜索命令见 [shard_locality.md](shard_locality.md)。

确实需要串行的小规模调试可使用原容量构建器；超过 100 万条必须显式增加
`--allow-serial-attachments`。这不是生产推荐命令。HDF5 适配器调用同一构建 API，
也受大数据串行保护约束。

## 观察、停止与恢复

| 文件 | 含义 |
|---|---|
| `status.json` | 上层图 / 容量布局准备状态 |
| `auto-attachments.json` | 自动并行切换进度或失败原因 |
| `parallel-attachments/status.json` | 8 机生成、合并、原生校验进度 |
| `continuation.json` | 分片物化、8 机构图、搜索或最终完成 |
| `logs/lower-<shard>.log` | 导入行数及 HNSW `main_graph.done/total` |
| `evidence/frozen-inputs.json` | 本次源代码、配置、二进制身份 |
| `evidence/build-resources-prepare.json` | 各机物理核、配额、NUMA、关联线程和物化预算 |
| `evidence/build-resources-lower.json` | 下层构图前重新检测的资源及各机实际线程数 |
| `evidence/matched-index-preflight.json` | 两种方法真实物理索引参数 |
| `results/COMPLETED.json` | 全部测量成功后才出现的完成标记 |

日志可能包含一整行、数十 MB 的构建 manifest。不要直接 `tail -n` 或打印整个
`build-manifest.json`；读取末尾固定字节数，或只输出需要的 JSON 字段。

停止本次任务：在本次根目录创建 `STOP`。控制器只终止核对过 PID、启动时间和
工作路径的自有进程，并释放实验锁。不要使用 `killall`、删除其他任务的锁，或
清理整个 `/dev/shm`。停止/失败不等于实验完成。

恢复先检查状态和日志。优先复用已经校验的 `prepared-parallel` 或索引归档，在
新目录重启；保留失败目录作为证据。当前入口不宣称对所有中断点都能自动续跑。
不要手工伪造完成标记，也不要仅修改 metadata 将旧图标成新参数。

## 历史运行数据与新资源策略的验证范围

以下数字来自原来每机 8 线程的运行，不是新自动资源配置的性能测量。
历史 1 亿条关联记录已经通过原生全量校验和 8 机样本逐字节校验。保留前缀
21,695,212 条，剩余 78,304,788 条的分发、生成、合并约 240 秒；包含全量校验
和续接约 426 秒。相对于此前单线程实测速度外推，分别约 24.8× / 13.9×。
基准由 1 个工作线程变为 64 个工作线程，不是“8 倍资源得到 24.8 倍”的超线性
结论，也不是严格相同行区间的串行重跑或 ANN QPS 测量。

本轮完整结果与证据位于：
`/mnt/rocksdb_nvme/shard-locality-matched8-20261117/`。
原启动器的各阶段在该运行分别执行过。新的自动资源选择、内存/配额限制、
并行校验、样本检查线程配置和分片输出经过小数据测试，并对 8 台机器做了
只读资源探测；尚未用新默认值重跑完整 100M 实验，不能据此承诺新的加速倍数。

```sh
python3 tests/tools/test_orion_fast_build.py
python3 tests/tools/test_orion_build_resources.py
python3 tests/tools/test_shard_locality.py
python3 tests/tools/test_orion_capacity_partition.py
```
