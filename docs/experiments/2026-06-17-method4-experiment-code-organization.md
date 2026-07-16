# Method4 实验代码目录组织说明

本文档说明当前 Method4 distributed Qdrant 实验相关代码如何组织。目标是方便快速定位：

- 去哪里跑实验；
- 去哪里改实验参数；
- 去哪里看 Qdrant server-side 优化实现；
- 去哪里看测试；
- 去哪里找实验结果和实验记录。

## 总体结构

当前实验代码分为 6 层：

| 层级 | 目录 / 文件 | 作用 |
|---|---|---|
| 实验驱动层 | `tools/*.py` | Python 实验 harness，负责构建 collection、生成 routing plan、跑 benchmark、写结果 |
| Docker 拓扑层 | `tools/compose/*.yaml` | controller + worker 的 Qdrant 集群部署 |
| Qdrant 查询调度层 | `src/common/query.rs` | controller 侧 shard-major execution、peer pre-merge fallback、最终 merge |
| Qdrant collection 搜索层 | `lib/collection/src/collection/search.rs` | per-shard specialization、worker-local peer pre-merge、remote shard RPC |
| gRPC / API 适配层 | `src/tonic/api/*.rs`, `lib/collection/src/shards/*.rs` | internal RPC 请求/响应转换和 remote shard 调用 |
| 测试与结果层 | `tests/tools`, `results`, `docs/experiments` | Python/Rust 测试、实验输出、实验文档 |

数据流可以概括为：

```text
HDF5 dataset
  -> tools/qdrant_two_level_routing_experiment.py
  -> Qdrant controller cluster
  -> src/common/query.rs
  -> lib/collection/src/collection/search.rs
  -> lower logical shard HNSW search
  -> results/<experiment>/<timestamp>/
  -> docs/experiments/*.md
```

## 1. 实验驱动层: `tools/`

### 主入口: `tools/qdrant_two_level_routing_experiment.py`

这是当前 Method4 / naive / placement / high-recall 实验的主 harness。

它负责：

- 读取 HDF5 数据集；
- 构建或复用 Qdrant collection；
- 按不同 `routing-mode` 生成 lower shard assignment；
- 执行 Method4 upper routing；
- 生成 `shard -> entry points`；
- 生成 per-shard dynamic EF；
- 发送 compact multi-EP / per-shard / naive search request；
- 做 tuning、正式评估、stability repeats；
- 输出 `summary.json`、`routing_tuning.csv`、`final_metrics.csv` 等结果文件。

常用参数分组：

| 参数组 | 参数 | 作用 |
|---|---|---|
| 数据与 collection | `--base-url`, `--collection`, `--hdf5-path`, `--reuse-existing`, `--recover-routing-from-collection` | 指定 Qdrant 和数据来源 |
| 构建参数 | `--hnsw-m`, `--ef-construct`, `--upper-m`, `--upper-ef-construction` | HNSW 构建参数 |
| Method4 构建 | `--sample-denominator`, `--k-overlap`, `--topology-iters`, `--disable-multi-assign`, `--disable-fission` | 控制原初 Method4 method=4 语义路径 |
| 搜索参数 | `--upper-k-candidates`, `--base-ef-candidates`, `--factor-candidates`, `--target-recall` | Method4 upper routing 和 dynamic EF 参数扫描 |
| 执行方式 | `--routed-execution-mode`, `--routed-planning-mode`, `--lower-execution-order`, `--search-dispatch-mode` | 控制 request 形态和分布式执行形态 |
| 物理放置 | `--shard-placement`, `--shard-placement-map`, `--shard-placement-map-name`, `--placement-peer-uri-contains` | 控制 logical shard 到 physical peer 的 placement |
| 分析输出 | `--placement-simulation`, `--physical-execution-trace`, `--output-dir` | 生成 placement / physical trace / benchmark 结果 |

最重要的 routing modes：

| `--routing-mode` | 含义 |
|---|---|
| `faithful_original_rest` | 当前 Method4 idea 主路径，尽量对齐原初 C++ method=4 的外部调度语义 |
| `naive_hash_all_shards` | naive all-shards baseline |
| `legacy_centroid` | 早期 centroid routing 逻辑，通常不作为当前主结果 |

最重要的 routed execution modes：

| `--routed-execution-mode` | 含义 |
|---|---|
| `compact_multi_ep` | 当前主路径；一个 query 携带 per-shard entry points 和 per-shard EF map |
| `per_shard_multi_ep` | 每个 routed shard 一个 search object，保留多入口语义 |
| `grouped_by_ef` | 按相同 EF group search，早期路径 |
| `compact_query_ef` | 一个 query 只有一个 compact EF，会丢失 per-shard EF 细节，主要用于对照 |

最重要的 planning modes：

| `--routed-planning-mode` | 含义 |
|---|---|
| `per_batch` | 每个 batch 内临时构造 route plan |
| `materialized` | 对 evaluation set 预先 materialize route plan |
| `compact_materialized` | 使用 compact routing manifest，减少 Python route plan / JSON hot path 开销 |

典型 Method4 高召回命令形态：

```bash
env PYTEST_ADDOPTS= LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
python3 tools/qdrant_two_level_routing_experiment.py \
  --base-url http://localhost:6833 \
  --collection qdrant_controller_idea_method4map_full_20260601 \
  --hdf5-path /home/taig/dry/faiss/datasets/glove-200-angular.hdf5 \
  --routing-mode faithful_original_rest \
  --num-shards 31 \
  --upper-k-candidates 160 \
  --base-ef-candidates 80 \
  --factor-candidates 8 \
  --target-recall 0.955 \
  --eval-query-count 3000 \
  --tuning-query-count 500 \
  --stability-repeats 2 \
  --batch-size 200 \
  --reuse-existing \
  --recover-routing-from-collection \
  --routed-execution-mode compact_multi_ep \
  --routed-planning-mode materialized \
  --routed-result-limit-mode top_k \
  --search-dispatch-mode coordinator \
  --shard-placement map \
  --shard-placement-map results/qdrant_goal_recall_idea_095_placement_simulation/20260601_075041/placement_simulation.json \
  --shard-placement-map-name method4_aware \
  --placement-peer-uri-contains qdrant_shard_ \
  --output-dir results/<experiment_name>
```

### 其他实验脚本

| 文件 | 作用 |
|---|---|
| `tools/qdrant_batch_search_benchmark.py` | Qdrant batch search benchmark，主要用于较朴素的 batch/search 对照 |
| `tools/qdrant_partition_scheme_comparison.py` | partition scheme 相关早期对比 |
| `tools/qdrant_custom_shard_adaptive_ef_experiment.py` | custom shard + adaptive EF 早期实验 |
| `tools/qdrant_custom_shard_nprobe_experiment.py` | custom shard nprobe 实验 |
| `tools/qdrant_centroid_nprobe_experiment.py` | centroid/nprobe 早期实验 |
| `tools/hnsw_experiment.py`, `tools/hnsw_stress_test.py` | HNSW 单机/压力实验，不是当前 Method4 distributed 主路径 |

当前投稿主线优先使用 `tools/qdrant_two_level_routing_experiment.py`。其他脚本主要用于历史对照或补充实验。

### Benchmark 矩阵入口: `tools/method4_benchmark_matrix.py`

这个脚本是 `qdrant_two_level_routing_experiment.py` 的上层 runner，用于把
Orion / naive / KMeans 的多组参数实验组织成一个矩阵。

它负责：

- 从 JSON 配置读取 defaults、cases 和 matrix；
- 展开不同 recall target、shard 数、batch size 等实验组合；
- 为 `orion`、`naive`、`kmeans` preset 自动补齐对应的 routing 参数；
- 生成每个 run 的完整命令；
- 默认只 dry-run，写出 `commands.sh` 和 `run_manifest.json`；
- 显式传 `--run` 后逐条执行命令；
- 从每个 run 的 `summary.json` 汇总出 `matrix_summary.csv`。

当前示例配置：

```text
tools/benchmark_configs/method4_orion_naive_kmeans_matrix.example.json
```

常用 dry-run：

```bash
python3 tools/method4_benchmark_matrix.py \
  --config tools/benchmark_configs/method4_orion_naive_kmeans_matrix.example.json \
  --run-id smoke
```

实际执行：

```bash
python3 tools/method4_benchmark_matrix.py \
  --config tools/benchmark_configs/method4_orion_naive_kmeans_matrix.example.json \
  --run-id full_001 \
  --run
```

输出目录形态：

```text
results/method4_benchmark_matrix/<matrix_name>/<run_id>/
  commands.sh
  run_manifest.json
  matrix_summary.csv
  orion__target_recall-0p95__batch_size-200/
    <timestamp>/
      summary.json
      routing_tuning.csv
      final_metrics.csv
      stability_runs.csv
```

当前 preset 语义：

| Preset | 对应设置 | 用途 |
|---|---|---|
| `orion` | `faithful_original_rest` + `compact_multi_ep` + `materialized` | 当前完整 Method4 / Orion 主路径 |
| `naive` | `naive_hash_all_shards` | naive all-shards baseline |
| `kmeans` | `faithful_original_rest` + `topology_iters=0` + `disable_fission` | Method4 框架下的 KMeans-only 分片 ablation |
| `kmeans_independent` | `cpp_kmeans_baseline` + `compact_multi_ep` + `materialized` | 独立 KMeans 分片 baseline，保留旧 `cpp_kmeans_baseline` 路径 |
| `legacy_centroid` | `legacy_centroid` | 早期 centroid routing，对照或历史实验用 |

注意：`kmeans` preset 不是旧 `legacy_centroid`。它保留 Method4 upper routing /
voting assignment / compact multi-EP 框架，只让 L1 shard map 停留在 balanced
KMeans 初始化结果，用于更公平地比较 Orion topology-aware refinement 的贡献。
如果需要复现独立 KMeans 分片 baseline，应使用 `kmeans_independent` preset。

## 2. Docker 拓扑层: `tools/compose/`

| 文件 | 作用 |
|---|---|
| `tools/compose/docker-compose.controller-cluster.yaml` | 当前 controller + 3 worker 主实验拓扑 |
| `tools/compose/docker-compose.idea-cluster.yaml` | idea cluster 早期/备用 compose |
| `tools/compose/docker-compose.naive-cluster.yaml` | naive cluster 早期/备用 compose |
| `tools/compose/docker-compose.yaml` | 通用 compose |

当前主线拓扑使用：

```text
tools/compose/docker-compose.controller-cluster.yaml
```

默认端口：

| 角色 | 容器 | HTTP | gRPC |
|---|---|---:|---:|
| Controller | `qdrant_controller` | 6833 | 6834 |
| Worker 1 | `qdrant_shard_1` | 6843 | 6844 |
| Worker 2 | `qdrant_shard_2` | 6853 | 6854 |
| Worker 3 | `qdrant_shard_3` | 6863 | 6864 |

常用环境变量：

| 变量 | 作用 |
|---|---|
| `QDRANT_CONTROLLER_IMAGE` | controller 使用的 Qdrant 镜像 |
| `QDRANT_SHARD_IMAGE` | worker 使用的 Qdrant 镜像 |
| `QDRANT_CONTROLLER_CPUSET` | controller CPU 绑定 |
| `QDRANT_SHARD_1_CPUSET` / `QDRANT_SHARD_2_CPUSET` / `QDRANT_SHARD_3_CPUSET` | worker CPU 绑定 |
| `QDRANT_DISABLE_SHARD_MAJOR_PEER_PREMERGE=1` | 关闭 worker-local peer pre-merge，用于 A/B |

## 3. Qdrant 查询调度层: `src/common/query.rs`

该文件是 controller 侧 Method4 server-native execution 的关键入口。

核心职责：

- 判断 batch search 是否应该走 shard-major path；
- 将 compact multi-EP request 按 logical shard 展开；
- 给每个 lower shard 写入自己的 `hnsw_entry_points` 和 `hnsw_ef`；
- 优先尝试 worker-local peer pre-merge；
- fallback 到旧的 logical-shard fan-out/fan-in；
- 做 source-id dedup aware merge；
- 保持 final global top-k merge 语义。

重点函数：

| 函数 | 作用 |
|---|---|
| `should_use_shard_major_search_batch` | 判断是否有 per-shard maps，需要走 shard-major |
| `specialize_core_search_for_shard` | 从 `hnsw_entry_points_by_shard` / `hnsw_ef_by_shard` 中取出当前 shard 的参数 |
| `merge_shard_major_candidates` | source-id dedup aware 的 final merge |
| `premerge_shard_major_candidates_by_peer` | peer-local partial merge 的语义辅助 |
| `do_search_batch_points_shard_major` | controller 侧 shard-major lower execution |
| `do_search_batch_points` | 普通 batch search 入口，会分流到 shard-major path |

注意：这里的优化不应该改变 Method4 的外部语义。它只改变分布式执行形态。

## 4. Qdrant collection 搜索层: `lib/collection/src/collection/search.rs`

该文件负责 collection 层的 per-shard specialization 和 worker-local peer pre-merge。

核心职责：

- 接收 controller 的 compact request；
- 对每个 logical shard 生成已经特化的 `CoreSearchRequest`；
- 对 remote worker 发送 `CoreSearchBatchByShardInternal`；
- worker 本地按 shard 执行 HNSW search；
- worker 本地把多个 logical shard result streams 预合并成 peer-level partial result；
- controller 再做最终 global merge。

重点函数：

| 函数 | 作用 |
|---|---|
| `specialize_search_batch_for_shard` | collection batch 层按 shard 注入 entry points 和 EF |
| `specialize_core_search_for_shard_major` | shard-major path 中单个 request 的 per-shard specialization |
| `core_search_batch_shard_major_peer_premerge` | controller 调用 remote workers 的 peer pre-merge path |
| `search_dedup_point_id` | 将 copied point id 映射回 source-id dedup key |

## 5. gRPC / remote shard 适配层

这些文件负责 internal RPC 和 remote shard 调用。

| 文件 | 作用 |
|---|---|
| `src/tonic/api/points_internal_api.rs` | internal gRPC API 入口，包含 `CoreSearchBatchByShardInternal` handler |
| `src/tonic/api/query_common.rs` | gRPC query conversion、worker-local by-shard search、merge 逻辑 |
| `lib/collection/src/shards/remote_shard.rs` | controller 调用 remote shard / worker 的客户端侧封装 |
| `lib/collection/src/shards/local_shard/query.rs` | local shard query 入口 |
| `lib/collection/src/collection_manager/segments_searcher.rs` | lower segment/HNSW search 入口，接收 `hnsw_entry_points` 和 `hnsw_ef` |

如果要审计“是否仍然 per logical shard search”，重点看：

```text
src/common/query.rs
lib/collection/src/collection/search.rs
src/tonic/api/query_common.rs
lib/collection/src/collection_manager/segments_searcher.rs
```

## 6. 测试目录

### Python harness 测试

主测试文件：

```text
tests/tools/test_qdrant_two_level_routing_experiment.py
```

覆盖内容包括：

- argument parsing；
- compact multi-EP request shape；
- shard-major expansion；
- materialized routing plan；
- placement map；
- source-id dedup block；
- physical execution trace / shard costs 等 harness 逻辑。

常用测试命令：

```bash
env PYTEST_ADDOPTS= pytest -o addopts= tests/tools/test_qdrant_two_level_routing_experiment.py -q
```

### Rust 侧测试

相关测试分散在代码文件内：

| 位置 | 关注点 |
|---|---|
| `src/common/query.rs` 内部 tests | shard-major specialization、merge、source-id dedup、peer pre-merge 语义 |
| `lib/collection/src/collection/search.rs` 内部 tests | per-shard entry points / EF specialization |
| `src/tonic/api/query_common.rs` 内部 tests | internal RPC by-shard merge 行为 |

常用测试命令：

```bash
cargo test -p qdrant shard_major_
cargo test -p qdrant core_search_by_shard_premerge_keeps_limit_plus_offset_per_peer_query
cargo test -p qdrant shard_major_peer_local_premerge_preserves_source_id_dedup
cargo test -p collection specialize_search_batch
```

## 7. 结果目录: `results/`

实验脚本统一把结果写到：

```text
results/<experiment_name>/<timestamp>/
```

典型文件：

| 文件 | 内容 |
|---|---|
| `summary.json` | 最完整的实验配置和汇总指标 |
| `routing_tuning.csv` | 参数扫描 / tuning 结果 |
| `final_metrics.csv` | 最终 eval result |
| `final_per_query_metrics.csv` | 开启 `--write-per-query-metrics` 后输出的每个 query hits@k / recall@k / retrieved ids / ground truth ids |
| `stability_runs.csv` | stability repeats |
| `builds.csv` | collection build / upload 统计 |
| `upper_sample_stats.csv` | upper sample / upper routing 统计 |
| `placement_simulation.json` | placement simulation 输出 |

当前重要结果目录示例：

| 目录 | 说明 |
|---|---|
| `results/qdrant_compare_current_idea_vs_naive_095_idea_b200/20260603_094930` | 当前 Method4 same-recall 主结果 |
| `results/qdrant_compare_current_idea_vs_naive_095_naive_ef76_b100/20260603_095227` | naive closest-recall 稳定对照 |
| `results/qdrant_compare_current_idea_vs_naive_095_naive_ef80_b100/20260603_095342` | naive higher-recall 对照 |
| `results/qdrant_goal_recall_idea_095_server_peer_premerge_fresh/20260603_092757` | worker-local peer pre-merge enabled A/B |
| `results/qdrant_goal_recall_idea_095_server_peer_premerge_disabled_fresh/20260603_092949` | worker-local peer pre-merge disabled A/B |
| `results/qdrant_goal_recall_idea_095_placement_simulation/20260601_075041` | method4-aware placement simulation |

不要手工修改 `results/` 中的原始结果。论文或文档中的二次整理应写入 `docs/experiments/`。

## 8. 实验文档目录: `docs/experiments/`

当前 Method4 投稿相关文档：

| 文件 | 作用 |
|---|---|
| `2026-06-04-current-method4-docker-distributed-design.md` | 当前 Docker 分布式架构说明 |
| `2026-06-03-current-method4-vs-naive-same-recall.md` | 当前 Method4 vs naive 同召回对比 |
| `2026-06-03-method4-worker-local-peer-premerge.md` | worker-local peer pre-merge 实现与 A/B |
| `2026-06-01-method4-aware-placement-simulation.md` | placement simulation |
| `2026-06-01-method4-aware-placement-deployment.md` | placement deployment |
| `2026-06-16-method4-distributed-submission-experiment-plan.md` | 长版投稿实验计划 |
| `2026-06-16-method4-submission-experiment-workboard.md` | 短版实验执行对照表 |
| `2026-06-17-method4-experiment-code-organization.md` | 本文档，说明实验代码目录组织 |

建议规则：

- 原始结果放 `results/`；
- 实验结论、解释和决策放 `docs/experiments/`；
- 新的投稿 claim / experiment checklist 优先更新 `2026-06-16-method4-submission-experiment-workboard.md`；
- 代码目录或运行入口变更时更新本文档。

## 9. 新增实验时应该放哪里

| 想做的事 | 推荐位置 |
|---|---|
| 新增 Method4 参数扫描 / ablation | 优先通过 `tools/method4_benchmark_matrix.py` 的 JSON 配置组织；底层能力不足时再扩展 `tools/qdrant_two_level_routing_experiment.py` |
| 新增只读分析，例如 oracle routing locality | 如果逻辑较大，新增 `tools/method4_routing_oracle_analysis.py`；如果很小，可先放进主 harness |
| 新增 Qdrant server-side 执行优化 | `src/common/query.rs` 和 `lib/collection/src/collection/search.rs` |
| 新增 internal RPC | `src/tonic/api/points_internal_api.rs`, `src/tonic/api/query_common.rs`, `lib/collection/src/shards/remote_shard.rs` |
| 新增 compose 拓扑 | `tools/compose/` |
| 新增 harness 单元测试 | `tests/tools/test_qdrant_two_level_routing_experiment.py` 或新建 `tests/tools/test_<script>.py` |
| 新增 Rust 语义测试 | 放在对应 Rust 文件的 `#[cfg(test)]` 模块 |
| 新增实验结果整理 | `docs/experiments/YYYY-MM-DD-<name>.md` |

## 10. 最常见定位问题

| 问题 | 看哪里 |
|---|---|
| Orion/naive/KMeans 三种实验矩阵怎么组织 | `tools/method4_benchmark_matrix.py` 和 `tools/benchmark_configs/` |
| Method4 参数怎么传 | `tools/qdrant_two_level_routing_experiment.py` 的 `parse_args()` |
| `upper_k/base_ef/factor` 如何生成 search request | `tools/qdrant_two_level_routing_experiment.py` 的 routed plan / compact multi-EP 相关函数 |
| `point_to_shards` 如何恢复或构建 | `tools/qdrant_two_level_routing_experiment.py` 的 `faithful_original_rest` 路径 |
| physical placement map 怎么用 | `load_shard_placement_map`, `placement_for_shard_key` |
| server 如何按 shard 注入 entry points / EF | `src/common/query.rs`, `lib/collection/src/collection/search.rs` |
| worker-local peer pre-merge 在哪里 | `lib/collection/src/collection/search.rs::core_search_batch_shard_major_peer_premerge` |
| 怎么关闭 peer pre-merge 做 A/B | 环境变量 `QDRANT_DISABLE_SHARD_MAJOR_PEER_PREMERGE=1` |
| source-id dedup 在哪里 | `src/common/query.rs`, `lib/collection/src/collection/search.rs`, `src/tonic/api/query_common.rs` |
| lower HNSW 最终在哪里被调用 | `lib/collection/src/collection_manager/segments_searcher.rs` |
| Python harness 测试在哪 | `tests/tools/test_qdrant_two_level_routing_experiment.py` |
| 实验结果看哪里 | `results/<experiment_name>/<timestamp>/summary.json` |
