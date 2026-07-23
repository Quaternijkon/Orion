# Method4 Claim 支撑实验数据：Excel 绘图数据表

> [!WARNING]
> **Historical single-host evidence.** The tables below summarize the
> 2026-07-09 single-machine simulation campaign. Do not merge or relabel them as
> native four-host results. Current native-v4 evidence is published separately
> in
> [`2026-07-23-orion-native-v4-four-node-results.md`](2026-07-23-orion-native-v4-four-node-results.md)
> and its adjacent small evidence bundle.

生成日期：2026-07-09

本文件把当前 GloVe 范围内可以支撑 Method4 论文/报告 claim 的派生实验数据整理成“Excel 可画图”的数据块。每个图表块都说明它支撑的论据、推荐图形、X 轴、建议系列、数据来源和不能越界使用的 caveat。

## 实验配置（当前 GloVe 结果；单机模拟，计算和存储隔离）

本文档汇总的结果来自主三方法对比、worker-count 和 shard-count 等多组实验，因此不能再用“1 core/shard”或单一 worker 数量概括全部结果。未在单张图表或其 metadata 中另行说明的共同环境如下；CPU/分片/worker 的可变项单列在后。

| 项目 | 当前配置或范围 |
| --- | --- |
| 平台 | Qdrant |
| 运行方式 | 单机模拟；计算与存储隔离 |
| 宿主 CPU | Intel Xeon Gold 6330 @ 2.00GHz（2 sockets × 28 cores/socket × 2 threads/core） |
| 数据集 | GloVe-200-angular（文本嵌入） |
| 向量维度 | 200 |
| Base size | 1.2M vectors |
| 距离度量 | Angular |
| 当前主召回范围 | Recall@10 目标 0.80、0.85、0.90、0.95；另有约 0.97 的正向补充和 0.99 的负结果边界 |

### 部署与测量范围

| 实验类别 | 实际配置 |
| --- | --- |
| 主三方法路由对比 | 以 46 个有效逻辑分片为主；具体分片形态、batch size、查询数和重复次数以各数据表的来源 metadata 为准。 |
| worker-count 在线扩展补充 | 46 个有效逻辑分片，1 个 controller + 1/2/3/4 个物理 worker，分片按 round-robin 放置；controller 固定绑核 4 个逻辑 CPU，每个 worker 固定绑核 4 个逻辑 CPU；batch size=100、每个设置 3 次重复、每次 3000 个查询。该补充仅包含 Orion/Method4，不包含 Naive 或 Plain K-means 的 worker-count 控制组。 |
| shard-count latency 补充 | 在固定 3-worker 部署下对比 31 与 46 个有效逻辑分片；这是逻辑分片数实验，不能与物理 worker-count 扩展混写。 |

### 对比方法

| 方法 | 设置与角色 |
| --- | --- |
| Naïve hash all-shards | Qdrant 生产式 hash/random 分片；查询 fan-out 到全部有效逻辑分片，不使用选择性路由。 |
| Plain K-means | 普通、无均衡约束的 k-means 分区；查询使用 nprobe 选择簇/分片，nprobe 随目标召回校准。 |
| Orion（Balanced K-means） | 带约束的均衡 k-means 分区与 Method4 路由；这是本文的目标方法，而非 baseline。 |

如果从 Markdown 复制仍无法让 Excel 自动分列，请直接使用导出的 Excel/TSV 文件：`docs/experiments/2026-07-09-method4-claim-support-excel-tables/method4_claim_support_excel_tables.xlsx`，或 `docs/experiments/2026-07-09-method4-claim-support-excel-tables/tsv/*.tsv`。这些文件只包含原始表头和数据行，不包含 Markdown 说明或三反引号。

使用约定：
- 所有可复制数据表都采用 TSV 格式：列与列之间使用真实 Tab 制表符分隔，行与行之间使用换行分隔。
- 复制到 Excel 时，只复制每个 `TSV 数据区` 代码块内部的内容，从表头行开始，不复制三反引号。
- Recall@10 使用 0-1 小数；QPS 为 queries/s；P50/P95/P99 为 batch latency 毫秒；visited 为平均访问 shard 数；EF-sum 为每 query 分配的 lower-search EF 总量。
- `*_delta_pct`、`改变量(%)`、`降低(%)` 等列已经是百分比值，例如 `-40.00` 表示下降 40%。TSV 中的 `NA` 表示该源表未提供或该指标不适用。
- 绘制 same-recall 或 target-neighborhood 对比时，必须同时展示/标注 Recall@10，不要只画 QPS 或 tail latency。
- 当前范围只纳入 GloVe 主实验和 GloVe 当前范围补充实验；reduced SIFT/L2、GIST、Deep 等非 GloVe 泛化实验不作为本文档的主图表数据。

## 0. Claim 支撑数据总索引

下面这张表用于先定位“哪份数据支撑哪个论据”。`是否进入图表` 为“否”的行主要是审计/边界/范围说明，仍保留以避免过度声称。
TSV 数据区（复制代码块内部，从下一行表头开始）：
```tsv
序号	Claim	支撑级别	数据组	是否进入图表	支撑/使用方式	安全论述
1	A	partial	orion_vs_simple_kmeans_same_avg_ef	是	Use as the main online Claim A routing-efficiency statement.	At matched Recall@10 targets and similar average EF per visited shard, Orion reaches similar recall while visiting fewer shards and having higher QPS than Simple KMeans nprobe.
2	A	partial	partition_oracle_topology_locality	是	Use for topology-convergence mechanism wording only.	Topology convergence improves offline locality over balanced KMeans-only on edge cut, GT-shard entropy, and routed shard count at upper_k 80/120/160.
3	A	partial	partition_online_latency_submatrix	是	Use as online latency evidence for the current Claim A scope with explicit rebuild and no-overclaim caveats.	Existing full-fission/balanced-KMeans rows and 2026-07-09 current-harness random/topology/load-recalibrated rebuild rows reduce tail latency versus paired naive near the 0.95 operating region.
4	A	partial	topology_no_fission_selected_cell	是	Use to explain topology/no-fission tradeoff.	The selected topology/no-fission online cell visits fewer shards than full fission at similar recall, but is slightly slower and has slightly worse tail latency.
5	A	partial	partition_family_coverage_audit	否：审计/完成状态	Use to state that the current-scope Claim A online-row gap is closed with caveats.	The previously missing Claim A random/topology/load-recalibrated online rows are now available as 2026-07-09 current-harness rebuilds, while strict stronger wording still has caveats.
6	A	partial_current_scope	partition_family_current_rebuilds_20260709	是	Use as the direct provenance row for the newly completed Claim A family online latency cells.	The 2026-07-09 current-harness rebuilds fill the previously missing random, topology, and load-recalibrated Claim A online latency cells.
7	B	complete	multi_assignment_online_frontier	是	Use as Claim B online frontier evidence with stability caveat.	Under expansion at or below about 2x, Orion multi-assignment improves the online recall-QPS frontier versus single assignment.
8	B	complete	multi_assignment_oracle_miss	是	Use as the routing-miss mechanism proof for Claim B.	Orion default and w2c2 reduce offline oracle_gt_miss@10 versus single assignment at upper_k 80/120/160.
9	C	supported_with_batch_latency_supplement	dynamic_vs_fixed_same_recall	是	Use as the main Claim C efficiency statement.	Dynamic EF improves same-recall QPS and reduces estimated lower EF-sum/query compared with fixed EF in Orion Claim C ablations.
10	C	supported_with_batch_latency_supplement	routed_ep_relevance_proxy	是	Use as the mechanism proof for Claim C.	Routed entry-point count is a useful shard relevance proxy: GT-hit shards have far higher routed EP count than non-hit shards and receive larger Dynamic EF budget share.
11	C	supported_with_batch_latency_supplement	dynamic_vs_fixed_batch_latency	是	Use for batch P95/P99 wording only.	At the selected 0.95 operating point, Dynamic EF improves client-observed batch latency versus fixed EF with the same visited-shard count.
12	D	partial_high_recall_caveat	method4_vs_naive_095	是	Use as 0.95 Claim D support.	At Recall@10 around 0.95, Method4 has higher same-recall QPS and visits fewer shards than naive.
13	D	partial_high_recall_caveat	method4_vs_naive_097_latency	是	Use as the strongest high-recall Claim D row.	At Recall@10 around 0.97, Method4 improves QPS and client-observed P95/P99 versus naive while visiting fewer shards.
14	D	partial_high_recall_caveat	method4_vs_naive_batch200_high_recall	是	Use to satisfy the Claim D high-recall batch_size=200 experiment shape and to discuss the planned baseline-instability reporting caveat.	At batch=200, the selected about-0.97 Method4 high-recall configuration preserves the Claim D QPS/tail advantage versus Naive, while the selected about-0.99 row remains below Recall@10 0.99 and below paired Naive recall.
15	D	partial_high_recall_caveat	method4_vs_naive_097_closest_robustness	是	Use as additional robustness support for Claim D through the ~0.97 neighborhood, with explicit target-neighborhood caveat.	An additional Recall@10 about 0.97 closest-neighborhood Method4 configuration keeps the Method4-vs-naive QPS and P95/P99 advantage under original order, two shuffled-query seeds, and cold/no-warmup matched-window evaluation.
16	D	partial_high_recall_caveat	method4_vs_naive_099_caveat	是	Use as the 0.99 caveat.	The current evidence does not support a strong Method4 dominance claim at Recall@10 around 0.99.
17	D	partial_high_recall_caveat	method4_vs_naive_095_robustness	是	Use as robustness support for the 0.95 operating point alongside the stronger high-recall 0.97 row.	At the selected Recall@10 around 0.955 row, Method4 keeps its same-run QPS and P95/P99 advantage over naive under original, two shuffled-query seeds, and cold/no-warmup conditions.
18	D	partial_high_recall_caveat	method4_vs_naive_095_curve_config_robustness	是	Use as additional robustness support for Claim D through the ~0.95 neighborhood, with explicit target-neighborhood caveat.	An additional Recall@10 about 0.95 curve configuration keeps the Method4-vs-naive QPS and P95/P99 advantage under original order, two shuffled-query seeds, and cold/no-warmup matched-window evaluation.
19	D	partial_high_recall_caveat	method4_vs_naive_080_085_robustness	是	Use as lower-recall robustness support alongside the selected 0.90, 0.955, and 0.97 robustness overlays.	At selected Recall@10 neighborhoods around 0.80 and 0.85, Method4 keeps same-run QPS/P95/P99 advantage over naive under original, two shuffled-query seeds, and cold/no-warmup conditions.
20	D	partial_high_recall_caveat	method4_vs_naive_090_robustness	是	Use as lower-recall robustness support alongside the selected 0.955 and 0.97 robustness overlays.	At the selected Recall@10 around 0.90 target neighborhood, Method4 keeps same-run QPS/P95/P99 advantage over naive under original, two shuffled-query seeds, and cold/no-warmup conditions.
21	E	supported_with_overhead_metrics_gap	compact_request_execution_latency	是	Use as Claim E runtime efficiency evidence.	compact_current preserves recall while improving QPS and P95/P99 versus grouped materialized and client-expanded request modes in the current-runtime matrix.
22	E	supported_with_overhead_metrics_gap	compact_request_payload_and_wire_bytes	是	Use for request/response body-size overhead wording.	compact_current substantially reduces serialized JSON request/response body bytes versus client-expanded request mode.
23	E	supported_with_overhead_metrics_gap	runtime_overhead_and_build_gap	是	Use for selected overhead reporting and explicit cost gap.	Current artifacts include selected planning time, Docker container CPU/network/memory, process RSS, host-interface byte counters, Docker bridge packet capture, physical-NIC negative-control packet capture, smoke-level build-stage time/RSS instrumentation, selected /proc controller/worker process-level CPU/RSS attribution, and an audit showing full-size Method4 build-stage time/RSS are missing.
24	E	supported_with_overhead_metrics_gap	host_interface_byte_counters	是	Use as host-interface byte-counter overhead support with caveat; do not call it packet-level physical NIC attribution.	A selected batch=200 host-interface supplement shows compact_current greatly reduces Docker bridge/veth byte counters versus grouped materialized and client-expanded execution modes, while physical NIC host counters remain small on the single-host Docker deployment.
25	F	complete_with_runtime_caveat	worker_local_premerge_fanin	是	Use as Claim F mechanism evidence.	Worker-local pre-merge reduces candidate fan-in shape from logical shard streams to peer-level streams.
26	F	complete_with_runtime_caveat	worker_local_premerge_same_image_server_ab	是	Use as the production QPS supplement for Claim F.	Same-image server A/B shows peer pre-merge enabled has higher QPS than disabled at the selected Recall@10 about 0.955 window.
27	F	complete_with_runtime_caveat	worker_local_premerge_enabled_10k_sanity	是	Use as a long-window enabled-path sanity supplement for Claim F, not as production tail-latency or A/B proof.	An enabled peer pre-merge 10000-query run keeps the compact server execution shape and Recall@10 about 0.954, providing a longer-window sanity check for the enabled path.
28	F	complete_with_runtime_caveat	premerge_batch_matrix	是	Use for fan-in/latency-shape evidence with runtime caveat.	In the direct-peer batch matrix, local pre-merge preserves recall and reduces candidate groups/returned candidates by 87.40%.
29	G	complete	method4_aware_placement	是	Use as Claim G complete evidence with modest-gain wording.	Method4-aware physical layout reduces predicted hot-worker load and gives modest matched online latency/QPS improvements.
30	G	complete	method4_aware_concurrency8_supplement	是	Use as concurrency-pressure supplement for Claim G, not as the primary matched-layout proof.	Under concurrent single-query load, the deployed Method4-aware placement improves QPS and P95/P99 versus deployed round-robin while preserving recall.
31	H	complete	semantic_invariant_audit	是	Use as Claim H semantic fidelity evidence.	The compact_multi_ep wrapper preserves the intended external Method4 routing/search semantics in the sampled audit.
32	CrossCut	partial_reduced_l2	reduced_sift100k_l2_generalization	否：非 GloVe 当前范围	Use only as reduced L2 external-validity and key-ablation support.	Reduced SIFT-100k Euclidean evidence provides smaller external-validity support above Recall@10 0.95, now including no-fission Orion, full-fission Orion, Simple KMeans, and Naive latency supplements.
33	CrossCut	basic_worker_count_complete_comparative_partial	scalability_boundary	是	Use as basic Method4 online worker-count scaling support; use comparative physical-worker wording only if Naive/KMeans controls are later run.	Current scalability evidence includes logical shard-count scaling on a fixed 3-worker deployment plus a basic GloVe Method4-only online worker-count scaling supplement across worker_count 1/2/3/4.
34	CrossCut	available_selected_latency_cells_with_caveats	strict_ablation_boundary	是	Use as selected strict ablation latency support with caveats; do not use as broad strict same-recall multi-assignment dominance evidence.	The strict unified ablation status now has selected full-size multi-assignment online P95/P99 cells for single/default/w2c2, but they should be reported with target-neighborhood caveats rather than as strict same-recall proof across all pairs.
35	E	supported_with_overhead_metrics_gap	qdrant_prometheus_metrics_supplement	是	Use as selected server-exposed metrics corroboration for Claim E overhead; do not use as internal CPU/memory attribution.	At selected batch=200, Qdrant /metrics server-exposed REST/process counters corroborate the compact_current efficiency shape versus grouped materialized and client-expanded execution modes.
36	D	partial_high_recall_caveat	method4_vs_naive_099_strategy_search_negative	是	Use as transparent 0.99 caveat; do not claim dominance at 0.99.	Follow-up 0.99 strategy-search formal runs reinforce that current Method4 lower-EF/high-upper-k retuning does not reach Recall@10 0.99 on the 3000-query window.
37	D	partial_high_recall_caveat	method4_vs_naive_099_broader_strategy_negative	是	Use to strengthen the transparent 0.99 caveat: broader upper-routing can improve selected tail metrics versus naive controls but still did not preserve Recall@10 0.99 on the formal 3000-query window.	A broader upper-routing 0.99 screen produced 500-query Method4 candidates above Recall@10 0.99, but both selected formal 3000-query repeats still fell below 0.99 recall and below paired naive recall.
38	D	partial_high_recall_caveat	method4_vs_naive_099_alternate_width_negative	是	Use to strengthen transparent 0.99 caveat for alternate wider upper-routing strategies.	A 2026-07-08 alternate wider-upper-routing 0.99 screen found a promising 500-query Method4 candidate, but formal validation still fell below Recall@10 0.99 and below paired naive recall.
39	E	supported_with_overhead_metrics_gap	docker_bridge_packet_capture	是	Use as selected packet-capture support for Claim E runtime overhead; do not present it as physical network or subsystem/function-level attribution.	A selected batch=200 Docker bridge packet-capture supplement shows compact_current substantially reduces bridge packet count and frame/payload bytes versus grouped materialized and client-expanded execution modes.
40	CrossCut	current_scope_resolved_with_caveats	current_environment_preflight	否：审计/完成状态	Use internally to explain current run state and separate resolved current-scope rows from optional stronger claims.	Continuation preflight and post-run audits distinguish historical blockers from current state: Claim A, worker-count, and selected strict multi-assignment GloVe rows are now resolved with caveats.
41	D	partial_high_recall_caveat	method4_vs_naive_099_boundary_query_order	是	Use to support transparent 0.99 boundary wording and query-order sensitivity caveat.	Two shuffled-query repeats of the closest 0.99 Method4-vs-naive boundary row confirm that the current Method4 setting remains below Recall@10 0.99 and below the paired naive_ef200 recall.
42	D	partial_high_recall_caveat	method4_vs_naive_099_boundary_warmup	是	Use to support transparent 0.99 boundary and warmup-sensitivity wording; do not use as a positive 0.99 same-recall claim.	A cold/no-warmup matched-window repeat of the closest 0.99 boundary row confirms that Method4 remains below Recall@10 0.99 and below the paired naive_ef200 recall; warmup changes latency/QPS shape but does not close the 0.99 same-recall gap.
43	CrossCut	current_scope_glove_selected_rows_complete_with_caveats	goal_completion_audit	否：审计/完成状态	Use internally for current-scope completion accounting; preserve caveats in paper wording.	The evidence package distinguishes completed/downgraded claim support from optional stronger claims; basic worker-count, Claim A current-scope rows, and selected strict multi-assignment latency rows are complete with caveats for the current GloVe-only/no-0.99 scope.
44	E	supported_with_overhead_metrics_gap	physical_nic_packet_capture_negative_control	是	Use to remove the prior physical-NIC permission blocker and to justify saying the selected Claim E packet traffic is Docker bridge/veth-local in this setup; do not use as a positive physical-network byte-reduction metric.	In the single-host Docker Claim E execution window, the external physical NICs carried no matching Docker-subnet Qdrant traffic.
45	E	supported_with_overhead_metrics_gap	build_stage_instrumented_smoke	是	Use only to say build-stage time/RSS instrumentation has been exercised on a fresh smoke build; do not use as full build-cost evidence.	A smoke-level fresh Method4/Qdrant collection build now has wall-time and max-RSS instrumentation via /usr/bin/time -v.
46	E	supported_with_overhead_metrics_gap	controller_process_attribution	是	Use as selected process-level resource attribution for Claim E runtime overhead at batch=200; do not call it server-internal subsystem profiling.	A selected Claim E batch=200 supplement provides process-level controller and worker Qdrant PID CPU/RSS attribution for the three execution modes.
47	B	complete_with_selected_latency_caveats	strict_multiassign_latency_20260709	是	Use as selected latency support for the multi-assignment ablation; keep frontier/oracle rows as the primary Claim B mechanism evidence.	Selected full-size strict multi-assignment cells now have 3-repeat client-observed batch P95/P99 latency evidence for single/default/w2c2 in the 0.95 target neighborhood.
```

## 0.1 Naive 覆盖审计（三种分片/路由方式）

这里按用户定义的三种分片/路由方式审计 workbook：Orion、Simple KMeans、Naive hash all-shards。只有 route-family 对比表需要补齐 Naive；机制 ablation、执行开销、premerge、placement、semantic invariant 等表不强行加入 Naive，以免混入不同实验问题。
TSV 数据区（复制代码块内部，从下一行表头开始）：
```tsv
chart_id	table_title	naive_status	action	naive_source	rationale
A-01	Orion vs Simple KMeans vs Naive online efficiency	supplemented	added 4 naive_hash_all_shards_s46 target rows for 0.80/0.85/0.90/0.95	results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_method4_vs_naive_same_recall.csv	Route-family comparison previously had Orion and Simple KMeans only.
B-01	Multi-assignment recall-QPS frontier with Naive all-shards baseline	supplemented	added 4 naive_hash_all_shards_s46 target rows for 0.80/0.85/0.90/0.95	results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_method4_vs_naive_same_recall.csv	Frontier table previously had Orion and Simple KMeans families only; Naive rows provide the third routing family baseline.
B-02	Multi-assignment oracle GT miss with Naive all-shards reference	supplemented_reference	added 3 naive_all_shards_46 oracle rows for upper_k 80/120/160	results/method4_claim_coverage_20260704/derived_claim_tables/claim_a_partition_oracle_matrix.csv	Naive rows provide the all-shards oracle reference baseline; they are not multi-assignment strategies.
A-02	partition oracle locality matrix	already_present	none	results/method4_claim_coverage_20260704/derived_claim_tables/claim_a_partition_oracle_matrix.csv	Contains naive_all_shards_46 rows.
A-03	partition online latency deltas	already_present_as_baseline	none	results/method4_claim_coverage_20260704/derived_claim_tables/claim_a_partition_online_latency_submatrix_deltas.csv	Uses baseline_label=naive_all_shards for route-family latency comparisons.
A-04	current-harness partition-family rebuilds	already_present_as_wide_columns	none	results/method4_claim_coverage_20260704/derived_claim_tables/claim_a_partition_family_current_rebuilds_20260709.csv	Contains paired Naive Recall/QPS/visited/P95/P99 columns.
D-01/D-02/D-03/D-04/D-05/D-06	Method4-vs-Naive claim D tables	already_present	none	multiple Claim D derived CSVs	These tables are already Method4-vs-Naive paired comparisons or 0.99 boundary comparisons.
X-02/X-03	shard scaling and cost/footprint tables	already_present	none	crosscut_shard_scaling_latency_095_deltas.csv; crosscut_cost_main_collections.csv	Shard scaling/cost tables include Naive rows or Naive pair comparisons.
B-03/C/E/F/G/H/X-01	ablation, overhead, premerge, placement, semantic, worker-count tables	not_applicable_or_not_route_family_table	none	NA	These tables test a mechanism within Method4/Orion, execution overhead, physical placement, semantic invariants, or Method4-only worker scaling; adding a Naive row would mix a different experimental question.
```

- 来源：`results/method4_claim_coverage_20260704/derived_claim_tables/naive_coverage_audit_20260709.csv`
- 边界/注意：`not_applicable_or_not_route_family_table` 表示该表不是三路由方式对比，不代表缺实验数据。

## Claim A：拓扑/分区感知路由能提升局部性和在线效率

### 图 A-01: Orion vs Simple KMeans vs Naive：相近召回目标下的在线效率
- 支撑论据：在 0.80/0.85/0.90/0.95 目标附近，Orion、Simple KMeans 和 Naive hash all-shards 三种分片/路由方式可以同表比较；Naive 固定访问全部 46 个 lower shards，作为全分片搜索基线。
- 推荐图形：按目标召回分组的簇状柱形图；QPS 与 visited shards 可分成两张图，Recall@10 用折线标注。
- Excel 绘图设置：X 轴=`目标召回 + 方法`；建议系列=`Recall@10, QPS, 平均访问shards, EF-sum/query`。百分比列按数值绘制，列名中的 `(%)` 表示百分比点/百分比值。

TSV 数据区（复制代码块内部，从下一行表头开始）：
```tsv
目标召回	方法	Recall@10	QPS	平均访问shards	平均EF/访问shard	EF-sum/query
0.80	Orion	0.80397	1009.95	6.37	43.22	275.20
0.85	Orion	0.85450	887.78	8.30	63.38	526.22
0.90	Orion	0.90627	702.02	14.54	76.02	1105.13
0.95	Orion	0.95423	391.25	21.12	154.67	3266.70
0.80	Simple KMeans nprobe	0.80600	870.09	12.00	48.00	576.00
0.85	Simple KMeans nprobe	0.85203	617.78	16.00	64.00	1024.00
0.90	Simple KMeans nprobe	0.90407	352.45	32.00	80.00	2560.00
0.95	Simple KMeans nprobe	0.95377	206.90	40.00	160.00	6400.00
0.80	Naive hash all-shards	0.80633	439.19	46.00	16.00	736.00
0.85	Naive hash all-shards	0.85640	377.08	46.00	24.00	1104.00
0.90	Naive hash all-shards	0.90683	329.43	46.00	40.00	1840.00
0.95	Naive hash all-shards	0.95283	261.93	46.00	76.00	3496.00
```

- 来源：`results/method4_claim_coverage_20260704/derived_claim_tables/claim_a_route_family_three_way_online_targets_20260709.csv`；原 Orion/Simple 来源 `results/method4_claim_coverage_20260704/derived_claim_tables/claim_a_online_orion_vs_simple_kmeans_same_avg_ef.csv`；Naive 来源 `results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_method4_vs_naive_same_recall.csv`
- 边界/注意：Naive 行是 hash 分散、全 46 分片搜索的基线；A-01 现在用于三种 route-family 的 target-neighborhood 在线效率图，绘图时保留 Recall@10。

### 图 A-02: 分区族 oracle 局部性矩阵
- 支撑论据：Topology convergence 相比 balanced KMeans-only 降低 edge cut、GT-shard entropy 和 routed shard count；random/naive 用作弱局部性对照。
- 推荐图形：折线图或簇状柱形图；每个分区族一条线，分别画 routed shards、GT miss、entropy、edge cut。
- Excel 绘图设置：X 轴=`upper_k`；建议系列=`平均routed shards, oracle GT miss@10, GT shard entropy, topology edge cut`。百分比列按数值绘制，列名中的 `(%)` 表示百分比点/百分比值。

TSV 数据区（复制代码块内部，从下一行表头开始）：
```tsv
分区族	upper_k	平均routed shards	P95 routed shards	oracle GT miss@10	GT shard entropy	topology edge cut	index expansion
balanced_kmeans_only_46	80	16.30	28.00	0.01457	0.942	0.514	1.184
balanced_kmeans_only_46	120	20.05	32.00	0.00843	0.942	0.514	1.184
balanced_kmeans_only_46	160	22.93	34.00	0.00537	0.942	0.514	1.184
full_method4_fission_31_to_46	80	17.03	29.00	0.01603	0.934	0.437	1.188
full_method4_fission_31_to_46	120	21.05	33.00	0.00947	0.934	0.437	1.188
full_method4_fission_31_to_46	160	24.10	36.00	0.00633	0.934	0.437	1.188
kmeans_topology_46	80	13.95	24.00	0.01227	0.811	0.351	1.147
kmeans_topology_46	120	17.27	28.00	0.00770	0.811	0.351	1.147
kmeans_topology_46	160	19.93	31.00	0.00553	0.811	0.351	1.147
kmeans_topology_load_recalibrated_46	80	13.95	24.00	0.01227	0.811	0.351	1.147
kmeans_topology_load_recalibrated_46	120	17.27	28.00	0.00770	0.811	0.351	1.147
kmeans_topology_load_recalibrated_46	160	19.93	31.00	0.00553	0.811	0.351	1.147
naive_all_shards_46	80	46.00	46.00	0.00000	2.173	NA	1.000
naive_all_shards_46	120	46.00	46.00	0.00000	2.173	NA	1.000
naive_all_shards_46	160	46.00	46.00	0.00000	2.173	NA	1.000
random_balanced_46	80	37.81	42.00	0.06343	2.004	0.881	1.199
random_balanced_46	120	42.34	45.00	0.02500	2.004	0.881	1.199
random_balanced_46	160	44.31	46.00	0.00950	2.004	0.881	1.199
```

- 来源：`results/method4_claim_coverage_20260704/derived_claim_tables/claim_a_partition_oracle_matrix.csv`
- 边界/注意：只支持拓扑收敛的机制论述；当前 full-fission 不支持“全面优于 balanced KMeans-only”的强 oracle 论述。

### 图 A-03: 分区族在线 latency 子矩阵 delta
- 支撑论据：已有 full-fission/balanced-KMeans 行和当前补充行在 0.95 附近相对 naive 降低 tail latency；topology/no-fission 显示“更少 visited 但不一定更快”的 tradeoff。
- 推荐图形：以对比项为 X 轴的柱形图；QPS delta、visited delta、P95/P99 delta 分面绘制。
- Excel 绘图设置：X 轴=`对比 + batch_size`；建议系列=`QPS变化(%), visited变化(%), P95变化(%), P99变化(%)`。百分比列按数值绘制，列名中的 `(%)` 表示百分比点/百分比值。

TSV 数据区（复制代码块内部，从下一行表头开始）：
```tsv
对比	batch_size	方法	基线	方法Recall	基线Recall	QPS变化(%)	visited变化(%)	P95变化(%)	P99变化(%)
balanced_kmeans_existing_vs_naive	100	balanced_kmeans_existing	naive_all_shards	0.95233	0.95290	42.49	-53.29	-32.35	-33.54
balanced_kmeans_existing_vs_naive	200	balanced_kmeans_existing	naive_all_shards	0.95212	0.95290	121.95	-53.32	-71.14	-72.56
full_fission_existing_vs_naive	100	full_fission_existing	naive_all_shards	0.95293	0.95290	53.53	-50.70	-39.65	-40.61
full_fission_existing_vs_naive	200	full_fission_existing	naive_all_shards	0.95317	0.95290	87.38	-50.71	-57.23	-59.66
kmeans_topology_46_vs_naive_current_rebuild	200	kmeans_topology_46	naive_all_shards	0.95037	0.95290	96.98	-59.49	-55.04	-57.88
kmeans_topology_load_recalibrated_46_vs_naive_current_rebuild	200	kmeans_topology_load_recalibrated_46	naive_all_shards	0.95023	0.95290	87.53	-59.52	-47.35	-47.98
random_balanced_46_vs_naive_current_rebuild	200	random_balanced_46	naive_all_shards	0.95867	0.95290	39.27	-4.01	-30.77	-32.47
topology_no_fission_vs_full_fission	200	topology_no_fission	full_fission_existing	0.95217	0.95317	-2.77	-16.46	2.71	2.52
```

- 来源：`results/method4_claim_coverage_20260704/derived_claim_tables/claim_a_partition_online_latency_submatrix_deltas.csv`
- 边界/注意：部分行来自 2026-07-09 current-harness rebuild，不是 2026-07-04 oracle artifact 的逐字节 replay；negative/折中行要保留。

### 图 A-04: 2026-07-09 当前 harness 分区族补充行
- 支撑论据：补齐 random_balanced_46、kmeans_topology_46、kmeans_topology_load_recalibrated_46 三个此前缺失的 Claim A 在线 latency cell。
- 推荐图形：簇状柱形图或 combo 图；每个分区族画 Method4/Naive QPS 与 P95/P99，Recall 作为标签。
- Excel 绘图设置：X 轴=`分区族`；建议系列=`Method4/Naive Recall, QPS, visited, P95/P99`。百分比列按数值绘制，列名中的 `(%)` 表示百分比点/百分比值。

TSV 数据区（复制代码块内部，从下一行表头开始）：
```tsv
分区族	Method4 Recall	Naive Recall	Method4 QPS	Naive QPS	QPS变化(%)	Method4 visited	Naive visited	visited变化(%)	Method4 P95 ms	Naive P95 ms	P95变化(%)	Method4 P99 ms	Naive P99 ms	P99变化(%)	临时索引清理状态
random_balanced_46	0.95867	0.95290	320.74	230.29	39.27	44.16	46.00	-4.01	645.63	932.65	-30.77	646.57	957.46	-32.47	collection_deleted_after_latency
kmeans_topology_46	0.95037	0.95290	420.19	213.31	96.98	18.63	46.00	-59.49	520.72	1158.09	-55.04	531.97	1262.95	-57.88	collection_deleted_after_latency
kmeans_topology_load_recalibrated_46	0.95023	0.95290	444.01	236.77	87.53	18.62	46.00	-59.52	470.23	893.21	-47.35	476.54	916.01	-47.98	collection_deleted_after_latency
```

- 来源：`results/method4_claim_coverage_20260704/derived_claim_tables/claim_a_partition_family_current_rebuilds_20260709.csv`
- 边界/注意：这些行关闭当前 GloVe 范围的在线-row 缺口，但仍是 current-harness rebuild；load-recalibrated no-fission route map 与 topology route map 等价，仅权重重校准。

## Claim B：Multi-assignment 改善 recall-QPS frontier 并降低 oracle GT miss

### 图 B-01: Multi-assignment recall-QPS frontier + Naive all-shards baseline
- 支撑论据：在 expansion <= 约 2x 时，Orion default / w2c2 相比 single assignment 提升在线 recall-QPS frontier；补充的 Naive hash all-shards 行给出第三种 route-family 的全分片搜索基线。
- 推荐图形：折线图；X 轴为目标召回，每个 strategy 一条 QPS 线；另画 Recall@10、visited shards 与 index expansion。
- Excel 绘图设置：X 轴=`目标召回`；建议系列=`最终Recall, QPS, 平均访问shards, EF-sum/query, index expansion`。百分比列按数值绘制，列名中的 `(%)` 表示百分比点/百分比值。

TSV 数据区（复制代码块内部，从下一行表头开始）：
```tsv
family	策略	目标召回	最终Recall	QPS	平均访问shards	EF-sum/query	index expansion
Orion	orion_default	0.80	0.79630	1127.40	6.34	241.00	1.185
Orion	orion_default	0.85	0.84590	943.10	7.33	526.00	1.185
Orion	orion_default	0.90	0.89630	755.70	11.40	1034.00	1.185
Orion	orion_default	0.95	0.94790	409.10	21.14	2802.00	1.185
Orion	orion_single	0.80	0.80120	499.80	15.32	3119.00	1.000
Orion	orion_single	0.85	0.84680	372.70	25.50	4104.00	1.000
Orion	orion_single	0.90	0.94960	322.50	28.12	3910.00	1.000
Orion	orion_single	0.95	0.97110	219.10	30.15	8418.00	1.000
Orion	orion_w2c2	0.80	0.79880	1069.80	8.11	215.00	1.499
Orion	orion_w2c2	0.85	0.84930	915.10	8.12	430.00	1.499
Orion	orion_w2c2	0.90	0.91240	632.60	14.39	1004.00	1.499
Orion	orion_w2c2	0.95	0.95010	458.00	21.30	2142.00	1.499
Orion	orion_w2c3	0.80	0.80650	1001.90	7.86	206.00	1.852
Orion	orion_w2c3	0.85	0.85100	865.40	9.43	397.00	1.852
Orion	orion_w2c3	0.90	0.92820	591.20	16.66	1150.00	1.852
Orion	orion_w2c3	0.95	0.96180	385.20	24.48	2449.00	1.852
Simple KMeans	simple_a1.000	0.80	0.81640	845.60	10.00	640.00	1.000
Simple KMeans	simple_a1.000	0.85	0.85260	617.80	16.00	1024.00	1.000
Simple KMeans	simple_a1.000	0.90	0.89710	478.60	20.00	1920.00	1.000
Simple KMeans	simple_a1.000	0.95	0.95320	223.10	24.00	5760.00	1.000
Simple KMeans	simple_a1.004	0.80	0.81260	700.00	8.00	512.00	1.191
Simple KMeans	simple_a1.004	0.85	0.86220	618.80	12.00	960.00	1.191
Simple KMeans	simple_a1.004	0.90	0.89430	429.20	20.00	1600.00	1.191
Simple KMeans	simple_a1.004	0.95	0.95250	219.50	24.00	4800.00	1.191
Simple KMeans	simple_a1.010	0.80	0.83080	726.10	8.00	512.00	1.566
Simple KMeans	simple_a1.010	0.85	0.85970	567.40	12.00	768.00	1.566
Simple KMeans	simple_a1.010	0.90	0.90470	409.90	20.00	1600.00	1.566
Simple KMeans	simple_a1.010	0.95	0.95770	199.00	24.00	4800.00	1.566
Simple KMeans	simple_a1.014	0.80	0.81900	706.70	8.00	384.00	1.887
Simple KMeans	simple_a1.014	0.85	0.88500	539.10	12.00	960.00	1.887
Simple KMeans	simple_a1.014	0.90	0.91040	400.80	20.00	1600.00	1.887
Simple KMeans	simple_a1.014	0.95	0.95330	220.00	24.00	3840.00	1.887
Naive	naive_hash_all_shards_s46	0.80	0.80633	439.19	46.00	736.00	1.000
Naive	naive_hash_all_shards_s46	0.85	0.85640	377.08	46.00	1104.00	1.000
Naive	naive_hash_all_shards_s46	0.90	0.90683	329.43	46.00	1840.00	1.000
Naive	naive_hash_all_shards_s46	0.95	0.95283	261.93	46.00	3496.00	1.000
```

- 来源：`results/method4_claim_coverage_20260704/derived_claim_tables/claim_b_frontier_with_naive_all_shards_20260709.csv`；原 Orion/Simple 来源 `results/method4_claim_coverage_20260704/derived_claim_tables/claim_b_multiassign_expansion_qps.csv`；Naive 来源 `results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_method4_vs_naive_same_recall.csv`
- 边界/注意：历史 frontier 多为 single final-eval run；Naive 行是全 46 分片搜索基线，不是 multi-assignment ablation；严格 latency 补充见 B-03。

### 图 B-02: Multi-assignment oracle GT miss + Naive all-shards reference
- 支撑论据：Orion default 和 w2c2 在 upper_k 80/120/160 上相对 single assignment 降低 oracle_gt_miss@10；补充的 Naive all-shards 行给出“搜索全部分片时 oracle miss 为 0”的全分片参考基线。
- 推荐图形：折线图；X 轴 upper_k，每个 strategy 一条 miss@10 线；coverage 可单独画，Naive 行可用虚线或灰色 reference 标记。
- Excel 绘图设置：X 轴=`upper_k`；建议系列=`oracle GT miss@10, oracle GT coverage@10, index expansion`。百分比列按数值绘制，列名中的 `(%)` 表示百分比点/百分比值。

TSV 数据区（复制代码块内部，从下一行表头开始）：
```tsv
策略	upper_k	oracle GT coverage@10	oracle GT miss@10	平均routed shards	P95 routed shards	index expansion
orion_default	80	0.98367	0.01633	17.19	30.00	1.184
orion_default	120	0.99043	0.00957	21.23	34.00	1.184
orion_default	160	0.99397	0.00603	24.29	37.00	1.184
orion_single	80	0.96113	0.03887	15.35	25.00	1.000
orion_single	120	0.97587	0.02413	18.85	29.00	1.000
orion_single	160	0.98333	0.01667	21.58	32.00	1.000
orion_w2c2	80	0.99393	0.00607	21.74	37.00	1.495
orion_w2c2	120	0.99670	0.00330	26.70	42.00	1.495
orion_w2c2	160	0.99787	0.00213	30.38	45.00	1.495
naive_all_shards_46	80	1.00000	0.00000	46.00	46.00	1.000
naive_all_shards_46	120	1.00000	0.00000	46.00	46.00	1.000
naive_all_shards_46	160	1.00000	0.00000	46.00	46.00	1.000
```

- 来源：`results/method4_claim_coverage_20260704/derived_claim_tables/claim_b_orion_oracle_gt_miss_with_naive_reference_20260709.csv`；原 Orion 来源 `results/method4_claim_coverage_20260704/derived_claim_tables/claim_b_orion_oracle_gt_miss.csv`；Naive 参考来源 `results/method4_claim_coverage_20260704/derived_claim_tables/claim_a_partition_oracle_matrix.csv`
- 边界/注意：Naive all-shards 行是全 46 分片搜索的 reference baseline，不是 multi-assignment 策略；它用于展示搜索全部分片的 oracle 上限和 routed-shard 代价。

### 图 B-03: 严格 multi-assignment selected latency cells
- 支撑论据：在 0.95 目标邻域，default/w2c2 的 selected full-size latency cell 显示更高 QPS、更低 EF-sum 和更低 tail latency。
- 推荐图形：簇状柱形图；策略为 X 轴，QPS 和 P95/P99 分图；Recall 作为数据标签。
- Excel 绘图设置：X 轴=`策略`；建议系列=`Recall@10, QPS, visited, EF-sum, P95/P99, index expansion`。百分比列按数值绘制，列名中的 `(%)` 表示百分比点/百分比值。

TSV 数据区（复制代码块内部，从下一行表头开始）：
```tsv
策略	配置	latency batch	Recall@10	QPS	visited	EF-sum	P95 ms	P99 ms	index expansion
single_assignment	single_095	200	0.97317	279.23	30.18	8421.04	736.45	739.58	1.000
orion_default_top_ties	default_095	200	0.95260	475.27	21.35	3088.52	432.73	435.41	1.186
orion_w2c2_vote_delta2_cap2	w2c2_095	200	0.94913	557.80	21.24	2127.40	375.01	376.76	1.499
```

- 来源：`results/method4_claim_coverage_20260704/derived_claim_tables/strict_multiassign_latency_20260709.csv`
- 边界/注意：这是 target-neighborhood latency evidence：single 高于 0.95，w2c2 略低于 0.95；不要写成严格 same-recall 全矩阵证明。

## Claim C：Dynamic EF 在相同/近似召回下提升预算效率和 batch tail latency

### 图 C-01: Dynamic EF vs Fixed EF：same/near-same recall 性能
- 支撑论据：Dynamic EF 在三个 same/near-same recall 对比中提高 QPS 约 27-31%，并降低 EF-sum/query 约 35-41%。
- 推荐图形：分组柱形图；数据表 1 画 QPS/EF-sum，数据表 2 画 QPS变化和 EF-sum变化。
- Excel 绘图设置：X 轴=`对比 + 方法`；建议系列=`Recall@10, QPS, visited, EF-sum/query`。百分比列按数值绘制，列名中的 `(%)` 表示百分比点/百分比值。

数据表 C-01a：summary。

TSV 数据区（复制代码块内部，从下一行表头开始）：
```tsv
对比	方法	Recall@10	QPS	visited	平均EF/visited	EF-sum/query
fixed_upper_k_120_isolated_095	Dynamic EF	0.94850	476.19	19.33	147.72	2855.37
frontier_robust_095	Dynamic EF	0.95153	485.08	17.10	182.21	3115.12
frontier_robust_097	Dynamic EF	0.97347	308.75	26.61	225.29	5994.48
fixed_upper_k_120_isolated_095	Fixed EF	0.94933	374.36	19.30	240.00	4632.88
frontier_robust_095	Fixed EF	0.95200	376.32	17.09	280.00	4786.51
frontier_robust_097	Fixed EF	0.97177	235.83	21.09	480.00	10125.12
```

数据表 C-01b：Dynamic minus Fixed delta。

TSV 数据区（复制代码块内部，从下一行表头开始）：
```tsv
对比	Recall差(Dyn-Fix)	QPS变化(%)	visited变化(%)	EF-sum变化(%)
fixed_upper_k_120_isolated_095	-0.00083	27.20	0.13	-38.37
frontier_robust_095	-0.00047	28.90	0.01	-34.92
frontier_robust_097	0.00170	30.92	26.14	-40.80
```

- 来源：`results/method4_claim_coverage_20260704/derived_claim_tables/claim_c_dynamic_vs_fixed_performance_summary.csv`；`results/method4_claim_coverage_20260704/derived_claim_tables/claim_c_dynamic_vs_fixed_deltas.csv`
- 边界/注意：这里的重点是 EF 预算效率，不要声称 Dynamic EF 总是减少 visited shards；frontier_097 的 visited 反而增加。

### 图 C-02: Dynamic EF batch latency supplement
- 支撑论据：在 Recall@10 基本相同的 batch endpoint 补充实验中，Dynamic EF 提升 QPS、降低 EF-sum，并降低 P95/P99 batch latency。
- 推荐图形：单行 waterfall/柱形图；绘制 QPS变化、EF-sum变化、P95/P99变化。
- Excel 绘图设置：X 轴=`对比`；建议系列=`QPS变化(%), EF-sum变化(%), P95变化(%), P99变化(%)`。百分比列按数值绘制，列名中的 `(%)` 表示百分比点/百分比值。

TSV 数据区（复制代码块内部，从下一行表头开始）：
```tsv
对比	Recall差(Dyn-Fix)	QPS变化(%)	visited变化(%)	EF-sum变化(%)	P95变化(%)	P99变化(%)	fixed repeats	dynamic repeats
latency_rerun_u80_095_b200	0.00017	26.75	0.00	-32.52	-21.00	-20.97	3	3
```

- 来源：`results/method4_claim_coverage_20260704/derived_claim_tables/claim_c_dynamic_vs_fixed_latency_deltas.csv`
- 边界/注意：这是 client-observed batch endpoint wall time，不是 server-internal lower-search trace。

### 图 C-03: routed EP count 相关性/机制证据
- 支撑论据：GT-hit shards 上的 routed EP count 高于非 hit shards，支持用 routed EP count 作为 Dynamic EF 预算分配的相关性代理。
- 推荐图形：柱形图；每个配置画 GT-hit vs non-hit EP count，也可画 budget share。
- Excel 绘图设置：X 轴=`配置`；建议系列=`GT-hit EP count, non-hit EP count, Dynamic/Fixed GT budget share`。百分比列按数值绘制，列名中的 `(%)` 表示百分比点/百分比值。

TSV 数据区（复制代码块内部，从下一行表头开始）：
```tsv
配置	upper_k	visited shards	GT hit shards上EP数	非GT-hit shards上EP数	EP hit-nonhit差	Dynamic GT budget share	Fixed GT budget share	visited GT copy coverage
fixed_width_claim_c_095	120	21.11	20.34	3.22	17.12	0.3493	0.1753	0.9854
frontier_095_dynamic	80	17.10	14.72	2.55	12.17	0.4324	0.2105	0.9752
frontier_097_dynamic	200	26.59	30.78	4.54	26.25	0.4264	0.1422	0.9932
```

- 来源：`results/method4_claim_coverage_20260704/derived_claim_tables/claim_c_routed_ep_relevance_summary.csv`
- 边界/注意：这是 relevance proxy 机制证据，不是独立的在线性能 A/B。

## Claim D：Method4 相对 Naive 在当前 0.80-0.95 范围和约 0.97 边界内提升效率

### 图 D-01: Method4 vs Naive：0.95 same-recall 主对比
- 支撑论据：在 0.955 附近，Method4 与 Naive Recall@10 接近，同时 QPS 更高并访问更少 shards。
- 推荐图形：簇状柱形图；X 轴为方法/配置，画 QPS、visited；Recall 用折线或标签。
- Excel 绘图设置：X 轴=`标签`；建议系列=`Recall@10, QPS, visited, EF-sum/query`。百分比列按数值绘制，列名中的 `(%)` 表示百分比点/百分比值。

TSV 数据区（复制代码块内部，从下一行表头开始）：
```tsv
标签	方法	Recall@10	QPS	visited	EF-sum/query	search req/query	candidate groups/query
Method4 current 0.955 batch200	Method4	0.95527	387.14	23.21	3274.33	1.00	1.00
curve target 0.8 orion	Method4	0.80430	1090.33	5.96	268.27	NA	NA
curve target 0.85 orion	Method4	0.86187	961.29	6.84	654.28	NA	NA
curve target 0.9 orion	Method4	0.90347	715.66	13.10	1045.39	NA	NA
curve target 0.95 orion	Method4	0.95117	390.71	22.69	2885.26	NA	NA
Naive stable closest 0.955 batch100 ef76	Naive	0.95477	272.37	46.00	3496.00	1.00	NA
Naive stable higher recall batch100 ef80	Naive	0.95733	264.64	46.00	3680.00	1.00	NA
curve target 0.8 naive	Naive	0.80633	439.19	46.00	736.00	NA	NA
curve target 0.85 naive	Naive	0.85640	377.08	46.00	1104.00	NA	NA
curve target 0.9 naive	Naive	0.90683	329.43	46.00	1840.00	NA	NA
curve target 0.95 naive	Naive	0.95283	261.93	46.00	3496.00	NA	NA
```

- 来源：`results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_method4_vs_naive_same_recall.csv`
- 边界/注意：这是 0.95 主支撑行；不同 batch_size/控制行不要混成不标注的平均值。

### 图 D-02: Method4 vs Naive：约 0.97 正向行与 0.99 边界行
- 支撑论据：约 0.97 strict/neighbor 行支持 high-recall 范围内 QPS、visited、tail latency 优势；0.99 行只作为边界/负结果。
- 推荐图形：柱形图；目标为 X 轴，画 delta 指标；0.99 行用不同颜色标为 boundary。
- Excel 绘图设置：X 轴=`目标`；建议系列=`QPS变化(%), visited变化(%), P95/P99变化(%), Recall差`。百分比列按数值绘制，列名中的 `(%)` 表示百分比点/百分比值。

TSV 数据区（复制代码块内部，从下一行表头开始）：
```tsv
目标	Method4配置	Naive配置	Method4 Recall	Naive Recall	QPS变化(%)	visited变化(%)	EF-sum变化(%)	P95变化(%)	P99变化(%)
~0.97 strict >=0.97	m4_160_80_20	naive_ef112	0.97180	0.97277	40.58	-46.39	7.19	-31.85	-32.42
~0.97 closest-neighborhood	m4_160_80_16	naive_ef104	0.96840	0.96953	43.90	-46.39	0.60	-33.29	-32.88
~0.99 neighborhood	m4_400_160_20	naive_ef200	0.98927	0.99017	-2.63	-25.50	56.76	0.50	0.71
~0.99 conservative naive higher recall	m4_400_160_20	naive_ef240	0.98927	0.99363	8.41	-25.50	30.64	-11.56	-11.24
```

- 来源：`results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_high_recall_latency_pairs.csv`
- 边界/注意：不要把 0.99 neighborhood 写成正向 dominance；Method4 recall 低于 Naive 且原始 0.99 行 QPS/tail 不占优。

### 图 D-03: Method4 vs Naive：batch=200 high-recall supplement
- 支撑论据：batch=200 形态下约 0.97 行给出强正向 same-batch 证据；约 0.99 行仍是边界证据。
- 推荐图形：两组柱形图；按目标画 QPS/visited/P95/P99 delta；Naive CV 作为误差/脚注。
- Excel 绘图设置：X 轴=`目标`；建议系列=`QPS变化(%), visited变化(%), P95/P99变化(%), Naive CV`。百分比列按数值绘制，列名中的 `(%)` 表示百分比点/百分比值。

TSV 数据区（复制代码块内部，从下一行表头开始）：
```tsv
对比	目标	batch_size	Method4 Recall	Naive Recall	QPS变化(%)	visited变化(%)	EF-sum变化(%)	P95变化(%)	P99变化(%)	Naive QPS CV	边界标记
batch200_097_m4_160_80_20_vs_naive_ef112	~0.97 strict batch=200	200	0.97183	0.97277	88.01	-46.40	7.18	-67.32	-69.33	0.307	supports_through_about_0.97_at_batch200; naive_batch200_tail_unstable_so_report_strict_same_batch_and_stable_batch100_context
batch200_099_m4_400_160_20_vs_naive_ef200	~0.99 boundary batch=200	200	0.98950	0.99017	30.53	-25.48	56.78	-54.26	-58.41	0.176	method4_recall_below_0.99_and_below_naive; use_as_batch200_boundary_evidence_not_0.99_dominance
```

- 来源：`results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_batch200_high_recall_deltas.csv`
- 边界/注意：Naive batch=200 控制行 tail/QPS 方差较高；0.99 行 Method4 仍低于 0.99 且低于 paired naive recall。

### 图 D-04: Method4 vs Naive：selected robustness overlays
- 支撑论据：在选定 0.80/0.85/0.90/0.955/0.95-curve/0.97-neighbor 场景中，Method4 相对 same-run Naive 保持 QPS/visited/tail latency 优势。
- 推荐图形：热力图或分组柱形图；X 轴为场景，颜色按目标组，画 QPS/P95/P99 delta。
- Excel 绘图设置：X 轴=`目标组 + 场景`；建议系列=`Recall@10, QPS变化(%), visited变化(%), P95/P99变化(%)`。百分比列按数值绘制，列名中的 `(%)` 表示百分比点/百分比值。

TSV 数据区（复制代码块内部，从下一行表头开始）：
```tsv
目标组	场景	Method4配置	Naive配置	Method4 Recall	Naive Recall	QPS变化(%)	visited变化(%)	EF-sum变化(%)	P95变化(%)	P99变化(%)
0.80/0.85	warm original ~0.80	m4_16_16_10	naive_ef16	0.80633	0.80747	258.35	-86.15	-62.36	-74.04	-75.22
0.80/0.85	shuffled seed 20260709 ~0.80	m4_16_16_10	naive_ef16	0.80627	0.80747	227.32	-86.12	-62.34	-72.24	-72.09
0.80/0.85	shuffled seed 20260712 ~0.80	m4_16_16_10	naive_ef16	0.80247	0.80747	213.41	-87.08	-63.36	-70.45	-70.72
0.80/0.85	warm original ~0.85	m4_20_64_10	naive_ef24	0.86273	0.85570	157.19	-84.03	-37.58	-63.84	-64.44
0.80/0.85	shuffled seed 20260709 ~0.85	m4_20_64_10	naive_ef24	0.86280	0.85570	199.61	-84.02	-37.54	-68.53	-70.70
0.80/0.85	shuffled seed 20260712 ~0.85	m4_20_64_10	naive_ef24	0.85730	0.85570	183.08	-85.15	-40.60	-66.88	-67.21
0.80/0.85	cold matched ~0.80	m4_16_16_10	naive_ef16	0.80600	0.80747	251.10	-86.13	-62.36	-73.05	-73.08
0.80/0.85	cold matched ~0.85	m4_20_64_10	naive_ef24	0.86223	0.85570	181.82	-84.03	-37.59	-67.73	-70.23
0.90	warm original ~0.90	m4_60_40_8	naive_ef40	0.90657	0.90680	143.77	-68.89	-40.16	-61.48	-62.31
0.90	shuffled seed 20260708 ~0.90	m4_60_40_8	naive_ef40	0.90537	0.90680	138.88	-68.90	-40.16	-60.31	-61.84
0.90	shuffled seed 20260711 ~0.90	m4_60_40_8	naive_ef40	0.90127	0.90680	154.44	-71.88	-43.25	-63.41	-62.87
0.90	cold matched ~0.90	m4_60_40_8	naive_ef40	0.90663	0.90680	140.85	-68.89	-40.16	-61.06	-61.93
0.955	warm original ~0.955	m4_160_80_8	naive_ef76	0.95527	0.95433	47.59	-48.69	-5.43	-37.74	-38.98
0.955	shuffled seed 20260707 ~0.955	m4_160_80_8	naive_ef76	0.95553	0.95433	78.85	-48.66	-5.41	-46.37	-45.98
0.955	shuffled seed 20260710 ~0.955	m4_160_80_8	naive_ef76	0.95467	0.95433	62.55	-49.55	-6.36	-43.87	-44.66
0.955	cold matched ~0.955	m4_160_80_8	naive_ef76	0.95530	0.95433	51.90	-48.69	-5.43	-39.96	-39.23
0.95 curve	warm original ~0.95 curve config	m4_curve_160_50_10	naive_ef76	0.95380	0.95433	69.05	-48.69	-15.54	-45.31	-44.12
0.95 curve	shuffled seed 20260714 ~0.95 curve config	m4_curve_160_50_10	naive_ef76	0.95370	0.95433	73.97	-48.70	-15.55	-46.53	-48.07
0.95 curve	shuffled seed 20260715 ~0.95 curve config	m4_curve_160_50_10	naive_ef76	0.95300	0.95433	72.18	-48.64	-15.52	-46.02	-46.91
0.95 curve	cold matched ~0.95 curve config	m4_curve_160_50_10	naive_ef76	0.95337	0.95433	72.50	-48.68	-15.55	-46.98	-47.69
0.97 closest	warm original ~0.97 closest-neighborhood	repeat_m4_160_80_16	naive_ef104	0.96867	0.96953	41.09	-48.65	-1.24	-28.09	-28.07
0.97 closest	shuffled seed 20260716 ~0.97 closest-neighborhood	repeat_m4_160_80_16	naive_ef104	0.96823	0.96953	54.97	-48.63	-1.22	-41.45	-43.32
0.97 closest	shuffled seed 20260717 ~0.97 closest-neighborhood	repeat_m4_160_80_16	naive_ef104	0.96847	0.96953	57.27	-48.65	-1.23	-41.27	-42.54
0.97 closest	cold matched ~0.97 closest-neighborhood	repeat_m4_160_80_16	naive_ef104	0.96850	0.96953	54.19	-48.63	-1.23	-38.10	-38.49
```

- 来源：`results/method4_claim_coverage_20260704/derived_claim_tables/robustness_080_085_method4_vs_naive_deltas.csv`；`results/method4_claim_coverage_20260704/derived_claim_tables/robustness_090_method4_vs_naive_deltas.csv`；`results/method4_claim_coverage_20260704/derived_claim_tables/robustness_095_method4_vs_naive_deltas.csv`；`results/method4_claim_coverage_20260704/derived_claim_tables/robustness_095_curve_config_method4_vs_naive_deltas.csv`；`results/method4_claim_coverage_20260704/derived_claim_tables/robustness_097_closest_method4_vs_naive_deltas.csv`
- 边界/注意：这些是 selected overlays，不是完整 recall/config 矩阵；部分行是 target-neighborhood 而非严格 same-recall。

### 图 D-05: 0.99 retune/strategy search：负结果边界数据
- 支撑论据：多个 0.99 retune 和更宽 upper-routing 候选在 500-query screening 后未能在 formal 3000-query 窗口保持 Method4 Recall@10 0.99；支撑“0.99 dominance 不成立/不纳入当前 scope”的边界论据。
- 推荐图形：散点图；X 轴为 Method4 Recall，Y 轴 QPS变化或 P95变化；用 0.99 竖线标出阈值。
- Excel 绘图设置：X 轴=`Method4 Recall`；建议系列=`Naive Recall, QPS变化(%), P95/P99变化(%)`。百分比列按数值绘制，列名中的 `(%)` 表示百分比点/百分比值。

TSV 数据区（复制代码块内部，从下一行表头开始）：
```tsv
来源表	对比	Method4配置	Naive配置	Method4 Recall	Naive Recall	QPS变化(%)	visited变化(%)	EF-sum变化(%)	P95变化(%)	P99变化(%)
claim_d_099_retune_deltas.csv	m4_480_70_16 vs naive_ef200	m4_480_70_16	naive_ef200	0.98667	0.99017	13.34	-21.33	21.01	-18.53	-22.36
claim_d_099_retune_deltas.csv	m4_520_60_16 vs naive_ef200	m4_520_60_16	naive_ef200	0.98730	0.99017	10.15	-19.90	25.35	-16.02	-20.73
claim_d_099_strategy_search_deltas.csv	m4_560_40_16 vs naive_ef200	m4_560_40_16	naive_ef200	0.98747	0.99017	6.18	-18.27	25.53	-11.36	-6.39
claim_d_099_strategy_search_deltas.csv	m4_560_40_16 vs naive_ef240	m4_560_40_16	naive_ef240	0.98747	0.99363	10.07	-18.27	4.61	-9.32	-5.29
claim_d_099_strategy_search_deltas.csv	m4_560_60_14 vs naive_ef200	m4_560_60_14	naive_ef200	0.98670	0.99017	9.25	-18.27	20.06	-13.38	-13.84
claim_d_099_strategy_search_deltas.csv	m4_560_60_14 vs naive_ef240	m4_560_60_14	naive_ef240	0.98670	0.99363	13.26	-18.27	0.05	-11.38	-12.83
claim_d_099_strategy_search_deltas.csv	m4_480_80_16 vs naive_ef200	m4_480_80_16	naive_ef200	0.98710	0.99017	14.70	-20.94	25.12	-19.08	-19.34
claim_d_099_strategy_search_deltas.csv	m4_480_80_16 vs naive_ef240	m4_480_80_16	naive_ef240	0.98710	0.99363	21.87	-20.94	4.27	-22.49	-25.55
claim_d_099_strategy_search_deltas.csv	m4_520_60_16 vs naive_ef200	m4_520_60_16	naive_ef200	0.98740	0.99017	12.59	-19.57	25.47	-17.72	-18.31
claim_d_099_strategy_search_deltas.csv	m4_520_60_16 vs naive_ef240	m4_520_60_16	naive_ef240	0.98740	0.99363	19.63	-19.57	4.56	-21.19	-24.60
claim_d_099_broader_strategy_formal_deltas.csv	m4_u720_b60_f12_vs_naive_ef200	m4_u720_b60_f12	naive_ef200	0.98800	0.99017	3.26	-13.85	31.41	-10.48	-11.89
claim_d_099_broader_strategy_formal_deltas.csv	m4_u720_b60_f12_vs_naive_ef240	m4_u720_b60_f12	naive_ef240	0.98800	0.99363	11.12	-13.85	9.51	-15.27	-16.39
claim_d_099_broader_strategy_formal_deltas.csv	m4_u720_b40_f14_vs_naive_ef200	m4_u720_b40_f14	naive_ef200	0.98870	0.99017	-0.64	-13.82	40.39	-5.50	-7.35
claim_d_099_broader_strategy_formal_deltas.csv	m4_u720_b40_f14_vs_naive_ef240	m4_u720_b40_f14	naive_ef240	0.98870	0.99363	8.24	-13.82	17.00	-13.50	-15.15
claim_d_099_alternate_width_formal_deltas.csv	m4_u960_b20_f10_vs_naive_ef200	m4_u960_b20_f10	naive_ef200	0.98760	0.99017	3.20	-9.66	26.66	-10.16	-11.89
claim_d_099_alternate_width_formal_deltas.csv	m4_u960_b20_f10_vs_naive_ef240	m4_u960_b20_f10	naive_ef240	0.98760	0.99363	12.28	-9.66	5.55	-16.84	-19.10
```

- 来源：`results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_099_retune_deltas.csv`；`results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_099_strategy_search_deltas.csv`；`results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_099_broader_strategy_formal_deltas.csv`；`results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_099_alternate_width_formal_deltas.csv`
- 边界/注意：这是透明负结果，不是正向 claim；只用于限定 Claim D 的适用范围。

### 图 D-06: 0.99 boundary：query-order / warmup sensitivity
- 支撑论据：0.99 边界行在 query order 和 cold/no-warmup 下仍未关闭 same-recall gap；支撑把 0.99 作为 boundary 而非主 claim。
- 推荐图形：柱形图；X 轴为敏感性场景，画 Recall变化、QPS变化、P95/P99变化。
- Excel 绘图设置：X 轴=`敏感性类型 + comparison`；建议系列=`Recall变化, QPS变化(%), P95/P99变化(%)`。百分比列按数值绘制，列名中的 `(%)` 表示百分比点/百分比值。

TSV 数据区（复制代码块内部，从下一行表头开始）：
```tsv
敏感性类型	方法	配置	baseline	comparison	Recall变化	QPS变化(%)	visited变化(%)	EF-sum变化(%)	P95变化(%)	P99变化(%)
query_order	Method4	m4_400_160_20	original query order	shuffled seed 20260713	-0.00003	-5.09	0.01	0.01	5.16	4.68
query_order	Naive	naive_ef200	original query order	shuffled seed 20260713	0.00000	-8.29	0.00	0.00	14.88	13.93
query_order	Method4	m4_400_160_20	original query order	shuffled seed 20260718	0.00017	-4.10	-0.06	-0.02	4.40	2.11
query_order	Naive	naive_ef200	original query order	shuffled seed 20260718	0.00000	-7.11	0.00	0.00	13.99	16.23
warmup	Method4	m4_400_160_20	original query order	cold no-warmup matched window	0.00010	-5.95	0.04	0.02	7.08	4.96
warmup	Naive	naive_ef200	original query order	cold no-warmup matched window	0.00000	-8.69	0.00	0.00	16.59	16.17
```

- 来源：`results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_099_boundary_query_order_sensitivity.csv`；`results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_099_boundary_warmup_sensitivity.csv`
- 边界/注意：只用于边界和敏感性说明；不要与 0.97 正向行混成平均。

## Claim E：compact request / execution mode 降低请求对象、字节和运行时开销

### 图 E-01: compact_current vs materialized/client-expanded：执行模式 latency delta
- 支撑论据：compact_current 在多个 batch size 上减少 search request/candidate group，并提升 QPS、降低 P95/P99。
- 推荐图形：按 batch_size 分组的柱形图；分别画 QPS变化、P95/P99变化、request/candidate 降低。
- Excel 绘图设置：X 轴=`batch_size + 对比`；建议系列=`QPS变化(%), P95/P99变化(%), search requests降低(%), candidate groups降低(%)`。百分比列按数值绘制，列名中的 `(%)` 表示百分比点/百分比值。

TSV 数据区（复制代码块内部，从下一行表头开始）：
```tsv
batch_size	对比	状态	Recall差	QPS变化(%)	P50变化(%)	P95变化(%)	P99变化(%)	search requests降低(%)	candidate groups降低(%)	returned candidates降低(%)
50	compact_vs_client_expanded_negative_control	ok	0.00000	161.50	-61.13	-60.73	-67.22	95.76	95.76	95.76
50	compact_vs_grouped_materialized	ok	0.00087	66.18	-38.91	-40.83	-47.32	89.78	89.78	89.78
100	compact_vs_client_expanded_negative_control	ok	0.00000	159.93	-60.56	-64.21	-65.58	95.76	95.76	95.76
100	compact_vs_grouped_materialized	ok	0.00087	76.61	-41.84	-48.27	-52.69	89.78	89.78	89.78
200	compact_vs_client_expanded_negative_control	ok	0.00000	185.02	-64.11	-66.68	-67.61	95.76	95.76	95.76
200	compact_vs_grouped_materialized	ok	0.00087	95.06	-47.21	-52.63	-55.02	89.78	89.78	89.78
400	compact_vs_client_expanded_negative_control	comparison_unavailable_due_to_error	NA	NA	NA	NA	NA	NA	NA	NA
400	compact_vs_grouped_materialized	ok	0.00087	170.27	-62.23	-68.29	-68.96	89.78	89.78	89.78
```

- 来源：`results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_execution_mode_latency_deltas.csv`
- 边界/注意：这是端到端执行模式证据，不是内部函数级 profiler。

### 图 E-02: JSON request/response body bytes
- 支撑论据：compact_current 显著减少请求 body、响应 body 和 total body bytes per query，是 Claim E “compact request”字节开销论据。
- 推荐图形：柱形图；X 轴 variant，数据表 1 画 request payload bytes/query，数据表 2 画 request/response/total body bytes/query。
- Excel 绘图设置：X 轴=`variant` 或 `batch_size + variant`；建议系列=`request bytes/query, response bytes/query, total body bytes/query, reduction vs baseline`。百分比列按数值绘制，列名中的 `(%)` 表示百分比值。

数据表 E-02a：request payload summary，覆盖 batch_size 50/100/200/400。

TSV 数据区（复制代码块内部，从下一行表头开始）：
```tsv
variant	batch_size	search requests	search req/query	request payload bytes/query	request payload bytes/batch mean	request payload降低vs client-expanded(%)
client_shard_major_expanded	50	70832	23.61	104064.30	5203215.25	0.00
compact_current	50	3000	1.00	7040.93	352046.47	93.23
grouped_by_ef_materialized	50	29294	9.76	42375.29	2118764.67	59.28
client_shard_major_expanded	100	70832	23.61	104064.16	10406416.50	0.00
compact_current	100	3000	1.00	7040.79	704078.93	93.23
grouped_by_ef_materialized	100	29294	9.76	42375.15	4237515.33	59.28
client_shard_major_expanded	200	70832	23.61	104064.10	20812819.00	0.00
compact_current	200	3000	1.00	7040.72	1408143.87	93.23
grouped_by_ef_materialized	200	29294	9.76	42375.08	8475016.67	59.28
client_shard_major_expanded	400	70832	23.61	104064.06	39024023.38	0.00
compact_current	400	3000	1.00	7040.69	2640257.50	93.23
grouped_by_ef_materialized	400	29294	9.76	42375.05	15890644.00	59.28
```

数据表 E-02b：request + response wire/body summary，覆盖 batch_size=200。

TSV 数据区（复制代码块内部，从下一行表头开始）：
```tsv
variant	batch_size	Recall@10	QPS	request body bytes/query	response body bytes/query	total body bytes/query	request降低vs client-expanded(%)	response降低vs client-expanded(%)	total降低vs client-expanded(%)	candidate groups/query	returned candidates/query
client_shard_major_expanded	200	0.95550	163.33	103915.48	24598.89	128514.37	0.00	0.00	0.00	23.58	235.76
compact_current	200	0.95550	449.01	7039.09	1042.87	8081.96	93.23	95.76	93.71	1.00	10.00
grouped_by_ef_materialized	200	0.95460	223.00	42375.90	10194.59	52570.50	59.22	58.56	59.09	9.77	97.65
```

- 来源：`results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_payload_bytes_summary.csv`；`results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_wire_bytes_summary.csv`
- 边界/注意：这些是 JSON body bytes，不含 HTTP framing、压缩或真实物理网络字节。

### 图 E-03: route planning/materialization time
- 支撑论据：client-side upper label 与 plan materialization 时间在 sub-ms/query 量级；用于解释 compact/grouped/client-expanded 的规划开销。
- 推荐图形：簇状柱形图；variant 为 X 轴，画 total planning ms/query 和 request/candidate counts。
- Excel 绘图设置：X 轴=`variant`；建议系列=`total planning ms/query, search requests/query, candidate groups/query`。百分比列按数值绘制，列名中的 `(%)` 表示百分比点/百分比值。

TSV 数据区（复制代码块内部，从下一行表头开始）：
```tsv
variant	query_count	lower order	upper label ms/query	plan build ms/query	total planning ms/query	search requests/query	candidate groups/query	visited shards	assigned EF-sum
grouped_by_ef_materialized	3000	query_major	0.0147	0.1525	0.1672	9.78	9.78	23.61	3306.75
compact_current	3000	query_major	0.0147	0.2353	0.2500	1.00	1.00	23.61	3306.75
client_shard_major_expanded	3000	shard_major	0.0147	0.2775	0.2922	23.61	23.61	23.61	3306.75
```

- 来源：`results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_planning_time_summary.csv`
- 边界/注意：只覆盖 client-side upper label + plan materialization，不是 Qdrant server 内部规划成本。

### 图 E-04: host interface byte counters
- 支撑论据：在 Docker bridge/veth/loopback 等 host interface 计数上，compact_current 相比 grouped/client-expanded 减少 bytes/query。
- 推荐图形：分组柱形图；role 为 X 轴，按 comparison 分组，画 bytes变化。
- Excel 绘图设置：X 轴=`role + comparison`；建议系列=`bytes变化(%), compact/baseline bytes/query, QPS变化(%)`。百分比列按数值绘制，列名中的 `(%)` 表示百分比点/百分比值。

TSV 数据区（复制代码块内部，从下一行表头开始）：
```tsv
comparison	role	compact bytes/query	baseline bytes/query	bytes变化(%)	compact QPS	baseline QPS	QPS变化(%)	compact search req/query	baseline search req/query
compact_current_vs_grouped_by_ef_materialized	docker_bridge	8604.09	54831.20	-84.31	438.41	219.82	99.44	1.00	9.77
compact_current_vs_grouped_by_ef_materialized	docker_veth	57651.56	161046.08	-64.20	438.41	219.82	99.44	1.00	9.77
compact_current_vs_grouped_by_ef_materialized	loopback	17252.27	109669.39	-84.27	438.41	219.82	99.44	1.00	9.77
compact_current_vs_grouped_by_ef_materialized	physical_nic	336.07	362.69	-7.34	438.41	219.82	99.44	1.00	9.77
compact_current_vs_grouped_by_ef_materialized	virtual_other	0.01	0.03	-78.95	438.41	219.82	99.44	1.00	9.77
compact_current_vs_client_shard_major_expanded	docker_bridge	8604.09	134186.78	-93.59	438.41	154.86	183.09	1.00	23.61
compact_current_vs_client_shard_major_expanded	docker_veth	57651.56	234646.22	-75.43	438.41	154.86	183.09	1.00	23.61
compact_current_vs_client_shard_major_expanded	loopback	17252.27	269724.85	-93.60	438.41	154.86	183.09	1.00	23.61
compact_current_vs_client_shard_major_expanded	physical_nic	336.07	540.87	-37.86	438.41	154.86	183.09	1.00	23.61
compact_current_vs_client_shard_major_expanded	virtual_other	0.01	0.01	-33.33	438.41	154.86	183.09	1.00	23.61
```

- 来源：`results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_host_interface_bytes_deltas.csv`
- 边界/注意：host-interface counter 不是 packet-level physical NIC 证据；physical_nic 行数值很小且需结合 E-06 的零包 negative control。

### 图 E-05: Docker bridge packet capture
- 支撑论据：tcpdump bridge capture 显示 compact_current 减少 frame bytes/query、TCP payload/query 和 packet count/query。
- 推荐图形：柱形图；baseline 为 X 轴，画 frame/TCP payload/packet count 变化。
- Excel 绘图设置：X 轴=`baseline`；建议系列=`frame bytes变化(%), TCP payload变化(%), packets/query变化(%)`。百分比列按数值绘制，列名中的 `(%)` 表示百分比点/百分比值。

TSV 数据区（复制代码块内部，从下一行表头开始）：
```tsv
method	baseline	batch_size	frame bytes/query(method)	frame bytes/query(base)	frame bytes变化(%)	TCP payload/query(method)	TCP payload变化(%)	packets/query变化(%)	QPS变化(%)	P95变化(%)	P99变化(%)
compact_current	grouped_by_ef_materialized	200	51579.46	129598.55	-60.20	51201.33	-56.68	-96.69	104.06	-57.61	-57.84
compact_current	client_shard_major_expanded	200	51579.46	208606.95	-75.27	51201.33	-74.83	-92.69	181.43	-66.81	-67.09
```

- 来源：`results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_packet_capture_bridge_deltas.csv`
- 边界/注意：这是 Docker bridge/local traffic 证据，不是物理网络 payload savings，也不是 subsystem profiler。

### 图 E-06: physical NIC packet capture negative control
- 支撑论据：在单机 Docker 设置下，Claim E 对应窗口没有匹配外部 physical NIC 包；支撑把 E-05 限定为 bridge/veth-local 证据。
- 推荐图形：简单柱形图；interface 为 X 轴，packet_count 和 frame bytes 应为 0。
- Excel 绘图设置：X 轴=`physical interface`；建议系列=`packet_count, frame bytes, tcp payload bytes`。百分比列按数值绘制，列名中的 `(%)` 表示百分比点/百分比值。

TSV 数据区（复制代码块内部，从下一行表头开始）：
```tsv
variant	physical interface	query_count	packet_count	frame bytes	tcp payload bytes	tcpdump dropped	capture scope
all_claim_e_execution_modes	ens111f0np0	27000	0	0	0	0	physical_nic_packet_capture
all_claim_e_execution_modes	ens6f0np0	27000	0	0	0	0	physical_nic_packet_capture
```

- 来源：`results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_packet_capture_physical_nic_totals.csv`
- 边界/注意：这是 negative control，不是正向物理网络字节节省指标。

### 图 E-07: Qdrant /metrics supplement
- 支撑论据：Prometheus /metrics window 显示 compact_current 相比 grouped/client-expanded 有更高 QPS、更低 batch latency、REST duration delta 和 minor page faults。
- 推荐图形：柱形图；comparison 为 X 轴，画 QPS变化、P95/P99变化、REST duration变化。
- Excel 绘图设置：X 轴=`comparison`；建议系列=`QPS变化(%), P95/P99变化(%), REST duration变化(%), minor faults变化(%)`。百分比列按数值绘制，列名中的 `(%)` 表示百分比点/百分比值。

TSV 数据区（复制代码块内部，从下一行表头开始）：
```tsv
comparison	Recall compact	Recall baseline	QPS compact	QPS baseline	QPS变化(%)	P95变化(%)	P99变化(%)	REST duration变化(%)	minor faults变化(%)
compact_current_vs_grouped_by_ef_materialized	0.95527	0.95427	386.62	201.00	92.34	-52.15	-52.29	-32.49	-92.13
compact_current_vs_client_shard_major_expanded	0.95527	0.95527	386.62	138.47	179.21	-66.33	-66.21	-26.18	-93.50
```

- 来源：`results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_qdrant_metrics_deltas.csv`
- 边界/注意：这是 exposed/process-level counter 证据，不是 Qdrant 内部 subsystem/function-level attribution。

### 图 E-08: controller process attribution
- 支撑论据：host /proc 级别的 controller process attribution 显示 compact_current 在 batch=200 窗口降低 controller CPU/RSS 等资源指标。
- 推荐图形：条形图；metric 为 X 轴，按 comparison 分组，画变化(%)。
- Excel 绘图设置：X 轴=`metric + comparison`；建议系列=`compact/baseline value, 变化(%)`。百分比列按数值绘制，列名中的 `(%)` 表示百分比点/百分比值。

TSV 数据区（复制代码块内部，从下一行表头开始）：
```tsv
comparison	role	metric	compact value	baseline value	变化(%)
compact_current_vs_grouped_by_ef_materialized	controller	process_cpu_total_delta_s	13.790	54.120	-74.52
compact_current_vs_grouped_by_ef_materialized	controller	process_cpu_percent_one_core	14.650	46.848	-68.73
compact_current_vs_grouped_by_ef_materialized	controller	process_cpu_ms_per_measured_query_window	1.532	6.013	-74.52
compact_current_vs_client_shard_major_expanded	controller	process_cpu_total_delta_s	13.790	44.850	-69.25
compact_current_vs_client_shard_major_expanded	controller	process_cpu_percent_one_core	14.650	33.972	-56.88
compact_current_vs_client_shard_major_expanded	controller	process_cpu_ms_per_measured_query_window	1.532	4.983	-69.25
```

- 来源：`results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_controller_process_attribution_deltas.csv`
- 边界/注意：这是进程级窗口归因，包含 setup/warmup/measured repeats/client orchestration；不要写成 server-internal subsystem profiling。

### 图 E-09: build-stage time/RSS smoke instrumentation
- 支撑论据：fresh-build smoke harness 证明 build-stage time/RSS instrumentation 路径可运行并记录 wall time / max RSS。
- 推荐图形：单行指标卡或柱形图；画 wall seconds 和 max RSS MiB。
- Excel 绘图设置：X 轴=`run label`；建议系列=`wall seconds, max RSS MiB, indexed vectors`。百分比列按数值绘制，列名中的 `(%)` 表示百分比点/百分比值。

TSV 数据区（复制代码块内部，从下一行表头开始）：
```tsv
run label	train_limit	logical points	assigned points	indexed vectors	active shards	eval queries	Recall@10	QPS	wall seconds	max RSS MiB	exit status
claim_e_build_smoke_20260707_085303	1000	1000	1381	1376	7	20	0.00000	711.97	37.82	64.203	0
```

- 来源：`results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_build_stage_instrumented_smoke_20260707.csv`
- 边界/注意：smoke-only，不是 full-size Method4 distributed build cost。

## Claim F：worker-local premerge 降低 fan-in / candidate stream 压力

### 图 F-01: worker-local premerge fan-in mechanism
- 支撑论据：direct-peer local premerge simulation 把 candidate groups / returned candidates per query 大幅降到 peer-local 数量级。
- 推荐图形：簇状柱形图；标签为 X 轴，画 candidate groups/query 和 returned candidates/query。
- Excel 绘图设置：X 轴=`标签`；建议系列=`candidate groups/query, returned candidates/query, QPS, Recall@10`。百分比列按数值绘制，列名中的 `(%)` 表示百分比点/百分比值。

TSV 数据区（复制代码块内部，从下一行表头开始）：
```tsv
标签	Recall@10	QPS	visited	search req/query	candidate groups/query	returned candidates/query
direct-peer logical-shard fan-in baseline	0.95557	154.30	23.21	23.21	23.21	232.08
direct-peer peer-local pre-merge simulation	0.95543	151.97	23.23	23.23	2.98	29.77
server peer pre-merge enabled fresh	0.95530	405.38	23.19	1.00	1.00	10.00
server peer pre-merge disabled fresh	0.95523	379.42	23.21	1.00	1.00	10.00
server peer pre-merge retained repeated	0.95517	423.15	23.23	1.00	1.00	10.00
```

- 来源：`results/method4_claim_coverage_20260704/derived_claim_tables/claim_f_worker_local_premerge.csv`
- 边界/注意：机制证据；direct-peer simulation 不能直接替代生产 enabled/disabled P95/P99 A/B。

### 图 F-02: premerge batch latency matrix
- 支撑论据：3 variants x 4 batch sizes x 3 repeats 矩阵显示 local premerge 保持 recall 并降低 candidate group/returned candidate 压力。
- 推荐图形：分组柱形图；batch_size 为 X 轴，每个 comparison 分组，画 candidate reduction 与 P95/P99 delta。
- Excel 绘图设置：X 轴=`batch_size + 对比`；建议系列=`candidate groups降低(%), returned candidates降低(%), P95/P99变化(%)`。百分比列按数值绘制，列名中的 `(%)` 表示百分比点/百分比值。

TSV 数据区（复制代码块内部，从下一行表头开始）：
```tsv
对比	batch_size	Recall差	QPS变化(%)	P50变化(%)	P95变化(%)	P99变化(%)	candidate groups降低(%)	returned candidates降低(%)
coordinator_current_vs_direct_peer_no_premerge	50	0.00000	112.70	-51.73	-64.16	-64.94	0.00	0.00
coordinator_current_vs_direct_peer_no_premerge	100	0.00000	149.48	-58.86	-65.09	-65.32	0.00	0.00
coordinator_current_vs_direct_peer_no_premerge	200	0.00000	164.37	-61.42	-66.08	-66.61	0.00	0.00
coordinator_current_vs_direct_peer_no_premerge	400	0.00000	171.64	-63.26	-64.33	-64.09	0.00	0.00
direct_peer_local_premerge_vs_no_premerge	50	0.00000	0.30	-0.18	-18.44	0.60	87.40	87.40
direct_peer_local_premerge_vs_no_premerge	100	0.00000	-0.93	1.56	-0.76	1.06	87.40	87.40
direct_peer_local_premerge_vs_no_premerge	200	0.00000	1.05	-1.67	-2.99	-1.47	87.40	87.40
direct_peer_local_premerge_vs_no_premerge	400	0.00000	-2.00	2.00	2.46	2.83	87.40	87.40
```

- 来源：`results/method4_claim_coverage_20260704/derived_claim_tables/claim_f_premerge_batch_latency_deltas.csv`
- 边界/注意：直接 peer simulation 与 current coordinator path 混合；用于 fan-in/latency-shape，不用于强生产 tail-latency enabled/disabled 声称。

## Claim G：Method4-aware physical placement 带来 modest online/placement gains

### 图 G-01: offline placement load balance
- 支撑论据：Method4-aware placement 相对 round_robin 降低 per-query max peer load，改善负载均衡。
- 推荐图形：折线/柱形图；worker_count 为 X 轴，placement 为系列，画 avg/P95 max peer load 改善。
- Excel 绘图设置：X 轴=`worker_count`；建议系列=`avg/P95 max peer load改善(%), peer load CV改善(%)`。百分比列按数值绘制，列名中的 `(%)` 表示百分比点/百分比值。

TSV 数据区（复制代码块内部，从下一行表头开始）：
```tsv
worker_count	placement	baseline	avg max peer load改善(%)	P95 max peer load改善(%)	max peer load改善(%)	peer load CV改善(%)	cross-worker pair delta pp
2	method4_aware	round_robin	6.45	6.21	2.31	98.39	0.77
2	size_balanced	round_robin	3.52	3.59	0.58	78.36	0.58
3	method4_aware	round_robin	3.47	4.87	2.97	75.03	0.71
3	size_balanced	round_robin	-5.23	-4.87	0.37	-8.13	-0.20
4	method4_aware	round_robin	9.46	8.00	1.65	88.46	0.78
4	size_balanced	round_robin	3.77	5.00	3.72	71.98	0.41
```

- 来源：`results/method4_claim_coverage_20260704/derived_claim_tables/claim_g_offline_placement_deltas.csv`
- 边界/注意：这是 offline placement simulation；online modest gains 见 G-02。

### 图 G-02: online batch latency placement A/B
- 支撑论据：Method4-aware placement 在线 batch latency matrix 给出小幅 QPS 和 tail-latency 改善。
- 推荐图形：簇状柱形图；batch_size 为 X 轴，placement 为系列，画 QPS/P95/P99 delta。
- Excel 绘图设置：X 轴=`batch_size + placement`；建议系列=`QPS变化(%), P95/P99变化(%), Recall差`。百分比列按数值绘制，列名中的 `(%)` 表示百分比点/百分比值。

TSV 数据区（复制代码块内部，从下一行表头开始）：
```tsv
batch_size	placement	baseline	Recall差	QPS变化(%)	mean latency变化(%)	P95变化(%)	P99变化(%)	visited变化(%)	EF-sum变化(%)
100	method4_aware	round_robin	-0.00043	2.05	-2.01	-0.43	-0.56	0.00	0.00
100	size_balanced	round_robin	0.00000	-1.27	1.42	0.36	1.11	0.00	0.00
200	method4_aware	round_robin	-0.00043	1.23	-1.22	-0.96	-0.98	0.00	0.00
200	size_balanced	round_robin	0.00000	-0.35	0.36	0.82	0.88	0.00	0.00
```

- 来源：`results/method4_claim_coverage_20260704/derived_claim_tables/claim_g_online_batch_latency_deltas.csv`
- 边界/注意：增益 modest，避免写成大幅或所有 workload 都成立。

### 图 G-03: concurrency=8 supplement
- 支撑论据：高并发补充行显示 Method4-aware 在 concurrency=8 下仍有小幅 QPS/tail 改善。
- 推荐图形：单行柱形图；画 QPS/P95/P99 delta。
- Excel 绘图设置：X 轴=`comparison`；建议系列=`QPS变化(%), P95/P99变化(%), Recall差`。百分比列按数值绘制，列名中的 `(%)` 表示百分比点/百分比值。

TSV 数据区（复制代码块内部，从下一行表头开始）：
```tsv
comparison	concurrency	Recall差	QPS变化(%)	mean latency变化(%)	P95变化(%)	P99变化(%)	visited变化(%)	EF-sum变化(%)
method4_aware_vs_round_robin	8	0.00120	2.30	-2.28	-5.27	-7.91	2.59	1.67
```

- 来源：`results/method4_claim_g_evidence_20260704/online_concurrency8_latency_deltas.csv`
- 边界/注意：补充证据，不是 primary matched-layout proof；deployed collection 不是完全相同 logical clone。

## Claim H：compact_multi_ep 语义保持

### 图 H-01: semantic invariant audit
- 支撑论据：compact_multi_ep wrapper audit 通过 9 个语义 invariant，支撑 compact request 不改变 Method4 预期语义。
- 推荐图形：pass/fail 矩阵或计数柱形图；invariant 为 X 轴，status 为系列。
- Excel 绘图设置：X 轴=`invariant`；建议系列=`status(pass/fail)`。百分比列按数值绘制，列名中的 `(%)` 表示百分比点/百分比值。

TSV 数据区（复制代码块内部，从下一行表头开始）：
```tsv
invariant	status	evidence
upper routing unchanged	pass	100-query compact_multi_ep semantic audit preserves 160 upper labels/query and emits one compact search/query
point_to_shards unchanged	pass	100-query audit reconstructs upper point shard membership from qdrant_controller_idea_method4map_full_20260601 and verifies every routed upper label has shard membership
multi-assignment preserved	pass	100-query audit verifies routed EP copy counts match point_to_shards membership; 99/100 queries include multi-assignment
per-shard entry points preserved	pass	100-query audit verifies compact hnsw_entry_points_by_shard keys and counts match expected routed logical shards
dynamic EF formula preserved	pass	100-query audit verifies hnsw_ef_by_shard equals base_ef + factor * routed_ep_count for every routed shard
lower search remains per logical shard	pass	100-query audit verifies compact shard_key, per-shard entry-point map, and per-shard EF map address identical logical shards
no adaptive shard pruning	pass	100-query audit verifies routed shard set equals the upper-label membership-derived shard set
source-id dedup preserved	pass	100-query audit verifies source_id_dedup_block_size=1183515 and encoded per-shard entry point IDs belong to their shard blocks
final global merge preserved	pass	100-query audit verifies compact search limit remains top_k=10 so final global merge semantics are preserved
```

- 来源：`results/method4_claim_coverage_20260704/derived_claim_tables/claim_h_semantic_invariants_current_evidence.csv`
- 边界/注意：外部 wrapper audit，不是 C++ 内部逐步 trace；如需 implementation-level equivalence 需额外 C++ reference trace。

## 横向证据：worker count、shard scaling、cost/footprint

### 图 X-01: basic Method4-only worker-count scaling
- 支撑论据：GloVe worker_count 1/2/3/4 在目标 0.80/0.85/0.90/0.95 上显示 Method4-only QPS scaling 和 P95/P99 下降。
- 推荐图形：折线图；X 轴 worker_count，每个 target 一条 QPS speedup 线；另画 P95/P99 delta。
- Excel 绘图设置：X 轴=`worker_count`；建议系列=`QPS speedup, P95变化vs1(%), P99变化vs1(%), Recall@10`。百分比列按数值绘制，列名中的 `(%)` 表示百分比点/百分比值。

TSV 数据区（复制代码块内部，从下一行表头开始）：
```tsv
target	worker_count	QPS	worker1 QPS	QPS speedup	QPS变化vs1(%)	P95 ms	P95变化vs1(%)	P99 ms	P99变化vs1(%)	Recall@10
0.080	1	907.28	907.28	1.000	0.00	126.07	0.00	134.55	0.00	0.80587
0.080	2	1009.28	907.28	1.112	11.24	108.19	-14.18	111.21	-17.34	0.80490
0.080	3	1135.91	907.28	1.252	25.20	98.83	-21.61	103.31	-23.22	0.80640
0.080	4	1234.45	907.28	1.361	36.06	91.65	-27.30	93.64	-30.41	0.80650
0.085	1	641.98	641.98	1.000	0.00	184.61	0.00	188.97	0.00	0.86183
0.085	2	786.17	641.98	1.225	22.46	137.99	-25.25	140.53	-25.63	0.86083
0.085	3	954.24	641.98	1.486	48.64	115.76	-37.29	117.74	-37.69	0.86343
0.085	4	1020.40	641.98	1.589	58.95	110.35	-40.22	112.68	-40.37	0.86213
0.090	1	432.76	432.76	1.000	0.00	250.97	0.00	258.37	0.00	0.90427
0.090	2	580.47	432.76	1.341	34.13	183.80	-26.76	187.83	-27.30	0.90423
0.090	3	706.18	432.76	1.632	63.18	152.88	-39.09	154.47	-40.21	0.90460
0.090	4	762.33	432.76	1.762	76.16	144.36	-42.48	148.79	-42.41	0.90550
0.095	1	218.63	218.63	1.000	0.00	488.70	0.00	513.15	0.00	0.95540
0.095	2	316.81	218.63	1.449	44.91	328.24	-32.83	371.26	-27.65	0.95377
0.095	3	382.40	218.63	1.749	74.91	297.39	-39.15	307.41	-40.09	0.95517
0.095	4	412.18	218.63	1.885	88.53	259.16	-46.97	262.64	-48.82	0.95470
```

- 来源：`results/method4_claim_coverage_20260704/derived_claim_tables/worker_count_online_scaling_deltas_20260708.csv`
- 边界/注意：这是 Method4-only worker-count scaling，没有 Naive/Simple KMeans worker-count 控制；只支持 basic scalability wording。

### 图 X-02: shard-count scaling latency supplement
- 支撑论据：31/46 shard-count 对比提供 selected latency scaling 证据，应与 worker-count scaling 分开表述。
- 推荐图形：簇状柱形图；comparison_label 为 X 轴，画 QPS/P95/P99 delta。
- Excel 绘图设置：X 轴=`comparison_label`；建议系列=`QPS变化(%), visited变化(%), P95/P99变化(%)`。百分比列按数值绘制，列名中的 `(%)` 表示百分比点/百分比值。

TSV 数据区（复制代码块内部，从下一行表头开始）：
```tsv
comparison_type	comparison_label	left_case	right_case	left Recall	right Recall	QPS变化(%)	left shards	right shards	visited变化(%)	EF-sum变化(%)	P95变化(%)	P99变化(%)
scale_s46_vs_s31	orion_s46_vs_orion_s31	orion_s46	orion_s31	0.95073	0.95187	-7.08	69	46	22.89	12.47	6.00	8.33
scale_s46_vs_s31	kmeans_s46_vs_kmeans_s31	kmeans_s46	kmeans_s31	0.95573	0.95603	4.20	46	31	24.45	12.24	-3.63	-6.62
scale_s46_vs_s31	naive_s46_vs_naive_s31	naive_s46	naive_s31	0.95290	0.95043	-13.65	46	31	48.39	17.47	21.80	18.92
orion_vs_simple_kmeans_same_base_shards	orion_vs_kmeans_base31	orion_s31	kmeans_s31	0.95187	0.95603	-5.92	46	31	20.21	21.45	8.36	4.75
orion_vs_simple_kmeans_same_base_shards	orion_vs_kmeans_base46	orion_s46	kmeans_s46	0.95073	0.95573	-16.10	69	46	18.71	21.70	19.18	21.53
orion_vs_naive_same_logical_shards	orion_vs_naive_base31	orion_s31	naive_s31	0.95187	0.95043	60.35	46	31	-37.76	4.64	-37.52	-39.80
orion_vs_naive_same_logical_shards	orion_vs_naive_base46	orion_s46	naive_s46	0.95073	0.95290	72.57	69	46	-48.45	0.18	-45.62	-45.16
simple_kmeans_vs_naive_same_logical_shards	kmeans_vs_naive_base31	kmeans_s31	naive_s31	0.95603	0.95043	70.44	31	31	-48.22	-13.84	-42.34	-42.53
simple_kmeans_vs_naive_same_logical_shards	kmeans_vs_naive_base46	kmeans_s46	naive_s46	0.95573	0.95290	105.69	46	46	-56.58	-17.68	-54.37	-54.87
```

- 来源：`results/method4_claim_coverage_20260704/derived_claim_tables/crosscut_shard_scaling_latency_095_deltas.csv`
- 边界/注意：这是 shard-count scaling，不是 physical worker-count scaling；两者不能混写。

### 图 X-03: cost/footprint main collections
- 支撑论据：主 collection 的 indexed vectors、index expansion、live disk footprint 与运行指标，为成本/空间/请求压力 claim 提供基础数据。
- 推荐图形：散点图；X 轴 index expansion 或 storage GiB，Y 轴 QPS/Recall；按 family 着色。
- Excel 绘图设置：X 轴=`family/strategy`；建议系列=`index expansion, storage GiB, Recall@10, QPS, candidate groups/query`。百分比列按数值绘制，列名中的 `(%)` 表示百分比点/百分比值。

TSV 数据区（复制代码块内部，从下一行表头开始）：
```tsv
label	family	strategy	collection	logical points	indexed vectors	index expansion	active shards	storage data shards GiB	Recall@10	QPS	visited shards	candidate groups/query
current_method4_map_095	Orion	current_method4map	qdrant_controller_idea_method4map_full_20260601	1183514	1400967	1.184	46	8.393	0.95527	396.78	23.21	1.00
bench095_orion_default_095	Orion	default_rr_s31	bench095_rr_orion_s31	1183514	1390579	1.175	46	8.455	0.95117	390.71	22.69	1.00
orion_multiassign_default_095	Orion	orion_default	bench_ma_orion_o118_default_s31	1183514	1402095	1.185	46	NA	0.94793	409.06	21.14	1.00
orion_multiassign_w2c2_095	Orion	orion_w2c2	bench_ma_orion_o149_w2c2_s31	1183514	1773921	1.499	54	NA	0.95007	457.98	21.30	1.00
orion_multiassign_w2c3_095	Orion	orion_w2c3	bench_ma_orion_o184_w2c3_s31	1183514	2191458	1.852	56	NA	0.96180	385.22	24.48	1.00
bench095_simple_kmeans_095	Simple KMeans	nprobe_s46	bench095_cpp_kmeans_s46	1183514	1183514	1.000	46	7.945	0.95440	231.20	24.00	1.00
simple_multiassign_a1014_095	Simple KMeans	simple_a1.014	bench_ma_simple_s189_a1014_s46	1183514	2233067	1.887	46	NA	0.95333	219.96	24.00	1.00
bench095_naive_all_shards_095	Naive	all_shards_s46	bench095_rr_naive_s46	1183514	1183514	1.000	46	7.482	0.95283	261.93	46.00	NA
current_naive_all_shards_095	Naive	current_naive46	qdrant_controller_naive46_full_20260521	1183514	1183514	1.000	46	7.482	0.95477	274.36	46.00	NA
```

- 来源：`results/method4_claim_coverage_20260704/derived_claim_tables/crosscut_cost_main_collections.csv`
- 边界/注意：full-size build time/RSS 仍不足；build-stage 只有 E-09 smoke instrumentation。

## 当前未作为主图表纳入的数据

TSV 数据区（复制代码块内部，从下一行表头开始）：
```tsv
Claim	数据组	支撑级别	原始使用建议	本文档处理
CrossCut	reduced_sift100k_l2_generalization	partial_reduced_l2	Use only as reduced L2 external-validity and key-ablation support.	非 GloVe 当前范围；只可作为历史 external-validity supplement
```

此外，`current_environment_preflight`、`goal_completion_audit`、`partition_family_coverage_audit` 等审计表在总索引中保留，用于说明当前范围是否关闭和 caveat，不建议直接画成论文主图。

## 数据源清单

- `results/method4_claim_coverage_20260704/derived_claim_tables/claim_support_data_index.csv`
- `results/method4_claim_coverage_20260704/derived_claim_tables/claim_a_online_orion_vs_simple_kmeans_same_avg_ef.csv`
- `results/method4_claim_coverage_20260704/derived_claim_tables/claim_a_partition_oracle_matrix.csv`
- `results/method4_claim_coverage_20260704/derived_claim_tables/claim_a_partition_online_latency_submatrix_deltas.csv`
- `results/method4_claim_coverage_20260704/derived_claim_tables/claim_a_partition_family_current_rebuilds_20260709.csv`
- `results/method4_claim_coverage_20260704/derived_claim_tables/claim_b_multiassign_expansion_qps.csv`
- `results/method4_claim_coverage_20260704/derived_claim_tables/claim_b_orion_oracle_gt_miss.csv`
- `results/method4_claim_coverage_20260704/derived_claim_tables/strict_multiassign_latency_20260709.csv`
- `results/method4_claim_coverage_20260704/derived_claim_tables/claim_c_dynamic_vs_fixed_performance_summary.csv`
- `results/method4_claim_coverage_20260704/derived_claim_tables/claim_c_dynamic_vs_fixed_deltas.csv`
- `results/method4_claim_coverage_20260704/derived_claim_tables/claim_c_dynamic_vs_fixed_latency_deltas.csv`
- `results/method4_claim_coverage_20260704/derived_claim_tables/claim_c_routed_ep_relevance_summary.csv`
- `results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_method4_vs_naive_same_recall.csv`
- `results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_high_recall_latency_pairs.csv`
- `results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_batch200_high_recall_deltas.csv`
- `results/method4_claim_coverage_20260704/derived_claim_tables/robustness_080_085_method4_vs_naive_deltas.csv`
- `results/method4_claim_coverage_20260704/derived_claim_tables/robustness_090_method4_vs_naive_deltas.csv`
- `results/method4_claim_coverage_20260704/derived_claim_tables/robustness_095_method4_vs_naive_deltas.csv`
- `results/method4_claim_coverage_20260704/derived_claim_tables/robustness_095_curve_config_method4_vs_naive_deltas.csv`
- `results/method4_claim_coverage_20260704/derived_claim_tables/robustness_097_closest_method4_vs_naive_deltas.csv`
- `results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_099_retune_deltas.csv`
- `results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_099_strategy_search_deltas.csv`
- `results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_099_broader_strategy_formal_deltas.csv`
- `results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_099_alternate_width_formal_deltas.csv`
- `results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_099_boundary_query_order_sensitivity.csv`
- `results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_099_boundary_warmup_sensitivity.csv`
- `results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_execution_mode_latency_deltas.csv`
- `results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_payload_bytes_summary.csv`
- `results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_wire_bytes_summary.csv`
- `results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_planning_time_summary.csv`
- `results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_host_interface_bytes_deltas.csv`
- `results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_packet_capture_bridge_deltas.csv`
- `results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_packet_capture_physical_nic_totals.csv`
- `results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_qdrant_metrics_deltas.csv`
- `results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_controller_process_attribution_deltas.csv`
- `results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_build_stage_instrumented_smoke_20260707.csv`
- `results/method4_claim_coverage_20260704/derived_claim_tables/claim_f_worker_local_premerge.csv`
- `results/method4_claim_coverage_20260704/derived_claim_tables/claim_f_premerge_batch_latency_deltas.csv`
- `results/method4_claim_coverage_20260704/derived_claim_tables/claim_g_offline_placement_deltas.csv`
- `results/method4_claim_coverage_20260704/derived_claim_tables/claim_g_online_batch_latency_deltas.csv`
- `results/method4_claim_coverage_20260704/derived_claim_tables/claim_h_semantic_invariants_current_evidence.csv`
- `results/method4_claim_coverage_20260704/derived_claim_tables/worker_count_online_scaling_deltas_20260708.csv`
- `results/method4_claim_coverage_20260704/derived_claim_tables/crosscut_shard_scaling_latency_095_deltas.csv`
- `results/method4_claim_coverage_20260704/derived_claim_tables/crosscut_cost_main_collections.csv`
- `results/method4_claim_g_evidence_20260704/online_concurrency8_latency_deltas.csv`
- `results/method4_claim_coverage_20260704/claim_evidence_audit.csv`
- `results/method4_claim_coverage_20260704/remaining_experiment_gaps.csv`

## Workbook 自解释附录

这个附录让 Excel 工作簿不仅能画图，也能在同一个文件里查看每张图的来源、边界、provenance 路径和默认实验配置。S-01/S-02/S-03/S-04 是自解释/审计 sheet，不建议作为论文主图。

### 表 S-01: 图表来源/边界/Provenance 自解释索引
- 支撑论据：把每个图表数据块的 claim、推荐图形、来源路径和 caveat 放入 Excel sheet，避免 xlsx 只能画图但缺少说明文字。
- 推荐图形：不建议画图；用于筛选和审阅。
- Excel 绘图设置：X 轴=NA；建议系列=NA。

TSV 数据区（复制代码块内部，从下一行表头开始）：
```tsv
chart_id	table_title	section	supports_claim	recommended_chart	excel_plot_settings	source_paths_or_files	caveat_or_boundary	tsv_data_blocks_in_section
A-01	Orion vs Simple KMeans vs Naive：相近召回目标下的在线效率	Claim A：拓扑/分区感知路由能提升局部性和在线效率	在 0.80/0.85/0.90/0.95 目标附近，Orion、Simple KMeans 和 Naive hash all-shards 三种分片/路由方式可以同表比较；Naive 固定访问全部 46 个 lower shards，作为全分片搜索基线。	按目标召回分组的簇状柱形图；QPS 与 visited shards 可分成两张图，Recall@10 用折线标注。	X 轴=目标召回 + 方法；建议系列=Recall@10, QPS, 平均访问shards, EF-sum/query。百分比列按数值绘制，列名中的 (%) 表示百分比点/百分比值。	results/method4_claim_coverage_20260704/derived_claim_tables/claim_a_route_family_three_way_online_targets_20260709.csv；原 Orion/Simple 来源 results/method4_claim_coverage_20260704/derived_claim_tables/claim_a_online_orion_vs_simple_kmeans_same_avg_ef.csv；Naive 来源 results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_method4_vs_naive_same_recall.csv	Naive 行是 hash 分散、全 46 分片搜索的基线；A-01 现在用于三种 route-family 的 target-neighborhood 在线效率图，绘图时保留 Recall@10。	1
A-02	分区族 oracle 局部性矩阵	Claim A：拓扑/分区感知路由能提升局部性和在线效率	Topology convergence 相比 balanced KMeans-only 降低 edge cut、GT-shard entropy 和 routed shard count；random/naive 用作弱局部性对照。	折线图或簇状柱形图；每个分区族一条线，分别画 routed shards、GT miss、entropy、edge cut。	X 轴=upper_k；建议系列=平均routed shards, oracle GT miss@10, GT shard entropy, topology edge cut。百分比列按数值绘制，列名中的 (%) 表示百分比点/百分比值。	results/method4_claim_coverage_20260704/derived_claim_tables/claim_a_partition_oracle_matrix.csv	只支持拓扑收敛的机制论述；当前 full-fission 不支持“全面优于 balanced KMeans-only”的强 oracle 论述。	1
A-03	分区族在线 latency 子矩阵 delta	Claim A：拓扑/分区感知路由能提升局部性和在线效率	已有 full-fission/balanced-KMeans 行和当前补充行在 0.95 附近相对 naive 降低 tail latency；topology/no-fission 显示“更少 visited 但不一定更快”的 tradeoff。	以对比项为 X 轴的柱形图；QPS delta、visited delta、P95/P99 delta 分面绘制。	X 轴=对比 + batch_size；建议系列=QPS变化(%), visited变化(%), P95变化(%), P99变化(%)。百分比列按数值绘制，列名中的 (%) 表示百分比点/百分比值。	results/method4_claim_coverage_20260704/derived_claim_tables/claim_a_partition_online_latency_submatrix_deltas.csv	部分行来自 2026-07-09 current-harness rebuild，不是 2026-07-04 oracle artifact 的逐字节 replay；negative/折中行要保留。	1
A-04	2026-07-09 当前 harness 分区族补充行	Claim A：拓扑/分区感知路由能提升局部性和在线效率	补齐 random_balanced_46、kmeans_topology_46、kmeans_topology_load_recalibrated_46 三个此前缺失的 Claim A 在线 latency cell。	簇状柱形图或 combo 图；每个分区族画 Method4/Naive QPS 与 P95/P99，Recall 作为标签。	X 轴=分区族；建议系列=Method4/Naive Recall, QPS, visited, P95/P99。百分比列按数值绘制，列名中的 (%) 表示百分比点/百分比值。	results/method4_claim_coverage_20260704/derived_claim_tables/claim_a_partition_family_current_rebuilds_20260709.csv	这些行关闭当前 GloVe 范围的在线-row 缺口，但仍是 current-harness rebuild；load-recalibrated no-fission route map 与 topology route map 等价，仅权重重校准。	1
B-01	Multi-assignment recall-QPS frontier + Naive all-shards baseline	Claim B：Multi-assignment 改善 recall-QPS frontier 并降低 oracle GT miss	在 expansion <= 约 2x 时，Orion default / w2c2 相比 single assignment 提升在线 recall-QPS frontier；补充的 Naive hash all-shards 行给出第三种 route-family 的全分片搜索基线。	折线图；X 轴为目标召回，每个 strategy 一条 QPS 线；另画 Recall@10、visited shards 与 index expansion。	X 轴=目标召回；建议系列=最终Recall, QPS, 平均访问shards, EF-sum/query, index expansion。百分比列按数值绘制，列名中的 (%) 表示百分比点/百分比值。	results/method4_claim_coverage_20260704/derived_claim_tables/claim_b_frontier_with_naive_all_shards_20260709.csv；原 Orion/Simple 来源 results/method4_claim_coverage_20260704/derived_claim_tables/claim_b_multiassign_expansion_qps.csv；Naive 来源 results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_method4_vs_naive_same_recall.csv	历史 frontier 多为 single final-eval run；Naive 行是全 46 分片搜索基线，不是 multi-assignment ablation；严格 latency 补充见 B-03。	1
B-02	Multi-assignment oracle GT miss + Naive all-shards reference	Claim B：Multi-assignment 改善 recall-QPS frontier 并降低 oracle GT miss	Orion default 和 w2c2 在 upper_k 80/120/160 上相对 single assignment 降低 oracle_gt_miss@10；补充的 Naive all-shards 行给出“搜索全部分片时 oracle miss 为 0”的全分片参考基线。	折线图；X 轴 upper_k，每个 strategy 一条 miss@10 线；coverage 可单独画，Naive 行可用虚线或灰色 reference 标记。	X 轴=upper_k；建议系列=oracle GT miss@10, oracle GT coverage@10, index expansion。百分比列按数值绘制，列名中的 (%) 表示百分比点/百分比值。	results/method4_claim_coverage_20260704/derived_claim_tables/claim_b_orion_oracle_gt_miss_with_naive_reference_20260709.csv；原 Orion 来源 results/method4_claim_coverage_20260704/derived_claim_tables/claim_b_orion_oracle_gt_miss.csv；Naive 参考来源 results/method4_claim_coverage_20260704/derived_claim_tables/claim_a_partition_oracle_matrix.csv	Naive all-shards 行是全 46 分片搜索的 reference baseline，不是 multi-assignment 策略；它用于展示搜索全部分片的 oracle 上限和 routed-shard 代价。	1
B-03	严格 multi-assignment selected latency cells	Claim B：Multi-assignment 改善 recall-QPS frontier 并降低 oracle GT miss	在 0.95 目标邻域，default/w2c2 的 selected full-size latency cell 显示更高 QPS、更低 EF-sum 和更低 tail latency。	簇状柱形图；策略为 X 轴，QPS 和 P95/P99 分图；Recall 作为数据标签。	X 轴=策略；建议系列=Recall@10, QPS, visited, EF-sum, P95/P99, index expansion。百分比列按数值绘制，列名中的 (%) 表示百分比点/百分比值。	results/method4_claim_coverage_20260704/derived_claim_tables/strict_multiassign_latency_20260709.csv	这是 target-neighborhood latency evidence：single 高于 0.95，w2c2 略低于 0.95；不要写成严格 same-recall 全矩阵证明。	1
C-01	Dynamic EF vs Fixed EF：same/near-same recall 性能	Claim C：Dynamic EF 在相同/近似召回下提升预算效率和 batch tail latency	Dynamic EF 在三个 same/near-same recall 对比中提高 QPS 约 27-31%，并降低 EF-sum/query 约 35-41%。	分组柱形图；数据表 1 画 QPS/EF-sum，数据表 2 画 QPS变化和 EF-sum变化。	X 轴=对比 + 方法；建议系列=Recall@10, QPS, visited, EF-sum/query。百分比列按数值绘制，列名中的 (%) 表示百分比点/百分比值。	results/method4_claim_coverage_20260704/derived_claim_tables/claim_c_dynamic_vs_fixed_performance_summary.csv；results/method4_claim_coverage_20260704/derived_claim_tables/claim_c_dynamic_vs_fixed_deltas.csv	这里的重点是 EF 预算效率，不要声称 Dynamic EF 总是减少 visited shards；frontier_097 的 visited 反而增加。	2
C-02	Dynamic EF batch latency supplement	Claim C：Dynamic EF 在相同/近似召回下提升预算效率和 batch tail latency	在 Recall@10 基本相同的 batch endpoint 补充实验中，Dynamic EF 提升 QPS、降低 EF-sum，并降低 P95/P99 batch latency。	单行 waterfall/柱形图；绘制 QPS变化、EF-sum变化、P95/P99变化。	X 轴=对比；建议系列=QPS变化(%), EF-sum变化(%), P95变化(%), P99变化(%)。百分比列按数值绘制，列名中的 (%) 表示百分比点/百分比值。	results/method4_claim_coverage_20260704/derived_claim_tables/claim_c_dynamic_vs_fixed_latency_deltas.csv	这是 client-observed batch endpoint wall time，不是 server-internal lower-search trace。	1
C-03	routed EP count 相关性/机制证据	Claim C：Dynamic EF 在相同/近似召回下提升预算效率和 batch tail latency	GT-hit shards 上的 routed EP count 高于非 hit shards，支持用 routed EP count 作为 Dynamic EF 预算分配的相关性代理。	柱形图；每个配置画 GT-hit vs non-hit EP count，也可画 budget share。	X 轴=配置；建议系列=GT-hit EP count, non-hit EP count, Dynamic/Fixed GT budget share。百分比列按数值绘制，列名中的 (%) 表示百分比点/百分比值。	results/method4_claim_coverage_20260704/derived_claim_tables/claim_c_routed_ep_relevance_summary.csv	这是 relevance proxy 机制证据，不是独立的在线性能 A/B。	1
D-01	Method4 vs Naive：0.95 same-recall 主对比	Claim D：Method4 相对 Naive 在当前 0.80-0.95 范围和约 0.97 边界内提升效率	在 0.955 附近，Method4 与 Naive Recall@10 接近，同时 QPS 更高并访问更少 shards。	簇状柱形图；X 轴为方法/配置，画 QPS、visited；Recall 用折线或标签。	X 轴=标签；建议系列=Recall@10, QPS, visited, EF-sum/query。百分比列按数值绘制，列名中的 (%) 表示百分比点/百分比值。	results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_method4_vs_naive_same_recall.csv	这是 0.95 主支撑行；不同 batch_size/控制行不要混成不标注的平均值。	1
D-02	Method4 vs Naive：约 0.97 正向行与 0.99 边界行	Claim D：Method4 相对 Naive 在当前 0.80-0.95 范围和约 0.97 边界内提升效率	约 0.97 strict/neighbor 行支持 high-recall 范围内 QPS、visited、tail latency 优势；0.99 行只作为边界/负结果。	柱形图；目标为 X 轴，画 delta 指标；0.99 行用不同颜色标为 boundary。	X 轴=目标；建议系列=QPS变化(%), visited变化(%), P95/P99变化(%), Recall差。百分比列按数值绘制，列名中的 (%) 表示百分比点/百分比值。	results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_high_recall_latency_pairs.csv	不要把 0.99 neighborhood 写成正向 dominance；Method4 recall 低于 Naive 且原始 0.99 行 QPS/tail 不占优。	1
D-03	Method4 vs Naive：batch=200 high-recall supplement	Claim D：Method4 相对 Naive 在当前 0.80-0.95 范围和约 0.97 边界内提升效率	batch=200 形态下约 0.97 行给出强正向 same-batch 证据；约 0.99 行仍是边界证据。	两组柱形图；按目标画 QPS/visited/P95/P99 delta；Naive CV 作为误差/脚注。	X 轴=目标；建议系列=QPS变化(%), visited变化(%), P95/P99变化(%), Naive CV。百分比列按数值绘制，列名中的 (%) 表示百分比点/百分比值。	results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_batch200_high_recall_deltas.csv	Naive batch=200 控制行 tail/QPS 方差较高；0.99 行 Method4 仍低于 0.99 且低于 paired naive recall。	1
D-04	Method4 vs Naive：selected robustness overlays	Claim D：Method4 相对 Naive 在当前 0.80-0.95 范围和约 0.97 边界内提升效率	在选定 0.80/0.85/0.90/0.955/0.95-curve/0.97-neighbor 场景中，Method4 相对 same-run Naive 保持 QPS/visited/tail latency 优势。	热力图或分组柱形图；X 轴为场景，颜色按目标组，画 QPS/P95/P99 delta。	X 轴=目标组 + 场景；建议系列=Recall@10, QPS变化(%), visited变化(%), P95/P99变化(%)。百分比列按数值绘制，列名中的 (%) 表示百分比点/百分比值。	results/method4_claim_coverage_20260704/derived_claim_tables/robustness_080_085_method4_vs_naive_deltas.csv；results/method4_claim_coverage_20260704/derived_claim_tables/robustness_090_method4_vs_naive_deltas.csv；results/method4_claim_coverage_20260704/derived_claim_tables/robustness_095_method4_vs_naive_deltas.csv；results/method4_claim_coverage_20260704/derived_claim_tables/robustness_095_curve_config_method4_vs_naive_deltas.csv；results/method4_claim_coverage_20260704/derived_claim_tables/robustness_097_closest_method4_vs_naive_deltas.csv	这些是 selected overlays，不是完整 recall/config 矩阵；部分行是 target-neighborhood 而非严格 same-recall。	1
D-05	0.99 retune/strategy search：负结果边界数据	Claim D：Method4 相对 Naive 在当前 0.80-0.95 范围和约 0.97 边界内提升效率	多个 0.99 retune 和更宽 upper-routing 候选在 500-query screening 后未能在 formal 3000-query 窗口保持 Method4 Recall@10 0.99；支撑“0.99 dominance 不成立/不纳入当前 scope”的边界论据。	散点图；X 轴为 Method4 Recall，Y 轴 QPS变化或 P95变化；用 0.99 竖线标出阈值。	X 轴=Method4 Recall；建议系列=Naive Recall, QPS变化(%), P95/P99变化(%)。百分比列按数值绘制，列名中的 (%) 表示百分比点/百分比值。	results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_099_retune_deltas.csv；results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_099_strategy_search_deltas.csv；results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_099_broader_strategy_formal_deltas.csv；results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_099_alternate_width_formal_deltas.csv	这是透明负结果，不是正向 claim；只用于限定 Claim D 的适用范围。	1
D-06	0.99 boundary：query-order / warmup sensitivity	Claim D：Method4 相对 Naive 在当前 0.80-0.95 范围和约 0.97 边界内提升效率	0.99 边界行在 query order 和 cold/no-warmup 下仍未关闭 same-recall gap；支撑把 0.99 作为 boundary 而非主 claim。	柱形图；X 轴为敏感性场景，画 Recall变化、QPS变化、P95/P99变化。	X 轴=敏感性类型 + comparison；建议系列=Recall变化, QPS变化(%), P95/P99变化(%)。百分比列按数值绘制，列名中的 (%) 表示百分比点/百分比值。	results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_099_boundary_query_order_sensitivity.csv；results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_099_boundary_warmup_sensitivity.csv	只用于边界和敏感性说明；不要与 0.97 正向行混成平均。	1
E-01	compact_current vs materialized/client-expanded：执行模式 latency delta	Claim E：compact request / execution mode 降低请求对象、字节和运行时开销	compact_current 在多个 batch size 上减少 search request/candidate group，并提升 QPS、降低 P95/P99。	按 batch_size 分组的柱形图；分别画 QPS变化、P95/P99变化、request/candidate 降低。	X 轴=batch_size + 对比；建议系列=QPS变化(%), P95/P99变化(%), search requests降低(%), candidate groups降低(%)。百分比列按数值绘制，列名中的 (%) 表示百分比点/百分比值。	results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_execution_mode_latency_deltas.csv	这是端到端执行模式证据，不是内部函数级 profiler。	1
E-02	JSON request/response body bytes	Claim E：compact request / execution mode 降低请求对象、字节和运行时开销	compact_current 显著减少请求 body、响应 body 和 total body bytes per query，是 Claim E “compact request”字节开销论据。	柱形图；X 轴 variant，数据表 1 画 request payload bytes/query，数据表 2 画 request/response/total body bytes/query。	X 轴=variant 或 batch_size + variant；建议系列=request bytes/query, response bytes/query, total body bytes/query, reduction vs baseline。百分比列按数值绘制，列名中的 (%) 表示百分比值。	results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_payload_bytes_summary.csv；results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_wire_bytes_summary.csv	这些是 JSON body bytes，不含 HTTP framing、压缩或真实物理网络字节。	2
E-03	route planning/materialization time	Claim E：compact request / execution mode 降低请求对象、字节和运行时开销	client-side upper label 与 plan materialization 时间在 sub-ms/query 量级；用于解释 compact/grouped/client-expanded 的规划开销。	簇状柱形图；variant 为 X 轴，画 total planning ms/query 和 request/candidate counts。	X 轴=variant；建议系列=total planning ms/query, search requests/query, candidate groups/query。百分比列按数值绘制，列名中的 (%) 表示百分比点/百分比值。	results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_planning_time_summary.csv	只覆盖 client-side upper label + plan materialization，不是 Qdrant server 内部规划成本。	1
E-04	host interface byte counters	Claim E：compact request / execution mode 降低请求对象、字节和运行时开销	在 Docker bridge/veth/loopback 等 host interface 计数上，compact_current 相比 grouped/client-expanded 减少 bytes/query。	分组柱形图；role 为 X 轴，按 comparison 分组，画 bytes变化。	X 轴=role + comparison；建议系列=bytes变化(%), compact/baseline bytes/query, QPS变化(%)。百分比列按数值绘制，列名中的 (%) 表示百分比点/百分比值。	results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_host_interface_bytes_deltas.csv	host-interface counter 不是 packet-level physical NIC 证据；physical_nic 行数值很小且需结合 E-06 的零包 negative control。	1
E-05	Docker bridge packet capture	Claim E：compact request / execution mode 降低请求对象、字节和运行时开销	tcpdump bridge capture 显示 compact_current 减少 frame bytes/query、TCP payload/query 和 packet count/query。	柱形图；baseline 为 X 轴，画 frame/TCP payload/packet count 变化。	X 轴=baseline；建议系列=frame bytes变化(%), TCP payload变化(%), packets/query变化(%)。百分比列按数值绘制，列名中的 (%) 表示百分比点/百分比值。	results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_packet_capture_bridge_deltas.csv	这是 Docker bridge/local traffic 证据，不是物理网络 payload savings，也不是 subsystem profiler。	1
E-06	physical NIC packet capture negative control	Claim E：compact request / execution mode 降低请求对象、字节和运行时开销	在单机 Docker 设置下，Claim E 对应窗口没有匹配外部 physical NIC 包；支撑把 E-05 限定为 bridge/veth-local 证据。	简单柱形图；interface 为 X 轴，packet_count 和 frame bytes 应为 0。	X 轴=physical interface；建议系列=packet_count, frame bytes, tcp payload bytes。百分比列按数值绘制，列名中的 (%) 表示百分比点/百分比值。	results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_packet_capture_physical_nic_totals.csv	这是 negative control，不是正向物理网络字节节省指标。	1
E-07	Qdrant /metrics supplement	Claim E：compact request / execution mode 降低请求对象、字节和运行时开销	Prometheus /metrics window 显示 compact_current 相比 grouped/client-expanded 有更高 QPS、更低 batch latency、REST duration delta 和 minor page faults。	柱形图；comparison 为 X 轴，画 QPS变化、P95/P99变化、REST duration变化。	X 轴=comparison；建议系列=QPS变化(%), P95/P99变化(%), REST duration变化(%), minor faults变化(%)。百分比列按数值绘制，列名中的 (%) 表示百分比点/百分比值。	results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_qdrant_metrics_deltas.csv	这是 exposed/process-level counter 证据，不是 Qdrant 内部 subsystem/function-level attribution。	1
E-08	controller process attribution	Claim E：compact request / execution mode 降低请求对象、字节和运行时开销	host /proc 级别的 controller process attribution 显示 compact_current 在 batch=200 窗口降低 controller CPU/RSS 等资源指标。	条形图；metric 为 X 轴，按 comparison 分组，画变化(%)。	X 轴=metric + comparison；建议系列=compact/baseline value, 变化(%)。百分比列按数值绘制，列名中的 (%) 表示百分比点/百分比值。	results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_controller_process_attribution_deltas.csv	这是进程级窗口归因，包含 setup/warmup/measured repeats/client orchestration；不要写成 server-internal subsystem profiling。	1
E-09	build-stage time/RSS smoke instrumentation	Claim E：compact request / execution mode 降低请求对象、字节和运行时开销	fresh-build smoke harness 证明 build-stage time/RSS instrumentation 路径可运行并记录 wall time / max RSS。	单行指标卡或柱形图；画 wall seconds 和 max RSS MiB。	X 轴=run label；建议系列=wall seconds, max RSS MiB, indexed vectors。百分比列按数值绘制，列名中的 (%) 表示百分比点/百分比值。	results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_build_stage_instrumented_smoke_20260707.csv	smoke-only，不是 full-size Method4 distributed build cost。	1
F-01	worker-local premerge fan-in mechanism	Claim F：worker-local premerge 降低 fan-in / candidate stream 压力	direct-peer local premerge simulation 把 candidate groups / returned candidates per query 大幅降到 peer-local 数量级。	簇状柱形图；标签为 X 轴，画 candidate groups/query 和 returned candidates/query。	X 轴=标签；建议系列=candidate groups/query, returned candidates/query, QPS, Recall@10。百分比列按数值绘制，列名中的 (%) 表示百分比点/百分比值。	results/method4_claim_coverage_20260704/derived_claim_tables/claim_f_worker_local_premerge.csv	机制证据；direct-peer simulation 不能直接替代生产 enabled/disabled P95/P99 A/B。	1
F-02	premerge batch latency matrix	Claim F：worker-local premerge 降低 fan-in / candidate stream 压力	3 variants x 4 batch sizes x 3 repeats 矩阵显示 local premerge 保持 recall 并降低 candidate group/returned candidate 压力。	分组柱形图；batch_size 为 X 轴，每个 comparison 分组，画 candidate reduction 与 P95/P99 delta。	X 轴=batch_size + 对比；建议系列=candidate groups降低(%), returned candidates降低(%), P95/P99变化(%)。百分比列按数值绘制，列名中的 (%) 表示百分比点/百分比值。	results/method4_claim_coverage_20260704/derived_claim_tables/claim_f_premerge_batch_latency_deltas.csv	直接 peer simulation 与 current coordinator path 混合；用于 fan-in/latency-shape，不用于强生产 tail-latency enabled/disabled 声称。	1
G-01	offline placement load balance	Claim G：Method4-aware physical placement 带来 modest online/placement gains	Method4-aware placement 相对 round_robin 降低 per-query max peer load，改善负载均衡。	折线/柱形图；worker_count 为 X 轴，placement 为系列，画 avg/P95 max peer load 改善。	X 轴=worker_count；建议系列=avg/P95 max peer load改善(%), peer load CV改善(%)。百分比列按数值绘制，列名中的 (%) 表示百分比点/百分比值。	results/method4_claim_coverage_20260704/derived_claim_tables/claim_g_offline_placement_deltas.csv	这是 offline placement simulation；online modest gains 见 G-02。	1
G-02	online batch latency placement A/B	Claim G：Method4-aware physical placement 带来 modest online/placement gains	Method4-aware placement 在线 batch latency matrix 给出小幅 QPS 和 tail-latency 改善。	簇状柱形图；batch_size 为 X 轴，placement 为系列，画 QPS/P95/P99 delta。	X 轴=batch_size + placement；建议系列=QPS变化(%), P95/P99变化(%), Recall差。百分比列按数值绘制，列名中的 (%) 表示百分比点/百分比值。	results/method4_claim_coverage_20260704/derived_claim_tables/claim_g_online_batch_latency_deltas.csv	增益 modest，避免写成大幅或所有 workload 都成立。	1
G-03	concurrency=8 supplement	Claim G：Method4-aware physical placement 带来 modest online/placement gains	高并发补充行显示 Method4-aware 在 concurrency=8 下仍有小幅 QPS/tail 改善。	单行柱形图；画 QPS/P95/P99 delta。	X 轴=comparison；建议系列=QPS变化(%), P95/P99变化(%), Recall差。百分比列按数值绘制，列名中的 (%) 表示百分比点/百分比值。	results/method4_claim_g_evidence_20260704/online_concurrency8_latency_deltas.csv	补充证据，不是 primary matched-layout proof；deployed collection 不是完全相同 logical clone。	1
H-01	semantic invariant audit	Claim H：compact_multi_ep 语义保持	compact_multi_ep wrapper audit 通过 9 个语义 invariant，支撑 compact request 不改变 Method4 预期语义。	pass/fail 矩阵或计数柱形图；invariant 为 X 轴，status 为系列。	X 轴=invariant；建议系列=status(pass/fail)。百分比列按数值绘制，列名中的 (%) 表示百分比点/百分比值。	results/method4_claim_coverage_20260704/derived_claim_tables/claim_h_semantic_invariants_current_evidence.csv	外部 wrapper audit，不是 C++ 内部逐步 trace；如需 implementation-level equivalence 需额外 C++ reference trace。	1
X-01	basic Method4-only worker-count scaling	横向证据：worker count、shard scaling、cost/footprint	GloVe worker_count 1/2/3/4 在目标 0.80/0.85/0.90/0.95 上显示 Method4-only QPS scaling 和 P95/P99 下降。	折线图；X 轴 worker_count，每个 target 一条 QPS speedup 线；另画 P95/P99 delta。	X 轴=worker_count；建议系列=QPS speedup, P95变化vs1(%), P99变化vs1(%), Recall@10。百分比列按数值绘制，列名中的 (%) 表示百分比点/百分比值。	results/method4_claim_coverage_20260704/derived_claim_tables/worker_count_online_scaling_deltas_20260708.csv	这是 Method4-only worker-count scaling，没有 Naive/Simple KMeans worker-count 控制；只支持 basic scalability wording。	1
X-02	shard-count scaling latency supplement	横向证据：worker count、shard scaling、cost/footprint	31/46 shard-count 对比提供 selected latency scaling 证据，应与 worker-count scaling 分开表述。	簇状柱形图；comparison_label 为 X 轴，画 QPS/P95/P99 delta。	X 轴=comparison_label；建议系列=QPS变化(%), visited变化(%), P95/P99变化(%)。百分比列按数值绘制，列名中的 (%) 表示百分比点/百分比值。	results/method4_claim_coverage_20260704/derived_claim_tables/crosscut_shard_scaling_latency_095_deltas.csv	这是 shard-count scaling，不是 physical worker-count scaling；两者不能混写。	1
X-03	cost/footprint main collections	横向证据：worker count、shard scaling、cost/footprint	主 collection 的 indexed vectors、index expansion、live disk footprint 与运行指标，为成本/空间/请求压力 claim 提供基础数据。	散点图；X 轴 index expansion 或 storage GiB，Y 轴 QPS/Recall；按 family 着色。	X 轴=family/strategy；建议系列=index expansion, storage GiB, Recall@10, QPS, candidate groups/query。百分比列按数值绘制，列名中的 (%) 表示百分比点/百分比值。	results/method4_claim_coverage_20260704/derived_claim_tables/crosscut_cost_main_collections.csv	full-size build time/RSS 仍不足；build-stage 只有 E-09 smoke instrumentation。	2
```

- 来源：`docs/experiments/2026-07-09-method4-claim-support-excel-chart-data.md`
- 边界/注意：这是说明性 provenance 索引，不替代原始 raw CSV/JSON；长文本列用于审阅而不是制图。

### 表 S-02: source_manifest raw/provenance 路径清单
- 支撑论据：把 `source_manifest.json` 中可解析的 repo-local raw/derived/provenance 路径导入 Excel，方便在 workbook 内检查路径、类型和行数/键数。
- 推荐图形：不建议画图；用于筛选和审阅。
- Excel 绘图设置：X 轴=NA；建议系列=NA。

TSV 数据区（复制代码块内部，从下一行表头开始）：
```tsv
manifest_key	artifact_group	path	exists	path_type	size_bytes	rows_keys_or_children
claim_a_kmeans_topology_current_build_dir	claim_a	results/method4_claim_a_partition_family_build_20260709/kmeans_topology_46_oracle_ef400/20260709_092502	yes	dir	NA	5
claim_a_kmeans_topology_current_build_summary_json	claim_a	results/method4_claim_a_partition_family_build_20260709/kmeans_topology_46_oracle_ef400/20260709_092502/summary.json	yes	json	4707	50
claim_a_kmeans_topology_current_builds_csv	claim_a	results/method4_claim_a_partition_family_build_20260709/kmeans_topology_46_oracle_ef400/20260709_092502/builds.csv	yes	csv	322	1
claim_a_kmeans_topology_current_latency_batches_csv	claim_a	results/method4_claim_a_partition_online_latency_20260709/kmeans_topology_46_ef400build/analysis_20260709_093136/claim_d_high_recall_latency_batches.csv	yes	csv	11185	90
claim_a_kmeans_topology_current_latency_dir	claim_a	results/method4_claim_a_partition_online_latency_20260709/kmeans_topology_46_ef400build/analysis_20260709_093136	yes	dir	NA	3
claim_a_kmeans_topology_current_latency_metadata_json	claim_a	results/method4_claim_a_partition_online_latency_20260709/kmeans_topology_46_ef400build/analysis_20260709_093136/run_metadata.json	yes	json	1279	29
claim_a_kmeans_topology_current_latency_summary_csv	claim_a	results/method4_claim_a_partition_online_latency_20260709/kmeans_topology_46_ef400build/analysis_20260709_093136/claim_d_high_recall_latency_summary.csv	yes	csv	1922	6
claim_a_kmeans_topology_load_recalibrated_current_build_dir	claim_a	results/method4_claim_a_partition_family_build_20260709/kmeans_topology_load_recalibrated_46_oracle_ef400/20260709_093432	yes	dir	NA	5
claim_a_kmeans_topology_load_recalibrated_current_build_summary_json	claim_a	results/method4_claim_a_partition_family_build_20260709/kmeans_topology_load_recalibrated_46_oracle_ef400/20260709_093432/summary.json	yes	json	4756	50
claim_a_kmeans_topology_load_recalibrated_current_builds_csv	claim_a	results/method4_claim_a_partition_family_build_20260709/kmeans_topology_load_recalibrated_46_oracle_ef400/20260709_093432/builds.csv	yes	csv	340	1
claim_a_kmeans_topology_load_recalibrated_current_latency_batches_csv	claim_a	results/method4_claim_a_partition_online_latency_20260709/kmeans_topology_load_recalibrated_46_ef400build/analysis_20260709_094451/claim_d_high_recall_latency_batches.csv	yes	csv	12801	90
claim_a_kmeans_topology_load_recalibrated_current_latency_dir	claim_a	results/method4_claim_a_partition_online_latency_20260709/kmeans_topology_load_recalibrated_46_ef400build/analysis_20260709_094451	yes	dir	NA	3
claim_a_kmeans_topology_load_recalibrated_current_latency_metadata_json	claim_a	results/method4_claim_a_partition_online_latency_20260709/kmeans_topology_load_recalibrated_46_ef400build/analysis_20260709_094451/run_metadata.json	yes	json	1333	29
claim_a_kmeans_topology_load_recalibrated_current_latency_summary_csv	claim_a	results/method4_claim_a_partition_online_latency_20260709/kmeans_topology_load_recalibrated_46_ef400build/analysis_20260709_094451/claim_d_high_recall_latency_summary.csv	yes	csv	1992	6
claim_a_partition_family_build_root_20260709	claim_a	results/method4_claim_a_partition_family_build_20260709	yes	dir	NA	5
claim_a_partition_family_current_rebuilds_20260709_csv	claim_a	results/method4_claim_coverage_20260704/derived_claim_tables/claim_a_partition_family_current_rebuilds_20260709.csv	yes	csv	2874	3
claim_a_partition_family_online_coverage_audit_csv	claim_a	results/method4_claim_coverage_20260704/derived_claim_tables/claim_a_partition_family_online_coverage_audit.csv	yes	csv	6128	7
claim_a_partition_online_latency_root	claim_a	results/method4_claim_a_partition_online_latency_20260704	yes	dir	NA	6
claim_a_partition_online_latency_root_20260709	claim_a	results/method4_claim_a_partition_online_latency_20260709	yes	dir	NA	4
claim_a_partition_online_latency_script	claim_a	tools/method4_claim_a_online_latency_summary.py	yes	py	27128	NA
claim_a_partition_online_latency_submatrix_csv	claim_a	results/method4_claim_coverage_20260704/derived_claim_tables/claim_a_partition_online_latency_submatrix.csv	yes	csv	11471	16
claim_a_partition_online_latency_submatrix_deltas_csv	claim_a	results/method4_claim_coverage_20260704/derived_claim_tables/claim_a_partition_online_latency_submatrix_deltas.csv	yes	csv	5229	8
claim_a_partition_oracle_dir	claim_a	results/method4_claim_a_partition_oracle_20260704/analysis_20260704_213047	yes	dir	NA	3
claim_a_partition_oracle_summary_csv	claim_a	results/method4_claim_a_partition_oracle_20260704/analysis_20260704_213047/partition_oracle_summary.csv	yes	csv	4364	18
claim_a_random_balanced_current_build_dir	claim_a	results/method4_claim_a_partition_family_build_20260709/random_balanced_46_oracle_ef400/20260709_091449	yes	dir	NA	5
claim_a_random_balanced_current_build_summary_json	claim_a	results/method4_claim_a_partition_family_build_20260709/random_balanced_46_oracle_ef400/20260709_091449/summary.json	yes	json	4694	50
claim_a_random_balanced_current_builds_csv	claim_a	results/method4_claim_a_partition_family_build_20260709/random_balanced_46_oracle_ef400/20260709_091449/builds.csv	yes	csv	322	1
claim_a_random_balanced_current_latency_batches_csv	claim_a	results/method4_claim_a_partition_online_latency_20260709/random_balanced_46_ef400build/analysis_20260709_092159/claim_d_high_recall_latency_batches.csv	yes	csv	11139	90
claim_a_random_balanced_current_latency_dir	claim_a	results/method4_claim_a_partition_online_latency_20260709/random_balanced_46_ef400build/analysis_20260709_092159	yes	dir	NA	3
claim_a_random_balanced_current_latency_metadata_json	claim_a	results/method4_claim_a_partition_online_latency_20260709/random_balanced_46_ef400build/analysis_20260709_092159/run_metadata.json	yes	json	1277	29
claim_a_random_balanced_current_latency_summary_csv	claim_a	results/method4_claim_a_partition_online_latency_20260709/random_balanced_46_ef400build/analysis_20260709_092159/claim_d_high_recall_latency_summary.csv	yes	csv	1895	6
claim_a_topology_no_fission_build_tune_dir	claim_a	results/strict_ablation_topology_no_fission_20260705/reuse_tune_095/20260705_012313	yes	dir	NA	5
claim_a_topology_no_fission_config_sensitivity_csv	claim_a	results/method4_claim_coverage_20260704/derived_claim_tables/claim_a_topology_no_fission_config_sensitivity.csv	yes	csv	2024	2
claim_a_topology_no_fission_latency_dir	claim_a	results/method4_claim_a_partition_online_latency_20260704/topology_no_fission_selected/analysis_20260705_012559	yes	dir	NA	3
claim_a_topology_no_fission_latency_summary_csv	claim_a	results/method4_claim_a_partition_online_latency_20260704/topology_no_fission_selected/analysis_20260705_012559/claim_d_high_recall_latency_summary.csv	yes	csv	2065	6
claim_b_online_frontier_keypoints_long_csv	claim_b	results/method4_benchmark_matrix/method4_multiassign_recall_qps_excel_20260630/final_eval_keypoints_long.csv	yes	csv	16310	32
claim_b_online_frontier_matrix_summaries.0	0	results/method4_benchmark_matrix/method4_multiassign_expansion_qps_20260629_orion_high_recover/eval3000_20260629/matrix_summary.csv	yes	csv	3285	8
claim_b_online_frontier_matrix_summaries.1	1	results/method4_benchmark_matrix/method4_multiassign_expansion_qps_20260629_orion_low/eval3000_20260629_v2/matrix_summary.csv	yes	csv	1481	8
claim_b_online_frontier_matrix_summaries.2	2	results/method4_benchmark_matrix/method4_multiassign_expansion_qps_20260629_orion_low_recover/eval3000_20260629/matrix_summary.csv	yes	csv	2146	5
claim_b_online_frontier_matrix_summaries.3	3	results/method4_benchmark_matrix/method4_multiassign_expansion_qps_20260629_simple_low/eval3000_20260629/matrix_summary.csv	yes	csv	3235	8
claim_b_online_frontier_matrix_summaries.4	4	results/method4_benchmark_matrix/method4_multiassign_expansion_qps_20260629_simple_s157_only/eval3000_20260629/matrix_summary.csv	yes	csv	1839	4
claim_b_online_frontier_matrix_summaries.5	5	results/method4_benchmark_matrix/method4_multiassign_expansion_qps_20260629_simple_s189_only/eval3000_20260629/matrix_summary.csv	yes	csv	1811	4
claim_b_online_frontier_matrix_summaries.6	6	results/method4_benchmark_matrix/method4_multiassign_recall_qps_current_20260630_orion_single_low/eval3000_20260630/matrix_summary.csv	yes	csv	1075	2
claim_b_online_frontier_quality_flags_csv	claim_b	results/method4_benchmark_matrix/method4_multiassign_recall_qps_excel_20260630/coverage_and_quality_flags.csv	yes	csv	2173	9
claim_b_online_frontier_rollup_dir	claim_b	results/method4_benchmark_matrix/method4_multiassign_recall_qps_excel_20260630	yes	dir	NA	9
claim_b_online_frontier_source_summary_jsons.0	0	results/method4_benchmark_matrix/method4_multiassign_expansion_qps_20260629_orion_low/eval3000_20260629_v2/orion_o100_single_r080/20260629_101020/summary.json	yes	json	5159	45
claim_b_online_frontier_source_summary_jsons.1	1	results/method4_benchmark_matrix/method4_multiassign_expansion_qps_20260629_orion_low/eval3000_20260629_v2/orion_o100_single_r085/20260629_101158/summary.json	yes	json	5176	45
claim_b_online_frontier_source_summary_jsons.2	2	results/method4_benchmark_matrix/method4_multiassign_expansion_qps_20260629_orion_low_recover/eval3000_20260629/orion_o100_single_r090/20260629_110713/summary.json	yes	json	4433	45
claim_b_online_frontier_source_summary_jsons.3	3	results/method4_benchmark_matrix/method4_multiassign_expansion_qps_20260629_orion_low_recover/eval3000_20260629/orion_o100_single_r095/20260629_110832/summary.json	yes	json	4422	45
claim_b_online_frontier_source_summary_jsons.4	4	results/method4_benchmark_matrix/method4_multiassign_expansion_qps_20260629_orion_default/eval3000_20260629/orion_o118_default_r080/20260629_101825/summary.json	yes	json	5189	45
claim_b_online_frontier_source_summary_jsons.5	5	results/method4_benchmark_matrix/method4_multiassign_expansion_qps_20260629_orion_low_recover/eval3000_20260629/orion_o118_default_r085/20260629_110937/summary.json	yes	json	4431	45
claim_b_online_frontier_source_summary_jsons.6	6	results/method4_benchmark_matrix/method4_multiassign_expansion_qps_20260629_orion_low_recover/eval3000_20260629/orion_o118_default_r090/20260629_111013/summary.json	yes	json	4456	45
claim_b_online_frontier_source_summary_jsons.7	7	results/method4_benchmark_matrix/method4_multiassign_expansion_qps_20260629_orion_low_recover/eval3000_20260629/orion_o118_default_r095/20260629_111048/summary.json	yes	json	4465	45
claim_b_online_frontier_source_summary_jsons.8	8	results/method4_benchmark_matrix/method4_multiassign_expansion_qps_20260629_orion_high_recover/eval3000_20260629/orion_o149_w2c2_r080/20260629_111612/summary.json	yes	json	5562	45
claim_b_online_frontier_source_summary_jsons.9	9	results/method4_benchmark_matrix/method4_multiassign_expansion_qps_20260629_orion_high_recover/eval3000_20260629/orion_o149_w2c2_r085/20260629_112512/summary.json	yes	json	4455	45
claim_b_online_frontier_source_summary_jsons.10	10	results/method4_benchmark_matrix/method4_multiassign_expansion_qps_20260629_orion_high_recover/eval3000_20260629/orion_o149_w2c2_r090/20260629_112556/summary.json	yes	json	4466	45
claim_b_online_frontier_source_summary_jsons.11	11	results/method4_benchmark_matrix/method4_multiassign_expansion_qps_20260629_orion_high_recover/eval3000_20260629/orion_o149_w2c2_r095/20260629_112639/summary.json	yes	json	4460	45
claim_b_online_frontier_source_summary_jsons.12	12	results/method4_benchmark_matrix/method4_multiassign_expansion_qps_20260629_orion_high_recover/eval3000_20260629/orion_o184_w2c3_r080/20260629_112739/summary.json	yes	json	5678	45
claim_b_online_frontier_source_summary_jsons.13	13	results/method4_benchmark_matrix/method4_multiassign_expansion_qps_20260629_orion_high_recover/eval3000_20260629/orion_o184_w2c3_r085/20260629_113813/summary.json	yes	json	4452	45
claim_b_online_frontier_source_summary_jsons.14	14	results/method4_benchmark_matrix/method4_multiassign_expansion_qps_20260629_orion_high_recover/eval3000_20260629/orion_o184_w2c3_r090/20260629_113908/summary.json	yes	json	4453	45
claim_b_online_frontier_source_summary_jsons.15	15	results/method4_benchmark_matrix/method4_multiassign_expansion_qps_20260629_orion_high_recover/eval3000_20260629/orion_o184_w2c3_r095/20260629_113958/summary.json	yes	json	4437	45
claim_b_online_frontier_source_summary_jsons.16	16	results/method4_benchmark_matrix/method4_multiassign_expansion_qps_20260629_simple_low/eval3000_20260629/simple_s100_a1000_r080/20260629_115220/summary.json	yes	json	3656	45
claim_b_online_frontier_source_summary_jsons.17	17	results/method4_benchmark_matrix/method4_multiassign_expansion_qps_20260629_simple_low/eval3000_20260629/simple_s100_a1000_r085/20260629_115739/summary.json	yes	json	3668	45
claim_b_online_frontier_source_summary_jsons.18	18	results/method4_benchmark_matrix/method4_multiassign_expansion_qps_20260629_simple_low/eval3000_20260629/simple_s100_a1000_r090/20260629_115756/summary.json	yes	json	3657	45
claim_b_online_frontier_source_summary_jsons.19	19	results/method4_benchmark_matrix/method4_multiassign_expansion_qps_20260629_simple_low/eval3000_20260629/simple_s100_a1000_r095/20260629_115820/summary.json	yes	json	3675	45
claim_b_online_frontier_source_summary_jsons.20	20	results/method4_benchmark_matrix/method4_multiassign_expansion_qps_20260629_simple_low/eval3000_20260629/simple_s119_a1004_r080/20260629_115914/summary.json	yes	json	3667	45
claim_b_online_frontier_source_summary_jsons.21	21	results/method4_benchmark_matrix/method4_multiassign_expansion_qps_20260629_simple_low/eval3000_20260629/simple_s119_a1004_r085/20260629_120549/summary.json	yes	json	3674	45
claim_b_online_frontier_source_summary_jsons.22	22	results/method4_benchmark_matrix/method4_multiassign_expansion_qps_20260629_simple_low/eval3000_20260629/simple_s119_a1004_r090/20260629_120616/summary.json	yes	json	3687	45
claim_b_online_frontier_source_summary_jsons.23	23	results/method4_benchmark_matrix/method4_multiassign_expansion_qps_20260629_simple_low/eval3000_20260629/simple_s119_a1004_r095/20260629_120652/summary.json	yes	json	3681	45
claim_b_online_frontier_source_summary_jsons.24	24	results/method4_benchmark_matrix/method4_multiassign_expansion_qps_20260629_simple_s157_only/eval3000_20260629/simple_s157_a1010_r080/20260629_121256/summary.json	yes	json	3679	45
claim_b_online_frontier_source_summary_jsons.25	25	results/method4_benchmark_matrix/method4_multiassign_expansion_qps_20260629_simple_s157_only/eval3000_20260629/simple_s157_a1010_r085/20260629_122117/summary.json	yes	json	3684	45
claim_b_online_frontier_source_summary_jsons.26	26	results/method4_benchmark_matrix/method4_multiassign_expansion_qps_20260629_simple_s157_only/eval3000_20260629/simple_s157_a1010_r090/20260629_122146/summary.json	yes	json	3685	45
claim_b_online_frontier_source_summary_jsons.27	27	results/method4_benchmark_matrix/method4_multiassign_expansion_qps_20260629_simple_s157_only/eval3000_20260629/simple_s157_a1010_r095/20260629_122223/summary.json	yes	json	3691	45
claim_b_online_frontier_source_summary_jsons.28	28	results/method4_benchmark_matrix/method4_multiassign_expansion_qps_20260629_simple_s189_only/eval3000_20260629/simple_s189_a1014_r080/20260629_122806/summary.json	yes	json	3666	45
claim_b_online_frontier_source_summary_jsons.29	29	results/method4_benchmark_matrix/method4_multiassign_expansion_qps_20260629_simple_s189_only/eval3000_20260629/simple_s189_a1014_r085/20260629_123753/summary.json	yes	json	3673	45
claim_b_online_frontier_source_summary_jsons.30	30	results/method4_benchmark_matrix/method4_multiassign_expansion_qps_20260629_simple_s189_only/eval3000_20260629/simple_s189_a1014_r090/20260629_123822/summary.json	yes	json	3685	45
claim_b_online_frontier_source_summary_jsons.31	31	results/method4_benchmark_matrix/method4_multiassign_expansion_qps_20260629_simple_s189_only/eval3000_20260629/simple_s189_a1014_r095/20260629_123900/summary.json	yes	json	3691	45
claim_b_online_frontier_strict_keypoints_long_csv	claim_b	results/method4_benchmark_matrix/method4_multiassign_recall_qps_excel_20260630_strict/final_eval_keypoints_long_strict.csv	yes	csv	18954	32
claim_b_online_frontier_strict_readme_md	claim_b	results/method4_benchmark_matrix/method4_multiassign_recall_qps_excel_20260630_strict/README.md	yes	md	1915	NA
claim_b_online_frontier_strict_rollup_dir	claim_b	results/method4_benchmark_matrix/method4_multiassign_recall_qps_excel_20260630_strict	yes	dir	NA	8
claim_b_online_frontier_strict_source_summary_jsons.0	0	results/method4_benchmark_matrix/method4_multiassign_recall_qps_current_20260630_orion_single_low/eval3000_20260630/orion_o100_single_r080_current/20260630_101917/summary.json	yes	json	5152	45
claim_b_online_frontier_strict_source_summary_jsons.1	1	results/method4_benchmark_matrix/method4_multiassign_recall_qps_current_20260630_orion_single_low/eval3000_20260630/orion_o100_single_r085_current/20260630_102621/summary.json	yes	json	4450	45
claim_b_online_frontier_strict_source_summary_jsons.2	2	results/method4_benchmark_matrix/method4_multiassign_expansion_qps_20260629_orion_low_recover/eval3000_20260629/orion_o100_single_r090/20260629_110713/summary.json	yes	json	4433	45
claim_b_online_frontier_strict_source_summary_jsons.3	3	results/method4_benchmark_matrix/method4_multiassign_expansion_qps_20260629_orion_low_recover/eval3000_20260629/orion_o100_single_r095/20260629_110832/summary.json	yes	json	4422	45
claim_b_online_frontier_strict_source_summary_jsons.4	4	results/method4_benchmark_matrix/method4_multiassign_recall_qps_strict_20260630_orion_default/eval3000_20260630/orion_o118_default_r080_strict/20260630_102914/summary.json	yes	json	5182	45
claim_b_online_frontier_strict_source_summary_jsons.5	5	results/method4_benchmark_matrix/method4_multiassign_recall_qps_strict_20260630_orion_default/eval3000_20260630/orion_o118_default_r085_strict/20260630_103628/summary.json	yes	json	4474	45
claim_b_online_frontier_strict_source_summary_jsons.6	6	results/method4_benchmark_matrix/method4_multiassign_recall_qps_strict_20260630_orion_default/eval3000_20260630/orion_o118_default_r090_strict/20260630_103703/summary.json	yes	json	4460	45
claim_b_online_frontier_strict_source_summary_jsons.7	7	results/method4_benchmark_matrix/method4_multiassign_recall_qps_strict_20260630_orion_default/eval3000_20260630/orion_o118_default_r095_strict/20260630_103739/summary.json	yes	json	4457	45
claim_b_online_frontier_strict_source_summary_jsons.8	8	results/method4_benchmark_matrix/method4_multiassign_recall_qps_strict_20260630_orion_w2c2/eval3000_20260630/orion_o149_w2c2_r080_strict/20260630_104031/summary.json	yes	json	5547	45
claim_b_online_frontier_strict_source_summary_jsons.9	9	results/method4_benchmark_matrix/method4_multiassign_recall_qps_strict_20260630_orion_w2c2/eval3000_20260630/orion_o149_w2c2_r085_strict/20260630_104916/summary.json	yes	json	4471	45
claim_b_online_frontier_strict_source_summary_jsons.10	10	results/method4_benchmark_matrix/method4_multiassign_expansion_qps_20260629_orion_high_recover/eval3000_20260629/orion_o149_w2c2_r090/20260629_112556/summary.json	yes	json	4466	45
claim_b_online_frontier_strict_source_summary_jsons.11	11	results/method4_benchmark_matrix/method4_multiassign_expansion_qps_20260629_orion_high_recover/eval3000_20260629/orion_o149_w2c2_r095/20260629_112639/summary.json	yes	json	4460	45
claim_b_online_frontier_strict_source_summary_jsons.12	12	results/method4_benchmark_matrix/method4_multiassign_expansion_qps_20260629_orion_high_recover/eval3000_20260629/orion_o184_w2c3_r080/20260629_112739/summary.json	yes	json	5678	45
claim_b_online_frontier_strict_source_summary_jsons.13	13	results/method4_benchmark_matrix/method4_multiassign_expansion_qps_20260629_orion_high_recover/eval3000_20260629/orion_o184_w2c3_r085/20260629_113813/summary.json	yes	json	4452	45
claim_b_online_frontier_strict_source_summary_jsons.14	14	results/method4_benchmark_matrix/method4_multiassign_expansion_qps_20260629_orion_high_recover/eval3000_20260629/orion_o184_w2c3_r090/20260629_113908/summary.json	yes	json	4453	45
claim_b_online_frontier_strict_source_summary_jsons.15	15	results/method4_benchmark_matrix/method4_multiassign_expansion_qps_20260629_orion_high_recover/eval3000_20260629/orion_o184_w2c3_r095/20260629_113958/summary.json	yes	json	4437	45
claim_b_online_frontier_strict_source_summary_jsons.16	16	results/method4_benchmark_matrix/method4_multiassign_expansion_qps_20260629_simple_low/eval3000_20260629/simple_s100_a1000_r080/20260629_115220/summary.json	yes	json	3656	45
claim_b_online_frontier_strict_source_summary_jsons.17	17	results/method4_benchmark_matrix/method4_multiassign_expansion_qps_20260629_simple_low/eval3000_20260629/simple_s100_a1000_r085/20260629_115739/summary.json	yes	json	3668	45
claim_b_online_frontier_strict_source_summary_jsons.18	18	results/method4_benchmark_matrix/method4_multiassign_recall_qps_strict_20260630_simple_a1000/eval3000_20260630/simple_s100_a1000_r090_strict/20260630_105107/summary.json	yes	json	3681	45
claim_b_online_frontier_strict_source_summary_jsons.19	19	results/method4_benchmark_matrix/method4_multiassign_expansion_qps_20260629_simple_low/eval3000_20260629/simple_s100_a1000_r095/20260629_115820/summary.json	yes	json	3675	45
claim_b_online_frontier_strict_source_summary_jsons.20	20	results/method4_benchmark_matrix/method4_multiassign_expansion_qps_20260629_simple_low/eval3000_20260629/simple_s119_a1004_r080/20260629_115914/summary.json	yes	json	3667	45
claim_b_online_frontier_strict_source_summary_jsons.21	21	results/method4_benchmark_matrix/method4_multiassign_expansion_qps_20260629_simple_low/eval3000_20260629/simple_s119_a1004_r085/20260629_120549/summary.json	yes	json	3674	45
claim_b_online_frontier_strict_source_summary_jsons.22	22	results/method4_benchmark_matrix/method4_multiassign_recall_qps_strict_20260630_simple_a1004/eval3000_20260630/simple_s119_a1004_r090_strict/20260630_105746/summary.json	yes	json	3691	45
claim_b_online_frontier_strict_source_summary_jsons.23	23	results/method4_benchmark_matrix/method4_multiassign_expansion_qps_20260629_simple_low/eval3000_20260629/simple_s119_a1004_r095/20260629_120652/summary.json	yes	json	3681	45
claim_b_online_frontier_strict_source_summary_jsons.24	24	results/method4_benchmark_matrix/method4_multiassign_expansion_qps_20260629_simple_s157_only/eval3000_20260629/simple_s157_a1010_r080/20260629_121256/summary.json	yes	json	3679	45
claim_b_online_frontier_strict_source_summary_jsons.25	25	results/method4_benchmark_matrix/method4_multiassign_expansion_qps_20260629_simple_s157_only/eval3000_20260629/simple_s157_a1010_r085/20260629_122117/summary.json	yes	json	3684	45
claim_b_online_frontier_strict_source_summary_jsons.26	26	results/method4_benchmark_matrix/method4_multiassign_expansion_qps_20260629_simple_s157_only/eval3000_20260629/simple_s157_a1010_r090/20260629_122146/summary.json	yes	json	3685	45
claim_b_online_frontier_strict_source_summary_jsons.27	27	results/method4_benchmark_matrix/method4_multiassign_expansion_qps_20260629_simple_s157_only/eval3000_20260629/simple_s157_a1010_r095/20260629_122223/summary.json	yes	json	3691	45
claim_b_online_frontier_strict_source_summary_jsons.28	28	results/method4_benchmark_matrix/method4_multiassign_expansion_qps_20260629_simple_s189_only/eval3000_20260629/simple_s189_a1014_r080/20260629_122806/summary.json	yes	json	3666	45
claim_b_online_frontier_strict_source_summary_jsons.29	29	results/method4_benchmark_matrix/method4_multiassign_expansion_qps_20260629_simple_s189_only/eval3000_20260629/simple_s189_a1014_r085/20260629_123753/summary.json	yes	json	3673	45
claim_b_online_frontier_strict_source_summary_jsons.30	30	results/method4_benchmark_matrix/method4_multiassign_expansion_qps_20260629_simple_s189_only/eval3000_20260629/simple_s189_a1014_r090/20260629_123822/summary.json	yes	json	3685	45
claim_b_online_frontier_strict_source_summary_jsons.31	31	results/method4_benchmark_matrix/method4_multiassign_expansion_qps_20260629_simple_s189_only/eval3000_20260629/simple_s189_a1014_r095/20260629_123900/summary.json	yes	json	3691	45
claim_b_oracle_miss_dir	claim_b	results/method4_claim_b_oracle_miss_20260704/analysis_20260704_211749	yes	dir	NA	3
claim_b_oracle_miss_metadata_json	claim_b	results/method4_claim_b_oracle_miss_20260704/analysis_20260704_211749/run_metadata.json	yes	json	1607	19
claim_b_oracle_miss_strategy_build_summary_csv	claim_b	results/method4_claim_b_oracle_miss_20260704/analysis_20260704_211749/strategy_build_summary.csv	yes	csv	392	3
claim_b_oracle_miss_summary_csv	claim_b	results/method4_claim_b_oracle_miss_20260704/analysis_20260704_211749/oracle_gt_miss_summary.csv	yes	csv	1674	9
claim_b_raw_provenance_audit_csv	claim_b	results/method4_claim_coverage_20260704/derived_claim_tables/claim_b_raw_provenance_audit_20260708.csv	yes	csv	42728	96
claim_c_evidence_dir	claim_c	results/method4_claim_c_evidence_20260704	yes	dir	NA	6
claim_c_evidence_manifest_json	claim_c	results/method4_claim_c_evidence_20260704/source_manifest.json	yes	json	2367	13
claim_c_latency_analysis_dir	claim_c	results/method4_claim_c_dynamic_vs_fixed_latency_20260704/analysis_20260704_235925	yes	dir	NA	3
claim_c_latency_deltas_csv	claim_c	results/method4_claim_coverage_20260704/derived_claim_tables/claim_c_dynamic_vs_fixed_latency_deltas.csv	yes	csv	883	1
claim_c_latency_matrix_csv	claim_c	results/method4_claim_coverage_20260704/derived_claim_tables/claim_c_dynamic_vs_fixed_latency_matrix.csv	yes	csv	1391	2
claim_c_latency_raw_batches_csv	claim_c	results/method4_claim_c_dynamic_vs_fixed_latency_20260704/analysis_20260704_235925/claim_d_high_recall_latency_batches.csv	yes	csv	11244	90
claim_c_latency_raw_metadata_json	claim_c	results/method4_claim_c_dynamic_vs_fixed_latency_20260704/analysis_20260704_235925/run_metadata.json	yes	json	836	16
claim_c_latency_raw_summary_csv	claim_c	results/method4_claim_c_dynamic_vs_fixed_latency_20260704/analysis_20260704_235925/claim_d_high_recall_latency_summary.csv	yes	csv	1795	6
claim_c_latency_summary_script	claim_c	tools/method4_claim_c_latency_summary.py	yes	py	7785	NA
claim_c_performance_deltas_csv	claim_c	results/method4_claim_c_evidence_20260704/dynamic_vs_fixed_deltas.csv	yes	csv	645	3
claim_c_performance_dynamic_upper_120_summary_json	claim_c	results/method4_claim_c_orion_dynamic_ef_20260625/20260625_122853/summary.json	yes	json	4027	42
claim_c_performance_fixed_upper_120_summary_json	claim_c	results/method4_claim_c_orion_fixed_ef_extended_20260625/20260625_122713/summary.json	yes	json	4043	42
claim_c_performance_frontier_dynamic_095_summary_json	claim_c	results/method4_claim_c_orion_frontier_dynamic_095_robust_confirm_20260625/20260625_132811/summary.json	yes	json	4025	42
claim_c_performance_frontier_dynamic_097_summary_json	claim_c	results/method4_claim_c_orion_frontier_dynamic_097_more_robust_confirm_20260625/20260625_132923/summary.json	yes	json	4057	42
claim_c_performance_frontier_fixed_095_summary_json	claim_c	results/method4_claim_c_orion_frontier_fixed_095_robust_confirm_20260625/20260625_132656/summary.json	yes	json	3996	42
claim_c_performance_frontier_fixed_097_summary_json	claim_c	results/method4_claim_c_orion_frontier_fixed_097_robust_confirm_20260625/20260625_132210/summary.json	yes	json	4031	42
claim_c_performance_summary_csv	claim_c	results/method4_claim_c_evidence_20260704/dynamic_vs_fixed_performance_summary.csv	yes	csv	1677	6
claim_c_raw_provenance_audit_csv	claim_c	results/method4_claim_coverage_20260704/derived_claim_tables/claim_c_raw_provenance_audit_20260708.csv	yes	csv	21709	58
claim_c_relevance_analysis_dir	claim_c	results/method4_claim_c_dynamic_ef_relevance_20260704/analysis_20260704_141217	yes	dir	NA	4
claim_c_relevance_budget_alignment_csv	claim_c	results/method4_claim_c_dynamic_ef_relevance_20260704/analysis_20260704_141217/budget_alignment_summary.csv	yes	csv	1371	3
claim_c_relevance_by_count_csv	claim_c	results/method4_claim_c_dynamic_ef_relevance_20260704/analysis_20260704_141217/routed_ep_relevance_by_count.csv	yes	csv	3753	27
claim_c_relevance_metadata_json	claim_c	results/method4_claim_c_dynamic_ef_relevance_20260704/analysis_20260704_141217/run_metadata.json	yes	json	1471	18
claim_c_routed_ep_relevance_summary_csv	claim_c	results/method4_claim_c_evidence_20260704/routed_ep_relevance_summary.csv	yes	csv	1371	3
claim_d_099_alternate_width_formal_deltas_csv	claim_d	results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_099_alternate_width_formal_deltas.csv	yes	csv	1742	2
claim_d_099_alternate_width_formal_dir	claim_d	results/method4_claim_d_099_alternate_width_formal_20260708/analysis_20260708_100010	yes	dir	NA	3
claim_d_099_alternate_width_formal_raw_metadata_json	claim_d	results/method4_claim_d_099_alternate_width_formal_20260708/analysis_20260708_100010/run_metadata.json	yes	json	1367	29
claim_d_099_alternate_width_formal_raw_summary_csv	claim_d	results/method4_claim_d_099_alternate_width_formal_20260708/analysis_20260708_100010/claim_d_high_recall_latency_summary.csv	yes	csv	2806	9
claim_d_099_alternate_width_formal_summary_csv	claim_d	results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_099_alternate_width_formal_summary.csv	yes	csv	2598	3
claim_d_099_alternate_width_manifest_json	claim_d	results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_099_alternate_width_manifest.json	yes	json	1650	12
claim_d_099_alternate_width_screen_dir	claim_d	results/method4_claim_d_099_alternate_width_screen_20260708/analysis_20260708_095804	yes	dir	NA	3
claim_d_099_alternate_width_screen_raw_metadata_json	claim_d	results/method4_claim_d_099_alternate_width_screen_20260708/analysis_20260708_095804/run_metadata.json	yes	json	1498	29
claim_d_099_alternate_width_screen_raw_summary_csv	claim_d	results/method4_claim_d_099_alternate_width_screen_20260708/analysis_20260708_095804/claim_d_high_recall_latency_summary.csv	yes	csv	2183	7
claim_d_099_alternate_width_screen_summary_csv	claim_d	results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_099_alternate_width_screen_summary.csv	yes	csv	4392	7
claim_d_099_boundary_method4_vs_naive_deltas_csv	claim_d	results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_099_boundary_method4_vs_naive_deltas.csv	yes	csv	3108	4
claim_d_099_boundary_query_order_manifest_json	claim_d	results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_099_boundary_query_order_manifest.json	yes	json	3931	14
claim_d_099_boundary_query_order_multiseed_summary_csv	claim_d	results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_099_boundary_query_order_multiseed_summary.csv	yes	csv	1039	2
claim_d_099_boundary_query_order_overlay_csv	claim_d	results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_099_boundary_query_order_overlay.csv	yes	csv	2489	6
claim_d_099_boundary_query_order_sensitivity_csv	claim_d	results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_099_boundary_query_order_sensitivity.csv	yes	csv	2077	4
claim_d_099_boundary_sensitivity_deltas_csv	claim_d	results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_099_boundary_sensitivity_deltas.csv	yes	csv	2988	6
claim_d_099_boundary_shuffle_batches_csv	claim_d	results/method4_099_boundary_shuffle_20260713/analysis_20260706_140200/claim_d_high_recall_latency_batches.csv	yes	csv	25042	180
claim_d_099_boundary_shuffle_dir	claim_d	results/method4_099_boundary_shuffle_20260713/analysis_20260706_140200	yes	dir	NA	3
claim_d_099_boundary_shuffle_metadata_json	claim_d	results/method4_099_boundary_shuffle_20260713/analysis_20260706_140200/run_metadata.json	yes	json	1343	29
claim_d_099_boundary_shuffle_seed_20260718_batches_csv	claim_d	results/method4_099_boundary_shuffle2_20260718/analysis_20260707_102446/claim_d_high_recall_latency_batches.csv	yes	csv	25060	180
claim_d_099_boundary_shuffle_seed_20260718_dir	claim_d	results/method4_099_boundary_shuffle2_20260718/analysis_20260707_102446	yes	dir	NA	3
claim_d_099_boundary_shuffle_seed_20260718_metadata_json	claim_d	results/method4_099_boundary_shuffle2_20260718/analysis_20260707_102446/run_metadata.json	yes	json	1343	29
claim_d_099_boundary_shuffle_seed_20260718_summary_csv	claim_d	results/method4_099_boundary_shuffle2_20260718/analysis_20260707_102446/claim_d_high_recall_latency_summary.csv	yes	csv	2018	6
claim_d_099_boundary_shuffle_summary_csv	claim_d	results/method4_099_boundary_shuffle_20260713/analysis_20260706_140200/claim_d_high_recall_latency_summary.csv	yes	csv	2050	6
claim_d_099_boundary_warmup_batches_csv	claim_d	results/method4_099_boundary_warmup_20260713/analysis_20260706_141159/claim_d_high_recall_latency_batches.csv	yes	csv	25049	180
claim_d_099_boundary_warmup_dir	claim_d	results/method4_099_boundary_warmup_20260713/analysis_20260706_141159	yes	dir	NA	3
claim_d_099_boundary_warmup_metadata_json	claim_d	results/method4_099_boundary_warmup_20260713/analysis_20260706_141159/run_metadata.json	yes	json	1343	29
claim_d_099_boundary_warmup_method4_vs_naive_deltas_csv	claim_d	results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_099_boundary_warmup_method4_vs_naive_deltas.csv	yes	csv	1762	2
claim_d_099_boundary_warmup_overlay_csv	claim_d	results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_099_boundary_warmup_overlay.csv	yes	csv	1896	4
claim_d_099_boundary_warmup_sensitivity_csv	claim_d	results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_099_boundary_warmup_sensitivity.csv	yes	csv	1141	2
claim_d_099_boundary_warmup_summary_csv	claim_d	results/method4_099_boundary_warmup_20260713/analysis_20260706_141159/claim_d_high_recall_latency_summary.csv	yes	csv	1992	6
claim_d_099_broader_strategy_formal_deltas_csv	claim_d	results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_099_broader_strategy_formal_deltas.csv	yes	csv	3249	4
claim_d_099_broader_strategy_formal_dir	claim_d	results/method4_claim_d_099_broader_formal_20260707/analysis_20260707_111008	yes	dir	NA	3
claim_d_099_broader_strategy_formal_summary_csv	claim_d	results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_099_broader_strategy_formal_summary.csv	yes	csv	4388	6
claim_d_099_broader_strategy_manifest_json	claim_d	results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_099_broader_strategy_manifest.json	yes	json	2596	11
claim_d_099_broader_strategy_screen_dir	claim_d	results/method4_claim_d_099_broader_screen_20260707/analysis_20260707_110802	yes	dir	NA	3
claim_d_099_broader_strategy_screen_summary_csv	claim_d	results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_099_broader_strategy_screen_summary.csv	yes	csv	3701	7
claim_d_099_broader_strategy_top_recall_formal_dir	claim_d	results/method4_claim_d_099_broader_top_recall_formal_20260707/analysis_20260707_112432	yes	dir	NA	3
claim_d_099_broader_strategy_top_recall_formal_raw_summary_csv	claim_d	results/method4_claim_d_099_broader_top_recall_formal_20260707/analysis_20260707_112432/claim_d_high_recall_latency_summary.csv	yes	csv	2797	9
claim_d_099_retune_deltas_csv	claim_d	results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_099_retune_deltas.csv	yes	csv	1370	2
claim_d_099_retune_formal_dir	claim_d	results/method4_claim_d_099_retune_20260705/analysis_20260705_022625	yes	dir	NA	3
claim_d_099_retune_root	claim_d	results/method4_claim_d_099_retune_20260705	yes	dir	NA	3
claim_d_099_retune_summary_csv	claim_d	results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_099_retune_summary.csv	yes	csv	1987	4
claim_d_099_retune_tuning_candidates_500q_csv	claim_d	results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_099_retune_tuning_candidates_500q.csv	yes	csv	5066	22
claim_d_099_strategy_search_deltas_csv	claim_d	results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_099_strategy_search_deltas.csv	yes	csv	5496	8
claim_d_099_strategy_search_dir	claim_d	results/method4_claim_d_099_strategy_search_20260705/analysis_20260705_074939	yes	dir	NA	3
claim_d_099_strategy_search_raw_batches_csv	claim_d	results/method4_claim_d_099_strategy_search_20260705/analysis_20260705_074939/claim_d_high_recall_latency_batches.csv	yes	csv	49834	360
claim_d_099_strategy_search_raw_metadata_json	claim_d	results/method4_claim_d_099_strategy_search_20260705/analysis_20260705_074939/run_metadata.json	yes	json	1394	29
claim_d_099_strategy_search_raw_summary_csv	claim_d	results/method4_claim_d_099_strategy_search_20260705/analysis_20260705_074939/claim_d_high_recall_latency_summary.csv	yes	csv	3652	12
claim_d_099_strategy_search_summary_csv	claim_d	results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_099_strategy_search_summary.csv	yes	csv	4757	8
claim_d_099_strategy_search_top_screen_deltas_csv	claim_d	results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_099_strategy_search_top_screen_deltas.csv	yes	csv	3036	4
claim_d_099_strategy_search_top_screen_dir	claim_d	results/method4_claim_d_099_strategy_search_20260706/top_screen_candidates/analysis_20260706_130013	yes	dir	NA	3
claim_d_099_strategy_search_top_screen_raw_batches_csv	claim_d	results/method4_claim_d_099_strategy_search_20260706/top_screen_candidates/analysis_20260706_130013/claim_d_high_recall_latency_batches.csv	yes	csv	49847	360
claim_d_099_strategy_search_top_screen_raw_metadata_json	claim_d	results/method4_claim_d_099_strategy_search_20260706/top_screen_candidates/analysis_20260706_130013/run_metadata.json	yes	json	1394	29
claim_d_099_strategy_search_top_screen_raw_summary_csv	claim_d	results/method4_claim_d_099_strategy_search_20260706/top_screen_candidates/analysis_20260706_130013/claim_d_high_recall_latency_summary.csv	yes	csv	3647	12
claim_d_099_strategy_search_top_screen_summary_csv	claim_d	results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_099_strategy_search_top_screen_summary.csv	yes	csv	2743	4
claim_d_batch200_high_recall_deltas_csv	claim_d	results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_batch200_high_recall_deltas.csv	yes	csv	2625	2
claim_d_batch200_high_recall_dir	claim_d	results/method4_claim_d_batch200_targets_20260708/analysis_20260708_104611	yes	dir	NA	3
claim_d_batch200_high_recall_manifest_json	claim_d	results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_batch200_high_recall_manifest.json	yes	json	2184	11
claim_d_batch200_high_recall_raw_batches_csv	claim_d	results/method4_claim_d_batch200_targets_20260708/analysis_20260708_104611/claim_d_high_recall_latency_batches.csv	yes	csv	26174	180
claim_d_batch200_high_recall_raw_metadata_json	claim_d	results/method4_claim_d_batch200_targets_20260708/analysis_20260708_104611/run_metadata.json	yes	json	1416	29
claim_d_batch200_high_recall_raw_summary_csv	claim_d	results/method4_claim_d_batch200_targets_20260708/analysis_20260708_104611/claim_d_high_recall_latency_summary.csv	yes	csv	3729	12
claim_d_batch200_high_recall_summary_csv	claim_d	results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_batch200_high_recall_summary.csv	yes	csv	6745	4
claim_d_high_recall_latency_dir	claim_d	results/method4_claim_d_high_recall_latency_20260704/analysis_20260704_215448	yes	dir	NA	3
claim_d_high_recall_latency_pairs_csv	claim_d	results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_high_recall_latency_pairs.csv	yes	csv	2726	4
claim_d_high_recall_latency_refine_dir	claim_d	results/method4_claim_d_high_recall_latency_refine_20260704/analysis_20260704_220036	yes	dir	NA	3
claim_d_high_recall_latency_script	claim_d	tools/method4_claim_d_high_recall_latency.py	yes	py	27786	NA
claim_e_build_stage_artifact_audit_csv	claim_e	results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_build_stage_artifact_audit.csv	yes	csv	1488	3
claim_e_build_stage_instrumented_smoke_csv	claim_e	results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_build_stage_instrumented_smoke_20260707.csv	yes	csv	1962	1
claim_e_build_stage_instrumented_smoke_manifest_json	claim_e	results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_build_stage_instrumented_smoke_manifest.json	yes	json	3409	10
claim_e_build_stage_smoke_builds_csv	claim_e	results/method4_claim_e_build_stage_smoke_20260707/claim_e_build_smoke_20260707_085303/20260707_085328/builds.csv	yes	csv	324	1
claim_e_build_stage_smoke_raw_stdout	claim_e	results/method4_claim_e_build_stage_smoke_20260707/raw_claim_e_build_smoke_20260707_085303/stdout.txt	yes	txt	515	NA
claim_e_build_stage_smoke_run_root	claim_e	results/method4_claim_e_build_stage_smoke_20260707/claim_e_build_smoke_20260707_085303/20260707_085328	yes	dir	NA	6
claim_e_build_stage_smoke_summary_json	claim_e	results/method4_claim_e_build_stage_smoke_20260707/claim_e_build_smoke_20260707_085303/20260707_085328/summary.json	yes	json	4614	50
claim_e_build_stage_smoke_time_stderr	claim_e	results/method4_claim_e_build_stage_smoke_20260707/raw_claim_e_build_smoke_20260707_085303/time_v_stderr.txt	yes	txt	1272	NA
claim_e_container_memory_batches_csv	claim_e	results/method4_claim_e_container_overhead_memory_20260705/analysis_20260705_040749/claim_e_container_overhead_batches.csv	yes	csv	32706	135
claim_e_container_memory_derived_deltas_csv	claim_e	results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_container_memory_deltas.csv	yes	csv	1508	2
claim_e_container_memory_derived_summary_csv	claim_e	results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_container_memory_summary.csv	yes	csv	2252	3
claim_e_container_memory_dir	claim_e	results/method4_claim_e_container_overhead_memory_20260705/analysis_20260705_040749	yes	dir	NA	4
claim_e_container_memory_samples_csv	claim_e	results/method4_claim_e_container_overhead_memory_20260705/analysis_20260705_040749/claim_e_container_overhead_samples.csv	yes	csv	26416	144
claim_e_container_memory_summary_csv	claim_e	results/method4_claim_e_container_overhead_memory_20260705/analysis_20260705_040749/claim_e_container_overhead_summary.csv	yes	csv	7609	9
claim_e_container_overhead_batches_csv	claim_e	results/method4_claim_e_container_overhead_20260705/analysis_20260705_024218/claim_e_container_overhead_batches.csv	yes	csv	32739	135
claim_e_container_overhead_derived_deltas_csv	claim_e	results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_container_overhead_deltas.csv	yes	csv	2143	2
claim_e_container_overhead_derived_summary_csv	claim_e	results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_container_overhead_summary.csv	yes	csv	3454	3
claim_e_container_overhead_dir	claim_e	results/method4_claim_e_container_overhead_20260705/analysis_20260705_024218	yes	dir	NA	4
claim_e_container_overhead_samples_csv	claim_e	results/method4_claim_e_container_overhead_20260705/analysis_20260705_024218/claim_e_container_overhead_samples.csv	yes	csv	19571	144
claim_e_container_overhead_script	claim_e	tools/method4_claim_e_container_overhead.py	yes	py	25522	NA
claim_e_container_overhead_summary_csv	claim_e	results/method4_claim_e_container_overhead_20260705/analysis_20260705_024218/claim_e_container_overhead_summary.csv	yes	csv	6073	9
claim_e_controller_process_attribution_deltas_csv	claim_e	results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_controller_process_attribution_deltas.csv	yes	csv	8255	42
claim_e_controller_process_attribution_manifest_json	claim_e	results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_controller_process_attribution_manifest.json	yes	json	3169	13
claim_e_controller_process_attribution_pid_map_json	claim_e	results/method4_claim_e_controller_process_attribution_20260707/batch200_all_variants_20260707_091900/process_pid_map.json	yes	json	385	4
claim_e_controller_process_attribution_raw_manifest_json	claim_e	results/method4_claim_e_controller_process_attribution_20260707/batch200_all_variants_20260707_091900/run_manifest.json	yes	json	7259	10
claim_e_controller_process_attribution_raw_run_rows_csv	claim_e	results/method4_claim_e_controller_process_attribution_20260707/batch200_all_variants_20260707_091900/raw_run_rows.csv	yes	csv	3675	3
claim_e_controller_process_attribution_run_root	claim_e	results/method4_claim_e_controller_process_attribution_20260707/batch200_all_variants_20260707_091900	yes	dir	NA	6
claim_e_controller_process_attribution_summary_csv	claim_e	results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_controller_process_attribution_summary.csv	yes	csv	13374	18
claim_e_cost_row_build_source_audit_csv	claim_e	results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_cost_row_build_source_audit.csv	yes	csv	6272	9
claim_e_execution_mode_latency_batches_csv	claim_e	results/method4_claim_e_execution_mode_latency_20260704/analysis_20260704_225847/claim_e_execution_mode_latency_batches.csv	yes	csv	236436	993
claim_e_execution_mode_latency_deltas_csv	claim_e	results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_execution_mode_latency_deltas.csv	yes	csv	4612	8
claim_e_execution_mode_latency_dir	claim_e	results/method4_claim_e_execution_mode_latency_20260704/analysis_20260704_225847	yes	dir	NA	3
claim_e_execution_mode_latency_matrix_csv	claim_e	results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_execution_mode_latency_matrix.csv	yes	csv	5044	12
claim_e_execution_mode_latency_script	claim_e	tools/method4_claim_e_execution_mode_latency.py	yes	py	19643	NA
claim_e_execution_mode_latency_summary_csv	claim_e	results/method4_claim_e_execution_mode_latency_20260704/analysis_20260704_225847/claim_e_execution_mode_latency_summary.csv	yes	csv	13414	36
claim_e_host_interface_bytes_derived_deltas_csv	claim_e	results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_host_interface_bytes_deltas.csv	yes	csv	5576	10
claim_e_host_interface_bytes_derived_summary_csv	claim_e	results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_host_interface_bytes_role_summary.csv	yes	csv	11586	15
claim_e_host_interface_bytes_dir	claim_e	results/method4_claim_e_host_interface_bytes_20260705/analysis_20260705_061525	yes	dir	NA	5
claim_e_host_interface_bytes_interfaces_csv	claim_e	results/method4_claim_e_host_interface_bytes_20260705/analysis_20260705_061525/claim_e_host_interface_bytes_interfaces.csv	yes	csv	40383	189
claim_e_host_interface_bytes_script	claim_e	tools/method4_claim_e_host_interface_bytes.py	yes	py	14241	NA
claim_e_host_interface_bytes_summary_csv	claim_e	results/method4_claim_e_host_interface_bytes_20260705/analysis_20260705_061525/claim_e_host_interface_bytes_summary.csv	yes	csv	10927	45
claim_e_host_interface_latency_summary_csv	claim_e	results/method4_claim_e_host_interface_bytes_20260705/analysis_20260705_061525/claim_e_host_interface_latency_summary.csv	yes	csv	4202	9
claim_e_packet_capture_bridge_deltas_csv	claim_e	results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_packet_capture_bridge_deltas.csv	yes	csv	1504	2
claim_e_packet_capture_bridge_summary_csv	claim_e	results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_packet_capture_bridge_summary.csv	yes	csv	5888	12
claim_e_packet_capture_bridge_totals_csv	claim_e	results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_packet_capture_bridge_totals.csv	yes	csv	2412	3
claim_e_packet_capture_dir	claim_e	results/method4_claim_e_packet_capture_20260706/per_variant_repeats3_20260706_122755	yes	dir	NA	3
claim_e_packet_capture_latency_summary_csv	claim_e	results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_packet_capture_latency_summary.csv	yes	csv	1376	3
claim_e_packet_capture_manifest_json	claim_e	results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_packet_capture_manifest.json	yes	json	1154	15
claim_e_packet_capture_physical_nic_execution_summary_csv	claim_e	results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_packet_capture_physical_nic_execution_summary.csv	yes	csv	1310	3
claim_e_packet_capture_physical_nic_manifest_json	claim_e	results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_packet_capture_physical_nic_manifest.json	yes	json	2533	28
claim_e_packet_capture_physical_nic_raw_execution_batches_csv	claim_e	results/method4_claim_e_packet_capture_physical_nic_20260707/all_variants_repeats3_preload_20260707_082833/execution_modes/analysis_20260707_082835/claim_e_execution_mode_latency_batches.csv	yes	csv	32745	135
claim_e_packet_capture_physical_nic_raw_execution_summary_csv	claim_e	results/method4_claim_e_packet_capture_physical_nic_20260707/all_variants_repeats3_preload_20260707_082833/execution_modes/analysis_20260707_082835/claim_e_execution_mode_latency_summary.csv	yes	csv	3824	9
claim_e_packet_capture_physical_nic_run_root	claim_e	results/method4_claim_e_packet_capture_physical_nic_20260707/all_variants_repeats3_preload_20260707_082833	yes	dir	NA	8
claim_e_packet_capture_physical_nic_summary_csv	claim_e	results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_packet_capture_physical_nic_summary.csv	yes	csv	1056	2
claim_e_packet_capture_physical_nic_totals_csv	claim_e	results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_packet_capture_physical_nic_totals.csv	yes	csv	1455	2
claim_e_packet_capture_script	claim_e	tools/method4_claim_e_packet_capture_summary.py	yes	py	10107	NA
claim_e_payload_bytes_batches_csv	claim_e	results/method4_claim_e_payload_bytes_20260705/analysis_20260705_014703/claim_e_payload_bytes_batches.csv	yes	csv	32723	339
claim_e_payload_bytes_derived_batches_csv	claim_e	results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_payload_bytes_batches.csv	yes	csv	32723	339
claim_e_payload_bytes_derived_summary_csv	claim_e	results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_payload_bytes_summary.csv	yes	csv	3422	12
claim_e_payload_bytes_dir	claim_e	results/method4_claim_e_payload_bytes_20260705/analysis_20260705_014703	yes	dir	NA	3
claim_e_payload_bytes_script	claim_e	tools/method4_claim_e_payload_bytes.py	yes	py	15193	NA
claim_e_payload_bytes_summary_csv	claim_e	results/method4_claim_e_payload_bytes_20260705/analysis_20260705_014703/claim_e_payload_bytes_summary.csv	yes	csv	3422	12
claim_e_planning_time_derived_summary_csv	claim_e	results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_planning_time_summary.csv	yes	csv	1404	3
claim_e_planning_time_dir	claim_e	results/method4_claim_e_planning_time_20260705/analysis_20260705_021347	yes	dir	NA	2
claim_e_planning_time_script	claim_e	tools/method4_claim_e_planning_time.py	yes	py	11982	NA
claim_e_planning_time_summary_csv	claim_e	results/method4_claim_e_planning_time_20260705/analysis_20260705_021347/claim_e_planning_time_summary.csv	yes	csv	1404	3
claim_e_process_rss_batches_csv	claim_e	results/method4_claim_e_process_rss_20260705/analysis_20260705_042151/claim_e_container_overhead_batches.csv	yes	csv	32728	135
claim_e_process_rss_derived_deltas_csv	claim_e	results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_process_rss_deltas.csv	yes	csv	1478	2
claim_e_process_rss_derived_summary_csv	claim_e	results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_process_rss_summary.csv	yes	csv	1907	3
claim_e_process_rss_dir	claim_e	results/method4_claim_e_process_rss_20260705/analysis_20260705_042151	yes	dir	NA	4
claim_e_process_rss_samples_csv	claim_e	results/method4_claim_e_process_rss_20260705/analysis_20260705_042151/claim_e_container_overhead_samples.csv	yes	csv	28715	144
claim_e_process_rss_summary_csv	claim_e	results/method4_claim_e_process_rss_20260705/analysis_20260705_042151/claim_e_container_overhead_summary.csv	yes	csv	8442	9
claim_e_qdrant_metrics_derived_deltas_csv	claim_e	results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_qdrant_metrics_deltas.csv	yes	csv	4736	2
claim_e_qdrant_metrics_derived_summary_csv	claim_e	results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_qdrant_metrics_summary.csv	yes	csv	7273	3
claim_e_qdrant_metrics_dir	claim_e	results/method4_claim_e_qdrant_metrics_20260705/analysis_20260705_073436	yes	dir	NA	4
claim_e_qdrant_metrics_raw_batches_csv	claim_e	results/method4_claim_e_qdrant_metrics_20260705/analysis_20260705_073436/claim_e_qdrant_metrics_batches.csv	yes	csv	32714	135
claim_e_qdrant_metrics_raw_metadata_json	claim_e	results/method4_claim_e_qdrant_metrics_20260705/analysis_20260705_073436/run_metadata.json	yes	json	1780	19
claim_e_qdrant_metrics_raw_samples_csv	claim_e	results/method4_claim_e_qdrant_metrics_20260705/analysis_20260705_073436/claim_e_qdrant_metrics_samples.csv	yes	csv	130751	1026
claim_e_qdrant_metrics_raw_summary_csv	claim_e	results/method4_claim_e_qdrant_metrics_20260705/analysis_20260705_073436/claim_e_qdrant_metrics_summary.csv	yes	csv	6677	9
claim_e_qdrant_metrics_script	claim_e	tools/method4_claim_e_qdrant_metrics.py	yes	py	18697	NA
claim_e_wire_bytes_batches_csv	claim_e	results/method4_claim_e_wire_bytes_20260705/analysis_20260705_020000/claim_e_wire_bytes_batches.csv	yes	csv	9445	45
claim_e_wire_bytes_derived_batches_csv	claim_e	results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_wire_bytes_batches.csv	yes	csv	9445	45
claim_e_wire_bytes_derived_summary_csv	claim_e	results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_wire_bytes_summary.csv	yes	csv	1655	3
claim_e_wire_bytes_dir	claim_e	results/method4_claim_e_wire_bytes_20260705/analysis_20260705_020000	yes	dir	NA	3
claim_e_wire_bytes_script	claim_e	tools/method4_claim_e_wire_bytes.py	yes	py	23907	NA
claim_e_wire_bytes_summary_csv	claim_e	results/method4_claim_e_wire_bytes_20260705/analysis_20260705_020000/claim_e_wire_bytes_summary.csv	yes	csv	1655	3
claim_evidence_audit_csv	claim_e	results/method4_claim_coverage_20260704/claim_evidence_audit.csv	yes	csv	44372	8
claim_f_direct_peer_premerge_baseline_summary_json	claim_f	results/qdrant_goal_recall_idea_095_direct_peer_premerge_ab_baseline/20260603_080131/summary.json	yes	json	4072	41
claim_f_direct_peer_premerge_premerged_summary_json	claim_f	results/qdrant_goal_recall_idea_095_direct_peer_premerge_ab_premerged/20260603_080247/summary.json	yes	json	4068	41
claim_f_premerge_batch_latency_batches_csv	claim_f	results/method4_claim_f_premerge_batch_latency_20260704/analysis_20260704_222700/claim_f_premerge_batch_latency_batches.csv	yes	csv	169013	1017
claim_f_premerge_batch_latency_deltas_csv	claim_f	results/method4_claim_coverage_20260704/derived_claim_tables/claim_f_premerge_batch_latency_deltas.csv	yes	csv	2225	8
claim_f_premerge_batch_latency_dir	claim_f	results/method4_claim_f_premerge_batch_latency_20260704/analysis_20260704_222700	yes	dir	NA	3
claim_f_premerge_batch_latency_matrix_csv	claim_f	results/method4_claim_coverage_20260704/derived_claim_tables/claim_f_premerge_batch_latency_matrix.csv	yes	csv	6437	12
claim_f_premerge_batch_latency_script	claim_f	tools/method4_claim_f_premerge_batch_latency.py	yes	py	21259	NA
claim_f_premerge_batch_latency_summary_csv	claim_f	results/method4_claim_f_premerge_batch_latency_20260704/analysis_20260704_222700/claim_f_premerge_batch_latency_summary.csv	yes	csv	12767	36
claim_f_server_peer_premerge_disabled_fresh_summary_json	claim_f	results/qdrant_goal_recall_idea_095_server_peer_premerge_disabled_fresh/20260603_092949/summary.json	yes	json	3988	41
claim_f_server_peer_premerge_enabled_10k_final_metrics_csv	claim_f	results/qdrant_goal_recall_idea_095_server_peer_premerge_10k/20260603_090738/final_metrics.csv	yes	csv	363	1
claim_f_server_peer_premerge_enabled_10k_summary_json	claim_f	results/qdrant_goal_recall_idea_095_server_peer_premerge_10k/20260603_090738/summary.json	yes	json	3956	41
claim_f_server_peer_premerge_enabled_fresh_summary_json	claim_f	results/qdrant_goal_recall_idea_095_server_peer_premerge_fresh/20260603_092757/summary.json	yes	json	3973	41
claim_f_server_peer_premerge_retained_summary_json	claim_f	results/qdrant_goal_recall_idea_095_server_peer_premerge/20260603_090530/summary.json	yes	json	4004	41
claim_f_worker_local_premerge_csv	claim_f	results/method4_claim_coverage_20260704/derived_claim_tables/claim_f_worker_local_premerge.csv	yes	csv	2151	5
claim_g_evidence_dir	claim_g	results/method4_claim_g_evidence_20260704	yes	dir	NA	10
claim_g_evidence_manifest_json	claim_g	results/method4_claim_g_evidence_20260704/source_manifest.json	yes	json	2727	12
claim_g_matched_method4_aware_deploy_dir	claim_g	results/method4_claim_g_matched_layout_deploy_20260704/20260704_170511	yes	dir	NA	3
claim_g_matched_method4_aware_deploy_metadata_json	claim_g	results/method4_claim_g_matched_layout_deploy_20260704/20260704_170511/run_metadata.json	yes	json	1833	15
claim_g_matched_size_balanced_deploy_dir	claim_g	results/method4_claim_g_matched_layout_deploy_20260704/20260704_171202	yes	dir	NA	3
claim_g_matched_size_balanced_deploy_metadata_json	claim_g	results/method4_claim_g_matched_layout_deploy_20260704/20260704_171202/run_metadata.json	yes	json	1837	15
claim_g_offline_physical_layout_raw_deltas_csv	claim_g	results/method4_claim_g_physical_layout_20260704/analysis_20260704_164039/placement_offline_deltas.csv	yes	csv	1646	6
claim_g_offline_physical_layout_raw_dir	claim_g	results/method4_claim_g_physical_layout_20260704/analysis_20260704_164039	yes	dir	NA	5
claim_g_offline_physical_layout_raw_metadata_json	claim_g	results/method4_claim_g_physical_layout_20260704/analysis_20260704_164039/run_metadata.json	yes	json	1087	19
claim_g_offline_physical_layout_raw_summary_csv	claim_g	results/method4_claim_g_physical_layout_20260704/analysis_20260704_164039/placement_offline_summary.csv	yes	csv	3976	9
claim_g_offline_placement_deltas_csv	claim_g	results/method4_claim_coverage_20260704/derived_claim_tables/claim_g_offline_placement_deltas.csv	yes	csv	1646	6
claim_g_offline_placement_summary_csv	claim_g	results/method4_claim_g_evidence_20260704/offline_placement_summary.csv	yes	csv	3976	9
claim_g_online_batch_latency_deltas_csv	claim_g	results/method4_claim_coverage_20260704/derived_claim_tables/claim_g_online_batch_latency_deltas.csv	yes	csv	1632	4
claim_g_online_batch_latency_raw_dir	claim_g	results/method4_claim_g_batch_latency_20260704/analysis_20260704_172015	yes	dir	NA	3
claim_g_online_batch_latency_raw_metadata_json	claim_g	results/method4_claim_g_batch_latency_20260704/analysis_20260704_172015/run_metadata.json	yes	json	1307	17
claim_g_online_batch_latency_raw_per_batch_csv	claim_g	results/method4_claim_g_batch_latency_20260704/analysis_20260704_172015/batch_latency_per_batch.csv	yes	csv	48687	405
claim_g_online_batch_latency_raw_runs_csv	claim_g	results/method4_claim_g_batch_latency_20260704/analysis_20260704_172015/batch_latency_runs.csv	yes	csv	4702	18
claim_g_online_batch_latency_summary_csv	claim_g	results/method4_claim_coverage_20260704/derived_claim_tables/claim_g_online_batch_latency_summary.csv	yes	csv	2871	6
claim_g_online_concurrency8_latency_deltas_csv	claim_g	results/method4_claim_g_evidence_20260704/online_concurrency8_latency_deltas.csv	yes	csv	762	1
claim_g_online_concurrency8_latency_summary_csv	claim_g	results/method4_claim_g_evidence_20260704/online_concurrency8_latency_summary.csv	yes	csv	1101	2
claim_g_online_concurrency8_raw_dir	claim_g	results/method4_claim_g_online_latency_20260704/analysis_20260704_165643	yes	dir	NA	3
claim_g_online_concurrency8_raw_metadata_json	claim_g	results/method4_claim_g_online_latency_20260704/analysis_20260704_165643/run_metadata.json	yes	json	1181	19
claim_g_online_concurrency8_raw_summary_csv	claim_g	results/method4_claim_g_online_latency_20260704/analysis_20260704_165643/online_single_query_latency_summary.csv	yes	csv	1545	6
claim_g_online_invalid_route_mismatch_diagnostic_dir	claim_g	results/method4_claim_g_online_latency_20260704/analysis_20260704_164441	yes	dir	NA	3
claim_g_online_low_pressure_diagnostic_dir	claim_g	results/method4_claim_g_online_latency_20260704/analysis_20260704_165135	yes	dir	NA	3
claim_g_physical_layout_doc	claim_g	docs/experiments/2026-07-04-method4-claim-g-physical-layout-evidence.md	yes	md	10931	NA
claim_g_raw_provenance_audit_csv	claim_g	results/method4_claim_coverage_20260704/derived_claim_tables/claim_g_raw_provenance_audit_20260708.csv	yes	csv	7630	22
claim_h_semantic_audit_dir	claim_h	results/method4_claim_h_semantic_audit_20260704/audit_20260704_210845	yes	dir	NA	4
claim_h_semantic_audit_summary_csv	claim_h	results/method4_claim_h_semantic_audit_20260704/audit_20260704_210845/semantic_invariant_summary.csv	yes	csv	673	9
claim_requirements_csv	claim	results/method4_claim_coverage_20260704/claim_requirements.csv	yes	csv	8223	8
claim_support_data_index_csv	claim_support	results/method4_claim_coverage_20260704/derived_claim_tables/claim_support_data_index.csv	yes	csv	66620	47
crosscut_cost_analysis_dir	crosscut	results/method4_claim_cost_analysis_20260704/analysis_20260704_232343	yes	dir	NA	3
crosscut_cost_analysis_script	crosscut	tools/method4_claim_cost_analysis.py	yes	py	22999	NA
crosscut_cost_main_collections_csv	crosscut	results/method4_claim_coverage_20260704/derived_claim_tables/crosscut_cost_main_collections.csv	yes	csv	5738	9
crosscut_cost_storage_du_csv	crosscut	results/method4_claim_coverage_20260704/derived_claim_tables/crosscut_cost_storage_du.csv	yes	csv	11042	76
crosscut_request_candidate_pressure_proxy_csv	crosscut	results/method4_claim_coverage_20260704/derived_claim_tables/crosscut_request_candidate_pressure_proxy.csv	yes	csv	10468	23
crosscut_request_candidate_pressure_proxy_script	crosscut	tools/method4_claim_cost_proxy_metrics.py	yes	py	6791	NA
crosscut_shard_scaling_latency_deltas_csv	crosscut	results/method4_claim_coverage_20260704/derived_claim_tables/crosscut_shard_scaling_latency_095_deltas.csv	yes	csv	6073	9
crosscut_shard_scaling_latency_kmeans_s31_dir	crosscut	results/method4_shard_scaling_latency_20260705/kmeans_s31/analysis_20260705_034019	yes	dir	NA	3
crosscut_shard_scaling_latency_kmeans_s46_dir	crosscut	results/method4_shard_scaling_latency_20260705/kmeans_s46/analysis_20260705_034149	yes	dir	NA	3
crosscut_shard_scaling_latency_matrix_csv	crosscut	results/method4_claim_coverage_20260704/derived_claim_tables/crosscut_shard_scaling_latency_095.csv	yes	csv	5260	6
crosscut_shard_scaling_latency_naive_s31_dir	crosscut	results/method4_shard_scaling_latency_20260705/naive_s31/analysis_20260705_031951	yes	dir	NA	3
crosscut_shard_scaling_latency_naive_s46_dir	crosscut	results/method4_shard_scaling_latency_20260705/naive_s46/analysis_20260705_032129	yes	dir	NA	3
crosscut_shard_scaling_latency_orion_s31_dir	crosscut	results/method4_shard_scaling_latency_20260705/orion_s31/analysis_20260705_031624	yes	dir	NA	3
crosscut_shard_scaling_latency_orion_s46_dir	crosscut	results/method4_shard_scaling_latency_20260705/orion_s46/analysis_20260705_031806	yes	dir	NA	3
crosscut_shard_scaling_latency_root	crosscut	results/method4_shard_scaling_latency_20260705	yes	dir	NA	6
crosscut_shard_scaling_latency_runner	crosscut	tools/method4_claim_d_high_recall_latency.py	yes	py	27786	NA
current_environment_preflight_20260707_build_smoke_csv	current	results/method4_claim_coverage_20260704/derived_claim_tables/current_environment_preflight_20260707_build_smoke.csv	yes	csv	571	1
current_environment_preflight_20260707_csv	current	results/method4_claim_coverage_20260704/derived_claim_tables/current_environment_preflight_20260707.csv	yes	csv	3837	6
current_environment_preflight_csv	current	results/method4_claim_coverage_20260704/derived_claim_tables/current_environment_preflight_20260706.csv	yes	csv	4042	9
current_experiment_readiness_recheck_20260707_csv	current	results/method4_claim_coverage_20260704/derived_claim_tables/current_experiment_readiness_recheck_20260707.csv	yes	csv	5251	9
current_experiment_readiness_recheck_20260708_csv	current	results/method4_claim_coverage_20260704/derived_claim_tables/current_experiment_readiness_recheck_20260708.csv	yes	csv	6820	11
current_hard_gap_blocker_audit_20260707_csv	current	results/method4_claim_coverage_20260704/derived_claim_tables/current_hard_gap_blocker_audit_20260707.csv	yes	csv	8506	6
current_hard_gap_blocker_audit_20260708_post_worker_count_csv	current	results/method4_claim_coverage_20260704/derived_claim_tables/current_hard_gap_blocker_audit_20260708_post_worker_count.csv	yes	csv	2264	5
current_hard_gap_blocker_audit_20260709_post_claim_a_csv	current	results/method4_claim_coverage_20260704/derived_claim_tables/current_hard_gap_blocker_audit_20260709_post_claim_a.csv	yes	csv	2391	4
current_hard_gap_blocker_audit_20260709_post_multiassign_csv	current	results/method4_claim_coverage_20260704/derived_claim_tables/current_hard_gap_blocker_audit_20260709_post_multiassign.csv	yes	csv	2590	4
current_hard_gap_blocker_audit_csv	current	results/method4_claim_coverage_20260704/derived_claim_tables/current_hard_gap_blocker_audit_20260706.csv	yes	csv	6140	6
current_unblock_status_20260707_csv	current	results/method4_claim_coverage_20260704/derived_claim_tables/current_unblock_status_20260707.csv	yes	csv	6945	13
current_unblock_status_20260708_csv	current	results/method4_claim_coverage_20260704/derived_claim_tables/current_unblock_status_20260708.csv	yes	csv	6450	14
derived_tables_dir	derived	results/method4_claim_coverage_20260704/derived_claim_tables	yes	dir	NA	173
docker_cleanup_opportunity_audit_20260708_csv	docker	results/method4_claim_coverage_20260704/derived_claim_tables/docker_cleanup_opportunity_audit_20260708.csv	yes	csv	2645	7
final_evidence_document_integrity_audit_20260708_csv	final	results/method4_claim_coverage_20260704/derived_claim_tables/final_evidence_document_integrity_audit_20260708.csv	yes	csv	16739	38
final_evidence_document_integrity_audit_csv	final	results/method4_claim_coverage_20260704/derived_claim_tables/final_evidence_document_integrity_audit_20260707.csv	yes	csv	8279	15
formal_experiment_prerequisite_audit_20260708_csv	formal	results/method4_claim_coverage_20260704/derived_claim_tables/formal_experiment_prerequisite_audit_20260708.csv	yes	csv	8801	21
goal_blocked_threshold_audit_20260708_csv	goal	results/method4_claim_coverage_20260704/derived_claim_tables/goal_blocked_threshold_audit_20260708.csv	yes	csv	4272	8
goal_completion_audit_20260707_csv	goal	results/method4_claim_coverage_20260704/derived_claim_tables/goal_completion_audit_20260707.csv	yes	csv	76913	23
goal_completion_audit_20260708_csv	goal	results/method4_claim_coverage_20260704/derived_claim_tables/goal_completion_audit_20260708.csv	yes	csv	76552	23
goal_completion_audit_20260709_csv	goal	results/method4_claim_coverage_20260704/derived_claim_tables/goal_completion_audit_20260709.csv	yes	csv	2111	2
goal_completion_audit_20260709_post_multiassign_csv	goal	results/method4_claim_coverage_20260704/derived_claim_tables/goal_completion_audit_20260709_post_multiassign.csv	yes	csv	3899	3
goal_completion_audit_csv	goal	results/method4_claim_coverage_20260704/derived_claim_tables/goal_completion_audit_20260706.csv	yes	csv	52218	23
hard_gap_artifact_audit_csv	hard	results/method4_claim_coverage_20260704/derived_claim_tables/hard_gap_artifact_audit_20260705.csv	yes	csv	5046	3
l2_distance_support_script	l2	tools/qdrant_two_level_routing_experiment.py	yes	py	221027	NA
l2_harness_readiness_smoke_csv	l2	results/method4_claim_coverage_20260704/derived_claim_tables/l2_harness_readiness_smoke_20260705.csv	yes	csv	738	1
l2_harness_readiness_smoke_dir	l2	results/method4_l2_harness_smoke_20260705/20260705_004225	yes	dir	NA	6
l2_harness_readiness_smoke_summary_json	l2	results/method4_l2_harness_smoke_20260705/20260705_004225/summary.json	yes	json	3421	50
l2_reduced_sift100k_dynamic_vs_fixed_coarse_dir	l2	results/method4_l2_reduced_sift100k_latency_20260705/dynamic_vs_fixed_nofission/analysis_20260705_051254	yes	dir	NA	3
l2_reduced_sift100k_dynamic_vs_fixed_ef_deltas_csv	l2	results/method4_claim_coverage_20260704/derived_claim_tables/l2_reduced_sift100k_dynamic_vs_fixed_ef_deltas.csv	yes	csv	5184	8
l2_reduced_sift100k_dynamic_vs_fixed_ef_matrix_csv	l2	results/method4_claim_coverage_20260704/derived_claim_tables/l2_reduced_sift100k_dynamic_vs_fixed_ef_matrix.csv	yes	csv	7394	11
l2_reduced_sift100k_dynamic_vs_fixed_fine_dir	l2	results/method4_l2_reduced_sift100k_latency_20260705/dynamic_vs_fixed_nofission_fine/analysis_20260705_051329	yes	dir	NA	3
l2_reduced_sift100k_full_fission_latency_deltas_csv	l2	results/method4_claim_coverage_20260704/derived_claim_tables/l2_reduced_sift100k_full_fission_latency_deltas.csv	yes	csv	893	1
l2_reduced_sift100k_full_fission_latency_matrix_csv	l2	results/method4_claim_coverage_20260704/derived_claim_tables/l2_reduced_sift100k_full_fission_latency_matrix.csv	yes	csv	1328	2
l2_reduced_sift100k_full_fission_method4_dir	l2	results/method4_l2_reduced_sift100k_latency_20260706/full_orion_method4_only/analysis_20260706_125222	yes	dir	NA	3
l2_reduced_sift100k_full_fission_method4_summary_csv	l2	results/method4_l2_reduced_sift100k_latency_20260706/full_orion_method4_only/analysis_20260706_125222/claim_d_high_recall_latency_summary.csv	yes	csv	1105	3
l2_reduced_sift100k_full_fission_naive_dir	l2	results/method4_l2_reduced_sift100k_latency_20260706/naive_only/analysis_20260706_125245	yes	dir	NA	3
l2_reduced_sift100k_full_fission_naive_summary_csv	l2	results/method4_l2_reduced_sift100k_latency_20260706/naive_only/analysis_20260706_125245/claim_d_high_recall_latency_summary.csv	yes	csv	1068	3
l2_reduced_sift100k_full_orion_dynamic_vs_fixed_dir	l2	results/method4_l2_reduced_sift100k_latency_20260705/full_orion_dynamic_vs_fixed/analysis_20260705_051940	yes	dir	NA	3
l2_reduced_sift100k_latency_deltas_csv	l2	results/method4_claim_coverage_20260704/derived_claim_tables/l2_reduced_sift100k_latency_deltas.csv	yes	csv	1940	3
l2_reduced_sift100k_latency_dir	l2	results/method4_l2_reduced_sift100k_latency_20260705/analysis_20260705_010022	yes	dir	NA	3
l2_reduced_sift100k_latency_matrix_csv	l2	results/method4_claim_coverage_20260704/derived_claim_tables/l2_reduced_sift100k_latency_matrix.csv	yes	csv	3041	6
l2_reduced_sift100k_main_comparison_csv	l2	results/method4_claim_coverage_20260704/derived_claim_tables/l2_reduced_sift100k_main_comparison.csv	yes	csv	1837	3
l2_reduced_sift100k_naive_3repeat_dir	l2	results/method4_l2_reduced_sift100k_20260705/naive_3repeat/20260705_005944	yes	dir	NA	6
l2_reduced_sift100k_orion_nofission_dir	l2	results/method4_l2_reduced_sift100k_20260705/20260705_005814	yes	dir	NA	6
l2_reduced_sift100k_simple_3repeat_dir	l2	results/method4_l2_reduced_sift100k_20260705/simple_3repeat/20260705_010000	yes	dir	NA	6
l2_reduced_sift100k_simple_latency_smoke_dir	l2	results/method4_l2_reduced_sift100k_latency_smoke_20260705/simple/analysis_20260705_035401	yes	dir	NA	3
l2_reduced_sift100k_simple_vs_naive_latency_dir	l2	results/method4_l2_reduced_sift100k_latency_20260705/simple_vs_naive/analysis_20260705_035458	yes	dir	NA	3
plan_document	plan	docs/experiments/2026-06-16-method4-distributed-submission-experiment-plan.md	yes	md	43175	NA
plan_execution_document	plan	docs/superpowers/plans/2026-07-04-method4-claim-coverage-completion.md	yes	md	7766	NA
remaining_experiment_execution_queue_csv	remaining	results/method4_claim_coverage_20260704/derived_claim_tables/remaining_experiment_execution_queue.csv	yes	csv	10435	4
remaining_experiment_gaps_csv	remaining	results/method4_claim_coverage_20260704/remaining_experiment_gaps.csv	yes	csv	12079	7
robustness_080_085_method4_vs_naive_deltas_csv	robustness	results/method4_claim_coverage_20260704/derived_claim_tables/robustness_080_085_method4_vs_naive_deltas.csv	yes	csv	5554	8
robustness_080_085_original_batches_csv	robustness	results/method4_robustness_080_085_original_20260705/analysis_20260705_070956/claim_d_high_recall_latency_batches.csv	yes	csv	49203	360
robustness_080_085_original_dir	robustness	results/method4_robustness_080_085_original_20260705/analysis_20260705_070956	yes	dir	NA	3
robustness_080_085_original_summary_csv	robustness	results/method4_robustness_080_085_original_20260705/analysis_20260705_070956/claim_d_high_recall_latency_summary.csv	yes	csv	3600	12
robustness_080_085_query_order_multiseed_summary_csv	robustness	results/method4_claim_coverage_20260704/derived_claim_tables/robustness_080_085_query_order_multiseed_summary.csv	yes	csv	3552	4
robustness_080_085_query_order_overlay_csv	robustness	results/method4_claim_coverage_20260704/derived_claim_tables/robustness_080_085_query_order_overlay.csv	yes	csv	9435	12
robustness_080_085_sensitivity_deltas_csv	robustness	results/method4_claim_coverage_20260704/derived_claim_tables/robustness_080_085_sensitivity_deltas.csv	yes	csv	6857	12
robustness_080_085_shuffle_seed_20260709_batches_csv	robustness	results/method4_robustness_080_085_shuffle_20260709/analysis_20260705_071249/claim_d_high_recall_latency_batches.csv	yes	csv	49188	360
robustness_080_085_shuffle_seed_20260709_dir	robustness	results/method4_robustness_080_085_shuffle_20260709/analysis_20260705_071249	yes	dir	NA	3
robustness_080_085_shuffle_seed_20260709_summary_csv	robustness	results/method4_robustness_080_085_shuffle_20260709/analysis_20260705_071249/claim_d_high_recall_latency_summary.csv	yes	csv	3613	12
robustness_080_085_shuffle_seed_20260712_batches_csv	robustness	results/method4_robustness_080_085_shuffle_20260712/analysis_20260706_133627/claim_d_high_recall_latency_batches.csv	yes	csv	49215	360
robustness_080_085_shuffle_seed_20260712_dir	robustness	results/method4_robustness_080_085_shuffle_20260712/analysis_20260706_133627	yes	dir	NA	3
robustness_080_085_shuffle_seed_20260712_metadata_json	robustness	results/method4_robustness_080_085_shuffle_20260712/analysis_20260706_133627/run_metadata.json	yes	json	1386	29
robustness_080_085_shuffle_seed_20260712_summary_csv	robustness	results/method4_robustness_080_085_shuffle_20260712/analysis_20260706_133627/claim_d_high_recall_latency_summary.csv	yes	csv	3593	12
robustness_080_085_warmup_batches_csv	robustness	results/method4_robustness_080_085_warmup_20260705/analysis_20260705_071529/claim_d_high_recall_latency_batches.csv	yes	csv	49199	360
robustness_080_085_warmup_dir	robustness	results/method4_robustness_080_085_warmup_20260705/analysis_20260705_071529	yes	dir	NA	3
robustness_080_085_warmup_overlay_csv	robustness	results/method4_claim_coverage_20260704/derived_claim_tables/robustness_080_085_warmup_overlay.csv	yes	csv	6336	8
robustness_080_085_warmup_summary_csv	robustness	results/method4_robustness_080_085_warmup_20260705/analysis_20260705_071529/claim_d_high_recall_latency_summary.csv	yes	csv	3654	12
robustness_090_method4_vs_naive_deltas_csv	robustness	results/method4_claim_coverage_20260704/derived_claim_tables/robustness_090_method4_vs_naive_deltas.csv	yes	csv	2896	4
robustness_090_original_batches_csv	robustness	results/method4_robustness_090_original_20260705/analysis_20260705_065005/claim_d_high_recall_latency_batches.csv	yes	csv	24825	180
robustness_090_original_dir	robustness	results/method4_robustness_090_original_20260705/analysis_20260705_065005	yes	dir	NA	3
robustness_090_original_summary_csv	robustness	results/method4_robustness_090_original_20260705/analysis_20260705_065005/claim_d_high_recall_latency_summary.csv	yes	csv	1980	6
robustness_090_query_order_multiseed_summary_csv	robustness	results/method4_claim_coverage_20260704/derived_claim_tables/robustness_090_query_order_multiseed_summary.csv	yes	csv	2084	2
robustness_090_query_order_overlay_csv	robustness	results/method4_claim_coverage_20260704/derived_claim_tables/robustness_090_query_order_overlay.csv	yes	csv	4941	6
robustness_090_sensitivity_deltas_csv	robustness	results/method4_claim_coverage_20260704/derived_claim_tables/robustness_090_sensitivity_deltas.csv	yes	csv	3387	6
robustness_090_shuffle_seed_20260708_batches_csv	robustness	results/method4_robustness_090_shuffle_20260708/analysis_20260705_065219/claim_d_high_recall_latency_batches.csv	yes	csv	24840	180
robustness_090_shuffle_seed_20260708_dir	robustness	results/method4_robustness_090_shuffle_20260708/analysis_20260705_065219	yes	dir	NA	3
robustness_090_shuffle_seed_20260708_summary_csv	robustness	results/method4_robustness_090_shuffle_20260708/analysis_20260705_065219/claim_d_high_recall_latency_summary.csv	yes	csv	2014	6
robustness_090_shuffle_seed_20260711_batches_csv	robustness	results/method4_robustness_090_shuffle_20260711/analysis_20260706_132643/claim_d_high_recall_latency_batches.csv	yes	csv	24853	180
robustness_090_shuffle_seed_20260711_dir	robustness	results/method4_robustness_090_shuffle_20260711/analysis_20260706_132643	yes	dir	NA	3
robustness_090_shuffle_seed_20260711_metadata_json	robustness	results/method4_robustness_090_shuffle_20260711/analysis_20260706_132643/run_metadata.json	yes	json	1335	29
robustness_090_shuffle_seed_20260711_summary_csv	robustness	results/method4_robustness_090_shuffle_20260711/analysis_20260706_132643/claim_d_high_recall_latency_summary.csv	yes	csv	1979	6
robustness_090_warmup_batches_csv	robustness	results/method4_robustness_090_warmup_20260705/analysis_20260705_065435/claim_d_high_recall_latency_batches.csv	yes	csv	24861	180
robustness_090_warmup_dir	robustness	results/method4_robustness_090_warmup_20260705/analysis_20260705_065435	yes	dir	NA	3
robustness_090_warmup_overlay_csv	robustness	results/method4_claim_coverage_20260704/derived_claim_tables/robustness_090_warmup_overlay.csv	yes	csv	3424	4
robustness_090_warmup_summary_csv	robustness	results/method4_robustness_090_warmup_20260705/analysis_20260705_065435/claim_d_high_recall_latency_summary.csv	yes	csv	2018	6
robustness_095_curve_config_manifest_json	robustness	results/method4_claim_coverage_20260704/derived_claim_tables/robustness_095_curve_config_manifest.json	yes	json	3645	13
robustness_095_curve_config_method4_vs_naive_deltas_csv	robustness	results/method4_claim_coverage_20260704/derived_claim_tables/robustness_095_curve_config_method4_vs_naive_deltas.csv	yes	csv	3113	4
robustness_095_curve_config_original_dir	robustness	results/method4_robustness_095_curve_config_original_20260707/analysis_20260707_093724	yes	dir	NA	3
robustness_095_curve_config_overlay_csv	robustness	results/method4_claim_coverage_20260704/derived_claim_tables/robustness_095_curve_config_overlay.csv	yes	csv	8795	8
robustness_095_curve_config_query_order_multiseed_summary_csv	robustness	results/method4_claim_coverage_20260704/derived_claim_tables/robustness_095_curve_config_query_order_multiseed_summary.csv	yes	csv	1892	2
robustness_095_curve_config_sensitivity_deltas_csv	robustness	results/method4_claim_coverage_20260704/derived_claim_tables/robustness_095_curve_config_sensitivity_deltas.csv	yes	csv	3753	6
robustness_095_curve_config_shuffle_seed_20260714_dir	robustness	results/method4_robustness_095_curve_config_shuffle_20260707/analysis_20260707_093953	yes	dir	NA	3
robustness_095_curve_config_shuffle_seed_20260715_dir	robustness	results/method4_robustness_095_curve_config_shuffle2_20260707/analysis_20260707_094525	yes	dir	NA	3
robustness_095_curve_config_warmup_dir	robustness	results/method4_robustness_095_curve_config_warmup_20260707/analysis_20260707_094220	yes	dir	NA	3
robustness_095_method4_vs_naive_deltas_csv	robustness	results/method4_claim_coverage_20260704/derived_claim_tables/robustness_095_method4_vs_naive_deltas.csv	yes	csv	2944	4
robustness_095_original_batches_csv	robustness	results/method4_robustness_095_original_20260705/analysis_20260705_063443/claim_d_high_recall_latency_batches.csv	yes	csv	24841	180
robustness_095_original_dir	robustness	results/method4_robustness_095_original_20260705/analysis_20260705_063443	yes	dir	NA	3
robustness_095_original_summary_csv	robustness	results/method4_robustness_095_original_20260705/analysis_20260705_063443/claim_d_high_recall_latency_summary.csv	yes	csv	2044	6
robustness_095_query_order_multiseed_summary_csv	robustness	results/method4_claim_coverage_20260704/derived_claim_tables/robustness_095_query_order_multiseed_summary.csv	yes	csv	1824	2
robustness_095_query_order_overlay_csv	robustness	results/method4_claim_coverage_20260704/derived_claim_tables/robustness_095_query_order_overlay.csv	yes	csv	5005	6
robustness_095_sensitivity_deltas_csv	robustness	results/method4_claim_coverage_20260704/derived_claim_tables/robustness_095_sensitivity_deltas.csv	yes	csv	3462	6
robustness_095_shuffle_seed_20260707_batches_csv	robustness	results/method4_robustness_095_shuffle_20260707/analysis_20260705_063719/claim_d_high_recall_latency_batches.csv	yes	csv	24837	180
robustness_095_shuffle_seed_20260707_dir	robustness	results/method4_robustness_095_shuffle_20260707/analysis_20260705_063719	yes	dir	NA	3
robustness_095_shuffle_seed_20260707_summary_csv	robustness	results/method4_robustness_095_shuffle_20260707/analysis_20260705_063719/claim_d_high_recall_latency_summary.csv	yes	csv	2042	6
robustness_095_shuffle_seed_20260710_batches_csv	robustness	results/method4_robustness_095_shuffle_20260710/analysis_20260706_131447/claim_d_high_recall_latency_batches.csv	yes	csv	24867	180
robustness_095_shuffle_seed_20260710_dir	robustness	results/method4_robustness_095_shuffle_20260710/analysis_20260706_131447	yes	dir	NA	3
robustness_095_shuffle_seed_20260710_metadata_json	robustness	results/method4_robustness_095_shuffle_20260710/analysis_20260706_131447/run_metadata.json	yes	json	1337	29
robustness_095_shuffle_seed_20260710_summary_csv	robustness	results/method4_robustness_095_shuffle_20260710/analysis_20260706_131447/claim_d_high_recall_latency_summary.csv	yes	csv	2054	6
robustness_095_warmup_batches_csv	robustness	results/method4_robustness_095_warmup_20260705/analysis_20260705_063957/claim_d_high_recall_latency_batches.csv	yes	csv	24859	180
robustness_095_warmup_dir	robustness	results/method4_robustness_095_warmup_20260705/analysis_20260705_063957	yes	dir	NA	3
robustness_095_warmup_overlay_csv	robustness	results/method4_claim_coverage_20260704/derived_claim_tables/robustness_095_warmup_overlay.csv	yes	csv	3445	4
robustness_095_warmup_summary_csv	robustness	results/method4_robustness_095_warmup_20260705/analysis_20260705_063957/claim_d_high_recall_latency_summary.csv	yes	csv	1983	6
robustness_097_closest_manifest_json	robustness	results/method4_claim_coverage_20260704/derived_claim_tables/robustness_097_closest_manifest.json	yes	json	3619	13
robustness_097_closest_method4_vs_naive_deltas_csv	robustness	results/method4_claim_coverage_20260704/derived_claim_tables/robustness_097_closest_method4_vs_naive_deltas.csv	yes	csv	3165	4
robustness_097_closest_original_batches_csv	robustness	results/method4_claim_d_high_recall_latency_refine_20260704/analysis_20260704_220036/claim_d_high_recall_latency_batches.csv	yes	csv	61402	450
robustness_097_closest_original_dir	robustness	results/method4_claim_d_high_recall_latency_refine_20260704/analysis_20260704_220036	yes	dir	NA	3
robustness_097_closest_original_metadata_json	robustness	results/method4_claim_d_high_recall_latency_refine_20260704/analysis_20260704_220036/run_metadata.json	yes	json	951	16
robustness_097_closest_original_summary_csv	robustness	results/method4_claim_d_high_recall_latency_refine_20260704/analysis_20260704_220036/claim_d_high_recall_latency_summary.csv	yes	csv	4086	15
robustness_097_closest_overlay_csv	robustness	results/method4_claim_coverage_20260704/derived_claim_tables/robustness_097_closest_overlay.csv	yes	csv	8876	8
robustness_097_closest_query_order_multiseed_summary_csv	robustness	results/method4_claim_coverage_20260704/derived_claim_tables/robustness_097_closest_query_order_multiseed_summary.csv	yes	csv	1915	2
robustness_097_closest_sensitivity_deltas_csv	robustness	results/method4_claim_coverage_20260704/derived_claim_tables/robustness_097_closest_sensitivity_deltas.csv	yes	csv	3873	6
robustness_097_closest_shuffle_seed_20260716_batches_csv	robustness	results/method4_robustness_097_closest_shuffle_20260707/analysis_20260707_095706/claim_d_high_recall_latency_batches.csv	yes	csv	25660	180
robustness_097_closest_shuffle_seed_20260716_dir	robustness	results/method4_robustness_097_closest_shuffle_20260707/analysis_20260707_095706	yes	dir	NA	3
robustness_097_closest_shuffle_seed_20260716_metadata_json	robustness	results/method4_robustness_097_closest_shuffle_20260707/analysis_20260707_095706/run_metadata.json	yes	json	1348	29
robustness_097_closest_shuffle_seed_20260716_summary_csv	robustness	results/method4_robustness_097_closest_shuffle_20260707/analysis_20260707_095706/claim_d_high_recall_latency_summary.csv	yes	csv	2065	6
robustness_097_closest_shuffle_seed_20260717_batches_csv	robustness	results/method4_robustness_097_closest_shuffle2_20260707/analysis_20260707_095948/claim_d_high_recall_latency_batches.csv	yes	csv	25668	180
robustness_097_closest_shuffle_seed_20260717_dir	robustness	results/method4_robustness_097_closest_shuffle2_20260707/analysis_20260707_095948	yes	dir	NA	3
robustness_097_closest_shuffle_seed_20260717_metadata_json	robustness	results/method4_robustness_097_closest_shuffle2_20260707/analysis_20260707_095948/run_metadata.json	yes	json	1348	29
robustness_097_closest_shuffle_seed_20260717_summary_csv	robustness	results/method4_robustness_097_closest_shuffle2_20260707/analysis_20260707_095948/claim_d_high_recall_latency_summary.csv	yes	csv	2047	6
robustness_097_closest_warmup_batches_csv	robustness	results/method4_robustness_097_closest_warmup_20260707/analysis_20260707_100229/claim_d_high_recall_latency_batches.csv	yes	csv	25666	180
robustness_097_closest_warmup_dir	robustness	results/method4_robustness_097_closest_warmup_20260707/analysis_20260707_100229	yes	dir	NA	3
robustness_097_closest_warmup_metadata_json	robustness	results/method4_robustness_097_closest_warmup_20260707/analysis_20260707_100229/run_metadata.json	yes	json	1348	29
robustness_097_closest_warmup_summary_csv	robustness	results/method4_robustness_097_closest_warmup_20260707/analysis_20260707_100229/claim_d_high_recall_latency_summary.csv	yes	csv	2012	6
robustness_query_order_multiseed_summary_csv	robustness	results/method4_claim_coverage_20260704/derived_claim_tables/robustness_query_order_multiseed_summary.csv	yes	csv	1779	2
robustness_shuffle_batches_csv	robustness	results/method4_robustness_shuffle_20260705/analysis_20260705_030323/claim_d_high_recall_latency_batches.csv	yes	csv	25031	180
robustness_shuffle_dir	robustness	results/method4_robustness_shuffle_20260705/analysis_20260705_030323	yes	dir	NA	3
robustness_shuffle_overlay_csv	robustness	results/method4_claim_coverage_20260704/derived_claim_tables/robustness_shuffled_query_overlay.csv	yes	csv	4399	6
robustness_shuffle_overlay_deltas_csv	robustness	results/method4_claim_coverage_20260704/derived_claim_tables/robustness_shuffled_query_overlay_deltas.csv	yes	csv	3893	7
robustness_shuffle_seed_20260705_batches_csv	robustness	results/method4_robustness_shuffle_20260705/analysis_20260705_030323/claim_d_high_recall_latency_batches.csv	yes	csv	25031	180
robustness_shuffle_seed_20260705_dir	robustness	results/method4_robustness_shuffle_20260705/analysis_20260705_030323	yes	dir	NA	3
robustness_shuffle_seed_20260705_summary_csv	robustness	results/method4_robustness_shuffle_20260705/analysis_20260705_030323/claim_d_high_recall_latency_summary.csv	yes	csv	2022	6
robustness_shuffle_seed_20260706_batches_csv	robustness	results/method4_robustness_shuffle_20260706/analysis_20260705_045135/claim_d_high_recall_latency_batches.csv	yes	csv	25008	180
robustness_shuffle_seed_20260706_dir	robustness	results/method4_robustness_shuffle_20260706/analysis_20260705_045135	yes	dir	NA	3
robustness_shuffle_seed_20260706_summary_csv	robustness	results/method4_robustness_shuffle_20260706/analysis_20260705_045135/claim_d_high_recall_latency_summary.csv	yes	csv	1974	6
robustness_shuffle_summary_csv	robustness	results/method4_robustness_shuffle_20260705/analysis_20260705_030323/claim_d_high_recall_latency_summary.csv	yes	csv	2022	6
robustness_warmup_batches_csv	robustness	results/method4_robustness_warmup_20260705/analysis_20260705_044122/claim_d_high_recall_latency_batches.csv	yes	csv	25023	180
robustness_warmup_dir	robustness	results/method4_robustness_warmup_20260705/analysis_20260705_044122	yes	dir	NA	3
robustness_warmup_overlay_csv	robustness	results/method4_claim_coverage_20260704/derived_claim_tables/robustness_warmup_overlay.csv	yes	csv	3334	4
robustness_warmup_overlay_deltas_csv	robustness	results/method4_claim_coverage_20260704/derived_claim_tables/robustness_warmup_overlay_deltas.csv	yes	csv	3094	4
robustness_warmup_summary_csv	robustness	results/method4_robustness_warmup_20260705/analysis_20260705_044122/claim_d_high_recall_latency_summary.csv	yes	csv	2068	6
strict_ablation_live_collection_audit_20260709_post_multiassign_csv	strict	results/method4_claim_coverage_20260704/derived_claim_tables/strict_ablation_live_collection_audit_20260709_post_multiassign.csv	yes	csv	2425	5
strict_ablation_live_collection_audit_csv	strict	results/method4_claim_coverage_20260704/derived_claim_tables/strict_ablation_live_collection_audit_20260705.csv	yes	csv	3415	9
strict_multiassign_default_build_dir	strict_multiassign	results/method4_benchmark_matrix/method4_strict_multiassign_latency_rebuild_20260709/default_build_20260709_preload/20260709_105830	yes	dir	NA	5
strict_multiassign_default_latency_dir	strict_multiassign	results/method4_strict_ablation_multiassign_latency_20260709/default/analysis_20260709_110559	yes	dir	NA	3
strict_multiassign_latency_20260709_csv	strict_multiassign	results/method4_claim_coverage_20260704/derived_claim_tables/strict_multiassign_latency_20260709.csv	yes	csv	2899	3
strict_multiassign_latency_root_20260709	strict_multiassign	results/method4_strict_ablation_multiassign_latency_20260709	yes	dir	NA	3
strict_multiassign_rebuild_config_json	strict_multiassign	tools/benchmark_configs/method4_strict_multiassign_latency_rebuild_20260709.json	yes	json	2588	6
strict_multiassign_rebuild_root_20260709	strict_multiassign	results/method4_benchmark_matrix/method4_strict_multiassign_latency_rebuild_20260709	yes	dir	NA	5
strict_multiassign_single_build_dir	strict_multiassign	results/method4_benchmark_matrix/method4_strict_multiassign_latency_rebuild_20260709/single_build_20260709_preload/20260709_101308	yes	dir	NA	5
strict_multiassign_single_latency_dir	strict_multiassign	results/method4_strict_ablation_multiassign_latency_20260709/single/analysis_20260709_105604	yes	dir	NA	3
strict_multiassign_w2c2_build_dir	strict_multiassign	results/method4_benchmark_matrix/method4_strict_multiassign_latency_rebuild_20260709/w2c2_build_20260709_preload/20260709_110827	yes	dir	NA	5
strict_multiassign_w2c2_latency_dir	strict_multiassign	results/method4_strict_ablation_multiassign_latency_20260709/w2c2/analysis_20260709_111731	yes	dir	NA	3
unified_ablation_status_matrix_csv	unified	results/method4_claim_coverage_20260704/derived_claim_tables/unified_ablation_status_matrix.csv	yes	csv	8165	7
unified_ablation_status_script	unified	tools/method4_unified_ablation_status.py	yes	py	12619	NA
unincorporated_artifact_review_20260707_csv	unincorporated	results/method4_claim_coverage_20260704/derived_claim_tables/unincorporated_artifact_review_20260707.csv	yes	csv	7296	10
worker_count_live_deployment_audit_csv	worker_count	results/method4_claim_coverage_20260704/derived_claim_tables/worker_count_live_deployment_audit_20260705.csv	yes	csv	6938	13
worker_count_online_scaling_deltas_csv	worker_count	results/method4_claim_coverage_20260704/derived_claim_tables/worker_count_online_scaling_deltas_20260708.csv	yes	csv	8572	16
worker_count_online_scaling_docker_stats_csv	worker_count	results/method4_claim_coverage_20260704/derived_claim_tables/worker_count_docker_stats_summary_20260708.csv	yes	csv	3004	14
worker_count_online_scaling_manifest_json	worker_count	results/method4_claim_coverage_20260704/derived_claim_tables/worker_count_online_scaling_manifest_20260708.json	yes	json	5771	19
worker_count_online_scaling_raw_rows_csv	worker_count	results/method4_claim_coverage_20260704/derived_claim_tables/worker_count_online_scaling_raw_rows_20260708.csv	yes	csv	41577	48
worker_count_online_scaling_root	worker_count	results/method4_worker_count_online_scaling_20260708/	yes	dir	NA	5
worker_count_online_scaling_runner	worker_count	tools/method4_worker_count_online_scaling.py	yes	py	19150	NA
worker_count_online_scaling_summary_csv	worker_count	results/method4_claim_coverage_20260704/derived_claim_tables/worker_count_online_scaling_summary_20260708.csv	yes	csv	14775	16
worker_count_online_scaling_tests	worker_count	tests/tools/test_method4_worker_count_online_scaling.py	yes	py	3092	NA
excel_claim_support_workbook_xlsx	excel	docs/experiments/2026-07-09-method4-claim-support-excel-tables/method4_claim_support_excel_tables.xlsx	yes	xlsx	96353	NA
excel_claim_support_manifest_tsv	excel	docs/experiments/2026-07-09-method4-claim-support-excel-tables/manifest.tsv	yes	tsv	10984	42
excel_claim_support_source_caveat_provenance_tsv	excel	docs/experiments/2026-07-09-method4-claim-support-excel-tables/tsv/40_S-01_图表来源_边界_Provenance_自解释索引.tsv	yes	tsv	27590	34
excel_claim_support_raw_provenance_paths_tsv	excel	docs/experiments/2026-07-09-method4-claim-support-excel-tables/tsv/41_S-02_source_manifest_raw_provenance_路径清单.tsv	yes	tsv	93636	538
excel_claim_support_completion_audit_tsv	excel	docs/experiments/2026-07-09-method4-claim-support-excel-tables/tsv/42_S-03_2026-07-09_当前范围完成审计.tsv	yes	tsv	1967	6
final_evidence_document_integrity_audit_20260709_post_excel_selfcontained_csv	final	results/method4_claim_coverage_20260704/derived_claim_tables/final_evidence_document_integrity_audit_20260709_post_excel_selfcontained.csv	yes	csv	3486	9
```

- 来源：`results/method4_claim_coverage_20260704/source_manifest.json`
- 边界/注意：此表列出 provenance 路径和基础存在性/规模信息，不把所有 raw 文件内容全文展开到 Excel。

### 表 S-03: 2026-07-09 当前范围完成审计
- 支撑论据：把最终判断拆成可筛选审计项，明确当前 GloVe 范围完成、以及原始强范围中哪些仅在强化 claim wording 时才需要补跑。
- 推荐图形：不建议画图；用于审阅。
- Excel 绘图设置：X 轴=NA；建议系列=NA。

TSV 数据区（复制代码块内部，从下一行表头开始）：
```tsv
audit_item	status	current_judgment	evidence	blocking_current_glove_scope	remaining_only_if_stronger_wording
xlsx_tsv_chart_data_roundtrip	pass	xlsx 对当前 Excel 制图文档的 TSV 数据没有发现缺失。	Markdown TSV blocks, exported TSV files, workbook data sheets, and manifest rows are verified to match.	no	none
xlsx_source_caveat_provenance_self_containment	supplemented	xlsx 现在包含 S-01 source/caveat/provenance sheet，补足原先只适合制图、不自带完整说明文字的问题。	40_S-01 workbook sheet; tsv/40_S-01_图表来源_边界_Provenance_自解释索引.tsv	no	none
xlsx_raw_provenance_manifest_paths	supplemented	xlsx 现在包含 S-02 raw provenance manifest path sheet，可在 Excel 内筛选 source_manifest 中的 raw/derived/provenance 路径。	538 source_manifest path rows are exported with exists/path_type/count fields.	no	none
xlsx_experiment_setup_self_containment	supplemented	xlsx 现在包含 S-04 实验配置 sheet，可在 Excel 内查看主实验默认平台、硬件、数据集、分片和 baseline 定义。	43_S-04 workbook sheet; tsv/43_S-04_实验配置.tsv	no	none
current_glove_scope_experiment_completion	complete_with_caveats	当前 GloVe-only 范围内没有活动 blocker，关键补充实验已完成并整理进文档/表格。	goal_completion_audit_20260709_post_multiassign.csv; current_hard_gap_blocker_audit_20260709_post_multiassign.csv	no	none
original_strong_full_matrix_scope	not_complete_by_design_for_current_scope	原始强范围/全矩阵实验没有全面完成；剩余项主要是强表述才需要的 follow-up。	remaining_experiment_gaps.csv; summary document Recommended Paper Claim Set and cautious wording sections.	no	0.99 dominance; full-fission Claim A dominance; strict same-recall single/default/w2c2; comparative worker-count controls; full-size build RSS/time; Qdrant subsystem profiling; non-GloVe generalization; matrix-wide robustness.
chart_id_uniqueness	pass	重复 C-01/E-02 数据块已在导出层拆为 C-01a/C-01b 和 E-02a/E-02b；当前未作为主图表纳入的数据使用 OUT-01，避免空 chart_id。	manifest.tsv chart_id column has no duplicates and no empty IDs after regeneration.	no	none
```

- 来源：`results/method4_claim_coverage_20260704/derived_claim_tables/goal_completion_audit_20260709_post_multiassign.csv`；`results/method4_claim_coverage_20260704/derived_claim_tables/current_hard_gap_blocker_audit_20260709_post_multiassign.csv`；`results/method4_claim_coverage_20260704/remaining_experiment_gaps.csv`
- 边界/注意：此表把当前范围和可选强表述 follow-up 分开；不应把可选强表述缺口解释为当前 GloVe-only 范围 blocker。

### 表 S-04: 实验配置
- 支撑论据：记录当前 GloVe 结果的共同运行环境、不同实验类别的资源/分片变量和对比方法定义，避免把 worker-count、shard-count 与主三方法对比混成单一配置。
- 推荐图形：不建议画图；用于审阅。
- Excel 绘图设置：X 轴=NA；建议系列=NA。

TSV 数据区（复制代码块内部，从下一行表头开始）：
```tsv
分类	项目	值	单位/说明
平台	平台	Qdrant	NA
运行	运行方式	单机模拟	计算与存储隔离
硬件	CPU 型号	Intel Xeon Gold 6330 @ 2.00GHz	NA
硬件	CPU sockets	2	sockets
硬件	CPU cores/socket	28	cores/socket
硬件	CPU threads/core	2	threads/core
数据	数据集	GloVe-200-angular	文本嵌入
数据	向量维度	200	dimensions
数据	Base size	1200000	vectors (1.2M)
数据	距离度量	Angular	NA
范围	主召回目标	0.80 / 0.85 / 0.90 / 0.95	Recall@10；另有约 0.97 正向补充与 0.99 负结果边界
主对比	有效逻辑分片数	46	主三方法路由对比为主；具体数据表可有不同分片形态
worker-count	物理 worker 数	1 / 2 / 3 / 4	1 controller + N workers；46 个有效逻辑分片按 round-robin 放置
worker-count	controller CPU 配额	4	logical CPUs；cpuset
worker-count	每个 worker CPU 配额	4	logical CPUs；cpuset
worker-count	测量	100 / 3 / 3000	batch size / repeats / queries per repeat；仅 Orion/Method4
shard-count	有效逻辑分片数	31 / 46	固定 3-worker 部署；不能作为 worker-count 结果解释
方法	Naïve hash all-shards	Qdrant 生产式 hash/random 分片	每个查询搜索全部有效逻辑分片
方法	Plain K-means	普通、无均衡约束的 k-means	查询以 nprobe 选择簇/分片，按目标召回校准
方法	Orion（Balanced K-means）	带约束的均衡 k-means 分区与 Method4 路由	目标方法，不是 baseline
```

- 来源：主机配置、`results/method4_worker_count_online_scaling_20260708/worker_*/docker-compose.worker-count.yaml`、`worker_count_online_scaling_manifest_20260708.json` 与各表来源 metadata。
- 边界/注意：不能把 46 个逻辑分片、1--4 个物理 worker、31/46 shard-count 以及其他表的分片形态误写为同一个固定部署；非 GloVe 结果不属于当前范围。
