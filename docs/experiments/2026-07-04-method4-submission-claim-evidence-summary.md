# 2026-07-04 Method4 Submission Claim Evidence Summary

## Scope

This document consolidates the current evidence for the claims listed in `docs/experiments/2026-06-16-method4-distributed-submission-experiment-plan.md`. It separates supported evidence from planned-but-not-yet-completed experiments. The goal is to avoid turning planned rows into paper claims unless raw artifacts exist and the experiment shape is sufficient.

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

Generated audit artifacts:

- `results/method4_claim_coverage_20260704/claim_requirements.csv`
- `results/method4_claim_coverage_20260704/claim_evidence_audit.csv`
- `results/method4_claim_coverage_20260704/remaining_experiment_gaps.csv`
- `results/method4_claim_coverage_20260704/derived_claim_tables/`
- `results/method4_claim_coverage_20260704/derived_claim_tables/claim_support_data_index.csv`
- `results/method4_claim_coverage_20260704/derived_claim_tables/remaining_experiment_execution_queue.csv`
- `results/method4_claim_coverage_20260704/derived_claim_tables/goal_completion_audit_20260706.csv`
- `results/method4_claim_coverage_20260704/derived_claim_tables/goal_completion_audit_20260707.csv`
- `results/method4_claim_coverage_20260704/derived_claim_tables/goal_completion_audit_20260708.csv`
- `results/method4_claim_coverage_20260704/derived_claim_tables/goal_completion_audit_20260709.csv`
- `results/method4_claim_coverage_20260704/derived_claim_tables/final_evidence_document_integrity_audit_20260707.csv`
- `results/method4_claim_coverage_20260704/derived_claim_tables/final_evidence_document_integrity_audit_20260708.csv`
- `results/method4_claim_coverage_20260704/derived_claim_tables/current_environment_preflight_20260706.csv`
- `results/method4_claim_coverage_20260704/derived_claim_tables/current_environment_preflight_20260707.csv`
- `results/method4_claim_coverage_20260704/derived_claim_tables/current_hard_gap_blocker_audit_20260706.csv`
- `results/method4_claim_coverage_20260704/derived_claim_tables/current_hard_gap_blocker_audit_20260707.csv`
- `results/method4_claim_coverage_20260704/derived_claim_tables/current_unblock_status_20260707.csv`
- `results/method4_claim_coverage_20260704/derived_claim_tables/current_unblock_status_20260708.csv`
- `results/method4_claim_coverage_20260704/derived_claim_tables/formal_experiment_prerequisite_audit_20260708.csv`
- `results/method4_claim_coverage_20260704/derived_claim_tables/goal_blocked_threshold_audit_20260708.csv`
- `results/method4_claim_coverage_20260704/derived_claim_tables/current_hard_gap_blocker_audit_20260708_post_worker_count.csv`
- `results/method4_claim_coverage_20260704/derived_claim_tables/current_hard_gap_blocker_audit_20260709_post_claim_a.csv`
- `results/method4_claim_coverage_20260704/derived_claim_tables/current_hard_gap_blocker_audit_20260709_post_multiassign.csv`
- `results/method4_claim_coverage_20260704/derived_claim_tables/goal_completion_audit_20260709_post_multiassign.csv`
- `results/method4_claim_coverage_20260704/derived_claim_tables/claim_a_partition_family_current_rebuilds_20260709.csv`
- `results/method4_claim_coverage_20260704/derived_claim_tables/worker_count_online_scaling_summary_20260708.csv`
- `results/method4_claim_coverage_20260704/derived_claim_tables/worker_count_online_scaling_deltas_20260708.csv`
- `results/method4_claim_coverage_20260704/derived_claim_tables/worker_count_docker_stats_summary_20260708.csv`
- `results/method4_claim_coverage_20260704/derived_claim_tables/worker_count_online_scaling_manifest_20260708.json`
- `results/method4_claim_coverage_20260704/derived_claim_tables/strict_multiassign_latency_20260709.csv`
- `results/method4_claim_coverage_20260704/derived_claim_tables/strict_ablation_live_collection_audit_20260709_post_multiassign.csv`
- `results/method4_claim_coverage_20260704/derived_claim_tables/docker_cleanup_opportunity_audit_20260708.csv`
- `results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_packet_capture_bridge_summary.csv`
- `results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_packet_capture_bridge_totals.csv`
- `results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_packet_capture_bridge_deltas.csv`
- `results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_packet_capture_physical_nic_summary.csv`
- `results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_packet_capture_physical_nic_totals.csv`
- `results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_packet_capture_physical_nic_execution_summary.csv`
- `results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_099_boundary_query_order_multiseed_summary.csv`
- `results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_099_boundary_query_order_manifest.json`
- `results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_099_boundary_warmup_overlay.csv`
- `results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_099_boundary_warmup_method4_vs_naive_deltas.csv`
- `results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_099_boundary_warmup_sensitivity.csv`
- `results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_099_boundary_sensitivity_deltas.csv`
- `results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_099_broader_strategy_screen_summary.csv`
- `results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_099_broader_strategy_formal_summary.csv`
- `results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_099_broader_strategy_formal_deltas.csv`
- `results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_099_broader_strategy_manifest.json`
- `results/method4_claim_coverage_20260704/derived_claim_tables/current_experiment_readiness_recheck_20260707.csv`
- `results/method4_claim_coverage_20260704/derived_claim_tables/current_experiment_readiness_recheck_20260708.csv`
- `results/method4_claim_coverage_20260704/derived_claim_tables/unincorporated_artifact_review_20260707.csv`
- `results/method4_claim_coverage_20260704/source_manifest.json`

The machine-readable claim support index is
`results/method4_claim_coverage_20260704/derived_claim_tables/claim_support_data_index.csv`.
It provides one row per major support point, including safe statement, key
numbers, primary sources, caveats, and intended paper use. It is an index over
the evidence package, not a replacement for the detailed claim sections below.

The remaining-experiment execution queue is
`results/method4_claim_coverage_20260704/derived_claim_tables/remaining_experiment_execution_queue.csv`.
It turns the still-missing experiments into concrete preflight checks,
execution recipes, expected outputs, update targets, and no-claim-before-run
boundaries. It is a run queue for future completion work, not evidence by
itself.

The requirement-level completion audit snapshot is
`results/method4_claim_coverage_20260704/derived_claim_tables/goal_completion_audit_20260708.csv`,
with 2026-07-09 increments at
`results/method4_claim_coverage_20260704/derived_claim_tables/goal_completion_audit_20260709.csv`
and
`results/method4_claim_coverage_20260704/derived_claim_tables/goal_completion_audit_20260709_post_multiassign.csv`.
Together they map claims A-H, cross-cut remaining gaps, current external
blockers, and the final-document deliverables to their source requirements,
current evidence, remaining caveats, next actions, and final-summary locations.
The post-multiassignment increment records that the current GloVe-only scope is
covered with caveats for the formerly missing worker-count, Claim A partition-
family, and selected strict multi-assignment latency rows. Recall@10 0.99
dominance and non-GloVe datasets remain out of current scope. Remaining caveats
are optional stronger-wording items: strict same-recall single/default/w2c2
latency, full-size build-stage time/RSS, Qdrant subsystem profiling, comparative
worker-count controls, and broader matrix-wide robustness.

The 2026-07-07 final evidence-document integrity audit is
`results/method4_claim_coverage_20260704/derived_claim_tables/final_evidence_document_integrity_audit_20260707.csv`.
It verified that the then-current cross-claim summary table had the required
columns and claims A-H, the embedded support-data appendix covered the then-
current 43 support-index rows, all support-index primary sources existed, all
`docs/` / `results/` / `tools/` paths referenced by that summary existed, and
the core CSV/JSON audit files parsed cleanly. The current support index has 47
rows after the Claim A and strict multi-assignment supplements.

The 2026-07-08 refresh of that integrity audit is
`results/method4_claim_coverage_20260704/derived_claim_tables/final_evidence_document_integrity_audit_20260708.csv`.
It adds coverage for the 2026-07-08 unblock/readiness artifacts, Claim B/C/G
raw-provenance audits, Claim D batch=200 and boundary supplements, and now the
post-worker-count evidence package while keeping the distinction between
documentation-package integrity and full original-scope experiment completion.

Continuation preflight on 2026-07-06 is recorded in
`results/method4_claim_coverage_20260704/derived_claim_tables/current_environment_preflight_20260706.csv`.
The current cluster is healthy and has no running Method4 benchmark process,
but the root filesystem has only 3.9G free, the live deployment is still fixed
at one controller plus three workers, and the full-size strict
multi-assignment plus Claim A random/topology partition-family collections are
not live. A follow-up preflight on 2026-07-07 is recorded in
`results/method4_claim_coverage_20260704/derived_claim_tables/current_environment_preflight_20260707.csv`.
The earlier non-sudo physical-NIC tcpdump attempt failed with `Operation not
permitted`, but the 2026-07-07 run verified `sudo -n tcpdump` on both UP
external NICs and collected a Claim E physical-NIC negative-control capture.
That capture records zero matching Docker-subnet packets on `ens6f0np0` and
`ens111f0np0` during the Claim E all-mode batch=200 window. Therefore the
physical-NIC permission blocker is removed for the single-host negative
control. In the original full-scope plan, the remaining hard gaps still required
collection restore/rebuild, worker-count redeploys, or stronger full-size
build/internal-resource instrumentation. The later 2026-07-08
worker-count supplement resolves that redeploy item for the current basic
Method4-only scope.

A fresh current-state hard-gap audit is recorded in
`results/method4_claim_coverage_20260704/derived_claim_tables/current_hard_gap_blocker_audit_20260707.csv`.
It was refreshed at `2026-07-07T10:41:06Z` and rechecked the live controller:
27 collections are live, including two small Claim E build-smoke collections,
but there is still no `random_*` partition collection, no `bench_ma_strict_*`
full-size multi-assignment collection, no GIST/Deep collection, and only
reduced SIFT-100k plus smoke L2 collections for L2. That GIST/Deep/L2 state is
now historical/out of current scope. It also records the fixed 3-worker
deployment before the later worker-count supplement, the current 3.9G-free root
filesystem, the resolved Claim
E physical-NIC negative-control status, and the smoke-only build-stage
time/RSS instrumentation. Full-size build-stage time/RSS remains missing. A
post-smoke preflight is also recorded in
`results/method4_claim_coverage_20260704/derived_claim_tables/current_environment_preflight_20260707_build_smoke.csv`;
it separately shows the same two Claim E build-smoke collections and confirms
the root filesystem was still at 3.9G free. This is not performance evidence;
it is historical proof that those then-remaining hard gaps needed external
storage/deployment/full-size build-instrumentation changes before they could be
closed.

A later runnable-state recheck is recorded in
`results/method4_claim_coverage_20260704/derived_claim_tables/current_experiment_readiness_recheck_20260707.csv`.
It rechecked the same live Qdrant controller at `2026-07-07T12:11:09Z` and
again found 27 live collections, no required Claim A random/topology/load
recalibrated collections, no full-size `bench_ma_strict_*` collections, no
full L2/GIST/Deep collection, the fixed one-controller/three-worker
deployment, and the same 3.9G-free root filesystem. It marks documentation and
read-only provenance audits as runnable now. This is a historical full-scope
snapshot: under the current narrowed scope, non-GloVe datasets and 0.99 work are
not blockers, and the later worker-count supplement resolves the basic redeploy
need.

The current unblock status is recorded in
`results/method4_claim_coverage_20260704/derived_claim_tables/current_unblock_status_20260707.csv`.
It was refreshed at `2026-07-07T11:00:14Z` after the interrupted continuation
was resumed. The goal API reported the original objective as active rather
than tool-blocked, no benchmark/tcpdump/pytest process was running, and Qdrant
was reachable with the same 27 live collections. This removes the session-level
ambiguity around the word "blocked"; it does not close the original full-scope
experiment gaps. At that historical 2026-07-07 checkpoint, the then-missing
Claim A partition-family online rows, multi-assignment P95/P99 rows, true
worker-count online scaling, full L2/GIST/Deep generalization, and full-size
build-stage time/RSS required the collection restore/rebuild, redeploy, storage,
or instrumentation changes listed in the execution queue before they could
support stronger claim wording. Later sections record the current-scope
resolutions for worker-count, Claim A, and selected strict multi-assignment
latency, while non-GloVe/0.99 work is out of scope and full-size build/resource
profiling remains optional stronger wording.

The 2026-07-08 unblock refresh is recorded in
`results/method4_claim_coverage_20260704/derived_claim_tables/current_unblock_status_20260708.csv`,
with the corresponding runnable-state table at
`results/method4_claim_coverage_20260704/derived_claim_tables/current_experiment_readiness_recheck_20260708.csv`.
At `2026-07-08T09:31:10Z`, the goal API again reported the original objective
as active rather than tool-blocked. Qdrant remained reachable with the same 27
live collections and the same fixed one-controller/three-worker deployment;
no matching Method4 benchmark, Qdrant experiment harness, `tcpdump`, or
`pytest` process was running. The root filesystem was tighter than the
previous check, with 3.4G free on `/` and 68G free on `/home`.

The 2026-07-08 local recovery search found references to the missing Claim A
and strict multi-assignment collection names in offline oracle tables, old
QPS/frontier summaries, configs, metrics samples, and prior audits, but did
not find a local Qdrant snapshot, restore directory, or hidden Docker volume
that can directly restore `random_balanced_46`, `kmeans_topology_46`,
`kmeans_topology_load_recalibrated_46`, or the `bench_ma_strict_*`
collections. This clears the session-level ambiguity around "blocked" and
rules out local artifact promotion as the immediate fix; it does not make the
missing formal experiments runnable. Under the current narrowed scope,
non-GloVe datasets and 0.99 dominance work are not blockers. The remaining
current-scope formal rows still require restored/rebuilt full-size collections,
and any new rebuild should use `/home`-backed temporary storage or approved
Docker cleanup rather than root-backed Docker named volumes.

The formal-experiment prerequisite audit is recorded in
`results/method4_claim_coverage_20260704/derived_claim_tables/formal_experiment_prerequisite_audit_20260708.csv`.
It expands the 2026-07-08 readiness check into per-item prerequisites for the
remaining formal gaps. Direct `GET /collections/<name>` checks returned HTTP
404 for `random_balanced_46`, `kmeans_topology_46`,
`kmeans_topology_load_recalibrated_46`,
`bench_ma_strict_orion_o100_single_s31`,
`bench_ma_strict_orion_o118_default_s31`, and
`bench_ma_strict_orion_o149_w2c2_s31`. The live
`smoke_multiassign_*`, reduced SIFT-100k, and Claim E build-smoke collections
are recorded as usable only for their existing smoke/reduced supplements, not
as substitutes for the missing full-size formal rows. The audit also records
that `sift-128-euclidean.hdf5` and `deep-image-96-angular.hdf5` exist locally,
while a GIST HDF5 was not listed in the dataset directory; dataset presence is
not itself a Qdrant result.

The persistent-goal blocked-threshold audit is recorded in
`results/method4_claim_coverage_20260704/derived_claim_tables/goal_blocked_threshold_audit_20260708.csv`.
It ties the resumed goal state to the repeated external blockers: missing
full-size formal collections, fixed three-worker deployment, and the 3.4G-free
root filesystem. This audit is not performance evidence and does not change
the claim support tables; it records why the original full-scope experiment
completion cannot advance further without restored/rebuilt collections,
storage cleanup or relocation, or controlled redeploys.

The post-worker-count scope audit is recorded in
`results/method4_claim_coverage_20260704/derived_claim_tables/current_hard_gap_blocker_audit_20260708_post_worker_count.csv`.
It supersedes the worker-count portion of the earlier blocker audit: the basic
GloVe Method4-only online worker-count supplement is now complete for
worker_count 1/2/3/4 and target neighborhoods 0.80/0.85/0.90/0.95. It also
records that non-GloVe datasets and 0.99 dominance work are out of the current
scope, while Claim A partition-family collections and strict multi-assignment
collections were still missing at that 2026-07-08 checkpoint.

The 2026-07-09 post-Claim-A blocker audit is recorded in
`results/method4_claim_coverage_20260704/derived_claim_tables/current_hard_gap_blocker_audit_20260709_post_claim_a.csv`.
It updates the current scope after rebuilding and measuring
`random_balanced_46`, `kmeans_topology_46`, and
`kmeans_topology_load_recalibrated_46` on `/home`-backed Qdrant storage. The
three current-harness online rows are now available and the temporary
collections were deleted after preserving results.

The later 2026-07-09 post-multiassignment blocker audit is recorded in
`results/method4_claim_coverage_20260704/derived_claim_tables/current_hard_gap_blocker_audit_20260709_post_multiassign.csv`.
It records that the selected full-size strict multi-assignment single/default/
w2c2 latency rows were rebuilt under `/home`-backed project Qdrant storage, run
for 3000 queries * 3 repeats at batch_size=200, indexed in
`strict_multiassign_latency_20260709.csv`, and then deleted after results were
preserved. No high-priority GloVe blocker remains for the requested current
scope; what remains are caveats for stronger optional wording.

Status vocabulary:

- `complete`: planned experiment shape is covered by raw artifacts and supports the claim wording.
- `complete_with_runtime_caveat`: planned matrix evidence exists, but production runtime enabled/disabled tail-latency A/B should be rerun for stronger runtime wording.
- `supported_with_*_gap`: the central claim is supported, but one planned metric family is missing and must be excluded or rerun.
- `complete_for_080_095_scope_with_099_excluded`: current Claim D wording is complete for the requested 0.80/0.85/0.90/0.95 scope while 0.99 dominance remains excluded.
- `basic_worker_count_complete_comparative_partial`: requested basic Method4-only worker-count scaling is complete, but cross-method worker-count controls are not present.
- `complete_with_selected_latency_caveats`: selected full-size latency cells exist, but must be reported with target-neighborhood or current-harness caveats rather than as strict same-recall proof.
- `current_scope_glove_selected_rows_complete_with_caveats`: the current GloVe-only/no-0.99 scope has the requested rows, while optional stronger wording still has caveats.
- `out_of_current_scope`: a historically planned experiment is no longer required by the current narrowed scope.
- `partial`: useful evidence exists, but the planned experiment shape or mechanism evidence is incomplete.
- `missing`: no adequate artifact found in the current workspace.

## Cross-Claim Summary Table

| Claim | Status | Primary contrast | Recall level | QPS/latency result | Work/load result | Artifact |
| --- | --- | --- | --- | --- | --- | --- |
| A | partial_current_scope_online_rows_complete_with_caveats | Orion / Full Method4 vs Simple KMeans nprobe and Naive; partition-family oracle, existing online submatrix, and 2026-07-09 current-harness family rebuilds | 0.80/0.85/0.90/0.95, with selected online submatrix around 0.95 | Orion same-avg-EF rows reach 1.16-1.99x Simple KMeans QPS; 2026-07-09 current rebuilds show random/topology/load-recalibrated QPS +39.3%/+97.0%/+87.5% and P95 -30.8%/-55.0%/-47.4% versus paired naive | Orion visits 47-55% fewer shards than Simple KMeans at same avg EF; offline topology rows improve edge cut, GT-shard entropy, routed shards, and upper_k 80/120 miss versus KMeans-only; topology rebuild rows visit about 18.6 shards vs 46 for naive | `results/method4_claim_coverage_20260704/derived_claim_tables/claim_a_online_orion_vs_simple_kmeans_same_avg_ef.csv`; `results/method4_claim_coverage_20260704/derived_claim_tables/claim_a_partition_oracle_matrix.csv`; `results/method4_claim_coverage_20260704/derived_claim_tables/claim_a_partition_online_latency_submatrix.csv`; `results/method4_claim_coverage_20260704/derived_claim_tables/claim_a_partition_family_current_rebuilds_20260709.csv` |
| B | complete_with_selected_latency_caveats | Orion multi-assignment default/w2c2 vs single assignment | Recall frontier under expansion <=2.0x; selected 0.95-neighborhood full-size latency cells | Online frontier evidence shows multi-assignment improves recall-QPS frontier; selected strict latency rows show single/default/w2c2 Recall@10 0.97317/0.95260/0.94913, QPS 279.23/475.27/557.80, and P95 736.45/432.73/375.01 ms | Orion default/w2c2 reduce oracle_gt_miss@10 versus single assignment at upper_k 80/120/160; selected strict rows reduce visited shards and EF-sum versus the single control in the target neighborhood | `results/method4_claim_coverage_20260704/derived_claim_tables/claim_b_multiassign_expansion_qps.csv`; `results/method4_claim_coverage_20260704/derived_claim_tables/claim_b_orion_oracle_gt_miss.csv`; `results/method4_claim_coverage_20260704/derived_claim_tables/claim_b_raw_provenance_audit_20260708.csv`; `results/method4_claim_coverage_20260704/derived_claim_tables/strict_multiassign_latency_20260709.csv`; `results/method4_benchmark_matrix/method4_strict_multiassign_latency_rebuild_20260709/`; `results/method4_strict_ablation_multiassign_latency_20260709/` |
| C | supported_with_batch_latency_supplement | Dynamic EF vs fixed EF with same routed shards | Same-recall rows around 0.95; latency supplement at Recall@10 about 0.949 | Same-recall Dynamic EF gives +27-31% QPS; batch-latency supplement gives +26.75% QPS and -21.00%/-20.97% P95/P99 versus fixed EF | EF-sum/query drops 35-41% in same-recall rows and 32.52% in the latency supplement; routed EP count predicts GT-hit shards | `results/method4_claim_coverage_20260704/derived_claim_tables/claim_c_dynamic_vs_fixed_latency_matrix.csv`; `results/method4_claim_coverage_20260704/derived_claim_tables/claim_c_dynamic_vs_fixed_latency_deltas.csv`; `results/method4_claim_c_evidence_20260704/`; `results/method4_claim_coverage_20260704/derived_claim_tables/claim_c_raw_provenance_audit_20260708.csv` |
| D | complete_for_080_095_scope_with_099_excluded | Method4 routed search vs Naive all-shards fixed EF | Current scope: 0.80/0.85/0.90/0.95, plus through-about-0.97 support; 0.99 is boundary/negative context only | Strict about-0.97 batch=100 pair: +40.58% QPS, -31.85% P95, -32.42% P99. Strict about-0.97 batch=200 pair: +88.01% QPS and -67.32%/-69.33% mean P95/P99 versus Naive, with Naive batch=200 instability caveat. 0.80/0.85/0.90/0.95 robustness rows remain positive in the selected target neighborhoods. | About-0.97 pair visits 46.39-46.40% fewer shards; selected 0.80/0.85/0.90/0.95 robustness rows keep QPS/tail advantage. Near-0.99 rows are retained only to avoid overclaiming. | `results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_high_recall_latency_pairs.csv`; `results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_batch200_high_recall_deltas.csv`; `results/method4_claim_coverage_20260704/derived_claim_tables/robustness_080_085_method4_vs_naive_deltas.csv`; `results/method4_claim_coverage_20260704/derived_claim_tables/robustness_090_method4_vs_naive_deltas.csv`; `results/method4_claim_coverage_20260704/derived_claim_tables/robustness_095_method4_vs_naive_deltas.csv` |
| E | supported_with_overhead_metrics_gap | compact_current execution vs grouped_by_ef_materialized and client_shard_major_expanded | Claim E runtime matrix at Recall@10-preserving 3000-query windows | compact_current improves QPS by +66-170% versus grouped and +159-185% versus client-expanded for batch sizes 50/100/200; selected REST/process metrics also show lower batch duration and lower controller process CPU/RSS | Request objects/query drop to 1.00; JSON request bytes/query -93.23%, response bytes/query -95.76%, total JSON body bytes/query -93.71% versus client-expanded; selected bridge/interface/process counters support lower overhead | `results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_execution_mode_latency_matrix.csv`; `results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_wire_bytes_summary.csv`; `results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_controller_process_attribution_summary.csv` |
| F | complete_with_runtime_caveat | Worker-local peer pre-merge vs direct-peer no-premerge/current coordinator path | Same-image 0.95-style Method4 windows and 3 variants * 4 batch sizes | Same-image server A/B shows +6.84% QPS with recall delta +0.000067; new batch matrix preserves recall and supports latency-shape/fan-in reduction wording | Physical fan-in drops from 23.208 logical streams/query to 2.977 physical peers/query; candidate groups/returned candidates drop 87.40% in the batch matrix | `results/method4_claim_coverage_20260704/derived_claim_tables/claim_f_worker_local_premerge.csv`; `results/method4_claim_coverage_20260704/derived_claim_tables/claim_f_premerge_batch_latency_matrix.csv`; `results/method4_claim_coverage_20260704/derived_claim_tables/crosscut_request_candidate_pressure_proxy.csv` |
| G | complete | method4-aware physical placement vs baseline placement | Matched online 18-run batch latency matrix, plus concurrency=8 single-query supplement | method4-aware improves QPS by 1.2-2.1% and reduces P95/P99 batch latency by 0.4-1.0% in the matched batch matrix; the concurrency=8 supplement shows +2.30% QPS and -5.27%/-7.91% P95/P99 | P95 max worker EF drops 4.9-8.0%; placement matrix supports more balanced Method4-aware load | `docs/experiments/2026-07-04-method4-claim-g-physical-layout-evidence.md`; `results/method4_claim_g_evidence_20260704/` |
| H | complete | compact_multi_ep semantic invariant audit vs expected Method4 semantics | 100-query semantic audit | All 9 semantic invariants pass; no performance claim is made from this audit | Confirms upper routing, point_to_shards, multi-assignment, per-shard entry points, dynamic EF, logical-shard lower search, no adaptive pruning, source-id dedup, and final global merge | `results/method4_claim_h_semantic_audit_20260704/audit_20260704_210845/semantic_invariant_summary.csv`; `results/method4_claim_coverage_20260704/derived_claim_tables/claim_h_semantic_invariants_current_evidence.csv` |
| CrossCut | basic_worker_count_complete_comparative_partial | Method4-only online worker-count scaling | GloVe target neighborhoods 0.80/0.85/0.90/0.95 | Worker_count=4 vs 1 QPS speedups are 1.36x/1.59x/1.76x/1.89x and P95 reductions are 27.30%/40.22%/42.48%/46.97% for targets 0.80/0.85/0.90/0.95 | 46 active custom shards in every run; Docker stats collected; temporary Qdrant storage used `/home` bind mounts and was removed. No Naive/Simple KMeans worker-count controls. | `results/method4_claim_coverage_20260704/derived_claim_tables/worker_count_online_scaling_summary_20260708.csv`; `results/method4_claim_coverage_20260704/derived_claim_tables/worker_count_online_scaling_deltas_20260708.csv`; `results/method4_claim_coverage_20260704/derived_claim_tables/worker_count_online_scaling_manifest_20260708.json` |

## Claim Support Data Index Appendix

| Claim | Evidence group | Support level | Safe statement | Key data | Caveats / missing | Primary sources |
| --- | --- | --- | --- | --- | --- | --- |
| A | orion_vs_simple_kmeans_same_avg_ef | partial | At matched Recall@10 targets and similar average EF per visited shard, Orion reaches similar recall while visiting fewer shards and having higher QPS than Simple KMeans nprobe. | Targets 0.80/0.85/0.90/0.95: Orion visited 6.37/8.30/14.54/21.12 shards vs Simple KMeans 12/16/32/40; Orion QPS 1009.9/887.8/702.0/391.3 vs Simple KMeans 870.1/617.8/352.4/206.9. | This supports online Orion-vs-simple routing efficiency, not the full partition-family matrix. | results/method4_claim_coverage_20260704/derived_claim_tables/claim_a_online_orion_vs_simple_kmeans_same_avg_ef.csv; docs/experiments/2026-06-26-orion-vs-simple-kmeans-same-avg-ef.md |
| A | partition_oracle_topology_locality | partial | Topology convergence improves offline locality over balanced KMeans-only on edge cut, GT-shard entropy, and routed shard count at upper_k 80/120/160. | Edge cut drops 0.514 to 0.351; GT-shard entropy drops 0.942 to 0.811; avg routed shards drops from 16.30/20.05/22.93 to 13.95/17.27/19.93 for upper_k 80/120/160. | Current full-fission partition does not beat balanced KMeans-only on oracle miss or routed shard count; do not claim full-fission oracle dominance over KMeans-only. | results/method4_claim_coverage_20260704/derived_claim_tables/claim_a_partition_oracle_matrix.csv; results/method4_claim_a_partition_oracle_20260704/analysis_20260704_213047/ |
| A | partition_online_latency_submatrix | partial | Existing full-fission/balanced-KMeans rows and 2026-07-09 current-harness random/topology/load-recalibrated rebuild rows reduce tail latency versus paired naive near the 0.95 operating region. | Existing rows: full-fission vs naive batch100 QPS +53.5%, P95 -39.7%, P99 -40.6%; batch200 QPS +87.4%, P95 -57.2%, P99 -59.7%; balanced-KMeans vs naive batch100 QPS +42.5%, P95 -32.4%, P99 -33.5%. New current rebuilds: random_balanced QPS +39.3%, P95 -30.8%, P99 -32.5%; kmeans_topology QPS +97.0%, visited -59.5%, P95 -55.0%, P99 -57.9%; load-recalibrated QPS +87.5%, visited -59.5%, P95 -47.4%, P99 -48.0% versus paired naive. | The 2026-07-09 family rows are current-harness rebuilds, not byte-identical replay of the 2026-07-04 oracle artifact; random still visits nearly all shards; load-recalibrated no-fission route map matches topology; balanced-KMeans batch200 naive has high variance. | results/method4_claim_coverage_20260704/derived_claim_tables/claim_a_partition_online_latency_submatrix.csv; results/method4_claim_coverage_20260704/derived_claim_tables/claim_a_partition_online_latency_submatrix_deltas.csv; results/method4_claim_coverage_20260704/derived_claim_tables/claim_a_partition_family_current_rebuilds_20260709.csv; results/method4_claim_a_partition_online_latency_20260704/; results/method4_claim_a_partition_online_latency_20260709/ |
| A | topology_no_fission_selected_cell | partial | The selected topology/no-fission online cell visits fewer shards than full fission at similar recall, but is slightly slower and has slightly worse tail latency. | Topology/no-fission Recall 0.9522, QPS 430.6, visited 18.94, P95/P99 479.6/483.3 ms. Relative to full-fission: visited -16.46%, QPS -2.77%, P95 +2.71%, P99 +2.52%. | Use as a selected ablation cell, not as evidence that no-fission dominates full fission online. | results/method4_claim_coverage_20260704/derived_claim_tables/claim_a_partition_online_latency_submatrix.csv; results/method4_claim_coverage_20260704/derived_claim_tables/claim_a_partition_online_latency_submatrix_deltas.csv |
| A | partition_family_coverage_audit | partial | The previously missing Claim A random/topology/load-recalibrated online rows are now available as 2026-07-09 current-harness rebuilds, while strict stronger wording still has caveats. | Coverage audit now marks random_balanced_46, kmeans_topology_46, and kmeans_topology_load_recalibrated_46 as available_current_harness_rebuild_20260709; all three temporary collections were deleted after latency results were saved. Remaining caveats are non-byte-identical rebuilds, load-recalibrated no-fission route-map equivalence, and no full-fission dominance over balanced KMeans-only. | This is an audit/provenance row, not standalone performance proof; it keeps the remaining full-fission and strict all-family wording caveats visible. | results/method4_claim_coverage_20260704/derived_claim_tables/claim_a_partition_family_online_coverage_audit.csv; results/method4_claim_coverage_20260704/derived_claim_tables/claim_a_partition_family_current_rebuilds_20260709.csv; results/method4_claim_coverage_20260704/remaining_experiment_gaps.csv |
| A | partition_family_current_rebuilds_20260709 | partial_current_scope | The 2026-07-09 current-harness rebuilds fill the previously missing random, topology, and load-recalibrated Claim A online latency cells. | random_balanced_46: Recall@10 0.9587 vs naive 0.9529, QPS +39.3%, P95 -30.8%, P99 -32.5%, visited 44.16 vs 46. kmeans_topology_46: Recall@10 0.9504 vs 0.9529, QPS +97.0%, visited -59.5%, P95 -55.0%, P99 -57.9%. kmeans_topology_load_recalibrated_46: Recall@10 0.9502 vs 0.9529, QPS +87.5%, visited -59.5%, P95 -47.4%, P99 -48.0%. | These are current-harness rebuilds with build upper_search_ef=400, not byte-identical replays of the 2026-07-04 oracle artifact. They close the missing online-row gap but do not establish full-fission superiority over balanced KMeans-only. | results/method4_claim_coverage_20260704/derived_claim_tables/claim_a_partition_family_current_rebuilds_20260709.csv; results/method4_claim_a_partition_family_build_20260709/random_balanced_46_oracle_ef400/20260709_091449/; results/method4_claim_a_partition_family_build_20260709/kmeans_topology_46_oracle_ef400/20260709_092502/; results/method4_claim_a_partition_family_build_20260709/kmeans_topology_load_recalibrated_46_oracle_ef400/20260709_093432/; results/method4_claim_a_partition_online_latency_20260709/random_balanced_46_ef400build/analysis_20260709_092159/; results/method4_claim_a_partition_online_latency_20260709/kmeans_topology_46_ef400build/analysis_20260709_093136/; results/method4_claim_a_partition_online_latency_20260709/kmeans_topology_load_recalibrated_46_ef400build/analysis_20260709_094451/ |
| B | multi_assignment_online_frontier | complete | Under expansion at or below about 2x, Orion multi-assignment improves the online recall-QPS frontier versus single assignment. | At target 0.80/0.85/0.90/0.95, Orion default QPS is 1127.4/943.1/755.7/409.1 at expansion 1.185. Orion w2c2 QPS is 1069.8/915.1/632.6/458.0 at expansion 1.499. Single assignment QPS is 499.8/372.7/322.5/219.1. | Online QPS rows are mostly single final-eval runs without stability repeats; selected strict latency rows are indexed separately in strict_multiassign_latency_20260709. | results/method4_claim_coverage_20260704/derived_claim_tables/claim_b_multiassign_expansion_qps.csv; docs/experiments/2026-06-29-method4-multiassign-expansion-qps.md; docs/experiments/2026-06-30-method4-multiassign-recall-qps-frontier.md; results/method4_claim_coverage_20260704/derived_claim_tables/claim_b_raw_provenance_audit_20260708.csv; results/method4_benchmark_matrix/method4_multiassign_recall_qps_excel_20260630/final_eval_keypoints_long.csv; results/method4_benchmark_matrix/method4_multiassign_recall_qps_excel_20260630/coverage_and_quality_flags.csv; results/method4_benchmark_matrix/method4_multiassign_recall_qps_excel_20260630_strict/final_eval_keypoints_long_strict.csv; results/method4_benchmark_matrix/method4_multiassign_recall_qps_excel_20260630_strict/README.md |
| B | multi_assignment_oracle_miss | complete | Orion default and w2c2 reduce offline oracle_gt_miss@10 versus single assignment at upper_k 80/120/160. | Single miss 3.89%/2.41%/1.67%; default miss 1.63%/0.96%/0.60%; w2c2 miss 0.61%/0.33%/0.21% for upper_k 80/120/160. | Offline oracle mechanism evidence does not execute lower HNSW search. | results/method4_claim_coverage_20260704/derived_claim_tables/claim_b_orion_oracle_gt_miss.csv; results/method4_claim_b_oracle_miss_20260704/analysis_20260704_211749/; results/method4_claim_coverage_20260704/derived_claim_tables/claim_b_raw_provenance_audit_20260708.csv; results/method4_claim_b_oracle_miss_20260704/analysis_20260704_211749/oracle_gt_miss_summary.csv; results/method4_claim_b_oracle_miss_20260704/analysis_20260704_211749/strategy_build_summary.csv; results/method4_claim_b_oracle_miss_20260704/analysis_20260704_211749/run_metadata.json |
| C | dynamic_vs_fixed_same_recall | supported_with_batch_latency_supplement | Dynamic EF improves same-recall QPS and reduces estimated lower EF-sum/query compared with fixed EF in Orion Claim C ablations. | Same-recall comparisons show QPS +27.20%/+28.90%/+30.92% and EF-sum -38.37%/-34.92%/-40.80% across fixed upper_k 120 isolated, frontier 0.95, and frontier 0.97 cases. | Dynamic EF does not always visit fewer shards; the 0.97 point visits more shards while using lower EF-sum. | results/method4_claim_coverage_20260704/derived_claim_tables/claim_c_dynamic_vs_fixed_performance_summary.csv; results/method4_claim_coverage_20260704/derived_claim_tables/claim_c_dynamic_vs_fixed_deltas.csv; docs/experiments/2026-07-04-method4-claim-c-dynamic-ef-evidence.md; results/method4_claim_coverage_20260704/derived_claim_tables/claim_c_raw_provenance_audit_20260708.csv; results/method4_claim_c_evidence_20260704/source_manifest.json; results/method4_claim_c_evidence_20260704/; results/method4_claim_c_orion_fixed_ef_extended_20260625/20260625_122713/; results/method4_claim_c_orion_dynamic_ef_20260625/20260625_122853/; results/method4_claim_c_orion_frontier_fixed_095_robust_confirm_20260625/20260625_132656/; results/method4_claim_c_orion_frontier_dynamic_095_robust_confirm_20260625/20260625_132811/; results/method4_claim_c_orion_frontier_fixed_097_robust_confirm_20260625/20260625_132210/; results/method4_claim_c_orion_frontier_dynamic_097_more_robust_confirm_20260625/20260625_132923/ |
| C | routed_ep_relevance_proxy | supported_with_batch_latency_supplement | Routed entry-point count is a useful shard relevance proxy: GT-hit shards have far higher routed EP count than non-hit shards and receive larger Dynamic EF budget share. | GT-hit EP count vs non-hit EP count: 20.34 vs 3.22, 14.72 vs 2.55, 30.78 vs 4.54. Dynamic GT budget share vs fixed share: 34.93% vs 17.53%, 43.24% vs 21.05%, 42.64% vs 14.22%. | Offline relevance analysis verifies the proxy, not server-internal timing. | results/method4_claim_coverage_20260704/derived_claim_tables/claim_c_routed_ep_relevance_summary.csv; results/method4_claim_c_dynamic_ef_relevance_20260704/analysis_20260704_141217/; results/method4_claim_coverage_20260704/derived_claim_tables/claim_c_raw_provenance_audit_20260708.csv; results/method4_claim_c_evidence_20260704/source_manifest.json; results/method4_claim_c_dynamic_ef_relevance_20260704/analysis_20260704_141217/budget_alignment_summary.csv; results/method4_claim_c_dynamic_ef_relevance_20260704/analysis_20260704_141217/routed_ep_relevance_by_count.csv |
| C | dynamic_vs_fixed_batch_latency | supported_with_batch_latency_supplement | At the selected 0.95 operating point, Dynamic EF improves client-observed batch latency versus fixed EF with the same visited-shard count. | Recall +0.000167; QPS +26.75%; visited shards +0.00%; EF-sum -32.52%; P95 -21.00%; P99 -20.97%. | Measures client-observed batch endpoint wall time, not server-internal lower-search trace latency. | results/method4_claim_coverage_20260704/derived_claim_tables/claim_c_dynamic_vs_fixed_latency_matrix.csv; results/method4_claim_coverage_20260704/derived_claim_tables/claim_c_dynamic_vs_fixed_latency_deltas.csv; results/method4_claim_c_dynamic_vs_fixed_latency_20260704/analysis_20260704_235925/; results/method4_claim_coverage_20260704/derived_claim_tables/claim_c_raw_provenance_audit_20260708.csv; results/method4_claim_c_evidence_20260704/source_manifest.json; results/method4_claim_c_dynamic_vs_fixed_latency_20260704/analysis_20260704_235925/claim_d_high_recall_latency_summary.csv; results/method4_claim_c_dynamic_vs_fixed_latency_20260704/analysis_20260704_235925/claim_d_high_recall_latency_batches.csv; results/method4_claim_c_dynamic_vs_fixed_latency_20260704/analysis_20260704_235925/run_metadata.json |
| D | method4_vs_naive_095 | partial_high_recall_caveat | At Recall@10 around 0.95, Method4 has higher same-recall QPS and visits fewer shards than naive. | Method4 Recall 0.955267, QPS 387.14, visited 23.214 vs naive Recall 0.954767, QPS 272.37, visited 46; QPS +42.14%. | This is a 0.95 operating point, not the highest-recall boundary. | results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_method4_vs_naive_same_recall.csv; docs/experiments/2026-06-03-current-method4-vs-naive-same-recall.md |
| D | method4_vs_naive_097_latency | partial_high_recall_caveat | At Recall@10 around 0.97, Method4 improves QPS and client-observed P95/P99 versus naive while visiting fewer shards. | Strict 0.97 pair: Method4 Recall 0.9718 vs naive 0.9728; QPS +40.58%; visited -46.39%; P95 -31.85%; P99 -32.42%. | Use through about 0.97; do not generalize to 0.99. | results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_high_recall_latency_pairs.csv; results/method4_claim_d_high_recall_latency_20260704/analysis_20260704_215448/; results/method4_claim_d_high_recall_latency_refine_20260704/analysis_20260704_220036/ |
| D | method4_vs_naive_batch200_high_recall | partial_high_recall_caveat | At batch=200, the selected about-0.97 Method4 high-recall configuration preserves the Claim D QPS/tail advantage versus Naive, while the selected about-0.99 row remains below Recall@10 0.99 and below paired Naive recall. | 3000 queries * 3 repeats, batch=200. About 0.97: m4_160_80_20_b200 Recall@10 0.97183 vs naive_ef112_b200 0.97277, QPS +88.01%, visited shards -46.40%, EF-sum +7.18%, P95/P99 mean -67.32%/-69.33%; median QPS/P95/P99 deltas are +61.62%/-42.81%/-42.86%. About 0.99: m4_400_160_20_b200 Recall@10 0.98950 vs naive_ef200_b200 0.99017, QPS +30.53%, visited shards -25.48%, EF-sum +56.78%, P95/P99 mean -54.26%/-58.41%; Method4 still does not reach 0.99 recall. | The batch=200 Naive controls show high QPS/tail variance, especially repeat 3 for naive_ef112 and repeats 1/3 for naive_ef200; report this as strict same-batch evidence together with the more stable batch=100 context. The about-0.99 row is boundary evidence only, not 0.99 Method4 dominance. | results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_batch200_high_recall_summary.csv; results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_batch200_high_recall_deltas.csv; results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_batch200_high_recall_manifest.json; results/method4_claim_d_batch200_targets_20260708/analysis_20260708_104611/; results/method4_claim_d_batch200_targets_20260708/analysis_20260708_104611/claim_d_high_recall_latency_summary.csv; results/method4_claim_d_batch200_targets_20260708/analysis_20260708_104611/run_metadata.json |
| D | method4_vs_naive_097_closest_robustness | partial_high_recall_caveat | An additional Recall@10 about 0.97 closest-neighborhood Method4 configuration keeps the Method4-vs-naive QPS and P95/P99 advantage under original order, two shuffled-query seeds, and cold/no-warmup matched-window evaluation. | repeat_m4_160_80_16 vs naive_ef104, 3000 queries * 3 repeats, batch_size=100. Across warm original, shuffled seeds 20260716/20260717, and cold matched runs: Method4 QPS +41.09% to +57.27%, visited shards -48.63% to -48.65%, EF-sum -1.22% to -1.24%, P95 -28.09% to -41.45%, P99 -28.07% to -43.32%. Method4 recall is 0.9682-0.9687 versus naive 0.9695, so this is target-neighborhood evidence rather than strict same-recall in every row. | This broadens selected robustness coverage to another near-0.97 configuration but is still not a full matrix-wide robustness sweep. The Method4 rows are slightly below same-run naive recall by 0.00087-0.00130, so use as target-neighborhood robustness support. | results/method4_claim_coverage_20260704/derived_claim_tables/robustness_097_closest_overlay.csv; results/method4_claim_coverage_20260704/derived_claim_tables/robustness_097_closest_method4_vs_naive_deltas.csv; results/method4_claim_coverage_20260704/derived_claim_tables/robustness_097_closest_sensitivity_deltas.csv; results/method4_claim_coverage_20260704/derived_claim_tables/robustness_097_closest_query_order_multiseed_summary.csv; results/method4_claim_coverage_20260704/derived_claim_tables/robustness_097_closest_manifest.json; results/method4_claim_d_high_recall_latency_refine_20260704/analysis_20260704_220036/; results/method4_robustness_097_closest_shuffle_20260707/analysis_20260707_095706/; results/method4_robustness_097_closest_shuffle2_20260707/analysis_20260707_095948/; results/method4_robustness_097_closest_warmup_20260707/analysis_20260707_100229/ |
| D | method4_vs_naive_099_caveat | partial_high_recall_caveat | The current evidence does not support a strong Method4 dominance claim at Recall@10 around 0.99. | Closest 0.99 neighborhood: Method4 Recall 0.9893 vs naive 0.9902; QPS -2.63%; visited -25.50%; P95 +0.50%; P99 +0.71%. Lower-EF/high-upper-k retune and strategy-search formal candidates reached only 0.98670-0.98747 recall vs same-run naive_ef200 at 0.99017. | This is a transparent negative/limit result. | results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_high_recall_latency_pairs.csv; results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_099_retune_summary.csv; results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_099_retune_deltas.csv; results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_099_strategy_search_summary.csv; results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_099_strategy_search_deltas.csv; results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_099_strategy_search_top_screen_summary.csv; results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_099_strategy_search_top_screen_deltas.csv |
| D | method4_vs_naive_095_robustness | partial_high_recall_caveat | At the selected Recall@10 around 0.955 row, Method4 keeps its same-run QPS and P95/P99 advantage over naive under original, two shuffled-query seeds, and cold/no-warmup conditions. | Warm original: QPS +47.59%, visited -48.69%, P95/P99 -37.74%/-38.98%. Shuffled seed 20260707: QPS +78.85%, visited -48.66%, P95/P99 -46.37%/-45.98%. Shuffled seed 20260710: QPS +62.55%, visited -49.55%, P95/P99 -43.87%/-44.66%. Cold matched: QPS +51.90%, visited -48.69%, P95/P99 -39.96%/-39.23%. Across the two shuffled seeds, Method4 query-order QPS vs warm original is +6.95% to +13.17%. | Selected 0.955 robustness supplement only; now two shuffled seeds plus cold/no-warmup, but still not a full matrix-wide robustness proof. | results/method4_claim_coverage_20260704/derived_claim_tables/robustness_095_query_order_overlay.csv; results/method4_claim_coverage_20260704/derived_claim_tables/robustness_095_query_order_multiseed_summary.csv; results/method4_claim_coverage_20260704/derived_claim_tables/robustness_095_warmup_overlay.csv; results/method4_claim_coverage_20260704/derived_claim_tables/robustness_095_method4_vs_naive_deltas.csv; results/method4_claim_coverage_20260704/derived_claim_tables/robustness_095_sensitivity_deltas.csv; results/method4_robustness_095_original_20260705/analysis_20260705_063443/; results/method4_robustness_095_shuffle_20260707/analysis_20260705_063719/; results/method4_robustness_095_shuffle_20260710/analysis_20260706_131447/; results/method4_robustness_095_warmup_20260705/analysis_20260705_063957/ |
| D | method4_vs_naive_095_curve_config_robustness | partial_high_recall_caveat | An additional Recall@10 about 0.95 curve configuration keeps the Method4-vs-naive QPS and P95/P99 advantage under original order, two shuffled-query seeds, and cold/no-warmup matched-window evaluation. | m4_curve_160_50_10 vs naive_ef76, 3000 queries * 3 repeats, batch_size=100. Across warm original, shuffled seeds 20260714/20260715, and cold matched runs: Method4 QPS +69.05% to +73.97%, visited shards -48.64% to -48.70%, EF-sum -15.52% to -15.55%, P95 -45.31% to -46.98%, P99 -44.12% to -48.07%. Method4 recall is 0.9530-0.9538 versus naive 0.9543, so this is target-neighborhood evidence rather than strict same-recall in every row. | This broadens selected robustness coverage to another near-0.95 curve configuration but is still not a full matrix-wide robustness sweep. The Method4 rows are slightly below same-run naive recall by 0.00053-0.00133, so use as target-neighborhood robustness support. | results/method4_claim_coverage_20260704/derived_claim_tables/robustness_095_curve_config_overlay.csv; results/method4_claim_coverage_20260704/derived_claim_tables/robustness_095_curve_config_method4_vs_naive_deltas.csv; results/method4_claim_coverage_20260704/derived_claim_tables/robustness_095_curve_config_sensitivity_deltas.csv; results/method4_claim_coverage_20260704/derived_claim_tables/robustness_095_curve_config_query_order_multiseed_summary.csv; results/method4_claim_coverage_20260704/derived_claim_tables/robustness_095_curve_config_manifest.json; results/method4_robustness_095_curve_config_original_20260707/analysis_20260707_093724/; results/method4_robustness_095_curve_config_shuffle_20260707/analysis_20260707_093953/; results/method4_robustness_095_curve_config_shuffle2_20260707/analysis_20260707_094525/; results/method4_robustness_095_curve_config_warmup_20260707/analysis_20260707_094220/ |
| D | method4_vs_naive_080_085_robustness | partial_high_recall_caveat | At selected Recall@10 neighborhoods around 0.80 and 0.85, Method4 keeps same-run QPS/P95/P99 advantage over naive under original, two shuffled-query seeds, and cold/no-warmup conditions. | Target 0.80: Method4 Recall about 0.802-0.806 vs naive 0.8075; QPS +215.80% to +258.35%, visited -86.12% to -87.08%, EF-sum -62.34% to -63.36%, P95/P99 at least -70.39%/-70.17%. Target 0.85: Method4 Recall about 0.857-0.863 vs naive 0.8557; QPS +157.19% to +199.61%, visited -84.02% to -85.15%, EF-sum -37.54% to -40.60%, P95/P99 at least -63.84%/-64.44%. | Selected 0.80/0.85 robustness supplement only; 0.85 rows are target-neighborhood comparisons where Method4 recall is about 0.0016-0.0071 above naive, not strict same-recall. Still not a full matrix-wide robustness proof. | results/method4_claim_coverage_20260704/derived_claim_tables/robustness_080_085_query_order_overlay.csv; results/method4_claim_coverage_20260704/derived_claim_tables/robustness_080_085_query_order_multiseed_summary.csv; results/method4_claim_coverage_20260704/derived_claim_tables/robustness_080_085_warmup_overlay.csv; results/method4_claim_coverage_20260704/derived_claim_tables/robustness_080_085_method4_vs_naive_deltas.csv; results/method4_claim_coverage_20260704/derived_claim_tables/robustness_080_085_sensitivity_deltas.csv; results/method4_robustness_080_085_original_20260705/analysis_20260705_070956/; results/method4_robustness_080_085_shuffle_20260709/analysis_20260705_071249/; results/method4_robustness_080_085_shuffle_20260712/analysis_20260706_133627/; results/method4_robustness_080_085_warmup_20260705/analysis_20260705_071529/ |
| D | method4_vs_naive_090_robustness | partial_high_recall_caveat | At the selected Recall@10 around 0.90 target neighborhood, Method4 keeps same-run QPS/P95/P99 advantage over naive under original, two shuffled-query seeds, and cold/no-warmup conditions. | Warm original: QPS +143.77%, visited -68.89%, EF-sum -40.16%, P95/P99 -61.48%/-62.31%. Shuffled seed 20260708: QPS +138.88%, visited -68.90%, EF-sum -40.16%, P95/P99 -60.31%/-61.84%. Shuffled seed 20260711: QPS +154.43%, visited -71.88%, EF-sum -43.25%, P95/P99 -63.41%/-62.88%. Cold matched: QPS +140.85%, visited -68.89%, EF-sum -40.16%, P95/P99 -61.06%/-61.93%. Across the two shuffled seeds, Method4 query-order QPS vs warm original is +1.05% to +10.96%. | Selected 0.90 target-neighborhood robustness supplement only; seed 20260711 has Method4 recall 0.9013 versus naive 0.9068, so do not call it strict same-recall. Still not a full matrix-wide robustness proof. | results/method4_claim_coverage_20260704/derived_claim_tables/robustness_090_query_order_overlay.csv; results/method4_claim_coverage_20260704/derived_claim_tables/robustness_090_query_order_multiseed_summary.csv; results/method4_claim_coverage_20260704/derived_claim_tables/robustness_090_warmup_overlay.csv; results/method4_claim_coverage_20260704/derived_claim_tables/robustness_090_method4_vs_naive_deltas.csv; results/method4_claim_coverage_20260704/derived_claim_tables/robustness_090_sensitivity_deltas.csv; results/method4_robustness_090_original_20260705/analysis_20260705_065005/; results/method4_robustness_090_shuffle_20260708/analysis_20260705_065219/; results/method4_robustness_090_shuffle_20260711/analysis_20260706_132643/; results/method4_robustness_090_warmup_20260705/analysis_20260705_065435/ |
| E | compact_request_execution_latency | supported_with_overhead_metrics_gap | compact_current preserves recall while improving QPS and P95/P99 versus grouped materialized and client-expanded request modes in the current-runtime matrix. | Versus grouped materialized, compact_current QPS +66% to +170% and P95 -41% to -68% across batch sizes 50/100/200/400. Versus client-expanded, QPS +159% to +185% and P95 -61% to -67% for batch sizes 50/100/200; batch400 client-expanded fails with BrokenPipe. | Current-runtime comparison does not recreate the historical old binary; batch400 client-expanded has no latency point due error. | results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_execution_mode_latency_matrix.csv; results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_execution_mode_latency_deltas.csv; results/method4_claim_e_execution_mode_latency_20260704/analysis_20260704_225847/ |
| E | compact_request_payload_and_wire_bytes | supported_with_overhead_metrics_gap | compact_current substantially reduces serialized JSON request/response body bytes versus client-expanded request mode. | Request objects reduce 95.76%; request body bytes/query reduce 93.23%; selected batch200 response body bytes/query reduce 95.76%; total JSON body bytes/query reduce 93.71%. | Measures JSON bodies only, not HTTP framing, compression, or physical NIC traffic. | results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_payload_bytes_summary.csv; results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_wire_bytes_summary.csv; results/method4_claim_e_payload_bytes_20260705/analysis_20260705_014703/; results/method4_claim_e_wire_bytes_20260705/analysis_20260705_020000/ |
| E | runtime_overhead_and_build_gap | supported_with_overhead_metrics_gap | Current artifacts include selected planning time, Docker container CPU/network/memory, process RSS, host-interface byte counters, Docker bridge packet capture, physical-NIC negative-control packet capture, smoke-level build-stage time/RSS instrumentation, selected /proc controller/worker process-level CPU/RSS attribution, and an audit showing full-size Method4 build-stage time/RSS are missing. | Planning time is 0.250 ms/query compact_current, 0.167 grouped, 0.292 client-expanded. Docker selected batch200 records compact_current controller RX bytes/query -85.57% vs grouped and -93.00% vs client-expanded, and lower sampled controller CPU. Host-interface supplement records compact_current docker_bridge bytes/query -84.31% vs grouped and -93.59% vs client-expanded. Docker bridge packet capture records compact_current frame bytes/query -60.20% vs grouped and -75.27% vs client-expanded, and packet count/query -96.69% and -92.69%. Build audit: 322 Method4/Qdrant builds.csv snapshots lack build-duration/RSS columns. Physical-NIC negative control: ens6f0np0 and ens111f0np0 captured 0 matching Docker-subnet packets, 0 frame bytes, 0 TCP payload bytes, and 0 kernel drops during the batch=200 all-mode Claim E run. | Container/process counters, host-interface counters, Docker bridge packet capture, and selected /proc PID sampling are not Qdrant subsystem/function-level attribution; full-size build-stage wall time/RSS remains missing, while smoke-level build-stage time/RSS instrumentation is present. The physical-NIC supplement is a zero-packet single-host negative control, not physical-network payload savings. | results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_planning_time_summary.csv; results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_container_overhead_summary.csv; results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_process_rss_summary.csv; results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_build_stage_artifact_audit.csv; results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_host_interface_bytes_role_summary.csv; results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_host_interface_bytes_deltas.csv; results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_packet_capture_bridge_totals.csv; results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_packet_capture_bridge_deltas.csv; results/method4_claim_e_packet_capture_20260706/per_variant_repeats3_20260706_122755/; tools/method4_claim_e_packet_capture_summary.py; results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_packet_capture_physical_nic_totals.csv; results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_packet_capture_physical_nic_summary.csv; results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_packet_capture_physical_nic_execution_summary.csv; results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_packet_capture_physical_nic_manifest.json |
| E | host_interface_byte_counters | supported_with_overhead_metrics_gap | A selected batch=200 host-interface supplement shows compact_current greatly reduces Docker bridge/veth byte counters versus grouped materialized and client-expanded execution modes, while physical NIC host counters remain small on the single-host Docker deployment. | At batch=200 with 3000 queries * 3 repeats, compact_current docker_bridge bytes/query is 8604.1 vs 54831.2 grouped (-84.31%) and 134186.8 client-expanded (-93.59%). docker_veth bytes/query is 57651.6 vs 161046.1 grouped (-64.20%) and 234646.2 client-expanded (-75.43%). physical_nic host counters are small: 336.1 bytes/query compact, 362.7 grouped, 540.9 client-expanded. | Linux host-interface counters are not packet capture or Qdrant subsystem/function-level attribution; docker_bridge/docker_veth/loopback are single-host Docker interfaces and physical_nic is the host external NIC role. | results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_host_interface_bytes_role_summary.csv; results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_host_interface_bytes_deltas.csv; results/method4_claim_e_host_interface_bytes_20260705/analysis_20260705_061525/; tools/method4_claim_e_host_interface_bytes.py |
| F | worker_local_premerge_fanin | complete_with_runtime_caveat | Worker-local pre-merge reduces candidate fan-in shape from logical shard streams to peer-level streams. | Physical trace shows logical streams/query can drop from 23.208 to 2.977 physical peers/query; direct simulation reduces candidate streams by about 87.17%. | Count-level fan-in evidence is not serialized/network byte measurement. | results/method4_claim_coverage_20260704/derived_claim_tables/claim_f_worker_local_premerge.csv; docs/experiments/2026-06-03-method4-physical-execution-trace.md; docs/experiments/2026-06-03-method4-direct-peer-premerge-simulation.md |
| F | worker_local_premerge_same_image_server_ab | complete_with_runtime_caveat | Same-image server A/B shows peer pre-merge enabled has higher QPS than disabled at the selected Recall@10 about 0.955 window. | Enabled fresh Recall@10 0.955300 and QPS 405.38 vs disabled fresh Recall@10 0.955233 and QPS 379.42; QPS +6.84%; recall delta +0.000067. | Single final-eval window from the same image and selected 0.955 operating point; use for production QPS wording only, not production batch-size P95/P99. | results/method4_claim_coverage_20260704/derived_claim_tables/claim_f_worker_local_premerge.csv; results/qdrant_goal_recall_idea_095_server_peer_premerge_fresh/20260603_092757/summary.json; results/qdrant_goal_recall_idea_095_server_peer_premerge_disabled_fresh/20260603_092949/summary.json; docs/experiments/2026-06-03-method4-worker-local-peer-premerge.md |
| F | worker_local_premerge_enabled_10k_sanity | complete_with_runtime_caveat | An enabled peer pre-merge 10000-query run keeps the compact server execution shape and Recall@10 about 0.954, providing a longer-window sanity check for the enabled path. | 10000 queries with upper_k=160, base_ef=80, factor=8: Recall@10 0.95403, QPS 429.98, avg visited shards 23.0935, search requests/query 1.0, candidate groups/query 1.0, returned candidates/query 10.0. | Enabled-path sanity only: no disabled 10000-query paired baseline and no P95/P99 batch-size matrix; 10000-query recall should not be directly compared with 3000-query recall because it evaluates additional queries. | results/qdrant_goal_recall_idea_095_server_peer_premerge_10k/20260603_090738/summary.json; results/qdrant_goal_recall_idea_095_server_peer_premerge_10k/20260603_090738/final_metrics.csv; docs/experiments/2026-06-03-method4-worker-local-peer-premerge.md |
| F | premerge_batch_matrix | complete_with_runtime_caveat | In the direct-peer batch matrix, local pre-merge preserves recall and reduces candidate groups/returned candidates by 87.40%. | Across batch sizes 50/100/200/400, direct-peer local pre-merge vs no-premerge has recall delta 0 and candidate group / returned candidate reduction 87.3989%. QPS deltas are small and mixed, while tail latency is mostly flat/noisy; coordinator current vs no-premerge shows much higher QPS due different path. | Direct-peer simulation still serializes logical shard results before local Python pre-merge; production enabled-vs-disabled P95/P99 restart A/B is not present. | results/method4_claim_coverage_20260704/derived_claim_tables/claim_f_premerge_batch_latency_matrix.csv; results/method4_claim_coverage_20260704/derived_claim_tables/claim_f_premerge_batch_latency_deltas.csv; results/method4_claim_f_premerge_batch_latency_20260704/analysis_20260704_222700/ |
| G | method4_aware_placement | complete | Method4-aware physical layout reduces predicted hot-worker load and gives modest matched online latency/QPS improvements. | Offline placement matrix: method4-aware lowers P95 max worker EF by 4.9% to 8.0%. Online matched batch latency: QPS +1.2% to +2.1% and P95/P99 -0.4% to -1.0% versus round-robin depending on batch size. | Online gains are modest; avoid claiming large wins for every workload. | results/method4_claim_coverage_20260704/derived_claim_tables/claim_g_offline_placement_deltas.csv; results/method4_claim_coverage_20260704/derived_claim_tables/claim_g_online_batch_latency_summary.csv; results/method4_claim_coverage_20260704/derived_claim_tables/claim_g_online_batch_latency_deltas.csv; results/method4_claim_g_evidence_20260704/; results/method4_claim_coverage_20260704/derived_claim_tables/claim_g_raw_provenance_audit_20260708.csv; results/method4_claim_g_evidence_20260704/source_manifest.json; results/method4_claim_g_physical_layout_20260704/analysis_20260704_164039/; results/method4_claim_g_matched_layout_deploy_20260704/20260704_170511/; results/method4_claim_g_matched_layout_deploy_20260704/20260704_171202/; results/method4_claim_g_batch_latency_20260704/analysis_20260704_172015/ |
| G | method4_aware_concurrency8_supplement | complete | Under concurrent single-query load, the deployed Method4-aware placement improves QPS and P95/P99 versus deployed round-robin while preserving recall. | Concurrency=8, 1000 measured queries + 50 warmup, 3 repeats: method4-aware Recall@10 0.9587 vs round_robin 0.9575; QPS +2.30%; mean latency -2.28%; P95 -5.27%; P99 -7.91%; max -10.21%. | Supplement only: deployed collections are not byte-identical logical clones and method4-aware visits slightly more shards/EF-sum; use the matched batch matrix as primary physical-only A/B. | results/method4_claim_g_evidence_20260704/online_concurrency8_latency_summary.csv; results/method4_claim_g_evidence_20260704/online_concurrency8_latency_deltas.csv; docs/experiments/2026-07-04-method4-claim-g-physical-layout-evidence.md; results/method4_claim_coverage_20260704/derived_claim_tables/claim_g_raw_provenance_audit_20260708.csv; results/method4_claim_g_evidence_20260704/source_manifest.json; results/method4_claim_g_online_latency_20260704/analysis_20260704_165643/ |
| H | semantic_invariant_audit | complete | The compact_multi_ep wrapper preserves the intended external Method4 routing/search semantics in the sampled audit. | 100-query audit passes all 9 invariants: upper routing, point_to_shards, multi-assignment, per-shard entry points, dynamic EF, logical-shard lower search, no adaptive pruning, source-id dedup, and final global merge. | This is an external wrapper audit, not a C++ internal step-by-step reference trace. | results/method4_claim_h_semantic_audit_20260704/audit_20260704_210845/semantic_invariant_summary.csv; results/method4_claim_coverage_20260704/derived_claim_tables/claim_h_semantic_invariants_current_evidence.csv |
| CrossCut | reduced_sift100k_l2_generalization | partial_reduced_l2 | Reduced SIFT-100k Euclidean evidence provides smaller external-validity support above Recall@10 0.95, now including no-fission Orion, full-fission Orion, Simple KMeans, and Naive latency supplements. | Orion no-fission Recall 0.9613, QPS 2285.6, visited 1.99, EF-sum 64.3. Full-fission Orion fresh latency row Recall 0.9545, QPS 1812.9, visited 2.88, EF-sum 172.5, P95/P99 66.8/67.5 ms; versus fresh Naive, QPS +74.4%, visited shards -82.0%, EF-sum -32.6%, P95/P99 -42.3%/-43.5%. Simple KMeans Recall 0.9676, QPS 2013.0, visited 4.00. Naive Recall 0.9859. Dynamic-vs-Fixed EF reduced-L2 supplements show lower Dynamic EF-sum and positive QPS/median-tail deltas at near matched recall. | Reduced subset only; not full SIFT1M/GIST/Deep. L2 latency pairs are above-threshold but not strict same-recall proofs; full-fission Orion uses a fission-expanded 25-shard collection while Naive uses 16 shards. | results/method4_claim_coverage_20260704/derived_claim_tables/l2_reduced_sift100k_main_comparison.csv; results/method4_claim_coverage_20260704/derived_claim_tables/l2_reduced_sift100k_latency_matrix.csv; results/method4_claim_coverage_20260704/derived_claim_tables/l2_reduced_sift100k_latency_deltas.csv; results/method4_claim_coverage_20260704/derived_claim_tables/l2_reduced_sift100k_full_fission_latency_matrix.csv; results/method4_claim_coverage_20260704/derived_claim_tables/l2_reduced_sift100k_full_fission_latency_deltas.csv; results/method4_claim_coverage_20260704/derived_claim_tables/l2_reduced_sift100k_dynamic_vs_fixed_ef_deltas.csv; results/method4_l2_reduced_sift100k_latency_20260706/full_orion_method4_only/analysis_20260706_125222/; results/method4_l2_reduced_sift100k_latency_20260706/naive_only/analysis_20260706_125245/ |
| CrossCut | scalability_boundary | basic_worker_count_complete_comparative_partial | Current scalability evidence includes logical shard-count scaling on a fixed 3-worker deployment plus a basic GloVe Method4-only online worker-count scaling supplement across worker_count 1/2/3/4. | Worker-count supplement: GloVe, Method4 only, targets 0.80/0.85/0.90/0.95, 3000 eval queries, 100 warmup queries, batch_size=100, 3 repeats, 46 active custom shards. Worker_count=4 vs 1 QPS speedups are 1.36x/1.59x/1.76x/1.89x, with P95 reductions 27.30%/40.22%/42.48%/46.97% and P99 reductions 30.41%/40.37%/42.41%/48.82% for targets 0.80/0.85/0.90/0.95. Shard-count scaling latency rows still cover Orion, Simple KMeans, and Naive 31/46 rows with 3 repeats on a fixed 3-worker deployment. | The worker-count supplement is Method4-only and uses temporary controlled deployments with /home bind-mounted storage; it does not include Naive/Simple KMeans worker-count controls. The earlier shard-count rows should still not be presented as physical worker-count scaling. | results/method4_claim_coverage_20260704/derived_claim_tables/crosscut_shard_scaling_latency_095.csv; results/method4_claim_coverage_20260704/derived_claim_tables/crosscut_shard_scaling_latency_095_deltas.csv; results/method4_claim_coverage_20260704/derived_claim_tables/worker_count_live_deployment_audit_20260705.csv; results/method4_claim_coverage_20260704/derived_claim_tables/worker_count_online_scaling_summary_20260708.csv; results/method4_claim_coverage_20260704/derived_claim_tables/worker_count_online_scaling_deltas_20260708.csv; results/method4_claim_coverage_20260704/derived_claim_tables/worker_count_online_scaling_raw_rows_20260708.csv; results/method4_claim_coverage_20260704/derived_claim_tables/worker_count_docker_stats_summary_20260708.csv; results/method4_claim_coverage_20260704/derived_claim_tables/worker_count_online_scaling_manifest_20260708.json; results/method4_worker_count_online_scaling_20260708/; tools/method4_worker_count_online_scaling.py |
| CrossCut | strict_ablation_boundary | available_selected_latency_cells_with_caveats | The strict unified ablation status now has selected full-size multi-assignment online P95/P99 cells for single/default/w2c2, but they should be reported with target-neighborhood caveats rather than as strict same-recall proof across all pairs. | single Recall@10 0.97317/QPS 279.23/visited 30.18/EF-sum 8421.04/P95 736.45/P99 739.58 ms; default Recall@10 0.95260/QPS 475.27/visited 21.35/EF-sum 3088.52/P95 432.73/P99 435.41 ms; w2c2 Recall@10 0.94913/QPS 557.80/visited 21.24/EF-sum 2127.40/P95 375.01/P99 376.76 ms. | Current-harness rebuilds, not byte-identical historical replays; single assignment is above 0.95 recall and w2c2 is slightly below 0.95 on the 3000-query latency window, so use as selected ablation/target-neighborhood latency evidence rather than strict same-recall proof across all pairs. | results/method4_claim_coverage_20260704/derived_claim_tables/unified_ablation_status_matrix.csv; results/method4_claim_coverage_20260704/derived_claim_tables/strict_multiassign_latency_20260709.csv; results/method4_claim_coverage_20260704/derived_claim_tables/strict_ablation_live_collection_audit_20260709_post_multiassign.csv; results/method4_benchmark_matrix/method4_strict_multiassign_latency_rebuild_20260709/; results/method4_strict_ablation_multiassign_latency_20260709/ |
| E | qdrant_prometheus_metrics_supplement | supported_with_overhead_metrics_gap | At selected batch=200, Qdrant /metrics server-exposed REST/process counters corroborate the compact_current efficiency shape versus grouped materialized and client-expanded execution modes. | 3000 queries * 3 repeats: compact_current QPS 386.6 vs 201.0 grouped (+92.34%) and 138.5 client-expanded (+179.21%); P95 564.2 ms vs 1179.1 (-52.15%) and 1675.5 (-66.33%); Qdrant REST batch duration delta 7.09 s vs 10.51 (-32.49%) and 9.61 (-26.18%); minor page-fault delta -92.13% and -93.50%. | Qdrant /metrics counters are server-exposed REST/process counters, not packet capture, build-stage time/RSS, or Qdrant subsystem-level CPU/memory attribution; collection_hardware_metric_cpu was exposed but stayed 0 in this run, and memory gauges are noisy process snapshots. | results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_qdrant_metrics_summary.csv; results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_qdrant_metrics_deltas.csv; results/method4_claim_e_qdrant_metrics_20260705/analysis_20260705_073436/; tools/method4_claim_e_qdrant_metrics.py |
| D | method4_vs_naive_099_strategy_search_negative | partial_high_recall_caveat | Follow-up 0.99 strategy-search formal runs reinforce that current Method4 lower-EF/high-upper-k retuning does not reach Recall@10 0.99 on the 3000-query window. | m4_560_40_16, m4_560_60_14, m4_480_80_16, and m4_520_60_16 screened at Recall@10 0.9900-0.9906 on 500 queries but formal 3000-query recall was only 0.98670-0.98747. Same-run naive_ef200 stayed at 0.99017. The 2026-07-06 top-screen candidates had +12.59% to +14.70% QPS versus same-run naive_ef200 and 17.7-19.1% lower P95, but at -0.00277 to -0.00307 lower recall. | Negative/boundary result for the current retuning family; it does not prove no possible Method4 strategy can reach 0.99, only that this lower-EF/high-upper-k search region did not. | results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_099_strategy_search_summary.csv; results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_099_strategy_search_deltas.csv; results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_099_strategy_search_top_screen_summary.csv; results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_099_strategy_search_top_screen_deltas.csv; results/method4_claim_d_099_strategy_search_20260705/analysis_20260705_074939/; results/method4_claim_d_099_strategy_search_20260706/top_screen_candidates/analysis_20260706_130013/ |
| D | method4_vs_naive_099_broader_strategy_negative | partial_high_recall_caveat | A broader upper-routing 0.99 screen produced 500-query Method4 candidates above Recall@10 0.99, but both selected formal 3000-query repeats still fell below 0.99 recall and below paired naive recall. | 500-query screen: five Method4 configs reached Recall@10 0.9918-0.9920. Formal 3000 queries * 3 repeats: high-QPS candidate m4_u720_b60_f12 reached Recall@10 0.9880 vs same-run naive_ef200 0.99017, with QPS +3.26%, visited shards -13.85%, EF-sum +31.41%, and P95/P99 -10.48%/-11.89%. Top-recall candidate m4_u720_b40_f14 reached Recall@10 0.9887 vs same-run naive_ef200 0.99017, with QPS -0.64%, visited shards -13.82%, EF-sum +40.39%, and P95/P99 -5.50%/-7.35%. | The 500-query screen overestimated formal recall for both selected candidates; neither formal candidate reaches 0.99 recall or supports Method4 dominance at 0.99. This is negative/boundary evidence for this broader upper-routing strategy, not proof that no future routing/EF strategy can reach 0.99. | results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_099_broader_strategy_screen_summary.csv; results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_099_broader_strategy_formal_summary.csv; results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_099_broader_strategy_formal_deltas.csv; results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_099_broader_strategy_manifest.json; results/method4_claim_d_099_broader_screen_20260707/analysis_20260707_110802/; results/method4_claim_d_099_broader_formal_20260707/analysis_20260707_111008/; results/method4_claim_d_099_broader_top_recall_formal_20260707/analysis_20260707_112432/ |
| D | method4_vs_naive_099_alternate_width_negative | partial_high_recall_caveat | A 2026-07-08 alternate wider-upper-routing 0.99 screen found a promising 500-query Method4 candidate, but formal validation still fell below Recall@10 0.99 and below paired naive recall. | 500-query screen: m4_u960_b20_f10 Recall@10 0.9916, QPS 168.09 vs naive_ef200 Recall@10 0.9902, QPS 166.57. Formal 3000 queries * 3 repeats: m4_u960_b20_f10 Recall@10 0.9876 vs naive_ef200 0.99017, with QPS +3.20%, visited shards -9.66%, EF-sum +26.66%, and P95/P99 -10.16%/-11.89%; versus naive_ef240, recall gap -0.00603, QPS +12.28%, P95/P99 -16.84%/-19.10%. | Boundary/negative only: the 500-query screen overestimated formal recall; this does not support 0.99 Method4 dominance. | results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_099_alternate_width_screen_summary.csv; results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_099_alternate_width_formal_summary.csv; results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_099_alternate_width_formal_deltas.csv; results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_099_alternate_width_manifest.json; results/method4_claim_d_099_alternate_width_screen_20260708/analysis_20260708_095804/; results/method4_claim_d_099_alternate_width_formal_20260708/analysis_20260708_100010/ |
| E | docker_bridge_packet_capture | supported_with_overhead_metrics_gap | A selected batch=200 Docker bridge packet-capture supplement shows compact_current substantially reduces bridge packet count and frame/payload bytes versus grouped materialized and client-expanded execution modes. | 3000 queries * 3 repeats per mode on br-b9aac8010880: compact_current frame bytes/query 51579.5 vs 129598.5 grouped (-60.20%) and 208607.0 client-expanded (-75.27%); TCP payload bytes/query 51201.3 vs 118184.2 (-56.68%) and 203434.9 (-74.83%); packets/query 5.73 vs 172.94 (-96.69%) and 78.35 (-92.69%). tcpdump reported 0 kernel-dropped packets for all captures. | Single-host Docker bridge packet capture only; not physical-NIC payload attribution and not Qdrant subsystem/function-level CPU/memory attribution. Capture snaplen was 128 bytes, with byte totals derived from pcap original frame length plus IP/TCP headers. | results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_packet_capture_bridge_totals.csv; results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_packet_capture_bridge_deltas.csv; results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_packet_capture_bridge_summary.csv; results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_packet_capture_manifest.json; results/method4_claim_e_packet_capture_20260706/per_variant_repeats3_20260706_122755/; tools/method4_claim_e_packet_capture_summary.py |
| CrossCut | current_environment_preflight | current_scope_resolved_with_caveats | Continuation preflight and post-run audits distinguish historical blockers from current state: Claim A, worker-count, and selected strict multi-assignment GloVe rows are now resolved with caveats. | 2026-07-09 post-multiassign audit records selected full-size bench_ma_strict_* latency rows as available and temporary collections deleted; non-GloVe and 0.99 dominance remain out of current scope. | This is blocker/preflight evidence, not standalone performance evidence. Current rows are rebuilds with caveats; optional stronger wordings still require retunes or additional instrumentation. | results/method4_claim_coverage_20260704/derived_claim_tables/current_environment_preflight_20260706.csv; results/method4_claim_coverage_20260704/derived_claim_tables/current_hard_gap_blocker_audit_20260706.csv; results/method4_claim_coverage_20260704/derived_claim_tables/current_environment_preflight_20260707.csv; results/method4_claim_coverage_20260704/derived_claim_tables/current_hard_gap_blocker_audit_20260707.csv; results/method4_claim_coverage_20260704/derived_claim_tables/current_hard_gap_blocker_audit_20260708_post_worker_count.csv; results/method4_claim_coverage_20260704/derived_claim_tables/current_hard_gap_blocker_audit_20260709_post_claim_a.csv; results/method4_claim_coverage_20260704/derived_claim_tables/claim_a_partition_family_current_rebuilds_20260709.csv; results/method4_claim_coverage_20260704/derived_claim_tables/current_hard_gap_blocker_audit_20260709_post_multiassign.csv; results/method4_claim_coverage_20260704/derived_claim_tables/strict_multiassign_latency_20260709.csv |
| D | method4_vs_naive_099_boundary_query_order | partial_high_recall_caveat | Two shuffled-query repeats of the closest 0.99 Method4-vs-naive boundary row confirm that the current Method4 setting remains below Recall@10 0.99 and below the paired naive_ef200 recall. | Shuffled seeds 20260713 and 20260718, 3000 queries * 3 repeats: seed 20260713 m4_400_160_20 Recall@10 0.98923 vs naive_ef200 0.99017, QPS +0.77%, visited shards -25.50%, EF-sum +56.77%, P95/P99 -8.00%/-7.46%; seed 20260718 Recall@10 0.98943 vs naive_ef200 0.99017, QPS +0.52%, visited shards -25.55%, EF-sum +56.73%, P95/P99 -7.96%/-11.52%. The multiseed summary records Method4 shuffled recall 0.98923-0.98943 versus naive 0.99017. This is boundary evidence, not 0.99 dominance. | Method4 recall remains below 0.99 and below naive_ef200; use only as 0.99 boundary/negative robustness evidence. It does not close the 0.99 same-recall claim gap. | results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_099_boundary_query_order_overlay.csv; results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_099_boundary_method4_vs_naive_deltas.csv; results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_099_boundary_query_order_sensitivity.csv; results/method4_099_boundary_shuffle_20260713/analysis_20260706_140200/; results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_099_boundary_query_order_multiseed_summary.csv; results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_099_boundary_query_order_manifest.json; results/method4_099_boundary_shuffle2_20260718/analysis_20260707_102446/ |
| D | method4_vs_naive_099_boundary_warmup | partial_high_recall_caveat | A cold/no-warmup matched-window repeat of the closest 0.99 boundary row confirms that Method4 remains below Recall@10 0.99 and below the paired naive_ef200 recall; warmup changes latency/QPS shape but does not close the 0.99 same-recall gap. | Cold no-warmup query_start_offset=100, 3000 queries * 3 repeats: m4_400_160_20 Recall@10 0.98937 vs naive_ef200 0.99017; QPS +0.29%; visited shards -25.47%; EF-sum +56.79%; P95/P99 -7.69%/-9.00%. Cold-vs-warm sensitivity: Method4 QPS -5.95%, P95/P99 +7.08%/+4.96%; Naive QPS -8.69%, P95/P99 +16.59%/+16.17%. | Method4 recall remains below 0.99 and below naive_ef200; this is cold/warm sensitivity and transparent 0.99 boundary evidence, not 0.99 dominance. | results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_099_boundary_warmup_overlay.csv; results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_099_boundary_warmup_method4_vs_naive_deltas.csv; results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_099_boundary_warmup_sensitivity.csv; results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_099_boundary_sensitivity_deltas.csv; results/method4_099_boundary_warmup_20260713/analysis_20260706_141159/ |
| CrossCut | goal_completion_audit | current_scope_glove_selected_rows_complete_with_caveats | The evidence package distinguishes completed/downgraded claim support from optional stronger claims; basic worker-count, Claim A current-scope rows, and selected strict multi-assignment latency rows are complete with caveats for the current GloVe-only/no-0.99 scope. | Current-scope update: worker-count complete; Claim A random/topology/load-recalibrated current-harness rows complete; strict multi-assignment single/default/w2c2 selected latency rows complete. Non-GloVe/full-L2/GIST/Deep and 0.99 dominance are out of current scope. | Completion is for the selected current GloVe scope, not for stronger optional wordings such as strict same-recall single/default/w2c2 latency, full-fission Claim A dominance, comparative worker-count controls, full-size build-stage RSS/time, Qdrant subsystem profiling, or non-GloVe generalization. | results/method4_claim_coverage_20260704/derived_claim_tables/goal_completion_audit_20260706.csv; results/method4_claim_coverage_20260704/derived_claim_tables/current_hard_gap_blocker_audit_20260706.csv; results/method4_claim_coverage_20260704/claim_evidence_audit.csv; results/method4_claim_coverage_20260704/remaining_experiment_gaps.csv; results/method4_claim_coverage_20260704/derived_claim_tables/goal_completion_audit_20260707.csv; results/method4_claim_coverage_20260704/derived_claim_tables/current_hard_gap_blocker_audit_20260707.csv; results/method4_claim_coverage_20260704/derived_claim_tables/goal_completion_audit_20260708.csv; results/method4_claim_coverage_20260704/derived_claim_tables/goal_completion_audit_20260709.csv; results/method4_claim_coverage_20260704/derived_claim_tables/current_hard_gap_blocker_audit_20260709_post_claim_a.csv; results/method4_claim_coverage_20260704/derived_claim_tables/worker_count_online_scaling_summary_20260708.csv; results/method4_claim_coverage_20260704/derived_claim_tables/claim_a_partition_family_current_rebuilds_20260709.csv; results/method4_claim_coverage_20260704/derived_claim_tables/goal_completion_audit_20260709_post_multiassign.csv; results/method4_claim_coverage_20260704/derived_claim_tables/strict_multiassign_latency_20260709.csv |
| E | physical_nic_packet_capture_negative_control | supported_with_overhead_metrics_gap | In the single-host Docker Claim E execution window, the external physical NICs carried no matching Docker-subnet Qdrant traffic. | During a batch=200 all-mode Claim E run (grouped_by_ef_materialized, compact_current, client_shard_major_expanded; 3000 queries * 3 repeats each; 27000 total query executions in the capture window), tcpdump on ens6f0np0 and ens111f0np0 with filter tcp and net 172.24.0.0/16 recorded 0 packets, 0 frame bytes, 0 TCP payload bytes, and 0 kernel drops on both NICs. Concurrent execution summary: compact_current QPS mean 434.93, grouped 218.15, client-expanded 152.42. | This is a physical-NIC negative control for a single-host Docker deployment; it shows Docker-subnet traffic stayed off external NICs. It is not Qdrant subsystem/function-level attribution and not proof of physical-network payload savings in a multi-host deployment. | results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_packet_capture_physical_nic_totals.csv; results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_packet_capture_physical_nic_summary.csv; results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_packet_capture_physical_nic_execution_summary.csv; results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_packet_capture_physical_nic_manifest.json; results/method4_claim_e_packet_capture_physical_nic_20260707/all_variants_repeats3_preload_20260707_082833; tools/method4_claim_e_packet_capture_summary.py |
| E | build_stage_instrumented_smoke | supported_with_overhead_metrics_gap | A smoke-level fresh Method4/Qdrant collection build now has wall-time and max-RSS instrumentation via /usr/bin/time -v. | train_limit=1000; 1000 logical points; 1381 assigned points; 1376 indexed vectors; 7 active custom shards across 4 peers; elapsed wall time 37.82 s; max RSS 65,744 KB; exit status 0. | Smoke only: timing wraps fresh collection creation/indexing plus 20-query tuning/eval, not a full-size Method4 distributed build and not pure build-only profiling. Full-size build-stage time/RSS and Qdrant subsystem/function-level CPU/memory attribution remain missing; selected process-level PID attribution is now present. | results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_build_stage_instrumented_smoke_20260707.csv; results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_build_stage_instrumented_smoke_manifest.json; results/method4_claim_e_build_stage_smoke_20260707/claim_e_build_smoke_20260707_085303/20260707_085328; results/method4_claim_e_build_stage_smoke_20260707/raw_claim_e_build_smoke_20260707_085303/stdout.txt; results/method4_claim_e_build_stage_smoke_20260707/raw_claim_e_build_smoke_20260707_085303/time_v_stderr.txt; results/method4_claim_e_build_stage_smoke_20260707/claim_e_build_smoke_20260707_085303/20260707_085328/summary.json; results/method4_claim_e_build_stage_smoke_20260707/claim_e_build_smoke_20260707_085303/20260707_085328/builds.csv; results/method4_claim_coverage_20260704/derived_claim_tables/current_environment_preflight_20260707_build_smoke.csv |
| E | controller_process_attribution | supported_with_overhead_metrics_gap | A selected Claim E batch=200 supplement provides process-level controller and worker Qdrant PID CPU/RSS attribution for the three execution modes. | compact_current controller CPU delta 13.79 s vs 54.12 s grouped (-74.52%) and 44.85 s client-expanded (-69.25%); controller mean RSS 297.46 MiB vs 406.57 MiB grouped (-26.84%) and 386.28 MiB client-expanded (-22.99%); cluster CPU delta 206.69 s vs 395.02 s grouped (-47.68%) and 261.58 s client-expanded (-20.98%). | This is process-level host /proc attribution for Qdrant container PIDs during the whole benchmark command window, including route-state recovery/setup, warmup, measured repeats, and client orchestration; it is not Qdrant subsystem/function-level CPU or memory attribution. Full-size build-stage time/RSS and Qdrant subsystem/function-level attribution remain missing. | results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_controller_process_attribution_summary.csv; results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_controller_process_attribution_deltas.csv; results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_controller_process_attribution_manifest.json; results/method4_claim_e_controller_process_attribution_20260707/batch200_all_variants_20260707_091900; results/method4_claim_e_controller_process_attribution_20260707/batch200_all_variants_20260707_091900/run_manifest.json; results/method4_claim_e_controller_process_attribution_20260707/batch200_all_variants_20260707_091900/raw_run_rows.csv; results/method4_claim_e_controller_process_attribution_20260707/batch200_all_variants_20260707_091900/process_pid_map.json |
| B | strict_multiassign_latency_20260709 | complete_with_selected_latency_caveats | Selected full-size strict multi-assignment cells now have 3-repeat client-observed batch P95/P99 latency evidence for single/default/w2c2 in the 0.95 target neighborhood. | single Recall@10 0.97317/QPS 279.23/visited 30.18/EF-sum 8421.04/P95 736.45/P99 739.58 ms; default Recall@10 0.95260/QPS 475.27/visited 21.35/EF-sum 3088.52/P95 432.73/P99 435.41 ms; w2c2 Recall@10 0.94913/QPS 557.80/visited 21.24/EF-sum 2127.40/P95 375.01/P99 376.76 ms. | Current-harness rebuilds, not byte-identical historical replays; single assignment is above 0.95 recall and w2c2 is slightly below 0.95 on the 3000-query latency window, so use as selected ablation/target-neighborhood latency evidence rather than strict same-recall proof across all pairs. | results/method4_claim_coverage_20260704/derived_claim_tables/strict_multiassign_latency_20260709.csv; results/method4_benchmark_matrix/method4_strict_multiassign_latency_rebuild_20260709/; results/method4_strict_ablation_multiassign_latency_20260709/; results/method4_claim_coverage_20260704/derived_claim_tables/strict_ablation_live_collection_audit_20260709_post_multiassign.csv |

## Claim Coverage Overview

| Claim | Status | Supported evidence | Missing / weak | Primary artifacts |
| --- | --- | --- | --- | --- |
| A | partial | Online Orion vs Simple KMeans vs Naive at 0.80/0.85/0.90/0.95; same avg EF comparison shows Orion visits 47-55% fewer shards than simple KMeans and has 1.16-1.99x QPS. Offline 6-partition oracle matrix over upper_k {80,120,160} shows topology convergence improves edge cut, GT-shard entropy, routed shards, and upper_k 80/120 oracle miss versus KMeans-only. Online P95/P99 submatrix for existing full-fission and balanced-KMeans live collections at Recall@10 ~=0.953 shows both reduce visited shards and tail latency versus naive. A selected topology/no-fission 0.95 cell is present: Recall@10 0.9522, QPS 430.6, visited shards 18.94, P95/P99 479.6/483.3 ms. Same-run sensitivity shows the alternative u160,b120,f8 point is worse than selected u160,b80,f10: QPS -3.63%, EF-sum +12.86%, and P95/P99 +4.21%/+4.53% versus selected. New 2026-07-09 current-harness rebuild rows now cover the previously missing random_balanced_46, kmeans_topology_46, and kmeans_topology_load_recalibrated_46 online latency cells: random has Recall@10 0.9587/QPS 320.7/visited 44.16/P95 645.6/P99 646.6 ms versus paired naive QPS 230.3/P95 932.6/P99 957.5; topology has Recall@10 0.9504/QPS 420.2/visited 18.63/P95 520.7/P99 532.0 versus paired naive QPS 213.3/P95 1158.1/P99 1262.9; load-recalibrated has Recall@10 0.9502/QPS 444.0/visited 18.62/P95 470.2/P99 476.5 versus paired naive QPS 236.8/P95 893.2/P99 916.0. A partition-family coverage audit now records that those current-scope online rows are available and that temporary collections were deleted after preserving results. | Strong full Method4/fission superiority over balanced KMeans-only is not supported by the offline matrix: current full fission has worse oracle miss and more routed shards than KMeans-only at upper_k 80/120/160. The 2026-07-09 random/topology/load-recalibrated online rows are current-harness rebuilds with matched Claim A family parameters and build upper_search_ef=400, but they are not byte-identical to the 2026-07-04 oracle artifact. The load-recalibrated no-fission route map matches the topology variant while weights are recalibrated, so it should not be interpreted as independent evidence that load recalibration changes no-fission routing. Balanced-KMeans batch=200 naive has high variance, and topology/no-fission remains a visited-shard vs QPS/tail-latency tradeoff relative to full fission. | docs/experiments/2026-06-26-orion-vs-simple-kmeans-same-avg-ef.md; results/method4_claim_coverage_20260704/derived_claim_tables/claim_a_online_orion_vs_simple_kmeans_same_avg_ef.csv; results/method4_claim_coverage_20260704/derived_claim_tables/claim_a_partition_oracle_matrix.csv; results/method4_claim_coverage_20260704/derived_claim_tables/claim_a_partition_online_latency_submatrix.csv; results/method4_claim_coverage_20260704/derived_claim_tables/claim_a_partition_online_latency_submatrix_deltas.csv; results/method4_claim_coverage_20260704/derived_claim_tables/claim_a_topology_no_fission_config_sensitivity.csv; results/method4_claim_coverage_20260704/derived_claim_tables/claim_a_partition_family_online_coverage_audit.csv; results/method4_claim_coverage_20260704/derived_claim_tables/claim_a_partition_family_current_rebuilds_20260709.csv; results/method4_claim_a_partition_oracle_20260704/analysis_20260704_213047/; results/method4_claim_a_partition_online_latency_20260704/; results/method4_claim_a_partition_online_latency_20260709/; results/method4_claim_a_partition_family_build_20260709/; results/strict_ablation_topology_no_fission_20260705/reuse_tune_095/20260705_012313/ |
| B | complete | Online expansion <=2.0x sweep and tuning-frontier analysis show Orion multi-assignment raises recall-QPS frontier; offline oracle analysis over 3000 queries shows Orion default/w2c2 reduce oracle_gt_miss@10 versus single assignment at upper_k {80,120,160}. Raw provenance audit indexes online frontier rollups/source summaries, strict quality rollup, and oracle-miss raw analysis. A new 2026-07-09 selected full-size strict latency supplement covers single/default/w2c2 at the 0.95 target neighborhood with 3000 queries * 3 repeats and batch_size=200: single Recall@10 0.97317/QPS 279.23/visited 30.18/EF-sum 8421.04/P95 736.45/P99 739.58 ms; default Recall@10 0.95260/QPS 475.27/visited 21.35/EF-sum 3088.52/P95 432.73/P99 435.41 ms; w2c2 Recall@10 0.94913/QPS 557.80/visited 21.24/EF-sum 2127.40/P95 375.01/P99 376.76 ms. | Historical online frontier QPS rows remain mostly single final-eval runs without stability repeats. The 2026-07-09 latency supplement is a current-harness rebuild rather than byte-identical historical replay; single is above 0.95 recall and w2c2 is slightly below 0.95, so it is selected ablation/target-neighborhood latency evidence rather than a strict same-recall proof across all pairs. | docs/experiments/2026-06-29-method4-multiassign-expansion-qps.md; docs/experiments/2026-06-30-method4-multiassign-recall-qps-frontier.md; results/method4_claim_coverage_20260704/derived_claim_tables/claim_b_multiassign_expansion_qps.csv; results/method4_claim_coverage_20260704/derived_claim_tables/claim_b_orion_oracle_gt_miss.csv; results/method4_claim_b_oracle_miss_20260704/analysis_20260704_211749/; results/method4_claim_coverage_20260704/derived_claim_tables/claim_b_raw_provenance_audit_20260708.csv; results/method4_benchmark_matrix/method4_multiassign_recall_qps_excel_20260630/final_eval_keypoints_long.csv; results/method4_benchmark_matrix/method4_multiassign_recall_qps_excel_20260630_strict/final_eval_keypoints_long_strict.csv; results/method4_claim_b_oracle_miss_20260704/analysis_20260704_211749/oracle_gt_miss_summary.csv; results/method4_claim_b_oracle_miss_20260704/analysis_20260704_211749/run_metadata.json; results/method4_claim_coverage_20260704/derived_claim_tables/strict_multiassign_latency_20260709.csv; results/method4_benchmark_matrix/method4_strict_multiassign_latency_rebuild_20260709/; results/method4_strict_ablation_multiassign_latency_20260709/ |
| C | supported_with_batch_latency_supplement | Same-recall Dynamic EF vs fixed EF: +27-31% QPS and -35-41% EF-sum/query; routed EP count strongly predicts GT-hit shards. New 3-repeat batch-latency supplement at Recall@10 ~=0.949 shows Dynamic EF has +26.75% QPS, -32.52% EF-sum, and -21.00%/-20.97% P95/P99 batch latency versus fixed EF with the same visited-shard count. A reduced SIFT-100k Euclidean no-fission Orion Dynamic-vs-Fixed EF supplement now exists: nearest fixed-EF point differs by -0.0004 recall dynamic-minus-fixed, Dynamic uses -10.18% EF-sum, mean QPS is +15.56%, and median P95/P99 are -1.86%/-4.00%; mean P95/P99 are noisy and should be caveated. A full-fission reduced-L2 Orion Dynamic-vs-Fixed EF supplement also exists: nearest fixed-EF point differs by -0.0008 recall dynamic-minus-fixed, Dynamic uses -16.87% EF-sum, mean/median QPS are +17.08%/+16.50%, mean P95 is roughly flat, median P95/P99 are -11.30%/-10.89%, and mean P99 is noisy. Raw provenance audit now indexes the same-recall performance runs, offline relevance analysis, and batch-latency supplement. | The latency supplement measures client-observed batch endpoint wall time, not server-internal lower-search trace time. Internal lower-search P99 still requires Qdrant/server instrumentation. Reduced-L2 supplement is smaller-scale and has noisy mean tail latency; use it only as external-validity support, not as a replacement for the main Claim C GloVe evidence. | docs/experiments/2026-07-04-method4-claim-c-dynamic-ef-evidence.md; results/method4_claim_c_evidence_20260704/; results/method4_claim_c_evidence_20260704/claim_c_dynamic_vs_fixed_latency_matrix.csv; results/method4_claim_c_evidence_20260704/claim_c_dynamic_vs_fixed_latency_deltas.csv; results/method4_claim_coverage_20260704/derived_claim_tables/claim_c_dynamic_vs_fixed_latency_matrix.csv; results/method4_claim_coverage_20260704/derived_claim_tables/claim_c_dynamic_vs_fixed_latency_deltas.csv; results/method4_claim_c_dynamic_vs_fixed_latency_20260704/analysis_20260704_235925/; results/method4_claim_coverage_20260704/derived_claim_tables/l2_reduced_sift100k_dynamic_vs_fixed_ef_matrix.csv; results/method4_claim_coverage_20260704/derived_claim_tables/l2_reduced_sift100k_dynamic_vs_fixed_ef_deltas.csv; results/method4_l2_reduced_sift100k_latency_20260705/dynamic_vs_fixed_nofission/analysis_20260705_051254; results/method4_l2_reduced_sift100k_latency_20260705/dynamic_vs_fixed_nofission_fine/analysis_20260705_051329; results/method4_l2_reduced_sift100k_latency_20260705/full_orion_dynamic_vs_fixed/analysis_20260705_051940; results/method4_claim_coverage_20260704/derived_claim_tables/claim_c_raw_provenance_audit_20260708.csv; results/method4_claim_c_evidence_20260704/source_manifest.json; results/method4_claim_c_orion_fixed_ef_extended_20260625/20260625_122713/; results/method4_claim_c_orion_dynamic_ef_20260625/20260625_122853/; results/method4_claim_c_orion_frontier_fixed_095_robust_confirm_20260625/20260625_132656/; results/method4_claim_c_orion_frontier_dynamic_095_robust_confirm_20260625/20260625_132811/; results/method4_claim_c_orion_frontier_fixed_097_robust_confirm_20260625/20260625_132210/; results/method4_claim_c_orion_frontier_dynamic_097_more_robust_confirm_20260625/20260625_132923/; results/method4_claim_c_dynamic_ef_relevance_20260704/analysis_20260704_141217/ |
| D | complete_for_080_095_scope_with_099_excluded | 0.95 same-recall stable comparison: Method4 Recall 0.955267/QPS 387.14/visited 23.214 vs naive Recall 0.954767/QPS 272.37/visited 46; +42.14% QPS. Curves cover 0.80-0.95. New 3000-query * 3-repeat batch-latency rows cover ~0.97 and ~0.99: at ~0.97 Method4 has +40.58% QPS, -46.39% visited shards, -31.85% P95, and -32.42% P99 versus naive. A 2026-07-05 targeted 0.99 retune tried 20 lower-EF/high-upper-k candidates in 500-query screening and two 3000-query * 3-repeat formal candidates. Selected 2026-07-05 shuffled-query overlays of the ~0.97 row cover two seeds, and selected robustness overlays now cover ~0.955, ~0.90, and combined ~0.80/~0.85 warm/shuffled/cold windows; the ~0.955 row now has two shuffled-query seeds (20260707 and 20260710), the ~0.90 row has two shuffled-query seeds (20260708 and 20260711), and the combined ~0.80/~0.85 row has two shuffled-query seeds (20260709 and 20260712). Two follow-up 0.99 strategy-search formal runs repeated four additional 500-query candidates that had reached Recall@10 0.9900-0.9906 in screening: m4_560_40_16, m4_560_60_14, m4_480_80_16, and m4_520_60_16. On the 3000-query * 3-repeat formal windows they reached only 0.98670-0.98747 recall, while same-run naive_ef200 remained at 0.99017, strengthening the 0.99 boundary/negative result. Two 0.99 boundary shuffled-query repeats (seeds 20260713 and 20260718) now cover m4_400_160_20 vs naive_ef200 at 3000 queries * 3 repeats: seed 20260713 records Method4 recall 0.98923 vs naive 0.99017, QPS +0.77%, visited -25.50%, EF-sum +56.77%, and P95/P99 -8.00%/-7.46%; seed 20260718 records Method4 recall 0.98943 vs naive 0.99017, QPS +0.52%, visited -25.55%, EF-sum +56.73%, and P95/P99 -7.96%/-11.52%. Both reinforce that this is boundary evidence rather than a 0.99 same-recall win. A matched cold/no-warmup 0.99 boundary repeat (query_start_offset=100, warmup_query_count=0) now covers the same measured HDF5 window as the original warm row: Method4 recall 0.98937 vs naive 0.99017, QPS +0.29%, visited -25.47%, EF-sum +56.79%, and P95/P99 -7.69%/-9.00%; it reinforces boundary/warmup sensitivity rather than 0.99 dominance. A 2026-07-07 additional ~0.95 curve-config robustness supplement now covers m4_curve_160_50_10 versus naive_ef76 under warm original order, shuffled seeds 20260714 and 20260715, and cold/no-warmup matched-window evaluation. Across those 3000-query * 3-repeat windows, Method4 QPS is +69.05% to +73.97%, visited shards -48.64% to -48.70%, EF-sum -15.52% to -15.55%, P95 -45.31% to -46.98%, and P99 -44.12% to -48.07% versus same-run naive; Method4 recall is 0.9530-0.9538 versus naive 0.9543, so these are target-neighborhood robustness rows. A 2026-07-07 additional ~0.97 closest-neighborhood robustness supplement covers repeat_m4_160_80_16 vs naive_ef104 under warm original, shuffled seeds 20260716/20260717, and cold matched-window scenarios; Method4 QPS is +41.09% to +57.27%, visited shards -48.63% to -48.65%, EF-sum -1.22% to -1.24%, P95 -28.09% to -41.45%, and P99 -28.07% to -43.32% versus same-run naive, with Method4 recall slightly below naive by 0.00087-0.00130. A 2026-07-07 broader upper-routing 500-query screen plus selected 3000-query * 3-repeat formal follow-up is now present; selected m4_u720_b60_f12 and m4_u720_b40_f14 did not preserve 0.99 recall formally (0.9880 and 0.9887 vs same-run naive_ef200 0.99017), strengthening the 0.99 caveat. A 2026-07-08 alternate wider-upper-routing 500-query screen selected m4_u960_b20_f10 at Recall@10 0.9916 and QPS 168.09, but the formal 3000-query * 3-repeat validation reached only Recall@10 0.9876 versus same-run naive_ef200 at 0.99017; it kept +3.20% QPS and -10.16%/-11.89% P95/P99 versus naive_ef200, so it strengthens the 0.99 boundary/negative result rather than closing the same-recall gap. New 2026-07-08 batch=200 high-recall supplement covers the planned batch=200 shape for the selected ~0.97 and ~0.99 Claim D pairs: m4_160_80_20_b200 vs naive_ef112_b200 gives Recall@10 0.97183/0.97277, QPS +88.01%, visited -46.40%, and P95/P99 mean -67.32%/-69.33%; m4_400_160_20_b200 vs naive_ef200_b200 gives Recall@10 0.98950/0.99017, QPS +30.53%, visited -25.48%, and P95/P99 mean -54.26%/-58.41%. | The strong high-recall claim is not supported at ~0.99. The original selected Method4 0.99 point still visits 25.50% fewer shards but has -2.63% QPS, +0.50% P95, and +0.71% P99 versus the closest naive point. Lower-EF/high-upper-k retune and strategy-search candidates improve QPS/tail versus same-run naive controls in several comparisons but do not reach 0.99 recall on 3000 queries: formal candidates reached only 0.98670-0.98747 while same-run naive_ef200 stayed at 0.99017. Method4 EF-sum remains higher than naive at selected near-0.99 operating points. Robustness overlays cover selected ~0.80, ~0.85, ~0.90, ~0.955, ~0.97, plus an additional ~0.95 curve configuration, not the entire recall/config matrix; the selected rows now have two shuffled-query seeds, but this is still not the entire recall/config matrix; the ~0.90 seed 20260711 row and the ~0.85 rows are target-neighborhood evidence, not strict same-recall. The two 0.99 boundary shuffled rows still do not close the 0.99 same-recall gap because Method4 remains below Recall@10 0.99 and below naive_ef200 recall in both seeds. The 0.99 boundary cold/no-warmup row also remains below Recall@10 0.99 and below naive_ef200 recall, so it does not close the 0.99 same-recall gap. The new repeat_m4_160_80_16 rows are target-neighborhood evidence, not strict same-recall, because Method4 recall is slightly below same-run naive recall. Broader upper-routing has now been checked for one selected candidate and still does not close the 0.99 same-recall gap. The 2026-07-08 alternate wider-routing row also failed to preserve 0.99 recall on the formal window: m4_u960_b20_f10 reached 0.9876 versus naive_ef200 0.99017 and naive_ef240 0.99363. This confirms that the additional width/EF neighborhood is still boundary/negative evidence, not 0.99 dominance. Batch=200 Naive controls show high tail/QPS variance, so the batch=200 deltas should be reported as strict same-batch evidence with an explicit baseline-instability caveat; the batch=200 0.99 Method4 row remains below 0.99 recall and below paired naive recall. Under the current scoped request, Recall@10 0.99 dominance is explicitly out of scope; use the 0.80/0.85/0.90/0.95 and through-about-0.97 evidence and keep 0.99 only as boundary/negative context. | docs/experiments/2026-06-03-current-method4-vs-naive-same-recall.md; results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_method4_vs_naive_same_recall.csv; results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_high_recall_latency_summary.csv; results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_high_recall_latency_pairs.csv; results/method4_claim_d_high_recall_latency_20260704/analysis_20260704_215448/; results/method4_claim_d_high_recall_latency_refine_20260704/analysis_20260704_220036/; results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_099_retune_summary.csv; results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_099_retune_deltas.csv; results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_099_retune_tuning_candidates_500q.csv; results/method4_claim_d_099_retune_20260705/analysis_20260705_022625/; results/method4_claim_coverage_20260704/derived_claim_tables/robustness_shuffled_query_overlay.csv; results/method4_claim_coverage_20260704/derived_claim_tables/robustness_shuffled_query_overlay_deltas.csv; results/method4_robustness_shuffle_20260705/analysis_20260705_030323/; results/method4_claim_coverage_20260704/derived_claim_tables/robustness_warmup_overlay.csv; results/method4_claim_coverage_20260704/derived_claim_tables/robustness_warmup_overlay_deltas.csv; results/method4_robustness_warmup_20260705/analysis_20260705_044122/; results/method4_robustness_shuffle_20260706/analysis_20260705_045135/; results/method4_claim_coverage_20260704/derived_claim_tables/robustness_query_order_multiseed_summary.csv; results/method4_claim_coverage_20260704/derived_claim_tables/robustness_095_query_order_overlay.csv; results/method4_claim_coverage_20260704/derived_claim_tables/robustness_095_warmup_overlay.csv; results/method4_claim_coverage_20260704/derived_claim_tables/robustness_095_method4_vs_naive_deltas.csv; results/method4_claim_coverage_20260704/derived_claim_tables/robustness_095_sensitivity_deltas.csv; results/method4_robustness_095_original_20260705/analysis_20260705_063443/; results/method4_robustness_095_shuffle_20260707/analysis_20260705_063719/; results/method4_robustness_095_warmup_20260705/analysis_20260705_063957/; results/method4_claim_coverage_20260704/derived_claim_tables/robustness_090_query_order_overlay.csv; results/method4_claim_coverage_20260704/derived_claim_tables/robustness_090_warmup_overlay.csv; results/method4_claim_coverage_20260704/derived_claim_tables/robustness_090_method4_vs_naive_deltas.csv; results/method4_claim_coverage_20260704/derived_claim_tables/robustness_090_sensitivity_deltas.csv; results/method4_robustness_090_original_20260705/analysis_20260705_065005/; results/method4_robustness_090_shuffle_20260708/analysis_20260705_065219/; results/method4_robustness_090_warmup_20260705/analysis_20260705_065435/; results/method4_claim_coverage_20260704/derived_claim_tables/robustness_080_085_query_order_overlay.csv; results/method4_claim_coverage_20260704/derived_claim_tables/robustness_080_085_warmup_overlay.csv; results/method4_claim_coverage_20260704/derived_claim_tables/robustness_080_085_method4_vs_naive_deltas.csv; results/method4_claim_coverage_20260704/derived_claim_tables/robustness_080_085_sensitivity_deltas.csv; results/method4_robustness_080_085_original_20260705/analysis_20260705_070956/; results/method4_robustness_080_085_shuffle_20260709/analysis_20260705_071249/; results/method4_robustness_080_085_warmup_20260705/analysis_20260705_071529/; results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_099_strategy_search_summary.csv; results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_099_strategy_search_deltas.csv; results/method4_claim_d_099_strategy_search_20260705/analysis_20260705_074939/; results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_099_strategy_search_top_screen_summary.csv; results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_099_strategy_search_top_screen_deltas.csv; results/method4_claim_d_099_strategy_search_20260706/top_screen_candidates/analysis_20260706_130013/; results/method4_claim_coverage_20260704/derived_claim_tables/robustness_095_query_order_multiseed_summary.csv; results/method4_robustness_095_shuffle_20260710/analysis_20260706_131447/; results/method4_claim_coverage_20260704/derived_claim_tables/robustness_090_query_order_multiseed_summary.csv; results/method4_robustness_090_shuffle_20260711/analysis_20260706_132643/; results/method4_claim_coverage_20260704/derived_claim_tables/robustness_080_085_query_order_multiseed_summary.csv; results/method4_robustness_080_085_shuffle_20260712/analysis_20260706_133627/; results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_099_boundary_query_order_overlay.csv; results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_099_boundary_method4_vs_naive_deltas.csv; results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_099_boundary_query_order_sensitivity.csv; results/method4_099_boundary_shuffle_20260713/analysis_20260706_140200/; results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_099_boundary_warmup_overlay.csv; results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_099_boundary_warmup_method4_vs_naive_deltas.csv; results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_099_boundary_warmup_sensitivity.csv; results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_099_boundary_sensitivity_deltas.csv; results/method4_099_boundary_warmup_20260713/analysis_20260706_141159/; results/method4_claim_coverage_20260704/derived_claim_tables/robustness_095_curve_config_overlay.csv; results/method4_claim_coverage_20260704/derived_claim_tables/robustness_095_curve_config_method4_vs_naive_deltas.csv; results/method4_claim_coverage_20260704/derived_claim_tables/robustness_095_curve_config_sensitivity_deltas.csv; results/method4_claim_coverage_20260704/derived_claim_tables/robustness_095_curve_config_query_order_multiseed_summary.csv; results/method4_claim_coverage_20260704/derived_claim_tables/robustness_095_curve_config_manifest.json; results/method4_robustness_095_curve_config_original_20260707/analysis_20260707_093724/; results/method4_robustness_095_curve_config_shuffle_20260707/analysis_20260707_093953/; results/method4_robustness_095_curve_config_shuffle2_20260707/analysis_20260707_094525/; results/method4_robustness_095_curve_config_warmup_20260707/analysis_20260707_094220/; results/method4_claim_coverage_20260704/derived_claim_tables/robustness_097_closest_overlay.csv; results/method4_claim_coverage_20260704/derived_claim_tables/robustness_097_closest_method4_vs_naive_deltas.csv; results/method4_claim_coverage_20260704/derived_claim_tables/robustness_097_closest_sensitivity_deltas.csv; results/method4_claim_coverage_20260704/derived_claim_tables/robustness_097_closest_query_order_multiseed_summary.csv; results/method4_claim_coverage_20260704/derived_claim_tables/robustness_097_closest_manifest.json; results/method4_robustness_097_closest_shuffle_20260707/analysis_20260707_095706/; results/method4_robustness_097_closest_shuffle2_20260707/analysis_20260707_095948/; results/method4_robustness_097_closest_warmup_20260707/analysis_20260707_100229/; results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_099_boundary_query_order_multiseed_summary.csv; results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_099_boundary_query_order_manifest.json; results/method4_099_boundary_shuffle2_20260718/analysis_20260707_102446/; results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_099_broader_strategy_screen_summary.csv; results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_099_broader_strategy_formal_summary.csv; results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_099_broader_strategy_formal_deltas.csv; results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_099_broader_strategy_manifest.json; results/method4_claim_d_099_broader_screen_20260707/analysis_20260707_110802/; results/method4_claim_d_099_broader_formal_20260707/analysis_20260707_111008/; results/method4_claim_d_099_broader_top_recall_formal_20260707/analysis_20260707_112432/; results/method4_claim_d_099_broader_top_recall_formal_20260707/analysis_20260707_112432/claim_d_high_recall_latency_summary.csv; results/method4_claim_d_099_broader_top_recall_formal_20260707/analysis_20260707_112432/run_metadata.json; results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_099_alternate_width_screen_summary.csv; results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_099_alternate_width_formal_summary.csv; results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_099_alternate_width_formal_deltas.csv; results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_099_alternate_width_manifest.json; results/method4_claim_d_099_alternate_width_screen_20260708/analysis_20260708_095804/; results/method4_claim_d_099_alternate_width_formal_20260708/analysis_20260708_100010/; results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_batch200_high_recall_summary.csv; results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_batch200_high_recall_deltas.csv; results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_batch200_high_recall_manifest.json; results/method4_claim_d_batch200_targets_20260708/analysis_20260708_104611/; results/method4_claim_d_batch200_targets_20260708/analysis_20260708_104611/claim_d_high_recall_latency_summary.csv; results/method4_claim_d_batch200_targets_20260708/analysis_20260708_104611/run_metadata.json |
| E | supported_with_overhead_metrics_gap | Current-runtime 3 execution modes * 4 batch sizes * 3 repeats matrix is present at 3000 eval queries/repeat. compact_current preserves recall, uses 1.00 request object/query, improves QPS by +66-170% and reduces P95 by 41-68% versus grouped_by_ef_materialized. Versus client_shard_major_expanded it improves QPS by +159-185% and reduces P95 by 61-67% for batch sizes 50/100/200; batch_size=400 client-expanded fails with BrokenPipe from large REST payload. Serialized request-body accounting shows compact_current reduces request objects by 95.76% and request body bytes/query by 93.23% versus client-expanded. A selected batch=200 wire-body supplement shows compact_current reduces response body bytes/query by 95.76% and total JSON body bytes/query by 93.71% versus client-expanded. A 3000-query client-side planning-time supplement records upper-label + route-plan materialization time: compact_current 0.250 ms/query, grouped_by_ef_materialized 0.167 ms/query, and client_shard_major_expanded 0.292 ms/query. A selected batch=200 Docker stats supplement records container-level overhead: compact_current reduces controller RX bytes/query by 85.57% versus grouped materialized and 93.00% versus client-expanded, and reduces sampled controller CPU average by 74.80% and 58.08%, respectively. A companion 2026-07-05 selected batch=200 Docker stats memory supplement records Docker MemUsage snapshots: compact_current controller memory averages 1153.0 MiB versus 1335.0 MiB for grouped_by_ef_materialized and 1287.5 MiB for client_shard_major_expanded; cluster memory is essentially flat around 39.3-39.5 GiB. A companion 2026-07-05 selected batch=200 Docker-top process RSS supplement records compact_current controller RSS averaging 1105.3 MiB versus 1275.6 MiB for grouped_by_ef_materialized and 1208.1 MiB for client_shard_major_expanded; cluster RSS is essentially flat around 36.0-36.1 GiB. A 2026-07-05 build-stage artifact audit scanned existing builds.csv schemas: 322 Method4/Qdrant build snapshots have no build-duration/RSS columns; 18 old HNSW build-timing files have total_build_secs but are not Method4 distributed evidence. A 2026-07-05 selected batch=200 host-interface byte-counter supplement records Linux /sys/class/net deltas during the same three execution modes: compact_current lowers docker_bridge bytes/query by 84.31% versus grouped_by_ef_materialized and 93.59% versus client_shard_major_expanded; docker_veth bytes/query drops by 64.20% and 75.43%, respectively. physical_nic host external counters are small, about 336 bytes/query for compact_current versus 363 grouped and 541 client-expanded, confirming that the single-host Docker experiment mostly exercises bridge/veth/loopback interfaces rather than external NIC traffic. A selected Qdrant /metrics supplement at batch=200 records server-exposed REST/process counters for the same three execution modes: compact_current has +92.34% QPS and -52.15% P95 versus grouped_by_ef_materialized, +179.21% QPS and -66.33% P95 versus client_shard_major_expanded, while Qdrant REST batch duration deltas are -32.49% and -26.18%, respectively; qdrant_search_batch_count_delta is 15 for every repeat. A selected 2026-07-06 Docker bridge packet-capture supplement at batch=200 with 3000 queries * 3 repeats records compact_current frame bytes/query -60.20% versus grouped_by_ef_materialized and -75.27% versus client_shard_major_expanded, TCP payload bytes/query -56.68% and -74.83%, and packet count/query -96.69% and -92.69%; tcpdump reported 0 kernel-dropped packets for all three captures. A 2026-07-07 physical-NIC negative-control packet capture ran the same three Claim E execution modes at batch=200 for 3000 queries * 3 repeats each while tcpdump captured both UP external NICs, ens6f0np0 and ens111f0np0, with filter `tcp and net 172.24.0.0/16`; both pcaps recorded 0 matching packets, 0 frame bytes, 0 TCP payload bytes, and 0 kernel drops. The concurrent execution summary completed all three modes: compact_current QPS mean 434.93, grouped_by_ef_materialized 218.15, and client_shard_major_expanded 152.42. This resolves the previous physical-NIC capture permission blocker as a single-host Docker negative control. A 2026-07-07 smoke-level fresh-build instrumentation run now wraps tools/qdrant_two_level_routing_experiment.py with /usr/bin/time -v: train_limit=1000, 20-query tuning/eval, 1000 logical points, 1381 assigned points, 1376 indexed vectors, 7 active custom shards across 4 peers, elapsed wall time 37.82 s, max RSS 65,744 KB, exit status 0. This proves the time/RSS instrumentation path for fresh builds, but it is not full-size Method4 distributed build cost and not pure build-only RSS attribution. A 2026-07-07 selected Claim E controller/worker process-level attribution supplement samples host /proc counters for the Qdrant controller and three shard worker PIDs during batch=200 execution-mode runs. compact_current controller Qdrant process CPU delta is 13.79 s versus 54.12 s grouped (-74.52%) and 44.85 s client-expanded (-69.25%); controller CPU ms per measured query-window is 1.532 versus 6.013 and 4.983. Controller RSS mean is 297.46 MiB versus 406.57 MiB grouped (-26.84%) and 386.28 MiB client-expanded (-22.99%). Cluster process CPU delta is 206.69 s versus 395.02 s grouped (-47.68%) and 261.58 s client-expanded (-20.98%). | Full-size Method4 distributed build-stage wall time/RSS remains missing; the smoke run is train_limit=1000 and wraps the whole fresh-build harness including tiny tuning/eval phases. Qdrant subsystem/function-level CPU/memory attribution remains missing; selected controller/worker process-level PID attribution is now present. The byte supplements measure client-side serialized JSON request bodies and raw JSON response bodies, not HTTP framing or compression. The Docker supplements measure container-level cumulative NetIO counters, sampled CPUPerc, Docker MemUsage, and Docker-top process RSS for the running Docker containers. The host-interface supplement measures Linux interface counters, where physical_nic rows are host external NIC counters and docker_bridge/docker_veth/loopback rows are single-host Docker interfaces. The packet-capture supplement measures TCP packets on the single-host Docker bridge, not physical-NIC payload attribution or Qdrant subsystem/function-level attribution. The 2026-07-07 physical-NIC capture is a negative control for the single-host Docker subnet; it should not be used as physical-network payload savings. | docs/experiments/2026-06-01-server-shard-major-routing.md; docs/experiments/2026-06-02-method4-fastpath-negative-controls.md; results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_compact_request_and_shard_major.csv; results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_execution_mode_latency_matrix.csv; results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_execution_mode_latency_deltas.csv; results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_payload_bytes_summary.csv; results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_wire_bytes_summary.csv; results/method4_claim_coverage_20260704/derived_claim_tables/crosscut_request_candidate_pressure_proxy.csv; results/method4_claim_e_execution_mode_latency_20260704/analysis_20260704_225847/; results/method4_claim_e_payload_bytes_20260705/analysis_20260705_014703/; results/method4_claim_e_wire_bytes_20260705/analysis_20260705_020000/; results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_planning_time_summary.csv; results/method4_claim_e_planning_time_20260705/analysis_20260705_021347/; results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_container_overhead_summary.csv; results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_container_overhead_deltas.csv; results/method4_claim_e_container_overhead_20260705/analysis_20260705_024218/; results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_container_memory_summary.csv; results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_container_memory_deltas.csv; results/method4_claim_e_container_overhead_memory_20260705/analysis_20260705_040749/; results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_process_rss_summary.csv; results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_process_rss_deltas.csv; results/method4_claim_e_process_rss_20260705/analysis_20260705_042151/; results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_build_stage_artifact_audit.csv; results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_cost_row_build_source_audit.csv; results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_host_interface_bytes_role_summary.csv; results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_host_interface_bytes_deltas.csv; results/method4_claim_e_host_interface_bytes_20260705/analysis_20260705_061525/; tools/method4_claim_e_host_interface_bytes.py; results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_qdrant_metrics_summary.csv; results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_qdrant_metrics_deltas.csv; results/method4_claim_e_qdrant_metrics_20260705/analysis_20260705_073436/; tools/method4_claim_e_qdrant_metrics.py; results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_packet_capture_bridge_totals.csv; results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_packet_capture_bridge_deltas.csv; results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_packet_capture_bridge_summary.csv; results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_packet_capture_manifest.json; results/method4_claim_e_packet_capture_20260706/per_variant_repeats3_20260706_122755/; tools/method4_claim_e_packet_capture_summary.py; results/method4_claim_coverage_20260704/derived_claim_tables/current_environment_preflight_20260706.csv; results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_packet_capture_physical_nic_totals.csv; results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_packet_capture_physical_nic_summary.csv; results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_packet_capture_physical_nic_execution_summary.csv; results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_packet_capture_physical_nic_manifest.json; results/method4_claim_e_packet_capture_physical_nic_20260707/all_variants_repeats3_preload_20260707_082833; results/method4_claim_coverage_20260704/derived_claim_tables/current_environment_preflight_20260707.csv; results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_build_stage_instrumented_smoke_20260707.csv; results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_build_stage_instrumented_smoke_manifest.json; results/method4_claim_e_build_stage_smoke_20260707/claim_e_build_smoke_20260707_085303/20260707_085328; results/method4_claim_e_build_stage_smoke_20260707/raw_claim_e_build_smoke_20260707_085303/stdout.txt; results/method4_claim_e_build_stage_smoke_20260707/raw_claim_e_build_smoke_20260707_085303/time_v_stderr.txt; results/method4_claim_e_build_stage_smoke_20260707/claim_e_build_smoke_20260707_085303/20260707_085328/summary.json; results/method4_claim_e_build_stage_smoke_20260707/claim_e_build_smoke_20260707_085303/20260707_085328/builds.csv; results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_controller_process_attribution_summary.csv; results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_controller_process_attribution_deltas.csv; results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_controller_process_attribution_manifest.json; results/method4_claim_e_controller_process_attribution_20260707/batch200_all_variants_20260707_091900; results/method4_claim_e_controller_process_attribution_20260707/batch200_all_variants_20260707_091900/run_manifest.json; results/method4_claim_e_controller_process_attribution_20260707/batch200_all_variants_20260707_091900/raw_run_rows.csv; results/method4_claim_e_controller_process_attribution_20260707/batch200_all_variants_20260707_091900/process_pid_map.json |
| F | complete_with_runtime_caveat | Physical trace shows fan-in can drop 23.208 logical streams/query to 2.977 physical peers/query; direct simulation reduces candidate streams by ~87.17%; same-image server A/B gives +6.84% QPS with recall delta +0.000067. New 3 variants * 4 batch sizes * 3 repeats matrix covers coordinator current, direct-peer no-premerge, and direct-peer local-premerge at 3000 eval queries/repeat; local premerge preserves recall and reduces candidate groups/returned candidates from 23.63/236.30 per query to 2.98/29.78 per query (-87.40%) across batch sizes 50/100/200/400. The cross-cut proxy table records the same count-level candidate pressure reduction. A 10000-query enabled peer pre-merge sanity run records Recall@10 0.95403, QPS 429.98, and compact server execution shape with search requests/query 1.0 and candidate groups/query 1.0. | The new matrix is a direct-peer simulation plus current coordinator path; it does not restart the server with QDRANT_DISABLE_SHARD_MAJOR_PEER_PREMERGE for batch-size P95/P99 A/B. Direct-peer simulation still serializes all logical shard results before local Python pre-merge, so use it for fan-in/latency-shape evidence, not production QPS proof. Candidate-pressure counts are not serialized/network byte metrics. | docs/experiments/2026-06-03-method4-physical-execution-trace.md; docs/experiments/2026-06-03-method4-direct-peer-premerge-simulation.md; docs/experiments/2026-06-03-method4-worker-local-peer-premerge.md; results/method4_claim_coverage_20260704/derived_claim_tables/claim_f_worker_local_premerge.csv; results/method4_claim_coverage_20260704/derived_claim_tables/claim_f_premerge_batch_latency_matrix.csv; results/method4_claim_coverage_20260704/derived_claim_tables/claim_f_premerge_batch_latency_deltas.csv; results/method4_claim_coverage_20260704/derived_claim_tables/crosscut_request_candidate_pressure_proxy.csv; results/method4_claim_f_premerge_batch_latency_20260704/analysis_20260704_222700/; tools/method4_claim_f_premerge_batch_latency.py; results/qdrant_goal_recall_idea_095_direct_peer_premerge_ab_baseline/20260603_080131/summary.json; results/qdrant_goal_recall_idea_095_direct_peer_premerge_ab_premerged/20260603_080247/summary.json; results/qdrant_goal_recall_idea_095_server_peer_premerge_fresh/20260603_092757/summary.json; results/qdrant_goal_recall_idea_095_server_peer_premerge_disabled_fresh/20260603_092949/summary.json; results/qdrant_goal_recall_idea_095_server_peer_premerge/20260603_090530/summary.json; results/qdrant_goal_recall_idea_095_server_peer_premerge_10k/20260603_090738/summary.json; results/qdrant_goal_recall_idea_095_server_peer_premerge_10k/20260603_090738/final_metrics.csv |
| G | complete | Offline 9-point placement matrix and matched online 18-run batch latency matrix completed; method4-aware lowers P95 max worker EF by 4.9-8.0%, improves QPS by 1.2-2.1%, and reduces P95/P99 batch latency by 0.4-1.0%. A concurrency=8 deployed-collection supplement also preserves recall and shows +2.30% QPS plus -5.27%/-7.91% P95/P99 versus round-robin. Raw provenance audit now indexes the offline/deploy/batch/concurrency source directories. | Online gains are modest; avoid claiming large wins for every workload. The concurrency=8 supplement is not the primary physical-only A/B because the deployed Method4-aware collection is not a byte-identical logical clone and visits slightly more shards/EF-sum. | docs/experiments/2026-07-04-method4-claim-g-physical-layout-evidence.md; results/method4_claim_g_evidence_20260704/; results/method4_claim_g_evidence_20260704/online_concurrency8_latency_summary.csv; results/method4_claim_g_evidence_20260704/online_concurrency8_latency_deltas.csv; results/method4_claim_coverage_20260704/derived_claim_tables/claim_g_raw_provenance_audit_20260708.csv; results/method4_claim_g_evidence_20260704/source_manifest.json; results/method4_claim_g_physical_layout_20260704/analysis_20260704_164039/; results/method4_claim_g_matched_layout_deploy_20260704/20260704_170511/; results/method4_claim_g_matched_layout_deploy_20260704/20260704_171202/; results/method4_claim_g_batch_latency_20260704/analysis_20260704_172015/ |
| H | complete | Standalone 100-query compact_multi_ep semantic invariant audit passes all 9 invariants: upper routing, point_to_shards, multi-assignment, per-shard entry points, dynamic EF, logical-shard lower search, no adaptive pruning, source-id dedup, and final global merge. | Optional C++ reference comparison was not run; current support is an external wrapper semantic audit, not an internal C++ step-by-step trace. | results/method4_claim_h_semantic_audit_20260704/audit_20260704_210845/semantic_invariant_summary.csv; results/method4_claim_coverage_20260704/derived_claim_tables/claim_h_semantic_invariants_current_evidence.csv |

Claim D addendum: a matched cold/no-warmup 0.99 boundary repeat now exists
for `m4_400_160_20` versus `naive_ef200` over the same measured HDF5 window
as the original warm row (`test[100:3100]`). It records Recall@10 0.98937
versus 0.99017, QPS +0.29%, visited shards -25.47%, EF-sum +56.79%, and
P95/P99 -7.69%/-9.00% for Method4 versus naive. This strengthens the 0.99
boundary/warmup-sensitivity caveat; it still does not close the 0.99
same-recall claim gap.

## Claim A: Topology-Aware Routing Locality

Current Claim A evidence supports two narrower statements, but not the original strongest full-fission wording.

First, online Orion-vs-Simple-KMeans evidence shows Orion reaches similar recall with fewer visited shards and higher QPS than the simple centroid nprobe baseline:

| Target | Recall O/S | QPS O/S | Visited O/S | EF-sum O/S | Source |
| --- | --- | --- | --- | --- | --- |
| 0.8 | 0.8040 / 0.8060 | 1009.9 / 870.1 | 6.37 / 12.00 | 275 / 576 | claim_a_online_orion_vs_simple_kmeans_same_avg_ef.csv |
| 0.85 | 0.8545 / 0.8520 | 887.8 / 617.8 | 8.30 / 16.00 | 526 / 1024 | claim_a_online_orion_vs_simple_kmeans_same_avg_ef.csv |
| 0.9 | 0.9063 / 0.9041 | 702.0 / 352.4 | 14.54 / 32.00 | 1105 / 2560 | claim_a_online_orion_vs_simple_kmeans_same_avg_ef.csv |
| 0.95 | 0.9542 / 0.9538 | 391.3 / 206.9 | 21.12 / 40.00 | 3267 / 6400 | claim_a_online_orion_vs_simple_kmeans_same_avg_ef.csv |

Second, the planned offline partition oracle matrix has now been run for six partition families at `upper_k` 80/120/160. The important rows are:

| upper_k | Partition | Avg routed | Oracle miss@10 | GT entropy | Edge cut |
| ---: | --- | ---: | ---: | ---: | ---: |
| 80 | random_balanced_46 | 37.81 | 6.34% | 2.004 | 0.881 |
| 80 | balanced_kmeans_only_46 | 16.30 | 1.46% | 0.942 | 0.514 |
| 80 | kmeans_topology_46 | 13.95 | 1.23% | 0.811 | 0.351 |
| 80 | full_method4_fission_31_to_46 | 17.03 | 1.60% | 0.934 | 0.437 |
| 80 | naive_all_shards_46 | 46.00 | 0.00% | 2.173 | N/A |
| 120 | random_balanced_46 | 42.34 | 2.50% | 2.004 | 0.881 |
| 120 | balanced_kmeans_only_46 | 20.05 | 0.84% | 0.942 | 0.514 |
| 120 | kmeans_topology_46 | 17.27 | 0.77% | 0.811 | 0.351 |
| 120 | full_method4_fission_31_to_46 | 21.05 | 0.95% | 0.934 | 0.437 |
| 120 | naive_all_shards_46 | 46.00 | 0.00% | 2.173 | N/A |
| 160 | random_balanced_46 | 44.31 | 0.95% | 2.004 | 0.881 |
| 160 | balanced_kmeans_only_46 | 22.93 | 0.54% | 0.942 | 0.514 |
| 160 | kmeans_topology_46 | 19.93 | 0.55% | 0.811 | 0.351 |
| 160 | full_method4_fission_31_to_46 | 24.10 | 0.63% | 0.934 | 0.437 |
| 160 | naive_all_shards_46 | 46.00 | 0.00% | 2.173 | N/A |

Source: `results/method4_claim_coverage_20260704/derived_claim_tables/claim_a_partition_oracle_matrix.csv`; raw run: `results/method4_claim_a_partition_oracle_20260704/analysis_20260704_213047/`.

Interpretation:

- Topology convergence improves locality over balanced KMeans-only: edge cut drops from 0.514 to 0.351, GT-shard entropy drops from 0.942 to 0.811, and avg routed shards are lower at all three `upper_k` values.
- Topology convergence also slightly reduces oracle miss at `upper_k=80` and `upper_k=120`, but is effectively tied/slightly worse at `upper_k=160`.
- `kmeans_topology_load_recalibrated_46` matches `kmeans_topology_46` in this artifact because the current implementation applies load recalibration through fission; without fission, the route map is identical.
- Current `full_method4_fission_31_to_46` does not beat balanced KMeans-only on oracle miss or routed shard count at these operating points.

Third, an online P95/P99 submatrix now combines the existing live
full-fission/Orion, balanced-KMeans/KMeans-only-ish, naive collections, the
selected topology/no-fission collection, and the 2026-07-09 current-harness
rebuilds for the three previously missing partition families. All formal rows
use 3000 eval queries and 3 repeats. The KMeans batch=200 historical naive
reference remains high-variance, so its median delta is more reliable than its
mean delta. The 2026-07-09 rows use build `upper_search_ef=400` to match the
Claim A oracle construction metadata, then online latency `upper_search_ef=160`;
they are not byte-identical replays of the 2026-07-04 oracle artifact.

| Pair | Batch | Recall method/baseline | QPS method | Visited method | P95 method | P99 method | QPS vs baseline | P95 vs baseline | P99 vs baseline |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| full_fission_existing vs naive | 100 | 0.9529 / 0.9529 | 417.1 | 22.68 | 253.1 ms | 259.1 ms | +53.5% | -39.7% | -40.6% |
| full_fission_existing vs naive | 200 | 0.9532 / 0.9529 | 442.9 | 22.67 | 467.0 ms | 471.4 ms | +87.4% | -57.2% | -59.7% |
| balanced_kmeans_existing vs naive | 100 | 0.9523 / 0.9529 | 385.4 | 21.49 | 277.5 ms | 282.7 ms | +42.5% | -32.4% | -33.5% |
| balanced_kmeans_existing vs naive | 200 | 0.9521 / 0.9529 | 440.6 | 21.47 | 472.1 ms | 476.1 ms | +90.5% median | -56.7% median | -57.9% median |
| random_balanced_46 current rebuild vs naive | 200 | 0.9587 / 0.9529 | 320.7 | 44.16 | 645.6 ms | 646.6 ms | +39.3% | -30.8% | -32.5% |
| kmeans_topology_46 current rebuild vs naive | 200 | 0.9504 / 0.9529 | 420.2 | 18.63 | 520.7 ms | 532.0 ms | +97.0% | -55.0% | -57.9% |
| kmeans_topology_load_recalibrated_46 current rebuild vs naive | 200 | 0.9502 / 0.9529 | 444.0 | 18.62 | 470.2 ms | 476.5 ms | +87.5% | -47.4% | -48.0% |
| topology_no_fission vs full_fission_existing | 200 | 0.9522 / 0.9532 | 430.6 | 18.94 | 479.6 ms | 483.3 ms | -2.8% | +2.7% | +2.5% |

Source:
`results/method4_claim_coverage_20260704/derived_claim_tables/claim_a_partition_online_latency_submatrix.csv`;
delta table:
`results/method4_claim_coverage_20260704/derived_claim_tables/claim_a_partition_online_latency_submatrix_deltas.csv`;
raw runs under:
`results/method4_claim_a_partition_online_latency_20260704/` and
`results/method4_claim_a_partition_online_latency_20260709/`;
2026-07-09 build provenance:
`results/method4_claim_coverage_20260704/derived_claim_tables/claim_a_partition_family_current_rebuilds_20260709.csv`.

Partition-family online coverage audit:

| Family | Offline oracle | Online latency | Live collection | Matrix status |
| --- | --- | --- | --- | --- |
| `random_balanced_46` | available | 2026-07-09 current-harness row available | deleted after latency | available with rebuild caveat |
| `balanced_kmeans_only_46` | available | existing row with caveat | `bench095_rr_kmeans_s46` | partial existing collection only |
| `kmeans_topology_46` | available | 2026-07-09 current-harness row available | deleted after latency | available with rebuild caveat |
| `kmeans_topology_load_recalibrated_46` | available but identical to topology in current artifact | 2026-07-09 current-harness row available | deleted after latency | available with route-map caveat |
| `full_method4_fission_31_to_46` | available | existing row | `bench095_rr_orion_s31` | partial existing collection only |
| `naive_all_shards_46` | available | reference row | `bench095_rr_naive_s46` | reference available |
| `topology_no_fission_46` | supplemental | selected cell | `bench095_rr_topology_no_fission_s46_20260705` | supplemental selected cell |

Source:
`results/method4_claim_coverage_20260704/derived_claim_tables/claim_a_partition_family_online_coverage_audit.csv`.

Topology/no-fission selected-config sensitivity:

| Config | Role | Recall | QPS | Visited | EF-sum | P95 | P99 | Delta vs selected |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `u160,b80,f10` | selected main cell | 0.9522 | 430.6 | 18.94 | 3227.8 | 479.6 ms | 483.3 ms | reference |
| `u160,b120,f8` | slower sensitivity candidate | 0.9516 | 414.9 | 18.94 | 3643.0 | 499.8 ms | 505.1 ms | QPS -3.63%; EF-sum +12.86%; P95/P99 +4.21%/+4.53% |

Source:
`results/method4_claim_coverage_20260704/derived_claim_tables/claim_a_topology_no_fission_config_sensitivity.csv`.

Supported wording:

```text
At matched Recall@10 and similar average EF per visited shard, Orion visits 47-55% fewer lower shards than Simple KMeans nprobe and achieves 1.16-1.99x QPS on the tested 0.80-0.95 recall targets. Offline oracle analysis also shows that topology convergence improves edge cut, GT-shard entropy, routed shard count, and upper_k 80/120 oracle miss relative to balanced KMeans-only. The 2026-07-09 current-harness online rows fill the random/topology/load-recalibrated latency cells and show topology-family rebuilds visiting about 18.6 shards versus 46 for paired naive with lower P95/P99.
```

Avoid claiming:

```text
The current full Method4/fission partition has proven better oracle locality than balanced KMeans-only.
```

Topology/no-fission tradeoff: the selected topology-converged no-fission cell
visits 16.5% fewer shards than the selected full-fission reference at similar
recall, but has 2.8% lower QPS and 2.7%/2.5% worse P95/P99 batch latency. Use
it as a completed selected ablation cell and not as evidence that no-fission
dominates full fission online. The same-run sensitivity table shows the
alternative `u160,b120,f8` no-fission candidate is worse than the selected
`u160,b80,f10` point on QPS, EF-sum, and tail latency, so it should remain a
transparency row rather than a competing main result.

Current Claim A gap status: the previously missing random/topology/load-
recalibrated online latency rows are now present for the current GloVe scope.
The remaining caveats are about wording, not missing current-scope raw rows:
the 2026-07-09 rows are current-harness rebuilds rather than byte-identical
2026-07-04 oracle replays, the load-recalibrated no-fission route map matches
the topology route map while weights are recalibrated, and the current fission
operating point should still be retuned before making a strong full-Method4
partition superiority claim over balanced KMeans-only.

## Claim B: Multi-Assignment

Claim B now has online frontier evidence, direct offline oracle evidence, and a selected full-size latency supplement. Under <=2.0x index expansion, Orion multi-assignment improves the recall-QPS frontier. A 3000-query offline oracle analysis shows that Orion default and w2c2 reduce `oracle_gt_miss@10` versus single assignment at `upper_k in {80,120,160}`. The 2026-07-09 strict supplement adds 3000-query * 3-repeat client-observed batch latency for selected single/default/w2c2 cells in the 0.95 target neighborhood.

Oracle mechanism evidence:

| Strategy | Expansion | upper_k | oracle_gt_coverage@10 | oracle_gt_miss@10 | Query all-GT covered | Avg routed shards |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| orion_single | 1.000 | 80 | 96.11% | 3.89% | 73.67% | 15.35 |
| orion_default | 1.184 | 80 | 98.37% | 1.63% | 87.30% | 17.19 |
| orion_w2c2 | 1.495 | 80 | 99.39% | 0.61% | 94.77% | 21.74 |
| orion_single | 1.000 | 120 | 97.59% | 2.41% | 81.33% | 18.85 |
| orion_default | 1.184 | 120 | 99.04% | 0.96% | 91.93% | 21.23 |
| orion_w2c2 | 1.495 | 120 | 99.67% | 0.33% | 97.03% | 26.70 |
| orion_single | 1.000 | 160 | 98.33% | 1.67% | 86.20% | 21.58 |
| orion_default | 1.184 | 160 | 99.40% | 0.60% | 94.73% | 24.29 |
| orion_w2c2 | 1.495 | 160 | 99.79% | 0.21% | 98.07% | 30.38 |

Source: results/method4_claim_coverage_20260704/derived_claim_tables/claim_b_orion_oracle_gt_miss.csv.

Selected full-size strict multi-assignment latency supplement:

| Strategy | Recall@10 | QPS | Visited shards | EF-sum/query | Expansion | P95 ms | P99 ms | Caveat |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| single | 0.97317 | 279.23 | 30.18 | 8421.04 | 1.000 | 736.45 | 739.58 | single-assignment control reaches recall well above 0.95; use as selected no-multiassign control rather than strict same-recall pair |
| default | 0.95260 | 475.27 | 21.35 | 3088.52 | 1.186 | 432.73 | 435.41 | current-harness rebuild; close to 0.95 selected default multiassign latency cell |
| w2c2 | 0.94913 | 557.80 | 21.24 | 2127.40 | 1.499 | 375.01 | 376.76 | current-harness rebuild lands slightly below 0.95 recall on the 3000-query latency window; use as target-neighborhood w2c2 selected cell |
| Family | Strategy | Target | Recall | QPS | Visited | Expansion | Params |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Orion | orion_single | 0.80 | 0.8012 | 499.8 | 15.32 | 1.000 | upper_k=80, base=120, factor=16 |
| Orion | orion_single | 0.85 | 0.8468 | 372.7 | 25.50 | 1.000 | upper_k=240, base=48, factor=12 |
| Orion | orion_single | 0.90 | 0.9496 | 322.5 | 28.12 | 1.000 | upper_k=320, base=48, factor=8 |
| Orion | orion_single | 0.95 | 0.9711 | 219.1 | 30.15 | 1.000 | upper_k=400, base=120, factor=12 |
| Orion | orion_default | 0.80 | 0.7963 | 1127.4 | 6.34 | 1.185 | upper_k=16, base=16, factor=8 |
| Orion | orion_default | 0.85 | 0.8459 | 943.1 | 7.33 | 1.185 | upper_k=20, base=48, factor=8 |
| Orion | orion_default | 0.90 | 0.8963 | 755.7 | 11.40 | 1.185 | upper_k=40, base=60, factor=8 |
| Orion | orion_default | 0.95 | 0.9479 | 409.1 | 21.14 | 1.185 | upper_k=120, base=70, factor=10 |
| Orion | orion_w2c2 | 0.80 | 0.7988 | 1069.8 | 8.11 | 1.499 | upper_k=16, base=16, factor=4 |
| Orion | orion_w2c2 | 0.85 | 0.8493 | 915.1 | 8.12 | 1.499 | upper_k=16, base=32, factor=8 |
| Orion | orion_w2c2 | 0.90 | 0.9124 | 632.6 | 14.39 | 1.499 | upper_k=40, base=40, factor=8 |
| Orion | orion_w2c2 | 0.95 | 0.9501 | 458.0 | 21.30 | 1.499 | upper_k=80, base=50, factor=10 |
| Orion | orion_w2c3 | 0.80 | 0.8065 | 1001.9 | 7.86 | 1.852 | upper_k=12, base=8, factor=8 |
| Orion | orion_w2c3 | 0.85 | 0.8510 | 865.4 | 9.43 | 1.852 | upper_k=16, base=32, factor=4 |
| Orion | orion_w2c3 | 0.90 | 0.9282 | 591.2 | 16.66 | 1.852 | upper_k=40, base=40, factor=8 |
| Orion | orion_w2c3 | 0.95 | 0.9618 | 385.2 | 24.48 | 1.852 | upper_k=80, base=50, factor=10 |
| Simple KMeans | simple_a1.000 | 0.90 | 0.8971 | 478.6 | 20.00 | 1.000 | nprobe=20, ef=96 |
| Simple KMeans | simple_a1.000 | 0.95 | 0.9532 | 223.1 | 24.00 | 1.000 | nprobe=24, ef=240 |
| Simple KMeans | simple_a1.004 | 0.90 | 0.8943 | 429.2 | 20.00 | 1.191 | nprobe=20, ef=80 |
| Simple KMeans | simple_a1.004 | 0.95 | 0.9525 | 219.5 | 24.00 | 1.191 | nprobe=24, ef=200 |
| Simple KMeans | simple_a1.010 | 0.90 | 0.9047 | 409.9 | 20.00 | 1.566 | nprobe=20, ef=80 |
| Simple KMeans | simple_a1.010 | 0.95 | 0.9577 | 199.0 | 24.00 | 1.566 | nprobe=24, ef=200 |
| Simple KMeans | simple_a1.014 | 0.90 | 0.9104 | 400.8 | 20.00 | 1.887 | nprobe=20, ef=80 |
| Simple KMeans | simple_a1.014 | 0.95 | 0.9533 | 220.0 | 24.00 | 1.887 | nprobe=24, ef=160 |

Supported wording:

```text
Moderate Orion multi-assignment raises the online recall-QPS frontier under <=2.0x expansion. Offline oracle analysis confirms the routing-miss mechanism: at upper_k 80/120/160, Orion default and w2c2 reduce oracle_gt_miss@10 versus single assignment, with w2c2 reaching 0.61% / 0.33% / 0.21% miss. The selected full-size latency supplement gives default and w2c2 higher QPS and lower client-observed P95/P99 than the single-assignment control in the 0.95 target neighborhood, but the rows should be described as current-harness target-neighborhood evidence rather than strict same-recall proof because single is above 0.95 recall and w2c2 is slightly below 0.95 on the latency window. Simple KMeans alpha multi-assignment does not show a broad online frontier lift.
```

Raw provenance is now explicit in `results/method4_claim_coverage_20260704/derived_claim_tables/claim_b_raw_provenance_audit_20260708.csv`. It indexes the two Claim B narrative docs, the promoted derived support tables, the main online frontier rollups including `final_eval_keypoints_long.csv`, the strict quality/provenance rollup including `final_eval_keypoints_long_strict.csv`, seven raw matrix-summary families with run manifests, 32 main source `summary.json` rows, 32 strict source `summary.json` rows, and the offline oracle-miss analysis at `results/method4_claim_b_oracle_miss_20260704/analysis_20260704_211749/`. The selected strict latency supplement is indexed separately in `results/method4_claim_coverage_20260704/derived_claim_tables/strict_multiassign_latency_20260709.csv`, with raw rebuilds under `results/method4_benchmark_matrix/method4_strict_multiassign_latency_rebuild_20260709/` and raw latency runs under `results/method4_strict_ablation_multiassign_latency_20260709/`.

## Claim C: Dynamic EF

Claim C has strong support for budget allocation, same-recall QPS, and client-
observed batch P95/P99 latency. Dynamic EF uses routed EP count as a shard
relevance proxy. Offline analysis confirms GT-hit shards receive far more
routed EPs and a larger budget share than uniform fixed EF would provide.

Same-recall Dynamic-vs-fixed deltas:

| Comparison | Recall delta | QPS delta | Visited delta | EF-sum delta |
| --- | --- | --- | --- | --- |
| fixed_upper_k_120_isolated_095 | -0.000833 | +27.20% | +0.13% | -38.37% |
| frontier_robust_095 | -0.000467 | +28.90% | +0.01% | -34.92% |
| frontier_robust_097 | +0.001700 | +30.92% | +26.14% | -40.80% |

Routed EP relevance:

| Config | GT-hit EP | Non-hit EP | Dynamic GT budget | Fixed GT budget |
| --- | --- | --- | --- | --- |
| fixed_width_claim_c_095 | 20.34 | 3.22 | 34.93% | 17.53% |
| frontier_095_dynamic | 14.72 | 2.55 | 43.24% | 21.05% |
| frontier_097_dynamic | 30.78 | 4.54 | 42.64% | 14.22% |

Batch-latency supplement:

| Method | Recall | QPS | Visited | EF-sum | P95 batch ms | P99 batch ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Fixed EF `upper_k=80; fixed_ef=280` | 0.949300 | 408.7 | 15.99 | 4477.1 | 514.4 | 519.5 |
| Dynamic EF `upper_k=80; base_ef=80; factor=20` | 0.949467 | 518.0 | 15.99 | 3021.0 | 406.3 | 410.5 |

Dynamic EF relative to fixed EF in this latency rerun: recall +0.000167, QPS
+26.75%, visited shards +0.00%, EF-sum -32.52%, P95 batch latency -21.00%,
and P99 batch latency -20.97%.

Reduced-L2 Dynamic-vs-Fixed EF supplement:

| Dataset | Orion variant | Dynamic config | Nearest fixed config | Recall delta | QPS delta mean / median | EF-sum delta | P95 delta mean / median | P99 delta mean / median |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| SIFT-100k Euclidean reduced | no-fission | `u8,b24,f2` | `u8,ef36` | -0.000400 | +15.56% / +0.92% | -10.18% | +3.73% / -1.86% | +6.84% / -4.00% |
| SIFT-100k Euclidean reduced | full-fission | `u8,b48,f4` | `u8,ef72` | -0.000800 | +17.08% / +16.50% | -16.87% | -0.08% / -11.30% | +3.81% / -10.89% |

Source:
`results/method4_claim_coverage_20260704/derived_claim_tables/l2_reduced_sift100k_dynamic_vs_fixed_ef_matrix.csv`;
deltas:
`results/method4_claim_coverage_20260704/derived_claim_tables/l2_reduced_sift100k_dynamic_vs_fixed_ef_deltas.csv`;
raw runs:
`results/method4_l2_reduced_sift100k_latency_20260705/dynamic_vs_fixed_nofission/analysis_20260705_051254/`;
`results/method4_l2_reduced_sift100k_latency_20260705/dynamic_vs_fixed_nofission_fine/analysis_20260705_051329/`;
`results/method4_l2_reduced_sift100k_latency_20260705/full_orion_dynamic_vs_fixed/analysis_20260705_051940/`.

This reduced-L2 supplement is smaller-scale and noisier than the main GloVe
Claim C evidence. Use it as external-validity support: at nearly matched
Recall@10, Dynamic EF uses lower EF-sum and has positive QPS/median-tail
deltas in both no-fission and full-fission Orion variants, but mean tail
latency is noisy.

Supported wording:

```text
Dynamic EF improves same-recall QPS by 27-31% and reduces estimated lower EF-sum/query by 35-41% compared with fixed EF. GT-hit shards have far higher routed EP counts than non-hit shards, so Dynamic EF allocates more lower search budget to more promising shards. A 3-repeat client-observed batch-latency supplement also shows Dynamic EF reducing P95/P99 batch latency by about 21% at the selected 0.95 operating point.
```

Boundary: the latency supplement measures client-observed batch endpoint wall
time, not server-internal lower-search trace time. Do not claim internal lower-
search P99 without additional Qdrant/server instrumentation.

Raw provenance is now explicit in `results/method4_claim_coverage_20260704/derived_claim_tables/claim_c_raw_provenance_audit_20260708.csv`. It indexes the six same-recall Dynamic-vs-Fixed performance `summary.json` sources from the June 25 runs, the packaged Claim C evidence manifest at `results/method4_claim_c_evidence_20260704/source_manifest.json`, the routed-EP relevance raw analysis at `results/method4_claim_c_dynamic_ef_relevance_20260704/analysis_20260704_141217/`, and the client-observed batch-latency raw supplement at `results/method4_claim_c_dynamic_vs_fixed_latency_20260704/analysis_20260704_235925/`.

## Claim D: Method4 vs Naive Same-Recall QPS

Current evidence supports 0.80-0.95 recall behavior, a stable 0.955 same-recall comparison, and high-recall batch-latency matrices. The high-recall result is mixed: the ~0.97 strict and closest-neighborhood points support the claim strongly at batch=100, and the new 2026-07-08 strict batch=200 supplement preserves the ~0.97 QPS/tail advantage while exposing Naive batch=200 tail instability. The closest ~0.99 point remains a quality/performance tradeoff: the new batch=200 0.99 row still reaches only 0.98950 recall versus paired `naive_ef200_b200` at 0.99017. A targeted 2026-07-05 0.99 retune and two follow-up 0.99 strategy-search formal runs tried lower-EF/high-upper-k Method4 settings that had screened near or above 0.99 recall on 500 queries. The formal candidates improved QPS/tail versus same-run naive controls but did not reach 0.99 recall on 3000 queries. A 2026-07-07 broader upper-routing screen also produced 500-query Method4 candidates above 0.99 recall, but both selected formal 3000-query repeats, one chosen for high screen QPS and one chosen for top screen recall, again fell below 0.99 recall and below paired naive recall. A 2026-07-08 alternate wider-upper-routing screen selected `m4_u960_b20_f10` after a positive 500-query result, but its 3000-query formal repeat reached only 0.9876 recall versus paired `naive_ef200` at 0.99017. Two shuffled-query repeats (seeds `20260713` and `20260718`) and a matched cold/no-warmup repeat of the closest original 0.99 boundary row also stayed below 0.99 recall, so the recommended boundary remains through ~0.97.

| Label | Method | Recall | QPS | Visited | EF-sum | Source |
| --- | --- | --- | --- | --- | --- | --- |
| Method4 current 0.955 batch200 | Method4 | 0.9553 | 387.1 | 23.21 | 3274 | results/qdrant_compare_current_idea_vs_naive_095_idea_b200/20260603_094930/summary.json |
| Naive stable closest 0.955 batch100 ef76 | Naive | 0.9548 | 272.4 | 46.00 | 3496 | results/qdrant_compare_current_idea_vs_naive_095_naive_ef76_b100/20260603_095227/summary.json |
| Naive stable higher recall batch100 ef80 | Naive | 0.9573 | 264.6 | 46.00 | 3680 | results/qdrant_compare_current_idea_vs_naive_095_naive_ef80_b100/20260603_095342/summary.json |
| curve target 0.8 orion | Method4 | 0.8043 | 1090.3 | 5.96 | 268 | results/method4_benchmark_matrix/method4_recall_sweep_simple_kmeans_nprobe_20260623/simple_nprobe_20260623/orion_r080_eff46/20260623_121829/summary.json |
| curve target 0.85 orion | Method4 | 0.8619 | 961.3 | 6.84 | 654 | results/method4_benchmark_matrix/method4_recall_sweep_simple_kmeans_nprobe_20260623/simple_nprobe_20260623/orion_r085_eff46/20260623_122023/summary.json |
| curve target 0.9 orion | Method4 | 0.9035 | 715.7 | 13.10 | 1045 | results/method4_benchmark_matrix/method4_recall_sweep_simple_kmeans_nprobe_20260623/simple_nprobe_20260623/orion_r090_eff46/20260623_124147/summary.json |
| curve target 0.95 orion | Method4 | 0.9512 | 390.7 | 22.69 | 2885 | results/method4_benchmark_matrix/method4_recall_sweep_simple_kmeans_nprobe_20260623/simple_nprobe_20260623/orion_r095_eff46/20260623_124839/summary.json |
| curve target 0.8 naive | Naive | 0.8063 | 439.2 | 46.00 | 736 | results/method4_benchmark_matrix/method4_recall_sweep_simple_kmeans_nprobe_20260623/simple_nprobe_20260623/naive_r080_s46/20260623_123515/summary.json |
| curve target 0.85 naive | Naive | 0.8564 | 377.1 | 46.00 | 1104 | results/method4_benchmark_matrix/method4_recall_sweep_simple_kmeans_nprobe_20260623/simple_nprobe_20260623/naive_r085_s46/20260623_123554/summary.json |
| curve target 0.9 naive | Naive | 0.9068 | 329.4 | 46.00 | 1840 | results/method4_benchmark_matrix/method4_recall_sweep_simple_kmeans_nprobe_20260623/simple_nprobe_20260623/naive_r090_s46/20260623_124517/summary.json |
| curve target 0.95 naive | Naive | 0.9528 | 261.9 | 46.00 | 3496 | results/method4_benchmark_matrix/method4_recall_sweep_simple_kmeans_nprobe_20260623/simple_nprobe_20260623/naive_r095_s46/20260623_124612/summary.json |

New high-recall batch-latency pairs:

| Target | Method4 / Naive config | Recall M/N | QPS M/N | QPS delta | Visited delta | EF-sum delta | P95 delta | P99 delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ~0.97 strict >=0.97 | `m4_160_80_20` / `naive_ef112` | 0.9718 / 0.9728 | 329.0 / 234.1 | +40.58% | -46.39% | +7.19% | -31.85% | -32.42% |
| ~0.97 closest-neighborhood | `m4_160_80_16` / `naive_ef104` | 0.9684 / 0.9695 | 349.6 / 242.9 | +43.90% | -46.39% | +0.60% | -33.29% | -32.88% |
| ~0.99 neighborhood | `m4_400_160_20` / `naive_ef200` | 0.9893 / 0.9902 | 173.0 / 177.7 | -2.63% | -25.50% | +56.76% | +0.50% | +0.71% |
| ~0.99 conservative naive higher recall | `m4_400_160_20` / `naive_ef240` | 0.9893 / 0.9936 | 173.0 / 159.6 | +8.41% | -25.50% | +30.64% | -11.56% | -11.24% |

Source: `results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_high_recall_latency_pairs.csv`; raw runs: `results/method4_claim_d_high_recall_latency_20260704/analysis_20260704_215448/` and `results/method4_claim_d_high_recall_latency_refine_20260704/analysis_20260704_220036/`.

Batch=200 high-recall supplement:

| Target | Method4 / Naive config | Recall M/N | QPS M/N | QPS delta | Visited delta | EF-sum delta | P95 delta | P99 delta | Interpretation |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| ~0.97 strict batch=200 | `m4_160_80_20_b200` / `naive_ef112_b200` | 0.9718 / 0.9728 | 342.3 / 182.0 | +88.01% | -46.40% | +7.18% | -67.32% | -69.33% | supports through-~0.97 Claim D wording at the planned batch=200 shape; Naive has high tail variance |
| ~0.99 boundary batch=200 | `m4_400_160_20_b200` / `naive_ef200_b200` | 0.9895 / 0.9902 | 183.8 / 140.8 | +30.53% | -25.48% | +56.78% | -54.26% | -58.41% | Method4 remains below 0.99 and below paired naive recall, so this is boundary evidence only |

Source: `results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_batch200_high_recall_summary.csv`; deltas: `results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_batch200_high_recall_deltas.csv`; manifest: `results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_batch200_high_recall_manifest.json`; raw run: `results/method4_claim_d_batch200_targets_20260708/analysis_20260708_104611/`.

The batch=200 Naive controls are unstable enough to require the plan's strict-same-batch caveat: `naive_ef112_b200` has QPS CV 0.307 and P95 CV 0.765, while `naive_ef200_b200` has QPS CV 0.176 and P95 CV 0.456. The mean deltas above are therefore useful as strict batch=200 evidence, but the paper should also keep the more stable batch=100 context. The batch=200 0.99 row improves QPS/tail in this unstable window but still does not satisfy Recall@10 >= 0.99 for Method4.

Targeted 0.99 retune check:

| Method | Config | Recall | QPS | Visited | EF-sum | P95 batch ms | P99 batch ms | Interpretation |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Method4 | `m4_480_70_16` | 0.9867 | 186.8 | 36.19 | 11132.7 | 557.0 | 567.8 | faster/tighter tail than same-run naive_ef200, but below 0.99 recall |
| Method4 | `m4_520_60_16` | 0.9873 | 181.5 | 36.85 | 11532.2 | 574.2 | 579.7 | faster/tighter tail than same-run naive_ef200, but below 0.99 recall |
| Naive | `naive_ef200` | 0.9902 | 164.8 | 46.00 | 9200.0 | 683.7 | 731.4 | same-run 0.99 baseline |
| Naive | `naive_ef240` | 0.9936 | 150.4 | 46.00 | 11040.0 | 742.4 | 757.0 | higher-recall conservative naive baseline |

Source: `results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_099_retune_summary.csv`;
deltas versus same-run `naive_ef200`:
`results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_099_retune_deltas.csv`;
500-query screening table:
`results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_099_retune_tuning_candidates_500q.csv`;
raw formal run:
`results/method4_claim_d_099_retune_20260705/analysis_20260705_022625/`.

Follow-up 0.99 strategy-search formal checks:

| Run | Method | Config | 500q screening recall | 3000q formal recall | QPS | Visited | EF-sum | P95 batch ms | P99 batch ms | Interpretation |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 2026-07-05 | Method4 | `m4_560_40_16` | 0.9902 | 0.9875 | 148.6 | 37.60 | 11549.1 | 743.0 | 805.9 | screened above 0.99, but formal 3000q recall fell below 0.99 |
| 2026-07-05 | Method4 | `m4_560_60_14` | 0.9900 | 0.9867 | 152.9 | 37.60 | 11045.4 | 726.1 | 741.8 | screened at 0.99, but formal 3000q recall fell below 0.99 |
| 2026-07-05 | Naive | `naive_ef200` | n/a | 0.9902 | 140.0 | 46.00 | 9200.0 | 838.2 | 860.9 | same-run 0.99 baseline |
| 2026-07-05 | Naive | `naive_ef240` | n/a | 0.9936 | 135.0 | 46.00 | 11040.0 | 819.3 | 850.9 | higher-recall conservative naive baseline |
| 2026-07-06 | Method4 | `m4_480_80_16` | 0.9904 | 0.9871 | 183.1 | 36.37 | 11511.1 | 572.6 | 577.9 | highest 500q recall candidate still fell below 0.99 on the formal window |
| 2026-07-06 | Method4 | `m4_520_60_16` | 0.9906 | 0.9874 | 179.7 | 37.00 | 11543.3 | 582.2 | 585.3 | highest 500q recall candidate still fell below 0.99 on the formal window |
| 2026-07-06 | Naive | `naive_ef200` | n/a | 0.9902 | 159.6 | 46.00 | 9200.0 | 707.6 | 716.5 | same-run 0.99 baseline |
| 2026-07-06 | Naive | `naive_ef240` | n/a | 0.9936 | 150.2 | 46.00 | 11040.0 | 738.8 | 776.2 | higher-recall conservative naive baseline |
| 2026-07-07 broader upper-routing | Method4 | `m4_u720_b60_f12` | 0.9918 | 0.9880 | 168.4 | 39.63 | 12089.9 | 617.6 | 626.8 | selected from a broader upper-routing screen, but formal 3000q recall fell below 0.99 |
| 2026-07-07 broader upper-routing | Naive | `naive_ef200` | n/a | 0.9902 | 163.0 | 46.00 | 9200.0 | 689.9 | 711.4 | same-run 0.99 baseline |
| 2026-07-07 broader upper-routing | Naive | `naive_ef240` | n/a | 0.9936 | 151.5 | 46.00 | 11040.0 | 728.9 | 749.6 | higher-recall conservative naive baseline |
| 2026-07-07 broader upper-routing top-recall | Method4 | `m4_u720_b40_f14` | 0.9920 | 0.9887 | 165.2 | 39.64 | 12916.3 | 629.0 | 631.1 | highest 500q recall candidate still fell below 0.99 on the formal window |
| 2026-07-07 broader upper-routing top-recall | Naive | `naive_ef200` | n/a | 0.9902 | 166.3 | 46.00 | 9200.0 | 665.7 | 681.2 | same-run 0.99 baseline |
| 2026-07-07 broader upper-routing top-recall | Naive | `naive_ef240` | n/a | 0.9936 | 152.6 | 46.00 | 11040.0 | 727.2 | 743.9 | higher-recall conservative naive baseline |
| 2026-07-08 alternate wider-routing | Method4 | `m4_u960_b20_f10` | 0.9916 | 0.9876 | 168.4 | 41.56 | 11652.6 | 617.9 | 622.2 | positive 500q candidate still fell below 0.99 on the formal window |
| 2026-07-08 alternate wider-routing | Naive | `naive_ef200` | n/a | 0.9902 | 163.2 | 46.00 | 9200.0 | 687.8 | 706.1 | same-run 0.99 baseline |
| 2026-07-08 alternate wider-routing | Naive | `naive_ef240` | n/a | 0.9936 | 150.0 | 46.00 | 11040.0 | 743.1 | 769.0 | higher-recall conservative naive baseline |

Source:
`results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_099_strategy_search_summary.csv`;
deltas:
`results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_099_strategy_search_deltas.csv`;
raw formal run:
`results/method4_claim_d_099_strategy_search_20260705/analysis_20260705_074939/`;
focused 2026-07-06 top-screen tables:
`results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_099_strategy_search_top_screen_summary.csv`;
`results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_099_strategy_search_top_screen_deltas.csv`;
2026-07-06 raw formal run:
`results/method4_claim_d_099_strategy_search_20260706/top_screen_candidates/analysis_20260706_130013/`.
2026-07-07 broader upper-routing screen and formal follow-up:
`results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_099_broader_strategy_screen_summary.csv`;
`results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_099_broader_strategy_formal_summary.csv`;
`results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_099_broader_strategy_formal_deltas.csv`;
`results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_099_broader_strategy_manifest.json`;
raw screen:
`results/method4_claim_d_099_broader_screen_20260707/analysis_20260707_110802/`;
raw formal run:
`results/method4_claim_d_099_broader_formal_20260707/analysis_20260707_111008/`.
raw top-recall formal run:
`results/method4_claim_d_099_broader_top_recall_formal_20260707/analysis_20260707_112432/`.
2026-07-08 alternate wider-routing screen and formal follow-up:
`results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_099_alternate_width_screen_summary.csv`;
`results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_099_alternate_width_formal_summary.csv`;
`results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_099_alternate_width_formal_deltas.csv`;
`results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_099_alternate_width_manifest.json`;
raw screen:
`results/method4_claim_d_099_alternate_width_screen_20260708/analysis_20260708_095804/`;
raw formal run:
`results/method4_claim_d_099_alternate_width_formal_20260708/analysis_20260708_100010/`.

These follow-ups strengthen the 0.99 caveat. Four lower-EF/high-upper-k
500-query candidates that looked promising at Recall@10 0.9900-0.9906 did not
retain 0.99 recall in the formal 3000-query window, while same-run
`naive_ef200` controls remained at Recall@10 0.9902. The broader
upper-routing screen was initially more promising: five Method4 configurations
reached Recall@10 0.9918-0.9920 on 500 queries. But the high-QPS formal
candidate `m4_u720_b60_f12` reached only Recall@10 0.9880, and the top-screen-
recall formal candidate `m4_u720_b40_f14` reached only Recall@10 0.9887. Both
remain below `naive_ef200` at 0.9902 and `naive_ef240` at 0.9936. The 2026-07-08
alternate wider-routing screen repeated the same pattern: `m4_u960_b20_f10`
screened at Recall@10 0.9916 over 500 queries, but formal validation reached
only 0.9876 versus `naive_ef200` at 0.99017. The high-QPS 2026-07-07 candidate
still had +3.26% QPS and -10.48%/-11.89% P95/P99 versus `naive_ef200`; the
2026-07-07 top-recall candidate had -0.64% QPS and -5.50%/-7.35% P95/P99; the
2026-07-08 alternate candidate had +3.20% QPS and -10.16%/-11.89% P95/P99.
These are boundary/negative results, not 0.99 dominance. The additional
shuffled-query boundary repeats below confirm that the original closest 0.99
Method4 setting also remains below 0.99 recall under two independent shuffled
query orders.

0.99 boundary query-order check:

| Scenario | Config pair | Recall M/N | QPS delta | Visited delta | EF-sum delta | P95/P99 delta | Interpretation |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| original query order | `m4_400_160_20` / `naive_ef200` | 0.9893 / 0.9902 | -2.63% | -25.50% | +56.76% | +0.50% / +0.71% | closest 0.99 row is a tradeoff, not a Method4 win |
| shuffled seed `20260713` | `m4_400_160_20` / `naive_ef200` | 0.9892 / 0.9902 | +0.77% | -25.50% | +56.77% | -8.00% / -7.46% | query order changes the QPS/tail shape but Method4 remains below 0.99 and below naive recall |
| shuffled seed `20260718` | `m4_400_160_20` / `naive_ef200` | 0.9894 / 0.9902 | +0.52% | -25.55% | +56.73% | -7.96% / -11.52% | second shuffled-query seed confirms Method4 remains below 0.99 and below naive recall |
| cold/no-warmup matched window | `m4_400_160_20` / `naive_ef200` | 0.9894 / 0.9902 | +0.29% | -25.47% | +56.79% | -7.69% / -9.00% | warmup changes the QPS/tail shape but Method4 remains below 0.99 and below naive recall |

The shuffled-query repeats use 3000 eval queries and 3 repeats at batch size
100, with seeds `20260713` and `20260718`. The multiseed summary records
Method4 shuffled recall 0.98923-0.98943 versus naive 0.99017. The
cold/no-warmup repeat uses `warmup_query_count=0` and
`query_start_offset=100`, so its measured HDF5 query window matches the warm
original row (`test[100:3100]`). These rows are useful as boundary robustness
evidence: the original slight QPS/tail deficit is not stable under shuffled or
cold conditions, but the claim-relevant recall boundary is stable. Method4
remains below Recall@10 0.99 and below `naive_ef200` in all follow-ups: 0.98923
and 0.98943 for shuffled, and 0.98937 for cold, versus naive 0.99017. The cold-vs-warm
sensitivity is Method4 QPS -5.95% with P95/P99 +7.08%/+4.96%; the paired naive
row changes by QPS -8.69% with P95/P99 +16.59%/+16.17%. This does not support
0.99 same-recall dominance.

Source matrices:
`results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_099_boundary_query_order_overlay.csv`;
`results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_099_boundary_method4_vs_naive_deltas.csv`;
`results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_099_boundary_query_order_sensitivity.csv`;
`results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_099_boundary_query_order_multiseed_summary.csv`;
`results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_099_boundary_query_order_manifest.json`;
`results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_099_boundary_warmup_overlay.csv`;
`results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_099_boundary_warmup_method4_vs_naive_deltas.csv`;
`results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_099_boundary_warmup_sensitivity.csv`;
`results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_099_boundary_sensitivity_deltas.csv`;
raw runs:
`results/method4_099_boundary_shuffle_20260713/analysis_20260706_140200/`;
`results/method4_099_boundary_shuffle2_20260718/analysis_20260707_102446/`;
`results/method4_099_boundary_warmup_20260713/analysis_20260706_141159/`.

Selected shuffled-query robustness overlays:

| Scenario | Method / config | Recall | QPS | Visited | EF-sum | P95 batch ms | P99 batch ms | Source |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| original query order | Method4 `m4_160_80_20` | 0.9718 | 329.0 | 24.66 | 5522.5 | 318.2 | 325.0 | `results/method4_claim_d_high_recall_latency_20260704/analysis_20260704_215448/claim_d_high_recall_latency_summary.csv` |
| original query order | Naive `naive_ef112` | 0.9728 | 234.1 | 46.00 | 5152.0 | 466.9 | 480.8 | `results/method4_claim_d_high_recall_latency_refine_20260704/analysis_20260704_220036/claim_d_high_recall_latency_summary.csv` |
| shuffled seed `20260705` | Method4 `m4_160_80_20` | 0.9722 | 328.0 | 23.60 | 5432.6 | 319.4 | 325.3 | `results/method4_robustness_shuffle_20260705/analysis_20260705_030323/claim_d_high_recall_latency_summary.csv` |
| shuffled seed `20260705` | Naive `naive_ef112` | 0.9728 | 213.2 | 46.00 | 5152.0 | 534.6 | 546.7 | `results/method4_robustness_shuffle_20260705/analysis_20260705_030323/claim_d_high_recall_latency_summary.csv` |
| shuffled seed `20260706` | Method4 `m4_160_80_20` | 0.9720 | 313.7 | 23.60 | 5432.5 | 343.4 | 348.9 | `results/method4_robustness_shuffle_20260706/analysis_20260705_045135/claim_d_high_recall_latency_summary.csv` |
| shuffled seed `20260706` | Naive `naive_ef112` | 0.9728 | 208.6 | 46.00 | 5152.0 | 548.2 | 558.0 | `results/method4_robustness_shuffle_20260706/analysis_20260705_045135/claim_d_high_recall_latency_summary.csv` |

The shuffled overlay keeps the original 100-query warmup prefix and shuffles
only the 3000 measured queries. The current overlay covers seeds `20260705` and
`20260706`, each with 3000 eval queries and 3 repeats. Across the two shuffled
seeds, Method4 remains ahead of same-run naive by at least +50.37% QPS, visits
48.69-48.71% fewer shards, and lowers P95/P99 by at least 37.36%/37.47%.
Method4's own two-seed shuffled sensitivity relative to the original-order row
is moderate: mean QPS -2.49%, worst-seed QPS -4.66%, worst-seed P95 +7.90%,
and worst-seed P99 +7.37%. The selected naive row is more sensitive over the
same two seeds: mean QPS -9.89%, worst-seed QPS -10.87%, worst-seed P95
+17.40%, and worst-seed P99 +16.04%.

Source matrix:
`results/method4_claim_coverage_20260704/derived_claim_tables/robustness_shuffled_query_overlay.csv`;
deltas:
`results/method4_claim_coverage_20260704/derived_claim_tables/robustness_shuffled_query_overlay_deltas.csv`;
multi-seed summary:
`results/method4_claim_coverage_20260704/derived_claim_tables/robustness_query_order_multiseed_summary.csv`;
raw shuffled run:
`results/method4_robustness_shuffle_20260705/analysis_20260705_030323/`;
second raw shuffled run:
`results/method4_robustness_shuffle_20260706/analysis_20260705_045135/`.

Selected cold/warm robustness overlay:

| Scenario | Warmup / measured window | Method / config | Recall | QPS | Visited | EF-sum | P95 batch ms | P99 batch ms | Source |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| warm original | warmup=100, measured HDF5 `test[100:3100]` | Method4 `m4_160_80_20` | 0.9718 | 329.0 | 24.66 | 5522.5 | 318.2 | 325.0 | `results/method4_claim_d_high_recall_latency_20260704/analysis_20260704_215448/claim_d_high_recall_latency_summary.csv` |
| warm original | warmup=100, measured HDF5 `test[100:3100]` | Naive `naive_ef112` | 0.9728 | 234.1 | 46.00 | 5152.0 | 466.9 | 480.8 | `results/method4_claim_d_high_recall_latency_refine_20260704/analysis_20260704_220036/claim_d_high_recall_latency_summary.csv` |
| cold matched | warmup=0, `query_start_offset=100`, measured HDF5 `test[100:3100]` | Method4 `m4_160_80_20` | 0.9720 | 320.0 | 23.62 | 5434.9 | 329.7 | 333.9 | `results/method4_robustness_warmup_20260705/analysis_20260705_044122/claim_d_high_recall_latency_summary.csv` |
| cold matched | warmup=0, `query_start_offset=100`, measured HDF5 `test[100:3100]` | Naive `naive_ef112` | 0.9728 | 209.7 | 46.00 | 5152.0 | 543.2 | 554.9 | `results/method4_robustness_warmup_20260705/analysis_20260705_044122/claim_d_high_recall_latency_summary.csv` |

The cold overlay removes the warmup prefix while keeping the exact same measured
query window as the original warm run. The latency runner was extended with
`--query-start-offset`; the cold run uses `warmup_query_count=0` and
`query_start_offset=100`, so both warm and cold rows measure HDF5
`test[100:3100]`. Under this matched cold condition, Method4 still improves QPS
by 52.59%, visits 48.64% fewer shards, and lowers P95/P99 batch latency by
39.29%/39.83% versus same-run naive. Method4's own cold sensitivity is modest:
QPS changes by -2.73%, P95 by +3.62%, and P99 by +2.76% relative to the warm
row. The same cold removal hurts the selected naive row more: QPS -10.39%, P95
+16.33%, and P99 +15.41%.

Source matrix:
`results/method4_claim_coverage_20260704/derived_claim_tables/robustness_warmup_overlay.csv`;
deltas:
`results/method4_claim_coverage_20260704/derived_claim_tables/robustness_warmup_overlay_deltas.csv`;
raw cold run:
`results/method4_robustness_warmup_20260705/analysis_20260705_044122/`.

Additional ~0.80/~0.85 robustness overlay:

| Target | Scenario | Config pair | Recall Method4 / Naive | QPS delta | Visited delta | EF-sum delta | P95/P99 delta |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| ~0.80 | warm original | `m4_16_16_10` / `naive_ef16` | 0.8063 / 0.8075 | +258.35% | -86.15% | -62.36% | -74.04% / -75.22% |
| ~0.80 | shuffled seed `20260709` | `m4_16_16_10` / `naive_ef16` | 0.8063 / 0.8075 | +227.32% | -86.12% | -62.34% | -72.24% / -72.09% |
| ~0.80 | shuffled seed `20260712` | `m4_16_16_10` / `naive_ef16` | 0.8025 / 0.8075 | +213.41% | -87.08% | -63.36% | -70.45% / -70.72% |
| ~0.80 | cold matched | `m4_16_16_10` / `naive_ef16` | 0.8060 / 0.8075 | +251.10% | -86.13% | -62.36% | -73.05% / -73.08% |
| ~0.85 | warm original | `m4_20_64_10` / `naive_ef24` | 0.8627 / 0.8557 | +157.19% | -84.03% | -37.58% | -63.84% / -64.44% |
| ~0.85 | shuffled seed `20260709` | `m4_20_64_10` / `naive_ef24` | 0.8628 / 0.8557 | +199.61% | -84.02% | -37.54% | -68.53% / -70.70% |
| ~0.85 | shuffled seed `20260712` | `m4_20_64_10` / `naive_ef24` | 0.8573 / 0.8557 | +183.08% | -85.15% | -40.60% | -66.88% / -67.21% |
| ~0.85 | cold matched | `m4_20_64_10` / `naive_ef24` | 0.8622 / 0.8557 | +181.82% | -84.03% | -37.59% | -67.73% / -70.23% |

This supplement adds lower-recall selected robustness points near Recall@10 0.80
and 0.85, also with 3000 eval queries and 3 repeats. Each target now has a warm
original rerun, two shuffled measured-query seeds, and a matched cold/no-warmup
overlay. The 0.80 warm, `20260709`, and cold rows are near same-recall with naive;
the 0.80 seed `20260712` row is target-neighborhood evidence because Method4
recall is 0.0050 lower than naive. The 0.85 rows should also be described as
target-neighborhood comparisons because Method4 recall is 0.0016-0.0071 higher
than naive. In every selected 0.80/0.85 overlay, Method4 keeps higher same-run
QPS, fewer visited shards, lower EF-sum/query, and lower P95/P99 batch latency
than naive.

Source matrices:
`results/method4_claim_coverage_20260704/derived_claim_tables/robustness_080_085_query_order_overlay.csv`;
`results/method4_claim_coverage_20260704/derived_claim_tables/robustness_080_085_query_order_multiseed_summary.csv`;
`results/method4_claim_coverage_20260704/derived_claim_tables/robustness_080_085_warmup_overlay.csv`;
deltas:
`results/method4_claim_coverage_20260704/derived_claim_tables/robustness_080_085_method4_vs_naive_deltas.csv`;
`results/method4_claim_coverage_20260704/derived_claim_tables/robustness_080_085_sensitivity_deltas.csv`;
raw runs:
`results/method4_robustness_080_085_original_20260705/analysis_20260705_070956/`;
`results/method4_robustness_080_085_shuffle_20260709/analysis_20260705_071249/`;
`results/method4_robustness_080_085_shuffle_20260712/analysis_20260706_133627/`;
`results/method4_robustness_080_085_warmup_20260705/analysis_20260705_071529/`.

Additional ~0.90 robustness overlay:

| Scenario | Method / config | Recall | QPS | Visited | EF-sum | P95 batch ms | P99 batch ms | Source |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| warm original | Method4 `m4_60_40_8` | 0.9066 | 658.5 | 14.31 | 1101.1 | 166.9 | 168.9 | `results/method4_robustness_090_original_20260705/analysis_20260705_065005/claim_d_high_recall_latency_summary.csv` |
| warm original | Naive `naive_ef40` | 0.9068 | 270.1 | 46.00 | 1840.0 | 433.3 | 448.1 | `results/method4_robustness_090_original_20260705/analysis_20260705_065005/claim_d_high_recall_latency_summary.csv` |
| shuffled seed `20260708` | Method4 `m4_60_40_8` | 0.9054 | 665.4 | 14.31 | 1101.0 | 164.1 | 167.3 | `results/method4_robustness_090_shuffle_20260708/analysis_20260705_065219/claim_d_high_recall_latency_summary.csv` |
| shuffled seed `20260708` | Naive `naive_ef40` | 0.9068 | 278.6 | 46.00 | 1840.0 | 413.5 | 438.5 | `results/method4_robustness_090_shuffle_20260708/analysis_20260705_065219/claim_d_high_recall_latency_summary.csv` |
| shuffled seed `20260711` | Method4 `m4_60_40_8` | 0.9013 | 730.7 | 12.93 | 1044.2 | 150.8 | 159.0 | `results/method4_robustness_090_shuffle_20260711/analysis_20260706_132643/claim_d_high_recall_latency_summary.csv` |
| shuffled seed `20260711` | Naive `naive_ef40` | 0.9068 | 287.2 | 46.00 | 1840.0 | 412.0 | 428.2 | `results/method4_robustness_090_shuffle_20260711/analysis_20260706_132643/claim_d_high_recall_latency_summary.csv` |
| cold matched | Method4 `m4_60_40_8` | 0.9066 | 671.0 | 14.31 | 1101.0 | 164.2 | 167.5 | `results/method4_robustness_090_warmup_20260705/analysis_20260705_065435/claim_d_high_recall_latency_summary.csv` |
| cold matched | Naive `naive_ef40` | 0.9068 | 278.6 | 46.00 | 1840.0 | 421.6 | 440.1 | `results/method4_robustness_090_warmup_20260705/analysis_20260705_065435/claim_d_high_recall_latency_summary.csv` |

This supplement adds a lower-recall selected robustness point near Recall@10
0.90, also with 3000 eval queries and 3 repeats. Under warm original, two
shuffled seeds, and cold matched scenarios, Method4 remains ahead of same-run
naive by +143.77%, +138.88%, +154.44%, and +140.85% QPS respectively, visits
68.89-71.88% fewer shards, uses 40.16-43.25% lower EF-sum/query, and lowers
P95/P99 by 60.31-63.41% / 61.84-62.88%. Across the two shuffled seeds,
Method4's own shuffled-vs-original sensitivity is +1.05% to +10.96% QPS,
-9.66% to -1.67% P95, and -5.88% to -0.94% P99; cold vs warm changes QPS by
+1.91% and P95/P99 by -1.64%/-0.82%. The `20260711` shuffled row is
target-neighborhood evidence: Method4 recall is 0.9013 versus naive 0.9068.

Source matrices:
`results/method4_claim_coverage_20260704/derived_claim_tables/robustness_090_query_order_overlay.csv`;
`results/method4_claim_coverage_20260704/derived_claim_tables/robustness_090_query_order_multiseed_summary.csv`;
`results/method4_claim_coverage_20260704/derived_claim_tables/robustness_090_warmup_overlay.csv`;
deltas:
`results/method4_claim_coverage_20260704/derived_claim_tables/robustness_090_method4_vs_naive_deltas.csv`;
`results/method4_claim_coverage_20260704/derived_claim_tables/robustness_090_sensitivity_deltas.csv`;
raw runs:
`results/method4_robustness_090_original_20260705/analysis_20260705_065005/`;
`results/method4_robustness_090_shuffle_20260708/analysis_20260705_065219/`;
`results/method4_robustness_090_shuffle_20260711/analysis_20260706_132643/`;
`results/method4_robustness_090_warmup_20260705/analysis_20260705_065435/`.

Additional ~0.955 robustness overlay:

| Scenario | Method / config | Recall | QPS | Visited | EF-sum | P95 batch ms | P99 batch ms | Source |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| warm original | Method4 `m4_160_80_8` | 0.9553 | 342.8 | 23.60 | 3306.2 | 314.4 | 318.9 | `results/method4_robustness_095_original_20260705/analysis_20260705_063443/claim_d_high_recall_latency_summary.csv` |
| warm original | Naive `naive_ef76` | 0.9543 | 232.3 | 46.00 | 3496.0 | 505.1 | 522.5 | `results/method4_robustness_095_original_20260705/analysis_20260705_063443/claim_d_high_recall_latency_summary.csv` |
| shuffled seed `20260707` | Method4 `m4_160_80_8` | 0.9555 | 366.6 | 23.62 | 3307.0 | 302.4 | 310.4 | `results/method4_robustness_095_shuffle_20260707/analysis_20260705_063719/claim_d_high_recall_latency_summary.csv` |
| shuffled seed `20260707` | Naive `naive_ef76` | 0.9543 | 205.0 | 46.00 | 3496.0 | 563.8 | 574.6 | `results/method4_robustness_095_shuffle_20260707/analysis_20260705_063719/claim_d_high_recall_latency_summary.csv` |
| shuffled seed `20260710` | Method4 `m4_160_80_8` | 0.9547 | 387.9 | 23.21 | 3273.7 | 272.0 | 275.5 | `results/method4_robustness_095_shuffle_20260710/analysis_20260706_131447/claim_d_high_recall_latency_summary.csv` |
| shuffled seed `20260710` | Naive `naive_ef76` | 0.9543 | 238.6 | 46.00 | 3496.0 | 484.6 | 497.8 | `results/method4_robustness_095_shuffle_20260710/analysis_20260706_131447/claim_d_high_recall_latency_summary.csv` |
| cold matched | Method4 `m4_160_80_8` | 0.9553 | 349.7 | 23.60 | 3306.2 | 316.0 | 329.6 | `results/method4_robustness_095_warmup_20260705/analysis_20260705_063957/claim_d_high_recall_latency_summary.csv` |
| cold matched | Naive `naive_ef76` | 0.9543 | 230.2 | 46.00 | 3496.0 | 526.4 | 542.4 | `results/method4_robustness_095_warmup_20260705/analysis_20260705_063957/claim_d_high_recall_latency_summary.csv` |

This supplement repeats the robustness shape at the original 0.955 operating
point with 3000 eval queries and 3 repeats. Under warm original, two shuffled
seeds, and cold matched scenarios, Method4 remains ahead of same-run naive by
+47.59%, +78.85%, +62.55%, and +51.90% QPS respectively, visits 48.66-49.55%
fewer shards, and lowers P95/P99 by 37.74-46.37% / 38.98-45.98%. Across the
two shuffled seeds, Method4's own shuffled-vs-original sensitivity is +6.95% to
+13.17% QPS, -13.49% to -3.83% P95, and -13.59% to -2.65% P99; cold vs warm
changes QPS by +2.03% and P95/P99 by +0.50%/+3.38%.

Source matrices:
`results/method4_claim_coverage_20260704/derived_claim_tables/robustness_095_query_order_overlay.csv`;
`results/method4_claim_coverage_20260704/derived_claim_tables/robustness_095_query_order_multiseed_summary.csv`;
`results/method4_claim_coverage_20260704/derived_claim_tables/robustness_095_warmup_overlay.csv`;
deltas:
`results/method4_claim_coverage_20260704/derived_claim_tables/robustness_095_method4_vs_naive_deltas.csv`;
`results/method4_claim_coverage_20260704/derived_claim_tables/robustness_095_sensitivity_deltas.csv`;
raw runs:
`results/method4_robustness_095_original_20260705/analysis_20260705_063443/`;
`results/method4_robustness_095_shuffle_20260707/analysis_20260705_063719/`;
`results/method4_robustness_095_shuffle_20260710/analysis_20260706_131447/`;
`results/method4_robustness_095_warmup_20260705/analysis_20260705_063957/`.

Additional ~0.95 curve-config robustness overlay:

| Scenario | Method / config | Recall | QPS | Visited | EF-sum | P95 batch ms | P99 batch ms | Source |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| warm original | Method4 `m4_curve_160_50_10` | 0.9538 | 407.3 | 23.60 | 2952.6 | 259.9 | 274.3 | `results/method4_robustness_095_curve_config_original_20260707/analysis_20260707_093724/claim_d_high_recall_latency_summary.csv` |
| warm original | Naive `naive_ef76` | 0.9543 | 240.9 | 46.00 | 3496.0 | 475.2 | 490.9 | `results/method4_robustness_095_curve_config_original_20260707/analysis_20260707_093724/claim_d_high_recall_latency_summary.csv` |
| shuffled seed `20260714` | Method4 `m4_curve_160_50_10` | 0.9537 | 422.4 | 23.60 | 2952.3 | 254.5 | 257.8 | `results/method4_robustness_095_curve_config_shuffle_20260707/analysis_20260707_093953/claim_d_high_recall_latency_summary.csv` |
| shuffled seed `20260714` | Naive `naive_ef76` | 0.9543 | 242.8 | 46.00 | 3496.0 | 476.0 | 496.5 | `results/method4_robustness_095_curve_config_shuffle_20260707/analysis_20260707_093953/claim_d_high_recall_latency_summary.csv` |
| shuffled seed `20260715` | Method4 `m4_curve_160_50_10` | 0.9530 | 412.5 | 23.63 | 2953.5 | 260.7 | 265.3 | `results/method4_robustness_095_curve_config_shuffle2_20260707/analysis_20260707_094525/claim_d_high_recall_latency_summary.csv` |
| shuffled seed `20260715` | Naive `naive_ef76` | 0.9543 | 239.6 | 46.00 | 3496.0 | 483.0 | 499.8 | `results/method4_robustness_095_curve_config_shuffle2_20260707/analysis_20260707_094525/claim_d_high_recall_latency_summary.csv` |
| cold matched | Method4 `m4_curve_160_50_10` | 0.9534 | 414.7 | 23.61 | 2952.2 | 256.9 | 261.1 | `results/method4_robustness_095_curve_config_warmup_20260707/analysis_20260707_094220/claim_d_high_recall_latency_summary.csv` |
| cold matched | Naive `naive_ef76` | 0.9543 | 240.4 | 46.00 | 3496.0 | 484.6 | 499.1 | `results/method4_robustness_095_curve_config_warmup_20260707/analysis_20260707_094220/claim_d_high_recall_latency_summary.csv` |

This supplement broadens selected robustness coverage to the `m4_160_50_10`
curve configuration near Recall@10 0.95, with 3000 eval queries and 3 repeats
for warm original order, two shuffled seeds, and a cold/no-warmup matched
window. Across the four scenarios, Method4 improves same-run QPS by +69.05% to
+73.97%, visits 48.64-48.70% fewer shards, uses 15.52-15.55% lower EF-sum,
and lowers P95/P99 by 45.31-46.98% / 44.12-48.07% versus naive. Method4 recall
is 0.9530-0.9538 versus same-run naive 0.9543, so this is target-neighborhood
robustness support, not a strict same-recall proof for every row.

Source matrices:
`results/method4_claim_coverage_20260704/derived_claim_tables/robustness_095_curve_config_overlay.csv`;
`results/method4_claim_coverage_20260704/derived_claim_tables/robustness_095_curve_config_query_order_multiseed_summary.csv`;
deltas:
`results/method4_claim_coverage_20260704/derived_claim_tables/robustness_095_curve_config_method4_vs_naive_deltas.csv`;
`results/method4_claim_coverage_20260704/derived_claim_tables/robustness_095_curve_config_sensitivity_deltas.csv`;
manifest:
`results/method4_claim_coverage_20260704/derived_claim_tables/robustness_095_curve_config_manifest.json`;
raw runs:
`results/method4_robustness_095_curve_config_original_20260707/analysis_20260707_093724/`;
`results/method4_robustness_095_curve_config_shuffle_20260707/analysis_20260707_093953/`;
`results/method4_robustness_095_curve_config_shuffle2_20260707/analysis_20260707_094525/`;
`results/method4_robustness_095_curve_config_warmup_20260707/analysis_20260707_094220/`.

Additional ~0.97 closest-neighborhood robustness overlay:

| Scenario | Method / config | Recall | QPS | Visited | EF-sum | P95 batch ms | P99 batch ms | Source |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| warm original | Method4 `repeat_m4_160_80_16` | 0.9687 | 342.7 | 23.62 | 4724.9 | 319.4 | 323.5 | `results/method4_claim_d_high_recall_latency_refine_20260704/analysis_20260704_220036/claim_d_high_recall_latency_summary.csv` |
| warm original | Naive `naive_ef104` | 0.9695 | 242.9 | 46.00 | 4784.0 | 444.2 | 449.7 | `results/method4_claim_d_high_recall_latency_refine_20260704/analysis_20260704_220036/claim_d_high_recall_latency_summary.csv` |
| shuffled seed `20260716` | Method4 `repeat_m4_160_80_16` | 0.9682 | 330.7 | 23.63 | 4725.8 | 319.1 | 322.4 | `results/method4_robustness_097_closest_shuffle_20260707/analysis_20260707_095706/claim_d_high_recall_latency_summary.csv` |
| shuffled seed `20260716` | Naive `naive_ef104` | 0.9695 | 213.4 | 46.00 | 4784.0 | 545.0 | 568.8 | `results/method4_robustness_097_closest_shuffle_20260707/analysis_20260707_095706/claim_d_high_recall_latency_summary.csv` |
| shuffled seed `20260717` | Method4 `repeat_m4_160_80_16` | 0.9685 | 338.9 | 23.62 | 4725.4 | 313.9 | 317.3 | `results/method4_robustness_097_closest_shuffle2_20260707/analysis_20260707_095948/claim_d_high_recall_latency_summary.csv` |
| shuffled seed `20260717` | Naive `naive_ef104` | 0.9695 | 215.5 | 46.00 | 4784.0 | 534.4 | 552.2 | `results/method4_robustness_097_closest_shuffle2_20260707/analysis_20260707_095948/claim_d_high_recall_latency_summary.csv` |
| cold matched | Method4 `repeat_m4_160_80_16` | 0.9685 | 337.0 | 23.63 | 4725.4 | 320.2 | 328.2 | `results/method4_robustness_097_closest_warmup_20260707/analysis_20260707_100229/claim_d_high_recall_latency_summary.csv` |
| cold matched | Naive `naive_ef104` | 0.9695 | 218.5 | 46.00 | 4784.0 | 517.3 | 533.7 | `results/method4_robustness_097_closest_warmup_20260707/analysis_20260707_100229/claim_d_high_recall_latency_summary.csv` |

This supplement covers `repeat_m4_160_80_16` versus `naive_ef104` under warm
original order, two shuffled seeds, and a cold/no-warmup matched-window
evaluation. Across the four scenarios, Method4 improves same-run QPS by
+41.09% to +57.27%, visits 48.63-48.65% fewer shards, uses about 1.2% lower
EF-sum/query, and lowers P95/P99 by at least 28.09%/28.07% versus naive.
Method4 recall is 0.9682-0.9687 versus same-run naive 0.9695, so this is
target-neighborhood robustness support rather than a strict same-recall proof
for every row. Method4's own query-order/cold sensitivity remains modest:
shuffled-vs-original QPS is -3.52% to -1.11%, and cold-vs-warm QPS is -1.68%.

Source matrices:
`results/method4_claim_coverage_20260704/derived_claim_tables/robustness_097_closest_overlay.csv`;
`results/method4_claim_coverage_20260704/derived_claim_tables/robustness_097_closest_query_order_multiseed_summary.csv`;
deltas:
`results/method4_claim_coverage_20260704/derived_claim_tables/robustness_097_closest_method4_vs_naive_deltas.csv`;
`results/method4_claim_coverage_20260704/derived_claim_tables/robustness_097_closest_sensitivity_deltas.csv`;
manifest:
`results/method4_claim_coverage_20260704/derived_claim_tables/robustness_097_closest_manifest.json`;
raw runs:
`results/method4_claim_d_high_recall_latency_refine_20260704/analysis_20260704_220036/`;
`results/method4_robustness_097_closest_shuffle_20260707/analysis_20260707_095706/`;
`results/method4_robustness_097_closest_shuffle2_20260707/analysis_20260707_095948/`;
`results/method4_robustness_097_closest_warmup_20260707/analysis_20260707_100229/`.

Supported wording:

```text
At Recall@10 about 0.955, Method4 visits 23.214 lower shards/query versus naive all-shards 46.0 and improves stable QPS by 42.14% against the closest stable naive point. In the new 3-repeat batch-latency experiment, the ~0.97 same-recall point preserves this advantage: Method4 improves QPS by 40.58%, visits 46.39% fewer shards, and lowers P95/P99 batch latency by 31.85%/32.42%.
Two selected shuffled-query overlays of the same ~0.97 row preserve the
advantage: Method4 improves QPS by at least 50.37%, visits about 48.7% fewer
shards, and lowers P95/P99 batch latency by at least 37.36%/37.47% versus
same-run naive.
A matched cold/no-warmup overlay of the same measured query window also
preserves the advantage: Method4 improves QPS by 52.59%, visits 48.64% fewer
shards, and lowers P95/P99 batch latency by 39.29%/39.83% versus same-run naive.
The additional ~0.97 closest-neighborhood robustness supplement covers
`repeat_m4_160_80_16` versus `naive_ef104` under warm original order, two
shuffled seeds, and cold matched-window evaluation: Method4 improves same-run
QPS by +41.09% to +57.27%, visits about 48.6% fewer shards, uses about 1.2%
lower EF-sum/query, and lowers P95/P99 by at least 28.09%/28.07%. These rows
are target-neighborhood evidence because Method4 recall is 0.9682-0.9687
versus same-run naive 0.9695.
The additional ~0.955 robustness supplement repeats the same pattern across
warm original, two shuffled seeds, and cold matched-window scenarios: Method4
improves same-run QPS by +47.59% to +78.85%, visits about 48.7% fewer shards,
and lowers P95/P99 by at least 37.74%/38.98% versus naive.
The additional ~0.95 curve-config robustness supplement covers
`m4_curve_160_50_10` versus `naive_ef76` under warm original order, two
shuffled seeds, and cold matched-window evaluation: Method4 improves same-run
QPS by +69.05% to +73.97%, visits about 48.7% fewer shards, uses about 15.5%
lower EF-sum/query, and lowers P95/P99 by at least 45.31%/44.12%. These rows
are target-neighborhood evidence because Method4 recall is 0.9530-0.9538
versus same-run naive 0.9543.
The additional ~0.90 robustness supplement also preserves the advantage:
Method4 improves same-run QPS by +138.88% to +154.44%, visits 68.89-71.88%
fewer shards, uses 40.16-43.25% lower EF-sum/query, and lowers P95/P99 by at
least 60.31%/61.84% versus naive. The `20260711` row is target-neighborhood
evidence, not strict same-recall.
The additional ~0.80/~0.85 robustness supplement extends the same selected-row
robustness pattern to lower recall targets: at ~0.80, Method4 improves QPS by
+213.41% to +258.35%, visits 86.1-87.1% fewer shards, uses 62.34-63.36% lower
EF-sum/query, and lowers P95/P99 by at least 70.45%/70.72%; at ~0.85, Method4
improves QPS by +157.19% to +199.61%, visits 84.0-85.2% fewer shards, uses
37.5-40.6% lower EF-sum/query, and lowers P95/P99 by at least 63.84%/64.44%.
The ~0.80 seed `20260712` row and all ~0.85 rows are target-neighborhood
comparisons, not strict same-recall pairs.
```

Boundary:

```text
Do not claim Method4 dominates naive at ~0.99 from the current operating points. At the closest ~0.99 pair, Method4 visits fewer shards but has slightly lower QPS and slightly worse P95/P99, because the selected Method4 dynamic-EF point uses much higher EF-sum.
The 2026-07-05/2026-07-06 lower-EF/high-upper-k retune and strategy-search
formal checks did not close this gap: six formal Method4 candidates were faster
or had tighter tail latency than same-run naive controls in several comparisons,
but they reached only 0.9867-0.9875 Recall@10, not the required 0.99
neighborhood.
```

## Claim E: Compact Request / Controller-Native Shard-Major

The current evidence now includes both the historical engineering-path rows and
a current-runtime batch-latency matrix. The new matrix covers three execution
shapes across batch sizes 50/100/200/400 with three repeats each at 3000 eval
queries/repeat:

- `grouped_by_ef_materialized`: materialized JSON route plan grouped by EF.
- `compact_current`: compact MultiEP request on the current server path.
- `client_shard_major_expanded`: client-expanded single-shard REST negative
  control.

The current-runtime matrix has 33 successful cells and 3 expected error cells:
`client_shard_major_expanded` at batch size 400 fails with `BrokenPipe` from
the large expanded REST payload. That failure is reported as payload-limit
evidence, not averaged into latency.

| Variant | Recall | QPS | Requests/query | Source |
| --- | --- | --- | --- | --- |
| compact materialized routing | 0.9551 | 331.7 | 1.00 | results/qdrant_goal_recall_idea_095_compact_materialized/20260601_112623/summary.json |
| client-side shard-major negative control | 0.9550 | 152.4 | 23.20 | results/qdrant_goal_recall_idea_095_shard_major/20260601_113943/summary.json |
| server-native shard-major | 0.9557 | 379.8 | 1.00 | results/qdrant_goal_recall_idea_095_server_shard_major/20260601_115939/summary.json |
| fast-path old AB | 0.9552 | 372.4 | 1.00 | results/qdrant_goal_recall_idea_095_server_shard_major_old_ab/20260602_133751/summary.json |
| fast-path candidate AB1 | 0.9553 | 381.9 | 1.00 | results/qdrant_goal_recall_idea_095_server_shard_major_fastpath/20260602_133000/summary.json |
| fast-path candidate AB2 | 0.9556 | 367.9 | 1.00 | results/qdrant_goal_recall_idea_095_server_shard_major_fastpath_ab2/20260602_133951/summary.json |

New current-runtime matrix summary:

| Variant | Batch coverage | Recall | Request objects/query | QPS range | P95 batch ms range | Status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| grouped_by_ef_materialized | 50/100/200/400 | 0.9545 | 9.79 | 176.8-225.6 | 275.9-2715.0 | ok |
| compact_current | 50/100/200/400 | 0.9554 | 1.00 | 339.9-477.9 | 163.2-860.8 | ok |
| client_shard_major_expanded | 50/100/200 | 0.9554 | 23.61 | 130.0-154.4 | 415.6-1416.9 | ok through batch 200; batch 400 BrokenPipe |

Source: `results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_execution_mode_latency_matrix.csv`; raw run: `results/method4_claim_e_execution_mode_latency_20260704/analysis_20260704_225847/`.

Delta summary:

| Batch | Comparison | QPS delta | P95 delta | P99 delta | Request-object reduction |
| ---: | --- | ---: | ---: | ---: | ---: |
| 50 | compact_current vs grouped_by_ef_materialized | +66.18% | -40.83% | -47.32% | 89.78% |
| 100 | compact_current vs grouped_by_ef_materialized | +76.61% | -48.27% | -52.69% | 89.78% |
| 200 | compact_current vs grouped_by_ef_materialized | +95.06% | -52.63% | -55.02% | 89.78% |
| 400 | compact_current vs grouped_by_ef_materialized | +170.27% | -68.29% | -68.96% | 89.78% |
| 50 | compact_current vs client_shard_major_expanded | +161.50% | -60.73% | -67.22% | 95.76% |
| 100 | compact_current vs client_shard_major_expanded | +159.93% | -64.21% | -65.58% | 95.76% |
| 200 | compact_current vs client_shard_major_expanded | +185.02% | -66.68% | -67.61% | 95.76% |
| 400 | compact_current vs client_shard_major_expanded | unavailable | unavailable | unavailable | client-expanded payload fails |

Source: `results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_execution_mode_latency_deltas.csv`.

Serialized request-body accounting supplement:

| Batch | Variant | Request objects/query | JSON request bytes/query | Reduction vs client-expanded |
| ---: | --- | ---: | ---: | ---: |
| 200 | client_shard_major_expanded | 23.61 | 104064.1 | baseline |
| 200 | grouped_by_ef_materialized | 9.76 | 42375.1 | 59.28% |
| 200 | compact_current | 1.00 | 7040.7 | 93.23% |

The same per-query byte ratios hold across batch sizes 50/100/200/400 because
the same 3000 measured query plans are grouped into different batch envelopes:
`compact_current` reduces request objects by 95.76% and serialized JSON request
body bytes/query by about 93.23% versus the client-expanded negative control.
It also reduces JSON request bytes/query by about 83.39% versus
`grouped_by_ef_materialized`.

Source:
`results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_payload_bytes_summary.csv`;
raw run:
`results/method4_claim_e_payload_bytes_20260705/analysis_20260705_014703/`.
The supplement reconstructs the request plans and serializes the
`/points/search/batch` JSON bodies without executing lower searches.

Selected request/response body supplement:

| Batch | Variant | Recall | Request bytes/query | Response bytes/query | Total body bytes/query | Total reduction vs client-expanded |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 200 | client_shard_major_expanded | 0.9555 | 103915.5 | 24598.9 | 128514.4 | baseline |
| 200 | grouped_by_ef_materialized | 0.9546 | 42375.9 | 10194.6 | 52570.5 | 59.09% |
| 200 | compact_current | 0.9555 | 7039.1 | 1042.9 | 8082.0 | 93.71% |

Source:
`results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_wire_bytes_summary.csv`;
raw run:
`results/method4_claim_e_wire_bytes_20260705/analysis_20260705_020000/`.
This run actually sends the selected batch=200 `/points/search/batch` requests
for each execution shape and records the serialized JSON request body plus the
raw JSON response body returned by Qdrant.

Client-side route-planning time supplement:

| Variant | Query count | Search requests/query | Returned candidates/query | Total planning ms/query |
| --- | ---: | ---: | ---: | ---: |
| grouped_by_ef_materialized | 3000 | 9.779 | 97.79 | 0.167 |
| compact_current | 3000 | 1.000 | 10.00 | 0.250 |
| client_shard_major_expanded | 3000 | 23.613 | 236.13 | 0.292 |

Source:
`results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_planning_time_summary.csv`;
raw run:
`results/method4_claim_e_planning_time_20260705/analysis_20260705_021347/`.
This supplement measures client-side upper-label computation plus route-plan
materialization only. It excludes dataset loading, upper-index build, Qdrant
scroll/recovery, and all lower-search execution.

Selected Docker container-level CPU/network supplement:

| Variant | QPS | Controller CPU avg % | Controller RX bytes/query | Controller TX bytes/query | Cluster RX bytes/query | Cluster TX bytes/query |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| grouped_by_ef_materialized | 222.3 | 51.58 | 63483.8 | 44294.7 | 97554.1 | 63258.5 |
| compact_current | 428.1 | 13.00 | 9161.9 | 24005.7 | 32110.4 | 25618.1 |
| client_shard_major_expanded | 152.1 | 31.01 | 130808.3 | 53751.3 | 159826.0 | 75033.2 |

`compact_current` relative reductions at batch=200:

| Comparison | QPS delta | P95 delta | Controller CPU avg delta | Controller RX reduction | Cluster RX reduction |
| --- | ---: | ---: | ---: | ---: | ---: |
| compact vs grouped_by_ef_materialized | +92.62% | -52.05% | -74.80% | 85.57% | 67.08% |
| compact vs client_shard_major_expanded | +181.43% | -66.06% | -58.08% | 93.00% | 79.91% |

Source:
`results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_container_overhead_summary.csv`;
deltas:
`results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_container_overhead_deltas.csv`;
raw run:
`results/method4_claim_e_container_overhead_20260705/analysis_20260705_024218/`.
This supplement samples Docker Engine raw stats for the controller and three
data-shard containers while measured batches execute. It reports container-level
cumulative NetIO deltas and sampled Docker `CPUPerc`; it is not a physical NIC
or packet-level capture and does not attribute CPU inside Qdrant subsystems.

Selected Docker container memory supplement:

| Variant | QPS | Controller mem avg MiB | Controller mem max MiB | Cluster mem avg GiB | Cluster mem max GiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| grouped_by_ef_materialized | 219.9 | 1335.0 | 1542.8 | 39.48 | 39.71 |
| compact_current | 430.8 | 1153.0 | 1191.3 | 39.32 | 39.44 |
| client_shard_major_expanded | 152.7 | 1287.5 | 1440.4 | 39.44 | 39.61 |

`compact_current` relative memory deltas at batch=200:

| Comparison | Controller avg mem delta | Controller max mem delta | Cluster avg mem delta | Cluster max mem delta |
| --- | ---: | ---: | ---: | ---: |
| compact vs grouped_by_ef_materialized | -13.63% | -22.79% | -0.41% | -0.68% |
| compact vs client_shard_major_expanded | -10.45% | -17.30% | -0.29% | -0.43% |

Source:
`results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_container_memory_summary.csv`;
deltas:
`results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_container_memory_deltas.csv`;
raw run:
`results/method4_claim_e_container_overhead_memory_20260705/analysis_20260705_040749/`.
This supplement records Docker `MemUsage` snapshots from the same controller
and shard containers while measured batches execute. It is container-level
memory accounting and is not process RSS attribution.

Selected Docker-top process RSS supplement:

| Variant | QPS | Controller RSS avg MiB | Controller RSS max MiB | Cluster RSS avg GiB | Cluster RSS max GiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| grouped_by_ef_materialized | 218.0 | 1275.6 | 1629.4 | 36.15 | 36.53 |
| compact_current | 430.8 | 1105.3 | 1167.1 | 35.99 | 36.14 |
| client_shard_major_expanded | 152.8 | 1208.1 | 1376.4 | 36.08 | 36.28 |

`compact_current` relative process RSS deltas at batch=200:

| Comparison | Controller avg RSS delta | Controller max RSS delta | Cluster avg RSS delta | Cluster max RSS delta |
| --- | ---: | ---: | ---: | ---: |
| compact vs grouped_by_ef_materialized | -13.34% | -28.37% | -0.42% | -1.08% |
| compact vs client_shard_major_expanded | -8.50% | -15.21% | -0.24% | -0.40% |

Source:
`results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_process_rss_summary.csv`;
deltas:
`results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_process_rss_deltas.csv`;
raw run:
`results/method4_claim_e_process_rss_20260705/analysis_20260705_042151/`.
This supplement samples Docker's `/containers/{name}/top` API for the
controller and shard containers while measured batches execute. It is process
RSS accounting for container processes, not Qdrant subsystem attribution or
build-stage peak RSS.

Selected Linux host-interface byte-counter supplement:

| Role | grouped bytes/query | compact bytes/query | client-expanded bytes/query |
| --- | ---: | ---: | ---: |
| docker_bridge | 54831.2 | 8604.1 | 134186.8 |
| docker_veth | 161046.1 | 57651.6 | 234646.2 |
| loopback | 109669.4 | 17252.3 | 269724.8 |
| physical_nic | 362.7 | 336.1 | 540.9 |

`compact_current` relative reductions at batch=200:

| Comparison | Role | Bytes/query delta | QPS delta |
| --- | --- | ---: | ---: |
| compact vs grouped_by_ef_materialized | docker_bridge | -84.31% | +99.44% |
| compact vs grouped_by_ef_materialized | docker_veth | -64.20% | +99.44% |
| compact vs client_shard_major_expanded | docker_bridge | -93.59% | +183.09% |
| compact vs client_shard_major_expanded | docker_veth | -75.43% | +183.09% |

Source:
`results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_host_interface_bytes_role_summary.csv`;
deltas:
`results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_host_interface_bytes_deltas.csv`;
raw run:
`results/method4_claim_e_host_interface_bytes_20260705/analysis_20260705_061525/`;
script:
`tools/method4_claim_e_host_interface_bytes.py`.

This supplement records Linux `/sys/class/net` RX/TX byte deltas before and
after each measured variant repeat. It is useful as host-interface byte-counter
evidence, but it is not packet capture or Qdrant subsystem/function-level attribution. Because
the experiment runs on a single-host Docker deployment, large deltas appear on
docker bridge/veth/loopback roles; `physical_nic` is only the host external NIC
role and stays small.

Selected Docker bridge packet-capture supplement:

| Variant | Frame bytes/query | TCP payload bytes/query | Packets/query | QPS | P95 batch ms | P99 batch ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| grouped_by_ef_materialized | 129598.5 | 118184.2 | 172.94 | 213.5 | 1116.9 | 1130.6 |
| compact_current | 51579.5 | 51201.3 | 5.73 | 435.7 | 473.4 | 476.7 |
| client_shard_major_expanded | 208607.0 | 203434.9 | 78.35 | 154.8 | 1426.7 | 1448.6 |

`compact_current` relative packet-capture deltas at batch=200:

| Comparison | Frame bytes/query delta | TCP payload bytes/query delta | Packet count/query delta | QPS delta | P95 delta | P99 delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| compact vs grouped_by_ef_materialized | -60.20% | -56.68% | -96.69% | +104.06% | -57.61% | -57.84% |
| compact vs client_shard_major_expanded | -75.27% | -74.83% | -92.69% | +181.43% | -66.81% | -67.09% |

Source:
`results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_packet_capture_bridge_totals.csv`;
deltas:
`results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_packet_capture_bridge_deltas.csv`;
per-direction summary:
`results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_packet_capture_bridge_summary.csv`;
run manifest:
`results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_packet_capture_manifest.json`;
raw run:
`results/method4_claim_e_packet_capture_20260706/per_variant_repeats3_20260706_122755/`;
script:
`tools/method4_claim_e_packet_capture_summary.py`.

This supplement captures TCP packets on the `br-b9aac8010880` Docker bridge
with filter `tcp and net 172.24.0.0/16` while running the three Claim E
execution modes at batch=200 with 3000 eval queries and 3 repeats per mode.
All three tcpdump logs report 0 kernel-dropped packets. The parser uses the
pcap original frame length, IP total length, and TCP payload length, so the
128-byte capture snaplen is sufficient for byte totals. This is true packet
capture evidence for the single-host Docker bridge, but it is not physical-NIC
capture and does not attribute CPU or memory inside Qdrant subsystems.

Selected physical-NIC packet-capture negative-control supplement:

| Interface | Capture scope | Query executions in capture window | Matching packets | Frame bytes | TCP payload bytes | Kernel drops |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| ens6f0np0 | physical_nic_packet_capture | 27000 | 0 | 0 | 0 | 0 |
| ens111f0np0 | physical_nic_packet_capture | 27000 | 0 | 0 | 0 | 0 |

Source:
`results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_packet_capture_physical_nic_totals.csv`;
per-interface summary:
`results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_packet_capture_physical_nic_summary.csv`;
concurrent execution summary:
`results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_packet_capture_physical_nic_execution_summary.csv`;
run manifest:
`results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_packet_capture_physical_nic_manifest.json`;
raw run:
`results/method4_claim_e_packet_capture_physical_nic_20260707/all_variants_repeats3_preload_20260707_082833/`.

This 2026-07-07 supplement captured both UP external NICs with filter
`tcp and net 172.24.0.0/16` while running all three Claim E execution modes at
batch=200 for 3000 eval queries * 3 repeats each. The concurrent execution
summary completed all three modes: `compact_current` QPS mean 434.93,
`grouped_by_ef_materialized` 218.15, and `client_shard_major_expanded` 152.42.
Both physical-NIC pcaps contain only the global pcap header and report 0
packets captured, 0 packets received by filter, and 0 kernel-dropped packets.
This removes the prior physical-NIC permission blocker for the single-host
negative control. It should be used only to say that Docker-subnet Claim E
traffic did not traverse these external NICs in this single-host deployment;
it is not physical-network payload-savings evidence and not Qdrant subsystem/function-level
resource attribution.

Selected Qdrant `/metrics` REST/process-counter supplement:

| Variant | QPS | P95 batch ms | P99 batch ms | Qdrant REST batch duration delta s | Qdrant REST avg duration ms/window | Minor page faults delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| grouped_by_ef_materialized | 201.0 | 1179.1 | 1196.9 | 10.51 | 700.5 | 403553.3 |
| compact_current | 386.6 | 564.2 | 571.1 | 7.09 | 472.9 | 31765.0 |
| client_shard_major_expanded | 138.5 | 1675.5 | 1690.1 | 9.61 | 640.7 | 488789.7 |

`compact_current` relative Qdrant `/metrics` deltas at batch=200:

| Comparison | QPS delta | P95 delta | P99 delta | REST batch duration delta | Minor page-fault delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| compact vs grouped_by_ef_materialized | +92.34% | -52.15% | -52.29% | -32.49% | -92.13% |
| compact vs client_shard_major_expanded | +179.21% | -66.33% | -66.21% | -26.18% | -93.50% |

Source:
`results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_qdrant_metrics_summary.csv`;
deltas:
`results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_qdrant_metrics_deltas.csv`;
raw run:
`results/method4_claim_e_qdrant_metrics_20260705/analysis_20260705_073436/`;
script:
`tools/method4_claim_e_qdrant_metrics.py`.

This supplement scrapes Qdrant's Prometheus `/metrics` endpoint before and
after each measured variant repeat. It confirms that each measured repeat
accounts for exactly 15 `/points/search/batch` REST calls. These are Qdrant-
exposed server/process counters, not packet capture, build-stage resource
accounting, or subsystem-level CPU/memory attribution. The
`collection_hardware_metric_cpu` metric is exposed in this deployment but
stayed at 0 in the measured windows; memory gauges are retained as raw process
snapshots and are not used for strong claims because they are noisy.

Supported wording:

```text
Keeping the compact MultiEP request shape on the current server path preserves
Recall@10 while reducing request objects/query from 9.79 to 1.00 versus the
grouped materialized route plan, and from 23.61 to 1.00 versus the client-
expanded shard-major negative control. In the current-runtime batch matrix this
improves QPS and P95/P99 latency at batch sizes 50/100/200/400 versus grouped
materialized routing, and at batch sizes 50/100/200 versus client-expanded
shard-major. Serialized request-body accounting shows the same compact path
also reduces JSON request bytes/query by 83.39% versus grouped materialized
routing and by 93.23% versus client-expanded shard-major. A selected batch=200
wire-body supplement also shows compact_current reducing JSON response
bytes/query by 95.76% and total JSON body bytes/query by 93.71% versus
client-expanded shard-major. Client-side upper-label plus route-plan
materialization stays below 0.3 ms/query for all three measured execution
shapes. A selected Docker stats supplement at batch=200 also shows compact
reducing controller container RX bytes/query by 85.57% versus grouped
materialized and 93.00% versus client-expanded, with lower sampled controller
CPU average. A companion Docker MemUsage supplement records lower controller
container memory snapshots for compact_current than for grouped materialized or
client-expanded at batch=200, while total cluster memory is essentially flat.
A Docker-top process RSS supplement shows the same shape for controller process
RSS. A selected Linux host-interface supplement also records lower docker
bridge/veth bytes/query for compact_current than for grouped materialized and
client-expanded modes at batch=200. A selected Docker bridge packet-capture
supplement records lower frame bytes/query, TCP payload bytes/query, and packet
count/query for compact_current than for grouped materialized and
client-expanded modes at batch=200. A selected Qdrant /metrics supplement
records lower REST batch duration deltas and lower process minor page-fault
deltas for compact_current than for grouped materialized and client-expanded
modes at batch=200. A physical-NIC negative-control supplement over the same
Claim E execution modes records zero matching Docker-subnet packets on both UP
external NICs, confirming that this single-host Docker run's packet traffic is
bridge/veth-local rather than external-NIC traffic. At batch size 400 the
client-expanded negative control fails with BrokenPipe, showing the payload
expansion limit rather than a usable latency point.
```

Boundary: the current-runtime matrix does not recreate the pre-shard-major
old binary. The byte supplements measure serialized JSON request bodies and
raw JSON response bodies only; they do not measure HTTP framing/compression
or Qdrant subsystem/function-level CPU/memory attribution. The planning-time supplement is
client-side upper-label plus plan materialization only, not lower-search
execution. The Docker supplements are container-level NetIO, sampled
`CPUPerc`/`MemUsage`, and Docker-top process RSS snapshots. The host-interface
supplement records Linux interface counters on a single-host Docker deployment;
the packet-capture supplement records TCP packets on the single-host Docker
bridge; the physical-NIC supplement is a zero-packet negative control for the
single-host Docker subnet; the Qdrant `/metrics` supplement records
server-exposed REST/process counters. These supplements are not
physical-network payload byte-savings evidence, build-stage peak RSS, or
Qdrant subsystem-level internal CPU/memory attribution.
Use the historical rows only as engineering-path evidence, not as the same
evaluation scope as the new current-runtime matrix.

## Claim F: Worker-Local Peer Pre-Merge

The physical trace, direct-peer simulation, same-image server A/B, enabled
10000-query sanity run, and the new batch-latency matrix together support Claim
F. The strongest completed evidence is now the fan-in mechanism and batch-shape
matrix; production enabled-vs-disabled P95/P99 still needs a server restart A/B
if the paper wants strong runtime tail-latency wording.

| Variant | Recall | QPS | Search req/query | Candidate groups/query | Returned candidates/query | Source |
| --- | --- | --- | --- | --- | --- | --- |
| direct-peer logical-shard fan-in baseline | 0.9556 | 154.3 | 23.21 | 23.21 | 232.08 | results/qdrant_goal_recall_idea_095_direct_peer_premerge_ab_baseline/20260603_080131/summary.json |
| direct-peer peer-local pre-merge simulation | 0.9554 | 152.0 | 23.23 | 2.98 | 29.77 | results/qdrant_goal_recall_idea_095_direct_peer_premerge_ab_premerged/20260603_080247/summary.json |
| server peer pre-merge enabled fresh | 0.9553 | 405.4 | 1.00 | 1.00 | 10.00 | results/qdrant_goal_recall_idea_095_server_peer_premerge_fresh/20260603_092757/summary.json |
| server peer pre-merge disabled fresh | 0.9552 | 379.4 | 1.00 | 1.00 | 10.00 | results/qdrant_goal_recall_idea_095_server_peer_premerge_disabled_fresh/20260603_092949/summary.json |
| server peer pre-merge retained repeated | 0.9552 | 423.2 | 1.00 | 1.00 | 10.00 | results/qdrant_goal_recall_idea_095_server_peer_premerge/20260603_090530/summary.json |
| server peer pre-merge enabled 10000-query sanity | 0.9540 | 430.0 | 1.00 | 1.00 | 10.00 | results/qdrant_goal_recall_idea_095_server_peer_premerge_10k/20260603_090738/summary.json |

The 10000-query enabled row is a longer-window sanity check for the enabled
server path. It should not be used as a paired A/B row because there is no
disabled 10000-query baseline, and its Recall@10 should not be directly
compared with the 3000-query rows because it evaluates additional queries.

New 3 variants * 4 batch sizes * 3 repeats matrix:

| Batch | Variant | Recall | QPS mean | QPS stdev | Candidate groups/query | P95 batch ms | P99 batch ms |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 50 | coordinator current | 0.9556 | 326.0 | 2.6 | 1.00 | 163.2 | 167.3 |
| 50 | direct-peer no pre-merge | 0.9556 | 153.3 | 1.1 | 23.63 | 455.4 | 477.0 |
| 50 | direct-peer local pre-merge | 0.9556 | 153.7 | 0.8 | 2.98 | 371.4 | 479.8 |
| 100 | coordinator current | 0.9556 | 404.7 | 11.2 | 1.00 | 264.9 | 271.9 |
| 100 | direct-peer no pre-merge | 0.9556 | 162.2 | 2.5 | 23.63 | 758.8 | 783.9 |
| 100 | direct-peer local pre-merge | 0.9556 | 160.7 | 4.3 | 2.98 | 753.1 | 792.2 |
| 200 | coordinator current | 0.9556 | 457.3 | 3.1 | 1.00 | 455.9 | 457.8 |
| 200 | direct-peer no pre-merge | 0.9556 | 173.0 | 1.9 | 23.63 | 1344.0 | 1371.0 |
| 200 | direct-peer local pre-merge | 0.9556 | 174.8 | 2.3 | 2.98 | 1303.8 | 1350.9 |
| 400 | coordinator current | 0.9556 | 498.2 | 4.9 | 1.00 | 822.4 | 839.1 |
| 400 | direct-peer no pre-merge | 0.9556 | 183.4 | 3.1 | 23.63 | 2305.3 | 2336.4 |
| 400 | direct-peer local pre-merge | 0.9556 | 179.7 | 2.7 | 2.98 | 2362.1 | 2402.6 |

Source: `results/method4_claim_coverage_20260704/derived_claim_tables/claim_f_premerge_batch_latency_matrix.csv`; raw run: `results/method4_claim_f_premerge_batch_latency_20260704/analysis_20260704_222700/`.

Delta summary:

| Batch | Comparison | QPS delta | P95 delta | P99 delta | Candidate reduction |
| ---: | --- | ---: | ---: | ---: | ---: |
| 50 | local pre-merge vs no pre-merge | +0.30% | -18.44% | +0.60% | 87.40% |
| 100 | local pre-merge vs no pre-merge | -0.93% | -0.76% | +1.06% | 87.40% |
| 200 | local pre-merge vs no pre-merge | +1.05% | -2.99% | -1.47% | 87.40% |
| 400 | local pre-merge vs no pre-merge | -2.00% | +2.46% | +2.83% | 87.40% |
| 50 | coordinator current vs direct-peer no pre-merge | +112.70% | -64.16% | -64.94% | N/A |
| 100 | coordinator current vs direct-peer no pre-merge | +149.48% | -65.09% | -65.32% | N/A |
| 200 | coordinator current vs direct-peer no pre-merge | +164.37% | -66.08% | -66.61% | N/A |
| 400 | coordinator current vs direct-peer no pre-merge | +171.64% | -64.33% | -64.09% | N/A |

Source: `results/method4_claim_coverage_20260704/derived_claim_tables/claim_f_premerge_batch_latency_deltas.csv`.

Supported wording:

```text
Method4 high-recall routing creates many logical shard streams that collapse
onto a few physical peers. Worker-local pre-merge reduces candidate stream
fan-in by about 87.4% in the direct-peer batch matrix, preserving Recall@10
across batch sizes 50/100/200/400. The stricter same-image server A/B also
shows a +6.84% QPS improvement at the selected 0.955 recall point. An enabled
10000-query sanity run keeps the compact server execution shape with Recall@10
0.95403 and QPS 429.98.
```

Boundary: the new batch matrix is still a direct-peer simulation for the
pre-merge on/off comparison. It intentionally demonstrates fan-in shape and
client-observed batch latency under the simulation, but it still sends and
serializes all logical shard results before Python-side local pre-merge. Use
the same-image server A/B for production QPS wording. Do not claim production
P95/P99 improvement until rerunning the server with peer pre-merge enabled vs
`QDRANT_DISABLE_SHARD_MAJOR_PEER_PREMERGE=1` across batch sizes.

## Claim G: Method4-Aware Physical Layout

Claim G is the cleanest completed planned matrix in the current workspace: the offline 9-point placement matrix and matched online 18-run batch-latency matrix both exist and passed integrity checks.

Offline method4-aware vs round-robin:

| Workers | Placement | Baseline | P95 max EF improvement | CV improvement | P95 shard-count improvement |
| --- | --- | --- | --- | --- | --- |
| 2 | method4_aware | round_robin | 6.21% | 98.39% | 0.00% |
| 3 | method4_aware | round_robin | 4.87% | 75.03% | 7.14% |
| 4 | method4_aware | round_robin | 8.00% | 88.46% | 9.09% |

Matched online placement deltas:

| Batch | Placement | Recall delta | QPS delta | P95 delta | P99 delta |
| --- | --- | --- | --- | --- | --- |
| 100 | size_balanced | +0.000000 | -1.27% | +0.36% | +1.11% |
| 100 | method4_aware | -0.000433 | +2.05% | -0.43% | -0.56% |
| 200 | size_balanced | +0.000000 | -0.35% | +0.82% | +0.88% |
| 200 | method4_aware | -0.000433 | +1.23% | -0.96% | -0.98% |

Concurrency=8 single-query supplement:

| Placement | Recall@10 | QPS | P95 latency | P99 latency | Visited shards | EF-sum/query |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| round_robin | 0.9575 | 331.72 | 34.19 ms | 38.70 ms | 22.642 | 3224.832 |
| method4_aware | 0.9587 | 339.36 | 32.39 ms | 35.64 ms | 23.228 | 3278.544 |

In that supplement, Method4-aware improves QPS by 2.30%, P95 by 5.27%,
and P99 by 7.91% versus deployed round-robin. This is useful concurrent-load
support, but it is not the primary physical-only A/B because the historical
deployed Method4-aware collection is not a byte-identical logical clone of the
round-robin source and it visits slightly more shards.

Supported wording:

```text
Across 2, 3, and 4 workers, Method4-aware layout lowers P95 max worker EF/query by 4.9-8.0% and worker-load CV by 75-98% versus round-robin. In the matched 3-worker online matrix, it preserves Recall@10, improves QPS by 1.2-2.1%, and reduces P95/P99 batch latency by 0.4-1.0%.
```

Boundary: online gains are consistent but modest; do not claim large latency reductions in every workload. Use the concurrency=8 row as a deployed-collection supplement, not as the primary matched-layout proof.

Raw provenance is now explicit in `results/method4_claim_coverage_20260704/derived_claim_tables/claim_g_raw_provenance_audit_20260708.csv`. It indexes the offline placement raw run at `results/method4_claim_g_physical_layout_20260704/analysis_20260704_164039/`, the matched Method4-aware and size-balanced deploy outputs at `results/method4_claim_g_matched_layout_deploy_20260704/20260704_170511/` and `results/method4_claim_g_matched_layout_deploy_20260704/20260704_171202/`, the primary matched batch-latency run at `results/method4_claim_g_batch_latency_20260704/analysis_20260704_172015/`, and the concurrency=8 supplement at `results/method4_claim_g_online_latency_20260704/analysis_20260704_165643/`. It also records the low-pressure diagnostic and the invalid route-mismatch diagnostic, but those diagnostic rows are not used as primary claim support.

## Claim H: Semantic Fidelity

Claim H is now backed by a standalone sampled semantic invariant audit. The audit reconstructs `compact_multi_ep` query plans for 100 queries on `qdrant_controller_idea_method4map_full_20260601`, using `upper_k=160`, `base_ef=80`, `factor=8`, `top_k=10`, and `source_id_dedup_block_size=1183515`.

Audit scale:

| Metric | Value |
| --- | ---: |
| Queries | 100 |
| Compact searches | 100 |
| Upper labels audited | 16000 |
| Visited logical shards audited | 2323 |
| Queries with multi-assignment | 99 |

Semantic invariant results:

| Invariant | Status | Source |
| --- | --- | --- |
| upper routing unchanged | pass | results/method4_claim_h_semantic_audit_20260704/audit_20260704_210845/semantic_invariant_summary.csv |
| point_to_shards unchanged | pass | results/method4_claim_h_semantic_audit_20260704/audit_20260704_210845/semantic_invariant_summary.csv |
| multi-assignment preserved | pass | results/method4_claim_h_semantic_audit_20260704/audit_20260704_210845/semantic_invariant_summary.csv |
| per-shard entry points preserved | pass | results/method4_claim_h_semantic_audit_20260704/audit_20260704_210845/semantic_invariant_summary.csv |
| dynamic EF formula preserved | pass | results/method4_claim_h_semantic_audit_20260704/audit_20260704_210845/semantic_invariant_summary.csv |
| lower search remains per logical shard | pass | results/method4_claim_h_semantic_audit_20260704/audit_20260704_210845/semantic_invariant_summary.csv |
| no adaptive shard pruning | pass | results/method4_claim_h_semantic_audit_20260704/audit_20260704_210845/semantic_invariant_summary.csv |
| source-id dedup preserved | pass | results/method4_claim_h_semantic_audit_20260704/audit_20260704_210845/semantic_invariant_summary.csv |
| final global merge preserved | pass | results/method4_claim_h_semantic_audit_20260704/audit_20260704_210845/semantic_invariant_summary.csv |

Supported wording:

```text
In a 100-query sampled compact_multi_ep semantic audit, all 9 external Method4 wrapper invariants pass: upper routing, point_to_shards routing, multi-assignment, per-shard entry points, per-shard Dynamic EF, logical-shard lower search, no adaptive shard pruning, source-id de-duplication, and final global merge semantics.
```

Boundary:

```text
This is an external wrapper semantic audit. It does not provide an internal C++ step-by-step reference trace.
```

## Cross-Cutting Evidence

### Shard-Count Scaling

This is partial scalability evidence, not a full worker-count scaling experiment.

| Case | Method | Logical shards | Effective shards | Recall | QPS | Visited | Source |
| --- | --- | --- | --- | --- | --- | --- | --- |
| orion_s31 | orion | 31 | 46 | 0.9520 | 386.9 | 19.33 | results/method4_benchmark_matrix/method4_shard_scaling_095_corrected_reuse_eval3000/corrected_001/orion_s31/20260617_112503/summary.json |
| orion_s46 | orion | 46 | 69 | 0.9500 | 380.6 | 23.74 | results/method4_benchmark_matrix/method4_shard_scaling_095_corrected_reuse_eval3000/corrected_001/orion_s46/20260617_112915/summary.json |
| kmeans_s31 | kmeans | 31 | 31 | 0.9561 | 456.2 | 16.05 | results/method4_benchmark_matrix/method4_shard_scaling_095_corrected_reuse_eval3000/corrected_001/kmeans_s31/20260617_113341/summary.json |
| kmeans_s46 | kmeans | 46 | 46 | 0.9556 | 426.8 | 20.01 | results/method4_benchmark_matrix/method4_shard_scaling_095_corrected_reuse_eval3000/corrected_001/kmeans_s46/20260617_113816/summary.json |
| naive_s31 | naive | 31 | 31 | 0.9505 | 276.7 | 31.00 | results/method4_benchmark_matrix/method4_shard_scaling_095_corrected_reuse_eval3000/corrected_001/naive_s31/20260617_114308/summary.json |
| naive_s46 | naive | 46 | 46 | 0.9528 | 261.6 | 46.00 | results/method4_benchmark_matrix/method4_shard_scaling_095_corrected_reuse_eval3000/corrected_001/naive_s46/20260617_114405/summary.json |

A 2026-07-05 latency supplement reruns the existing Orion, Simple KMeans
`cpp_kmeans_baseline`, and Naive shard-count scaling rows with 3000 eval
queries, 100 warmup queries, batch size 100, and 3 repeats. It reuses live
collections and does not rebuild indexes.

| Case | Method | Logical shards | Effective shards | Recall | QPS | Visited | P95 batch ms | P99 batch ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| orion_s31 | Orion | 31 | 46 | 0.9519 | 441.9 | 19.30 | 245.7 | 246.9 |
| orion_s46 | Orion | 46 | 69 | 0.9507 | 410.7 | 23.71 | 260.5 | 267.5 |
| kmeans_s31 | Simple KMeans | 31 | 31 | 0.9560 | 469.7 | 16.05 | 226.8 | 235.7 |
| kmeans_s46 | Simple KMeans | 46 | 46 | 0.9557 | 489.5 | 19.98 | 218.6 | 220.1 |
| naive_s31 | Naive | 31 | 31 | 0.9504 | 275.6 | 31.00 | 393.3 | 410.2 |
| naive_s46 | Naive | 46 | 46 | 0.9529 | 238.0 | 46.00 | 479.0 | 487.8 |

The selected scaling comparison shows graceful Orion degradation from base
31 to base 46: QPS changes by -7.08%, P95 by +6.00%, and P99 by +8.33% while
visited shards increase by 22.89%. Simple KMeans `cpp_kmeans_baseline` is also
now covered: from 31 to 46 shards, QPS changes by +4.20%, visited shards by
+24.45%, P95 by -3.63%, and P99 by -6.62%. Naive degrades more sharply over
its 31 to 46 shard scaling row: QPS changes by -13.65%, P95 by +21.80%, and
P99 by +18.92%. At the same base logical-shard count, Orion still improves QPS
by 60.35%/72.57% and lowers P95 by 37.52%/45.62% versus Naive for
base31/base46, respectively; Simple KMeans also beats Naive in this selected
tail-latency supplement. Do not use this cross-cut cell to claim Orion latency
superiority over the selected `cpp_kmeans_baseline` operating point, because
the KMeans rows are faster and visit fewer shards here.

Source matrix:
`results/method4_claim_coverage_20260704/derived_claim_tables/crosscut_shard_scaling_latency_095.csv`;
deltas:
`results/method4_claim_coverage_20260704/derived_claim_tables/crosscut_shard_scaling_latency_095_deltas.csv`;
raw runs:
`results/method4_shard_scaling_latency_20260705/`.

Worker-count boundary audit:
`results/method4_claim_coverage_20260704/derived_claim_tables/worker_count_live_deployment_audit_20260705.csv`.
It confirms the live online rows above run on a fixed 3-worker deployment
(one controller peer plus three shard-worker peers). Offline placement proxy
rows cover worker_count 2/3/4, and the fixed-deployment rows above still must
not be presented as physical worker-count scaling.

The requested 2026-07-08 basic online worker-count supplement is now recorded in
`results/method4_claim_coverage_20260704/derived_claim_tables/worker_count_online_scaling_summary_20260708.csv`,
with deltas in
`results/method4_claim_coverage_20260704/derived_claim_tables/worker_count_online_scaling_deltas_20260708.csv`,
raw repeat rows in
`results/method4_claim_coverage_20260704/derived_claim_tables/worker_count_online_scaling_raw_rows_20260708.csv`,
Docker stats in
`results/method4_claim_coverage_20260704/derived_claim_tables/worker_count_docker_stats_summary_20260708.csv`,
and run provenance in
`results/method4_claim_coverage_20260704/derived_claim_tables/worker_count_online_scaling_manifest_20260708.json`.
It uses GloVe only, Method4 only, worker_count 1/2/3/4, 46 active custom
shards, 3000 eval queries, 100 warmup queries, batch_size=100, and 3 repeats.

| Target | worker_count=1 QPS / P95 / P99 | worker_count=4 QPS / P95 / P99 | wc4 QPS speedup | wc4 P95/P99 delta |
| --- | ---: | ---: | ---: | ---: |
| 0.80 | 907.28 / 126.07 ms / 134.55 ms | 1234.45 / 91.65 ms / 93.64 ms | 1.36x | -27.30% / -30.41% |
| 0.85 | 641.98 / 184.61 ms / 188.97 ms | 1020.40 / 110.35 ms / 112.68 ms | 1.59x | -40.22% / -40.37% |
| 0.90 | 432.76 / 250.97 ms / 258.37 ms | 762.33 / 144.36 ms / 148.79 ms | 1.76x | -42.48% / -42.41% |
| 0.95 | 218.63 / 488.70 ms / 513.15 ms | 412.18 / 259.16 ms / 262.64 ms | 1.89x | -46.97% / -48.82% |

Boundary: this is a basic Method4-only scaling supplement. It does not include
Naive or Simple KMeans worker-count controls, so use it for Method4 online
worker-count behavior only. The temporary Qdrant storage for these runs was
bind-mounted under `/home` and removed after each worker-count run; the raw
result tree remains at `results/method4_worker_count_online_scaling_20260708/`.

### Docker Cleanup Audit

The project-related Docker cleanup audit is recorded in
`results/method4_claim_coverage_20260704/derived_claim_tables/docker_cleanup_opportunity_audit_20260708.csv`.
No Docker prune, volume deletion, or image deletion was performed. The audit
records these current cleanup boundaries:

| Item | State | Size estimate | Action |
| --- | --- | ---: | --- |
| Current `qdrant/qdrant:method4-peer-premerge` image | in use | 334 MB image, 138.4 MB unique | keep while current evidence cluster is running |
| Current `qdrant-controller_*` volumes | in use | about 144.2 GB total | keep; these are the live evidence-cluster volumes |
| Old exited `qdrant-idea_*` volumes | approval-required candidate | about 29.0 GB | delete only after explicit approval |
| Old exited `qdrant-naive_*` volumes | approval-required candidate | about 30.9 GB | delete only after explicit approval |
| Docker build cache | approval-required candidate | 4.659 GB | `docker builder prune` only after approval |
| Worker-count temporary storage | already removed | no `docker_storage` dirs remain | no action |

Large unrelated Docker resources also exist, including the 53.3 GB
`postmill-populated-exposed-withimg` image, a 29.7 GB exited Milvus writable
layer, and a 20.5 GB running `fuzzycache` writable layer. They are outside this
project-specific cleanup scope and should not be touched unless a global Docker
cleanup is requested.

### Cost Analysis

The cost table now records the parts that can be verified from current
artifacts: indexed vector copies, expansion ratio, request-object counters,
Claim E serialized JSON request/response body bytes, Claim E client-side
route-planning time, selected 0.95 performance rows, and live disk footprint
for collections still present in the current Docker cluster. Disk is measured
read-only with
`docker exec ... du -sb /qdrant/storage/collections/<collection>` and the main
disk number sums only the three data-shard containers; controller metadata is
reported separately in the raw storage table.

| Case | Family | Expansion | Live disk GiB | Recall | QPS | Visited | Source |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| bench095_orion_default_095 | Orion | 1.175 | 8.455 | 0.9512 | 390.7 | 22.69 | crosscut_cost_main_collections.csv |
| bench095_simple_kmeans_095 | Simple KMeans | 1.000 | 7.945 | 0.9544 | 231.2 | 24.00 | crosscut_cost_main_collections.csv |
| bench095_naive_all_shards_095 | Naive | 1.000 | 7.482 | 0.9528 | 261.9 | 46.00 | crosscut_cost_main_collections.csv |
| current_method4_map_095 | Orion | 1.184 | 8.393 | 0.9553 | 396.8 | 23.21 | crosscut_cost_main_collections.csv |
| current_naive_all_shards_095 | Naive | 1.000 | 7.482 | 0.9548 | 274.4 | 46.00 | crosscut_cost_main_collections.csv |
| orion_multiassign_default_095 | Orion | 1.185 | N/A | 0.9479 | 409.1 | 21.14 | crosscut_cost_main_collections.csv |
| orion_multiassign_w2c2_095 | Orion | 1.499 | N/A | 0.9501 | 458.0 | 21.30 | crosscut_cost_main_collections.csv |
| orion_multiassign_w2c3_095 | Orion | 1.852 | N/A | 0.9618 | 385.2 | 24.48 | crosscut_cost_main_collections.csv |
| simple_multiassign_a1014_095 | Simple KMeans | 1.887 | N/A | 0.9533 | 220.0 | 24.00 | crosscut_cost_main_collections.csv |

Sources: `results/method4_claim_coverage_20260704/derived_claim_tables/crosscut_cost_main_collections.csv`; raw live-disk rows:
`results/method4_claim_coverage_20260704/derived_claim_tables/crosscut_cost_storage_du.csv`; raw run:
`results/method4_claim_cost_analysis_20260704/analysis_20260704_232343/`.

The coverage package also now includes a cross-cut request/candidate pressure
proxy table derived from the completed Claim E/F latency matrices:
`results/method4_claim_coverage_20260704/derived_claim_tables/crosscut_request_candidate_pressure_proxy.csv`.

| Scope | Variant | Batch | Search req/query | Candidate groups/query | Returned candidates/query | Reduction vs baseline |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Claim E compact request | compact_current vs client-expanded | 200 | 1.00 vs 23.61 | 1.00 vs 23.61 | 10.00 vs 236.09 | 95.76% |
| Claim E compact request | compact_current vs grouped materialized | 400 | 1.00 vs 9.79 | 1.00 vs 9.79 | 10.00 vs 97.87 | 89.78% |
| Claim F local pre-merge | direct_peer_local_premerge vs no_premerge | 200 | 23.63 vs 23.63 | 2.98 vs 23.63 | 29.78 vs 236.30 | 87.40% candidate pressure |

This proxy table is useful for request-object and candidate-fan-in pressure.
For Claim E request size, the coverage package now also includes serialized
JSON request-body accounting:
`results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_payload_bytes_summary.csv`.
For the selected batch=200 row it also includes client-observed JSON response
body accounting:
`results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_wire_bytes_summary.csv`.
For client-side routing overhead, it includes upper-label plus route-plan
materialization time:
`results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_planning_time_summary.csv`.
For selected container-level runtime overhead, it includes Docker stats CPU,
network, and memory counters:
`results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_container_overhead_summary.csv`.
`results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_container_memory_summary.csv`.
For selected process-level runtime overhead, it includes Docker-top RSS
snapshots:
`results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_process_rss_summary.csv`.
For selected host-interface byte counters, it includes Linux `/sys/class/net`
RX/TX deltas:
`results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_host_interface_bytes_role_summary.csv`.
`results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_host_interface_bytes_deltas.csv`.
For selected Docker bridge packet capture, it includes pcap-derived frame,
IP, TCP payload, and packet-count totals:
`results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_packet_capture_bridge_totals.csv`.
`results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_packet_capture_bridge_deltas.csv`.
For selected physical-NIC packet-capture negative control, it includes
per-interface pcap totals and the concurrent execution summary:
`results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_packet_capture_physical_nic_totals.csv`.
`results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_packet_capture_physical_nic_execution_summary.csv`.
For selected Qdrant server-exposed REST/process counters, it includes
Prometheus `/metrics` before/after deltas:
`results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_qdrant_metrics_summary.csv`.
`results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_qdrant_metrics_deltas.csv`.
For selected controller/worker process-level attribution, it includes host
`/proc` CPU/RSS samples for the Qdrant container PIDs:
`results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_controller_process_attribution_summary.csv`.
`results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_controller_process_attribution_deltas.csv`.
For build-stage resource accounting, it now includes an artifact audit of
existing `builds.csv` schemas:
`results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_build_stage_artifact_audit.csv`.
The per-cost-row source audit is:
`results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_cost_row_build_source_audit.csv`.
It also includes a smoke-level fresh-build time/RSS instrumentation run:
`results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_build_stage_instrumented_smoke_20260707.csv`.
The smoke manifest is:
`results/method4_claim_coverage_20260704/derived_claim_tables/claim_e_build_stage_instrumented_smoke_manifest.json`.

| Scope | Variant | Batch | Request bytes/query | Response bytes/query | Total body bytes/query |
| --- | --- | ---: | ---: | ---: | ---: |
| Claim E wire bodies | client_shard_major_expanded | 200 | 103915.5 | 24598.9 | 128514.4 |
| Claim E wire bodies | grouped_by_ef_materialized | 200 | 42375.9 | 10194.6 | 52570.5 |
| Claim E wire bodies | compact_current | 200 | 7039.1 | 1042.9 | 8082.0 |

| Scope | Variant | Total planning ms/query | Request objects/query |
| --- | --- | ---: | ---: |
| Claim E planning time | grouped_by_ef_materialized | 0.167 | 9.779 |
| Claim E planning time | compact_current | 0.250 | 1.000 |
| Claim E planning time | client_shard_major_expanded | 0.292 | 23.613 |

| Scope | Variant | Controller CPU avg % | Controller RX bytes/query | Cluster RX bytes/query |
| --- | --- | ---: | ---: | ---: |
| Claim E container overhead | grouped_by_ef_materialized | 51.58 | 63483.8 | 97554.1 |
| Claim E container overhead | compact_current | 13.00 | 9161.9 | 32110.4 |
| Claim E container overhead | client_shard_major_expanded | 31.01 | 130808.3 | 159826.0 |

| Scope | Variant | Controller mem avg MiB | Controller mem max MiB | Cluster mem avg GiB |
| --- | --- | ---: | ---: | ---: |
| Claim E container memory | grouped_by_ef_materialized | 1335.0 | 1542.8 | 39.48 |
| Claim E container memory | compact_current | 1153.0 | 1191.3 | 39.32 |
| Claim E container memory | client_shard_major_expanded | 1287.5 | 1440.4 | 39.44 |

| Scope | Variant | Controller RSS avg MiB | Controller RSS max MiB | Cluster RSS avg GiB |
| --- | --- | ---: | ---: | ---: |
| Claim E process RSS | grouped_by_ef_materialized | 1275.6 | 1629.4 | 36.15 |
| Claim E process RSS | compact_current | 1105.3 | 1167.1 | 35.99 |
| Claim E process RSS | client_shard_major_expanded | 1208.1 | 1376.4 | 36.08 |

| Scope | Role | grouped bytes/query | compact bytes/query | client-expanded bytes/query |
| --- | --- | ---: | ---: | ---: |
| Claim E host-interface | docker_bridge | 54831.2 | 8604.1 | 134186.8 |
| Claim E host-interface | docker_veth | 161046.1 | 57651.6 | 234646.2 |
| Claim E host-interface | loopback | 109669.4 | 17252.3 | 269724.8 |
| Claim E host-interface | physical_nic | 362.7 | 336.1 | 540.9 |

| Scope | Variant | Frame bytes/query | TCP payload bytes/query | Packets/query |
| --- | --- | ---: | ---: | ---: |
| Claim E bridge pcap | grouped_by_ef_materialized | 129598.5 | 118184.2 | 172.94 |
| Claim E bridge pcap | compact_current | 51579.5 | 51201.3 | 5.73 |
| Claim E bridge pcap | client_shard_major_expanded | 208607.0 | 203434.9 | 78.35 |

| Scope | Physical interface | Matching packets | Frame bytes | TCP payload bytes | Kernel drops |
| --- | --- | ---: | ---: | ---: | ---: |
| Claim E physical NIC negative control | ens6f0np0 | 0 | 0 | 0 | 0 |
| Claim E physical NIC negative control | ens111f0np0 | 0 | 0 | 0 | 0 |

| Scope | Variant | REST batch duration delta s | REST avg duration ms/window | Minor page faults delta |
| --- | --- | ---: | ---: | ---: |
| Claim E Qdrant metrics | grouped_by_ef_materialized | 10.51 | 700.5 | 403553.3 |
| Claim E Qdrant metrics | compact_current | 7.09 | 472.9 | 31765.0 |
| Claim E Qdrant metrics | client_shard_major_expanded | 9.61 | 640.7 | 488789.7 |

| Scope | Variant | Controller CPU delta s | Controller CPU ms/query-window | Controller RSS mean MiB | Cluster CPU delta s |
| --- | --- | ---: | ---: | ---: | ---: |
| Claim E `/proc` process attribution | grouped_by_ef_materialized | 54.12 | 6.013 | 406.57 | 395.02 |
| Claim E `/proc` process attribution | compact_current | 13.79 | 1.532 | 297.46 | 206.69 |
| Claim E `/proc` process attribution | client_shard_major_expanded | 44.85 | 4.983 | 386.28 | 261.58 |

The process-attribution supplement records compact_current controller CPU
delta -74.52% versus grouped and -69.25% versus client-expanded. Cluster
Qdrant process CPU delta is -47.68% and -20.98%, respectively. These are
host `/proc` process counters over the whole benchmark command window; the
per-query CPU normalization uses 9000 measured queries per variant but the CPU
numerator also includes route-state recovery/setup and warmup.

Build-stage artifact audit:

| Artifact family | Files | Has build wall time | Has build RSS | Applicability |
| --- | ---: | --- | --- | --- |
| Method4/Qdrant collection snapshots | 322 | no | no | Direct count/placement snapshots only |
| Standalone HNSW build timing | 18 | yes | no | Not Method4 distributed build evidence |
| Legacy collection snapshots | 7 | no | no | Indirect or not Method4 |

Build-stage instrumentation smoke:

| Scope | Collection | Timer scope | Logical points | Assigned points | Active shards | Peers | Wall time s | Max RSS MiB |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Claim E build-stage smoke | `claim_e_build_smoke_20260707_085303` | Fresh collection build harness plus 20-query tuning/eval | 1000 | 1381 | 7 | 4 | 37.82 | 64.203 |

The row-level audit covers all 9 rows in the cross-cut cost table and marks
all 9 as unsafe for filling `build_time_sec` or `memory_rss_bytes` from current
artifacts. Their `builds.csv` sources are post-build count/placement snapshots,
not timed build logs.

Boundary: full-size Method4 build-stage time/RSS and Qdrant
subsystem/function-level CPU/memory attribution are still missing. The
process-attribution supplement is useful process-level evidence for controller
and worker Qdrant PIDs, but it is not an internal profiler trace. The
build-stage smoke only proves the fresh-build time/RSS instrumentation path and
must not be used as full Method4 distributed build-cost evidence. The
2026-07-07 physical-NIC supplement resolves the prior capture-permission
blocker as a single-host negative control, but because it records zero matching
packets on external NICs it must not be used as physical-network payload
byte-savings evidence. The Docker bridge pcap supplement remains the
packet-capture evidence for the single-host Docker bridge, not physical-network
payload attribution or Qdrant subsystem/function-level attribution.
The new JSON byte tables measure request/response bodies only, not HTTP
framing/compression. The planning-time table measures only client-side
upper-label computation plus route-plan materialization and executes no lower
searches. The Docker tables measure container/process-level counters. The
host-interface table measures Linux interface counters on a single-host Docker
deployment; it is not Qdrant subsystem attribution. The Qdrant `/metrics`
table measures server-exposed REST/process counters and is likewise not
subsystem-level CPU/memory attribution. Use these tables for index-expansion,
live-disk, request-object, candidate-pressure, Claim E JSON body-size, client-
side planning-overhead, selected bridge packet capture, selected physical-NIC
negative-control capture, and selected runtime overhead reporting, not for a
full end-to-end build/runtime resource-cost claim.

### Unified Ablation Matrix Status

The original plan asks for a strict unified ablation matrix: one selected
operating point, one runtime image, matched controls, and common columns such as
Recall@10, QPS, P95/P99, visited shards, RPC/query, controller streams, and
index expansion. Current artifacts now include selected full-size multi-
assignment P95/P99 latency cells, so the former current-scope blocker is closed.
The result still should not be presented as a falsely perfect same-recall matrix:
the strict multi-assignment rows are current-harness rebuilds, the single control
lands above 0.95 recall, and w2c2 lands slightly below 0.95 on the 3000-query
latency window.

To avoid mixing incompatible experiment scopes, the coverage package contains a
status matrix rather than a single over-flattened performance matrix:
`results/method4_claim_coverage_20260704/derived_claim_tables/unified_ablation_status_matrix.csv`.

| Ablation group | Status | Decision | Caveat |
| --- | --- | --- | --- |
| dynamic_ef | available | include | Latency supplement is client-observed batch endpoint time, not server-internal lower-search trace. |
| compact_request_execution | available_with_overhead_metric_gap | include_with_caveat | Does not recreate historical old binary; client-observed request/response body bytes, client-side route-planning time, Docker container-level CPU/network counters, physical-NIC negative-control capture, and selected process-level PID attribution are present; full-size build-stage time/RSS and Qdrant subsystem/function-level CPU/memory attribution are missing. |
| worker_local_premerge | available_with_runtime_caveat | include_with_caveat | Batch matrix is direct-peer simulation/current coordinator path, not production enabled-vs-disabled P95/P99 server restart A/B. |
| physical_placement | available | include_as_placement_submatrix | Online gains are modest and this is a matched placement submatrix, not a mechanism-off single row. |
| multi_assignment | available_selected_latency_cells_with_caveats | include_as_selected_online_ablation_cell_with_caveats | Rows are current-harness rebuilds using /home-backed Qdrant storage and are not byte-identical historical replays. The single-assignment control is above 0.95 recall, while w2c2 lands slightly below 0.95 on the 3000-query latency window; use this as selected ablation/target-neighborhood latency evidence, not a strict same-recall proof for every pair. |
| topology_no_fission | available_selected_latency_cell | include_as_selected_online_ablation_cell | Selected 0.95 client-observed batch latency cell plus same-run config sensitivity, not the full random/KMeans/topology/full-fission online partition-family matrix. |

Selected strict multi-assignment latency supplement:

| Strategy | Config | Recall@10 | QPS | Visited shards | EF-sum/query | P95 ms | P99 ms | Cleanup |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| single | `upper_k=400,base_ef=120,factor=12` | 0.97317 | 279.23 | 30.18 | 8421.04 | 736.45 | 739.58 | deleted_after_latency |
| default | `upper_k=120,base_ef=70,factor=12` | 0.95260 | 475.27 | 21.35 | 3088.52 | 432.73 | 435.41 | deleted_after_latency |
| w2c2 | `upper_k=80,base_ef=50,factor=10` | 0.94913 | 557.80 | 21.24 | 2127.40 | 375.01 | 376.76 | deleted_after_latency |

No further strict multi-assignment experiment is required for the current
selected GloVe ablation scope. Retune/rerun only if the paper needs strict
same-recall single/default/w2c2 wording or broader ablation-matrix coverage.
Production pre-merge enabled-vs-disabled P95/P99 is optional unless the paper
wants a strict runtime on/off row rather than the existing fan-in/direct-peer
evidence.

### Hard Gap Artifact Audit

The 2026-07-05 read-only audit is recorded in
`results/method4_claim_coverage_20260704/derived_claim_tables/hard_gap_artifact_audit_20260705.csv`.
It checked whether the then-remaining hard gaps could be closed from existing
artifacts rather than by new runs. Later 2026-07-08/2026-07-09 supplements
supersede its current-state judgment while preserving the historical audit
trail.

| Gap | Current interpretation | Current next step |
| --- | --- | --- |
| Worker-count online scaling | Basic Method4-only supplement is complete for GloVe worker_count 1/2/3/4 and targets 0.80/0.85/0.90/0.95. | No next step for current basic scope; run Naive/Simple KMeans worker-count controls only for stronger comparative physical-worker claims. |
| L2 multi-dataset generalization | Out of current scope; reduced SIFT-100k Euclidean evidence remains available as historical external-validity support. | No current action: latest scope requires GloVe only and excludes GIST/Deep/full-L2 experiments. |
| Strict multi-assignment and topology/no-fission ablation cells | Selected topology/no-fission and selected full-size strict multi-assignment single/default/w2c2 latency cells now exist with caveats. | No current-scope action; retune/rerun only for strict same-recall single/default/w2c2 wording or broader all-ablation coverage. |

The corresponding run queue is
`results/method4_claim_coverage_20260704/derived_claim_tables/remaining_experiment_execution_queue.csv`.
It now marks `claim_a_partition_family_online_matrix` and
`strict_ablation_multiassign_latency` as `resolved_current_scope`, while keeping
Claim E full-size resource attribution and broader robustness as optional
stronger-wording work.

The historical strict-ablation live collection audit is recorded in
`results/method4_claim_coverage_20260704/derived_claim_tables/strict_ablation_live_collection_audit_20260705.csv`.
At that point the full-size `bench_ma_strict_*` collections were missing and the
`smoke_multiassign_*` collections were too small to substitute. The post-run
audit at
`results/method4_claim_coverage_20260704/derived_claim_tables/strict_ablation_live_collection_audit_20260709_post_multiassign.csv`
updates the state: `bench_ma_strict_orion_o100_single_s31`,
`bench_ma_strict_orion_o118_default_s31`, and
`bench_ma_strict_orion_o149_w2c2_s31` were rebuilt under `/home`-backed project
Qdrant storage, used for selected 3-repeat latency, and deleted after the raw
and derived results were preserved. The smoke collections remain smoke-only and
are not substitutes for the full-size evidence.

The companion worker-count deployment audit is recorded in
`results/method4_claim_coverage_20260704/derived_claim_tables/worker_count_live_deployment_audit_20260705.csv`.
It remains useful to avoid misreading fixed-deployment shard-count rows as
physical worker-count scaling. The later `worker_count_online_scaling_*_20260708`
artifacts close the requested basic Method4-only worker-count supplement, but not
cross-method worker-count controls.

Current-state progression:

- `current_hard_gap_blocker_audit_20260707.csv` and the 2026-07-08 readiness
  checks record the historical state before storage relocation and rebuilds:
  missing Claim A and strict multi-assignment collections, fixed worker-count
  deployment, root-space pressure, and out-of-scope non-GloVe dataset gaps.
- `current_hard_gap_blocker_audit_20260708_post_worker_count.csv` resolves the
  basic GloVe Method4-only worker-count item for the requested scope.
- `current_hard_gap_blocker_audit_20260709_post_claim_a.csv` resolves the Claim A
  random/topology/load-recalibrated online rows with current-harness rebuild
  caveats.
- `current_hard_gap_blocker_audit_20260709_post_multiassign.csv` resolves the
  selected strict multi-assignment latency rows with target-neighborhood and
  current-harness rebuild caveats, and records project Qdrant storage under
  `/home/taig/dry/qdrant/qdrant_storage` bind mounts.

| Gap | Current state | Feasible without external change |
| --- | --- | --- |
| Claim A partition-family online matrix | 2026-07-09 current-harness rebuilds produced preserved online rows for `random_balanced_46`, `kmeans_topology_46`, and `kmeans_topology_load_recalibrated_46`; temporary collections were deleted after latency runs. | yes_current_scope_with_rebuild_caveats |
| Strict multi-assignment latency | 2026-07-09 current-harness rebuilds produced selected full-size single/default/w2c2 latency rows; temporary `bench_ma_strict_*` collections were deleted after results were preserved. | yes_current_scope_with_selected_latency_caveats |
| Worker-count online scaling | Basic GloVe Method4-only worker_count 1/2/3/4 runs are complete; no Naive/Simple KMeans worker-count controls were run. | yes_basic_method4_only; no_for_cross_method_controls |
| Full L2 / multi-dataset generalization | Out of current scope; latest instruction requires GloVe only. | yes_out_of_scope |
| Claim E physical-NIC capture | Resolved as a single-host negative-control supplement: sudo tcpdump on `ens6f0np0` and `ens111f0np0` during the Claim E all-mode batch=200 window recorded 0 matching Docker-subnet packets, 0 frame bytes, 0 TCP payload bytes, and 0 kernel drops. | yes_resolved_20260707 |
| Claim E build-stage time/RSS | Smoke-level fresh-build instrumentation is present: `/usr/bin/time -v` over `claim_e_build_smoke_20260707_085303` recorded 37.82 s wall time and 65,744 KB max RSS. Full-size Method4 distributed build-stage time/RSS still requires future instrumented builds. | yes_smoke_resolved_20260707; no_for_full_size |

Requirement-level completion audits are recorded in
`goal_completion_audit_20260708.csv`, `goal_completion_audit_20260709.csv`, and
`goal_completion_audit_20260709_post_multiassign.csv`. The latest audit says the
current GloVe-only/no-0.99 scope has the requested basic worker-count, Claim A,
and selected strict multi-assignment rows complete with caveats. Original
full-scope completion is still not claimed for stronger optional wordings such
as strict same-recall single/default/w2c2 latency, full-size build-stage time/RSS,
Qdrant subsystem/function-level profiling, comparative worker-count controls,
non-GloVe generalization, or 0.99 Method4 dominance.


L2 harness readiness smoke:

| Artifact | Dataset | Collection | Distance config | Result | Caveat |
| --- | --- | --- | --- | --- | --- |
| `results/method4_claim_coverage_20260704/derived_claim_tables/l2_harness_readiness_smoke_20260705.csv` | `sift-128-euclidean.hdf5` | `smoke_l2_euclid_naive_20260705` | Qdrant `Euclid`; upper hnswlib `l2`; no normalization; lower-score-is-better merge | passed; live collection has 1000 indexed vectors and `distance=Euclid` | Smoke only: `train_limit=1000` uses a prefix subset while ground truth is full SIFT, so Recall@10 is not a valid L2 claim result |

Reduced SIFT-100k Euclidean evidence:

| Method | Recall@10 | QPS mean | Visited shards | EF-sum/query | Expansion | Params |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Orion no-fission | 0.9613 | 2285.6 +/- 124.4 | 1.99 | 64.3 | 1.052 | `upper_k=8; base_ef=24; factor=2` |
| Simple KMeans nprobe | 0.9676 | 2013.0 +/- 61.9 | 4.00 | 128.0 | 1.000 | `nprobe=4; ef=32` |
| Naive all-shards | 0.9859 | 898.2 +/- 5.5 | 16.00 | 256.0 | 1.000 | `ef=16` |

Source:
`results/method4_claim_coverage_20260704/derived_claim_tables/l2_reduced_sift100k_main_comparison.csv`;
raw runs under `results/method4_l2_reduced_sift100k_20260705/`.

The reduced dataset is
`/home/taig/dry/faiss/datasets/sift-128-euclidean-subset100k-eval1000.hdf5`.
It uses the first 100,000 SIFT train vectors and first 1,000 test queries, with
exact squared-L2 top-100 neighbors recomputed over that train subset. This fixes
the `train_limit`/full-ground-truth problem in the earlier smoke run. The
experiment uses `vector_distance=euclid`, hnswlib `l2`, Qdrant `Euclid`, no
vector normalization, and lower-score-is-better merge semantics.

Reduced L2 batch-latency supplement:

| Method | Recall@10 | QPS mean | Visited shards | EF-sum/query | P95 batch ms | P99 batch ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Orion no-fission | 0.9613 | 2386.7 +/- 363.6 | 1.99 | 64.3 | 48.8 | 49.6 |
| Naive all-shards, paired with Orion | 0.9859 | 907.7 +/- 19.4 | 16.00 | 256.0 | 119.7 | 120.4 |
| Simple KMeans nprobe | 0.9602 | 2638.1 +/- 822.5 | 4.00 | 128.0 | 53.7 | 57.5 |
| Naive all-shards, paired with Simple KMeans | 0.9859 | 941.0 +/- 70.8 | 16.00 | 256.0 | 117.5 | 121.6 |
| Orion full-fission | 0.9545 | 1812.9 +/- 174.8 | 2.88 | 172.5 | 66.8 | 67.5 |
| Naive all-shards, paired with full-fission Orion | 0.9859 | 1039.5 +/- 215.6 | 16.00 | 256.0 | 115.8 | 119.4 |

Source:
`results/method4_claim_coverage_20260704/derived_claim_tables/l2_reduced_sift100k_latency_matrix.csv`;
delta table:
`results/method4_claim_coverage_20260704/derived_claim_tables/l2_reduced_sift100k_latency_deltas.csv`;
raw run:
`results/method4_l2_reduced_sift100k_latency_20260705/analysis_20260705_010022/`;
SimpleKMeans-vs-Naive raw run:
`results/method4_l2_reduced_sift100k_latency_20260705/simple_vs_naive/analysis_20260705_035458/`;
full-fission focused tables:
`results/method4_claim_coverage_20260704/derived_claim_tables/l2_reduced_sift100k_full_fission_latency_matrix.csv`;
`results/method4_claim_coverage_20260704/derived_claim_tables/l2_reduced_sift100k_full_fission_latency_deltas.csv`;
full-fission Method4 raw run:
`results/method4_l2_reduced_sift100k_latency_20260706/full_orion_method4_only/analysis_20260706_125222/`;
fresh full-fission comparison Naive raw run:
`results/method4_l2_reduced_sift100k_latency_20260706/naive_only/analysis_20260706_125245/`.

Reduced L2 Dynamic-vs-Fixed EF supplement:

| Group | Orion variant | Dynamic config | Fixed config | Recall dynamic / fixed | QPS mean dynamic / fixed | EF-sum dynamic / fixed | P95 mean dynamic / fixed | P99 mean dynamic / fixed |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| nearest same-recall | no-fission | `u8,b24,f2` | `u8,ef36` | 0.9613 / 0.9617 | 2636.1 / 2281.1 | 64.3 / 71.6 | 55.0 / 53.0 | 59.1 / 55.4 |
| fixed higher recall | no-fission | `u8,b24,f2` | `u8,ef40` | 0.9613 / 0.9663 | 2636.1 / 2135.8 | 64.3 / 79.6 | 55.0 / 57.4 | 59.1 / 58.8 |
| fixed higher recall | no-fission | `u8,b24,f2` | `u8,ef44` | 0.9613 / 0.9688 | 2636.1 / 2118.1 | 64.3 / 87.6 | 55.0 / 53.6 | 59.1 / 54.9 |
| nearest same-recall | full-fission | `u8,b48,f4` | `u8,ef72` | 0.9545 / 0.9553 | 1957.5 / 1671.9 | 172.5 / 207.5 | 69.3 / 69.4 | 74.2 / 71.5 |

Source:
`results/method4_claim_coverage_20260704/derived_claim_tables/l2_reduced_sift100k_dynamic_vs_fixed_ef_matrix.csv`;
deltas:
`results/method4_claim_coverage_20260704/derived_claim_tables/l2_reduced_sift100k_dynamic_vs_fixed_ef_deltas.csv`.

Interpretation:

- This is positive reduced-L2 external-validity evidence: at Recall@10 above
  0.95, Orion no-fission visits 87.6% fewer shards than Naive, uses 74.9% lower
  EF-sum/query, improves QPS by 162.9%, and reduces client-observed P95/P99
  batch latency by 59.3%/58.8%.
- The Simple KMeans nprobe reduced-L2 latency supplement also stays above
  Recall@10 0.95 and, against its paired Naive run, visits 75.0% fewer shards,
  uses 50.0% lower EF-sum/query, improves QPS by 180.4%, and reduces
  client-observed P95/P99 batch latency by 54.3%/52.7%.
- The full-fission Orion reduced-L2 latency supplement uses the fission-expanded
  25-shard collection and stays above Recall@10 0.95. Against a fresh 16-shard
  Naive all-shards run over the same 1,000-query window, it visits 82.0% fewer
  shards, uses 32.6% lower EF-sum/query, improves QPS by 74.4%, and reduces
  client-observed P95/P99 batch latency by 42.3%/43.5%.
- The latency pairs are not exactly same recall: Orion is 0.9613, Simple KMeans
  is 0.9602, full-fission Orion is 0.9545, and the paired Naive rows are
  0.9859. Use them as reduced L2 efficiency results above the 0.95 threshold,
  not as strict same-recall L2 proofs.
- The Dynamic-vs-Fixed EF supplement closes one reduced-L2 internal ablation
  gap for the no-fission and full-fission Orion collections. The nearest
  fixed-EF rows have almost identical recall and show Dynamic EF using lower
  EF-sum with positive QPS and median-tail deltas; mean tail latency remains
  noisy, so keep this as a small external-validity supplement, not a replacement
  for the main Claim C GloVe evidence.
- This does not close the full multi-dataset generalization requirement. It is
  not full SIFT1M, GIST, or Deep, and it does not include the full set of key
  ablations on L2.

### Remaining Experiment Gaps

| Area | Status | Gap | Preferred next experiment |
| --- | --- | --- | --- |
| Minimum paper package | partial_current_scope | 0.97 same-recall Method4-vs-naive latency evidence is present and positive. A 2026-07-08 batch=200 high-recall supplement now covers the planned batch=200 shape for selected 0.97 and 0.99 pairs: m4_160_80_20_b200 vs naive_ef112_b200 has Recall@10 0.97183/0.97277, QPS +88.01%, visited -46.40%, and P95/P99 mean -67.32%/-69.33%; m4_400_160_20_b200 vs naive_ef200_b200 has Recall@10 0.98950/0.99017, QPS +30.53%, visited -25.48%, and P95/P99 mean -54.26%/-58.41%, but remains below 0.99 recall and below naive recall. The batch=200 naive controls are tail-unstable, so use this as strict same-batch evidence with a baseline-instability caveat. The closest 0.99 point remains a tradeoff, and 2026-07-05/2026-07-06 lower-EF/high-upper-k retune plus strategy-search formal runs did not produce a 3000-query Method4 candidate at 0.99 recall. The follow-ups formally repeated m4_560_40_16, m4_560_60_14, m4_480_80_16, and m4_520_60_16 after 500-query screening at Recall@10 0.9900-0.9906, but formal recall was only 0.98670-0.98747 while same-run naive_ef200 stayed at 0.99017. A 2026-07-07 broader upper-routing screen found five 500-query Method4 rows at Recall@10 0.9918-0.9920. Two selected formal repeats still failed to close the gap: high-QPS candidate m4_u720_b60_f12 reached 0.9880 recall over 3000 queries * 3 repeats versus same-run naive_ef200 at 0.99017, and top-recall candidate m4_u720_b40_f14 reached 0.9887 versus same-run naive_ef200 at 0.99017. The high-QPS candidate had +3.26% QPS and -10.48%/-11.89% P95/P99 versus naive_ef200; the top-recall candidate had -0.64% QPS and -5.50%/-7.35% P95/P99 versus naive_ef200. An additional 2026-07-08 alternate wider-routing screen selected m4_u960_b20_f10 after 500-query Recall@10 0.9916, but formal validation reached only 0.9876 versus naive_ef200 0.99017; it had +3.20% QPS and -10.16%/-11.89% P95/P99, so it is another boundary/negative row, not a 0.99 dominance result. None of these supports a 0.99 Method4 dominance claim. Two 0.99 boundary shuffled-query repeats (seeds 20260713 and 20260718) confirm the boundary: Method4 m4_400_160_20 remains below 0.99 recall (0.98923 and 0.98943) and below naive_ef200 recall (0.99017), despite slightly better QPS/tail in both shuffled runs. A matched cold/no-warmup 0.99 boundary repeat using query_start_offset=100 also keeps Method4 below 0.99 recall (0.98937) and below naive_ef200 recall (0.99017), while showing +0.29% QPS and -7.69%/-9.00% P95/P99; this is sensitivity evidence, not a 0.99 same-recall win. Current scoped Claim D wording no longer requires a 0.99 dominance result; use only the 0.80/0.85/0.90/0.95 and through-about-0.97 evidence, with 0.99 retained only as transparent boundary/negative context. | No 0.99 follow-up is required for the current paper scope; only reopen 0.99 if the claim scope changes. |
| Claim A partition-family online | current_scope_online_rows_complete_with_caveats | Offline partition oracle matrix is complete, and the online P95/P99 submatrix now includes the prior full-fission, balanced-KMeans, naive, and selected topology/no-fission rows plus 2026-07-09 current-harness rebuild rows for random_balanced_46, kmeans_topology_46, and kmeans_topology_load_recalibrated_46. The new current-harness rows used build upper_search_ef=400 to match Claim A oracle construction metadata and online upper_search_ef=160; all three temporary collections were deleted after latency results were preserved. Remaining caveats: the new rebuilds are not byte-identical to the 2026-07-04 oracle artifact, load-recalibrated no-fission routing is effectively the topology route map with recalibrated weights, balanced-KMeans batch=200 naive has high variance, and current full fission still does not beat balanced KMeans-only in oracle miss/routed shards. | No further Claim A random/topology/load-recalibrated online row is required for the current GloVe-only scope; only rerun a single fully matched all-family sweep or retune fission if stronger full-partition superiority wording is needed. |
| Ablation | selected_strict_multiassign_latency_complete_with_caveats | Dynamic EF, compact request/execution mode, worker-local pre-merge, physical placement, selected topology/no-fission 0.95 online P95/P99, and selected full-size strict multi-assignment latency cells now have usable evidence with caveats. The 2026-07-09 strict multi-assignment supplement rebuilt single/default/w2c2 full-size collections under /home-backed Qdrant storage, ran 3000-query * 3-repeat batch=200 latency, and deleted temporary collections after preserving results. Results: single Recall@10 0.97317/QPS 279.23/visited 30.18/EF-sum 8421.04/P95 736.45/P99 739.58 ms; default Recall@10 0.95260/QPS 475.27/visited 21.35/EF-sum 3088.52/P95 432.73/P99 435.41 ms; w2c2 Recall@10 0.94913/QPS 557.80/visited 21.24/EF-sum 2127.40/P95 375.01/P99 376.76 ms. Caveat: Current-harness rebuilds, not byte-identical historical replays; single assignment is above 0.95 recall and w2c2 is slightly below 0.95 on the 3000-query latency window, so use as selected ablation/target-neighborhood latency evidence rather than strict same-recall proof across all pairs. | No further strict multi-assignment experiment is required for the current selected GloVe ablation scope. Retune/rerun only if the paper needs strict same-recall single/default/w2c2 wording or broader ablation-matrix coverage. |
| Scalability | basic_worker_count_complete_comparative_partial | Shard-count scaling has 31/46 summary rows plus selected 3-repeat Orion, Simple KMeans, and Naive batch-latency supplements. A new 2026-07-08 basic online Method4-only worker-count experiment covers GloVe worker_count {1,2,3,4}, targets 0.80/0.85/0.90/0.95, 3000 eval queries, 100 warmup queries, batch_size=100, and 3 repeats. Worker_count=4 vs 1 improves QPS by 1.36x/1.59x/1.76x/1.89x and reduces mean P95 by 27.30%/40.22%/42.48%/46.97% for targets 0.80/0.85/0.90/0.95. This closes the requested basic worker-count supplement, but it is Method4-only and does not include Naive or Simple KMeans worker-count controls. | No further worker-count experiment is required for the current basic Method4-only scope. Run Naive/Simple KMeans worker-count controls only if stronger comparative physical-worker scaling claims are needed. |
| Cost analysis | partial | Indexed vectors / expansion, live-disk footprint, count-level request/candidate pressure proxies, Claim E serialized JSON request-body bytes, selected Claim E JSON response-body bytes, client-side route-planning time, selected Docker container-level CPU/network/memory counters, selected Docker-top process RSS snapshots, selected host-interface byte counters, selected Docker bridge packet capture, selected physical-NIC negative-control packet capture, selected Qdrant /metrics REST/process counters, smoke-level build-stage time/RSS instrumentation, and selected Qdrant controller/worker process-level CPU/RSS attribution are now recorded. The host-interface supplement shows compact_current reduces docker_bridge bytes/query by 84.31% versus grouped materialized and 93.59% versus client-expanded at batch=200. The Docker bridge packet-capture supplement shows compact_current reduces frame bytes/query by 60.20% versus grouped materialized and 75.27% versus client-expanded, with packet count/query down 96.69% and 92.69%; tcpdump reported 0 kernel-dropped packets. The Qdrant /metrics supplement records compact_current REST batch duration deltas -32.49% versus grouped and -26.18% versus client-expanded, but it is server-exposed REST/process counter evidence rather than subsystem attribution. A build-stage artifact audit confirms Method4/Qdrant builds.csv snapshots contain count/placement metadata but no build-duration/RSS fields; old HNSW timing rows are not Method4 distributed evidence. Full-size build time/RSS and Qdrant subsystem/function-level CPU/memory attribution remain incomplete; selected process-level PID attribution is now present; the new build-stage timing/RSS run is smoke-only and wraps a tiny fresh-build harness. The physical-NIC negative-control capture is complete for this single-host Docker setup, but it records zero matching external-NIC packets rather than physical-network payload savings. A 2026-07-07 sudo tcpdump run on ens6f0np0 and ens111f0np0 removed the prior permission blocker and produced a zero-packet physical-NIC negative control for the Claim E single-host window. A 2026-07-07 smoke-level fresh-build instrumentation run now wraps tools/qdrant_two_level_routing_experiment.py with /usr/bin/time -v: train_limit=1000, 20-query tuning/eval, 1000 logical points, 1381 assigned points, 1376 indexed vectors, 7 active custom shards across 4 peers, elapsed wall time 37.82 s, max RSS 65,744 KB, exit status 0. This proves the time/RSS instrumentation path for fresh builds, but it is not full-size Method4 distributed build cost and not pure build-only RSS attribution. A 2026-07-07 selected Claim E controller/worker process-level attribution supplement samples host /proc counters for the Qdrant controller and three shard worker PIDs during batch=200 execution-mode runs. compact_current controller Qdrant process CPU delta is 13.79 s versus 54.12 s grouped (-74.52%) and 44.85 s client-expanded (-69.25%); controller CPU ms per measured query-window is 1.532 versus 6.013 and 4.983. Controller RSS mean is 297.46 MiB versus 406.57 MiB grouped (-26.84%) and 386.28 MiB client-expanded (-22.99%). Cluster process CPU delta is 206.69 s versus 395.02 s grouped (-47.68%) and 261.58 s client-expanded (-20.98%). This is process-level host /proc attribution for Qdrant container PIDs during the whole benchmark command window, including route-state recovery/setup, warmup, measured repeats, and client orchestration; it is not Qdrant subsystem/function-level CPU or memory attribution. | Run full-size fresh Method4 distributed builds with build-stage timers and /usr/bin/time -v; collect Qdrant subsystem/function-level CPU/memory profiles only if stronger resource attribution is required; collect additional physical/per-flow capture only for multi-host or physical-network payload wording. |
| Multi-dataset generalization | out_of_current_scope | Per current instruction, only the GloVe dataset is required. Existing reduced SIFT-100k/L2 artifacts remain historical external-validity supplements, but full SIFT/GIST/Deep/L2 generalization is no longer a blocking experiment for the current scope. | No action for the current GloVe-only scope. |
| Robustness | partial | Selected Method4-vs-naive robustness overlays now cover the ~0.97 final-paper row, an additional ~0.97 closest-neighborhood configuration repeat_m4_160_80_16 vs naive_ef104, additional ~0.955, ~0.90, ~0.85, and ~0.80 rows, plus an additional ~0.95 curve configuration m4_curve_160_50_10 vs naive_ef76. Each selected row has 3000 eval queries and 3 repeats; every selected row now has two shuffled-query seeds and matched cold/no-warmup coverage. The new ~0.97 closest supplement records Method4 QPS +41.09% to +57.27%, visited shards -48.63% to -48.65%, EF-sum -1.22% to -1.24%, P95 -28.09% to -41.45%, and P99 -28.07% to -43.32% versus same-run naive, with Method4 recall 0.9682-0.9687 versus naive 0.9695. The new ~0.95 curve-config supplement records Method4 QPS +69.05% to +73.97%, visited shards -48.64% to -48.70%, EF-sum -15.52% to -15.55%, P95 -45.31% to -46.98%, and P99 -44.12% to -48.07% versus same-run naive, with Method4 recall 0.9530-0.9538 versus naive 0.9543. Method4 remains ahead of same-run naive in every selected overlay, but some rows are target-neighborhood rather than strict same-recall. The cold overlays use query_start_offset=100 and warmup_query_count=0 so their measured HDF5 query window matches the warm run test[100:3100]. Many broader matrix sweeps still have stability_repeats=0 and do not have query-order or cold/warm overlays. | Broaden query-order and cold/warm overlays across more recall targets/configs only if stronger matrix-wide robustness wording is needed. |

## Recommended Paper Claim Set

Use strong wording for:

- Claim C budget efficiency and client-observed batch P95/P99 improvement, while avoiding claims about server-internal lower-search P99 trace time.
- Claim B Orion multi-assignment frontier lift, oracle_gt_miss@10 reduction, and selected full-size strict single/default/w2c2 latency supplement, noting the latency rows are current-harness target-neighborhood evidence rather than strict same-recall proof across all three strategies.
- Claim D same-recall QPS/tail-latency advantage for the current 0.80/0.85/0.90/0.95 scope and through about 0.97, now with selected robustness overlays at ~0.80, ~0.85, ~0.90, ~0.955, the strict ~0.97 row, an additional ~0.97 closest-neighborhood configuration, and a strict batch=200 high-recall supplement. Keep the existing ~0.99 rows only as transparent boundary/negative context, not as a required claim target.
- Claim E compact request/current-runtime execution-mode efficiency, using the request/candidate pressure proxy table for count-level overhead, the serialized JSON request/response body byte tables for body-size overhead, the planning-time table for client-side routing overhead, the Docker supplements for selected container-level CPU/network/memory plus process RSS overhead, the selected Linux host-interface byte counters, the selected Docker bridge packet-capture supplement, the 2026-07-07 physical-NIC negative-control supplement, the selected Qdrant `/metrics` REST/process counters, the selected `/proc` controller/worker process-attribution supplement, and the smoke-level build-stage time/RSS instrumentation row, while keeping the full-size-build-RSS/subsystem-attribution, physical-network-payload, and old-binary caveats.
- Claim F fan-in reduction, completed direct-peer batch matrix, and server QPS A/B; keep production enabled-vs-disabled P95/P99 as future unless rerun.
- Claim G physical layout hot-worker evidence, matched batch-latency evidence, and the concurrency=8 deployed-collection supplement.
- Claim H external wrapper semantic fidelity, with the boundary that no C++ internal reference trace was run.
- Basic Method4-only worker-count scaling on GloVe for worker_count 1/2/3/4 at target neighborhoods 0.80/0.85/0.90/0.95, with the explicit boundary that no Naive/Simple KMeans worker-count controls were run.
- Cross-cutting cost reporting for index expansion, current live disk footprint, request/candidate pressure counts, Claim E serialized JSON request/response body bytes, Claim E client-side planning time, selected Docker container-level CPU/network/memory plus process RSS counters, selected Linux host-interface byte counters, selected Docker bridge packet capture, selected physical-NIC negative-control capture, selected Qdrant `/metrics` REST/process counters, selected `/proc` controller/worker process attribution, and smoke-level build-stage time/RSS instrumentation, with an artifact audit confirming that full-size Method4 build time/RSS and subsystem/function-level attribution remain missing.

Use cautious or downgraded wording for:

- Claim A topology-locality: online Orion-vs-simple-KMeans evidence exists, the offline oracle matrix supports topology convergence, and the online P95/P99 submatrix now includes existing full-fission/balanced-KMeans rows plus 2026-07-09 current-harness random/topology/load-recalibrated rebuild rows. A selected topology/no-fission online cell exists and visits fewer shards than full fission, but it is slightly worse on QPS/P95/P99. Current full fission does not beat balanced KMeans-only on oracle miss/routed shards; the new family rows close the missing online-cell gap for current scope but should be described as current-harness rebuilds, not byte-identical oracle replays.
- Strict unified ablation if worded as strict same-recall single/default/w2c2 proof: selected full-size P95/P99 latency rows now exist, but the single control is above 0.95 recall and w2c2 is slightly below 0.95 on the latency window, so a retune/rerun is needed only for that stronger wording.
- Comparative worker-count scaling across methods: the current worker-count supplement is Method4-only.
- Claim E controller-native overhead if worded as physical-network payload savings or Qdrant subsystem/function-level controller CPU/memory attribution: those metrics are still missing even though the current-runtime QPS/P95/P99 matrix, count-level proxy table, serialized JSON request/response body byte tables, client-side route-planning table, Docker container/process-level overhead tables, selected `/proc` Qdrant process-attribution tables, selected Linux host-interface byte-counter tables, selected Docker bridge packet-capture tables, selected physical-NIC negative-control tables, and selected Qdrant `/metrics` REST/process counter tables are now present.
- Cost analysis if worded as complete resource accounting: the build-stage artifact audit found no historical full-size Method4 build-duration/RSS fields, the new build-stage timer/RSS run is smoke-only, and Qdrant subsystem/function-level CPU/memory attribution still requires an instrumented rerun; the physical-NIC capture is complete only as a zero-packet single-host negative control.
- Non-GloVe or 0.99-dominance claims: both are out of the current requested scope.

## Source Manifest

The generated data package for this document is `results/method4_claim_coverage_20260704/source_manifest.json`.
The Claim A 2026-07-09 current-harness rebuild supplement is indexed at
`results/method4_claim_coverage_20260704/derived_claim_tables/claim_a_partition_family_current_rebuilds_20260709.csv`,
with raw builds under `results/method4_claim_a_partition_family_build_20260709/`
and raw latency runs under `results/method4_claim_a_partition_online_latency_20260709/`.
The Claim B raw provenance audit is indexed at `results/method4_claim_coverage_20260704/derived_claim_tables/claim_b_raw_provenance_audit_20260708.csv`, with online frontier rollups/source summaries, the strict quality/provenance rollup, and oracle-miss raw analysis promoted into the main manifest; the strict rollup remains quality/provenance evidence and does not silently replace the current Claim B key data. The selected strict multi-assignment latency supplement is indexed at `results/method4_claim_coverage_20260704/derived_claim_tables/strict_multiassign_latency_20260709.csv`, with raw rebuilds under `results/method4_benchmark_matrix/method4_strict_multiassign_latency_rebuild_20260709/`, raw latency runs under `results/method4_strict_ablation_multiassign_latency_20260709/`, and a post-cleanup live-collection audit at `results/method4_claim_coverage_20260704/derived_claim_tables/strict_ablation_live_collection_audit_20260709_post_multiassign.csv`.
The Claim C raw provenance audit is indexed at `results/method4_claim_coverage_20260704/derived_claim_tables/claim_c_raw_provenance_audit_20260708.csv`, with the same-recall performance, relevance-analysis, and batch-latency raw sources promoted into the main manifest.
The Claim G raw provenance audit is indexed at `results/method4_claim_coverage_20260704/derived_claim_tables/claim_g_raw_provenance_audit_20260708.csv`, with raw offline/deploy/batch/concurrency paths also promoted into the main manifest.
The Claim D batch=200 high-recall supplement is indexed at `results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_batch200_high_recall_summary.csv`, `results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_batch200_high_recall_deltas.csv`, and `results/method4_claim_coverage_20260704/derived_claim_tables/claim_d_batch200_high_recall_manifest.json`, with raw output under `results/method4_claim_d_batch200_targets_20260708/analysis_20260708_104611/`.
The worker-count supplement is indexed at `results/method4_claim_coverage_20260704/derived_claim_tables/worker_count_online_scaling_summary_20260708.csv`, `results/method4_claim_coverage_20260704/derived_claim_tables/worker_count_online_scaling_deltas_20260708.csv`, `results/method4_claim_coverage_20260704/derived_claim_tables/worker_count_docker_stats_summary_20260708.csv`, and `results/method4_claim_coverage_20260704/derived_claim_tables/worker_count_online_scaling_manifest_20260708.json`, with raw output under `results/method4_worker_count_online_scaling_20260708/`.
The Docker cleanup opportunity audit is indexed at `results/method4_claim_coverage_20260704/derived_claim_tables/docker_cleanup_opportunity_audit_20260708.csv`; it records cleanup candidates and keep/delete boundaries but no cleanup action.
The current package also includes `results/method4_claim_coverage_20260704/derived_claim_tables/unincorporated_artifact_review_20260707.csv`, which records high-value raw artifacts that were reviewed but either promoted, covered by an existing derived table, superseded by stronger formal evidence, or kept out of claim support because they are probe/smoke-only or lack a paired baseline.
The latest runnable-state recheck is recorded at `results/method4_claim_coverage_20260704/derived_claim_tables/current_experiment_readiness_recheck_20260708.csv`; the previous `20260707` recheck remains in the package for provenance.
The latest formal-experiment prerequisite checklist is `results/method4_claim_coverage_20260704/derived_claim_tables/formal_experiment_prerequisite_audit_20260708.csv`.
The persistent-goal blocked-threshold audit is `results/method4_claim_coverage_20260704/derived_claim_tables/goal_blocked_threshold_audit_20260708.csv`.
The post-worker-count current-scope blocker audit is `results/method4_claim_coverage_20260704/derived_claim_tables/current_hard_gap_blocker_audit_20260708_post_worker_count.csv`.
The post-Claim-A current-scope blocker audit is `results/method4_claim_coverage_20260704/derived_claim_tables/current_hard_gap_blocker_audit_20260709_post_claim_a.csv`, and the corresponding completion increment is `results/method4_claim_coverage_20260704/derived_claim_tables/goal_completion_audit_20260709.csv`. The latest post-multiassignment current-scope blocker audit is `results/method4_claim_coverage_20260704/derived_claim_tables/current_hard_gap_blocker_audit_20260709_post_multiassign.csv`, and the latest completion increment is `results/method4_claim_coverage_20260704/derived_claim_tables/goal_completion_audit_20260709_post_multiassign.csv`.
The Excel-friendly claim-support workbook is `docs/experiments/2026-07-09-method4-claim-support-excel-tables/method4_claim_support_excel_tables.xlsx`, with TSV exports and manifest under `docs/experiments/2026-07-09-method4-claim-support-excel-tables/`. The workbook now includes self-explanation sheets `S-01` source/caveat/provenance, `S-02` source-manifest raw/provenance paths, and `S-03` current-scope completion audit. The corresponding post-Excel self-contained integrity audit is `results/method4_claim_coverage_20260704/derived_claim_tables/final_evidence_document_integrity_audit_20260709_post_excel_selfcontained.csv`.
