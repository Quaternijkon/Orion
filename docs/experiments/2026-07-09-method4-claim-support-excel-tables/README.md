# Method4 Claim 支撑实验数据 Excel 文件

> [!WARNING]
> **Historical single-host workbook.** This workbook and the TSV exports retain
> the 2026-07-09 single-machine evidence exactly as published. They are not
> updated with native four-host data. See
> [`../2026-07-23-orion-native-v4-four-node-results.md`](../2026-07-23-orion-native-v4-four-node-results.md)
> for the separate native-v4 result set.

这里是从 `docs/experiments/2026-07-09-method4-claim-support-excel-chart-data.md` 导出的原始 Excel 友好文件。

- `method4_claim_support_excel_tables.xlsx`：一个 Excel 工作簿，每个 TSV 数据区一个 sheet；其中完整数字单元格已存为 Excel 数值类型，推荐直接打开这个文件绘图。
- `manifest.tsv`：所有导出表的索引，Tab 分隔。
- `tsv/*.tsv`：每个数据区的原始 TSV 文件，只包含表头和数据行，不包含 Markdown 说明、三反引号或代码块标记。

2026-07-09 追加说明：A-01、B-01 与 B-02 已补充 Naive hash all-shards baseline/reference 行；Naive 定义为 hash/random 分散到各分片、查询时搜索全部分片。工作簿还包含 `NAIVE_AUDIT` sheet，说明哪些表已补 Naive、哪些表已包含 Naive、哪些表不是三路由方式对比而不适用。

2026-07-09 自解释补充：工作簿新增 `S-01`、`S-02`、`S-03`、`S-04` 四个审阅 sheet。`S-01` 汇总每张图的支撑论据、推荐图形、来源和 caveat；`S-02` 扁平化 `source_manifest.json` 中的 raw/derived/provenance 路径并记录存在性、类型和基础规模；`S-03` 把当前 GloVe 范围完成状态和强表述才需要的 follow-up 分开列出；`S-04` 记录当前 GloVe 结果的共同环境、worker-count/shard-count 的可变部署条件和三种对比方法定义。`C-01`、`E-02` 的多数据块导出也已拆成 `C-01a/C-01b` 和 `E-02a/E-02b`，避免 chart_id 重复。

当前配置摘要：Qdrant 单机模拟、计算与存储隔离；Intel Xeon Gold 6330 @ 2.00GHz（2 sockets × 28 cores/socket × 2 threads/core）；GloVe-200-angular、200 维、Base size 1.2M、Angular 距离。主三方法对比以 46 个有效逻辑分片为主；worker-count 使用 1 个 controller 与 1--4 个各 4 逻辑 CPU 的 worker；shard-count 补充比较固定 3-worker 下的 31/46 个有效逻辑分片。完整定义见 `S-04` sheet。

完整性审计文件：`results/method4_claim_coverage_20260704/derived_claim_tables/final_evidence_document_integrity_audit_20260709_post_excel_selfcontained.csv`。

如果从 `.tsv` 复制到 Excel，请复制文件内容本身。每列之间是真实 Tab，行之间是 CRLF 换行。
