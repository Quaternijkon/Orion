#!/usr/bin/env python3
"""Summarize the native HashAll fixed-recall physical scale-out run."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


HOSTS = ["10.10.1.1", "10.10.1.2", "10.10.1.3", "10.10.1.4"]


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()

    primary_files = {
        1: args.root / "bench" / "m1-clean.json",
        2: args.root / "bench" / "m2-clean.json",
        3: args.root / "bench" / "m3-clean.json",
        4: args.root / "bench" / "m4-clean-confirm.json",
    }
    records = {machine_count: load(path) for machine_count, path in primary_files.items()}
    first_m4 = load(args.root / "bench" / "m4-clean.json")
    rows = []
    baseline_qps = records[1]["qps_mean"]
    previous_qps = None
    for machine_count in range(1, 5):
        record = records[machine_count]
        concurrency_source = first_m4 if machine_count == 4 else record
        repeats = record["repeats"]
        cpu_by_host = {
            host: statistics.fmean(
                repeat["cpu_average_cores"][host] for repeat in repeats
            )
            for host in HOSTS
        }
        total_cpu = sum(cpu_by_host.values())
        concurrency_four = next(
            (
                row["qps"]
                for row in concurrency_source["concurrency_sweep"]
                if row["concurrency"] == 4
            ),
            None,
        )
        row = {
            "machine_count": machine_count,
            "selected_ef": record["parameters"]["selected_hnsw_ef"],
            "tuning_recall_at_10": next(
                item["recall_at_10"]
                for item in record["tuning_sweep"]
                if item["hnsw_ef"] == record["parameters"]["selected_hnsw_ef"]
            ),
            "heldout_recall_at_10": record["heldout_recall"]["recall_at_10"],
            "selected_concurrency": record["parameters"]["selected_concurrency"],
            "qps_mean": record["qps_mean"],
            "qps_stdev": record["qps_stdev"],
            "qps_cv": record["qps_cv"],
            "qps_vs_m1": record["qps_mean"] / baseline_qps,
            "qps_vs_previous": (
                None if previous_qps is None else record["qps_mean"] / previous_qps
            ),
            "fixed_concurrency_4_qps_short_scan": concurrency_four,
            "cpu_average_cores_total": total_cpu,
            "coordinator_cpu_average_cores": cpu_by_host[HOSTS[0]],
            "cpu_usec_per_query": total_cpu / record["qps_mean"] * 1_000_000.0,
            "qps_per_cpu_core": record["qps_mean"] / total_cpu,
            "source": str(primary_files[machine_count]),
        }
        rows.append(row)
        previous_qps = record["qps_mean"]

    m4_relative_difference = abs(
        first_m4["qps_mean"] - records[4]["qps_mean"]
    ) / records[4]["qps_mean"]
    audit = {
        "status": "PASS",
        "gates": {
            "all_native_auto_sharding": all(
                record["request_contract"]["sharding_method"] == "auto"
                for record in records.values()
            ),
            "all_server_side_fanout": all(
                record["request_contract"]["standard_coordinator_request"]
                and not record["request_contract"]["client_side_fanout"]
                and not record["request_contract"]["shard_selector_present"]
                for record in records.values()
            ),
            "all_heldout_recall_at_least_0_90": all(
                record["heldout_recall"]["recall_at_10"] >= 0.90
                for record in records.values()
            ),
            "all_ef_bounded_10_to_64": all(
                10 <= record["parameters"]["selected_hnsw_ef"] <= 64
                for record in records.values()
            ),
            "all_qps_cv_below_0_03": all(
                record["qps_cv"] < 0.03 for record in records.values()
            ),
            "m4_confirmation_relative_difference_below_0_01": (
                m4_relative_difference < 0.01
            ),
            "all_prepare_points_and_indexed_equal_1m": all(
                load(args.root / "prepare" / f"m{machine_count}.json")[
                    "collection_info"
                ]["points_count"]
                == 1_000_000
                and load(args.root / "prepare" / f"m{machine_count}.json")[
                    "collection_info"
                ]["indexed_vectors_count"]
                == 1_000_000
                for machine_count in range(1, 5)
            ),
            "all_actual_placements_use_m_distinct_peers": all(
                len(
                    set(
                        load(args.root / "prepare" / f"m{machine_count}.json")[
                            "actual_placement"
                        ].values()
                    )
                )
                == machine_count
                for machine_count in range(1, 5)
            ),
        },
        "m4_first_qps": first_m4["qps_mean"],
        "m4_confirm_qps": records[4]["qps_mean"],
        "m4_relative_difference": m4_relative_difference,
    }
    if not all(audit["gates"].values()):
        audit["status"] = "FAIL"

    with (args.root / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.root / "summary.json").write_text(
        json.dumps(
            {
                "experiment": "native HashAll physical M=1..4 at Recall@10 >= 0.90",
                "rows": rows,
                "audit": audit,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (args.root / "audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8"
    )

    table_lines = [
        "| 真实机器/Shard 数 | ef | Held-out Recall@10 | 饱和并发 | QPS（均值 ± 标准差） | 相对 1 台 | 边际变化 | 集群 CPU μs/query |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        marginal = (
            "—"
            if row["qps_vs_previous"] is None
            else f"{(row['qps_vs_previous'] - 1) * 100:+.1f}%"
        )
        table_lines.append(
            "| {machine_count} | {selected_ef} | {heldout_recall_at_10:.4f} | "
            "{selected_concurrency} | {qps_mean:,.0f} ± {qps_stdev:,.0f} | "
            "{qps_vs_m1:.3f}× | {marginal} | {cpu_usec_per_query:.0f} |".format(
                marginal=marginal, **row
            )
        )
    fixed_four = " → ".join(
        f"{row['fixed_concurrency_4_qps_short_scan'] / 1000:.1f}k"
        for row in rows
    )
    summary_md = f"""# 原生 HashAll：1～4 台真实机器固定召回率结果

{chr(10).join(table_lines)}

## 确定性结论

- 在 SIFT1M、Recall@10≥0.90、原生 Qdrant 服务端 HashAll、每 shard 一台真实机器的条件下，饱和 QPS 从 1 台的 68.6k 增至 4 台的 82.3k，总提升约 19.9%，不是线性扩展。
- 最大边际收益发生在 1→2 台（+13.2%）；2→3 仅 +1.6%；3→4 约 +4.3%。因此准确描述是“小幅正向扩展，并在 2 台后明显趋于平台”，不是 QPS 随机器数下降，也不是 2×/3×/4×。
- ef 随切分从 24 降到 18/16/14，始终限制在 10～64；更小 shard 和更低 ef 确实使单 shard 搜索变快。
- 但 HashAll 的每个 query 必须在全部 M 个 shard 上各执行一次 HNSW。集群累计 CPU 成本从约 237 μs/query 增到 675 μs/query（约 2.85×），而 QPS 只增加约 1.20×，所以资源效率显著下降。
- coordinator 开销是次要但真实的：M=4 时 coordinator 平均约占 15.7 核，三个远端 shard 分别约占 13.7、13.2、12.9 核。主要限制仍是 broadcast-to-all 的工作放大，而不是 Orion fork 实现回归。

## 为什么旧曲线会下降

短并发扫描若固定 concurrency=4，M=1～4 的 QPS 为 {fixed_four}；更高 M 的请求等待更多 shard，固定在途请求数不足以压满集群。允许饱和并发后（M=1 选 4，M=2～4 选 8），容量曲线才变为 68.6k → 77.7k → 78.9k → 82.3k。旧实验还固定 ef=24，使高 M 的 Recall 从约 0.90 上升到约 0.96，也额外压低了高 M QPS。

## 口径与审计

- 数据集：SIFT1M，HNSW m=32、efConstruct=200、top-10。
- tuning query：0～999；held-out query：1000～9999。
- 原生 `sharding_method=auto`；客户端只向 coordinator 发送一次 `/points/search/batch`，无 shard selector、无客户端 fan-out/merge。
- batch=200；正式点 M=1～3 为 5×20 秒；M=4 用独立 7×30 秒确认，首轮与确认均值仅相差 {m4_relative_difference * 100:.2f}%。
- 审计状态：{audit['status']}。
"""
    (args.root / "SUMMARY_zh.md").write_text(summary_md, encoding="utf-8")
    print(json.dumps({"status": audit["status"], "rows": rows}, indent=2))
    return 0 if audit["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
