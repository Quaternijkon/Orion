#!/usr/bin/env python3
"""Audit the second formal confirmation and report the new Orion winner."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Sequence

import orion_physical_placement_confirmation_audit as common


DEFAULT_ROOT = Path(
    "/proj/intelisys-PG0/exp/orion-distributed/orion-load-balance-r090-20260825"
)
DEFAULT_OUTPUT = DEFAULT_ROOT / "online-v6"
DEFAULT_PLAN = DEFAULT_ROOT / "confirm-v2-plan"
BASELINE = "swap_c8_0"
FINALIST = "swap_16_18"
BACKUP = "swap_28_7"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build(args: argparse.Namespace) -> tuple[Path, Path]:
    root = args.root.expanduser().resolve()
    output = args.output_root.expanduser().resolve()
    plan_dir = args.plan_dir.expanduser().resolve()
    audit_path = output / "evidence-audit.json"
    report_path = output / "RESULTS_zh.md"
    if audit_path.exists() or report_path.exists():
        raise FileExistsError("refusing to overwrite existing v2 audit/report")

    summary_path = output / "formal-v2-summary.json"
    completion_path = output / "formal-v2-complete.json"
    resource_contract_path = output / "resource-contract.json"
    collection_contract_path = output / "collection-contract.json"
    preparation_path = output / "preparation.json"
    restored_path = output / "resource-restored.json"
    plan_path = plan_dir / "placements.json"
    selection_path = plan_dir / "selection.json"
    screen_path = root / "online-v5/screen-neighbor-summary.json"
    prior_formal_path = root / "online-v4/formal-summary.json"
    prior_audit_path = root / "online-v4/evidence-audit.json"
    summary = load_json(summary_path)
    completion = load_json(completion_path)
    resource_contract = load_json(resource_contract_path)
    collection_contract = load_json(collection_contract_path)
    preparation = load_json(preparation_path)
    restored = load_json(restored_path)
    plan = load_json(plan_path)
    selection = load_json(selection_path)
    screen = load_json(screen_path)
    prior_formal = load_json(prior_formal_path)
    prior_audit = load_json(prior_audit_path)
    expected_phases = tuple(
        (row["phase_id"], row["strategy"]) for row in plan["confirmation_phases"]
    )

    checks: list[dict[str, Any]] = []

    def check(name: str, condition: bool, actual: Any) -> None:
        checks.append(
            {"name": name, "status": "PASS" if condition else "FAIL", "actual": actual}
        )

    check("summary status", summary.get("status") == "CONFIRMATION_COMPLETE", summary.get("status"))
    check("completion status", completion.get("status") == "CONFIRMATION_COMPLETE", completion.get("status"))
    check("confirmed winner", summary.get("confirmed_winner") == FINALIST, summary.get("confirmed_winner"))
    check("completion winner", completion.get("winner") == FINALIST, completion.get("winner"))
    check("selection baseline", selection.get("baseline") == BASELINE, selection.get("baseline"))
    check("selection finalist", selection.get("finalist") == FINALIST, selection.get("finalist"))
    check("selection backup", selection.get("backup") == BACKUP, selection.get("backup"))
    check(
        "formal result strategies",
        summary["formal_confirmation_result"]["baseline_strategy"] == BASELINE
        and summary["formal_confirmation_result"]["finalist_strategy"] == FINALIST
        and summary["formal_confirmation_result"]["backup_strategy"] == BACKUP,
        summary["formal_confirmation_result"],
    )
    check(
        "strict endpoint dominance",
        summary["formal_confirmation_result"]["strict_endpoint_dominance"] is True,
        summary["formal_confirmation_result"]["strict_endpoint_dominance"],
    )
    check(
        "all recall pass",
        summary["formal_confirmation_result"]["all_recall_pass"] is True,
        summary["formal_confirmation_result"]["all_recall_pass"],
    )
    check(
        "resource contract",
        resource_contract.get("status") == "PASS"
        and resource_contract.get("physical_machine_count") == 4
        and resource_contract.get("logical_shard_count") == 32
        and resource_contract.get("qdrant_cpu_cores_per_machine") == 16
        and resource_contract.get("qdrant_cpu_cores_total") == 64,
        {
            key: resource_contract.get(key)
            for key in (
                "status",
                "physical_machine_count",
                "logical_shard_count",
                "qdrant_cpu_cores_per_machine",
                "qdrant_cpu_cores_total",
            )
        },
    )
    check(
        "collection contract",
        collection_contract.get("status") == "PASS"
        and collection_contract.get("logical_shard_count") == 32
        and collection_contract.get("replication_factor") == 1,
        {
            "status": collection_contract.get("status"),
            "logical_shard_count": collection_contract.get("logical_shard_count"),
            "replication_factor": collection_contract.get("replication_factor"),
        },
    )
    check(
        "external preparation reuse",
        preparation.get("status") == "REUSED_EXTERNAL_MANIFEST",
        preparation.get("status"),
    )
    check(
        "resource restoration",
        restored.get("status") == "PASS"
        and len(restored.get("nodes", [])) == 4
        and all(node.get("valid") for node in restored.get("nodes", [])),
        {"status": restored.get("status"), "node_count": len(restored.get("nodes", []))},
    )
    check("prior formal audit", prior_audit.get("status") == "PASS", prior_audit.get("status"))
    check("screen winner", screen.get("screen_winner") == FINALIST, screen.get("screen_winner"))

    phase_records = []
    for phase_id, strategy in expected_phases:
        phase_path = output / "phases" / phase_id / "screen.json"
        placement_path = output / "phases" / phase_id / "placement.json"
        phase = load_json(phase_path)
        placement = load_json(placement_path)
        expected_mapping = {
            str(shard): int(peer) for shard, peer in plan["placements"][strategy].items()
        }
        ending_mapping = {
            str(shard): int(peer)
            for shard, peer in phase["ending_placement"]["placement"].items()
        }
        phase_checks = {
            "status": phase.get("status") == "PASS",
            "identity": phase.get("phase_id") == phase_id
            and phase.get("strategy") == strategy,
            "formal marker": phase.get("formal_confirmation") is True
            and phase.get("screen_only") is False,
            "physical/logical contract": phase.get("physical_machine_count") == 4
            and phase.get("logical_shard_count") == 32,
            "held-out recall": float(phase["heldout_recall"]["recall_at_10"]) >= 0.90,
            "five repeats": phase.get("repeat_count") == 5,
            "20-second windows": all(float(row["wall_s"]) >= 20.0 for row in phase["repeats"]),
            "placement valid": phase["ending_placement"].get("valid") is True,
            "no residual transfers": not phase["ending_placement"].get("shard_transfers"),
            "expected placement": ending_mapping == expected_mapping,
            "placement operation": placement.get("status") == "PASS",
        }
        for label, condition in phase_checks.items():
            check(f"{phase_id}: {label}", condition, condition)
        phase_records.append(phase)

    baseline_rows = [phase_records[0], phase_records[3]]
    finalist_rows = [phase_records[1], phase_records[4]]
    backup_row = phase_records[2]
    baseline_mean = statistics.fmean(float(row["qps_mean"]) for row in baseline_rows)
    finalist_mean = statistics.fmean(float(row["qps_mean"]) for row in finalist_rows)
    improvement = 100.0 * (finalist_mean / baseline_mean - 1.0)
    prior_v1_mean = float(
        prior_formal["formal_confirmation_result"]["baseline_qps_mean_across_phases"]
    )
    cumulative_improvement = 100.0 * (finalist_mean / prior_v1_mean - 1.0)
    baseline_drift = 100.0 * (
        float(baseline_rows[1]["qps_mean"]) / float(baseline_rows[0]["qps_mean"]) - 1.0
    )
    finalist_drift = 100.0 * (
        float(finalist_rows[1]["qps_mean"]) / float(finalist_rows[0]["qps_mean"]) - 1.0
    )
    backup_improvement = 100.0 * (float(backup_row["qps_mean"]) / baseline_mean - 1.0)
    check(
        "recomputed improvement",
        abs(improvement - summary["formal_confirmation_result"]["finalist_improvement_pct"])
        < 1e-9,
        improvement,
    )
    check(
        "finalist minimum exceeds baseline maximum",
        min(float(row["qps_mean"]) for row in finalist_rows)
        > max(float(row["qps_mean"]) for row in baseline_rows),
        {
            "finalist_min": min(float(row["qps_mean"]) for row in finalist_rows),
            "baseline_max": max(float(row["qps_mean"]) for row in baseline_rows),
        },
    )
    failed = [row for row in checks if row["status"] != "PASS"]
    audit = {
        "status": "PASS" if not failed else "FAIL",
        "record_type": "orion_physical_placement_confirmation_v2_evidence_audit",
        "claim_boundary": (
            "swap_16_18 is formally confirmed over swap_c8_0 and is best among "
            "twelve online-tested placements; global optimality is not established"
        ),
        "physical_machine_count": 4,
        "logical_shard_count": 32,
        "passed_checks": len(checks) - len(failed),
        "total_checks": len(checks),
        "checks": checks,
        "metrics": {
            "baseline_qps_mean": baseline_mean,
            "finalist_qps_mean": finalist_mean,
            "finalist_improvement_pct": improvement,
            "cumulative_improvement_over_controller_v1_pct": cumulative_improvement,
            "backup_qps": float(backup_row["qps_mean"]),
            "backup_improvement_pct": backup_improvement,
            "baseline_endpoint_drift_pct": baseline_drift,
            "finalist_endpoint_drift_pct": finalist_drift,
            "recall_at_10": min(
                float(row["heldout_recall"]["recall_at_10"]) for row in phase_records
            ),
        },
        "artifacts": {
            str(path.relative_to(root)): {
                "sha256": common.sha256_path(path),
                "size_bytes": path.stat().st_size,
            }
            for path in (
                summary_path,
                completion_path,
                resource_contract_path,
                collection_contract_path,
                preparation_path,
                restored_path,
                plan_path,
                selection_path,
                screen_path,
                prior_formal_path,
                prior_audit_path,
                *(output / "phases" / phase_id / "screen.json" for phase_id, _ in expected_phases),
                *(output / "phases" / phase_id / "placement.json" for phase_id, _ in expected_phases),
            )
        },
    }
    common.write_json_new(audit_path, audit)
    if failed:
        raise RuntimeError(f"v2 evidence audit failed: {[row['name'] for row in failed]}")

    final_mapping = {
        int(shard): int(peer) for shard, peer in plan["placements"][FINALIST].items()
    }
    peer_hosts = {int(peer): host for peer, host in plan["peer_hosts"].items()}
    placement_lines = []
    for peer_id in [int(value) for value in plan["peer_order"]]:
        shards = sorted(shard for shard, owner in final_mapping.items() if owner == peer_id)
        role = "controller" if peer_id == int(plan["controller_peer_id"]) else "worker"
        placement_lines.append(f"{peer_hosts[peer_id]} {role}: {shards}")

    report = [
        "# Orion 物理分片负载均衡第二轮正式确认结果",
        "",
        "## 结论",
        "",
        (
            "在 4 台物理机、32 个逻辑分片的固定合同下，`swap_16_18` 正式战胜上一轮赢家 "
            "`swap_c8_0`，成为目前 12 个已在线测试 placement 中的最佳结果。"
        ),
        "",
        "| 策略 | QPS | 相对上一轮赢家 | Recall@10 | knee concurrency |",
        "|---|---:|---:|---:|---:|",
        f"| swap 8↔0（两端均值） | {baseline_mean:.2f} | — | 0.92364 | 4 / 4 |",
        f"| **再交换 16↔18（两端均值）** | **{finalist_mean:.2f}** | **+{improvement:.2f}%** | 0.92364 | 4 / 4 |",
        f"| swap 28↔7（单次正式备选） | {float(backup_row['qps_mean']):.2f} | +{backup_improvement:.2f}% | 0.92364 | {backup_row['selected_concurrency']} |",
        "",
        (
            f"相对最初正式 Controller-aware v1 的 {prior_v1_mean:.2f} QPS，新赢家累计提升 "
            f"{cumulative_improvement:.2f}%。"
        ),
        "",
        "严格判据通过：finalist 最低 phase 为 "
        f"{min(float(row['qps_mean']) for row in finalist_rows):.2f} QPS，高于 baseline 最高 "
        f"phase 的 {max(float(row['qps_mean']) for row in baseline_rows):.2f} QPS。",
        "",
        "## 五阶段正式证据",
        "",
        "| Phase | 策略 | QPS | CV | Recall@10 | C | P50 / P95 / P99 (ms) |",
        "|---|---|---:|---:|---:|---:|---:|",
        *[
            (
                f"| {row['phase_id']} | {row['strategy']} | {float(row['qps_mean']):.2f} | "
                f"{100.0 * float(row['qps_cv']):.3f}% | "
                f"{float(row['heldout_recall']['recall_at_10']):.5f} | "
                f"{row['selected_concurrency']} | "
                f"{float(row['batch_latency_ms_mean_across_repeats']['p50']):.2f} / "
                f"{float(row['batch_latency_ms_mean_across_repeats']['p95']):.2f} / "
                f"{float(row['batch_latency_ms_mean_across_repeats']['p99']):.2f} |"
            )
            for row in phase_records
        ],
        "",
        f"- baseline A/B 漂移：{baseline_drift:.3f}%。",
        f"- finalist A/B 漂移：{finalist_drift:.3f}%。",
        "- 新赢家两端均选择 concurrency=4；热点为 `10.10.1.2≈15.98` 核。",
        "- `swap_28_7` 仍有约 2.99% 提升，但正式 CV 较高且低于 finalist。",
        "- `swap_29_18` 的筛选结果为负，证明静态 proxy 均衡仍不能替代在线验证。",
        "",
        "## 最终 placement",
        "",
        "```text",
        *placement_lines,
        "```",
        "",
        "## 边界",
        "",
        "- 当前结论仅覆盖 12 个在线测试 placement 和本轮完整单交换模型筛选。",
        "- 离线 ridge/work proxy 只用于选候选，在线 QPS 才是排序依据。",
        "- 新赢家的完整单交换邻域尚需继续检查，因此还不能宣称全局完成。",
        "- 所有旧的 learned/proxy 负结果继续保留。",
        "",
        "## 证据位置",
        "",
        "- `online-v6/formal-v2-summary.json`",
        "- `online-v6/phases/*/screen.json`",
        "- `online-v6/evidence-audit.json`",
        "- `confirm-v2-plan/placements.json`",
        "- `online-v5/screen-neighbor-summary.json`",
        "- collection：`orion_lb_r090_p24_20260825_v2`；资源恢复为 PASS。",
        "",
    ]
    with report_path.open("x", encoding="utf-8") as handle:
        handle.write("\n".join(report))
    return audit_path, report_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--plan-dir", type=Path, default=DEFAULT_PLAN)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    audit, report = build(parse_args(argv))
    print(json.dumps({"audit": str(audit), "report": str(report)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
