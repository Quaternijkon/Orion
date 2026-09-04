#!/usr/bin/env python3
"""Audit the worker-only screen and its failed formal challenger confirmation."""

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
BASELINE = "swap_16_18"
FINALIST = "swap_16_17"
BACKUP = "swap_16_10"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def add_check(checks: list[dict[str, Any]], name: str, condition: bool, actual: Any) -> None:
    checks.append({"name": name, "status": "PASS" if condition else "FAIL", "actual": actual})


def audit_phase(
    checks: list[dict[str, Any]],
    output: Path,
    plan: dict[str, Any],
    phase_id: str,
    strategy: str,
    *,
    formal: bool,
) -> dict[str, Any]:
    phase = load_json(output / "phases" / phase_id / "screen.json")
    placement = load_json(output / "phases" / phase_id / "placement.json")
    expected = {str(shard): int(peer) for shard, peer in plan["placements"][strategy].items()}
    ending = {
        str(shard): int(peer)
        for shard, peer in phase["ending_placement"]["placement"].items()
    }
    expected_repeats = 5 if formal else 3
    expected_window = 20.0 if formal else 10.0
    phase_checks = {
        "status": phase.get("status") == "PASS",
        "identity": phase.get("phase_id") == phase_id and phase.get("strategy") == strategy,
        "mode marker": phase.get("formal_confirmation") is formal
        and phase.get("screen_only") is (not formal),
        "physical/logical contract": phase.get("physical_machine_count") == 4
        and phase.get("logical_shard_count") == 32,
        "held-out recall": float(phase["heldout_recall"]["recall_at_10"]) >= 0.90,
        "repeat count": phase.get("repeat_count") == expected_repeats,
        "measurement windows": all(
            float(row["wall_s"]) >= expected_window for row in phase["repeats"]
        ),
        "saturation knee": phase["saturation_selection"].get("knee_observed") is True,
        "placement valid": phase["ending_placement"].get("valid") is True,
        "no residual transfers": not phase["ending_placement"].get("shard_transfers"),
        "expected placement": ending == expected,
        "placement operation": placement.get("status") == "PASS",
    }
    for label, condition in phase_checks.items():
        add_check(checks, f"{phase_id}: {label}", condition, condition)
    return phase


def common_contract_checks(
    checks: list[dict[str, Any]], output: Path, plan: dict[str, Any]
) -> tuple[dict[str, Any], ...]:
    resource = load_json(output / "resource-contract.json")
    collection = load_json(output / "collection-contract.json")
    preparation = load_json(output / "preparation.json")
    restored = load_json(output / "resource-restored.json")
    add_check(
        checks,
        "plan physical/logical contract",
        plan.get("physical_machine_count") == 4
        and plan.get("logical_shard_count") == 32
        and len(plan.get("peer_order", [])) == 4,
        {
            "physical_machine_count": plan.get("physical_machine_count"),
            "logical_shard_count": plan.get("logical_shard_count"),
            "peer_count": len(plan.get("peer_order", [])),
        },
    )
    add_check(
        checks,
        "resource contract",
        resource.get("status") == "PASS"
        and resource.get("physical_machine_count") == 4
        and resource.get("logical_shard_count") == 32
        and resource.get("qdrant_cpu_cores_per_machine") == 16
        and resource.get("qdrant_cpu_cores_total") == 64,
        {
            key: resource.get(key)
            for key in (
                "status",
                "physical_machine_count",
                "logical_shard_count",
                "qdrant_cpu_cores_per_machine",
                "qdrant_cpu_cores_total",
            )
        },
    )
    add_check(
        checks,
        "collection contract",
        collection.get("status") == "PASS"
        and collection.get("logical_shard_count") == 32
        and collection.get("replication_factor") == 1,
        {
            "status": collection.get("status"),
            "logical_shard_count": collection.get("logical_shard_count"),
            "replication_factor": collection.get("replication_factor"),
        },
    )
    add_check(
        checks,
        "external preparation reuse",
        preparation.get("status") == "REUSED_EXTERNAL_MANIFEST",
        preparation.get("status"),
    )
    add_check(
        checks,
        "resource restoration",
        restored.get("status") == "PASS"
        and len(restored.get("nodes", [])) == 4
        and all(node.get("valid") for node in restored.get("nodes", [])),
        {"status": restored.get("status"), "node_count": len(restored.get("nodes", []))},
    )
    return resource, collection, preparation, restored


def build_screen(root: Path) -> tuple[Path, Path]:
    output = root / "online-v8"
    plan_dir = root / "refine-v8-worker"
    audit_path = output / "evidence-audit.json"
    report_path = output / "RESULTS_zh.md"
    if audit_path.exists() or report_path.exists():
        raise FileExistsError("refusing to overwrite existing online-v8 audit/report")
    summary_path = output / "screen-worker-summary.json"
    completion_path = output / "screen-worker-complete.json"
    plan_path = plan_dir / "placements.json"
    manifest_path = plan_dir / "manifest.json"
    metrics_path = plan_dir / "candidate-metrics.json"
    rationale_path = plan_dir / "RATIONALE_zh.md"
    prior_audit_path = root / "online-v7/evidence-audit.json"
    summary = load_json(summary_path)
    completion = load_json(completion_path)
    plan = load_json(plan_path)
    manifest = load_json(manifest_path)
    metrics = load_json(metrics_path)
    prior_audit = load_json(prior_audit_path)
    phases = tuple((row["phase_id"], row["strategy"]) for row in plan["screen_phases"])
    checks: list[dict[str, Any]] = []
    add_check(checks, "summary status", summary.get("status") == "SCREEN_COMPLETE", summary.get("status"))
    add_check(checks, "completion status", completion.get("status") == "SCREEN_COMPLETE", completion.get("status"))
    add_check(checks, "screen finalist", summary.get("screen_winner") == FINALIST, summary.get("screen_winner"))
    add_check(checks, "completion finalist", completion.get("winner") == FINALIST, completion.get("winner"))
    add_check(checks, "prior negative audit", prior_audit.get("status") == "PASS", prior_audit.get("status"))
    add_check(
        checks,
        "worker-only plan manifest",
        manifest.get("status") == "OFFLINE_REFINEMENT_ONLY"
        and manifest.get("candidate_count") == 4
        and manifest.get("controller_placement_unchanged") is True,
        {
            "status": manifest.get("status"),
            "candidate_count": manifest.get("candidate_count"),
            "controller_placement_unchanged": manifest.get("controller_placement_unchanged"),
        },
    )
    baseline_controller = sorted(
        shard
        for shard, peer in plan["placements"][BASELINE].items()
        if int(peer) == int(plan["controller_peer_id"])
    )
    for strategy, mapping in plan["placements"].items():
        controller = sorted(
            shard for shard, peer in mapping.items() if int(peer) == int(plan["controller_peer_id"])
        )
        counts = {int(peer): 0 for peer in plan["peer_order"]}
        for peer in mapping.values():
            counts[int(peer)] += 1
        add_check(
            checks,
            f"{strategy}: controller unchanged and 8 shards per peer",
            controller == baseline_controller and set(counts.values()) == {8},
            {"controller_shards": controller, "counts": counts},
        )
    for strategy in ("swap_16_10", "swap_25_10", "swap_16_17", "swap_25_17"):
        add_check(
            checks,
            f"{strategy}: positive work shift from hotspot",
            float(metrics[strategy]["work_units_shifted_from_hot_worker"]) > 0
            and metrics[strategy].get("controller_placement_unchanged") is True,
            metrics[strategy]["work_units_shifted_from_hot_worker"],
        )
    common_contract_checks(checks, output, plan)
    records = [audit_phase(checks, output, plan, phase_id, strategy, formal=False) for phase_id, strategy in phases]
    baseline_rows = [records[0], records[-1]]
    candidate_rows = records[1:-1]
    baseline_mean = statistics.fmean(float(row["qps_mean"]) for row in baseline_rows)
    baseline_min = min(float(row["qps_mean"]) for row in baseline_rows)
    baseline_max = max(float(row["qps_mean"]) for row in baseline_rows)
    finalist = next(row for row in candidate_rows if row["strategy"] == FINALIST)
    improvement = 100.0 * (float(finalist["qps_mean"]) / baseline_mean - 1.0)
    drift = 100.0 * (float(baseline_rows[1]["qps_mean"]) / float(baseline_rows[0]["qps_mean"]) - 1.0)
    add_check(checks, "screen improvement recomputed", abs(improvement - 0.07721148982300363) < 1e-9, improvement)
    add_check(
        checks,
        "screen signal is not endpoint dominant",
        float(finalist["qps_mean"]) <= baseline_max,
        {"finalist": finalist["qps_mean"], "baseline_max": baseline_max},
    )
    losers = [row for row in candidate_rows if row["strategy"] != FINALIST]
    add_check(
        checks,
        "other worker-only candidates lose to baseline minimum",
        all(float(row["qps_mean"]) < baseline_min for row in losers),
        {row["strategy"]: row["qps_mean"] for row in losers},
    )
    failed = [row for row in checks if row["status"] != "PASS"]
    artifact_paths = (
        summary_path,
        completion_path,
        output / "resource-contract.json",
        output / "collection-contract.json",
        output / "preparation.json",
        output / "resource-restored.json",
        plan_path,
        manifest_path,
        metrics_path,
        rationale_path,
        prior_audit_path,
        *(output / "phases" / phase_id / "screen.json" for phase_id, _ in phases),
        *(output / "phases" / phase_id / "placement.json" for phase_id, _ in phases),
    )
    audit = {
        "status": "PASS" if not failed else "FAIL",
        "record_type": "orion_physical_placement_worker_screen_evidence_audit",
        "claim_boundary": "swap_16_17 is a weak screen finalist only and requires formal confirmation",
        "physical_machine_count": 4,
        "logical_shard_count": 32,
        "passed_checks": len(checks) - len(failed),
        "total_checks": len(checks),
        "checks": checks,
        "metrics": {
            "baseline_qps_mean": baseline_mean,
            "baseline_endpoint_drift_pct": drift,
            "finalist_qps": float(finalist["qps_mean"]),
            "finalist_screen_improvement_pct": improvement,
            "recall_at_10": min(float(row["heldout_recall"]["recall_at_10"]) for row in records),
        },
        "artifacts": {
            str(path.relative_to(root)): {"sha256": common.sha256_path(path), "size_bytes": path.stat().st_size}
            for path in artifact_paths
        },
    }
    common.write_json_new(audit_path, audit)
    if failed:
        raise RuntimeError(f"online-v8 evidence audit failed: {[row['name'] for row in failed]}")
    deltas = {
        row["strategy"]: 100.0 * (float(row["qps_mean"]) / baseline_mean - 1.0)
        for row in candidate_rows
    }
    report = [
        "# Orion worker-only 微调在线筛选结果",
        "",
        "在 4 台物理机、32 个逻辑分片、总计 64 个 Qdrant 核的固定合同下，四个 worker-only 候选仅 `swap_16_17` 出现极弱正信号。",
        "",
        "| 策略 | QPS | 相对本轮 baseline A/B | Recall@10 |",
        "|---|---:|---:|---:|",
        f"| `swap_16_18` A/B 均值 | {baseline_mean:.2f} | — | 0.92364 |",
        *[
            f"| `{row['strategy']}` | {float(row['qps_mean']):.2f} | {deltas[row['strategy']]:+.2f}% | {float(row['heldout_recall']['recall_at_10']):.5f} |"
            for row in candidate_rows
        ],
        "",
        f"`swap_16_17` 仅提升 {improvement:.3f}%，且低于 baseline B 端点，因此只能进入正式确认，不能视为新赢家。",
        f"baseline A/B 漂移 {drift:+.3f}%；审计 `{len(checks)}/{len(checks)} PASS`；资源恢复 PASS。",
        "",
        "边界：controller ownership 完全不变；离线 work shift 只用于提名，在线 QPS 才用于排序。",
        "",
    ]
    with report_path.open("x", encoding="utf-8") as handle:
        handle.write("\n".join(report))
    return audit_path, report_path


def unique_online_placements(root: Path) -> tuple[int, int]:
    files = list((root / "online-v1/phases").glob("*/benchmark.json"))
    for version in range(2, 10):
        files.extend((root / f"online-v{version}/phases").glob("*/screen.json"))
    mappings = set()
    for path in files:
        payload = load_json(path)
        mappings.add(
            tuple(
                sorted(
                    (int(shard), int(peer))
                    for shard, peer in payload["ending_placement"]["placement"].items()
                )
            )
        )
    return len(files), len(mappings)


def build_confirmation(root: Path) -> tuple[Path, Path]:
    output = root / "online-v9"
    plan_dir = root / "confirm-v3-plan"
    audit_path = output / "evidence-audit.json"
    report_path = output / "RESULTS_zh.md"
    if audit_path.exists() or report_path.exists():
        raise FileExistsError("refusing to overwrite existing online-v9 audit/report")
    summary_path = output / "formal-v3-summary.json"
    completion_path = output / "formal-v3-complete.json"
    restore_path = output / "restore-current-winner.json"
    plan_path = plan_dir / "placements.json"
    selection_path = plan_dir / "selection.json"
    manifest_path = plan_dir / "manifest.json"
    screen_audit_path = root / "online-v8/evidence-audit.json"
    prior_summary_path = root / "online-v6/formal-v2-summary.json"
    prior_audit_path = root / "online-v6/evidence-audit.json"
    summary = load_json(summary_path)
    completion = load_json(completion_path)
    restore = load_json(restore_path)
    plan = load_json(plan_path)
    selection = load_json(selection_path)
    screen_audit = load_json(screen_audit_path)
    prior_summary = load_json(prior_summary_path)
    prior_audit = load_json(prior_audit_path)
    phases = tuple((row["phase_id"], row["strategy"]) for row in plan["confirmation_phases"])
    checks: list[dict[str, Any]] = []
    add_check(checks, "summary status", summary.get("status") == "CONFIRMATION_COMPLETE", summary.get("status"))
    add_check(checks, "completion status", completion.get("status") == "CONFIRMATION_COMPLETE", completion.get("status"))
    add_check(checks, "challenger not confirmed", summary.get("confirmed_winner") is None, summary.get("confirmed_winner"))
    add_check(checks, "completion winner null", completion.get("winner") is None, completion.get("winner"))
    add_check(
        checks,
        "selection identities",
        selection.get("baseline") == BASELINE
        and selection.get("finalist") == FINALIST
        and selection.get("backup") == BACKUP,
        {key: selection.get(key) for key in ("baseline", "finalist", "backup")},
    )
    formal_result = summary["formal_confirmation_result"]
    add_check(checks, "all recall pass", formal_result.get("all_recall_pass") is True, formal_result.get("all_recall_pass"))
    add_check(checks, "strict endpoint dominance rejected", formal_result.get("strict_endpoint_dominance") is False, formal_result.get("strict_endpoint_dominance"))
    add_check(checks, "formal confirmation rejected", formal_result.get("confirmed_over_baseline") is False, formal_result.get("confirmed_over_baseline"))
    add_check(checks, "screen audit", screen_audit.get("status") == "PASS", screen_audit.get("status"))
    add_check(checks, "prior formal audit", prior_audit.get("status") == "PASS", prior_audit.get("status"))
    add_check(checks, "prior formal winner retained", prior_summary.get("confirmed_winner") == BASELINE, prior_summary.get("confirmed_winner"))
    common_contract_checks(checks, output, plan)
    records = [audit_phase(checks, output, plan, phase_id, strategy, formal=True) for phase_id, strategy in phases]
    baseline_rows = [records[0], records[3]]
    finalist_rows = [records[1], records[4]]
    backup = records[2]
    baseline_mean = statistics.fmean(float(row["qps_mean"]) for row in baseline_rows)
    finalist_mean = statistics.fmean(float(row["qps_mean"]) for row in finalist_rows)
    improvement = 100.0 * (finalist_mean / baseline_mean - 1.0)
    baseline_drift = 100.0 * (float(baseline_rows[1]["qps_mean"]) / float(baseline_rows[0]["qps_mean"]) - 1.0)
    finalist_drift = 100.0 * (float(finalist_rows[1]["qps_mean"]) / float(finalist_rows[0]["qps_mean"]) - 1.0)
    add_check(checks, "formal improvement recomputed", abs(improvement - float(formal_result["finalist_improvement_pct"])) < 1e-9, improvement)
    add_check(
        checks,
        "baseline mean exceeds finalist mean",
        baseline_mean > finalist_mean,
        {"baseline_mean": baseline_mean, "finalist_mean": finalist_mean},
    )
    expected_restore = {str(shard): int(peer) for shard, peer in plan["placements"][BASELINE].items()}
    restored_mapping = {
        str(shard): int(peer)
        for shard, peer in restore["ending_placement"]["placement"].items()
    }
    add_check(
        checks,
        "live placement restored to current winner",
        restore.get("status") == "PASS"
        and restore.get("strategy") == BASELINE
        and restore["ending_placement"].get("valid") is True
        and not restore["ending_placement"].get("shard_transfers")
        and restored_mapping == expected_restore,
        {"status": restore.get("status"), "strategy": restore.get("strategy")},
    )
    phase_count, placement_count = unique_online_placements(root)
    add_check(checks, "online placement inventory", phase_count == 47 and placement_count == 20, {"phase_count": phase_count, "unique_placement_count": placement_count})
    failed = [row for row in checks if row["status"] != "PASS"]
    artifact_paths = (
        summary_path,
        completion_path,
        restore_path,
        output / "resource-contract.json",
        output / "collection-contract.json",
        output / "preparation.json",
        output / "resource-restored.json",
        plan_path,
        selection_path,
        manifest_path,
        screen_audit_path,
        prior_summary_path,
        prior_audit_path,
        *(output / "phases" / phase_id / "screen.json" for phase_id, _ in phases),
        *(output / "phases" / phase_id / "placement.json" for phase_id, _ in phases),
    )
    audit = {
        "status": "PASS" if not failed else "FAIL",
        "record_type": "orion_physical_placement_worker_confirmation_evidence_audit",
        "claim_boundary": (
            "swap_16_18 remains the formal winner and best among 20 online-tested placements; "
            "single-swap local and global optimality are not established"
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
            "backup_qps": float(backup["qps_mean"]),
            "baseline_endpoint_drift_pct": baseline_drift,
            "finalist_endpoint_drift_pct": finalist_drift,
            "recall_at_10": min(float(row["heldout_recall"]["recall_at_10"]) for row in records),
            "online_phase_count": phase_count,
            "unique_online_placement_count": placement_count,
        },
        "artifacts": {
            str(path.relative_to(root)): {"sha256": common.sha256_path(path), "size_bytes": path.stat().st_size}
            for path in artifact_paths
        },
    }
    common.write_json_new(audit_path, audit)
    if failed:
        raise RuntimeError(f"online-v9 evidence audit failed: {[row['name'] for row in failed]}")
    report = [
        "# Orion worker-only 候选正式确认结果",
        "",
        "## 结论",
        "",
        (
            "在 4 台物理机、32 个逻辑分片、每台 Qdrant 16 核（总计 64 核）的固定合同下，"
            "`swap_16_17` 未通过正式确认；`swap_16_18` 继续保持正式赢家。"
        ),
        "",
        "| 策略 | 正式 QPS | 相对当前赢家 | Recall@10 |",
        "|---|---:|---:|---:|",
        f"| **`swap_16_18` A/B 均值** | **{baseline_mean:.2f}** | — | 0.92364 |",
        f"| `swap_16_17` A/B 均值 | {finalist_mean:.2f} | {improvement:+.2f}% | 0.92364 |",
        f"| `swap_16_10` 单次 backup | {float(backup['qps_mean']):.2f} | {100.0 * (float(backup['qps_mean']) / baseline_mean - 1.0):+.2f}% | 0.92364 |",
        "",
        f"baseline A/B 漂移 {baseline_drift:+.3f}%，finalist A/B 漂移 {finalist_drift:+.3f}%。",
        "严格端点支配为 false，且 finalist 均值低于 baseline 约 0.19%。",
        f"审计 `{len(checks)}/{len(checks)} PASS`；资源恢复 PASS；live placement 已恢复为 `swap_16_18`，无 shard transfer。",
        "",
        "## 当前证据边界",
        "",
        f"- 累计 {phase_count} 个在线 phase、{placement_count} 个唯一 physical placement。",
        "- `swap_16_18` 是这些已测试 placement 中的正式最佳结果，不是单交换局部最优或全局最优证明。",
        "- 完整 384 邻域模型前四名和四个 CPU-directed worker-only 候选均保留其负结果。",
        "- Recall@10 全部为 0.92364；离线模型与 work proxy 不替代在线 QPS。",
        "",
    ]
    with report_path.open("x", encoding="utf-8") as handle:
        handle.write("\n".join(report))
    return audit_path, report_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    root = parse_args(argv).root.expanduser().resolve()
    screen_audit, screen_report = build_screen(root)
    confirmation_audit, confirmation_report = build_confirmation(root)
    print(
        json.dumps(
            {
                "screen_audit": str(screen_audit),
                "screen_report": str(screen_report),
                "confirmation_audit": str(confirmation_audit),
                "confirmation_report": str(confirmation_report),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
