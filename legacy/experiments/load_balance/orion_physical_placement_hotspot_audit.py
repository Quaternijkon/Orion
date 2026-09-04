#!/usr/bin/env python3
"""Audit the shard-24-preserving hotspot screen and failed confirmation."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Sequence

import orion_physical_placement_confirmation_audit as audit_common
import orion_physical_placement_worker_audit as worker_common


DEFAULT_ROOT = Path(
    "/proj/intelisys-PG0/exp/orion-distributed/orion-load-balance-r090-20260825"
)
BASELINE = "swap_24_21"
FINALIST = "hotspot_10_18"
BACKUP = "hotspot_14_12"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def inventory(root: Path) -> tuple[int, int]:
    files = list((root / "online-v1/phases").glob("*/benchmark.json"))
    for version in range(2, 15):
        files.extend((root / f"online-v{version}/phases").glob("*/screen.json"))
    mappings = {
        tuple(
            sorted(
                (int(shard), int(peer))
                for shard, peer in load_json(path)["ending_placement"]["placement"].items()
            )
        )
        for path in files
    }
    return len(files), len(mappings)


def build_screen(root: Path) -> tuple[Path, Path]:
    output = root / "online-v13"
    plan_dir = root / "refine-v13-hotspot"
    audit_path = output / "evidence-audit.json"
    report_path = output / "RESULTS_zh.md"
    if audit_path.exists() or report_path.exists():
        raise FileExistsError("refusing to overwrite existing online-v13 audit/report")
    summary_path = output / "screen-hotspot-summary.json"
    completion_path = output / "screen-hotspot-complete.json"
    plan_path = plan_dir / "placements.json"
    manifest_path = plan_dir / "manifest.json"
    metrics_path = plan_dir / "candidate-metrics.json"
    universe_path = plan_dir / "candidate-universe.json"
    rationale_path = plan_dir / "RATIONALE_zh.md"
    prior_audit_path = root / "online-v12/evidence-audit.json"
    summary = load_json(summary_path)
    completion = load_json(completion_path)
    plan = load_json(plan_path)
    manifest = load_json(manifest_path)
    prior_audit = load_json(prior_audit_path)
    phases = tuple((row["phase_id"], row["strategy"]) for row in plan["screen_phases"])
    checks: list[dict[str, Any]] = []
    add = lambda name, condition, actual: worker_common.add_check(checks, name, condition, actual)
    add("summary status", summary.get("status") == "SCREEN_COMPLETE", summary.get("status"))
    add("completion status", completion.get("status") == "SCREEN_COMPLETE", completion.get("status"))
    add("screen finalist", summary.get("screen_winner") == FINALIST, summary.get("screen_winner"))
    add("completion finalist", completion.get("winner") == FINALIST, completion.get("winner"))
    add("prior neighbor audit", prior_audit.get("status") == "PASS", prior_audit.get("status"))
    add(
        "hotspot plan contract",
        manifest.get("status") == "OFFLINE_REFINEMENT_ONLY"
        and manifest.get("eligible_candidate_count") == 9
        and manifest.get("selected_candidate_count") == 4,
        {
            "status": manifest.get("status"),
            "eligible": manifest.get("eligible_candidate_count"),
            "selected": manifest.get("selected_candidate_count"),
        },
    )
    baseline_shard_24 = int(plan["placements"][BASELINE]["24"])
    for strategy, mapping in plan["placements"].items():
        add(
            f"{strategy}: shard 24 preserved",
            int(mapping["24"]) == baseline_shard_24,
            int(mapping["24"]),
        )
    for name, record in manifest["outputs"].items():
        path = plan_dir / name
        add(
            f"offline output hash: {name}",
            path.is_file() and audit_common.sha256_path(path) == record["sha256"],
            audit_common.sha256_path(path) if path.is_file() else None,
        )
    worker_common.common_contract_checks(checks, output, plan)
    records = [
        worker_common.audit_phase(checks, output, plan, phase_id, strategy, formal=False)
        for phase_id, strategy in phases
    ]
    baseline_rows = [records[0], records[-1]]
    candidate_rows = records[1:-1]
    baseline_mean = statistics.fmean(float(row["qps_mean"]) for row in baseline_rows)
    baseline_max = max(float(row["qps_mean"]) for row in baseline_rows)
    by_strategy = {row["strategy"]: row for row in candidate_rows}
    improvement = 100.0 * (float(by_strategy[FINALIST]["qps_mean"]) / baseline_mean - 1.0)
    drift = 100.0 * (float(baseline_rows[1]["qps_mean"]) / float(baseline_rows[0]["qps_mean"]) - 1.0)
    add(
        "screen finalist exceeds both baseline endpoints",
        float(by_strategy[FINALIST]["qps_mean"]) > baseline_max,
        {"finalist": by_strategy[FINALIST]["qps_mean"], "baseline_max": baseline_max},
    )
    add(
        "other hotspot candidates lose to baseline mean",
        all(float(row["qps_mean"]) < baseline_mean for row in candidate_rows if row["strategy"] != FINALIST),
        {row["strategy"]: row["qps_mean"] for row in candidate_rows if row["strategy"] != FINALIST},
    )
    add("baseline endpoint drift below one percent", abs(drift) < 1.0, drift)
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
        universe_path,
        rationale_path,
        prior_audit_path,
        *(output / "phases" / phase_id / "screen.json" for phase_id, _ in phases),
        *(output / "phases" / phase_id / "placement.json" for phase_id, _ in phases),
    )
    audit = {
        "status": "PASS" if not failed else "FAIL",
        "record_type": "orion_physical_placement_hotspot_screen_evidence_audit",
        "claim_boundary": "hotspot_10_18 is a screen finalist only and requires formal confirmation",
        "physical_machine_count": 4,
        "logical_shard_count": 32,
        "passed_checks": len(checks) - len(failed),
        "total_checks": len(checks),
        "checks": checks,
        "metrics": {
            "baseline_qps_mean": baseline_mean,
            "baseline_endpoint_drift_pct": drift,
            "finalist_qps": float(by_strategy[FINALIST]["qps_mean"]),
            "finalist_screen_improvement_pct": improvement,
            "recall_at_10": min(float(row["heldout_recall"]["recall_at_10"]) for row in records),
        },
        "artifacts": {
            str(path.relative_to(root)): {"sha256": audit_common.sha256_path(path), "size_bytes": path.stat().st_size}
            for path in artifact_paths
        },
    }
    audit_common.write_json_new(audit_path, audit)
    if failed:
        raise RuntimeError(f"online-v13 evidence audit failed: {[row['name'] for row in failed]}")
    deltas = {
        row["strategy"]: 100.0 * (float(row["qps_mean"]) / baseline_mean - 1.0)
        for row in candidate_rows
    }
    report = [
        "# swap_24_21 热点卸载筛选结果",
        "",
        "在 4 台物理机、32 个逻辑分片、64 个 Qdrant 核的固定合同下，仅 `hotspot_10_18` 出现小幅正信号。",
        "",
        "| 策略 | QPS | 相对 baseline A/B | Recall@10 |",
        "|---|---:|---:|---:|",
        f"| `swap_24_21` A/B 均值 | {baseline_mean:.2f} | — | 0.92364 |",
        *[
            f"| `{row['strategy']}` | {float(row['qps_mean']):.2f} | {deltas[row['strategy']]:+.2f}% | {float(row['heldout_recall']['recall_at_10']):.5f} |"
            for row in candidate_rows
        ],
        "",
        f"`hotspot_10_18` 筛选提升 {improvement:.3f}%，超过 baseline 两端，因此进入正式确认。",
        f"baseline A/B 漂移 {drift:+.3f}%；审计 `{len(checks)}/{len(checks)} PASS`；资源恢复 PASS。",
        "",
    ]
    with report_path.open("x", encoding="utf-8") as handle:
        handle.write("\n".join(report))
    return audit_path, report_path


def build_confirmation(root: Path) -> tuple[Path, Path]:
    output = root / "online-v14"
    plan_dir = root / "confirm-v5-plan"
    audit_path = output / "evidence-audit.json"
    report_path = output / "RESULTS_zh.md"
    if audit_path.exists() or report_path.exists():
        raise FileExistsError("refusing to overwrite existing online-v14 audit/report")
    summary_path = output / "formal-v5-summary.json"
    completion_path = output / "formal-v5-complete.json"
    restore_path = output / "restore-current-winner.json"
    plan_path = plan_dir / "placements.json"
    selection_path = plan_dir / "selection.json"
    manifest_path = plan_dir / "manifest.json"
    screen_audit_path = root / "online-v13/evidence-audit.json"
    prior_summary_path = root / "online-v11/formal-v4-summary.json"
    prior_audit_path = root / "online-v11/evidence-audit.json"
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
    add = lambda name, condition, actual: worker_common.add_check(checks, name, condition, actual)
    add("summary status", summary.get("status") == "CONFIRMATION_COMPLETE", summary.get("status"))
    add("completion status", completion.get("status") == "CONFIRMATION_COMPLETE", completion.get("status"))
    add("challenger not confirmed", summary.get("confirmed_winner") is None, summary.get("confirmed_winner"))
    add("completion winner null", completion.get("winner") is None, completion.get("winner"))
    add(
        "selection identities",
        selection.get("baseline") == BASELINE
        and selection.get("finalist") == FINALIST
        and selection.get("backup") == BACKUP,
        {key: selection.get(key) for key in ("baseline", "finalist", "backup")},
    )
    result = summary["formal_confirmation_result"]
    add("all recall pass", result.get("all_recall_pass") is True, result.get("all_recall_pass"))
    add("strict endpoint dominance rejected", result.get("strict_endpoint_dominance") is False, result.get("strict_endpoint_dominance"))
    add("formal confirmation rejected", result.get("confirmed_over_baseline") is False, result.get("confirmed_over_baseline"))
    add("screen audit", screen_audit.get("status") == "PASS", screen_audit.get("status"))
    add("prior formal audit", prior_audit.get("status") == "PASS", prior_audit.get("status"))
    add("prior formal winner retained", prior_summary.get("confirmed_winner") == BASELINE, prior_summary.get("confirmed_winner"))
    worker_common.common_contract_checks(checks, output, plan)
    records = [
        worker_common.audit_phase(checks, output, plan, phase_id, strategy, formal=True)
        for phase_id, strategy in phases
    ]
    baseline_rows = [records[0], records[3]]
    finalist_rows = [records[1], records[4]]
    backup = records[2]
    baseline_mean = statistics.fmean(float(row["qps_mean"]) for row in baseline_rows)
    finalist_mean = statistics.fmean(float(row["qps_mean"]) for row in finalist_rows)
    improvement = 100.0 * (finalist_mean / baseline_mean - 1.0)
    baseline_drift = 100.0 * (float(baseline_rows[1]["qps_mean"]) / float(baseline_rows[0]["qps_mean"]) - 1.0)
    finalist_drift = 100.0 * (float(finalist_rows[1]["qps_mean"]) / float(finalist_rows[0]["qps_mean"]) - 1.0)
    add("formal improvement recomputed", abs(improvement - float(result["finalist_improvement_pct"])) < 1e-9, improvement)
    add(
        "strict failure recomputed",
        min(float(row["qps_mean"]) for row in finalist_rows)
        <= max(float(row["qps_mean"]) for row in baseline_rows),
        {
            "finalist_min": min(float(row["qps_mean"]) for row in finalist_rows),
            "baseline_max": max(float(row["qps_mean"]) for row in baseline_rows),
        },
    )
    expected = {str(shard): int(peer) for shard, peer in plan["placements"][BASELINE].items()}
    ending = {
        str(shard): int(peer)
        for shard, peer in restore["ending_placement"]["placement"].items()
    }
    add(
        "live placement restored to formal winner",
        restore.get("status") == "PASS"
        and restore.get("strategy") == BASELINE
        and restore["ending_placement"].get("valid") is True
        and not restore["ending_placement"].get("shard_transfers")
        and ending == expected,
        {"status": restore.get("status"), "strategy": restore.get("strategy")},
    )
    phase_count, placement_count = inventory(root)
    add("online placement inventory", phase_count == 75 and placement_count == 32, {"phase_count": phase_count, "unique_placement_count": placement_count})
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
        "record_type": "orion_physical_placement_hotspot_confirmation_evidence_audit",
        "claim_boundary": (
            "hotspot_10_18 is not confirmed; swap_24_21 remains best among 32 online-tested "
            "placements and global optimality is not established"
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
            str(path.relative_to(root)): {"sha256": audit_common.sha256_path(path), "size_bytes": path.stat().st_size}
            for path in artifact_paths
        },
    }
    audit_common.write_json_new(audit_path, audit)
    if failed:
        raise RuntimeError(f"online-v14 evidence audit failed: {[row['name'] for row in failed]}")
    report = [
        "# hotspot_10_18 正式确认结果",
        "",
        "在 4 台物理机、32 个逻辑分片、64 个 Qdrant 核的固定合同下，`hotspot_10_18` 未通过严格端点确认，正式赢家仍为 `swap_24_21`。",
        "",
        "| 策略 | 正式 QPS | 相对当前赢家 | Recall@10 |",
        "|---|---:|---:|---:|",
        f"| **`swap_24_21` A/B 均值** | **{baseline_mean:.2f}** | — | 0.92364 |",
        f"| `hotspot_10_18` A/B 均值 | {finalist_mean:.2f} | {improvement:+.2f}% | 0.92364 |",
        f"| `hotspot_14_12` backup | {float(backup['qps_mean']):.2f} | {100.0 * (float(backup['qps_mean']) / baseline_mean - 1.0):+.2f}% | 0.92364 |",
        "",
        (
            f"端点判据失败：finalist 最低 {min(float(row['qps_mean']) for row in finalist_rows):.2f} "
            f"≤ baseline 最高 {max(float(row['qps_mean']) for row in baseline_rows):.2f} QPS。"
        ),
        f"虽然 finalist 均值仅高 {improvement:.3f}%，但 A/B 漂移 {finalist_drift:+.3f}% 且不能端点支配，因此不更新赢家。",
        f"审计 `{len(checks)}/{len(checks)} PASS`；live placement 已恢复为 `swap_24_21`，无 shard transfer。",
        "",
        f"当前累计 {placement_count} 个唯一在线 physical placement、{phase_count} 个 phase。",
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
    formal_audit, formal_report = build_confirmation(root)
    print(
        json.dumps(
            {
                "screen_audit": str(screen_audit),
                "screen_report": str(screen_report),
                "formal_audit": str(formal_audit),
                "formal_report": str(formal_report),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
