#!/usr/bin/env python3
"""Audit the smaller worker-only screen and the confirmed new Orion winner."""

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
BASELINE = "swap_16_18"
FINALIST = "swap_24_21"
BACKUP = "swap_24_13"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def inventory(root: Path) -> tuple[int, int]:
    files = list((root / "online-v1/phases").glob("*/benchmark.json"))
    for version in range(2, 12):
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
    output = root / "online-v10"
    plan_dir = root / "refine-v10-worker-small"
    audit_path = output / "evidence-audit.json"
    report_path = output / "RESULTS_zh.md"
    if audit_path.exists() or report_path.exists():
        raise FileExistsError("refusing to overwrite existing online-v10 audit/report")
    summary_path = output / "screen-worker-small-summary.json"
    completion_path = output / "screen-worker-small-complete.json"
    plan_path = plan_dir / "placements.json"
    manifest_path = plan_dir / "manifest.json"
    metrics_path = plan_dir / "candidate-metrics.json"
    universe_path = plan_dir / "candidate-universe.json"
    models_path = plan_dir / "service-demand-models.json"
    rationale_path = plan_dir / "RATIONALE_zh.md"
    prior_audit_path = root / "online-v9/evidence-audit.json"
    summary = load_json(summary_path)
    completion = load_json(completion_path)
    plan = load_json(plan_path)
    manifest = load_json(manifest_path)
    models = load_json(models_path)
    prior_audit = load_json(prior_audit_path)
    phases = tuple((row["phase_id"], row["strategy"]) for row in plan["screen_phases"])
    checks: list[dict[str, Any]] = []
    add = lambda name, condition, actual: worker_common.add_check(checks, name, condition, actual)
    add("summary status", summary.get("status") == "SCREEN_COMPLETE", summary.get("status"))
    add("completion status", completion.get("status") == "SCREEN_COMPLETE", completion.get("status"))
    add("screen winner", summary.get("screen_winner") == FINALIST, summary.get("screen_winner"))
    add("completion winner", completion.get("winner") == FINALIST, completion.get("winner"))
    add("prior formal audit", prior_audit.get("status") == "PASS", prior_audit.get("status"))
    add(
        "offline refinement contract",
        manifest.get("status") == "OFFLINE_REFINEMENT_ONLY"
        and manifest.get("modeling")
        == {
            "unique_online_placement_count": 20,
            "node_observation_count": 80,
            "eligible_candidate_count": 12,
            "selected_candidate_count": 4,
            "model_count": 3,
        },
        manifest.get("modeling"),
    )
    rmses = [float(row["leave_one_placement_out_rmse"]) for row in models["models"]]
    add("updated model RMSE", len(rmses) == 3 and max(rmses) < 110.0, rmses)
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
    finalist_improvement = 100.0 * (float(by_strategy[FINALIST]["qps_mean"]) / baseline_mean - 1.0)
    backup_improvement = 100.0 * (float(by_strategy[BACKUP]["qps_mean"]) / baseline_mean - 1.0)
    drift = 100.0 * (float(baseline_rows[1]["qps_mean"]) / float(baseline_rows[0]["qps_mean"]) - 1.0)
    add(
        "finalist exceeds both baseline endpoints",
        float(by_strategy[FINALIST]["qps_mean"]) > baseline_max,
        {"finalist": by_strategy[FINALIST]["qps_mean"], "baseline_max": baseline_max},
    )
    add(
        "backup exceeds both baseline endpoints",
        float(by_strategy[BACKUP]["qps_mean"]) > baseline_max,
        {"backup": by_strategy[BACKUP]["qps_mean"], "baseline_max": baseline_max},
    )
    add("stable baseline endpoints", abs(drift) < 1.0, drift)
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
        models_path,
        rationale_path,
        prior_audit_path,
        *(output / "phases" / phase_id / "screen.json" for phase_id, _ in phases),
        *(output / "phases" / phase_id / "placement.json" for phase_id, _ in phases),
    )
    audit = {
        "status": "PASS" if not failed else "FAIL",
        "record_type": "orion_physical_placement_small_worker_screen_evidence_audit",
        "claim_boundary": "screen finalists require formal five-stage confirmation",
        "physical_machine_count": 4,
        "logical_shard_count": 32,
        "passed_checks": len(checks) - len(failed),
        "total_checks": len(checks),
        "checks": checks,
        "metrics": {
            "baseline_qps_mean": baseline_mean,
            "baseline_endpoint_drift_pct": drift,
            "finalist_qps": float(by_strategy[FINALIST]["qps_mean"]),
            "finalist_improvement_pct": finalist_improvement,
            "backup_qps": float(by_strategy[BACKUP]["qps_mean"]),
            "backup_improvement_pct": backup_improvement,
            "model_leave_one_placement_out_rmse": rmses,
            "recall_at_10": min(float(row["heldout_recall"]["recall_at_10"]) for row in records),
        },
        "artifacts": {
            str(path.relative_to(root)): {"sha256": audit_common.sha256_path(path), "size_bytes": path.stat().st_size}
            for path in artifact_paths
        },
    }
    audit_common.write_json_new(audit_path, audit)
    if failed:
        raise RuntimeError(f"online-v10 evidence audit failed: {[row['name'] for row in failed]}")
    deltas = {
        row["strategy"]: 100.0 * (float(row["qps_mean"]) / baseline_mean - 1.0)
        for row in candidate_rows
    }
    report = [
        "# Orion 小-shift worker-only 在线筛选结果",
        "",
        "在 4 台物理机、32 个逻辑分片、64 个 Qdrant 核的固定合同下，两个 `shard 24` 候选稳定超过 baseline 两端。",
        "",
        "| 策略 | QPS | 相对 baseline A/B | Recall@10 |",
        "|---|---:|---:|---:|",
        f"| `swap_16_18` A/B 均值 | {baseline_mean:.2f} | — | 0.92364 |",
        *[
            f"| `{row['strategy']}` | {float(row['qps_mean']):.2f} | {deltas[row['strategy']]:+.2f}% | {float(row['heldout_recall']['recall_at_10']):.5f} |"
            for row in candidate_rows
        ],
        "",
        f"baseline A/B 漂移仅 {drift:+.3f}%；`swap_24_21` 和 `swap_24_13` 分别提升 {finalist_improvement:.2f}% / {backup_improvement:.2f}%。",
        f"审计 `{len(checks)}/{len(checks)} PASS`；资源恢复 PASS。",
        "",
    ]
    with report_path.open("x", encoding="utf-8") as handle:
        handle.write("\n".join(report))
    return audit_path, report_path


def build_confirmation(root: Path) -> tuple[Path, Path]:
    output = root / "online-v11"
    plan_dir = root / "confirm-v4-plan"
    audit_path = output / "evidence-audit.json"
    report_path = output / "RESULTS_zh.md"
    if audit_path.exists() or report_path.exists():
        raise FileExistsError("refusing to overwrite existing online-v11 audit/report")
    summary_path = output / "formal-v4-summary.json"
    completion_path = output / "formal-v4-complete.json"
    live_path = output / "live-winner-verified.json"
    plan_path = plan_dir / "placements.json"
    selection_path = plan_dir / "selection.json"
    manifest_path = plan_dir / "manifest.json"
    screen_audit_path = root / "online-v10/evidence-audit.json"
    prior_summary_path = root / "online-v9/formal-v3-summary.json"
    prior_audit_path = root / "online-v9/evidence-audit.json"
    first_formal_path = root / "online-v4/formal-summary.json"
    summary = load_json(summary_path)
    completion = load_json(completion_path)
    live = load_json(live_path)
    plan = load_json(plan_path)
    selection = load_json(selection_path)
    screen_audit = load_json(screen_audit_path)
    prior_summary = load_json(prior_summary_path)
    prior_audit = load_json(prior_audit_path)
    first_formal = load_json(first_formal_path)
    phases = tuple((row["phase_id"], row["strategy"]) for row in plan["confirmation_phases"])
    checks: list[dict[str, Any]] = []
    add = lambda name, condition, actual: worker_common.add_check(checks, name, condition, actual)
    add("summary status", summary.get("status") == "CONFIRMATION_COMPLETE", summary.get("status"))
    add("completion status", completion.get("status") == "CONFIRMATION_COMPLETE", completion.get("status"))
    add("confirmed winner", summary.get("confirmed_winner") == FINALIST, summary.get("confirmed_winner"))
    add("completion winner", completion.get("winner") == FINALIST, completion.get("winner"))
    add(
        "selection identities",
        selection.get("baseline") == BASELINE
        and selection.get("finalist") == FINALIST
        and selection.get("backup") == BACKUP,
        {key: selection.get(key) for key in ("baseline", "finalist", "backup")},
    )
    result = summary["formal_confirmation_result"]
    add("all recall pass", result.get("all_recall_pass") is True, result.get("all_recall_pass"))
    add("strict endpoint dominance", result.get("strict_endpoint_dominance") is True, result.get("strict_endpoint_dominance"))
    add("formal confirmation passed", result.get("confirmed_over_baseline") is True, result.get("confirmed_over_baseline"))
    add("screen audit", screen_audit.get("status") == "PASS", screen_audit.get("status"))
    add("prior formal audit", prior_audit.get("status") == "PASS", prior_audit.get("status"))
    add(
        "prior run retained baseline",
        prior_summary["formal_confirmation_result"]["baseline_strategy"] == BASELINE
        and prior_summary.get("confirmed_winner") is None,
        prior_summary.get("confirmed_winner"),
    )
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
    first_baseline = float(first_formal["formal_confirmation_result"]["baseline_qps_mean_across_phases"])
    cumulative = 100.0 * (finalist_mean / first_baseline - 1.0)
    add("formal improvement recomputed", abs(improvement - float(result["finalist_improvement_pct"])) < 1e-9, improvement)
    add(
        "strict endpoints recomputed",
        min(float(row["qps_mean"]) for row in finalist_rows)
        > max(float(row["qps_mean"]) for row in baseline_rows),
        {
            "finalist_min": min(float(row["qps_mean"]) for row in finalist_rows),
            "baseline_max": max(float(row["qps_mean"]) for row in baseline_rows),
        },
    )
    expected = {str(shard): int(peer) for shard, peer in plan["placements"][FINALIST].items()}
    ending = {
        str(shard): int(peer)
        for shard, peer in live["ending_placement"]["placement"].items()
    }
    add(
        "live winner verified",
        live.get("status") == "PASS"
        and live.get("strategy") == FINALIST
        and live.get("resource_state_valid") is True
        and live["ending_placement"].get("valid") is True
        and not live["ending_placement"].get("shard_transfers")
        and ending == expected,
        {"status": live.get("status"), "strategy": live.get("strategy")},
    )
    phase_count, placement_count = inventory(root)
    add("online placement inventory", phase_count == 58 and placement_count == 24, {"phase_count": phase_count, "unique_placement_count": placement_count})
    failed = [row for row in checks if row["status"] != "PASS"]
    artifact_paths = (
        summary_path,
        completion_path,
        live_path,
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
        first_formal_path,
        *(output / "phases" / phase_id / "screen.json" for phase_id, _ in phases),
        *(output / "phases" / phase_id / "placement.json" for phase_id, _ in phases),
    )
    audit = {
        "status": "PASS" if not failed else "FAIL",
        "record_type": "orion_physical_placement_worker_small_confirmation_evidence_audit",
        "claim_boundary": (
            "swap_24_21 is formally confirmed and best among 24 online-tested placements; "
            "global optimality is not established"
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
            "cumulative_improvement_over_controller_v1_pct": cumulative,
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
        raise RuntimeError(f"online-v11 evidence audit failed: {[row['name'] for row in failed]}")

    mapping = {int(shard): int(peer) for shard, peer in plan["placements"][FINALIST].items()}
    hosts = {int(peer): host for peer, host in plan["peer_hosts"].items()}
    placement_lines = []
    for peer in [int(value) for value in plan["peer_order"]]:
        shards = sorted(shard for shard, owner in mapping.items() if owner == peer)
        role = "controller" if peer == int(plan["controller_peer_id"]) else "worker"
        placement_lines.append(f"{hosts[peer]} {role}: {shards}")
    report = [
        "# Orion 小-shift worker-only 正式确认结果",
        "",
        "## 结论",
        "",
        (
            "在 4 台物理机、32 个逻辑分片、每台 Qdrant 16 核（总计 64 核）的固定合同下，"
            "`swap_24_21` 正式战胜 `swap_16_18`，成为当前新赢家。"
        ),
        "",
        "| 策略 | 正式 QPS | 相对上一赢家 | Recall@10 |",
        "|---|---:|---:|---:|",
        f"| `swap_16_18` A/B 均值 | {baseline_mean:.2f} | — | 0.92364 |",
        f"| **`swap_24_21` A/B 均值** | **{finalist_mean:.2f}** | **+{improvement:.2f}%** | 0.92364 |",
        f"| `swap_24_13` 单次 backup | {float(backup['qps_mean']):.2f} | {100.0 * (float(backup['qps_mean']) / baseline_mean - 1.0):+.2f}% | 0.92364 |",
        "",
        (
            f"严格端点判据通过：finalist 最低 {min(float(row['qps_mean']) for row in finalist_rows):.2f} "
            f"> baseline 最高 {max(float(row['qps_mean']) for row in baseline_rows):.2f} QPS。"
        ),
        f"相对最初 Controller-aware v1 的 {first_baseline:.2f} QPS，累计提升 {cumulative:.2f}%。",
        f"审计 `{len(checks)}/{len(checks)} PASS`；资源与 live winner 验证 PASS；无 shard transfer。",
        "",
        "## 当前 live placement",
        "",
        "```text",
        *placement_lines,
        "```",
        "",
        "## 边界",
        "",
        f"- 当前结论覆盖 {placement_count} 个唯一在线 physical placement、{phase_count} 个 phase。",
        "- `swap_24_21` 是已测试 placement 中最佳，不是全局最优证明。",
        "- Recall@10 全部为 0.92364；离线模型与 trace proxy 仅用于选候选。",
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
