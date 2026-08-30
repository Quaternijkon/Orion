#!/usr/bin/env python3
"""Audit the first complete-neighborhood screen around swap_24_21."""

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


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def inventory(root: Path) -> tuple[int, int]:
    files = list((root / "online-v1/phases").glob("*/benchmark.json"))
    for version in range(2, 13):
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


def build(args: argparse.Namespace) -> tuple[Path, Path]:
    root = args.root.expanduser().resolve()
    output = root / "online-v12"
    plan_dir = root / "refine-v12-neighbor"
    audit_path = output / "evidence-audit.json"
    report_path = output / "RESULTS_zh.md"
    if audit_path.exists() or report_path.exists():
        raise FileExistsError("refusing to overwrite existing online-v12 audit/report")
    summary_path = output / "screen-new-winner-neighbor-summary.json"
    completion_path = output / "screen-new-winner-neighbor-complete.json"
    plan_path = plan_dir / "placements.json"
    manifest_path = plan_dir / "manifest.json"
    metrics_path = plan_dir / "candidate-metrics.json"
    neighborhood_path = plan_dir / "one-swap-neighborhood.json"
    models_path = plan_dir / "service-demand-models.json"
    rationale_path = plan_dir / "RATIONALE_zh.md"
    prior_summary_path = root / "online-v11/formal-v4-summary.json"
    prior_audit_path = root / "online-v11/evidence-audit.json"
    summary = load_json(summary_path)
    completion = load_json(completion_path)
    plan = load_json(plan_path)
    manifest = load_json(manifest_path)
    neighborhood = load_json(neighborhood_path)
    models = load_json(models_path)
    prior_summary = load_json(prior_summary_path)
    prior_audit = load_json(prior_audit_path)
    phases = tuple((row["phase_id"], row["strategy"]) for row in plan["screen_phases"])
    checks: list[dict[str, Any]] = []
    add = lambda name, condition, actual: worker_common.add_check(checks, name, condition, actual)
    add("summary status", summary.get("status") == "SCREEN_COMPLETE", summary.get("status"))
    add("completion status", completion.get("status") == "SCREEN_COMPLETE", completion.get("status"))
    add("screen winner retained", summary.get("screen_winner") == BASELINE, summary.get("screen_winner"))
    add("completion winner retained", completion.get("winner") == BASELINE, completion.get("winner"))
    add("prior formal winner", prior_summary.get("confirmed_winner") == BASELINE, prior_summary.get("confirmed_winner"))
    add("prior formal audit", prior_audit.get("status") == "PASS", prior_audit.get("status"))
    add(
        "complete-neighborhood manifest",
        manifest.get("status") == "OFFLINE_REFINEMENT_ONLY"
        and manifest.get("modeling")
        == {
            "unique_online_placement_count": 24,
            "node_observation_count": 96,
            "one_swap_neighbors_evaluated": 384,
            "already_tested_neighbor_count": 1,
            "selected_candidate_count": 4,
            "model_count": 3,
        },
        manifest.get("modeling"),
    )
    add(
        "complete one-swap enumeration",
        neighborhood.get("baseline") == BASELINE
        and neighborhood.get("evaluated_swaps") == 384
        and len(neighborhood.get("neighbors", [])) == 384,
        {
            "baseline": neighborhood.get("baseline"),
            "evaluated_swaps": neighborhood.get("evaluated_swaps"),
            "neighbor_count": len(neighborhood.get("neighbors", [])),
        },
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
    baseline_drift = 100.0 * (float(baseline_rows[1]["qps_mean"]) / float(baseline_rows[0]["qps_mean"]) - 1.0)
    add(
        "all selected neighbors lose to baseline mean",
        all(float(row["qps_mean"]) < baseline_mean for row in candidate_rows),
        {row["strategy"]: row["qps_mean"] for row in candidate_rows},
    )
    add("baseline endpoint drift below one percent", abs(baseline_drift) < 1.0, baseline_drift)
    add(
        "final phase restores winner placement",
        records[-1]["ending_placement"].get("valid") is True
        and not records[-1]["ending_placement"].get("shard_transfers"),
        records[-1]["ending_placement"].get("shard_transfers"),
    )
    phase_count, placement_count = inventory(root)
    add("online placement inventory", phase_count == 64 and placement_count == 28, {"phase_count": phase_count, "unique_placement_count": placement_count})
    deltas = {
        row["strategy"]: 100.0 * (float(row["qps_mean"]) / baseline_mean - 1.0)
        for row in candidate_rows
    }
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
        neighborhood_path,
        models_path,
        rationale_path,
        prior_summary_path,
        prior_audit_path,
        *(output / "phases" / phase_id / "screen.json" for phase_id, _ in phases),
        *(output / "phases" / phase_id / "placement.json" for phase_id, _ in phases),
    )
    audit = {
        "status": "PASS" if not failed else "FAIL",
        "record_type": "orion_physical_placement_new_winner_neighbor_screen_evidence_audit",
        "claim_boundary": (
            "the four updated-model leaders around swap_24_21 are contradicted online; "
            "this is not a full online audit of all 384 neighbors"
        ),
        "physical_machine_count": 4,
        "logical_shard_count": 32,
        "passed_checks": len(checks) - len(failed),
        "total_checks": len(checks),
        "checks": checks,
        "metrics": {
            "baseline_qps_mean": baseline_mean,
            "baseline_endpoint_drift_pct": baseline_drift,
            "candidate_delta_pct": deltas,
            "recall_at_10": min(float(row["heldout_recall"]["recall_at_10"]) for row in records),
            "model_leave_one_placement_out_rmse": rmses,
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
        raise RuntimeError(f"online-v12 evidence audit failed: {[row['name'] for row in failed]}")
    report = [
        "# swap_24_21 新赢家邻域在线筛选结果",
        "",
        "在 4 台物理机、32 个逻辑分片、64 个 Qdrant 核的固定合同下，新赢家完整单交换邻域的模型前四名全部在线失败。",
        "",
        "| 策略 | QPS | 相对同轮 baseline A/B | Recall@10 |",
        "|---|---:|---:|---:|",
        f"| **`swap_24_21` A/B 均值** | **{baseline_mean:.2f}** | — | 0.92364 |",
        *[
            f"| `{row['strategy']}` | {float(row['qps_mean']):.2f} | {deltas[row['strategy']]:+.2f}% | {float(row['heldout_recall']['recall_at_10']):.5f} |"
            for row in candidate_rows
        ],
        "",
        f"baseline A/B 漂移 {baseline_drift:+.3f}%；审计 `{len(checks)}/{len(checks)} PASS`；资源恢复 PASS。",
        "`neighbor_24_7` 的 CV 达 5.17%，同时 QPS 明显下降，因此不构成可用候选。",
        "",
        "## 结论边界",
        "",
        f"- 当前累计 {placement_count} 个唯一在线 physical placement、{phase_count} 个 phase。",
        "- `swap_24_21` 仍是已测试 placement 中的正式最佳结果。",
        "- 只在线否定了模型前四名；其余邻居未逐一在线测试，不能宣称单交换局部或全局最优。",
        "- 384 邻域已完整离线枚举，模型/trace 排名仍不能替代在线 QPS。",
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
    audit, report = build(parse_args(argv))
    print(json.dumps({"audit": str(audit), "report": str(report)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
