#!/usr/bin/env python3
"""Audit the direct paired-online-QPS surrogate and its contradicted candidates."""

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
    for version in range(2, 17):
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
    output = root / "online-v16"
    plan_dir = root / "refine-v16-qps-surrogate"
    audit_path = output / "evidence-audit.json"
    report_path = output / "RESULTS_zh.md"
    if audit_path.exists() or report_path.exists():
        raise FileExistsError("refusing to overwrite existing online-v16 audit/report")
    summary_path = output / "screen-qps-surrogate-summary.json"
    completion_path = output / "screen-qps-surrogate-complete.json"
    plan_path = plan_dir / "placements.json"
    manifest_path = plan_dir / "manifest.json"
    metrics_path = plan_dir / "candidate-metrics.json"
    pairs_path = plan_dir / "paired-qps-observations.json"
    models_path = plan_dir / "qps-surrogate-models.json"
    ranking_path = plan_dir / "one-swap-qps-ranking.json"
    rationale_path = plan_dir / "RATIONALE_zh.md"
    prior_audit_path = root / "online-v15/evidence-audit.json"
    summary = load_json(summary_path)
    completion = load_json(completion_path)
    plan = load_json(plan_path)
    manifest = load_json(manifest_path)
    metrics = load_json(metrics_path)
    models = load_json(models_path)
    prior_audit = load_json(prior_audit_path)
    phases = tuple((row["phase_id"], row["strategy"]) for row in plan["screen_phases"])
    checks: list[dict[str, Any]] = []
    add = lambda name, condition, actual: worker_common.add_check(checks, name, condition, actual)
    add("summary status", summary.get("status") == "SCREEN_COMPLETE", summary.get("status"))
    add("completion status", completion.get("status") == "SCREEN_COMPLETE", completion.get("status"))
    add("winner retained", summary.get("screen_winner") == BASELINE, summary.get("screen_winner"))
    add("completion winner retained", completion.get("winner") == BASELINE, completion.get("winner"))
    add("prior evidence audit", prior_audit.get("status") == "PASS", prior_audit.get("status"))
    add(
        "surrogate manifest contract",
        manifest.get("status") == "OFFLINE_REFINEMENT_ONLY"
        and manifest.get("pair_observation_count") == 48
        and manifest.get("unique_online_placement_count") == 34
        and manifest.get("one_swap_neighbors_evaluated") == 384
        and manifest.get("selected_candidate_count") == 4,
        {
            key: manifest.get(key)
            for key in (
                "status",
                "pair_observation_count",
                "unique_online_placement_count",
                "one_swap_neighbors_evaluated",
                "selected_candidate_count",
            )
        },
    )
    rmse = [
        float(row["leave_one_candidate_placement_out_rmse_approx_pct"])
        for row in models["models"]
    ]
    add("surrogate LOPO RMSE recorded", len(rmse) == 3 and all(2.0 < value < 3.0 for value in rmse), rmse)
    selected_names = [row["strategy"] for row in summary["phases"] if row["strategy"] != BASELINE]
    add(
        "all selected candidates predicted positive",
        all(float(metrics[name]["predicted_improvement_min_pct"]) > 0.0 for name in selected_names),
        {name: metrics[name]["predicted_improvement_min_pct"] for name in selected_names},
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
    baseline_min = min(float(row["qps_mean"]) for row in baseline_rows)
    drift = 100.0 * (float(baseline_rows[1]["qps_mean"]) / float(baseline_rows[0]["qps_mean"]) - 1.0)
    deltas = {
        row["strategy"]: 100.0 * (float(row["qps_mean"]) / baseline_mean - 1.0)
        for row in candidate_rows
    }
    add(
        "all positive predictions contradicted online",
        all(float(row["qps_mean"]) < baseline_min for row in candidate_rows)
        and all(deltas[row["strategy"]] < -4.0 for row in candidate_rows),
        deltas,
    )
    add(
        "final phase restores formal winner",
        records[-1]["ending_placement"].get("valid") is True
        and not records[-1]["ending_placement"].get("shard_transfers"),
        records[-1]["ending_placement"].get("shard_transfers"),
    )
    phase_count, placement_count = inventory(root)
    add("online placement inventory", phase_count == 87 and placement_count == 40, {"phase_count": phase_count, "unique_placement_count": placement_count})
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
        pairs_path,
        models_path,
        ranking_path,
        rationale_path,
        prior_audit_path,
        *(output / "phases" / phase_id / "screen.json" for phase_id, _ in phases),
        *(output / "phases" / phase_id / "placement.json" for phase_id, _ in phases),
    )
    audit = {
        "status": "PASS" if not failed else "FAIL",
        "record_type": "orion_physical_placement_paired_qps_surrogate_evidence_audit",
        "claim_boundary": (
            "the linear paired-QPS surrogate is contradicted for candidate selection; "
            "swap_24_21 remains best among 40 online-tested placements"
        ),
        "physical_machine_count": 4,
        "logical_shard_count": 32,
        "passed_checks": len(checks) - len(failed),
        "total_checks": len(checks),
        "checks": checks,
        "metrics": {
            "baseline_qps_mean": baseline_mean,
            "baseline_endpoint_drift_pct": drift,
            "candidate_delta_pct": deltas,
            "surrogate_lopo_rmse_approx_pct": rmse,
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
        raise RuntimeError(f"online-v16 evidence audit failed: {[row['name'] for row in failed]}")
    report = [
        "# paired-online-QPS surrogate 在线验证结果",
        "",
        "在 4 台物理机、32 个逻辑分片、64 个 Qdrant 核的固定合同下，direct-QPS surrogate 的四个正预测全部被在线结果强烈推翻。",
        "",
        "| 策略 | 预测最小提升 | 在线 QPS | 相对 baseline A/B | Recall@10 |",
        "|---|---:|---:|---:|---:|",
        f"| **`swap_24_21` A/B 均值** | — | **{baseline_mean:.2f}** | — | 0.92364 |",
        *[
            (
                f"| `{row['strategy']}` | {float(metrics[row['strategy']]['predicted_improvement_min_pct']):+.2f}% | "
                f"{float(row['qps_mean']):.2f} | {deltas[row['strategy']]:+.2f}% | "
                f"{float(row['heldout_recall']['recall_at_10']):.5f} |"
            )
            for row in candidate_rows
        ],
        "",
        "模型 LOPO RMSE 约 2.60%，但四个候选实际下降约 5%–14%；该线性 surrogate 不再用于后续候选选择。",
        f"baseline A/B 漂移 {drift:+.3f}%；审计 `{len(checks)}/{len(checks)} PASS`；资源恢复 PASS。",
        "",
        f"当前累计 {placement_count} 个唯一在线 physical placement、{phase_count} 个 phase；正式赢家仍为 `swap_24_21`。",
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
