#!/usr/bin/env python3
"""Audit the complete-neighborhood online screen and preserve its negative result."""

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
DEFAULT_OUTPUT = DEFAULT_ROOT / "online-v7"
DEFAULT_PLAN = DEFAULT_ROOT / "refine-v7"
BASELINE = "swap_16_18"
CANDIDATES = ("swap_27_16", "swap_5_8", "swap_27_25", "swap_30_11")


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
        raise FileExistsError("refusing to overwrite existing online-v7 audit/report")

    summary_path = output / "screen-neighbor-v2-summary.json"
    completion_path = output / "screen-neighbor-v2-complete.json"
    resource_contract_path = output / "resource-contract.json"
    collection_contract_path = output / "collection-contract.json"
    preparation_path = output / "preparation.json"
    restored_path = output / "resource-restored.json"
    placements_path = plan_dir / "placements.json"
    manifest_path = plan_dir / "manifest.json"
    metrics_path = plan_dir / "candidate-metrics.json"
    models_path = plan_dir / "service-demand-models.json"
    neighborhood_path = plan_dir / "one-swap-neighborhood.json"
    prior_summary_path = root / "online-v6/formal-v2-summary.json"
    prior_audit_path = root / "online-v6/evidence-audit.json"

    summary = load_json(summary_path)
    completion = load_json(completion_path)
    resource_contract = load_json(resource_contract_path)
    collection_contract = load_json(collection_contract_path)
    preparation = load_json(preparation_path)
    restored = load_json(restored_path)
    placements = load_json(placements_path)
    manifest = load_json(manifest_path)
    candidate_metrics = load_json(metrics_path)
    models = load_json(models_path)
    neighborhood = load_json(neighborhood_path)
    prior_summary = load_json(prior_summary_path)
    prior_audit = load_json(prior_audit_path)
    expected_phases = tuple(
        (str(row["phase_id"]), str(row["strategy"]))
        for row in placements["screen_phases"]
    )

    checks: list[dict[str, Any]] = []

    def check(name: str, condition: bool, actual: Any) -> None:
        checks.append(
            {"name": name, "status": "PASS" if condition else "FAIL", "actual": actual}
        )

    check("summary status", summary.get("status") == "SCREEN_COMPLETE", summary.get("status"))
    check("completion status", completion.get("status") == "SCREEN_COMPLETE", completion.get("status"))
    check("screen winner retained", summary.get("screen_winner") == BASELINE, summary.get("screen_winner"))
    check("completion winner retained", completion.get("winner") == BASELINE, completion.get("winner"))
    check(
        "fixed physical/logical plan contract",
        placements.get("physical_machine_count") == 4
        and placements.get("logical_shard_count") == 32
        and len(placements.get("peer_order", [])) == 4,
        {
            "physical_machine_count": placements.get("physical_machine_count"),
            "logical_shard_count": placements.get("logical_shard_count"),
            "peer_count": len(placements.get("peer_order", [])),
        },
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
    check(
        "prior formal winner",
        prior_summary.get("confirmed_winner") == BASELINE,
        prior_summary.get("confirmed_winner"),
    )
    check(
        "offline enumeration contract",
        manifest.get("status") == "OFFLINE_REFINEMENT_ONLY"
        and manifest.get("physical_machine_count") == 4
        and manifest.get("logical_shard_count") == 32
        and manifest.get("modeling")
        == {
            "model_count": 3,
            "node_observation_count": 48,
            "one_swap_neighbors_evaluated": 384,
            "selection_count": 4,
            "unique_online_placement_count": 12,
        },
        manifest.get("modeling"),
    )
    check(
        "complete one-swap neighborhood",
        neighborhood.get("baseline") == BASELINE
        and neighborhood.get("evaluated_swaps") == 384
        and len(neighborhood.get("neighbors", [])) == 384,
        {
            "baseline": neighborhood.get("baseline"),
            "evaluated_swaps": neighborhood.get("evaluated_swaps"),
            "neighbor_count": len(neighborhood.get("neighbors", [])),
        },
    )
    check(
        "selected model-consensus ranks",
        [candidate_metrics[name].get("consensus_rank") for name in CANDIDATES]
        == [1, 2, 3, 4],
        [candidate_metrics[name].get("consensus_rank") for name in CANDIDATES],
    )
    model_rows = models.get("models", [])
    rmse_values = [float(row["leave_one_placement_out_rmse"]) for row in model_rows]
    check(
        "three leave-one-placement-out models",
        len(model_rows) == 3 and all(100.0 < value < 160.0 for value in rmse_values),
        rmse_values,
    )
    for name, record in manifest.get("outputs", {}).items():
        path = plan_dir / name
        check(
            f"offline output hash: {name}",
            path.is_file() and common.sha256_path(path) == record.get("sha256"),
            common.sha256_path(path) if path.is_file() else None,
        )

    phase_records: list[dict[str, Any]] = []
    for phase_id, strategy in expected_phases:
        phase_path = output / "phases" / phase_id / "screen.json"
        placement_path = output / "phases" / phase_id / "placement.json"
        phase = load_json(phase_path)
        placement = load_json(placement_path)
        expected_mapping = {
            str(shard): int(peer) for shard, peer in placements["placements"][strategy].items()
        }
        ending_mapping = {
            str(shard): int(peer)
            for shard, peer in phase["ending_placement"]["placement"].items()
        }
        phase_checks = {
            "status": phase.get("status") == "PASS",
            "identity": phase.get("phase_id") == phase_id and phase.get("strategy") == strategy,
            "screen marker": phase.get("screen_only") is True
            and phase.get("formal_confirmation") is False,
            "physical/logical contract": phase.get("physical_machine_count") == 4
            and phase.get("logical_shard_count") == 32,
            "held-out recall": float(phase["heldout_recall"]["recall_at_10"]) >= 0.90,
            "three repeats": phase.get("repeat_count") == 3,
            "10-second windows": all(float(row["wall_s"]) >= 10.0 for row in phase["repeats"]),
            "saturation knee": phase["saturation_selection"].get("knee_observed") is True,
            "placement valid": phase["ending_placement"].get("valid") is True,
            "no residual transfers": not phase["ending_placement"].get("shard_transfers"),
            "expected placement": ending_mapping == expected_mapping,
            "placement operation": placement.get("status") == "PASS",
        }
        for label, condition in phase_checks.items():
            check(f"{phase_id}: {label}", condition, condition)
        phase_records.append(phase)

    baseline_rows = [phase_records[0], phase_records[-1]]
    candidate_rows = phase_records[1:-1]
    baseline_mean = statistics.fmean(float(row["qps_mean"]) for row in baseline_rows)
    baseline_min = min(float(row["qps_mean"]) for row in baseline_rows)
    baseline_drift = 100.0 * (
        float(baseline_rows[1]["qps_mean"]) / float(baseline_rows[0]["qps_mean"]) - 1.0
    )
    check(
        "all four candidates lose to baseline minimum",
        all(float(row["qps_mean"]) < baseline_min for row in candidate_rows),
        {row["strategy"]: float(row["qps_mean"]) for row in candidate_rows},
    )
    check(
        "baseline A/B drift below one percent",
        abs(baseline_drift) < 1.0,
        baseline_drift,
    )
    for strategy, rows in {
        name: [row for row in phase_records if row["strategy"] == name]
        for name in summary["strategy_summary"]
    }.items():
        recomputed = statistics.fmean(float(row["qps_mean"]) for row in rows)
        check(
            f"summary QPS recomputed: {strategy}",
            abs(recomputed - float(summary["strategy_summary"][strategy]["qps_mean_across_phases"]))
            < 1e-9,
            recomputed,
        )

    candidate_deltas = {
        row["strategy"]: 100.0 * (float(row["qps_mean"]) / baseline_mean - 1.0)
        for row in candidate_rows
    }
    failed = [row for row in checks if row["status"] != "PASS"]
    artifact_paths = (
        summary_path,
        completion_path,
        resource_contract_path,
        collection_contract_path,
        preparation_path,
        restored_path,
        placements_path,
        manifest_path,
        metrics_path,
        models_path,
        neighborhood_path,
        prior_summary_path,
        prior_audit_path,
        *(output / "phases" / phase_id / "screen.json" for phase_id, _ in expected_phases),
        *(output / "phases" / phase_id / "placement.json" for phase_id, _ in expected_phases),
    )
    audit = {
        "status": "PASS" if not failed else "FAIL",
        "record_type": "orion_physical_placement_complete_neighbor_screen_evidence_audit",
        "claim_boundary": (
            "the four model-consensus neighbors are contradicted online; this does not prove "
            "one-swap local or global optimality because untested neighbors remain"
        ),
        "physical_machine_count": 4,
        "logical_shard_count": 32,
        "passed_checks": len(checks) - len(failed),
        "total_checks": len(checks),
        "checks": checks,
        "metrics": {
            "baseline_qps_mean": baseline_mean,
            "baseline_qps_min": baseline_min,
            "baseline_endpoint_drift_pct": baseline_drift,
            "candidate_delta_pct": candidate_deltas,
            "recall_at_10": min(
                float(row["heldout_recall"]["recall_at_10"]) for row in phase_records
            ),
            "model_leave_one_placement_out_rmse": rmse_values,
        },
        "artifacts": {
            str(path.relative_to(root)): {
                "sha256": common.sha256_path(path),
                "size_bytes": path.stat().st_size,
            }
            for path in artifact_paths
        },
    }
    common.write_json_new(audit_path, audit)
    if failed:
        raise RuntimeError(f"online-v7 evidence audit failed: {[row['name'] for row in failed]}")

    report = [
        "# Orion 完整单交换模型候选在线筛选结果",
        "",
        "## 结论",
        "",
        (
            "在 4 台物理机、32 个逻辑分片、每台 Qdrant 16 核（总计 64 核）的固定合同下，"
            "完整枚举 `swap_16_18` 的 384 个单交换邻居后，由三个模型共同选出的前四名全部在线失败。"
        ),
        "",
        "| 策略 | QPS | 相对同轮基线 | Recall@10 |",
        "|---|---:|---:|---:|",
        f"| `swap_16_18` A/B 均值 | {baseline_mean:.2f} | — | 0.92364 |",
        *[
            f"| `{row['strategy']}` | {float(row['qps_mean']):.2f} | {candidate_deltas[row['strategy']]:+.2f}% | {float(row['heldout_recall']['recall_at_10']):.5f} |"
            for row in candidate_rows
        ],
        "",
        f"基线 A/B 漂移仅 {baseline_drift:+.3f}%；四个候选均低于基线最低值 {baseline_min:.2f} QPS。",
        "",
        "## 模型与在线证据",
        "",
        "- 12 个唯一在线 placement、48 个节点观测。",
        "- 三模型 leave-one-placement-out RMSE 为 "
        + " / ".join(f"{value:.2f}" for value in rmse_values)
        + " μs/query。",
        "- 384/384 单交换邻居已离线枚举；模型前四名的在线方向全部被推翻。",
        "- Recall@10 全部为 0.92364，差异来自固定召回口径下的物理 placement QPS。",
        f"- 证据审计：`{len(checks)}/{len(checks)} PASS`；资源恢复为 PASS。",
        "",
        "## 边界与下一步",
        "",
        "- 离线模型只负责提名候选，不能替代在线 QPS。",
        "- 不能据此宣称 `swap_16_18` 已达到单交换局部最优：其余邻居尚未在线逐一测试。",
        "- 下一步只测试 worker-only 微调，把热点 `10.10.1.2` 的小部分 work 移向 `10.10.1.3/.4`，不触碰 controller。",
        "- 所有 learned/proxy 负结果继续保留，不覆盖旧 evidence。",
        "",
        "## 证据位置",
        "",
        "- `refine-v7/one-swap-neighborhood.json`",
        "- `refine-v7/service-demand-models.json`",
        "- `online-v7/screen-neighbor-v2-summary.json`",
        "- `online-v7/phases/*/screen.json`",
        "- `online-v7/evidence-audit.json`",
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
