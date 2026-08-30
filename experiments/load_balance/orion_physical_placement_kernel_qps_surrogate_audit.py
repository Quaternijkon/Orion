#!/usr/bin/env python3
"""Audit the interaction-aware paired-QPS kernel refinement and stop gate."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

import orion_physical_placement_confirmation_audit as common


DEFAULT_ROOT = Path(
    "/proj/intelisys-PG0/exp/orion-distributed/orion-load-balance-r090-20260825"
)
DEFAULT_PLAN = DEFAULT_ROOT / "refine-v17-kernel-qps"
BASELINE = "swap_24_21"
EXPECTED_MODELS = ("hamming_rbf", "work_weighted_rbf", "colocation_rbf")
EXPECTED_SELECTED = (
    "kernel_neighbor_30_10",
    "kernel_neighbor_27_10",
    "kernel_neighbor_30_11",
    "kernel_neighbor_10_25",
)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def add_check(
    checks: list[dict[str, Any]], name: str, condition: bool, actual: Any
) -> None:
    checks.append(
        {"name": name, "status": "PASS" if condition else "FAIL", "actual": actual}
    )


def placement_key(mapping: dict[str, int] | dict[int, int]) -> tuple[tuple[int, int], ...]:
    return tuple(sorted((int(shard), int(peer)) for shard, peer in mapping.items()))


def online_inventory(root: Path) -> tuple[int, int]:
    paths = list((root / "online-v1/phases").glob("*/benchmark.json"))
    for version in range(2, 17):
        paths.extend((root / f"online-v{version}/phases").glob("*/screen.json"))
    mappings = {
        placement_key(load_json(path)["ending_placement"]["placement"])
        for path in paths
    }
    return len(paths), len(mappings)


def build(args: argparse.Namespace) -> tuple[Path, Path]:
    root = args.root.expanduser().resolve()
    plan_dir = args.plan_dir.expanduser().resolve()
    audit_path = plan_dir / "evidence-audit.json"
    report_path = plan_dir / "RESULTS_zh.md"
    if audit_path.exists() or report_path.exists():
        raise FileExistsError("refusing to overwrite existing kernel audit/report")

    manifest_path = plan_dir / "manifest.json"
    placements_path = plan_dir / "placements.json"
    metrics_path = plan_dir / "candidate-metrics.json"
    pairs_path = plan_dir / "paired-qps-observations.json"
    models_path = plan_dir / "kernel-qps-models.json"
    ranking_path = plan_dir / "one-swap-kernel-qps-ranking.json"
    rationale_path = plan_dir / "RATIONALE_zh.md"
    prior_audit_path = root / "online-v16/evidence-audit.json"
    prior_models_path = root / "refine-v16-qps-surrogate/qps-surrogate-models.json"
    prior_summary_path = root / "online-v16/screen-qps-surrogate-summary.json"

    manifest = load_json(manifest_path)
    placements = load_json(placements_path)
    metrics = load_json(metrics_path)
    pairs = load_json(pairs_path)
    models_payload = load_json(models_path)
    ranking = load_json(ranking_path)
    prior_audit = load_json(prior_audit_path)
    prior_models = load_json(prior_models_path)["models"]
    prior_summary = load_json(prior_summary_path)
    models = models_payload["models"]
    neighbors = ranking["neighbors"]
    untested = [row for row in neighbors if not row["already_online_tested"]]
    tested = [row for row in neighbors if row["already_online_tested"]]

    checks: list[dict[str, Any]] = []
    add_check(
        checks,
        "prior online evidence audit",
        prior_audit.get("status") == "PASS"
        and prior_audit.get("passed_checks") == prior_audit.get("total_checks")
        and prior_audit.get("metrics", {}).get("unique_online_placement_count") == 40,
        {
            "status": prior_audit.get("status"),
            "passed_checks": prior_audit.get("passed_checks"),
            "total_checks": prior_audit.get("total_checks"),
            "unique_online_placement_count": prior_audit.get("metrics", {}).get(
                "unique_online_placement_count"
            ),
        },
    )
    add_check(
        checks,
        "formal winner retained before refinement",
        prior_summary.get("screen_winner") == BASELINE,
        prior_summary.get("screen_winner"),
    )
    add_check(
        checks,
        "offline manifest contract",
        manifest.get("status") == "OFFLINE_REFINEMENT_ONLY"
        and manifest.get("physical_machine_count") == 4
        and manifest.get("logical_shard_count") == 32
        and manifest.get("pair_observation_count") == 52
        and manifest.get("unique_paired_placement_count") == 38
        and manifest.get("total_online_placement_count") == 40
        and manifest.get("one_swap_neighbors_evaluated") == 384
        and manifest.get("selected_candidate_count") == 4,
        {
            key: manifest.get(key)
            for key in (
                "status",
                "physical_machine_count",
                "logical_shard_count",
                "pair_observation_count",
                "unique_paired_placement_count",
                "total_online_placement_count",
                "one_swap_neighbors_evaluated",
                "selected_candidate_count",
            )
        },
    )
    add_check(
        checks,
        "paired-QPS observation inventory",
        pairs.get("pair_count") == 52
        and pairs.get("unique_paired_placement_count") == 38
        and len(pairs.get("pairs", [])) == 52,
        {
            "pair_count": pairs.get("pair_count"),
            "unique_paired_placement_count": pairs.get(
                "unique_paired_placement_count"
            ),
            "stored_pairs": len(pairs.get("pairs", [])),
        },
    )
    add_check(
        checks,
        "kernel model inventory",
        tuple(model.get("name") for model in models) == EXPECTED_MODELS
        and all(model.get("pair_observation_count") == 52 for model in models)
        and all(model.get("basis_placement_count") == 38 for model in models),
        [
            {
                "name": model.get("name"),
                "pair_observation_count": model.get("pair_observation_count"),
                "basis_placement_count": model.get("basis_placement_count"),
            }
            for model in models
        ],
    )
    rmse = {
        model["name"]: float(
            model["leave_one_candidate_placement_out_rmse_approx_pct"]
        )
        for model in models
    }
    sign_accuracy = {
        model["name"]: float(
            model["leave_one_candidate_placement_out_sign_accuracy"]
        )
        for model in models
    }
    prior_rmse = {
        model["name"]: float(
            model["leave_one_candidate_placement_out_rmse_approx_pct"]
        )
        for model in prior_models
    }
    best_kernel_rmse = min(rmse.values())
    best_linear_rmse = min(prior_rmse.values())
    rmse_gain_points = best_linear_rmse - best_kernel_rmse
    add_check(
        checks,
        "kernel cross-validation metrics finite",
        len(rmse) == 3
        and all(math.isfinite(value) and value > 0.0 for value in rmse.values())
        and all(0.0 <= value <= 1.0 for value in sign_accuracy.values()),
        {"rmse_pct": rmse, "sign_accuracy": sign_accuracy},
    )
    add_check(
        checks,
        "kernel improvement over linear model is marginal",
        0.0 < rmse_gain_points < 0.05
        and rmse["hamming_rbf"] > best_linear_rmse
        and rmse["colocation_rbf"] > best_linear_rmse,
        {
            "best_kernel_rmse_pct": best_kernel_rmse,
            "best_linear_rmse_pct": best_linear_rmse,
            "gain_percentage_points": rmse_gain_points,
            "kernel_rmse_pct": rmse,
        },
    )
    add_check(
        checks,
        "one-swap ranking inventory",
        ranking.get("evaluated_swaps") == 384
        and len(neighbors) == 384
        and len(tested) == 17
        and len(untested) == 367,
        {
            "evaluated_swaps": ranking.get("evaluated_swaps"),
            "stored_neighbors": len(neighbors),
            "online_tested_neighbors": len(tested),
            "untested_neighbors": len(untested),
        },
    )
    positive_counts = {
        name: sum(
            float(row["predicted_improvement_pct"][name]) > 0.0 for row in untested
        )
        for name in EXPECTED_MODELS
    }
    max_untested_prediction = {
        name: max(
            float(row["predicted_improvement_pct"][name]) for row in untested
        )
        for name in EXPECTED_MODELS
    }
    add_check(
        checks,
        "no kernel predicts a positive untested neighbor",
        all(count == 0 for count in positive_counts.values())
        and all(value < 0.0 for value in max_untested_prediction.values()),
        {
            "positive_untested_neighbor_count": positive_counts,
            "max_untested_prediction_pct": max_untested_prediction,
        },
    )
    selected = tuple(manifest.get("selected_strategies", []))
    selected_rows = [metrics[name] for name in selected]
    add_check(
        checks,
        "selected consensus candidates are expected ranks 5 through 8",
        selected == EXPECTED_SELECTED
        and [int(row["consensus_rank"]) for row in selected_rows] == [5, 6, 7, 8]
        and all(row["already_online_tested"] is False for row in selected_rows),
        {
            "selected": selected,
            "consensus_ranks": [row["consensus_rank"] for row in selected_rows],
            "already_online_tested": [
                row["already_online_tested"] for row in selected_rows
            ],
        },
    )
    selected_predictions = {
        name: {
            model: float(metrics[name]["predicted_improvement_pct"][model])
            for model in EXPECTED_MODELS
        }
        for name in selected
    }
    add_check(
        checks,
        "all selected candidates have negative predictions from every kernel",
        all(
            value < 0.0
            for predictions in selected_predictions.values()
            for value in predictions.values()
        ),
        selected_predictions,
    )
    add_check(
        checks,
        "offline evidence does not clear online-screen gate",
        max(value for values in selected_predictions.values() for value in values.values())
        <= 0.0,
        {
            "gate": "at least one credible positive prediction larger than model error",
            "best_selected_prediction_pct": max(
                value
                for values in selected_predictions.values()
                for value in values.values()
            ),
            "best_kernel_lopo_rmse_pct": best_kernel_rmse,
            "decision": "NO_ONLINE_V17",
        },
    )
    add_check(
        checks,
        "baseline placement is structurally valid",
        BASELINE in placements.get("placements", {})
        and len(placements["placements"][BASELINE]) == 32
        and set(placements["placements"][BASELINE].values())
        == set(placements.get("peer_order", [])),
        {
            "shard_count": len(placements.get("placements", {}).get(BASELINE, {})),
            "peers": sorted(
                set(placements.get("placements", {}).get(BASELINE, {}).values())
            ),
        },
    )
    final_phase = prior_summary["phases"][-1]
    final_phase_path = (
        root / "online-v16/phases" / final_phase["phase_id"] / "screen.json"
    )
    final_record = load_json(final_phase_path)
    add_check(
        checks,
        "live-ending placement remains formal winner",
        final_phase.get("strategy") == BASELINE
        and final_record.get("strategy") == BASELINE
        and final_record["ending_placement"].get("valid") is True
        and not final_record["ending_placement"].get("shard_transfers")
        and placement_key(final_record["ending_placement"]["placement"])
        == placement_key(placements["placements"][BASELINE]),
        {
            "phase_id": final_phase.get("phase_id"),
            "strategy": final_record.get("strategy"),
            "valid": final_record["ending_placement"].get("valid"),
            "shard_transfers": final_record["ending_placement"].get(
                "shard_transfers"
            ),
        },
    )
    phase_count, placement_count = online_inventory(root)
    add_check(
        checks,
        "online inventory unchanged",
        phase_count == 87 and placement_count == 40,
        {"phase_count": phase_count, "unique_placement_count": placement_count},
    )
    add_check(
        checks,
        "online-v17 was not started",
        not (root / "online-v17").exists(),
        str(root / "online-v17"),
    )
    for name, record in manifest["outputs"].items():
        path = plan_dir / name
        add_check(
            checks,
            f"offline output hash: {name}",
            path.is_file() and common.sha256_path(path) == record["sha256"],
            common.sha256_path(path) if path.is_file() else None,
        )

    failed = [row for row in checks if row["status"] != "PASS"]
    artifact_paths = (
        manifest_path,
        placements_path,
        metrics_path,
        pairs_path,
        models_path,
        ranking_path,
        rationale_path,
        prior_audit_path,
        prior_models_path,
        prior_summary_path,
        final_phase_path,
    )
    audit = {
        "status": "PASS" if not failed else "FAIL",
        "record_type": "orion_physical_placement_kernel_qps_surrogate_stop_gate_audit",
        "decision": "NO_ONLINE_V17",
        "evidence_class": "WEAK_NEGATIVE_OFFLINE_EVIDENCE",
        "claim_boundary": (
            "the tested interaction-aware kernel surrogates do not justify another "
            "online screen; swap_24_21 remains best among 40 online-tested physical "
            "placements, not a proven global optimum"
        ),
        "physical_machine_count": 4,
        "logical_shard_count": 32,
        "passed_checks": len(checks) - len(failed),
        "total_checks": len(checks),
        "checks": checks,
        "metrics": {
            "pair_observation_count": 52,
            "unique_paired_placement_count": 38,
            "online_phase_count": phase_count,
            "unique_online_placement_count": placement_count,
            "online_tested_one_swap_neighbor_count": len(tested),
            "untested_one_swap_neighbor_count": len(untested),
            "kernel_lopo_rmse_approx_pct": rmse,
            "kernel_lopo_sign_accuracy": sign_accuracy,
            "best_linear_lopo_rmse_approx_pct": best_linear_rmse,
            "best_kernel_lopo_rmse_approx_pct": best_kernel_rmse,
            "best_kernel_rmse_gain_percentage_points": rmse_gain_points,
            "positive_untested_neighbor_count": positive_counts,
            "max_untested_prediction_pct": max_untested_prediction,
            "selected_candidate_predictions_pct": selected_predictions,
        },
        "artifacts": {
            str(path): {
                "sha256": common.sha256_path(path),
                "size_bytes": path.stat().st_size,
            }
            for path in artifact_paths
        },
    }
    common.write_json_new(audit_path, audit)
    if failed:
        raise RuntimeError(
            f"kernel-QPS stop-gate audit failed: {[row['name'] for row in failed]}"
        )

    report = [
        "# interaction-aware paired-QPS kernel 结论",
        "",
        "结论：不启动 `online-v17`。kernel surrogate 没有产生值得在线验证的正向信号，按弱否定性离线证据封存。",
        "",
        "| 模型 | LOPO RMSE | sign accuracy | 相对最佳线性 RMSE |",
        "|---|---:|---:|---:|",
        *[
            (
                f"| `{name}` | {rmse[name]:.3f}% | {100.0 * sign_accuracy[name]:.1f}% | "
                f"{rmse[name] - best_linear_rmse:+.3f} pp |"
            )
            for name in EXPECTED_MODELS
        ],
        "",
        (
            f"最佳 kernel RMSE 为 {best_kernel_rmse:.3f}%，仅比最佳线性模型 "
            f"{best_linear_rmse:.3f}% 低 {rmse_gain_points:.3f} 个百分点；另外两种核退化到约 3.01%。"
        ),
        "",
        "| 未测候选 | swap | 三核预测范围 | consensus rank |",
        "|---|---:|---:|---:|",
        *[
            (
                f"| `{name}` | {metrics[name]['left_shard']}↔{metrics[name]['right_shard']} | "
                f"{min(selected_predictions[name].values()):+.2f}% … "
                f"{max(selected_predictions[name].values()):+.2f}% | "
                f"{metrics[name]['consensus_rank']} |"
            )
            for name in selected
        ],
        "",
        (
            "367 个未在线测试的单交换邻居中，三种核各自预测为正的数量均为 0；"
            "因此不存在预测提升超过交叉验证误差的候选。"
        ),
        "",
        "边界：这不是对全部 placement 的全局最优证明，也不能证明所有未测邻居必然更差；它只说明当前 kernel 模型不足以支持继续消耗在线实验时间。",
        (
            f"当前正式赢家仍为 `swap_24_21`，范围限定为 {placement_count} 个已在线测试的 "
            f"physical placement；在线库存仍为 {phase_count} 个 phase。"
        ),
        f"审计：`{len(checks)}/{len(checks)} PASS`；最终 placement 有效且无 shard transfer。",
        "",
    ]
    with report_path.open("x", encoding="utf-8") as handle:
        handle.write("\n".join(report))
    return audit_path, report_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--plan-dir", type=Path, default=DEFAULT_PLAN)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    audit, report = build(parse_args(argv))
    print(json.dumps({"audit": str(audit), "report": str(report)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
