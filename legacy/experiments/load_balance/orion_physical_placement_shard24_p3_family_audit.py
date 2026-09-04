#!/usr/bin/env python3
"""Audit the exhaustive shard-24-to-worker-.3 counterpart family screen."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Sequence

import orion_physical_placement_confirmation_audit as common
import orion_physical_placement_worker_audit as worker_common


DEFAULT_ROOT = Path(
    "/proj/intelisys-PG0/exp/orion-distributed/orion-load-balance-r090-20260825"
)
BASELINE = "swap_24_21"
FAMILY_WINNER = "shard24_p3_counterpart_13"
EXPECTED_COUNTERPARTS = {1, 2, 12, 13, 17, 22, 26, 31}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def placement_key(mapping: dict[str, int]) -> tuple[tuple[int, int], ...]:
    return tuple(sorted((int(shard), int(peer)) for shard, peer in mapping.items()))


def online_inventory(root: Path) -> tuple[int, int]:
    paths = list((root / "online-v1/phases").glob("*/benchmark.json"))
    for version in (*range(2, 17), 18, 19):
        paths.extend((root / f"online-v{version}/phases").glob("*/screen.json"))
    mappings = {
        placement_key(load_json(path)["ending_placement"]["placement"])
        for path in paths
    }
    return len(paths), len(mappings)


def build(args: argparse.Namespace) -> tuple[Path, Path]:
    root = args.root.expanduser().resolve()
    output = root / "online-v19"
    plan_dir = root / "refine-v19-shard24-p3-family"
    audit_path = output / "evidence-audit.json"
    report_path = output / "RESULTS_zh.md"
    if audit_path.exists() or report_path.exists():
        raise FileExistsError("refusing to overwrite existing online-v19 audit/report")

    summary_path = output / "screen-shard24-p3-family-summary.json"
    completion_path = output / "screen-shard24-p3-family-complete.json"
    plan_path = plan_dir / "placements.json"
    manifest_path = plan_dir / "manifest.json"
    metrics_path = plan_dir / "candidate-metrics.json"
    universe_path = plan_dir / "family-universe.json"
    rationale_path = plan_dir / "RATIONALE_zh.md"
    prior_audit_path = root / "online-v18/evidence-audit.json"

    summary = load_json(summary_path)
    completion = load_json(completion_path)
    plan = load_json(plan_path)
    manifest = load_json(manifest_path)
    metrics = load_json(metrics_path)
    universe = load_json(universe_path)
    prior_audit = load_json(prior_audit_path)
    phases = tuple((row["phase_id"], row["strategy"]) for row in plan["screen_phases"])

    checks: list[dict[str, Any]] = []
    add = lambda name, condition, actual: worker_common.add_check(
        checks, name, condition, actual
    )
    add(
        "summary status",
        summary.get("status") == "SCREEN_COMPLETE",
        summary.get("status"),
    )
    add(
        "completion status",
        completion.get("status") == "SCREEN_COMPLETE",
        completion.get("status"),
    )
    add(
        "current winner retained",
        summary.get("screen_winner") == BASELINE,
        summary.get("screen_winner"),
    )
    add(
        "completion winner retained",
        completion.get("winner") == BASELINE,
        completion.get("winner"),
    )
    add(
        "prior family audit",
        prior_audit.get("status") == "PASS"
        and prior_audit.get("metrics", {}).get("unique_online_placement_count") == 47,
        {
            "status": prior_audit.get("status"),
            "unique_online_placement_count": prior_audit.get("metrics", {}).get(
                "unique_online_placement_count"
            ),
        },
    )
    add(
        "finite-family manifest contract",
        manifest.get("status") == "OFFLINE_REFINEMENT_ONLY"
        and manifest.get("physical_machine_count") == 4
        and manifest.get("logical_shard_count") == 32
        and manifest.get("family_size") == 8
        and manifest.get("previously_online_tested_member_count") == 1
        and manifest.get("new_candidate_count") == 7
        and manifest.get("selected_candidate_count") == 8
        and manifest.get("surrogate_used_for_selection") is False,
        {
            key: manifest.get(key)
            for key in (
                "status",
                "physical_machine_count",
                "logical_shard_count",
                "family_size",
                "previously_online_tested_member_count",
                "new_candidate_count",
                "selected_candidate_count",
                "surrogate_used_for_selection",
            )
        },
    )
    selected = tuple(manifest["selected_strategies"])
    counterparts = {int(metrics[name]["counterpart_shard"]) for name in selected}
    previously_tested = [
        int(metrics[name]["counterpart_shard"])
        for name in selected
        if metrics[name]["previously_online_tested"]
    ]
    add(
        "all .3 family members selected exactly once",
        len(selected) == len(set(selected)) == 8
        and counterparts == EXPECTED_COUNTERPARTS
        and previously_tested == [13]
        and universe.get("selection_rule") == "complete enumeration of all eight members",
        {
            "selected_count": len(selected),
            "counterparts": sorted(counterparts),
            "previously_tested": previously_tested,
            "selection_rule": universe.get("selection_rule"),
        },
    )
    add(
        "fixed shuffled phase order",
        universe.get("phase_order_seed") == 3_248_142
        and universe.get("phase_counterpart_order")
        == [2, 17, 12, 31, 26, 1, 22, 13]
        and len(phases) == 10
        and phases[0] == ("swap-24-21-p3-family-a", BASELINE)
        and phases[-1] == ("swap-24-21-p3-family-b", BASELINE),
        {
            "seed": universe.get("phase_order_seed"),
            "counterpart_order": universe.get("phase_counterpart_order"),
            "phase_count": len(phases),
            "first": phases[0],
            "last": phases[-1],
        },
    )
    for name, record in manifest["outputs"].items():
        path = plan_dir / name
        add(
            f"offline output hash: {name}",
            path.is_file() and common.sha256_path(path) == record["sha256"],
            common.sha256_path(path) if path.is_file() else None,
        )

    worker_common.common_contract_checks(checks, output, plan)
    records = [
        worker_common.audit_phase(
            checks, output, plan, phase_id, strategy, formal=False
        )
        for phase_id, strategy in phases
    ]
    baseline_rows = [records[0], records[-1]]
    candidate_rows = records[1:-1]
    baseline_mean = statistics.fmean(float(row["qps_mean"]) for row in baseline_rows)
    baseline_min = min(float(row["qps_mean"]) for row in baseline_rows)
    baseline_max = max(float(row["qps_mean"]) for row in baseline_rows)
    drift = 100.0 * (
        float(baseline_rows[1]["qps_mean"]) / float(baseline_rows[0]["qps_mean"])
        - 1.0
    )
    deltas = {
        row["strategy"]: 100.0 * (float(row["qps_mean"]) / baseline_mean - 1.0)
        for row in candidate_rows
    }
    endpoint_gaps = {
        row["strategy"]: 100.0 * (float(row["qps_mean"]) / baseline_min - 1.0)
        for row in candidate_rows
    }
    add(
        "all eight .3 family members below current-winner minimum endpoint",
        len(candidate_rows) == 8
        and all(float(row["qps_mean"]) < baseline_min for row in candidate_rows),
        {
            "baseline_min": baseline_min,
            "candidate_qps": {
                row["strategy"]: row["qps_mean"] for row in candidate_rows
            },
        },
    )
    family_winner = max(candidate_rows, key=lambda row: float(row["qps_mean"]))
    add(
        "counterpart 13 is complete .3-family winner",
        family_winner["strategy"] == FAMILY_WINNER
        and int(metrics[FAMILY_WINNER]["counterpart_shard"]) == 13
        and len(candidate_rows) == universe.get("family_size"),
        {
            "strategy": family_winner["strategy"],
            "qps": family_winner["qps_mean"],
            "family_size": universe.get("family_size"),
            "online_covered_members": len(candidate_rows),
        },
    )
    historical_values = [
        float(metrics[FAMILY_WINNER]["historical_screen_qps"]),
        float(metrics[FAMILY_WINNER]["historical_confirmation_qps"]),
    ]
    historical_mean = statistics.fmean(historical_values)
    anchor_delta = 100.0 * (
        float(family_winner["qps_mean"]) / historical_mean - 1.0
    )
    add(
        "historical counterpart-13 signal reproduced",
        abs(anchor_delta) < 0.5,
        {
            "historical_qps": historical_values,
            "historical_mean": historical_mean,
            "current_qps": family_winner["qps_mean"],
            "delta_pct": anchor_delta,
        },
    )
    add(
        "no formal confirmation needed",
        float(family_winner["qps_mean"]) < baseline_min,
        {
            "criterion": "candidate must exceed both current-winner endpoints",
            "baseline_endpoints": [
                float(baseline_rows[0]["qps_mean"]),
                float(baseline_rows[1]["qps_mean"]),
            ],
            "best_family_candidate_qps": float(family_winner["qps_mean"]),
        },
    )
    add(
        "final phase restores formal winner",
        records[-1]["ending_placement"].get("valid") is True
        and not records[-1]["ending_placement"].get("shard_transfers")
        and placement_key(records[-1]["ending_placement"]["placement"])
        == placement_key(plan["placements"][BASELINE]),
        {
            "valid": records[-1]["ending_placement"].get("valid"),
            "shard_transfers": records[-1]["ending_placement"].get(
                "shard_transfers"
            ),
        },
    )
    phase_count, placement_count = online_inventory(root)
    add(
        "online placement inventory",
        phase_count == 106 and placement_count == 54,
        {"phase_count": phase_count, "unique_placement_count": placement_count},
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
        universe_path,
        rationale_path,
        prior_audit_path,
        *(output / "phases" / phase_id / "screen.json" for phase_id, _ in phases),
        *(output / "phases" / phase_id / "placement.json" for phase_id, _ in phases),
    )
    audit = {
        "status": "PASS" if not failed else "FAIL",
        "record_type": "orion_physical_placement_shard24_p3_family_evidence_audit",
        "claim_boundary": (
            "counterpart 13 is best across the completely online-tested eight-member "
            "shard-24-to-worker-.3 family, while swap_24_21 beats every member and "
            "remains best among 54 online-tested physical placements; this is not a "
            "global-optimality proof"
        ),
        "physical_machine_count": 4,
        "logical_shard_count": 32,
        "passed_checks": len(checks) - len(failed),
        "total_checks": len(checks),
        "checks": checks,
        "metrics": {
            "baseline_qps_mean": baseline_mean,
            "baseline_qps_min": baseline_min,
            "baseline_qps_max": baseline_max,
            "baseline_endpoint_drift_pct": drift,
            "candidate_delta_vs_baseline_mean_pct": deltas,
            "candidate_gap_vs_baseline_min_pct": endpoint_gaps,
            "p3_family_winner": family_winner["strategy"],
            "p3_family_winner_qps": float(family_winner["qps_mean"]),
            "p3_family_winner_historical_delta_pct": anchor_delta,
            "recall_at_10": min(
                float(row["heldout_recall"]["recall_at_10"]) for row in records
            ),
            "family_size": 8,
            "family_members_online_tested": 8,
            "online_phase_count": phase_count,
            "unique_online_placement_count": placement_count,
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
        raise RuntimeError(
            f"online-v19 evidence audit failed: {[row['name'] for row in failed]}"
        )

    ordered_candidates = sorted(
        candidate_rows, key=lambda row: float(row["qps_mean"]), reverse=True
    )
    report = [
        "# shard 24→`.3` counterpart 有限族在线穷举结果",
        "",
        "在 4 台物理机、32 个逻辑分片、64 个 Qdrant 核的固定合同下，counterpart 13 是 `.3` 有限族最佳，但该族全部 8 个成员都低于当前正式赢家 `swap_24_21`。",
        "",
        "| 策略 | counterpart | QPS | 相对 winner A/B | Recall@10 |",
        "|---|---:|---:|---:|---:|",
        f"| **`swap_24_21` A/B 均值** | — | **{baseline_mean:.2f}** | — | 0.92364 |",
        *[
            (
                f"| `{row['strategy']}` | {int(metrics[row['strategy']]['counterpart_shard'])} | "
                f"{float(row['qps_mean']):.2f} | {deltas[row['strategy']]:+.2f}% | "
                f"{float(row['heldout_recall']['recall_at_10']):.5f} |"
            )
            for row in ordered_candidates
        ],
        "",
        (
            f"winner A/B 为 {float(baseline_rows[0]['qps_mean']):.2f} / "
            f"{float(baseline_rows[1]['qps_mean']):.2f} QPS，漂移 {drift:+.3f}%。"
        ),
        (
            f"counterpart 13 本轮为 {float(family_winner['qps_mean']):.2f} QPS，"
            f"相对其历史均值 {historical_mean:.2f} 仅 {anchor_delta:+.2f}%，但仍低于 "
            f"winner 最低端点 {baseline_min:.2f}。"
        ),
        "",
        "结论边界：`.3` family 8/8 已在线覆盖，因此可声明 counterpart 13 是该有限族内实测最佳；`swap_24_21` 则仍是全部已测试 placement 中最佳，不能宣称全局最优。",
        (
            f"累计在线证据为 {placement_count} 个唯一 physical placement、{phase_count} 个 phase。"
        ),
        f"审计：`{len(checks)}/{len(checks)} PASS`；资源恢复和最终 live placement 均 PASS，无 shard transfer。",
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
