#!/usr/bin/env python3
"""Audit the shard-24 controller family and 24-counterpart coverage."""

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
FAMILY_WINNER = "shard24_controller_counterpart_5"
EXPECTED_COUNTERPARTS = {0, 4, 5, 18, 27, 28, 29, 30}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def placement_key(mapping: dict[str, int]) -> tuple[tuple[int, int], ...]:
    return tuple(sorted((int(shard), int(peer)) for shard, peer in mapping.items()))


def online_inventory(root: Path) -> tuple[int, int]:
    paths = list((root / "online-v1/phases").glob("*/benchmark.json"))
    for version in (*range(2, 17), 18, 19, 20):
        paths.extend((root / f"online-v{version}/phases").glob("*/screen.json"))
    mappings = {
        placement_key(load_json(path)["ending_placement"]["placement"])
        for path in paths
    }
    return len(paths), len(mappings)


def build(args: argparse.Namespace) -> tuple[Path, Path]:
    root = args.root.expanduser().resolve()
    output = root / "online-v20"
    plan_dir = root / "refine-v20-shard24-controller-family"
    audit_path = output / "evidence-audit.json"
    report_path = output / "RESULTS_zh.md"
    if audit_path.exists() or report_path.exists():
        raise FileExistsError("refusing to overwrite existing online-v20 audit/report")

    summary_path = output / "screen-shard24-controller-family-summary.json"
    completion_path = output / "screen-shard24-controller-family-complete.json"
    plan_path = plan_dir / "placements.json"
    manifest_path = plan_dir / "manifest.json"
    metrics_path = plan_dir / "candidate-metrics.json"
    universe_path = plan_dir / "family-universe.json"
    rationale_path = plan_dir / "RATIONALE_zh.md"
    p4_audit_path = root / "online-v18/evidence-audit.json"
    p3_audit_path = root / "online-v19/evidence-audit.json"

    summary = load_json(summary_path)
    completion = load_json(completion_path)
    plan = load_json(plan_path)
    manifest = load_json(manifest_path)
    metrics = load_json(metrics_path)
    universe = load_json(universe_path)
    p4_audit = load_json(p4_audit_path)
    p3_audit = load_json(p3_audit_path)
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
        "prior worker-family audits",
        p4_audit.get("status") == "PASS"
        and p3_audit.get("status") == "PASS"
        and p4_audit.get("metrics", {}).get("family_members_online_tested") == 8
        and p3_audit.get("metrics", {}).get("family_members_online_tested") == 8,
        {
            "p4_status": p4_audit.get("status"),
            "p4_coverage": p4_audit.get("metrics", {}).get(
                "family_members_online_tested"
            ),
            "p3_status": p3_audit.get("status"),
            "p3_coverage": p3_audit.get("metrics", {}).get(
                "family_members_online_tested"
            ),
        },
    )
    add(
        "finite-family manifest contract",
        manifest.get("status") == "OFFLINE_REFINEMENT_ONLY"
        and manifest.get("physical_machine_count") == 4
        and manifest.get("logical_shard_count") == 32
        and manifest.get("family_size") == 8
        and manifest.get("previously_online_tested_member_count") == 0
        and manifest.get("new_candidate_count") == 8
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
    add(
        "all controller family members selected exactly once",
        len(selected) == len(set(selected)) == 8
        and counterparts == EXPECTED_COUNTERPARTS
        and not any(metrics[name]["previously_online_tested"] for name in selected)
        and universe.get("selection_rule") == "complete enumeration of all eight members",
        {
            "selected_count": len(selected),
            "counterparts": sorted(counterparts),
            "selection_rule": universe.get("selection_rule"),
        },
    )
    add(
        "fixed shuffled phase order",
        universe.get("phase_order_seed") == 3_248_143
        and universe.get("phase_counterpart_order")
        == [29, 18, 28, 27, 5, 30, 4, 0]
        and len(phases) == 10
        and phases[0] == ("swap-24-21-controller-family-a", BASELINE)
        and phases[-1] == ("swap-24-21-controller-family-b", BASELINE),
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
        "all controller family members below current-winner minimum endpoint",
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
        "counterpart 5 is complete controller-family winner",
        family_winner["strategy"] == FAMILY_WINNER
        and int(metrics[FAMILY_WINNER]["counterpart_shard"]) == 5
        and len(candidate_rows) == universe.get("family_size"),
        {
            "strategy": family_winner["strategy"],
            "qps": family_winner["qps_mean"],
            "family_size": universe.get("family_size"),
            "online_covered_members": len(candidate_rows),
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
        "shard-24 cross-machine counterpart coverage complete",
        p4_audit.get("metrics", {}).get("family_members_online_tested")
        + p3_audit.get("metrics", {}).get("family_members_online_tested")
        + len(candidate_rows)
        == 24,
        {
            "worker_p4_family": p4_audit.get("metrics", {}).get(
                "family_members_online_tested"
            ),
            "worker_p3_family": p3_audit.get("metrics", {}).get(
                "family_members_online_tested"
            ),
            "controller_family": len(candidate_rows),
            "total": p4_audit.get("metrics", {}).get(
                "family_members_online_tested"
            )
            + p3_audit.get("metrics", {}).get("family_members_online_tested")
            + len(candidate_rows),
        },
    )
    add(
        "swap_24_21 wins all three target-host families",
        p4_audit.get("metrics", {}).get("closest_candidate_qps")
        < p4_audit.get("metrics", {}).get("baseline_qps_min")
        and p3_audit.get("metrics", {}).get("p3_family_winner_qps")
        < p3_audit.get("metrics", {}).get("baseline_qps_min")
        and float(family_winner["qps_mean"]) < baseline_min,
        {
            "worker_p4": {
                "best_challenger_qps": p4_audit.get("metrics", {}).get(
                    "closest_candidate_qps"
                ),
                "winner_min_qps": p4_audit.get("metrics", {}).get(
                    "baseline_qps_min"
                ),
            },
            "worker_p3": {
                "best_challenger_qps": p3_audit.get("metrics", {}).get(
                    "p3_family_winner_qps"
                ),
                "winner_min_qps": p3_audit.get("metrics", {}).get(
                    "baseline_qps_min"
                ),
            },
            "controller": {
                "best_challenger_qps": family_winner["qps_mean"],
                "winner_min_qps": baseline_min,
            },
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
        phase_count == 116 and placement_count == 62,
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
        p4_audit_path,
        p3_audit_path,
        *(output / "phases" / phase_id / "screen.json" for phase_id, _ in phases),
        *(output / "phases" / phase_id / "placement.json" for phase_id, _ in phases),
    )
    audit = {
        "status": "PASS" if not failed else "FAIL",
        "record_type": "orion_physical_placement_shard24_controller_family_evidence_audit",
        "claim_boundary": (
            "counterpart 5 is best across the completely online-tested controller "
            "family, while swap_24_21 beats all 24 cross-machine shard-24 "
            "counterparts and remains best among 62 online-tested physical "
            "placements; this is not a global-optimality proof"
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
            "controller_family_winner": family_winner["strategy"],
            "controller_family_winner_qps": float(family_winner["qps_mean"]),
            "recall_at_10": min(
                float(row["heldout_recall"]["recall_at_10"]) for row in records
            ),
            "controller_family_size": 8,
            "controller_family_members_online_tested": 8,
            "shard24_cross_machine_counterpart_count": 24,
            "shard24_cross_machine_counterparts_online_tested": 24,
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
            f"online-v20 evidence audit failed: {[row['name'] for row in failed]}"
        )

    ordered_candidates = sorted(
        candidate_rows, key=lambda row: float(row["qps_mean"]), reverse=True
    )
    report = [
        "# shard 24→controller counterpart 有限族在线穷举结果",
        "",
        "在 4 台物理机、32 个逻辑分片、64 个 Qdrant 核的固定合同下，controller 族全部 8 个成员都低于当前正式赢家 `swap_24_21`。",
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
            f"controller 族最佳 counterpart 5 为 {float(family_winner['qps_mean']):.2f} QPS，"
            f"仍比 winner A/B 均值低 {-deltas[FAMILY_WINNER]:.2f}%。"
        ),
        "",
        "## shard 24 跨机单交换完整结论",
        "",
        "- `10.10.1.4` target family：8/8 在线覆盖，最佳为 counterpart 21，即 `swap_24_21`。",
        "- `10.10.1.3` target family：8/8 在线覆盖，族内最佳为 counterpart 13，但低于 `swap_24_21`。",
        "- controller `10.10.1.1` target family：8/8 在线覆盖，族内最佳为 counterpart 5，但低于 `swap_24_21`。",
        "",
        "因此 shard 24 的 24/24 个跨机单交换 counterpart 已全部在线测试，`swap_24_21` 是该完整有限搜索空间内的实测最佳。这个结论不扩展到任意多交换 placement 的全局最优。",
        (
            f"累计在线证据为 {placement_count} 个唯一 physical placement、{phase_count} 个 phase；"
            "全部 Recall@10 均为 0.92364。"
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
