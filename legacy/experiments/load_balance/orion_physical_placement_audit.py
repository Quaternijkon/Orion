#!/usr/bin/env python3
"""Independently audit and report an Orion physical-placement A/B run."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


DEFAULT_ROOT = Path(
    "/proj/intelisys-PG0/exp/orion-distributed/"
    "orion-load-balance-r090-20260825/online-v1"
)
DEFAULT_PLAN = Path(
    "/proj/intelisys-PG0/exp/orion-distributed/"
    "orion-load-balance-r090-20260825/plan"
)
PHASES = (
    ("round-robin-a", "round_robin"),
    ("controller-tail-a", "controller_aware_tail"),
    ("size-balanced-a", "size_balanced"),
    ("round-robin-b", "round_robin"),
    ("controller-tail-b", "controller_aware_tail"),
)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_text(path: Path, value: str) -> None:
    with path.open("x", encoding="utf-8") as handle:
        handle.write(value)


def percent_gain(new: float, old: float) -> float:
    return 100.0 * (new / old - 1.0)


def audit(root: Path, plan_dir: Path) -> dict[str, Any]:
    root = root.expanduser().resolve()
    plan_dir = plan_dir.expanduser().resolve()
    summary = load_json(root / "summary.json")
    plan_manifest = load_json(plan_dir / "manifest.json")
    placements = load_json(plan_dir / "placements.json")
    phase_results = {
        phase_id: load_json(root / "phases" / phase_id / "benchmark.json")
        for phase_id, _strategy in PHASES
    }
    resource = load_json(root / "resource-contract.json")
    cleanup = load_json(root / "collection-cleanup.json")
    restored = load_json(root / "resource-restored.json")

    fixed_contracts = [result["fixed_contract"] for result in phase_results.values()]
    first_contract = fixed_contracts[0]
    expected_concurrency = [1, 2, 4, 8, 16, 32, 64]
    checks: dict[str, bool] = {
        "plan_is_offline_only": plan_manifest.get("status") == "OFFLINE_PLAN_ONLY",
        "five_phase_files_present": len(phase_results) == 5,
        "phase_strategy_sequence_exact": all(
            phase_results[phase_id]["strategy"] == strategy
            for phase_id, strategy in PHASES
        ),
        "all_phase_status_pass": all(
            result.get("status") == "PASS" for result in phase_results.values()
        ),
        "all_fixed_contracts_identical": all(
            contract == first_contract for contract in fixed_contracts
        ),
        "fixed_artifact_generation": first_contract.get("artifact_generation") == 3_248_141,
        "fixed_artifact_sha256": first_contract.get("artifact_sha256")
        == "8936c14a242af48ccb675825f5a193b3307585a2a588087622bac3276577708e",
        "fixed_router_parameters": all(
            first_contract.get(key) == value
            for key, value in {
                "upper_k": 48,
                "upper_search_ef": 48,
                "dynamic_ef_base": 50,
                "dynamic_ef_factor": 14,
            }.items()
        ),
        "fixed_batch_200": first_contract.get("batch_size") == 200,
        "fixed_resource_64_cores": resource.get("qdrant_cpu_cores_total") == 64
        and resource.get("qdrant_cpu_cores_per_machine") == 16,
        "all_four_physical_machines": all(
            result.get("physical_machine_count") == 4 for result in phase_results.values()
        ),
        "all_32_logical_shards": all(
            result.get("logical_shard_count") == 32 for result in phase_results.values()
        ),
        "all_9000_heldout_queries": all(
            result["heldout_recall"].get("query_count") == 9000
            for result in phase_results.values()
        ),
        "all_recall_at_least_090": all(
            result["heldout_recall"]["recall_at_10"] >= 0.90
            for result in phase_results.values()
        ),
        "recall_invariant_across_placements": len(
            {result["heldout_recall"]["recall_at_10"] for result in phase_results.values()}
        )
        == 1,
        "all_concurrency_grids_complete": all(
            [row["concurrency"] for row in result["concurrency_sweep"]]
            == expected_concurrency
            for result in phase_results.values()
        ),
        "all_saturation_knees_observed": all(
            result["saturation_selection"]["knee_observed"]
            for result in phase_results.values()
        ),
        "all_at_least_five_repeats": all(
            result["repeat_count"] >= 5 for result in phase_results.values()
        ),
        "all_cv_at_most_005": all(
            result["qps_cv"] <= 0.05 for result in phase_results.values()
        ),
        "all_qps_finite_positive": all(
            math.isfinite(result["qps_mean"]) and result["qps_mean"] > 0
            for result in phase_results.values()
        ),
        "all_start_placements_valid": all(
            result["placement"]["proof"]["valid"] for result in phase_results.values()
        ),
        "all_end_placements_valid": all(
            result["ending_placement"]["valid"] for result in phase_results.values()
        ),
        "all_placements_stable_during_measurement": all(
            result["placement"]["proof"]["expected_placement"]
            == result["ending_placement"]["placement"]
            for result in phase_results.values()
        ),
        "summary_winner_controller_aware": summary.get("online_winner")
        == "controller_aware_tail",
        "strict_ab_confirmation_passed": summary.get(
            "controller_aware_tail_confirmed_over_round_robin"
        )
        is True,
        "collection_cleanup_empty": cleanup.get("status") == "PASS"
        and cleanup.get("remaining_collections") == [],
        "all_resources_restored": all(
            row.get("valid") is True and row.get("exact_docker_metadata_restored") is True
            for row in restored.get("nodes", [])
        )
        and len(restored.get("nodes", [])) == 4,
    }
    for key, value in plan_manifest["outputs"].items():
        if key.endswith("_sha256"):
            continue
        declared_path = Path(value)
        declared_digest = plan_manifest["outputs"].get(key + "_sha256")
        checks[f"plan_output_checksum_{key}"] = (
            declared_path.is_file()
            and declared_digest is not None
            and sha256_path(declared_path) == declared_digest
        )

    rr = summary["strategy_summary"]["round_robin"]
    candidate = summary["strategy_summary"]["controller_aware_tail"]
    size = summary["strategy_summary"]["size_balanced"]
    metrics = {
        "round_robin_qps_mean_across_phases": rr["qps_mean_across_phases"],
        "controller_aware_tail_qps_mean_across_phases": candidate[
            "qps_mean_across_phases"
        ],
        "size_balanced_qps": size["qps_mean_across_phases"],
        "controller_aware_gain_over_round_robin_percent": percent_gain(
            candidate["qps_mean_across_phases"], rr["qps_mean_across_phases"]
        ),
        "size_balanced_gain_over_round_robin_percent": percent_gain(
            size["qps_mean_across_phases"], rr["qps_mean_across_phases"]
        ),
        "controller_aware_gain_over_size_balanced_percent": percent_gain(
            candidate["qps_mean_across_phases"], size["qps_mean_across_phases"]
        ),
        "round_robin_ab_drift_percent": 100.0
        * abs(rr["qps_max"] - rr["qps_min"])
        / rr["qps_mean_across_phases"],
        "controller_aware_ab_drift_percent": 100.0
        * abs(candidate["qps_max"] - candidate["qps_min"])
        / candidate["qps_mean_across_phases"],
    }
    checks["round_robin_ab_drift_below_1pct"] = metrics[
        "round_robin_ab_drift_percent"
    ] < 1.0
    checks["controller_aware_ab_drift_below_1pct"] = metrics[
        "controller_aware_ab_drift_percent"
    ] < 1.0
    checks["controller_candidate_min_exceeds_round_robin_max"] = (
        candidate["qps_min"] > rr["qps_max"]
    )
    checks["controller_candidate_exceeds_size_balanced"] = (
        candidate["qps_mean_across_phases"] > size["qps_mean_across_phases"]
    )

    evidence_files = [
        root / "summary.csv",
        root / "summary.json",
        root / "completion-audit.json",
        root / "execution-complete.json",
        root / "collection-contract.json",
        root / "resource-contract.json",
        root / "collection-cleanup.json",
        root / "resource-restored.json",
        plan_dir / "manifest.json",
        plan_dir / "placements.json",
        plan_dir / "candidate-metrics.csv",
    ]
    for phase_id, _strategy in PHASES:
        evidence_files.extend(
            [
                root / "phases" / phase_id / "placement.json",
                root / "phases" / phase_id / "benchmark.json",
            ]
        )
    evidence = {
        str(path): {"sha256": sha256_path(path), "size_bytes": path.stat().st_size}
        for path in evidence_files
    }

    audit_record = {
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "PASS" if all(checks.values()) else "FAIL",
        "claim_boundary": (
            "controller_aware_tail is the highest-QPS strategy among the three tested "
            "physical placements on four machines; this is not a proof of global optimality"
        ),
        "checks": checks,
        "passed": sum(checks.values()),
        "total": len(checks),
        "online_metrics": metrics,
        "candidate_placement": placements["placements"]["controller_aware_tail"],
        "candidate_shards_by_host": {
            placements["peer_hosts"][str(peer_id)]: sorted(
                int(shard_id)
                for shard_id, owner in placements["placements"][
                    "controller_aware_tail"
                ].items()
                if int(owner) == int(peer_id)
            )
            for peer_id in placements["peer_order"]
        },
        "evidence_files": evidence,
    }
    if audit_record["status"] != "PASS":
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"evidence audit failed: {failed}")
    return audit_record


def report_markdown(audit_record: dict[str, Any], root: Path) -> str:
    summary = load_json(root / "summary.json")
    rows = summary["rows"]
    lines = [
        "# Orion 物理分片负载均衡在线实验结果",
        "",
        "固定条件：GloVe-200-angular、Cosine、top-10、32 个逻辑分片、4 台物理机、"
        "每台 Qdrant 16 核（总计 64 核）、artifact generation 3248141、batch=200。",
        "",
        "| 阶段 | 策略 | Recall@10 | knee 并发度 | QPS | CV | P95 batch 延迟 (ms) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    labels = {
        "round_robin": "Round-robin",
        "controller_aware_tail": "Controller-aware tail",
        "size_balanced": "Size-balanced",
    }
    for row in rows:
        lines.append(
            f"| {row['phase_id']} | {labels[row['strategy']]} | "
            f"{row['recall_at_10']:.5f} | {row['selected_concurrency']} | "
            f"{row['qps_mean']:.2f} | {100.0 * row['qps_cv']:.3f}% | "
            f"{row['batch_latency_p95_ms']:.2f} |"
        )
    metrics = audit_record["online_metrics"]
    lines.extend(
        [
            "",
            "## 结论",
            "",
            "在本次测试的三种物理 placement 中，`controller_aware_tail` 的在线 QPS 最高。"
            f"两次测量均值为 {metrics['controller_aware_tail_qps_mean_across_phases']:.2f} QPS，"
            f"相对两次 Round-robin 均值提升 {metrics['controller_aware_gain_over_round_robin_percent']:.2f}%，"
            f"相对 Size-balanced 提升 {metrics['controller_aware_gain_over_size_balanced_percent']:.2f}%。",
            "",
            "两次 Round-robin 的漂移仅 "
            f"{metrics['round_robin_ab_drift_percent']:.4f}%，两次候选的漂移为 "
            f"{metrics['controller_aware_ab_drift_percent']:.4f}%；全部五个阶段的 "
            "Recall@10 均为 0.92366。严格确认规则（候选两次结果均高于 Round-robin 两次结果）通过。",
            "",
            "该结论限定为当前数据集、artifact、路由参数、64 核资源合同、4 台物理机和已测试的三种 placement；"
            "它不构成对所有可能分片映射的全局最优证明。",
            "",
            "## 最佳 placement",
            "",
        ]
    )
    for host, shards in audit_record["candidate_shards_by_host"].items():
        role = "controller" if host == "10.10.1.1" else "worker"
        lines.append(f"- `{host}` ({role}): `{shards}`")
    lines.extend(
        [
            "",
            f"证据审计：`{audit_record['passed']}/{audit_record['total']}` checks PASS。",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--plan-dir", type=Path, default=DEFAULT_PLAN)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.root.expanduser().resolve()
    audit_path = root / "evidence-audit.json"
    report_path = root / "RESULTS_zh.md"
    record = audit(root, args.plan_dir)
    write_json(audit_path, record)
    write_text(report_path, report_markdown(record, root))
    print(
        json.dumps(
            {
                "evidence_audit": str(audit_path),
                "report": str(report_path),
                "checks": f"{record['passed']}/{record['total']}",
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
