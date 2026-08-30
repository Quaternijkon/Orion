#!/usr/bin/env python3
"""Audit and report the formal local Orion placement confirmation evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any, Sequence


DEFAULT_ROOT = Path(
    "/proj/intelisys-PG0/exp/orion-distributed/orion-load-balance-r090-20260825"
)
DEFAULT_OUTPUT = DEFAULT_ROOT / "online-v4"
DEFAULT_PLAN = DEFAULT_ROOT / "confirm-v1-plan"
BASELINE = "controller_aware_v1"
FINALIST = "swap_c8_0"
BACKUP = "swap_19_15"
EXPECTED_PHASES = (
    ("controller-v1-confirm-a", BASELINE),
    ("swap-c8-0-confirm-a", FINALIST),
    ("swap-19-15-confirm", BACKUP),
    ("controller-v1-confirm-b", BASELINE),
    ("swap-c8-0-confirm-b", FINALIST),
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


def write_json_new(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def pct(value: float) -> str:
    return f"{value:.2f}%"


def build(args: argparse.Namespace) -> tuple[Path, Path]:
    root = args.root.expanduser().resolve()
    output = args.output_root.expanduser().resolve()
    plan_dir = args.plan_dir.expanduser().resolve()
    audit_path = output / "evidence-audit.json"
    report_path = output / "RESULTS_zh.md"
    if audit_path.exists() or report_path.exists():
        raise FileExistsError("refusing to overwrite existing audit/report")

    summary_path = output / "formal-summary.json"
    completion_path = output / "formal-complete.json"
    resource_contract_path = output / "resource-contract.json"
    collection_contract_path = output / "collection-contract.json"
    preparation_path = output / "preparation.json"
    restored_path = output / "resource-restored.json"
    plan_path = plan_dir / "placements.json"
    selection_path = plan_dir / "selection.json"
    screen_v3_path = root / "online-v3/screen-local-summary.json"
    screen_v2_path = root / "online-v2/screen-summary.json"
    summary = load_json(summary_path)
    completion = load_json(completion_path)
    resource_contract = load_json(resource_contract_path)
    collection_contract = load_json(collection_contract_path)
    preparation = load_json(preparation_path)
    restored = load_json(restored_path)
    plan = load_json(plan_path)
    selection = load_json(selection_path)
    screen_v3 = load_json(screen_v3_path)
    screen_v2 = load_json(screen_v2_path)

    checks: list[dict[str, Any]] = []

    def check(name: str, condition: bool, actual: Any) -> None:
        checks.append(
            {"name": name, "status": "PASS" if condition else "FAIL", "actual": actual}
        )

    check("summary status", summary.get("status") == "CONFIRMATION_COMPLETE", summary.get("status"))
    check(
        "completion status",
        completion.get("status") == "CONFIRMATION_COMPLETE",
        completion.get("status"),
    )
    check("confirmed winner", summary.get("confirmed_winner") == FINALIST, summary.get("confirmed_winner"))
    check("completion winner", completion.get("winner") == FINALIST, completion.get("winner"))
    check(
        "strict endpoint dominance",
        summary["formal_confirmation_result"]["strict_endpoint_dominance"] is True,
        summary["formal_confirmation_result"]["strict_endpoint_dominance"],
    )
    check(
        "all recall pass",
        summary["formal_confirmation_result"]["all_recall_pass"] is True,
        summary["formal_confirmation_result"]["all_recall_pass"],
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
    check(
        "confirmation plan order",
        tuple(
            (row["phase_id"], row["strategy"])
            for row in plan["confirmation_phases"]
        )
        == EXPECTED_PHASES,
        plan["confirmation_phases"],
    )
    check("screen-selected finalist", selection.get("finalist") == FINALIST, selection.get("finalist"))
    check("screen-selected backup", selection.get("backup") == BACKUP, selection.get("backup"))

    phase_records = []
    for index, (phase_id, strategy) in enumerate(EXPECTED_PHASES):
        phase_path = output / "phases" / phase_id / "screen.json"
        placement_path = output / "phases" / phase_id / "placement.json"
        phase = load_json(phase_path)
        placement = load_json(placement_path)
        expected_mapping = {
            str(shard): int(peer)
            for shard, peer in plan["placements"][strategy].items()
        }
        ending_mapping = {
            str(shard): int(peer)
            for shard, peer in phase["ending_placement"]["placement"].items()
        }
        checks_for_phase = {
            "status": phase.get("status") == "PASS",
            "phase identity": phase.get("phase_id") == phase_id
            and phase.get("strategy") == strategy,
            "formal marker": phase.get("formal_confirmation") is True
            and phase.get("screen_only") is False,
            "physical/logical contract": phase.get("physical_machine_count") == 4
            and phase.get("logical_shard_count") == 32,
            "held-out recall": float(phase["heldout_recall"]["recall_at_10"]) >= 0.90,
            "repeat count": phase.get("repeat_count") == 5,
            "20-second repeats": all(float(row["wall_s"]) >= 20.0 for row in phase["repeats"]),
            "placement valid": phase["ending_placement"].get("valid") is True,
            "no residual transfers": not phase["ending_placement"].get("shard_transfers"),
            "expected placement": ending_mapping == expected_mapping,
            "placement operation": placement.get("status") == "PASS",
        }
        for label, condition in checks_for_phase.items():
            check(f"{phase_id}: {label}", condition, condition)
        phase_records.append(phase)

    baseline = [phase_records[0], phase_records[3]]
    finalist = [phase_records[1], phase_records[4]]
    backup = phase_records[2]
    baseline_mean = statistics.fmean(float(row["qps_mean"]) for row in baseline)
    finalist_mean = statistics.fmean(float(row["qps_mean"]) for row in finalist)
    baseline_drift = 100.0 * (
        float(baseline[1]["qps_mean"]) / float(baseline[0]["qps_mean"]) - 1.0
    )
    finalist_drift = 100.0 * (
        float(finalist[1]["qps_mean"]) / float(finalist[0]["qps_mean"]) - 1.0
    )
    improvement = 100.0 * (finalist_mean / baseline_mean - 1.0)
    backup_improvement = 100.0 * (float(backup["qps_mean"]) / baseline_mean - 1.0)
    check(
        "recomputed finalist improvement",
        abs(improvement - summary["formal_confirmation_result"]["finalist_improvement_pct"])
        < 1e-9,
        improvement,
    )
    check(
        "finalist minimum exceeds baseline maximum",
        min(float(row["qps_mean"]) for row in finalist)
        > max(float(row["qps_mean"]) for row in baseline),
        {
            "finalist_min": min(float(row["qps_mean"]) for row in finalist),
            "baseline_max": max(float(row["qps_mean"]) for row in baseline),
        },
    )

    failed = [row for row in checks if row["status"] != "PASS"]
    audit = {
        "status": "PASS" if not failed else "FAIL",
        "record_type": "orion_physical_placement_confirmation_evidence_audit",
        "claim_boundary": (
            "swap_c8_0 is confirmed over controller_aware_v1 and is best among eight "
            "online-tested placements; global optimality is not established"
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
            "backup_improvement_pct": backup_improvement,
            "baseline_endpoint_drift_pct": baseline_drift,
            "finalist_endpoint_drift_pct": finalist_drift,
            "recall_at_10": min(
                float(row["heldout_recall"]["recall_at_10"]) for row in phase_records
            ),
        },
        "artifacts": {
            str(path.relative_to(root)): {"sha256": sha256_path(path), "size_bytes": path.stat().st_size}
            for path in (
                summary_path,
                completion_path,
                resource_contract_path,
                collection_contract_path,
                preparation_path,
                restored_path,
                plan_path,
                selection_path,
                screen_v3_path,
                screen_v2_path,
                *(output / "phases" / phase_id / "screen.json" for phase_id, _ in EXPECTED_PHASES),
                *(output / "phases" / phase_id / "placement.json" for phase_id, _ in EXPECTED_PHASES),
            )
        },
    }
    write_json_new(audit_path, audit)
    if failed:
        raise RuntimeError(f"evidence audit failed: {[row['name'] for row in failed]}")

    strategy_rows = [
        (
            "Controller-aware v1（两端均值）",
            baseline_mean,
            0.0,
            min(float(row["heldout_recall"]["recall_at_10"]) for row in baseline),
            "2 / 2",
            "10.10.1.2≈15.97 核",
        ),
        (
            "swap 8↔0（两端均值）",
            finalist_mean,
            improvement,
            min(float(row["heldout_recall"]["recall_at_10"]) for row in finalist),
            "4 / 4",
            "controller≈15.97 核",
        ),
        (
            "swap 19↔15（单次正式备选）",
            float(backup["qps_mean"]),
            backup_improvement,
            float(backup["heldout_recall"]["recall_at_10"]),
            str(backup["selected_concurrency"]),
            "10.10.1.4≈15.96 核",
        ),
    ]
    final_mapping = {
        int(shard): int(peer) for shard, peer in plan["placements"][FINALIST].items()
    }
    peer_hosts = {int(peer): host for peer, host in plan["peer_hosts"].items()}
    placement_lines = []
    for peer_id in [int(value) for value in plan["peer_order"]]:
        shards = sorted(shard for shard, owner in final_mapping.items() if owner == peer_id)
        role = "controller" if peer_id == int(plan["controller_peer_id"]) else "worker"
        placement_lines.append(f"{peer_hosts[peer_id]} {role}: {shards}")

    report = [
        "# Orion 物理分片负载均衡正式确认结果",
        "",
        "## 结论",
        "",
        (
            "在固定的 4 台物理机、32 个逻辑分片合同下，`swap_c8_0`（将 shard 8 与 "
            "shard 0 对换）正式战胜原 Controller-aware v1。它是目前 8 个已在线测试 "
            "placement 中的最佳结果，但这不是数学意义上的全局最优证明。"
        ),
        "",
        "| 策略 | QPS | 相对 v1 | Recall@10 | knee concurrency | 饱和节点 |",
        "|---|---:|---:|---:|---:|---|",
        *[
            f"| {name} | {qps_value:.2f} | {delta:+.2f}% | {recall:.5f} | {concurrency} | {cpu} |"
            for name, qps_value, delta, recall, concurrency, cpu in strategy_rows
        ],
        "",
        (
            f"严格判据通过：finalist 最低 phase 为 "
            f"{min(float(row['qps_mean']) for row in finalist):.2f} QPS，高于 baseline 最高 "
            f"phase 的 {max(float(row['qps_mean']) for row in baseline):.2f} QPS。"
        ),
        "",
        "## 固定实验合同",
        "",
        "- 数据集：GloVe-200-angular，Cosine，top-10。",
        "- 4 台物理机，32 个逻辑分片；每台 Qdrant 16 核，总计 64 核。",
        "- Orion artifact：`generation-3248141.json`，`upper_k=48, base_ef=50, factor=14`。",
        "- batch=200；每阶段重新测 9,000-query held-out Recall。",
        "- 每阶段完整 concurrency knee sweep，随后 5 次×20 秒重复测量。",
        "",
        "## 五阶段正式证据",
        "",
        "| Phase | 策略 | QPS | CV | Recall@10 | C | P50 / P95 / P99 (ms) |",
        "|---|---|---:|---:|---:|---:|---:|",
        *[
            (
                f"| {row['phase_id']} | {row['strategy']} | {float(row['qps_mean']):.2f} | "
                f"{100.0 * float(row['qps_cv']):.3f}% | "
                f"{float(row['heldout_recall']['recall_at_10']):.5f} | "
                f"{row['selected_concurrency']} | "
                f"{float(row['batch_latency_ms_mean_across_repeats']['p50']):.2f} / "
                f"{float(row['batch_latency_ms_mean_across_repeats']['p95']):.2f} / "
                f"{float(row['batch_latency_ms_mean_across_repeats']['p99']):.2f} |"
            )
            for row in phase_records
        ],
        "",
        f"- v1 A/B 漂移：{pct(baseline_drift)}。",
        f"- finalist A/B 漂移：{pct(finalist_drift)}。",
        (
            "- `swap_c8_0` 在 knee concurrency=4 下获得更高 QPS，同时 controller 接近 "
            "16 核饱和；其延迟不能与 concurrency=2 的 v1 直接解释为同并发延迟改善。"
        ),
        (
            "- `swap_19_15` 的单次正式观测提升约 "
            f"{backup_improvement:.2f}%，且 controller 压力较低，但没有执行双端正式确认。"
        ),
        "",
        "## 最终 placement",
        "",
        "```text",
        *placement_lines,
        "```",
        "",
        "## 负结果与边界",
        "",
        (
            "- 第二轮全局 learned candidates 已被在线证伪：`learned_robust` 和 "
            "`learned_robust_workcap5500` 都低于当轮 v1，不能因本轮成功而删除该负结果。"
        ),
        (
            "- `swap_6_3` 在局部筛选中只比同轮 v1 两端均值高约 "
            f"{100.0 * (screen_v3['strategy_summary']['swap_6_3']['qps_mean_across_phases'] / screen_v3['strategy_summary'][BASELINE]['qps_mean_across_phases'] - 1.0):.2f}%，未进入正式确认。"
        ),
        "- 离线 ridge/work proxy 仅用于选候选；在线 QPS 才是排序依据。",
        (
            "- 当前结论是“8 个已测试 placement 中最佳，并正式战胜 v1”；完整双交换空间、"
            "更多随机种子、其他数据集和其他资源合同仍未穷尽。"
        ),
        "",
        "## 证据位置",
        "",
        "- `online-v4/formal-summary.json`：正式判据与聚合结果。",
        "- `online-v4/phases/*/screen.json`：逐阶段 Recall、sweep、5×20 秒、CPU 与延迟。",
        "- `online-v4/evidence-audit.json`：完整门禁审计。",
        "- `confirm-v1-plan/placements.json`：正式阶段顺序与 placement。",
        "- `online-v3/screen-local-summary.json`：三个局部交换的初筛结果。",
        "- `online-v2/screen-summary.json`：全局 learned candidates 的负结果。",
        "- collection 保留为 `orion_lb_r090_p24_20260825_v2`；资源恢复为 PASS。",
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
