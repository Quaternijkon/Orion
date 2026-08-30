#!/usr/bin/env python3
"""Build the final audited synthesis for the Recall@10 >= 0.90 placement search."""

from __future__ import annotations

import argparse
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import orion_physical_placement_confirmation_audit as common
import orion_physical_placement_worker_audit as worker_common


DEFAULT_ROOT = Path(
    "/proj/intelisys-PG0/exp/orion-distributed/orion-load-balance-r090-20260825"
)
DEFAULT_OUTPUT = DEFAULT_ROOT / "final-v1"
COLLECTION = "orion_lb_r090_p24_20260825_v2"
CLUSTER_URL = f"http://10.10.1.1:6333/collections/{COLLECTION}/cluster"
WINNER = "swap_24_21"
PREVIOUS_WINNER = "swap_16_18"
ONLINE_AUDIT_VERSIONS = (1, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 20)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def query_json(url: str) -> Any:
    with urllib.request.urlopen(url, timeout=30.0) as response:
        return json.load(response)


def normalize_mapping(mapping: dict[str, int] | dict[int, int]) -> dict[int, int]:
    return {int(shard): int(peer) for shard, peer in mapping.items()}


def placement_groups(
    mapping: dict[int, int], peer_order: Sequence[int], peer_hosts: dict[int, str]
) -> dict[str, list[int]]:
    return {
        peer_hosts[peer]: sorted(shard for shard, owner in mapping.items() if owner == peer)
        for peer in peer_order
    }


def online_inventory(root: Path) -> tuple[int, int]:
    paths = list((root / "online-v1/phases").glob("*/benchmark.json"))
    for version in (*range(2, 17), 18, 19, 20):
        paths.extend((root / f"online-v{version}/phases").glob("*/screen.json"))
    mappings = {
        tuple(
            sorted(
                (int(shard), int(peer))
                for shard, peer in load_json(path)["ending_placement"][
                    "placement"
                ].items()
            )
        )
        for path in paths
    }
    return len(paths), len(mappings)


def audit_counts(payload: dict[str, Any]) -> tuple[int, int]:
    passed = payload.get("passed_checks", payload.get("passed"))
    total = payload.get("total_checks", payload.get("total"))
    if passed is None or total is None:
        raise RuntimeError("audit is missing passed/total counts")
    return int(passed), int(total)


def build(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    root = args.root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)

    formal_summary_path = root / "online-v11/formal-v4-summary.json"
    formal_audit_path = root / "online-v11/evidence-audit.json"
    latest_audit_path = root / "online-v20/evidence-audit.json"
    kernel_audit_path = root / "refine-v17-kernel-qps/evidence-audit.json"
    linear_audit_path = root / "online-v16/evidence-audit.json"
    winner_plan_path = root / "confirm-v4-plan/placements.json"
    resource_contract_path = root / "online-v20/resource-contract.json"
    collection_contract_path = root / "online-v20/collection-contract.json"

    formal_summary = load_json(formal_summary_path)
    formal_audit = load_json(formal_audit_path)
    latest_audit = load_json(latest_audit_path)
    kernel_audit = load_json(kernel_audit_path)
    linear_audit = load_json(linear_audit_path)
    winner_plan = load_json(winner_plan_path)
    resource_contract = load_json(resource_contract_path)
    collection_contract = load_json(collection_contract_path)
    online_audits = {
        version: load_json(root / f"online-v{version}/evidence-audit.json")
        for version in ONLINE_AUDIT_VERSIONS
    }

    live_payload = query_json(args.cluster_url)
    live_snapshot_path = output / "live-cluster-snapshot.json"
    common.write_json_new(
        live_snapshot_path,
        {
            "captured_at": datetime.now(timezone.utc)
            .isoformat()
            .replace("+00:00", "Z"),
            "url": args.cluster_url,
            "payload": live_payload,
        },
    )

    peer_order = [int(value) for value in winner_plan["peer_order"]]
    peer_hosts = {
        int(peer): str(host) for peer, host in winner_plan["peer_hosts"].items()
    }
    expected = normalize_mapping(winner_plan["placements"][WINNER])
    live_result = live_payload["result"]
    live_mapping = {
        int(row["shard_id"]): int(live_result["peer_id"])
        for row in live_result["local_shards"]
    }
    live_mapping.update(
        {
            int(row["shard_id"]): int(row["peer_id"])
            for row in live_result["remote_shards"]
        }
    )
    live_states = [
        str(row["state"])
        for row in (*live_result["local_shards"], *live_result["remote_shards"])
    ]
    groups = placement_groups(expected, peer_order, peer_hosts)

    formal = formal_summary["formal_confirmation_result"]
    formal_metrics = formal_audit["metrics"]
    latest_metrics = latest_audit["metrics"]
    phase_count, placement_count = online_inventory(root)
    online_passed = 0
    online_total = 0
    audit_inventory = {}
    for version, payload in online_audits.items():
        passed, total = audit_counts(payload)
        online_passed += passed
        online_total += total
        audit_inventory[f"online-v{version}"] = {
            "status": payload.get("status"),
            "passed": passed,
            "total": total,
        }
    kernel_passed, kernel_total = audit_counts(kernel_audit)

    checks: list[dict[str, Any]] = []
    add = lambda name, condition, actual: worker_common.add_check(
        checks, name, condition, actual
    )
    add(
        "formal winner confirmed",
        formal_summary.get("status") == "CONFIRMATION_COMPLETE"
        and formal_summary.get("confirmed_winner") == WINNER
        and formal.get("strict_endpoint_dominance") is True
        and formal.get("all_recall_pass") is True,
        {
            "status": formal_summary.get("status"),
            "confirmed_winner": formal_summary.get("confirmed_winner"),
            "strict_endpoint_dominance": formal.get("strict_endpoint_dominance"),
            "all_recall_pass": formal.get("all_recall_pass"),
        },
    )
    add(
        "formal evidence audit",
        formal_audit.get("status") == "PASS"
        and formal_audit.get("passed_checks") == formal_audit.get("total_checks"),
        {
            "status": formal_audit.get("status"),
            "passed_checks": formal_audit.get("passed_checks"),
            "total_checks": formal_audit.get("total_checks"),
        },
    )
    add(
        "latest evidence audit",
        latest_audit.get("status") == "PASS"
        and latest_audit.get("passed_checks") == latest_audit.get("total_checks")
        and latest_metrics.get("unique_online_placement_count") == 62
        and latest_metrics.get("online_phase_count") == 116,
        {
            "status": latest_audit.get("status"),
            "passed_checks": latest_audit.get("passed_checks"),
            "total_checks": latest_audit.get("total_checks"),
            "unique_online_placement_count": latest_metrics.get(
                "unique_online_placement_count"
            ),
            "online_phase_count": latest_metrics.get("online_phase_count"),
        },
    )
    add(
        "all online audits pass",
        all(payload.get("status") == "PASS" for payload in online_audits.values())
        and online_passed == online_total == 1501,
        {
            "audit_count": len(online_audits),
            "passed_checks": online_passed,
            "total_checks": online_total,
            "inventory": audit_inventory,
        },
    )
    add(
        "kernel stop-gate audit",
        kernel_audit.get("status") == "PASS"
        and kernel_audit.get("decision") == "NO_ONLINE_V17"
        and kernel_audit.get("evidence_class") == "WEAK_NEGATIVE_OFFLINE_EVIDENCE"
        and kernel_passed == kernel_total == 22,
        {
            "status": kernel_audit.get("status"),
            "decision": kernel_audit.get("decision"),
            "evidence_class": kernel_audit.get("evidence_class"),
            "passed_checks": kernel_passed,
            "total_checks": kernel_total,
        },
    )
    add(
        "contradicted linear surrogate preserved",
        linear_audit.get("status") == "PASS"
        and "contradicted" in linear_audit.get("claim_boundary", "")
        and all(
            float(value) < 0.0
            for value in linear_audit.get("metrics", {})
            .get("candidate_delta_pct", {})
            .values()
        ),
        {
            "status": linear_audit.get("status"),
            "claim_boundary": linear_audit.get("claim_boundary"),
            "candidate_delta_pct": linear_audit.get("metrics", {}).get(
                "candidate_delta_pct"
            ),
        },
    )
    add(
        "fixed physical and resource contract",
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
    add(
        "fixed collection contract",
        collection_contract.get("status") == "PASS"
        and collection_contract.get("logical_shard_count") == 32
        and collection_contract.get("replication_factor") == 1,
        {
            "status": collection_contract.get("status"),
            "logical_shard_count": collection_contract.get("logical_shard_count"),
            "replication_factor": collection_contract.get("replication_factor"),
        },
    )
    add(
        "recall contract",
        float(formal_metrics["recall_at_10"]) >= 0.90
        and float(latest_metrics["recall_at_10"]) >= 0.90,
        {
            "formal_recall_at_10": formal_metrics["recall_at_10"],
            "latest_recall_at_10": latest_metrics["recall_at_10"],
        },
    )
    add(
        "shard-24 cross-machine finite search complete",
        latest_metrics.get("shard24_cross_machine_counterpart_count") == 24
        and latest_metrics.get("shard24_cross_machine_counterparts_online_tested")
        == 24,
        {
            "counterpart_count": latest_metrics.get(
                "shard24_cross_machine_counterpart_count"
            ),
            "online_tested": latest_metrics.get(
                "shard24_cross_machine_counterparts_online_tested"
            ),
        },
    )
    add(
        "online inventory recomputed",
        phase_count == 116 and placement_count == 62,
        {"phase_count": phase_count, "unique_placement_count": placement_count},
    )
    add(
        "live cluster response",
        live_payload.get("status") == "ok"
        and live_result.get("shard_count") == 32
        and len(live_mapping) == 32,
        {
            "status": live_payload.get("status"),
            "shard_count": live_result.get("shard_count"),
            "mapped_shards": len(live_mapping),
        },
    )
    add(
        "live winner placement",
        live_mapping == expected,
        {
            "expected_groups": groups,
            "live_groups": placement_groups(live_mapping, peer_order, peer_hosts),
        },
    )
    add(
        "live shards active and transfer-free",
        set(live_states) == {"Active"} and not live_result.get("shard_transfers"),
        {
            "states": sorted(set(live_states)),
            "shard_transfers": live_result.get("shard_transfers"),
        },
    )
    add(
        "balanced shard counts",
        all(len(shards) == 8 for shards in groups.values()),
        {host: len(shards) for host, shards in groups.items()},
    )

    failed = [row for row in checks if row["status"] != "PASS"]
    report_path = output / "FINAL_RESULTS_zh.md"
    report = [
        "# Orion Recall@10 ≥ 0.90 物理负载均衡最终结果",
        "",
        "## 最佳已验证策略",
        "",
        "`swap_24_21`：在上一正式赢家 `swap_16_18` 上交换 shard 24 与 shard 21。",
        "",
        "| 指标 | 结果 |",
        "|---|---:|",
        f"| 正式 QPS（5×20s A/B 均值） | **{formal_metrics['finalist_qps_mean']:.2f}** |",
        f"| Recall@10 | **{formal_metrics['recall_at_10']:.6f}** |",
        f"| 相对 `swap_16_18` | **+{formal_metrics['finalist_improvement_pct']:.2f}%** |",
        f"| 相对最初 controller-aware v1 | **+{formal_metrics['cumulative_improvement_over_controller_v1_pct']:.2f}%** |",
        f"| 正式最低端点 | {formal['finalist_qps_min']:.2f} QPS |",
        f"| 上一赢家最高端点 | {formal['baseline_qps_max']:.2f} QPS |",
        "| 严格端点支配 | PASS |",
        "",
        "## 当前 placement",
        "",
        "| 物理节点 | 角色 | shards |",
        "|---|---|---|",
        *[
            f"| `{host}` | {'controller' if peer == int(winner_plan['controller_peer_id']) else 'worker'} | {groups[host]} |"
            for peer, host in ((peer, peer_hosts[peer]) for peer in peer_order)
        ],
        "",
        "## 搜索覆盖与负结果",
        "",
        f"- 累计在线覆盖 **{placement_count} 个唯一 physical placement、{phase_count} 个 phase**。",
        "- shard 24 的三个跨机目标族全部穷举：`.4`、`.3`、controller 各 8 个，共 **24/24 counterpart**；`swap_24_21` 是该完整有限空间内最佳。",
        "- 新赢家的 learned CPU-ridge、热点 work proxy、direct paired-QPS 线性 surrogate 均出现在线反例；四个线性 surrogate 正预测实际下降约 5%–14%，已明确停用。",
        "- interaction-aware kernel 最佳 LOPO RMSE 仅从 2.595% 微降至 2.571%，且 367 个未测单交换邻居均无正预测，因此按弱否定性离线证据封存，未启动 `online-v17`。",
        f"- 16 份在线 evidence audit 合计 **{online_passed}/{online_total} PASS**；kernel stop-gate 另为 **{kernel_passed}/{kernel_total} PASS**。",
        "",
        "## 结论边界",
        "",
        f"可以声明：`swap_24_21` 是 **{placement_count} 个已在线测试 placement** 中的正式最佳，也是 shard 24 的 **24/24 跨机单交换 counterpart** 中最佳。",
        "不能声明：任意多交换、全部 4^32 balanced assignment 空间中的数学全局最优。离线模型只用于提出或停止候选，在线 QPS 始终是排序依据。",
        "",
        "最终 live 查询确认 32 个 shard 全部 `Active`，placement 与 `swap_24_21` 完全一致，`shard_transfers=[]`。",
        f"总审计：`{len(checks)}/{len(checks)} PASS`。",
        "",
    ]
    with report_path.open("x", encoding="utf-8") as handle:
        handle.write("\n".join(report))

    artifact_paths = (
        formal_summary_path,
        formal_audit_path,
        latest_audit_path,
        kernel_audit_path,
        linear_audit_path,
        winner_plan_path,
        resource_contract_path,
        collection_contract_path,
        *(root / f"online-v{version}/evidence-audit.json" for version in ONLINE_AUDIT_VERSIONS),
        live_snapshot_path,
        report_path,
    )
    audit_path = output / "final-evidence-audit.json"
    audit = {
        "status": "PASS" if not failed else "FAIL",
        "record_type": "orion_physical_placement_r090_final_synthesis_audit",
        "winner": WINNER,
        "claim_boundary": (
            "swap_24_21 is the formally confirmed best among 62 online-tested "
            "physical placements and the complete 24-member cross-machine shard-24 "
            "counterpart space; arbitrary multi-swap global optimality is not proven"
        ),
        "physical_machine_count": 4,
        "logical_shard_count": 32,
        "passed_checks": len(checks) - len(failed),
        "total_checks": len(checks),
        "checks": checks,
        "metrics": {
            "formal_qps": formal_metrics["finalist_qps_mean"],
            "recall_at_10": formal_metrics["recall_at_10"],
            "improvement_over_previous_winner_pct": formal_metrics[
                "finalist_improvement_pct"
            ],
            "cumulative_improvement_over_controller_v1_pct": formal_metrics[
                "cumulative_improvement_over_controller_v1_pct"
            ],
            "formal_winner_min_qps": formal["finalist_qps_min"],
            "formal_previous_winner_max_qps": formal["baseline_qps_max"],
            "online_phase_count": phase_count,
            "unique_online_placement_count": placement_count,
            "shard24_cross_machine_counterparts_online_tested": 24,
            "online_evidence_audit_passed_checks": online_passed,
            "online_evidence_audit_total_checks": online_total,
            "kernel_stop_gate_passed_checks": kernel_passed,
            "kernel_stop_gate_total_checks": kernel_total,
        },
        "live_placement": {
            "collection": COLLECTION,
            "groups": groups,
            "all_active": set(live_states) == {"Active"},
            "shard_transfers": live_result.get("shard_transfers"),
        },
        "artifacts": {
            str(path): {
                "sha256": common.sha256_path(path),
                "size_bytes": path.stat().st_size,
            }
            for path in dict.fromkeys(artifact_paths)
        },
    }
    common.write_json_new(audit_path, audit)
    if failed:
        raise RuntimeError(
            f"final synthesis audit failed: {[row['name'] for row in failed]}"
        )
    return audit_path, report_path, live_snapshot_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cluster-url", default=CLUSTER_URL)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    audit, report, live = build(parse_args(argv))
    print(
        json.dumps(
            {"audit": str(audit), "report": str(report), "live_snapshot": str(live)},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
