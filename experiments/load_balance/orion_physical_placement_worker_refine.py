#!/usr/bin/env python3
"""Build a worker-only refinement plan around the confirmed Orion placement."""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import orion_physical_placement_local_refine as local
import orion_physical_placement_refine as refine


DEFAULT_ROOT = Path(
    "/proj/intelisys-PG0/exp/orion-distributed/orion-load-balance-r090-20260825"
)
DEFAULT_LAYOUT = Path(
    "/users/dry/orion-distributed/native-20260818-scale4-32-r080-095-v1/"
    "artifacts/scale32/orion-r090-u48-b50-f14-g3248141"
)
DEFAULT_OUTPUT = DEFAULT_ROOT / "refine-v8-worker"
BASELINE = "swap_16_18"
CANDIDATE_SWAPS = {
    "swap_16_10": (16, 10),
    "swap_25_10": (25, 10),
    "swap_16_17": (16, 17),
    "swap_25_17": (25, 17),
}
SCREEN_PHASES = (
    ("swap-16-18-worker-a", BASELINE),
    ("swap-16-10", "swap_16_10"),
    ("swap-25-10", "swap_25_10"),
    ("swap-16-17", "swap_16_17"),
    ("swap-25-17", "swap_25_17"),
    ("swap-16-18-worker-b", BASELINE),
)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build(args: argparse.Namespace) -> Path:
    root = args.root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    layout_dir = args.layout_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)

    source_plan_path = root / "confirm-v2-plan/placements.json"
    source_summary_path = root / "online-v6/formal-v2-summary.json"
    source_audit_path = root / "online-v6/evidence-audit.json"
    neighbor_summary_path = root / "online-v7/screen-neighbor-v2-summary.json"
    neighbor_audit_path = root / "online-v7/evidence-audit.json"
    trace_path = root / "trace/p24-per-query.json"
    layout_manifest_path = layout_dir / "build-manifest.json"

    source_plan = load_json(source_plan_path)
    source_summary = load_json(source_summary_path)
    source_audit = load_json(source_audit_path)
    neighbor_summary = load_json(neighbor_summary_path)
    neighbor_audit = load_json(neighbor_audit_path)
    trace = load_json(trace_path)
    layout_manifest = load_json(layout_manifest_path)
    if source_summary.get("confirmed_winner") != BASELINE:
        raise RuntimeError("formal source summary does not confirm swap_16_18")
    if source_audit.get("status") != "PASS" or neighbor_audit.get("status") != "PASS":
        raise RuntimeError("source evidence audit is not PASS")
    if neighbor_summary.get("screen_winner") != BASELINE:
        raise RuntimeError("complete-neighbor screen did not retain swap_16_18")

    peer_order = [int(value) for value in source_plan["peer_order"]]
    peer_hosts = {int(peer): str(host) for peer, host in source_plan["peer_hosts"].items()}
    controller_peer = int(source_plan["controller_peer_id"])
    host_to_peer = {host: peer for peer, host in peer_hosts.items()}
    baseline = local.normalize_mapping(source_plan["placements"][BASELINE])
    if set(baseline) != set(range(32)) or set(baseline.values()) != set(peer_order):
        raise RuntimeError("invalid baseline placement")

    baseline_cpu_rows = neighbor_summary["strategy_summary"][BASELINE]["cpu_cores_by_phase"]
    cpu_by_host = {
        host: statistics.fmean(float(row[host]) for row in baseline_cpu_rows)
        for host in peer_hosts.values()
    }
    hot_host = "10.10.1.2"
    if max(
        (host for host in cpu_by_host if host != peer_hosts[controller_peer]),
        key=cpu_by_host.__getitem__,
    ) != hot_host:
        raise RuntimeError(f"expected worker hotspot {hot_host}, observed {cpu_by_host}")

    counts = [int(value) for value in layout_manifest["routing"]["shard_counts"]]
    shard_features = refine.shard_features(trace, counts)
    work_by_shard = [1000.0 * row["work_1000"] for row in shard_features]
    candidate_mappings = {BASELINE: baseline}
    candidate_metrics: dict[str, Any] = {
        BASELINE: {
            "groups": local.placement_groups(baseline, peer_order),
            "selection_reason": "formally confirmed current winner and A/B control",
            "cpu_cores_by_host": cpu_by_host,
        }
    }

    controller_shards = sorted(shard for shard, peer in baseline.items() if peer == controller_peer)
    for name, (hot_shard, cool_shard) in CANDIDATE_SWAPS.items():
        hot_peer = baseline[hot_shard]
        cool_peer = baseline[cool_shard]
        if peer_hosts[hot_peer] != hot_host:
            raise RuntimeError(f"{name}: shard {hot_shard} is not on {hot_host}")
        if peer_hosts[cool_peer] not in {"10.10.1.3", "10.10.1.4"}:
            raise RuntimeError(f"{name}: shard {cool_shard} is not on a cooler worker")
        if controller_peer in {hot_peer, cool_peer}:
            raise RuntimeError(f"{name}: worker-only candidate touches controller")
        mapping = local.swap_mapping(baseline, hot_shard, cool_shard)
        if sorted(shard for shard, peer in mapping.items() if peer == controller_peer) != controller_shards:
            raise RuntimeError(f"{name}: controller placement changed")
        groups = local.placement_groups(mapping, peer_order)
        if any(len(group) != 8 for group in groups):
            raise RuntimeError(f"{name}: shard count per peer changed")
        work_shift = work_by_shard[hot_shard] - work_by_shard[cool_shard]
        if work_shift <= 0:
            raise RuntimeError(f"{name}: proposed work shift is not away from hotspot")
        candidate_mappings[name] = mapping
        candidate_metrics[name] = {
            "groups": groups,
            "swap_from_baseline": {
                "hot_shard": hot_shard,
                "cool_shard": cool_shard,
                "source_host": peer_hosts[hot_peer],
                "target_host": peer_hosts[cool_peer],
            },
            "selection_reason": (
                "high-concurrency CPU-directed worker-only micro-adjustment; move work "
                "off 10.10.1.2 without changing controller ownership"
            ),
            "source_cpu_cores": cpu_by_host[peer_hosts[hot_peer]],
            "target_cpu_cores": cpu_by_host[peer_hosts[cool_peer]],
            "hot_shard_work_units": work_by_shard[hot_shard],
            "cool_shard_work_units": work_by_shard[cool_shard],
            "work_units_shifted_from_hot_worker": work_shift,
            "moved_physical_vector_copies": counts[hot_shard] + counts[cool_shard],
            "controller_placement_unchanged": True,
        }

    placements_path = output / "placements.json"
    local.write_json_new(
        placements_path,
        {
            "format_version": 1,
            "physical_machine_count": 4,
            "logical_shard_count": 32,
            "controller_peer_id": controller_peer,
            "peer_order": peer_order,
            "peer_hosts": source_plan["peer_hosts"],
            "screen_phases": [
                {"phase_id": phase_id, "strategy": strategy}
                for phase_id, strategy in SCREEN_PHASES
            ],
            "placements": {
                name: {str(shard): peer for shard, peer in sorted(mapping.items())}
                for name, mapping in candidate_mappings.items()
            },
        },
    )
    metrics_path = output / "candidate-metrics.json"
    local.write_json_new(metrics_path, candidate_metrics)
    rationale_path = output / "RATIONALE_zh.md"
    rationale_lines = [
        "# Orion worker-only placement 微调计划",
        "",
        "固定合同：4 台物理机、32 个逻辑分片、每台 Qdrant 16 核，总计 64 核；Recall@10 目标不低于 0.90。",
        "",
        f"`swap_16_18` 同轮 A/B 的 worker 热点为 `{hot_host}`（平均 {cpu_by_host[hot_host]:.3f} 核）。",
        "本计划保持 controller 的 8 个 shard 完全不变，只把热点 worker 的较重 shard 与 `.3/.4` 的较轻 shard 对换。",
        "",
        "| 候选 | 源 → 目标 | work shift | 源/目标 CPU（核） | 搬移向量副本 |",
        "|---|---|---:|---:|---:|",
        *[
            (
                f"| `{name}` | {row['swap_from_baseline']['source_host']} → "
                f"{row['swap_from_baseline']['target_host']} | "
                f"{row['work_units_shifted_from_hot_worker']:.2f} | "
                f"{row['source_cpu_cores']:.3f} / {row['target_cpu_cores']:.3f} | "
                f"{row['moved_physical_vector_copies']} |"
            )
            for name, row in candidate_metrics.items()
            if name != BASELINE
        ],
        "",
        "顺序：baseline A → 16↔10 → 25↔10 → 16↔17 → 25↔17 → baseline B。",
        "离线 work shift 只说明候选方向；最终排序以在线 held-out Recall 和 QPS 为准。",
        "",
    ]
    with rationale_path.open("x", encoding="utf-8") as handle:
        handle.write("\n".join(rationale_lines))

    outputs = (placements_path, metrics_path, rationale_path)
    manifest_path = output / "manifest.json"
    local.write_json_new(
        manifest_path,
        {
            "format_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "status": "OFFLINE_REFINEMENT_ONLY",
            "claim_boundary": (
                "CPU-directed worker-only candidate plan; work shift is a selector and "
                "online QPS remains authoritative"
            ),
            "physical_machine_count": 4,
            "logical_shard_count": 32,
            "baseline": BASELINE,
            "candidate_count": len(CANDIDATE_SWAPS),
            "controller_placement_unchanged": True,
            "inputs": {
                "source_plan": {"path": str(source_plan_path), "sha256": local.sha256_path(source_plan_path)},
                "source_summary": {"path": str(source_summary_path), "sha256": local.sha256_path(source_summary_path)},
                "source_audit": {"path": str(source_audit_path), "sha256": local.sha256_path(source_audit_path)},
                "neighbor_summary": {"path": str(neighbor_summary_path), "sha256": local.sha256_path(neighbor_summary_path)},
                "neighbor_audit": {"path": str(neighbor_audit_path), "sha256": local.sha256_path(neighbor_audit_path)},
                "trace": {"path": str(trace_path), "sha256": local.sha256_path(trace_path)},
                "layout_manifest": {"path": str(layout_manifest_path), "sha256": local.sha256_path(layout_manifest_path)},
            },
            "outputs": {
                path.name: {"path": str(path), "sha256": local.sha256_path(path)}
                for path in outputs
            },
        },
    )
    return manifest_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--layout-dir", type=Path, default=DEFAULT_LAYOUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    manifest = build(parse_args(argv))
    print(json.dumps({"manifest": str(manifest)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
