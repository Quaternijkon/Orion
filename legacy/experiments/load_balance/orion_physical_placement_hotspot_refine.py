#!/usr/bin/env python3
"""Build a shard-24-preserving hotspot-offload plan around swap_24_21."""

from __future__ import annotations

import argparse
import json
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
DEFAULT_OUTPUT = DEFAULT_ROOT / "refine-v13-hotspot"
BASELINE = "swap_24_21"
MAX_WORK_SHIFT = 180.0
TARGET_HOSTS = ("10.10.1.1", "10.10.1.3")


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

    source_plan_path = root / "confirm-v4-plan/placements.json"
    source_summary_path = root / "online-v11/formal-v4-summary.json"
    source_audit_path = root / "online-v12/evidence-audit.json"
    neighborhood_path = root / "refine-v12-neighbor/one-swap-neighborhood.json"
    trace_path = root / "trace/p24-per-query.json"
    layout_manifest_path = layout_dir / "build-manifest.json"
    source_plan = load_json(source_plan_path)
    source_summary = load_json(source_summary_path)
    source_audit = load_json(source_audit_path)
    neighborhood = load_json(neighborhood_path)
    trace = load_json(trace_path)
    layout_manifest = load_json(layout_manifest_path)
    if source_summary.get("confirmed_winner") != BASELINE:
        raise RuntimeError("formal source does not confirm swap_24_21")
    if source_audit.get("status") != "PASS":
        raise RuntimeError("new-winner neighbor audit is not PASS")
    if neighborhood.get("baseline") != BASELINE or neighborhood.get("evaluated_swaps") != 384:
        raise RuntimeError("unexpected source neighborhood")

    peer_order = [int(value) for value in source_plan["peer_order"]]
    peer_hosts = {int(peer): str(host) for peer, host in source_plan["peer_hosts"].items()}
    baseline = local.normalize_mapping(source_plan["placements"][BASELINE])
    source_peer = next(peer for peer, host in peer_hosts.items() if host == "10.10.1.4")
    target_peers = {peer for peer, host in peer_hosts.items() if host in TARGET_HOSTS}
    counts = [int(value) for value in layout_manifest["routing"]["shard_counts"]]
    work_by_shard = [
        1000.0 * row["work_1000"]
        for row in refine.shard_features(trace, counts)
    ]
    row_by_pair = {
        tuple(sorted((int(row["left_shard"]), int(row["right_shard"])))): row
        for row in neighborhood["neighbors"]
    }

    eligible = []
    for source_shard, owner in baseline.items():
        if owner != source_peer or source_shard == 24:
            continue
        for target_shard, target_peer in baseline.items():
            if target_peer not in target_peers:
                continue
            work_shift = work_by_shard[source_shard] - work_by_shard[target_shard]
            if not (0.0 < work_shift <= MAX_WORK_SHIFT):
                continue
            row = row_by_pair[tuple(sorted((source_shard, target_shard)))]
            if row["already_online_tested"]:
                continue
            eligible.append(
                {
                    **row,
                    "source_shard": source_shard,
                    "target_shard": target_shard,
                    "source_host": peer_hosts[source_peer],
                    "target_host": peer_hosts[target_peer],
                    "work_units_shifted_from_source": work_shift,
                }
            )
    selected = []
    for target_host in TARGET_HOSTS:
        rows = sorted(
            (row for row in eligible if row["target_host"] == target_host),
            key=lambda row: (
                row["model_worst_rank"],
                row["model_rank_sum"],
                row["trace_p95_rank"],
                row["moved_physical_vector_copies"],
            ),
        )
        selected.extend(rows[:2])
    selected.sort(key=lambda row: row["consensus_rank"])
    if len(selected) != 4:
        raise RuntimeError(f"expected four selected hotspot candidates, got {len(selected)}")

    placements = {BASELINE: baseline}
    metrics: dict[str, Any] = {
        BASELINE: {
            "groups": local.placement_groups(baseline, peer_order),
            "selection_reason": "formally confirmed current winner and A/B control",
        }
    }
    names = []
    for row in selected:
        name = f"hotspot_{row['source_shard']}_{row['target_shard']}"
        names.append(name)
        mapping = local.swap_mapping(
            baseline, row["source_shard"], row["target_shard"]
        )
        if mapping[24] != baseline[24]:
            raise RuntimeError(f"{name} moved shard 24")
        placements[name] = mapping
        metrics[name] = {
            **row,
            "groups": local.placement_groups(mapping, peer_order),
            "selection_reason": (
                "top-two model consensus for its target host among untested shard-24-"
                f"preserving .4 offloads with work shift <= {MAX_WORK_SHIFT:.0f}"
            ),
            "shard_24_preserved": True,
        }

    phases = [("swap-24-21-hotspot-a", BASELINE)] + [
        (name.replace("_", "-"), name) for name in names
    ] + [("swap-24-21-hotspot-b", BASELINE)]
    placements_path = output / "placements.json"
    local.write_json_new(
        placements_path,
        {
            "format_version": 1,
            "physical_machine_count": 4,
            "logical_shard_count": 32,
            "controller_peer_id": source_plan["controller_peer_id"],
            "peer_order": peer_order,
            "peer_hosts": source_plan["peer_hosts"],
            "screen_phases": [
                {"phase_id": phase_id, "strategy": strategy}
                for phase_id, strategy in phases
            ],
            "placements": {
                name: {str(shard): peer for shard, peer in sorted(mapping.items())}
                for name, mapping in placements.items()
            },
        },
    )
    universe_path = output / "candidate-universe.json"
    local.write_json_new(
        universe_path,
        {
            "eligibility": {
                "source_host": "10.10.1.4",
                "target_hosts": list(TARGET_HOSTS),
                "shard_24_preserved": True,
                "positive_work_shift_required": True,
                "maximum_work_shift": MAX_WORK_SHIFT,
                "already_online_tested_excluded": True,
            },
            "eligible_count": len(eligible),
            "selection_rule": "top two per target host by updated model consensus",
            "candidates": sorted(eligible, key=lambda row: row["consensus_rank"]),
        },
    )
    metrics_path = output / "candidate-metrics.json"
    local.write_json_new(metrics_path, metrics)
    rationale_path = output / "RATIONALE_zh.md"
    lines = [
        "# swap_24_21 热点卸载候选",
        "",
        "保留带来正式提升的 shard 24→10.10.1.4，只从 `.4` 向 controller 或 `.3` 卸载少量 work。",
        "每个目标节点选更新模型共识前两名；在线 QPS 仍是唯一排序依据。",
        "",
        "| 候选 | source→target | work shift | global consensus | model worst/sum | trace p95 |",
        "|---|---|---:|---:|---:|---:|",
        *[
            (
                f"| `{name}` | {row['source_shard']}↔{row['target_shard']} "
                f"({row['source_host']}→{row['target_host']}) | "
                f"{row['work_units_shifted_from_source']:.2f} | {row['consensus_rank']} | "
                f"{row['model_worst_rank']}/{row['model_rank_sum']} | {row['trace_p95_rank']} |"
            )
            for name, row in metrics.items()
            if name != BASELINE
        ],
        "",
    ]
    with rationale_path.open("x", encoding="utf-8") as handle:
        handle.write("\n".join(lines))

    outputs = (placements_path, universe_path, metrics_path, rationale_path)
    manifest_path = output / "manifest.json"
    local.write_json_new(
        manifest_path,
        {
            "format_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "status": "OFFLINE_REFINEMENT_ONLY",
            "claim_boundary": (
                "shard-24-preserving hotspot-offload candidate plan; offline rankings "
                "do not establish online QPS"
            ),
            "physical_machine_count": 4,
            "logical_shard_count": 32,
            "eligible_candidate_count": len(eligible),
            "selected_candidate_count": len(names),
            "selected_strategies": names,
            "inputs": {
                "source_plan": {"path": str(source_plan_path), "sha256": local.sha256_path(source_plan_path)},
                "source_summary": {"path": str(source_summary_path), "sha256": local.sha256_path(source_summary_path)},
                "source_audit": {"path": str(source_audit_path), "sha256": local.sha256_path(source_audit_path)},
                "neighborhood": {"path": str(neighborhood_path), "sha256": local.sha256_path(neighborhood_path)},
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
