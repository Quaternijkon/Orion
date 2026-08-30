#!/usr/bin/env python3
"""Build the remaining high-value shard-24-preserving hotspot candidates."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import orion_physical_placement_local_refine as local


DEFAULT_ROOT = Path(
    "/proj/intelisys-PG0/exp/orion-distributed/orion-load-balance-r090-20260825"
)
DEFAULT_OUTPUT = DEFAULT_ROOT / "refine-v15-hotspot-followup"
BASELINE = "swap_24_21"
SELECTED_PAIRS = ((6, 29), (23, 30), (6, 5), (20, 22))


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build(args: argparse.Namespace) -> Path:
    root = args.root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)

    source_plan_path = root / "confirm-v4-plan/placements.json"
    source_audit_path = root / "online-v14/evidence-audit.json"
    universe_path = root / "refine-v13-hotspot/candidate-universe.json"
    source_plan = load_json(source_plan_path)
    source_audit = load_json(source_audit_path)
    universe = load_json(universe_path)
    if source_audit.get("status") != "PASS":
        raise RuntimeError("hotspot confirmation audit is not PASS")
    baseline = local.normalize_mapping(source_plan["placements"][BASELINE])
    peer_order = [int(value) for value in source_plan["peer_order"]]
    rows = {
        (int(row["source_shard"]), int(row["target_shard"])): row
        for row in universe["candidates"]
    }
    missing = [pair for pair in SELECTED_PAIRS if pair not in rows]
    if missing:
        raise RuntimeError(f"selected pairs missing from hotspot universe: {missing}")

    placements = {BASELINE: baseline}
    metrics: dict[str, Any] = {
        BASELINE: {
            "groups": local.placement_groups(baseline, peer_order),
            "selection_reason": "formally confirmed current winner and A/B control",
        }
    }
    names = []
    for source_shard, target_shard in SELECTED_PAIRS:
        row = rows[(source_shard, target_shard)]
        name = f"hotspot2_{source_shard}_{target_shard}"
        names.append(name)
        mapping = local.swap_mapping(baseline, source_shard, target_shard)
        if mapping[24] != baseline[24]:
            raise RuntimeError(f"{name} moved shard 24")
        placements[name] = mapping
        metrics[name] = {
            **row,
            "groups": local.placement_groups(mapping, peer_order),
            "selection_reason": (
                "remaining controller-offload candidate ordered by model consensus"
                if row["target_host"] == "10.10.1.1"
                else "remaining target-.3 candidate with best trace-p95 rank"
            ),
            "shard_24_preserved": True,
        }

    phases = [("swap-24-21-hotspot2-a", BASELINE)] + [
        (name.replace("_", "-"), name) for name in names
    ] + [("swap-24-21-hotspot2-b", BASELINE)]
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
    metrics_path = output / "candidate-metrics.json"
    local.write_json_new(metrics_path, metrics)
    rationale_path = output / "RATIONALE_zh.md"
    lines = [
        "# swap_24_21 热点卸载后续候选",
        "",
        "不重复已正式否定的 hotspot_10_18；测试剩余三个 controller 卸载候选，以及 `.3` 方向 trace-p95 最优候选。",
        "所有候选保留 shard 24，在线 QPS 为唯一排序依据。",
        "",
        "| 候选 | swap | target | work shift | global consensus | trace p95 |",
        "|---|---|---|---:|---:|---:|",
        *[
            (
                f"| `{name}` | {row['source_shard']}↔{row['target_shard']} | "
                f"{row['target_host']} | {row['work_units_shifted_from_source']:.2f} | "
                f"{row['consensus_rank']} | {row['trace_p95_rank']} |"
            )
            for name, row in metrics.items()
            if name != BASELINE
        ],
        "",
    ]
    with rationale_path.open("x", encoding="utf-8") as handle:
        handle.write("\n".join(lines))

    outputs = (placements_path, metrics_path, rationale_path)
    manifest_path = output / "manifest.json"
    local.write_json_new(
        manifest_path,
        {
            "format_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "status": "OFFLINE_REFINEMENT_ONLY",
            "claim_boundary": "hotspot follow-up plan; offline ranks do not establish online QPS",
            "physical_machine_count": 4,
            "logical_shard_count": 32,
            "selected_candidate_count": len(names),
            "selected_strategies": names,
            "inputs": {
                "source_plan": {"path": str(source_plan_path), "sha256": local.sha256_path(source_plan_path)},
                "source_audit": {"path": str(source_audit_path), "sha256": local.sha256_path(source_audit_path)},
                "candidate_universe": {"path": str(universe_path), "sha256": local.sha256_path(universe_path)},
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
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    manifest = build(parse_args(argv))
    print(json.dumps({"manifest": str(manifest)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
