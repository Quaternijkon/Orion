#!/usr/bin/env python3
"""Enumerate the shard-24-to-controller counterpart family."""

from __future__ import annotations

import argparse
import json
import random
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
DEFAULT_OUTPUT = DEFAULT_ROOT / "refine-v20-shard24-controller-family"
BASELINE = "swap_24_21"
PREVIOUS_WINNER = "swap_16_18"
SHARD = 24
PHASE_ORDER_SEED = 3_248_143


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def mapping_key(mapping: dict[int, int]) -> tuple[tuple[int, int], ...]:
    return tuple(sorted(mapping.items()))


def all_online_mappings(root: Path) -> set[tuple[tuple[int, int], ...]]:
    paths = list((root / "online-v1/phases").glob("*/benchmark.json"))
    for version in (*range(2, 17), 18, 19):
        paths.extend((root / f"online-v{version}/phases").glob("*/screen.json"))
    result = set()
    for path in paths:
        payload = load_json(path)
        result.add(
            tuple(
                sorted(
                    (int(shard), int(peer))
                    for shard, peer in payload["ending_placement"][
                        "placement"
                    ].items()
                )
            )
        )
    if len(result) != 54:
        raise RuntimeError(f"expected 54 online placements, got {len(result)}")
    return result


def build(args: argparse.Namespace) -> Path:
    root = args.root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    layout_dir = args.layout_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)

    winner_plan_path = root / "confirm-v4-plan/placements.json"
    previous_plan_path = root / "confirm-v3-plan/placements.json"
    source_audit_path = root / "online-v19/evidence-audit.json"
    trace_path = root / "trace/p24-per-query.json"
    layout_manifest_path = layout_dir / "build-manifest.json"

    winner_plan = load_json(winner_plan_path)
    previous_plan = load_json(previous_plan_path)
    source_audit = load_json(source_audit_path)
    trace = load_json(trace_path)
    layout_manifest = load_json(layout_manifest_path)
    if source_audit.get("status") != "PASS":
        raise RuntimeError("latest family audit is not PASS")

    peer_order = [int(value) for value in winner_plan["peer_order"]]
    peer_hosts = {
        int(peer): str(host) for peer, host in winner_plan["peer_hosts"].items()
    }
    controller = int(winner_plan["controller_peer_id"])
    if peer_hosts[controller] != "10.10.1.1":
        raise RuntimeError("unexpected controller host")
    worker_p2 = next(peer for peer, host in peer_hosts.items() if host == "10.10.1.2")
    winner = local.normalize_mapping(winner_plan["placements"][BASELINE])
    previous = local.normalize_mapping(previous_plan["placements"][PREVIOUS_WINNER])
    if previous[SHARD] != worker_p2:
        raise RuntimeError("unexpected shard-24 owner in previous winner")

    counterpart_shards = sorted(
        shard for shard, peer in previous.items() if peer == controller
    )
    if len(counterpart_shards) != 8:
        raise RuntimeError("unexpected controller shard group")
    seen = all_online_mappings(root)
    already_seen = [
        counterpart
        for counterpart in counterpart_shards
        if mapping_key(local.swap_mapping(previous, SHARD, counterpart)) in seen
    ]
    if already_seen:
        raise RuntimeError(f"controller family unexpectedly already tested: {already_seen}")

    counts = [int(value) for value in layout_manifest["routing"]["shard_counts"]]
    work_by_shard = [
        1000.0 * row["work_1000"]
        for row in refine.shard_features(trace, counts)
    ]

    placements = {BASELINE: winner}
    metrics: dict[str, Any] = {
        BASELINE: {
            "groups": local.placement_groups(winner, peer_order),
            "formally_confirmed": True,
            "selection_reason": "current formally confirmed winner and A/B control",
        }
    }
    family_rows = []
    names_by_counterpart: dict[int, str] = {}
    for counterpart in counterpart_shards:
        name = f"shard24_controller_counterpart_{counterpart}"
        names_by_counterpart[counterpart] = name
        mapping = local.swap_mapping(previous, SHARD, counterpart)
        if mapping[SHARD] != controller or mapping[counterpart] != worker_p2:
            raise RuntimeError(
                f"invalid controller family mapping for counterpart {counterpart}"
            )
        row = {
            "counterpart_shard": counterpart,
            "shard_24_from_host": "10.10.1.2",
            "shard_24_to_host": "10.10.1.1",
            "counterpart_from_host": "10.10.1.1",
            "counterpart_to_host": "10.10.1.2",
            "counterpart_work_units": work_by_shard[counterpart],
            "shard_24_work_units": work_by_shard[SHARD],
            "work_shift_from_p2_to_controller": work_by_shard[SHARD]
            - work_by_shard[counterpart],
            "moved_physical_vector_copies_from_previous_winner": counts[SHARD]
            + counts[counterpart],
            "previously_online_tested": False,
            "groups": local.placement_groups(mapping, peer_order),
            "selection_reason": (
                "complete enumeration of the finite shard-24-to-controller "
                "counterpart family; no surrogate ranking used"
            ),
        }
        placements[name] = mapping
        metrics[name] = row
        family_rows.append({"strategy": name, **row})

    phase_counterparts = list(counterpart_shards)
    random.Random(PHASE_ORDER_SEED).shuffle(phase_counterparts)
    phases = [("swap-24-21-controller-family-a", BASELINE)] + [
        (
            f"shard24-controller-counterpart-{counterpart}",
            names_by_counterpart[counterpart],
        )
        for counterpart in phase_counterparts
    ] + [("swap-24-21-controller-family-b", BASELINE)]

    placements_path = output / "placements.json"
    local.write_json_new(
        placements_path,
        {
            "format_version": 1,
            "physical_machine_count": 4,
            "logical_shard_count": 32,
            "controller_peer_id": winner_plan["controller_peer_id"],
            "peer_order": peer_order,
            "peer_hosts": winner_plan["peer_hosts"],
            "screen_phases": [
                {"phase_id": phase_id, "strategy": strategy}
                for phase_id, strategy in phases
            ],
            "placements": {
                name: {
                    str(shard): peer for shard, peer in sorted(mapping.items())
                }
                for name, mapping in placements.items()
            },
        },
    )
    universe_path = output / "family-universe.json"
    local.write_json_new(
        universe_path,
        {
            "family_definition": (
                "start from swap_16_18; move shard 24 from 10.10.1.2 to the "
                "controller 10.10.1.1 and move exactly one original controller "
                "shard back"
            ),
            "family_size": 8,
            "previously_online_tested_member_count": 0,
            "new_member_count": 8,
            "selection_rule": "complete enumeration of all eight members",
            "surrogate_used_for_selection": False,
            "phase_order_seed": PHASE_ORDER_SEED,
            "phase_counterpart_order": phase_counterparts,
            "candidates": family_rows,
        },
    )
    metrics_path = output / "candidate-metrics.json"
    local.write_json_new(metrics_path, metrics)
    rationale_path = output / "RATIONALE_zh.md"
    lines = [
        "# shard 24→controller counterpart 有限族穷举计划",
        "",
        "`.4` 与 `.3` 两个 worker 目标族已完整在线覆盖。这里测试最后一个跨机方向：把 shard 24 从 `10.10.1.2` 移到 controller `10.10.1.1`，并分别把 controller 原有 8 个 shard 回迁到 `.2`。",
        "",
        "8 个成员此前均未在线测试；候选选择为有限族完整枚举，不使用 surrogate。完成后，shard 24 的 24 个跨机单交换 counterpart 将全部有在线证据。",
        "",
        "| 候选 | counterpart work | shard24-counterpart work shift | moved copies |",
        "|---|---:|---:|---:|",
        *[
            (
                f"| `{row['strategy']}` | {row['counterpart_work_units']:.2f} | "
                f"{row['work_shift_from_p2_to_controller']:+.2f} | "
                f"{row['moved_physical_vector_copies_from_previous_winner']} |"
            )
            for row in family_rows
        ],
        "",
        f"在线 phase 顺序由固定 seed `{PHASE_ORDER_SEED}` 打乱：`{phase_counterparts}`。",
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
            "created_at": datetime.now(timezone.utc).isoformat().replace(
                "+00:00", "Z"
            ),
            "status": "OFFLINE_REFINEMENT_ONLY",
            "claim_boundary": (
                "complete candidate plan for the eight-member shard-24-to-controller "
                "counterpart family; online QPS remains authoritative"
            ),
            "physical_machine_count": 4,
            "logical_shard_count": 32,
            "family_size": 8,
            "previously_online_tested_member_count": 0,
            "new_candidate_count": 8,
            "selected_candidate_count": 8,
            "selected_strategies": [row["strategy"] for row in family_rows],
            "surrogate_used_for_selection": False,
            "phase_order_seed": PHASE_ORDER_SEED,
            "inputs": {
                "winner_plan": {
                    "path": str(winner_plan_path),
                    "sha256": local.sha256_path(winner_plan_path),
                },
                "previous_plan": {
                    "path": str(previous_plan_path),
                    "sha256": local.sha256_path(previous_plan_path),
                },
                "source_audit": {
                    "path": str(source_audit_path),
                    "sha256": local.sha256_path(source_audit_path),
                },
                "layout_manifest": {
                    "path": str(layout_manifest_path),
                    "sha256": local.sha256_path(layout_manifest_path),
                },
            },
            "outputs": {
                path.name: {
                    "path": str(path),
                    "sha256": local.sha256_path(path),
                }
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
