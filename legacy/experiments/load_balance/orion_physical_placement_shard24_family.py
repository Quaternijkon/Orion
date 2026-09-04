#!/usr/bin/env python3
"""Enumerate the untested shard-24-to-worker-.4 counterpart family."""

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
DEFAULT_OUTPUT = DEFAULT_ROOT / "refine-v18-shard24-p4-family"
BASELINE = "swap_24_21"
PREVIOUS_WINNER = "swap_16_18"
SHARD = 24
WINNING_COUNTERPART = 21
PHASE_ORDER_SEED = 3_248_141


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def mapping_key(mapping: dict[int, int]) -> tuple[tuple[int, int], ...]:
    return tuple(sorted(mapping.items()))


def all_online_mappings(root: Path) -> set[tuple[tuple[int, int], ...]]:
    paths = list((root / "online-v1/phases").glob("*/benchmark.json"))
    for version in range(2, 17):
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
    if len(result) != 40:
        raise RuntimeError(f"expected 40 online placements, got {len(result)}")
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
    formal_audit_path = root / "online-v11/evidence-audit.json"
    latest_audit_path = root / "online-v16/evidence-audit.json"
    kernel_audit_path = root / "refine-v17-kernel-qps/evidence-audit.json"
    kernel_ranking_path = (
        root / "refine-v17-kernel-qps/one-swap-kernel-qps-ranking.json"
    )
    trace_path = root / "trace/p24-per-query.json"
    layout_manifest_path = layout_dir / "build-manifest.json"

    winner_plan = load_json(winner_plan_path)
    previous_plan = load_json(previous_plan_path)
    formal_audit = load_json(formal_audit_path)
    latest_audit = load_json(latest_audit_path)
    kernel_audit = load_json(kernel_audit_path)
    kernel_ranking = load_json(kernel_ranking_path)
    trace = load_json(trace_path)
    layout_manifest = load_json(layout_manifest_path)
    if formal_audit.get("status") != "PASS":
        raise RuntimeError("formal winner audit is not PASS")
    if latest_audit.get("status") != "PASS":
        raise RuntimeError("latest online evidence audit is not PASS")
    if kernel_audit.get("status") != "PASS" or kernel_audit.get("decision") != "NO_ONLINE_V17":
        raise RuntimeError("kernel stop-gate audit is not PASS")

    peer_order = [int(value) for value in winner_plan["peer_order"]]
    peer_hosts = {
        int(peer): str(host) for peer, host in winner_plan["peer_hosts"].items()
    }
    worker_p4 = next(peer for peer, host in peer_hosts.items() if host == "10.10.1.4")
    worker_p2 = next(peer for peer, host in peer_hosts.items() if host == "10.10.1.2")
    winner = local.normalize_mapping(winner_plan["placements"][BASELINE])
    previous = local.normalize_mapping(previous_plan["placements"][PREVIOUS_WINNER])
    if previous[SHARD] != worker_p2 or winner[SHARD] != worker_p4:
        raise RuntimeError("unexpected shard-24 movement in source placements")
    if previous[WINNING_COUNTERPART] != worker_p4 or winner[WINNING_COUNTERPART] != worker_p2:
        raise RuntimeError("unexpected winning counterpart movement")
    if local.swap_mapping(previous, SHARD, WINNING_COUNTERPART) != winner:
        raise RuntimeError("formal winner is not the expected shard-24 family member")

    original_p4_shards = sorted(
        shard for shard, peer in previous.items() if peer == worker_p4
    )
    if len(original_p4_shards) != 8 or WINNING_COUNTERPART not in original_p4_shards:
        raise RuntimeError("unexpected prior .4 shard group")
    remaining_counterparts = [
        shard for shard in original_p4_shards if shard != WINNING_COUNTERPART
    ]
    if len(remaining_counterparts) != 7:
        raise RuntimeError("expected seven untested shard-24 counterparts")

    counts = [int(value) for value in layout_manifest["routing"]["shard_counts"]]
    work_by_shard = [
        1000.0 * row["work_1000"]
        for row in refine.shard_features(trace, counts)
    ]
    kernel_by_pair = {
        tuple(sorted((int(row["left_shard"]), int(row["right_shard"])))): row
        for row in kernel_ranking["neighbors"]
    }
    seen = all_online_mappings(root)

    placements = {BASELINE: winner}
    metrics: dict[str, Any] = {
        BASELINE: {
            "counterpart_shard": WINNING_COUNTERPART,
            "groups": local.placement_groups(winner, peer_order),
            "formally_confirmed": True,
            "selection_reason": "current formally confirmed winner and A/B control",
        }
    }
    names_by_counterpart: dict[int, str] = {}
    family_rows = []
    for counterpart in remaining_counterparts:
        mapping = local.swap_mapping(previous, SHARD, counterpart)
        equivalent = local.swap_mapping(winner, WINNING_COUNTERPART, counterpart)
        if mapping != equivalent:
            raise RuntimeError(f"family equivalence failed for counterpart {counterpart}")
        if mapping_key(mapping) in seen:
            raise RuntimeError(f"counterpart {counterpart} was already tested online")
        if mapping[SHARD] != worker_p4 or mapping[counterpart] != worker_p2:
            raise RuntimeError(f"invalid family mapping for counterpart {counterpart}")
        name = f"shard24_p4_counterpart_{counterpart}"
        names_by_counterpart[counterpart] = name
        placements[name] = mapping
        kernel_row = kernel_by_pair[
            tuple(sorted((WINNING_COUNTERPART, counterpart)))
        ]
        row = {
            "counterpart_shard": counterpart,
            "equivalent_swap_from_current_winner": [
                WINNING_COUNTERPART,
                counterpart,
            ],
            "shard_24_from_host": "10.10.1.2",
            "shard_24_to_host": "10.10.1.4",
            "counterpart_from_host": "10.10.1.4",
            "counterpart_to_host": "10.10.1.2",
            "counterpart_work_units": work_by_shard[counterpart],
            "winning_counterpart_work_units": work_by_shard[
                WINNING_COUNTERPART
            ],
            "additional_work_shift_from_p4_to_p2_vs_winner": (
                work_by_shard[counterpart] - work_by_shard[WINNING_COUNTERPART]
            ),
            "moved_physical_vector_copies_from_winner": (
                counts[counterpart] + counts[WINNING_COUNTERPART]
            ),
            "kernel_predictions_pct_preserved_not_used_for_selection": kernel_row[
                "predicted_improvement_pct"
            ],
            "kernel_consensus_rank_preserved_not_used_for_selection": kernel_row[
                "consensus_rank"
            ],
            "already_online_tested": False,
            "groups": local.placement_groups(mapping, peer_order),
            "selection_reason": (
                "exhaustive remaining member of the finite shard-24-to-.4 "
                "counterpart family; no surrogate ranking used"
            ),
        }
        metrics[name] = row
        family_rows.append({"strategy": name, **row})

    phase_counterparts = list(remaining_counterparts)
    random.Random(PHASE_ORDER_SEED).shuffle(phase_counterparts)
    phases = [("swap-24-21-family-a", BASELINE)] + [
        (f"shard24-p4-counterpart-{counterpart}", names_by_counterpart[counterpart])
        for counterpart in phase_counterparts
    ] + [("swap-24-21-family-b", BASELINE)]

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
                "start from swap_16_18; move shard 24 from 10.10.1.2 to "
                "10.10.1.4 and move exactly one original 10.10.1.4 shard back"
            ),
            "family_size": 8,
            "formally_confirmed_member": {
                "strategy": BASELINE,
                "counterpart_shard": WINNING_COUNTERPART,
            },
            "remaining_member_count": len(family_rows),
            "selection_rule": "exhaustive enumeration of all remaining members",
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
        "# shard 24→`.4` counterpart 有限族穷举计划",
        "",
        "正式提升来自把 shard 24 从 `10.10.1.2` 移到 `10.10.1.4`。在原 `swap_16_18` 的 `.4` 组中，恰有 8 个 shard 可作为回迁到 `.2` 的 counterpart；当前赢家使用 shard 21。",
        "",
        "本计划一次在线筛完其余 7 个成员，从而得到这个关键结构族内的完整实测排序。候选选择不使用已经被在线证据否定的 learned/QPS/kernel surrogate；相关负预测只原样保留。",
        "",
        "| 候选 | current-winner swap | counterpart work | 相对 shard 21 的 work shift | kernel range（不用于选择） |",
        "|---|---:|---:|---:|---:|",
        *[
            (
                f"| `{row['strategy']}` | 21↔{row['counterpart_shard']} | "
                f"{row['counterpart_work_units']:.2f} | "
                f"{row['additional_work_shift_from_p4_to_p2_vs_winner']:+.2f} | "
                f"{min(row['kernel_predictions_pct_preserved_not_used_for_selection'].values()):+.2f}% … "
                f"{max(row['kernel_predictions_pct_preserved_not_used_for_selection'].values()):+.2f}% |"
            )
            for row in family_rows
        ],
        "",
        f"在线 phase 顺序由固定 seed `{PHASE_ORDER_SEED}` 打乱：`{phase_counterparts}`；前后均测 `swap_24_21` A/B。",
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
                "complete candidate plan for the eight-member shard-24-to-.4 "
                "counterpart family; online QPS remains authoritative"
            ),
            "physical_machine_count": 4,
            "logical_shard_count": 32,
            "family_size": 8,
            "previously_confirmed_member_count": 1,
            "selected_candidate_count": len(family_rows),
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
                "formal_audit": {
                    "path": str(formal_audit_path),
                    "sha256": local.sha256_path(formal_audit_path),
                },
                "latest_online_audit": {
                    "path": str(latest_audit_path),
                    "sha256": local.sha256_path(latest_audit_path),
                },
                "kernel_stop_gate_audit": {
                    "path": str(kernel_audit_path),
                    "sha256": local.sha256_path(kernel_audit_path),
                },
                "kernel_ranking": {
                    "path": str(kernel_ranking_path),
                    "sha256": local.sha256_path(kernel_ranking_path),
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
