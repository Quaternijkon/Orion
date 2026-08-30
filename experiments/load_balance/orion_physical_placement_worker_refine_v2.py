#!/usr/bin/env python3
"""Select smaller worker-only swaps using all current online observations."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

import orion_physical_placement_local_refine as local
import orion_physical_placement_neighbor_refine as neighbor
import orion_physical_placement_refine as refine


DEFAULT_ROOT = Path(
    "/proj/intelisys-PG0/exp/orion-distributed/orion-load-balance-r090-20260825"
)
DEFAULT_LAYOUT = Path(
    "/users/dry/orion-distributed/native-20260818-scale4-32-r080-095-v1/"
    "artifacts/scale32/orion-r090-u48-b50-f14-g3248141"
)
DEFAULT_OUTPUT = DEFAULT_ROOT / "refine-v10-worker-small"
BASELINE = "swap_16_18"
MAX_WORK_SHIFT = 120.0
ALIASES = {"controller_aware_tail": "controller_aware_v1"}
OBSERVED_STRATEGIES = (
    "round_robin",
    "size_balanced",
    "controller_aware_v1",
    "learned_robust_workcap5500",
    "learned_robust",
    "swap_19_15",
    "swap_6_3",
    "swap_c8_0",
    "swap_28_7",
    "swap_29_25",
    "swap_16_18",
    "swap_29_18",
    "swap_27_16",
    "swap_5_8",
    "swap_27_25",
    "swap_30_11",
    "swap_16_10",
    "swap_25_10",
    "swap_16_17",
    "swap_25_17",
)
EXCLUDED_PAIRS = {(16, 10), (25, 10), (16, 17), (25, 17)}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def phase_files(root: Path) -> list[Path]:
    result = list((root / "online-v1/phases").glob("*/benchmark.json"))
    for version in range(2, 10):
        result.extend((root / f"online-v{version}/phases").glob("*/screen.json"))
    return sorted(result)


def gather_observations(
    root: Path, peer_order: Sequence[int], peer_hosts: dict[int, str]
) -> tuple[list[dict[str, Any]], dict[str, dict[int, int]], dict[str, Any]]:
    values: dict[str, list[list[float]]] = {}
    mappings: dict[str, dict[int, int]] = {}
    sources: dict[str, list[dict[str, Any]]] = {}
    for path in phase_files(root):
        payload = load_json(path)
        strategy = ALIASES.get(str(payload["strategy"]), str(payload["strategy"]))
        if strategy not in OBSERVED_STRATEGIES:
            continue
        concurrency_one = next(
            row for row in payload["concurrency_sweep"] if int(row["concurrency"]) == 1
        )
        demand = [
            1_000_000.0
            * float(concurrency_one["cpu_average_cores"][peer_hosts[peer_id]])
            / float(concurrency_one["qps"])
            for peer_id in peer_order
        ]
        mapping = {
            int(shard): int(peer)
            for shard, peer in payload["ending_placement"]["placement"].items()
        }
        if strategy in mappings and mapping != mappings[strategy]:
            raise RuntimeError(f"strategy {strategy} changed placement at {path}")
        mappings[strategy] = mapping
        values.setdefault(strategy, []).append(demand)
        sources.setdefault(strategy, []).append(
            {
                "path": str(path),
                "sha256": local.sha256_path(path),
                "node_cpu_microseconds_per_query": demand,
            }
        )
    if set(values) != set(OBSERVED_STRATEGIES):
        raise RuntimeError(
            f"observation strategies mismatch: {sorted(values)} != {sorted(OBSERVED_STRATEGIES)}"
        )
    observations = []
    provenance = {}
    for strategy in OBSERVED_STRATEGIES:
        averaged = np.mean(np.asarray(values[strategy], dtype=np.float64), axis=0)
        provenance[strategy] = {
            "replicate_count": len(values[strategy]),
            "sources": sources[strategy],
            "mean_node_cpu_microseconds_per_query": [float(value) for value in averaged],
        }
        mapping = mappings[strategy]
        for node_index, peer_id in enumerate(peer_order):
            observations.append(
                {
                    "strategy": strategy,
                    "peer_id": peer_id,
                    "host": peer_hosts[peer_id],
                    "controller": node_index == 0,
                    "shards": sorted(shard for shard, owner in mapping.items() if owner == peer_id),
                    "cpu_microseconds_per_query": float(averaged[node_index]),
                }
            )
    if len(observations) != 80:
        raise RuntimeError(f"expected 80 node observations, got {len(observations)}")
    return observations, mappings, provenance


def build(args: argparse.Namespace) -> Path:
    root = args.root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    layout_dir = args.layout_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)

    source_plan_path = root / "confirm-v2-plan/placements.json"
    source_summary_path = root / "online-v9/formal-v3-summary.json"
    source_audit_path = root / "online-v9/evidence-audit.json"
    trace_path = root / "trace/p24-per-query.json"
    proxy_manifest_path = root / "plan/manifest.json"
    layout_manifest_path = layout_dir / "build-manifest.json"
    source_plan = load_json(source_plan_path)
    source_summary = load_json(source_summary_path)
    source_audit = load_json(source_audit_path)
    trace = load_json(trace_path)
    layout_manifest = load_json(layout_manifest_path)
    if source_summary["formal_confirmation_result"]["baseline_strategy"] != BASELINE:
        raise RuntimeError("latest formal run does not retain swap_16_18 as baseline")
    if source_summary.get("confirmed_winner") is not None:
        raise RuntimeError("latest formal challenger unexpectedly succeeded")
    if source_audit.get("status") != "PASS":
        raise RuntimeError("latest formal evidence audit is not PASS")

    peer_order = [int(value) for value in source_plan["peer_order"]]
    peer_hosts = {int(peer): str(host) for peer, host in source_plan["peer_hosts"].items()}
    controller_peer = int(source_plan["controller_peer_id"])
    hot_peer = next(peer for peer, host in peer_hosts.items() if host == "10.10.1.2")
    target_peers = {
        peer for peer, host in peer_hosts.items() if host in {"10.10.1.3", "10.10.1.4"}
    }
    counts = [int(value) for value in layout_manifest["routing"]["shard_counts"]]
    shard_features = refine.shard_features(trace, counts)
    work_by_shard = [1000.0 * row["work_1000"] for row in shard_features]
    controller_overhead = float(
        load_json(proxy_manifest_path)["work_proxy"]["controller_overhead_model"][
            "controller_overhead_work_units"
        ]
    )
    observations, observed_mappings, provenance = gather_observations(
        root, peer_order, peer_hosts
    )
    baseline = observed_mappings[BASELINE]
    if baseline != local.normalize_mapping(source_plan["placements"][BASELINE]):
        raise RuntimeError("confirmed baseline mapping changed")
    models = [
        neighbor.fit_model(
            name,
            include_proxy,
            standardize,
            observations,
            work_by_shard,
            controller_overhead,
            observed_strategies=OBSERVED_STRATEGIES,
        )
        for name, include_proxy, standardize in neighbor.MODEL_SPECS
    ]

    eligible = []
    hot_shards = sorted(shard for shard, peer in baseline.items() if peer == hot_peer)
    target_shards = sorted(shard for shard, peer in baseline.items() if peer in target_peers)
    for hot_shard in hot_shards:
        for target_shard in target_shards:
            target_peer = baseline[target_shard]
            pair = (hot_shard, target_shard)
            work_shift = work_by_shard[hot_shard] - work_by_shard[target_shard]
            if pair in EXCLUDED_PAIRS or not (0.0 < work_shift <= MAX_WORK_SHIFT):
                continue
            mapping = local.swap_mapping(baseline, hot_shard, target_shard)
            groups = local.placement_groups(mapping, peer_order)
            predictions = {
                model["name"]: local.predict_mapping(
                    model,
                    mapping,
                    peer_order,
                    peer_hosts,
                    work_by_shard,
                    controller_overhead,
                )
                for model in models
            }
            trace_metrics = refine.trace_peak_metrics(
                trace, counts, groups, controller_overhead
            )
            eligible.append(
                {
                    "hot_shard": hot_shard,
                    "target_shard": target_shard,
                    "target_peer": target_peer,
                    "target_host": peer_hosts[target_peer],
                    "work_units_shifted_from_hot_worker": work_shift,
                    "moved_physical_vector_copies": counts[hot_shard] + counts[target_shard],
                    "ridge_predictions": predictions,
                    **trace_metrics,
                }
            )
    if len(eligible) < 4:
        raise RuntimeError(f"too few eligible small-shift candidates: {len(eligible)}")
    for model in models:
        model_name = model["name"]
        ranked = sorted(
            eligible,
            key=lambda row: (
                row["ridge_predictions"][model_name]["worst_node_cpu_microseconds_per_query"],
                row["hot_shard"],
                row["target_shard"],
            ),
        )
        for rank, row in enumerate(ranked, start=1):
            row.setdefault("model_ranks", {})[model_name] = rank
    trace_ranked = sorted(
        eligible,
        key=lambda row: (
            row["per_query_node_peak_p95"],
            row["per_query_node_peak_p99"],
            row["hot_shard"],
            row["target_shard"],
        ),
    )
    for rank, row in enumerate(trace_ranked, start=1):
        row["trace_p95_rank"] = rank
    for row in eligible:
        ranks = list(row["model_ranks"].values())
        row["model_worst_rank"] = max(ranks)
        row["model_rank_sum"] = sum(ranks)
    ranked = sorted(
        eligible,
        key=lambda row: (
            row["model_worst_rank"],
            row["model_rank_sum"],
            row["trace_p95_rank"],
            row["moved_physical_vector_copies"],
            row["hot_shard"],
            row["target_shard"],
        ),
    )
    for rank, row in enumerate(ranked, start=1):
        row["constrained_consensus_rank"] = rank

    selected = []
    for target_host in ("10.10.1.3", "10.10.1.4"):
        selected.extend([row for row in ranked if row["target_host"] == target_host][:2])
    selected.sort(key=lambda row: row["constrained_consensus_rank"])
    if len(selected) != 4 or len({(row["hot_shard"], row["target_shard"]) for row in selected}) != 4:
        raise RuntimeError("small-shift selection is not four unique candidates")

    candidate_mappings = {BASELINE: baseline}
    candidate_metrics: dict[str, Any] = {
        BASELINE: {
            "groups": local.placement_groups(baseline, peer_order),
            "selection_reason": "current formally retained winner and A/B control",
        }
    }
    selected_names = []
    for row in selected:
        name = f"swap_{row['hot_shard']}_{row['target_shard']}"
        selected_names.append(name)
        mapping = local.swap_mapping(baseline, row["hot_shard"], row["target_shard"])
        candidate_mappings[name] = mapping
        candidate_metrics[name] = {
            **row,
            "groups": local.placement_groups(mapping, peer_order),
            "selection_reason": (
                "top-two updated-model consensus within its target worker among untested "
                f"positive work shifts <= {MAX_WORK_SHIFT:.0f}"
            ),
            "controller_placement_unchanged": True,
        }

    placements_path = output / "placements.json"
    phases = [("swap-16-18-small-a", BASELINE)] + [
        (name.replace("_", "-"), name) for name in selected_names
    ] + [("swap-16-18-small-b", BASELINE)]
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
                for phase_id, strategy in phases
            ],
            "placements": {
                name: {str(shard): peer for shard, peer in sorted(mapping.items())}
                for name, mapping in candidate_mappings.items()
            },
        },
    )
    universe_path = output / "candidate-universe.json"
    local.write_json_new(
        universe_path,
        {
            "eligibility": {
                "source_host": "10.10.1.2",
                "target_hosts": ["10.10.1.3", "10.10.1.4"],
                "positive_work_shift_required": True,
                "maximum_work_shift": MAX_WORK_SHIFT,
                "tested_pairs_excluded": [list(pair) for pair in sorted(EXCLUDED_PAIRS)],
            },
            "eligible_count": len(ranked),
            "selection_rule": (
                "top two per target host by worst model rank, rank sum, trace p95 rank, "
                "then moved copies"
            ),
            "candidates": ranked,
        },
    )
    metrics_path = output / "candidate-metrics.json"
    local.write_json_new(metrics_path, candidate_metrics)
    models_path = output / "service-demand-models.json"
    local.write_json_new(
        models_path,
        {
            "claim_boundary": "updated candidate selector only; online QPS remains authoritative",
            "observed_strategies": list(OBSERVED_STRATEGIES),
            "observations": observations,
            "observation_provenance": provenance,
            "models": models,
        },
    )
    rationale_path = output / "RATIONALE_zh.md"
    lines = [
        "# Orion 更小 worker-only shift 计划",
        "",
        "使用 20 个唯一在线 placement、80 个节点观测重新拟合三种 ridge selector。",
        f"仅保留未测试、controller 不变、从 `.2` 向 `.3/.4` 且 0 < work shift ≤ {MAX_WORK_SHIFT:.0f} 的单交换。",
        "每个目标 worker 选模型共识前两名；离线排序不构成 QPS 结论。",
        "",
        "| 候选 | 目标 | work shift | 模型 worst-rank / rank-sum | trace p95 rank | 搬移副本 |",
        "|---|---|---:|---:|---:|---:|",
        *[
            (
                f"| `{name}` | {row['target_host']} | "
                f"{row['work_units_shifted_from_hot_worker']:.2f} | "
                f"{row['model_worst_rank']} / {row['model_rank_sum']} | "
                f"{row['trace_p95_rank']} | {row['moved_physical_vector_copies']} |"
            )
            for name, row in candidate_metrics.items()
            if name != BASELINE
        ],
        "",
    ]
    with rationale_path.open("x", encoding="utf-8") as handle:
        handle.write("\n".join(lines))

    outputs = (placements_path, universe_path, metrics_path, models_path, rationale_path)
    manifest_path = output / "manifest.json"
    local.write_json_new(
        manifest_path,
        {
            "format_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "status": "OFFLINE_REFINEMENT_ONLY",
            "claim_boundary": (
                "smaller worker-only single-swap candidate plan; models and trace metrics "
                "do not establish online QPS"
            ),
            "physical_machine_count": 4,
            "logical_shard_count": 32,
            "modeling": {
                "unique_online_placement_count": len(OBSERVED_STRATEGIES),
                "node_observation_count": len(observations),
                "eligible_candidate_count": len(ranked),
                "selected_candidate_count": len(selected_names),
                "model_count": len(models),
            },
            "selected_strategies": selected_names,
            "inputs": {
                "source_plan": {"path": str(source_plan_path), "sha256": local.sha256_path(source_plan_path)},
                "source_summary": {"path": str(source_summary_path), "sha256": local.sha256_path(source_summary_path)},
                "source_audit": {"path": str(source_audit_path), "sha256": local.sha256_path(source_audit_path)},
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
