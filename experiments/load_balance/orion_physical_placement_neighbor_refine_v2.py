#!/usr/bin/env python3
"""Refine swap_16_18 over its complete one-swap neighborhood."""

from __future__ import annotations

import argparse
import csv
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
DEFAULT_OUTPUT = DEFAULT_ROOT / "refine-v7"
BASELINE = "swap_16_18"
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
)
CANDIDATE_SWAPS = {
    "swap_27_16": (27, 16),
    "swap_5_8": (5, 8),
    "swap_27_25": (27, 25),
    "swap_30_11": (30, 11),
}
CANDIDATE_REASON = {
    "swap_27_16": "best updated-model consensus; shifts observed worker hotspot toward controller",
    "swap_5_8": "second consensus choice and nearly work-proxy neutral",
    "swap_27_25": "third consensus choice with an alternate worker-to-controller shift",
    "swap_30_11": "fourth consensus choice with a low deterministic work-proxy rank",
}
SCREEN_PHASES = (
    ("swap-16-18-neighbor-a", BASELINE),
    ("swap-27-16", "swap_27_16"),
    ("swap-5-8", "swap_5_8"),
    ("swap-27-25", "swap_27_25"),
    ("swap-30-11", "swap_30_11"),
    ("swap-16-18-neighbor-b", BASELINE),
)
ALIASES = {"controller_aware_tail": "controller_aware_v1"}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def phase_files(root: Path) -> list[Path]:
    result = list((root / "online-v1/phases").glob("*/benchmark.json"))
    for version in range(2, 7):
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
                    "shards": sorted(
                        shard for shard, owner in mapping.items() if owner == peer_id
                    ),
                    "cpu_microseconds_per_query": float(averaged[node_index]),
                }
            )
    if len(observations) != 48:
        raise RuntimeError(f"expected 48 node observations, got {len(observations)}")
    return observations, mappings, provenance


def build(args: argparse.Namespace) -> Path:
    root = args.root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    layout_dir = args.layout_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)

    source_plan_path = root / "confirm-v2-plan/placements.json"
    source_summary_path = root / "online-v6/formal-v2-summary.json"
    trace_path = root / "trace/p24-per-query.json"
    proxy_manifest_path = root / "plan/manifest.json"
    layout_manifest_path = layout_dir / "build-manifest.json"
    source_plan = load_json(source_plan_path)
    source_summary = load_json(source_summary_path)
    trace = load_json(trace_path)
    layout_manifest = load_json(layout_manifest_path)
    if source_summary.get("confirmed_winner") != BASELINE:
        raise RuntimeError("formal source summary does not confirm swap_16_18")
    peer_order = [int(value) for value in source_plan["peer_order"]]
    peer_hosts = {
        int(peer): str(host) for peer, host in source_plan["peer_hosts"].items()
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
        raise RuntimeError("confirmed winner placement changed")
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

    neighbors = neighbor.enumerate_neighbors(baseline, peer_order)
    rows = []
    for left_shard, right_shard, mapping in neighbors:
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
        loads = neighbor.proxy_loads(
            mapping, peer_order, work_by_shard, controller_overhead
        )
        rows.append(
            {
                "left_shard": left_shard,
                "right_shard": right_shard,
                "ridge_predictions": predictions,
                "work_proxy_loads": loads,
                "work_proxy_peak": max(loads),
                "moved_physical_vector_copies": counts[left_shard] + counts[right_shard],
            }
        )
    for model in models:
        name = model["name"]
        ranked = sorted(
            rows,
            key=lambda row: (
                row["ridge_predictions"][name][
                    "worst_node_cpu_microseconds_per_query"
                ],
                row["left_shard"],
                row["right_shard"],
            ),
        )
        for rank, row in enumerate(ranked, start=1):
            row.setdefault("model_ranks", {})[name] = rank
    proxy_ranked = sorted(
        rows,
        key=lambda row: (
            row["work_proxy_peak"], row["left_shard"], row["right_shard"]
        ),
    )
    for rank, row in enumerate(proxy_ranked, start=1):
        row["work_proxy_rank"] = rank
    for row in rows:
        ranks = list(row["model_ranks"].values())
        row["model_rank_sum"] = sum(ranks)
        row["model_worst_rank"] = max(ranks)
    neighborhood = sorted(
        rows,
        key=lambda row: (
            row["model_worst_rank"],
            row["model_rank_sum"],
            row["work_proxy_rank"],
            row["left_shard"],
            row["right_shard"],
        ),
    )
    for rank, row in enumerate(neighborhood, start=1):
        row["consensus_rank"] = rank
    row_by_pair = {
        (row["left_shard"], row["right_shard"]): row for row in neighborhood
    }

    candidate_mappings = {BASELINE: baseline}
    candidate_metrics: dict[str, Any] = {}
    baseline_groups = local.placement_groups(baseline, peer_order)
    baseline_loads = neighbor.proxy_loads(
        baseline, peer_order, work_by_shard, controller_overhead
    )
    candidate_metrics[BASELINE] = {
        "groups": baseline_groups,
        "selection_reason": "formally confirmed current winner",
        "ridge_predictions": {
            model["name"]: local.predict_mapping(
                model,
                baseline,
                peer_order,
                peer_hosts,
                work_by_shard,
                controller_overhead,
            )
            for model in models
        },
        "work_proxy_loads": baseline_loads,
        "work_proxy_peak": max(baseline_loads),
        "moved_shard_count_from_baseline": 0,
        "moved_physical_vector_copies_from_baseline": 0,
        **refine.trace_peak_metrics(trace, counts, baseline_groups, controller_overhead),
    }
    for name, pair in CANDIDATE_SWAPS.items():
        row = row_by_pair[pair]
        mapping = local.swap_mapping(baseline, *pair)
        candidate_mappings[name] = mapping
        groups = local.placement_groups(mapping, peer_order)
        candidate_metrics[name] = {
            "groups": groups,
            "swap_from_baseline": {
                "left_shard": pair[0],
                "right_shard": pair[1],
            },
            "selection_reason": CANDIDATE_REASON[name],
            "consensus_rank": row["consensus_rank"],
            "model_ranks": row["model_ranks"],
            "model_rank_sum": row["model_rank_sum"],
            "model_worst_rank": row["model_worst_rank"],
            "work_proxy_rank": row["work_proxy_rank"],
            "ridge_predictions": row["ridge_predictions"],
            "work_proxy_loads": row["work_proxy_loads"],
            "work_proxy_peak": row["work_proxy_peak"],
            "moved_shard_count_from_baseline": 2,
            "moved_physical_vector_copies_from_baseline": row[
                "moved_physical_vector_copies"
            ],
            **refine.trace_peak_metrics(trace, counts, groups, controller_overhead),
        }
    expected = ["swap_27_16", "swap_5_8", "swap_27_25", "swap_30_11"]
    actual = [
        name
        for name, _row in sorted(
            ((name, candidate_metrics[name]) for name in expected),
            key=lambda item: item[1]["consensus_rank"],
        )
    ]
    if actual != expected:
        raise RuntimeError(f"candidate consensus order changed: {actual}")

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
                for phase_id, strategy in SCREEN_PHASES
            ],
            "placements": {
                name: {str(shard): peer for shard, peer in sorted(mapping.items())}
                for name, mapping in candidate_mappings.items()
            },
        },
    )
    models_path = output / "service-demand-models.json"
    local.write_json_new(
        models_path,
        {
            "claim_boundary": "updated models select one-swap neighbors only; online QPS is authoritative",
            "observed_strategies": list(OBSERVED_STRATEGIES),
            "observations": observations,
            "observation_provenance": provenance,
            "models": models,
        },
    )
    neighborhood_path = output / "one-swap-neighborhood.json"
    local.write_json_new(
        neighborhood_path,
        {
            "baseline": BASELINE,
            "evaluated_swaps": len(neighborhood),
            "ranking_rule": [
                "minimize worst rank across three ridge models",
                "then minimize rank sum",
                "then deterministic work-proxy rank",
            ],
            "neighbors": neighborhood,
        },
    )
    metrics_path = output / "candidate-metrics.json"
    local.write_json_new(metrics_path, candidate_metrics)
    csv_path = output / "candidate-metrics.csv"
    model_names = [model["name"] for model in models]
    with csv_path.open("x", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "strategy",
            "consensus_rank",
            "model_worst_rank",
            "model_rank_sum",
            "work_proxy_rank",
            "work_proxy_peak",
            *[f"{name}_worst_cpu_us_per_query" for name in model_names],
            "moved_copies_from_baseline",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for name, row in candidate_metrics.items():
            writer.writerow(
                {
                    "strategy": name,
                    "consensus_rank": row.get("consensus_rank"),
                    "model_worst_rank": row.get("model_worst_rank"),
                    "model_rank_sum": row.get("model_rank_sum"),
                    "work_proxy_rank": row.get("work_proxy_rank"),
                    "work_proxy_peak": row["work_proxy_peak"],
                    **{
                        f"{model_name}_worst_cpu_us_per_query": row[
                            "ridge_predictions"
                        ][model_name]["worst_node_cpu_microseconds_per_query"]
                        for model_name in model_names
                    },
                    "moved_copies_from_baseline": row[
                        "moved_physical_vector_copies_from_baseline"
                    ],
                }
            )

    manifest_path = output / "manifest.json"
    outputs = [
        placements_path,
        models_path,
        neighborhood_path,
        metrics_path,
        csv_path,
    ]
    local.write_json_new(
        manifest_path,
        {
            "format_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "status": "OFFLINE_REFINEMENT_ONLY",
            "claim_boundary": (
                "complete one-swap enumeration around swap_16_18; models do not "
                "establish online QPS or global optimality"
            ),
            "physical_machine_count": 4,
            "logical_shard_count": 32,
            "modeling": {
                "unique_online_placement_count": len(OBSERVED_STRATEGIES),
                "node_observation_count": len(observations),
                "one_swap_neighbors_evaluated": len(neighborhood),
                "model_count": len(models),
                "selection_count": len(CANDIDATE_SWAPS),
            },
            "inputs": {
                "source_plan": {
                    "path": str(source_plan_path),
                    "sha256": local.sha256_path(source_plan_path),
                },
                "source_summary": {
                    "path": str(source_summary_path),
                    "sha256": local.sha256_path(source_summary_path),
                },
                "trace": {"path": str(trace_path), "sha256": local.sha256_path(trace_path)},
                "layout_manifest": {
                    "path": str(layout_manifest_path),
                    "sha256": local.sha256_path(layout_manifest_path),
                },
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
