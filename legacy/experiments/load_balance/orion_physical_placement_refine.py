#!/usr/bin/env python3
"""Refine Orion physical placement from measured online CPU service demand.

The first placement study showed that the original EF*logN proxy did not rank
size-balanced versus round-robin correctly.  This refinement therefore fits
several low-concurrency CPU-microseconds-per-query models from the three live
placements, then audits two robust MILP incumbents against those models, the
original trace proxy, and their complete one-swap neighborhoods.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np


DEFAULT_ROOT = Path(
    "/proj/intelisys-PG0/exp/orion-distributed/orion-load-balance-r090-20260825"
)
DEFAULT_LAYOUT = Path(
    "/users/dry/orion-distributed/native-20260818-scale4-32-r080-095-v1/"
    "artifacts/scale32/orion-r090-u48-b50-f14-g3248141"
)
DEFAULT_OUTPUT = DEFAULT_ROOT / "refine-v2"
EXPECTED_SHARDS = 32

# Incumbents produced by HiGHS MILP over 128 binary shard-to-node variables and
# one minimax variable.  Both are re-evaluated from source measurements below;
# no objective values are trusted from these constants.
ROBUST_GROUPS = (
    (4, 9, 10, 18, 19, 20, 24, 25),
    (5, 6, 7, 13, 17, 23, 26, 30),
    (1, 3, 11, 12, 14, 27, 28, 31),
    (0, 2, 8, 15, 16, 21, 22, 29),
)
ROBUST_WORK_CAP_5500_GROUPS = (
    (2, 10, 12, 14, 18, 28, 29, 30),
    (0, 1, 17, 20, 22, 23, 24, 26),
    (4, 6, 7, 8, 11, 15, 19, 21),
    (3, 5, 9, 13, 16, 25, 27, 31),
)
MILP_PROVENANCE = {
    "learned_robust": {
        "time_limit_seconds": 90,
        "relative_gap": 0.0011367824408107764,
        "solver_status": "time_limit_with_feasible_incumbent",
        "work_proxy_cap": None,
    },
    "learned_robust_workcap5500": {
        "time_limit_seconds": 60,
        "relative_gap": 0.001800749444819373,
        "solver_status": "time_limit_with_feasible_incumbent",
        "work_proxy_cap": 5500.0,
    },
}


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
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * quantile
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (position - lower) * (ordered[upper] - ordered[lower])


def validate_groups(groups: Sequence[Sequence[int]]) -> tuple[tuple[int, ...], ...]:
    normalized = tuple(tuple(sorted(int(shard) for shard in group)) for group in groups)
    if len(normalized) != 4 or any(len(group) != 8 for group in normalized):
        raise ValueError("placement groups must be strict 8/8/8/8")
    flattened = [shard for group in normalized for shard in group]
    if sorted(flattened) != list(range(EXPECTED_SHARDS)):
        raise ValueError("placement groups must cover shards 0..31 exactly once")
    return normalized


def shard_features(trace: dict[str, Any], counts: Sequence[int]) -> list[dict[str, float]]:
    query_count = len(trace["per_query"])
    targets: list[list[dict[str, Any]]] = [[] for _ in counts]
    for query in trace["per_query"]:
        for target in query["targets"]:
            targets[int(target["shard_id"])].append(target)
    result = []
    for shard_id, rows in enumerate(targets):
        ef_per_query = sum(float(row["ef"]) for row in rows) / query_count
        entry_points_per_query = (
            sum(len(row["entry_points"]) for row in rows) / query_count
        )
        result.append(
            {
                "shard_id": shard_id,
                "physical_vector_copies": float(counts[shard_id]),
                "points_100k": float(counts[shard_id]) / 100_000.0,
                "query_visits_per_query": len(rows) / query_count,
                "ef_per_query": ef_per_query,
                "entry_points_per_query": entry_points_per_query,
                "entry_points_10": entry_points_per_query / 10.0,
                "work_1000": ef_per_query * math.log2(counts[shard_id]) / 1000.0,
            }
        )
    return result


def aggregate_features(
    features: Sequence[dict[str, float]], shards: Sequence[int]
) -> dict[str, float]:
    keys = (
        "physical_vector_copies",
        "points_100k",
        "query_visits_per_query",
        "ef_per_query",
        "entry_points_per_query",
        "entry_points_10",
        "work_1000",
    )
    return {key: sum(features[shard][key] for shard in shards) for key in keys}


def observations(
    root: Path,
    placements: dict[str, Any],
    features: Sequence[dict[str, float]],
) -> list[dict[str, Any]]:
    peer_order = [int(value) for value in placements["peer_order"]]
    peer_hosts = {int(peer): host for peer, host in placements["peer_hosts"].items()}
    phases = ("round-robin-a", "size-balanced-a", "controller-tail-a")
    rows = []
    for phase in phases:
        benchmark_path = root / "online-v1/phases" / phase / "benchmark.json"
        benchmark = load_json(benchmark_path)
        concurrency_one = next(
            row for row in benchmark["concurrency_sweep"] if row["concurrency"] == 1
        )
        placement = {
            int(shard): int(peer)
            for shard, peer in benchmark["ending_placement"]["placement"].items()
        }
        for node_index, peer_id in enumerate(peer_order):
            host = peer_hosts[peer_id]
            shards = sorted(shard for shard, owner in placement.items() if owner == peer_id)
            aggregate = aggregate_features(features, shards)
            rows.append(
                {
                    "phase": phase,
                    "peer_id": peer_id,
                    "host": host,
                    "controller": 1.0 if node_index == 0 else 0.0,
                    "shards": shards,
                    **aggregate,
                    "qps": concurrency_one["qps"],
                    "cpu_cores": concurrency_one["cpu_average_cores"][host],
                    "cpu_microseconds_per_query": (
                        1_000_000.0
                        * concurrency_one["cpu_average_cores"][host]
                        / concurrency_one["qps"]
                    ),
                    "source": str(benchmark_path),
                }
            )
    return rows


def fit_model(
    name: str,
    feature_names: Sequence[str],
    rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    design = np.asarray(
        [[1.0, *(float(row[key]) for key in feature_names)] for row in rows],
        dtype=np.float64,
    )
    target = np.asarray(
        [float(row["cpu_microseconds_per_query"]) for row in rows], dtype=np.float64
    )
    coefficients, _residuals, rank, singular_values = np.linalg.lstsq(
        design, target, rcond=None
    )
    predicted = design @ coefficients
    residual = target - predicted
    return {
        "name": name,
        "target": "CPU microseconds per query at client concurrency 1",
        "features": ["intercept", *feature_names],
        "coefficients": [float(value) for value in coefficients],
        "rank": int(rank),
        "singular_values": [float(value) for value in singular_values],
        "rmse": float(math.sqrt(np.mean(np.square(residual)))),
        "mae": float(np.mean(np.abs(residual))),
        "predicted": [float(value) for value in predicted],
        "observed": [float(value) for value in target],
        "residual": [float(value) for value in residual],
    }


def model_prediction(
    model: dict[str, Any],
    aggregate: dict[str, float],
    *,
    controller: bool,
) -> float:
    values = {**aggregate, "controller": 1.0 if controller else 0.0}
    coefficients = model["coefficients"]
    return float(
        coefficients[0]
        + sum(
            coefficient * float(values[feature])
            for coefficient, feature in zip(
                coefficients[1:], model["features"][1:], strict=True
            )
        )
    )


def group_objective(
    groups: Sequence[Sequence[int]],
    features: Sequence[dict[str, float]],
    models: Sequence[dict[str, Any]],
    *,
    work_cap: float | None,
    controller_overhead_work: float,
) -> tuple[float, list[float], list[float]]:
    predictions = []
    proxy_loads = []
    for node_index, shards in enumerate(groups):
        aggregate = aggregate_features(features, shards)
        for model in models:
            predictions.append(
                model_prediction(model, aggregate, controller=node_index == 0)
            )
        proxy_loads.append(
            1000.0 * aggregate["work_1000"]
            + (controller_overhead_work if node_index == 0 else 0.0)
        )
    if work_cap is not None and max(proxy_loads) > work_cap + 1e-9:
        return math.inf, predictions, proxy_loads
    return max(predictions), predictions, proxy_loads


def one_swap_audit(
    groups: Sequence[Sequence[int]],
    objective: Callable[[Sequence[Sequence[int]]], float],
) -> dict[str, Any]:
    baseline = objective(groups)
    best = baseline
    best_swap = None
    evaluated = 0
    for left, right in itertools.combinations(range(4), 2):
        for left_shard in groups[left]:
            for right_shard in groups[right]:
                candidate = [list(group) for group in groups]
                candidate[left].remove(left_shard)
                candidate[right].remove(right_shard)
                candidate[left].append(right_shard)
                candidate[right].append(left_shard)
                value = objective(candidate)
                evaluated += 1
                if value < best - 1e-9:
                    best = value
                    best_swap = {
                        "left_bin": left,
                        "right_bin": right,
                        "left_shard": left_shard,
                        "right_shard": right_shard,
                    }
    return {
        "evaluated_swaps": evaluated,
        "baseline_objective": baseline,
        "best_neighbor_objective": best,
        "improving_swap": best_swap,
        "one_swap_local_optimum": best_swap is None,
    }


def assign_workers_for_sequence(
    candidate_groups: dict[str, Sequence[Sequence[int]]],
    peer_order: Sequence[int],
    baseline: dict[int, int],
    counts: Sequence[int],
) -> dict[str, dict[int, int]]:
    names = list(candidate_groups)
    options: dict[str, list[dict[int, int]]] = {}
    for name, groups in candidate_groups.items():
        mappings = []
        for permutation in itertools.permutations(peer_order[1:]):
            mapping = {shard: peer_order[0] for shard in groups[0]}
            for shards, peer_id in zip(groups[1:], permutation, strict=True):
                mapping.update({shard: peer_id for shard in shards})
            mappings.append(mapping)
        options[name] = mappings

    best_score = None
    best_maps = None
    for chosen in itertools.product(*(options[name] for name in names)):
        chain = [baseline, *chosen, baseline]
        moved_count = 0
        moved_copies = 0
        for before, after in zip(chain, chain[1:]):
            moved = [shard for shard in range(EXPECTED_SHARDS) if before[shard] != after[shard]]
            moved_count += len(moved)
            moved_copies += sum(int(counts[shard]) for shard in moved)
        score = (moved_count, moved_copies)
        if best_score is None or score < best_score:
            best_score = score
            best_maps = {name: mapping for name, mapping in zip(names, chosen, strict=True)}
    assert best_maps is not None
    return best_maps


def trace_peak_metrics(
    trace: dict[str, Any],
    counts: Sequence[int],
    groups: Sequence[Sequence[int]],
    controller_overhead_work: float,
) -> dict[str, float]:
    assignment = {shard: node for node, group in enumerate(groups) for shard in group}
    node_totals = [0.0, 0.0, 0.0, 0.0]
    peaks = []
    for query in trace["per_query"]:
        loads = [controller_overhead_work, 0.0, 0.0, 0.0]
        for target in query["targets"]:
            shard = int(target["shard_id"])
            loads[assignment[shard]] += float(target["ef"]) * math.log2(counts[shard])
        peaks.append(max(loads))
        for node, load in enumerate(loads):
            node_totals[node] += load
    query_count = len(trace["per_query"])
    average_loads = [value / query_count for value in node_totals]
    return {
        "average_node_load_peak": max(average_loads),
        "per_query_node_peak_mean": statistics.fmean(peaks),
        "per_query_node_peak_p95": percentile(peaks, 0.95),
        "per_query_node_peak_p99": percentile(peaks, 0.99),
    }


def build(args: argparse.Namespace) -> Path:
    root = args.root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    old_plan_path = root / "plan/placements.json"
    trace_path = root / "trace/p24-per-query.json"
    layout_manifest_path = args.layout_dir / "build-manifest.json"
    old_plan = load_json(old_plan_path)
    trace = load_json(trace_path)
    layout_manifest = load_json(layout_manifest_path)
    counts = [int(value) for value in layout_manifest["routing"]["shard_counts"]]
    features = shard_features(trace, counts)
    observed = observations(root, old_plan, features)
    models = [
        fit_model("visits_plus_controller", ("query_visits_per_query", "controller"), observed),
        fit_model(
            "points_work_plus_controller",
            ("points_100k", "work_1000", "controller"),
            observed,
        ),
        fit_model(
            "points_entrypoints_plus_controller",
            ("points_100k", "entry_points_10", "controller"),
            observed,
        ),
    ]
    controller_overhead = float(
        load_json(root / "plan/manifest.json")["work_proxy"][
            "controller_overhead_model"
        ]["controller_overhead_work_units"]
    )
    groups = {
        "learned_robust_workcap5500": validate_groups(ROBUST_WORK_CAP_5500_GROUPS),
        "learned_robust": validate_groups(ROBUST_GROUPS),
    }
    old_mapping = {
        int(shard): int(peer)
        for shard, peer in old_plan["placements"]["controller_aware_tail"].items()
    }
    peer_order = [int(value) for value in old_plan["peer_order"]]
    candidate_maps = assign_workers_for_sequence(groups, peer_order, old_mapping, counts)
    candidate_maps = {"controller_aware_v1": old_mapping, **candidate_maps}

    metrics: dict[str, Any] = {}
    old_groups = tuple(
        tuple(sorted(shard for shard, peer in old_mapping.items() if peer == peer_id))
        for peer_id in peer_order
    )
    all_groups = {"controller_aware_v1": old_groups, **groups}
    for name, candidate_groups in all_groups.items():
        cap = MILP_PROVENANCE.get(name, {}).get("work_proxy_cap")
        objective, predictions, proxy_loads = group_objective(
            candidate_groups,
            features,
            models,
            work_cap=cap,
            controller_overhead_work=controller_overhead,
        )
        mapping = candidate_maps[name]
        moved = [
            shard for shard in range(EXPECTED_SHARDS) if mapping[shard] != old_mapping[shard]
        ]
        metrics[name] = {
            "groups": candidate_groups,
            "learned_ensemble_max_cpu_microseconds_per_query": objective,
            "learned_model_node_predictions": predictions,
            "work_proxy_loads": proxy_loads,
            **trace_peak_metrics(trace, counts, candidate_groups, controller_overhead),
            "moved_shard_count_from_v1": len(moved),
            "moved_physical_vector_copies_from_v1": sum(counts[shard] for shard in moved),
            "milp": MILP_PROVENANCE.get(name),
        }
        if name in groups:
            metrics[name]["one_swap_audit"] = one_swap_audit(
                candidate_groups,
                lambda candidate, cap=cap: group_objective(
                    candidate,
                    features,
                    models,
                    work_cap=cap,
                    controller_overhead_work=controller_overhead,
                )[0],
            )
            if not metrics[name]["one_swap_audit"]["one_swap_local_optimum"]:
                raise RuntimeError(f"{name} is not a one-swap local optimum")

    placements_path = output / "placements.json"
    write_json_new(
        placements_path,
        {
            "format_version": 1,
            "physical_machine_count": 4,
            "logical_shard_count": EXPECTED_SHARDS,
            "controller_peer_id": old_plan["controller_peer_id"],
            "peer_order": peer_order,
            "peer_hosts": old_plan["peer_hosts"],
            "screen_order": list(candidate_maps),
            "placements": {
                name: {str(shard): peer for shard, peer in mapping.items()}
                for name, mapping in candidate_maps.items()
            },
        },
    )
    model_path = output / "service-demand-models.json"
    write_json_new(model_path, {"observations": observed, "models": models})
    metrics_path = output / "candidate-metrics.json"
    write_json_new(metrics_path, metrics)
    csv_path = output / "candidate-metrics.csv"
    rows = []
    for name, row in metrics.items():
        rows.append(
            {
                "strategy": name,
                "learned_ensemble_max_cpu_us_per_query": row[
                    "learned_ensemble_max_cpu_microseconds_per_query"
                ],
                "work_proxy_average_node_peak": row["average_node_load_peak"],
                "work_proxy_per_query_peak_mean": row["per_query_node_peak_mean"],
                "work_proxy_per_query_peak_p95": row["per_query_node_peak_p95"],
                "work_proxy_per_query_peak_p99": row["per_query_node_peak_p99"],
                "moved_shards_from_v1": row["moved_shard_count_from_v1"],
                "moved_copies_from_v1": row["moved_physical_vector_copies_from_v1"],
            }
        )
    with csv_path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    manifest_path = output / "manifest.json"
    write_json_new(
        manifest_path,
        {
            "format_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "status": "OFFLINE_REFINEMENT_ONLY",
            "claim_boundary": (
                "The learned service-demand ensemble and one-swap audits only select "
                "online screening candidates; they do not establish QPS."
            ),
            "inputs": {
                "old_placement_plan": str(old_plan_path),
                "old_placement_plan_sha256": sha256_path(old_plan_path),
                "trace": str(trace_path),
                "trace_sha256": sha256_path(trace_path),
                "layout_manifest": str(layout_manifest_path),
                "layout_manifest_sha256": sha256_path(layout_manifest_path),
                "online_v1_summary": str(root / "online-v1/summary.json"),
                "online_v1_summary_sha256": sha256_path(root / "online-v1/summary.json"),
            },
            "modeling": {
                "fit_source": "concurrency=1 rows from three unique online placements",
                "observation_count": len(observed),
                "model_count": len(models),
                "objective": "minimize worst predicted node CPU microseconds per query",
                "strict_shards_per_node": 8,
                "candidate_one_swap_neighborhood_size": 384,
            },
            "outputs": {
                "placements": str(placements_path),
                "placements_sha256": sha256_path(placements_path),
                "service_demand_models": str(model_path),
                "service_demand_models_sha256": sha256_path(model_path),
                "candidate_metrics_json": str(metrics_path),
                "candidate_metrics_json_sha256": sha256_path(metrics_path),
                "candidate_metrics_csv": str(csv_path),
                "candidate_metrics_csv_sha256": sha256_path(csv_path),
            },
        },
    )
    print(json.dumps({"manifest": str(manifest_path)}, indent=2))
    return manifest_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--layout-dir", type=Path, default=DEFAULT_LAYOUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    build(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
