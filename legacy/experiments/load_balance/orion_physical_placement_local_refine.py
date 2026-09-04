#!/usr/bin/env python3
"""Build auditable one-swap Orion placement candidates around the online winner.

The global learned placements in refine-v2 did not beat controller-aware v1
online.  This follow-up deliberately stays in the winner's one-swap
neighborhood.  Ridge models are candidate selectors only: the generated plan
must still be ranked by an online QPS screen at the fixed recall contract.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

import orion_physical_placement_refine as refine


DEFAULT_ROOT = Path(
    "/proj/intelisys-PG0/exp/orion-distributed/orion-load-balance-r090-20260825"
)
DEFAULT_LAYOUT = Path(
    "/users/dry/orion-distributed/native-20260818-scale4-32-r080-095-v1/"
    "artifacts/scale32/orion-r090-u48-b50-f14-g3248141"
)
DEFAULT_OUTPUT = DEFAULT_ROOT / "refine-v3"
RIDGE_ALPHA = 1e-6
EXPECTED_SHARDS = 32

LOCAL_SWAPS = {
    "swap_19_15": (19, 15),
    "swap_6_3": (6, 3),
    "swap_c8_0": (8, 0),
}

SCREEN_PHASES = (
    ("controller-v1-local-a", "controller_aware_v1"),
    ("swap-19-15", "swap_19_15"),
    ("swap-6-3", "swap_6_3"),
    ("swap-c8-0", "swap_c8_0"),
    ("controller-v1-local-b", "controller_aware_v1"),
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
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def normalize_mapping(raw: dict[str, Any]) -> dict[int, int]:
    mapping = {int(shard): int(peer) for shard, peer in raw.items()}
    if set(mapping) != set(range(EXPECTED_SHARDS)):
        raise ValueError("placement must cover shards 0..31 exactly once")
    return mapping


def placement_groups(
    mapping: dict[int, int], peer_order: Sequence[int]
) -> tuple[tuple[int, ...], ...]:
    groups = tuple(
        tuple(sorted(shard for shard, peer in mapping.items() if peer == peer_id))
        for peer_id in peer_order
    )
    return refine.validate_groups(groups)


def concurrency_one_service_demand(
    path: Path, peer_order: Sequence[int], peer_hosts: dict[int, str]
) -> list[float]:
    payload = load_json(path)
    row = next(
        value for value in payload["concurrency_sweep"] if value["concurrency"] == 1
    )
    return [
        1_000_000.0 * float(row["cpu_average_cores"][peer_hosts[peer_id]])
        / float(row["qps"])
        for peer_id in peer_order
    ]


def observation_sources(root: Path) -> dict[str, list[Path]]:
    return {
        "round_robin": sorted((root / "online-v1/phases").glob("round-robin-*/benchmark.json")),
        "size_balanced": [root / "online-v1/phases/size-balanced-a/benchmark.json"],
        "controller_aware_v1": [
            *sorted((root / "online-v1/phases").glob("controller-tail-*/benchmark.json")),
            *sorted((root / "online-v2/phases").glob("controller-v1-*/screen.json")),
        ],
        "learned_robust_workcap5500": [
            root / "online-v2/phases/learned-workcap5500/screen.json"
        ],
        "learned_robust": [root / "online-v2/phases/learned-robust/screen.json"],
    }


def build_observations(
    root: Path,
    peer_order: Sequence[int],
    peer_hosts: dict[int, str],
    mappings: dict[str, dict[int, int]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    observations = []
    provenance: dict[str, Any] = {}
    for strategy, paths in observation_sources(root).items():
        if not paths or any(not path.is_file() for path in paths):
            raise FileNotFoundError(f"missing online observations for {strategy}: {paths}")
        replicate_values = [
            concurrency_one_service_demand(path, peer_order, peer_hosts) for path in paths
        ]
        averaged = np.mean(np.asarray(replicate_values, dtype=np.float64), axis=0)
        mapping = mappings[strategy]
        provenance[strategy] = {
            "replicate_count": len(paths),
            "sources": [
                {"path": str(path), "sha256": sha256_path(path)} for path in paths
            ],
            "replicate_node_cpu_microseconds_per_query": replicate_values,
        }
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
    if len(observations) != 20:
        raise RuntimeError(f"expected 20 unique-placement node observations, got {len(observations)}")
    return observations, provenance


def model_design(
    rows: Sequence[dict[str, Any]],
    work_by_shard: Sequence[float],
    controller_overhead: float,
    *,
    include_proxy: bool,
) -> tuple[np.ndarray, list[str]]:
    names = [f"shard_{shard}" for shard in range(EXPECTED_SHARDS)] + ["controller"]
    if include_proxy:
        names.append("work_proxy_load")
    design = []
    for row in rows:
        owned = set(int(shard) for shard in row["shards"])
        values = [1.0 if shard in owned else 0.0 for shard in range(EXPECTED_SHARDS)]
        values.append(1.0 if row["controller"] else 0.0)
        if include_proxy:
            values.append(
                sum(work_by_shard[shard] for shard in owned)
                + (controller_overhead if row["controller"] else 0.0)
            )
        design.append(values)
    return np.asarray(design, dtype=np.float64), names


def fit_centered_ridge(
    name: str,
    observations: Sequence[dict[str, Any]],
    work_by_shard: Sequence[float],
    controller_overhead: float,
    *,
    include_proxy: bool,
) -> dict[str, Any]:
    design, feature_names = model_design(
        observations,
        work_by_shard,
        controller_overhead,
        include_proxy=include_proxy,
    )
    target = np.asarray(
        [float(row["cpu_microseconds_per_query"]) for row in observations],
        dtype=np.float64,
    )
    design_mean = np.mean(design, axis=0)
    target_mean = float(np.mean(target))
    centered = design - design_mean
    coefficients = np.linalg.pinv(
        centered.T @ centered + RIDGE_ALPHA * np.eye(design.shape[1])
    ) @ centered.T @ (target - target_mean)
    intercept = target_mean - float(design_mean @ coefficients)
    predicted = design @ coefficients + intercept
    residual = target - predicted
    return {
        "name": name,
        "claim_boundary": "candidate selector only; cannot establish online QPS",
        "target": "mean CPU microseconds per query at client concurrency 1",
        "unique_placement_count": 5,
        "observation_count": len(observations),
        "ridge_alpha": RIDGE_ALPHA,
        "fit_intercept": True,
        "centered_features": True,
        "feature_names": feature_names,
        "coefficients": [float(value) for value in coefficients],
        "intercept": intercept,
        "rmse": float(math.sqrt(np.mean(np.square(residual)))),
        "mae": float(np.mean(np.abs(residual))),
        "observed": [float(value) for value in target],
        "fitted": [float(value) for value in predicted],
        "residual": [float(value) for value in residual],
        "include_work_proxy": include_proxy,
    }


def predict_mapping(
    model: dict[str, Any],
    mapping: dict[int, int],
    peer_order: Sequence[int],
    peer_hosts: dict[int, str],
    work_by_shard: Sequence[float],
    controller_overhead: float,
) -> dict[str, Any]:
    rows = [
        {
            "controller": node_index == 0,
            "shards": sorted(
                shard for shard, owner in mapping.items() if owner == peer_id
            ),
        }
        for node_index, peer_id in enumerate(peer_order)
    ]
    design, names = model_design(
        rows,
        work_by_shard,
        controller_overhead,
        include_proxy=bool(model["include_work_proxy"]),
    )
    if names != model["feature_names"]:
        raise RuntimeError("model feature order changed")
    values = design @ np.asarray(model["coefficients"]) + float(model["intercept"])
    by_host = {peer_hosts[peer_id]: float(values[index]) for index, peer_id in enumerate(peer_order)}
    return {
        "by_host": by_host,
        "worst_node_cpu_microseconds_per_query": max(by_host.values()),
    }


def swap_mapping(base: dict[int, int], left: int, right: int) -> dict[int, int]:
    if base[left] == base[right]:
        raise ValueError(f"swap {left}<->{right} must cross physical nodes")
    candidate = dict(base)
    candidate[left], candidate[right] = candidate[right], candidate[left]
    return candidate


def build(args: argparse.Namespace) -> Path:
    root = args.root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)

    old_plan_path = root / "plan/placements.json"
    refine_plan_path = root / "refine-v2/placements.json"
    layout_manifest_path = args.layout_dir.expanduser().resolve() / "build-manifest.json"
    trace_path = root / "trace/p24-per-query.json"
    proxy_manifest_path = root / "plan/manifest.json"
    old_plan = load_json(old_plan_path)
    refine_plan = load_json(refine_plan_path)
    layout_manifest = load_json(layout_manifest_path)
    trace = load_json(trace_path)
    peer_order = [int(value) for value in refine_plan["peer_order"]]
    peer_hosts = {int(peer): str(host) for peer, host in refine_plan["peer_hosts"].items()}
    counts = [int(value) for value in layout_manifest["routing"]["shard_counts"]]
    features = refine.shard_features(trace, counts)
    work_by_shard = [1000.0 * row["work_1000"] for row in features]
    controller_overhead = float(
        load_json(proxy_manifest_path)["work_proxy"]["controller_overhead_model"][
            "controller_overhead_work_units"
        ]
    )

    source_mappings = {
        "round_robin": normalize_mapping(old_plan["placements"]["round_robin"]),
        "size_balanced": normalize_mapping(old_plan["placements"]["size_balanced"]),
        **{
            name: normalize_mapping(mapping)
            for name, mapping in refine_plan["placements"].items()
        },
    }
    base = source_mappings["controller_aware_v1"]
    candidate_mappings = {"controller_aware_v1": base}
    candidate_mappings.update(
        {
            name: swap_mapping(base, left, right)
            for name, (left, right) in LOCAL_SWAPS.items()
        }
    )
    for mapping in candidate_mappings.values():
        placement_groups(mapping, peer_order)

    observations, observation_provenance = build_observations(
        root, peer_order, peer_hosts, source_mappings
    )
    models = [
        fit_centered_ridge(
            "direct_shard_cost_ridge",
            observations,
            work_by_shard,
            controller_overhead,
            include_proxy=False,
        ),
        fit_centered_ridge(
            "proxy_augmented_shard_cost_ridge",
            observations,
            work_by_shard,
            controller_overhead,
            include_proxy=True,
        ),
    ]

    metrics: dict[str, Any] = {}
    base_groups = placement_groups(base, peer_order)
    for name, mapping in candidate_mappings.items():
        groups = placement_groups(mapping, peer_order)
        moved = [
            shard for shard in range(EXPECTED_SHARDS) if mapping[shard] != base[shard]
        ]
        proxy_loads = [
            sum(work_by_shard[shard] for shard in group)
            + (controller_overhead if node_index == 0 else 0.0)
            for node_index, group in enumerate(groups)
        ]
        metrics[name] = {
            "groups": groups,
            "swap_from_controller_aware_v1": (
                None
                if name == "controller_aware_v1"
                else {"left_shard": LOCAL_SWAPS[name][0], "right_shard": LOCAL_SWAPS[name][1]}
            ),
            "moved_shard_count_from_v1": len(moved),
            "moved_physical_vector_copies_from_v1": sum(counts[shard] for shard in moved),
            "work_proxy_loads": proxy_loads,
            "work_proxy_peak": max(proxy_loads),
            "work_proxy_peak_delta_from_v1": max(proxy_loads)
            - max(
                sum(work_by_shard[shard] for shard in group)
                + (controller_overhead if node_index == 0 else 0.0)
                for node_index, group in enumerate(base_groups)
            ),
            **refine.trace_peak_metrics(trace, counts, groups, controller_overhead),
            "ridge_predictions": {
                model["name"]: predict_mapping(
                    model,
                    mapping,
                    peer_order,
                    peer_hosts,
                    work_by_shard,
                    controller_overhead,
                )
                for model in models
            },
        }

    placements_path = output / "placements.json"
    write_json_new(
        placements_path,
        {
            "format_version": 1,
            "physical_machine_count": 4,
            "logical_shard_count": EXPECTED_SHARDS,
            "controller_peer_id": refine_plan["controller_peer_id"],
            "peer_order": peer_order,
            "peer_hosts": refine_plan["peer_hosts"],
            "screen_order": [strategy for _phase, strategy in SCREEN_PHASES],
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
    model_path = output / "service-demand-models.json"
    write_json_new(
        model_path,
        {
            "claim_boundary": "offline ridge models select local candidates only; online QPS is authoritative",
            "observations": observations,
            "observation_provenance": observation_provenance,
            "models": models,
        },
    )
    metrics_path = output / "candidate-metrics.json"
    write_json_new(metrics_path, metrics)
    csv_path = output / "candidate-metrics.csv"
    with csv_path.open("x", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "strategy",
            "direct_ridge_worst_cpu_us_per_query",
            "proxy_augmented_ridge_worst_cpu_us_per_query",
            "work_proxy_peak",
            "work_proxy_peak_delta_from_v1",
            "moved_shards_from_v1",
            "moved_copies_from_v1",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for name, row in metrics.items():
            writer.writerow(
                {
                    "strategy": name,
                    "direct_ridge_worst_cpu_us_per_query": row["ridge_predictions"][
                        "direct_shard_cost_ridge"
                    ]["worst_node_cpu_microseconds_per_query"],
                    "proxy_augmented_ridge_worst_cpu_us_per_query": row[
                        "ridge_predictions"
                    ]["proxy_augmented_shard_cost_ridge"][
                        "worst_node_cpu_microseconds_per_query"
                    ],
                    "work_proxy_peak": row["work_proxy_peak"],
                    "work_proxy_peak_delta_from_v1": row[
                        "work_proxy_peak_delta_from_v1"
                    ],
                    "moved_shards_from_v1": row["moved_shard_count_from_v1"],
                    "moved_copies_from_v1": row[
                        "moved_physical_vector_copies_from_v1"
                    ],
                }
            )

    manifest_path = output / "manifest.json"
    outputs = [placements_path, model_path, metrics_path, csv_path]
    write_json_new(
        manifest_path,
        {
            "format_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "status": "OFFLINE_REFINEMENT_ONLY",
            "claim_boundary": "three one-swap candidates around the stable online winner; no offline model establishes QPS or global optimality",
            "physical_machine_count": 4,
            "logical_shard_count": EXPECTED_SHARDS,
            "modeling": {
                "fit_source": "concurrency=1 service demand averaged within five unique online placements",
                "unique_placement_count": 5,
                "observation_count": len(observations),
                "ridge_alpha": RIDGE_ALPHA,
                "candidate_scope": "three preselected one-swap neighbors of controller-aware v1",
            },
            "inputs": {
                "old_plan": {"path": str(old_plan_path), "sha256": sha256_path(old_plan_path)},
                "refine_v2_plan": {
                    "path": str(refine_plan_path),
                    "sha256": sha256_path(refine_plan_path),
                },
                "layout_manifest": {
                    "path": str(layout_manifest_path),
                    "sha256": sha256_path(layout_manifest_path),
                },
                "trace": {"path": str(trace_path), "sha256": sha256_path(trace_path)},
                "proxy_manifest": {
                    "path": str(proxy_manifest_path),
                    "sha256": sha256_path(proxy_manifest_path),
                },
            },
            "outputs": {
                path.name: {"path": str(path), "sha256": sha256_path(path)}
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
