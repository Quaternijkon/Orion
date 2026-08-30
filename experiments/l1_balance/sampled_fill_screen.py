#!/usr/bin/env python3
"""Screen a deterministic sampled-L0 proxy mass for frozen CNBR 9/4.

The experiment deliberately remains outside the production owner builder.  It
uses sampled full-attachment rows only to estimate one mass per upper node,
keeps the existing upper self-navigation target rows and CNBR repair rule, and
then evaluates the frozen result against full attachments as ground truth.
"""

from __future__ import annotations

import argparse
import csv
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.l1_balance import native_cnbr_core as core  # noqa: E402
from experiments.l1_balance.evaluate_cnbr_frozen_owners import (  # noqa: E402
    assignment_views,
    query_gate_results,
)
from experiments.l1_balance.run_offline_screen import (  # noqa: E402
    checked_binary_matrix,
    load_upper,
    local_hits,
    query_metrics,
)


DEFAULT_FRACTIONS = (0.001, 0.0025, 0.005, 0.01, 0.02, 0.05)
DEFAULT_SEEDS = (1, 7, 19, 42, 100)
NUM_PARTITIONS = 32
MULTI_ASSIGNMENT_CONTRACT = {
    "enabled": True,
    "min_max_vote": 2,
    "vote_delta": 0,
    "max_shards": 0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-spec",
        action="append",
        required=True,
        metavar="NAME:PHASE_A_MANIFEST:PHASE_B_MANIFEST",
    )
    parser.add_argument(
        "--fractions", nargs="+", type=float, default=list(DEFAULT_FRACTIONS)
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument(
        "--sample-weights",
        nargs="+",
        type=float,
        default=[1.0],
        help="Weight of normalized sampled mass versus the upper-self prior",
    )
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def deterministic_sample_indices(
    row_count: int, fraction: float, seed: int
) -> np.ndarray:
    if row_count <= 0:
        raise ValueError("row_count must be positive")
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    sample_count = max(1, int(round(row_count * fraction)))
    rng = np.random.default_rng(int(seed))
    return np.sort(rng.choice(row_count, size=sample_count, replace=False))


def occurrence_mass(sample_local: np.ndarray, upper_count: int) -> np.ndarray:
    if sample_local.ndim != 2 or len(sample_local) <= 0:
        raise ValueError("sample_local must be a non-empty matrix")
    if np.any(sample_local < 0) or np.any(sample_local >= upper_count):
        raise ValueError("sample_local contains an invalid upper node")
    return np.bincount(
        np.asarray(sample_local, dtype=np.int64).reshape(-1),
        minlength=upper_count,
    ).astype(np.uint64, copy=False)


def blended_proxy_mass(
    sample_mass: np.ndarray,
    upper_mass: np.ndarray,
    sample_weight: float,
) -> np.ndarray:
    """Blend normalized sample evidence with the current upper-only prior.

    Both components are normalized to the same total using integer arithmetic;
    the common denominator is intentionally retained because CNBR is invariant
    to uniform mass scaling.
    """

    sample = np.asarray(sample_mass)
    upper = np.asarray(upper_mass)
    if sample.shape != upper.shape or sample.ndim != 1:
        raise ValueError("sample and upper masses must have the same vector shape")
    if sample.dtype.kind not in "iu" or upper.dtype.kind not in "iu":
        raise TypeError("sample and upper masses must contain integers")
    weight = Fraction(str(float(sample_weight))).limit_denominator(1000)
    if weight < 0 or weight > 1:
        raise ValueError("sample_weight must be in [0, 1]")
    sample_total = int(np.sum(sample, dtype=np.uint64))
    upper_total = int(np.sum(upper, dtype=np.uint64))
    if sample_total <= 0 or upper_total <= 0:
        raise ValueError("sample and upper masses must have positive totals")
    numerator = weight.numerator
    denominator = weight.denominator
    result = (
        (denominator - numerator)
        * np.asarray(upper, dtype=np.uint64)
        * sample_total
        + numerator
        * np.asarray(sample, dtype=np.uint64)
        * upper_total
    )
    return np.asarray(result, dtype=np.uint64)


def load_distribution(loads: np.ndarray, prefix: str) -> dict[str, float | int]:
    values = np.asarray(loads, dtype=np.float64)
    mean = float(np.mean(values))
    return {
        f"{prefix}_min": int(np.min(values)),
        f"{prefix}_max": int(np.max(values)),
        f"{prefix}_mean": mean,
        f"{prefix}_cv": float(np.std(values) / mean),
        f"{prefix}_max_over_mean": float(np.max(values) / mean),
        f"{prefix}_min_over_mean": float(np.min(values) / mean),
        f"{prefix}_empty_shards": int(np.count_nonzero(values == 0)),
    }


def _average_ranks(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=np.float64)
    start = 0
    while start < len(array):
        stop = start + 1
        while stop < len(array) and array[order[stop]] == array[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks


def prediction_metrics(
    sampled_loads: np.ndarray, full_loads: np.ndarray
) -> dict[str, float]:
    sampled = np.asarray(sampled_loads, dtype=np.float64)
    full = np.asarray(full_loads, dtype=np.float64)
    if sampled.shape != full.shape or sampled.ndim != 1:
        raise ValueError("sampled and full loads must have the same vector shape")
    if float(np.sum(sampled)) <= 0.0 or np.any(full <= 0.0):
        raise ValueError("prediction loads must be positive")
    scaled = sampled * (float(np.sum(full)) / float(np.sum(sampled)))
    relative = np.abs(scaled - full) / full
    pearson = float(np.corrcoef(sampled, full)[0, 1])
    spearman = float(
        np.corrcoef(_average_ranks(sampled), _average_ranks(full))[0, 1]
    )
    return {
        "prediction_mape": float(np.mean(relative)),
        "prediction_max_abs_relative_error": float(np.max(relative)),
        "prediction_rmse_over_full_mean": float(
            np.sqrt(np.mean((scaled - full) ** 2)) / np.mean(full)
        ),
        "prediction_pearson": pearson,
        "prediction_spearman": spearman,
    }


def owner_sha256(owner: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(owner, dtype="<i4").tobytes(order="C")
    ).hexdigest()


def _parse_dataset_spec(raw: str) -> tuple[str, Path, Path]:
    parts = raw.split(":", 2)
    if len(parts) != 3 or not parts[0]:
        raise ValueError(
            "dataset-spec must be NAME:PHASE_A_MANIFEST:PHASE_B_MANIFEST"
        )
    return (
        parts[0],
        Path(parts[1]).expanduser().resolve(),
        Path(parts[2]).expanduser().resolve(),
    )


def _resolve_relative(path: str, parent: Path) -> Path:
    value = Path(path).expanduser()
    return (parent / value).resolve() if not value.is_absolute() else value.resolve()


def _evaluate_owner(
    owner: np.ndarray,
    attachment_local: np.ndarray,
    query_local: np.ndarray,
    ground_truth: np.ndarray,
    labels: np.ndarray,
    *,
    dynamic_ef_base: int,
    dynamic_ef_factor: int,
) -> tuple[dict[str, Any], np.ndarray]:
    membership, copy_count, physical_loads, _primary, primary_loads = assignment_views(
        owner, attachment_local, NUM_PARTITIONS
    )
    metrics: dict[str, Any] = {
        "logical_point_count": int(len(attachment_local)),
        "physical_point_count": int(np.sum(physical_loads)),
        "expansion_ratio": float(np.mean(copy_count)),
        **load_distribution(primary_loads, "primary_load"),
        **load_distribution(physical_loads, "physical_copy_load"),
    }
    metrics.update(
        query_metrics(
            owner,
            query_local,
            membership[labels],
            membership,
            ground_truth,
            dynamic_ef_base,
            dynamic_ef_factor,
        )
    )
    return metrics, np.asarray(physical_loads, dtype=np.int64).copy()


def _load_dataset(
    name: str, phase_a_path: Path, phase_b_path: Path
) -> dict[str, Any]:
    phase_a = json.loads(phase_a_path.read_text(encoding="utf-8"))
    phase_b = json.loads(phase_b_path.read_text(encoding="utf-8"))
    construction = phase_a["construction_inputs"]
    artifact_path = Path(construction["artifact"]["path"]).resolve()
    (
        artifact,
        adjacency,
        _vectors,
        labels,
        _entry,
        _edge_left,
        _edge_right,
        _navigator_sha,
    ) = load_upper(artifact_path)
    upper_count = len(labels)
    if int(artifact["shard_count"]) != NUM_PARTITIONS:
        raise ValueError(f"{name}: expected exactly {NUM_PARTITIONS} logical shards")

    parent = phase_a_path.parent
    owners = {
        owner_name: np.asarray(
            checked_binary_matrix(
                _resolve_relative(record["owner"]["path"], parent),
                rows=upper_count,
                width=1,
                dtype="<i4",
            )
        ).reshape(-1).copy()
        for owner_name, record in phase_a["owners"].items()
        if owner_name in {"N_native", "C_CNBR"}
    }
    if set(owners) != {"N_native", "C_CNBR"}:
        raise ValueError(f"{name}: frozen N_native/C_CNBR owners are missing")

    upper_mass_record = phase_a["mass"]["values"]
    upper_mass = np.asarray(
        checked_binary_matrix(
            _resolve_relative(upper_mass_record["path"], parent),
            rows=upper_count,
            width=1,
            dtype="<u8",
        )
    ).reshape(-1).copy()

    self_record = construction["self_navigation"]
    self_hits = checked_binary_matrix(
        Path(self_record["path"]).resolve(),
        rows=upper_count,
        width=int(self_record["top_k"]),
        dtype="<u8",
    )
    navigation_local = local_hits(self_hits, labels, upper_count)

    inputs = phase_b["post_freeze_evaluation_inputs"]
    attachment_manifest = json.loads(
        Path(inputs["attachments_manifest"]).read_text(encoding="utf-8")
    )
    row_count = int(attachment_manifest["row_count"])
    attachment_k = int(attachment_manifest["top_k"])
    attachment_hits = checked_binary_matrix(
        Path(inputs["attachments"]).resolve(),
        rows=row_count,
        width=attachment_k,
        dtype="<u8",
    )
    mapping_start = time.perf_counter()
    attachment_local = local_hits(attachment_hits, labels, row_count)
    full_attachment_mapping_wall_seconds = time.perf_counter() - mapping_start

    query_manifest = json.loads(
        Path(inputs["query_hits_manifest"]).read_text(encoding="utf-8")
    )
    query_rows = int(query_manifest["row_count"])
    query_hits = checked_binary_matrix(
        Path(inputs["query_hits"]).resolve(),
        rows=query_rows,
        width=int(query_manifest["top_k"]),
        dtype="<u8",
    )
    query_local = local_hits(query_hits, labels, query_rows)
    ground_truth_manifest = json.loads(
        Path(inputs["ground_truth_manifest"]).read_text(encoding="utf-8")
    )
    ground_truth = checked_binary_matrix(
        Path(inputs["ground_truth"]).resolve(),
        rows=query_rows,
        width=int(ground_truth_manifest["width"]),
        dtype="<u4",
    )

    performance_path = phase_a_path.with_name("phase-a.performance.json")
    performance = json.loads(performance_path.read_text(encoding="utf-8"))
    full_attachment_seconds = float(attachment_manifest["elapsed_seconds"])
    self_navigation_seconds = float(
        json.loads(Path(self_record["manifest_path"]).read_text(encoding="utf-8"))[
            "elapsed_seconds"
        ]
    )
    validation_seconds = float(
        performance["shared"][
            "required_input_validation_and_mass_replay_wall_seconds"
        ]
    )
    current_cnbr_seconds = float(performance["owners"]["C_CNBR"]["wall_seconds"])
    return {
        "name": name,
        "phase_a_path": phase_a_path,
        "phase_b_path": phase_b_path,
        "artifact": artifact,
        "adjacency": adjacency,
        "labels": labels,
        "navigation_local": navigation_local,
        "owners": owners,
        "upper_mass": upper_mass,
        "attachment_local": attachment_local,
        "query_local": query_local,
        "ground_truth": ground_truth,
        "row_count": row_count,
        "upper_count": upper_count,
        "query_count": query_rows,
        "full_attachment_mapping_wall_seconds": full_attachment_mapping_wall_seconds,
        "full_attachment_seconds": full_attachment_seconds,
        "self_navigation_seconds": self_navigation_seconds,
        "validation_seconds": validation_seconds,
        "current_cnbr_seconds": current_cnbr_seconds,
    }


def _screen_dataset(
    dataset: dict[str, Any],
    fractions: list[float],
    seeds: list[int],
    sample_weights: list[float],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    name = dataset["name"]
    n_owner = dataset["owners"]["N_native"]
    current_owner = dataset["owners"]["C_CNBR"]
    artifact = dataset["artifact"]
    evaluation_args = {
        "attachment_local": dataset["attachment_local"],
        "query_local": dataset["query_local"],
        "ground_truth": dataset["ground_truth"],
        "labels": dataset["labels"],
        "dynamic_ef_base": int(artifact["dynamic_ef_base"]),
        "dynamic_ef_factor": int(artifact["dynamic_ef_factor"]),
    }
    baseline_start = time.perf_counter()
    n_metrics, n_full_loads = _evaluate_owner(n_owner, **evaluation_args)
    current_metrics, current_full_loads = _evaluate_owner(
        current_owner, **evaluation_args
    )
    baseline_wall_seconds = time.perf_counter() - baseline_start
    current_cost_ratio = (
        dataset["self_navigation_seconds"]
        + dataset["validation_seconds"]
        + dataset["current_cnbr_seconds"]
    ) / dataset["full_attachment_seconds"]
    baseline = {
        "N_native": n_metrics,
        "C_CNBR": current_metrics,
        "N_native_owner_sha256": owner_sha256(n_owner),
        "C_CNBR_owner_sha256": owner_sha256(current_owner),
        "baseline_evaluation_wall_seconds": baseline_wall_seconds,
        "current_cnbr_incremental_cost_ratio": current_cost_ratio,
    }

    rows: list[dict[str, Any]] = []
    for fraction in fractions:
        for seed in seeds:
            indices = deterministic_sample_indices(
                dataset["row_count"], fraction, seed
            )
            sample_start = time.perf_counter()
            sample_local = np.asarray(dataset["attachment_local"][indices]).copy()
            raw_sample_mass = occurrence_mass(sample_local, dataset["upper_count"])
            sample_preprocess_seconds = time.perf_counter() - sample_start
            for sample_weight in sample_weights:
                mass = blended_proxy_mass(
                    raw_sample_mass, dataset["upper_mass"], sample_weight
                )
                repair_start = time.perf_counter()
                result = core.build_cnbr_owner_with_external_mass_for_research(
                    dataset["adjacency"],
                    dataset["navigation_local"],
                    mass,
                    n_owner,
                    NUM_PARTITIONS,
                )
                repair_seconds = time.perf_counter() - repair_start
                owner = np.asarray(result.owner, dtype=np.int32)

                sample_eval_start = time.perf_counter()
                (
                    _sample_membership,
                    sample_copy_count,
                    sample_physical_loads,
                    _sample_primary,
                    sample_primary_loads,
                ) = assignment_views(owner, sample_local, NUM_PARTITIONS)
                sample_evaluation_seconds = time.perf_counter() - sample_eval_start

                full_eval_start = time.perf_counter()
                full_metrics, full_physical_loads = _evaluate_owner(
                    owner, **evaluation_args
                )
                full_evaluation_seconds = time.perf_counter() - full_eval_start
                query_gates = query_gate_results(
                    full_metrics, n_metrics, dataset["query_count"]
                )
                graph_pass = all(
                    bool(record["pass"])
                    for record in result.topology_gates.values()
                )
                query_pass = all(
                    bool(record["pass"]) for record in query_gates.values()
                )

                actual_fraction = len(indices) / dataset["row_count"]
                sample_navigation_seconds = (
                    actual_fraction * dataset["full_attachment_seconds"]
                )
                incremental_seconds = (
                    dataset["self_navigation_seconds"]
                    + dataset["validation_seconds"]
                    + sample_navigation_seconds
                    + sample_preprocess_seconds
                    + repair_seconds
                )
                incremental_cost_ratio = (
                    incremental_seconds / dataset["full_attachment_seconds"]
                )
                better_max = float(
                    full_metrics["physical_copy_load_max_over_mean"]
                ) < float(current_metrics["physical_copy_load_max_over_mean"])
                no_worse_cv = float(full_metrics["physical_copy_load_cv"]) <= float(
                    current_metrics["physical_copy_load_cv"]
                )
                cost_pass = incremental_cost_ratio <= 0.05
                row: dict[str, Any] = {
                    "dataset": name,
                    "estimator": "normalized_sample_upper_prior_blend",
                    "sample_weight": float(sample_weight),
                    "upper_prior_weight": float(1.0 - sample_weight),
                    "fraction_requested": float(fraction),
                    "fraction_actual": float(actual_fraction),
                    "seed": int(seed),
                    "sample_row_count": int(len(indices)),
                    "sample_index_sha256": hashlib.sha256(
                        np.ascontiguousarray(indices, dtype="<i8").tobytes(order="C")
                    ).hexdigest(),
                    "raw_sample_mass_total": int(
                        np.sum(raw_sample_mass, dtype=np.uint64)
                    ),
                    "sample_mass_total": int(np.sum(mass, dtype=np.uint64)),
                    "sample_mass_nonzero_nodes": int(np.count_nonzero(raw_sample_mass)),
                    "sample_mass_nonzero_fraction": float(
                        np.count_nonzero(raw_sample_mass) / len(raw_sample_mass)
                    ),
                    "sample_mass_max": int(np.max(raw_sample_mass)),
                    "sample_mass_cv": float(
                        np.std(raw_sample_mass) / np.mean(raw_sample_mass)
                    ),
                    "owner_sha256": owner_sha256(owner),
                    "l1_min_size": int(
                        np.min(np.bincount(owner, minlength=NUM_PARTITIONS))
                    ),
                    "l1_max_size": int(
                        np.max(np.bincount(owner, minlength=NUM_PARTITIONS))
                    ),
                    "moved_l1_count": int(len(result.moved_nodes)),
                    "round_count": int(len(result.rounds)),
                    "proposal_count": int(
                        sum(int(record["proposal_count"]) for record in result.rounds)
                    ),
                    "committed_move_count": int(
                        sum(
                            int(record["committed_move_count"])
                            for record in result.rounds
                        )
                    ),
                    "graph_topology_all_pass": graph_pass,
                    "query_topology_all_pass": query_pass,
                    "sample_preprocess_wall_seconds": sample_preprocess_seconds,
                    "sample_repair_wall_seconds": repair_seconds,
                    "sample_evaluation_wall_seconds": sample_evaluation_seconds,
                    "full_evaluation_wall_seconds": full_evaluation_seconds,
                    "sample_navigation_wall_seconds_estimate": sample_navigation_seconds,
                    "incremental_wall_seconds_estimate": incremental_seconds,
                    "incremental_cost_ratio_estimate": incremental_cost_ratio,
                    "incremental_cost_gate_pass": cost_pass,
                    "current_cnbr_incremental_cost_ratio": current_cost_ratio,
                    "physical_max_over_mean_better_than_current_cnbr": better_max,
                    "physical_cv_no_worse_than_current_cnbr": no_worse_cv,
                    "offline_eligible": bool(
                        graph_pass
                        and query_pass
                        and cost_pass
                        and better_max
                        and no_worse_cv
                    ),
                    "sample_expansion_ratio": float(np.mean(sample_copy_count)),
                    **load_distribution(
                        np.asarray(result.partition_masses, dtype=np.int64),
                        "proxy_mass_load",
                    ),
                    **{
                        f"graph_{metric}": value
                        for metric, value in result.topology_metrics.items()
                    },
                    **load_distribution(sample_primary_loads, "sample_primary_load"),
                    **load_distribution(
                        sample_physical_loads, "sample_physical_copy_load"
                    ),
                    **full_metrics,
                    **prediction_metrics(sample_physical_loads, full_physical_loads),
                }
                for metric, gate in result.topology_gates.items():
                    row[f"graph_gate_{metric}_pass"] = bool(gate["pass"])
                for metric, gate in query_gates.items():
                    row[f"query_gate_{metric}_pass"] = bool(gate["pass"])
                rows.append(row)
                print(
                    f"{name} fraction={fraction:g} seed={seed} "
                    f"weight={sample_weight:g} "
                    f"max/mean={row['physical_copy_load_max_over_mean']:.4f} "
                    f"query_gate={query_pass} cost={incremental_cost_ratio:.4%} "
                    f"eligible={row['offline_eligible']}",
                    flush=True,
                )
                del _sample_membership

    return rows, baseline


def _aggregate(
    rows: list[dict[str, Any]], seeds: list[int]
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, float, float], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(
            (
                str(row["dataset"]),
                float(row["fraction_requested"]),
                float(row["sample_weight"]),
            ),
            [],
        ).append(row)
    summary: list[dict[str, Any]] = []
    for (dataset, fraction, sample_weight), group in sorted(groups.items()):
        values = lambda key: np.asarray([float(row[key]) for row in group])
        summary.append(
            {
                "dataset": dataset,
                "fraction": fraction,
                "sample_weight": sample_weight,
                "upper_prior_weight": 1.0 - sample_weight,
                "seed_count": len(group),
                "expected_seed_count": len(seeds),
                "offline_eligible_seed_count": int(
                    sum(bool(row["offline_eligible"]) for row in group)
                ),
                "all_seeds_offline_eligible": bool(
                    len(group) == len(seeds)
                    and all(bool(row["offline_eligible"]) for row in group)
                ),
                "query_gate_pass_seed_count": int(
                    sum(bool(row["query_topology_all_pass"]) for row in group)
                ),
                "cost_gate_pass_seed_count": int(
                    sum(bool(row["incremental_cost_gate_pass"]) for row in group)
                ),
                "physical_max_over_mean_min": float(
                    np.min(values("physical_copy_load_max_over_mean"))
                ),
                "physical_max_over_mean_median": float(
                    np.median(values("physical_copy_load_max_over_mean"))
                ),
                "physical_max_over_mean_max": float(
                    np.max(values("physical_copy_load_max_over_mean"))
                ),
                "physical_cv_median": float(
                    np.median(values("physical_copy_load_cv"))
                ),
                "prediction_pearson_median": float(
                    np.median(values("prediction_pearson"))
                ),
                "prediction_spearman_median": float(
                    np.median(values("prediction_spearman"))
                ),
                "prediction_mape_median": float(
                    np.median(values("prediction_mape"))
                ),
                "incremental_cost_ratio_max": float(
                    np.max(values("incremental_cost_ratio_estimate"))
                ),
                "moved_l1_count_median": float(
                    np.median(values("moved_l1_count"))
                ),
            }
        )
    return summary


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output_dir}")
    fractions = sorted(set(float(value) for value in args.fractions))
    seeds = list(dict.fromkeys(int(value) for value in args.seeds))
    sample_weights = sorted(set(float(value) for value in args.sample_weights))
    if not fractions or not seeds or not sample_weights:
        raise ValueError("at least one fraction, seed, and sample weight are required")
    if any(weight < 0.0 or weight > 1.0 for weight in sample_weights):
        raise ValueError("sample weights must be in [0, 1]")

    output_dir.mkdir(parents=True, exist_ok=False)
    all_rows: list[dict[str, Any]] = []
    baselines: dict[str, Any] = {}
    dataset_inputs: dict[str, Any] = {}
    for raw_spec in args.dataset_spec:
        name, phase_a_path, phase_b_path = _parse_dataset_spec(raw_spec)
        if name in baselines:
            raise ValueError(f"duplicate dataset name: {name}")
        dataset = _load_dataset(name, phase_a_path, phase_b_path)
        rows, baseline = _screen_dataset(
            dataset, fractions, seeds, sample_weights
        )
        all_rows.extend(rows)
        baselines[name] = baseline
        dataset_inputs[name] = {
            "phase_a_manifest": str(phase_a_path),
            "phase_b_manifest": str(phase_b_path),
            "row_count": dataset["row_count"],
            "upper_count": dataset["upper_count"],
            "full_attachment_seconds": dataset["full_attachment_seconds"],
            "self_navigation_seconds": dataset["self_navigation_seconds"],
            "required_validation_seconds": dataset["validation_seconds"],
            "current_cnbr_repair_seconds": dataset["current_cnbr_seconds"],
            "full_attachment_mapping_wall_seconds": dataset[
                "full_attachment_mapping_wall_seconds"
            ],
        }

    aggregate = _aggregate(all_rows, seeds)
    manifest = {
        "format_version": 1,
        "experiment": "cnbr-deterministic-sampled-l0-fill-screen-v1",
        "status": "offline_research_only",
        "design_boundary": {
            "satisfies_current_balanced_md_upper_only_contract": False,
            "reason": "sampled L0 attachments are read before owner freeze",
            "production_path_modified": False,
            "multi_assignment_unchanged": True,
        },
        "estimator": {
            "sampling": "uniform_without_replacement",
            "mass": "convex blend after equal-total normalization",
            "sample_component": "raw occurrence count over sampled L0 top10 attachments",
            "prior_component": "current upper-self-navigation raw occurrence mass",
            "repair": "frozen CNBR trigger 9/4",
            "target_candidates": "existing frozen upper self-navigation top10",
        },
        "cost_contract": {
            "formula": "(upper_self_navigation + required_validation + sampled_navigation_estimate + sample_preprocess + CNBR_repair) / full_attachment",
            "sampled_navigation_estimate": "sample_fraction * measured_full_attachment_wall_seconds",
            "gate": 0.05,
        },
        "multi_assignment": MULTI_ASSIGNMENT_CONTRACT,
        "fractions": fractions,
        "sample_weights": sample_weights,
        "seeds": seeds,
        "datasets": dataset_inputs,
        "baselines": baselines,
        "rows": all_rows,
        "aggregate": aggregate,
    }
    (output_dir / "screen-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_csv(output_dir / "results.csv", all_rows)
    _write_csv(output_dir / "aggregate.csv", aggregate)


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
