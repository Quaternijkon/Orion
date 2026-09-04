#!/usr/bin/env python3
"""Audit and aggregate the C23 E3 local-search and E4 oracle-fan-out matrix."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.c23.scripts import c23_e1, c23_protocol


LOCAL_RECALL_THRESHOLDS = (0.80, 0.90, 0.95)


def summarize(values: np.ndarray, prefix: str) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        raise ValueError(f"cannot summarize empty {prefix} values")
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_p95": float(np.percentile(values, 95)),
        f"{prefix}_std": float(np.std(values)),
    }


def write_csv_new(path: str | Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    target = Path(path).expanduser().resolve()
    if target.exists():
        raise FileExistsError(f"refusing to overwrite: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_e2_per_query(path: str | Path) -> dict[tuple[str, int, int], dict[str, Any]]:
    result = {}
    with Path(path).expanduser().resolve().open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (
                str(row["partition_method"]),
                int(row["logical_shards"]),
                int(row["query_id"]),
            )
            if key in result:
                raise ValueError(f"duplicate E2 per-query row: {key}")
            result[key] = {
                "global_trace_length": int(row["global_trace_length"]),
                "path_shard_count": int(row["path_shard_count"]),
                "path_transition_count": int(row["path_transition_count"]),
            }
    return result


def expected_truth_groups(
    truth: np.ndarray,
    assignments: np.ndarray,
    measurement_start: int,
) -> tuple[list[dict[int, list[int]]], np.ndarray]:
    groups: list[dict[int, list[int]]] = []
    participation = np.zeros(int(np.max(assignments)) + 1, dtype=np.int64)
    for row in truth:
        per_shard: dict[int, list[int]] = defaultdict(list)
        for point_id in row:
            shard_id = int(assignments[int(point_id)])
            per_shard[shard_id].append(int(point_id))
        for shard_id in per_shard:
            participation[shard_id] += 1
        groups.append(dict(per_shard))
    if not groups:
        raise ValueError(f"no measurement truth groups from query {measurement_start}")
    return groups, participation


def minimum_shards_for_nine(contributions: np.ndarray) -> int:
    cumulative = 0
    for shard_count, contribution in enumerate(np.sort(contributions)[::-1], 1):
        cumulative += int(contribution)
        if cumulative >= 9:
            return shard_count
    return len(contributions) + 1


def validate_result_ids(result_ids: list[int], assignments: np.ndarray, shard_id: int) -> None:
    if len(result_ids) != len(set(result_ids)):
        raise ValueError("local search returned duplicate IDs")
    if any(point_id < 0 or point_id >= len(assignments) for point_id in result_ids):
        raise ValueError("local result ID lies outside the point range")
    if any(int(assignments[point_id]) != shard_id for point_id in result_ids):
        raise ValueError("local search returned an ID from a different shard")


def analyze_configuration(
    dataset: str,
    spec: c23_protocol.DatasetSpec,
    measurement_start: int,
    common_ef: int,
    configuration: dict[str, Any],
    run: dict[str, Any],
    assignments: np.ndarray,
    truth: np.ndarray,
    e2_rows: dict[tuple[str, int, int], dict[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    method = str(configuration["partition_method"])
    partition_seed = int(configuration["partition_seed"])
    logical_shards = int(configuration["logical_shards"])
    run_dir = Path(run["run_dir"]).expanduser().resolve()
    root_manifest_path = run_dir / "manifest.json"
    root_manifest = json.loads(root_manifest_path.read_text(encoding="utf-8"))
    expected_root = {
        "dataset": dataset,
        "dataset_sha256": spec.sha256,
        "partition_method": method,
        "partition_seed": partition_seed,
        "partition_sha256": configuration["raw_assignment_sha256"],
        "logical_shards": logical_shards,
        "point_count": len(assignments),
        "dimension": spec.dimension,
        "hnsw_m": c23_protocol.HNSW_M,
        "hnsw_ef_construction": c23_protocol.HNSW_EF_CONSTRUCTION,
        "hnsw_full_scan_threshold_kb": 10,
        "hnsw_graph_seed": c23_protocol.HNSW_GRAPH_BUILD_SEED,
        "measurement_start": measurement_start,
        "measurement_query_count": len(truth),
        "e3_ef_values": [10, 20, 40, 80, 160, 320],
        "e4_ef_search": common_ef,
    }
    mismatches = {
        key: {"expected": value, "actual": root_manifest.get(key)}
        for key, value in expected_root.items()
        if root_manifest.get(key) != value
    }
    if mismatches:
        raise ValueError(f"local root-manifest mismatch for {run_dir}: {mismatches}")
    if c23_e1.sha256_path(configuration["raw_assignment_path"]) != str(
        configuration["raw_assignment_sha256"]
    ):
        raise ValueError("raw assignment checksum mismatch")
    if len(root_manifest["shards"]) != logical_shards:
        raise ValueError("local root manifest has incomplete shard coverage")

    truth_groups, participation = expected_truth_groups(truth, assignments, measurement_start)
    pair_keys = [
        (query_offset, shard_id)
        for query_offset, groups in enumerate(truth_groups)
        for shard_id in sorted(groups)
    ]
    pair_index = {key: index for index, key in enumerate(pair_keys)}
    ef_values = np.asarray(root_manifest["e3_ef_values"], dtype=np.int64)
    ef_index = {int(ef): index for index, ef in enumerate(ef_values)}
    pair_count = len(pair_keys)
    ef_count = len(ef_values)
    recall = np.full((pair_count, ef_count), np.nan, dtype=np.float64)
    recovered_count = np.zeros((pair_count, ef_count), dtype=np.int16)
    target_count = np.asarray(
        [len(truth_groups[query][shard]) for query, shard in pair_keys],
        dtype=np.int16,
    )
    distance = np.zeros((pair_count, ef_count), dtype=np.uint64)
    nodes = np.zeros((pair_count, ef_count), dtype=np.uint64)
    latency = np.zeros((pair_count, ef_count), dtype=np.uint64)
    capped_rank = np.zeros((pair_count, ef_count), dtype=np.float64)
    seen_e3 = np.zeros((pair_count, ef_count), dtype=np.bool_)
    e4_recovered = np.zeros((len(truth), logical_shards), dtype=np.int16)
    e4_distance = np.zeros((len(truth), logical_shards), dtype=np.uint64)
    e4_seen = np.zeros((len(truth), logical_shards), dtype=np.bool_)
    shard_audit_rows: list[dict[str, Any]] = []

    total_local_points = 0
    for shard_record in root_manifest["shards"]:
        shard_id = int(shard_record["shard_id"])
        shard_manifest_path = Path(shard_record["manifest"])
        if c23_e1.sha256_path(shard_manifest_path) != str(shard_record["manifest_sha256"]):
            raise ValueError(f"shard-manifest checksum mismatch: {shard_manifest_path}")
        shard_manifest = json.loads(shard_manifest_path.read_text(encoding="utf-8"))
        local_count = int(shard_manifest["local_point_count"])
        expected_count = int(np.count_nonzero(assignments == shard_id))
        if local_count != expected_count:
            raise ValueError(f"local point count mismatch for shard {shard_id}")
        total_local_points += local_count
        average_vector_bytes = int(shard_manifest["hnsw_average_vector_bytes"])
        expected_entry = c23_protocol.expected_hnsw_entry_point_parameters(
            local_count,
            int(shard_manifest["dimension"]),
            int(shard_manifest["hnsw_full_scan_threshold_kb"]),
        )
        recorded_entry = {
            "average_vector_bytes": average_vector_bytes,
            "full_scan_threshold_points": int(
                shard_manifest["hnsw_full_scan_threshold_points"]
            ),
            "entry_points_num": int(shard_manifest["hnsw_entry_points_num"]),
        }
        if recorded_entry != expected_entry:
            raise ValueError(f"entry-point calculation mismatch for shard {shard_id}")
        graph_files = [Path(path) for path in shard_manifest["files"]["graph"]]
        graph_hashes = {str(path): c23_e1.sha256_path(path) for path in graph_files}

        e3_path = Path(shard_manifest["files"]["e3_query_shard"])
        e4_path = Path(shard_manifest["files"]["e4_query_shard"])
        if c23_e1.sha256_path(e3_path) != str(
            shard_manifest["files"]["e3_query_shard_sha256"]
        ):
            raise ValueError(f"E3 checksum mismatch for shard {shard_id}")
        if c23_e1.sha256_path(e4_path) != str(
            shard_manifest["files"]["e4_query_shard_sha256"]
        ):
            raise ValueError(f"E4 checksum mismatch for shard {shard_id}")

        e3_rows = 0
        with e3_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                query_id = int(row["query_id"])
                query_offset = query_id - measurement_start
                row_shard = int(row["shard_id"])
                ef = int(row["ef_search"])
                if row_shard != shard_id or not (0 <= query_offset < len(truth)):
                    raise ValueError("invalid E3 query or shard ID")
                key = (query_offset, shard_id)
                if key not in pair_index or ef not in ef_index:
                    raise ValueError("unexpected E3 query-shard or EF row")
                pair = pair_index[key]
                column = ef_index[ef]
                if seen_e3[pair, column]:
                    raise ValueError("duplicate E3 query-shard-EF row")
                expected_targets = truth_groups[query_offset][shard_id]
                targets = [int(value) for value in row["ground_truth_ids_in_shard"]]
                if targets != expected_targets:
                    raise ValueError("E3 local ground truth differs from assignment-derived truth")
                results = [int(value) for value in row["result_ids"]]
                validate_result_ids(results, assignments, shard_id)
                ranks = [results.index(target) + 1 if target in results else 0 for target in targets]
                if ranks != [int(value) for value in row["local_target_ranks"]]:
                    raise ValueError("E3 local target ranks do not reproduce")
                recovered = sum(rank > 0 for rank in ranks)
                if recovered != int(row["local_recovered_ground_truth"]):
                    raise ValueError("E3 recovered target count does not reproduce")
                reproduced_recall = recovered / len(targets)
                if not math.isclose(reproduced_recall, float(row["local_target_recall"]), abs_tol=1e-12):
                    raise ValueError("E3 local target recall does not reproduce")
                work = int(row["distance_computations"])
                visited = int(row["graph_nodes_visited"])
                local_latency = int(row["local_latency_us"])
                if work <= 0 or visited <= 0 or local_latency < 0:
                    raise ValueError("E3 row has invalid work or latency")
                recall[pair, column] = reproduced_recall
                recovered_count[pair, column] = recovered
                distance[pair, column] = work
                nodes[pair, column] = visited
                latency[pair, column] = local_latency
                capped_rank[pair, column] = float(
                    np.mean([rank if rank else 11 for rank in ranks])
                )
                seen_e3[pair, column] = True
                e3_rows += 1

        e4_rows = 0
        with e4_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                query_id = int(row["query_id"])
                query_offset = query_id - measurement_start
                row_shard = int(row["shard_id"])
                if row_shard != shard_id or not (0 <= query_offset < len(truth)):
                    raise ValueError("invalid E4 query or shard ID")
                if e4_seen[query_offset, shard_id]:
                    raise ValueError("duplicate E4 query-shard row")
                if int(row["ef_search"]) != common_ef:
                    raise ValueError(f"E4 row does not use common EF={common_ef}")
                results = [int(value) for value in row["result_ids"]]
                validate_result_ids(results, assignments, shard_id)
                truth_set = {int(value) for value in truth[query_offset]}
                reproduced = [point_id for point_id in results if point_id in truth_set]
                if reproduced != [int(value) for value in row["recovered_ground_truth_ids"]]:
                    raise ValueError("E4 recovered ground-truth IDs do not reproduce in rank order")
                work = int(row["distance_computations"])
                visited = int(row["graph_nodes_visited"])
                local_latency = int(row["local_latency_us"])
                if work <= 0 or visited <= 0 or local_latency < 0:
                    raise ValueError("E4 row has invalid work or latency")
                e4_recovered[query_offset, shard_id] = len(reproduced)
                e4_distance[query_offset, shard_id] = work
                e4_seen[query_offset, shard_id] = True
                e4_rows += 1

        if e3_rows != int(shard_manifest["e3_row_count"]):
            raise ValueError(f"E3 row count mismatch for shard {shard_id}")
        if e4_rows != int(shard_manifest["e4_row_count"]) or e4_rows != len(truth):
            raise ValueError(f"E4 row count mismatch for shard {shard_id}")
        shard_audit_rows.append(
            {
                "partition_method": method,
                "partition_seed": partition_seed,
                "logical_shards": logical_shards,
                "shard_id": shard_id,
                "local_point_count": local_count,
                "entry_points_num": recorded_entry["entry_points_num"],
                "construction_seconds": float(shard_manifest["construction_seconds"]),
                "total_shard_seconds": float(shard_manifest["total_shard_seconds"]),
                "e3_row_count": e3_rows,
                "e4_row_count": e4_rows,
                "graph_files_sha256_json": json.dumps(graph_hashes, sort_keys=True),
                "manifest_path": str(shard_manifest_path),
                "manifest_sha256": c23_e1.sha256_path(shard_manifest_path),
            }
        )

    if total_local_points != len(assignments):
        raise ValueError("local shard point counts do not cover the full partition")
    if not np.all(seen_e3):
        raise ValueError(f"missing {np.size(seen_e3) - np.count_nonzero(seen_e3)} E3 rows")
    if not np.all(e4_seen):
        raise ValueError(f"missing {np.size(e4_seen) - np.count_nonzero(e4_seen)} E4 rows")

    e3_summary: list[dict[str, Any]] = []
    for column, ef in enumerate(ef_values):
        recalls = recall[:, column]
        e3_summary.append(
            {
                "dataset": dataset,
                "partition_method": method,
                "partition_seed": partition_seed,
                "logical_shards": logical_shards,
                "query_shard_pair_count": pair_count,
                "ef_search": int(ef),
                **summarize(recalls, "local_target_recall"),
                "local_target_recall_target_weighted": float(
                    np.sum(recovered_count[:, column]) / np.sum(target_count)
                ),
                **summarize(distance[:, column], "distance_computations"),
                **summarize(nodes[:, column], "graph_nodes_visited"),
                **summarize(latency[:, column], "local_latency_us"),
                **summarize(capped_rank[:, column], "capped_local_target_rank"),
            }
        )

    fixed_recall: list[dict[str, Any]] = []
    for threshold in LOCAL_RECALL_THRESHOLDS:
        meets = recall >= threshold
        success = np.any(meets, axis=1)
        first = np.argmax(meets, axis=1)
        rows = np.flatnonzero(success)
        columns = first[rows]
        required_ef = ef_values[columns]
        required_distance = distance[rows, columns]
        required_nodes = nodes[rows, columns]
        row = {
            "dataset": dataset,
            "partition_method": method,
            "partition_seed": partition_seed,
            "logical_shards": logical_shards,
            "local_recall_threshold": threshold,
            "query_shard_pair_count": pair_count,
            "success_count": int(np.count_nonzero(success)),
            "success_fraction": float(np.mean(success)),
        }
        if len(rows):
            row.update(summarize(required_ef, "required_ef_search"))
            row.update(summarize(required_distance, "required_distance_computations"))
            row.update(summarize(required_nodes, "required_graph_nodes_visited"))
        else:
            for prefix in (
                "required_ef_search",
                "required_distance_computations",
                "required_graph_nodes_visited",
            ):
                row.update({f"{prefix}_{suffix}": "" for suffix in ("mean", "median", "p95", "std")})
        fixed_recall.append(row)

    e4_per_query: list[dict[str, Any]] = []
    p_exact_values = []
    p_hnsw_values = []
    delta_values = []
    w90_values = []
    reached_values = []
    common_ef_local_recall = np.asarray(
        [
            e4_recovered[query_offset, shard_id]
            / len(truth_groups[query_offset][shard_id])
            for query_offset, shard_id in pair_keys
        ],
        dtype=np.float64,
    )
    common_ef_local_distance = np.asarray(
        [e4_distance[query_offset, shard_id] for query_offset, shard_id in pair_keys],
        dtype=np.uint64,
    )
    for query_offset, groups in enumerate(truth_groups):
        exact_contributions = np.zeros(logical_shards, dtype=np.int16)
        for shard_id, point_ids in groups.items():
            exact_contributions[shard_id] = len(point_ids)
        p_exact = minimum_shards_for_nine(exact_contributions)
        contributions = e4_recovered[query_offset]
        order = sorted(
            range(logical_shards),
            key=lambda shard_id: (
                -int(contributions[shard_id]),
                int(e4_distance[query_offset, shard_id]),
                shard_id,
            ),
        )
        cumulative = 0
        work = 0
        selected = 0
        for shard_id in order:
            selected += 1
            cumulative += int(contributions[shard_id])
            work += int(e4_distance[query_offset, shard_id])
            if cumulative >= 9:
                break
        reached = cumulative >= 9
        p_hnsw = selected if reached else logical_shards + 1
        delta = p_hnsw - p_exact
        query_id = measurement_start + query_offset
        e2 = e2_rows[(method, logical_shards, query_id)]
        relevant_shards = sorted(groups)
        relevant_local_recalls = [
            e4_recovered[query_offset, shard_id] / len(groups[shard_id])
            for shard_id in relevant_shards
        ]
        relevant_work = [
            int(e4_distance[query_offset, shard_id]) for shard_id in relevant_shards
        ]
        e4_per_query.append(
            {
                "query_id": query_id,
                "partition_method": method,
                "partition_seed": partition_seed,
                "logical_shards": logical_shards,
                "global_trace_length": e2["global_trace_length"],
                "path_shard_count": e2["path_shard_count"],
                "path_transition_count": e2["path_transition_count"],
                "ground_truth_shards": json.dumps(
                    [int(assignments[int(point_id)]) for point_id in truth[query_offset]]
                ),
                "P_exact": p_exact,
                "P_HNSW": p_hnsw,
                "Delta_P": delta,
                "W90": work,
                "hnsw_reached_90": reached,
                "common_ef_mean_local_target_recall": float(
                    np.mean(relevant_local_recalls)
                ),
                "common_ef_target_weighted_recall": float(
                    np.sum(e4_recovered[query_offset]) / 10.0
                ),
                "common_ef_mean_relevant_distance_computations": float(
                    np.mean(relevant_work)
                ),
            }
        )
        p_exact_values.append(p_exact)
        p_hnsw_values.append(p_hnsw)
        delta_values.append(delta)
        w90_values.append(work)
        reached_values.append(reached)
    e4_summary = {
        "dataset": dataset,
        "partition_method": method,
        "partition_seed": partition_seed,
        "logical_shards": logical_shards,
        "measurement_query_count": len(truth),
        **summarize(np.asarray(p_exact_values), "P_exact"),
        **summarize(np.asarray(p_hnsw_values), "P_HNSW"),
        **summarize(np.asarray(delta_values), "Delta_P"),
        **summarize(np.asarray(w90_values), "W90"),
        "hnsw_reached_90_fraction": float(np.mean(reached_values)),
        "common_ef_search": common_ef,
        **summarize(common_ef_local_recall, "common_ef_local_target_recall"),
        **summarize(common_ef_local_distance, "common_ef_local_distance_computations"),
    }
    participation_rows = [
        {
            "dataset": dataset,
            "partition_method": method,
            "partition_seed": partition_seed,
            "logical_shards": logical_shards,
            "shard_id": shard_id,
            "vector_count": int(np.count_nonzero(assignments == shard_id)),
            "query_participation_count": int(participation[shard_id]),
            "query_participation_fraction": float(participation[shard_id] / len(truth)),
        }
        for shard_id in range(logical_shards)
    ]
    config_audit = {
        "partition_method": method,
        "partition_seed": partition_seed,
        "logical_shards": logical_shards,
        "run_dir": str(run_dir),
        "root_manifest": str(root_manifest_path),
        "root_manifest_sha256": c23_e1.sha256_path(root_manifest_path),
        "query_shard_pair_count": pair_count,
        "checks": {
            "root_manifest_contract": "PASS",
            "partition_checksum_and_coverage": "PASS",
            "complete_shard_manifests": "PASS",
            "production_entry_point_parameters": "PASS",
            "graph_files_present_and_hashed": "PASS",
            "e3_checksums_schema_coverage_recall_ranks": "PASS",
            "e4_checksums_schema_coverage_recovery": "PASS",
        },
    }
    return (
        e3_summary,
        fixed_recall,
        e4_summary,
        e4_per_query,
        participation_rows,
        shard_audit_rows + [config_audit],
    )


def plot_method_curves(
    rows: list[dict[str, Any]],
    *,
    x_key: str,
    y_key: str,
    xlabel: str,
    ylabel: str,
    path: Path,
) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite figure: {path}")
    fig, axes = plt.subplots(2, 3, figsize=(11.0, 6.6), sharex=False, sharey=True)
    for axis, logical_shards in zip(axes.flat, c23_protocol.LOGICAL_SHARDS):
        for method in c23_e1.METHOD_ORDER:
            selected = sorted(
                (
                    row
                    for row in rows
                    if row["partition_method"] == method
                    and int(row["logical_shards"]) == logical_shards
                ),
                key=lambda row: float(row[x_key]),
            )
            axis.plot(
                [float(row[x_key]) for row in selected],
                [float(row[y_key]) for row in selected],
                label=c23_e1.METHOD_LABELS[method],
                linewidth=1.4,
                markersize=4,
                **c23_e1.METHOD_STYLES[method],
            )
        axis.set_title(f"M={logical_shards}")
        axis.set_xlabel(xlabel)
        axis.grid(True, alpha=0.2)
    axes[0, 0].set_ylabel(ylabel)
    axes[1, 0].set_ylabel(ylabel)
    axes[1, 2].axis("off")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right", bbox_to_anchor=(0.98, 0.08), frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_e4_lines(rows: list[dict[str, Any]], y_key: str, ylabel: str, path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite figure: {path}")
    fig, axis = plt.subplots(figsize=(6.6, 4.2))
    for method in c23_e1.METHOD_ORDER:
        selected = sorted(
            (row for row in rows if row["partition_method"] == method),
            key=lambda row: int(row["logical_shards"]),
        )
        axis.plot(
            [int(row["logical_shards"]) for row in selected],
            [float(row[y_key]) for row in selected],
            label=c23_e1.METHOD_LABELS[method],
            linewidth=1.8,
            markersize=5,
            **c23_e1.METHOD_STYLES[method],
        )
    axis.set_xlabel("Logical shard count")
    axis.set_ylabel(ylabel)
    axis.set_xticks(c23_protocol.LOGICAL_SHARDS)
    axis.grid(True, alpha=0.25)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_figures(
    e3_rows: list[dict[str, Any]],
    fixed_rows: list[dict[str, Any]],
    e4_rows: list[dict[str, Any]],
    e4_per_query: list[dict[str, Any]],
    figure_dir: str | Path,
) -> dict[str, str]:
    target = Path(figure_dir).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    paths = {
        "local_recall_vs_ef": target / "fig_c23_local_recall_vs_ef.pdf",
        "local_recall_vs_distance": target / "fig_c23_local_recall_vs_distance_computations.pdf",
        "local_work_fixed_recall": target / "fig_c23_local_work_at_fixed_recall.pdf",
        "exact_oracle_fanout": target / "fig_c23_exact_oracle_fanout.pdf",
        "hnsw_oracle_fanout": target / "fig_c23_hnsw_oracle_fanout.pdf",
        "fanout_cdf": target / "fig_c23_fanout_cdf.pdf",
        "fanout_decomposition": target / "fig_c23_fanout_decomposition.pdf",
        "work_to_90": target / "fig_c23_work_to_90_recall.pdf",
    }
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite figures: {existing}")
    plot_method_curves(
        e3_rows,
        x_key="ef_search",
        y_key="local_target_recall_mean",
        xlabel="efSearch",
        ylabel="Mean local target recall",
        path=paths["local_recall_vs_ef"],
    )
    plot_method_curves(
        e3_rows,
        x_key="distance_computations_mean",
        y_key="local_target_recall_mean",
        xlabel="Mean distance computations/query-shard",
        ylabel="Mean local target recall",
        path=paths["local_recall_vs_distance"],
    )
    plot_method_curves(
        fixed_rows,
        x_key="local_recall_threshold",
        y_key="required_distance_computations_mean",
        xlabel="Local target-recall threshold",
        ylabel="Mean required distance computations",
        path=paths["local_work_fixed_recall"],
    )
    plot_e4_lines(e4_rows, "P_exact_mean", "Mean exact-oracle fan-out", paths["exact_oracle_fanout"])
    plot_e4_lines(e4_rows, "P_HNSW_mean", "Mean HNSW-oracle fan-out", paths["hnsw_oracle_fanout"])
    plot_e4_lines(e4_rows, "Delta_P_mean", "Mean additional HNSW fan-out", paths["fanout_decomposition"])
    plot_e4_lines(e4_rows, "W90_mean", "Mean distance computations to 90% recall", paths["work_to_90"])

    fig, axes = plt.subplots(2, 3, figsize=(11.0, 6.6), sharey=True)
    for axis, logical_shards in zip(axes.flat, c23_protocol.LOGICAL_SHARDS):
        for method in c23_e1.METHOD_ORDER:
            values = np.sort(
                np.asarray(
                    [
                        int(row["P_HNSW"])
                        for row in e4_per_query
                        if row["partition_method"] == method
                        and int(row["logical_shards"]) == logical_shards
                    ]
                )
            )
            y = np.arange(1, len(values) + 1, dtype=np.float64) / len(values)
            axis.step(
                values,
                y,
                where="post",
                label=c23_e1.METHOD_LABELS[method],
                linewidth=1.4,
                linestyle=c23_e1.METHOD_STYLES[method]["linestyle"],
            )
        axis.set_title(f"M={logical_shards}")
        axis.set_xlabel("HNSW oracle fan-out")
        axis.grid(True, alpha=0.2)
    axes[0, 0].set_ylabel("CDF")
    axes[1, 0].set_ylabel("CDF")
    axes[1, 2].axis("off")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right", bbox_to_anchor=(0.98, 0.08), frameon=False)
    fig.tight_layout()
    fig.savefig(paths["fanout_cdf"])
    plt.close(fig)
    return {name: str(path) for name, path in paths.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(c23_protocol.DATASETS), required=True)
    parser.add_argument("--input-manifest", required=True)
    parser.add_argument("--run-registry", required=True)
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--e2-per-query", required=True)
    parser.add_argument("--measurement-start", type=int, default=1000)
    parser.add_argument("--common-ef", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--figure-dir", required=True)
    parser.add_argument("--audit-output", required=True)
    args = parser.parse_args()

    input_manifest_path = Path(args.input_manifest).expanduser().resolve()
    input_manifest = json.loads(input_manifest_path.read_text(encoding="utf-8"))
    spec = c23_protocol.DATASETS[args.dataset]
    if str(input_manifest["dataset"]) != args.dataset:
        raise ValueError("local input-manifest dataset mismatch")
    registry_path = Path(args.run_registry).expanduser().resolve()
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    if int(input_manifest["configuration_count"]) != 20 or int(registry["configuration_count"]) != 20:
        raise ValueError("local analysis requires the complete 20-configuration SIFT matrix")
    configurations = {
        str(row["configuration_id"]): row for row in input_manifest["configurations"]
    }
    runs = {str(row["configuration_id"]): row for row in registry["runs"]}
    if set(configurations) != set(runs):
        raise ValueError("input and run-registry configuration coverage differs")
    truth_values = np.memmap(args.ground_truth, dtype="<u4", mode="r")
    if len(truth_values) != spec.expected_query_rows * 10:
        raise ValueError("ground-truth raw shape mismatch")
    truth = truth_values.reshape(spec.expected_query_rows, 10)[args.measurement_start :]
    e2 = load_e2_per_query(args.e2_per_query)
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output}")
    output.mkdir(parents=True, exist_ok=False)

    all_e3: list[dict[str, Any]] = []
    all_fixed: list[dict[str, Any]] = []
    all_e4: list[dict[str, Any]] = []
    all_e4_query: list[dict[str, Any]] = []
    all_participation: list[dict[str, Any]] = []
    shard_audits: list[dict[str, Any]] = []
    config_audits: list[dict[str, Any]] = []
    for index, configuration_id in enumerate(sorted(configurations), 1):
        configuration = configurations[configuration_id]
        assignments = np.memmap(
            configuration["raw_assignment_path"],
            dtype="<u4",
            mode="r",
        )
        result = analyze_configuration(
            args.dataset,
            spec,
            args.measurement_start,
            args.common_ef,
            configuration,
            runs[configuration_id],
            assignments,
            truth,
            e2,
        )
        e3_rows, fixed_rows, e4_row, e4_query_rows, participation_rows, audits = result
        all_e3.extend(e3_rows)
        all_fixed.extend(fixed_rows)
        all_e4.append(e4_row)
        all_e4_query.extend(e4_query_rows)
        all_participation.extend(participation_rows)
        for row in audits:
            if "shard_id" in row:
                shard_audits.append(row)
            else:
                config_audits.append(row)
        print(f"[{index}/20] audited and aggregated {configuration_id}", flush=True)

    paths = {
        "e3_summary": output / "e3_summary.csv",
        "e3_fixed_recall": output / "e3_fixed_recall.csv",
        "e4_summary": output / "e4_summary.csv",
        "e4_per_query": output / "e4_per_query.csv",
        "query_participation": output / "query_participation_by_shard.csv",
        "shard_audit": output / "local_shard_audit.csv",
    }
    write_csv_new(paths["e3_summary"], all_e3)
    write_csv_new(paths["e3_fixed_recall"], all_fixed)
    write_csv_new(paths["e4_summary"], all_e4)
    write_csv_new(paths["e4_per_query"], all_e4_query)
    write_csv_new(paths["query_participation"], all_participation)
    write_csv_new(paths["shard_audit"], shard_audits)
    figures = save_figures(all_e3, all_fixed, all_e4, all_e4_query, args.figure_dir)
    manifest = {
        "protocol_version": c23_protocol.PROTOCOL_VERSION,
        "timestamp": c23_e1.utc_timestamp(),
        "git_commit": c23_protocol.git_commit(REPO_ROOT),
        "experiment": f"C23 {args.dataset} E3 local navigability and E4 oracle fan-out",
        "dataset": args.dataset,
        "dataset_sha256": spec.sha256,
        "configuration_count": len(config_audits),
        "local_shard_count": len(shard_audits),
        "measurement_query_count": len(truth),
        "input_manifest": str(input_manifest_path),
        "input_manifest_sha256": c23_e1.sha256_path(input_manifest_path),
        "run_registry": str(registry_path),
        "run_registry_sha256": c23_e1.sha256_path(registry_path),
        "checks": {
            "complete_20_configuration_matrix": "PASS",
            "all_local_shard_manifests_and_graphs": "PASS",
            "all_e3_rows_checksums_coverage_recall_ranks": "PASS",
            "all_e4_rows_checksums_coverage_recovery": "PASS",
            f"common_e4_ef_search_{args.common_ef}": "PASS",
            "exact_fanout_reproduced_from_ground_truth_membership": "PASS",
        },
        "configuration_audits": config_audits,
        "files": {
            name: {"path": str(path), "sha256": c23_e1.sha256_path(path)}
            for name, path in paths.items()
        }
        | {
            "figures": {
                name: {"path": path, "sha256": c23_e1.sha256_path(path)}
                for name, path in figures.items()
            }
        },
    }
    manifest_path = output / "manifest.json"
    c23_e1.write_json_new(manifest_path, manifest)
    c23_e1.write_json_new(
        args.audit_output,
        {
            **manifest,
            "manifest": str(manifest_path),
            "manifest_sha256": c23_e1.sha256_path(manifest_path),
            "e3_summary": all_e3,
            "e3_fixed_recall": all_fixed,
            "e4_summary": all_e4,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
