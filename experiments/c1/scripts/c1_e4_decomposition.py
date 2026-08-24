#!/usr/bin/env python3

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import os
import shutil
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from c1_protocol import repository_commit, sha256_path, utc_timestamp


METHODS = ("random", "kmeans")
LOGICAL_SHARD_COUNTS = (1, 2, 4, 8, 16, 32)
PHYSICAL_MACHINE_COUNTS = (1, 2, 4)
TARGET_RECALL = 0.90
DATASET_LABELS = {
    "sift1m": "SIFT1M",
    "glove-200-angular": "GloVe-200-angular",
}
PER_QUERY_FIELDS = (
    "query_id",
    "dataset",
    "partition_method",
    "logical_shard_count",
    "actual_fanout",
    "oracle_minimum_fanout",
    "selected_ef_search",
    "observed_distance_computations",
    "observed_nodes_visited",
    "observed_worker_cpu_time_us",
    "source_e3_per_search",
)
SUMMARY_FIELDS = (
    "dataset",
    "partition_method",
    "logical_shard_count",
    "query_count",
    "actual_fanout_mean",
    "oracle_fanout_mean",
    "selected_ef_search",
    "observed_distance_mean",
    "observed_distance_p95",
    "normalized_observed_distance_to_m1",
    "observed_nodes_mean",
    "observed_nodes_p95",
    "observed_worker_cpu_time_us_mean",
    "observed_worker_cpu_time_us_p95",
    "mean_local_distance_per_searched_shard",
    "modeled_distance_mean",
    "normalized_modeled_distance_to_m1",
    "model_relative_error",
    "ideal_query_work_reduction",
    "physical_e1_normalized_qps",
    "physical_work_bound_projection",
    "m_gt_4_projection",
    "projection_status",
    "source_e3_per_search",
    "source_e3_per_search_sha256",
    "status",
)
MODEL_ACCURACY_FIELDS = (
    "dataset",
    "partition_method",
    "configuration_count",
    "pearson_model_vs_observed",
    "spearman_model_vs_observed",
    "mean_relative_error",
    "maximum_relative_error",
    "physical_point_count",
    "pearson_physical_projection_vs_e1",
    "spearman_physical_projection_vs_e1",
    "physical_projection_mean_relative_error",
    "physical_projection_trend_agreement",
    "m_gt_4_projection_status",
    "c1_d_aggregate_cost_status",
    "physical_attribution_status",
)


def atomic_write_text(path: str | Path, text: str) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, destination)
    return destination


def write_json_atomic(path: str | Path, payload: Any) -> Path:
    return atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def register_manifest_record_once(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path).expanduser().resolve()
    existing = []
    if destination.is_file():
        existing = [
            json.loads(line)
            for line in destination.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    matches = [
        row
        for row in existing
        if row.get("experiment_id") == payload.get("experiment_id")
    ]
    if matches:
        if len(matches) == 1 and matches[0] == payload:
            return
        raise ValueError(
            f"manifest already contains a different record for {payload.get('experiment_id')}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_csv_atomic(
    path: str | Path, fieldnames: Sequence[str], rows: Iterable[dict[str, Any]]
) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fieldnames})
    os.replace(temporary, destination)
    return destination


def load_json(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {source}")
    return payload


def load_external_projection_gates(
    paths: Sequence[str | Path], *, current_dataset: str
) -> tuple[list[dict[str, Any]], bool]:
    gates: list[dict[str, Any]] = []
    seen_datasets: set[str] = set()
    for path in paths:
        source = Path(path).expanduser().resolve()
        payload = load_json(source)
        if payload.get("record_type") != "e4_aggregate_work_decomposition":
            raise ValueError(f"external projection gate is not an E4 record: {source}")
        dataset = str(payload.get("dataset") or "")
        if not dataset:
            raise ValueError(f"external projection gate has no dataset: {source}")
        if dataset == current_dataset:
            raise ValueError(
                f"external projection gate must use another dataset: {source}"
            )
        if dataset in seen_datasets:
            raise ValueError(f"duplicate external projection-gate dataset: {dataset}")
        status = str(payload.get("m_gt_4_projection_status") or "")
        if status not in {
            "AVAILABLE",
            "WITHHELD_PHYSICAL_SANITY_CHECK_FAILED",
        }:
            raise ValueError(
                f"external projection gate has invalid M>4 status: {source}"
            )
        experiment_id = str(payload.get("experiment_id") or "")
        if not experiment_id:
            raise ValueError(f"external projection gate has no experiment ID: {source}")
        seen_datasets.add(dataset)
        gates.append(
            {
                "dataset": dataset,
                "experiment_id": experiment_id,
                "m_gt_4_projection_status": status,
                "source_record": str(source),
                "source_record_sha256": sha256_path(source),
            }
        )
    return gates, all(
        gate["m_gt_4_projection_status"] == "AVAILABLE" for gate in gates
    )


def percentile(values: Sequence[int | float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def average_ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    position = 0
    while position < len(order):
        stop = position + 1
        while stop < len(order) and values[order[stop]] == values[order[position]]:
            stop += 1
        average = (position + 1 + stop) / 2.0
        for offset in range(position, stop):
            ranks[order[offset]] = average
        position = stop
    return ranks


def pearson(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        raise ValueError("correlation requires equal sequences of length at least two")
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    left_norm = math.sqrt(sum((x - left_mean) ** 2 for x in left))
    right_norm = math.sqrt(sum((y - right_mean) ** 2 for y in right))
    if left_norm == 0.0 or right_norm == 0.0:
        raise ValueError("correlation is undefined for a constant sequence")
    return numerator / (left_norm * right_norm)


def spearman(left: Sequence[float], right: Sequence[float]) -> float:
    return pearson(average_ranks(left), average_ranks(right))


def source_component(value: str, label: str) -> str:
    if not value or Path(value).name != value:
        raise ValueError(f"{label} source stem must be one filename component")
    return value


def load_e2_rows(
    path: Path,
    dataset: str,
    *,
    e2_source_stem: str,
    e3_source_stem: str,
) -> dict[tuple[str, int, int], dict[str, Any]]:
    expected_name = f"{e2_source_stem}-{dataset}-fanout.csv"
    if path.name != expected_name:
        raise ValueError(f"E2 source filename mismatch: {path.name} != {expected_name}")
    required = {
        "query_id",
        "dataset",
        "partition_method",
        "logical_shard_count",
        "oracle_minimum_fanout",
        "actual_selected_fanout",
        "actual_selected_shard_ids",
        "selected_ef_search",
        "source_e3_per_search",
    }
    rows: dict[tuple[str, int, int], dict[str, Any]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"E2 source missing fields {sorted(missing)}: {path}")
        for row in reader:
            if row["dataset"] != dataset:
                raise ValueError(f"E2 dataset mismatch: {path}")
            method = row["partition_method"]
            logical_shards = int(row["logical_shard_count"])
            query_id = int(row["query_id"])
            key = (method, logical_shards, query_id)
            if key in rows:
                raise ValueError(f"duplicate E2 query row: {key}")
            selected_shards = ast.literal_eval(row["actual_selected_shard_ids"])
            if not isinstance(selected_shards, list):
                raise ValueError(f"invalid E2 selected shard list: {key}")
            actual_fanout = int(row["actual_selected_fanout"])
            if len(selected_shards) != actual_fanout or len(set(selected_shards)) != actual_fanout:
                raise ValueError(f"invalid E2 fan-out accounting: {key}")
            source_e3 = Path(row["source_e3_per_search"]).resolve()
            expected_e3_name = (
                f"{e3_source_stem}-{dataset}-{method}-m{logical_shards}.csv"
            )
            if source_e3.name != expected_e3_name:
                raise ValueError(
                    f"E3 source filename mismatch: {source_e3.name} != {expected_e3_name}"
                )
            rows[key] = {
                "actual_fanout": actual_fanout,
                "oracle_fanout": int(row["oracle_minimum_fanout"]),
                "selected_shards": list(map(int, selected_shards)),
                "ef_search": int(row["selected_ef_search"]),
                "source_e3": str(source_e3),
            }
    expected = len(METHODS) * len(LOGICAL_SHARD_COUNTS) * 9_000
    if len(rows) != expected:
        raise ValueError(f"unexpected E2 row count: {len(rows)} != {expected}")
    return rows


def load_local_work(path: Path, dataset: str) -> dict[tuple[str, int], dict[str, Any]]:
    rows: dict[tuple[str, int], dict[str, Any]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["dataset"] != dataset:
                continue
            key = (row["partition_method"], int(row["logical_shard_count"]))
            if key in rows:
                raise ValueError(f"duplicate local-work row: {key}")
            rows[key] = row
    expected = {(method, m) for method in METHODS for m in LOGICAL_SHARD_COUNTS}
    if set(rows) != expected:
        raise ValueError(f"local-work configuration mismatch: {set(rows) ^ expected}")
    return rows


def load_local_work_summaries(
    runs_root: Path,
    dataset: str,
    *,
    e3_source_stem: str,
) -> tuple[dict[tuple[str, int], dict[str, Any]], dict[str, str]]:
    rows: dict[tuple[str, int], dict[str, Any]] = {}
    hashes: dict[str, str] = {}
    for method in METHODS:
        for logical_shards in LOGICAL_SHARD_COUNTS:
            path = (
                runs_root
                / f"{e3_source_stem}-{dataset}-{method}-m{logical_shards}-summary.json"
            ).resolve()
            payload = load_json(path)
            key = (method, logical_shards)
            if payload.get("dataset") != dataset:
                raise ValueError(f"E3 summary dataset mismatch: {path}")
            if payload.get("partition_method") != method:
                raise ValueError(f"E3 summary method mismatch: {path}")
            if int(payload.get("logical_shards") or 0) != logical_shards:
                raise ValueError(f"E3 summary shard-count mismatch: {path}")
            if not str(payload.get("result_status") or "").startswith("VALID_E3"):
                raise ValueError(f"E3 summary is not valid: {path}")
            if int(payload.get("query_count") or 0) != 9_000:
                raise ValueError(f"E3 summary query-count mismatch: {path}")
            if payload.get("benchmark_cpu_affinity") != list(range(20, 32)):
                raise ValueError(f"E3 summary benchmark affinity mismatch: {path}")
            if payload.get("deterministic_graph_construction") is not True:
                raise ValueError(f"E3 summary is not deterministic: {path}")
            if int(payload.get("hnsw_max_indexing_threads") or 0) != 1:
                raise ValueError(f"E3 summary did not use one indexing thread: {path}")
            if int(payload.get("max_optimization_threads") or 0) != 1:
                raise ValueError(f"E3 summary did not use one optimizer thread: {path}")
            rows[key] = payload
            hashes[str(path)] = sha256_path(path)
    return rows, hashes


def kmeans_partition_anomalies(partition_directory: Path, dataset: str) -> list[str]:
    dataset_token = dataset.replace("-", "_")
    anomalies: list[str] = []
    for logical_shards in LOGICAL_SHARD_COUNTS:
        artifact = (partition_directory / f"kmeans-m{logical_shards}.npz").resolve()
        with np.load(artifact, allow_pickle=False) as data:
            metadata = json.loads(str(np.asarray(data["metadata_json"]).item()))
        metadata_dataset = metadata.get("dataset") or {}
        if not isinstance(metadata_dataset, dict) or metadata_dataset.get("name") != dataset:
            raise ValueError(f"partition dataset mismatch: {artifact}")
        if int(metadata.get("logical_shards") or 0) != logical_shards:
            raise ValueError(f"partition shard-count mismatch: {artifact}")
        kmeans = metadata.get("kmeans") or {}
        history = list(kmeans.get("history") or [])
        maximum_iterations = int(kmeans.get("maximum_iterations") or 0)
        tolerance = float(kmeans.get("tolerance") or 0.0)
        final_movement = (
            float(history[-1]["max_centroid_movement"]) if history else None
        )
        converged = bool(
            history
            and final_movement is not None
            and final_movement <= tolerance
            and len(history) <= maximum_iterations
        )
        if not converged and len(history) == maximum_iterations:
            anomalies.append(
                f"{dataset_token}_kmeans_m{logical_shards}_reached_"
                f"{maximum_iterations}_iteration_cap_without_convergence"
            )
    return anomalies


def load_physical_e1(
    path: Path, dataset: str, *, source_stem: str | None = None
) -> dict[tuple[str, int], dict[str, Any]]:
    rows: dict[tuple[str, int], dict[str, Any]] = {}
    if path.suffix.lower() == ".json":
        payload = load_json(path)
        if (
            payload.get("record_type") != "e1_physical_scaleout_analysis"
            or payload.get("dataset") != dataset
            or payload.get("status") != "VALID_E1"
        ):
            raise ValueError(f"invalid physical E1 summary: {path}")
        source_rows = payload.get("configurations")
        if not isinstance(source_rows, list):
            raise ValueError(f"physical E1 summary has no configurations: {path}")
    else:
        with path.open(newline="", encoding="utf-8") as handle:
            source_rows = list(csv.DictReader(handle))
    for row in source_rows:
        if row["dataset"] != dataset:
            continue
        key = (row["partition_method"], int(row["physical_machine_count"]))
        if key in rows:
            raise ValueError(f"duplicate physical E1 row: {key}")
        if row["status"] != "VALID_E1" or row["bottleneck_flags"]:
            raise ValueError(f"invalid physical E1 row: {key}")
        if source_stem and not row["experiment_id"].startswith(
            f"{source_stem}-{dataset}-"
        ):
            raise ValueError(f"physical E1 source-stem mismatch: {key}")
        rows[key] = row
    expected = {(method, m) for method in METHODS for m in PHYSICAL_MACHINE_COUNTS}
    if set(rows) != expected:
        raise ValueError(f"physical E1 configuration mismatch: {set(rows) ^ expected}")
    return rows


def aggregate_configuration(
    *,
    dataset: str,
    method: str,
    logical_shards: int,
    e2_rows: dict[tuple[str, int, int], dict[str, Any]],
    local_work: dict[str, Any],
    per_query_rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[int], list[int], list[int]]:
    keys = [
        (method, logical_shards, query_id)
        for query_id in range(1_000, 10_000)
    ]
    if any(key not in e2_rows for key in keys):
        raise ValueError(f"E2 query coverage mismatch for {method} M={logical_shards}")
    source_paths = {e2_rows[key]["source_e3"] for key in keys}
    if len(source_paths) != 1:
        raise ValueError(f"E2 mixes E3 sources for {method} M={logical_shards}")
    source = Path(source_paths.pop()).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    grouped: dict[int, dict[str, Any]] = defaultdict(
        lambda: {"distance": 0, "nodes": 0, "cpu": 0, "shards": [], "ef": set()}
    )
    with source.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row["dataset"] != dataset or row["partition_method"] != method:
                raise ValueError(f"E3 dataset/method mismatch: {source}")
            if int(row["logical_shards"]) != logical_shards:
                raise ValueError(f"E3 shard-count mismatch: {source}")
            query_id = int(row["query_id"])
            if not 1_000 <= query_id < 10_000:
                raise ValueError(f"E3 query outside measurement range: {query_id}")
            distance = int(row["distance_computations"])
            nodes = int(row["nodes_visited"])
            cpu = int(row["worker_cpu_time_us"])
            if distance <= 0 or nodes <= 0 or cpu <= 0:
                raise ValueError(f"non-positive E3 counter for query {query_id}: {source}")
            record = grouped[query_id]
            record["distance"] += distance
            record["nodes"] += nodes
            record["cpu"] += cpu
            record["shards"].append(int(row["shard_id"]))
            record["ef"].add(int(row["ef_search"]))
    if set(grouped) != set(range(1_000, 10_000)):
        raise ValueError(f"E3 query coverage mismatch: {source}")
    distances: list[int] = []
    nodes_visited: list[int] = []
    cpu_times: list[int] = []
    fanouts: list[int] = []
    oracle_fanouts: list[int] = []
    ef_values: set[int] = set()
    for query_id in range(1_000, 10_000):
        e2 = e2_rows[(method, logical_shards, query_id)]
        observed = grouped[query_id]
        shards = observed["shards"]
        if len(shards) != len(set(shards)):
            raise ValueError(f"duplicate E3 shard event: {(method, logical_shards, query_id)}")
        if sorted(shards) != sorted(e2["selected_shards"]):
            raise ValueError(f"E2/E3 selected-shard mismatch: {(method, logical_shards, query_id)}")
        if len(shards) != e2["actual_fanout"]:
            raise ValueError(f"E2/E3 fan-out mismatch: {(method, logical_shards, query_id)}")
        if observed["ef"] != {e2["ef_search"]}:
            raise ValueError(f"E2/E3 efSearch mismatch: {(method, logical_shards, query_id)}")
        distances.append(int(observed["distance"]))
        nodes_visited.append(int(observed["nodes"]))
        cpu_times.append(int(observed["cpu"]))
        fanouts.append(int(e2["actual_fanout"]))
        oracle_fanouts.append(int(e2["oracle_fanout"]))
        ef_values.add(int(e2["ef_search"]))
        per_query_rows.append(
            {
                "query_id": query_id,
                "dataset": dataset,
                "partition_method": method,
                "logical_shard_count": logical_shards,
                "actual_fanout": e2["actual_fanout"],
                "oracle_minimum_fanout": e2["oracle_fanout"],
                "selected_ef_search": e2["ef_search"],
                "observed_distance_computations": observed["distance"],
                "observed_nodes_visited": observed["nodes"],
                "observed_worker_cpu_time_us": observed["cpu"],
                "source_e3_per_search": str(source),
            }
        )
    if len(ef_values) != 1:
        raise ValueError(f"mixed E4 efSearch values for {method} M={logical_shards}")
    total_events = sum(fanouts)
    local_distance = sum(distances) / total_events
    recorded_local = float(local_work["mean_distance_computations_per_searched_shard"])
    if not math.isclose(local_distance, recorded_local, rel_tol=1e-12):
        raise ValueError(f"local-work mean mismatch for {method} M={logical_shards}")
    fanout_mean = statistics.fmean(fanouts)
    model = fanout_mean * recorded_local
    observed_mean = statistics.fmean(distances)
    if not math.isclose(model, observed_mean, rel_tol=1e-12):
        raise ValueError(f"decomposition identity mismatch for {method} M={logical_shards}")
    summary = {
        "dataset": dataset,
        "partition_method": method,
        "logical_shard_count": logical_shards,
        "query_count": len(distances),
        "actual_fanout_mean": fanout_mean,
        "oracle_fanout_mean": statistics.fmean(oracle_fanouts),
        "selected_ef_search": ef_values.pop(),
        "observed_distance_mean": observed_mean,
        "observed_distance_p95": percentile(distances, 95),
        "observed_nodes_mean": statistics.fmean(nodes_visited),
        "observed_nodes_p95": percentile(nodes_visited, 95),
        "observed_worker_cpu_time_us_mean": statistics.fmean(cpu_times),
        "observed_worker_cpu_time_us_p95": percentile(cpu_times, 95),
        "mean_local_distance_per_searched_shard": recorded_local,
        "modeled_distance_mean": model,
        "model_relative_error": abs(model - observed_mean) / observed_mean,
        "source_e3_per_search": str(source),
        "source_e3_per_search_sha256": sha256_path(source),
        "status": "VALID_E4",
    }
    return summary, distances, nodes_visited, cpu_times


def save_pdf_atomic(figure: Any, path: Path) -> Path:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    figure.savefig(
        temporary,
        format="pdf",
        bbox_inches="tight",
        metadata={
            "Title": destination.stem,
            "Creator": "Orion C1 E4 aggregate-work analyzer",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    os.replace(temporary, destination)
    return destination


def copy_atomic(source: Path, destination: Path) -> Path:
    target = destination.expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    shutil.copyfile(source, temporary)
    os.replace(temporary, target)
    return target


def generate_figures(
    figures_dir: str | Path,
    summaries: Sequence[dict[str, Any]],
    projection_checks: dict[str, dict[str, Any]],
    *,
    dataset: str,
) -> dict[str, Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory = Path(figures_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    colors = {"random": "#3366cc", "kmeans": "#d97706"}
    labels = {"random": "Random", "kmeans": "K-Means"}

    figure, axes = plt.subplots(1, 2, figsize=(11.2, 4.2), sharey=True)
    for axis, method in zip(axes, METHODS, strict=True):
        rows = sorted(
            (row for row in summaries if row["partition_method"] == method),
            key=lambda row: row["logical_shard_count"],
        )
        shards = [row["logical_shard_count"] for row in rows]
        axis.plot(
            shards,
            [row["normalized_modeled_distance_to_m1"] for row in rows],
            linestyle="--",
            marker="x",
            color="#d62728",
            label="Fan-out x local-work model",
        )
        axis.plot(
            shards,
            [row["normalized_observed_distance_to_m1"] for row in rows],
            marker="o",
            markerfacecolor="white",
            color=colors[method],
            label="Observed aggregate work",
        )
        axis.plot(
            shards,
            [1.0 / shard for shard in shards],
            linestyle=":",
            color="#555555",
            label="Ideal 1/M work",
        )
        axis.set_xscale("log", base=2)
        axis.set_yscale("log", base=2)
        axis.set_xticks(LOGICAL_SHARD_COUNTS, [str(value) for value in LOGICAL_SHARD_COUNTS])
        axis.set_xlabel("Logical shard count")
        axis.set_title(labels[method])
        axis.grid(True, which="both", alpha=0.25)
    axes[0].set_ylabel("Normalized distance computations/query")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, legend_labels, frameon=False, loc="upper center", ncols=3)
    dataset_label = DATASET_LABELS.get(dataset, dataset)
    figure.suptitle(f"{dataset_label} aggregate graph-search work", y=1.02)
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    aggregate = save_pdf_atomic(figure, directory / "fig_c1_aggregate_work.pdf")
    plt.close(figure)
    canonical_aggregate = copy_atomic(
        aggregate, directory / "c1_fig5_aggregate_work_decomposition.pdf"
    )

    figure, axis = plt.subplots(figsize=(6.2, 5.0))
    all_values = [float(row["observed_distance_mean"]) for row in summaries]
    low = min(all_values) * 0.85
    high = max(all_values) * 1.15
    axis.plot([low, high], [low, high], linestyle="--", color="#555555", label="y = x")
    for method in METHODS:
        rows = sorted(
            (row for row in summaries if row["partition_method"] == method),
            key=lambda row: row["logical_shard_count"],
        )
        axis.scatter(
            [row["observed_distance_mean"] for row in rows],
            [row["modeled_distance_mean"] for row in rows],
            color=colors[method],
            label=labels[method],
        )
        for row in rows:
            axis.annotate(
                f"M={row['logical_shard_count']}",
                (row["observed_distance_mean"], row["modeled_distance_mean"]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
            )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlim(low, high)
    axis.set_ylim(low, high)
    axis.set_xlabel("Observed aggregate distance computations/query")
    axis.set_ylabel("Fan-out x local-work model")
    axis.set_title(f"{dataset_label} modeled versus observed aggregate work")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    model = save_pdf_atomic(figure, directory / "fig_c1_model_vs_observed.pdf")
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(11.2, 4.2), sharey=False)
    for axis, method in zip(axes, METHODS, strict=True):
        check = projection_checks[method]
        projection_available = check["m_gt_4_projection_status"] == "AVAILABLE"
        projection_x = (
            LOGICAL_SHARD_COUNTS if projection_available else PHYSICAL_MACHINE_COUNTS
        )
        projection_y = (
            check["work_bound_projection"]
            if projection_available
            else check["physical_work_bound_projection"]
        )
        axis.plot(
            projection_x,
            projection_y,
            marker="s",
            linestyle="--",
            color="#8c564b",
            label="Work-bound projection",
        )
        axis.plot(
            PHYSICAL_MACHINE_COUNTS,
            check["physical_e1_normalized_qps"],
            marker="o",
            color=colors[method],
            label="Measured E1 throughput",
        )
        if projection_available:
            axis.set_xscale("log", base=2)
            axis.set_xticks(
                LOGICAL_SHARD_COUNTS, [str(value) for value in LOGICAL_SHARD_COUNTS]
            )
        else:
            axis.set_xticks(PHYSICAL_MACHINE_COUNTS)
        axis.set_xlabel("Physical workers")
        axis.set_title(labels[method])
        axis.grid(True, alpha=0.25)
        if projection_available:
            annotation = "M>4 projection available:\nall physical sanity checks passed"
        elif check["trend_agreement"]:
            annotation = (
                "Method trend agrees; M>4 withheld:\n"
                "global release gate failed"
            )
        else:
            annotation = "M>4 projection withheld:\nmethod sanity check failed"
        axis.text(
            0.03,
            0.05,
            annotation,
            transform=axis.transAxes,
            fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "#999999", "alpha": 0.9},
        )
    axes[0].set_ylabel("Normalized QPS")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    figure.suptitle(f"{dataset_label} work-bound projection sanity check", y=0.98)
    figure.legend(
        handles,
        legend_labels,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        ncols=2,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.82))
    projected = save_pdf_atomic(figure, directory / "fig_c1_projected_scaling.pdf")
    plt.close(figure)
    canonical_projected = copy_atomic(
        projected, directory / "c1_fig6_projected_logical_scaling.pdf"
    )
    return {
        "protocol_aggregate_work": aggregate,
        "canonical_aggregate_work": canonical_aggregate,
        "protocol_model_vs_observed": model,
        "protocol_projected_scaling": projected,
        "canonical_projected_scaling": canonical_projected,
    }


def execute(args: argparse.Namespace) -> dict[str, Any]:
    runs_root = Path(args.runs_root).expanduser().resolve()
    e2_per_query_path = Path(args.e2_per_query).expanduser().resolve()
    physical_e1_path = Path(args.physical_e1_summary).expanduser().resolve()
    e2_rows = load_e2_rows(
        e2_per_query_path,
        args.dataset,
        e2_source_stem=args.e2_source_stem,
        e3_source_stem=args.e3_source_stem,
    )
    local_work_hashes: dict[str, str]
    if args.local_work_summary:
        local_work_path = Path(args.local_work_summary).expanduser().resolve()
        local_work = load_local_work(local_work_path, args.dataset)
        local_work_hashes = {str(local_work_path): sha256_path(local_work_path)}
        graph_build_seed = None
    else:
        local_work, local_work_hashes = load_local_work_summaries(
            runs_root,
            args.dataset,
            e3_source_stem=args.e3_source_stem,
        )
        graph_build_seeds = {
            int(row["graph_build_seed"]) for row in local_work.values()
        }
        if len(graph_build_seeds) != 1:
            raise ValueError(
                f"E4 sources mix graph-build seeds: {sorted(graph_build_seeds)}"
            )
        graph_build_seed = graph_build_seeds.pop()
    physical_e1 = load_physical_e1(
        physical_e1_path, args.dataset, source_stem=args.e1_source_stem
    )

    per_query_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for method in METHODS:
        for logical_shards in LOGICAL_SHARD_COUNTS:
            summary, _distances, _nodes, _cpu = aggregate_configuration(
                dataset=args.dataset,
                method=method,
                logical_shards=logical_shards,
                e2_rows=e2_rows,
                local_work=local_work[(method, logical_shards)],
                per_query_rows=per_query_rows,
            )
            summaries.append(summary)
    if len(per_query_rows) != 108_000:
        raise ValueError(f"unexpected E4 per-query row count: {len(per_query_rows)}")

    projection_checks: dict[str, dict[str, Any]] = {}
    model_accuracy: list[dict[str, Any]] = []
    for method in METHODS:
        rows = sorted(
            (row for row in summaries if row["partition_method"] == method),
            key=lambda row: row["logical_shard_count"],
        )
        observed = [float(row["observed_distance_mean"]) for row in rows]
        modeled = [float(row["modeled_distance_mean"]) for row in rows]
        observed_base = observed[0]
        modeled_base = modeled[0]
        candidate_projection = [
            row["logical_shard_count"] * observed_base / float(row["observed_distance_mean"])
            for row in rows
        ]
        physical_rows = [row for row in rows if row["logical_shard_count"] <= 4]
        physical_projection = candidate_projection[: len(PHYSICAL_MACHINE_COUNTS)]
        actual_e1 = [
            float(physical_e1[(method, row["logical_shard_count"])]["normalized_qps_to_m1"])
            for row in physical_rows
        ]
        projection_pearson = pearson(physical_projection, actual_e1)
        projection_spearman = spearman(physical_projection, actual_e1)
        projection_mre = statistics.fmean(
            abs(projected - actual) / actual
            for projected, actual in zip(physical_projection, actual_e1, strict=True)
        )
        trend_agreement = projection_pearson > 0.0 and projection_spearman > 0.0
        projection_checks[method] = {
            "physical_machine_counts": list(PHYSICAL_MACHINE_COUNTS),
            "work_bound_projection": candidate_projection,
            "physical_work_bound_projection": physical_projection,
            "physical_e1_normalized_qps": actual_e1,
            "pearson": projection_pearson,
            "spearman": projection_spearman,
            "mean_relative_error": projection_mre,
            "trend_agreement": trend_agreement,
            "method_m_gt_4_projection_status": (
                "AVAILABLE" if trend_agreement else "WITHHELD_PHYSICAL_SANITY_CHECK_FAILED"
            ),
        }
        relative_errors = [
            abs(model - observation) / observation
            for model, observation in zip(modeled, observed, strict=True)
        ]
        accuracy = {
            "dataset": args.dataset,
            "partition_method": method,
            "configuration_count": len(rows),
            "pearson_model_vs_observed": pearson(modeled, observed),
            "spearman_model_vs_observed": spearman(modeled, observed),
            "mean_relative_error": statistics.fmean(relative_errors),
            "maximum_relative_error": max(relative_errors),
            "physical_point_count": len(physical_rows),
            "pearson_physical_projection_vs_e1": projection_pearson,
            "spearman_physical_projection_vs_e1": projection_spearman,
            "physical_projection_mean_relative_error": projection_mre,
            "physical_projection_trend_agreement": trend_agreement,
            "m_gt_4_projection_status": None,
            "c1_d_aggregate_cost_status": "SUPPORTS",
            "physical_attribution_status": (
                "SUPPORTS" if trend_agreement else "INSUFFICIENT"
            ),
        }
        model_accuracy.append(accuracy)
        for index, row in enumerate(rows):
            row["normalized_observed_distance_to_m1"] = (
                float(row["observed_distance_mean"]) / observed_base
            )
            row["normalized_modeled_distance_to_m1"] = (
                float(row["modeled_distance_mean"]) / modeled_base
            )
            row["ideal_query_work_reduction"] = 1.0 / int(row["logical_shard_count"])
            if int(row["logical_shard_count"]) <= 4:
                row["physical_e1_normalized_qps"] = actual_e1[index]
                row["physical_work_bound_projection"] = physical_projection[index]
            else:
                row["physical_e1_normalized_qps"] = None
                row["physical_work_bound_projection"] = None

    dataset_all_trends_agree = all(
        check["trend_agreement"] for check in projection_checks.values()
    )
    external_projection_gates, external_gates_pass = load_external_projection_gates(
        args.external_projection_gate_record,
        current_dataset=args.dataset,
    )
    all_trends_agree = dataset_all_trends_agree and external_gates_pass
    dataset_projection_status = (
        "AVAILABLE"
        if dataset_all_trends_agree
        else "WITHHELD_PHYSICAL_SANITY_CHECK_FAILED"
    )
    global_projection_status = (
        "AVAILABLE"
        if all_trends_agree
        else "WITHHELD_PHYSICAL_SANITY_CHECK_FAILED"
    )
    projection_release_gate = (
        "all_partition_methods_and_external_datasets_must_pass"
        if external_projection_gates
        else "all_partition_methods_must_pass"
    )
    dataset_physical_attribution_status = (
        "SUPPORTS" if dataset_all_trends_agree else "INSUFFICIENT"
    )
    physical_attribution_status = "SUPPORTS" if all_trends_agree else "INSUFFICIENT"
    for method in METHODS:
        check = projection_checks[method]
        check["dataset_m_gt_4_projection_status"] = dataset_projection_status
        check["m_gt_4_projection_status"] = global_projection_status
        accuracy = next(
            row for row in model_accuracy if row["partition_method"] == method
        )
        accuracy["m_gt_4_projection_status"] = global_projection_status
        accuracy["physical_attribution_status"] = physical_attribution_status
        rows = sorted(
            (row for row in summaries if row["partition_method"] == method),
            key=lambda row: row["logical_shard_count"],
        )
        for index, row in enumerate(rows):
            if int(row["logical_shard_count"]) <= 4:
                row["m_gt_4_projection"] = None
                row["projection_status"] = (
                    "PHYSICAL_SANITY_CHECK_PASSED"
                    if check["trend_agreement"]
                    else "PHYSICAL_SANITY_CHECK_FAILED"
                )
            elif all_trends_agree:
                row["m_gt_4_projection"] = check["work_bound_projection"][index]
                row["projection_status"] = "AVAILABLE_PHYSICAL_SANITY_CHECK_PASSED"
            else:
                row["m_gt_4_projection"] = None
                row["projection_status"] = "WITHHELD_PHYSICAL_SANITY_CHECK_FAILED"

    per_query_output = write_csv_atomic(
        args.per_query_output, PER_QUERY_FIELDS, per_query_rows
    )
    summary_csv_output = write_csv_atomic(
        args.summary_csv_output, SUMMARY_FIELDS, summaries
    )
    model_accuracy_output = write_csv_atomic(
        args.model_accuracy_output, MODEL_ACCURACY_FIELDS, model_accuracy
    )
    figures = generate_figures(
        args.figures_dir, summaries, projection_checks, dataset=args.dataset
    )
    timestamp = utc_timestamp()
    common_e1 = load_json(
        runs_root / f"{args.e1_source_stem}-{args.dataset}-common-m1.json"
    )
    dataset_sha256 = str(common_e1["dataset_sha256"])
    status = (
        "VALID_E4"
        if all_trends_agree
        else "VALID_E4_WITH_PHYSICAL_PROJECTION_SANITY_CHECK_FAILURE"
    )
    experiment_id = args.experiment_id or f"stage5-e4-{args.dataset}-aggregate-work"
    correction = {}
    if args.supersedes_experiment_id:
        correction = {
            "supersedes_experiment_id": args.supersedes_experiment_id,
            "correction_reason": args.correction_reason,
        }
    partition_directory = Path(common_e1["partition_artifact"]).resolve().parent
    anomalies = kmeans_partition_anomalies(partition_directory, args.dataset)
    if not dataset_all_trends_agree:
        anomalies.append(
            "work_bound_projection_fails_global_physical_sanity_check_at_m1_m2_m4"
        )
        anomalies.extend(
            f"{method}_work_bound_projection_anticorrelates_with_measured_e1_at_m1_m2_m4"
            for method, check in projection_checks.items()
            if check["pearson"] < 0.0 and check["spearman"] < 0.0
        )
    if not external_gates_pass:
        anomalies.append("work_bound_projection_fails_cross_dataset_release_gate")
        anomalies.extend(
            "external_"
            f"{gate['dataset'].replace('-', '_')}_work_bound_projection_gate_failed"
            for gate in external_projection_gates
            if gate["m_gt_4_projection_status"] != "AVAILABLE"
        )
    summary_payload = {
        "timestamp": timestamp,
        "record_type": "e4_aggregate_work_decomposition",
        "experiment_id": experiment_id,
        "e1_source_stem": args.e1_source_stem,
        "e2_source_stem": args.e2_source_stem,
        "e3_source_stem": args.e3_source_stem,
        **correction,
        "git_commit": repository_commit(Path(__file__).resolve().parents[3]),
        "dataset": args.dataset,
        "dataset_sha256": dataset_sha256,
        "target_recall": TARGET_RECALL,
        "query_range": [1_000, 10_000],
        "query_count_per_configuration": 9_000,
        "per_query_row_count": len(per_query_rows),
        "configuration_count": len(summaries),
        "partition_methods": list(METHODS),
        "logical_shard_counts": list(LOGICAL_SHARD_COUNTS),
        "graph_build_seed": graph_build_seed,
        "deterministic_graph_construction": graph_build_seed is not None,
        "hnsw_max_indexing_threads": 1 if graph_build_seed is not None else None,
        "max_optimization_threads": 1 if graph_build_seed is not None else None,
        "configuration_summaries": summaries,
        "model_accuracy": model_accuracy,
        "projection_sanity_checks": projection_checks,
        "external_projection_gate_records": external_projection_gates,
        "projection_release_gate": projection_release_gate,
        "c1_d_aggregate_cost_status": "SUPPORTS",
        "dataset_physical_attribution_status": dataset_physical_attribution_status,
        "dataset_m_gt_4_projection_status": dataset_projection_status,
        "physical_attribution_status": physical_attribution_status,
        "m_gt_4_projection_status": global_projection_status,
        "anomalies": anomalies,
        "status": status,
    }
    summary_json_output = write_json_atomic(args.summary_json_output, summary_payload)
    evidence = {
        "per_query": sha256_path(per_query_output),
        "aggregate_work_summary_csv": sha256_path(summary_csv_output),
        "model_accuracy_csv": sha256_path(model_accuracy_output),
        "summary_json": sha256_path(summary_json_output),
        "e2_per_query": sha256_path(e2_per_query_path),
        "local_work_sources": local_work_hashes,
        "physical_e1_summary": sha256_path(physical_e1_path),
        **{name: sha256_path(path) for name, path in figures.items()},
        "e3_per_search": {
            str(Path(row["source_e3_per_search"])): row["source_e3_per_search_sha256"]
            for row in summaries
        },
    }
    record = {
        "timestamp": timestamp,
        "record_type": "e4_aggregate_work_decomposition",
        "experiment_id": summary_payload["experiment_id"],
        "e1_source_stem": args.e1_source_stem,
        "e2_source_stem": args.e2_source_stem,
        "e3_source_stem": args.e3_source_stem,
        **correction,
        "git_commit": summary_payload["git_commit"],
        "dataset": args.dataset,
        "dataset_checksum": summary_payload["dataset_sha256"],
        "target_recall": TARGET_RECALL,
        "query_range": summary_payload["query_range"],
        "query_count_per_configuration": 9_000,
        "per_query_row_count": len(per_query_rows),
        "configuration_count": len(summaries),
        "partition_methods": list(METHODS),
        "logical_shard_counts": list(LOGICAL_SHARD_COUNTS),
        "graph_build_seed": graph_build_seed,
        "deterministic_graph_construction": graph_build_seed is not None,
        "hnsw_max_indexing_threads": 1 if graph_build_seed is not None else None,
        "max_optimization_threads": 1 if graph_build_seed is not None else None,
        "observed_distance_mean": {
            method: {
                str(row["logical_shard_count"]): row["observed_distance_mean"]
                for row in summaries
                if row["partition_method"] == method
            }
            for method in METHODS
        },
        "observed_worker_cpu_time_us_mean": {
            method: {
                str(row["logical_shard_count"]): row[
                    "observed_worker_cpu_time_us_mean"
                ]
                for row in summaries
                if row["partition_method"] == method
            }
            for method in METHODS
        },
        "model_accuracy": {
            row["partition_method"]: {
                key: row[key]
                for key in (
                    "pearson_model_vs_observed",
                    "spearman_model_vs_observed",
                    "mean_relative_error",
                    "maximum_relative_error",
                )
            }
            for row in model_accuracy
        },
        "projection_sanity_checks": projection_checks,
        "external_projection_gate_records": external_projection_gates,
        "projection_release_gate": projection_release_gate,
        "c1_d_aggregate_cost_status": "SUPPORTS",
        "dataset_physical_attribution_status": dataset_physical_attribution_status,
        "dataset_m_gt_4_projection_status": dataset_projection_status,
        "physical_attribution_status": physical_attribution_status,
        "m_gt_4_projection_status": global_projection_status,
        "anomalies": anomalies,
        "evidence_sha256": evidence,
        "status": status,
    }
    record_output = write_json_atomic(args.record_output, record)
    register_manifest_record_once(args.manifest_jsonl, record)
    result = {
        "per_query_output": str(per_query_output),
        "summary_csv_output": str(summary_csv_output),
        "model_accuracy_output": str(model_accuracy_output),
        "summary_json_output": str(summary_json_output),
        "record_output": str(record_output),
        "figures": {name: str(path) for name, path in figures.items()},
        "record": record,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute C1 E4 observed aggregate work and decomposition accuracy"
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--runs-root", default="experiments/c1/runs")
    parser.add_argument(
        "--e1-source-stem",
        default="stage4-e1",
        help="physical E1 source filename stem before '-<dataset>-<layout>.json'",
    )
    parser.add_argument(
        "--e2-source-stem",
        default="stage3-e2",
        help="E2 per-query filename stem before '-<dataset>-fanout.csv'",
    )
    parser.add_argument(
        "--e3-source-stem",
        default="stage2-e3",
        help="E3 per-search and summary filename stem before '-<dataset>-<method>-m<M>'",
    )
    parser.add_argument("--experiment-id")
    parser.add_argument("--supersedes-experiment-id")
    parser.add_argument("--correction-reason")
    parser.add_argument("--e2-per-query", required=True)
    parser.add_argument(
        "--local-work-summary",
        help="legacy combined E3 local-work CSV; omit to load E3 summary JSON files by stem",
    )
    parser.add_argument("--physical-e1-summary", required=True)
    parser.add_argument("--per-query-output", required=True)
    parser.add_argument("--summary-csv-output", required=True)
    parser.add_argument("--model-accuracy-output", required=True)
    parser.add_argument("--summary-json-output", required=True)
    parser.add_argument("--record-output", required=True)
    parser.add_argument(
        "--manifest-jsonl", default="experiments/c1/runs/manifest.jsonl"
    )
    parser.add_argument(
        "--external-projection-gate-record",
        action="append",
        default=[],
        help=(
            "prior E4 record from another dataset; repeat to require every dataset "
            "to pass before exposing M>4 projections"
        ),
    )
    parser.add_argument("--figures-dir", default="experiments/c1/figures")
    args = parser.parse_args(argv)
    for attribute, label in (
        ("e1_source_stem", "E1"),
        ("e2_source_stem", "E2"),
        ("e3_source_stem", "E3"),
    ):
        try:
            source_component(getattr(args, attribute), label)
        except ValueError as error:
            parser.error(str(error))
    if bool(args.supersedes_experiment_id) != bool(args.correction_reason):
        parser.error(
            "supersedes experiment ID and correction reason must be provided together"
        )
    return args


def main(argv: Sequence[str] | None = None) -> int:
    execute(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
