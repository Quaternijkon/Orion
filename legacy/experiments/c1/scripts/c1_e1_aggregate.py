#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import statistics
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from c1_protocol import repository_commit, sha256_path, utc_timestamp


METHODS = ("random", "kmeans")
PHYSICAL_MACHINE_COUNTS = (1, 2, 4)
TARGET_RECALL = 0.90
DATASET_LABELS = {
    "sift1m": "SIFT1M",
    "glove-200-angular": "GloVe-200-angular",
}
EXPECTED_CONFIGURATIONS = {
    "sift1m": {
        ("random", 1): (1, 24),
        ("random", 2): (2, 24),
        ("random", 4): (4, 16),
        ("kmeans", 1): (1, 24),
        ("kmeans", 2): (1, 24),
        ("kmeans", 4): (1, 48),
    },
    "glove-200-angular": {
        ("random", 1): (1, 384),
        ("random", 2): (2, 256),
        ("random", 4): (4, 128),
        ("kmeans", 1): (1, 384),
        ("kmeans", 2): (2, 256),
        ("kmeans", 4): (3, 256),
    },
}
CSV_FIELDS = (
    "dataset",
    "partition_method",
    "physical_machine_count",
    "logical_shard_count",
    "experiment_id",
    "shared_unsharded_baseline",
    "routing_fanout",
    "ef_search",
    "selected_concurrency",
    "repetition_count",
    "qps_mean",
    "qps_sample_std",
    "qps_ci95_lower",
    "qps_ci95_upper",
    "qps_coefficient_of_variation",
    "normalized_qps_to_m1",
    "scaling_efficiency",
    "achieved_recall_mean",
    "achieved_recall_min",
    "offered_qps_mean",
    "mean_latency_us",
    "p50_latency_us",
    "p95_latency_us",
    "p99_latency_us",
    "mean_shards_per_query",
    "p95_shards_per_query",
    "mean_distance_computations_per_query",
    "mean_nodes_visited_per_query",
    "mean_worker_cpu_time_us_per_query",
    "mean_routing_latency_us",
    "aggregator_cpu_utilization_pct_of_reserved_cores",
    "routing_share_of_routing_plus_worker_cpu_pct",
    "max_worker_cpu_utilization_pct",
    "max_network_utilization_pct",
    "max_aggregator_queue_depth",
    "bottleneck_flags",
    "source_result",
    "source_result_sha256",
    "partition_artifact",
    "partition_artifact_sha256",
    "anomalies",
    "status",
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
        raise ValueError(f"expected a JSON object: {source}")
    return payload


def mean(repetitions: Sequence[dict[str, Any]], field: str) -> float:
    return statistics.fmean(float(row[field]) for row in repetitions)


def load_partition_anomalies(source: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
    if source["partition_method"] != "kmeans":
        return [], {}
    artifact = Path(source["partition_artifact"]).expanduser().resolve()
    if sha256_path(artifact) != source["partition_artifact_sha256"]:
        raise ValueError(f"partition artifact hash mismatch: {artifact}")
    with np.load(artifact, allow_pickle=False) as data:
        metadata = json.loads(str(np.asarray(data["metadata_json"]).item()))
    kmeans = metadata.get("kmeans") or {}
    history = list(kmeans.get("history") or [])
    maximum_iterations = int(kmeans.get("maximum_iterations") or 0)
    tolerance = float(kmeans.get("tolerance") or 0.0)
    last_movement = (
        float(history[-1]["max_centroid_movement"]) if history else None
    )
    converged = bool(
        history
        and last_movement is not None
        and last_movement <= tolerance
        and len(history) <= maximum_iterations
    )
    details = {
        "converged": converged,
        "iterations": len(history),
        "maximum_iterations": maximum_iterations,
        "tolerance": tolerance,
        "last_max_centroid_movement": last_movement,
        "shard_counts": list(map(int, metadata.get("shard_counts") or [])),
    }
    anomalies: list[str] = []
    if not converged and len(history) == maximum_iterations:
        anomalies.append("kmeans_reached_iteration_cap_without_convergence")
    return anomalies, details


def validate_source(
    path: Path,
    source: dict[str, Any],
    *,
    dataset: str,
    expected_method: str,
    expected_machines: int,
) -> None:
    if source.get("status") != "VALID_E1":
        raise ValueError(f"source is not VALID_E1: {path}")
    if source.get("dataset") != dataset:
        raise ValueError(f"dataset mismatch: {path}")
    actual_method = str(source.get("partition_method"))
    if expected_machines == 1:
        if actual_method not in METHODS:
            raise ValueError(f"invalid shared M=1 method: {path}")
    elif actual_method != expected_method:
        raise ValueError(f"partition method mismatch: {path}")
    if int(source.get("physical_machine_count") or 0) != expected_machines:
        raise ValueError(f"physical-machine mismatch: {path}")
    if int(source.get("logical_shard_count") or 0) != expected_machines:
        raise ValueError(f"logical-shard mismatch: {path}")
    try:
        expected_fanout, expected_ef = EXPECTED_CONFIGURATIONS[dataset][
            (expected_method, expected_machines)
        ]
    except KeyError as error:
        raise ValueError(f"unsupported E1 dataset/configuration: {dataset} {error}") from error
    if int(source.get("fanout") or 0) != expected_fanout:
        raise ValueError(f"routing fan-out mismatch: {path}")
    if int(source.get("ef_search") or 0) != expected_ef:
        raise ValueError(f"efSearch mismatch: {path}")
    if source.get("benchmark_cpu_affinity") != list(range(20, 32)):
        raise ValueError(f"benchmark affinity mismatch: {path}")
    worker_affinity = source.get("worker_cpu_affinity") or {}
    if not worker_affinity or set(worker_affinity.values()) != {"0-19"}:
        raise ValueError(f"worker affinity mismatch: {path}")
    repetitions = list(source.get("repetitions") or [])
    count = int(source.get("repetition_count") or 0)
    if count != len(repetitions) or count < 3:
        raise ValueError(f"invalid repetition count: {path}")
    cv = float(source.get("qps_coefficient_of_variation") or 0.0)
    if cv > 0.05 and count < 5:
        raise ValueError(f"CV exceeds 5 percent without five repetitions: {path}")
    completed_qps = [float(row["completed_qps"]) for row in repetitions]
    recorded = source.get("qps_confidence_interval_95") or {}
    if not math.isclose(statistics.fmean(completed_qps), float(recorded["mean"]), rel_tol=1e-12):
        raise ValueError(f"QPS mean mismatch: {path}")
    if not math.isclose(
        statistics.stdev(completed_qps), float(recorded["sample_std"]), rel_tol=1e-12
    ):
        raise ValueError(f"QPS sample standard-deviation mismatch: {path}")
    for repetition in repetitions:
        if repetition.get("run_status") != "VALID_E1":
            raise ValueError(f"invalid repetition: {path}")
        if float(repetition["achieved_recall"]) < TARGET_RECALL:
            raise ValueError(f"recall below target: {path}")
        if repetition.get("bottleneck_flags"):
            raise ValueError(f"bottleneck flag present: {path}")
        if repetition.get("persistent_aggregator_queue_growth"):
            raise ValueError(f"persistent aggregator queue growth: {path}")


def configuration_row(
    source_path: Path,
    source: dict[str, Any],
    *,
    method: str,
    physical_machines: int,
    baseline_qps: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    repetitions = list(source["repetitions"])
    qps = source["qps_confidence_interval_95"]
    normalized_qps = float(qps["mean"]) / baseline_qps
    anomalies, anomaly_details = load_partition_anomalies(source)
    flags = sorted(
        {str(flag) for repetition in repetitions for flag in repetition["bottleneck_flags"]}
    )
    max_queue_depth = max(
        int(repetition["aggregator_waiting_queue_depth_max"])
        for repetition in repetitions
    )
    row = {
        "dataset": source["dataset"],
        "partition_method": method,
        "physical_machine_count": physical_machines,
        "logical_shard_count": physical_machines,
        "experiment_id": source["experiment_id"],
        "shared_unsharded_baseline": physical_machines == 1,
        "routing_fanout": int(source["fanout"]),
        "ef_search": int(source["ef_search"]),
        "selected_concurrency": int(source["selected_concurrency"]),
        "repetition_count": int(source["repetition_count"]),
        "qps_mean": float(qps["mean"]),
        "qps_sample_std": float(qps["sample_std"]),
        "qps_ci95_lower": float(qps["lower"]),
        "qps_ci95_upper": float(qps["upper"]),
        "qps_coefficient_of_variation": float(source["qps_coefficient_of_variation"]),
        "normalized_qps_to_m1": normalized_qps,
        "scaling_efficiency": normalized_qps / physical_machines,
        "achieved_recall_mean": mean(repetitions, "achieved_recall"),
        "achieved_recall_min": min(float(rep["achieved_recall"]) for rep in repetitions),
        "offered_qps_mean": mean(repetitions, "offered_qps"),
        "mean_latency_us": mean(repetitions, "mean_latency_us"),
        "p50_latency_us": mean(repetitions, "p50_latency_us"),
        "p95_latency_us": mean(repetitions, "p95_latency_us"),
        "p99_latency_us": mean(repetitions, "p99_latency_us"),
        "mean_shards_per_query": mean(repetitions, "mean_shards_per_query"),
        "p95_shards_per_query": mean(repetitions, "p95_shards_per_query"),
        "mean_distance_computations_per_query": mean(
            repetitions, "mean_distance_computations_per_query"
        ),
        "mean_nodes_visited_per_query": mean(repetitions, "mean_nodes_visited_per_query"),
        "mean_worker_cpu_time_us_per_query": mean(
            repetitions, "mean_worker_cpu_time_us_per_query"
        ),
        "mean_routing_latency_us": mean(repetitions, "mean_routing_latency_us"),
        "aggregator_cpu_utilization_pct_of_reserved_cores": mean(
            repetitions, "aggregator_cpu_utilization_pct_of_reserved_cores"
        ),
        "routing_share_of_routing_plus_worker_cpu_pct": mean(
            repetitions, "routing_share_of_routing_plus_worker_cpu_pct"
        ),
        "max_worker_cpu_utilization_pct": max(
            float(rep["resource_utilization"]["max_worker_cpu_utilization_pct"])
            for rep in repetitions
        ),
        "max_network_utilization_pct": max(
            float(rep["resource_utilization"]["max_network_utilization_pct"])
            for rep in repetitions
        ),
        "max_aggregator_queue_depth": max_queue_depth,
        "bottleneck_flags": ";".join(flags),
        "source_result": str(source_path),
        "source_result_sha256": sha256_path(source_path),
        "partition_artifact": source["partition_artifact"],
        "partition_artifact_sha256": source["partition_artifact_sha256"],
        "anomalies": ";".join(anomalies),
        "status": "VALID_E1",
    }
    details = {
        **row,
        "qps_repetitions": [float(rep["completed_qps"]) for rep in repetitions],
        "recall_repetitions": [float(rep["achieved_recall"]) for rep in repetitions],
        "logical_to_physical_mapping": source["logical_to_physical_mapping"],
        "saturation_selection": source["saturation_selection"],
        "worker_cpu_affinity": source["worker_cpu_affinity"],
        "benchmark_cpu_affinity": source["benchmark_cpu_affinity"],
        "partition_anomaly_details": anomaly_details,
    }
    return row, details


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
            "Creator": "Orion C1 E1 physical scale-out aggregator",
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
    rows: Sequence[dict[str, Any]],
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

    figure, axis = plt.subplots(figsize=(6.6, 4.2))
    axis.plot(
        PHYSICAL_MACHINE_COUNTS,
        PHYSICAL_MACHINE_COUNTS,
        color="#555555",
        linestyle="--",
        marker="o",
        label="Ideal linear",
    )
    for method in METHODS:
        method_rows = sorted(
            (row for row in rows if row["partition_method"] == method),
            key=lambda row: row["physical_machine_count"],
        )
        y = [float(row["normalized_qps_to_m1"]) for row in method_rows]
        lower = [
            y_value - float(row["qps_ci95_lower"]) / float(method_rows[0]["qps_mean"])
            for row, y_value in zip(method_rows, y, strict=True)
        ]
        upper = [
            float(row["qps_ci95_upper"]) / float(method_rows[0]["qps_mean"]) - y_value
            for row, y_value in zip(method_rows, y, strict=True)
        ]
        axis.errorbar(
            PHYSICAL_MACHINE_COUNTS,
            y,
            yerr=[lower, upper],
            color=colors[method],
            marker="o",
            capsize=3,
            label=labels[method],
        )
    axis.set_xticks(PHYSICAL_MACHINE_COUNTS)
    axis.set_xlabel("Physical workers")
    axis.set_ylabel("Normalized throughput QPS(M) / QPS(1)")
    dataset_label = DATASET_LABELS.get(dataset, dataset)
    axis.set_title(f"{dataset_label} physical scale-out at Recall@10 >= 0.90")
    axis.set_ylim(bottom=0)
    axis.grid(True, alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    physical = save_pdf_atomic(figure, directory / "fig_c1_physical_scaleout.pdf")
    plt.close(figure)
    canonical = copy_atomic(physical, directory / "c1_fig1_physical_scaleout.pdf")

    figure, axis = plt.subplots(figsize=(6.6, 4.2))
    axis.plot(
        PHYSICAL_MACHINE_COUNTS,
        [1.0] * len(PHYSICAL_MACHINE_COUNTS),
        color="#555555",
        linestyle="--",
        label="Ideal linear",
    )
    for method in METHODS:
        method_rows = sorted(
            (row for row in rows if row["partition_method"] == method),
            key=lambda row: row["physical_machine_count"],
        )
        axis.plot(
            PHYSICAL_MACHINE_COUNTS,
            [float(row["scaling_efficiency"]) for row in method_rows],
            color=colors[method],
            marker="o",
            label=labels[method],
        )
    axis.set_xticks(PHYSICAL_MACHINE_COUNTS)
    axis.set_xlabel("Physical workers")
    axis.set_ylabel("Scaling efficiency QPS(M) / (M x QPS(1))")
    axis.set_title(f"{dataset_label} physical scaling efficiency")
    axis.set_ylim(0, 1.05)
    axis.grid(True, alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    efficiency = save_pdf_atomic(figure, directory / "fig_c1_scaling_efficiency.pdf")
    plt.close(figure)
    return {
        "protocol_physical_scaleout": physical,
        "canonical_physical_scaleout": canonical,
        "protocol_scaling_efficiency": efficiency,
    }


def execute(args: argparse.Namespace) -> dict[str, Any]:
    runs_root = Path(args.runs_root).expanduser().resolve()
    source_prefix = f"{args.source_stem}-{args.dataset}"
    common_path = runs_root / f"{source_prefix}-common-m1.json"
    source_paths = {
        ("random", 1): common_path,
        ("kmeans", 1): common_path,
        ("random", 2): runs_root / f"{source_prefix}-random-m2.json",
        ("random", 4): runs_root / f"{source_prefix}-random-m4.json",
        ("kmeans", 2): runs_root / f"{source_prefix}-kmeans-m2.json",
        ("kmeans", 4): runs_root / f"{source_prefix}-kmeans-m4.json",
    }
    sources: dict[tuple[str, int], dict[str, Any]] = {}
    for key, path in source_paths.items():
        source = load_json(path)
        validate_source(
            path,
            source,
            dataset=args.dataset,
            expected_method=key[0],
            expected_machines=key[1],
        )
        sources[key] = source
    dataset_hashes = {str(source["dataset_sha256"]) for source in sources.values()}
    if len(dataset_hashes) != 1:
        raise ValueError("E1 sources do not share one dataset checksum")
    baseline_qps = float(sources[("random", 1)]["qps_confidence_interval_95"]["mean"])
    rows: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    for method in METHODS:
        for physical_machines in PHYSICAL_MACHINE_COUNTS:
            source_path = source_paths[(method, physical_machines)].resolve()
            row, detail = configuration_row(
                source_path,
                sources[(method, physical_machines)],
                method=method,
                physical_machines=physical_machines,
                baseline_qps=baseline_qps,
            )
            rows.append(row)
            details.append(detail)
    csv_output = write_csv_atomic(args.csv_output, CSV_FIELDS, rows)
    figures = generate_figures(args.figures_dir, rows, dataset=args.dataset)
    timestamp = utc_timestamp()
    correction = {}
    if args.supersedes_experiment_id:
        correction = {
            "supersedes_experiment_id": args.supersedes_experiment_id,
            "correction_reason": args.correction_reason,
        }
    sublinear = all(
        float(row["normalized_qps_to_m1"]) < int(row["physical_machine_count"])
        for row in rows
        if int(row["physical_machine_count"]) > 1
    )
    dataset_label = DATASET_LABELS.get(args.dataset, args.dataset)
    conclusion = {
        "random_normalized_qps": {
            str(row["physical_machine_count"]): row["normalized_qps_to_m1"]
            for row in rows
            if row["partition_method"] == "random"
        },
        "kmeans_normalized_qps": {
            str(row["physical_machine_count"]): row["normalized_qps_to_m1"]
            for row in rows
            if row["partition_method"] == "kmeans"
        },
        "random_scaling_efficiency": {
            str(row["physical_machine_count"]): row["scaling_efficiency"]
            for row in rows
            if row["partition_method"] == "random"
        },
        "kmeans_scaling_efficiency": {
            str(row["physical_machine_count"]): row["scaling_efficiency"]
            for row in rows
            if row["partition_method"] == "kmeans"
        },
        "c1_a_status": "SUPPORTS" if sublinear else "INSUFFICIENT",
        "interpretation": (
            f"{dataset_label} physical throughput is sub-linear for both partition methods "
            "at the measured M=2 and M=4 points. Random uses broadcast fan-out, while "
            "K-Means uses its fixed-recall centroid-ranked fan-out; both remain below "
            "ideal linear scaling."
        ),
    }
    summary = {
        "timestamp": timestamp,
        "record_type": "e1_physical_scaleout_analysis",
        "experiment_id": f"{source_prefix}-physical-scaleout",
        "source_stem": args.source_stem,
        **correction,
        "git_commit": repository_commit(Path(__file__).resolve().parents[3]),
        "dataset": args.dataset,
        "dataset_sha256": dataset_hashes.pop(),
        "target_recall": TARGET_RECALL,
        "partition_methods": list(METHODS),
        "physical_machine_counts": list(PHYSICAL_MACHINE_COUNTS),
        "configuration_count": len(details),
        "unique_physical_layout_count": 5,
        "shared_unsharded_baseline": True,
        "configurations": details,
        "conclusion": conclusion,
        "status": "VALID_E1",
    }
    summary_output = write_json_atomic(args.summary_output, summary)
    evidence = {
        "physical_scaleout_csv": sha256_path(csv_output),
        "summary_json": sha256_path(summary_output),
        **{name: sha256_path(path) for name, path in figures.items()},
        "source_results": {
            str(path.resolve()): sha256_path(path)
            for path in sorted(set(source_paths.values()))
        },
    }
    dataset_token = args.dataset.replace("-", "_")
    anomalies: list[str] = []
    for detail in details:
        if "kmeans_reached_iteration_cap_without_convergence" not in str(
            detail["anomalies"]
        ):
            continue
        method = str(detail["partition_method"])
        machines = int(detail["physical_machine_count"])
        anomaly_details = detail["partition_anomaly_details"]
        iterations = int(anomaly_details["iterations"])
        counts = "_".join(map(str, anomaly_details["shard_counts"]))
        anomalies.extend(
            [
                f"{dataset_token}_{method}_m{machines}_reached_{iterations}_iteration_cap_without_convergence",
                f"{dataset_token}_{method}_m{machines}_shard_counts_{counts}",
            ]
        )
    record = {
        "timestamp": timestamp,
        "record_type": "e1_physical_scaleout_analysis",
        "experiment_id": summary["experiment_id"],
        "source_stem": args.source_stem,
        **correction,
        "git_commit": summary["git_commit"],
        "dataset": args.dataset,
        "dataset_checksum": summary["dataset_sha256"],
        "target_recall": TARGET_RECALL,
        "partition_methods": list(METHODS),
        "physical_machine_counts": list(PHYSICAL_MACHINE_COUNTS),
        "configuration_count": len(details),
        "unique_physical_layout_count": 5,
        "shared_unsharded_baseline": True,
        "repetition_counts": {
            f"{row['partition_method']}-m{row['physical_machine_count']}": row[
                "repetition_count"
            ]
            for row in rows
        },
        "qps_mean": {
            method: {
                str(row["physical_machine_count"]): row["qps_mean"]
                for row in rows
                if row["partition_method"] == method
            }
            for method in METHODS
        },
        "qps_sample_std": {
            method: {
                str(row["physical_machine_count"]): row["qps_sample_std"]
                for row in rows
                if row["partition_method"] == method
            }
            for method in METHODS
        },
        "measurement_recall": {
            method: {
                str(row["physical_machine_count"]): row["achieved_recall_mean"]
                for row in rows
                if row["partition_method"] == method
            }
            for method in METHODS
        },
        "normalized_qps": {
            "random": conclusion["random_normalized_qps"],
            "kmeans": conclusion["kmeans_normalized_qps"],
        },
        "scaling_efficiency": {
            "random": conclusion["random_scaling_efficiency"],
            "kmeans": conclusion["kmeans_scaling_efficiency"],
        },
        "bottleneck_validation": {
            "all_configurations_unflagged": all(not row["bottleneck_flags"] for row in rows),
            "maximum_aggregator_cpu_pct": max(
                float(row["aggregator_cpu_utilization_pct_of_reserved_cores"]) for row in rows
            ),
            "maximum_routing_share_pct": max(
                float(row["routing_share_of_routing_plus_worker_cpu_pct"]) for row in rows
            ),
            "maximum_network_utilization_pct": max(
                float(row["max_network_utilization_pct"]) for row in rows
            ),
            "maximum_aggregator_queue_depth": max(
                int(row["max_aggregator_queue_depth"]) for row in rows
            ),
        },
        "anomalies": anomalies,
        "c1_a_status": conclusion["c1_a_status"],
        "evidence_sha256": evidence,
        "status": "VALID_E1",
    }
    record_output = write_json_atomic(args.record_output, record)
    result = {
        "csv_output": str(csv_output),
        "summary_output": str(summary_output),
        "record_output": str(record_output),
        "figures": {name: str(path) for name, path in figures.items()},
        "record": record,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate and validate C1 E1 physical scale-out results"
    )
    parser.add_argument("--dataset", choices=sorted(EXPECTED_CONFIGURATIONS), required=True)
    parser.add_argument("--runs-root", default="experiments/c1/runs")
    parser.add_argument(
        "--source-stem",
        default="stage4-e1",
        help="source filename stem before '-<dataset>-<layout>.json'",
    )
    parser.add_argument("--supersedes-experiment-id")
    parser.add_argument("--correction-reason")
    parser.add_argument("--csv-output", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--record-output", required=True)
    parser.add_argument("--figures-dir", default="experiments/c1/figures")
    args = parser.parse_args(argv)
    if not args.source_stem or Path(args.source_stem).name != args.source_stem:
        parser.error("source stem must be a non-empty filename component")
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
