#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
from pathlib import Path
from typing import Any, Iterable, Sequence

from c1_protocol import repository_commit, sha256_path, utc_timestamp


DATASETS = ("sift1m", "glove-200-angular")
METHODS = ("random", "kmeans")
LOGICAL_SHARDS = (1, 2, 4, 8, 16, 32)
PHYSICAL_WORKERS = (1, 2, 4)
SENSITIVITY_TARGETS = (0.89, 0.90, 0.91)
LABELS = {"sift1m": "SIFT1M", "glove-200-angular": "GloVe-200-angular"}
COLORS = {"random": "#3366cc", "kmeans": "#d97706"}
METHOD_LABELS = {"random": "Random", "kmeans": "K-Means"}
DATASET_VECTOR_COUNTS = {"sift1m": 1_000_000, "glove-200-angular": 1_183_514}
REPRESENTATIVE_CDF_SHARDS = (4, 16, 32)
SHARD_LINESTYLES = {4: "-", 16: "--", 32: ":"}


def load_json(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {source}")
    return payload


def source_component(value: str, label: str) -> str:
    if not value or Path(value).name != value:
        raise ValueError(f"{label} source stem must be one filename component")
    return value


def validate_section31_proof(path: Path) -> dict[str, Any]:
    proof = load_json(path)
    required = {
        "graph_build_seed",
        "image_id",
        "max_indexing_threads",
        "max_optimization_threads",
        "identical_build_count",
        "graph_content_sha256",
        "source_commit",
    }
    if not required.issubset(proof):
        raise ValueError(f"Section 31 proof is incomplete: {path}")
    if (
        proof.get("deterministic_construction_verified") is not True
        or proof.get("authoritative_e2_e4_rerun_complete") is not True
        or int(proof["identical_build_count"]) < 2
        or int(proof["max_indexing_threads"]) != 1
        or int(proof["max_optimization_threads"]) != 1
    ):
        raise ValueError(f"Section 31 proof is not complete: {path}")
    return proof


def atomic_write_text(path: str | Path, value: str) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, destination)
    return destination


def write_json(path: str | Path, payload: Any) -> Path:
    return atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def register_manifest_record_once(path: Path, payload: dict[str, Any]) -> None:
    existing = []
    if path.is_file():
        existing = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    matches = [
        record
        for record in existing
        if record.get("experiment_id") == payload.get("experiment_id")
    ]
    if matches:
        if len(matches) == 1 and matches[0] == payload:
            return
        raise ValueError(
            f"manifest already contains a different record for {payload.get('experiment_id')}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_csv(path: str | Path) -> tuple[list[str], list[dict[str, str]]]:
    source = Path(path).expanduser().resolve()
    with source.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {source}")
        return list(reader.fieldnames), list(reader)


def write_csv(
    path: str | Path, fieldnames: Sequence[str], rows: Iterable[dict[str, Any]]
) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
    os.replace(temporary, destination)
    return destination


def combine_csvs(
    paths: Sequence[Path], output: Path, *, sort_key: Any
) -> tuple[Path, list[dict[str, str]]]:
    header: list[str] | None = None
    rows: list[dict[str, str]] = []
    for path in paths:
        current_header, current_rows = read_csv(path)
        if header is None:
            header = current_header
        elif current_header != header:
            raise ValueError(f"CSV schema mismatch: {path}")
        rows.extend(current_rows)
    if header is None:
        raise ValueError("no CSV inputs")
    rows.sort(key=sort_key)
    return write_csv(output, header, rows), rows


def tuning_path(
    runs: Path,
    dataset: str,
    method: str,
    shards: int,
    *,
    e3_source_stem: str,
) -> Path:
    stem = runs / (
        f"{source_component(e3_source_stem, 'E3')}-{dataset}-{method}-m{shards}"
    )
    candidates = (
        Path(str(stem) + "-tuning-pinned.json"),
        Path(str(stem) + "-tuning-prefix-pinned.json"),
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"no deterministic tuning artifact found among {candidates}")


def select_sensitivity_candidate(
    tuning: dict[str, Any], *, fanout: int, ef_search: int
) -> dict[str, Any]:
    matches = [
        row
        for row in tuning.get("candidates") or []
        if int(row["fanout"]) == fanout and int(row["ef_search"]) == ef_search
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one sensitivity candidate for P={fanout}, ef={ef_search}; "
            f"found {len(matches)}"
        )
    return dict(matches[0])


def build_sensitivity_rows(
    runs: Path,
    *,
    e2_source_stem: str,
    e3_source_stem: str,
    graph_build_seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    source_hashes: dict[str, str] = {}
    for dataset in DATASETS:
        e2_path = runs / f"{e2_source_stem}-{dataset}-fanout-summary.json"
        e2 = load_json(e2_path)
        if (
            e2.get("deterministic_graph_construction") is not True
            or int(e2.get("graph_build_seed") or -1) != graph_build_seed
        ):
            raise ValueError(f"E2 source is not the authoritative deterministic build: {e2_path}")
        source_hashes[str(e2_path.resolve())] = sha256_path(e2_path)
        configurations = {
            (row["partition_method"], int(row["logical_shard_count"])): row
            for row in e2["configuration_summaries"]
        }
        for method in METHODS:
            for shards in (4, 16):
                source = tuning_path(
                    runs,
                    dataset,
                    method,
                    shards,
                    e3_source_stem=e3_source_stem,
                )
                tuning = load_json(source)
                source_hash = sha256_path(source)
                source_hashes[str(source.resolve())] = source_hash
                sensitivity = configurations[(method, shards)]["sensitivity"]
                for target in SENSITIVITY_TARGETS:
                    selected = sensitivity[f"{target:.2f}"]
                    fanout = int(selected["globally_selected_fanout"])
                    ef_search = int(selected["globally_selected_ef_search"])
                    candidate = select_sensitivity_candidate(
                        tuning, fanout=fanout, ef_search=ef_search
                    )
                    aggregate = float(candidate["mean_aggregate_distance_computations"])
                    recall = float(candidate["recall_at_10"])
                    if recall < target:
                        raise ValueError(
                            f"sensitivity candidate misses target {target}: {source}"
                        )
                    rows.append(
                        {
                            "dataset": dataset,
                            "partition_method": method,
                            "logical_shard_count": shards,
                            "sensitivity_target_recall": target,
                            "selected_fanout": fanout,
                            "selected_ef_search": ef_search,
                            "achieved_tuning_recall": recall,
                            "mean_aggregate_distance_computations": aggregate,
                            "mean_local_distance_computations": aggregate / fanout,
                            "source_tuning": str(source.resolve()),
                            "source_tuning_sha256": source_hash,
                        }
                    )
    checks: dict[str, Any] = {}
    for dataset in DATASETS:
        checks[dataset] = {}
        for method in METHODS:
            checks[dataset][method] = {}
            for target in SENSITIVITY_TARGETS:
                selected = {
                    int(row["logical_shard_count"]): row
                    for row in rows
                    if row["dataset"] == dataset
                    and row["partition_method"] == method
                    and float(row["sensitivity_target_recall"]) == target
                }
                lower = selected[4]
                upper = selected[16]
                local_ratio = (
                    float(upper["mean_local_distance_computations"])
                    / float(lower["mean_local_distance_computations"])
                )
                fanout_non_decreasing = int(upper["selected_fanout"]) >= int(
                    lower["selected_fanout"]
                )
                aggregate_non_decreasing = float(
                    upper["mean_aggregate_distance_computations"]
                ) >= float(lower["mean_aggregate_distance_computations"])
                checks[dataset][method][f"{target:.2f}"] = {
                    "m16_to_m4_local_work_ratio": local_ratio,
                    "ideal_m16_to_m4_local_work_ratio": 0.25,
                    "local_work_decreases_slower_than_ideal": local_ratio > 0.25,
                    "fanout_non_decreasing_m4_to_m16": fanout_non_decreasing,
                    "aggregate_work_non_decreasing_m4_to_m16": aggregate_non_decreasing,
                    "qualitative_c1_trend_preserved": (
                        local_ratio > 0.25
                        and fanout_non_decreasing
                        and aggregate_non_decreasing
                    ),
                }
    contradicted_checks = [
        f"{dataset}/{method}/{target}"
        for dataset, dataset_checks in checks.items()
        for method, method_checks in dataset_checks.items()
        for target, item in method_checks.items()
        if not item["qualitative_c1_trend_preserved"]
    ]
    rows.sort(
        key=lambda row: (
            DATASETS.index(str(row["dataset"])),
            METHODS.index(str(row["partition_method"])),
            int(row["logical_shard_count"]),
            float(row["sensitivity_target_recall"]),
        )
    )
    return rows, {
        "checks": checks,
        "all_trends_preserved": not contradicted_checks,
        "contradicted_checks": contradicted_checks,
        "source_sha256": source_hashes,
    }


LOCAL_FIELDS = (
    "experiment_id",
    "timestamp",
    "status",
    "dataset",
    "partition_method",
    "logical_shard_count",
    "routing_fanout",
    "ef_search",
    "selected_tuning_recall",
    "query_count",
    "searched_shard_events",
    "mean_distance_computations_per_searched_shard",
    "p95_distance_computations_per_searched_shard",
    "normalized_distance_to_m1",
    "mean_nodes_visited_per_searched_shard",
    "p95_nodes_visited_per_searched_shard",
    "mean_worker_cpu_time_us_per_searched_shard",
    "mean_worker_cpu_wall_time_us_per_searched_shard",
    "mean_local_search_latency_us_per_searched_shard",
    "p95_local_search_latency_us_per_searched_shard",
    "mean_shard_point_count",
    "mean_vector_storage_size_bytes",
    "controller_graph_index_size_bytes",
    "controller_total_local_index_size_bytes",
    "machine_llc_size_bytes",
    "cache_regime",
    "source_summary",
    "source_summary_sha256",
)


def build_local_work_rows(
    runs: Path, *, e3_source_stem: str, graph_build_seed: int
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    baselines: dict[tuple[str, str], float] = {}
    summaries: dict[tuple[str, str, int], tuple[Path, dict[str, Any]]] = {}
    for dataset in DATASETS:
        for method in METHODS:
            for shards in LOGICAL_SHARDS:
                path = runs / (
                    f"{e3_source_stem}-{dataset}-{method}-m{shards}-summary.json"
                )
                summary = load_json(path)
                if (
                    summary.get("deterministic_graph_construction") is not True
                    or int(summary.get("graph_build_seed") or -1) != graph_build_seed
                    or int(summary.get("hnsw_max_indexing_threads") or 0) != 1
                    or int(summary.get("max_optimization_threads") or 0) != 1
                ):
                    raise ValueError(
                        f"E3 source is not the authoritative deterministic build: {path}"
                    )
                summaries[(dataset, method, shards)] = (path, summary)
                if shards == 1:
                    baselines[(dataset, method)] = float(
                        summary["mean_distance_computations_per_searched_shard"]
                    )
    for dataset in DATASETS:
        for method in METHODS:
            for shards in LOGICAL_SHARDS:
                path, summary = summaries[(dataset, method, shards)]
                resources = list((summary.get("shard_resources") or {}).values())
                if not resources:
                    raise ValueError(f"missing shard resources: {path}")
                controller = [
                    row for row in resources if row.get("physical_host") == "10.10.1.1"
                ]
                graph_sizes = [
                    int(row["graph_index_size_bytes"])
                    for row in controller
                    if row.get("graph_index_size_bytes") is not None
                ]
                total_sizes = [
                    int(row["total_local_index_size_bytes"])
                    for row in controller
                    if row.get("total_local_index_size_bytes") is not None
                ]
                mean_distance = float(
                    summary["mean_distance_computations_per_searched_shard"]
                )
                vector_sizes = [int(row["vector_storage_size_bytes"]) for row in resources]
                llc_sizes = [int(row["machine_llc_size_bytes"]) for row in resources]
                rows.append(
                    {
                        "experiment_id": (
                            f"{e3_source_stem}-{dataset}-{method}-m{shards}"
                        ),
                        "timestamp": summary["timestamp"],
                        "status": summary["result_status"],
                        "dataset": dataset,
                        "partition_method": method,
                        "logical_shard_count": shards,
                        "routing_fanout": summary["selected_fanout"],
                        "ef_search": summary["ef_search"],
                        "selected_tuning_recall": summary["selected_tuning_recall"],
                        "query_count": summary["query_count"],
                        "searched_shard_events": summary["searched_shard_events"],
                        "mean_distance_computations_per_searched_shard": mean_distance,
                        "p95_distance_computations_per_searched_shard": summary[
                            "p95_distance_computations_per_searched_shard"
                        ],
                        "normalized_distance_to_m1": mean_distance
                        / baselines[(dataset, method)],
                        "mean_nodes_visited_per_searched_shard": summary[
                            "mean_nodes_visited_per_searched_shard"
                        ],
                        "p95_nodes_visited_per_searched_shard": summary[
                            "p95_nodes_visited_per_searched_shard"
                        ],
                        "mean_worker_cpu_time_us_per_searched_shard": summary[
                            "mean_worker_cpu_time_us_per_searched_shard"
                        ],
                        "mean_worker_cpu_wall_time_us_per_searched_shard": summary[
                            "mean_worker_cpu_wall_time_us_per_searched_shard"
                        ],
                        "mean_local_search_latency_us_per_searched_shard": summary[
                            "mean_local_search_latency_us_per_searched_shard"
                        ],
                        "p95_local_search_latency_us_per_searched_shard": summary[
                            "p95_local_search_latency_us_per_searched_shard"
                        ],
                        "mean_shard_point_count": sum(
                            int(row["point_count"]) for row in resources
                        )
                        / len(resources),
                        "mean_vector_storage_size_bytes": sum(vector_sizes)
                        / len(vector_sizes),
                        "controller_graph_index_size_bytes": max(graph_sizes)
                        if graph_sizes
                        else "",
                        "controller_total_local_index_size_bytes": max(total_sizes)
                        if total_sizes
                        else "",
                        "machine_llc_size_bytes": min(llc_sizes),
                        "cache_regime": (
                            "all_shard_vector_stores_larger_than_llc"
                            if all(
                                vector > llc
                                for vector, llc in zip(vector_sizes, llc_sizes, strict=True)
                            )
                            else "mixed_or_below_llc"
                        ),
                        "source_summary": str(path.resolve()),
                        "source_summary_sha256": sha256_path(path),
                    }
                )
    return rows


def evaluate_final_mechanisms(
    fanout: Sequence[dict[str, str]],
    local: Sequence[dict[str, Any]],
    e4_records: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    local_index = {
        (
            str(row["dataset"]),
            str(row["partition_method"]),
            int(row["logical_shard_count"]),
        ): row
        for row in local
    }
    slow_local_checks: dict[str, dict[str, dict[str, Any]]] = {}
    slow_local_supported = True
    for dataset in DATASETS:
        slow_local_checks[dataset] = {}
        for method in METHODS:
            baseline = local_index[(dataset, method, 1)]
            distance_baseline = float(
                baseline["mean_distance_computations_per_searched_shard"]
            )
            nodes_baseline = float(baseline["mean_nodes_visited_per_searched_shard"])
            method_checks: dict[str, Any] = {}
            for shards in LOGICAL_SHARDS[1:]:
                row = local_index[(dataset, method, shards)]
                distance_ratio = float(
                    row["mean_distance_computations_per_searched_shard"]
                ) / distance_baseline
                nodes_ratio = (
                    float(row["mean_nodes_visited_per_searched_shard"])
                    / nodes_baseline
                )
                passes = (
                    distance_ratio > 1.0 / shards
                    and nodes_ratio > 1.0 / shards
                )
                method_checks[str(shards)] = {
                    "distance_ratio_to_m1": distance_ratio,
                    "nodes_ratio_to_m1": nodes_ratio,
                    "ideal_ratio": 1.0 / shards,
                    "slower_than_ideal": passes,
                }
                slow_local_supported &= passes
            slow_local_checks[dataset][method] = method_checks

    sift_kmeans = sorted(
        (
            row
            for row in fanout
            if row["dataset"] == "sift1m" and row["partition_method"] == "kmeans"
        ),
        key=lambda row: int(row["logical_shard_count"]),
    )
    if [int(row["logical_shard_count"]) for row in sift_kmeans] != list(
        LOGICAL_SHARDS
    ):
        raise ValueError("SIFT K-Means fan-out rows are incomplete")
    sift_kmeans_fanout = {
        str(row["logical_shard_count"]): float(row["actual_fanout_mean"])
        for row in sift_kmeans
    }
    final_sift_kmeans_fraction = sift_kmeans_fanout["32"] / 32.0
    generalized_fanout_contradicted = final_sift_kmeans_fraction <= 0.125
    physical_attribution_insufficient = all(
        record.get("physical_attribution_status") == "INSUFFICIENT"
        for record in e4_records.values()
    )
    if not generalized_fanout_contradicted:
        raise ValueError(
            "deterministic SIFT K-Means no longer satisfies the recorded small-fan-out contradiction"
        )
    if not physical_attribution_insufficient:
        raise ValueError(
            "deterministic E4 records no longer support the recorded physical-attribution boundary"
        )
    return {
        "slow_local_work_supported": slow_local_supported,
        "slow_local_work_checks": slow_local_checks,
        "sift1m_kmeans_actual_fanout": sift_kmeans_fanout,
        "sift1m_kmeans_m32_normalized_fanout": final_sift_kmeans_fraction,
        "generalized_high_fanout_contradicted": generalized_fanout_contradicted,
        "physical_attribution_insufficient": physical_attribution_insufficient,
    }


def save_pdf(figure: Any, path: Path) -> Path:
    destination = path.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    figure.savefig(
        temporary,
        format="pdf",
        bbox_inches="tight",
        metadata={
            "Title": destination.stem,
            "Creator": "Orion C1 cross-dataset finalizer",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    os.replace(temporary, destination)
    return destination


def copy_atomic(source: Path, destination: Path) -> Path:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copyfile(source, temporary)
    os.replace(temporary, destination)
    return destination


def generate_figures(
    figures: Path,
    physical: list[dict[str, str]],
    fanout: list[dict[str, str]],
    local: list[dict[str, Any]],
    aggregate: list[dict[str, str]],
) -> dict[str, Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2), sharey=True)
    for axis, dataset in zip(axes, DATASETS, strict=True):
        for method in METHODS:
            rows = sorted(
                (r for r in physical if r["dataset"] == dataset and r["partition_method"] == method),
                key=lambda r: int(r["physical_machine_count"]),
            )
            axis.errorbar(
                [int(r["physical_machine_count"]) for r in rows],
                [float(r["normalized_qps_to_m1"]) for r in rows],
                yerr=[
                    float(r["qps_sample_std"]) / float(rows[0]["qps_mean"])
                    for r in rows
                ],
                marker="o",
                capsize=3,
                color=COLORS[method],
                label=METHOD_LABELS[method],
            )
        axis.plot(PHYSICAL_WORKERS, PHYSICAL_WORKERS, ":", color="#555555", label="Ideal linear")
        axis.set_xticks(PHYSICAL_WORKERS)
        axis.set_xlabel("Physical workers")
        axis.set_title(LABELS[dataset])
        axis.grid(True, alpha=0.25)
    axes[0].set_ylabel("Normalized measured QPS")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper center", ncols=3)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    outputs["c1_fig1_physical_scaleout"] = save_pdf(fig, figures / "c1_fig1_physical_scaleout.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2), sharey=True)
    for axis, dataset in zip(axes, DATASETS, strict=True):
        for method in METHODS:
            rows = sorted(
                (
                    row
                    for row in physical
                    if row["dataset"] == dataset
                    and row["partition_method"] == method
                ),
                key=lambda row: int(row["physical_machine_count"]),
            )
            axis.plot(
                [int(row["physical_machine_count"]) for row in rows],
                [float(row["scaling_efficiency"]) for row in rows],
                marker="o",
                color=COLORS[method],
                label=METHOD_LABELS[method],
            )
        axis.axhline(1.0, linestyle=":", color="#555555", label="Ideal efficiency")
        axis.set_xticks(PHYSICAL_WORKERS)
        axis.set_ylim(bottom=0.0, top=1.08)
        axis.set_xlabel("Physical workers")
        axis.set_title(LABELS[dataset])
        axis.grid(True, alpha=0.25)
    axes[0].set_ylabel("Scaling efficiency E(M)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper center", ncols=3)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    outputs["fig_c1_scaling_efficiency"] = save_pdf(
        fig, figures / "fig_c1_scaling_efficiency.pdf"
    )
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2), sharey=True)
    for axis, dataset in zip(axes, DATASETS, strict=True):
        for method in METHODS:
            rows = sorted(
                (r for r in fanout if r["dataset"] == dataset and r["partition_method"] == method),
                key=lambda r: int(r["logical_shard_count"]),
            )
            x = [int(r["logical_shard_count"]) for r in rows]
            axis.plot(x, [float(r["actual_fanout_mean"]) for r in rows], marker="o", color=COLORS[method], label=f"{METHOD_LABELS[method]} actual")
            axis.plot(x, [float(r["oracle_fanout_mean"]) for r in rows], "--", color=COLORS[method], alpha=0.65, label=f"{METHOD_LABELS[method]} oracle")
        axis.set_xscale("log", base=2)
        axis.set_xticks(LOGICAL_SHARDS, [str(x) for x in LOGICAL_SHARDS])
        axis.set_xlabel("Logical shards")
        axis.set_title(LABELS[dataset])
        axis.grid(True, alpha=0.25)
    axes[0].set_ylabel("Mean shard fan-out")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper center", ncols=4)
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    outputs["c1_fig2_fanout_vs_logical_shards"] = save_pdf(fig, figures / "c1_fig2_fanout_vs_logical_shards.pdf")
    plt.close(fig)

    def plot_discrete_cdf(
        axis: Any,
        counts_json: str,
        *,
        color: str,
        linestyle: str,
        label: str,
    ) -> None:
        counts = {int(key): int(value) for key, value in json.loads(counts_json).items()}
        total = sum(counts.values())
        cumulative = 0
        sorted_values = sorted(counts)
        x_values: list[int] = [sorted_values[0]]
        y_values: list[float] = [0.0]
        for value in sorted_values:
            cumulative += counts[value]
            x_values.append(value)
            y_values.append(cumulative / total)
        axis.step(
            x_values,
            y_values,
            where="post",
            color=color,
            linestyle=linestyle,
            label=label,
        )

    for distribution_field, output_name, x_label in (
        (
            "oracle_distribution_counts",
            "c1_fig3_required_fanout_cdf",
            "Oracle minimum fan-out",
        ),
        (
            "actual_distribution_counts",
            "fig_c1_actual_fanout_cdf",
            "Actual serving fan-out",
        ),
    ):
        fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.5), sharey=True)
        for axis, dataset in zip(axes, DATASETS, strict=True):
            for method in METHODS:
                for shards in REPRESENTATIVE_CDF_SHARDS:
                    row = next(
                        row
                        for row in fanout
                        if row["dataset"] == dataset
                        and row["partition_method"] == method
                        and int(row["logical_shard_count"]) == shards
                    )
                    plot_discrete_cdf(
                        axis,
                        row[distribution_field],
                        color=COLORS[method],
                        linestyle=SHARD_LINESTYLES[shards],
                        label=f"{METHOD_LABELS[method]} M={shards}",
                    )
            axis.set_xlabel(x_label)
            axis.set_title(LABELS[dataset])
            axis.grid(True, alpha=0.25)
        axes[0].set_ylabel("Empirical CDF")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, frameon=False, loc="upper center", ncols=3)
        fig.tight_layout(rect=(0, 0, 1, 0.82))
        outputs[output_name] = save_pdf(fig, figures / f"{output_name}.pdf")
        plt.close(fig)

    def normalized_local_values(
        dataset: str, method: str, field: str
    ) -> list[float]:
        rows = sorted(
            (
                row
                for row in local
                if row["dataset"] == dataset
                and row["partition_method"] == method
            ),
            key=lambda row: int(row["logical_shard_count"]),
        )
        values = [float(row[field]) for row in rows]
        baseline = values[0]
        if baseline <= 0:
            raise ValueError(
                f"non-positive M=1 local-work baseline for {dataset}/{method}/{field}"
            )
        return [value / baseline for value in values]

    def normalized_log_reference(dataset: str) -> list[float]:
        vector_count = DATASET_VECTOR_COUNTS[dataset]
        baseline = math.log(vector_count)
        return [math.log(vector_count / shards) / baseline for shards in LOGICAL_SHARDS]

    local_metrics = (
        (
            "mean_distance_computations_per_searched_shard",
            "Normalized distance computations/searched shard",
            "fig_c1_local_distance_computations",
            True,
        ),
        (
            "mean_nodes_visited_per_searched_shard",
            "Normalized nodes visited/searched shard",
            "fig_c1_local_nodes_visited",
            False,
        ),
        (
            "mean_worker_cpu_time_us_per_searched_shard",
            "Normalized worker CPU time/searched shard",
            "fig_c1_local_cpu_time",
            False,
        ),
    )
    for field, y_label, output_name, include_log in local_metrics:
        fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2), sharey=True)
        for axis, dataset in zip(axes, DATASETS, strict=True):
            for method in METHODS:
                axis.plot(
                    LOGICAL_SHARDS,
                    normalized_local_values(dataset, method, field),
                    marker="o",
                    color=COLORS[method],
                    label=f"Measured {METHOD_LABELS[method]}",
                )
            axis.plot(
                LOGICAL_SHARDS,
                [1 / shards for shards in LOGICAL_SHARDS],
                ":",
                color="#555555",
                label="Ideal 1/M",
            )
            if include_log:
                axis.plot(
                    LOGICAL_SHARDS,
                    normalized_log_reference(dataset),
                    "-.",
                    color="#2f855a",
                    label="Normalized log(N/M) reference",
                )
            axis.set_xscale("log", base=2)
            axis.set_yscale("log", base=2)
            axis.set_xticks(LOGICAL_SHARDS, [str(value) for value in LOGICAL_SHARDS])
            axis.set_xlabel("Logical shards")
            axis.set_title(LABELS[dataset])
            axis.grid(True, which="both", alpha=0.25)
        axes[0].set_ylabel(y_label)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            frameon=False,
            loc="upper center",
            ncols=4 if include_log else 3,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.90))
        outputs[output_name] = save_pdf(fig, figures / f"{output_name}.pdf")
        plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(15.2, 7.7), sharex=True, sharey="col")
    for row_index, dataset in enumerate(DATASETS):
        for column_index, (field, y_label, _output_name, include_log) in enumerate(
            local_metrics
        ):
            axis = axes[row_index][column_index]
            for method in METHODS:
                axis.plot(
                    LOGICAL_SHARDS,
                    normalized_local_values(dataset, method, field),
                    marker="o",
                    color=COLORS[method],
                    label=f"Measured {METHOD_LABELS[method]}",
                )
            axis.plot(
                LOGICAL_SHARDS,
                [1 / shards for shards in LOGICAL_SHARDS],
                ":",
                color="#555555",
                label="Ideal 1/M",
            )
            if include_log:
                axis.plot(
                    LOGICAL_SHARDS,
                    normalized_log_reference(dataset),
                    "-.",
                    color="#2f855a",
                    label="Normalized log(N/M) reference",
                )
            axis.set_xscale("log", base=2)
            axis.set_yscale("log", base=2)
            axis.set_xticks(LOGICAL_SHARDS, [str(value) for value in LOGICAL_SHARDS])
            axis.grid(True, which="both", alpha=0.25)
            if column_index == 0:
                axis.set_ylabel(f"{LABELS[dataset]}\nNormalized work")
            if row_index == len(DATASETS) - 1:
                axis.set_xlabel("Logical shards")
            if row_index == 0:
                axis.set_title(
                    (
                        "Distance computations"
                        if column_index == 0
                        else "Nodes visited"
                        if column_index == 1
                        else "Worker CPU time"
                    )
                )
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper center", ncols=4)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    outputs["c1_fig4_local_search_work"] = save_pdf(
        fig, figures / "c1_fig4_local_search_work.pdf"
    )
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2), sharey=True)
    for axis, dataset in zip(axes, DATASETS, strict=True):
        for method in METHODS:
            rows = sorted((r for r in aggregate if r["dataset"] == dataset and r["partition_method"] == method), key=lambda r: int(r["logical_shard_count"]))
            axis.plot(LOGICAL_SHARDS, [float(r["normalized_observed_distance_to_m1"]) for r in rows], marker="o", color=COLORS[method], label=f"{METHOD_LABELS[method]} observed")
            axis.plot(LOGICAL_SHARDS, [float(r["normalized_modeled_distance_to_m1"]) for r in rows], "x--", color=COLORS[method], alpha=0.7, label=f"{METHOD_LABELS[method]} model")
        axis.plot(
            LOGICAL_SHARDS,
            [1 / shards for shards in LOGICAL_SHARDS],
            ":",
            color="#555555",
            label="Ideal 1/M query-work reduction",
        )
        axis.set_xscale("log", base=2)
        axis.set_yscale("log", base=2)
        axis.set_xticks(LOGICAL_SHARDS, [str(x) for x in LOGICAL_SHARDS])
        axis.set_xlabel("Logical shards")
        axis.set_title(LABELS[dataset])
        axis.grid(True, which="both", alpha=0.25)
    axes[0].set_ylabel("Normalized aggregate distance computations/query")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper center", ncols=3)
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    outputs["c1_fig5_aggregate_work_decomposition"] = save_pdf(fig, figures / "c1_fig5_aggregate_work_decomposition.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.5), sharex=True, sharey=True)
    all_values = [
        float(row[field])
        for row in aggregate
        for field in (
            "normalized_observed_distance_to_m1",
            "normalized_modeled_distance_to_m1",
        )
    ]
    lower = min(all_values)
    upper = max(all_values)
    for axis, dataset in zip(axes, DATASETS, strict=True):
        for method in METHODS:
            rows = sorted(
                (
                    row
                    for row in aggregate
                    if row["dataset"] == dataset
                    and row["partition_method"] == method
                ),
                key=lambda row: int(row["logical_shard_count"]),
            )
            x_values = [
                float(row["normalized_observed_distance_to_m1"]) for row in rows
            ]
            y_values = [
                float(row["normalized_modeled_distance_to_m1"]) for row in rows
            ]
            axis.plot(
                x_values,
                y_values,
                marker="o",
                linestyle="none",
                color=COLORS[method],
                label=METHOD_LABELS[method],
            )
            for row, x_value, y_value in zip(rows, x_values, y_values, strict=True):
                axis.annotate(
                    f"M={row['logical_shard_count']}",
                    (x_value, y_value),
                    xytext=(4, 3),
                    textcoords="offset points",
                    fontsize=7,
                )
        axis.plot([lower, upper], [lower, upper], ":", color="#555555", label="Exact agreement")
        axis.set_xscale("log", base=2)
        axis.set_yscale("log", base=2)
        axis.set_xlabel("Normalized observed aggregate work")
        axis.set_title(LABELS[dataset])
        axis.grid(True, which="both", alpha=0.25)
    axes[0].set_ylabel("Normalized fan-out x local-work model")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper center", ncols=3)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    outputs["fig_c1_model_vs_observed"] = save_pdf(
        fig, figures / "fig_c1_model_vs_observed.pdf"
    )
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10.8, 8.0), sharex=True)
    for row_index, dataset in enumerate(DATASETS):
        for column_index, method in enumerate(METHODS):
            axis = axes[row_index][column_index]
            rows = sorted((r for r in aggregate if r["dataset"] == dataset and r["partition_method"] == method and int(r["logical_shard_count"]) <= 4), key=lambda r: int(r["logical_shard_count"]))
            axis.plot(
                PHYSICAL_WORKERS,
                [float(r["physical_work_bound_projection"]) for r in rows],
                "s--",
                color="#8c564b",
                label="Projected from measured aggregate graph-search work",
            )
            axis.plot(PHYSICAL_WORKERS, [float(r["physical_e1_normalized_qps"]) for r in rows], "o-", color=COLORS[method], label="Measured E1")
            axis.set_xticks(PHYSICAL_WORKERS)
            axis.set_title(f"{LABELS[dataset]} - {METHOD_LABELS[method]}")
            axis.grid(True, alpha=0.25)
            axis.text(0.03, 0.05, "M>4 withheld:\ncross-dataset gate failed", transform=axis.transAxes, fontsize=8, bbox={"facecolor": "white", "edgecolor": "#999999", "alpha": 0.9})
    for axis in axes[-1]:
        axis.set_xlabel("Physical workers")
    for axis in axes[:, 0]:
        axis.set_ylabel("Normalized QPS")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper center", ncols=2)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    outputs["c1_fig6_projected_logical_scaling"] = save_pdf(fig, figures / "c1_fig6_projected_logical_scaling.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(2, 4, figsize=(16.0, 7.4))
    for row_index, dataset in enumerate(DATASETS):
        for method in METHODS:
            e1 = sorted((r for r in physical if r["dataset"] == dataset and r["partition_method"] == method), key=lambda r: int(r["physical_machine_count"]))
            e2 = sorted((r for r in fanout if r["dataset"] == dataset and r["partition_method"] == method), key=lambda r: int(r["logical_shard_count"]))
            e3 = sorted(
                (
                    r
                    for r in local
                    if r["dataset"] == dataset
                    and r["partition_method"] == method
                ),
                key=lambda r: int(r["logical_shard_count"]),
            )
            e4 = sorted((r for r in aggregate if r["dataset"] == dataset and r["partition_method"] == method), key=lambda r: int(r["logical_shard_count"]))
            axes[row_index][0].plot(PHYSICAL_WORKERS, [float(r["normalized_qps_to_m1"]) for r in e1], marker="o", color=COLORS[method], label=METHOD_LABELS[method])
            axes[row_index][1].plot(LOGICAL_SHARDS, [float(r["actual_fanout_mean"]) for r in e2], marker="o", color=COLORS[method])
            axes[row_index][2].plot(LOGICAL_SHARDS, [float(r["normalized_distance_to_m1"]) for r in e3], marker="o", color=COLORS[method])
            axes[row_index][3].plot(LOGICAL_SHARDS, [float(r["normalized_observed_distance_to_m1"]) for r in e4], marker="o", color=COLORS[method])
            axes[row_index][3].plot(LOGICAL_SHARDS, [float(r["normalized_modeled_distance_to_m1"]) for r in e4], "x--", color=COLORS[method], alpha=0.7)
        axes[row_index][0].plot(PHYSICAL_WORKERS, PHYSICAL_WORKERS, ":", color="#555555")
        axes[row_index][2].plot(LOGICAL_SHARDS, [1 / m for m in LOGICAL_SHARDS], ":", color="#555555")
        axes[row_index][2].plot(
            LOGICAL_SHARDS,
            normalized_log_reference(dataset),
            "-.",
            color="#2f855a",
        )
        axes[row_index][3].plot(
            LOGICAL_SHARDS,
            [1 / m for m in LOGICAL_SHARDS],
            ":",
            color="#555555",
        )
        axes[row_index][0].set_ylabel(LABELS[dataset])
        for column in (1, 2, 3):
            axes[row_index][column].set_xscale("log", base=2)
            axes[row_index][column].set_xticks(LOGICAL_SHARDS, [str(x) for x in LOGICAL_SHARDS])
        axes[row_index][2].set_yscale("log", base=2)
        axes[row_index][3].set_yscale("log", base=2)
        for axis in axes[row_index]:
            axis.grid(True, which="both", alpha=0.25)
    for axis, title in zip(axes[0], ("Normalized physical QPS", "Actual fan-out", "Local distance work/shard", "Observed and modeled aggregate work"), strict=True):
        axis.set_title(title)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper center", ncols=2)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    outputs["c1_combined_motivation"] = save_pdf(fig, figures / "c1_combined_motivation.pdf")
    plt.close(fig)

    aliases = {
        "c1_fig1_physical_scaleout": "fig_c1_physical_scaleout.pdf",
        "c1_fig2_fanout_vs_logical_shards": "fig_c1_fanout_vs_shards.pdf",
        "c1_fig3_required_fanout_cdf": "fig_c1_oracle_fanout_cdf.pdf",
        "c1_fig4_local_search_work": "fig_c1_local_search_work.pdf",
        "c1_fig5_aggregate_work_decomposition": "fig_c1_aggregate_work.pdf",
        "c1_fig6_projected_logical_scaling": "fig_c1_projected_scaling.pdf",
    }
    for source_name, alias in aliases.items():
        copy_atomic(outputs[source_name], figures / alias)
    return outputs


def latex_table(
    physical: list[dict[str, str]], aggregate: list[dict[str, str]]
) -> str:
    aggregate_index = {
        (r["dataset"], r["partition_method"], int(r["logical_shard_count"])): r
        for r in aggregate
    }
    lines = [
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r"Dataset & Partition & Workers & Recall & QPS & Efficiency & Fan-out & Dist./query \\",
        r"\midrule",
    ]
    for row in physical:
        key = (
            row["dataset"],
            row["partition_method"],
            int(row["physical_machine_count"]),
        )
        work = aggregate_index[key]
        lines.append(
            f"{LABELS[row['dataset']]} & {METHOD_LABELS[row['partition_method']]} & "
            f"{row['physical_machine_count']} & {float(row['achieved_recall_mean']):.4f} & "
            f"{float(row['qps_mean']):.1f} & {float(row['scaling_efficiency']):.3f} & "
            f"{float(work['actual_fanout_mean']):.1f} & "
            f"{float(work['observed_distance_mean']):.1f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    return "\n".join(lines)


def execute(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.root).expanduser().resolve()
    runs = root / "runs"
    figures = root / "figures"
    if not args.record_stem or Path(args.record_stem).name != args.record_stem:
        raise ValueError("record stem must be one filename component")
    e2_source_stem = source_component(args.e2_source_stem, "E2")
    e3_source_stem = source_component(args.e3_source_stem, "E3")
    e4_sift_stem = source_component(args.e4_sift_stem, "SIFT E4")
    e4_glove_stem = source_component(args.e4_glove_stem, "GloVe E4")
    superseded_record_name = source_component(args.superseded_record, "superseded record")
    section31_proof_name = source_component(args.section31_proof, "Section 31 proof")
    historical_record = runs / superseded_record_name
    if not historical_record.is_file():
        raise FileNotFoundError(historical_record)
    historical_payload = load_json(historical_record)
    historical_record_sha256 = sha256_path(historical_record)
    section31_path = runs / section31_proof_name
    section31 = validate_section31_proof(section31_path)
    graph_build_seed = int(section31["graph_build_seed"])
    e4_stems = {
        "sift1m": e4_sift_stem,
        "glove-200-angular": e4_glove_stem,
    }
    e4_records: dict[str, dict[str, Any]] = {}
    for dataset, stem in e4_stems.items():
        record_path = runs / f"{stem}-aggregate-work-record.json"
        record = load_json(record_path)
        if (
            record.get("dataset") != dataset
            or record.get("deterministic_graph_construction") is not True
            or int(record.get("graph_build_seed") or -1) != graph_build_seed
            or int(record.get("hnsw_max_indexing_threads") or 0) != 1
            or int(record.get("max_optimization_threads") or 0) != 1
            or record.get("e2_source_stem") != e2_source_stem
            or record.get("e3_source_stem") != e3_source_stem
        ):
            raise ValueError(
                f"E4 source is not authoritative deterministic evidence: {record_path}"
            )
        e4_records[dataset] = record
    dataset_index = {dataset: index for index, dataset in enumerate(DATASETS)}
    method_index = {method: index for index, method in enumerate(METHODS)}
    physical_output, physical = combine_csvs(
        [
            runs / "stage6-corrected-sift1m-c1_physical_scaleout.csv",
            runs / "stage8-corrected-glove-200-angular-c1_physical_scaleout.csv",
        ],
        runs / "c1_physical_scaleout.csv",
        sort_key=lambda row: (
            dataset_index[row["dataset"]],
            method_index[row["partition_method"]],
            int(row["physical_machine_count"]),
        ),
    )
    fanout_output, fanout = combine_csvs(
        [
            runs / f"{e2_source_stem}-sift1m-c1_fanout_summary.csv",
            runs / f"{e2_source_stem}-glove-200-angular-c1_fanout_summary.csv",
        ],
        runs / "c1_fanout_summary.csv",
        sort_key=lambda row: (
            dataset_index[row["dataset"]],
            method_index[row["partition_method"]],
            int(row["logical_shard_count"]),
        ),
    )
    aggregate_output, aggregate = combine_csvs(
        [
            runs / f"{e4_sift_stem}-c1_aggregate_work_summary.csv",
            runs / f"{e4_glove_stem}-c1_aggregate_work_summary.csv",
        ],
        runs / "c1_aggregate_work_summary.csv",
        sort_key=lambda row: (
            dataset_index[row["dataset"]],
            method_index[row["partition_method"]],
            int(row["logical_shard_count"]),
        ),
    )
    model_output, model = combine_csvs(
        [
            runs / f"{e4_sift_stem}-c1_model_accuracy.csv",
            runs / f"{e4_glove_stem}-c1_model_accuracy.csv",
        ],
        runs / "c1_model_accuracy.csv",
        sort_key=lambda row: (
            dataset_index[row["dataset"]], method_index[row["partition_method"]]
        ),
    )
    local = build_local_work_rows(
        runs,
        e3_source_stem=e3_source_stem,
        graph_build_seed=graph_build_seed,
    )
    local_output = write_csv(runs / "c1_local_work_summary.csv", LOCAL_FIELDS, local)
    sensitivity, sensitivity_details = build_sensitivity_rows(
        runs,
        e2_source_stem=e2_source_stem,
        e3_source_stem=e3_source_stem,
        graph_build_seed=graph_build_seed,
    )
    sensitivity_fields = list(sensitivity[0])
    sensitivity_output = write_csv(
        runs / "c1_recall_sensitivity.csv", sensitivity_fields, sensitivity
    )
    mechanism_checks = evaluate_final_mechanisms(fanout, local, e4_records)
    figure_outputs = generate_figures(figures, physical, fanout, local, aggregate)
    latex_output = atomic_write_text(
        runs / "c1_physical_scale_table.tex", latex_table(physical, aggregate)
    )
    timestamp = utc_timestamp()
    dataset_projection_status = {
        dataset: record["dataset_m_gt_4_projection_status"]
        for dataset, record in e4_records.items()
    }
    cross_dataset_projection_statuses = {
        str(record["m_gt_4_projection_status"])
        for record in e4_records.values()
    }
    if len(cross_dataset_projection_statuses) != 1:
        raise ValueError(
            "final E4 records disagree on the cross-dataset projection status"
        )
    cross_dataset_projection_status = cross_dataset_projection_statuses.pop()
    c1_c_status = (
        "SUPPORTED"
        if mechanism_checks["slow_local_work_supported"]
        else "CONTRADICTED"
    )
    e5_status = (
        "OBSERVED_TREND_STABLE"
        if sensitivity_details["all_trends_preserved"]
        else "CONTRADICTED_QUALITATIVE_TREND_REVERSAL"
    )
    anomalies = [
        "sift1m_random_work_bound_projection_anticorrelates_with_measured_e1",
        "cross_dataset_m_gt_4_throughput_projection_withheld",
        "sift1m_kmeans_fanout_is_nearly_constant_and_small",
        "glove_random_m4_automatic_c32_formal_repetitions_exceeded_client_cpu_gate",
        "kmeans_partitions_reached_iteration_cap_without_convergence",
    ]
    if not sensitivity_details["all_trends_preserved"]:
        anomalies.append("e5_recall_sensitivity_contains_qualitative_trend_reversal")
    summary = {
        "timestamp": timestamp,
        "record_type": "c1_cross_dataset_deterministic_final_summary",
        "experiment_id": args.record_stem,
        "git_commit": repository_commit(Path(__file__).resolve().parents[3]),
        "datasets": list(DATASETS),
        "target_recall": 0.90,
        "subclaim_status": {
            "c1_a_physical_sublinear_scaling": "SUPPORTED",
            "c1_b_high_logical_shard_fanout": "CONTRADICTED",
            "c1_c_slow_local_work_decrease": c1_c_status,
            "c1_d_fanout_times_local_work_decomposition": "INSUFFICIENT",
        },
        "observed_mechanism_status": {
            "random_fanout": "SUPPORTED",
            "kmeans_fanout_sift1m": "CONTRADICTED_NEARLY_CONSTANT_SMALL",
            "kmeans_fanout_glove_200_angular": "SUPPORTED",
            "slow_local_work_decrease": c1_c_status,
            "aggregate_decomposition_identity": "SUPPORTED",
            "physical_throughput_attribution": "INSUFFICIENT",
        },
        "c1_status": "INSUFFICIENT",
        "e5_recall_sensitivity_status": e5_status,
        "e5_checks": sensitivity_details["checks"],
        "e5_contradicted_checks": sensitivity_details["contradicted_checks"],
        "mechanism_checks": mechanism_checks,
        "dataset_projection_status": dataset_projection_status,
        "cross_dataset_m_gt_4_projection_status": cross_dataset_projection_status,
        "result_scope": {
            "m_1_2_4": "measured_physical_scaleout",
            "m_8_16_32": "logical_shard_mechanism_measurements_only",
            "m_gt_4_throughput_projection": "withheld",
        },
        "section31_index_construction_status": (
            "VERIFIED_DETERMINISTIC_SINGLE_BUILD_AUTHORITATIVE_RERUN"
        ),
        "authoritative_source_stems": {
            "e2": e2_source_stem,
            "e3": e3_source_stem,
            "e4": e4_stems,
        },
        "graph_build_seed": graph_build_seed,
        "section31_proof_sha256": sha256_path(section31_path),
        "paper_ready_claim": None,
        "supersedes": {
            "experiment_id": historical_payload["experiment_id"],
            "record_path": str(historical_record.resolve()),
            "record_sha256": historical_record_sha256,
            "reason": "section31_deterministic_e2_e3_e4_authoritative_rerun",
        },
        "anomalies": anomalies,
        "status": "C1_PROTOCOL_COMPLETE_RESULT_INSUFFICIENT",
    }
    summary_output = write_json(runs / f"{args.record_stem}-summary.json", summary)
    evidence_paths = {
        "physical_scaleout_csv": physical_output,
        "fanout_summary_csv": fanout_output,
        "local_work_summary_csv": local_output,
        "aggregate_work_summary_csv": aggregate_output,
        "model_accuracy_csv": model_output,
        "recall_sensitivity_csv": sensitivity_output,
        "physical_scale_latex": latex_output,
        "section31_deterministic_build_proof": section31_path,
        "summary_json": summary_output,
        **figure_outputs,
    }
    record = {
        key: summary[key]
        for key in (
            "timestamp",
            "record_type",
            "experiment_id",
            "git_commit",
            "datasets",
            "target_recall",
            "subclaim_status",
            "observed_mechanism_status",
            "c1_status",
            "e5_recall_sensitivity_status",
            "e5_checks",
            "e5_contradicted_checks",
            "mechanism_checks",
            "dataset_projection_status",
            "cross_dataset_m_gt_4_projection_status",
            "result_scope",
            "section31_index_construction_status",
            "authoritative_source_stems",
            "graph_build_seed",
            "section31_proof_sha256",
            "supersedes",
            "anomalies",
            "status",
        )
    }
    record["configuration_counts"] = {
        "physical_e1": len(physical),
        "fanout_e2": len(fanout),
        "local_work_e3": len(local),
        "aggregate_work_e4": len(aggregate),
        "recall_sensitivity_e5": len(sensitivity),
    }
    record["evidence_sha256"] = {
        name: sha256_path(path) for name, path in evidence_paths.items()
    }
    record["sensitivity_source_sha256"] = sensitivity_details["source_sha256"]
    record_output = write_json(runs / f"{args.record_stem}-record.json", record)
    register_manifest_record_once(runs / "manifest.jsonl", record)
    result = {
        "summary_output": str(summary_output),
        "record_output": str(record_output),
        "outputs": {name: str(path) for name, path in evidence_paths.items()},
        "record": record,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Finalize cross-dataset C1 outputs")
    parser.add_argument("--root", default="experiments/c1")
    parser.add_argument("--e2-source-stem", default="stage12-e2-deterministic")
    parser.add_argument("--e3-source-stem", default="stage12-e3-deterministic")
    parser.add_argument(
        "--e4-sift-stem", default="stage12-e4-deterministic-sift1m-final"
    )
    parser.add_argument(
        "--e4-glove-stem", default="stage12-e4-deterministic-glove-200-angular"
    )
    parser.add_argument(
        "--section31-proof", default="section31-deterministic-build-proof.json"
    )
    parser.add_argument(
        "--superseded-record", default="stage11-c1-report-repair-record.json"
    )
    parser.add_argument("--record-stem", default="stage12-c1-deterministic-final")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    execute(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
