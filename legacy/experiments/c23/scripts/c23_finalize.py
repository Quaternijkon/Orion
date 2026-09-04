#!/usr/bin/env python3
"""Generate the final two-dataset C23 tables, figures, verdicts, and audit."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.c23.scripts import c23_e1, c23_protocol


DATASETS = ("sift1m", "glove-200-angular")
LOGICAL_SHARDS = (2, 4, 8, 16, 32)
METHODS = ("random", "kmeans", "orion", "orion_no_refinement")
METHOD_LABELS = {
    "random": "Random",
    "kmeans": "K-Means",
    "orion": "Orion",
    "orion_no_refinement": "Orion-NoRefinement",
}
METHOD_COLORS = {
    "random": "#777777",
    "kmeans": "#d97706",
    "orion": "#2563eb",
    "orion_no_refinement": "#059669",
}
METHOD_MARKERS = {
    "random": "o",
    "kmeans": "s",
    "orion": "^",
    "orion_no_refinement": "D",
}
DATASET_LABELS = {"sift1m": "SIFT1M", "glove-200-angular": "GloVe-200"}
DATASET_INFO = {
    "sift1m": {
        "dimension": 128,
        "distance_metric": "euclid",
        "common_ef": 24,
    },
    "glove-200-angular": {
        "dimension": 200,
        "distance_metric": "cosine",
        "common_ef": 320,
    },
}

REQUIRED_TABLES = (
    "c23_topology_summary.csv",
    "c23_local_navigability_summary.csv",
    "c23_exact_fanout_summary.csv",
    "c23_hnsw_fanout_summary.csv",
    "c23_fanout_decomposition.csv",
    "c23_partition_balance.csv",
    "c23_correlation_analysis.csv",
    "c23_construction_overhead.csv",
)
REQUIRED_FIGURES = (
    "c23_fig1_traversal_weighted_edge_cut.pdf",
    "c23_fig2_local_navigability.pdf",
    "c23_fig3_exact_fanout.pdf",
    "c23_fig4_hnsw_fanout.pdf",
    "c23_fig5_fanout_decomposition.pdf",
    "c23_fig6_work_to_90_recall.pdf",
    "c23_fig7_partition_balance.pdf",
    "c23_combined_causal_chain.pdf",
)

RUN_SUMMARY_FIELDS = (
    "experiment_id",
    "timestamp",
    "git_commit",
    "dataset",
    "dataset_checksum",
    "partition_method",
    "partition_seed",
    "logical_shards",
    "physical_hosts",
    "logical_to_physical_mapping",
    "vector_count",
    "dimension",
    "distance_metric",
    "HNSW_M",
    "HNSW_efConstruction",
    "efSearch",
    "navigation_sample_size",
    "navigation_sample_rate",
    "k_nav",
    "shard_size_mean",
    "shard_size_std",
    "shard_size_min",
    "shard_size_max",
    "edge_cut_ratio",
    "traversal_weighted_edge_cut",
    "mean_retained_degree_ratio",
    "largest_component_fraction",
    "mean_path_shards",
    "mean_path_transitions",
    "mean_local_target_recall",
    "mean_local_distance_computations",
    "P_exact_mean",
    "P_exact_p95",
    "P_HNSW_mean",
    "P_HNSW_p95",
    "Delta_P_mean",
    "W90_mean",
    "W90_p95",
)


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).expanduser().resolve().open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        return list(csv.DictReader(handle))


def write_csv_new(path: str | Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    target = Path(path).expanduser().resolve()
    if target.exists():
        raise FileExistsError(f"refusing to overwrite: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json_new(path: str | Path, value: Any) -> None:
    target = Path(path).expanduser().resolve()
    if target.exists():
        raise FileExistsError(f"refusing to overwrite: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_run_summary(path: str | Path, rows: list[dict[str, Any]]) -> None:
    target = Path(path).expanduser().resolve()
    if target.exists():
        existing_lines = target.read_text(encoding="utf-8").splitlines()
        if len(existing_lines) != 1:
            raise FileExistsError(
                f"refusing to replace populated run summary: {target}"
            )
        if existing_lines[0].split(",") != list(RUN_SUMMARY_FIELDS):
            raise ValueError("existing run-summary header differs from protocol schema")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RUN_SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def append_jsonl(path: str | Path, value: Any) -> None:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True, separators=(",", ":")))
        handle.write("\n")


def f(row: dict[str, Any], name: str) -> float:
    value = row[name]
    if value in (None, ""):
        return float("nan")
    return float(value)


def i(row: dict[str, Any], name: str) -> int:
    return int(float(row[name]))


def config_key(row: dict[str, Any]) -> tuple[str, str, int]:
    dataset = str(row["dataset"])
    method = str(row.get("partition_method", row.get("method")))
    return dataset, method, i(row, "logical_shards")


def load_artifact_metadata(path: str | Path) -> dict[str, Any]:
    with np.load(Path(path).expanduser().resolve(), allow_pickle=False) as archive:
        return json.loads(str(np.asarray(archive["metadata_json"]).item()))


def method_rows(
    rows: Iterable[dict[str, Any]], dataset: str, method: str
) -> list[dict[str, Any]]:
    return sorted(
        [
            row
            for row in rows
            if row["dataset"] == dataset and row["partition_method"] == method
        ],
        key=lambda row: int(row["logical_shards"]),
    )


def configure_axis(ax: plt.Axes, title: str, ylabel: str, xlabel: str = "Logical shards") -> None:
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    if xlabel == "Logical shards":
        ax.set_xticks(LOGICAL_SHARDS)
        ax.set_xscale("log", base=2)


def save_figure_new(fig: plt.Figure, path: str | Path) -> None:
    target = Path(path).expanduser().resolve()
    if target.exists():
        raise FileExistsError(f"refusing to overwrite: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, format="pdf", bbox_inches="tight")
    plt.close(fig)


def plot_lines(
    ax: plt.Axes,
    rows: list[dict[str, Any]],
    dataset: str,
    metric: str,
    *,
    methods: Iterable[str] = METHODS,
) -> None:
    for method in methods:
        selected = method_rows(rows, dataset, method)
        ax.plot(
            [int(row["logical_shards"]) for row in selected],
            [float(row[metric]) for row in selected],
            label=METHOD_LABELS[method],
            color=METHOD_COLORS[method],
            marker=METHOD_MARKERS[method],
            linewidth=1.8,
            markersize=4.5,
        )


def build_tables(
    analysis_root: Path, construction_csv: Path
) -> dict[str, list[dict[str, Any]]]:
    e1_rows: list[dict[str, str]] = []
    e2_rows: list[dict[str, str]] = []
    e3_rows: list[dict[str, str]] = []
    fixed_rows: list[dict[str, str]] = []
    e4_rows: list[dict[str, str]] = []
    participation_rows: list[dict[str, str]] = []
    bootstrap_rows: list[dict[str, str]] = []
    for dataset in DATASETS:
        dataset_root = analysis_root / dataset
        e1_rows.extend(read_csv(dataset_root / "e1-v1/e1_metrics.csv"))
        e2_rows.extend(read_csv(dataset_root / "e2-v1/e2_summary.csv"))
        e3_rows.extend(read_csv(dataset_root / "local-v1/e3_summary.csv"))
        fixed_rows.extend(read_csv(dataset_root / "local-v1/e3_fixed_recall.csv"))
        e4_rows.extend(read_csv(dataset_root / "local-v1/e4_summary.csv"))
        participation_rows.extend(
            read_csv(dataset_root / "local-v1/query_participation_by_shard.csv")
        )
        bootstrap_rows.extend(
            read_csv(dataset_root / "bootstrap-v1/paired_bootstrap_ci.csv")
        )

    e1 = {config_key(row): row for row in e1_rows}
    e2 = {config_key(row): row for row in e2_rows}
    e4 = {config_key(row): row for row in e4_rows}
    expected_keys = {
        (dataset, method, shards)
        for dataset in DATASETS
        for method in METHODS
        for shards in LOGICAL_SHARDS
    }
    if set(e1) != expected_keys or set(e2) != expected_keys or set(e4) != expected_keys:
        raise ValueError("E1/E2/E4 configuration coverage is not exactly 40 rows")

    common_e3: dict[tuple[str, str, int], dict[str, str]] = {}
    any_e3: dict[tuple[str, str, int], dict[str, str]] = {}
    for row in e3_rows:
        key = config_key(row)
        any_e3[key] = row
        if i(row, "ef_search") == DATASET_INFO[key[0]]["common_ef"]:
            common_e3[key] = row
    fixed: dict[tuple[str, str, int, str], dict[str, str]] = {}
    for row in fixed_rows:
        threshold = f"{float(row['local_recall_threshold']):.2f}"
        fixed[(*config_key(row), threshold)] = row

    topology: list[dict[str, Any]] = []
    local: list[dict[str, Any]] = []
    exact: list[dict[str, Any]] = []
    hnsw: list[dict[str, Any]] = []
    decomposition: list[dict[str, Any]] = []
    causal: list[dict[str, Any]] = []
    for key in sorted(expected_keys, key=lambda x: (x[0], METHODS.index(x[1]), x[2])):
        e1_row, e2_row, e4_row = e1[key], e2[key], e4[key]
        e3_row = common_e3.get(key)
        dataset, method, shards = key
        topology.append(
            {
                "dataset": dataset,
                "partition_method": method,
                "partition_seed": i(e1_row, "partition_seed"),
                "logical_shards": shards,
                "point_count": i(e1_row, "point_count"),
                "edge_count": i(e1_row, "edge_count"),
                "edge_cut_ratio": f(e1_row, "edge_cut_ratio"),
                "traversal_weighted_edge_cut": f(
                    e2_row, "traversal_weighted_edge_cut"
                ),
                "mean_intra_shard_degree": f(e1_row, "mean_intra_shard_degree"),
                "median_intra_shard_degree": f(
                    e1_row, "median_intra_shard_degree"
                ),
                "p10_intra_shard_degree": f(e1_row, "p10_intra_shard_degree"),
                "mean_retained_degree_ratio": f(
                    e1_row, "mean_retained_degree_ratio"
                ),
                "fraction_losing_gt_25pct_neighbors": f(
                    e1_row, "fraction_losing_gt_25pct_neighbors"
                ),
                "fraction_losing_gt_50pct_neighbors": f(
                    e1_row, "fraction_losing_gt_50pct_neighbors"
                ),
                "fraction_losing_gt_75pct_neighbors": f(
                    e1_row, "fraction_losing_gt_75pct_neighbors"
                ),
                "largest_component_fraction_unweighted_shard_mean": f(
                    e1_row, "largest_component_fraction_unweighted_shard_mean"
                ),
                "largest_component_fraction_vector_weighted_mean": f(
                    e1_row, "largest_component_fraction_vector_weighted_mean"
                ),
                "isolated_node_fraction_vector_weighted_mean": f(
                    e1_row, "isolated_node_fraction_vector_weighted_mean"
                ),
                "path_shards_mean": f(e2_row, "path_shards_mean"),
                "path_transitions_mean": f(e2_row, "path_transitions_mean"),
                "dominant_shard_concentration_mean": f(
                    e2_row, "dominant_shard_concentration_mean"
                ),
                "partition_artifact_path": e1_row["partition_artifact_path"],
                "partition_artifact_sha256": e1_row[
                    "partition_artifact_sha256"
                ],
            }
        )
        local_row: dict[str, Any] = {
            "dataset": dataset,
            "partition_method": method,
            "partition_seed": i(e4_row, "partition_seed"),
            "logical_shards": shards,
            "common_ef_search": i(e4_row, "common_ef_search"),
            "query_shard_pair_count": i(any_e3[key], "query_shard_pair_count"),
            "local_target_recall_mean": f(
                e4_row, "common_ef_local_target_recall_mean"
            ),
            "local_target_recall_median": f(
                e4_row, "common_ef_local_target_recall_median"
            ),
            "local_target_recall_p95": f(
                e4_row, "common_ef_local_target_recall_p95"
            ),
            "local_target_recall_std": f(
                e4_row, "common_ef_local_target_recall_std"
            ),
            "distance_computations_mean": f(
                e4_row, "common_ef_local_distance_computations_mean"
            ),
            "distance_computations_median": f(
                e4_row, "common_ef_local_distance_computations_median"
            ),
            "distance_computations_p95": f(
                e4_row, "common_ef_local_distance_computations_p95"
            ),
            "distance_computations_std": f(
                e4_row, "common_ef_local_distance_computations_std"
            ),
            "graph_nodes_visited_mean": ""
            if e3_row is None
            else f(e3_row, "graph_nodes_visited_mean"),
            "local_latency_us_mean_secondary": ""
            if e3_row is None
            else f(e3_row, "local_latency_us_mean"),
        }
        for threshold in ("0.80", "0.90", "0.95"):
            row = fixed[(*key, threshold)]
            suffix = threshold.replace("0.", "")
            local_row[f"fixed_recall_{suffix}_success_fraction"] = f(
                row, "success_fraction"
            )
            local_row[f"fixed_recall_{suffix}_required_ef_search_mean"] = f(
                row, "required_ef_search_mean"
            )
            local_row[
                f"fixed_recall_{suffix}_required_distance_computations_mean"
            ] = f(row, "required_distance_computations_mean")
            local_row[
                f"fixed_recall_{suffix}_required_graph_nodes_visited_mean"
            ] = f(row, "required_graph_nodes_visited_mean")
        local.append(local_row)
        exact.append(
            {
                "dataset": dataset,
                "partition_method": method,
                "partition_seed": i(e4_row, "partition_seed"),
                "logical_shards": shards,
                "measurement_query_count": i(e4_row, "measurement_query_count"),
                "P_exact_mean": f(e4_row, "P_exact_mean"),
                "P_exact_median": f(e4_row, "P_exact_median"),
                "P_exact_p95": f(e4_row, "P_exact_p95"),
                "P_exact_std": f(e4_row, "P_exact_std"),
            }
        )
        hnsw.append(
            {
                "dataset": dataset,
                "partition_method": method,
                "partition_seed": i(e4_row, "partition_seed"),
                "logical_shards": shards,
                "measurement_query_count": i(e4_row, "measurement_query_count"),
                "common_ef_search": i(e4_row, "common_ef_search"),
                "P_HNSW_mean": f(e4_row, "P_HNSW_mean"),
                "P_HNSW_median": f(e4_row, "P_HNSW_median"),
                "P_HNSW_p95": f(e4_row, "P_HNSW_p95"),
                "P_HNSW_std": f(e4_row, "P_HNSW_std"),
                "hnsw_reached_90_fraction": f(
                    e4_row, "hnsw_reached_90_fraction"
                ),
            }
        )
        decomposition.append(
            {
                "dataset": dataset,
                "partition_method": method,
                "partition_seed": i(e4_row, "partition_seed"),
                "logical_shards": shards,
                "P_exact_mean": f(e4_row, "P_exact_mean"),
                "P_HNSW_mean": f(e4_row, "P_HNSW_mean"),
                "Delta_P_mean": f(e4_row, "Delta_P_mean"),
                "Delta_P_median": f(e4_row, "Delta_P_median"),
                "Delta_P_p95": f(e4_row, "Delta_P_p95"),
                "Delta_P_std": f(e4_row, "Delta_P_std"),
                "W90_mean": f(e4_row, "W90_mean"),
                "W90_median": f(e4_row, "W90_median"),
                "W90_p95": f(e4_row, "W90_p95"),
                "W90_std": f(e4_row, "W90_std"),
                "hnsw_reached_90_fraction": f(
                    e4_row, "hnsw_reached_90_fraction"
                ),
            }
        )
        causal.append(
            {
                "dataset": dataset,
                "partition_method": method,
                "logical_shards": shards,
                "edge_cut_ratio": f(e1_row, "edge_cut_ratio"),
                "traversal_weighted_cut": f(
                    e2_row, "traversal_weighted_edge_cut"
                ),
                "retained_neighbor_degree": f(
                    e1_row, "mean_intra_shard_degree"
                ),
                "retained_neighbor_degree_ratio": f(
                    e1_row, "mean_retained_degree_ratio"
                ),
                "local_target_recall": f(
                    e4_row, "common_ef_local_target_recall_mean"
                ),
                "distance_computations_at_fixed_local_recall": local_row[
                    "fixed_recall_90_required_distance_computations_mean"
                ],
                "distance_computations_at_fixed_local_recall_080": local_row[
                    "fixed_recall_80_required_distance_computations_mean"
                ],
                "distance_computations_at_fixed_local_recall_090": local_row[
                    "fixed_recall_90_required_distance_computations_mean"
                ],
                "distance_computations_at_fixed_local_recall_095": local_row[
                    "fixed_recall_95_required_distance_computations_mean"
                ],
                "P_exact": f(e4_row, "P_exact_mean"),
                "P_HNSW": f(e4_row, "P_HNSW_mean"),
                "Delta_P": f(e4_row, "Delta_P_mean"),
                "W90": f(e4_row, "W90_mean"),
            }
        )

    e1_by_key = {config_key(row): row for row in topology}
    balance: list[dict[str, Any]] = []
    for row in participation_rows:
        key = config_key(row)
        config = e1_by_key[key]
        mean_size = float(config["point_count"]) / key[2]
        std_size = f(e1[key], "shard_size_std")
        shard_id = i(row, "shard_id")
        balance.append(
            {
                "dataset": key[0],
                "partition_method": key[1],
                "partition_seed": i(row, "partition_seed"),
                "logical_shards": key[2],
                "physical_hosts": 4,
                "logical_to_physical_rule": "logical_shard_id modulo 4",
                "shard_id": shard_id,
                "physical_host_id": shard_id % 4,
                "vector_count": i(row, "vector_count"),
                "shard_size_min": i(e1[key], "shard_size_min"),
                "shard_size_max": i(e1[key], "shard_size_max"),
                "shard_size_mean": mean_size,
                "shard_size_std": std_size,
                "shard_size_coefficient_of_variation": std_size / mean_size,
                "max_size_over_mean": f(e1[key], "max_shard_size_over_mean"),
                "query_participation_count": i(row, "query_participation_count"),
                "query_participation_fraction": f(
                    row, "query_participation_fraction"
                ),
            }
        )

    correlation_relations = [
        (
            "topology_disruption_vs_local_target_recall",
            "traversal_weighted_cut",
            "local_target_recall",
            "negative",
        ),
        (
            "topology_disruption_vs_fixed_recall_work_080",
            "traversal_weighted_cut",
            "distance_computations_at_fixed_local_recall_080",
            "positive",
        ),
        (
            "topology_disruption_vs_fixed_recall_work_090",
            "traversal_weighted_cut",
            "distance_computations_at_fixed_local_recall_090",
            "positive",
        ),
        (
            "topology_disruption_vs_fixed_recall_work_095",
            "traversal_weighted_cut",
            "distance_computations_at_fixed_local_recall_095",
            "positive",
        ),
        (
            "local_target_recall_vs_hnsw_fanout",
            "local_target_recall",
            "P_HNSW",
            "negative",
        ),
        (
            "ann_fanout_penalty_vs_topology_disruption",
            "Delta_P",
            "traversal_weighted_cut",
            "positive",
        ),
    ]
    correlations: list[dict[str, Any]] = []
    for scope in ("pooled", *DATASETS):
        selected = causal if scope == "pooled" else [r for r in causal if r["dataset"] == scope]
        for relation, x_name, y_name, expected_direction in correlation_relations:
            pairs = [
                (float(row[x_name]), float(row[y_name]))
                for row in selected
                if math.isfinite(float(row[x_name])) and math.isfinite(float(row[y_name]))
            ]
            x = np.asarray([pair[0] for pair in pairs], dtype=np.float64)
            y = np.asarray([pair[1] for pair in pairs], dtype=np.float64)
            pearson = pearsonr(x, y)
            spearman = spearmanr(x, y)
            correlations.append(
                {
                    "scope": scope,
                    "relation": relation,
                    "x_metric": x_name,
                    "y_metric": y_name,
                    "expected_direction_under_claim": expected_direction,
                    "configuration_count": len(pairs),
                    "pearson_r": float(pearson.statistic),
                    "pearson_p_value": float(pearson.pvalue),
                    "spearman_rho": float(spearman.statistic),
                    "spearman_p_value": float(spearman.pvalue),
                    "interpretation": "consistency diagnostic only; not formal causality",
                }
            )

    construction = read_csv(construction_csv)
    if len(construction) != 40:
        raise ValueError("construction-overhead input does not contain 40 rows")
    return {
        "topology": topology,
        "local": local,
        "exact": exact,
        "hnsw": hnsw,
        "decomposition": decomposition,
        "balance": balance,
        "correlations": correlations,
        "causal": causal,
        "construction": construction,
        "e3_raw": e3_rows,
        "bootstrap": bootstrap_rows,
    }


def generate_figures(tables: dict[str, list[dict[str, Any]]], figure_dir: Path) -> None:
    topology = tables["topology"]
    local = tables["local"]
    exact = tables["exact"]
    hnsw = tables["hnsw"]
    decomposition = tables["decomposition"]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.1), constrained_layout=True)
    for ax, dataset in zip(axes, DATASETS, strict=True):
        plot_lines(ax, topology, dataset, "traversal_weighted_edge_cut")
        configure_axis(ax, DATASET_LABELS[dataset], "Traversal-weighted edge cut")
    axes[0].legend(fontsize=8)
    save_figure_new(fig, figure_dir / REQUIRED_FIGURES[0])

    fig, axes = plt.subplots(2, 5, figsize=(17, 7.2), constrained_layout=True, sharex=False)
    for row_index, dataset in enumerate(DATASETS):
        for col_index, shards in enumerate(LOGICAL_SHARDS):
            ax = axes[row_index, col_index]
            for method in METHODS:
                selected = sorted(
                    [
                        row
                        for row in tables["e3_raw"]
                        if row["dataset"] == dataset
                        and row["partition_method"] == method
                        and i(row, "logical_shards") == shards
                    ],
                    key=lambda row: i(row, "ef_search"),
                )
                ax.plot(
                    [f(row, "distance_computations_mean") for row in selected],
                    [f(row, "local_target_recall_mean") for row in selected],
                    color=METHOD_COLORS[method],
                    marker=METHOD_MARKERS[method],
                    linewidth=1.3,
                    markersize=3.3,
                    label=METHOD_LABELS[method],
                )
            ax.set_title(f"{DATASET_LABELS[dataset]}, M={shards}", fontsize=9)
            ax.grid(True, alpha=0.25)
            if col_index == 0:
                ax.set_ylabel("Local target recall")
            if row_index == 1:
                ax.set_xlabel("Distance computations")
    axes[0, 0].legend(fontsize=7)
    save_figure_new(fig, figure_dir / REQUIRED_FIGURES[1])

    for filename, rows, metric, ylabel in (
        (REQUIRED_FIGURES[2], exact, "P_exact_mean", "Mean exact oracle fan-out"),
        (REQUIRED_FIGURES[3], hnsw, "P_HNSW_mean", "Mean HNSW oracle fan-out"),
        (REQUIRED_FIGURES[5], decomposition, "W90_mean", "Mean distance computations to 90% recall"),
    ):
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.1), constrained_layout=True)
        for ax, dataset in zip(axes, DATASETS, strict=True):
            plot_lines(ax, rows, dataset, metric)
            configure_axis(ax, DATASET_LABELS[dataset], ylabel)
        axes[0].legend(fontsize=8)
        save_figure_new(fig, figure_dir / filename)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), constrained_layout=True)
    width = 0.18
    x = np.arange(len(LOGICAL_SHARDS), dtype=np.float64)
    for ax, dataset in zip(axes, DATASETS, strict=True):
        for index, method in enumerate(METHODS):
            selected = method_rows(decomposition, dataset, method)
            offsets = x + (index - 1.5) * width
            exact_values = np.asarray([float(row["P_exact_mean"]) for row in selected])
            delta_values = np.asarray([float(row["Delta_P_mean"]) for row in selected])
            ax.bar(
                offsets,
                exact_values,
                width,
                color=METHOD_COLORS[method],
                alpha=0.88,
                label=METHOD_LABELS[method] if dataset == DATASETS[0] else None,
            )
            ax.bar(
                offsets,
                delta_values,
                width,
                bottom=exact_values,
                color=METHOD_COLORS[method],
                alpha=0.35,
                hatch="//",
            )
        ax.set_xticks(x, [str(value) for value in LOGICAL_SHARDS])
        ax.set_xlabel("Logical shards")
        ax.set_ylabel("Mean fan-out")
        ax.set_title(f"{DATASET_LABELS[dataset]}: solid P_exact + hatched Delta_P")
        ax.grid(True, axis="y", alpha=0.25)
    axes[0].legend(fontsize=8, ncol=2)
    save_figure_new(fig, figure_dir / REQUIRED_FIGURES[4])

    balance_configs: list[dict[str, Any]] = []
    for row in topology:
        mean_size = float(row["point_count"]) / int(row["logical_shards"])
        source = next(
            candidate
            for candidate in tables["balance"]
            if config_key(candidate) == config_key(row)
        )
        balance_configs.append(
            {
                "dataset": row["dataset"],
                "partition_method": row["partition_method"],
                "logical_shards": row["logical_shards"],
                "max_size_over_mean": source["max_size_over_mean"],
                "shard_size_cv": source["shard_size_std"] / mean_size,
            }
        )
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5), constrained_layout=True)
    for col, dataset in enumerate(DATASETS):
        for method in METHODS:
            selected = method_rows(balance_configs, dataset, method)
            axes[0, col].plot(
                [int(row["logical_shards"]) for row in selected],
                [float(row["max_size_over_mean"]) for row in selected],
                color=METHOD_COLORS[method],
                marker=METHOD_MARKERS[method],
                label=METHOD_LABELS[method],
            )
            axes[1, col].plot(
                [int(row["logical_shards"]) for row in selected],
                [float(row["shard_size_cv"]) for row in selected],
                color=METHOD_COLORS[method],
                marker=METHOD_MARKERS[method],
            )
        configure_axis(
            axes[0, col], DATASET_LABELS[dataset], "Max shard size / mean"
        )
        configure_axis(
            axes[1, col], DATASET_LABELS[dataset], "Shard-size coefficient of variation"
        )
    axes[0, 0].legend(fontsize=8)
    save_figure_new(fig, figure_dir / REQUIRED_FIGURES[6])

    fig, axes = plt.subplots(2, 4, figsize=(17, 7.7), constrained_layout=True)
    for row_index, dataset in enumerate(DATASETS):
        plot_lines(axes[row_index, 0], topology, dataset, "traversal_weighted_edge_cut")
        configure_axis(axes[row_index, 0], DATASET_LABELS[dataset], "TWCut")
        for method in METHODS:
            selected = method_rows(local, dataset, method)
            axes[row_index, 1].plot(
                [float(row["distance_computations_mean"]) for row in selected],
                [float(row["local_target_recall_mean"]) for row in selected],
                color=METHOD_COLORS[method],
                marker=METHOD_MARKERS[method],
                label=METHOD_LABELS[method],
            )
        configure_axis(
            axes[row_index, 1],
            DATASET_LABELS[dataset],
            "Local target recall",
            "Distance computations at common ef",
        )
        plot_lines(axes[row_index, 2], exact, dataset, "P_exact_mean")
        configure_axis(axes[row_index, 2], DATASET_LABELS[dataset], "P_exact")
        plot_lines(axes[row_index, 3], hnsw, dataset, "P_HNSW_mean")
        configure_axis(axes[row_index, 3], DATASET_LABELS[dataset], "P_HNSW")
    axes[0, 0].legend(fontsize=7)
    save_figure_new(fig, figure_dir / REQUIRED_FIGURES[7])


def build_run_summary(
    tables: dict[str, list[dict[str, Any]]], timestamp: str, git_commit: str
) -> list[dict[str, Any]]:
    topology = {config_key(row): row for row in tables["topology"]}
    local = {config_key(row): row for row in tables["local"]}
    decomposition = {config_key(row): row for row in tables["decomposition"]}
    construction = {config_key(row): row for row in tables["construction"]}
    rows: list[dict[str, Any]] = []
    for key in sorted(topology, key=lambda x: (x[0], METHODS.index(x[1]), x[2])):
        dataset, method, shards = key
        top = topology[key]
        loc = local[key]
        fan = decomposition[key]
        metadata = load_artifact_metadata(top["partition_artifact_path"])
        is_orion = method in ("orion", "orion_no_refinement")
        mapping = {
            "physical_host_count": 4,
            "rule": "logical_shard_id modulo 4",
            "microbenchmark_execution": "one logical shard at a time",
        }
        rows.append(
            {
                "experiment_id": f"c23-20260821-v1-{dataset}-{method}-m{shards}",
                "timestamp": timestamp,
                "git_commit": git_commit,
                "dataset": dataset,
                "dataset_checksum": metadata["dataset_sha256"],
                "partition_method": method,
                "partition_seed": top["partition_seed"],
                "logical_shards": shards,
                "physical_hosts": 4,
                "logical_to_physical_mapping": json.dumps(
                    mapping, separators=(",", ":"), sort_keys=True
                ),
                "vector_count": top["point_count"],
                "dimension": DATASET_INFO[dataset]["dimension"],
                "distance_metric": DATASET_INFO[dataset]["distance_metric"],
                "HNSW_M": 32,
                "HNSW_efConstruction": 200,
                "efSearch": DATASET_INFO[dataset]["common_ef"],
                "navigation_sample_size": metadata.get("navigation_sample_size", "")
                if is_orion
                else "",
                "navigation_sample_rate": metadata.get("navigation_sample_rate", "")
                if is_orion
                else "",
                "k_nav": metadata.get("k_nav", "") if is_orion else "",
                "shard_size_mean": float(top["point_count"]) / shards,
                "shard_size_std": next(
                    float(row["shard_size_std"])
                    for row in tables["balance"]
                    if config_key(row) == key
                ),
                "shard_size_min": min(
                    int(row["vector_count"])
                    for row in tables["balance"]
                    if config_key(row) == key
                ),
                "shard_size_max": max(
                    int(row["vector_count"])
                    for row in tables["balance"]
                    if config_key(row) == key
                ),
                "edge_cut_ratio": top["edge_cut_ratio"],
                "traversal_weighted_edge_cut": top[
                    "traversal_weighted_edge_cut"
                ],
                "mean_retained_degree_ratio": top["mean_retained_degree_ratio"],
                "largest_component_fraction": top[
                    "largest_component_fraction_vector_weighted_mean"
                ],
                "mean_path_shards": top["path_shards_mean"],
                "mean_path_transitions": top["path_transitions_mean"],
                "mean_local_target_recall": loc["local_target_recall_mean"],
                "mean_local_distance_computations": loc[
                    "distance_computations_mean"
                ],
                "P_exact_mean": fan["P_exact_mean"],
                "P_exact_p95": next(
                    row["P_exact_p95"]
                    for row in tables["exact"]
                    if config_key(row) == key
                ),
                "P_HNSW_mean": fan["P_HNSW_mean"],
                "P_HNSW_p95": next(
                    row["P_HNSW_p95"]
                    for row in tables["hnsw"]
                    if config_key(row) == key
                ),
                "Delta_P_mean": fan["Delta_P_mean"],
                "W90_mean": fan["W90_mean"],
                "W90_p95": fan["W90_p95"],
            }
        )
        if key not in construction:
            raise ValueError(f"construction row missing for {key}")
    return rows


def build_verdicts(tables: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    topology = {config_key(row): row for row in tables["topology"]}
    local = {config_key(row): row for row in tables["local"]}
    fanout = {config_key(row): row for row in tables["decomposition"]}
    dataset_results: dict[str, Any] = {}
    for dataset in DATASETS:
        twcut_deltas = {
            str(shards): topology[(dataset, "orion", shards)][
                "traversal_weighted_edge_cut"
            ]
            - topology[(dataset, "kmeans", shards)]["traversal_weighted_edge_cut"]
            for shards in LOGICAL_SHARDS
        }
        recall_deltas = {
            str(shards): local[(dataset, "orion", shards)][
                "local_target_recall_mean"
            ]
            - local[(dataset, "kmeans", shards)]["local_target_recall_mean"]
            for shards in LOGICAL_SHARDS
        }
        hnsw_deltas = {
            str(shards): fanout[(dataset, "orion", shards)]["P_HNSW_mean"]
            - fanout[(dataset, "kmeans", shards)]["P_HNSW_mean"]
            for shards in LOGICAL_SHARDS
        }
        exact_deltas = {
            str(shards): fanout[(dataset, "orion", shards)]["P_exact_mean"]
            - fanout[(dataset, "kmeans", shards)]["P_exact_mean"]
            for shards in LOGICAL_SHARDS
        }
        w90_deltas = {
            str(shards): fanout[(dataset, "orion", shards)]["W90_mean"]
            - fanout[(dataset, "kmeans", shards)]["W90_mean"]
            for shards in LOGICAL_SHARDS
        }
        bootstrap_hnsw = [
            row
            for row in tables["bootstrap"]
            if row["dataset"] == dataset and row["metric"] == "P_HNSW"
        ]
        dataset_results[dataset] = {
            "C2_status": "CONTRADICTED",
            "C3_status": "CONTRADICTED",
            "orion_minus_kmeans_twcut": twcut_deltas,
            "orion_minus_kmeans_common_ef_local_target_recall": recall_deltas,
            "orion_minus_kmeans_P_exact": exact_deltas,
            "orion_minus_kmeans_P_HNSW": hnsw_deltas,
            "orion_minus_kmeans_W90": w90_deltas,
            "P_HNSW_bootstrap_conclusions": {
                str(i(row, "logical_shards")): row["conclusion"]
                for row in bootstrap_hnsw
            },
        }
    return {
        "dataset_verdicts": dataset_results,
        "final_verdicts": {
            "C2": {
                "status": "CONTRADICTED",
                "reason": (
                    "K-Means has lower traversal-weighted cut than Full Orion at "
                    "M=4,8,16,32 on both datasets, while Random combines extreme "
                    "cut with competitive local navigability; the proposed monotonic "
                    "topology-to-navigability mechanism does not hold."
                ),
            },
            "C3": {
                "status": "CONTRADICTED",
                "reason": (
                    "Full Orion requires higher mean HNSW oracle fan-out than K-Means "
                    "throughout the evaluated range on GloVe and at M=2,4,8,16,32 "
                    "on SIFT (M=2 is bootstrap-inconclusive on SIFT); it never provides "
                    "the claimed meaningful fan-out reduction."
                ),
            },
        },
        "ablation_conclusion": (
            "Topology refinement consistently lowers TWCut relative to "
            "Orion-NoRefinement, but does not establish downstream fan-out superiority "
            "over K-Means; the required two-part ablation question is answered NO."
        ),
    }


def all_checks_pass(path: Path) -> bool:
    value = json.loads(path.read_text(encoding="utf-8"))
    statuses: list[str] = []

    def visit(item: Any) -> None:
        if isinstance(item, dict):
            for key, child in item.items():
                if key == "checks" and isinstance(child, dict):
                    statuses.extend(
                        status for status in child.values() if isinstance(status, str)
                    )
                visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)

    visit(value)
    return bool(statuses) and all(status == "PASS" for status in statuses)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-root", required=True)
    parser.add_argument("--construction-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--table-dir", required=True)
    parser.add_argument("--figure-dir", required=True)
    parser.add_argument("--run-summary", required=True)
    parser.add_argument("--run-manifest-jsonl", required=True)
    parser.add_argument("--audit-output", required=True)
    args = parser.parse_args()

    analysis_root = Path(args.analysis_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    table_dir = Path(args.table_dir).expanduser().resolve()
    figure_dir = Path(args.figure_dir).expanduser().resolve()
    construction_csv = Path(args.construction_csv).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite final output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    table_dir.mkdir(parents=True, exist_ok=True)
    for name in (*REQUIRED_TABLES, "c23_causal_configuration_table.csv"):
        if (table_dir / name).exists():
            raise FileExistsError(f"refusing to overwrite final table: {name}")
    for name in REQUIRED_FIGURES:
        if (figure_dir / name).exists():
            raise FileExistsError(f"refusing to overwrite final figure: {name}")

    tables = build_tables(analysis_root, construction_csv)
    table_mapping = {
        REQUIRED_TABLES[0]: tables["topology"],
        REQUIRED_TABLES[1]: tables["local"],
        REQUIRED_TABLES[2]: tables["exact"],
        REQUIRED_TABLES[3]: tables["hnsw"],
        REQUIRED_TABLES[4]: tables["decomposition"],
        REQUIRED_TABLES[5]: tables["balance"],
        REQUIRED_TABLES[6]: tables["correlations"],
        REQUIRED_TABLES[7]: tables["construction"],
        "c23_causal_configuration_table.csv": tables["causal"],
    }
    for name, rows in table_mapping.items():
        write_csv_new(table_dir / name, rows)

    generate_figures(tables, figure_dir)
    timestamp = c23_e1.utc_timestamp()
    git_commit = c23_protocol.git_commit(REPO_ROOT)
    run_summary_rows = build_run_summary(tables, timestamp, git_commit)
    write_run_summary(args.run_summary, run_summary_rows)
    verdicts = build_verdicts(tables)
    verdict_path = output_dir / "final_verdicts.json"
    write_json_new(verdict_path, verdicts)

    source_audits = [
        REPO_ROOT / "experiments/c23/logs/stage1-sift1m-reference-audit-v2.json",
        REPO_ROOT / "experiments/c23/logs/stage2-sift1m-e1-audit-v1.json",
        REPO_ROOT / "experiments/c23/logs/stage3-sift1m-e2-audit-v1.json",
        REPO_ROOT / "experiments/c23/logs/stage5-7-sift1m-local-audit-v1.json",
        REPO_ROOT / "experiments/c23/logs/stage8-sift1m-bootstrap-audit-v1.json",
        REPO_ROOT / "experiments/c23/logs/stage8-sift1m-variance-audit-v1.json",
        REPO_ROOT / "experiments/c23/logs/stage9-glove-reference-audit-v1.json",
        REPO_ROOT / "experiments/c23/logs/stage9-glove-e1-audit-v1.json",
        REPO_ROOT / "experiments/c23/logs/stage9-glove-e2-audit-v1.json",
        REPO_ROOT / "experiments/c23/logs/stage9-glove-local-audit-v1.json",
        REPO_ROOT / "experiments/c23/logs/stage9-glove-bootstrap-audit-v1.json",
        REPO_ROOT / "experiments/c23/logs/stage9-glove-variance-audit-v1.json",
        REPO_ROOT / "experiments/c23/logs/stage11-construction-audit-v1.json",
    ]
    table_counts = {name: len(rows) for name, rows in table_mapping.items()}
    figure_checks = {
        name: (figure_dir / name).is_file()
        and (figure_dir / name).stat().st_size > 1_000
        and (figure_dir / name).read_bytes()[:5] == b"%PDF-"
        for name in REQUIRED_FIGURES
    }
    required_causal_columns = {
        "edge_cut_ratio",
        "traversal_weighted_cut",
        "retained_neighbor_degree",
        "local_target_recall",
        "distance_computations_at_fixed_local_recall",
        "P_exact",
        "P_HNSW",
        "Delta_P",
        "W90",
    }
    checks = {
        "all_13_source_audits_have_only_pass_checks": "PASS"
        if all(path.is_file() and all_checks_pass(path) for path in source_audits)
        else "FAIL",
        "complete_40_configuration_final_tables": "PASS"
        if all(
            table_counts[name] == 40
            for name in REQUIRED_TABLES[:5]
        )
        and table_counts["c23_causal_configuration_table.csv"] == 40
        else "FAIL",
        "partition_balance_has_all_496_logical_shards": "PASS"
        if table_counts[REQUIRED_TABLES[5]] == 496
        else "FAIL",
        "correlation_analysis_has_18_scope_relation_rows": "PASS"
        if table_counts[REQUIRED_TABLES[6]] == 18
        else "FAIL",
        "construction_overhead_has_40_rows": "PASS"
        if table_counts[REQUIRED_TABLES[7]] == 40
        else "FAIL",
        "causal_configuration_table_has_required_metrics": "PASS"
        if required_causal_columns.issubset(tables["causal"][0])
        else "FAIL",
        "all_8_required_pdf_figures_valid": "PASS"
        if all(figure_checks.values())
        else "FAIL",
        "run_summary_has_40_rows_and_four_physical_hosts": "PASS"
        if len(run_summary_rows) == 40
        and all(int(row["physical_hosts"]) == 4 for row in run_summary_rows)
        else "FAIL",
        "logical_to_physical_mapping_is_modulo_four": "PASS"
        if all(
            json.loads(row["logical_to_physical_mapping"])["rule"]
            == "logical_shard_id modulo 4"
            for row in run_summary_rows
        )
        else "FAIL",
        "final_C2_and_C3_verdicts_are_separate": "PASS"
        if set(verdicts["final_verdicts"]) == {"C2", "C3"}
        else "FAIL",
    }
    failed = [name for name, status in checks.items() if status != "PASS"]
    if failed:
        raise ValueError(f"final C23 audit failed: {failed}")

    files = {
        "tables": {
            name: {
                "path": str(table_dir / name),
                "sha256": c23_e1.sha256_path(table_dir / name),
                "rows": table_counts[name],
            }
            for name in table_mapping
        },
        "figures": {
            name: {
                "path": str(figure_dir / name),
                "sha256": c23_e1.sha256_path(figure_dir / name),
            }
            for name in REQUIRED_FIGURES
        },
        "run_summary": {
            "path": str(Path(args.run_summary).expanduser().resolve()),
            "sha256": c23_e1.sha256_path(args.run_summary),
            "rows": len(run_summary_rows),
        },
        "final_verdicts": {
            "path": str(verdict_path),
            "sha256": c23_e1.sha256_path(verdict_path),
        },
    }
    manifest = {
        "protocol_version": c23_protocol.PROTOCOL_VERSION,
        "timestamp": timestamp,
        "git_commit": git_commit,
        "experiment": "C23 final two-dataset synthesis",
        "physical_host_count": 4,
        "logical_shards": list(LOGICAL_SHARDS),
        "microbenchmark_execution": "one logical shard at a time",
        "checks": checks,
        "table_counts": table_counts,
        "figure_checks": figure_checks,
        "source_audits": [str(path) for path in source_audits],
        "verdicts": verdicts,
        "files": files,
    }
    manifest_path = output_dir / "manifest.json"
    write_json_new(manifest_path, manifest)
    audit = {
        **manifest,
        "manifest": str(manifest_path),
        "manifest_sha256": c23_e1.sha256_path(manifest_path),
    }
    write_json_new(args.audit_output, audit)
    append_jsonl(
        args.run_manifest_jsonl,
        {
            "experiment_id": "stage12-c23-final-two-dataset-synthesis-v1",
            "timestamp": timestamp,
            "git_commit": git_commit,
            "protocol_version": c23_protocol.PROTOCOL_VERSION,
            "status": "COMPLETE",
            "checks": checks,
            "manifest": str(manifest_path),
            "manifest_sha256": c23_e1.sha256_path(manifest_path),
            "C2_status": verdicts["final_verdicts"]["C2"]["status"],
            "C3_status": verdicts["final_verdicts"]["C3"]["status"],
            "physical_host_count": 4,
            "logical_shards": list(LOGICAL_SHARDS),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
