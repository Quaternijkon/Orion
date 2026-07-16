#!/usr/bin/env python3
"""Aggregate existing Claim A online partition-family latency runs."""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path
from typing import Any, NamedTuple


SUMMARY_NAME = "claim_d_high_recall_latency_summary.csv"
HIGH_QPS_CV_THRESHOLD = 0.20
CLAIM_A_20260709_ROOT = "results/method4_claim_a_partition_online_latency_20260709"


class RunGroup(NamedTuple):
    comparison_group: str
    partition_label: str
    method_label: str
    method: str
    config_label: str
    collection: str
    routing_source_collection: str
    batch_size: int
    source_dirs: tuple[str, ...]
    source_root: str = ""
    base_note: str = ""


class TopologyNoFissionSensitivitySpec(NamedTuple):
    config_label: str
    selection_rank: int
    selection_role: str
    selection_note: str


DEFAULT_GROUPS: tuple[RunGroup, ...] = (
    RunGroup(
        comparison_group="full_fission_vs_naive",
        partition_label="full_fission_existing",
        method_label="full_fission_existing",
        method="Method4",
        config_label="full_fission_u160_b80_f8",
        collection="bench095_rr_orion_s31",
        routing_source_collection="bench095_rr_orion_s31",
        batch_size=100,
        source_dirs=("full_fission_vs_naive_b100/analysis_20260704_233349",),
        base_note="existing live full-fission/Orion collection; not full partition-family matrix",
    ),
    RunGroup(
        comparison_group="full_fission_vs_naive",
        partition_label="naive_reference_for_full_fission",
        method_label="naive_all_shards",
        method="Naive",
        config_label="naive_ef76",
        collection="bench095_rr_naive_s46",
        routing_source_collection="",
        batch_size=100,
        source_dirs=("full_fission_vs_naive_b100/analysis_20260704_233349",),
        base_note="paired naive all-shards reference from same run",
    ),
    RunGroup(
        comparison_group="full_fission_vs_naive",
        partition_label="full_fission_existing",
        method_label="full_fission_existing",
        method="Method4",
        config_label="full_fission_u160_b80_f8",
        collection="bench095_rr_orion_s31",
        routing_source_collection="bench095_rr_orion_s31",
        batch_size=200,
        source_dirs=("full_fission_vs_naive/analysis_20260704_233127",),
        base_note="existing live full-fission/Orion collection; not full partition-family matrix",
    ),
    RunGroup(
        comparison_group="full_fission_vs_naive",
        partition_label="naive_reference_for_full_fission",
        method_label="naive_all_shards",
        method="Naive",
        config_label="naive_ef76",
        collection="bench095_rr_naive_s46",
        routing_source_collection="",
        batch_size=200,
        source_dirs=("full_fission_vs_naive/analysis_20260704_233127",),
        base_note="paired naive all-shards reference from same run",
    ),
    RunGroup(
        comparison_group="balanced_kmeans_vs_naive",
        partition_label="balanced_kmeans_existing",
        method_label="balanced_kmeans_existing",
        method="Method4",
        config_label="balanced_kmeans_u160_b80_f8",
        collection="bench095_rr_kmeans_s46",
        routing_source_collection="bench095_rr_kmeans_s46",
        batch_size=100,
        source_dirs=("balanced_kmeans_vs_naive_b100/analysis_20260704_233859",),
        base_note="existing live balanced KMeans/KMeans-only-ish collection; not full partition-family matrix",
    ),
    RunGroup(
        comparison_group="balanced_kmeans_vs_naive",
        partition_label="naive_reference_for_balanced_kmeans",
        method_label="naive_all_shards",
        method="Naive",
        config_label="naive_ef76",
        collection="bench095_rr_naive_s46",
        routing_source_collection="",
        batch_size=100,
        source_dirs=("balanced_kmeans_vs_naive_b100/analysis_20260704_233859",),
        base_note="paired naive all-shards reference from same run",
    ),
    RunGroup(
        comparison_group="balanced_kmeans_vs_naive",
        partition_label="balanced_kmeans_existing",
        method_label="balanced_kmeans_existing",
        method="Method4",
        config_label="balanced_kmeans_u160_b80_f8",
        collection="bench095_rr_kmeans_s46",
        routing_source_collection="bench095_rr_kmeans_s46",
        batch_size=200,
        source_dirs=(
            "balanced_kmeans_vs_naive/analysis_20260704_233609",
            "balanced_kmeans_vs_naive_b200_confirm/analysis_20260704_234615",
        ),
        base_note="existing live balanced KMeans/KMeans-only-ish collection; confirm run added after first-run naive outlier",
    ),
    RunGroup(
        comparison_group="balanced_kmeans_vs_naive",
        partition_label="naive_reference_for_balanced_kmeans",
        method_label="naive_all_shards",
        method="Naive",
        config_label="naive_ef76",
        collection="bench095_rr_naive_s46",
        routing_source_collection="",
        batch_size=200,
        source_dirs=(
            "balanced_kmeans_vs_naive/analysis_20260704_233609",
            "balanced_kmeans_vs_naive_b200_confirm/analysis_20260704_234615",
        ),
        base_note="paired naive all-shards reference; batch=200 has repeated slow-tail outliers and should be read with variance caveat",
    ),
    RunGroup(
        comparison_group="topology_no_fission_vs_full_fission",
        partition_label="topology_no_fission",
        method_label="topology_no_fission",
        method="Method4",
        config_label="topology_no_fission_u160_b80_f10",
        collection="bench095_rr_topology_no_fission_s46_20260705",
        routing_source_collection="bench095_rr_topology_no_fission_s46_20260705",
        batch_size=200,
        source_dirs=("topology_no_fission_selected/analysis_20260705_012559",),
        base_note=(
            "selected topology/no-fission online latency cell; faithful_original_rest; "
            "topology_iters=50; disable_fission; GloVe cosine; 46 shards; client-observed batch latency"
        ),
    ),
    RunGroup(
        comparison_group="topology_no_fission_vs_full_fission",
        partition_label="full_fission_existing",
        method_label="full_fission_existing",
        method="Method4",
        config_label="full_fission_u160_b80_f8",
        collection="bench095_rr_orion_s31",
        routing_source_collection="bench095_rr_orion_s31",
        batch_size=200,
        source_dirs=("full_fission_vs_naive/analysis_20260704_233127",),
        base_note="existing live full-fission/Orion reference reused as the selected topology/no-fission comparison baseline",
    ),
    RunGroup(
        comparison_group="random_balanced_46_current_rebuild_vs_naive",
        partition_label="random_balanced_46",
        method_label="random_balanced_46",
        method="Method4",
        config_label="random_balanced_u160_b80_f8",
        collection="random_balanced_46",
        routing_source_collection="random_balanced_46",
        batch_size=200,
        source_dirs=("random_balanced_46_ef400build/analysis_20260709_092159",),
        source_root=CLAIM_A_20260709_ROOT,
        base_note=(
            "current-harness rebuild of Claim A random-balanced partition family; "
            "build upper_search_ef=400 to match oracle construction metadata; collection deleted after latency run"
        ),
    ),
    RunGroup(
        comparison_group="random_balanced_46_current_rebuild_vs_naive",
        partition_label="naive_reference_for_random_balanced_46",
        method_label="naive_all_shards",
        method="Naive",
        config_label="naive_ef76",
        collection="bench095_rr_naive_s46",
        routing_source_collection="",
        batch_size=200,
        source_dirs=("random_balanced_46_ef400build/analysis_20260709_092159",),
        source_root=CLAIM_A_20260709_ROOT,
        base_note="paired naive all-shards reference from the same current-harness random-balanced run",
    ),
    RunGroup(
        comparison_group="kmeans_topology_46_current_rebuild_vs_naive",
        partition_label="kmeans_topology_46",
        method_label="kmeans_topology_46",
        method="Method4",
        config_label="kmeans_topology_u160_b80_f10",
        collection="kmeans_topology_46",
        routing_source_collection="kmeans_topology_46",
        batch_size=200,
        source_dirs=("kmeans_topology_46_ef400build/analysis_20260709_093136",),
        source_root=CLAIM_A_20260709_ROOT,
        base_note=(
            "current-harness rebuild of Claim A KMeans topology partition family; "
            "build upper_search_ef=400 to match oracle construction metadata; collection deleted after latency run"
        ),
    ),
    RunGroup(
        comparison_group="kmeans_topology_46_current_rebuild_vs_naive",
        partition_label="naive_reference_for_kmeans_topology_46",
        method_label="naive_all_shards",
        method="Naive",
        config_label="naive_ef76",
        collection="bench095_rr_naive_s46",
        routing_source_collection="",
        batch_size=200,
        source_dirs=("kmeans_topology_46_ef400build/analysis_20260709_093136",),
        source_root=CLAIM_A_20260709_ROOT,
        base_note="paired naive all-shards reference from the same current-harness topology run",
    ),
    RunGroup(
        comparison_group="kmeans_topology_load_recalibrated_46_current_rebuild_vs_naive",
        partition_label="kmeans_topology_load_recalibrated_46",
        method_label="kmeans_topology_load_recalibrated_46",
        method="Method4",
        config_label="kmeans_topology_load_recalibrated_u160_b80_f10",
        collection="kmeans_topology_load_recalibrated_46",
        routing_source_collection="kmeans_topology_load_recalibrated_46",
        batch_size=200,
        source_dirs=("kmeans_topology_load_recalibrated_46_ef400build/analysis_20260709_094451",),
        source_root=CLAIM_A_20260709_ROOT,
        base_note=(
            "current-harness rebuild of Claim A load-recalibrated topology family; route map matches topology "
            "without fission, and collection was deleted after latency run"
        ),
    ),
    RunGroup(
        comparison_group="kmeans_topology_load_recalibrated_46_current_rebuild_vs_naive",
        partition_label="naive_reference_for_kmeans_topology_load_recalibrated_46",
        method_label="naive_all_shards",
        method="Naive",
        config_label="naive_ef76",
        collection="bench095_rr_naive_s46",
        routing_source_collection="",
        batch_size=200,
        source_dirs=("kmeans_topology_load_recalibrated_46_ef400build/analysis_20260709_094451",),
        source_root=CLAIM_A_20260709_ROOT,
        base_note="paired naive all-shards reference from the same current-harness load-recalibrated run",
    ),
)


DEFAULT_DELTAS: tuple[tuple[str, str, str], ...] = (
    ("full_fission_existing_vs_naive", "full_fission_existing", "naive_all_shards"),
    ("balanced_kmeans_existing_vs_naive", "balanced_kmeans_existing", "naive_all_shards"),
    ("topology_no_fission_vs_full_fission", "topology_no_fission", "full_fission_existing"),
    ("random_balanced_46_vs_naive_current_rebuild", "random_balanced_46", "naive_all_shards"),
    ("kmeans_topology_46_vs_naive_current_rebuild", "kmeans_topology_46", "naive_all_shards"),
    (
        "kmeans_topology_load_recalibrated_46_vs_naive_current_rebuild",
        "kmeans_topology_load_recalibrated_46",
        "naive_all_shards",
    ),
)


TOPOLOGY_NO_FISSION_SOURCE_DIRS = ("topology_no_fission_selected/analysis_20260705_012559",)


TOPOLOGY_NO_FISSION_SENSITIVITY_SPECS: tuple[TopologyNoFissionSensitivitySpec, ...] = (
    TopologyNoFissionSensitivitySpec(
        config_label="topology_no_fission_u160_b80_f10",
        selection_rank=1,
        selection_role="selected_main_cell",
        selection_note="Selected for the Claim A submatrix: higher QPS, lower EF-sum, and lower tail latency.",
    ),
    TopologyNoFissionSensitivitySpec(
        config_label="topology_no_fission_u160_b120_f8",
        selection_rank=2,
        selection_role="slower_sensitivity_candidate",
        selection_note="Sensitivity candidate from the same run; not used as the main row because it is slower and uses more EF-sum.",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate Claim A online partition latency submatrix.")
    parser.add_argument(
        "--input-root",
        default="results/method4_claim_a_partition_online_latency_20260704",
        help="Root containing Claim A online latency run directories.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/method4_claim_coverage_20260704/derived_claim_tables",
        help="Directory for derived CSV outputs.",
    )
    return parser.parse_args()


def float_value(value: Any) -> float:
    if value in ("", None):
        return 0.0
    return float(value)


def read_summary_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def stdev(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def pct_delta(value: float, baseline: float) -> float:
    if baseline == 0:
        return 0.0
    return (value - baseline) / baseline * 100.0


def unique_int_value(rows: list[dict[str, str]], field: str) -> int | str:
    values = {
        int(float(row[field]))
        for row in rows
        if row.get(field) not in ("", None)
    }
    if len(values) == 1:
        return next(iter(values))
    if not values:
        return ""
    return ";".join(str(value) for value in sorted(values))


def selected_rows(root: Path, group: RunGroup) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    source_root = Path(group.source_root) if group.source_root else root
    for source_dir in group.source_dirs:
        summary_path = source_root / source_dir / SUMMARY_NAME
        if not summary_path.exists():
            raise FileNotFoundError(summary_path)
        for row in read_summary_rows(summary_path):
            if row.get("method") != group.method:
                continue
            if row.get("config_label") != group.config_label:
                continue
            if int(float(row.get("batch_size") or 0)) != group.batch_size:
                continue
            rows.append(row)
    if not rows:
        raise RuntimeError(f"no rows for {group.method_label} {group.config_label} batch={group.batch_size}")
    return rows


def source_dir_labels(group: RunGroup) -> str:
    if not group.source_root:
        return ";".join(group.source_dirs)
    return ";".join(str(Path(group.source_root) / source_dir) for source_dir in group.source_dirs)


def summarize_metric(rows: list[dict[str, str]], field: str) -> tuple[float, float, float]:
    values = [float_value(row.get(field)) for row in rows]
    return mean(values), stdev(values), median(values)


def aggregate_group(root: Path, group: RunGroup) -> dict[str, Any]:
    rows = selected_rows(root, group)
    qps_values = [float_value(row.get("qps")) for row in rows]
    qps_mean = mean(qps_values)
    qps_stdev = stdev(qps_values)
    qps_cv = qps_stdev / qps_mean if qps_mean else 0.0
    notes = [group.base_note] if group.base_note else []
    if qps_cv > HIGH_QPS_CV_THRESHOLD:
        notes.append(f"high_qps_variance_cv={qps_cv:.3f}")

    recall_mean, recall_stdev, recall_median = summarize_metric(rows, "recall_at_10")
    p50_mean, p50_stdev, p50_median = summarize_metric(rows, "batch_latency_p50_ms")
    p95_mean, p95_stdev, p95_median = summarize_metric(rows, "batch_latency_p95_ms")
    p99_mean, p99_stdev, p99_median = summarize_metric(rows, "batch_latency_p99_ms")
    visited_mean, visited_stdev, visited_median = summarize_metric(rows, "avg_visited_shards")
    ef_mean, ef_stdev, ef_median = summarize_metric(rows, "avg_assigned_ef_sum")

    return {
        "comparison_group": group.comparison_group,
        "partition_label": group.partition_label,
        "method_label": group.method_label,
        "method": group.method,
        "collection": group.collection,
        "routing_source_collection": group.routing_source_collection,
        "batch_size": group.batch_size,
        "config_label": group.config_label,
        "recall_at_10_mean": recall_mean,
        "recall_at_10_stdev": recall_stdev,
        "recall_at_10_median": recall_median,
        "qps_mean": qps_mean,
        "qps_stdev": qps_stdev,
        "qps_median": median(qps_values),
        "qps_cv": qps_cv,
        "batch_latency_p50_ms_mean": p50_mean,
        "batch_latency_p50_ms_stdev": p50_stdev,
        "batch_latency_p50_ms_median": p50_median,
        "batch_latency_p95_ms_mean": p95_mean,
        "batch_latency_p95_ms_stdev": p95_stdev,
        "batch_latency_p95_ms_median": p95_median,
        "batch_latency_p99_ms_mean": p99_mean,
        "batch_latency_p99_ms_stdev": p99_stdev,
        "batch_latency_p99_ms_median": p99_median,
        "avg_visited_shards_mean": visited_mean,
        "avg_visited_shards_stdev": visited_stdev,
        "avg_visited_shards_median": visited_median,
        "avg_assigned_ef_sum_mean": ef_mean,
        "avg_assigned_ef_sum_stdev": ef_stdev,
        "avg_assigned_ef_sum_median": ef_median,
        "repeats": len(rows),
        "source_dirs": source_dir_labels(group),
        "notes": "; ".join(notes),
    }


def row_key(row: dict[str, Any]) -> tuple[str, str, int]:
    return str(row["comparison_group"]), str(row["method_label"]), int(row["batch_size"])


def build_delta_rows(
    summary_rows: list[dict[str, Any]],
    delta_specs: list[tuple[str, str, str]] | tuple[tuple[str, str, str], ...] = DEFAULT_DELTAS,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for comparison_label, method_label, baseline_label in delta_specs:
        candidate_groups = sorted(
            {
                str(row["comparison_group"])
                for row in summary_rows
                if row.get("method_label") in {method_label, baseline_label}
            }
        )
        for comparison_group in candidate_groups:
            group_rows = [row for row in summary_rows if row.get("comparison_group") == comparison_group]
            batch_sizes = sorted({int(row["batch_size"]) for row in group_rows})
            by_key = {row_key(row): row for row in group_rows}
            for batch_size in batch_sizes:
                method = by_key.get((comparison_group, method_label, batch_size))
                baseline = by_key.get((comparison_group, baseline_label, batch_size))
                if method is None or baseline is None:
                    continue
                out.append(build_delta_row(comparison_label, batch_size, method_label, baseline_label, method, baseline))
    return out


def build_delta_row(
    comparison_label: str,
    batch_size: int,
    method_label: str,
    baseline_label: str,
    method: dict[str, Any],
    baseline: dict[str, Any],
) -> dict[str, Any]:
    notes = "; ".join(
        note
        for note in [str(method.get("notes", "")), str(baseline.get("notes", ""))]
        if note
    )
    return {
        "comparison_label": comparison_label,
        "batch_size": batch_size,
        "method_label": method_label,
        "baseline_label": baseline_label,
        "method_recall_at_10_mean": method["recall_at_10_mean"],
        "baseline_recall_at_10_mean": baseline["recall_at_10_mean"],
        "recall_delta_mean": method["recall_at_10_mean"] - baseline["recall_at_10_mean"],
        "qps_delta_mean": method["qps_mean"] - baseline["qps_mean"],
        "qps_delta_pct_mean": pct_delta(method["qps_mean"], baseline["qps_mean"]),
        "qps_delta_pct_median": pct_delta(method["qps_median"], baseline["qps_median"]),
        "visited_shards_delta_mean": method["avg_visited_shards_mean"] - baseline["avg_visited_shards_mean"],
        "visited_shards_delta_pct_mean": pct_delta(
            method["avg_visited_shards_mean"], baseline["avg_visited_shards_mean"]
        ),
        "ef_sum_delta_mean": method["avg_assigned_ef_sum_mean"] - baseline["avg_assigned_ef_sum_mean"],
        "ef_sum_delta_pct_mean": pct_delta(
            method["avg_assigned_ef_sum_mean"], baseline["avg_assigned_ef_sum_mean"]
        ),
        "p95_latency_delta_ms_mean": method["batch_latency_p95_ms_mean"] - baseline["batch_latency_p95_ms_mean"],
        "p95_latency_delta_pct_mean": pct_delta(
            method["batch_latency_p95_ms_mean"], baseline["batch_latency_p95_ms_mean"]
        ),
        "p95_latency_delta_pct_median": pct_delta(
            method["batch_latency_p95_ms_median"], baseline["batch_latency_p95_ms_median"]
        ),
        "p99_latency_delta_ms_mean": method["batch_latency_p99_ms_mean"] - baseline["batch_latency_p99_ms_mean"],
        "p99_latency_delta_pct_mean": pct_delta(
            method["batch_latency_p99_ms_mean"], baseline["batch_latency_p99_ms_mean"]
        ),
        "p99_latency_delta_pct_median": pct_delta(
            method["batch_latency_p99_ms_median"], baseline["batch_latency_p99_ms_median"]
        ),
        "method_repeats": method.get("repeats", ""),
        "baseline_repeats": baseline.get("repeats", ""),
        "notes": notes,
    }


def topology_no_fission_run_group(spec: TopologyNoFissionSensitivitySpec) -> RunGroup:
    return RunGroup(
        comparison_group="topology_no_fission_config_sensitivity",
        partition_label="topology_no_fission",
        method_label="topology_no_fission",
        method="Method4",
        config_label=spec.config_label,
        collection="bench095_rr_topology_no_fission_s46_20260705",
        routing_source_collection="bench095_rr_topology_no_fission_s46_20260705",
        batch_size=200,
        source_dirs=TOPOLOGY_NO_FISSION_SOURCE_DIRS,
        base_note="topology/no-fission selected-run config sensitivity; faithful_original_rest; topology_iters=50; disable_fission",
    )


def aggregate_topology_no_fission_sensitivity_config(
    root: Path,
    spec: TopologyNoFissionSensitivitySpec,
) -> dict[str, Any]:
    group = topology_no_fission_run_group(spec)
    raw_rows = selected_rows(root, group)
    summary = aggregate_group(root, group)
    notes = "; ".join(note for note in [str(summary.get("notes", "")), spec.selection_note] if note)
    return {
        "selection_rank": spec.selection_rank,
        "config_label": spec.config_label,
        "selection_role": spec.selection_role,
        "recall_at_10_mean": summary["recall_at_10_mean"],
        "recall_at_10_stdev": summary["recall_at_10_stdev"],
        "qps_mean": summary["qps_mean"],
        "qps_stdev": summary["qps_stdev"],
        "qps_median": summary["qps_median"],
        "avg_visited_shards_mean": summary["avg_visited_shards_mean"],
        "avg_assigned_ef_sum_mean": summary["avg_assigned_ef_sum_mean"],
        "batch_latency_p95_ms_mean": summary["batch_latency_p95_ms_mean"],
        "batch_latency_p95_ms_stdev": summary["batch_latency_p95_ms_stdev"],
        "batch_latency_p95_ms_median": summary["batch_latency_p95_ms_median"],
        "batch_latency_p99_ms_mean": summary["batch_latency_p99_ms_mean"],
        "batch_latency_p99_ms_stdev": summary["batch_latency_p99_ms_stdev"],
        "batch_latency_p99_ms_median": summary["batch_latency_p99_ms_median"],
        "qps_delta_vs_selected": 0.0,
        "qps_delta_pct_vs_selected": 0.0,
        "ef_sum_delta_vs_selected": 0.0,
        "ef_sum_delta_pct_vs_selected": 0.0,
        "p95_latency_delta_ms_vs_selected": 0.0,
        "p95_latency_delta_pct_vs_selected": 0.0,
        "p99_latency_delta_ms_vs_selected": 0.0,
        "p99_latency_delta_pct_vs_selected": 0.0,
        "repeats": summary["repeats"],
        "batch_size": summary["batch_size"],
        "upper_k": unique_int_value(raw_rows, "upper_k"),
        "base_ef": unique_int_value(raw_rows, "base_ef"),
        "factor": unique_int_value(raw_rows, "factor"),
        "collection": summary["collection"],
        "source_dirs": summary["source_dirs"],
        "notes": notes,
    }


def build_topology_no_fission_sensitivity_rows(root: Path) -> list[dict[str, Any]]:
    rows = [
        aggregate_topology_no_fission_sensitivity_config(root, spec)
        for spec in TOPOLOGY_NO_FISSION_SENSITIVITY_SPECS
    ]
    rows.sort(key=lambda row: int(row["selection_rank"]))
    selected = rows[0]
    for row in rows:
        row["qps_delta_vs_selected"] = row["qps_mean"] - selected["qps_mean"]
        row["qps_delta_pct_vs_selected"] = pct_delta(row["qps_mean"], selected["qps_mean"])
        row["ef_sum_delta_vs_selected"] = row["avg_assigned_ef_sum_mean"] - selected["avg_assigned_ef_sum_mean"]
        row["ef_sum_delta_pct_vs_selected"] = pct_delta(
            row["avg_assigned_ef_sum_mean"],
            selected["avg_assigned_ef_sum_mean"],
        )
        row["p95_latency_delta_ms_vs_selected"] = (
            row["batch_latency_p95_ms_mean"] - selected["batch_latency_p95_ms_mean"]
        )
        row["p95_latency_delta_pct_vs_selected"] = pct_delta(
            row["batch_latency_p95_ms_mean"],
            selected["batch_latency_p95_ms_mean"],
        )
        row["p99_latency_delta_ms_vs_selected"] = (
            row["batch_latency_p99_ms_mean"] - selected["batch_latency_p99_ms_mean"]
        )
        row["p99_latency_delta_pct_vs_selected"] = pct_delta(
            row["batch_latency_p99_ms_mean"],
            selected["batch_latency_p99_ms_mean"],
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"no rows to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    root = Path(args.input_root)
    output_dir = Path(args.output_dir)
    summary_rows = [aggregate_group(root, group) for group in DEFAULT_GROUPS]
    delta_rows = build_delta_rows(summary_rows)
    topology_no_fission_sensitivity_rows = build_topology_no_fission_sensitivity_rows(root)
    write_csv(output_dir / "claim_a_partition_online_latency_submatrix.csv", summary_rows)
    write_csv(output_dir / "claim_a_partition_online_latency_submatrix_deltas.csv", delta_rows)
    write_csv(
        output_dir / "claim_a_topology_no_fission_config_sensitivity.csv",
        topology_no_fission_sensitivity_rows,
    )
    print(output_dir / "claim_a_partition_online_latency_submatrix.csv")
    print(output_dir / "claim_a_partition_online_latency_submatrix_deltas.csv")
    print(output_dir / "claim_a_topology_no_fission_config_sensitivity.csv")


if __name__ == "__main__":
    main()
