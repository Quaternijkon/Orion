#!/usr/bin/env python3
"""Build the compact six-point Method4 distributed submission report.

The input directory must be one completed benchmark-matrix run containing
``matrix_summary.csv`` and exactly the six target/method pairs formed by
{0.95, 0.97} x {Orion, Simple KMeans, Naive}.  Report statistics are always
recomputed from each case's ``stability_runs.csv``; tuning/final-row summary
statistics are deliberately not reused.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results/method4_distributed_submission/final_report"
TARGETS = (0.95, 0.97)
METHODS = ("Orion", "Simple KMeans", "Naive")
METHOD_COLORS = {
    "Orion": "#2f6fbb",
    "Simple KMeans": "#e28b27",
    "Naive": "#737373",
}

CSV_FIELDS = [
    "target_recall",
    "method",
    "case_name",
    "params",
    "recall_mean",
    "recall_stdev",
    "recall_delta_to_target",
    "recall_match_status",
    "qps_mean",
    "qps_stdev",
    "qps_cv_pct",
    "p50_ms_mean",
    "p50_ms_stdev",
    "p95_ms_mean",
    "p95_ms_stdev",
    "p99_ms_mean",
    "p99_ms_stdev",
    "stability_repeats",
    "eval_query_count",
    "warmup_query_count",
    "batch_size",
    "avg_visited_shards",
    "avg_ef_per_visited_shard",
    "ef_sum_per_query",
    "avg_physical_peers",
    "avg_search_requests_per_query",
    "search_batch_calls",
    "fixed_ef_shard_chunk_size",
    "indexed_vectors",
    "logical_points",
    "index_expansion_ratio",
    "segments_count",
    "cluster_peer_count",
    "cluster_shard_count",
    "cluster_active_shards",
    "controller_local_lower_shards",
    "worker_shard_counts",
    "placement_valid",
    "routing_metadata_status",
    "routing_metadata_verified",
    "routing_metadata_expected_fingerprint",
    "routing_metadata_actual_fingerprint",
    "runtime_health_valid",
    "transport_log_clean",
    "transport_resources_valid",
    "p2p_connections_start",
    "p2p_connections_end",
    "p2p_connections_delta",
    "fd_count_start",
    "fd_count_end",
    "fd_count_delta",
    "expected_steady_state_p2p_connections",
    "resource_contract_valid",
    "resource_contract",
    "client_cpu_affinity",
    "search_dispatch_mode",
    "lower_execution_order",
    "routed_execution_mode",
    "routed_planning_mode",
    "image_digest",
    "repository_commit",
    "repository_dirty",
    "dataset_sha256",
    "source_main_run_dir",
    "summary_path",
    "stability_path",
    "builds_path",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "main_run_dir",
        type=Path,
        help="Completed six-case matrix directory containing matrix_summary.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Report destination (default: {DEFAULT_OUTPUT_DIR}).",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def as_float(value: Any, *, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric, got {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite, got {value!r}")
    return result


def as_int(value: Any, *, field: str) -> int:
    return int(as_float(value, field=field))


def optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return as_int(value, field="optional integer")


def metric_stats(rows: list[dict[str, str]], field: str) -> tuple[float, float]:
    values = [as_float(row.get(field), field=field) for row in rows]
    if not values:
        raise ValueError(f"no values for {field}")
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def resolve_input_path(raw: str | Path, main_run_dir: Path) -> Path:
    candidate = Path(raw)
    options = [
        candidate,
        REPO_ROOT / candidate,
        main_run_dir / candidate,
    ]
    for option in options:
        resolved = option.resolve()
        if resolved.exists():
            return resolved
    raise FileNotFoundError(f"could not resolve result path {raw!r}")


def method_name(matrix_row: dict[str, str], summary: dict[str, Any]) -> str:
    preset = str(matrix_row.get("preset") or "")
    routing_mode = str(summary.get("routing_mode") or matrix_row.get("routing_mode") or "")
    if preset == "orion" or routing_mode == "faithful_original_rest":
        return "Orion"
    if preset == "kmeans_simple_nprobe" or routing_mode == "kmeans_simple_nprobe":
        return "Simple KMeans"
    if preset == "naive" or routing_mode == "naive_hash_all_shards":
        return "Naive"
    raise ValueError(
        f"unsupported method for case {matrix_row.get('case_name')!r}: "
        f"preset={preset!r}, routing_mode={routing_mode!r}"
    )


def method_params(method: str, summary: dict[str, Any], final_row: dict[str, Any]) -> str:
    base_ef = as_int(final_row.get("base_ef"), field="base_ef")
    if method == "Orion":
        return (
            f"upper_k={as_int(final_row.get('upper_k'), field='upper_k')},"
            f"base={base_ef},factor={as_int(final_row.get('factor'), field='factor')},"
            f"planning={summary.get('routed_planning_mode') or 'unknown'}"
        )
    if method == "Simple KMeans":
        details = summary.get("kmeans_simple_nprobe") or {}
        alpha = as_float(details.get("multi_assign_alpha", 1.0), field="multi_assign_alpha")
        return (
            f"nprobe={as_int(final_row.get('nprobe'), field='nprobe')},"
            f"fixed_ef={base_ef},alpha={alpha:g}"
        )
    return f"fixed_ef={base_ef},all_shards=46"


def deployment_resource_contract(summary: dict[str, Any]) -> tuple[bool, str]:
    manifest = summary.get("deployment_manifest") or {}
    nodes = manifest.get("nodes") or []
    expected = {
        "controller": ("0-7", 8, 4),
        "qdrant_shard_1": ("0-19", 16, 4),
        "qdrant_shard_2": ("0-19", 16, 4),
        "qdrant_shard_3": ("0-19", 16, 4),
    }
    observed: dict[str, Any] = {}
    valid = len(nodes) == 4
    for node in nodes:
        role = str(node.get("role") or "")
        nofile = node.get("nofile") or {}
        observed[role] = {
            "cpuset": node.get("cpuset"),
            "max_search_threads": node.get("max_search_threads"),
            "optimizer_cpu_budget": node.get("optimizer_cpu_budget"),
            "nofile_soft": nofile.get("soft"),
            "nofile_hard": nofile.get("hard"),
        }
        role_expected = expected.get(role)
        valid = valid and role_expected is not None
        if role_expected is not None:
            valid = valid and (
                str(node.get("cpuset")) == role_expected[0]
                and int(node.get("max_search_threads", -1)) == role_expected[1]
                and int(node.get("optimizer_cpu_budget", -1)) == role_expected[2]
                and int(nofile.get("soft", -1)) == 65536
                and int(nofile.get("hard", -1)) == 65536
            )
    affinity = [int(cpu) for cpu in summary.get("process_affinity") or []]
    valid = valid and affinity == list(range(8, 20))
    observed["benchmark_client"] = {"affinity": affinity}
    return bool(valid), canonical_json(observed)


def build_aggregate_row(
    matrix_row: dict[str, str],
    summary_path: Path,
    main_run_dir: Path,
) -> dict[str, Any]:
    summary = load_json(summary_path)
    result_dir = summary_path.parent
    stability_path = result_dir / "stability_runs.csv"
    builds_path = result_dir / "builds.csv"
    if not stability_path.exists():
        raise FileNotFoundError(stability_path)
    if not builds_path.exists():
        raise FileNotFoundError(builds_path)

    stability_rows = load_csv(stability_path)
    if len(stability_rows) < 2:
        raise ValueError(f"{stability_path} must contain repeated runs")
    builds_rows = load_csv(builds_path)
    if len(builds_rows) != 1:
        raise ValueError(f"{builds_path} must contain exactly one build row")
    build = builds_rows[0]

    recall_mean, recall_stdev = metric_stats(stability_rows, "recall_at_k")
    qps_mean, qps_stdev = metric_stats(stability_rows, "qps")
    p50_mean, p50_stdev = metric_stats(stability_rows, "latency_p50_ms")
    p95_mean, p95_stdev = metric_stats(stability_rows, "latency_p95_ms")
    p99_mean, p99_stdev = metric_stats(stability_rows, "latency_p99_ms")
    target_recall = as_float(matrix_row.get("target_recall"), field="target_recall")
    recall_delta = recall_mean - target_recall
    recall_status = "strict" if abs(recall_delta) <= 0.003 + 1e-12 else "nearest"

    final_row = summary.get("final_row") or {}
    method = method_name(matrix_row, summary)
    cluster = summary.get("collection_cluster") or {}
    metadata = summary.get("collection_routing_build_metadata") or {}
    runtime = summary.get("runtime_health_audit") or {}
    log_audit = runtime.get("log_audit") or {}
    transport = runtime.get("transport_resources") or {}
    transport_start = transport.get("start") or {}
    transport_end = transport.get("end") or {}
    transport_delta = transport.get("delta") or {}
    indexed_vectors = as_int(
        build.get("indexed_vectors_count") or build.get("points_count"),
        field="indexed_vectors_count",
    )
    logical_points = as_int(
        build.get("logical_points_count") or indexed_vectors,
        field="logical_points_count",
    )
    resource_valid, resource_details = deployment_resource_contract(summary)
    worker_shard_counts = cluster.get("cluster_shards_per_peer") or {}

    return {
        "target_recall": target_recall,
        "method": method,
        "case_name": matrix_row.get("case_name") or matrix_row.get("run_id"),
        "params": method_params(method, summary, final_row),
        "recall_mean": recall_mean,
        "recall_stdev": recall_stdev,
        "recall_delta_to_target": recall_delta,
        "recall_match_status": recall_status,
        "qps_mean": qps_mean,
        "qps_stdev": qps_stdev,
        "qps_cv_pct": 100.0 * qps_stdev / qps_mean if qps_mean else math.inf,
        "p50_ms_mean": p50_mean,
        "p50_ms_stdev": p50_stdev,
        "p95_ms_mean": p95_mean,
        "p95_ms_stdev": p95_stdev,
        "p99_ms_mean": p99_mean,
        "p99_ms_stdev": p99_stdev,
        "stability_repeats": len(stability_rows),
        "eval_query_count": as_int(stability_rows[0].get("query_count"), field="query_count"),
        "warmup_query_count": optional_int(summary.get("warmup_query_count")),
        "batch_size": optional_int(matrix_row.get("batch_size")),
        "avg_visited_shards": as_float(
            final_row.get("avg_visited_shards"), field="avg_visited_shards"
        ),
        "avg_ef_per_visited_shard": as_float(
            final_row.get("avg_assigned_ef_per_visited_shard"),
            field="avg_assigned_ef_per_visited_shard",
        ),
        "ef_sum_per_query": as_float(
            final_row.get("estimated_ef_sum_per_query"), field="estimated_ef_sum_per_query"
        ),
        "avg_physical_peers": as_float(
            final_row.get("avg_physical_peers_per_query"),
            field="avg_physical_peers_per_query",
        ),
        "avg_search_requests_per_query": as_float(
            final_row.get("avg_search_requests_per_query"),
            field="avg_search_requests_per_query",
        ),
        "search_batch_calls": as_int(
            final_row.get("search_batch_calls"), field="search_batch_calls"
        ),
        "fixed_ef_shard_chunk_size": as_int(
            summary.get("fixed_ef_shard_chunk_size", 0), field="fixed_ef_shard_chunk_size"
        ),
        "indexed_vectors": indexed_vectors,
        "logical_points": logical_points,
        "index_expansion_ratio": indexed_vectors / logical_points,
        "segments_count": optional_int(build.get("segments_count")),
        "cluster_peer_count": as_int(
            cluster.get("cluster_peer_count") or build.get("cluster_peer_count"),
            field="cluster_peer_count",
        ),
        "cluster_shard_count": as_int(
            cluster.get("cluster_shard_count") or build.get("cluster_shard_count"),
            field="cluster_shard_count",
        ),
        "cluster_active_shards": as_int(
            cluster.get("cluster_active_shards") or build.get("cluster_active_shards"),
            field="cluster_active_shards",
        ),
        "controller_local_lower_shards": as_int(
            cluster.get("cluster_local_shards", build.get("cluster_local_shards", 0)),
            field="cluster_local_shards",
        ),
        "worker_shard_counts": canonical_json(worker_shard_counts),
        "placement_valid": bool(
            summary.get("placement_valid", cluster.get("cluster_placement_valid", False))
        ),
        "routing_metadata_status": metadata.get("status") or build.get("routing_build_metadata_status"),
        "routing_metadata_verified": bool(
            metadata.get("verified", str(build.get("routing_build_metadata_verified")).lower() == "true")
        ),
        "routing_metadata_expected_fingerprint": metadata.get("expected_fingerprint")
        or build.get("routing_build_metadata_expected_fingerprint"),
        "routing_metadata_actual_fingerprint": metadata.get("actual_fingerprint")
        or build.get("routing_build_metadata_actual_fingerprint"),
        "runtime_health_valid": bool(runtime.get("valid", False)),
        "transport_log_clean": bool(log_audit.get("valid", False)),
        "transport_resources_valid": bool(transport.get("valid", False)),
        "p2p_connections_start": optional_int(transport_start.get("p2p_established")),
        "p2p_connections_end": optional_int(transport_end.get("p2p_established")),
        "p2p_connections_delta": optional_int(transport_delta.get("p2p_established")),
        "fd_count_start": optional_int(transport_start.get("fd_count")),
        "fd_count_end": optional_int(transport_end.get("fd_count")),
        "fd_count_delta": optional_int(transport_delta.get("fd_count")),
        "expected_steady_state_p2p_connections": optional_int(
            transport.get("expected_steady_state_p2p_connections")
        ),
        "resource_contract_valid": resource_valid,
        "resource_contract": resource_details,
        "client_cpu_affinity": canonical_json(summary.get("process_affinity") or []),
        "search_dispatch_mode": summary.get("search_dispatch_mode"),
        "lower_execution_order": summary.get("lower_execution_order"),
        "routed_execution_mode": summary.get("routed_execution_mode"),
        "routed_planning_mode": summary.get("routed_planning_mode"),
        "image_digest": summary.get("image_digest"),
        "repository_commit": (summary.get("repository") or {}).get("commit"),
        "repository_dirty": bool((summary.get("repository") or {}).get("dirty", False)),
        "dataset_sha256": summary.get("dataset_sha256"),
        "source_main_run_dir": str(main_run_dir),
        "summary_path": str(summary_path),
        "stability_path": str(stability_path),
        "builds_path": str(builds_path),
    }


def load_aggregate(main_run_dir: Path) -> list[dict[str, Any]]:
    matrix_path = main_run_dir / "matrix_summary.csv"
    if not matrix_path.exists():
        raise FileNotFoundError(matrix_path)
    matrix_rows = load_csv(matrix_path)
    if len(matrix_rows) != 6:
        raise ValueError(f"expected exactly six matrix rows in {matrix_path}, found {len(matrix_rows)}")

    rows: list[dict[str, Any]] = []
    for matrix_row in matrix_rows:
        raw_summary = matrix_row.get("summary_path")
        if not raw_summary:
            raise ValueError(f"case {matrix_row.get('case_name')!r} has no summary_path")
        summary_path = resolve_input_path(raw_summary, main_run_dir)
        rows.append(build_aggregate_row(matrix_row, summary_path, main_run_dir))

    expected = {(target, method) for target in TARGETS for method in METHODS}
    actual = {(round(float(row["target_recall"]), 2), str(row["method"])) for row in rows}
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise ValueError(f"invalid six-point matrix; missing={missing}, unexpected={unexpected}")
    rows = sorted(
        rows,
        key=lambda row: (float(row["target_recall"]), METHODS.index(row["method"])),
    )
    validate_submission_rows(rows)
    return rows


def validate_submission_rows(rows: list[dict[str, Any]]) -> None:
    """Reject a six-point report unless every formal distributed gate passes."""
    failures: list[str] = []

    def fail(row: dict[str, Any], message: str) -> None:
        failures.append(
            f"target={float(row['target_recall']):.2f} method={row['method']}: {message}"
        )

    required_true = (
        "placement_valid",
        "runtime_health_valid",
        "transport_log_clean",
        "transport_resources_valid",
        "resource_contract_valid",
    )
    for row in rows:
        if row.get("recall_match_status") != "strict":
            fail(row, "recall point is not a strict target match")
        for field in required_true:
            if row.get(field) is not True:
                fail(row, f"{field} is not true")

        exact_values = {
            "stability_repeats": 3,
            "eval_query_count": 3000,
            "warmup_query_count": 500,
            "batch_size": 200,
            "cluster_peer_count": 4,
            "cluster_shard_count": 46,
            "cluster_active_shards": 46,
            "controller_local_lower_shards": 0,
            "fixed_ef_shard_chunk_size": 0,
            "expected_steady_state_p2p_connections": 6,
            "p2p_connections_start": 6,
            "p2p_connections_end": 6,
            "p2p_connections_delta": 0,
        }
        for field, expected_value in exact_values.items():
            if row.get(field) != expected_value:
                fail(
                    row,
                    f"{field}={row.get(field)!r}, expected {expected_value!r}",
                )

        try:
            worker_counts = json.loads(str(row.get("worker_shard_counts") or ""))
        except json.JSONDecodeError:
            worker_counts = None
        if not isinstance(worker_counts, dict) or sorted(worker_counts.values()) != [15, 15, 16]:
            fail(row, f"worker shard placement is not 15/15/16: {worker_counts!r}")

    for field in (
        "image_digest",
        "repository_commit",
        "dataset_sha256",
        "resource_contract",
        "source_main_run_dir",
    ):
        values = {str(row.get(field) or "") for row in rows}
        if "" in values or len(values) != 1:
            failures.append(f"{field} is missing or inconsistent across cases: {sorted(values)!r}")

    if failures:
        detail = "\n - ".join(failures)
        raise ValueError(f"submission acceptance gate failed:\n - {detail}")


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=CSV_FIELDS,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def annotate_recall(axis: Any, x: float, y: float, row: dict[str, Any], offset: float) -> None:
    axis.text(
        x,
        y + offset,
        f"R={float(row['recall_mean']):.4f}\n{row['recall_match_status']}",
        ha="center",
        va="bottom",
        fontsize=7.5,
    )


def save_figure(fig: Any, output_dir: Path, stem: str) -> None:
    fig.savefig(output_dir / f"{stem}.png", dpi=200, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")


def write_plots(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    cache_dir = Path("/tmp/method4-matplotlib-cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib and numpy are required to generate the submission figures"
        ) from exc

    row_by_key = {(float(row["target_recall"]), row["method"]): row for row in rows}

    # Same-recall QPS.
    x = np.arange(len(TARGETS), dtype=float)
    width = 0.24
    fig, axis = plt.subplots(figsize=(8.2, 5.0))
    for method_idx, method in enumerate(METHODS):
        method_rows = [row_by_key[(target, method)] for target in TARGETS]
        positions = x + (method_idx - 1) * width
        values = [float(row["qps_mean"]) for row in method_rows]
        errors = [float(row["qps_stdev"]) for row in method_rows]
        bars = axis.bar(
            positions,
            values,
            width,
            yerr=errors,
            capsize=4,
            color=METHOD_COLORS[method],
            label=method,
        )
        for position, bar, row in zip(positions, bars, method_rows):
            annotate_recall(axis, float(position), float(bar.get_height()), row, max(values) * 0.015)
    axis.set_xticks(x, [f"Target {target:.2f}" for target in TARGETS])
    axis.set_ylabel("QPS (mean ± sample stdev)")
    axis.set_title("Four-node same-recall throughput")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(ncol=3, loc="upper right")
    qps_ceiling = max(
        float(row["qps_mean"]) + float(row["qps_stdev"])
        for row in rows
    )
    axis.set_ylim(0, qps_ceiling * 1.18)
    fig.tight_layout()
    save_figure(fig, output_dir, "same_recall_qps")
    plt.close(fig)

    # P95/P99 grouped by target and method.
    labels = [f"{float(row['target_recall']):.2f}\n{row['method']}" for row in rows]
    positions = np.arange(len(rows), dtype=float)
    tail_width = 0.36
    fig, axis = plt.subplots(figsize=(10.5, 5.2))
    p95 = [float(row["p95_ms_mean"]) for row in rows]
    p99 = [float(row["p99_ms_mean"]) for row in rows]
    p95_err = [float(row["p95_ms_stdev"]) for row in rows]
    p99_err = [float(row["p99_ms_stdev"]) for row in rows]
    axis.bar(
        positions - tail_width / 2,
        p95,
        tail_width,
        yerr=p95_err,
        capsize=3,
        color="#4c78a8",
        label="P95",
    )
    p99_bars = axis.bar(
        positions + tail_width / 2,
        p99,
        tail_width,
        yerr=p99_err,
        capsize=3,
        color="#e45756",
        label="P99",
    )
    for position, bar, row in zip(positions, p99_bars, rows):
        annotate_recall(axis, float(position), float(bar.get_height()), row, max(p99) * 0.015)
    axis.set_xticks(positions, labels)
    axis.set_ylabel("Batch latency (ms; mean ± sample stdev)")
    axis.set_title("Four-node same-recall tail latency")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    tail_ceiling = max(
        float(row["p99_ms_mean"]) + float(row["p99_ms_stdev"])
        for row in rows
    )
    axis.set_ylim(0, tail_ceiling * 1.14)
    fig.tight_layout()
    save_figure(fig, output_dir, "same_recall_tail_latency")
    plt.close(fig)

    # Logical work: visited shards and EF-sum.
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.0))
    visited = [float(row["avg_visited_shards"]) for row in rows]
    ef_sum = [float(row["ef_sum_per_query"]) for row in rows]
    for axis, values, ylabel, title in (
        (axes[0], visited, "Visited logical shards/query", "Logical shard fan-out"),
        (axes[1], ef_sum, "Estimated EF-sum/query", "Assigned lower-search work"),
    ):
        bars = axis.bar(
            positions,
            values,
            color=[METHOD_COLORS[row["method"]] for row in rows],
        )
        for position, bar, row in zip(positions, bars, rows):
            annotate_recall(axis, float(position), float(bar.get_height()), row, max(values) * 0.015)
        axis.set_xticks(positions, labels)
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
        axis.set_ylim(0, max(values) * 1.20)
    fig.suptitle("Four-node same-recall logical work")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_figure(fig, output_dir, "same_recall_work")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    main_run_dir = args.main_run_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_aggregate(main_run_dir)
    csv_path = output_dir / "final_six_point_aggregate.csv"
    write_csv(csv_path, rows)
    write_plots(output_dir, rows)
    print(f"wrote {csv_path}")
    for row in rows:
        print(
            f"{float(row['target_recall']):.2f} {row['method']}: "
            f"recall={float(row['recall_mean']):.6f} "
            f"qps={float(row['qps_mean']):.3f}±{float(row['qps_stdev']):.3f} "
            f"status={row['recall_match_status']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
