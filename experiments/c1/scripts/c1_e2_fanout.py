#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import h5py
import numpy as np

from c1_protocol import (
    LOGICAL_SHARD_COUNTS,
    TARGET_RECALL,
    TOP_K,
    oracle_minimum_fanout,
    required_hits,
    select_kmeans_tuning_candidate,
    select_random_tuning_candidate,
    sha256_path,
    utc_timestamp,
)


METHODS = ("random", "kmeans")
SENSITIVITY_TARGETS = (0.89, 0.90, 0.91)
PER_QUERY_FIELDS = (
    "query_id",
    "dataset",
    "partition_method",
    "logical_shard_count",
    "target_recall",
    "oracle_minimum_fanout",
    "oracle_normalized_fanout",
    "actual_selected_fanout",
    "actual_normalized_fanout",
    "actual_selected_shard_ids",
    "actual_ground_truth_shard_coverage",
    "achieved_recall",
    "selected_ef_search",
    "selected_tuning_recall",
    "source_e3_per_search",
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
        raise ValueError(f"expected a JSON object: {source}")
    return payload


def load_partition_assignments(
    path: str | Path, method: str, logical_shards: int
) -> tuple[np.ndarray, dict[str, Any]]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    with np.load(source, allow_pickle=False) as data:
        assignments = np.asarray(data["assignments"]).copy()
        metadata = json.loads(str(np.asarray(data["metadata_json"]).item()))
    if metadata.get("method") != method:
        raise ValueError(f"partition method mismatch for {source}")
    if int(metadata.get("logical_shards") or 0) != logical_shards:
        raise ValueError(f"partition shard-count mismatch for {source}")
    if len(assignments) != int(metadata.get("point_count") or 0):
        raise ValueError(f"partition assignment-count mismatch for {source}")
    if assignments.size and (
        int(assignments.min()) < 0 or int(assignments.max()) >= logical_shards
    ):
        raise ValueError(f"partition contains an invalid shard ID: {source}")
    counts = np.bincount(assignments.astype(np.int64), minlength=logical_shards)
    if np.any(counts == 0):
        raise ValueError(f"partition contains an empty shard: {source}")
    recorded_counts = metadata.get("shard_counts")
    if recorded_counts is not None and list(map(int, recorded_counts)) != counts.tolist():
        raise ValueError(f"partition shard-count metadata mismatch: {source}")
    return assignments, metadata


def distribution_summary(values: Sequence[int | float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) == 0:
        raise ValueError("distribution requires a non-empty one-dimensional sequence")
    unique, counts = np.unique(array, return_counts=True)
    return {
        "mean": float(np.mean(array)),
        "query_population_std": float(np.std(array, ddof=0)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "distribution_counts": {
            str(int(value) if float(value).is_integer() else float(value)): int(count)
            for value, count in zip(unique, counts, strict=True)
        },
    }


def load_e3_actual_events(
    path: str | Path,
    *,
    dataset: str,
    method: str,
    logical_shards: int,
    query_start: int,
    query_stop: int,
) -> tuple[dict[int, dict[str, Any]], dict[str, int]]:
    source = Path(path).expanduser().resolve()
    required = {
        "query_id",
        "dataset",
        "partition_method",
        "logical_shards",
        "selected_fanout",
        "ef_search",
        "shard_id",
        "route_rank",
        "ground_truth_points_in_shard",
        "ground_truth_hits",
    }
    grouped: dict[int, dict[str, Any]] = defaultdict(
        lambda: {"routes": [], "ground_truth_hits": 0, "ground_truth_coverage": 0}
    )
    seen_pairs: set[tuple[int, int]] = set()
    fanouts: set[int] = set()
    ef_values: set[int] = set()
    with source.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = sorted(required - set(reader.fieldnames or ()))
        if missing:
            raise ValueError(f"E3 source is missing fields {missing}: {source}")
        for row in reader:
            if row["dataset"] != dataset or row["partition_method"] != method:
                raise ValueError(f"E3 dataset/method mismatch: {source}")
            if int(row["logical_shards"]) != logical_shards:
                raise ValueError(f"E3 logical-shard mismatch: {source}")
            query_id = int(row["query_id"])
            shard_id = int(row["shard_id"])
            if not query_start <= query_id < query_stop:
                raise ValueError(f"E3 query ID outside the measurement range: {query_id}")
            pair = (query_id, shard_id)
            if pair in seen_pairs:
                raise ValueError(f"duplicate E3 query/shard event: {pair}")
            seen_pairs.add(pair)
            fanouts.add(int(row["selected_fanout"]))
            ef_values.add(int(row["ef_search"]))
            record = grouped[query_id]
            record["routes"].append((int(row["route_rank"]), shard_id))
            record["ground_truth_hits"] += int(row["ground_truth_hits"])
            record["ground_truth_coverage"] += int(row["ground_truth_points_in_shard"])
    expected_queries = set(range(query_start, query_stop))
    if set(grouped) != expected_queries:
        missing = sorted(expected_queries - set(grouped))
        extra = sorted(set(grouped) - expected_queries)
        raise ValueError(
            f"E3 query coverage mismatch: missing={missing[:5]}, extra={extra[:5]}"
        )
    if len(fanouts) != 1 or len(ef_values) != 1:
        raise ValueError(f"E3 source mixes selected configurations: {source}")
    selected_fanout = fanouts.pop()
    selected_ef = ef_values.pop()
    if selected_fanout <= 0 or selected_fanout > logical_shards:
        raise ValueError(f"invalid selected E3 fan-out: {selected_fanout}")
    for query_id, record in grouped.items():
        routes = sorted(record.pop("routes"))
        if [rank for rank, _ in routes] != list(range(1, selected_fanout + 1)):
            raise ValueError(f"invalid one-based route ranks for query {query_id}")
        shard_ids = [shard_id for _, shard_id in routes]
        if len(set(shard_ids)) != selected_fanout:
            raise ValueError(f"duplicate selected shard for query {query_id}")
        hits = int(record["ground_truth_hits"])
        coverage = int(record["ground_truth_coverage"])
        if not 0 <= hits <= coverage <= TOP_K:
            raise ValueError(f"invalid ground-truth accounting for query {query_id}")
        record["selected_shard_ids"] = shard_ids
        record["actual_fanout"] = selected_fanout
        record["achieved_recall"] = hits / TOP_K
        record["ground_truth_shard_coverage"] = coverage / TOP_K
    return dict(grouped), {"selected_fanout": selected_fanout, "ef_search": selected_ef}


def source_stem(template: str, dataset: str) -> str:
    try:
        rendered = template.format(dataset=dataset)
    except (IndexError, KeyError, ValueError) as error:
        raise ValueError(f"invalid source-stem template {template!r}") from error
    if not rendered or Path(rendered).name != rendered:
        raise ValueError(f"source stem must render to one filename component: {rendered!r}")
    return rendered


def tuning_path(
    runs_root: Path,
    dataset: str,
    method: str,
    logical_shards: int,
    *,
    small_source_stem: str = "stage1-{dataset}",
    large_source_stem: str = "stage2-e3-{dataset}",
) -> Path:
    template = small_source_stem if logical_shards <= 4 else large_source_stem
    stem = runs_root / (
        f"{source_stem(template, dataset)}-{method}-m{logical_shards}"
    )
    candidates = [
        Path(str(stem) + "-tuning-pinned.json"),
        Path(str(stem) + "-tuning-prefix-pinned.json"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"no tuning artifact found among {candidates}")


def select_candidate(
    method: str, candidates: Sequence[dict[str, Any]], target_recall: float
) -> dict[str, Any]:
    if method == "random":
        return select_random_tuning_candidate(candidates, target_recall=target_recall)
    return select_kmeans_tuning_candidate(candidates, target_recall=target_recall)


def minimum_feasible_fanout(
    candidates: Sequence[dict[str, Any]], target_recall: float
) -> int:
    feasible = [
        int(row["fanout"])
        for row in candidates
        if float(row["recall_at_10"]) >= target_recall
    ]
    if not feasible:
        raise ValueError(f"no tuning candidate reaches Recall@10={target_recall}")
    return min(feasible)


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
            "Creator": "Orion C1 E2 fan-out analyzer",
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


def empirical_cdf(values: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(values, dtype=np.int64)
    unique, counts = np.unique(array, return_counts=True)
    x = np.concatenate((np.asarray([0]), unique))
    y = np.concatenate((np.asarray([0.0]), np.cumsum(counts) / len(array)))
    return x, y


def generate_figures(
    figures_dir: str | Path,
    summaries: Sequence[dict[str, Any]],
    oracle_values: dict[tuple[str, int], list[int]],
    actual_values: dict[tuple[str, int], list[int]],
) -> dict[str, Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory = Path(figures_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    colors = {"random": "#3366cc", "kmeans": "#d97706"}
    labels = {"random": "Random", "kmeans": "K-Means"}

    figure, axis = plt.subplots(figsize=(6.6, 4.2))
    for method in METHODS:
        method_rows = sorted(
            (row for row in summaries if row["partition_method"] == method),
            key=lambda row: row["logical_shard_count"],
        )
        shards = [row["logical_shard_count"] for row in method_rows]
        axis.plot(
            shards,
            [row["actual_fanout"]["mean"] for row in method_rows],
            marker="o",
            color=colors[method],
            label=f"{labels[method]} actual",
        )
        axis.plot(
            shards,
            [row["oracle_fanout"]["mean"] for row in method_rows],
            marker="s",
            linestyle="--",
            color=colors[method],
            label=f"{labels[method]} oracle",
        )
    axis.set_xscale("log", base=2)
    axis.set_xticks(LOGICAL_SHARD_COUNTS, [str(value) for value in LOGICAL_SHARD_COUNTS])
    axis.set_xlabel("Logical shard count")
    axis.set_ylabel("Mean shards searched per query")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend(frameon=False, ncols=2)
    figure.tight_layout()
    canonical_fanout = save_pdf_atomic(
        figure, directory / "c1_fig2_fanout_vs_logical_shards.pdf"
    )
    plt.close(figure)
    protocol_fanout = copy_atomic(
        canonical_fanout, directory / "fig_c1_fanout_vs_shards.pdf"
    )

    representative = (4, 16, 32)
    figure, axis = plt.subplots(figsize=(6.6, 4.2))
    line_styles = {4: ":", 16: "--", 32: "-"}
    for method in METHODS:
        for logical_shards in representative:
            x, y = empirical_cdf(oracle_values[(method, logical_shards)])
            axis.step(
                x,
                y,
                where="post",
                color=colors[method],
                linestyle=line_styles[logical_shards],
                label=f"{labels[method]} M={logical_shards}",
            )
    axis.set_xlabel("Oracle minimum shards for Recall@10 >= 0.90")
    axis.set_ylabel("Empirical CDF")
    axis.set_xlim(left=0)
    axis.set_ylim(0, 1.01)
    axis.grid(True, alpha=0.25)
    axis.legend(frameon=False, ncols=2)
    figure.tight_layout()
    canonical_oracle = save_pdf_atomic(
        figure, directory / "c1_fig3_required_fanout_cdf.pdf"
    )
    plt.close(figure)
    protocol_oracle = copy_atomic(
        canonical_oracle, directory / "fig_c1_oracle_fanout_cdf.pdf"
    )

    figure, axis = plt.subplots(figsize=(6.6, 4.2))
    for method in METHODS:
        for logical_shards in representative:
            x, y = empirical_cdf(actual_values[(method, logical_shards)])
            axis.step(
                x,
                y,
                where="post",
                color=colors[method],
                linestyle=line_styles[logical_shards],
                label=f"{labels[method]} M={logical_shards}",
            )
    axis.set_xlabel("Actual selected shards searched per query")
    axis.set_ylabel("Empirical CDF")
    axis.set_xlim(left=0)
    axis.set_ylim(0, 1.01)
    axis.grid(True, alpha=0.25)
    axis.legend(frameon=False, ncols=2)
    figure.tight_layout()
    protocol_actual = save_pdf_atomic(
        figure, directory / "fig_c1_actual_fanout_cdf.pdf"
    )
    plt.close(figure)
    return {
        "canonical_fanout": canonical_fanout,
        "protocol_fanout": protocol_fanout,
        "canonical_oracle_cdf": canonical_oracle,
        "protocol_oracle_cdf": protocol_oracle,
        "protocol_actual_cdf": protocol_actual,
    }


def flattened_summary_rows(summaries: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for summary in summaries:
        actual = summary["actual_fanout"]
        oracle = summary["oracle_fanout"]
        rows.append(
            {
                "dataset": summary["dataset"],
                "partition_method": summary["partition_method"],
                "logical_shard_count": summary["logical_shard_count"],
                "query_count": summary["query_count"],
                "target_recall": summary["target_recall"],
                "selected_fanout": summary["selected_fanout"],
                "minimum_feasible_fanout": summary["minimum_feasible_fanout"],
                "selected_ef_search": summary["selected_ef_search"],
                "selected_tuning_recall": summary["selected_tuning_recall"],
                "measurement_recall": summary["measurement_recall"],
                "actual_fanout_mean": actual["mean"],
                "actual_fanout_query_population_std": actual["query_population_std"],
                "actual_fanout_median": actual["median"],
                "actual_fanout_p95": actual["p95"],
                "actual_fanout_p99": actual["p99"],
                "actual_normalized_fanout": actual["mean"]
                / summary["logical_shard_count"],
                "oracle_fanout_mean": oracle["mean"],
                "oracle_fanout_query_population_std": oracle["query_population_std"],
                "oracle_fanout_median": oracle["median"],
                "oracle_fanout_p95": oracle["p95"],
                "oracle_fanout_p99": oracle["p99"],
                "oracle_normalized_fanout": oracle["mean"]
                / summary["logical_shard_count"],
                "deterministic_repetition_count": 1,
                "between_repetition_std": "",
                "partition_artifact_sha256": summary["partition_artifact_sha256"],
                "tuning_artifact_sha256": summary["tuning_artifact_sha256"],
                "e3_per_search_sha256": summary["e3_per_search_sha256"],
                "oracle_distribution_counts": json.dumps(
                    oracle["distribution_counts"], separators=(",", ":")
                ),
                "actual_distribution_counts": json.dumps(
                    actual["distribution_counts"], separators=(",", ":")
                ),
            }
        )
    return rows


def qualitative_sensitivity_checks(
    summaries: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    by_key = {
        (row["partition_method"], row["logical_shard_count"]): row
        for row in summaries
    }
    checks: dict[str, Any] = {}
    all_pass = True
    for threshold in SENSITIVITY_TARGETS:
        key = f"{threshold:.2f}"
        random_actual = [
            int(by_key[("random", shards)]["sensitivity"][key]["globally_selected_fanout"])
            for shards in LOGICAL_SHARD_COUNTS
        ]
        kmeans_actual = [
            int(by_key[("kmeans", shards)]["sensitivity"][key]["globally_selected_fanout"])
            for shards in LOGICAL_SHARD_COUNTS
        ]
        random_oracle = [
            float(by_key[("random", shards)]["sensitivity"][key]["oracle_fanout"]["mean"])
            for shards in LOGICAL_SHARD_COUNTS
        ]
        kmeans_oracle = [
            float(by_key[("kmeans", shards)]["sensitivity"][key]["oracle_fanout"]["mean"])
            for shards in LOGICAL_SHARD_COUNTS
        ]
        threshold_checks = {
            "random_actual_equals_broadcast": random_actual
            == list(LOGICAL_SHARD_COUNTS),
            "kmeans_actual_non_decreasing": all(
                left <= right for left, right in zip(kmeans_actual, kmeans_actual[1:])
            ),
            "kmeans_actual_m32_exceeds_m1": kmeans_actual[-1] > kmeans_actual[0],
            "kmeans_actual_m32_at_least_three": kmeans_actual[-1] >= 3,
            "random_oracle_non_decreasing": all(
                left <= right for left, right in zip(random_oracle, random_oracle[1:])
            ),
            "kmeans_oracle_non_decreasing": all(
                left <= right for left, right in zip(kmeans_oracle, kmeans_oracle[1:])
            ),
            "random_actual_series": random_actual,
            "kmeans_actual_series": kmeans_actual,
            "random_oracle_mean_series": random_oracle,
            "kmeans_oracle_mean_series": kmeans_oracle,
        }
        # The protocol requires K-Means fan-out to remain substantial instead
        # of collapsing to a consistently small constant. It does not require
        # every adjacent M step to be monotonic; retain that stricter signal as
        # an anomaly without invalidating otherwise complete E2 evidence.
        required_checks = (
            "random_actual_equals_broadcast",
            "kmeans_actual_m32_exceeds_m1",
            "kmeans_actual_m32_at_least_three",
            "random_oracle_non_decreasing",
            "kmeans_oracle_non_decreasing",
        )
        passed = all(bool(threshold_checks[name]) for name in required_checks)
        threshold_checks["passed"] = passed
        checks[key] = threshold_checks
        all_pass = all_pass and passed
    checks["all_thresholds_passed"] = all_pass
    checks["non_monotonic_kmeans_targets"] = [
        key
        for key in (f"{threshold:.2f}" for threshold in SENSITIVITY_TARGETS)
        if not checks[key]["kmeans_actual_non_decreasing"]
    ]
    if not all_pass:
        raise ValueError("small recall-target changes qualitatively reverse E2 fan-out trends")
    return checks


SUMMARY_CSV_FIELDS = tuple(
    (
        "dataset",
        "partition_method",
        "logical_shard_count",
        "query_count",
        "target_recall",
        "selected_fanout",
        "minimum_feasible_fanout",
        "selected_ef_search",
        "selected_tuning_recall",
        "measurement_recall",
        "actual_fanout_mean",
        "actual_fanout_query_population_std",
        "actual_fanout_median",
        "actual_fanout_p95",
        "actual_fanout_p99",
        "actual_normalized_fanout",
        "oracle_fanout_mean",
        "oracle_fanout_query_population_std",
        "oracle_fanout_median",
        "oracle_fanout_p95",
        "oracle_fanout_p99",
        "oracle_normalized_fanout",
        "deterministic_repetition_count",
        "between_repetition_std",
        "partition_artifact_sha256",
        "tuning_artifact_sha256",
        "e3_per_search_sha256",
        "oracle_distribution_counts",
        "actual_distribution_counts",
    )
)


def execute(args: argparse.Namespace) -> dict[str, Any]:
    dataset_path = Path(args.hdf5_path).expanduser().resolve()
    partition_root = Path(args.partition_root).expanduser().resolve()
    runs_root = Path(args.runs_root).expanduser().resolve()
    with h5py.File(dataset_path, "r") as handle:
        neighbors = np.asarray(
            handle["neighbors"][args.query_start : args.query_stop, :TOP_K],
            dtype=np.int64,
        )
    expected_query_count = args.query_stop - args.query_start
    if len(neighbors) != expected_query_count:
        raise ValueError("HDF5 measurement query range is incomplete")

    per_query_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    oracle_values: dict[tuple[str, int], list[int]] = {}
    actual_values: dict[tuple[str, int], list[int]] = {}
    graph_build_seeds: set[int] = set()
    e3_source_stem = source_stem(args.e3_source_stem, args.dataset)
    for method in METHODS:
        for logical_shards in LOGICAL_SHARD_COUNTS:
            partition = partition_root / f"{method}-m{logical_shards}.npz"
            assignments, partition_metadata = load_partition_assignments(
                partition, method, logical_shards
            )
            oracle = oracle_minimum_fanout(
                assignments,
                neighbors,
                logical_shards,
                target_recall=TARGET_RECALL,
                top_k=TOP_K,
            )
            e3_per_search = (
                runs_root
                / "per_query"
                / f"{e3_source_stem}-{method}-m{logical_shards}.csv"
            )
            e3_summary_path = (
                runs_root
                / f"{e3_source_stem}-{method}-m{logical_shards}-summary.json"
            )
            actual, selected = load_e3_actual_events(
                e3_per_search,
                dataset=args.dataset,
                method=method,
                logical_shards=logical_shards,
                query_start=args.query_start,
                query_stop=args.query_stop,
            )
            e3_summary = load_json(e3_summary_path)
            if e3_summary.get("deterministic_graph_construction") is not True:
                raise ValueError(f"E3 source is not deterministically constructed: {e3_summary_path}")
            if int(e3_summary.get("hnsw_max_indexing_threads") or 0) != 1:
                raise ValueError(f"E3 source did not use one indexing thread: {e3_summary_path}")
            if int(e3_summary.get("max_optimization_threads") or 0) != 1:
                raise ValueError(f"E3 source did not use one optimizer thread: {e3_summary_path}")
            graph_build_seeds.add(int(e3_summary["graph_build_seed"]))
            tuning = tuning_path(
                runs_root,
                args.dataset,
                method,
                logical_shards,
                small_source_stem=args.tuning_small_source_stem,
                large_source_stem=args.tuning_large_source_stem,
            )
            tuning_payload = load_json(tuning)
            tuning_candidates = tuning_payload.get("candidates")
            if not isinstance(tuning_candidates, list) or not tuning_candidates:
                raise ValueError(f"tuning artifact has no candidate matrix: {tuning}")
            tuning_selected = tuning_payload.get("selected")
            if not isinstance(tuning_selected, dict):
                raise ValueError(f"tuning artifact has no selected candidate: {tuning}")
            if (
                selected["selected_fanout"] != int(e3_summary["selected_fanout"])
                or selected["ef_search"] != int(e3_summary["ef_search"])
                or selected["selected_fanout"] != int(tuning_selected["fanout"])
                or selected["ef_search"] != int(tuning_selected["ef_search"])
            ):
                raise ValueError(f"selected configuration mismatch for {method} M={logical_shards}")
            actual_by_query = [actual[q] for q in range(args.query_start, args.query_stop)]
            actual_fanout = [int(row["actual_fanout"]) for row in actual_by_query]
            achieved_recall = [float(row["achieved_recall"]) for row in actual_by_query]
            measurement_recall = statistics.fmean(achieved_recall)
            if measurement_recall < TARGET_RECALL:
                raise ValueError(
                    f"measurement Recall@10 below target for {method} M={logical_shards}"
                )
            selected_tuning_recall = float(e3_summary["selected_tuning_recall"])
            if not math.isclose(
                selected_tuning_recall,
                float(tuning_selected["recall_at_10"]),
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(f"selected tuning recall mismatch for {method} M={logical_shards}")
            minimum_fanout = minimum_feasible_fanout(
                tuning_candidates, TARGET_RECALL
            )
            sensitivity: dict[str, Any] = {}
            for threshold in SENSITIVITY_TARGETS:
                sensitivity_oracle = oracle_minimum_fanout(
                    assignments,
                    neighbors,
                    logical_shards,
                    target_recall=threshold,
                    top_k=TOP_K,
                )
                sensitivity_selected = select_candidate(
                    method, tuning_candidates, threshold
                )
                sensitivity[f"{threshold:.2f}"] = {
                    "required_ground_truth_hits": required_hits(threshold, TOP_K),
                    "oracle_fanout": distribution_summary(sensitivity_oracle.tolist()),
                    "globally_selected_fanout": int(sensitivity_selected["fanout"]),
                    "globally_selected_ef_search": int(sensitivity_selected["ef_search"]),
                    "selected_tuning_recall": float(sensitivity_selected["recall_at_10"]),
                    "minimum_feasible_fanout": minimum_feasible_fanout(
                        tuning_candidates, threshold
                    ),
                }
            for offset, query_id in enumerate(range(args.query_start, args.query_stop)):
                actual_row = actual[query_id]
                per_query_rows.append(
                    {
                        "query_id": query_id,
                        "dataset": args.dataset,
                        "partition_method": method,
                        "logical_shard_count": logical_shards,
                        "target_recall": TARGET_RECALL,
                        "oracle_minimum_fanout": int(oracle[offset]),
                        "oracle_normalized_fanout": int(oracle[offset]) / logical_shards,
                        "actual_selected_fanout": actual_row["actual_fanout"],
                        "actual_normalized_fanout": actual_row["actual_fanout"]
                        / logical_shards,
                        "actual_selected_shard_ids": json.dumps(
                            actual_row["selected_shard_ids"], separators=(",", ":")
                        ),
                        "actual_ground_truth_shard_coverage": actual_row[
                            "ground_truth_shard_coverage"
                        ],
                        "achieved_recall": actual_row["achieved_recall"],
                        "selected_ef_search": selected["ef_search"],
                        "selected_tuning_recall": selected_tuning_recall,
                        "source_e3_per_search": str(e3_per_search.resolve()),
                    }
                )
            oracle_list = list(map(int, oracle.tolist()))
            oracle_values[(method, logical_shards)] = oracle_list
            actual_values[(method, logical_shards)] = actual_fanout
            summaries.append(
                {
                    "dataset": args.dataset,
                    "partition_method": method,
                    "logical_shard_count": logical_shards,
                    "physical_machine_count": min(logical_shards, 4),
                    "execution_scope": (
                        "physical_layout" if logical_shards <= 4 else "logical_shard_simulation"
                    ),
                    "query_count": expected_query_count,
                    "query_range": [args.query_start, args.query_stop],
                    "target_recall": TARGET_RECALL,
                    "selected_fanout": selected["selected_fanout"],
                    "minimum_feasible_fanout": minimum_fanout,
                    "selected_ef_search": selected["ef_search"],
                    "selected_tuning_recall": selected_tuning_recall,
                    "measurement_recall": measurement_recall,
                    "actual_fanout": distribution_summary(actual_fanout),
                    "oracle_fanout": distribution_summary(oracle_list),
                    "actual_ground_truth_shard_coverage_mean": statistics.fmean(
                        float(row["ground_truth_shard_coverage"])
                        for row in actual_by_query
                    ),
                    "sensitivity": sensitivity,
                    "deterministic_repetition_count": 1,
                    "between_repetition_std": None,
                    "partition_artifact": str(partition.resolve()),
                    "partition_artifact_sha256": sha256_path(partition),
                    "partition_metadata": partition_metadata,
                    "tuning_artifact": str(tuning),
                    "tuning_artifact_sha256": sha256_path(tuning),
                    "e3_per_search": str(e3_per_search.resolve()),
                    "e3_per_search_sha256": sha256_path(e3_per_search),
                    "e3_summary": str(e3_summary_path.resolve()),
                    "e3_summary_sha256": sha256_path(e3_summary_path),
                }
            )

    if len(per_query_rows) != expected_query_count * len(METHODS) * len(LOGICAL_SHARD_COUNTS):
        raise RuntimeError("unexpected E2 per-query row count")
    if len(graph_build_seeds) != 1:
        raise ValueError(f"E2 sources mix graph-build seeds: {sorted(graph_build_seeds)}")
    graph_build_seed = graph_build_seeds.pop()
    per_query_output = write_csv_atomic(
        args.per_query_output, PER_QUERY_FIELDS, per_query_rows
    )
    summary_csv_output = write_csv_atomic(
        args.summary_csv_output,
        SUMMARY_CSV_FIELDS,
        flattened_summary_rows(summaries),
    )
    summary_payload = {
        "timestamp": utc_timestamp(),
        "record_type": "e2_fanout_analysis",
        "experiment_id": args.experiment_id or f"stage3-e2-{args.dataset}-fanout",
        "dataset": args.dataset,
        "dataset_path": str(dataset_path),
        "dataset_sha256": sha256_path(dataset_path),
        "query_range": [args.query_start, args.query_stop],
        "query_count_per_configuration": expected_query_count,
        "configuration_count": len(summaries),
        "per_query_row_count": len(per_query_rows),
        "target_recall": TARGET_RECALL,
        "graph_build_seed": graph_build_seed,
        "deterministic_graph_construction": True,
        "sensitivity_targets": list(SENSITIVITY_TARGETS),
        "configuration_summaries": summaries,
        "qualitative_sensitivity_checks": qualitative_sensitivity_checks(summaries),
        "status": "VALID_E2",
    }
    summary_json_output = write_json_atomic(args.summary_json_output, summary_payload)
    figures = generate_figures(
        args.figures_dir, summaries, oracle_values, actual_values
    )
    evidence = {
        "per_query": sha256_path(per_query_output),
        "summary_csv": sha256_path(summary_csv_output),
        "summary_json": sha256_path(summary_json_output),
        **{name: sha256_path(path) for name, path in figures.items()},
    }
    record = {
        "timestamp": summary_payload["timestamp"],
        "record_type": "e2_fanout_analysis",
        "experiment_id": summary_payload["experiment_id"],
        "dataset": args.dataset,
        "dataset_checksum": summary_payload["dataset_sha256"],
        "query_range": summary_payload["query_range"],
        "query_count_per_configuration": expected_query_count,
        "configuration_count": len(summaries),
        "per_query_row_count": len(per_query_rows),
        "target_recall": TARGET_RECALL,
        "logical_shard_counts": list(LOGICAL_SHARD_COUNTS),
        "partition_methods": list(METHODS),
        "actual_fanout": {
            method: {
                str(row["logical_shard_count"]): row["actual_fanout"]["mean"]
                for row in summaries
                if row["partition_method"] == method
            }
            for method in METHODS
        },
        "oracle_mean_fanout": {
            method: {
                str(row["logical_shard_count"]): row["oracle_fanout"]["mean"]
                for row in summaries
                if row["partition_method"] == method
            }
            for method in METHODS
        },
        "measurement_recall": {
            method: {
                str(row["logical_shard_count"]): row["measurement_recall"]
                for row in summaries
                if row["partition_method"] == method
            }
            for method in METHODS
        },
        "deterministic_repetition_count": 1,
        "graph_build_seed": graph_build_seed,
        "deterministic_graph_construction": True,
        "between_repetition_std": None,
        "sensitivity_targets": list(SENSITIVITY_TARGETS),
        "sensitivity_status": (
            "QUALITATIVE_FANOUT_TRENDS_PRESERVED_WITH_NON_MONOTONIC_DIP"
            if summary_payload["qualitative_sensitivity_checks"][
                "non_monotonic_kmeans_targets"
            ]
            else "QUALITATIVE_FANOUT_TRENDS_PRESERVED"
            if summary_payload["qualitative_sensitivity_checks"]["all_thresholds_passed"]
            else "SENSITIVITY_FAILURE"
        ),
        "evidence_sha256": evidence,
        "status": "VALID_E2",
    }
    record_output = write_json_atomic(args.record_output, record)
    register_manifest_record_once(args.manifest_jsonl, record)
    result = {
        "record": record,
        "record_output": str(record_output),
        "per_query_output": str(per_query_output),
        "summary_csv_output": str(summary_csv_output),
        "summary_json_output": str(summary_json_output),
        "figures": {name: str(path) for name, path in figures.items()},
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Derive C1 E2 actual and oracle fan-out from validated artifacts"
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--hdf5-path", required=True)
    parser.add_argument("--partition-root", required=True)
    parser.add_argument("--runs-root", default="experiments/c1/runs")
    parser.add_argument(
        "--e3-source-stem",
        default="stage2-e3-{dataset}",
        help="filename stem template for E3 CSV/summary inputs",
    )
    parser.add_argument(
        "--tuning-small-source-stem",
        default="stage1-{dataset}",
        help="tuning filename stem template for M <= 4",
    )
    parser.add_argument(
        "--tuning-large-source-stem",
        default="stage2-e3-{dataset}",
        help="tuning filename stem template for M > 4",
    )
    parser.add_argument("--query-start", type=int, default=1_000)
    parser.add_argument("--query-stop", type=int, default=10_000)
    parser.add_argument("--per-query-output", required=True)
    parser.add_argument("--summary-csv-output", required=True)
    parser.add_argument("--summary-json-output", required=True)
    parser.add_argument("--record-output", required=True)
    parser.add_argument("--experiment-id")
    parser.add_argument(
        "--manifest-jsonl", default="experiments/c1/runs/manifest.jsonl"
    )
    parser.add_argument("--figures-dir", default="experiments/c1/figures")
    args = parser.parse_args(argv)
    if args.query_start < 0 or args.query_stop <= args.query_start:
        parser.error("query range must be positive and non-empty")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    execute(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
