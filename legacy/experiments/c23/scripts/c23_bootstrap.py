#!/usr/bin/env python3
"""Compute paired query-bootstrap CIs for Orion versus K-Means C23 headline metrics."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.c23.scripts import c23_e1, c23_e2, c23_protocol


def paired_bootstrap(
    orion: np.ndarray,
    kmeans: np.ndarray,
    *,
    seed: int,
    replicates: int,
    batch_size: int = 100,
) -> tuple[float, float, float]:
    orion = np.asarray(orion, dtype=np.float64)
    kmeans = np.asarray(kmeans, dtype=np.float64)
    if orion.shape != kmeans.shape or orion.ndim != 1 or not len(orion):
        raise ValueError("paired bootstrap inputs must be equal non-empty vectors")
    difference = orion - kmeans
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(replicates, dtype=np.float64)
    for start in range(0, replicates, batch_size):
        end = min(start + batch_size, replicates)
        indices = rng.integers(0, len(difference), size=(end - start, len(difference)))
        bootstrap[start:end] = np.mean(difference[indices], axis=1)
    return (
        float(np.mean(difference)),
        float(np.percentile(bootstrap, 2.5)),
        float(np.percentile(bootstrap, 97.5)),
    )


def write_csv_new(path: str | Path, rows: list[dict[str, Any]]) -> None:
    target = Path(path).expanduser().resolve()
    if target.exists():
        raise FileExistsError(f"refusing to overwrite: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_metric_csv(
    path: str | Path,
    metrics: tuple[str, ...],
) -> dict[tuple[str, int, int], dict[str, float]]:
    result = {}
    with Path(path).expanduser().resolve().open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            method = str(row["partition_method"])
            if method not in ("kmeans", "orion"):
                continue
            key = (method, int(row["logical_shards"]), int(row["query_id"]))
            result[key] = {metric: float(row[metric]) for metric in metrics}
    return result


def query_twcut(
    trace_path: str | Path,
    assignments: np.ndarray,
    measurement_start: int,
    measurement_count: int,
) -> np.ndarray:
    values = np.empty((len(assignments), measurement_count), dtype=np.float64)
    with Path(trace_path).expanduser().resolve().open("r", encoding="utf-8") as handle:
        for row_index, line in enumerate(handle):
            trace = json.loads(line)
            if int(trace["query_id"]) != measurement_start + row_index:
                raise ValueError("unexpected query sequence in reference traces")
            keys = c23_e2.trace_edge_keys(trace["visited_edges"], assignments.shape[1])
            if not len(keys):
                raise ValueError("trace contains no unique edges")
            sources, destinations = c23_e2.decode_edge_keys(keys)
            values[:, row_index] = np.mean(
                assignments[:, sources] != assignments[:, destinations],
                axis=1,
            )
    if row_index + 1 != measurement_count:
        raise ValueError("reference trace count differs from measurement count")
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(c23_protocol.DATASETS), required=True)
    parser.add_argument("--input-manifest", required=True)
    parser.add_argument("--reference-run", required=True)
    parser.add_argument("--e2-per-query", required=True)
    parser.add_argument("--e4-per-query", required=True)
    parser.add_argument("--replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--audit-output", required=True)
    args = parser.parse_args()

    input_manifest_path = Path(args.input_manifest).expanduser().resolve()
    input_manifest = json.loads(input_manifest_path.read_text(encoding="utf-8"))
    selected = [
        row
        for row in input_manifest["configurations"]
        if row["partition_method"] in ("kmeans", "orion")
    ]
    selected.sort(
        key=lambda row: (
            int(row["logical_shards"]),
            0 if row["partition_method"] == "kmeans" else 1,
        )
    )
    assignment_matrix = np.stack(
        [np.memmap(row["raw_assignment_path"], dtype="<u4", mode="r") for row in selected]
    )
    reference_manifest = json.loads(
        (Path(args.reference_run).expanduser().resolve() / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    measurement_start = int(reference_manifest["measurement_start"])
    measurement_count = int(reference_manifest["measurement_query_count"])
    twcut = query_twcut(
        reference_manifest["files"]["measurement_traces"],
        assignment_matrix,
        measurement_start,
        measurement_count,
    )
    twcut_index = {
        (str(row["partition_method"]), int(row["logical_shards"])): index
        for index, row in enumerate(selected)
    }
    e2 = load_metric_csv(
        args.e2_per_query,
        ("path_shard_count", "path_transition_count"),
    )
    e4_metrics = (
        "common_ef_mean_local_target_recall",
        "common_ef_mean_relevant_distance_computations",
        "P_exact",
        "P_HNSW",
        "Delta_P",
        "W90",
    )
    e4 = load_metric_csv(args.e4_per_query, e4_metrics)
    metric_directions = {
        "query_traversal_weighted_edge_cut": "lower_favors_orion",
        "path_shard_count": "lower_favors_orion",
        "path_transition_count": "lower_favors_orion",
        "common_ef_mean_local_target_recall": "higher_favors_orion",
        "common_ef_mean_relevant_distance_computations": "lower_favors_orion",
        "P_exact": "lower_favors_orion",
        "P_HNSW": "lower_favors_orion",
        "Delta_P": "lower_favors_orion",
        "W90": "lower_favors_orion",
    }
    rows = []
    for logical_shards in c23_protocol.LOGICAL_SHARDS:
        query_ids = np.arange(
            measurement_start,
            measurement_start + measurement_count,
            dtype=np.int64,
        )
        metric_values: dict[str, tuple[np.ndarray, np.ndarray]] = {
            "query_traversal_weighted_edge_cut": (
                twcut[twcut_index[("orion", logical_shards)]],
                twcut[twcut_index[("kmeans", logical_shards)]],
            ),
            "path_shard_count": (
                np.asarray(
                    [e2[("orion", logical_shards, int(query))]["path_shard_count"] for query in query_ids]
                ),
                np.asarray(
                    [e2[("kmeans", logical_shards, int(query))]["path_shard_count"] for query in query_ids]
                ),
            ),
            "path_transition_count": (
                np.asarray(
                    [e2[("orion", logical_shards, int(query))]["path_transition_count"] for query in query_ids]
                ),
                np.asarray(
                    [e2[("kmeans", logical_shards, int(query))]["path_transition_count"] for query in query_ids]
                ),
            ),
        }
        for metric in e4_metrics:
            metric_values[metric] = (
                np.asarray([e4[("orion", logical_shards, int(query))][metric] for query in query_ids]),
                np.asarray([e4[("kmeans", logical_shards, int(query))][metric] for query in query_ids]),
            )
        for metric, (orion_values, kmeans_values) in metric_values.items():
            observed, ci_low, ci_high = paired_bootstrap(
                orion_values,
                kmeans_values,
                seed=args.seed + logical_shards,
                replicates=args.replicates,
            )
            direction = metric_directions[metric]
            if direction == "lower_favors_orion":
                conclusion = "ORION_FAVORED" if ci_high < 0 else "KMEANS_FAVORED" if ci_low > 0 else "INCONCLUSIVE"
            else:
                conclusion = "ORION_FAVORED" if ci_low > 0 else "KMEANS_FAVORED" if ci_high < 0 else "INCONCLUSIVE"
            rows.append(
                {
                    "dataset": args.dataset,
                    "logical_shards": logical_shards,
                    "metric": metric,
                    "difference_definition": "Orion minus K-Means",
                    "favorable_direction": direction,
                    "orion_mean": float(np.mean(orion_values)),
                    "kmeans_mean": float(np.mean(kmeans_values)),
                    "mean_difference": observed,
                    "bootstrap_ci95_low": ci_low,
                    "bootstrap_ci95_high": ci_high,
                    "paired_query_count": len(orion_values),
                    "bootstrap_replicates": args.replicates,
                    "bootstrap_seed": args.seed + logical_shards,
                    "conclusion": conclusion,
                }
            )
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output}")
    output.mkdir(parents=True, exist_ok=False)
    result_path = output / "paired_bootstrap_ci.csv"
    write_csv_new(result_path, rows)
    manifest = {
        "protocol_version": c23_protocol.PROTOCOL_VERSION,
        "timestamp": c23_e1.utc_timestamp(),
        "git_commit": c23_protocol.git_commit(REPO_ROOT),
        "dataset": args.dataset,
        "comparison": "Orion versus K-Means paired over identical measurement queries",
        "configuration_count": len(c23_protocol.LOGICAL_SHARDS),
        "metric_count": len(metric_directions),
        "bootstrap_replicates": args.replicates,
        "checks": {
            "complete_query_pairing": "PASS",
            "query_twcut_recomputed_from_raw_traces": "PASS",
            "deterministic_bootstrap": "PASS",
        },
        "files": {
            "paired_bootstrap_ci": str(result_path),
            "paired_bootstrap_ci_sha256": c23_e1.sha256_path(result_path),
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
            "rows": rows,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
