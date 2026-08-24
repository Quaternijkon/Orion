#!/usr/bin/env python3
"""Compute C23 E2 search-path preservation metrics from global HNSW traces."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
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


def compute_path_metrics(assignments: np.ndarray, visited_nodes: np.ndarray) -> tuple[int, int, float]:
    assignments = np.asarray(assignments, dtype=np.int32)
    visited_nodes = np.asarray(visited_nodes, dtype=np.int64)
    if not len(visited_nodes):
        raise ValueError("global trace has no visited nodes")
    shards = assignments[visited_nodes]
    path_shard_count = int(np.unique(shards).size)
    path_transition_count = int(np.count_nonzero(shards[1:] != shards[:-1]))
    dominant_shard_concentration = float(
        np.max(np.bincount(shards.astype(np.int64))) / len(shards)
    )
    return path_shard_count, path_transition_count, dominant_shard_concentration


def trace_edge_keys(visited_edges: list[dict[str, Any]], point_count: int) -> np.ndarray:
    if not visited_edges:
        return np.empty(0, dtype=np.uint64)
    sources = np.fromiter(
        (int(edge["source_node"]) for edge in visited_edges),
        dtype=np.uint64,
        count=len(visited_edges),
    )
    destinations = np.fromiter(
        (int(edge["destination_node"]) for edge in visited_edges),
        dtype=np.uint64,
        count=len(visited_edges),
    )
    if int(np.max(sources)) >= point_count or int(np.max(destinations)) >= point_count:
        raise ValueError("trace edge endpoint lies outside the point range")
    return np.unique((sources << np.uint64(32)) | destinations)


def decode_edge_keys(keys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    keys = np.asarray(keys, dtype=np.uint64)
    return (
        (keys >> np.uint64(32)).astype(np.int64),
        (keys & np.uint64(0xFFFF_FFFF)).astype(np.int64),
    )


def cut_counts(
    assignment_matrix: np.ndarray,
    keys: np.ndarray,
    *,
    chunk_size: int = 1_000_000,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    result = np.zeros(len(assignment_matrix), dtype=np.uint64)
    if weights is not None and len(weights) != len(keys):
        raise ValueError("edge weights differ from key count")
    for start in range(0, len(keys), chunk_size):
        end = min(start + chunk_size, len(keys))
        sources, destinations = decode_edge_keys(keys[start:end])
        cut = assignment_matrix[:, sources] != assignment_matrix[:, destinations]
        if weights is None:
            result += np.count_nonzero(cut, axis=1).astype(np.uint64)
        else:
            result += np.sum(
                cut.astype(np.uint64) * weights[start:end][None, :],
                axis=1,
                dtype=np.uint64,
            )
    return result


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


def save_figures(
    summary_rows: list[dict[str, Any]],
    hot_rows: list[dict[str, Any]],
    path_values: list[list[int]],
    configurations: list[dict[str, Any]],
    figure_dir: str | Path,
) -> dict[str, str]:
    target = Path(figure_dir).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    paths = {
        "traversal_weighted_cut": target / "fig_c23_traversal_weighted_cut.pdf",
        "path_shard_count_cdf": target / "fig_c23_path_shard_count_cdf.pdf",
        "hot_edge_retention": target / "fig_c23_hot_edge_retention.pdf",
    }
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite figures: {existing}")

    fig, axis = plt.subplots(figsize=(6.6, 4.2))
    for method in c23_e1.METHOD_ORDER:
        selected = sorted(
            (row for row in summary_rows if row["partition_method"] == method),
            key=lambda row: int(row["logical_shards"]),
        )
        axis.plot(
            [int(row["logical_shards"]) for row in selected],
            [float(row["traversal_weighted_edge_cut"]) for row in selected],
            label=c23_e1.METHOD_LABELS[method],
            linewidth=1.8,
            markersize=5,
            **c23_e1.METHOD_STYLES[method],
        )
    axis.set_xlabel("Logical shard count")
    axis.set_ylabel("Traversal-weighted edge-cut ratio")
    axis.set_xticks(c23_protocol.LOGICAL_SHARDS)
    axis.grid(True, alpha=0.25)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(paths["traversal_weighted_cut"])
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(11.0, 6.6), sharey=True)
    for panel, logical_shards in zip(axes.flat, c23_protocol.LOGICAL_SHARDS):
        for config_index, configuration in enumerate(configurations):
            if int(configuration["logical_shards"]) != logical_shards:
                continue
            values = np.sort(np.asarray(path_values[config_index], dtype=np.int64))
            y = np.arange(1, len(values) + 1, dtype=np.float64) / len(values)
            method = str(configuration["method"])
            panel.step(
                values,
                y,
                where="post",
                label=c23_e1.METHOD_LABELS[method],
                linewidth=1.4,
                linestyle=c23_e1.METHOD_STYLES[method]["linestyle"],
            )
        panel.set_title(f"M={logical_shards}")
        panel.set_xlabel("Unique shards on global search path")
        panel.grid(True, alpha=0.2)
    axes[0, 0].set_ylabel("CDF")
    axes[1, 0].set_ylabel("CDF")
    axes[1, 2].axis("off")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right", bbox_to_anchor=(0.98, 0.08), frameon=False)
    fig.tight_layout()
    fig.savefig(paths["path_shard_count_cdf"])
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(11.0, 6.6), sharex=True, sharey=True)
    scopes = ("top_1pct", "top_5pct", "top_10pct", "all_observed")
    scope_labels = ("Top 1%", "Top 5%", "Top 10%", "All observed")
    for panel, logical_shards in zip(axes.flat, c23_protocol.LOGICAL_SHARDS):
        for method in c23_e1.METHOD_ORDER:
            selected = {
                str(row["edge_scope"]): row
                for row in hot_rows
                if row["partition_method"] == method
                and int(row["logical_shards"]) == logical_shards
            }
            panel.plot(
                range(len(scopes)),
                [float(selected[scope]["edge_retention_ratio"]) for scope in scopes],
                label=c23_e1.METHOD_LABELS[method],
                linewidth=1.4,
                markersize=4,
                **c23_e1.METHOD_STYLES[method],
            )
        panel.set_title(f"M={logical_shards}")
        panel.set_xticks(range(len(scopes)), scope_labels, rotation=20)
        panel.grid(True, alpha=0.2)
    axes[0, 0].set_ylabel("Edge retention ratio")
    axes[1, 0].set_ylabel("Edge retention ratio")
    axes[1, 2].axis("off")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right", bbox_to_anchor=(0.98, 0.08), frameon=False)
    fig.tight_layout()
    fig.savefig(paths["hot_edge_retention"])
    plt.close(fig)
    return {name: str(path) for name, path in paths.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(c23_protocol.DATASETS), required=True)
    parser.add_argument("--reference-run", required=True)
    parser.add_argument("--baseline-root", required=True)
    parser.add_argument("--orion-build-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--figure-dir", required=True)
    parser.add_argument("--audit-output", required=True)
    args = parser.parse_args()

    spec = c23_protocol.DATASETS[args.dataset]
    reference_root = Path(args.reference_run).expanduser().resolve()
    reference_manifest_path = reference_root / "manifest.json"
    reference_manifest = json.loads(reference_manifest_path.read_text(encoding="utf-8"))
    point_count = int(reference_manifest["point_count"])
    trace_path = Path(reference_manifest["files"]["measurement_traces"])
    if c23_e1.sha256_path(trace_path) != str(
        reference_manifest["files"]["measurement_traces_sha256"]
    ):
        raise ValueError("measurement trace checksum mismatch")
    matrix = c23_e1.load_partition_matrix(
        dataset=args.dataset,
        point_count=point_count,
        dataset_sha256=spec.sha256,
        baseline_root=args.baseline_root,
        orion_manifest_path=args.orion_build_manifest,
    )
    configurations = [
        {
            "method": row["method"],
            "logical_shards": row["logical_shards"],
            "partition_seed": int(row["metadata"]["partition_seed"]),
            "artifact_path": row["artifact_path"],
            "artifact_sha256": row["artifact_sha256"],
        }
        for row in matrix
    ]
    assignment_matrix = np.stack([row["assignments"] for row in matrix])
    configuration_count = len(configurations)
    weighted_cut_numerator = np.zeros(configuration_count, dtype=np.uint64)
    trace_edge_incidence_count = 0
    per_query_rows: list[dict[str, Any]] = []
    path_values: list[list[int]] = [[] for _ in range(configuration_count)]
    transition_values: list[list[int]] = [[] for _ in range(configuration_count)]
    concentration_values: list[list[float]] = [[] for _ in range(configuration_count)]
    unique_query_edge_chunks: list[np.ndarray] = []
    expected_query = int(reference_manifest["measurement_start"])
    measurement_count = int(reference_manifest["measurement_query_count"])

    with trace_path.open("r", encoding="utf-8") as handle:
        for row_index, line in enumerate(handle):
            trace = json.loads(line)
            query_id = int(trace["query_id"])
            if query_id != expected_query + row_index:
                raise ValueError(f"unexpected trace query ID at row {row_index}: {query_id}")
            nodes = np.asarray(trace["visited_node_ids"], dtype=np.int64)
            if not len(nodes) or int(np.max(nodes)) >= point_count or int(np.min(nodes)) < 0:
                raise ValueError(f"invalid visited-node sequence for query {query_id}")
            edge_keys = trace_edge_keys(trace["visited_edges"], point_count)
            unique_query_edge_chunks.append(edge_keys)
            trace_edge_incidence_count += len(edge_keys)
            if len(edge_keys):
                sources, destinations = decode_edge_keys(edge_keys)
                weighted_cut_numerator += np.count_nonzero(
                    assignment_matrix[:, sources] != assignment_matrix[:, destinations],
                    axis=1,
                ).astype(np.uint64)

            trace_shards = assignment_matrix[:, nodes]
            path_counts = np.asarray(
                [np.unique(shards).size for shards in trace_shards],
                dtype=np.int64,
            )
            transitions = np.count_nonzero(
                trace_shards[:, 1:] != trace_shards[:, :-1],
                axis=1,
            )
            concentrations = np.asarray(
                [
                    np.max(np.bincount(shards.astype(np.int64))) / len(shards)
                    for shards in trace_shards
                ],
                dtype=np.float64,
            )
            for config_index, configuration in enumerate(configurations):
                path_value = int(path_counts[config_index])
                transition_value = int(transitions[config_index])
                concentration_value = float(concentrations[config_index])
                path_values[config_index].append(path_value)
                transition_values[config_index].append(transition_value)
                concentration_values[config_index].append(concentration_value)
                per_query_rows.append(
                    {
                        "query_id": query_id,
                        "partition_method": configuration["method"],
                        "partition_seed": configuration["partition_seed"],
                        "logical_shards": configuration["logical_shards"],
                        "global_trace_length": len(nodes),
                        "trace_unique_edge_count": len(edge_keys),
                        "path_shard_count": path_value,
                        "path_transition_count": transition_value,
                        "dominant_shard_concentration": concentration_value,
                    }
                )
            if (row_index + 1) % 1000 == 0:
                print(f"processed {row_index + 1}/{measurement_count} traces", flush=True)
    if len(unique_query_edge_chunks) != measurement_count:
        raise ValueError(
            f"measurement trace count mismatch: {len(unique_query_edge_chunks)} != {measurement_count}"
        )
    if trace_edge_incidence_count == 0:
        raise ValueError("measurement traces contain no edges")

    all_query_edge_keys = np.concatenate(unique_query_edge_chunks)
    observed_edge_keys, edge_frequencies = np.unique(
        all_query_edge_keys,
        return_counts=True,
    )
    del all_query_edge_keys, unique_query_edge_chunks
    edge_frequencies = edge_frequencies.astype(np.uint64, copy=False)
    reproduced_weighted_cut = cut_counts(
        assignment_matrix,
        observed_edge_keys,
        weights=edge_frequencies,
    )
    if not np.array_equal(reproduced_weighted_cut, weighted_cut_numerator):
        raise ValueError("frequency-weighted edge cut does not reproduce direct query incidence")

    summary_rows: list[dict[str, Any]] = []
    for config_index, configuration in enumerate(configurations):
        summary_rows.append(
            {
                "dataset": args.dataset,
                "dataset_sha256": spec.sha256,
                "partition_method": configuration["method"],
                "partition_seed": configuration["partition_seed"],
                "logical_shards": configuration["logical_shards"],
                "measurement_query_count": measurement_count,
                "trace_edge_incidence_count": trace_edge_incidence_count,
                "unique_observed_trace_edges": len(observed_edge_keys),
                "traversal_weighted_edge_cut": float(
                    weighted_cut_numerator[config_index] / trace_edge_incidence_count
                ),
                **summarize(np.asarray(path_values[config_index]), "path_shards"),
                **summarize(np.asarray(transition_values[config_index]), "path_transitions"),
                **summarize(
                    np.asarray(concentration_values[config_index]),
                    "dominant_shard_concentration",
                ),
                "partition_artifact_path": configuration["artifact_path"],
                "partition_artifact_sha256": configuration["artifact_sha256"],
            }
        )

    ranking = np.argsort(edge_frequencies, kind="stable")[::-1]
    scopes = (("top_1pct", 0.01), ("top_5pct", 0.05), ("top_10pct", 0.10))
    hot_rows: list[dict[str, Any]] = []
    scope_indices: list[tuple[str, np.ndarray]] = []
    for label, fraction in scopes:
        count = max(1, math.ceil(len(observed_edge_keys) * fraction))
        scope_indices.append((label, ranking[:count]))
    scope_indices.append(("all_observed", np.arange(len(observed_edge_keys), dtype=np.int64)))
    for scope_label, indices in scope_indices:
        selected_keys = observed_edge_keys[indices]
        selected_frequencies = edge_frequencies[indices]
        cuts = cut_counts(assignment_matrix, selected_keys)
        for config_index, configuration in enumerate(configurations):
            cut_ratio = float(cuts[config_index] / len(selected_keys))
            hot_rows.append(
                {
                    "dataset": args.dataset,
                    "partition_method": configuration["method"],
                    "partition_seed": configuration["partition_seed"],
                    "logical_shards": configuration["logical_shards"],
                    "edge_scope": scope_label,
                    "edge_count": len(selected_keys),
                    "minimum_query_frequency": int(np.min(selected_frequencies)),
                    "maximum_query_frequency": int(np.max(selected_frequencies)),
                    "edge_cut_ratio": cut_ratio,
                    "edge_retention_ratio": 1.0 - cut_ratio,
                }
            )

    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    summary_path = output_dir / "e2_summary.csv"
    per_query_path = output_dir / "e2_per_query.csv"
    hot_path = output_dir / "e2_hot_edges.csv"
    frequency_path = output_dir / "e2_observed_edge_frequency.npz"
    write_csv_new(summary_path, summary_rows)
    write_csv_new(per_query_path, per_query_rows)
    write_csv_new(hot_path, hot_rows)
    np.savez_compressed(
        frequency_path,
        edge_keys=observed_edge_keys,
        query_frequencies=edge_frequencies,
    )
    figures = save_figures(summary_rows, hot_rows, path_values, configurations, args.figure_dir)
    manifest = {
        "protocol_version": c23_protocol.PROTOCOL_VERSION,
        "timestamp": c23_e1.utc_timestamp(),
        "git_commit": c23_protocol.git_commit(REPO_ROOT),
        "experiment": "C23 E2 search-path preservation",
        "dataset": args.dataset,
        "dataset_sha256": spec.sha256,
        "measurement_query_count": measurement_count,
        "configuration_count": configuration_count,
        "trace_edge_frequency_definition": (
            "number of measurement queries containing directed edge (u,v), counted at most once "
            "per query and aggregated across trace layers"
        ),
        "trace_edge_incidence_count": trace_edge_incidence_count,
        "unique_observed_trace_edges": len(observed_edge_keys),
        "reference_manifest": str(reference_manifest_path),
        "reference_manifest_sha256": c23_e1.sha256_path(reference_manifest_path),
        "trace_path": str(trace_path),
        "trace_sha256": c23_e1.sha256_path(trace_path),
        "checks": {
            "complete_partition_matrix": "PASS",
            "complete_query_sequence": "PASS",
            "trace_endpoint_ranges": "PASS",
            "per_query_unique_edge_frequency": "PASS",
            "weighted_cut_frequency_reproduction": "PASS",
        },
        "files": {
            "summary": str(summary_path),
            "summary_sha256": c23_e1.sha256_path(summary_path),
            "per_query": str(per_query_path),
            "per_query_sha256": c23_e1.sha256_path(per_query_path),
            "hot_edges": str(hot_path),
            "hot_edges_sha256": c23_e1.sha256_path(hot_path),
            "observed_edge_frequency": str(frequency_path),
            "observed_edge_frequency_sha256": c23_e1.sha256_path(frequency_path),
            "figures": {
                name: {"path": path, "sha256": c23_e1.sha256_path(path)}
                for name, path in figures.items()
            },
        },
    }
    manifest_path = output_dir / "manifest.json"
    c23_e1.write_json_new(manifest_path, manifest)
    c23_e1.write_json_new(
        args.audit_output,
        {
            **manifest,
            "manifest": str(manifest_path),
            "manifest_sha256": c23_e1.sha256_path(manifest_path),
            "summary": summary_rows,
            "hot_edges": hot_rows,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
