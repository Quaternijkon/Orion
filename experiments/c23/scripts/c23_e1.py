#!/usr/bin/env python3
"""Compute C23 E1 structural-topology metrics from accepted raw artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.c23.scripts import c23_protocol


METHOD_ORDER = ("random", "kmeans", "orion", "orion_no_refinement")
METHOD_LABELS = {
    "random": "Random",
    "kmeans": "K-Means",
    "orion": "Orion",
    "orion_no_refinement": "Orion-NoRefinement",
}
METHOD_STYLES = {
    "random": {"marker": "o", "linestyle": "-"},
    "kmeans": {"marker": "s", "linestyle": "-"},
    "orion": {"marker": "^", "linestyle": "-"},
    "orion_no_refinement": {"marker": "D", "linestyle": "--"},
}


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_path(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_new(path: str | Path, value: Any) -> None:
    target = Path(path).expanduser().resolve()
    if target.exists():
        raise FileExistsError(f"refusing to overwrite: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


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


def load_partition_artifact(
    path: str | Path,
    *,
    expected_method: str,
    expected_shards: int,
    expected_points: int,
    expected_dataset_sha256: str,
    expected_sha256: str | None = None,
) -> tuple[np.ndarray, dict[str, Any], str]:
    source = Path(path).expanduser().resolve()
    actual_sha256 = sha256_path(source)
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise ValueError(f"partition checksum mismatch for {source}")
    assignments, metadata = c23_protocol.load_partition(source, expected_method)
    if len(assignments) != expected_points:
        raise ValueError(f"partition point count mismatch for {source}")
    if int(metadata["logical_shards"]) != expected_shards:
        raise ValueError(f"partition shard count mismatch for {source}")
    if str(metadata["dataset_sha256"]) != expected_dataset_sha256:
        raise ValueError(f"partition dataset binding mismatch for {source}")
    if expected_method.startswith("orion"):
        if not bool(metadata.get("disjoint")):
            raise ValueError(f"Orion partition is not marked disjoint: {source}")
        if bool(metadata.get("multi_assignment")) or bool(metadata.get("fission")):
            raise ValueError(f"Orion partition enables excluded behavior: {source}")
        expected_refinement = expected_method == "orion"
        if bool(metadata.get("topology_refinement_enabled")) != expected_refinement:
            raise ValueError(f"Orion refinement flag mismatch for {source}")
    return assignments, metadata, actual_sha256


def load_partition_matrix(
    *,
    dataset: str,
    point_count: int,
    dataset_sha256: str,
    baseline_root: str | Path,
    orion_manifest_path: str | Path,
) -> list[dict[str, Any]]:
    baseline = Path(baseline_root).expanduser().resolve() / dataset
    manifest_path = Path(orion_manifest_path).expanduser().resolve()
    build_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if str(build_manifest["dataset"]) != dataset:
        raise ValueError("Orion build-manifest dataset mismatch")
    orion_rows = {
        (str(row["method"]), int(row["logical_shards"])): row
        for row in build_manifest["artifacts"]
    }
    expected_orion = {
        (method, shard_count)
        for method in ("orion", "orion_no_refinement")
        for shard_count in c23_protocol.LOGICAL_SHARDS
    }
    if set(orion_rows) != expected_orion:
        raise ValueError(
            f"Orion matrix coverage mismatch: {sorted(orion_rows)} != {sorted(expected_orion)}"
        )

    matrix: list[dict[str, Any]] = []
    for method in METHOD_ORDER:
        for shard_count in c23_protocol.LOGICAL_SHARDS:
            if method in ("random", "kmeans"):
                path = baseline / f"{method}-m{shard_count}.npz"
                expected_sha256 = None
            else:
                source_row = orion_rows[(method, shard_count)]
                path = Path(source_row["artifact_path"])
                expected_sha256 = str(source_row["artifact_sha256"])
            assignments, metadata, actual_sha256 = load_partition_artifact(
                path,
                expected_method=method,
                expected_shards=shard_count,
                expected_points=point_count,
                expected_dataset_sha256=dataset_sha256,
                expected_sha256=expected_sha256,
            )
            matrix.append(
                {
                    "method": method,
                    "logical_shards": shard_count,
                    "assignments": assignments,
                    "metadata": metadata,
                    "artifact_path": str(Path(path).resolve()),
                    "artifact_sha256": actual_sha256,
                }
            )
    if len(matrix) != len(METHOD_ORDER) * len(c23_protocol.LOGICAL_SHARDS):
        raise AssertionError("partition matrix is incomplete")
    return matrix


def compute_e1_for_partition(
    assignments: np.ndarray,
    edge_sources: np.ndarray,
    edge_destinations: np.ndarray,
    original_out_degree: np.ndarray,
    logical_shards: int,
    *,
    edge_chunk_size: int = 2_000_000,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    assignments = np.asarray(assignments, dtype=np.int32)
    point_count = len(assignments)
    edge_count = len(edge_sources)
    if edge_count != len(edge_destinations):
        raise ValueError("edge arrays differ in length")
    if len(original_out_degree) != point_count:
        raise ValueError("degree array differs from point count")
    shard_counts = np.bincount(assignments.astype(np.int64), minlength=logical_shards)
    if len(shard_counts) != logical_shards or np.any(shard_counts == 0):
        raise ValueError("partition contains empty or invalid shards")

    intra_out_degree = np.zeros(point_count, dtype=np.int64)
    cut_edges = 0
    for start in range(0, edge_count, edge_chunk_size):
        end = min(start + edge_chunk_size, edge_count)
        sources = edge_sources[start:end]
        destinations = edge_destinations[start:end]
        same = assignments[sources] == assignments[destinations]
        cut_edges += int(len(same) - np.count_nonzero(same))
        intra_out_degree += np.bincount(
            sources[same].astype(np.int64, copy=False),
            minlength=point_count,
        )

    positive_degree = original_out_degree > 0
    if not np.all(positive_degree):
        raise ValueError("reference graph contains zero-out-degree nodes")
    retained_ratio = intra_out_degree / original_out_degree
    loss_ratio = 1.0 - retained_ratio

    same_edge = assignments[edge_sources] == assignments[edge_destinations]
    intra_sources = np.asarray(edge_sources[same_edge], dtype=np.int64)
    intra_destinations = np.asarray(edge_destinations[same_edge], dtype=np.int64)
    graph = coo_matrix(
        (
            np.ones(len(intra_sources), dtype=np.bool_),
            (intra_sources, intra_destinations),
        ),
        shape=(point_count, point_count),
        dtype=np.bool_,
    ).tocsr()
    graph.sum_duplicates()
    component_count, labels = connected_components(
        graph,
        directed=False,
        return_labels=True,
    )
    del graph, intra_sources, intra_destinations, same_edge

    component_sizes = np.bincount(labels, minlength=component_count)
    first_node = np.full(component_count, point_count, dtype=np.int64)
    np.minimum.at(first_node, labels, np.arange(point_count, dtype=np.int64))
    component_shards = assignments[first_node]
    if np.any(component_shards < 0) or np.any(component_shards >= logical_shards):
        raise AssertionError("component assigned to an invalid shard")
    components_per_shard = np.bincount(
        component_shards.astype(np.int64),
        minlength=logical_shards,
    )
    largest_component = np.zeros(logical_shards, dtype=np.int64)
    np.maximum.at(largest_component, component_shards, component_sizes)
    isolated_per_shard = np.bincount(
        assignments[component_sizes[labels] == 1].astype(np.int64),
        minlength=logical_shards,
    )
    largest_fraction = largest_component / shard_counts
    isolated_fraction = isolated_per_shard / shard_counts

    connectivity_rows = []
    for shard_id in range(logical_shards):
        connectivity_rows.append(
            {
                "shard_id": shard_id,
                "vector_count": int(shard_counts[shard_id]),
                "connected_components": int(components_per_shard[shard_id]),
                "largest_component_fraction": float(largest_fraction[shard_id]),
                "isolated_node_fraction": float(isolated_fraction[shard_id]),
            }
        )

    weights = shard_counts.astype(np.float64)
    metrics = {
        "edge_count": edge_count,
        "cut_edge_count": cut_edges,
        "edge_cut_ratio": float(cut_edges / edge_count),
        "mean_intra_shard_degree": float(np.mean(intra_out_degree)),
        "median_intra_shard_degree": float(np.median(intra_out_degree)),
        "p10_intra_shard_degree": float(np.percentile(intra_out_degree, 10)),
        "mean_retained_degree_ratio": float(np.mean(retained_ratio)),
        "median_retained_degree_ratio": float(np.median(retained_ratio)),
        "p10_retained_degree_ratio": float(np.percentile(retained_ratio, 10)),
        "fraction_losing_gt_25pct_neighbors": float(np.mean(loss_ratio > 0.25)),
        "fraction_losing_gt_50pct_neighbors": float(np.mean(loss_ratio > 0.50)),
        "fraction_losing_gt_75pct_neighbors": float(np.mean(loss_ratio > 0.75)),
        "connected_components_unweighted_shard_mean": float(np.mean(components_per_shard)),
        "connected_components_vector_weighted_mean": float(
            np.average(components_per_shard, weights=weights)
        ),
        "largest_component_fraction_unweighted_shard_mean": float(
            np.mean(largest_fraction)
        ),
        "largest_component_fraction_vector_weighted_mean": float(
            np.average(largest_fraction, weights=weights)
        ),
        "isolated_node_fraction_unweighted_shard_mean": float(np.mean(isolated_fraction)),
        "isolated_node_fraction_vector_weighted_mean": float(
            np.average(isolated_fraction, weights=weights)
        ),
    }
    return metrics, connectivity_rows


def save_figures(rows: list[dict[str, Any]], figure_dir: str | Path) -> dict[str, str]:
    target = Path(figure_dir).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    figure_specs = {
        "edge_cut": target / "fig_c23_edge_cut.pdf",
        "retained_degree": target / "fig_c23_retained_degree.pdf",
        "connectivity": target / "fig_c23_connectivity.pdf",
    }
    existing = [str(path) for path in figure_specs.values() if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite figures: {existing}")

    metrics = {
        "edge_cut": ("edge_cut_ratio", "Reference L0 edge-cut ratio"),
        "retained_degree": (
            "mean_retained_degree_ratio",
            "Mean retained neighbor-degree ratio",
        ),
        "connectivity": (
            "largest_component_fraction_vector_weighted_mean",
            "Vector-weighted largest-component fraction",
        ),
    }
    for figure_name, (metric, ylabel) in metrics.items():
        fig, axis = plt.subplots(figsize=(6.6, 4.2))
        for method in METHOD_ORDER:
            selected = sorted(
                (row for row in rows if row["partition_method"] == method),
                key=lambda row: int(row["logical_shards"]),
            )
            axis.plot(
                [int(row["logical_shards"]) for row in selected],
                [float(row[metric]) for row in selected],
                label=METHOD_LABELS[method],
                linewidth=1.8,
                markersize=5,
                **METHOD_STYLES[method],
            )
        axis.set_xlabel("Logical shard count")
        axis.set_ylabel(ylabel)
        axis.set_xticks(c23_protocol.LOGICAL_SHARDS)
        axis.grid(True, alpha=0.25)
        axis.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(figure_specs[figure_name])
        plt.close(fig)
    return {name: str(path) for name, path in figure_specs.items()}


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
    if int(reference_manifest["point_count"]) != spec.expected_train_rows:
        raise ValueError("reference point count differs from dataset contract")
    if int(reference_manifest["dimension"]) != spec.dimension:
        raise ValueError("reference dimension differs from dataset contract")

    node_path = Path(reference_manifest["files"]["node_ids"])
    edge_path = Path(reference_manifest["files"]["l0_edges"])
    if sha256_path(node_path) != str(reference_manifest["files"]["node_ids_sha256"]):
        raise ValueError("reference node-ID checksum mismatch")
    if sha256_path(edge_path) != str(reference_manifest["files"]["l0_edges_sha256"]):
        raise ValueError("reference L0-edge checksum mismatch")
    point_count = int(reference_manifest["point_count"])
    node_ids = np.memmap(node_path, dtype="<u4", mode="r")
    if len(node_ids) != point_count or not np.array_equal(
        node_ids,
        np.arange(point_count, dtype=np.uint32),
    ):
        raise ValueError("E1 currently requires reference node IDs to equal row offsets")
    edges = np.memmap(edge_path, dtype="<u4", mode="r")
    if len(edges) % 2:
        raise ValueError("L0 edge file contains a partial pair")
    edges = edges.reshape(-1, 2)
    if len(edges) != int(reference_manifest["l0_edge_count"]):
        raise ValueError("L0 edge count differs from reference manifest")
    edge_sources = edges[:, 0]
    edge_destinations = edges[:, 1]
    if int(np.max(edges)) >= point_count:
        raise ValueError("L0 edge endpoint lies outside the reference node range")
    original_out_degree = np.bincount(
        edge_sources.astype(np.int64, copy=False),
        minlength=point_count,
    )

    matrix = load_partition_matrix(
        dataset=args.dataset,
        point_count=point_count,
        dataset_sha256=spec.sha256,
        baseline_root=args.baseline_root,
        orion_manifest_path=args.orion_build_manifest,
    )
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)

    metric_rows: list[dict[str, Any]] = []
    connectivity_rows: list[dict[str, Any]] = []
    for configuration in matrix:
        method = str(configuration["method"])
        logical_shards = int(configuration["logical_shards"])
        assignments = configuration["assignments"]
        metadata = configuration["metadata"]
        metrics, per_shard = compute_e1_for_partition(
            assignments,
            edge_sources,
            edge_destinations,
            original_out_degree,
            logical_shards,
        )
        shard_counts = np.bincount(assignments.astype(np.int64), minlength=logical_shards)
        row = {
            "dataset": args.dataset,
            "dataset_sha256": spec.sha256,
            "partition_method": method,
            "partition_seed": int(metadata["partition_seed"]),
            "logical_shards": logical_shards,
            "point_count": point_count,
            "shard_size_mean": float(np.mean(shard_counts)),
            "shard_size_std": float(np.std(shard_counts)),
            "shard_size_min": int(np.min(shard_counts)),
            "shard_size_max": int(np.max(shard_counts)),
            "max_shard_size_over_mean": float(np.max(shard_counts) / np.mean(shard_counts)),
            "partition_artifact_path": configuration["artifact_path"],
            "partition_artifact_sha256": configuration["artifact_sha256"],
            **metrics,
        }
        metric_rows.append(row)
        for shard_row in per_shard:
            connectivity_rows.append(
                {
                    "dataset": args.dataset,
                    "partition_method": method,
                    "partition_seed": int(metadata["partition_seed"]),
                    "logical_shards": logical_shards,
                    **shard_row,
                }
            )
        print(
            f"completed {method} M={logical_shards}: "
            f"edge_cut={metrics['edge_cut_ratio']:.9f}",
            flush=True,
        )

    metrics_path = output_dir / "e1_metrics.csv"
    connectivity_path = output_dir / "e1_connectivity_by_shard.csv"
    write_csv_new(metrics_path, metric_rows)
    write_csv_new(connectivity_path, connectivity_rows)
    figures = save_figures(metric_rows, args.figure_dir)
    manifest = {
        "protocol_version": c23_protocol.PROTOCOL_VERSION,
        "timestamp": utc_timestamp(),
        "git_commit": c23_protocol.git_commit(REPO_ROOT),
        "experiment": "C23 E1 structural topology preservation",
        "dataset": args.dataset,
        "dataset_sha256": spec.sha256,
        "point_count": point_count,
        "reference_edge_count": len(edges),
        "reference_manifest": str(reference_manifest_path),
        "reference_manifest_sha256": sha256_path(reference_manifest_path),
        "baseline_root": str(Path(args.baseline_root).expanduser().resolve()),
        "orion_build_manifest": str(Path(args.orion_build_manifest).expanduser().resolve()),
        "orion_build_manifest_sha256": sha256_path(args.orion_build_manifest),
        "configuration_count": len(metric_rows),
        "software": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "checks": {
            "complete_partition_matrix": "PASS",
            "partition_checksums_and_dataset_binding": "PASS",
            "orion_disjoint_and_exclusions": "PASS",
            "reference_checksums_and_cardinality": "PASS",
            "zero_out_degree_nodes": int(np.count_nonzero(original_out_degree == 0)),
        },
        "files": {
            "metrics": str(metrics_path),
            "metrics_sha256": sha256_path(metrics_path),
            "connectivity_by_shard": str(connectivity_path),
            "connectivity_by_shard_sha256": sha256_path(connectivity_path),
            "figures": {
                name: {"path": path, "sha256": sha256_path(path)}
                for name, path in figures.items()
            },
        },
    }
    manifest_path = output_dir / "manifest.json"
    write_json_new(manifest_path, manifest)
    audit = {
        **manifest,
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_path(manifest_path),
        "metrics": metric_rows,
    }
    write_json_new(args.audit_output, audit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
