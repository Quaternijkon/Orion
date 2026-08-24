#!/usr/bin/env python3
"""Build and evaluate the protocol-required three-seed C23 partition variance matrix."""

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

from experiments.c1.scripts import c1_protocol
from experiments.c23.scripts import c23_e1, c23_e2, c23_protocol


def write_csv_new(path: str | Path, rows: list[dict[str, Any]]) -> None:
    target = Path(path).expanduser().resolve()
    if target.exists():
        raise FileExistsError(f"refusing to overwrite: {target}")
    with target.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_artifact(path: str | Path) -> tuple[np.ndarray, dict[str, Any]]:
    with np.load(Path(path), allow_pickle=False) as archive:
        assignments = np.asarray(archive["assignments"], dtype=np.int32)
        metadata = json.loads(str(np.asarray(archive["metadata_json"]).item()))
    return assignments, metadata


def build_matrix(args: argparse.Namespace, output: Path) -> list[dict[str, Any]]:
    spec = c23_protocol.DATASETS[args.dataset]
    c1_spec = c1_protocol.DATASETS[args.dataset]
    rows: list[dict[str, Any]] = []
    primary_root = Path(args.primary_baseline_root).expanduser().resolve() / args.dataset
    for method in ("random", "kmeans"):
        primary = primary_root / f"{method}-m32.npz"
        _, metadata = load_artifact(primary)
        rows.append(
            {
                "method": method,
                "seed_label": str(metadata["partition_seed"]),
                "partition_seed": int(metadata["partition_seed"]),
                "navigation_sample_seed": "",
                "artifact_path": str(primary),
                "artifact_sha256": c23_e1.sha256_path(primary),
            }
        )
        for seed in (20260821, 20260822):
            path = output / f"{method}-m32-seed{seed}.npz"
            record = c1_protocol.partition_artifact(
                args.hdf5_path,
                c1_spec,
                method,
                32,
                path,
                seed=seed,
                sample_size=100_000,
                iterations=12,
                tolerance=1e-4,
                batch_size=16_384,
            )
            rows.append(
                {
                    "method": method,
                    "seed_label": str(seed),
                    "partition_seed": seed,
                    "navigation_sample_seed": "",
                    "artifact_path": str(path),
                    "artifact_sha256": str(record["artifact_sha256"]),
                }
            )

    primary_orion = json.loads(
        Path(args.primary_orion_manifest).expanduser().resolve().read_text(encoding="utf-8")
    )
    for method in ("orion", "orion_no_refinement"):
        record = next(
            row
            for row in primary_orion["artifacts"]
            if row["method"] == method and int(row["logical_shards"]) == 32
        )
        rows.append(
            {
                "method": method,
                "seed_label": "upper100-kmeans1",
                "partition_seed": 1,
                "navigation_sample_seed": 100,
                "artifact_path": str(record["artifact_path"]),
                "artifact_sha256": str(record["artifact_sha256"]),
            }
        )
    for upper_seed, kmeans_seed in ((101, 2), (102, 3)):
        seed_root = output / f"orion-upper{upper_seed}-kmeans{kmeans_seed}"
        result = c23_protocol.build_orion_matrix(
            args.hdf5_path,
            spec,
            seed_root,
            logical_shards=(32,),
            upper_sample_seed=upper_seed,
            kmeans_seed=kmeans_seed,
            include_no_refinement=True,
        )
        for record in result["artifacts"]:
            rows.append(
                {
                    "method": str(record["method"]),
                    "seed_label": f"upper{upper_seed}-kmeans{kmeans_seed}",
                    "partition_seed": kmeans_seed,
                    "navigation_sample_seed": upper_seed,
                    "artifact_path": str(record["artifact_path"]),
                    "artifact_sha256": str(record["artifact_sha256"]),
                }
            )
    expected = {method: 3 for method in c23_e1.METHOD_ORDER}
    observed = {
        method: sum(row["method"] == method for row in rows) for method in c23_e1.METHOD_ORDER
    }
    if observed != expected:
        raise ValueError(f"variance matrix coverage mismatch: {observed}")
    return rows


def edge_cut(assignments: np.ndarray, edges: np.ndarray, chunk_size: int = 2_000_000) -> float:
    cut = 0
    for start in range(0, len(edges), chunk_size):
        chunk = edges[start : start + chunk_size]
        cut += int(
            np.count_nonzero(assignments[chunk[:, 0]] != assignments[chunk[:, 1]])
        )
    return cut / len(edges)


def summarize_groups(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for method in c23_e1.METHOD_ORDER:
        selected = [row for row in rows if row["method"] == method]
        summary: dict[str, Any] = {"method": method, "seed_count": len(selected)}
        for metric in (
            "edge_cut_ratio",
            "traversal_weighted_edge_cut",
            "shard_size_cv",
            "max_size_over_mean",
        ):
            values = np.asarray([float(row[metric]) for row in selected])
            summary.update(
                {
                    f"{metric}_mean": float(np.mean(values)),
                    f"{metric}_std": float(np.std(values)),
                    f"{metric}_min": float(np.min(values)),
                    f"{metric}_max": float(np.max(values)),
                    f"{metric}_range": float(np.ptp(values)),
                }
            )
        result.append(summary)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(c23_protocol.DATASETS), required=True)
    parser.add_argument("--hdf5-path", required=True)
    parser.add_argument("--primary-baseline-root", required=True)
    parser.add_argument("--primary-orion-manifest", required=True)
    parser.add_argument("--reference-run", required=True)
    parser.add_argument("--e2-frequency", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--audit-output", required=True)
    args = parser.parse_args()

    output = Path(args.output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output}")
    output.mkdir(parents=True, exist_ok=False)
    matrix = build_matrix(args, output)
    reference_manifest = json.loads(
        (Path(args.reference_run).expanduser().resolve() / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    edge_values = np.memmap(
        reference_manifest["files"]["l0_edges"], dtype="<u4", mode="r"
    )
    edges = edge_values.reshape(-1, 2)
    with np.load(args.e2_frequency, allow_pickle=False) as archive:
        observed_keys = np.asarray(archive["edge_keys"], dtype=np.uint64)
        frequencies = np.asarray(archive["query_frequencies"], dtype=np.uint64)
    sources, destinations = c23_e2.decode_edge_keys(observed_keys)
    total_frequency = int(np.sum(frequencies, dtype=np.uint64))
    metric_rows = []
    for index, row in enumerate(matrix, 1):
        assignments, metadata = load_artifact(row["artifact_path"])
        if c23_e1.sha256_path(row["artifact_path"]) != row["artifact_sha256"]:
            raise ValueError("variance partition checksum mismatch")
        if len(assignments) != int(reference_manifest["point_count"]):
            raise ValueError("variance partition point-count mismatch")
        counts = np.bincount(assignments.astype(np.int64), minlength=32)
        cut = assignments[sources] != assignments[destinations]
        twcut = float(np.sum(frequencies[cut], dtype=np.uint64) / total_frequency)
        metric_rows.append(
            {
                **row,
                "logical_shards": 32,
                "point_count": len(assignments),
                "edge_cut_ratio": edge_cut(assignments, edges),
                "traversal_weighted_edge_cut": twcut,
                "shard_size_mean": float(np.mean(counts)),
                "shard_size_std": float(np.std(counts)),
                "shard_size_cv": float(np.std(counts) / np.mean(counts)),
                "shard_size_min": int(np.min(counts)),
                "shard_size_max": int(np.max(counts)),
                "max_size_over_mean": float(np.max(counts) / np.mean(counts)),
                "metadata_partition_seed": int(metadata["partition_seed"]),
            }
        )
        print(f"[{index}/12] measured {row['method']} {row['seed_label']}", flush=True)
    summary_rows = summarize_groups(metric_rows)
    metrics_path = output / "variance_metrics.csv"
    summary_path = output / "variance_summary.csv"
    write_csv_new(metrics_path, metric_rows)
    write_csv_new(summary_path, summary_rows)
    manifest = {
        "protocol_version": c23_protocol.PROTOCOL_VERSION,
        "timestamp": c23_e1.utc_timestamp(),
        "git_commit": c23_protocol.git_commit(REPO_ROOT),
        "dataset": args.dataset,
        "logical_shards": 32,
        "seed_count_per_method": 3,
        "configuration_count": len(metric_rows),
        "decision": (
            "PENDING: compare within-method ranges with between-method effects before accepting "
            "one fixed seed for the expensive full matrix."
        ),
        "checks": {
            "three_seeds_per_method": "PASS",
            "partition_checksums_and_coverage": "PASS",
            "reference_edge_and_frequency_metrics": "PASS",
        },
        "files": {
            "metrics": str(metrics_path),
            "metrics_sha256": c23_e1.sha256_path(metrics_path),
            "summary": str(summary_path),
            "summary_sha256": c23_e1.sha256_path(summary_path),
        },
        "summary": summary_rows,
    }
    manifest_path = output / "manifest.json"
    c23_e1.write_json_new(manifest_path, manifest)
    c23_e1.write_json_new(
        args.audit_output,
        {
            **manifest,
            "manifest": str(manifest_path),
            "manifest_sha256": c23_e1.sha256_path(manifest_path),
            "metrics": metric_rows,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
