#!/usr/bin/env python3
"""Core C2-C3 dataset, partition, and evidence helpers.

Formal measurements live outside the repository under ``--artifact-root``. Small manifests,
summaries, and final paper artifacts are registered under ``experiments/c23``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import resource
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import orion_native_layout
from tools import qdrant_two_level_routing_experiment as orion


PROTOCOL_VERSION = "c23-20260821-v1"
LOGICAL_SHARDS = (2, 4, 8, 16, 32)
TUNING_QUERIES = slice(0, 1000)
MEASUREMENT_QUERIES = slice(1000, None)
HNSW_M = 32
HNSW_EF_CONSTRUCTION = 200
HNSW_GRAPH_BUILD_SEED = 20260821
EF_SEARCH_GRID = (10, 20, 40, 80, 160, 320)


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    filename: str
    metric: str
    hnsw_space: str
    dimension: int
    expected_train_rows: int
    expected_query_rows: int
    sha256: str


DATASETS = {
    "sift1m": DatasetSpec(
        name="sift1m",
        filename="sift-128-euclidean.hdf5",
        metric="euclid",
        hnsw_space="l2",
        dimension=128,
        expected_train_rows=1_000_000,
        expected_query_rows=10_000,
        sha256="dd6f0a6ed6b7ebb8934680f861a33ed01ff33991eaee4fd60914d854a0ca5984",
    ),
    "glove-200-angular": DatasetSpec(
        name="glove-200-angular",
        filename="glove-200-angular.hdf5",
        metric="cosine",
        hnsw_space="cosine",
        dimension=200,
        expected_train_rows=1_183_514,
        expected_query_rows=10_000,
        sha256="4839085e5a8bb293434a1a66e1aa0193afc3f07c6797a85f1dbd91656172da20",
    ),
}


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_path(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(repo: str | Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value * 1024 if os.uname().sysname != "Darwin" else value


def dataset_audit(path: str | Path, spec: DatasetSpec) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    checksum = sha256_path(source)
    if checksum != spec.sha256:
        raise ValueError(f"dataset checksum mismatch for {spec.name}: {checksum}")
    with h5py.File(source, "r") as handle:
        required = {"train", "test", "neighbors"}
        missing = sorted(required.difference(handle.keys()))
        if missing:
            raise ValueError(f"dataset {spec.name} is missing HDF5 arrays: {missing}")
        train_shape = tuple(int(value) for value in handle["train"].shape)
        test_shape = tuple(int(value) for value in handle["test"].shape)
        truth_shape = tuple(int(value) for value in handle["neighbors"].shape)
        if train_shape != (spec.expected_train_rows, spec.dimension):
            raise ValueError(f"unexpected train shape for {spec.name}: {train_shape}")
        if test_shape != (spec.expected_query_rows, spec.dimension):
            raise ValueError(f"unexpected test shape for {spec.name}: {test_shape}")
        if truth_shape[0] != spec.expected_query_rows or truth_shape[1] < 10:
            raise ValueError(f"unexpected ground-truth shape for {spec.name}: {truth_shape}")
        truth_sample = np.asarray(handle["neighbors"][: min(1000, truth_shape[0]), :10])
        if truth_sample.size and (
            int(truth_sample.min()) < 0 or int(truth_sample.max()) >= spec.expected_train_rows
        ):
            raise ValueError(f"ground-truth IDs are outside the train array for {spec.name}")
    measurement_count = max(0, spec.expected_query_rows - 1000)
    return {
        "protocol_version": PROTOCOL_VERSION,
        "timestamp": utc_timestamp(),
        "dataset": asdict(spec),
        "path": str(source),
        "size_bytes": source.stat().st_size,
        "sha256": checksum,
        "train_shape": train_shape,
        "test_shape": test_shape,
        "neighbors_shape": truth_shape,
        "tuning_query_range": [0, 1000],
        "measurement_query_range": [1000, spec.expected_query_rows],
        "measurement_query_count": measurement_count,
        "measurement_count_note": (
            "The official dataset exposes 10,000 total queries. A disjoint 1,000-query tuning "
            "split therefore leaves 9,000 measurement queries; no additional official queries "
            "are available."
        ),
    }


def load_partition(path: str | Path, expected_method: str | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    source = Path(path).expanduser().resolve()
    with np.load(source, allow_pickle=False) as archive:
        assignments = np.asarray(archive["assignments"]).astype(np.int32, copy=True)
        metadata = json.loads(str(np.asarray(archive["metadata_json"]).item()))
    method = str(metadata.get("method"))
    if expected_method is not None and method != expected_method:
        raise ValueError(f"partition method mismatch: expected {expected_method}, found {method}")
    logical_shards = int(metadata["logical_shards"])
    point_count = int(metadata["point_count"])
    if assignments.shape != (point_count,):
        raise ValueError(f"partition assignment shape mismatch: {assignments.shape}")
    if assignments.size and (
        int(assignments.min()) < 0 or int(assignments.max()) >= logical_shards
    ):
        raise ValueError("partition contains an invalid shard ID")
    counts = np.bincount(assignments.astype(np.int64), minlength=logical_shards)
    if np.any(counts == 0):
        raise ValueError(f"partition contains empty shards: {np.flatnonzero(counts == 0).tolist()}")
    if counts.astype(int).tolist() != [int(value) for value in metadata["shard_counts"]]:
        raise ValueError("partition shard counts differ from metadata")
    return assignments, metadata


def audit_c1_baseline_matrix(partition_root: str | Path) -> dict[str, Any]:
    root = Path(partition_root).expanduser().resolve()
    rows: list[dict[str, Any]] = []
    for dataset_name, spec in DATASETS.items():
        for method in ("random", "kmeans"):
            for logical_shards in LOGICAL_SHARDS:
                artifact = root / dataset_name / f"{method}-m{logical_shards}.npz"
                assignments, metadata = load_partition(artifact, method)
                if int(metadata["logical_shards"]) != logical_shards:
                    raise ValueError(f"logical-shard mismatch in {artifact}")
                if str(metadata.get("dataset_sha256")) != spec.sha256:
                    raise ValueError(f"dataset binding mismatch in {artifact}")
                if len(assignments) != spec.expected_train_rows:
                    raise ValueError(f"point-count mismatch in {artifact}")
                counts = np.bincount(assignments.astype(np.int64), minlength=logical_shards)
                mean = float(np.mean(counts))
                std = float(np.std(counts))
                rows.append(
                    {
                        "dataset": dataset_name,
                        "partition_method": method,
                        "partition_seed": int(metadata["partition_seed"]),
                        "logical_shards": logical_shards,
                        "artifact_path": str(artifact),
                        "artifact_sha256": sha256_path(artifact),
                        "point_count": len(assignments),
                        "shard_size_min": int(np.min(counts)),
                        "shard_size_max": int(np.max(counts)),
                        "shard_size_mean": mean,
                        "shard_size_std": std,
                        "shard_size_cv": float(std / mean),
                        "max_size_over_mean": float(np.max(counts) / mean),
                        "kmeans_reached_iteration_cap": bool(
                            method == "kmeans"
                            and metadata.get("kmeans")
                            and len(metadata["kmeans"].get("history") or [])
                            >= int(metadata["kmeans"].get("maximum_iterations") or 0)
                        ),
                    }
                )
    if len(rows) != len(DATASETS) * 2 * len(LOGICAL_SHARDS):
        raise AssertionError("baseline audit did not cover the complete matrix")
    seed_sets = {
        f"{dataset}:{method}": sorted(
            {
                int(row["partition_seed"])
                for row in rows
                if row["dataset"] == dataset and row["partition_method"] == method
            }
        )
        for dataset in DATASETS
        for method in ("random", "kmeans")
    }
    return {
        "protocol_version": PROTOCOL_VERSION,
        "timestamp": utc_timestamp(),
        "partition_root": str(root),
        "configuration_count": len(rows),
        "expected_configuration_count": 20,
        "seed_sets": seed_sets,
        "randomness_validation_status": (
            "PENDING: the reusable C1 primary matrix contains one fixed seed per method. "
            "C23 must run the protocol-required >=3-seed construction-variance validation "
            "before deciding whether one fixed seed is acceptable for later expensive stages."
        ),
        "rows": rows,
    }


def expected_hnsw_entry_point_parameters(
    point_count: int,
    dimension: int,
    full_scan_threshold_kb: int,
) -> dict[str, int]:
    """Independently reproduce Qdrant's dense-float HNSW entry-point calculation."""
    average_vector_bytes = dimension * np.dtype(np.float32).itemsize
    full_scan_threshold_points = (
        full_scan_threshold_kb * 1024 // average_vector_bytes
        if average_vector_bytes
        else 1
    )
    entry_points_num = max(
        1,
        (point_count // full_scan_threshold_points if full_scan_threshold_points else 0) * 10,
    )
    return {
        "average_vector_bytes": average_vector_bytes,
        "full_scan_threshold_points": full_scan_threshold_points,
        "entry_points_num": entry_points_num,
    }


def audit_reference_run(run_dir: str | Path) -> dict[str, Any]:
    root = Path(run_dir).expanduser().resolve()
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = manifest["files"]
    checked_files: dict[str, dict[str, Any]] = {}
    for label in ("node_ids", "l0_edges", "tuning", "measurement_traces"):
        path = Path(files[label]).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256_path(path)
        expected = str(files[f"{label}_sha256"])
        if actual != expected:
            raise ValueError(f"{label} checksum mismatch: {actual} != {expected}")
        checked_files[label] = {
            "path": str(path),
            "sha256": actual,
            "size_bytes": path.stat().st_size,
        }
    point_count = int(manifest["point_count"])
    if checked_files["node_ids"]["size_bytes"] != point_count * 4:
        raise ValueError("node_ids byte length does not match point_count")
    edge_count = int(manifest["l0_edge_count"])
    if checked_files["l0_edges"]["size_bytes"] != edge_count * 8:
        raise ValueError("L0 edge byte length does not match l0_edge_count")

    entry_parameters = expected_hnsw_entry_point_parameters(
        point_count,
        int(manifest["dimension"]),
        int(manifest["hnsw_full_scan_threshold_kb"]),
    )
    recorded_entry_parameters = {
        "average_vector_bytes": int(manifest["hnsw_average_vector_bytes"]),
        "full_scan_threshold_points": int(manifest["hnsw_full_scan_threshold_points"]),
        "entry_points_num": int(manifest["hnsw_entry_points_num"]),
    }
    if recorded_entry_parameters != entry_parameters:
        raise ValueError(
            "HNSW entry-point parameters differ from production calculation: "
            f"{recorded_entry_parameters} != {entry_parameters}"
        )

    tuning = list(manifest["tuning"])
    selected_ef = int(manifest["selected_ef_search"])
    eligible = [
        int(row["ef_search"])
        for row in tuning
        if float(row["recall_at_k"]) >= float(manifest["target_recall"])
    ]
    if not eligible or selected_ef != min(eligible):
        raise ValueError("selected EF is not the smallest tuning candidate meeting recall")

    trace_path = Path(files["measurement_traces"])
    trace_rows = 0
    recall_sum = 0.0
    required = {
        "query_id",
        "visited_node_ids",
        "visited_edges",
        "distance_computations",
        "result_ids",
        "ground_truth_ids",
        "recall_at_k",
        "ef_search",
        "local_latency_us",
    }
    with trace_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            missing = sorted(required.difference(row))
            if missing:
                raise ValueError(f"trace row {line_number} is missing fields: {missing}")
            visited_nodes = list(row["visited_node_ids"])
            visited_edges = list(row["visited_edges"])
            if int(row["distance_computations"]) != len(visited_nodes):
                raise ValueError(f"trace row {line_number} distance count mismatch")
            if [int(edge["traversal_order"]) for edge in visited_edges] != list(
                range(len(visited_edges))
            ):
                raise ValueError(f"trace row {line_number} traversal order is not contiguous")
            if int(row["ef_search"]) != selected_ef:
                raise ValueError(f"trace row {line_number} uses a non-selected EF")
            result_ids = [int(value) for value in row["result_ids"]]
            ground_truth = {int(value) for value in row["ground_truth_ids"]}
            recomputed = sum(value in ground_truth for value in result_ids[:10]) / 10.0
            if not math.isclose(recomputed, float(row["recall_at_k"]), abs_tol=1e-12):
                raise ValueError(f"trace row {line_number} recall mismatch")
            if int(row["distance_computations"]) <= 0 or int(row["local_latency_us"]) < 0:
                raise ValueError(f"trace row {line_number} has invalid work or latency")
            recall_sum += recomputed
            trace_rows += 1
    expected_rows = int(manifest["measurement_query_count"])
    if trace_rows != expected_rows:
        raise ValueError(f"trace row count mismatch: {trace_rows} != {expected_rows}")
    measured_recall = recall_sum / trace_rows
    if not math.isclose(
        measured_recall,
        float(manifest["measurement_recall_at_k"]),
        abs_tol=1e-12,
    ):
        raise ValueError("manifest measurement recall does not reproduce from traces")
    return {
        "protocol_version": PROTOCOL_VERSION,
        "timestamp": utc_timestamp(),
        "run_dir": str(root),
        "manifest_sha256": sha256_path(manifest_path),
        "point_count": point_count,
        "l0_edge_count": edge_count,
        "selected_ef_search": selected_ef,
        "measurement_query_count": trace_rows,
        "measurement_recall_at_k": measured_recall,
        "checked_files": checked_files,
        "checks": {
            "file_checksums": "PASS",
            "graph_cardinality": "PASS",
            "production_entry_point_parameters": "PASS",
            "smallest_common_ef_selection": "PASS",
            "trace_schema": "PASS",
            "trace_distance_counts": "PASS",
            "trace_edge_order": "PASS",
            "trace_recall_reproduction": "PASS",
        },
    }


def export_reference_input(
    hdf5_path: str | Path,
    spec: DatasetSpec,
    output_dir: str | Path,
    *,
    chunk_size: int = 16_384,
) -> dict[str, Any]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    audit = dataset_audit(hdf5_path, spec)
    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite reference input: {output}")
    output.mkdir(parents=True, exist_ok=False)
    outputs = {
        "vectors": output / "vectors.f32le",
        "queries": output / "queries.f32le",
        "ground_truth": output / "ground_truth.u32le",
    }
    with h5py.File(Path(hdf5_path).expanduser().resolve(), "r") as handle:
        with outputs["vectors"].open("xb") as writer:
            train = handle["train"]
            for start in range(0, len(train), chunk_size):
                np.asarray(train[start : start + chunk_size], dtype="<f4").tofile(writer)
        with outputs["queries"].open("xb") as writer:
            queries = handle["test"]
            for start in range(0, len(queries), chunk_size):
                np.asarray(queries[start : start + chunk_size], dtype="<f4").tofile(writer)
        with outputs["ground_truth"].open("xb") as writer:
            truth = handle["neighbors"]
            for start in range(0, len(truth), chunk_size):
                np.asarray(truth[start : start + chunk_size, :10], dtype="<u4").tofile(writer)
    records = {
        label: {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_path(path),
        }
        for label, path in outputs.items()
    }
    expected_sizes = {
        "vectors": spec.expected_train_rows * spec.dimension * 4,
        "queries": spec.expected_query_rows * spec.dimension * 4,
        "ground_truth": spec.expected_query_rows * 10 * 4,
    }
    for label, expected in expected_sizes.items():
        if records[label]["size_bytes"] != expected:
            raise ValueError(
                f"exported {label} size mismatch: {records[label]['size_bytes']} != {expected}"
            )
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "timestamp": utc_timestamp(),
        "dataset": asdict(spec),
        "source_dataset": audit,
        "raw_format": {
            "vectors": "row-major little-endian float32",
            "queries": "row-major little-endian float32",
            "ground_truth": "row-major little-endian uint32, width 10",
        },
        "files": records,
    }
    write_json_new(output / "manifest.json", manifest)
    return manifest


def save_partition(
    destination: str | Path,
    assignments: np.ndarray,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    target = Path(destination).expanduser().resolve()
    if target.exists():
        raise FileExistsError(f"refusing to overwrite partition artifact: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    assignments = np.asarray(assignments, dtype=np.int32)
    logical_shards = int(metadata["logical_shards"])
    counts = np.bincount(assignments.astype(np.int64), minlength=logical_shards)
    if np.any(counts == 0):
        raise ValueError(f"refusing partition with empty shards: {counts.tolist()}")
    if len(assignments) != int(metadata["point_count"]):
        raise ValueError("assignment count differs from metadata point_count")
    metadata = dict(metadata)
    metadata["shard_counts"] = counts.astype(int).tolist()
    temporary = target.with_suffix(target.suffix + ".tmp.npz")
    np.savez_compressed(
        temporary,
        assignments=assignments,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True, allow_nan=False)),
    )
    os.replace(temporary, target)
    record = dict(metadata)
    record["artifact_path"] = str(target)
    record["artifact_sha256"] = sha256_path(target)
    return record


def _orion_assignments(
    train: np.ndarray,
    upper_indices: np.ndarray,
    point_to_l1s: list[list[int]],
    logical_shards: int,
    *,
    kmeans_seed: int,
    kmeans_iters: int,
    topology_iters: int,
    refine: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    nearest_l1 = np.asarray([row[0] for row in point_to_l1s], dtype=np.int64)
    l1_weights_map = np.bincount(nearest_l1, minlength=len(train))
    upper_weights = l1_weights_map[upper_indices].astype(np.int64, copy=False)

    started = time.perf_counter()
    initial = orion.initial_l1_shards_by_balanced_kmeans(
        train,
        upper_indices,
        upper_weights,
        logical_shards,
        kmeans_iters,
        kmeans_seed,
    )
    initial_seconds = time.perf_counter() - started

    refined = list(initial)
    refinement_iterations = 0
    refinement_seconds = 0.0
    if refine:
        started = time.perf_counter()
        refined, refinement_iterations = orion.converge_l1_topology(
            point_to_l1s,
            upper_indices,
            upper_weights,
            refined,
            len(train),
            logical_shards,
            topology_iters,
        )
        refinement_seconds = time.perf_counter() - started

    started = time.perf_counter()
    assignments, point_to_shards = orion.assign_points_by_l1_vote(
        point_to_l1s,
        refined,
        logical_shards,
        False,
    )
    assignment_seconds = time.perf_counter() - started
    if any(len(shards) != 1 for shards in point_to_shards):
        raise RuntimeError("C23 Orion partition must be disjoint")
    return assignments.astype(np.int32, copy=False), {
        "initial_label_seconds": initial_seconds,
        "topology_refinement_seconds": refinement_seconds,
        "topology_refinement_iterations": refinement_iterations,
        "unsampled_assignment_seconds": assignment_seconds,
    }


def build_orion_matrix(
    hdf5_path: str | Path,
    spec: DatasetSpec,
    output_root: str | Path,
    *,
    logical_shards: Iterable[int] = LOGICAL_SHARDS,
    upper_sample_seed: int = 100,
    kmeans_seed: int = 1,
    upper_m: int = 32,
    upper_ef_construction: int = 100,
    attachment_search_ef: int = 100,
    k_nav: int = 10,
    sample_denominator: int = 32,
    kmeans_iters: int = 10,
    topology_iters: int = 50,
    batch_size: int = 10_000,
    include_no_refinement: bool = True,
) -> dict[str, Any]:
    audit = dataset_audit(hdf5_path, spec)
    output = Path(output_root).expanduser().resolve() / spec.name
    output.mkdir(parents=True, exist_ok=True)
    train, dataset_record = orion_native_layout.load_train_vectors(
        Path(hdf5_path).expanduser().resolve(),
        None,
        spec.metric,
    )
    upper_indices = orion.global_upper_indices(len(train), sample_denominator, upper_sample_seed)
    if len(upper_indices) < max(int(value) for value in logical_shards):
        raise ValueError("navigation sample is smaller than requested shard count")

    started = time.perf_counter()
    upper_index = orion.build_upper_index(
        train[upper_indices],
        upper_indices.astype(np.int64, copy=False),
        train.shape[1],
        upper_m,
        upper_ef_construction,
        attachment_search_ef,
        spec.hnsw_space,
    )
    navigation_graph_seconds = time.perf_counter() - started
    navigation_path = output / (
        f"navigation-seed{upper_sample_seed}-m{upper_m}-ef{upper_ef_construction}.hnsw"
    )
    if navigation_path.exists():
        raise FileExistsError(f"refusing to overwrite navigation graph: {navigation_path}")
    upper_index.save_index(str(navigation_path))

    started = time.perf_counter()
    point_to_l1s = orion.compute_point_to_l1s(upper_index, train, k_nav, batch_size)
    attachment_seconds = time.perf_counter() - started
    attachment_path = output / f"point-to-l1-k{k_nav}-seed{upper_sample_seed}.npy"
    if attachment_path.exists():
        raise FileExistsError(f"refusing to overwrite attachment matrix: {attachment_path}")
    attachment_array = np.asarray(point_to_l1s, dtype=np.int32)
    np.save(attachment_path, attachment_array, allow_pickle=False)
    upper_path = output / f"upper-indices-seed{upper_sample_seed}.npy"
    if upper_path.exists():
        raise FileExistsError(f"refusing to overwrite upper indices: {upper_path}")
    np.save(upper_path, upper_indices, allow_pickle=False)

    artifacts: list[dict[str, Any]] = []
    variants = (("orion", True), ("orion_no_refinement", False))
    if not include_no_refinement:
        variants = variants[:1]
    for shard_count in [int(value) for value in logical_shards]:
        for method, refine in variants:
            assignments, timings = _orion_assignments(
                train,
                upper_indices,
                point_to_l1s,
                shard_count,
                kmeans_seed=kmeans_seed,
                kmeans_iters=kmeans_iters,
                topology_iters=topology_iters,
                refine=refine,
            )
            metadata = {
                "protocol_version": PROTOCOL_VERSION,
                "timestamp": utc_timestamp(),
                "dataset": asdict(spec),
                "dataset_path": audit["path"],
                "dataset_sha256": audit["sha256"],
                "method": method,
                "logical_shards": shard_count,
                "partition_seed": kmeans_seed,
                "point_count": len(train),
                "dimension": int(train.shape[1]),
                "distance_metric": spec.metric,
                "disjoint": True,
                "multi_assignment": False,
                "fission": False,
                "navigation_sample_size": int(len(upper_indices)),
                "navigation_sample_rate": float(len(upper_indices) / len(train)),
                "navigation_sample_seed": upper_sample_seed,
                "navigation_graph_path": str(navigation_path),
                "navigation_graph_sha256": sha256_path(navigation_path),
                "navigation_graph_memory_bytes": navigation_path.stat().st_size,
                "navigation_graph_construction_seconds": navigation_graph_seconds,
                "attachment_search_seconds": attachment_seconds,
                "attachment_search_ef": attachment_search_ef,
                "k_nav": k_nav,
                "upper_m": upper_m,
                "upper_ef_construction": upper_ef_construction,
                "kmeans_iterations": kmeans_iters,
                "topology_iteration_limit": topology_iters,
                "topology_refinement_enabled": refine,
                "peak_construction_memory_bytes": peak_rss_bytes(),
                "construction_timings": timings,
                "attachment_matrix_path": str(attachment_path),
                "attachment_matrix_sha256": sha256_path(attachment_path),
                "upper_indices_path": str(upper_path),
                "upper_indices_sha256": sha256_path(upper_path),
            }
            destination = output / f"{method}-m{shard_count}-seed{kmeans_seed}.npz"
            artifacts.append(save_partition(destination, assignments, metadata))
    return {
        "protocol_version": PROTOCOL_VERSION,
        "timestamp": utc_timestamp(),
        "dataset": spec.name,
        "dataset_record": dataset_record,
        "navigation_graph": str(navigation_path),
        "attachment_matrix": str(attachment_path),
        "artifacts": artifacts,
    }


def parse_shards(value: str) -> tuple[int, ...]:
    result = tuple(int(item) for item in value.split(",") if item.strip())
    invalid = sorted(set(result).difference(LOGICAL_SHARDS))
    if not result or invalid:
        raise argparse.ArgumentTypeError(f"shards must be a subset of {LOGICAL_SHARDS}")
    return result


def write_json_new(path: str | Path, value: Any) -> None:
    target = Path(path).expanduser().resolve()
    if target.exists():
        raise FileExistsError(f"refusing to overwrite: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit_parser = subparsers.add_parser("audit-dataset")
    audit_parser.add_argument("--dataset", choices=tuple(DATASETS), required=True)
    audit_parser.add_argument("--hdf5-path", required=True)
    audit_parser.add_argument("--output", required=True)

    partition_parser = subparsers.add_parser("audit-partition")
    partition_parser.add_argument("--artifact", required=True)
    partition_parser.add_argument("--method", default=None)

    baseline_parser = subparsers.add_parser("audit-c1-baselines")
    baseline_parser.add_argument("--partition-root", required=True)
    baseline_parser.add_argument("--output", required=True)

    reference_parser = subparsers.add_parser("audit-reference")
    reference_parser.add_argument("--run-dir", required=True)
    reference_parser.add_argument("--output", required=True)

    export_parser = subparsers.add_parser("export-reference-input")
    export_parser.add_argument("--dataset", choices=tuple(DATASETS), required=True)
    export_parser.add_argument("--hdf5-path", required=True)
    export_parser.add_argument("--output-dir", required=True)
    export_parser.add_argument("--chunk-size", type=int, default=16_384)

    orion_parser = subparsers.add_parser("build-orion-matrix")
    orion_parser.add_argument("--dataset", choices=tuple(DATASETS), required=True)
    orion_parser.add_argument("--hdf5-path", required=True)
    orion_parser.add_argument("--artifact-root", required=True)
    orion_parser.add_argument("--shards", type=parse_shards, default=LOGICAL_SHARDS)
    orion_parser.add_argument("--upper-sample-seed", type=int, default=100)
    orion_parser.add_argument("--kmeans-seed", type=int, default=1)
    orion_parser.add_argument("--skip-no-refinement", action="store_true")
    orion_parser.add_argument("--output", required=True)

    args = parser.parse_args(argv)
    if args.command == "audit-dataset":
        write_json_new(args.output, dataset_audit(args.hdf5_path, DATASETS[args.dataset]))
    elif args.command == "audit-partition":
        assignments, metadata = load_partition(args.artifact, args.method)
        print(
            json.dumps(
                {
                    "artifact": str(Path(args.artifact).resolve()),
                    "sha256": sha256_path(args.artifact),
                    "point_count": len(assignments),
                    "metadata": metadata,
                },
                sort_keys=True,
            )
        )
    elif args.command == "audit-c1-baselines":
        write_json_new(args.output, audit_c1_baseline_matrix(args.partition_root))
    elif args.command == "audit-reference":
        write_json_new(args.output, audit_reference_run(args.run_dir))
    elif args.command == "export-reference-input":
        manifest = export_reference_input(
            args.hdf5_path,
            DATASETS[args.dataset],
            args.output_dir,
            chunk_size=args.chunk_size,
        )
        print(json.dumps(manifest["files"], sort_keys=True))
    elif args.command == "build-orion-matrix":
        result = build_orion_matrix(
            args.hdf5_path,
            DATASETS[args.dataset],
            args.artifact_root,
            logical_shards=args.shards,
            upper_sample_seed=args.upper_sample_seed,
            kmeans_seed=args.kmeans_seed,
            include_no_refinement=not args.skip_no_refinement,
        )
        result["git_commit"] = git_commit(Path(__file__).resolve().parents[3])
        write_json_new(args.output, result)
    else:
        raise AssertionError(args.command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
