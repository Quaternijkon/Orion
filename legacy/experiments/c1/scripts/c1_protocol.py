#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import h5py
import numpy as np


PROTOCOL_VERSION = 1
PARTITION_SEED = 20260820
TUNING_QUERY_COUNT = 1_000
TOP_K = 10
TARGET_RECALL = 0.90
LOGICAL_SHARD_COUNTS = (1, 2, 4, 8, 16, 32)


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    distance: str
    qdrant_distance: str
    normalize: bool


DATASETS = {
    "sift1m": DatasetSpec(
        name="sift1m",
        distance="euclidean",
        qdrant_distance="Euclid",
        normalize=False,
    ),
    "glove-200-angular": DatasetSpec(
        name="glove-200-angular",
        distance="angular",
        qdrant_distance="Cosine",
        normalize=True,
    ),
}


def utc_timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_path(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def repository_commit(repo: str | Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()


def normalize_rows(rows: np.ndarray) -> np.ndarray:
    values = np.asarray(rows, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return values / norms


def preprocess_rows(rows: np.ndarray, spec: DatasetSpec) -> np.ndarray:
    values = np.asarray(rows, dtype=np.float32)
    return normalize_rows(values) if spec.normalize else values


def dataset_audit(path: str | Path, spec: DatasetSpec) -> dict[str, Any]:
    dataset_path = Path(path).expanduser().resolve()
    if not dataset_path.is_file():
        raise FileNotFoundError(dataset_path)
    with h5py.File(dataset_path, "r") as handle:
        missing = sorted({"train", "test", "neighbors"} - set(handle.keys()))
        if missing:
            raise ValueError(f"dataset is missing HDF5 keys: {missing}")
        train = handle["train"]
        test = handle["test"]
        neighbors = handle["neighbors"]
        if train.ndim != 2 or test.ndim != 2 or neighbors.ndim != 2:
            raise ValueError("train, test, and neighbors must be two-dimensional")
        if train.shape[1] != test.shape[1]:
            raise ValueError("train and test dimensions differ")
        if test.shape[0] != neighbors.shape[0]:
            raise ValueError("test and neighbors row counts differ")
        if neighbors.shape[1] < TOP_K:
            raise ValueError(f"neighbors must provide at least {TOP_K} columns")
        if test.shape[0] <= TUNING_QUERY_COUNT:
            raise ValueError("dataset does not have queries remaining after the tuning split")
        attrs = {
            str(key): value.item() if hasattr(value, "item") else value
            for key, value in handle.attrs.items()
        }
        reported_distance = str(attrs.get("distance", "")).lower()
        if reported_distance and reported_distance != spec.distance:
            raise ValueError(
                f"dataset distance {reported_distance!r} does not match {spec.distance!r}"
            )
        shapes = {
            "train": list(map(int, train.shape)),
            "test": list(map(int, test.shape)),
            "neighbors": list(map(int, neighbors.shape)),
        }
        dtypes = {
            "train": str(train.dtype),
            "test": str(test.dtype),
            "neighbors": str(neighbors.dtype),
        }

    return {
        "protocol_version": PROTOCOL_VERSION,
        "timestamp": utc_timestamp(),
        "dataset": asdict(spec),
        "path": str(dataset_path),
        "size_bytes": dataset_path.stat().st_size,
        "sha256": sha256_path(dataset_path),
        "hdf5_shapes": shapes,
        "hdf5_dtypes": dtypes,
        "hdf5_attrs": attrs,
        "tuning_query_range": [0, TUNING_QUERY_COUNT],
        "measurement_query_range": [TUNING_QUERY_COUNT, shapes["test"][0]],
        "tuning_query_count": TUNING_QUERY_COUNT,
        "measurement_query_count": shapes["test"][0] - TUNING_QUERY_COUNT,
        "query_sets_disjoint": True,
    }


def deterministic_random_assignments(
    point_count: int,
    logical_shards: int,
    seed: int = PARTITION_SEED,
) -> np.ndarray:
    if point_count <= 0:
        raise ValueError("point_count must be positive")
    if logical_shards <= 0:
        raise ValueError("logical_shards must be positive")
    rng = np.random.default_rng(seed)
    dtype = np.uint8 if logical_shards <= 256 else np.uint16
    return rng.integers(0, logical_shards, size=point_count, dtype=dtype)


def centroid_assignments(
    rows: np.ndarray,
    centroids: np.ndarray,
    spec: DatasetSpec,
) -> np.ndarray:
    points = preprocess_rows(rows, spec)
    centers = preprocess_rows(centroids, spec)
    if points.ndim != 2 or centers.ndim != 2 or points.shape[1] != centers.shape[1]:
        raise ValueError("rows and centroids must be compatible two-dimensional arrays")
    if spec.normalize:
        return np.argmax(points @ centers.T, axis=1).astype(np.int32, copy=False)
    point_norms = np.einsum("ij,ij->i", points, points)[:, None]
    center_norms = np.einsum("ij,ij->i", centers, centers)[None, :]
    distances = point_norms + center_norms - 2.0 * (points @ centers.T)
    return np.argmin(distances, axis=1).astype(np.int32, copy=False)


def _minimum_distance_to_centroids(
    rows: np.ndarray,
    centroids: np.ndarray,
    spec: DatasetSpec,
) -> np.ndarray:
    points = preprocess_rows(rows, spec)
    centers = preprocess_rows(centroids, spec)
    if spec.normalize:
        return np.maximum(0.0, 1.0 - np.max(points @ centers.T, axis=1))
    point_norms = np.einsum("ij,ij->i", points, points)[:, None]
    center_norms = np.einsum("ij,ij->i", centers, centers)[None, :]
    distances = point_norms + center_norms - 2.0 * (points @ centers.T)
    return np.maximum(0.0, np.min(distances, axis=1))


def kmeans_plus_plus(
    sample: np.ndarray,
    logical_shards: int,
    spec: DatasetSpec,
    seed: int = PARTITION_SEED,
) -> np.ndarray:
    points = preprocess_rows(sample, spec)
    if logical_shards <= 0 or logical_shards > len(points):
        raise ValueError("logical_shards must be within the sample row count")
    rng = np.random.default_rng(seed)
    centroids = np.empty((logical_shards, points.shape[1]), dtype=np.float32)
    first = int(rng.integers(0, len(points)))
    centroids[0] = points[first]
    closest = _minimum_distance_to_centroids(points, centroids[:1], spec)
    for index in range(1, logical_shards):
        total = float(np.sum(closest, dtype=np.float64))
        if not math.isfinite(total) or total <= 0.0:
            chosen = int(rng.integers(0, len(points)))
        else:
            chosen = int(rng.choice(len(points), p=closest / total))
        centroids[index] = points[chosen]
        newest = _minimum_distance_to_centroids(points, centroids[index : index + 1], spec)
        closest = np.minimum(closest, newest)
    return preprocess_rows(centroids, spec)


def fit_kmeans(
    rows: np.ndarray,
    logical_shards: int,
    spec: DatasetSpec,
    *,
    seed: int = PARTITION_SEED,
    iterations: int = 12,
    tolerance: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    points = preprocess_rows(rows, spec)
    centroids = kmeans_plus_plus(points, logical_shards, spec, seed)
    history: list[dict[str, Any]] = []
    assignments = np.zeros(len(points), dtype=np.int32)
    for iteration in range(iterations):
        assignments = centroid_assignments(points, centroids, spec)
        counts = np.bincount(assignments, minlength=logical_shards).astype(np.int64)
        sums = np.zeros_like(centroids, dtype=np.float64)
        np.add.at(sums, assignments, points)
        updated = centroids.copy()
        nonempty = counts > 0
        updated[nonempty] = (sums[nonempty] / counts[nonempty, None]).astype(np.float32)
        updated = preprocess_rows(updated, spec)
        movement = float(np.max(np.linalg.norm(updated - centroids, axis=1)))
        history.append(
            {
                "iteration": iteration + 1,
                "max_centroid_movement": movement,
                "empty_clusters": int(np.sum(~nonempty)),
                "min_cluster_size": int(np.min(counts)),
                "max_cluster_size": int(np.max(counts)),
            }
        )
        centroids = updated
        if movement <= tolerance:
            break
    assignments = centroid_assignments(points, centroids, spec)
    return assignments, centroids, history


def _sample_hdf5_rows(
    dataset: h5py.Dataset,
    sample_size: int,
    seed: int,
) -> np.ndarray:
    if sample_size >= len(dataset):
        return dataset[:].astype(np.float32, copy=True)
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(dataset), size=sample_size, replace=False)
    order = np.argsort(indices)
    sorted_rows = dataset[indices[order]].astype(np.float32, copy=True)
    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))
    return sorted_rows[inverse]


def fit_hdf5_kmeans(
    hdf5_path: str | Path,
    logical_shards: int,
    spec: DatasetSpec,
    *,
    seed: int = PARTITION_SEED,
    sample_size: int = 100_000,
    iterations: int = 12,
    tolerance: float = 1e-4,
    batch_size: int = 16_384,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    with h5py.File(hdf5_path, "r") as handle:
        train = handle["train"]
        sample = _sample_hdf5_rows(train, min(sample_size, len(train)), seed)
        centroids = kmeans_plus_plus(sample, logical_shards, spec, seed)
        history: list[dict[str, Any]] = []
        for iteration in range(iterations):
            sums = np.zeros((logical_shards, train.shape[1]), dtype=np.float64)
            counts = np.zeros(logical_shards, dtype=np.int64)
            for start in range(0, len(train), batch_size):
                batch = preprocess_rows(train[start : start + batch_size], spec)
                assigned = centroid_assignments(batch, centroids, spec)
                np.add.at(sums, assigned, batch)
                counts += np.bincount(assigned, minlength=logical_shards)
            updated = centroids.copy()
            nonempty = counts > 0
            updated[nonempty] = (sums[nonempty] / counts[nonempty, None]).astype(
                np.float32
            )
            updated = preprocess_rows(updated, spec)
            movement = float(np.max(np.linalg.norm(updated - centroids, axis=1)))
            history.append(
                {
                    "iteration": iteration + 1,
                    "max_centroid_movement": movement,
                    "empty_clusters": int(np.sum(~nonempty)),
                    "min_cluster_size": int(np.min(counts)),
                    "max_cluster_size": int(np.max(counts)),
                }
            )
            centroids = updated
            if movement <= tolerance:
                break

        assignment_dtype = np.uint8 if logical_shards <= 256 else np.uint16
        assignments = np.empty(len(train), dtype=assignment_dtype)
        for start in range(0, len(train), batch_size):
            stop = min(start + batch_size, len(train))
            assignments[start:stop] = centroid_assignments(
                train[start:stop], centroids, spec
            ).astype(assignment_dtype, copy=False)
    return assignments, centroids, history


def required_hits(target_recall: float = TARGET_RECALL, top_k: int = TOP_K) -> int:
    if not 0.0 < target_recall <= 1.0:
        raise ValueError("target_recall must be within (0, 1]")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    return int(math.ceil(target_recall * top_k - 1e-12))


def oracle_minimum_fanout(
    assignments: np.ndarray,
    neighbors: np.ndarray,
    logical_shards: int,
    *,
    target_recall: float = TARGET_RECALL,
    top_k: int = TOP_K,
) -> np.ndarray:
    mapping = np.asarray(assignments)
    ground_truth = np.asarray(neighbors)
    if ground_truth.ndim != 2 or ground_truth.shape[1] < top_k:
        raise ValueError("neighbors does not contain enough ground-truth columns")
    top = ground_truth[:, :top_k].astype(np.int64, copy=False)
    if np.min(top) < 0 or np.max(top) >= len(mapping):
        raise ValueError("ground-truth IDs are outside the assignment array")
    shard_ids = mapping[top]
    needed = required_hits(target_recall, top_k)
    result = np.empty(len(top), dtype=np.uint16)
    for query_index, row in enumerate(shard_ids):
        counts = np.bincount(row.astype(np.int64), minlength=logical_shards)
        descending = np.sort(counts)[::-1]
        result[query_index] = int(np.searchsorted(np.cumsum(descending), needed) + 1)
    return result


def recall_per_query(
    predicted: Sequence[Sequence[int]],
    ground_truth: np.ndarray,
    top_k: int = TOP_K,
) -> np.ndarray:
    truth = np.asarray(ground_truth)
    if len(predicted) != len(truth):
        raise ValueError("predicted and ground_truth row counts differ")
    recalls = np.empty(len(truth), dtype=np.float64)
    for index, result_ids in enumerate(predicted):
        result = set(map(int, result_ids[:top_k]))
        expected = set(map(int, truth[index, :top_k]))
        recalls[index] = len(result & expected) / top_k
    return recalls


def distance_computations_from_usage(cpu_units: int, dimension: int) -> int:
    if cpu_units < 0 or dimension <= 0:
        raise ValueError("cpu_units must be non-negative and dimension must be positive")
    unit = dimension * np.dtype(np.float32).itemsize
    quotient, remainder = divmod(int(cpu_units), unit)
    if remainder:
        raise ValueError(
            f"hardware CPU units {cpu_units} are not divisible by dense score unit {unit}"
        )
    return quotient


def select_random_tuning_candidate(
    rows: Sequence[dict[str, Any]], target_recall: float = TARGET_RECALL
) -> dict[str, Any]:
    feasible = [row for row in rows if float(row["recall_at_10"]) >= target_recall]
    if not feasible:
        raise ValueError("no Random tuning candidate reaches the recall target")
    return min(feasible, key=lambda row: (int(row["ef_search"]), -float(row["recall_at_10"])))


def select_kmeans_tuning_candidate(
    rows: Sequence[dict[str, Any]], target_recall: float = TARGET_RECALL
) -> dict[str, Any]:
    feasible = [row for row in rows if float(row["recall_at_10"]) >= target_recall]
    if not feasible:
        raise ValueError("no K-Means tuning candidate reaches the recall target")
    best_work = min(float(row["mean_aggregate_distance_computations"]) for row in feasible)
    near_ties = [
        row
        for row in feasible
        if float(row["mean_aggregate_distance_computations"]) <= best_work * 1.02
    ]
    return min(
        near_ties,
        key=lambda row: (
            int(row["fanout"]),
            float(row["mean_aggregate_distance_computations"]),
            int(row["ef_search"]),
        ),
    )


def percentile(values: Sequence[float], percent: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), percent))


def partition_artifact(
    hdf5_path: str | Path,
    spec: DatasetSpec,
    method: str,
    logical_shards: int,
    output_path: str | Path,
    *,
    seed: int = PARTITION_SEED,
    sample_size: int = 100_000,
    iterations: int = 12,
    tolerance: float = 1e-4,
    batch_size: int = 16_384,
) -> dict[str, Any]:
    audit = dataset_audit(hdf5_path, spec)
    point_count = int(audit["hdf5_shapes"]["train"][0])
    if method == "random":
        assignments = deterministic_random_assignments(point_count, logical_shards, seed)
        centroids = np.empty((0, int(audit["hdf5_shapes"]["train"][1])), dtype=np.float32)
        history: list[dict[str, Any]] = []
    elif method == "kmeans":
        assignments, centroids, history = fit_hdf5_kmeans(
            hdf5_path,
            logical_shards,
            spec,
            seed=seed,
            sample_size=sample_size,
            iterations=iterations,
            tolerance=tolerance,
            batch_size=batch_size,
        )
    else:
        raise ValueError(f"unsupported partition method: {method}")

    counts = np.bincount(assignments.astype(np.int64), minlength=logical_shards)
    metadata = {
        "protocol_version": PROTOCOL_VERSION,
        "timestamp": utc_timestamp(),
        "dataset": asdict(spec),
        "dataset_path": audit["path"],
        "dataset_sha256": audit["sha256"],
        "method": method,
        "logical_shards": logical_shards,
        "partition_seed": seed,
        "point_count": point_count,
        "dimension": int(audit["hdf5_shapes"]["train"][1]),
        "shard_counts": counts.astype(int).tolist(),
        "kmeans": {
            "sample_size": sample_size,
            "maximum_iterations": iterations,
            "tolerance": tolerance,
            "batch_size": batch_size,
            "history": history,
        }
        if method == "kmeans"
        else None,
    }
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp.npz")
    np.savez_compressed(
        temporary,
        assignments=assignments,
        centroids=centroids,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    os.replace(temporary, destination)
    metadata["artifact_path"] = str(destination)
    metadata["artifact_sha256"] = sha256_path(destination)
    return metadata


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="C1 protocol data and partition utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit_parser = subparsers.add_parser("dataset-audit")
    audit_parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    audit_parser.add_argument("--hdf5-path", required=True)
    audit_parser.add_argument("--output")

    partition_parser = subparsers.add_parser("partition")
    partition_parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    partition_parser.add_argument("--hdf5-path", required=True)
    partition_parser.add_argument("--method", choices=("random", "kmeans"), required=True)
    partition_parser.add_argument(
        "--logical-shards", type=int, choices=LOGICAL_SHARD_COUNTS, required=True
    )
    partition_parser.add_argument("--output", required=True)
    partition_parser.add_argument("--seed", type=int, default=PARTITION_SEED)
    partition_parser.add_argument("--sample-size", type=int, default=100_000)
    partition_parser.add_argument("--iterations", type=int, default=12)
    partition_parser.add_argument("--tolerance", type=float, default=1e-4)
    partition_parser.add_argument("--batch-size", type=int, default=16_384)
    return parser.parse_args(argv)


def write_json_output(payload: dict[str, Any], destination: str | None) -> None:
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if destination:
        path = Path(destination).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(rendered + "\n", encoding="utf-8")
        os.replace(temporary, path)
    print(rendered)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    spec = DATASETS[args.dataset]
    if args.command == "dataset-audit":
        write_json_output(dataset_audit(args.hdf5_path, spec), args.output)
        return 0
    if args.command == "partition":
        payload = partition_artifact(
            args.hdf5_path,
            spec,
            args.method,
            args.logical_shards,
            args.output,
            seed=args.seed,
            sample_size=args.sample_size,
            iterations=args.iterations,
            tolerance=args.tolerance,
            batch_size=args.batch_size,
        )
        write_json_output(payload, None)
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    sys.exit(main())
