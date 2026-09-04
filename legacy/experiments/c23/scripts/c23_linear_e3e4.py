#!/usr/bin/env python3
"""Build and measure resource-isolated virtual C23 shards on four hosts."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import http.client
import json
import os
import sys
import tempfile
import threading
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import h5py
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import c23_linear_resources as resources


TOP_K = 10
TARGET_HITS = 9
COLLECTION = "c23_linear"
INDEXING_THRESHOLD_KIB = 10
TAIL_COMPACTION_THRESHOLD_KIB = 1
INDEX_GATE_STABLE_SECONDS = 5.0
METHODS = ("random", "kmeans", "orion", "orion_no_refinement")
EF_GRID = (10, 16, 20, 24, 32, 40, 80, 160, 320)
DATASETS = {
    "sift1m": {
        "filename": "sift-128-euclidean.hdf5",
        "dimension": 128,
        "distance": "Euclid",
        "normalize": False,
        "train_rows": 1_000_000,
        "test_rows": 10_000,
        "sha256": "dd6f0a6ed6b7ebb8934680f861a33ed01ff33991eaee4fd60914d854a0ca5984",
    },
    "glove-200-angular": {
        "filename": "glove-200-angular.hdf5",
        "dimension": 200,
        "distance": "Cosine",
        "normalize": True,
        "train_rows": 1_183_514,
        "test_rows": 10_000,
        "sha256": "4839085e5a8bb293434a1a66e1aa0193afc3f07c6797a85f1dbd91656172da20",
    },
}


@dataclass(frozen=True)
class SearchResult:
    shard_id: int
    result_ids: tuple[int, ...]
    distance_computations: int
    graph_nodes_visited: int
    worker_cpu_time_us: int
    worker_cpu_wall_time_us: int
    latency_us: float
    response_bytes: int


class JsonHttpClient:
    def __init__(self) -> None:
        self._local = threading.local()

    def _connections(self) -> dict[tuple[str, int], http.client.HTTPConnection]:
        value = getattr(self._local, "connections", None)
        if value is None:
            value = {}
            self._local.connections = value
        return value

    def request(
        self,
        base_url: str,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
        *,
        timeout: float = 900.0,
    ) -> tuple[dict[str, Any], int]:
        parsed = urllib.parse.urlsplit(base_url)
        if parsed.scheme != "http" or not parsed.hostname:
            raise ValueError(f"invalid base URL: {base_url}")
        key = (parsed.hostname, parsed.port or 80)
        encoded = None if body is None else json.dumps(body, separators=(",", ":")).encode()
        headers = {"Accept": "application/json"}
        if encoded is not None:
            headers["Content-Type"] = "application/json"
        last_error: BaseException | None = None
        for attempt in range(2):
            connection = self._connections().get(key)
            if connection is None:
                connection = http.client.HTTPConnection(*key, timeout=timeout)
                self._connections()[key] = connection
            connection.timeout = timeout
            try:
                connection.request(method, path, body=encoded, headers=headers)
                response = connection.getresponse()
                raw = response.read()
                if response.status >= 400:
                    raise RuntimeError(
                        f"{method} {base_url}{path} returned HTTP {response.status}: "
                        f"{raw.decode(errors='replace')}"
                    )
                payload = json.loads(raw.decode())
                if not isinstance(payload, dict):
                    raise RuntimeError("Qdrant response is not a JSON object")
                status = payload.get("status")
                if isinstance(status, dict) and status.get("error"):
                    raise RuntimeError(f"Qdrant error: {status['error']}")
                return payload, len(raw)
            except (BrokenPipeError, ConnectionError, http.client.HTTPException, OSError) as error:
                last_error = error
                stale = self._connections().pop(key, None)
                if stale is not None:
                    stale.close()
                if attempt:
                    raise RuntimeError(f"request failed: {base_url}{path}: {error}") from error
        raise AssertionError(last_error)


HTTP = JsonHttpClient()


def sha256_path(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: str | Path, payload: Any) -> None:
    resources.write_json_atomic(path, payload)


def preprocess(rows: np.ndarray, *, normalize: bool) -> np.ndarray:
    values = np.asarray(rows, dtype=np.float32)
    if normalize:
        norms = np.linalg.norm(values, axis=1, keepdims=True)
        values = values / np.maximum(norms, np.finfo(np.float32).tiny)
    return np.ascontiguousarray(values, dtype=np.float32)


def parse_cpuset(value: str) -> list[int]:
    cpus: set[int] = set()
    for raw_part in value.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "-" in part:
            raw_start, raw_stop = part.split("-", 1)
            start, stop = int(raw_start), int(raw_stop)
            if start < 0 or stop < start:
                raise ValueError(f"invalid CPU range: {part}")
            cpus.update(range(start, stop + 1))
        else:
            cpus.add(int(part))
    if not cpus:
        raise ValueError("CPU set must not be empty")
    return sorted(cpus)


def validate_client_affinity(topology: dict[str, Any]) -> dict[str, Any]:
    expected = parse_cpuset(str(topology["benchmark_client_cpuset"]))
    actual = sorted(os.sched_getaffinity(0))
    task_affinities: dict[str, list[int]] = {}
    for task in sorted(
        Path(f"/proc/{os.getpid()}/task").iterdir(), key=lambda path: int(path.name)
    ):
        task_affinities[task.name] = sorted(os.sched_getaffinity(int(task.name)))
    all_threads_match = all(value == expected for value in task_affinities.values())
    if actual != expected or not all_threads_match:
        raise RuntimeError(
            f"benchmark client affinity mismatch: expected={expected} actual={actual} "
            f"threads={task_affinities}"
        )
    return {
        "pid": os.getpid(),
        "expected": expected,
        "actual": actual,
        "task_affinities": task_affinities,
        "status": "PASS",
    }


def distance_computations(cpu_units: int, dimension: int) -> int:
    unit = dimension * np.dtype(np.float32).itemsize
    quotient, remainder = divmod(int(cpu_units), unit)
    if remainder:
        raise ValueError(
            f"hardware CPU units {cpu_units} are not divisible by dense-score unit {unit}"
        )
    return quotient


def partition_artifact(
    topology: dict[str, Any], dataset: str, method: str, logical_shards: int
) -> Path | None:
    if logical_shards == 1:
        return None
    if method in {"random", "kmeans"}:
        return (
            Path(topology["legacy_partition_root"])
            / dataset
            / f"{method}-m{logical_shards}.npz"
        )
    if method in {"orion", "orion_no_refinement"}:
        return (
            Path(topology["orion_partition_roots"][dataset])
            / f"{method}-m{logical_shards}-seed1.npz"
        )
    raise ValueError(f"unsupported partition method: {method}")


def load_assignments(
    topology: dict[str, Any], dataset: str, method: str, logical_shards: int
) -> tuple[np.ndarray, dict[str, Any]]:
    spec = DATASETS[dataset]
    artifact = partition_artifact(topology, dataset, method, logical_shards)
    if artifact is None:
        assignments = np.zeros(int(spec["train_rows"]), dtype=np.int32)
        return assignments, {
            "method": "common_m1",
            "logical_shards": 1,
            "point_count": int(spec["train_rows"]),
            "partition_seed": None,
            "artifact": None,
            "artifact_sha256": None,
        }
    with np.load(artifact, allow_pickle=False) as archive:
        assignments = np.asarray(archive["assignments"], dtype=np.int32).copy()
        metadata = json.loads(str(np.asarray(archive["metadata_json"]).item()))
    if assignments.shape != (int(spec["train_rows"]),):
        raise ValueError(f"partition point count mismatch: {artifact}")
    if int(metadata["logical_shards"]) != logical_shards:
        raise ValueError(f"partition shard count mismatch: {artifact}")
    if str(metadata["method"]) != method:
        raise ValueError(f"partition method mismatch: {artifact}")
    if str(metadata.get("dataset_sha256")) != str(spec["sha256"]):
        raise ValueError(f"partition dataset binding mismatch: {artifact}")
    counts = np.bincount(assignments.astype(np.int64), minlength=logical_shards)
    if len(counts) != logical_shards or np.any(counts == 0):
        raise ValueError(f"partition has empty or invalid shards: {artifact}")
    metadata = dict(metadata)
    metadata.update(
        {
            "artifact": str(artifact.resolve()),
            "artifact_sha256": sha256_path(artifact),
        }
    )
    return assignments, metadata


def dataset_path(topology: dict[str, Any], dataset: str) -> Path:
    return Path(topology["dataset_root"]) / str(DATASETS[dataset]["filename"])


def audit_dataset(path: Path, dataset: str) -> dict[str, Any]:
    spec = DATASETS[dataset]
    checksum = sha256_path(path)
    if checksum != spec["sha256"]:
        raise ValueError(f"dataset checksum mismatch: {path}: {checksum}")
    with h5py.File(path, "r") as handle:
        train_shape = tuple(map(int, handle["train"].shape))
        test_shape = tuple(map(int, handle["test"].shape))
        truth_shape = tuple(map(int, handle["neighbors"].shape))
    expected_train = (int(spec["train_rows"]), int(spec["dimension"]))
    expected_test = (int(spec["test_rows"]), int(spec["dimension"]))
    if train_shape != expected_train or test_shape != expected_test or truth_shape[0] != expected_test[0]:
        raise ValueError(
            f"dataset shape mismatch: train={train_shape} test={test_shape} truth={truth_shape}"
        )
    return {
        "path": str(path.resolve()),
        "sha256": checksum,
        "train_shape": train_shape,
        "test_shape": test_shape,
        "truth_shape": truth_shape,
    }


def encoded_collection() -> str:
    return urllib.parse.quote(COLLECTION, safe="")


def create_collection(base_url: str, dimension: int, distance: str, hnsw: dict[str, Any]) -> None:
    HTTP.request(
        base_url,
        "PUT",
        f"/collections/{encoded_collection()}",
        {
            "vectors": {"size": dimension, "distance": distance},
            "shard_number": 1,
            "replication_factor": 1,
            "write_consistency_factor": 1,
            "hnsw_config": {
                "m": int(hnsw["m"]),
                "ef_construct": int(hnsw["ef_construct"]),
                "full_scan_threshold": int(hnsw["full_scan_threshold"]),
                "max_indexing_threads": int(hnsw["max_indexing_threads"]),
            },
            "optimizers_config": {
                "default_segment_number": 1,
                "indexing_threshold": INDEXING_THRESHOLD_KIB,
                "max_segment_size": 2_000_000,
                "max_optimization_threads": int(hnsw["max_optimization_threads"]),
            },
        },
        timeout=300.0,
    )


def upload_points(base_url: str, point_ids: np.ndarray, vectors: np.ndarray) -> int:
    points = [
        {"id": int(point_id), "vector": vector.tolist()}
        for point_id, vector in zip(point_ids, vectors, strict=True)
    ]
    HTTP.request(
        base_url,
        "PUT",
        f"/collections/{encoded_collection()}/points?wait=true",
        {"points": points},
        timeout=900.0,
    )
    return len(points)


def create_and_upload(
    placements: Sequence[dict[str, Any]],
    vectors: np.ndarray | h5py.Dataset,
    assignments: np.ndarray,
    *,
    dimension: int,
    distance: str,
    normalize: bool,
    hnsw: dict[str, Any],
    chunk_size: int,
    upload_batch_size: int,
) -> dict[str, Any]:
    if len(placements) != int(assignments.max()) + 1:
        raise ValueError("placement and assignment shard counts differ")
    started = time.perf_counter()
    for placement in placements:
        create_collection(str(placement["base_url"]), dimension, distance, hnsw)
    counts = np.bincount(assignments.astype(np.int64), minlength=len(placements))
    uploaded = np.zeros(len(placements), dtype=np.int64)
    with ThreadPoolExecutor(max_workers=min(64, max(4, len(placements) * 2))) as executor:
        for chunk_start in range(0, len(assignments), chunk_size):
            chunk_stop = min(chunk_start + chunk_size, len(assignments))
            rows = preprocess(vectors[chunk_start:chunk_stop], normalize=normalize)
            labels = assignments[chunk_start:chunk_stop]
            futures = []
            for shard_id, placement in enumerate(placements):
                offsets = np.flatnonzero(labels == shard_id)
                for batch_start in range(0, len(offsets), upload_batch_size):
                    selected = offsets[batch_start : batch_start + upload_batch_size]
                    if not len(selected):
                        continue
                    point_ids = selected.astype(np.int64) + chunk_start
                    futures.append(
                        (
                            shard_id,
                            executor.submit(
                                upload_points,
                                str(placement["base_url"]),
                                point_ids,
                                rows[selected],
                            ),
                        )
                    )
            for shard_id, future in futures:
                uploaded[shard_id] += future.result()
            if chunk_stop == len(assignments) or chunk_stop % max(chunk_size * 10, 1) == 0:
                print(f"upload: {chunk_stop}/{len(assignments)} points", flush=True)
    if not np.array_equal(uploaded, counts):
        raise RuntimeError(f"uploaded counts differ: expected={counts} actual={uploaded}")
    return {
        "elapsed_seconds": time.perf_counter() - started,
        "expected_points_per_shard": counts.astype(int).tolist(),
        "uploaded_points_per_shard": uploaded.astype(int).tolist(),
        "total_uploaded_points": int(uploaded.sum()),
    }


def collection_info(base_url: str) -> dict[str, Any]:
    payload, _ = HTTP.request(
        base_url,
        "GET",
        f"/collections/{encoded_collection()}",
        timeout=60.0,
    )
    return dict(payload["result"])


def update_indexing_threshold(base_url: str, threshold_kib: int) -> None:
    if threshold_kib <= 0:
        raise ValueError("indexing threshold must be positive")
    HTTP.request(
        base_url,
        "PATCH",
        f"/collections/{encoded_collection()}",
        {"optimizers_config": {"indexing_threshold": int(threshold_kib)}},
        timeout=60.0,
    )


def collection_segment_telemetry(base_url: str) -> dict[str, Any]:
    payload, _ = HTTP.request(
        base_url,
        "GET",
        "/telemetry?details_level=4",
        timeout=60.0,
    )
    result = payload.get("result")
    if not isinstance(result, dict):
        raise RuntimeError(f"telemetry result is missing for {base_url}")
    collections = (result.get("collections") or {}).get("collections") or []
    matching = [row for row in collections if row.get("id") == COLLECTION]
    if len(matching) != 1:
        raise RuntimeError(
            f"expected one {COLLECTION!r} telemetry row at {base_url}, got {len(matching)}"
        )
    collection = matching[0]
    optimizer = (collection.get("config") or {}).get("optimizer_config") or {}
    threshold = optimizer.get("indexing_threshold")
    if threshold is None:
        raise RuntimeError(f"telemetry indexing threshold is missing for {base_url}")
    segments: list[dict[str, Any]] = []
    for shard in collection.get("shards") or []:
        local = shard.get("local") or {}
        for segment in local.get("segments") or []:
            info = segment.get("info") or {}
            segments.append(
                {
                    "shard_id": int(shard.get("id") or 0),
                    "segment_type": str(info.get("segment_type") or "").lower(),
                    "num_vectors": int(info.get("num_vectors") or 0),
                    "num_indexed_vectors": int(info.get("num_indexed_vectors") or 0),
                }
            )
    return {
        "collection_id": str(collection.get("id")),
        "indexing_threshold_kib": int(threshold),
        "segments": segments,
    }


def single_hnsw_layout_matches(telemetry: dict[str, Any], expected: int) -> bool:
    segments = telemetry.get("segments") or []
    indexed = [row for row in segments if row.get("segment_type") == "indexed"]
    plain = [row for row in segments if row.get("segment_type") == "plain"]
    return (
        len(segments) == 2
        and len(indexed) == 1
        and len(plain) == 1
        and int(indexed[0].get("num_vectors") or 0) == expected
        and int(indexed[0].get("num_indexed_vectors") or 0) == expected
        and int(plain[0].get("num_vectors") or 0) == 0
        and int(plain[0].get("num_indexed_vectors") or 0) == 0
    )


def plain_tail_layout_matches(telemetry: dict[str, Any], expected: int) -> bool:
    segments = telemetry.get("segments") or []
    indexed = [row for row in segments if row.get("segment_type") == "indexed"]
    plain = [row for row in segments if row.get("segment_type") == "plain"]
    if len(segments) != 2 or len(indexed) != 1 or len(plain) != 1:
        return False
    indexed_vectors = int(indexed[0].get("num_vectors") or 0)
    plain_vectors = int(plain[0].get("num_vectors") or 0)
    return (
        int(telemetry.get("indexing_threshold_kib") or 0) == INDEXING_THRESHOLD_KIB
        and indexed_vectors > 0
        and int(indexed[0].get("num_indexed_vectors") or 0) == indexed_vectors
        and plain_vectors > 0
        and int(plain[0].get("num_indexed_vectors") or 0) == 0
        and indexed_vectors + plain_vectors == expected
    )


def collection_base_index_ready(info: dict[str, Any], expected: int) -> bool:
    return (
        int(info.get("points_count") or 0) == expected
        and int(info.get("segments_count") or 0) == 2
        and str(info.get("status") or "").lower() == "green"
        and info.get("optimizer_status") == "ok"
    )


def _map_placements(
    placements: Sequence[dict[str, Any]], function: Any
) -> list[Any]:
    with ThreadPoolExecutor(max_workers=len(placements)) as executor:
        return list(
            executor.map(lambda row: function(str(row["base_url"])), placements)
        )


def wait_indexed(
    placements: Sequence[dict[str, Any]], expected_counts: Sequence[int], *, timeout: float
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    stable_since: float | None = None
    last: list[dict[str, Any]] = []
    last_telemetry: list[dict[str, Any]] = []
    affected_indices: list[int] = []
    thresholds_restored = False
    compaction: dict[str, Any] = {
        "status": "NOT_REQUIRED",
        "affected_shards": [],
    }
    while time.monotonic() < deadline:
        last = _map_placements(placements, collection_info)
        base_ready = all(
            collection_base_index_ready(info, expected)
            for info, expected in zip(last, expected_counts, strict=True)
        )
        if base_ready:
            last_telemetry = _map_placements(placements, collection_segment_telemetry)
            final_ready = all(
                int(telemetry.get("indexing_threshold_kib") or 0)
                == INDEXING_THRESHOLD_KIB
                and single_hnsw_layout_matches(telemetry, expected)
                for telemetry, expected in zip(
                    last_telemetry, expected_counts, strict=True
                )
            )
            if final_ready:
                stable_since = stable_since or time.monotonic()
            else:
                stable_since = None
                if not affected_indices:
                    affected_indices = [
                        index
                        for index, (telemetry, expected) in enumerate(
                            zip(last_telemetry, expected_counts, strict=True)
                        )
                        if not single_hnsw_layout_matches(telemetry, expected)
                    ]
                    unexpected = [
                        index
                        for index in affected_indices
                        if not plain_tail_layout_matches(
                            last_telemetry[index], expected_counts[index]
                        )
                    ]
                    if unexpected:
                        raise RuntimeError(
                            "unexpected segment layout before tail compaction: "
                            f"shards={unexpected} telemetry={last_telemetry}"
                        )
                    affected_placements = [placements[index] for index in affected_indices]
                    _map_placements(
                        affected_placements,
                        lambda base_url: update_indexing_threshold(
                            base_url, TAIL_COMPACTION_THRESHOLD_KIB
                        ),
                    )
                    compaction = {
                        "status": "IN_PROGRESS",
                        "reason": "green_two_segment_plain_tail_below_indexing_threshold",
                        "affected_shards": [
                            int(placements[index].get("shard_id", index))
                            for index in affected_indices
                        ],
                        "original_indexing_threshold_kib": INDEXING_THRESHOLD_KIB,
                        "temporary_indexing_threshold_kib": TAIL_COMPACTION_THRESHOLD_KIB,
                        "before": [last_telemetry[index] for index in affected_indices],
                    }
                elif not thresholds_restored:
                    temporary_ready = all(
                        int(last_telemetry[index].get("indexing_threshold_kib") or 0)
                        == TAIL_COMPACTION_THRESHOLD_KIB
                        and single_hnsw_layout_matches(
                            last_telemetry[index], expected_counts[index]
                        )
                        for index in affected_indices
                    )
                    if temporary_ready:
                        compaction["temporary_layout"] = [
                            last_telemetry[index] for index in affected_indices
                        ]
                        affected_placements = [
                            placements[index] for index in affected_indices
                        ]
                        _map_placements(
                            affected_placements,
                            lambda base_url: update_indexing_threshold(
                                base_url, INDEXING_THRESHOLD_KIB
                            ),
                        )
                        compaction["restored_indexing_threshold_kib"] = (
                            INDEXING_THRESHOLD_KIB
                        )
                        thresholds_restored = True
            if (
                final_ready
                and stable_since is not None
                and time.monotonic() - stable_since >= INDEX_GATE_STABLE_SECONDS
            ):
                if affected_indices:
                    compaction["status"] = "PASS"
                    compaction["final_layout"] = [
                        last_telemetry[index] for index in affected_indices
                    ]
                return {
                    "status": "PASS",
                    "expected_total_segments": 2 * len(placements),
                    "observed_total_segments": sum(int(info["segments_count"]) for info in last),
                    "expected_nonempty_hnsw_segments": len(placements),
                    "collection_info": last,
                    "segment_telemetry": last_telemetry,
                    "tail_compaction": compaction,
                }
        else:
            stable_since = None
        time.sleep(1)
    raise TimeoutError(
        "collections did not finish indexing: "
        f"collection_info={last} segment_telemetry={last_telemetry} "
        f"tail_compaction={compaction}"
    )


def search_one(
    placement: dict[str, Any], query: np.ndarray, ef_search: int, dimension: int
) -> SearchResult:
    started = time.perf_counter_ns()
    payload, response_bytes = HTTP.request(
        str(placement["base_url"]),
        "POST",
        f"/collections/{encoded_collection()}/points/search",
        {
            "vector": np.asarray(query, dtype=np.float32).tolist(),
            "limit": TOP_K,
            "with_payload": False,
            "with_vector": False,
            "params": {"hnsw_ef": ef_search, "exact": False},
        },
        timeout=300.0,
    )
    latency_us = (time.perf_counter_ns() - started) / 1_000
    hardware = ((payload.get("usage") or {}).get("hardware") or {})
    required = {"cpu", "graph_nodes_visited", "cpu_time_us", "cpu_wall_time_us"}
    missing = sorted(required - set(hardware))
    if missing:
        raise RuntimeError(f"search response is missing hardware fields: {missing}")
    return SearchResult(
        shard_id=int(placement["shard_id"]),
        result_ids=tuple(int(row["id"]) for row in payload.get("result") or []),
        distance_computations=distance_computations(int(hardware["cpu"]), dimension),
        graph_nodes_visited=int(hardware["graph_nodes_visited"]),
        worker_cpu_time_us=int(hardware["cpu_time_us"]),
        worker_cpu_wall_time_us=int(hardware["cpu_wall_time_us"]),
        latency_us=latency_us,
        response_bytes=response_bytes,
    )


def minimum_shards(contributions: np.ndarray) -> int:
    cumulative = 0
    for index, value in enumerate(sorted(map(int, contributions), reverse=True), 1):
        cumulative += value
        if cumulative >= TARGET_HITS:
            return index
    return len(contributions) + 1


def evaluate_one_query(
    query_id: int,
    query: np.ndarray,
    truth: np.ndarray,
    truth_assignments: np.ndarray,
    placements: Sequence[dict[str, Any]],
    ef_search: int,
    dimension: int,
    request_executor: ThreadPoolExecutor,
) -> dict[str, Any]:
    started = time.perf_counter_ns()
    futures = [
        request_executor.submit(search_one, placement, query, ef_search, dimension)
        for placement in placements
    ]
    results = [future.result() for future in futures]
    full_wall_us = (time.perf_counter_ns() - started) / 1_000
    results.sort(key=lambda row: row.shard_id)
    truth_set = set(map(int, truth[:TOP_K]))
    contributions = np.asarray(
        [len(truth_set.intersection(row.result_ids)) for row in results], dtype=np.int16
    )
    exact = np.bincount(truth_assignments.astype(np.int64), minlength=len(placements))
    p_exact = minimum_shards(exact)
    order = sorted(
        range(len(results)),
        key=lambda shard_id: (
            -int(contributions[shard_id]),
            int(results[shard_id].distance_computations),
            shard_id,
        ),
    )
    cumulative = 0
    work = 0
    selected = 0
    for shard_id in order:
        selected += 1
        cumulative += int(contributions[shard_id])
        work += int(results[shard_id].distance_computations)
        if cumulative >= TARGET_HITS:
            break
    reached = cumulative >= TARGET_HITS
    p_hnsw = selected if reached else len(results) + 1
    return {
        "query_id": query_id,
        "contributions": contributions,
        "distance": np.asarray([row.distance_computations for row in results], dtype=np.uint64),
        "nodes": np.asarray([row.graph_nodes_visited for row in results], dtype=np.uint64),
        "worker_cpu": np.asarray([row.worker_cpu_time_us for row in results], dtype=np.uint64),
        "worker_wall": np.asarray([row.worker_cpu_wall_time_us for row in results], dtype=np.uint64),
        "latency": np.asarray([row.latency_us for row in results], dtype=np.float64),
        "response_bytes": np.asarray([row.response_bytes for row in results], dtype=np.uint64),
        "p_exact": p_exact,
        "p_hnsw": p_hnsw,
        "delta_p": p_hnsw - p_exact,
        "w90": work,
        "reached": reached,
        "full_wall_us": full_wall_us,
    }


def evaluate_queries(
    placements: Sequence[dict[str, Any]],
    queries: np.ndarray,
    truth: np.ndarray,
    assignments: np.ndarray,
    *,
    query_start: int,
    ef_search: int,
    dimension: int,
    query_concurrency: int,
    request_workers: int,
    progress_every: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    query_count = len(queries)
    logical_shards = len(placements)
    arrays = {
        "query_id": np.arange(query_start, query_start + query_count, dtype=np.int32),
        "contributions": np.zeros((query_count, logical_shards), dtype=np.int16),
        "distance": np.zeros((query_count, logical_shards), dtype=np.uint64),
        "nodes": np.zeros((query_count, logical_shards), dtype=np.uint64),
        "worker_cpu": np.zeros((query_count, logical_shards), dtype=np.uint64),
        "worker_wall": np.zeros((query_count, logical_shards), dtype=np.uint64),
        "latency_us": np.zeros((query_count, logical_shards), dtype=np.float64),
        "response_bytes": np.zeros((query_count, logical_shards), dtype=np.uint64),
        "p_exact": np.zeros(query_count, dtype=np.int16),
        "p_hnsw": np.zeros(query_count, dtype=np.int16),
        "delta_p": np.zeros(query_count, dtype=np.int16),
        "w90": np.zeros(query_count, dtype=np.uint64),
        "reached": np.zeros(query_count, dtype=np.bool_),
        "full_wall_us": np.zeros(query_count, dtype=np.float64),
    }
    started = time.perf_counter()
    completed = 0
    with ThreadPoolExecutor(max_workers=request_workers) as request_executor:
        with ThreadPoolExecutor(max_workers=query_concurrency) as query_executor:
            window = max(query_concurrency * 4, 1)
            for window_start in range(0, query_count, window):
                window_stop = min(window_start + window, query_count)
                futures = {
                    query_executor.submit(
                        evaluate_one_query,
                        query_start + offset,
                        queries[offset],
                        truth[offset],
                        assignments[truth[offset, :TOP_K]],
                        placements,
                        ef_search,
                        dimension,
                        request_executor,
                    ): offset
                    for offset in range(window_start, window_stop)
                }
                for future in as_completed(futures):
                    offset = futures[future]
                    row = future.result()
                    for source, target in (
                        ("contributions", "contributions"),
                        ("distance", "distance"),
                        ("nodes", "nodes"),
                        ("worker_cpu", "worker_cpu"),
                        ("worker_wall", "worker_wall"),
                        ("latency", "latency_us"),
                        ("response_bytes", "response_bytes"),
                        ("p_exact", "p_exact"),
                        ("p_hnsw", "p_hnsw"),
                        ("delta_p", "delta_p"),
                        ("w90", "w90"),
                        ("reached", "reached"),
                        ("full_wall_us", "full_wall_us"),
                    ):
                        arrays[target][offset] = row[source]
                    completed += 1
                    if completed % progress_every == 0 or completed == query_count:
                        print(f"search: {completed}/{query_count} queries", flush=True)
    elapsed = time.perf_counter() - started
    summary = {
        "query_count": query_count,
        "logical_shards": logical_shards,
        "ef_search": ef_search,
        "elapsed_seconds": elapsed,
        "completed_qps": query_count / max(elapsed, 1e-9),
        "achieved_recall_at_10": float(
            np.mean(np.sum(arrays["contributions"], axis=1) / TOP_K)
        ),
        "P_exact_mean": float(np.mean(arrays["p_exact"])),
        "P_HNSW_mean": float(np.mean(arrays["p_hnsw"])),
        "P_HNSW_median": float(np.median(arrays["p_hnsw"])),
        "P_HNSW_p95": float(np.percentile(arrays["p_hnsw"], 95)),
        "Delta_P_mean": float(np.mean(arrays["delta_p"])),
        "W90_mean": float(np.mean(arrays["w90"])),
        "hnsw_reached_90_fraction": float(np.mean(arrays["reached"])),
        "mean_full_fanout_wall_us": float(np.mean(arrays["full_wall_us"])),
        "p95_full_fanout_wall_us": float(np.percentile(arrays["full_wall_us"], 95)),
        "mean_total_worker_cpu_us": float(np.mean(np.sum(arrays["worker_cpu"], axis=1))),
        "mean_total_distance_computations": float(np.mean(np.sum(arrays["distance"], axis=1))),
    }
    return arrays, summary


def evaluate_one_e3_query(
    query_id: int,
    query: np.ndarray,
    truth: np.ndarray,
    truth_assignments: np.ndarray,
    placements: Sequence[dict[str, Any]],
    ef_search: int,
    dimension: int,
    request_executor: ThreadPoolExecutor,
) -> dict[str, Any]:
    grouped: dict[int, set[int]] = {}
    for point_id, shard_id in zip(truth[:TOP_K], truth_assignments[:TOP_K], strict=True):
        grouped.setdefault(int(shard_id), set()).add(int(point_id))
    futures = {
        shard_id: request_executor.submit(
            search_one,
            placements[shard_id],
            query,
            ef_search,
            dimension,
        )
        for shard_id in grouped
    }
    rows: dict[int, dict[str, Any]] = {}
    for shard_id, future in futures.items():
        result = future.result()
        rows[shard_id] = {
            "target_count": len(grouped[shard_id]),
            "recovered": len(grouped[shard_id].intersection(result.result_ids)),
            "distance": result.distance_computations,
            "nodes": result.graph_nodes_visited,
            "worker_cpu": result.worker_cpu_time_us,
            "latency_us": result.latency_us,
        }
    return {"query_id": query_id, "rows": rows}


def evaluate_local_navigability(
    placements: Sequence[dict[str, Any]],
    queries: np.ndarray,
    truth: np.ndarray,
    assignments: np.ndarray,
    *,
    query_start: int,
    ef_values: Sequence[int],
    common_ef: int,
    dimension: int,
    query_concurrency: int,
    request_workers: int,
    progress_every: int,
    e4_arrays: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    query_count = len(queries)
    logical_shards = len(placements)
    ef_values = tuple(int(value) for value in ef_values)
    if common_ef not in ef_values:
        raise ValueError("E3 ef grid must include the common E4 efSearch")
    target_count = np.zeros((query_count, logical_shards), dtype=np.int16)
    for offset in range(query_count):
        target_count[offset] = np.bincount(
            assignments[truth[offset, :TOP_K]].astype(np.int64),
            minlength=logical_shards,
        )
    shape = (len(ef_values), query_count, logical_shards)
    arrays = {
        "ef_values": np.asarray(ef_values, dtype=np.int32),
        "query_id": np.arange(query_start, query_start + query_count, dtype=np.int32),
        "target_count": target_count,
        "recovered": np.full(shape, -1, dtype=np.int16),
        "distance": np.zeros(shape, dtype=np.uint64),
        "nodes": np.zeros(shape, dtype=np.uint64),
        "worker_cpu": np.zeros(shape, dtype=np.uint64),
        "latency_us": np.zeros(shape, dtype=np.float64),
    }
    relevant = target_count > 0
    common_index = ef_values.index(common_ef)
    arrays["recovered"][common_index][relevant] = e4_arrays["contributions"][relevant]
    for name, source in (
        ("distance", "distance"),
        ("nodes", "nodes"),
        ("worker_cpu", "worker_cpu"),
        ("latency_us", "latency_us"),
    ):
        arrays[name][common_index][relevant] = e4_arrays[source][relevant]

    with ThreadPoolExecutor(max_workers=request_workers) as request_executor:
        with ThreadPoolExecutor(max_workers=query_concurrency) as query_executor:
            for ef_index, ef_search in enumerate(ef_values):
                if ef_search == common_ef:
                    print(f"E3 ef={ef_search}: reused common-EF E4 searches", flush=True)
                    continue
                completed = 0
                window = max(query_concurrency * 4, 1)
                for window_start in range(0, query_count, window):
                    window_stop = min(window_start + window, query_count)
                    futures = {
                        query_executor.submit(
                            evaluate_one_e3_query,
                            query_start + offset,
                            queries[offset],
                            truth[offset],
                            assignments[truth[offset, :TOP_K]],
                            placements,
                            ef_search,
                            dimension,
                            request_executor,
                        ): offset
                        for offset in range(window_start, window_stop)
                    }
                    for future in as_completed(futures):
                        offset = futures[future]
                        row = future.result()
                        for shard_id, local in row["rows"].items():
                            arrays["recovered"][ef_index, offset, shard_id] = local["recovered"]
                            arrays["distance"][ef_index, offset, shard_id] = local["distance"]
                            arrays["nodes"][ef_index, offset, shard_id] = local["nodes"]
                            arrays["worker_cpu"][ef_index, offset, shard_id] = local["worker_cpu"]
                            arrays["latency_us"][ef_index, offset, shard_id] = local["latency_us"]
                        completed += 1
                        if completed % progress_every == 0 or completed == query_count:
                            print(
                                f"E3 ef={ef_search}: {completed}/{query_count} queries",
                                flush=True,
                            )
    summaries: list[dict[str, Any]] = []
    target_values = target_count[relevant].astype(np.float64)
    for ef_index, ef_search in enumerate(ef_values):
        recovered = arrays["recovered"][ef_index][relevant].astype(np.float64)
        if np.any(recovered < 0):
            raise RuntimeError(f"E3 ef={ef_search} is missing relevant query-shard pairs")
        summaries.append(
            {
                "ef_search": ef_search,
                "relevant_query_shard_pairs": int(np.sum(relevant)),
                "mean_local_target_recall": float(np.mean(recovered / target_values)),
                "target_weighted_recall": float(np.sum(recovered) / np.sum(target_values)),
                "mean_distance_computations": float(
                    np.mean(arrays["distance"][ef_index][relevant])
                ),
                "mean_graph_nodes_visited": float(
                    np.mean(arrays["nodes"][ef_index][relevant])
                ),
                "mean_worker_cpu_time_us": float(
                    np.mean(arrays["worker_cpu"][ef_index][relevant])
                ),
            }
        )
    return arrays, summaries


def run_calibration(args: argparse.Namespace) -> dict[str, Any]:
    topology = resources.load_topology(args.topology)
    client_affinity = validate_client_affinity(topology)
    dataset_source = dataset_path(topology, args.dataset)
    dataset_audit = audit_dataset(dataset_source, args.dataset)
    assignments, partition = load_assignments(topology, args.dataset, "kmeans", 1)
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output: {output}")
    output.mkdir(parents=True, exist_ok=False)
    state_path = output / "resource-state.json"
    spec = DATASETS[args.dataset]
    ef_grid = tuple(int(value) for value in args.ef_grid.split(","))
    if not ef_grid or any(value <= 0 for value in ef_grid) or tuple(sorted(set(ef_grid))) != ef_grid:
        raise ValueError("ef grid must be unique, positive, and increasing")
    run_id = resources.sanitize_run_id(args.run_id)
    candidates: list[dict[str, Any]] = []
    after: dict[str, Any] | None = None
    with resources.isolated_shard_cluster(
        topology,
        run_id,
        1,
        state_output=state_path,
    ) as cluster:
        placements = [row["placement"] for row in cluster["started"]]
        with h5py.File(dataset_source, "r") as handle:
            upload = create_and_upload(
                placements,
                handle["train"],
                assignments,
                dimension=int(spec["dimension"]),
                distance=str(spec["distance"]),
                normalize=bool(spec["normalize"]),
                hnsw=topology["hnsw"],
                chunk_size=args.upload_chunk_size,
                upload_batch_size=args.upload_batch_size,
            )
            indexed = wait_indexed(
                placements,
                upload["expected_points_per_shard"],
                timeout=args.index_timeout,
            )
            start = args.query_start
            stop = min(start + args.query_count, int(spec["test_rows"]))
            queries = preprocess(handle["test"][start:stop], normalize=bool(spec["normalize"]))
            truth = np.asarray(handle["neighbors"][start:stop, :TOP_K], dtype=np.int32)
        for ef_search in ef_grid:
            arrays, summary = evaluate_queries(
                placements,
                queries,
                truth,
                assignments,
                query_start=start,
                ef_search=ef_search,
                dimension=int(spec["dimension"]),
                query_concurrency=args.query_concurrency,
                request_workers=args.request_workers,
                progress_every=args.progress_every,
            )
            per_query_path = output / f"tuning-ef{ef_search}.npz"
            save_npz_atomic(per_query_path, arrays)
            candidates.append(
                {
                    "ef_search": ef_search,
                    "summary": summary,
                    "per_query_path": str(per_query_path),
                    "per_query_sha256": sha256_path(per_query_path),
                    "meets_recall_target": summary["achieved_recall_at_10"] >= 0.90,
                }
            )
        after = resources.verify_cluster(topology, run_id, 1)
        if after["status"] != "PASS":
            raise RuntimeError("resource verification failed after calibration")
        before = cluster["verification"]
    cleanup = json.loads(state_path.read_text(encoding="utf-8"))
    feasible = [row for row in candidates if row["meets_recall_target"]]
    selected = int(feasible[0]["ef_search"]) if feasible else None
    payload = {
        "protocol_version": resources.PROTOCOL_VERSION,
        "timestamp": resources.utc_timestamp(),
        "record_type": "c23_linear_m1_common_ef_calibration",
        "run_id": run_id,
        "dataset": args.dataset,
        "dataset_audit": dataset_audit,
        "partition": partition,
        "logical_shards": 1,
        "query_range": [args.query_start, args.query_start + args.query_count],
        "target_recall_at_10": 0.90,
        "selection_rule": "smallest common efSearch with M=1 tuning Recall@10 >= 0.90",
        "ef_grid": list(ef_grid),
        "candidates": candidates,
        "selected_ef_search": selected,
        "benchmark_client_affinity": client_affinity,
        "resource_before": before,
        "upload": upload,
        "index_gate": indexed,
        "resource_after": after,
        "cleanup": cleanup,
        "checks": {
            "resource_before": "PASS" if before["status"] == "PASS" else "FAIL",
            "benchmark_client_affinity": client_affinity["status"],
            "upload": "PASS",
            "single_hnsw_per_shard": indexed["status"],
            "all_ef_candidates_complete": "PASS" if len(candidates) == len(ef_grid) else "FAIL",
            "feasible_common_ef": "PASS" if selected is not None else "FAIL",
            "resource_after": "PASS" if after and after["status"] == "PASS" else "FAIL",
            "cleanup": "PASS"
            if cleanup.get("restored", {}).get("status") == "PASS"
            else "FAIL",
        },
    }
    payload["status"] = (
        "PASS" if all(value == "PASS" for value in payload["checks"].values()) else "FAIL"
    )
    write_json_atomic(output / "manifest.json", payload)
    return payload


def save_npz_atomic(path: str | Path, arrays: dict[str, np.ndarray]) -> None:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    os.close(fd)
    try:
        with open(temporary, "wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary)
        raise


def run_real_configuration(args: argparse.Namespace) -> dict[str, Any]:
    topology = resources.load_topology(args.topology)
    client_affinity = validate_client_affinity(topology)
    if args.logical_shards == 1:
        method = "common_m1"
    else:
        method = args.method
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}")
    dataset_source = dataset_path(topology, args.dataset)
    dataset_audit = audit_dataset(dataset_source, args.dataset)
    assignments, partition = load_assignments(
        topology,
        args.dataset,
        "kmeans" if method == "common_m1" else method,
        args.logical_shards,
    )
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output: {output}")
    output.mkdir(parents=True, exist_ok=False)
    state_path = output / "resource-state.json"
    per_query_path = output / "per-query.npz"
    e3_per_query_path = output / "e3-per-query.npz"
    spec = DATASETS[args.dataset]
    run_id = resources.sanitize_run_id(args.run_id)
    resource_after_upload: dict[str, Any] | None = None
    resource_after_index: dict[str, Any] | None = None
    resource_after_e4: dict[str, Any] | None = None
    resource_after_e3: dict[str, Any] | None = None
    with resources.isolated_shard_cluster(
        topology,
        run_id,
        args.logical_shards,
        state_output=state_path,
    ) as cluster:
        placements = [row["placement"] for row in cluster["started"]]
        with h5py.File(dataset_source, "r") as handle:
            upload = create_and_upload(
                placements,
                handle["train"],
                assignments,
                dimension=int(spec["dimension"]),
                distance=str(spec["distance"]),
                normalize=bool(spec["normalize"]),
                hnsw=topology["hnsw"],
                chunk_size=args.upload_chunk_size,
                upload_batch_size=args.upload_batch_size,
            )
            resource_after_upload = resources.verify_cluster(
                topology, run_id, args.logical_shards
            )
            if resource_after_upload["status"] != "PASS":
                raise RuntimeError("resource verification failed after upload")
            indexed = wait_indexed(
                placements,
                upload["expected_points_per_shard"],
                timeout=args.index_timeout,
            )
            resource_after_index = resources.verify_cluster(
                topology, run_id, args.logical_shards
            )
            if resource_after_index["status"] != "PASS":
                raise RuntimeError("resource verification failed after indexing")
            start = args.query_start
            stop = min(start + args.query_count, int(spec["test_rows"]))
            queries = preprocess(handle["test"][start:stop], normalize=bool(spec["normalize"]))
            truth = np.asarray(handle["neighbors"][start:stop, :TOP_K], dtype=np.int32)
        arrays, query_summary = evaluate_queries(
            placements,
            queries,
            truth,
            assignments,
            query_start=start,
            ef_search=args.ef_search,
            dimension=int(spec["dimension"]),
            query_concurrency=args.query_concurrency,
            request_workers=args.request_workers,
            progress_every=args.progress_every,
        )
        save_npz_atomic(per_query_path, arrays)
        resource_after_e4 = resources.verify_cluster(
            topology, run_id, args.logical_shards
        )
        if resource_after_e4["status"] != "PASS":
            raise RuntimeError("resource verification failed after E4")
        e3_values = tuple(int(value) for value in args.e3_ef_grid.split(","))
        if tuple(sorted(set(e3_values))) != e3_values or any(value <= 0 for value in e3_values):
            raise ValueError("E3 ef grid must be unique, positive, and increasing")
        e3_arrays, e3_summary = evaluate_local_navigability(
            placements,
            queries,
            truth,
            assignments,
            query_start=start,
            ef_values=e3_values,
            common_ef=args.ef_search,
            dimension=int(spec["dimension"]),
            query_concurrency=args.query_concurrency,
            request_workers=args.request_workers,
            progress_every=args.progress_every,
            e4_arrays=arrays,
        )
        save_npz_atomic(e3_per_query_path, e3_arrays)
        resource_after_e3 = resources.verify_cluster(
            topology, run_id, args.logical_shards
        )
        if resource_after_e3["status"] != "PASS":
            raise RuntimeError("resource verification failed after E3")
        active = cluster["verification"]
    cleanup = json.loads(state_path.read_text(encoding="utf-8"))
    manifest = {
        "protocol_version": resources.PROTOCOL_VERSION,
        "timestamp": resources.utc_timestamp(),
        "record_type": "c23_linear_resource_real_configuration",
        "run_id": run_id,
        "dataset": args.dataset,
        "dataset_audit": dataset_audit,
        "partition_method": method,
        "partition": partition,
        "logical_shards": args.logical_shards,
        "ef_search": args.ef_search,
        "query_range": [args.query_start, args.query_start + args.query_count],
        "resource_contract": resources.resource_contract(topology, args.logical_shards),
        "benchmark_client_affinity": client_affinity,
        "resource_verification_before": active,
        "resource_phase_snapshots": {
            "before_upload": active,
            "after_upload": resource_after_upload,
            "after_index": resource_after_index,
            "after_e4": resource_after_e4,
            "after_e3": resource_after_e3,
        },
        "upload": upload,
        "index_gate": indexed,
        "query_summary": query_summary,
        "per_query_path": str(per_query_path),
        "per_query_sha256": sha256_path(per_query_path),
        "e3_ef_values": list(e3_values),
        "e3_summary": e3_summary,
        "e3_per_query_path": str(e3_per_query_path),
        "e3_per_query_sha256": sha256_path(e3_per_query_path),
        "resource_verification_after": resource_after_e3,
        "cleanup": cleanup,
        "checks": {
            "dataset": "PASS",
            "partition": "PASS",
            "benchmark_client_affinity": client_affinity["status"],
            "resource_before": "PASS" if active["status"] == "PASS" else "FAIL",
            "resource_after_upload": "PASS"
            if resource_after_upload and resource_after_upload["status"] == "PASS"
            else "FAIL",
            "resource_after_index": "PASS"
            if resource_after_index and resource_after_index["status"] == "PASS"
            else "FAIL",
            "upload": "PASS",
            "single_hnsw_per_shard": indexed["status"],
            "hardware_counters": "PASS",
            "resource_after_e4": "PASS"
            if resource_after_e4 and resource_after_e4["status"] == "PASS"
            else "FAIL",
            "complete_e3_grid": "PASS"
            if len(e3_summary) == len(e3_values)
            else "FAIL",
            "resource_after_e3": "PASS"
            if resource_after_e3 and resource_after_e3["status"] == "PASS"
            else "FAIL",
            "resource_after": "PASS"
            if resource_after_e3 and resource_after_e3["status"] == "PASS"
            else "FAIL",
            "cleanup": "PASS"
            if cleanup.get("restored", {}).get("status") == "PASS"
            else "FAIL",
        },
    }
    manifest["status"] = (
        "PASS" if all(value == "PASS" for value in manifest["checks"].values()) else "FAIL"
    )
    write_json_atomic(output / "manifest.json", manifest)
    return manifest


def synthetic_data(seed: int, point_count: int, query_count: int, dimension: int, shards: int):
    rng = np.random.default_rng(seed)
    vectors = rng.normal(size=(point_count, dimension)).astype(np.float32)
    queries = rng.normal(size=(query_count, dimension)).astype(np.float32)
    assignments = np.arange(point_count, dtype=np.int32) % shards
    distances = (
        np.sum(queries * queries, axis=1)[:, None]
        + np.sum(vectors * vectors, axis=1)[None, :]
        - 2 * (queries @ vectors.T)
    )
    truth = np.argpartition(distances, TOP_K - 1, axis=1)[:, :TOP_K]
    truth = np.take_along_axis(
        truth,
        np.argsort(np.take_along_axis(distances, truth, axis=1), axis=1),
        axis=1,
    ).astype(np.int32)
    return vectors, queries, truth, assignments


def run_synthetic_smoke(args: argparse.Namespace) -> dict[str, Any]:
    topology = resources.load_topology(args.topology)
    client_affinity = validate_client_affinity(topology)
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output: {output}")
    output.mkdir(parents=True, exist_ok=False)
    vectors, queries, truth, assignments = synthetic_data(
        args.seed,
        args.point_count,
        args.query_count,
        args.dimension,
        args.logical_shards,
    )
    state_path = output / "resource-state.json"
    per_query_path = output / "per-query.npz"
    run_id = resources.sanitize_run_id(args.run_id)
    with resources.isolated_shard_cluster(
        topology,
        run_id,
        args.logical_shards,
        state_output=state_path,
    ) as cluster:
        placements = [row["placement"] for row in cluster["started"]]
        upload = create_and_upload(
            placements,
            vectors,
            assignments,
            dimension=args.dimension,
            distance="Euclid",
            normalize=False,
            hnsw=topology["hnsw"],
            chunk_size=args.upload_chunk_size,
            upload_batch_size=args.upload_batch_size,
        )
        indexed = wait_indexed(
            placements,
            upload["expected_points_per_shard"],
            timeout=args.index_timeout,
        )
        arrays, query_summary = evaluate_queries(
            placements,
            queries,
            truth,
            assignments,
            query_start=0,
            ef_search=args.ef_search,
            dimension=args.dimension,
            query_concurrency=args.query_concurrency,
            request_workers=args.request_workers,
            progress_every=max(1, args.query_count // 4),
        )
        save_npz_atomic(per_query_path, arrays)
        after = resources.verify_cluster(topology, run_id, args.logical_shards)
        before = cluster["verification"]
    cleanup = json.loads(state_path.read_text(encoding="utf-8"))
    payload = {
        "protocol_version": resources.PROTOCOL_VERSION,
        "timestamp": resources.utc_timestamp(),
        "record_type": "c23_linear_e3e4_synthetic_smoke",
        "run_id": run_id,
        "logical_shards": args.logical_shards,
        "point_count": args.point_count,
        "query_count": args.query_count,
        "dimension": args.dimension,
        "ef_search": args.ef_search,
        "benchmark_client_affinity": client_affinity,
        "resource_before": before,
        "upload": upload,
        "index_gate": indexed,
        "query_summary": query_summary,
        "per_query_path": str(per_query_path),
        "per_query_sha256": sha256_path(per_query_path),
        "resource_after": after,
        "cleanup": cleanup,
        "checks": {
            "resource_before": "PASS" if before["status"] == "PASS" else "FAIL",
            "benchmark_client_affinity": client_affinity["status"],
            "upload": "PASS" if upload["total_uploaded_points"] == args.point_count else "FAIL",
            "single_hnsw_per_shard": indexed["status"],
            "positive_graph_search": "PASS"
            if np.all(np.sum(arrays["nodes"], axis=1) > 0)
            else "FAIL",
            "positive_worker_cpu": "PASS"
            if np.all(np.sum(arrays["worker_cpu"], axis=1) > 0)
            else "FAIL",
            "resource_after": "PASS" if after["status"] == "PASS" else "FAIL",
            "cleanup": "PASS"
            if cleanup.get("restored", {}).get("status") == "PASS"
            else "FAIL",
        },
    }
    payload["status"] = (
        "PASS" if all(value == "PASS" for value in payload["checks"].values()) else "FAIL"
    )
    write_json_atomic(output / "manifest.json", payload)
    return payload


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--topology", default=str(resources.DEFAULT_TOPOLOGY))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--logical-shards", type=int, required=True)
    parser.add_argument("--ef-search", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--upload-chunk-size", type=int, default=8192)
    parser.add_argument("--upload-batch-size", type=int, default=256)
    parser.add_argument("--index-timeout", type=float, default=10_800)
    parser.add_argument("--query-concurrency", type=int, default=16)
    parser.add_argument("--request-workers", type=int, default=128)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    smoke = subparsers.add_parser("synthetic-smoke")
    add_common_arguments(smoke)
    smoke.add_argument("--seed", type=int, default=20260824)
    smoke.add_argument("--point-count", type=int, default=8192)
    smoke.add_argument("--query-count", type=int, default=32)
    smoke.add_argument("--dimension", type=int, default=32)

    run = subparsers.add_parser("run-config")
    add_common_arguments(run)
    run.add_argument("--dataset", choices=tuple(DATASETS), required=True)
    run.add_argument("--method", choices=METHODS, required=True)
    run.add_argument("--query-start", type=int, default=1000)
    run.add_argument("--query-count", type=int, default=9000)
    run.add_argument("--progress-every", type=int, default=100)
    run.add_argument("--e3-ef-grid", default=",".join(map(str, EF_GRID)))

    calibration = subparsers.add_parser("calibrate")
    calibration.add_argument("--topology", default=str(resources.DEFAULT_TOPOLOGY))
    calibration.add_argument("--run-id", required=True)
    calibration.add_argument("--dataset", choices=tuple(DATASETS), required=True)
    calibration.add_argument("--ef-grid", default=",".join(map(str, EF_GRID)))
    calibration.add_argument("--query-start", type=int, default=0)
    calibration.add_argument("--query-count", type=int, default=1000)
    calibration.add_argument("--output", required=True)
    calibration.add_argument("--upload-chunk-size", type=int, default=8192)
    calibration.add_argument("--upload-batch-size", type=int, default=256)
    calibration.add_argument("--index-timeout", type=float, default=10_800)
    calibration.add_argument("--query-concurrency", type=int, default=16)
    calibration.add_argument("--request-workers", type=int, default=64)
    calibration.add_argument("--progress-every", type=int, default=100)

    args = parser.parse_args()
    if args.command == "synthetic-smoke":
        if args.logical_shards not in resources.ALLOWED_LOGICAL_SHARDS:
            parser.error(f"logical shards must be one of {resources.ALLOWED_LOGICAL_SHARDS}")
        if args.ef_search <= 0:
            parser.error("ef-search must be positive")
        payload = run_synthetic_smoke(args)
    elif args.command == "run-config":
        if args.logical_shards not in resources.ALLOWED_LOGICAL_SHARDS:
            parser.error(f"logical shards must be one of {resources.ALLOWED_LOGICAL_SHARDS}")
        if args.ef_search <= 0:
            parser.error("ef-search must be positive")
        payload = run_real_configuration(args)
    elif args.command == "calibrate":
        payload = run_calibration(args)
    else:
        raise AssertionError(args.command)
    print(json.dumps({"status": payload["status"], "output": str(Path(args.output).resolve())}))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
