#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import http.client
import json
import math
import os
import shlex
import statistics
import subprocess
import sys
import threading
import time
import urllib.parse
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import h5py
import numpy as np

from c1_protocol import (
    DATASETS,
    PARTITION_SEED,
    TARGET_RECALL,
    TOP_K,
    TUNING_QUERY_COUNT,
    DatasetSpec,
    distance_computations_from_usage,
    oracle_minimum_fanout,
    preprocess_rows,
    recall_per_query,
    repository_commit,
    select_kmeans_tuning_candidate,
    select_random_tuning_candidate,
    sha256_path,
    utc_timestamp,
)


HNSW_M = 32
HNSW_EF_CONSTRUCTION = 200
HNSW_MAX_INDEXING_THREADS = 1
MAX_OPTIMIZATION_THREADS = 1
DEFAULT_EF_GRID = (16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512)
PER_QUERY_FIELDS = (
    "query_id",
    "dataset",
    "partition_method",
    "logical_shards",
    "physical_hosts",
    "target_recall",
    "achieved_recall",
    "queried_shards",
    "queried_shard_ids",
    "routing_latency_us",
    "routing_cpu_time_us",
    "local_search_latency_us_per_shard",
    "distance_computations_per_shard",
    "nodes_visited_per_shard",
    "worker_cpu_time_us_per_shard",
    "worker_cpu_wall_time_us_per_shard",
    "response_bytes_per_shard",
    "end_to_end_latency_us",
    "oracle_minimum_fanout",
    "result_ids",
)
E3_SEARCH_FIELDS = (
    "query_id",
    "dataset",
    "partition_method",
    "logical_shards",
    "selected_fanout",
    "ef_search",
    "shard_id",
    "route_rank",
    "physical_host",
    "shard_point_count",
    "ground_truth_points_in_shard",
    "ground_truth_hits",
    "recall_contribution",
    "distance_computations",
    "nodes_visited",
    "worker_cpu_time_us",
    "worker_cpu_wall_time_us",
    "local_search_latency_us",
    "response_bytes",
    "result_ids",
)


@dataclass(frozen=True)
class Node:
    role: str
    ssh_host: str
    private_ip: str
    cpuset: str
    peer_id: int
    base_url: str


@dataclass(frozen=True)
class PartitionData:
    path: str
    sha256: str
    assignments: np.ndarray
    centroids: np.ndarray
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ShardSearch:
    shard_id: int
    latency_us: float
    distance_computations: int
    nodes_visited: int
    worker_cpu_time_us: int
    worker_cpu_wall_time_us: int
    response_bytes: int
    results: list[tuple[int, float]]


class JsonHttpClient:
    """Small persistent HTTP/1.1 client with one connection per thread and host."""

    def __init__(self) -> None:
        self._local = threading.local()

    def _connections(self) -> dict[tuple[str, int], http.client.HTTPConnection]:
        connections = getattr(self._local, "connections", None)
        if connections is None:
            connections = {}
            self._local.connections = connections
        return connections

    def _connection(self, base_url: str, timeout: float) -> http.client.HTTPConnection:
        parsed = urllib.parse.urlsplit(base_url)
        if parsed.scheme != "http" or not parsed.hostname:
            raise ValueError(f"C1 requires an http base URL, got {base_url!r}")
        port = parsed.port or 80
        key = (parsed.hostname, port)
        connections = self._connections()
        connection = connections.get(key)
        if connection is None:
            connection = http.client.HTTPConnection(parsed.hostname, port, timeout=timeout)
            connections[key] = connection
        else:
            connection.timeout = timeout
        return connection

    def request(
        self,
        base_url: str,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
        *,
        timeout: float = 600.0,
    ) -> tuple[dict[str, Any], int]:
        encoded = None if body is None else json.dumps(body, separators=(",", ":")).encode()
        headers = {"Accept": "application/json"}
        if encoded is not None:
            headers["Content-Type"] = "application/json"
        last_error: BaseException | None = None
        for attempt in range(2):
            connection = self._connection(base_url, timeout)
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
                    raise RuntimeError(f"{method} {path} returned non-object JSON")
                status = payload.get("status")
                if isinstance(status, dict) and status.get("error"):
                    raise RuntimeError(f"{method} {path} failed: {status['error']}")
                return payload, len(raw)
            except (BrokenPipeError, ConnectionError, http.client.HTTPException, OSError) as exc:
                last_error = exc
                parsed = urllib.parse.urlsplit(base_url)
                key = (str(parsed.hostname), parsed.port or 80)
                stale = self._connections().pop(key, None)
                if stale is not None:
                    stale.close()
                if attempt == 0:
                    continue
                raise RuntimeError(f"{method} {base_url}{path} failed: {exc}") from exc
        raise AssertionError(last_error)


HTTP = JsonHttpClient()


def encoded_collection(name: str) -> str:
    return urllib.parse.quote(name, safe="")


def load_topology(path: str | Path) -> dict[str, Any]:
    topology_path = Path(path).expanduser().resolve()
    data = json.loads(topology_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("topology root must be an object")
    controller = data.get("controller")
    workers = data.get("workers")
    if not isinstance(controller, dict) or not isinstance(workers, list) or len(workers) != 3:
        raise ValueError("C1 topology must contain one controller and three workers")
    return data


def parse_cpu_set(value: str) -> list[int]:
    cpus: set[int] = set()
    for raw_part in value.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "-" in part:
            raw_start, raw_stop = part.split("-", 1)
            start, stop = int(raw_start), int(raw_stop)
            if start < 0 or stop < start:
                raise ValueError(f"invalid CPU range: {part!r}")
            cpus.update(range(start, stop + 1))
        else:
            cpu = int(part)
            if cpu < 0:
                raise ValueError(f"invalid CPU number: {part!r}")
            cpus.add(cpu)
    if not cpus:
        raise ValueError("CPU set must not be empty")
    return sorted(cpus)


def current_cpu_affinity() -> list[int]:
    if not hasattr(os, "sched_getaffinity"):
        raise RuntimeError("this platform cannot report process CPU affinity")
    return sorted(os.sched_getaffinity(0))


def validate_benchmark_affinity(topology_path: str | Path) -> list[int]:
    topology = load_topology(topology_path)
    expected = parse_cpu_set(str(topology["benchmark_client_cpuset"]))
    actual = current_cpu_affinity()
    if actual != expected:
        raise RuntimeError(
            f"benchmark CPU affinity mismatch: actual={actual}, expected={expected}"
        )
    return actual


def discover_nodes(topology: dict[str, Any]) -> list[Node]:
    port = int((topology.get("ports") or {}).get("http") or 6333)
    raw_nodes = [topology["controller"], *topology["workers"]]
    nodes: list[Node] = []
    for raw in raw_nodes:
        private_ip = str(raw["private_ip"])
        base_url = f"http://{private_ip}:{port}"
        payload, _ = HTTP.request(base_url, "GET", "/cluster", timeout=30.0)
        result = payload.get("result") or {}
        nodes.append(
            Node(
                role=str(raw["role"]),
                ssh_host=str(raw["ssh_host"]),
                private_ip=private_ip,
                cpuset=str(raw["cpuset"]),
                peer_id=int(result["peer_id"]),
                base_url=base_url,
            )
        )
    peer_ids = {node.peer_id for node in nodes}
    if len(peer_ids) != 4:
        raise RuntimeError(f"expected four unique C1 peers, got {sorted(peer_ids)}")
    cluster, _ = HTTP.request(nodes[0].base_url, "GET", "/cluster", timeout=30.0)
    visible = set(map(int, ((cluster.get("result") or {}).get("peers") or {}).keys()))
    if visible != peer_ids:
        raise RuntimeError(
            f"cluster membership mismatch: discovered={sorted(peer_ids)}, visible={sorted(visible)}"
        )
    return nodes


def load_partition(path: str | Path, expected_method: str, logical_shards: int) -> PartitionData:
    artifact_path = Path(path).expanduser().resolve()
    if not artifact_path.is_file():
        raise FileNotFoundError(artifact_path)
    with np.load(artifact_path, allow_pickle=False) as data:
        assignments = np.asarray(data["assignments"]).copy()
        centroids = np.asarray(data["centroids"], dtype=np.float32).copy()
        metadata = json.loads(str(np.asarray(data["metadata_json"]).item()))
    if metadata.get("method") != expected_method:
        raise ValueError(
            f"partition method mismatch: artifact={metadata.get('method')}, expected={expected_method}"
        )
    if int(metadata.get("logical_shards") or 0) != logical_shards:
        raise ValueError("partition logical-shard count mismatch")
    if len(assignments) != int(metadata.get("point_count") or 0):
        raise ValueError("partition assignment count does not match metadata")
    if assignments.size and (int(assignments.min()) < 0 or int(assignments.max()) >= logical_shards):
        raise ValueError("partition assignment contains an invalid shard ID")
    if expected_method == "kmeans" and centroids.shape[0] != logical_shards:
        raise ValueError("K-Means artifact has the wrong centroid count")
    if expected_method == "random" and centroids.shape[0] != 0:
        raise ValueError("Random artifact must not contain centroids")
    return PartitionData(
        path=str(artifact_path),
        sha256=sha256_path(artifact_path),
        assignments=assignments,
        centroids=centroids,
        metadata=metadata,
    )


def shard_key(collection: str, shard_id: int) -> str:
    return f"{collection}__s{shard_id:02d}"


def round_robin_placement(logical_shards: int, nodes: Sequence[Node]) -> dict[int, Node]:
    if logical_shards <= 0:
        raise ValueError("logical_shards must be positive")
    if len(nodes) != 4:
        raise ValueError("C1 placement requires exactly four physical nodes")
    if logical_shards <= 4:
        return {shard_id: nodes[shard_id] for shard_id in range(logical_shards)}
    if logical_shards % 4:
        raise ValueError("logical shard simulation must divide evenly across four nodes")
    return {shard_id: nodes[shard_id % 4] for shard_id in range(logical_shards)}


def collection_info(base_url: str, collection: str) -> dict[str, Any]:
    payload, _ = HTTP.request(
        base_url, "GET", f"/collections/{encoded_collection(collection)}", timeout=60.0
    )
    return dict(payload["result"])


def collection_cluster_info(base_url: str, collection: str) -> dict[str, Any]:
    payload, _ = HTTP.request(
        base_url,
        "GET",
        f"/collections/{encoded_collection(collection)}/cluster",
        timeout=60.0,
    )
    return dict(payload["result"])


def collection_exists(base_url: str, collection: str) -> bool:
    try:
        collection_info(base_url, collection)
        return True
    except RuntimeError as exc:
        if "HTTP 404" in str(exc) or "doesn't exist" in str(exc):
            return False
        raise


def delete_collection(base_url: str, collection: str) -> None:
    if collection_exists(base_url, collection):
        HTTP.request(
            base_url,
            "DELETE",
            f"/collections/{encoded_collection(collection)}",
            timeout=300.0,
        )


def create_collection(
    base_url: str,
    collection: str,
    dimension: int,
    distance: str,
    *,
    max_segment_size_kb: int | None = None,
) -> None:
    optimizers_config: dict[str, Any] = {
        "default_segment_number": 1,
        "indexing_threshold": 10,
        "max_optimization_threads": MAX_OPTIMIZATION_THREADS,
    }
    if max_segment_size_kb is not None:
        if max_segment_size_kb <= 0:
            raise ValueError("max_segment_size_kb must be positive")
        optimizers_config["max_segment_size"] = max_segment_size_kb
    HTTP.request(
        base_url,
        "PUT",
        f"/collections/{encoded_collection(collection)}",
        {
            "vectors": {"size": dimension, "distance": distance},
            "shard_number": 1,
            "sharding_method": "custom",
            "replication_factor": 1,
            "write_consistency_factor": 1,
            "hnsw_config": {
                "m": HNSW_M,
                "ef_construct": HNSW_EF_CONSTRUCTION,
                "full_scan_threshold": 10,
                "max_indexing_threads": HNSW_MAX_INDEXING_THREADS,
            },
            "optimizers_config": optimizers_config,
        },
        timeout=300.0,
    )


def create_shards(
    base_url: str,
    collection: str,
    logical_shards: int,
    placement: dict[int, Node],
) -> None:
    path = f"/collections/{encoded_collection(collection)}/shards"
    for shard_id in range(logical_shards):
        HTTP.request(
            base_url,
            "PUT",
            path,
            {
                "shard_key": shard_key(collection, shard_id),
                "shards_number": 1,
                "replication_factor": 1,
                "placement": [placement[shard_id].peer_id],
            },
            timeout=300.0,
        )


def _json_points(ids: np.ndarray, rows: np.ndarray) -> list[dict[str, Any]]:
    return [
        {"id": int(point_id), "vector": vector.tolist()}
        for point_id, vector in zip(ids, rows, strict=True)
    ]


def upload_partitioned_dataset(
    hdf5_path: str | Path,
    spec: DatasetSpec,
    collection: str,
    partition: PartitionData,
    placement: dict[int, Node],
    *,
    batch_size: int,
) -> dict[str, Any]:
    logical_shards = int(partition.metadata["logical_shards"])
    uploaded = 0
    started = time.time()
    with h5py.File(hdf5_path, "r") as handle:
        train = handle["train"]
        if len(train) != len(partition.assignments):
            raise ValueError("dataset row count does not match partition assignments")
        for shard_id in range(logical_shards):
            indices = np.flatnonzero(partition.assignments == shard_id)
            node = placement[shard_id]
            path = f"/collections/{encoded_collection(collection)}/points?wait=true"
            for start in range(0, len(indices), batch_size):
                batch_ids = indices[start : start + batch_size]
                rows = preprocess_rows(train[batch_ids], spec)
                HTTP.request(
                    node.base_url,
                    "PUT",
                    path,
                    {
                        "points": _json_points(batch_ids, rows),
                        "shard_key": shard_key(collection, shard_id),
                    },
                    timeout=900.0,
                )
                uploaded += len(batch_ids)
                if uploaded % max(batch_size * 20, 1) == 0 or uploaded == len(train):
                    print(
                        f"upload {collection}: {uploaded}/{len(train)} points",
                        flush=True,
                    )
    return {
        "uploaded_points": uploaded,
        "upload_seconds": time.time() - started,
        "upload_points_per_second": uploaded / max(time.time() - started, 1e-9),
    }


def wait_collection_indexed(
    base_url: str,
    collection: str,
    expected_points: int,
    *,
    timeout: float = 10_800.0,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    stable_since: float | None = None
    last: dict[str, Any] = {}
    while time.monotonic() < deadline:
        last = collection_info(base_url, collection)
        cluster = collection_cluster_info(base_url, collection)
        points = int(last.get("points_count") or 0)
        indexed = int(last.get("indexed_vectors_count") or 0)
        optimizer = last.get("optimizer_status")
        optimizer_ok = optimizer == "ok" or (
            isinstance(optimizer, dict) and optimizer.get("ok") is True
        )
        shards = [*(cluster.get("local_shards") or []), *(cluster.get("remote_shards") or [])]
        active = bool(shards) and all(shard.get("state") == "Active" for shard in shards)
        ready = (
            points == expected_points
            and indexed >= expected_points
            and str(last.get("status") or "").lower() == "green"
            and optimizer_ok
            and active
            and not (cluster.get("shard_transfers") or [])
        )
        if ready:
            stable_since = stable_since or time.monotonic()
            if time.monotonic() - stable_since >= 5.0:
                return last
        else:
            stable_since = None
        time.sleep(1.0)
    raise TimeoutError(f"collection {collection!r} did not finish indexing; last={last}")


def actual_shard_placement(base_url: str, collection: str) -> dict[str, int]:
    cluster = collection_cluster_info(base_url, collection)
    local_peer = int(cluster["peer_id"])
    result: dict[str, int] = {}
    for shard in cluster.get("local_shards") or []:
        if shard.get("shard_key") is not None:
            result[str(shard["shard_key"])] = int(shard.get("peer_id") or local_peer)
    for shard in cluster.get("remote_shards") or []:
        if shard.get("shard_key") is not None:
            result[str(shard["shard_key"])] = int(shard["peer_id"])
    return result


def ensure_collection(
    topology_path: str | Path,
    hdf5_path: str | Path,
    dataset: str,
    method: str,
    logical_shards: int,
    partition_path: str | Path,
    collection: str,
    *,
    batch_size: int,
    replace: bool,
    max_segment_size_kb: int | None = None,
) -> dict[str, Any]:
    spec = DATASETS[dataset]
    topology = load_topology(topology_path)
    graph_build_seed = topology.get("hnsw_graph_build_seed")
    if (
        isinstance(graph_build_seed, bool)
        or not isinstance(graph_build_seed, int)
        or graph_build_seed < 0
        or graph_build_seed > 2**64 - 1
    ):
        raise ValueError(
            "C1 topology requires hnsw_graph_build_seed as an unsigned 64-bit integer"
        )
    nodes = discover_nodes(topology)
    placement = round_robin_placement(logical_shards, nodes)
    partition = load_partition(partition_path, method, logical_shards)
    controller = nodes[0].base_url
    with h5py.File(hdf5_path, "r") as handle:
        point_count, dimension = map(int, handle["train"].shape)
    if point_count != len(partition.assignments):
        raise ValueError("partition does not cover the complete base dataset")
    if replace:
        delete_collection(controller, collection)
    created = not collection_exists(controller, collection)
    upload: dict[str, Any] | None = None
    if created:
        create_collection(
            controller,
            collection,
            dimension,
            spec.qdrant_distance,
            max_segment_size_kb=max_segment_size_kb,
        )
        create_shards(controller, collection, logical_shards, placement)
        upload = upload_partitioned_dataset(
            hdf5_path,
            spec,
            collection,
            partition,
            placement,
            batch_size=batch_size,
        )
    info = wait_collection_indexed(controller, collection, point_count)
    actual = actual_shard_placement(controller, collection)
    expected = {
        shard_key(collection, shard_id): placement[shard_id].peer_id
        for shard_id in range(logical_shards)
    }
    if actual != expected:
        raise RuntimeError(f"collection placement mismatch: actual={actual}, expected={expected}")
    config = info.get("config") or {}
    vector_config = ((config.get("params") or {}).get("vectors") or {})
    hnsw = config.get("hnsw_config") or {}
    optimizer = config.get("optimizer_config") or {}
    if int(vector_config.get("size") or 0) != dimension:
        raise RuntimeError("collection dimension mismatch")
    if str(vector_config.get("distance") or "").lower() != spec.qdrant_distance.lower():
        raise RuntimeError("collection distance mismatch")
    if int(hnsw.get("m") or 0) != HNSW_M or int(hnsw.get("ef_construct") or 0) != HNSW_EF_CONSTRUCTION:
        raise RuntimeError("collection HNSW build configuration mismatch")
    if int(hnsw.get("max_indexing_threads") or 0) != HNSW_MAX_INDEXING_THREADS:
        raise RuntimeError("collection HNSW indexing-thread configuration mismatch")
    if int(optimizer.get("max_optimization_threads") or 0) != MAX_OPTIMIZATION_THREADS:
        raise RuntimeError("collection optimization-thread configuration mismatch")
    if max_segment_size_kb is not None and int(optimizer.get("max_segment_size") or 0) != max_segment_size_kb:
        raise RuntimeError("collection max-segment-size configuration mismatch")
    return {
        "timestamp": utc_timestamp(),
        "collection": collection,
        "created": created,
        "dataset": dataset,
        "dataset_path": str(Path(hdf5_path).resolve()),
        "dataset_sha256": sha256_path(hdf5_path),
        "partition_method": method,
        "partition_artifact": partition.path,
        "partition_artifact_sha256": partition.sha256,
        "partition_seed": int(partition.metadata["partition_seed"]),
        "logical_shards": logical_shards,
        "physical_hosts": min(logical_shards, 4),
        "logical_to_physical_mapping": {
            str(shard_id): placement[shard_id].private_ip
            for shard_id in range(logical_shards)
        },
        "logical_to_peer_mapping": {
            str(shard_id): placement[shard_id].peer_id for shard_id in range(logical_shards)
        },
        "point_count": point_count,
        "dimension": dimension,
        "distance": spec.qdrant_distance,
        "hnsw_m": HNSW_M,
        "hnsw_ef_construction": HNSW_EF_CONSTRUCTION,
        "hnsw_max_indexing_threads": HNSW_MAX_INDEXING_THREADS,
        "max_optimization_threads": MAX_OPTIMIZATION_THREADS,
        "max_segment_size_kb": max_segment_size_kb,
        "graph_build_seed": graph_build_seed,
        "deterministic_graph_construction": True,
        "collection_info": info,
        "nodes": [asdict(node) for node in nodes],
        "upload": upload,
    }


def rank_kmeans_shards(query: np.ndarray, centroids: np.ndarray, spec: DatasetSpec) -> np.ndarray:
    prepared = preprocess_rows(np.asarray(query, dtype=np.float32)[None, :], spec)[0]
    centers = preprocess_rows(centroids, spec)
    if spec.normalize:
        return np.argsort(-(centers @ prepared), kind="stable")
    delta = centers - prepared
    return np.argsort(np.einsum("ij,ij->i", delta, delta), kind="stable")


def selected_shards_for_query(
    query: np.ndarray,
    spec: DatasetSpec,
    method: str,
    logical_shards: int,
    fanout: int,
    centroids: np.ndarray,
) -> list[int]:
    if method == "random":
        if fanout != logical_shards:
            raise ValueError("Random serving must broadcast to every shard")
        return list(range(logical_shards))
    if method == "kmeans":
        if not 1 <= fanout <= logical_shards:
            raise ValueError("fanout must be within [1, logical_shards]")
        return rank_kmeans_shards(query, centroids, spec)[:fanout].astype(int).tolist()
    raise ValueError(f"unsupported partition method: {method}")


def merge_shard_results(
    shard_results: Sequence[ShardSearch], spec: DatasetSpec, top_k: int = TOP_K
) -> list[int]:
    best: dict[int, float] = {}
    for shard in shard_results:
        for point_id, score in shard.results:
            previous = best.get(point_id)
            if previous is None:
                best[point_id] = score
            elif spec.normalize and score > previous:
                best[point_id] = score
            elif not spec.normalize and score < previous:
                best[point_id] = score
    ordered = sorted(best.items(), key=lambda item: item[1], reverse=spec.normalize)
    return [point_id for point_id, _score in ordered[:top_k]]


def search_one_shard(
    node: Node,
    collection: str,
    shard_id: int,
    query: np.ndarray,
    ef_search: int,
    dimension: int,
    *,
    require_worker_cpu_time: bool,
) -> ShardSearch:
    started = time.perf_counter_ns()
    payload, response_bytes = HTTP.request(
        node.base_url,
        "POST",
        f"/collections/{encoded_collection(collection)}/points/search",
        {
            "vector": np.asarray(query, dtype=np.float32).tolist(),
            "limit": TOP_K,
            "with_payload": False,
            "with_vector": False,
            "shard_key": [shard_key(collection, shard_id)],
            "params": {"hnsw_ef": ef_search, "exact": False},
        },
        timeout=300.0,
    )
    latency_us = (time.perf_counter_ns() - started) / 1_000.0
    usage = payload.get("usage")
    hardware = usage.get("hardware") if isinstance(usage, dict) else None
    if not isinstance(hardware, dict):
        raise RuntimeError("search response is missing usage.hardware")
    required = {"cpu", "graph_nodes_visited"}
    if require_worker_cpu_time:
        required.update({"cpu_time_us", "cpu_wall_time_us"})
    missing = sorted(required - set(hardware))
    if missing:
        raise RuntimeError(f"search response is missing hardware fields: {missing}")
    distance_computations = distance_computations_from_usage(int(hardware["cpu"]), dimension)
    results = [
        (int(item["id"]), float(item["score"])) for item in (payload.get("result") or [])
    ]
    return ShardSearch(
        shard_id=shard_id,
        latency_us=latency_us,
        distance_computations=distance_computations,
        nodes_visited=int(hardware["graph_nodes_visited"]),
        worker_cpu_time_us=int(hardware.get("cpu_time_us") or 0),
        worker_cpu_wall_time_us=int(hardware.get("cpu_wall_time_us") or 0),
        response_bytes=response_bytes,
        results=results,
    )


def execute_query(
    query_id: int,
    query: np.ndarray,
    ground_truth: np.ndarray,
    spec: DatasetSpec,
    method: str,
    logical_shards: int,
    fanout: int,
    ef_search: int,
    collection: str,
    placement: dict[int, Node],
    centroids: np.ndarray,
    oracle_fanout: int,
    request_executor: ThreadPoolExecutor,
    *,
    require_worker_cpu_time: bool,
) -> dict[str, Any]:
    e2e_started = time.perf_counter_ns()
    routing_cpu_started = time.thread_time_ns()
    routing_started = time.perf_counter_ns()
    selected = selected_shards_for_query(
        query,
        spec,
        method,
        logical_shards,
        fanout,
        centroids,
    )
    routing_latency_us = (time.perf_counter_ns() - routing_started) / 1_000.0
    routing_cpu_time_us = (time.thread_time_ns() - routing_cpu_started) / 1_000.0
    futures: list[Future[ShardSearch]] = [
        request_executor.submit(
            search_one_shard,
            placement[shard_id],
            collection,
            shard_id,
            query,
            ef_search,
            int(query.shape[0]),
            require_worker_cpu_time=require_worker_cpu_time,
        )
        for shard_id in selected
    ]
    shard_results = [future.result() for future in futures]
    shard_results.sort(key=lambda item: selected.index(item.shard_id))
    result_ids = merge_shard_results(shard_results, spec)
    achieved = float(recall_per_query([result_ids], ground_truth[None, :])[0])
    return {
        "query_id": query_id,
        "dataset": spec.name,
        "partition_method": method,
        "logical_shards": logical_shards,
        "physical_hosts": len({placement[shard_id].private_ip for shard_id in selected}),
        "target_recall": TARGET_RECALL,
        "achieved_recall": achieved,
        "queried_shards": len(selected),
        "queried_shard_ids": selected,
        "routing_latency_us": routing_latency_us,
        "routing_cpu_time_us": routing_cpu_time_us,
        "local_search_latency_us_per_shard": [item.latency_us for item in shard_results],
        "distance_computations_per_shard": [
            item.distance_computations for item in shard_results
        ],
        "nodes_visited_per_shard": [item.nodes_visited for item in shard_results],
        "worker_cpu_time_us_per_shard": [
            item.worker_cpu_time_us for item in shard_results
        ],
        "worker_cpu_wall_time_us_per_shard": [
            item.worker_cpu_wall_time_us for item in shard_results
        ],
        "response_bytes_per_shard": [item.response_bytes for item in shard_results],
        "end_to_end_latency_us": (time.perf_counter_ns() - e2e_started) / 1_000.0,
        "oracle_minimum_fanout": oracle_fanout,
        "result_ids": result_ids,
    }


def prefix_metrics_from_ranked_searches(
    shard_results: Sequence[ShardSearch],
    ground_truth: np.ndarray,
    spec: DatasetSpec,
) -> list[dict[str, Any]]:
    if not shard_results:
        raise ValueError("prefix reuse requires at least one ranked shard search")
    metrics: list[dict[str, Any]] = []
    for fanout in range(1, len(shard_results) + 1):
        prefix = shard_results[:fanout]
        result_ids = merge_shard_results(prefix, spec)
        metrics.append(
            {
                "fanout": fanout,
                "recall_at_10": float(
                    recall_per_query([result_ids], ground_truth[None, :])[0]
                ),
                "aggregate_distance_computations": sum(
                    item.distance_computations for item in prefix
                ),
                "aggregate_nodes_visited": sum(item.nodes_visited for item in prefix),
                "aggregate_worker_cpu_time_us": sum(
                    item.worker_cpu_time_us for item in prefix
                ),
            }
        )
    return metrics


def execute_kmeans_prefix_query(
    query: np.ndarray,
    ground_truth: np.ndarray,
    spec: DatasetSpec,
    logical_shards: int,
    ef_search: int,
    collection: str,
    placement: dict[int, Node],
    centroids: np.ndarray,
    request_executor: ThreadPoolExecutor,
    *,
    require_worker_cpu_time: bool,
) -> list[dict[str, Any]]:
    ranked_shards = selected_shards_for_query(
        query,
        spec,
        "kmeans",
        logical_shards,
        logical_shards,
        centroids,
    )
    futures: list[Future[ShardSearch]] = [
        request_executor.submit(
            search_one_shard,
            placement[shard_id],
            collection,
            shard_id,
            query,
            ef_search,
            int(query.shape[0]),
            require_worker_cpu_time=require_worker_cpu_time,
        )
        for shard_id in ranked_shards
    ]
    by_shard = {result.shard_id: result for result in (future.result() for future in futures)}
    ordered = [by_shard[shard_id] for shard_id in ranked_shards]
    return prefix_metrics_from_ranked_searches(ordered, ground_truth, spec)


def evaluate_kmeans_prefix_tuning_grid(
    topology_path: str | Path,
    hdf5_path: str | Path,
    partition_path: str | Path,
    dataset: str,
    logical_shards: int,
    collection: str,
    ef_grid: Sequence[int],
    tuning_query_count: int,
    *,
    query_concurrency: int,
    request_workers: int,
    require_worker_cpu_time: bool,
    require_benchmark_affinity: bool,
    experiment_id: str,
    manifest_jsonl: str | Path,
) -> list[dict[str, Any]]:
    benchmark_affinity = (
        validate_benchmark_affinity(topology_path)
        if require_benchmark_affinity
        else current_cpu_affinity()
    )
    spec = DATASETS[dataset]
    partition = load_partition(partition_path, "kmeans", logical_shards)
    topology = load_topology(topology_path)
    nodes = discover_nodes(topology)
    placement = round_robin_placement(logical_shards, nodes)
    with h5py.File(hdf5_path, "r") as handle:
        vector_count, dimension = map(int, handle["train"].shape)
        stop = min(tuning_query_count, len(handle["test"]))
        if stop <= 0:
            raise ValueError("requested tuning query range is empty")
        queries = preprocess_rows(handle["test"][:stop], spec)
        truth = np.asarray(handle["neighbors"][:stop, :TOP_K], dtype=np.int32)

    dataset_checksum = sha256_path(hdf5_path)
    git_commit = repository_commit(Path(__file__).resolve().parents[3])
    candidates_by_key: dict[tuple[int, int], dict[str, Any]] = {}
    for ef_search in ef_grid:
        rows_by_fanout: dict[int, list[dict[str, Any] | None]] = {
            fanout: [None] * len(queries)
            for fanout in range(1, logical_shards + 1)
        }
        with ThreadPoolExecutor(max_workers=request_workers) as request_executor:
            with ThreadPoolExecutor(max_workers=query_concurrency) as query_executor:
                futures = {
                    query_executor.submit(
                        execute_kmeans_prefix_query,
                        queries[offset],
                        truth[offset],
                        spec,
                        logical_shards,
                        ef_search,
                        collection,
                        placement,
                        partition.centroids,
                        request_executor,
                        require_worker_cpu_time=require_worker_cpu_time,
                    ): offset
                    for offset in range(len(queries))
                }
                completed = 0
                for future in as_completed(futures):
                    offset = futures[future]
                    prefix_metrics = future.result()
                    if len(prefix_metrics) != logical_shards:
                        raise RuntimeError("prefix tuning did not return every fan-out")
                    for row in prefix_metrics:
                        rows_by_fanout[int(row["fanout"])][offset] = row
                    completed += 1
                    if completed % 100 == 0 or completed == len(queries):
                        print(
                            f"prefix-tune {collection} ef={ef_search}: "
                            f"{completed}/{len(queries)} queries",
                            flush=True,
                        )

        for fanout in range(1, logical_shards + 1):
            rows = rows_by_fanout[fanout]
            if any(row is None for row in rows):
                raise RuntimeError(
                    f"prefix tuning did not record all queries for fan-out {fanout}"
                )
            typed_rows = [row for row in rows if row is not None]
            recall = statistics.fmean(float(row["recall_at_10"]) for row in typed_rows)
            candidate = {
                "experiment_id": experiment_id,
                "record_type": "tuning_candidate",
                "timestamp": utc_timestamp(),
                "git_commit": git_commit,
                "dataset": dataset,
                "dataset_checksum": dataset_checksum,
                "partition_method": "kmeans",
                "partition_seed": PARTITION_SEED,
                "logical_shard_count": logical_shards,
                "physical_machine_count": min(logical_shards, 4),
                "fanout": fanout,
                "ef_search": ef_search,
                "recall_at_10": recall,
                "mean_aggregate_distance_computations": statistics.fmean(
                    int(row["aggregate_distance_computations"]) for row in typed_rows
                ),
                "mean_nodes_visited_per_query": statistics.fmean(
                    int(row["aggregate_nodes_visited"]) for row in typed_rows
                ),
                "mean_worker_cpu_time_us_per_query": statistics.fmean(
                    int(row["aggregate_worker_cpu_time_us"]) for row in typed_rows
                ),
                "query_count": len(typed_rows),
                "target_recall": TARGET_RECALL,
                "status": "VALID" if recall >= TARGET_RECALL else "INVALID_RECALL",
                "benchmark_cpu_affinity": benchmark_affinity,
                "query_concurrency": query_concurrency,
                "request_workers": request_workers,
            }
            candidate.update(
                {
                    "graph_build_seed": topology.get("hnsw_graph_build_seed"),
                    "logical_to_physical_mapping": {
                        str(shard_id): placement[shard_id].private_ip
                        for shard_id in range(logical_shards)
                    },
                    "vector_count": vector_count,
                    "dimension": dimension,
                    "distance_metric": spec.qdrant_distance,
                    "k": TOP_K,
                    "achieved_recall": recall,
                    "routing_fanout": fanout,
                    "efSearch": ef_search,
                    "HNSW_M": HNSW_M,
                    "HNSW_efConstruction": HNSW_EF_CONSTRUCTION,
                    "warmup_query_count": 0,
                    "CPU_affinity": {
                        "aggregator": benchmark_affinity,
                        "workers": {
                            node.private_ip: node.cpuset
                            for node in nodes[: min(logical_shards, 4)]
                        },
                    },
                    "worker_threads": {
                        node.private_ip: int(topology_node["max_search_threads"])
                        for node, topology_node in zip(
                            nodes,
                            [topology["controller"], *topology["workers"]],
                            strict=True,
                        )
                    },
                    "aggregator_threads": query_concurrency,
                    "machine_hostname": [
                        node.ssh_host for node in nodes[: min(logical_shards, 4)]
                    ],
                    "index_memory_bytes": (
                        "not_recorded_in_prefix_reuse_tuning_candidate"
                    ),
                    "qps": "not_applicable_prefix_reuse_candidate",
                    "mean_latency": "not_applicable_prefix_reuse_candidate",
                    "p50_latency": "not_applicable_prefix_reuse_candidate",
                    "p95_latency": "not_applicable_prefix_reuse_candidate",
                    "p99_latency": "not_applicable_prefix_reuse_candidate",
                    "mean_shards_per_query": fanout,
                    "p95_shards_per_query": fanout,
                    "distance_computations_per_query": candidate[
                        "mean_aggregate_distance_computations"
                    ],
                    "nodes_visited_per_query": candidate[
                        "mean_nodes_visited_per_query"
                    ],
                    "worker_cpu_time_per_query": candidate[
                        "mean_worker_cpu_time_us_per_query"
                    ],
                    "routing_cpu_time_per_query": (
                        "not_applicable_prefix_reuse_candidate"
                    ),
                    "aggregator_cpu_utilization": (
                        "not_applicable_non_throughput_fixed_quality_run"
                    ),
                    "worker_cpu_utilization": (
                        "not_applicable_non_throughput_fixed_quality_run"
                    ),
                    "network_bytes_per_query": (
                        "not_applicable_prefix_reuse_candidate"
                    ),
                }
            )
            candidates_by_key[(fanout, ef_search)] = candidate
            append_jsonl(manifest_jsonl, candidate)

    return [
        candidates_by_key[(fanout, ef_search)]
        for fanout in range(1, logical_shards + 1)
        for ef_search in ef_grid
    ]


def percentile(values: Sequence[float], percent: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), percent))


def summarize_rows(rows: Sequence[dict[str, Any]], elapsed_seconds: float) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot summarize an empty query set")
    latencies = [float(row["end_to_end_latency_us"]) for row in rows]
    shard_counts = [int(row["queried_shards"]) for row in rows]
    distance = [sum(map(int, row["distance_computations_per_shard"])) for row in rows]
    nodes = [sum(map(int, row["nodes_visited_per_shard"])) for row in rows]
    worker_cpu = [sum(map(int, row["worker_cpu_time_us_per_shard"])) for row in rows]
    routing = [float(row["routing_latency_us"]) for row in rows]
    routing_cpu = [float(row["routing_cpu_time_us"]) for row in rows]
    network = [sum(map(int, row["response_bytes_per_shard"])) for row in rows]
    return {
        "query_count": len(rows),
        "achieved_recall": statistics.fmean(float(row["achieved_recall"]) for row in rows),
        "mean_shards_per_query": statistics.fmean(shard_counts),
        "median_shards_per_query": statistics.median(shard_counts),
        "p95_shards_per_query": percentile(shard_counts, 95),
        "mean_distance_computations_per_query": statistics.fmean(distance),
        "p95_distance_computations_per_query": percentile(distance, 95),
        "mean_nodes_visited_per_query": statistics.fmean(nodes),
        "p95_nodes_visited_per_query": percentile(nodes, 95),
        "mean_worker_cpu_time_us_per_query": statistics.fmean(worker_cpu),
        "mean_routing_latency_us": statistics.fmean(routing),
        "mean_routing_cpu_time_us": statistics.fmean(routing_cpu),
        "mean_latency_us": statistics.fmean(latencies),
        "p50_latency_us": percentile(latencies, 50),
        "p95_latency_us": percentile(latencies, 95),
        "p99_latency_us": percentile(latencies, 99),
        "completed_qps": len(rows) / max(elapsed_seconds, 1e-9),
        "network_bytes_per_query": statistics.fmean(network),
        "oracle_fanout_mean": statistics.fmean(
            int(row["oracle_minimum_fanout"]) for row in rows
        ),
        "oracle_fanout_median": statistics.median(
            int(row["oracle_minimum_fanout"]) for row in rows
        ),
        "oracle_fanout_p95": percentile(
            [int(row["oracle_minimum_fanout"]) for row in rows], 95
        ),
    }


def protocol_metadata_from_query_summary(summary: dict[str, Any]) -> dict[str, Any]:
    """Normalize one fixed-quality query run to the Section 32 field names."""
    return {
        "graph_build_seed": summary["graph_build_seed"],
        "logical_to_physical_mapping": summary["logical_to_physical_mapping"],
        "vector_count": summary["vector_count"],
        "dimension": summary["dimension"],
        "distance_metric": summary["distance_metric"],
        "k": summary["k"],
        "achieved_recall": summary["achieved_recall"],
        "routing_fanout": summary["routing_fanout"],
        "efSearch": summary["efSearch"],
        "HNSW_M": summary["HNSW_M"],
        "HNSW_efConstruction": summary["HNSW_efConstruction"],
        "warmup_query_count": summary["warmup_query_count"],
        "CPU_affinity": summary["CPU_affinity"],
        "worker_threads": summary["worker_threads"],
        "aggregator_threads": summary["aggregator_threads"],
        "machine_hostname": summary["machine_hostname"],
        "index_memory_bytes": "not_recorded_in_fixed_quality_query_summary",
        "qps": summary["completed_qps"],
        "mean_latency": summary["mean_latency_us"],
        "p50_latency": summary["p50_latency_us"],
        "p95_latency": summary["p95_latency_us"],
        "p99_latency": summary["p99_latency_us"],
        "mean_shards_per_query": summary["mean_shards_per_query"],
        "p95_shards_per_query": summary["p95_shards_per_query"],
        "distance_computations_per_query": summary[
            "mean_distance_computations_per_query"
        ],
        "nodes_visited_per_query": summary["mean_nodes_visited_per_query"],
        "worker_cpu_time_per_query": summary["mean_worker_cpu_time_us_per_query"],
        "routing_cpu_time_per_query": summary["mean_routing_cpu_time_us"],
        "aggregator_cpu_utilization": (
            "not_applicable_non_throughput_fixed_quality_run"
        ),
        "worker_cpu_utilization": "not_applicable_non_throughput_fixed_quality_run",
        "network_bytes_per_query": summary["network_bytes_per_query"],
    }


def evaluate_queries(
    topology_path: str | Path,
    hdf5_path: str | Path,
    partition_path: str | Path,
    dataset: str,
    method: str,
    logical_shards: int,
    collection: str,
    fanout: int,
    ef_search: int,
    query_start: int,
    query_count: int,
    *,
    query_concurrency: int,
    request_workers: int,
    require_worker_cpu_time: bool,
    require_benchmark_affinity: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if method == "random" and fanout != logical_shards:
        raise ValueError("Random serving must broadcast to every shard")
    if not 1 <= fanout <= logical_shards:
        raise ValueError("fanout must be within [1, logical_shards]")
    benchmark_affinity = (
        validate_benchmark_affinity(topology_path)
        if require_benchmark_affinity
        else current_cpu_affinity()
    )
    spec = DATASETS[dataset]
    partition = load_partition(partition_path, method, logical_shards)
    topology = load_topology(topology_path)
    nodes = discover_nodes(topology)
    placement = round_robin_placement(logical_shards, nodes)
    with h5py.File(hdf5_path, "r") as handle:
        vector_count, dimension = map(int, handle["train"].shape)
        test_count = len(handle["test"])
        stop = min(query_start + query_count, test_count)
        if query_start < 0 or stop <= query_start:
            raise ValueError("requested query range is empty or invalid")
        queries = preprocess_rows(handle["test"][query_start:stop], spec)
        truth = np.asarray(handle["neighbors"][query_start:stop, :TOP_K], dtype=np.int32)
    oracle = oracle_minimum_fanout(
        partition.assignments, truth, logical_shards, target_recall=TARGET_RECALL
    )
    rows: list[dict[str, Any] | None] = [None] * len(queries)
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=request_workers) as request_executor:
        with ThreadPoolExecutor(max_workers=query_concurrency) as query_executor:
            futures = {
                query_executor.submit(
                    execute_query,
                    query_start + offset,
                    queries[offset],
                    truth[offset],
                    spec,
                    method,
                    logical_shards,
                    fanout,
                    ef_search,
                    collection,
                    placement,
                    partition.centroids,
                    int(oracle[offset]),
                    request_executor,
                    require_worker_cpu_time=require_worker_cpu_time,
                ): offset
                for offset in range(len(queries))
            }
            completed = 0
            for future in as_completed(futures):
                offset = futures[future]
                rows[offset] = future.result()
                completed += 1
                if completed % 100 == 0 or completed == len(queries):
                    print(
                        f"measure {collection} P={fanout} ef={ef_search}: "
                        f"{completed}/{len(queries)} queries",
                        flush=True,
                    )
    elapsed = time.perf_counter() - started
    typed_rows = [row for row in rows if row is not None]
    if len(typed_rows) != len(queries):
        raise RuntimeError("one or more query results were not recorded")
    summary = summarize_rows(typed_rows, elapsed)
    summary.update(
        {
            "timestamp": utc_timestamp(),
            "dataset": dataset,
            "partition_method": method,
            "logical_shards": logical_shards,
            "physical_hosts": min(logical_shards, 4),
            "collection": collection,
            "fanout": fanout,
            "ef_search": ef_search,
            "target_recall": TARGET_RECALL,
            "query_start": query_start,
            "query_stop": query_start + len(queries),
            "elapsed_seconds": elapsed,
            "benchmark_cpu_affinity": benchmark_affinity,
            "query_concurrency": query_concurrency,
            "request_workers": request_workers,
            "git_commit": repository_commit(Path(__file__).resolve().parents[3]),
            "dataset_checksum": sha256_path(hdf5_path),
            "partition_seed": int(partition.metadata["partition_seed"]),
            "graph_build_seed": topology.get("hnsw_graph_build_seed"),
            "physical_machine_count": min(logical_shards, 4),
            "logical_shard_count": logical_shards,
            "logical_to_physical_mapping": {
                str(shard_id): placement[shard_id].private_ip
                for shard_id in range(logical_shards)
            },
            "vector_count": vector_count,
            "dimension": dimension,
            "distance_metric": spec.qdrant_distance,
            "k": TOP_K,
            "routing_fanout": fanout,
            "efSearch": ef_search,
            "HNSW_M": HNSW_M,
            "HNSW_efConstruction": HNSW_EF_CONSTRUCTION,
            "hnsw_max_indexing_threads": HNSW_MAX_INDEXING_THREADS,
            "max_optimization_threads": MAX_OPTIMIZATION_THREADS,
            "warmup_query_count": 0,
            "CPU_affinity": {
                "aggregator": benchmark_affinity,
                "workers": {
                    node.private_ip: node.cpuset for node in nodes[: min(logical_shards, 4)]
                },
            },
            "worker_threads": {
                node.private_ip: int(topology_node["max_search_threads"])
                for node, topology_node in zip(
                    nodes,
                    [topology["controller"], *topology["workers"]],
                    strict=True,
                )
            },
            "aggregator_threads": query_concurrency,
            "machine_hostname": [node.ssh_host for node in nodes[: min(logical_shards, 4)]],
            "result_status": (
                "VALID" if float(summary["achieved_recall"]) >= TARGET_RECALL else "INVALID_RECALL"
            ),
        }
    )
    return typed_rows, summary


def build_e3_route_plan(
    queries: np.ndarray,
    spec: DatasetSpec,
    method: str,
    logical_shards: int,
    fanout: int,
    centroids: np.ndarray,
) -> dict[int, list[tuple[int, int]]]:
    plan: dict[int, list[tuple[int, int]]] = {
        shard_id: [] for shard_id in range(logical_shards)
    }
    for offset, query in enumerate(queries):
        selected = selected_shards_for_query(
            query,
            spec,
            method,
            logical_shards,
            fanout,
            centroids,
        )
        for route_rank, shard_id in enumerate(selected, start=1):
            plan[shard_id].append((offset, route_rank))
    return plan


def local_llc_size_bytes(
    path: str | Path = "/sys/devices/system/cpu/cpu0/cache/index3/size",
) -> int | None:
    source = Path(path)
    if not source.is_file():
        return None
    value = source.read_text(encoding="utf-8").strip().upper()
    multipliers = {"K": 1024, "M": 1024**2, "G": 1024**3}
    if value and value[-1] in multipliers:
        return int(value[:-1]) * multipliers[value[-1]]
    return int(value)


def local_graph_index_size_bytes(
    storage_root: str | Path,
    collection: str,
    physical_shard_id: int,
) -> int | None:
    segment_root = (
        Path(storage_root).expanduser().resolve()
        / "collections"
        / collection
        / str(physical_shard_id)
        / "segments"
    )
    if not segment_root.is_dir():
        return None
    files = [
        path
        for path in segment_root.rglob("*")
        if path.is_file() and "vector_index" in path.parts
    ]
    return sum(path.stat().st_size for path in files) if files else None


def command_on_node(node: Node, script: str) -> list[str]:
    if node.ssh_host in {"localhost", "127.0.0.1"}:
        return ["bash", "-lc", script]
    return [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=10",
        node.ssh_host,
        "bash",
        "-c",
        shlex.quote(script),
    ]


def graph_index_size_bytes_on_node(
    node: Node,
    storage_root: str | Path,
    collection: str,
    physical_shard_id: int,
) -> int | None:
    segment_root = (
        Path(storage_root)
        / "collections"
        / collection
        / str(physical_shard_id)
        / "segments"
    )
    script = (
        "set -euo pipefail; "
        f"root={shlex.quote(str(segment_root))}; "
        "if [ ! -d \"$root\" ]; then printf 'missing\\n'; exit 0; fi; "
        "find \"$root\" -type f -path '*/vector_index/*' -printf '%s\\n' "
        "| awk '{sum += $1} END {print sum + 0}'"
    )
    result = subprocess.run(
        command_on_node(node, script),
        check=True,
        capture_output=True,
        text=True,
        timeout=60.0,
    )
    value = result.stdout.strip()
    if not value or value == "missing":
        return None
    return int(value)


def llc_size_bytes_on_node(node: Node) -> int | None:
    script = "cat /sys/devices/system/cpu/cpu0/cache/index3/size 2>/dev/null || true"
    result = subprocess.run(
        command_on_node(node, script),
        check=True,
        capture_output=True,
        text=True,
        timeout=30.0,
    )
    value = result.stdout.strip().upper()
    if not value:
        return None
    multipliers = {"K": 1024, "M": 1024**2, "G": 1024**3}
    if value[-1] in multipliers:
        return int(value[:-1]) * multipliers[value[-1]]
    return int(value)


def collect_collection_resource_facts(
    nodes: Sequence[Node],
    collection: str,
    *,
    controller_storage_root: str | Path | None,
    storage_roots_by_role: dict[str, str | Path] | None = None,
) -> dict[str, dict[str, Any]]:
    facts: dict[str, dict[str, Any]] = {}
    for node in nodes:
        llc_bytes = llc_size_bytes_on_node(node)
        payload, _ = HTTP.request(
            node.base_url,
            "GET",
            "/telemetry?details_level=3",
            timeout=60.0,
        )
        telemetry = payload.get("result") or {}
        resident_bytes = int((telemetry.get("memory") or {}).get("resident_bytes") or 0)
        collections = (telemetry.get("collections") or {}).get("collections") or []
        target = next(
            (item for item in collections if str(item.get("id") or "") == collection),
            None,
        )
        if target is None:
            continue
        for shard in target.get("shards") or []:
            local = shard.get("local")
            shard_key_value = shard.get("key")
            if not isinstance(local, dict) or shard_key_value is None:
                continue
            graph_bytes = None
            graph_size_source = "unavailable_in_qdrant_telemetry"
            if storage_roots_by_role is not None:
                storage_root = storage_roots_by_role.get(node.role)
                if storage_root is None:
                    raise ValueError(f"missing storage root for role {node.role}")
                graph_bytes = graph_index_size_bytes_on_node(
                    node,
                    storage_root,
                    collection,
                    int(shard["id"]),
                )
                graph_size_source = "node_storage_files"
            elif controller_storage_root is not None and node.role == "controller":
                graph_bytes = local_graph_index_size_bytes(
                    controller_storage_root,
                    collection,
                    int(shard["id"]),
                )
                graph_size_source = "controller_storage_files"
            vector_bytes = int(local.get("vectors_size_bytes") or 0)
            facts[str(shard_key_value)] = {
                "physical_host": node.private_ip,
                "physical_peer_id": node.peer_id,
                "physical_shard_id": int(shard["id"]),
                "point_count": int(local.get("num_points") or 0),
                "vector_storage_size_bytes": vector_bytes,
                "graph_index_size_bytes": graph_bytes,
                "total_local_index_size_bytes": (
                    vector_bytes + graph_bytes if graph_bytes is not None else None
                ),
                "qdrant_resident_set_size_bytes": resident_bytes,
                "machine_llc_size_bytes": llc_bytes,
                "graph_size_source": graph_size_source,
            }
    return facts


def summarize_e3_search_rows(
    rows: Sequence[dict[str, Any]],
    *,
    query_count: int,
    logical_shards: int,
) -> dict[str, Any]:
    if query_count <= 0:
        raise ValueError("E3 query_count must be positive")
    by_shard: dict[int, list[dict[str, Any]]] = {
        shard_id: [] for shard_id in range(logical_shards)
    }
    for row in rows:
        by_shard[int(row["shard_id"])].append(row)

    shard_summaries: list[dict[str, Any]] = []
    for shard_id in range(logical_shards):
        shard_rows = by_shard[shard_id]
        shard_summaries.append(
            {
                "shard_id": shard_id,
                "point_count": (
                    int(shard_rows[0]["shard_point_count"]) if shard_rows else None
                ),
                "searched_query_count": len(shard_rows),
                "selection_frequency": len(shard_rows) / query_count,
                "mean_distance_computations": (
                    statistics.fmean(
                        int(row["distance_computations"]) for row in shard_rows
                    )
                    if shard_rows
                    else None
                ),
                "mean_nodes_visited": (
                    statistics.fmean(int(row["nodes_visited"]) for row in shard_rows)
                    if shard_rows
                    else None
                ),
                "mean_worker_cpu_time_us": (
                    statistics.fmean(
                        int(row["worker_cpu_time_us"]) for row in shard_rows
                    )
                    if shard_rows
                    else None
                ),
                "mean_worker_cpu_wall_time_us": (
                    statistics.fmean(
                        int(row["worker_cpu_wall_time_us"]) for row in shard_rows
                    )
                    if shard_rows
                    else None
                ),
                "mean_local_search_latency_us": (
                    statistics.fmean(
                        float(row["local_search_latency_us"]) for row in shard_rows
                    )
                    if shard_rows
                    else None
                ),
            }
        )

    if not rows:
        raise ValueError("E3 route selected no shard searches")
    distances = [int(row["distance_computations"]) for row in rows]
    nodes = [int(row["nodes_visited"]) for row in rows]
    worker_cpu = [int(row["worker_cpu_time_us"]) for row in rows]
    worker_wall = [int(row["worker_cpu_wall_time_us"]) for row in rows]
    local_latency = [float(row["local_search_latency_us"]) for row in rows]
    response_bytes = [int(row["response_bytes"]) for row in rows]
    return {
        "query_count": query_count,
        "searched_shard_events": len(rows),
        "mean_shards_per_query": len(rows) / query_count,
        "mean_distance_computations_per_searched_shard": statistics.fmean(distances),
        "p95_distance_computations_per_searched_shard": percentile(distances, 95),
        "mean_nodes_visited_per_searched_shard": statistics.fmean(nodes),
        "p95_nodes_visited_per_searched_shard": percentile(nodes, 95),
        "mean_worker_cpu_time_us_per_searched_shard": statistics.fmean(worker_cpu),
        "mean_worker_cpu_wall_time_us_per_searched_shard": statistics.fmean(worker_wall),
        "mean_local_search_latency_us_per_searched_shard": statistics.fmean(local_latency),
        "p50_local_search_latency_us_per_searched_shard": percentile(
            local_latency, 50
        ),
        "p95_local_search_latency_us_per_searched_shard": percentile(local_latency, 95),
        "p99_local_search_latency_us_per_searched_shard": percentile(
            local_latency, 99
        ),
        "mean_distance_computations_per_query": sum(distances) / query_count,
        "mean_nodes_visited_per_query": sum(nodes) / query_count,
        "mean_worker_cpu_time_us_per_query": sum(worker_cpu) / query_count,
        "network_bytes_per_query": sum(response_bytes) / query_count,
        "shards": shard_summaries,
    }


def e3_measurement_recall(
    rows: Sequence[dict[str, Any]],
    *,
    query_start: int,
    query_count: int,
) -> float:
    if query_count <= 0:
        raise ValueError("E3 query_count must be positive")
    query_stop = query_start + query_count
    contributions = {query_id: 0.0 for query_id in range(query_start, query_stop)}
    seen: set[int] = set()
    for row in rows:
        query_id = int(row["query_id"])
        if query_id not in contributions:
            raise ValueError(f"E3 row has out-of-range query ID {query_id}")
        contribution = float(row["recall_contribution"])
        if not 0.0 <= contribution <= 1.0:
            raise ValueError(
                f"E3 row has invalid recall contribution {contribution}"
            )
        contributions[query_id] += contribution
        seen.add(query_id)
    missing = sorted(set(contributions) - seen)
    if missing:
        raise ValueError(
            f"E3 rows are missing {len(missing)} measurement queries; first={missing[0]}"
        )
    invalid = {
        query_id: value
        for query_id, value in contributions.items()
        if not 0.0 <= value <= 1.0 + 1e-12
    }
    if invalid:
        first_query = min(invalid)
        raise ValueError(
            f"E3 aggregate recall contribution exceeds one for query {first_query}: "
            f"{invalid[first_query]}"
        )
    return statistics.fmean(contributions.values())


def evaluate_e3_local_searches(
    topology_path: str | Path,
    hdf5_path: str | Path,
    partition_path: str | Path,
    dataset: str,
    method: str,
    logical_shards: int,
    collection: str,
    fanout: int,
    ef_search: int,
    selected_tuning_recall: float,
    query_start: int,
    query_count: int,
    *,
    require_worker_cpu_time: bool,
    require_benchmark_affinity: bool,
    controller_storage_root: str | Path | None,
    run_id: str | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not TARGET_RECALL <= selected_tuning_recall <= 1.0:
        raise ValueError(
            "selected tuning recall must be within [target_recall, 1.0] for valid E3"
        )
    benchmark_affinity = (
        validate_benchmark_affinity(topology_path)
        if require_benchmark_affinity
        else current_cpu_affinity()
    )
    spec = DATASETS[dataset]
    partition = load_partition(partition_path, method, logical_shards)
    topology = load_topology(topology_path)
    nodes = discover_nodes(topology)
    placement = round_robin_placement(logical_shards, nodes)
    with h5py.File(hdf5_path, "r") as handle:
        vector_count, dimension = map(int, handle["train"].shape)
        test_count = len(handle["test"])
        stop = min(query_start + query_count, test_count)
        if query_start < 0 or stop <= query_start:
            raise ValueError("requested E3 query range is empty or invalid")
        queries = preprocess_rows(handle["test"][query_start:stop], spec)
        truth = np.asarray(handle["neighbors"][query_start:stop, :TOP_K], dtype=np.int32)
    plan = build_e3_route_plan(
        queries,
        spec,
        method,
        logical_shards,
        fanout,
        partition.centroids,
    )
    shard_counts = np.bincount(
        partition.assignments.astype(np.int64), minlength=logical_shards
    )
    storage_roots_by_role = None
    if run_id is not None:
        if not run_id or Path(run_id).name != run_id:
            raise ValueError("run ID must be one safe path component")
        storage_roots_by_role = {
            node.role: Path(topology["local_storage_root"])
            / run_id
            / node.role
            / "storage"
            for node in nodes
        }
    resources = collect_collection_resource_facts(
        nodes,
        collection,
        controller_storage_root=controller_storage_root,
        storage_roots_by_role=storage_roots_by_role,
    )
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for shard_id in range(logical_shards):
        selected_queries = plan[shard_id]
        node = placement[shard_id]
        for completed, (offset, route_rank) in enumerate(selected_queries, start=1):
            search = search_one_shard(
                node,
                collection,
                shard_id,
                queries[offset],
                ef_search,
                int(queries.shape[1]),
                require_worker_cpu_time=require_worker_cpu_time,
            )
            if search.distance_computations <= 0 or search.nodes_visited <= 0:
                raise RuntimeError(
                    f"E3 shard {shard_id} returned non-positive graph-work counters"
                )
            if require_worker_cpu_time and (
                search.worker_cpu_time_us <= 0 or search.worker_cpu_wall_time_us <= 0
            ):
                raise RuntimeError(
                    f"E3 shard {shard_id} returned non-positive worker-time counters"
                )
            truth_in_shard = truth[offset][
                partition.assignments[truth[offset]] == shard_id
            ]
            result_ids = [point_id for point_id, _score in search.results]
            hits = len(set(map(int, truth_in_shard)) & set(result_ids))
            rows.append(
                {
                    "query_id": query_start + offset,
                    "dataset": dataset,
                    "partition_method": method,
                    "logical_shards": logical_shards,
                    "selected_fanout": fanout,
                    "ef_search": ef_search,
                    "shard_id": shard_id,
                    "route_rank": route_rank,
                    "physical_host": node.private_ip,
                    "shard_point_count": int(shard_counts[shard_id]),
                    "ground_truth_points_in_shard": len(truth_in_shard),
                    "ground_truth_hits": hits,
                    "recall_contribution": hits / TOP_K,
                    "distance_computations": search.distance_computations,
                    "nodes_visited": search.nodes_visited,
                    "worker_cpu_time_us": search.worker_cpu_time_us,
                    "worker_cpu_wall_time_us": search.worker_cpu_wall_time_us,
                    "local_search_latency_us": search.latency_us,
                    "response_bytes": search.response_bytes,
                    "result_ids": result_ids,
                }
            )
            if completed % 500 == 0 or completed == len(selected_queries):
                print(
                    f"E3 {collection} shard={shard_id} ef={ef_search}: "
                    f"{completed}/{len(selected_queries)} searches",
                    flush=True,
                )
    expected_events = len(queries) * fanout
    if len(rows) != expected_events:
        raise RuntimeError(
            f"E3 searched-shard event mismatch: actual={len(rows)}, expected={expected_events}"
        )
    measurement_recall = e3_measurement_recall(
        rows,
        query_start=query_start,
        query_count=len(queries),
    )
    summary = summarize_e3_search_rows(
        rows,
        query_count=len(queries),
        logical_shards=logical_shards,
    )
    summary.update(
        {
            "timestamp": utc_timestamp(),
            "git_commit": repository_commit(Path(__file__).resolve().parents[3]),
            "dataset": dataset,
            "dataset_checksum": sha256_path(hdf5_path),
            "partition_method": method,
            "partition_seed": int(partition.metadata["partition_seed"]),
            "logical_shards": logical_shards,
            "physical_hosts": min(logical_shards, 4),
            "collection": collection,
            "selected_fanout": fanout,
            "ef_search": ef_search,
            "selected_tuning_recall": selected_tuning_recall,
            "measurement_recall": measurement_recall,
            "target_recall": TARGET_RECALL,
            "query_start": query_start,
            "query_stop": query_start + len(queries),
            "elapsed_seconds": time.perf_counter() - started,
            "benchmark_cpu_affinity": benchmark_affinity,
            "query_concurrency_per_shard": 1,
            "competing_logical_shards": 0,
            "partition_artifact_sha256": partition.sha256,
            "graph_build_seed": topology.get("hnsw_graph_build_seed"),
            "hnsw_max_indexing_threads": HNSW_MAX_INDEXING_THREADS,
            "max_optimization_threads": MAX_OPTIMIZATION_THREADS,
            "deterministic_graph_construction": (
                topology.get("hnsw_graph_build_seed") is not None
                and HNSW_MAX_INDEXING_THREADS == 1
                and MAX_OPTIMIZATION_THREADS == 1
            ),
            "physical_machine_count": min(logical_shards, 4),
            "logical_shard_count": logical_shards,
            "logical_to_physical_mapping": {
                str(shard_id): placement[shard_id].private_ip
                for shard_id in range(logical_shards)
            },
            "vector_count": vector_count,
            "dimension": dimension,
            "distance_metric": spec.qdrant_distance,
            "k": TOP_K,
            "achieved_recall": measurement_recall,
            "routing_fanout": fanout,
            "efSearch": ef_search,
            "HNSW_M": HNSW_M,
            "HNSW_efConstruction": HNSW_EF_CONSTRUCTION,
            "warmup_query_count": 0,
            "CPU_affinity": {
                "aggregator": benchmark_affinity,
                "workers": {
                    node.private_ip: node.cpuset for node in nodes[: min(logical_shards, 4)]
                },
            },
            "worker_threads": {
                node.private_ip: int(topology_node["max_search_threads"])
                for node, topology_node in zip(
                    nodes,
                    [topology["controller"], *topology["workers"]],
                    strict=True,
                )
            },
            "aggregator_threads": "not_applicable_isolated_e3",
            "machine_hostname": [
                node.ssh_host for node in nodes[: min(logical_shards, 4)]
            ],
            "index_memory_bytes": {
                shard_id: resource["total_local_index_size_bytes"]
                for shard_id, resource in {
                    str(shard_id): resources.get(shard_key(collection, shard_id), {})
                    for shard_id in range(logical_shards)
                }.items()
            },
            "qps": "not_applicable_isolated_e3",
            "mean_latency": summary[
                "mean_local_search_latency_us_per_searched_shard"
            ],
            "p50_latency": summary[
                "p50_local_search_latency_us_per_searched_shard"
            ],
            "p95_latency": summary[
                "p95_local_search_latency_us_per_searched_shard"
            ],
            "p99_latency": summary[
                "p99_local_search_latency_us_per_searched_shard"
            ],
            "p95_shards_per_query": fanout,
            "distance_computations_per_query": summary[
                "mean_distance_computations_per_query"
            ],
            "nodes_visited_per_query": summary["mean_nodes_visited_per_query"],
            "worker_cpu_time_per_query": summary[
                "mean_worker_cpu_time_us_per_query"
            ],
            "routing_cpu_time_per_query": "not_applicable_isolated_e3",
            "aggregator_cpu_utilization": "not_applicable_isolated_e3",
            "worker_cpu_utilization": "not_applicable_isolated_e3",
            "network_bytes_per_query": summary["network_bytes_per_query"],
            "shard_resources": {
                str(shard_id): resources.get(
                    shard_key(collection, shard_id),
                    {
                        "physical_host": placement[shard_id].private_ip,
                        "point_count": int(shard_counts[shard_id]),
                        "vector_storage_size_bytes": None,
                        "graph_index_size_bytes": None,
                        "total_local_index_size_bytes": None,
                        "qdrant_resident_set_size_bytes": None,
                        "machine_llc_size_bytes": local_llc_size_bytes(),
                        "graph_size_source": "telemetry_record_missing",
                    },
                )
                for shard_id in range(logical_shards)
            },
            "result_status": (
                "VALID_E3" if measurement_recall >= TARGET_RECALL else "INVALID_RECALL"
            ),
        }
    )
    return rows, summary


def write_json_atomic(path: str | Path, payload: Any) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, destination)
    return destination


def write_per_query_csv(path: str | Path, rows: Sequence[dict[str, Any]]) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PER_QUERY_FIELDS)
        writer.writeheader()
        for row in rows:
            encoded = dict(row)
            for key in (
                "queried_shard_ids",
                "local_search_latency_us_per_shard",
                "distance_computations_per_shard",
                "nodes_visited_per_shard",
                "worker_cpu_time_us_per_shard",
                "worker_cpu_wall_time_us_per_shard",
                "response_bytes_per_shard",
                "result_ids",
            ):
                encoded[key] = json.dumps(encoded[key], separators=(",", ":"))
            writer.writerow({key: encoded[key] for key in PER_QUERY_FIELDS})
    os.replace(temporary, destination)
    return destination


def write_e3_search_csv(path: str | Path, rows: Sequence[dict[str, Any]]) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=E3_SEARCH_FIELDS)
        writer.writeheader()
        for row in rows:
            encoded = dict(row)
            encoded["result_ids"] = json.dumps(
                encoded["result_ids"], separators=(",", ":")
            )
            writer.writerow({key: encoded[key] for key in E3_SEARCH_FIELDS})
    os.replace(temporary, destination)
    return destination


def append_jsonl(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def parse_int_grid(value: str) -> list[int]:
    result = [int(item) for item in value.split(",") if item.strip()]
    if not result or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError("grid must contain positive comma-separated integers")
    return sorted(set(result))


def command_prepare(args: argparse.Namespace) -> int:
    payload = ensure_collection(
        args.topology,
        args.hdf5_path,
        args.dataset,
        args.method,
        args.logical_shards,
        args.partition_artifact,
        args.collection,
        batch_size=args.upload_batch_size,
        replace=args.replace,
    )
    write_json_atomic(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def command_measure(args: argparse.Namespace) -> int:
    rows, summary = evaluate_queries(
        args.topology,
        args.hdf5_path,
        args.partition_artifact,
        args.dataset,
        args.method,
        args.logical_shards,
        args.collection,
        args.fanout,
        args.ef_search,
        args.query_start,
        args.query_count,
        query_concurrency=args.query_concurrency,
        request_workers=args.request_workers,
        require_worker_cpu_time=args.require_worker_cpu_time,
        require_benchmark_affinity=args.require_benchmark_affinity,
    )
    write_per_query_csv(args.per_query_output, rows)
    write_json_atomic(args.summary_output, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def command_e3_measure(args: argparse.Namespace) -> int:
    rows, summary = evaluate_e3_local_searches(
        args.topology,
        args.hdf5_path,
        args.partition_artifact,
        args.dataset,
        args.method,
        args.logical_shards,
        args.collection,
        args.fanout,
        args.ef_search,
        args.selected_tuning_recall,
        args.query_start,
        args.query_count,
        require_worker_cpu_time=args.require_worker_cpu_time,
        require_benchmark_affinity=args.require_benchmark_affinity,
        controller_storage_root=args.controller_storage_root,
        run_id=args.run_id,
    )
    summary["experiment_id"] = args.experiment_id
    summary["record_type"] = "e3_local_search_measurement"
    write_e3_search_csv(args.per_search_output, rows)
    write_json_atomic(args.summary_output, summary)
    append_jsonl(args.manifest_jsonl, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["result_status"] == "VALID_E3" else 2


def command_tune(args: argparse.Namespace) -> int:
    if args.kmeans_prefix_reuse:
        if args.method != "kmeans":
            raise ValueError("--kmeans-prefix-reuse is valid only for K-Means tuning")
        candidates = evaluate_kmeans_prefix_tuning_grid(
            args.topology,
            args.hdf5_path,
            args.partition_artifact,
            args.dataset,
            args.logical_shards,
            args.collection,
            args.ef_grid,
            args.tuning_query_count,
            query_concurrency=args.query_concurrency,
            request_workers=args.request_workers,
            require_worker_cpu_time=args.require_worker_cpu_time,
            require_benchmark_affinity=args.require_benchmark_affinity,
            experiment_id=args.experiment_id,
            manifest_jsonl=args.manifest_jsonl,
        )
    else:
        candidates = []
        fanouts = [args.logical_shards] if args.method == "random" else list(
            range(1, args.logical_shards + 1)
        )
        for fanout in fanouts:
            for ef_search in args.ef_grid:
                _rows, summary = evaluate_queries(
                    args.topology,
                    args.hdf5_path,
                    args.partition_artifact,
                    args.dataset,
                    args.method,
                    args.logical_shards,
                    args.collection,
                    fanout,
                    ef_search,
                    0,
                    args.tuning_query_count,
                    query_concurrency=args.query_concurrency,
                    request_workers=args.request_workers,
                    require_worker_cpu_time=args.require_worker_cpu_time,
                    require_benchmark_affinity=args.require_benchmark_affinity,
                )
                candidate = {
                    "experiment_id": args.experiment_id,
                    "record_type": "tuning_candidate",
                    "timestamp": utc_timestamp(),
                    "git_commit": repository_commit(Path(__file__).resolve().parents[3]),
                    "dataset": args.dataset,
                    "dataset_checksum": sha256_path(args.hdf5_path),
                    "partition_method": args.method,
                    "partition_seed": PARTITION_SEED,
                    "logical_shard_count": args.logical_shards,
                    "physical_machine_count": min(args.logical_shards, 4),
                    "fanout": fanout,
                    "ef_search": ef_search,
                    "recall_at_10": summary["achieved_recall"],
                    "mean_aggregate_distance_computations": summary[
                        "mean_distance_computations_per_query"
                    ],
                    "mean_nodes_visited_per_query": summary[
                        "mean_nodes_visited_per_query"
                    ],
                    "mean_worker_cpu_time_us_per_query": summary[
                        "mean_worker_cpu_time_us_per_query"
                    ],
                    "query_count": summary["query_count"],
                    "target_recall": TARGET_RECALL,
                    "status": summary["result_status"],
                    "benchmark_cpu_affinity": summary["benchmark_cpu_affinity"],
                    "query_concurrency": summary["query_concurrency"],
                    "request_workers": summary["request_workers"],
                }
                candidate.update(protocol_metadata_from_query_summary(summary))
                candidates.append(candidate)
                append_jsonl(args.manifest_jsonl, candidate)
    selected = (
        select_random_tuning_candidate(candidates)
        if args.method == "random"
        else select_kmeans_tuning_candidate(candidates)
    )
    output = {"candidates": candidates, "selected": selected}
    if args.kmeans_prefix_reuse:
        output["tuning_execution"] = {
            "mode": "exact_ranked_prefix_reuse",
            "candidate_count": len(candidates),
            "shard_search_requests": (
                args.tuning_query_count * args.logical_shards * len(args.ef_grid)
            ),
        }
    write_json_atomic(args.output, output)
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


def common_configuration_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--topology", default="experiments/c1/topology-amd-4node.json")
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--hdf5-path", required=True)
    parser.add_argument("--method", choices=("random", "kmeans"), required=True)
    parser.add_argument("--logical-shards", type=int, choices=(1, 2, 4, 8, 16, 32), required=True)
    parser.add_argument("--partition-artifact", required=True)
    parser.add_argument("--collection", required=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute C1 Random/K-Means graph ANN baselines")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    common_configuration_arguments(prepare)
    prepare.add_argument("--upload-batch-size", type=int, default=256)
    prepare.add_argument("--replace", action="store_true")
    prepare.add_argument("--output", required=True)
    prepare.set_defaults(func=command_prepare)

    measure = subparsers.add_parser("measure")
    common_configuration_arguments(measure)
    measure.add_argument("--fanout", type=int, required=True)
    measure.add_argument("--ef-search", type=int, required=True)
    measure.add_argument("--query-start", type=int, required=True)
    measure.add_argument("--query-count", type=int, required=True)
    measure.add_argument("--query-concurrency", type=int, default=16)
    measure.add_argument("--request-workers", type=int, default=128)
    measure.add_argument("--require-worker-cpu-time", action="store_true")
    measure.add_argument("--require-benchmark-affinity", action="store_true")
    measure.add_argument("--per-query-output", required=True)
    measure.add_argument("--summary-output", required=True)
    measure.set_defaults(func=command_measure)

    e3_measure = subparsers.add_parser("e3-measure")
    common_configuration_arguments(e3_measure)
    e3_measure.add_argument("--fanout", type=int, required=True)
    e3_measure.add_argument("--ef-search", type=int, required=True)
    e3_measure.add_argument("--selected-tuning-recall", type=float, required=True)
    e3_measure.add_argument("--query-start", type=int, required=True)
    e3_measure.add_argument("--query-count", type=int, required=True)
    e3_measure.add_argument("--require-worker-cpu-time", action="store_true")
    e3_measure.add_argument("--require-benchmark-affinity", action="store_true")
    e3_measure.add_argument("--controller-storage-root")
    e3_measure.add_argument("--run-id")
    e3_measure.add_argument("--experiment-id", required=True)
    e3_measure.add_argument(
        "--manifest-jsonl", default="experiments/c1/runs/manifest.jsonl"
    )
    e3_measure.add_argument("--per-search-output", required=True)
    e3_measure.add_argument("--summary-output", required=True)
    e3_measure.set_defaults(func=command_e3_measure)

    tune = subparsers.add_parser("tune")
    common_configuration_arguments(tune)
    tune.add_argument("--experiment-id", required=True)
    tune.add_argument("--ef-grid", type=parse_int_grid, default=list(DEFAULT_EF_GRID))
    tune.add_argument("--tuning-query-count", type=int, default=TUNING_QUERY_COUNT)
    tune.add_argument("--query-concurrency", type=int, default=16)
    tune.add_argument("--request-workers", type=int, default=128)
    tune.add_argument("--require-worker-cpu-time", action="store_true")
    tune.add_argument("--require-benchmark-affinity", action="store_true")
    tune.add_argument("--kmeans-prefix-reuse", action="store_true")
    tune.add_argument("--manifest-jsonl", default="experiments/c1/runs/manifest.jsonl")
    tune.add_argument("--output", required=True)
    tune.set_defaults(func=command_tune)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    positive = {
        name: getattr(args, name)
        for name in (
            "upload_batch_size",
            "fanout",
            "ef_search",
            "query_count",
            "query_concurrency",
            "request_workers",
            "tuning_query_count",
        )
        if hasattr(args, name)
    }
    invalid = {name: value for name, value in positive.items() if value <= 0}
    if invalid:
        raise ValueError(f"positive integer arguments required: {invalid}")
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
