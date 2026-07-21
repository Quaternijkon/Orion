#!/usr/bin/env python3
"""Validate and summarize a native Orion compact-transport chunk sweep.

The sweep root must contain the seven fixed arms::

    enabled-all/{benchmark/{summary.json,stability_runs.csv,run_manifest.json},transport-probe.json}
    enabled-{16,8,4,2,1}/{...}
    disabled-all/{...}

The analyzer is deliberately strict and specific to the fixed four-node
GloVe-200 protocol: it validates benchmark and probe manifests, placement and
cluster health, proves benchmark recall and fixed-probe result equivalence,
validates the expected gRPC method/status deltas, reports P95/P99 batch latency,
enforces a 5% QPS-CV ceiling, and only then creates a new output directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
import struct
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit


ARM_SETTINGS = (
    ("enabled-all", "enabled", "all"),
    ("enabled-16", "enabled", "16"),
    ("enabled-8", "enabled", "8"),
    ("enabled-4", "enabled", "4"),
    ("enabled-2", "enabled", "2"),
    ("enabled-1", "enabled", "1"),
    ("disabled-all", "disabled", "all"),
)
ENABLED_ARMS = tuple(name for name, mode, _chunk in ARM_SETTINGS if mode == "enabled")
DISABLED_ARM = "disabled-all"
QPS_CV_LIMIT = 0.05
TOP_TWO_TIE_LIMIT = 0.02
BENCHMARK_SCHEMA_VERSION = 1
PROBE_SCHEMA_VERSION = 2
BENCHMARK_WARMUP_QUERY_COUNT = 500
BENCHMARK_QUERY_COUNT = 1000
PROBE_WARMUP_QUERY_COUNT = 500
PROBE_QUERY_COUNT = 200
TOP_K = 10
BATCH_SIZE = 200
STABILITY_REPEATS = 3
EXPECTED_METHOD = "orion"
EXPECTED_API = "search"
EXPECTED_SHARD_COUNT = 46
EXPECTED_REPLICATION_FACTOR = 1
EXPECTED_WORKER_COUNT = 3
EXPECTED_DISABLED_RPC_COUNTS = (15, 16, 15)
EXPECTED_CONTROLLER_IP = "10.10.1.1"
EXPECTED_WORKER_IPS = ("10.10.1.2", "10.10.1.3", "10.10.1.4")
EXPECTED_CONTROLLER_HTTP_URL = f"http://{EXPECTED_CONTROLLER_IP}:6333"
EXPECTED_WORKER_HTTP_URLS = tuple(
    f"http://{private_ip}:6333" for private_ip in EXPECTED_WORKER_IPS
)
EXPECTED_PEER_URIS = (
    f"http://{EXPECTED_CONTROLLER_IP}:6335",
    *(f"http://{private_ip}:6335" for private_ip in EXPECTED_WORKER_IPS),
)
EXPECTED_PROCESS_AFFINITY = tuple(range(8, 20))
EXPECTED_REQUEST_CONTRACT = {
    "standard_coordinator_request": True,
    "shard_selector": False,
    "entry_point_hints": False,
    "per_shard_ef": False,
    "source_id_hint": False,
    "scalar_hnsw_ef": None,
}
EXPECTED_PROBE_REQUEST_CONTRACT = (
    "ordinary coordinator batch; no shard selector, entry points, per-shard EF, "
    "or source-ID hint"
)
EXPECTED_DATASET_SHA256 = (
    "4839085e5a8bb293434a1a66e1aa0193afc3f07c6797a85f1dbd91656172da20"
)
EXPECTED_DATASET_SIZE_BYTES = 962_819_488
EXPECTED_DATASET_SHAPES = {
    "train": (1_183_514, 200),
    "test": (10_000, 200),
    "neighbors": (10_000, 100),
}
CORE_SEARCH_BATCH = "/qdrant.PointsInternal/CoreSearchBatch"
LEGACY_BY_SHARD = "/qdrant.PointsInternal/CoreSearchBatchByShard"
COMPACT_BY_SHARD = "/qdrant.PointsInternal/CoreSearchBatchByShardCompact"
TELEMETRY_METHODS = (CORE_SEARCH_BATCH, LEGACY_BY_SHARD, COMPACT_BY_SHARD)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
F32_HEX_RE = re.compile(r"^[0-9a-f]{8}$")
GIT_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
IMAGE_ID_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "sweep_root",
        help="Completed chunk-sweep root containing all seven fixed arms.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="New directory to create for the three summary artifacts.",
    )
    return parser.parse_args(argv)


def require_mapping(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError(f"{context} must be a JSON object")
    return value


def require_list(value: Any, context: str) -> list[Any]:
    if not isinstance(value, list):
        raise RuntimeError(f"{context} must be a JSON array")
    return value


def require_int(value: Any, context: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool):
        raise RuntimeError(f"{context} must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{context} must be an integer") from exc
    if str(value).strip() != str(parsed):
        raise RuntimeError(f"{context} must be an integer")
    if parsed < minimum:
        raise RuntimeError(f"{context} must be at least {minimum}")
    return parsed


def require_float(value: Any, context: str, *, positive: bool = False) -> float:
    if isinstance(value, bool):
        raise RuntimeError(f"{context} must be a finite number")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{context} must be a finite number") from exc
    if not math.isfinite(parsed):
        raise RuntimeError(f"{context} must be a finite number")
    if positive and parsed <= 0:
        raise RuntimeError(f"{context} must be positive")
    return parsed


def require_sha256(value: Any, context: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise RuntimeError(f"{context} must be a lowercase SHA-256 digest")
    return value


def require_string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"{context} must be a non-empty string")
    return value


def require_exact(value: Any, expected: Any, context: str) -> Any:
    if value != expected:
        raise RuntimeError(f"{context} must be {expected!r}, got {value!r}")
    return value


def require_shape(value: Any, context: str) -> tuple[int, int]:
    shape = require_list(value, context)
    if len(shape) != 2:
        raise RuntimeError(f"{context} must contain exactly two dimensions")
    return (
        require_int(shape[0], f"{context}[0]", minimum=1),
        require_int(shape[1], f"{context}[1]", minimum=1),
    )


def require_git_commit(value: Any, context: str) -> str:
    if not isinstance(value, str) or GIT_COMMIT_RE.fullmatch(value) is None:
        raise RuntimeError(f"{context} must be a lowercase 40-character Git commit")
    return value


def require_image_id(value: Any, context: str) -> str:
    if not isinstance(value, str) or IMAGE_ID_RE.fullmatch(value) is None:
        raise RuntimeError(f"{context} must be a sha256: Docker image ID")
    return value


def normalized_url(value: Any, context: str, *, port: int) -> str:
    raw = require_string(value, context).rstrip("/")
    parsed = urlsplit(raw)
    if parsed.scheme != "http" or not parsed.hostname or parsed.port != port:
        raise RuntimeError(f"{context} must be an http URL on port {port}")
    if parsed.path or parsed.query or parsed.fragment or parsed.username:
        raise RuntimeError(f"{context} must contain only scheme, host, and port")
    return f"http://{parsed.hostname}:{port}"


def validate_dataset(
    value: Any,
    context: str,
    *,
    shape_fields: dict[str, str] | None = None,
) -> dict[str, Any]:
    dataset = require_mapping(value, context)
    path = require_string(dataset.get("path"), f"{context} path")
    sha256 = require_sha256(dataset.get("sha256"), f"{context} sha256")
    require_exact(sha256, EXPECTED_DATASET_SHA256, f"{context} sha256")
    size_bytes = require_int(
        dataset.get("size_bytes"), f"{context} size_bytes", minimum=1
    )
    require_exact(
        size_bytes, EXPECTED_DATASET_SIZE_BYTES, f"{context} size_bytes"
    )
    fields = shape_fields or {
        "train": "train_shape",
        "test": "test_shape",
        "neighbors": "neighbors_shape",
    }
    shape_source: dict[str, Any]
    if "hdf5_shapes" in dataset:
        shape_source = require_mapping(
            dataset.get("hdf5_shapes"), f"{context} hdf5_shapes"
        )
    else:
        shape_source = dataset
    shapes: dict[str, tuple[int, int]] = {}
    for name, field in fields.items():
        shape = require_shape(shape_source.get(field), f"{context} {field}")
        require_exact(shape, EXPECTED_DATASET_SHAPES[name], f"{context} {field}")
        shapes[name] = shape
    return {
        "path": path,
        "sha256": sha256,
        "size_bytes": size_bytes,
        "shapes": shapes,
    }


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"required input not found: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"invalid JSON in {path}: {exc}") from exc
    return require_mapping(value, str(path))


def load_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"required input not found: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise RuntimeError(f"CSV has no header: {path}")
        rows = list(reader)
    if not rows:
        raise RuntimeError(f"CSV has no data rows: {path}")
    return rows


def parse_shard_placement(value: Any, context: str) -> dict[int, int]:
    raw = require_mapping(value, context)
    placement: dict[int, int] = {}
    for raw_shard_id, raw_peer_id in raw.items():
        shard_id = require_int(raw_shard_id, f"{context} shard ID")
        peer_id = require_int(raw_peer_id, f"{context} shard {shard_id} peer ID")
        if shard_id in placement:
            raise RuntimeError(f"{context} contains duplicate shard ID {shard_id}")
        placement[shard_id] = peer_id
    if set(placement) != set(range(EXPECTED_SHARD_COUNT)):
        raise RuntimeError(
            f"{context} must contain exactly shards 0..{EXPECTED_SHARD_COUNT - 1}"
        )
    return placement


def validate_benchmark_manifest(
    arm: str,
    mode: str,
    chunk: str,
    benchmark_root: Path,
    *,
    collection: str,
    method: str,
    api: str,
) -> dict[str, Any]:
    manifest_path = benchmark_root / "run_manifest.json"
    manifest = load_json(manifest_path)
    require_exact(
        manifest.get("schema_version"),
        BENCHMARK_SCHEMA_VERSION,
        f"{arm} benchmark manifest schema_version",
    )
    require_exact(manifest.get("method"), method, f"{arm} manifest method")
    require_exact(manifest.get("api"), api, f"{arm} manifest API")
    require_exact(
        manifest.get("collection"), collection, f"{arm} manifest collection"
    )
    base_url = normalized_url(
        manifest.get("base_url"), f"{arm} manifest base_url", port=6333
    )
    require_exact(
        base_url, EXPECTED_CONTROLLER_HTTP_URL, f"{arm} manifest base_url"
    )
    require_exact(
        manifest.get("request_contract"),
        EXPECTED_REQUEST_CONTRACT,
        f"{arm} benchmark request contract",
    )
    affinity = tuple(
        require_int(cpu, f"{arm} process_affinity CPU")
        for cpu in require_list(
            manifest.get("process_affinity"), f"{arm} process_affinity"
        )
    )
    require_exact(
        affinity, EXPECTED_PROCESS_AFFINITY, f"{arm} benchmark process_affinity"
    )
    dataset = validate_dataset(manifest.get("dataset"), f"{arm} benchmark dataset")

    parameters = require_mapping(manifest.get("parameters"), f"{arm} parameters")
    required_parameters = {
        "method": EXPECTED_METHOD,
        "api": EXPECTED_API,
        "base_url": EXPECTED_CONTROLLER_HTTP_URL,
        "collection": collection,
        "warmup_query_count": BENCHMARK_WARMUP_QUERY_COUNT,
        "eval_query_count": BENCHMARK_QUERY_COUNT,
        "top_k": TOP_K,
        "batch_size": BATCH_SIZE,
        "stability_repeats": STABILITY_REPEATS,
        "vector_distance": "cosine",
        "vector_name": "",
        "hnsw_ef": None,
        "orion_route_trace": False,
        "write_per_query_metrics": True,
    }
    for field, expected in required_parameters.items():
        require_exact(
            parameters.get(field), expected, f"{arm} benchmark parameter {field}"
        )

    repository = require_mapping(manifest.get("repository"), f"{arm} repository")
    repository_binding = require_mapping(
        manifest.get("repository_binding"), f"{arm} repository_binding"
    )
    commit = require_git_commit(
        repository.get("commit"), f"{arm} repository commit"
    )
    require_exact(
        repository.get("tracked_dirty"), False, f"{arm} repository tracked_dirty"
    )
    require_exact(
        repository_binding.get("tracked_dirty"),
        False,
        f"{arm} repository binding tracked_dirty",
    )
    require_exact(
        repository_binding.get("benchmark_commit"),
        commit,
        f"{arm} benchmark commit binding",
    )
    require_exact(
        repository_binding.get("deployment_commit"),
        commit,
        f"{arm} deployment commit binding",
    )

    deployment = require_mapping(manifest.get("deployment"), f"{arm} deployment")
    deployment_manifest_path = require_string(
        deployment.get("path"), f"{arm} deployment manifest path"
    )
    deployment_repository = require_mapping(
        deployment.get("repository"), f"{arm} deployment repository"
    )
    require_exact(
        deployment_repository.get("commit"),
        commit,
        f"{arm} deployment repository commit",
    )
    require_exact(
        deployment_repository.get("tracked_dirty"),
        False,
        f"{arm} deployment repository tracked_dirty",
    )
    image = require_mapping(deployment.get("image"), f"{arm} deployment image")
    image_id = require_image_id(image.get("id"), f"{arm} deployment image ID")
    image_tag = require_string(image.get("tag"), f"{arm} deployment image tag")

    nodes = require_list(deployment.get("nodes"), f"{arm} deployment nodes")
    if len(nodes) != EXPECTED_WORKER_COUNT + 1:
        raise RuntimeError(f"{arm} deployment must contain one controller and 3 workers")
    nodes_by_role: dict[str, dict[str, Any]] = {}
    for raw_node in nodes:
        node = require_mapping(raw_node, f"{arm} deployment node")
        role = require_string(node.get("role"), f"{arm} deployment node role")
        if role in nodes_by_role:
            raise RuntimeError(f"{arm} deployment contains duplicate role {role}")
        nodes_by_role[role] = node
        require_exact(
            node.get("image_id"), image_id, f"{arm} {role} deployment image ID"
        )
    expected_roles = {"controller", *(f"qdrant_shard_{i}" for i in range(1, 4))}
    if set(nodes_by_role) != expected_roles:
        raise RuntimeError(f"{arm} deployment node roles differ from fixed topology")
    controller_node = nodes_by_role["controller"]
    for field, expected in {
        "private_ip": EXPECTED_CONTROLLER_IP,
        "cpuset": "0-7",
        "peer_premerge_mode": mode,
        "peer_premerge_shards_per_rpc": chunk,
    }.items():
        require_exact(
            controller_node.get(field), expected, f"{arm} controller node {field}"
        )
    for index, private_ip in enumerate(EXPECTED_WORKER_IPS, start=1):
        role = f"qdrant_shard_{index}"
        node = nodes_by_role[role]
        for field, expected in {
            "private_ip": private_ip,
            "cpuset": "0-19",
            "peer_premerge_mode": "not_applicable",
            "peer_premerge_shards_per_rpc": "not_applicable",
        }.items():
            require_exact(node.get(field), expected, f"{arm} {role} node {field}")

    topology = require_mapping(manifest.get("topology"), f"{arm} topology")
    require_exact(
        topology.get("benchmark_client_cpuset"),
        "8-19",
        f"{arm} benchmark client cpuset",
    )
    require_exact(
        normalized_url(
            topology.get("controller_uri"), f"{arm} controller URI", port=6335
        ),
        EXPECTED_PEER_URIS[0],
        f"{arm} controller URI",
    )
    worker_uris = tuple(
        normalized_url(uri, f"{arm} worker URI", port=6335)
        for uri in require_list(topology.get("worker_uris"), f"{arm} worker URIs")
    )
    require_exact(worker_uris, EXPECTED_PEER_URIS[1:], f"{arm} worker URIs")
    topology_workers = require_list(topology.get("workers"), f"{arm} topology workers")
    if len(topology_workers) != EXPECTED_WORKER_COUNT:
        raise RuntimeError(f"{arm} topology must contain exactly 3 workers")
    for index, (raw_worker, private_ip) in enumerate(
        zip(topology_workers, EXPECTED_WORKER_IPS, strict=True), start=1
    ):
        worker = require_mapping(raw_worker, f"{arm} topology worker {index}")
        require_exact(
            worker.get("private_ip"), private_ip, f"{arm} topology worker {index} IP"
        )
        require_exact(
            worker.get("cpuset"), "0-19", f"{arm} topology worker {index} cpuset"
        )

    cluster = require_mapping(
        manifest.get("cluster_preflight"), f"{arm} cluster_preflight"
    )
    require_exact(cluster.get("peer_count"), 4, f"{arm} cluster peer_count")
    require_exact(
        cluster.get("consensus_thread_status"),
        "working",
        f"{arm} consensus thread status",
    )
    require_exact(
        cluster.get("pending_operations"), 0, f"{arm} pending operations"
    )
    require_exact(
        cluster.get("message_send_failures"), {}, f"{arm} message send failures"
    )
    controller_peer_id = require_int(
        cluster.get("controller_peer_id"), f"{arm} controller peer ID", minimum=1
    )
    require_exact(
        cluster.get("peer_id"), controller_peer_id, f"{arm} cluster peer ID"
    )
    worker_peer_ids = tuple(
        require_int(value, f"{arm} worker peer ID", minimum=1)
        for value in require_list(cluster.get("worker_peer_ids"), f"{arm} worker peers")
    )
    if len(worker_peer_ids) != EXPECTED_WORKER_COUNT or len(set(worker_peer_ids)) != 3:
        raise RuntimeError(f"{arm} must contain exactly 3 unique worker peer IDs")
    if controller_peer_id in worker_peer_ids:
        raise RuntimeError(f"{arm} controller peer ID appears in worker peer IDs")
    raw_peers = require_mapping(cluster.get("peers"), f"{arm} cluster peers")
    peers: dict[int, str] = {
        require_int(raw_peer_id, f"{arm} cluster peer ID", minimum=1): normalized_url(
            uri, f"{arm} cluster peer URI", port=6335
        )
        for raw_peer_id, uri in raw_peers.items()
    }
    if set(peers) != {controller_peer_id, *worker_peer_ids}:
        raise RuntimeError(f"{arm} cluster peer map differs from controller/workers")
    require_exact(
        peers[controller_peer_id], EXPECTED_PEER_URIS[0], f"{arm} controller peer URI"
    )
    if {peers[peer_id] for peer_id in worker_peer_ids} != set(EXPECTED_PEER_URIS[1:]):
        raise RuntimeError(f"{arm} worker peer URIs differ from fixed topology")

    collection_cluster = require_mapping(
        manifest.get("collection_cluster"), f"{arm} collection_cluster"
    )
    require_exact(
        collection_cluster.get("peer_id"),
        controller_peer_id,
        f"{arm} collection controller peer ID",
    )
    require_exact(
        collection_cluster.get("shard_count"),
        EXPECTED_SHARD_COUNT,
        f"{arm} collection shard_count",
    )
    require_exact(
        collection_cluster.get("local_shards"),
        [],
        f"{arm} controller local shards",
    )
    require_exact(
        collection_cluster.get("shard_transfers"),
        [],
        f"{arm} collection shard transfers",
    )
    remote_shards = require_list(
        collection_cluster.get("remote_shards"), f"{arm} remote shards"
    )
    if len(remote_shards) != EXPECTED_SHARD_COUNT:
        raise RuntimeError(f"{arm} must contain exactly 46 remote shards")
    remote_placement: dict[int, int] = {}
    for raw_shard in remote_shards:
        shard = require_mapping(raw_shard, f"{arm} remote shard")
        shard_id = require_int(shard.get("shard_id"), f"{arm} remote shard ID")
        peer_id = require_int(
            shard.get("peer_id"), f"{arm} remote shard {shard_id} peer ID", minimum=1
        )
        require_exact(
            shard.get("state"), "Active", f"{arm} remote shard {shard_id} state"
        )
        if peer_id not in worker_peer_ids:
            raise RuntimeError(f"{arm} remote shard {shard_id} is not on a worker")
        if shard_id in remote_placement:
            raise RuntimeError(f"{arm} remote shard {shard_id} has duplicate replicas")
        remote_placement[shard_id] = peer_id
    if set(remote_placement) != set(range(EXPECTED_SHARD_COUNT)):
        raise RuntimeError(f"{arm} remote shard IDs must be exactly 0..45")

    placement_proof = require_mapping(
        manifest.get("placement_proof"), f"{arm} placement_proof"
    )
    require_exact(placement_proof.get("valid"), True, f"{arm} placement valid")
    require_exact(
        placement_proof.get("controller_peer_id"),
        controller_peer_id,
        f"{arm} placement controller peer ID",
    )
    require_exact(
        placement_proof.get("shard_count"),
        EXPECTED_SHARD_COUNT,
        f"{arm} placement shard_count",
    )
    require_exact(
        placement_proof.get("replication_factor"),
        EXPECTED_REPLICATION_FACTOR,
        f"{arm} placement replication_factor",
    )
    require_exact(
        placement_proof.get("shard_transfers"),
        [],
        f"{arm} placement shard transfers",
    )
    placement = parse_shard_placement(
        placement_proof.get("placement"), f"{arm} placement"
    )
    expected_placement = parse_shard_placement(
        placement_proof.get("expected_placement"), f"{arm} expected placement"
    )
    if placement != expected_placement or placement != remote_placement:
        raise RuntimeError(f"{arm} placement proof differs from active remote shards")
    raw_counts = require_mapping(
        placement_proof.get("shards_per_worker"), f"{arm} shards_per_worker"
    )
    declared_counts = {
        require_int(raw_peer_id, f"{arm} placement worker peer ID", minimum=1):
        require_int(raw_count, f"{arm} placement worker shard count", minimum=1)
        for raw_peer_id, raw_count in raw_counts.items()
    }
    actual_counts = {
        peer_id: sum(assigned == peer_id for assigned in placement.values())
        for peer_id in worker_peer_ids
    }
    if declared_counts != actual_counts or sorted(actual_counts.values()) != [15, 15, 16]:
        raise RuntimeError(f"{arm} worker shard counts must be a 15/15/16 split")

    collection_info = require_mapping(
        manifest.get("collection_info"), f"{arm} collection_info"
    )
    config = require_mapping(collection_info.get("config"), f"{arm} collection config")
    params = require_mapping(config.get("params"), f"{arm} collection params")
    require_exact(
        params.get("shard_number"), EXPECTED_SHARD_COUNT, f"{arm} shard_number"
    )
    require_exact(
        params.get("replication_factor"),
        EXPECTED_REPLICATION_FACTOR,
        f"{arm} replication_factor",
    )
    require_exact(params.get("sharding_method"), "auto", f"{arm} sharding_method")
    vectors = require_mapping(params.get("vectors"), f"{arm} vector schema")
    require_exact(vectors.get("size"), 200, f"{arm} vector dimension")
    require_exact(vectors.get("distance"), "Cosine", f"{arm} vector distance")
    require_exact(collection_info.get("status"), "green", f"{arm} collection status")
    require_exact(
        collection_info.get("optimizer_status"), "ok", f"{arm} optimizer status"
    )
    update_queue = require_mapping(
        collection_info.get("update_queue"), f"{arm} update_queue"
    )
    require_exact(update_queue.get("length"), 0, f"{arm} update queue length")
    points_count = require_int(
        collection_info.get("points_count"), f"{arm} points_count", minimum=1
    )
    require_exact(
        collection_info.get("indexed_vectors_count"),
        points_count,
        f"{arm} indexed vector count",
    )

    auto_policy = require_mapping(
        config.get("auto_shard_policy"), f"{arm} auto shard policy"
    )
    require_exact(auto_policy.get("type"), "orion", f"{arm} auto shard policy type")
    artifact_sha256 = require_sha256(
        auto_policy.get("artifact_sha256"), f"{arm} auto shard artifact SHA-256"
    )
    generation = require_int(
        auto_policy.get("generation"), f"{arm} auto shard generation", minimum=1
    )
    metadata = require_mapping(config.get("metadata"), f"{arm} collection metadata")
    prepare = require_mapping(
        metadata.get("native_auto_shard_prepare"), f"{arm} prepare provenance envelope"
    )
    require_exact(
        prepare.get("schema_version"), 2, f"{arm} prepare provenance schema_version"
    )
    require_sha256(
        prepare.get("provenance_sha256"), f"{arm} prepare provenance SHA-256"
    )
    provenance = require_mapping(
        prepare.get("provenance"), f"{arm} collection provenance"
    )
    require_exact(provenance.get("method"), "orion", f"{arm} provenance method")
    require_exact(
        provenance.get("shard_count"), EXPECTED_SHARD_COUNT, f"{arm} provenance shards"
    )
    require_exact(
        provenance.get("physical_point_count"),
        points_count,
        f"{arm} provenance physical points",
    )
    provenance_schema = require_mapping(
        provenance.get("vector_schema"), f"{arm} provenance vector schema"
    )
    require_exact(provenance_schema.get("dimension"), 200, f"{arm} provenance dimension")
    require_exact(provenance_schema.get("distance"), "Cosine", f"{arm} provenance distance")
    require_exact(provenance_schema.get("vector_name"), "", f"{arm} provenance vector name")

    live_policy = require_mapping(manifest.get("live_policy"), f"{arm} live policy")
    require_exact(live_policy.get("type"), "orion", f"{arm} live policy type")
    require_exact(
        live_policy.get("artifact_sha256"), artifact_sha256, f"{arm} live artifact SHA"
    )
    require_exact(live_policy.get("generation"), generation, f"{arm} live generation")
    artifact = require_mapping(manifest.get("artifact"), f"{arm} artifact proof")
    require_exact(artifact.get("status"), "verified", f"{arm} artifact status")
    require_exact(artifact.get("sha256"), artifact_sha256, f"{arm} artifact SHA")
    require_exact(artifact.get("generation"), generation, f"{arm} artifact generation")
    require_exact(
        artifact.get("shard_count"), EXPECTED_SHARD_COUNT, f"{arm} artifact shards"
    )
    require_exact(
        artifact.get("physical_point_count"), points_count, f"{arm} artifact points"
    )

    readiness = require_mapping(
        manifest.get("indexing_readiness"), f"{arm} indexing readiness"
    )
    require_exact(readiness.get("fully_indexed"), True, f"{arm} fully indexed")
    require_exact(
        readiness.get("completion_mode"), "fully_indexed", f"{arm} completion mode"
    )
    require_exact(readiness.get("status"), "green", f"{arm} readiness status")
    require_exact(
        readiness.get("optimizer_status"), "ok", f"{arm} readiness optimizer"
    )
    require_exact(readiness.get("points_count"), points_count, f"{arm} readiness points")
    require_exact(
        readiness.get("indexed_vectors_count"), points_count, f"{arm} readiness indexed"
    )
    require_exact(
        readiness.get("shard_transfers"), [], f"{arm} readiness shard transfers"
    )
    readiness_queue = require_mapping(
        readiness.get("update_queue"), f"{arm} readiness update queue"
    )
    require_exact(readiness_queue.get("length"), 0, f"{arm} readiness queue length")

    peer_id_by_uri = {uri: peer_id for peer_id, uri in peers.items()}
    placement_by_worker_url: dict[str, int] = {}
    for worker_http_url, worker_peer_uri in zip(
        EXPECTED_WORKER_HTTP_URLS, EXPECTED_PEER_URIS[1:], strict=True
    ):
        worker_peer_id = peer_id_by_uri[worker_peer_uri]
        placement_by_worker_url[worker_http_url] = actual_counts[worker_peer_id]
    require_exact(
        tuple(placement_by_worker_url.values()),
        EXPECTED_DISABLED_RPC_COUNTS,
        f"{arm} ordered worker shard split",
    )
    return {
        "manifest_path": str(manifest_path),
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "base_url": base_url,
        "dataset": dataset,
        "commit": commit,
        "image_tag": image_tag,
        "image_id": image_id,
        "deployment_manifest_path": deployment_manifest_path,
        "worker_urls": list(EXPECTED_WORKER_HTTP_URLS),
        "controller_peer_id": controller_peer_id,
        "worker_peer_ids": list(worker_peer_ids),
        "placement": placement,
        "placement_by_worker_url": placement_by_worker_url,
        "artifact_sha256": artifact_sha256,
        "generation": generation,
        "points_count": points_count,
    }


def validate_benchmark(
    arm: str, mode: str, chunk: str, arm_root: Path
) -> dict[str, Any]:
    benchmark_root = arm_root / "benchmark"
    summary_path = benchmark_root / "summary.json"
    stability_path = benchmark_root / "stability_runs.csv"
    summary = load_json(summary_path)
    rows = load_csv(stability_path)
    final = require_mapping(summary.get("final_metrics"), f"{arm} final_metrics")

    method = require_string(final.get("method"), f"{arm} benchmark method")
    api = require_string(final.get("api"), f"{arm} benchmark API")
    require_exact(method, EXPECTED_METHOD, f"{arm} benchmark method")
    require_exact(api, EXPECTED_API, f"{arm} benchmark API")
    query_count = require_int(
        final.get("query_count"), f"{arm} benchmark query_count", minimum=1
    )
    top_k = require_int(final.get("top_k"), f"{arm} benchmark top_k", minimum=1)
    batch_size = require_int(
        final.get("batch_size"), f"{arm} benchmark batch_size", minimum=1
    )
    repeats = require_int(
        final.get("stability_repeats"),
        f"{arm} benchmark stability_repeats",
        minimum=1,
    )
    require_exact(
        query_count, BENCHMARK_QUERY_COUNT, f"{arm} benchmark query_count"
    )
    require_exact(top_k, TOP_K, f"{arm} benchmark top_k")
    require_exact(batch_size, BATCH_SIZE, f"{arm} benchmark batch_size")
    require_exact(repeats, STABILITY_REPEATS, f"{arm} benchmark stability_repeats")
    if repeats != len(rows):
        raise RuntimeError(
            f"{arm} benchmark declares {repeats} repeats but CSV has {len(rows)} rows"
        )

    qps_values: list[float] = []
    recall_values: list[float] = []
    latency_p95_values: list[float] = []
    latency_p99_values: list[float] = []
    run_labels: list[str] = []
    for row_number, row in enumerate(rows, start=2):
        context = f"{arm} stability row {row_number}"
        missing = [
            field
            for field in (
                "run",
                "method",
                "api",
                "query_count",
                "top_k",
                "batch_size",
                "recall_at_k",
                "qps",
                "latency_p95_ms",
                "latency_p99_ms",
            )
            if field not in row
        ]
        if missing:
            raise RuntimeError(f"{context} is missing fields: {', '.join(missing)}")
        if row["method"] != method or row["api"] != api:
            raise RuntimeError(f"{context} method/API differs from summary.json")
        row_contract = (
            require_int(row["query_count"], f"{context} query_count", minimum=1),
            require_int(row["top_k"], f"{context} top_k", minimum=1),
            require_int(row["batch_size"], f"{context} batch_size", minimum=1),
        )
        if row_contract != (query_count, top_k, batch_size):
            raise RuntimeError(f"{context} query/top-k/batch contract differs from summary")
        run_label = str(row["run"])
        if not run_label:
            raise RuntimeError(f"{context} has an empty run label")
        run_labels.append(run_label)
        recall = require_float(row["recall_at_k"], f"{context} recall_at_k")
        if not 0.0 <= recall <= 1.0:
            raise RuntimeError(f"{context} recall_at_k must be in [0, 1]")
        recall_values.append(recall)
        qps_values.append(require_float(row["qps"], f"{context} qps", positive=True))
        latency_p95_values.append(
            require_float(
                row["latency_p95_ms"], f"{context} latency_p95_ms", positive=True
            )
        )
        latency_p99_values.append(
            require_float(
                row["latency_p99_ms"], f"{context} latency_p99_ms", positive=True
            )
        )
    if len(set(run_labels)) != len(run_labels):
        raise RuntimeError(f"{arm} stability run labels must be unique")

    qps_mean = statistics.fmean(qps_values)
    qps_stdev = statistics.stdev(qps_values) if len(qps_values) > 1 else 0.0
    recall_mean = statistics.fmean(recall_values)
    recall_stdev = (
        statistics.stdev(recall_values) if len(recall_values) > 1 else 0.0
    )
    latency_p95_mean = statistics.fmean(latency_p95_values)
    latency_p95_stdev = (
        statistics.stdev(latency_p95_values)
        if len(latency_p95_values) > 1
        else 0.0
    )
    latency_p99_mean = statistics.fmean(latency_p99_values)
    latency_p99_stdev = (
        statistics.stdev(latency_p99_values)
        if len(latency_p99_values) > 1
        else 0.0
    )
    reported = {
        "qps": require_float(final.get("qps"), f"{arm} summary qps", positive=True),
        "qps_stdev": require_float(
            final.get("qps_stdev"), f"{arm} summary qps_stdev"
        ),
        "recall": require_float(
            final.get("recall_at_k"), f"{arm} summary recall_at_k"
        ),
        "recall_stdev": require_float(
            final.get("recall_stdev"), f"{arm} summary recall_stdev"
        ),
        "latency_p95_ms": require_float(
            final.get("latency_p95_ms"),
            f"{arm} summary latency_p95_ms",
            positive=True,
        ),
        "latency_p95_ms_stdev": require_float(
            final.get("latency_p95_ms_stdev"),
            f"{arm} summary latency_p95_ms_stdev",
        ),
        "latency_p99_ms": require_float(
            final.get("latency_p99_ms"),
            f"{arm} summary latency_p99_ms",
            positive=True,
        ),
        "latency_p99_ms_stdev": require_float(
            final.get("latency_p99_ms_stdev"),
            f"{arm} summary latency_p99_ms_stdev",
        ),
    }
    recomputed = {
        "qps": qps_mean,
        "qps_stdev": qps_stdev,
        "recall": recall_mean,
        "recall_stdev": recall_stdev,
        "latency_p95_ms": latency_p95_mean,
        "latency_p95_ms_stdev": latency_p95_stdev,
        "latency_p99_ms": latency_p99_mean,
        "latency_p99_ms_stdev": latency_p99_stdev,
    }
    for field, value in recomputed.items():
        if reported[field] != value:
            raise RuntimeError(
                f"{arm} summary {field}={reported[field]!r} does not match "
                f"stability CSV value {value!r}"
            )
    qps_cv = qps_stdev / qps_mean
    if qps_cv > QPS_CV_LIMIT:
        raise RuntimeError(
            f"{arm} QPS CV {qps_cv:.6f} exceeds the {QPS_CV_LIMIT:.0%} limit"
        )
    collection = require_string(summary.get("collection"), f"{arm} summary collection")
    require_exact(summary.get("method"), method, f"{arm} summary method")
    require_exact(summary.get("stability_runs"), repeats, f"{arm} summary stability_runs")
    manifest = validate_benchmark_manifest(
        arm,
        mode,
        chunk,
        benchmark_root,
        collection=collection,
        method=method,
        api=api,
    )
    return {
        "summary_path": str(summary_path),
        "stability_path": str(stability_path),
        "collection": collection,
        "method": method,
        "api": api,
        "query_count": query_count,
        "top_k": top_k,
        "batch_size": batch_size,
        "repeats": repeats,
        "run_labels": run_labels,
        "recall_values": recall_values,
        "recall_mean": recall_mean,
        "recall_stdev": recall_stdev,
        "qps_values": qps_values,
        "qps_mean": qps_mean,
        "qps_stdev": qps_stdev,
        "qps_cv": qps_cv,
        "latency_p95_values": latency_p95_values,
        "latency_p95_ms": latency_p95_mean,
        "latency_p95_ms_stdev": latency_p95_stdev,
        "latency_p99_values": latency_p99_values,
        "latency_p99_ms": latency_p99_mean,
        "latency_p99_ms_stdev": latency_p99_stdev,
        "manifest": manifest,
    }


def canonical_result_hashes(results: list[Any]) -> tuple[str, str]:
    ids = [[point["id"] for point in row] for row in results]
    encoded_ids = json.dumps(
        ids, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    encoded_results = json.dumps(
        results, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded_ids).hexdigest(), hashlib.sha256(
        encoded_results
    ).hexdigest()


def validate_probe_results(arm: str, probe: dict[str, Any]) -> dict[str, Any]:
    query_count = require_int(
        probe.get("query_count"), f"{arm} probe query_count", minimum=1
    )
    top_k = require_int(probe.get("top_k"), f"{arm} probe top_k", minimum=1)
    require_exact(query_count, PROBE_QUERY_COUNT, f"{arm} probe query_count")
    require_exact(top_k, TOP_K, f"{arm} probe top_k")
    lengths = require_list(probe.get("result_row_lengths"), f"{arm} row lengths")
    parsed_lengths = [
        require_int(value, f"{arm} row length {index}", minimum=0)
        for index, value in enumerate(lengths)
    ]
    if len(parsed_lengths) != query_count:
        raise RuntimeError(
            f"{arm} has {len(parsed_lengths)} result row lengths for {query_count} queries"
        )
    if any(length != top_k for length in parsed_lengths):
        raise RuntimeError(f"{arm} probe does not contain exactly top-k results per query")

    results = require_list(probe.get("results"), f"{arm} probe results")
    if len(results) != query_count:
        raise RuntimeError(f"{arm} probe result row count does not match query_count")
    for query_index, raw_row in enumerate(results):
        row = require_list(raw_row, f"{arm} result row {query_index}")
        if len(row) != top_k or len(row) != parsed_lengths[query_index]:
            raise RuntimeError(
                f"{arm} result row {query_index} does not contain exactly top-k results"
            )
        ids: list[int] = []
        for result_index, raw_point in enumerate(row):
            point = require_mapping(
                raw_point, f"{arm} result {query_index}/{result_index}"
            )
            if set(point) != {"id", "score_f32_le_hex"}:
                raise RuntimeError(
                    f"{arm} result {query_index}/{result_index} has unexpected fields"
                )
            point_id = require_int(
                point["id"], f"{arm} result {query_index}/{result_index} id"
            )
            ids.append(point_id)
            score_hex = point["score_f32_le_hex"]
            if not isinstance(score_hex, str) or F32_HEX_RE.fullmatch(score_hex) is None:
                raise RuntimeError(
                    f"{arm} result {query_index}/{result_index} has invalid float32 bytes"
                )
            score = struct.unpack("<f", bytes.fromhex(score_hex))[0]
            if not math.isfinite(score):
                raise RuntimeError(
                    f"{arm} result {query_index}/{result_index} has non-finite score"
                )
        if len(set(ids)) != len(ids):
            raise RuntimeError(f"{arm} result row {query_index} contains duplicate IDs")

    ids_sha256, ids_scores_sha256 = canonical_result_hashes(results)
    reported_ids = require_sha256(probe.get("ids_sha256"), f"{arm} ids_sha256")
    reported_scores = require_sha256(
        probe.get("ids_scores_sha256"), f"{arm} ids_scores_sha256"
    )
    if ids_sha256 != reported_ids:
        raise RuntimeError(f"{arm} ids_sha256 does not match its stored result rows")
    if ids_scores_sha256 != reported_scores:
        raise RuntimeError(
            f"{arm} ids_scores_sha256 does not match its stored result rows"
        )
    return {
        "query_count": query_count,
        "top_k": top_k,
        "row_lengths": parsed_lengths,
        "ids_sha256": ids_sha256,
        "ids_scores_sha256": ids_scores_sha256,
    }


def parse_telemetry_snapshot(
    arm: str,
    probe: dict[str, Any],
    field: str,
    worker_urls: tuple[str, ...],
) -> dict[str, dict[str, dict[str, int]]]:
    snapshot = require_mapping(probe.get(field), f"{arm} {field}")
    if set(snapshot) != set(worker_urls):
        raise RuntimeError(f"{arm} {field} worker set differs from worker_urls")
    parsed: dict[str, dict[str, dict[str, int]]] = {}
    for worker in worker_urls:
        worker_snapshot = require_mapping(
            snapshot[worker], f"{arm} {field} for {worker}"
        )
        if set(worker_snapshot) != set(TELEMETRY_METHODS):
            raise RuntimeError(f"{arm} {field} methods are incomplete for {worker}")
        parsed[worker] = {}
        for method in TELEMETRY_METHODS:
            raw_status_counts = require_mapping(
                worker_snapshot[method], f"{arm} {field} {worker} {method}"
            )
            status_counts: dict[str, int] = {}
            for status, raw_count in raw_status_counts.items():
                if not isinstance(status, str) or not status.isdigit():
                    raise RuntimeError(
                        f"{arm} {field} {worker} {method} has invalid status {status!r}"
                    )
                status_counts[status] = require_int(
                    raw_count,
                    f"{arm} {field} {worker} {method} status={status}",
                )
            parsed[worker][method] = status_counts
    return parsed


def validate_telemetry(
    arm: str, probe: dict[str, Any], worker_urls: tuple[str, ...]
) -> dict[str, dict[str, int]]:
    methods = require_list(probe.get("telemetry_methods"), f"{arm} telemetry_methods")
    if tuple(methods) != TELEMETRY_METHODS:
        raise RuntimeError(f"{arm} telemetry method list differs from the required set")
    before = parse_telemetry_snapshot(arm, probe, "telemetry_before", worker_urls)
    after = parse_telemetry_snapshot(arm, probe, "telemetry_after", worker_urls)
    delta = parse_telemetry_snapshot(arm, probe, "telemetry_delta", worker_urls)

    counts: dict[str, dict[str, int]] = {}
    for worker in worker_urls:
        counts[worker] = {}
        for method in TELEMETRY_METHODS:
            statuses = set(before[worker][method]) | set(after[worker][method])
            expected_delta: dict[str, int] = {}
            for status in statuses:
                count = after[worker][method].get(status, 0) - before[worker][method].get(
                    status, 0
                )
                if count < 0:
                    raise RuntimeError(
                        f"{arm} telemetry counter decreased for {worker} {method} status={status}"
                    )
                if count:
                    expected_delta[status] = count
            actual_delta = {
                status: count
                for status, count in delta[worker][method].items()
                if count
            }
            for status, count in actual_delta.items():
                if status != "0" and count:
                    raise RuntimeError(
                        f"{arm} telemetry has non-zero gRPC status {status} for {method}"
                    )
            if actual_delta != expected_delta:
                raise RuntimeError(
                    f"{arm} telemetry_delta does not equal telemetry_after-before for "
                    f"{worker} {method}"
                )
            counts[worker][method] = sum(actual_delta.values())
    return counts


def validate_probe_cluster_state(arm: str, value: Any, context: str) -> dict[str, Any]:
    cluster = require_mapping(value, context)
    require_exact(cluster.get("peer_count"), 4, f"{context} peer_count")
    require_exact(
        cluster.get("pending_operations"), 0, f"{context} pending_operations"
    )
    require_exact(
        cluster.get("message_send_failures"), {}, f"{context} message_send_failures"
    )
    consensus = require_mapping(
        cluster.get("consensus_thread_status"), f"{context} consensus_thread_status"
    )
    require_exact(
        consensus.get("consensus_thread_status"),
        "working",
        f"{context} consensus status",
    )
    peer_id = require_int(cluster.get("peer_id"), f"{context} peer_id", minimum=1)
    peer_uris = tuple(
        normalized_url(uri, f"{context} peer URI", port=6335)
        for uri in require_list(cluster.get("peer_uris"), f"{context} peer_uris")
    )
    require_exact(
        peer_uris, EXPECTED_PEER_URIS, f"{context} ordered peer URIs"
    )
    return {"peer_id": peer_id, "peer_uris": peer_uris}


def validate_probe_containers(
    arm: str,
    value: Any,
    context: str,
    *,
    image_tag: str,
    image_id: str,
) -> dict[str, dict[str, Any]]:
    containers = require_mapping(value, context)
    expected_roles = {"controller", *(f"qdrant_shard_{i}" for i in range(1, 4))}
    if set(containers) != expected_roles:
        raise RuntimeError(f"{context} roles differ from fixed topology")
    parsed: dict[str, dict[str, Any]] = {}
    for role, raw_container in containers.items():
        container = require_mapping(raw_container, f"{context} {role}")
        expected_cpuset = "0-7" if role == "controller" else "0-19"
        require_exact(container.get("running"), True, f"{context} {role} running")
        require_exact(
            container.get("network_mode"), "host", f"{context} {role} network mode"
        )
        require_exact(
            container.get("cpuset"), expected_cpuset, f"{context} {role} cpuset"
        )
        require_exact(
            container.get("image_tag"), image_tag, f"{context} {role} image tag"
        )
        require_exact(
            container.get("image_id"), image_id, f"{context} {role} image ID"
        )
        parsed[role] = dict(container)
    return parsed


def validate_probe(
    arm: str, mode: str, chunk: str, arm_root: Path
) -> dict[str, Any]:
    probe_path = arm_root / "transport-probe.json"
    probe = load_json(probe_path)
    require_exact(
        probe.get("schema_version"),
        PROBE_SCHEMA_VERSION,
        f"{arm} probe schema_version",
    )
    run_id = require_string(probe.get("run_id"), f"{arm} probe run_id")
    collection = require_string(probe.get("collection"), f"{arm} probe collection")
    api = require_string(probe.get("api"), f"{arm} probe API")
    require_exact(api, EXPECTED_API, f"{arm} probe API")
    base_url = normalized_url(probe.get("base_url"), f"{arm} probe base_url", port=6333)
    require_exact(base_url, EXPECTED_CONTROLLER_HTTP_URL, f"{arm} probe base_url")
    worker_urls = tuple(
        normalized_url(url, f"{arm} worker URL", port=6333)
        for url in require_list(probe.get("worker_urls"), f"{arm} worker_urls")
    )
    if len(worker_urls) != EXPECTED_WORKER_COUNT or len(set(worker_urls)) != 3:
        raise RuntimeError(f"{arm} probe must contain exactly 3 unique worker URLs")
    require_exact(worker_urls, EXPECTED_WORKER_HTTP_URLS, f"{arm} worker URLs")
    dataset = validate_dataset(
        probe.get("dataset"),
        f"{arm} probe dataset",
        shape_fields={
            "train": "train",
            "test": "test",
            "neighbors": "neighbors",
        },
    )
    deployment = require_mapping(probe.get("deployment"), f"{arm} deployment")
    actual_mode = require_string(
        deployment.get("peer_premerge_mode"), f"{arm} peer premerge mode"
    )
    actual_chunk = require_string(
        deployment.get("peer_premerge_shards_per_rpc"),
        f"{arm} peer premerge shards per RPC",
    )
    if actual_mode != mode or actual_chunk != chunk:
        raise RuntimeError(
            f"{arm} deployment is {actual_mode}/{actual_chunk}, expected {mode}/{chunk}"
        )
    deployment_commit = require_git_commit(
        deployment.get("commit"), f"{arm} probe deployment commit"
    )
    image_id = require_image_id(
        deployment.get("image_id"), f"{arm} probe deployment image ID"
    )
    image_tag = require_string(
        deployment.get("image_tag"), f"{arm} probe deployment image tag"
    )
    deployment_manifest_path = require_string(
        deployment.get("manifest_path"), f"{arm} probe deployment manifest path"
    )
    deployment_manifest_sha256 = require_sha256(
        deployment.get("manifest_sha256"), f"{arm} deployment manifest SHA-256"
    )
    require_string(deployment.get("topology_path"), f"{arm} topology path")
    cluster_before = validate_probe_cluster_state(
        arm, deployment.get("cluster_before"), f"{arm} cluster_before"
    )
    cluster_after = validate_probe_cluster_state(
        arm, deployment.get("cluster_after"), f"{arm} cluster_after"
    )
    if cluster_before != cluster_after:
        raise RuntimeError(f"{arm} cluster identity changed during the probe")
    controller_peer_id_before = require_int(
        deployment.get("controller_peer_id_before"),
        f"{arm} controller_peer_id_before",
        minimum=1,
    )
    controller_peer_id_after = require_int(
        deployment.get("controller_peer_id_after"),
        f"{arm} controller_peer_id_after",
        minimum=1,
    )
    require_exact(
        controller_peer_id_before,
        cluster_before["peer_id"],
        f"{arm} controller peer ID before",
    )
    require_exact(
        controller_peer_id_after,
        controller_peer_id_before,
        f"{arm} controller peer ID after",
    )
    containers_before = validate_probe_containers(
        arm,
        deployment.get("containers_before"),
        f"{arm} containers_before",
        image_tag=image_tag,
        image_id=image_id,
    )
    containers_after = validate_probe_containers(
        arm,
        deployment.get("containers_after"),
        f"{arm} containers_after",
        image_tag=image_tag,
        image_id=image_id,
    )
    if containers_before != containers_after:
        raise RuntimeError(f"{arm} container identity/configuration changed during probe")
    placement_before_sha256 = require_sha256(
        deployment.get("collection_placement_before_sha256"),
        f"{arm} collection placement before SHA-256",
    )
    placement_after_sha256 = require_sha256(
        deployment.get("collection_placement_after_sha256"),
        f"{arm} collection placement after SHA-256",
    )
    require_exact(
        placement_after_sha256,
        placement_before_sha256,
        f"{arm} collection placement hash after probe",
    )

    require_exact(
        probe.get("request_contract"),
        EXPECTED_PROBE_REQUEST_CONTRACT,
        f"{arm} probe request contract",
    )
    require_exact(
        probe.get("vector_distance"), "cosine", f"{arm} probe vector distance"
    )
    require_exact(probe.get("vector_name"), "", f"{arm} probe vector name")
    require_exact(
        probe.get("query_dtype"), "float32-le", f"{arm} probe query dtype"
    )
    require_exact(
        require_int(
            probe.get("query_dimension"), f"{arm} query_dimension", minimum=1
        ),
        200,
        f"{arm} probe query_dimension",
    )
    query_offset = require_int(probe.get("query_offset"), f"{arm} query_offset")
    require_exact(query_offset, 0, f"{arm} probe query_offset")
    batch_size = require_int(
        probe.get("batch_size"), f"{arm} probe batch_size", minimum=1
    )
    require_exact(batch_size, BATCH_SIZE, f"{arm} probe batch_size")
    warmup_count = require_int(
        probe.get("warmup_query_count"), f"{arm} warmup_query_count"
    )
    require_exact(
        warmup_count, PROBE_WARMUP_QUERY_COUNT, f"{arm} probe warmup_query_count"
    )
    result_proof = validate_probe_results(arm, probe)
    require_exact(
        result_proof["query_count"],
        batch_size,
        f"{arm} one-batch probe query_count",
    )
    warmup_sha = require_sha256(
        probe.get("warmup_query_sha256"), f"{arm} warmup_query_sha256"
    )
    query_sha = require_sha256(probe.get("query_sha256"), f"{arm} query_sha256")
    wall_s = require_float(probe.get("wall_s"), f"{arm} probe wall_s", positive=True)
    qps = require_float(probe.get("qps"), f"{arm} probe qps", positive=True)
    expected_qps = result_proof["query_count"] / wall_s
    if not math.isclose(qps, expected_qps, rel_tol=1e-12, abs_tol=0.0):
        raise RuntimeError(f"{arm} probe qps does not equal query_count/wall_s")
    telemetry = validate_telemetry(arm, probe, worker_urls)
    method_totals = {
        method: sum(worker[method] for worker in telemetry.values())
        for method in TELEMETRY_METHODS
    }
    if mode == "enabled":
        if method_totals[COMPACT_BY_SHARD] <= 0:
            raise RuntimeError(f"{arm} did not execute any compact RPCs")
        if method_totals[CORE_SEARCH_BATCH] or method_totals[LEGACY_BY_SHARD]:
            raise RuntimeError(f"{arm} enabled mode used a non-compact search RPC")
    else:
        if method_totals[CORE_SEARCH_BATCH] <= 0:
            raise RuntimeError(f"{arm} did not execute ordinary CoreSearchBatch RPCs")
        if method_totals[LEGACY_BY_SHARD] or method_totals[COMPACT_BY_SHARD]:
            raise RuntimeError(f"{arm} disabled mode used a by-shard search RPC")

    return {
        "probe_path": str(probe_path),
        "schema_version": PROBE_SCHEMA_VERSION,
        "mode": mode,
        "chunk": chunk,
        "run_id": run_id,
        "collection": collection,
        "base_url": base_url,
        "worker_urls": list(worker_urls),
        "api": api,
        "dataset": dataset,
        "vector_distance": "cosine",
        "vector_name": "",
        "query_dtype": "float32-le",
        "query_dimension": 200,
        "query_offset": query_offset,
        "query_count": result_proof["query_count"],
        "warmup_query_count": warmup_count,
        "top_k": result_proof["top_k"],
        "batch_size": batch_size,
        "query_sha256": query_sha,
        "warmup_query_sha256": warmup_sha,
        "ids_sha256": result_proof["ids_sha256"],
        "ids_scores_sha256": result_proof["ids_scores_sha256"],
        "row_lengths": result_proof["row_lengths"],
        "telemetry": telemetry,
        "method_totals": method_totals,
        "wall_s": wall_s,
        "qps": qps,
        "request_contract": EXPECTED_PROBE_REQUEST_CONTRACT,
        "deployment_commit": deployment_commit,
        "deployment_manifest_path": deployment_manifest_path,
        "deployment_manifest_sha256": deployment_manifest_sha256,
        "image_tag": image_tag,
        "image_id": image_id,
        "controller_peer_id": controller_peer_id_before,
        "cluster_contract": cluster_before,
        "placement_sha256": placement_before_sha256,
    }


def exact_contract(value: dict[str, Any], fields: tuple[str, ...]) -> tuple[Any, ...]:
    return tuple(value[field] for field in fields)


def validate_cross_arm_equivalence(arms: dict[str, dict[str, Any]]) -> None:
    reference = arms[DISABLED_ARM]
    benchmark_fields = (
        "collection",
        "method",
        "api",
        "query_count",
        "top_k",
        "batch_size",
        "repeats",
        "run_labels",
        "recall_values",
        "recall_mean",
        "recall_stdev",
    )
    benchmark_manifest_fields = (
        "schema_version",
        "base_url",
        "dataset",
        "commit",
        "image_tag",
        "image_id",
        "deployment_manifest_path",
        "worker_urls",
        "controller_peer_id",
        "worker_peer_ids",
        "placement",
        "placement_by_worker_url",
        "artifact_sha256",
        "generation",
        "points_count",
    )
    probe_fields = (
        "schema_version",
        "run_id",
        "collection",
        "base_url",
        "worker_urls",
        "api",
        "dataset",
        "vector_distance",
        "vector_name",
        "query_dtype",
        "query_dimension",
        "query_offset",
        "query_count",
        "warmup_query_count",
        "top_k",
        "batch_size",
        "query_sha256",
        "warmup_query_sha256",
        "ids_sha256",
        "ids_scores_sha256",
        "row_lengths",
        "deployment_commit",
        "deployment_manifest_path",
        "image_tag",
        "image_id",
        "request_contract",
        "controller_peer_id",
        "cluster_contract",
        "placement_sha256",
    )
    benchmark_reference = exact_contract(reference["benchmark"], benchmark_fields)
    benchmark_manifest_reference = exact_contract(
        reference["benchmark"]["manifest"], benchmark_manifest_fields
    )
    probe_reference = exact_contract(reference["probe"], probe_fields)
    for arm, value in arms.items():
        benchmark = value["benchmark"]
        benchmark_manifest = benchmark["manifest"]
        probe = value["probe"]
        if benchmark["collection"] != probe["collection"]:
            raise RuntimeError(f"{arm} benchmark/probe collection mismatch")
        if benchmark["api"] != probe["api"] or benchmark["top_k"] != probe["top_k"]:
            raise RuntimeError(f"{arm} benchmark/probe API or top-k mismatch")
        if benchmark_manifest["base_url"] != probe["base_url"]:
            raise RuntimeError(f"{arm} benchmark/probe controller API mismatch")
        if benchmark_manifest["worker_urls"] != probe["worker_urls"]:
            raise RuntimeError(f"{arm} benchmark/probe worker URL mismatch")
        if benchmark_manifest["dataset"] != probe["dataset"]:
            raise RuntimeError(f"{arm} benchmark/probe dataset mismatch")
        if benchmark_manifest["commit"] != probe["deployment_commit"]:
            raise RuntimeError(f"{arm} benchmark/probe commit mismatch")
        if (
            benchmark_manifest["image_tag"] != probe["image_tag"]
            or benchmark_manifest["image_id"] != probe["image_id"]
        ):
            raise RuntimeError(f"{arm} benchmark/probe image mismatch")
        if (
            benchmark_manifest["deployment_manifest_path"]
            != probe["deployment_manifest_path"]
        ):
            raise RuntimeError(f"{arm} benchmark/probe deployment manifest mismatch")
        if benchmark_manifest["controller_peer_id"] != probe["controller_peer_id"]:
            raise RuntimeError(f"{arm} benchmark/probe controller peer mismatch")
        if (
            benchmark_manifest["placement_by_worker_url"]
            != arms[DISABLED_ARM]["benchmark"]["manifest"][
                "placement_by_worker_url"
            ]
        ):
            raise RuntimeError(f"{arm} benchmark placement differs from disabled-all")
        if exact_contract(benchmark, benchmark_fields) != benchmark_reference:
            raise RuntimeError(
                f"{arm} benchmark query contract or exact recall differs from {DISABLED_ARM}"
            )
        if (
            exact_contract(benchmark_manifest, benchmark_manifest_fields)
            != benchmark_manifest_reference
        ):
            raise RuntimeError(
                f"{arm} benchmark manifest binding differs from {DISABLED_ARM}"
            )
        if exact_contract(probe, probe_fields) != probe_reference:
            raise RuntimeError(
                f"{arm} query/warmup/results transport proof differs from {DISABLED_ARM}"
            )


def expected_compact_calls(disabled_calls: int, chunk: str) -> int:
    if disabled_calls == 0:
        return 0
    if chunk == "all":
        return 1
    return math.ceil(disabled_calls / int(chunk))


def validate_rpc_relationship(
    arms: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    disabled_telemetry = arms[DISABLED_ARM]["probe"]["telemetry"]
    disabled_by_worker = {
        worker: counts[CORE_SEARCH_BATCH]
        for worker, counts in disabled_telemetry.items()
    }
    disabled_total = sum(disabled_by_worker.values())
    require_exact(
        tuple(disabled_by_worker),
        EXPECTED_WORKER_HTTP_URLS,
        "disabled-all telemetry worker order",
    )
    require_exact(
        tuple(disabled_by_worker.values()),
        EXPECTED_DISABLED_RPC_COUNTS,
        "disabled-all ordinary RPC counts",
    )
    require_exact(
        disabled_total, EXPECTED_SHARD_COUNT, "disabled-all ordinary RPC total"
    )
    placement_counts = arms[DISABLED_ARM]["benchmark"]["manifest"][
        "placement_by_worker_url"
    ]
    if disabled_by_worker != placement_counts:
        raise RuntimeError(
            "disabled-all ordinary RPC counts do not match the benchmark placement proof"
        )
    enabled: dict[str, Any] = {}
    ordered_totals: list[int] = []
    for arm in ENABLED_ARMS:
        probe = arms[arm]["probe"]
        actual_by_worker = {
            worker: counts[COMPACT_BY_SHARD]
            for worker, counts in probe["telemetry"].items()
        }
        if set(actual_by_worker) != set(disabled_by_worker):
            raise RuntimeError(f"{arm} telemetry worker set differs from disabled-all")
        expected_by_worker = {
            worker: expected_compact_calls(count, probe["chunk"])
            for worker, count in disabled_by_worker.items()
        }
        if actual_by_worker != expected_by_worker:
            raise RuntimeError(
                f"{arm} compact RPC counts {actual_by_worker} do not match the "
                f"disabled-derived expectation {expected_by_worker}"
            )
        actual_total = sum(actual_by_worker.values())
        if actual_total > disabled_total:
            raise RuntimeError(f"{arm} compact RPC count exceeds disabled-all RPC count")
        ordered_totals.append(actual_total)
        enabled[arm] = {
            "chunk": probe["chunk"],
            "actual_by_worker": actual_by_worker,
            "expected_by_worker": expected_by_worker,
            "actual_total": actual_total,
            "expected_total": sum(expected_by_worker.values()),
            "ratio_to_disabled": actual_total / disabled_total,
        }
    if ordered_totals != sorted(ordered_totals):
        raise RuntimeError(
            "compact RPC counts must be non-decreasing as the chunk size shrinks"
        )
    if ordered_totals[-1] != disabled_total:
        raise RuntimeError("enabled-1 compact RPC count must equal disabled-all RPC count")
    return {
        "valid": True,
        "disabled": {
            "arm": DISABLED_ARM,
            "ordinary_by_worker": disabled_by_worker,
            "ordinary_total": disabled_total,
        },
        "enabled": enabled,
    }


def chunk_rank(chunk: str) -> int:
    return sys.maxsize if chunk == "all" else int(chunk)


def choose_chunk(arms: dict[str, dict[str, Any]]) -> dict[str, Any]:
    ranked = sorted(
        (arms[arm] for arm in ENABLED_ARMS),
        key=lambda value: (
            value["benchmark"]["qps_mean"],
            chunk_rank(value["probe"]["chunk"]),
        ),
        reverse=True,
    )
    best, runner_up = ranked[:2]
    best_qps = best["benchmark"]["qps_mean"]
    relative_gap = (best_qps - runner_up["benchmark"]["qps_mean"]) / best_qps
    tie_applied = relative_gap <= TOP_TWO_TIE_LIMIT
    selected = (
        max((best, runner_up), key=lambda value: chunk_rank(value["probe"]["chunk"]))
        if tie_applied
        else best
    )
    disabled = arms[DISABLED_ARM]
    ranked_payload = []
    for rank, value in enumerate(ranked, start=1):
        qps = value["benchmark"]["qps_mean"]
        ranked_payload.append(
            {
                "rank": rank,
                "arm": value["arm"],
                "chunk": value["probe"]["chunk"],
                "qps_mean": qps,
                "qps_stdev": value["benchmark"]["qps_stdev"],
                "qps_cv": value["benchmark"]["qps_cv"],
                "relative_gap_to_best": (best_qps - qps) / best_qps,
            }
        )
    selected_qps = selected["benchmark"]["qps_mean"]
    disabled_qps = disabled["benchmark"]["qps_mean"]
    return {
        "schema_version": 2,
        "selection_rule": (
            "rank enabled arms by stability-run mean QPS; when the top two are "
            "within 2% of the best, choose the larger chunk of those two"
        ),
        "qps_cv_limit": QPS_CV_LIMIT,
        "top_two_tie_limit": TOP_TWO_TIE_LIMIT,
        "ranked_enabled": ranked_payload,
        "raw_best_arm": best["arm"],
        "raw_best_chunk": best["probe"]["chunk"],
        "runner_up_arm": runner_up["arm"],
        "runner_up_chunk": runner_up["probe"]["chunk"],
        "top_two_relative_gap": relative_gap,
        "tie_applied": tie_applied,
        "selected_arm": selected["arm"],
        "selected_chunk": selected["probe"]["chunk"],
        "selected_qps_mean": selected_qps,
        "selected_qps_cv": selected["benchmark"]["qps_cv"],
        "disabled_arm": DISABLED_ARM,
        "disabled_qps_mean": disabled_qps,
        "selected_qps_relative_to_disabled": selected_qps / disabled_qps,
        "selected_qps_delta_pct_vs_disabled":
            (selected_qps / disabled_qps - 1.0) * 100.0,
    }


def build_transport_equivalence(
    sweep_root: Path,
    arms: dict[str, dict[str, Any]],
    rpc_relationship: dict[str, Any],
) -> dict[str, Any]:
    reference = arms[DISABLED_ARM]
    per_arm: dict[str, Any] = {}
    for arm, value in arms.items():
        probe = value["probe"]
        benchmark = value["benchmark"]
        per_arm[arm] = {
            "mode": probe["mode"],
            "chunk": probe["chunk"],
            "query_equal": probe["query_sha256"]
                == reference["probe"]["query_sha256"],
            "warmup_equal": probe["warmup_query_sha256"]
                == reference["probe"]["warmup_query_sha256"],
            "ids_equal": probe["ids_sha256"]
                == reference["probe"]["ids_sha256"],
            "ids_scores_equal": probe["ids_scores_sha256"]
                == reference["probe"]["ids_scores_sha256"],
            "recall_equal": benchmark["recall_values"]
                == reference["benchmark"]["recall_values"],
            "latency_p95_ms": benchmark["latency_p95_ms"],
            "latency_p95_ms_stdev": benchmark["latency_p95_ms_stdev"],
            "latency_p99_ms": benchmark["latency_p99_ms"],
            "latency_p99_ms_stdev": benchmark["latency_p99_ms_stdev"],
            "top_k_rows_valid": True,
            "status_zero_only": True,
            "benchmark_manifest_path": benchmark["manifest"]["manifest_path"],
            "method_totals": probe["method_totals"],
            "summary_path": benchmark["summary_path"],
            "stability_path": benchmark["stability_path"],
            "transport_probe_path": probe["probe_path"],
        }
    return {
        "schema_version": 2,
        "sweep_root": str(sweep_root),
        "reference_arm": DISABLED_ARM,
        "equivalent": True,
        "strict_protocol": {
            "benchmark_warmup_query_count": BENCHMARK_WARMUP_QUERY_COUNT,
            "benchmark_query_count": BENCHMARK_QUERY_COUNT,
            "probe_warmup_query_count": PROBE_WARMUP_QUERY_COUNT,
            "probe_query_count": PROBE_QUERY_COUNT,
            "top_k": TOP_K,
            "batch_size": BATCH_SIZE,
            "stability_repeats": STABILITY_REPEATS,
            "worker_count": EXPECTED_WORKER_COUNT,
            "shard_count": EXPECTED_SHARD_COUNT,
            "replication_factor": EXPECTED_REPLICATION_FACTOR,
            "disabled_rpc_counts": list(EXPECTED_DISABLED_RPC_COUNTS),
            "dataset_sha256": EXPECTED_DATASET_SHA256,
            "commit": reference["benchmark"]["manifest"]["commit"],
            "image_id": reference["benchmark"]["manifest"]["image_id"],
        },
        "benchmark_contract": {
            "collection": reference["benchmark"]["collection"],
            "warmup_query_count": BENCHMARK_WARMUP_QUERY_COUNT,
            "query_count": reference["benchmark"]["query_count"],
            "top_k": reference["benchmark"]["top_k"],
            "batch_size": reference["benchmark"]["batch_size"],
            "stability_repeats": reference["benchmark"]["repeats"],
            "recall_at_k": reference["benchmark"]["recall_mean"],
            "recall_sequence": reference["benchmark"]["recall_values"],
        },
        "probe_contract": {
            "query_offset": reference["probe"]["query_offset"],
            "query_count": reference["probe"]["query_count"],
            "warmup_query_count": reference["probe"]["warmup_query_count"],
            "top_k": reference["probe"]["top_k"],
            "batch_size": reference["probe"]["batch_size"],
            "query_sha256": reference["probe"]["query_sha256"],
            "warmup_query_sha256": reference["probe"]["warmup_query_sha256"],
            "ids_sha256": reference["probe"]["ids_sha256"],
            "ids_scores_sha256": reference["probe"]["ids_scores_sha256"],
        },
        "rpc_relationship": rpc_relationship,
        "arms": per_arm,
    }


def summary_rows(
    arms: dict[str, dict[str, Any]],
    rpc_relationship: dict[str, Any],
    selection: dict[str, Any],
) -> list[dict[str, Any]]:
    disabled_total = rpc_relationship["disabled"]["ordinary_total"]
    rows: list[dict[str, Any]] = []
    for arm, mode, chunk in ARM_SETTINGS:
        value = arms[arm]
        benchmark = value["benchmark"]
        probe = value["probe"]
        if mode == "enabled":
            active_count = probe["method_totals"][COMPACT_BY_SHARD]
            expected_count = rpc_relationship["enabled"][arm]["expected_total"]
        else:
            active_count = probe["method_totals"][CORE_SEARCH_BATCH]
            expected_count = disabled_total
        rows.append(
            {
                "arm": arm,
                "peer_premerge_mode": mode,
                "shards_per_rpc": chunk,
                "benchmark_query_count": benchmark["query_count"],
                "benchmark_top_k": benchmark["top_k"],
                "benchmark_batch_size": benchmark["batch_size"],
                "stability_repeats": benchmark["repeats"],
                "recall_at_k": benchmark["recall_mean"],
                "qps_mean": benchmark["qps_mean"],
                "qps_stdev": benchmark["qps_stdev"],
                "qps_cv": benchmark["qps_cv"],
                "qps_cv_pct": benchmark["qps_cv"] * 100.0,
                "latency_p95_ms": benchmark["latency_p95_ms"],
                "latency_p95_ms_stdev": benchmark["latency_p95_ms_stdev"],
                "latency_p99_ms": benchmark["latency_p99_ms"],
                "latency_p99_ms_stdev": benchmark["latency_p99_ms_stdev"],
                "probe_query_count": probe["query_count"],
                "probe_warmup_query_count": probe["warmup_query_count"],
                "probe_top_k": probe["top_k"],
                "probe_batch_size": probe["batch_size"],
                "query_sha256": probe["query_sha256"],
                "warmup_query_sha256": probe["warmup_query_sha256"],
                "ids_sha256": probe["ids_sha256"],
                "ids_scores_sha256": probe["ids_scores_sha256"],
                "core_search_batch_rpc_count": probe["method_totals"][
                    CORE_SEARCH_BATCH
                ],
                "legacy_by_shard_rpc_count": probe["method_totals"][LEGACY_BY_SHARD],
                "compact_rpc_count": probe["method_totals"][COMPACT_BY_SHARD],
                "active_rpc_count": active_count,
                "expected_active_rpc_count": expected_count,
                "rpc_ratio_to_disabled": active_count / disabled_total,
                "transport_equivalent": True,
                "selected": arm == selection["selected_arm"],
            }
        )
    return rows


def write_json(path: Path, value: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(sweep_root: str | Path, output_dir: str | Path) -> Path:
    root = Path(sweep_root).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"sweep root not found: {root}")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output}")

    arms: dict[str, dict[str, Any]] = {}
    for arm, mode, chunk in ARM_SETTINGS:
        arm_root = root / arm
        if not arm_root.is_dir():
            raise FileNotFoundError(f"required sweep arm not found: {arm_root}")
        arms[arm] = {
            "arm": arm,
            "benchmark": validate_benchmark(arm, mode, chunk, arm_root),
            "probe": validate_probe(arm, mode, chunk, arm_root),
        }
    validate_cross_arm_equivalence(arms)
    rpc_relationship = validate_rpc_relationship(arms)
    selection = choose_chunk(arms)
    transport = build_transport_equivalence(root, arms, rpc_relationship)
    rows = summary_rows(arms, rpc_relationship, selection)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(exist_ok=False)
    write_csv(output / "chunk_sweep_summary.csv", rows)
    write_json(output / "transport_equivalence.json", transport)
    write_json(output / "chunk_selection.json", selection)
    return output


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        output = analyze(args.sweep_root, args.output_dir)
    except (ValueError, RuntimeError, FileNotFoundError, FileExistsError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"output_dir": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
