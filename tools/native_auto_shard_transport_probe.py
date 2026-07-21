#!/usr/bin/env python3
"""Probe native auto-shard transport without changing collection semantics.

The probe sends ordinary coordinator Search/Query batches, records exact ordered
external IDs and score bit-patterns, and captures selected worker-side gRPC
telemetry counters before and after the measured window.  It is intended for
Orion peer-premerge/chunking A/B validation; it never sends shard selectors,
entry points, per-shard EF values, or source-ID hints.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import tempfile
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import qdrant_two_level_routing_experiment as experiment  # noqa: E402
from tools import method4_distributed_cluster as cluster  # noqa: E402


DEFAULT_TELEMETRY_METHODS = (
    "/qdrant.PointsInternal/CoreSearchBatch",
    "/qdrant.PointsInternal/CoreSearchBatchByShard",
    "/qdrant.PointsInternal/CoreSearchBatchByShardCompact",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a fixed standard-API batch and capture exact result/transport proof."
        )
    )
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--topology", required=True)
    parser.add_argument("--deployment-manifest", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--collection", required=True)
    parser.add_argument("--hdf5-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--worker-url", action="append", default=[])
    parser.add_argument(
        "--telemetry-method",
        action="append",
        default=None,
        help="gRPC method path to count; repeatable.",
    )
    parser.add_argument("--query-offset", type=int, default=0)
    parser.add_argument("--query-count", type=int, default=200)
    parser.add_argument("--warmup-query-count", type=int, default=500)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--api", choices=("search", "query"), default="search")
    parser.add_argument(
        "--vector-distance",
        choices=("cosine", "euclid", "l2"),
        default="cosine",
    )
    parser.add_argument("--vector-name", default="")
    parser.add_argument("--request-timeout-secs", type=float, default=600.0)
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if args.query_offset < 0:
        raise ValueError("--query-offset must be non-negative")
    positive = {
        "query-count": args.query_count,
        "top-k": args.top_k,
        "batch-size": args.batch_size,
        "request-timeout-secs": args.request_timeout_secs,
    }
    for name, value in positive.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"--{name} must be positive")
        if not math.isfinite(float(value)) or value <= 0:
            raise ValueError(f"--{name} must be positive")
    if args.warmup_query_count < 0:
        raise ValueError("--warmup-query-count must be non-negative")
    if len(set(url.rstrip("/") for url in args.worker_url)) != len(args.worker_url):
        raise ValueError("--worker-url values must be unique")
    methods = args.telemetry_method or list(DEFAULT_TELEMETRY_METHODS)
    if not methods or any(not isinstance(method, str) or not method.startswith("/") for method in methods):
        raise ValueError("--telemetry-method values must be absolute gRPC method paths")
    if len(set(methods)) != len(methods):
        raise ValueError("--telemetry-method values must be unique")


def output_path(value: str | Path) -> Path:
    path = Path(value).expanduser().resolve()
    if path == REPO_ROOT or REPO_ROOT in path.parents:
        raise ValueError("probe output must be outside the repository")
    if path.exists():
        raise FileExistsError(f"refusing to overwrite probe output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def runtime_container_proof(
    topology: dict[str, Any], inspected: dict[str, dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    proof: dict[str, dict[str, Any]] = {}
    for node in cluster.all_nodes(topology):
        role = str(node["role"])
        value = inspected[role]
        proof[role] = {
            "container_id": str(value.get("Id") or ""),
            "image_id": str(value.get("Image") or ""),
            "image_tag": str((value.get("Config") or {}).get("Image") or ""),
            "running": bool((value.get("State") or {}).get("Running")),
            "cpuset": str((value.get("HostConfig") or {}).get("CpusetCpus") or ""),
            "network_mode": str(
                (value.get("HostConfig") or {}).get("NetworkMode") or ""
            ),
        }
    return proof


def validate_live_cluster(
    topology: dict[str, Any], run_id: str, collection_name: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    snapshot = cluster.cluster_snapshot(topology)
    errors = cluster.cluster_validation_errors(topology, snapshot)
    collections = cluster.run_collection_placements(topology, run_id)
    errors.extend(cluster.collection_validation_errors(topology, snapshot, collections))
    if collection_name not in collections:
        errors.append(
            f"requested collection {collection_name!r} is not a run-scoped collection for {run_id}"
        )
    if errors:
        raise RuntimeError("cluster validation failed: " + "; ".join(errors))
    return snapshot, collections


def load_deployment_context(args: argparse.Namespace) -> dict[str, Any]:
    topology_path = Path(args.topology).expanduser().resolve()
    topology = cluster.load_topology(topology_path)
    run_id = cluster.validate_run_id(args.run_id)
    manifest_path = Path(args.deployment_manifest).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"deployment manifest not found: {manifest_path}")
    expected_manifest_path = cluster.manifest_path(topology, run_id).resolve()
    if manifest_path != expected_manifest_path:
        raise ValueError(
            "--deployment-manifest must be the run-scoped manifest: "
            f"expected {expected_manifest_path}, got {manifest_path}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    image_tag, image_id, mode, shards_per_rpc, deployment_commit = (
        cluster.validate_peer_premerge_transition_manifest(
            topology,
            run_id,
            manifest,
        )
    )

    expected_base_url = cluster.http_url(topology["controller"], topology).rstrip("/")
    if args.base_url.rstrip("/") != expected_base_url:
        raise ValueError(
            f"--base-url must be the controller standard API {expected_base_url}"
        )
    expected_worker_urls = [
        cluster.http_url(node, topology).rstrip("/") for node in topology["workers"]
    ]
    if args.worker_url:
        actual_worker_urls = sorted(url.rstrip("/") for url in args.worker_url)
        if actual_worker_urls != sorted(expected_worker_urls):
            raise ValueError(
                "--worker-url values must exactly match the three topology workers"
            )
    args.worker_url = expected_worker_urls

    runtime_args = SimpleNamespace(
        dry_run=False,
        ssh_user=None,
        ssh_option=[],
    )
    inspected, actual_mode, actual_shards_per_rpc = (
        cluster.inspect_peer_premerge_transition_runtime(
            topology,
            run_id,
            image_tag,
            image_id,
            mode,
            shards_per_rpc,
            runtime_args,
        )
    )
    snapshot, collections = validate_live_cluster(topology, run_id, args.collection)
    return {
        "topology_path": str(topology_path),
        "topology": topology,
        "run_id": run_id,
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "manifest": manifest,
        "deployment_commit": deployment_commit,
        "image_tag": image_tag,
        "image_id": image_id,
        "peer_premerge_mode": actual_mode,
        "peer_premerge_shards_per_rpc": actual_shards_per_rpc,
        "runtime_args": runtime_args,
        "containers": inspected,
        "cluster": snapshot,
        "collections": collections,
    }


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def collection_placement_signature(collections: dict[str, Any]) -> dict[str, Any]:
    signature: dict[str, Any] = {}
    for name, payload in sorted(collections.items()):
        cluster_payload = (payload or {}).get("cluster") or {}
        signature[name] = {
            "peer_id": cluster_payload.get("peer_id"),
            "local_shards": sorted(
                cluster_payload.get("local_shards") or [],
                key=lambda item: (
                    int((item or {}).get("shard_id", -1)),
                    str((item or {}).get("peer_id") or ""),
                ),
            ),
            "remote_shards": sorted(
                cluster_payload.get("remote_shards") or [],
                key=lambda item: (
                    int((item or {}).get("shard_id", -1)),
                    str((item or {}).get("peer_id") or ""),
                ),
            ),
            "shard_transfers": cluster_payload.get("shard_transfers") or [],
        }
    return signature


def capture_unchanged_deployment(
    context: dict[str, Any], collection_name: str
) -> dict[str, Any]:
    topology = context["topology"]
    inspected, mode, shards_per_rpc = cluster.inspect_peer_premerge_transition_runtime(
        topology,
        context["run_id"],
        context["image_tag"],
        context["image_id"],
        context["peer_premerge_mode"],
        context["peer_premerge_shards_per_rpc"],
        context["runtime_args"],
    )
    snapshot, collections = validate_live_cluster(
        topology, context["run_id"], collection_name
    )
    return {
        "peer_premerge_mode": mode,
        "peer_premerge_shards_per_rpc": shards_per_rpc,
        "containers": inspected,
        "cluster": snapshot,
        "collections": collections,
    }


def assert_deployment_unchanged(
    before: dict[str, Any], after: dict[str, Any]
) -> None:
    before_containers = runtime_container_proof(before["topology"], before["containers"])
    after_containers = runtime_container_proof(before["topology"], after["containers"])
    if before_containers != after_containers:
        raise RuntimeError("container runtime identity changed during transport probe")
    before_cluster = before["cluster"].get("result") or {}
    after_cluster = after["cluster"].get("result") or {}
    if before_cluster.get("peer_id") != after_cluster.get("peer_id"):
        raise RuntimeError("controller peer ID changed during transport probe")
    before_placement = collection_placement_signature(before["collections"])
    after_placement = collection_placement_signature(after["collections"])
    if before_placement != after_placement:
        raise RuntimeError("collection placement changed during transport probe")


def load_queries(
    hdf5_path: str | Path,
    *,
    offset: int,
    count: int,
    vector_distance: str,
) -> Any:
    path = Path(hdf5_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"HDF5 dataset not found: {path}")
    with experiment.h5py.File(path, "r") as handle:
        if "test" not in handle:
            raise ValueError("HDF5 dataset is missing test vectors")
        shape = tuple(int(value) for value in handle["test"].shape)
        if len(shape) != 2:
            raise ValueError("HDF5 test vectors must be two-dimensional")
        end = offset + count
        if end > shape[0]:
            raise ValueError(
                f"requested query range [{offset}, {end}) exceeds test count {shape[0]}"
            )
        queries = handle["test"][offset:end].astype(
            experiment.np.float32,
            copy=True,
        )
    return experiment.prepare_vectors_for_distance(queries, vector_distance)


def telemetry_method_status_counts(
    payload: dict[str, Any], method: str
) -> dict[str, int]:
    root = payload.get("result", payload)
    responses = (((root.get("requests") or {}).get("grpc") or {}).get("responses") or {})
    status_statistics = responses.get(method) or {}
    if not isinstance(status_statistics, dict):
        raise RuntimeError(f"invalid telemetry status map for {method}")
    counts: dict[str, int] = {}
    for status, statistics in status_statistics.items():
        if not isinstance(statistics, dict):
            raise RuntimeError(f"invalid telemetry statistics for {method}")
        count = statistics.get("count", 0)
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise RuntimeError(f"invalid telemetry count for {method}: {count!r}")
        counts[str(status)] = count
    return counts


def telemetry_method_count(payload: dict[str, Any], method: str) -> int:
    return sum(telemetry_method_status_counts(payload, method).values())


def telemetry_snapshot(
    worker_urls: list[str], methods: list[str], timeout: float
) -> dict[str, dict[str, dict[str, int]]]:
    snapshot: dict[str, dict[str, dict[str, int]]] = {}
    for raw_url in worker_urls:
        url = raw_url.rstrip("/")
        payload = experiment.request_json(
            url,
            "GET",
            "/telemetry?details_level=2",
            timeout=timeout,
        )
        snapshot[url] = {
            method: telemetry_method_status_counts(payload, method)
            for method in methods
        }
    return snapshot


def telemetry_delta(
    before: dict[str, dict[str, dict[str, int]]],
    after: dict[str, dict[str, dict[str, int]]],
) -> dict[str, dict[str, dict[str, int]]]:
    if set(before) != set(after):
        raise RuntimeError("telemetry worker set changed during probe")
    delta: dict[str, dict[str, dict[str, int]]] = {}
    for worker in sorted(before):
        if set(before[worker]) != set(after[worker]):
            raise RuntimeError(f"telemetry method set changed for {worker}")
        delta[worker] = {}
        for method in before[worker]:
            statuses = set(before[worker][method]) | set(after[worker][method])
            delta[worker][method] = {}
            for status in sorted(statuses):
                difference = after[worker][method].get(status, 0) - before[worker][
                    method
                ].get(status, 0)
                if difference < 0:
                    raise RuntimeError(
                        f"telemetry counter decreased for {worker} {method} "
                        f"status={status}; the worker likely restarted during the probe"
                    )
                delta[worker][method][status] = difference
    return delta


def canonical_result_proof(
    rows: list[list[tuple[float, int]]],
) -> dict[str, Any]:
    results = [
        [
            {
                "id": int(point_id),
                "score_f32_le_hex": experiment.np.asarray(
                    score, dtype="<f4"
                ).tobytes().hex(),
            }
            for score, point_id in row
        ]
        for row in rows
    ]
    ids = [[point["id"] for point in row] for row in results]
    encoded_ids = json.dumps(
        ids,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    encoded_results = json.dumps(
        results,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return {
        "ids_sha256": hashlib.sha256(encoded_ids).hexdigest(),
        "ids_scores_sha256": hashlib.sha256(encoded_results).hexdigest(),
        "results": results,
    }


def validate_result_rows(
    rows: list[list[tuple[float, int]]], *, query_count: int, top_k: int
) -> list[int]:
    if len(rows) != query_count:
        raise RuntimeError(
            f"probe returned {len(rows)} rows for {query_count} queries"
        )
    lengths: list[int] = []
    for query_index, row in enumerate(rows):
        lengths.append(len(row))
        if len(row) != top_k:
            raise RuntimeError(
                f"probe query {query_index} returned {len(row)} points; expected exactly {top_k}"
            )
        point_ids = [int(point_id) for _score, point_id in row]
        if len(set(point_ids)) != len(point_ids):
            raise RuntimeError(f"probe query {query_index} returned duplicate point IDs")
        if any(not math.isfinite(float(score)) for score, _point_id in row):
            raise RuntimeError(f"probe query {query_index} returned a non-finite score")
    return lengths


def execute_batches(
    args: argparse.Namespace,
    queries: Any,
) -> tuple[list[list[tuple[float, int]]], float]:
    rows: list[list[tuple[float, int]]] = []
    started = time.perf_counter()
    for start in range(0, len(queries), args.batch_size):
        rows.extend(
            experiment.standard_dense_vector_batch(
                args.base_url,
                args.collection,
                queries[start : start + args.batch_size],
                args.top_k,
                api=args.api,
                vector_name=args.vector_name,
                timeout=args.request_timeout_secs,
            )
        )
    return rows, time.perf_counter() - started


def run_probe(args: argparse.Namespace) -> Path:
    validate_args(args)
    destination = output_path(args.output)
    context = load_deployment_context(args)
    methods = list(args.telemetry_method or DEFAULT_TELEMETRY_METHODS)
    dataset_path = Path(args.hdf5_path).expanduser().resolve()
    dataset_manifest = context["manifest"].get("dataset") or {}
    manifest_dataset_path = Path(str(dataset_manifest.get("path") or "")).resolve()
    if manifest_dataset_path != dataset_path:
        raise RuntimeError(
            "probe HDF5 path does not match the deployment manifest dataset: "
            f"manifest={manifest_dataset_path}, probe={dataset_path}"
        )
    if dataset_manifest.get("size_bytes") != dataset_path.stat().st_size:
        raise RuntimeError("probe HDF5 size does not match the deployment manifest")

    measured_queries = load_queries(
        dataset_path,
        offset=args.query_offset,
        count=args.query_count,
        vector_distance=args.vector_distance,
    )
    warmup_query_sha256 = None
    if args.warmup_query_count:
        warmup_queries = load_queries(
            dataset_path,
            offset=0,
            count=args.warmup_query_count,
            vector_distance=args.vector_distance,
        )
        warmup_rows, _warmup_wall_s = execute_batches(args, warmup_queries)
        validate_result_rows(
            warmup_rows,
            query_count=args.warmup_query_count,
            top_k=args.top_k,
        )
        warmup_bytes = experiment.np.asarray(warmup_queries, dtype="<f4").tobytes()
        warmup_query_sha256 = hashlib.sha256(warmup_bytes).hexdigest()

    before = telemetry_snapshot(
        args.worker_url,
        methods,
        args.request_timeout_secs,
    )
    rows, wall_s = execute_batches(args, measured_queries)
    after = telemetry_snapshot(
        args.worker_url,
        methods,
        args.request_timeout_secs,
    )
    row_lengths = validate_result_rows(
        rows,
        query_count=args.query_count,
        top_k=args.top_k,
    )
    after_context = capture_unchanged_deployment(context, args.collection)
    assert_deployment_unchanged(context, after_context)
    proof = canonical_result_proof(rows)
    query_bytes = experiment.np.asarray(measured_queries, dtype="<f4").tobytes()
    before_container_proof = runtime_container_proof(
        context["topology"], context["containers"]
    )
    after_container_proof = runtime_container_proof(
        context["topology"], after_context["containers"]
    )
    before_placement = collection_placement_signature(context["collections"])
    after_placement = collection_placement_signature(after_context["collections"])
    payload = {
        "schema_version": 2,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "run_id": context["run_id"],
        "collection": args.collection,
        "base_url": args.base_url.rstrip("/"),
        "worker_urls": list(args.worker_url),
        "api": args.api,
        "deployment": {
            "topology_path": context["topology_path"],
            "manifest_path": context["manifest_path"],
            "manifest_sha256": context["manifest_sha256"],
            "commit": context["deployment_commit"],
            "image_tag": context["image_tag"],
            "image_id": context["image_id"],
            "peer_premerge_mode": context["peer_premerge_mode"],
            "peer_premerge_shards_per_rpc": context[
                "peer_premerge_shards_per_rpc"
            ],
            "containers_before": before_container_proof,
            "containers_after": after_container_proof,
            "controller_peer_id_before": (context["cluster"].get("result") or {}).get(
                "peer_id"
            ),
            "controller_peer_id_after": (
                after_context["cluster"].get("result") or {}
            ).get("peer_id"),
            "cluster_before": cluster.transition_cluster_proof(context["cluster"]),
            "cluster_after": cluster.transition_cluster_proof(
                after_context["cluster"]
            ),
            "collection_placement_before_sha256": canonical_sha256(
                before_placement
            ),
            "collection_placement_after_sha256": canonical_sha256(after_placement),
        },
        "dataset": {
            "path": str(dataset_path),
            "size_bytes": dataset_path.stat().st_size,
            "sha256": dataset_manifest.get("sha256"),
            "hdf5_shapes": dataset_manifest.get("hdf5_shapes"),
        },
        "vector_distance": args.vector_distance,
        "vector_name": args.vector_name,
        "query_dtype": "float32-le",
        "query_dimension": int(measured_queries.shape[1]),
        "query_offset": args.query_offset,
        "query_count": args.query_count,
        "warmup_query_count": args.warmup_query_count,
        "warmup_query_sha256": warmup_query_sha256,
        "top_k": args.top_k,
        "result_row_lengths": row_lengths,
        "batch_size": args.batch_size,
        "wall_s": wall_s,
        "qps": args.query_count / wall_s,
        "query_sha256": hashlib.sha256(query_bytes).hexdigest(),
        "request_contract": (
            "ordinary coordinator batch; no shard selector, entry points, "
            "per-shard EF, or source-ID hint"
        ),
        "telemetry_methods": methods,
        "telemetry_before": before,
        "telemetry_after": after,
        "telemetry_delta": telemetry_delta(before, after),
        **proof,
    }
    temporary_path: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
        )
        temporary_path = Path(temporary_name)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(
                json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
                + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, destination)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    print(json.dumps({"probe": str(destination)}, indent=2))
    return destination


def main(argv: list[str] | None = None) -> int:
    try:
        run_probe(parse_args(argv))
    except (
        ValueError,
        FileNotFoundError,
        FileExistsError,
        RuntimeError,
        TimeoutError,
        OSError,
        subprocess.CalledProcessError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
