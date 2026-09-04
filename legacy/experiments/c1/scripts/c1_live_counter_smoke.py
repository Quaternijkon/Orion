#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Sequence

import numpy as np


def utc_timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def request_json(
    base_url: str,
    method: str,
    path: str,
    body: dict[str, Any] | None = None,
    *,
    timeout: float = 120.0,
) -> dict[str, Any]:
    encoded = None
    headers = {"Accept": "application/json"}
    if body is not None:
        encoded = json.dumps(body, separators=(",", ":")).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=encoded,
        headers=headers,
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {path} returned HTTP {exc.code}: {detail}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"{method} {path} returned a non-object JSON payload")
    status = payload.get("status")
    if isinstance(status, dict) and status.get("error"):
        raise RuntimeError(f"{method} {path} failed: {status['error']}")
    return payload


def collection_path(collection: str) -> str:
    return f"/collections/{urllib.parse.quote(collection, safe='')}"


def wait_indexed(
    base_url: str,
    collection: str,
    expected_points: int,
    timeout: float,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    last: dict[str, Any] = {}
    while time.monotonic() < deadline:
        last = request_json(base_url, "GET", collection_path(collection))["result"]
        points = int(last.get("points_count") or 0)
        indexed = int(last.get("indexed_vectors_count") or 0)
        status = str(last.get("status") or "")
        optimizer_status = last.get("optimizer_status")
        optimizer_ok = optimizer_status == "ok" or (
            isinstance(optimizer_status, dict) and optimizer_status.get("ok") is True
        )
        if (
            points == expected_points
            and indexed >= expected_points
            and status == "green"
            and optimizer_ok
        ):
            return last
        time.sleep(1.0)
    raise TimeoutError(
        f"collection {collection!r} did not finish indexing within {timeout}s; last={last}"
    )


def hardware_usage(response: dict[str, Any]) -> dict[str, int]:
    usage = response.get("usage")
    hardware = usage.get("hardware") if isinstance(usage, dict) else None
    if not isinstance(hardware, dict):
        raise RuntimeError("response did not contain usage.hardware")
    required = {
        "cpu",
        "cpu_time_us",
        "cpu_wall_time_us",
        "graph_nodes_visited",
    }
    missing = sorted(required - set(hardware))
    if missing:
        raise RuntimeError(f"usage.hardware is missing fields: {missing}")
    return {str(key): int(value) for key, value in hardware.items()}


def distance_computations(cpu_units: int, dimension: int) -> int:
    score_unit = dimension * np.dtype(np.float32).itemsize
    quotient, remainder = divmod(cpu_units, score_unit)
    if remainder:
        raise RuntimeError(
            f"hardware CPU units {cpu_units} are not divisible by dense score unit {score_unit}"
        )
    return quotient


def execute_smoke(args: argparse.Namespace) -> dict[str, Any]:
    rng = np.random.default_rng(args.seed)
    vectors = rng.normal(size=(args.point_count, args.dimension)).astype(np.float32)
    vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    query = vectors[0].tolist()
    collection = args.collection
    shard_key = "c1_smoke_shard"
    collection_url = collection_path(collection)
    cluster = request_json(args.base_url, "GET", "/cluster")["result"]
    peer_id = int(cluster["peer_id"])

    try:
        request_json(args.base_url, "DELETE", collection_url)
    except RuntimeError as exc:
        if "HTTP 404" not in str(exc):
            raise

    created = request_json(
        args.base_url,
        "PUT",
        collection_url,
        {
            "vectors": {"size": args.dimension, "distance": "Cosine"},
            "shard_number": 1,
            "sharding_method": "custom",
            "replication_factor": 1,
            "write_consistency_factor": 1,
            "hnsw_config": {
                "m": 16,
                "ef_construct": 100,
                # Qdrant validates this threshold in KiB and requires at least 10.
                # The smoke collection is much larger than 10 KiB, so approximate
                # searches are still forced through the HNSW index.
                "full_scan_threshold": 10,
                "max_indexing_threads": 2,
            },
            "optimizers_config": {
                "default_segment_number": 1,
                "indexing_threshold": 1,
                "max_optimization_threads": 2,
            },
        },
    )
    request_json(
        args.base_url,
        "PUT",
        collection_url + "/shards",
        {
            "shard_key": shard_key,
            "shards_number": 1,
            "replication_factor": 1,
            "placement": [peer_id],
        },
    )
    for start in range(0, args.point_count, args.batch_size):
        stop = min(start + args.batch_size, args.point_count)
        request_json(
            args.base_url,
            "PUT",
            collection_url + "/points?wait=true",
            {
                "points": [
                    {"id": index, "vector": vectors[index].tolist()}
                    for index in range(start, stop)
                ],
                "shard_key": shard_key,
            },
            timeout=args.timeout,
        )
    indexed = wait_indexed(
        args.base_url,
        collection,
        args.point_count,
        args.timeout,
    )

    common_search = {
        "vector": query,
        "limit": 10,
        "with_payload": False,
        "with_vector": False,
        "shard_key": [shard_key],
    }
    approximate = request_json(
        args.base_url,
        "POST",
        collection_url + "/points/search",
        {**common_search, "params": {"hnsw_ef": 64, "exact": False}},
        timeout=args.timeout,
    )
    exact = request_json(
        args.base_url,
        "POST",
        collection_url + "/points/search",
        {**common_search, "params": {"exact": True}},
        timeout=args.timeout,
    )
    approximate_hw = hardware_usage(approximate)
    exact_hw = hardware_usage(exact)
    approximate_distances = distance_computations(
        approximate_hw["cpu"], args.dimension
    )
    exact_distances = distance_computations(exact_hw["cpu"], args.dimension)
    checks = {
        "approximate_usage_present": True,
        "approximate_cpu_divides_exactly": True,
        "approximate_distance_computations_positive": approximate_distances > 0,
        "approximate_graph_nodes_positive": approximate_hw["graph_nodes_visited"] > 0,
        "approximate_worker_cpu_time_positive": approximate_hw["cpu_time_us"] > 0,
        "approximate_worker_wall_time_positive": approximate_hw["cpu_wall_time_us"] > 0,
        "exact_cpu_divides_exactly": True,
        "exact_distance_computations_positive": exact_distances > 0,
        "exact_graph_nodes_zero": exact_hw["graph_nodes_visited"] == 0,
        "exact_worker_cpu_time_positive": exact_hw["cpu_time_us"] > 0,
        "exact_worker_wall_time_positive": exact_hw["cpu_wall_time_us"] > 0,
    }
    payload = {
        "timestamp": utc_timestamp(),
        "base_url": args.base_url,
        "collection": collection,
        "peer_id": peer_id,
        "seed": args.seed,
        "dimension": args.dimension,
        "point_count": args.point_count,
        "created_response": created,
        "indexed_collection": indexed,
        "approximate": {
            "hardware": approximate_hw,
            "distance_computations": approximate_distances,
            "result_count": len(approximate.get("result") or []),
            "response": approximate,
        },
        "exact": {
            "hardware": exact_hw,
            "distance_computations": exact_distances,
            "result_count": len(exact.get("result") or []),
            "response": exact,
        },
        "checks": checks,
        "passed": all(checks.values()),
    }
    if not payload["passed"]:
        raise RuntimeError(f"live counter smoke failed: {checks}")
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="C1 live HNSW work-counter smoke test")
    parser.add_argument("--base-url", default="http://10.10.1.1:6333")
    parser.add_argument("--collection", default="c1_stage0_counter_smoke")
    parser.add_argument("--output", required=True)
    parser.add_argument("--dimension", type=int, default=128)
    parser.add_argument("--point-count", type=int, default=2_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--keep-collection", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any]
    try:
        payload = execute_smoke(args)
    finally:
        if not args.keep_collection:
            try:
                request_json(
                    args.base_url,
                    "DELETE",
                    collection_path(args.collection),
                )
            except Exception as exc:
                print(f"warning: failed to remove smoke collection: {exc}", flush=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
