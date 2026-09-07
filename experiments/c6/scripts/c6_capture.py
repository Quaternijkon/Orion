#!/usr/bin/env python3
"""Capture a frozen C6 query/shard/efSearch result cube from Qdrant.

This runner never rebuilds a collection.  It consumes a checksum-addressed Orion
route trace, searches every represented candidate shard independently at the
requested EF grid, and appends one durable JSONL record per completed search.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import math
import os
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from c6_protocol import (  # noqa: E402
    DurableJsonlBatchWriter,
    EF_GRID,
    LocalSearchTrace,
    append_jsonl,
    canonical_json_sha256,
    distance_computations_from_usage,
    policy_config_from_dict,
    policy_efs,
    policy_shards,
    route_queries_from_trace,
    sha256_path,
    utc_timestamp,
    write_json_atomic,
)


_HTTP_CONNECTIONS = threading.local()


def parse_int_grid(value: str) -> tuple[int, ...]:
    try:
        values = tuple(sorted(set(int(item) for item in value.split(",") if item)))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("EF grid must contain integers") from exc
    if not values or values[0] <= 0:
        raise argparse.ArgumentTypeError("EF grid must contain positive integers")
    return values


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--collection", required=True)
    parser.add_argument("--route-trace", required=True)
    parser.add_argument("--hdf5-path", required=True)
    parser.add_argument("--query-start", type=int, required=True)
    parser.add_argument("--query-count", type=int, required=True)
    parser.add_argument("--shard-key-map", required=True)
    parser.add_argument(
        "--shard-base-url-map",
        help="Optional JSON object mapping logical shard IDs to direct owner HTTP URLs.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--ef-grid", type=parse_int_grid, default=EF_GRID)
    parser.add_argument(
        "--policy-matrix",
        help="Optional JSON object mapping P0-P3 to candidate configuration arrays.",
    )
    parser.add_argument(
        "--oracle-high-ef",
        type=int,
        default=max(EF_GRID),
        help="High fixed EF used by the P4 prefix oracle.",
    )
    parser.add_argument(
        "--skip-oracle-grid",
        action="store_true",
        help="Capture only deployable policy allocations from --policy-matrix.",
    )
    parser.add_argument(
        "--exclude-keys-from",
        action="append",
        default=[],
        help="Existing trace JSONL whose covered keys must not be recaptured.",
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--request-workers", type=int, default=16)
    parser.add_argument(
        "--max-in-flight",
        type=int,
        default=0,
        help="Bound submitted futures; 0 uses four times --request-workers.",
    )
    parser.add_argument("--fsync-interval", type=int, default=100)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--normalize-queries", action="store_true")
    parser.add_argument(
        "--entry-point-id-mode",
        choices=("raw", "raw_plus_one", "copy_block"),
        default="copy_block",
    )
    parser.add_argument(
        "--result-id-mode",
        choices=("payload_source_id", "raw", "raw_minus_one", "copy_block"),
        default="payload_source_id",
    )
    parser.add_argument("--source-id-dedup-block-size", type=int)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if args.query_start < 0 or args.query_count <= 0:
        raise ValueError("query range must be non-negative and non-empty")
    if (
        args.top_k <= 0
        or args.request_workers <= 0
        or args.max_in_flight < 0
        or args.fsync_interval <= 0
        or args.timeout_seconds <= 0
    ):
        raise ValueError("top-k, workers, and timeout must be positive")
    if args.oracle_high_ef <= 0:
        raise ValueError("oracle-high-ef must be positive")
    if not args.run_id or Path(args.run_id).name != args.run_id:
        raise ValueError("run-id must be one safe path component")
    if args.entry_point_id_mode == "copy_block" or args.result_id_mode == "copy_block":
        if args.source_id_dedup_block_size is None:
            raise ValueError("copy_block ID mode requires --source-id-dedup-block-size")
    if (
        args.source_id_dedup_block_size is not None
        and args.source_id_dedup_block_size <= 1
    ):
        raise ValueError("source-id dedup block size must be greater than one")


def request_json(
    base_url: str,
    method: str,
    path: str,
    body: Mapping[str, Any] | None,
    *,
    timeout: float,
) -> tuple[dict[str, Any], int]:
    data = None
    headers = {"Accept": "application/json"}
    if body is not None:
        data = json.dumps(body, separators=(",", ":"), allow_nan=False).encode()
        headers["Content-Type"] = "application/json"
    parsed = urllib.parse.urlsplit(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError(f"invalid base URL: {base_url!r}")
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    target = f"{parsed.path.rstrip('/')}{path}" or "/"
    key = (parsed.scheme, parsed.hostname, port)
    headers["Connection"] = "keep-alive"
    connections = getattr(_HTTP_CONNECTIONS, "connections", None)
    if connections is None:
        connections = {}
        _HTTP_CONNECTIONS.connections = connections
    url = f"{base_url.rstrip('/')}{path}"
    for attempt in range(2):
        connection = connections.get(key)
        if connection is None:
            connection_type = (
                http.client.HTTPSConnection
                if parsed.scheme == "https"
                else http.client.HTTPConnection
            )
            connection = connection_type(parsed.hostname, port, timeout=timeout)
            connections[key] = connection
        elif connection.sock is not None:
            connection.sock.settimeout(timeout)
        try:
            connection.request(method, target, body=data, headers=headers)
            response = connection.getresponse()
            raw = response.read()
            status = response.status
            if response.will_close:
                connection.close()
                connections.pop(key, None)
            if status >= 400:
                detail = raw.decode("utf-8", errors="replace")
                raise RuntimeError(f"{method} {url} failed: {status} {detail}")
            return json.loads(raw), len(raw)
        except RuntimeError:
            raise
        except (OSError, http.client.HTTPException) as exc:
            connection.close()
            connections.pop(key, None)
            if attempt:
                raise RuntimeError(f"{method} {url} failed: {exc}") from exc
    raise AssertionError("unreachable HTTP retry state")


def normalize_rows(rows: np.ndarray) -> np.ndarray:
    values = np.asarray(rows, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return values / norms


def load_queries(
    hdf5_path: str | Path,
    query_start: int,
    query_count: int,
    *,
    normalize: bool,
) -> tuple[np.ndarray, int]:
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("c6_capture requires h5py") from exc
    with h5py.File(hdf5_path, "r") as handle:
        if "test" not in handle:
            raise ValueError("HDF5 dataset is missing test queries")
        stop = query_start + query_count
        if stop > len(handle["test"]):
            raise ValueError("query range exceeds the HDF5 test set")
        queries = handle["test"][query_start:stop].astype(np.float32, copy=True)
    if normalize:
        queries = normalize_rows(queries)
    queries = np.ascontiguousarray(queries, dtype=np.dtype("<f4"))
    return queries, int(queries.shape[1])


def query_bytes_sha256(queries: np.ndarray) -> str:
    values = np.ascontiguousarray(queries, dtype=np.dtype("<f4"))
    return hashlib.sha256(values.tobytes(order="C")).hexdigest()


def load_shard_key_map(path: str | Path, expected_shards: set[int]) -> dict[int, str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("shard-key map must be a JSON object")
    result: dict[int, str] = {}
    for raw_id, raw_key in payload.items():
        try:
            shard_id = int(raw_id)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid shard ID {raw_id!r}") from exc
        if str(shard_id) != str(raw_id) or shard_id < 0:
            raise ValueError(f"invalid canonical shard ID {raw_id!r}")
        if not isinstance(raw_key, str) or not raw_key:
            raise ValueError(f"invalid shard key for shard {shard_id}")
        result[shard_id] = raw_key
    missing = expected_shards - set(result)
    if missing:
        raise ValueError(
            f"shard-key map IDs differ: missing={sorted(missing)}, actual={sorted(result)}"
        )
    return result


def load_shard_base_url_map(
    path: str | Path | None, expected_shards: set[int]
) -> dict[int, str]:
    if path is None:
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("shard base-URL map must be a JSON object")
    result: dict[int, str] = {}
    for raw_id, raw_url in payload.items():
        try:
            shard_id = int(raw_id)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid shard base-URL ID {raw_id!r}") from exc
        if str(shard_id) != str(raw_id) or shard_id < 0:
            raise ValueError(f"invalid canonical shard base-URL ID {raw_id!r}")
        if not isinstance(raw_url, str) or not raw_url.startswith(("http://", "https://")):
            raise ValueError(f"invalid base URL for shard {shard_id}")
        result[shard_id] = raw_url.rstrip("/")
    missing = expected_shards - set(result)
    if missing:
        raise ValueError(f"shard base-URL map is missing shards: {sorted(missing)}")
    return result


def encode_entry_point(
    source_id: int,
    shard_id: int,
    *,
    mode: str,
    block_size: int | None,
) -> int:
    if source_id < 0 or shard_id < 0:
        raise ValueError("numeric entry-point IDs and shard IDs must be non-negative")
    if mode == "raw":
        return source_id
    if mode == "raw_plus_one":
        return source_id + 1
    assert mode == "copy_block" and block_size is not None
    return shard_id * block_size + source_id + 1


def decode_result_id(
    item: Mapping[str, Any],
    *,
    mode: str,
    block_size: int | None,
) -> int | str:
    raw_id = item.get("id")
    if mode == "payload_source_id":
        payload = item.get("payload")
        if not isinstance(payload, dict) or "source_id" not in payload:
            raise RuntimeError("search result is missing payload.source_id")
        return int(payload["source_id"])
    if isinstance(raw_id, str):
        if mode != "raw":
            raise RuntimeError("non-raw result ID modes require numeric point IDs")
        return raw_id
    if isinstance(raw_id, bool) or not isinstance(raw_id, int):
        raise RuntimeError("search result has an invalid point ID")
    if mode == "raw":
        return raw_id
    if mode == "raw_minus_one":
        if raw_id <= 0:
            raise RuntimeError("raw_minus_one result ID must be positive")
        return raw_id - 1
    assert mode == "copy_block" and block_size is not None
    if raw_id <= 0:
        raise RuntimeError("copy-block result ID must be positive")
    source_id = (raw_id - 1) % block_size
    if source_id >= block_size - 1:
        raise RuntimeError("copy-block result ID encodes the reserved block sentinel")
    return source_id


def parse_search_response(
    payload: Mapping[str, Any],
    *,
    query_id: int,
    shard_id: int,
    ef_search: int,
    dimension: int,
    latency_us: float,
    response_bytes: int,
    result_id_mode: str,
    block_size: int | None,
) -> LocalSearchTrace:
    usage = payload.get("usage")
    hardware = usage.get("hardware") if isinstance(usage, dict) else None
    if not isinstance(hardware, dict):
        raise RuntimeError("search response is missing usage.hardware")
    required = {"cpu", "graph_nodes_visited", "cpu_time_us", "cpu_wall_time_us"}
    missing = sorted(required - set(hardware))
    if missing:
        raise RuntimeError(f"search response is missing hardware fields: {missing}")
    raw_results = payload.get("result")
    if not isinstance(raw_results, list):
        raise RuntimeError("search response result must be an array")
    result_ids = tuple(
        decode_result_id(item, mode=result_id_mode, block_size=block_size)
        for item in raw_results
    )
    scores = tuple(float(item["score"]) for item in raw_results)
    if any(not math.isfinite(score) for score in scores):
        raise RuntimeError("search response contains a non-finite score")
    return LocalSearchTrace(
        query_id=query_id,
        shard_id=shard_id,
        ef_search=ef_search,
        result_ids=result_ids,
        result_scores=scores,
        distance_computations=distance_computations_from_usage(
            int(hardware["cpu"]), dimension
        ),
        nodes_visited=int(hardware["graph_nodes_visited"]),
        worker_cpu_time_us=int(hardware["cpu_time_us"]),
        worker_cpu_wall_time_us=int(hardware["cpu_wall_time_us"]),
        latency_us=latency_us,
        response_bytes=response_bytes,
    )


def search_one(
    args: argparse.Namespace,
    query_id: int,
    query: np.ndarray,
    dimension: int,
    shard_id: int,
    shard_key: str,
    entry_points: Sequence[int | str],
    ef_search: int,
) -> LocalSearchTrace:
    if not all(isinstance(point_id, int) and not isinstance(point_id, bool) for point_id in entry_points):
        raise ValueError("the custom-shard capture path currently requires numeric entry points")
    encoded_entry_points = [
        encode_entry_point(
            int(point_id),
            shard_id,
            mode=args.entry_point_id_mode,
            block_size=args.source_id_dedup_block_size,
        )
        for point_id in entry_points
    ]
    body: dict[str, Any] = {
        "vector": np.asarray(query, dtype=np.float32).tolist(),
        "limit": args.top_k,
        "with_payload": ["source_id"]
        if args.result_id_mode == "payload_source_id"
        else False,
        "with_vector": False,
        "shard_key": [shard_key],
        "params": {"hnsw_ef": ef_search, "exact": False},
    }
    if encoded_entry_points:
        body["hnsw_entry_points"] = encoded_entry_points
    if args.source_id_dedup_block_size is not None:
        body["source_id_dedup_block_size"] = args.source_id_dedup_block_size
    started = time.perf_counter_ns()
    payload, response_bytes = request_json(
        args.shard_base_urls.get(shard_id, args.base_url),
        "POST",
        f"/collections/{urllib.parse.quote(args.collection, safe='')}/points/search",
        body,
        timeout=args.timeout_seconds,
    )
    latency_us = (time.perf_counter_ns() - started) / 1_000.0
    return parse_search_response(
        payload,
        query_id=query_id,
        shard_id=shard_id,
        ef_search=ef_search,
        dimension=dimension,
        latency_us=latency_us,
        response_bytes=response_bytes,
        result_id_mode=args.result_id_mode,
        block_size=args.source_id_dedup_block_size,
    )


def trace_record(trace: LocalSearchTrace, *, official_query_id: int) -> dict[str, Any]:
    return {
        "query_id": trace.query_id,
        "official_query_id": official_query_id,
        "shard_id": trace.shard_id,
        "ef_search": trace.ef_search,
        "result_ids": list(trace.result_ids),
        "result_scores": list(trace.result_scores),
        "distance_computations": trace.distance_computations,
        "nodes_visited": trace.nodes_visited,
        "worker_cpu_time_us": trace.worker_cpu_time_us,
        "worker_cpu_wall_time_us": trace.worker_cpu_wall_time_us,
        "latency_us": trace.latency_us,
        "response_bytes": trace.response_bytes,
    }


def load_completed_keys(path: Path) -> set[tuple[int, int, int]]:
    if not path.is_file():
        return set()
    result: set[tuple[int, int, int]] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
                key = (int(row["query_id"]), int(row["shard_id"]), int(row["ef_search"]))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid result JSONL line {line_number}") from exc
            if key in result:
                raise ValueError(f"duplicate completed result key {key}")
            result.add(key)
    return result


def load_policy_matrix(
    path: str | Path | None, *, logical_shards: int
) -> list[Any]:
    if path is None:
        return []
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("policy matrix must be a JSON object")
    configs = []
    for policy in ("P0", "P1", "P2", "P3"):
        raw_rows = payload.get(policy, [])
        if not isinstance(raw_rows, list):
            raise ValueError(f"policy matrix {policy} entry must be an array")
        for raw in raw_rows:
            if not isinstance(raw, dict):
                raise ValueError(f"policy matrix {policy} candidate must be an object")
            row = dict(raw)
            if row.get("policy", policy) != policy:
                raise ValueError(f"policy matrix candidate is stored under the wrong key: {policy}")
            row["policy"] = policy
            config = policy_config_from_dict(row)
            config.validate(logical_shards)
            configs.append(config)
    return configs


def required_search_keys(
    routes: Sequence[Any],
    *,
    ef_grid: Sequence[int],
    oracle_high_ef: int,
    policy_configs: Sequence[Any],
    include_oracle_grid: bool = True,
) -> set[tuple[int, int, int]]:
    result: set[tuple[int, int, int]] = set()
    for route in routes:
        if include_oracle_grid:
            for shard_id in route.fixed_policy_shard_order():
                for ef_search in ef_grid:
                    result.add((route.query_id, shard_id, int(ef_search)))
                result.add((route.query_id, shard_id, int(oracle_high_ef)))
        for config in policy_configs:
            shards = policy_shards(route, config)
            efs = policy_efs(route, shards, config)
            result.update(
                (route.query_id, int(shard_id), int(ef_search))
                for shard_id, ef_search in zip(shards, efs, strict=True)
            )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    validate_args(args)
    route_trace_path = Path(args.route_trace).expanduser().resolve()
    route_trace = json.loads(route_trace_path.read_text(encoding="utf-8"))
    routes = route_queries_from_trace(route_trace)
    if len(routes) != args.query_count:
        raise ValueError("route trace query count differs from --query-count")
    queries, dimension = load_queries(
        args.hdf5_path,
        args.query_start,
        args.query_count,
        normalize=args.normalize_queries,
    )
    query_sha256 = query_bytes_sha256(queries)
    expected_query_sha256 = str((route_trace.get("queries") or {}).get("sha256") or "")
    if query_sha256 != expected_query_sha256:
        raise ValueError(
            "selected HDF5 query bytes differ from the frozen Orion route trace"
        )
    expected_shards = {
        shard_id for route in routes for shard_id in route.ranked_candidate_shards
    }
    shard_keys = load_shard_key_map(args.shard_key_map, expected_shards)
    args.shard_base_urls = load_shard_base_url_map(
        args.shard_base_url_map, expected_shards
    )
    logical_shards = int((route_trace.get("artifact") or {}).get("shard_count") or 0)
    if logical_shards <= 0:
        raise ValueError("route trace artifact is missing shard_count")
    policy_configs = load_policy_matrix(
        args.policy_matrix, logical_shards=logical_shards
    )
    all_requirements = required_search_keys(
        routes,
        ef_grid=args.ef_grid,
        oracle_high_ef=args.oracle_high_ef,
        policy_configs=policy_configs,
        include_oracle_grid=not args.skip_oracle_grid,
    )
    external_keys = set().union(
        *(load_completed_keys(Path(path)) for path in args.exclude_keys_from)
    ) if args.exclude_keys_from else set()
    externally_covered = all_requirements & external_keys
    requirements = all_requirements - externally_covered
    output_root = Path(args.output_dir).expanduser().resolve()
    run_dir = output_root / args.run_id
    if run_dir.exists() and not args.resume:
        raise FileExistsError(f"run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    results_path = run_dir / "local_search_traces.jsonl"
    completed = load_completed_keys(results_path) if args.resume else set()
    manifest_path = output_root / "manifest.jsonl"
    run_identity = {
        "record_type": "c6_capture_start",
        "protocol_version": 1,
        "timestamp": utc_timestamp(),
        "run_id": args.run_id,
        "base_url": args.base_url,
        "collection": args.collection,
        "route_trace": {
            "path": str(route_trace_path),
            "sha256": sha256_path(route_trace_path),
            "canonical_sha256": canonical_json_sha256(route_trace),
        },
        "dataset": {
            "path": str(Path(args.hdf5_path).expanduser().resolve()),
            "sha256": sha256_path(args.hdf5_path),
            "query_start": args.query_start,
            "query_count": args.query_count,
            "query_sha256": query_sha256,
            "dimension": dimension,
        },
        "ef_grid": list(args.ef_grid),
        "oracle_high_ef": args.oracle_high_ef,
        "oracle_grid_included": not args.skip_oracle_grid,
        "policy_matrix": (
            {
                "path": str(Path(args.policy_matrix).expanduser().resolve()),
                "sha256": sha256_path(args.policy_matrix),
                "candidate_count": len(policy_configs),
            }
            if args.policy_matrix
            else None
        ),
        "excluded_key_sources": [
            {
                "path": str(Path(path).expanduser().resolve()),
                "sha256": sha256_path(path),
            }
            for path in args.exclude_keys_from
        ],
        "externally_covered_searches": len(externally_covered),
        "shard_key_map_sha256": sha256_path(args.shard_key_map),
        "shard_base_url_map": (
            {
                "path": str(Path(args.shard_base_url_map).expanduser().resolve()),
                "sha256": sha256_path(args.shard_base_url_map),
            }
            if args.shard_base_url_map
            else None
        ),
        "entry_point_id_mode": args.entry_point_id_mode,
        "result_id_mode": args.result_id_mode,
        "source_id_dedup_block_size": args.source_id_dedup_block_size,
    }
    if not args.resume:
        append_jsonl(manifest_path, run_identity)
        write_json_atomic(run_dir / "run_manifest.json", run_identity)
    else:
        append_jsonl(
            manifest_path,
            {
                "record_type": "c6_capture_resume",
                "protocol_version": 1,
                "timestamp": utc_timestamp(),
                "run_id": args.run_id,
                "completed_searches_before_resume": len(completed),
                "shard_base_url_map": run_identity["shard_base_url_map"],
                "request_workers": args.request_workers,
                "max_in_flight": args.max_in_flight or args.request_workers * 4,
                "fsync_interval": args.fsync_interval,
            },
        )

    evidence_by_query = [route.evidence_by_shard() for route in routes]
    unexpected_completed = completed - requirements
    if unexpected_completed:
        first = min(unexpected_completed)
        raise ValueError(
            f"resume trace contains a key outside current requirements: {first}"
        )
    scheduled = len(requirements) - len(completed)
    max_in_flight = args.max_in_flight or args.request_workers * 4
    if max_in_flight < args.request_workers:
        raise ValueError("max-in-flight must be at least request-workers")
    remaining = (key for key in requirements if key not in completed)
    pending: dict[Future[LocalSearchTrace], tuple[int, int, int]] = {}

    def submit_key(
        executor: ThreadPoolExecutor, key: tuple[int, int, int]
    ) -> None:
        query_id, shard_id, ef_search = key
        evidence = evidence_by_query[query_id]
        future = executor.submit(
            search_one,
            args,
            query_id,
            queries[query_id],
            dimension,
            shard_id,
            shard_keys[shard_id],
            evidence[shard_id].entry_points if shard_id in evidence else (),
            ef_search,
        )
        pending[future] = key

    with ThreadPoolExecutor(max_workers=args.request_workers) as executor:
        for _ in range(min(max_in_flight, scheduled)):
            submit_key(executor, next(remaining))
        completed_now = 0
        with DurableJsonlBatchWriter(
            results_path, fsync_interval=args.fsync_interval
        ) as writer:
            while pending:
                done, _not_done = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    key = pending.pop(future)
                    trace = future.result()
                    if key != (trace.query_id, trace.shard_id, trace.ef_search):
                        raise RuntimeError(
                            "completed search identity differs from scheduled key"
                        )
                    writer.append(
                        trace_record(
                            trace,
                            official_query_id=args.query_start + trace.query_id,
                        )
                    )
                    completed_now += 1
                    try:
                        next_key = next(remaining)
                    except StopIteration:
                        pass
                    else:
                        submit_key(executor, next_key)
                    if completed_now % 1000 == 0 or completed_now == scheduled:
                        print(
                            f"captured {completed_now}/{scheduled} new local searches",
                            flush=True,
                        )

    total_completed = len(load_completed_keys(results_path))
    expected_total = len(requirements)
    if total_completed != expected_total:
        raise RuntimeError(
            f"capture is incomplete: actual={total_completed}, expected={expected_total}"
        )
    completion = {
        "record_type": "c6_capture_complete",
        "protocol_version": 1,
        "timestamp": utc_timestamp(),
        "run_id": args.run_id,
        "scheduled_new_searches": scheduled,
        "total_searches": total_completed,
        "expected_total_searches": expected_total,
        "externally_covered_searches": len(externally_covered),
        "total_requirement_searches": len(all_requirements),
        "results_path": str(results_path),
        "results_sha256": sha256_path(results_path),
        "shard_base_url_map_sha256": (
            sha256_path(args.shard_base_url_map) if args.shard_base_url_map else None
        ),
    }
    append_jsonl(manifest_path, completion)
    write_json_atomic(run_dir / "completion.json", completion)
    print(json.dumps(completion, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
