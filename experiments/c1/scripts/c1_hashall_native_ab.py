#!/usr/bin/env python3
"""Saturating native HashAll benchmark using standard coordinator batch requests."""

from __future__ import annotations

import argparse
import concurrent.futures
import http.client
import json
import statistics
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path

import h5py
import numpy as np


def request_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=30.0) as response:
        return json.load(response)


def load_queries(path: Path, query_count: int, top_k: int) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as handle:
        queries = np.asarray(handle["test"][:query_count], dtype=np.float32)
        neighbors = np.asarray(handle["neighbors"][:query_count, :top_k], dtype=np.int64)
    return queries, neighbors


def make_bodies(
    queries: np.ndarray,
    *,
    batch_size: int,
    top_k: int,
    hnsw_ef: int,
) -> list[bytes]:
    bodies: list[bytes] = []
    for start in range(0, len(queries), batch_size):
        rows = queries[start : start + batch_size]
        searches = [
            {
                "vector": row.tolist(),
                "limit": top_k,
                "with_payload": False,
                "with_vector": False,
                "params": {"hnsw_ef": hnsw_ef, "exact": False},
            }
            for row in rows
        ]
        bodies.append(
            json.dumps(
                {"searches": searches},
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        )
    return bodies


def post_batch(host: str, port: int, path: str, body: bytes) -> bytes:
    connection = http.client.HTTPConnection(host, port, timeout=300.0)
    try:
        connection.request(
            "POST",
            path,
            body=body,
            headers={"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        payload = response.read()
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status}: {payload[:1000]!r}")
        return payload
    finally:
        connection.close()


def recall_probe(
    host: str,
    port: int,
    path: str,
    bodies: list[bytes],
    neighbors: np.ndarray,
    *,
    batch_size: int,
    top_k: int,
) -> dict:
    result_ids: list[list[int]] = []
    for body in bodies:
        payload = json.loads(post_batch(host, port, path, body))
        rows = payload.get("result")
        if not isinstance(rows, list):
            raise RuntimeError("search batch response has no result rows")
        result_ids.extend(
            [[int(point["id"]) for point in row[:top_k]] for row in rows]
        )
    result_ids = result_ids[: len(neighbors)]
    hits = sum(
        len(set(ids).intersection(int(value) for value in truth[:top_k]))
        for ids, truth in zip(result_ids, neighbors, strict=True)
    )
    return {
        "query_count": len(result_ids),
        "batch_size": batch_size,
        "hits": hits,
        "recall_at_k": hits / (len(result_ids) * top_k),
    }


def timed_run(
    host: str,
    port: int,
    path: str,
    bodies: list[bytes],
    *,
    concurrency: int,
    duration_s: float,
    batch_size: int,
) -> dict:
    barrier = threading.Barrier(concurrency + 1)

    def worker(worker_id: int) -> tuple[int, int]:
        connection = http.client.HTTPConnection(host, port, timeout=300.0)
        completed = 0
        response_bytes = 0
        body_index = worker_id % len(bodies)
        barrier.wait()
        deadline = time.perf_counter() + duration_s
        try:
            while time.perf_counter() < deadline:
                body = bodies[body_index]
                body_index = (body_index + concurrency) % len(bodies)
                connection.request(
                    "POST",
                    path,
                    body=body,
                    headers={"Content-Type": "application/json"},
                )
                response = connection.getresponse()
                payload = response.read()
                if response.status != 200:
                    raise RuntimeError(f"HTTP {response.status}: {payload[:1000]!r}")
                completed += 1
                response_bytes += len(payload)
        finally:
            connection.close()
        return completed, response_bytes

    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(worker, worker_id) for worker_id in range(concurrency)]
        barrier.wait()
        rows = [future.result() for future in futures]
    wall_s = time.perf_counter() - started
    completed_batches = sum(row[0] for row in rows)
    query_count = completed_batches * batch_size
    return {
        "wall_s": wall_s,
        "completed_batches": completed_batches,
        "query_count": query_count,
        "qps": query_count / wall_s,
        "response_bytes": sum(row[1] for row in rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--collection", required=True)
    parser.add_argument("--hdf5-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--machine-count", type=int, required=True)
    parser.add_argument("--hnsw-ef", type=int, default=24)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--query-count", type=int, default=10_000)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--warmup-seconds", type=float, default=5.0)
    parser.add_argument("--measure-seconds", type=float, default=15.0)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    parsed = urllib.parse.urlparse(args.base_url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 6333
    endpoint = (
        f"/collections/{urllib.parse.quote(args.collection, safe='')}/points/search/batch"
    )
    queries, neighbors = load_queries(args.hdf5_path, args.query_count, args.top_k)
    bodies = make_bodies(
        queries,
        batch_size=args.batch_size,
        top_k=args.top_k,
        hnsw_ef=args.hnsw_ef,
    )
    if any(len(queries[start : start + args.batch_size]) != args.batch_size for start in range(0, len(queries), args.batch_size)):
        raise ValueError("query_count must be divisible by batch_size")

    recall = recall_probe(
        host,
        port,
        endpoint,
        bodies[: max(1, 1000 // args.batch_size)],
        neighbors[:1000],
        batch_size=args.batch_size,
        top_k=args.top_k,
    )
    warmup = timed_run(
        host,
        port,
        endpoint,
        bodies,
        concurrency=args.concurrency,
        duration_s=args.warmup_seconds,
        batch_size=args.batch_size,
    )
    repeats = [
        timed_run(
            host,
            port,
            endpoint,
            bodies,
            concurrency=args.concurrency,
            duration_s=args.measure_seconds,
            batch_size=args.batch_size,
        )
        for _ in range(args.repeats)
    ]
    qps_values = [row["qps"] for row in repeats]
    record = {
        "record_type": "native_hashall_same_index_ab",
        "variant": args.variant,
        "machine_count": args.machine_count,
        "base_url": args.base_url,
        "collection": args.collection,
        "request_contract": {
            "sharding_method": "auto",
            "standard_coordinator_request": True,
            "shard_selector_present": False,
            "client_side_fanout": False,
            "endpoint": endpoint,
        },
        "parameters": {
            "hnsw_ef": args.hnsw_ef,
            "top_k": args.top_k,
            "batch_size": args.batch_size,
            "query_count_source": args.query_count,
            "concurrency": args.concurrency,
            "warmup_seconds": args.warmup_seconds,
            "measure_seconds": args.measure_seconds,
            "repeats": args.repeats,
        },
        "server": request_json(args.base_url + "/"),
        "collection_info": request_json(
            args.base_url
            + f"/collections/{urllib.parse.quote(args.collection, safe='')}"
        )["result"],
        "collection_cluster": request_json(
            args.base_url
            + f"/collections/{urllib.parse.quote(args.collection, safe='')}/cluster"
        )["result"],
        "recall_probe": recall,
        "warmup": warmup,
        "repeats": repeats,
        "qps_mean": statistics.fmean(qps_values),
        "qps_stdev": statistics.stdev(qps_values) if len(qps_values) > 1 else 0.0,
        "generated_at_unix": time.time(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "variant": args.variant,
        "machine_count": args.machine_count,
        "recall_at_10": recall["recall_at_k"],
        "qps_mean": record["qps_mean"],
        "qps_stdev": record["qps_stdev"],
        "output": str(args.output),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
