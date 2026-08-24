#!/usr/bin/env python3
"""Calibrate ef and measure saturated native HashAll throughput at fixed recall."""

from __future__ import annotations

import argparse
import concurrent.futures
import http.client
import json
import statistics
import subprocess
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


def load_dataset(path: Path, top_k: int) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as handle:
        queries = np.asarray(handle["test"], dtype=np.float32)
        neighbors = np.asarray(handle["neighbors"][:, :top_k], dtype=np.int64)
    if len(queries) < 10_000 or len(neighbors) < 10_000:
        raise ValueError("the fixed-recall protocol requires at least 10,000 queries")
    return queries[:10_000], neighbors[:10_000]


def make_bodies(
    queries: np.ndarray,
    *,
    batch_size: int,
    top_k: int,
    hnsw_ef: int,
) -> list[bytes]:
    if len(queries) % batch_size:
        raise ValueError("query count must be divisible by batch size")
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
                {"searches": searches}, separators=(",", ":"), allow_nan=False
            ).encode("utf-8")
        )
    return bodies


def post_batch(host: str, port: int, path: str, body: bytes) -> bytes:
    connection = http.client.HTTPConnection(host, port, timeout=300.0)
    try:
        connection.request(
            "POST", path, body=body, headers={"Content-Type": "application/json"}
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
    queries: np.ndarray,
    neighbors: np.ndarray,
    *,
    batch_size: int,
    top_k: int,
    hnsw_ef: int,
) -> dict:
    result_ids: list[list[int]] = []
    for body in make_bodies(
        queries, batch_size=batch_size, top_k=top_k, hnsw_ef=hnsw_ef
    ):
        payload = json.loads(post_batch(host, port, path, body))
        rows = payload.get("result")
        if not isinstance(rows, list):
            raise RuntimeError("search batch response has no result rows")
        result_ids.extend(
            [[int(point["id"]) for point in row[:top_k]] for row in rows]
        )
    hits = sum(
        len(set(ids).intersection(int(value) for value in truth[:top_k]))
        for ids, truth in zip(result_ids, neighbors, strict=True)
    )
    return {
        "hnsw_ef": hnsw_ef,
        "query_count": len(result_ids),
        "hits": hits,
        "recall_at_10": hits / (len(result_ids) * top_k),
    }


def parse_int_csv(value: str) -> list[int]:
    result = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not result:
        raise ValueError("expected at least one integer")
    return result


def cpu_usage_usec(host: str, local_host: str, container: str) -> int:
    docker_command = [
        "sudo",
        "-n",
        "docker",
        "exec",
        container,
        "cat",
        "/sys/fs/cgroup/cpu.stat",
    ]
    command = docker_command if host == local_host else ["ssh", host, *docker_command]
    output = subprocess.run(
        command, check=True, capture_output=True, text=True, timeout=30.0
    ).stdout
    for line in output.splitlines():
        fields = line.split()
        if len(fields) == 2 and fields[0] == "usage_usec":
            return int(fields[1])
    raise RuntimeError(f"usage_usec absent for {host}/{container}")


def cpu_snapshot(
    node_hosts: list[str], local_host: str, containers: list[str]
) -> dict[str, int]:
    if len(node_hosts) != len(containers):
        raise ValueError("node host and container counts differ")
    return {
        host: cpu_usage_usec(host, local_host, container)
        for host, container in zip(node_hosts, containers, strict=True)
    }


def percentiles(values: list[float]) -> dict[str, float]:
    data = np.asarray(values, dtype=np.float64)
    return {
        "p50": float(np.percentile(data, 50)),
        "p95": float(np.percentile(data, 95)),
        "p99": float(np.percentile(data, 99)),
        "max": float(np.max(data)),
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
    node_hosts: list[str],
    local_host: str,
    containers: list[str],
) -> dict:
    barrier = threading.Barrier(concurrency + 1)

    def worker(worker_id: int) -> tuple[int, int, list[float]]:
        connection = http.client.HTTPConnection(host, port, timeout=300.0)
        completed = 0
        response_bytes = 0
        latencies_ms: list[float] = []
        body_index = worker_id % len(bodies)
        barrier.wait()
        deadline = time.perf_counter() + duration_s
        try:
            while time.perf_counter() < deadline:
                body = bodies[body_index]
                body_index = (body_index + concurrency) % len(bodies)
                request_started = time.perf_counter()
                connection.request(
                    "POST",
                    path,
                    body=body,
                    headers={"Content-Type": "application/json"},
                )
                response = connection.getresponse()
                payload = response.read()
                latencies_ms.append((time.perf_counter() - request_started) * 1000.0)
                if response.status != 200:
                    raise RuntimeError(f"HTTP {response.status}: {payload[:1000]!r}")
                completed += 1
                response_bytes += len(payload)
        finally:
            connection.close()
        return completed, response_bytes, latencies_ms

    cpu_before = cpu_snapshot(node_hosts, local_host, containers)
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(worker, worker_id) for worker_id in range(concurrency)]
        barrier.wait()
        rows = [future.result() for future in futures]
    wall_s = time.perf_counter() - started
    cpu_after = cpu_snapshot(node_hosts, local_host, containers)
    completed_batches = sum(row[0] for row in rows)
    query_count = completed_batches * batch_size
    latency_values = [value for row in rows for value in row[2]]
    cpu_delta = {host: cpu_after[host] - cpu_before[host] for host in node_hosts}
    cpu_cores = {
        host: delta / (wall_s * 1_000_000.0) for host, delta in cpu_delta.items()
    }
    return {
        "wall_s": wall_s,
        "completed_batches": completed_batches,
        "query_count": query_count,
        "qps": query_count / wall_s,
        "response_bytes": sum(row[1] for row in rows),
        "batch_latency_ms": percentiles(latency_values),
        "cpu_usage_delta_usec": cpu_delta,
        "cpu_average_cores": cpu_cores,
        "cpu_average_cores_total": sum(cpu_cores.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--collection", required=True)
    parser.add_argument("--hdf5-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--machine-count", type=int, required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--target-recall", type=float, default=0.90)
    parser.add_argument("--tuning-margin", type=float, default=0.002)
    parser.add_argument(
        "--ef-candidates", default="10,12,14,16,18,20,22,24,28,32,40,48,64"
    )
    parser.add_argument("--concurrency-candidates", default="1,2,4,8")
    parser.add_argument("--sweep-seconds", type=float, default=6.0)
    parser.add_argument("--warmup-seconds", type=float, default=10.0)
    parser.add_argument("--measure-seconds", type=float, default=20.0)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--node-hosts", default="10.10.1.1,10.10.1.2,10.10.1.3,10.10.1.4"
    )
    parser.add_argument("--local-host", default="10.10.1.1")
    parser.add_argument(
        "--containers",
        default="hashall-ab-node0,hashall-ab-node1,hashall-ab-node2,hashall-ab-node3",
    )
    args = parser.parse_args()

    ef_candidates = parse_int_csv(args.ef_candidates)
    if min(ef_candidates) < args.top_k or max(ef_candidates) > 64:
        raise ValueError("ef candidates must remain within [top_k, 64]")
    concurrency_candidates = parse_int_csv(args.concurrency_candidates)
    node_hosts = [item.strip() for item in args.node_hosts.split(",") if item.strip()]
    containers = [item.strip() for item in args.containers.split(",") if item.strip()]

    parsed = urllib.parse.urlparse(args.base_url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 6333
    endpoint = (
        f"/collections/{urllib.parse.quote(args.collection, safe='')}/points/search/batch"
    )
    queries, neighbors = load_dataset(args.hdf5_path, args.top_k)
    tuning_queries, tuning_neighbors = queries[:1000], neighbors[:1000]
    heldout_queries, heldout_neighbors = queries[1000:10_000], neighbors[1000:10_000]

    tuning_sweep: list[dict] = []
    selected_ef: int | None = None
    tuning_target = args.target_recall + args.tuning_margin
    for hnsw_ef in ef_candidates:
        row = recall_probe(
            host,
            port,
            endpoint,
            tuning_queries,
            tuning_neighbors,
            batch_size=args.batch_size,
            top_k=args.top_k,
            hnsw_ef=hnsw_ef,
        )
        tuning_sweep.append(row)
        if selected_ef is None and row["recall_at_10"] >= tuning_target:
            selected_ef = hnsw_ef
    if selected_ef is None:
        raise RuntimeError(f"no ef candidate reached tuning recall {tuning_target}")

    selected_index = ef_candidates.index(selected_ef)
    heldout: dict | None = None
    selection_reason = "smallest ef reaching tuning target with safety margin"
    while selected_index < len(ef_candidates):
        selected_ef = ef_candidates[selected_index]
        heldout = recall_probe(
            host,
            port,
            endpoint,
            heldout_queries,
            heldout_neighbors,
            batch_size=args.batch_size,
            top_k=args.top_k,
            hnsw_ef=selected_ef,
        )
        if heldout["recall_at_10"] >= args.target_recall:
            break
        selection_reason = "advanced to next bounded ef because held-out recall missed target"
        selected_index += 1
    if heldout is None or heldout["recall_at_10"] < args.target_recall:
        raise RuntimeError("bounded ef candidates did not pass held-out recall")

    throughput_bodies = make_bodies(
        heldout_queries,
        batch_size=args.batch_size,
        top_k=args.top_k,
        hnsw_ef=selected_ef,
    )
    concurrency_sweep = []
    for concurrency in concurrency_candidates:
        row = timed_run(
            host,
            port,
            endpoint,
            throughput_bodies,
            concurrency=concurrency,
            duration_s=args.sweep_seconds,
            batch_size=args.batch_size,
            node_hosts=node_hosts,
            local_host=args.local_host,
            containers=containers,
        )
        row["concurrency"] = concurrency
        concurrency_sweep.append(row)
    best_sweep_qps = max(row["qps"] for row in concurrency_sweep)
    selected_concurrency = min(
        row["concurrency"]
        for row in concurrency_sweep
        if row["qps"] >= best_sweep_qps * 0.98
    )

    warmup = timed_run(
        host,
        port,
        endpoint,
        throughput_bodies,
        concurrency=selected_concurrency,
        duration_s=args.warmup_seconds,
        batch_size=args.batch_size,
        node_hosts=node_hosts,
        local_host=args.local_host,
        containers=containers,
    )
    repeats = [
        timed_run(
            host,
            port,
            endpoint,
            throughput_bodies,
            concurrency=selected_concurrency,
            duration_s=args.measure_seconds,
            batch_size=args.batch_size,
            node_hosts=node_hosts,
            local_host=args.local_host,
            containers=containers,
        )
        for _ in range(args.repeats)
    ]
    qps_values = [row["qps"] for row in repeats]
    qps_mean = statistics.fmean(qps_values)
    qps_stdev = statistics.stdev(qps_values) if len(qps_values) > 1 else 0.0
    record = {
        "record_type": "native_hashall_fixed_recall",
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
        "protocol": {
            "dataset": "SIFT1M",
            "tuning_query_range": [0, 1000],
            "heldout_query_range": [1000, 10000],
            "target_recall_at_10": args.target_recall,
            "tuning_margin": args.tuning_margin,
            "ef_lower_bound": args.top_k,
            "ef_upper_bound": 64,
            "selection_rule": selection_reason,
        },
        "parameters": {
            "selected_hnsw_ef": selected_ef,
            "top_k": args.top_k,
            "batch_size": args.batch_size,
            "selected_concurrency": selected_concurrency,
            "sweep_seconds": args.sweep_seconds,
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
        "tuning_sweep": tuning_sweep,
        "heldout_recall": heldout,
        "concurrency_sweep": concurrency_sweep,
        "warmup": warmup,
        "repeats": repeats,
        "qps_mean": qps_mean,
        "qps_stdev": qps_stdev,
        "qps_cv": qps_stdev / qps_mean,
        "generated_at_unix": time.time(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    print(
        json.dumps(
            {
                "variant": args.variant,
                "machine_count": args.machine_count,
                "selected_hnsw_ef": selected_ef,
                "tuning_recall_at_10": next(
                    row["recall_at_10"]
                    for row in tuning_sweep
                    if row["hnsw_ef"] == selected_ef
                ),
                "heldout_recall_at_10": heldout["recall_at_10"],
                "selected_concurrency": selected_concurrency,
                "qps_mean": qps_mean,
                "qps_stdev": qps_stdev,
                "qps_cv": record["qps_cv"],
                "output": str(args.output),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
