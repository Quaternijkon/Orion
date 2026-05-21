#!/usr/bin/env python3

from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import statistics
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import aiohttp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Concurrent load generator for Qdrant vector search. Uses HDF5 queries and "
            "reports throughput, latency, and optional Docker CPU samples."
        )
    )
    parser.add_argument("--base-url", default="http://localhost:6333", help="Qdrant base URL.")
    parser.add_argument("--collection", required=True, help="Collection name to query.")
    parser.add_argument(
        "--hdf5-path",
        required=True,
        help="Path to the HDF5 dataset that provides query vectors.",
    )
    parser.add_argument(
        "--hdf5-query-key",
        default="test",
        help="HDF5 dataset key for queries. Default: %(default)s",
    )
    parser.add_argument(
        "--query-count",
        type=int,
        default=1000,
        help="How many HDF5 queries to load and cycle through. Default: %(default)s",
    )
    parser.add_argument(
        "--total-requests",
        type=int,
        default=4000,
        help="Total number of search requests to send. Default: %(default)s",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=16,
        help="Number of concurrent client workers. Default: %(default)s",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Search limit. Default: %(default)s",
    )
    parser.add_argument(
        "--query-ef",
        type=int,
        default=0,
        help="HNSW ef to use. Ignored when --exact is set.",
    )
    parser.add_argument(
        "--exact",
        action="store_true",
        help="Use exact=true search instead of hnsw_ef.",
    )
    parser.add_argument(
        "--timeout-sec",
        type=float,
        default=60.0,
        help="Per-request timeout. Default: %(default)s",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle loaded queries once before the run.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed used when --shuffle is enabled. Default: %(default)s",
    )
    parser.add_argument(
        "--docker-container",
        default="",
        help="Optional Docker container name for CPU sampling during the run.",
    )
    parser.add_argument(
        "--docker-sample-interval-sec",
        type=float,
        default=1.0,
        help="Docker CPU sample interval. Default: %(default)s",
    )
    parser.add_argument(
        "--output-json",
        default="",
        help="Optional path to write the result JSON.",
    )
    return parser.parse_args()


def percentile(values: list[float], percent: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (percent / 100.0) * (len(sorted_values) - 1)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return sorted_values[int(rank)]
    lower_value = sorted_values[lower]
    upper_value = sorted_values[upper]
    weight = rank - lower
    return lower_value + (upper_value - lower_value) * weight


def parse_cpu_percent(text: str) -> float:
    return float(text.strip().rstrip("%"))


def load_queries(hdf5_path: str, query_key: str, query_count: int) -> list[list[float]]:
    import h5py
    import numpy as np

    path = Path(hdf5_path)
    if not path.exists():
        raise FileNotFoundError(f"HDF5 dataset not found: {path}")

    with h5py.File(path, "r") as handle:
        query_set = handle[query_key]
        if query_count <= 0 or query_count > int(query_set.shape[0]):
            raise ValueError(
                f"--query-count must be within 1..{int(query_set.shape[0])} for query split {query_key}"
            )
        queries = query_set[:query_count].astype("float32", copy=True)

    # glove-200-angular should be normalized for cosine / angular search
    norms = np.linalg.norm(queries, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    queries = queries / norms
    return [row.tolist() for row in queries]


@dataclass
class StressResult:
    base_url: str
    collection: str
    exact: bool
    query_ef: int | None
    concurrency: int
    total_requests: int
    query_count: int
    throughput_rps: float
    latency_mean_ms: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    server_time_mean_ms: float
    wall_time_sec: float
    docker_cpu_mean_pct: float | None
    docker_cpu_max_pct: float | None
    docker_samples: int


async def docker_cpu_sampler(
    container_name: str,
    interval_sec: float,
    stop_event: asyncio.Event,
    samples: list[float],
) -> None:
    if not container_name:
        return

    while not stop_event.is_set():
        try:
            proc = await asyncio.create_subprocess_exec(
                "docker",
                "stats",
                "--no-stream",
                container_name,
                "--format",
                "{{.CPUPerc}}",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _stderr = await proc.communicate()
            if proc.returncode == 0:
                text = stdout.decode().strip()
                if text:
                    samples.append(parse_cpu_percent(text))
        except Exception:
            pass

        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval_sec)
        except asyncio.TimeoutError:
            continue


async def run_stress(args: argparse.Namespace) -> StressResult:
    queries = load_queries(args.hdf5_path, args.hdf5_query_key, args.query_count)
    if args.shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(queries)

    latencies_ms: list[float] = []
    server_times_ms: list[float] = []
    cpu_samples: list[float] = []
    counter_lock = asyncio.Lock()
    request_counter = {"value": 0}

    stop_event = asyncio.Event()
    sampler_task = None
    if args.docker_container:
        sampler_task = asyncio.create_task(
            docker_cpu_sampler(
                args.docker_container,
                args.docker_sample_interval_sec,
                stop_event,
                cpu_samples,
            )
        )

    timeout = aiohttp.ClientTimeout(total=args.timeout_sec)
    connector = aiohttp.TCPConnector(limit=args.concurrency * 2, ttl_dns_cache=300)

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        async def worker(worker_id: int) -> None:
            while True:
                async with counter_lock:
                    index = request_counter["value"]
                    if index >= args.total_requests:
                        return
                    request_counter["value"] += 1

                query = queries[index % len(queries)]
                if args.exact:
                    params: dict[str, Any] = {"exact": True}
                else:
                    params = {"hnsw_ef": args.query_ef}

                body = {
                    "vector": query,
                    "limit": args.top_k,
                    "params": params,
                    "with_payload": False,
                    "with_vector": False,
                }

                started = time.perf_counter()
                async with session.post(
                    f"{args.base_url.rstrip('/')}/collections/{args.collection}/points/search",
                    json=body,
                ) as response:
                    response.raise_for_status()
                    payload = await response.json()
                latency_ms = (time.perf_counter() - started) * 1000.0
                latencies_ms.append(latency_ms)
                server_times_ms.append(float(payload.get("time", 0.0)) * 1000.0)

        started = time.perf_counter()
        await asyncio.gather(*(worker(i) for i in range(args.concurrency)))
        wall_time_sec = time.perf_counter() - started

    stop_event.set()
    if sampler_task is not None:
        await sampler_task

    return StressResult(
        base_url=args.base_url,
        collection=args.collection,
        exact=args.exact,
        query_ef=None if args.exact else args.query_ef,
        concurrency=args.concurrency,
        total_requests=args.total_requests,
        query_count=args.query_count,
        throughput_rps=(args.total_requests / wall_time_sec) if wall_time_sec > 0 else 0.0,
        latency_mean_ms=statistics.mean(latencies_ms) if latencies_ms else 0.0,
        latency_p50_ms=percentile(latencies_ms, 50),
        latency_p95_ms=percentile(latencies_ms, 95),
        latency_p99_ms=percentile(latencies_ms, 99),
        server_time_mean_ms=statistics.mean(server_times_ms) if server_times_ms else 0.0,
        wall_time_sec=wall_time_sec,
        docker_cpu_mean_pct=statistics.mean(cpu_samples) if cpu_samples else None,
        docker_cpu_max_pct=max(cpu_samples) if cpu_samples else None,
        docker_samples=len(cpu_samples),
    )


def main() -> int:
    args = parse_args()
    if args.concurrency <= 0:
        raise ValueError("--concurrency must be greater than 0")
    if args.total_requests <= 0:
        raise ValueError("--total-requests must be greater than 0")
    if args.query_count <= 0:
        raise ValueError("--query-count must be greater than 0")
    if not args.exact and args.query_ef <= 0:
        raise ValueError("--query-ef must be greater than 0 unless --exact is used")

    result = asyncio.run(run_stress(args))
    result_json = json.dumps(asdict(result), indent=2)
    print(result_json)

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(result_json + "\n", encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
