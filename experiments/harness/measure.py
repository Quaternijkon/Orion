"""Fixed-recall throughput measurement for the containerized Qdrant cluster.

Design constraints, all of which exist because the previous harness violated
them and produced a client-bound artifact (see ../STAGE0.md):

* The server coordinates fan-out. The client issues one request per query (or
  per batch) against a distributed collection, so client cost per query does
  not grow with the number of shards touched.
* Request bodies are serialized before the timer starts. The measured path only
  writes prepared bytes and reads response bytes.
* Responses are parsed, and recall is scored, after the timer stops. Recall
  never appears in the measured path.
* Offered concurrency is the number of in-flight requests, independent of how
  many client processes exist.
* The load generator runs on cores disjoint from every peer cpuset, and the
  run refuses to start otherwise.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import multiprocessing as mp
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import orjson

from cluster import (
    Peer,
    TestbedError,
    assert_disjoint,
    assert_symmetric,
    discover_peers,
    host_busy_usec,
    process_cpu_usec,
)

WORKER_STARTED_TIMEOUT_S = 120.0


@dataclass(frozen=True)
class Gates:
    """Validity gates from STAGE0.md.

    G1 asks whether *some* server component is saturated, which is what makes a
    throughput reading a statement about the server. It deliberately does not
    require every peer to be saturated: in scatter-gather the coordinator role
    carries extra work, so uniform saturation is not achievable by construction.
    Load imbalance is reported separately as a diagnostic, not as a validity
    condition.
    """

    server_saturation_pct: float
    least_loaded_peer_pct: float
    peer_imbalance_ratio: float
    client_headroom_pct: float
    host_interference_pct: float
    g1_server_saturation: bool
    g2_client_headroom: bool
    g4_host_interference: bool

    @property
    def passed(self) -> bool:
        # G3 spans two runs and is evaluated by the Stage 0 runner, not here.
        return (
            self.g1_server_saturation
            and self.g2_client_headroom
            and self.g4_host_interference
        )


def build_request_bodies(
    queries: np.ndarray,
    *,
    collection: str,
    top_k: int,
    hnsw_ef: int,
    batch: int,
) -> tuple[list[bytes], str, list[tuple[int, ...]]]:
    """Pre-serialize every request. Returns bodies, url path, and query id groups."""
    vectors = queries.astype(np.float32).tolist()
    bodies: list[bytes] = []
    groups: list[tuple[int, ...]] = []
    if batch == 1:
        path = f"/collections/{collection}/points/search"
        for index, vector in enumerate(vectors):
            bodies.append(
                orjson.dumps(
                    {
                        "vector": vector,
                        "limit": top_k,
                        "params": {"hnsw_ef": hnsw_ef},
                        "with_payload": False,
                        "with_vector": False,
                    }
                )
            )
            groups.append((index,))
        return bodies, path, groups

    path = f"/collections/{collection}/points/search/batch"
    for start in range(0, len(vectors), batch):
        window = vectors[start : start + batch]
        bodies.append(
            orjson.dumps(
                {
                    "searches": [
                        {
                            "vector": vector,
                            "limit": top_k,
                            "params": {"hnsw_ef": hnsw_ef},
                            "with_payload": False,
                            "with_vector": False,
                        }
                        for vector in window
                    ]
                }
            )
        )
        groups.append(tuple(range(start, start + len(window))))
    return bodies, path, groups


def parse_ids(payloads: Sequence[bytes], batch: int, top_k: int) -> list[list[int]]:
    """Extract result ids from raw responses. Runs after the timer stops."""
    results: list[list[int]] = []
    for payload in payloads:
        document = orjson.loads(payload)
        if "result" not in document:
            raise RuntimeError(f"search failed: {document}")
        result = document["result"]
        hit_lists = result if batch > 1 else [result]
        for hits in hit_lists:
            ids = [int(hit["id"]) for hit in hits]
            if len(ids) > top_k:
                ids = ids[:top_k]
            results.append(ids)
    return results


async def drive_requests(
    *,
    endpoints: Sequence[str],
    path: str,
    bodies: Sequence[bytes],
    inflight: int,
    warmup_requests: int,
) -> tuple[list[bytes], list[float], float, float]:
    """Issue every prepared body, returning raw responses and latencies."""
    import aiohttp

    responses: list[bytes | None] = [None] * len(bodies)
    latencies_us: list[float] = [0.0] * len(bodies)
    headers = {"Content-Type": "application/json"}
    next_index = 0
    lock = asyncio.Lock()
    connector = aiohttp.TCPConnector(limit=0, ttl_dns_cache=3600)
    timeout = aiohttp.ClientTimeout(total=120)

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        urls = [f"{endpoint}{path}" for endpoint in endpoints]

        async def warm(worker: int) -> None:
            url = urls[worker % len(urls)]
            for offset in range(worker, warmup_requests, inflight):
                async with session.post(
                    url, data=bodies[offset % len(bodies)], headers=headers
                ) as response:
                    await response.read()

        if warmup_requests > 0:
            await asyncio.gather(*(warm(worker) for worker in range(inflight)))

        async def run(worker: int) -> None:
            nonlocal next_index
            url = urls[worker % len(urls)]
            while True:
                async with lock:
                    index = next_index
                    next_index += 1
                if index >= len(bodies):
                    return
                started = time.perf_counter_ns()
                async with session.post(
                    url, data=bodies[index], headers=headers
                ) as response:
                    payload = await response.read()
                latencies_us[index] = (time.perf_counter_ns() - started) / 1_000.0
                responses[index] = payload

        wall_started = time.perf_counter()
        await asyncio.gather(*(run(worker) for worker in range(inflight)))
        wall_elapsed = time.perf_counter() - wall_started

    if any(payload is None for payload in responses):
        raise RuntimeError("a request produced no response")
    return [payload for payload in responses if payload is not None], (
        latencies_us
    ), wall_started, wall_elapsed


def pad_ids(ids: Sequence[Sequence[int]], top_k: int) -> np.ndarray:
    """Pack ragged result lists into a rectangular array, padding with -1."""
    packed = np.full((len(ids), top_k), -1, dtype=np.int64)
    for row, hits in enumerate(ids):
        packed[row, : len(hits)] = hits
    return packed


def worker_main(
    worker_id: int,
    cores: Sequence[int],
    config: dict[str, Any],
    body_slice: tuple[int, int],
    ready: Any,
    start_signal: Any,
    result_queue: Any,
) -> None:
    """Run one client process: pin cores, drive its slice, return a summary."""
    try:
        os.sched_setaffinity(0, set(cores))
        queries = np.load(config["queries_path"], mmap_mode="r")
        start, stop = body_slice
        bodies, path, groups = build_request_bodies(
            np.asarray(queries[start:stop]),
            collection=config["collection"],
            top_k=config["top_k"],
            hnsw_ef=config["hnsw_ef"],
            batch=config["batch"],
        )
        repeat = max(int(config["repeat"]), 1)
        if repeat > 1:
            bodies = bodies * repeat
            groups = groups * repeat
        ready.put((worker_id, os.getpid()))
        start_signal.wait()  # the coordinator samples CPU counters before this fires

        payloads, latencies_us, _, wall_elapsed = asyncio.run(
            drive_requests(
                endpoints=config["endpoints"],
                path=path,
                bodies=bodies,
                inflight=config["inflight_per_worker"],
                warmup_requests=config["warmup_per_worker"],
            )
        )
        ids = parse_ids(payloads, config["batch"], config["top_k"])
        offsets = [start + index for group in groups for index in group]
        output_dir = Path(config["output_dir"])
        np.save(output_dir / f"ids_worker{worker_id}.npy", pad_ids(ids, config["top_k"]))
        np.save(
            output_dir / f"offsets_worker{worker_id}.npy",
            np.asarray(offsets, dtype=np.int64),
        )
        result_queue.put(
            {
                "worker_id": worker_id,
                "requests": len(bodies),
                "queries": len(ids),
                "wall_elapsed_s": wall_elapsed,
                "latencies_us": latencies_us,
            }
        )
    except BaseException as error:  # noqa: BLE001 - must reach the coordinator
        import traceback

        result_queue.put(
            {
                "worker_id": worker_id,
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            }
        )
        ready.put((worker_id, os.getpid()))


def evaluate_gates(
    *,
    peers: Sequence[Peer],
    peer_usage_delta_usec: dict[str, int],
    client_usage_delta_usec: int,
    host_busy_delta_usec: int,
    elapsed_us: float,
    client_core_count: int,
) -> tuple[Gates, dict[str, float]]:
    per_peer_pct = {
        peer.name: peer_usage_delta_usec[peer.name] / elapsed_us / len(peer.cores) * 100
        for peer in peers
    }
    min_peer_pct = min(per_peer_pct.values())
    max_peer_pct = max(per_peer_pct.values())
    client_pct = client_usage_delta_usec / elapsed_us / client_core_count * 100
    measured_usec = sum(peer_usage_delta_usec.values()) + client_usage_delta_usec
    host_cores = os.cpu_count() or 1
    interference_pct = (
        max(host_busy_delta_usec - measured_usec, 0) / elapsed_us / host_cores * 100
    )
    gates = Gates(
        server_saturation_pct=max_peer_pct,
        least_loaded_peer_pct=min_peer_pct,
        peer_imbalance_ratio=min_peer_pct / max_peer_pct if max_peer_pct else 0.0,
        client_headroom_pct=client_pct,
        host_interference_pct=interference_pct,
        g1_server_saturation=max_peer_pct >= 85.0,
        g2_client_headroom=client_pct < 50.0,
        g4_host_interference=interference_pct < 5.0,
    )
    return gates, per_peer_pct


def parse_cores(spec: str) -> tuple[int, ...]:
    from cluster import parse_cpuset

    return parse_cpuset(spec)


def drain_worker_errors(result_queue: Any) -> None:
    """Print any diagnostics a dying client process managed to report."""
    while True:
        try:
            summary = result_queue.get(timeout=2.0)
        except Exception:  # noqa: BLE001 - queue drained
            return
        if "error" in summary:
            print(f"worker {summary['worker_id']} failed:\n{summary['traceback']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", required=True)
    parser.add_argument(
        "--queries-path",
        required=True,
        help="npy file of query vectors, prepared by prepare_dataset.py",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--client-cores", required=True, help="e.g. 64-95")
    parser.add_argument("--procs", type=int, default=4)
    parser.add_argument("--inflight", type=int, default=64)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--hnsw-ef", type=int, required=True)
    parser.add_argument("--queries", type=int, default=0, help="0 uses every query")
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="replay the query set this many times to reach a steady state",
    )
    parser.add_argument("--warmup", type=int, default=2000)
    parser.add_argument(
        "--coordinator",
        choices=("all", "controller"),
        default="all",
        help="'all' spreads entry points across peers; 'controller' pins one",
    )
    parser.add_argument(
        "--allow-asymmetric",
        action="store_true",
        help="permit unequal peer core counts (diagnostics only)",
    )
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    peers = discover_peers()
    client_cores = parse_cores(args.client_cores)
    assert_disjoint(peers, client_cores)
    if not args.allow_asymmetric:
        assert_symmetric(peers)
    if args.procs > len(client_cores):
        raise TestbedError("more client processes than client cores")
    if args.inflight % args.procs:
        raise TestbedError("--inflight must divide evenly across --procs")

    endpoints = (
        [peer.base_url for peer in peers]
        if args.coordinator == "all"
        else [peers[0].base_url]
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_queries = int(np.load(args.queries_path, mmap_mode="r").shape[0])
    if args.queries:
        total_queries = min(total_queries, args.queries)
    bounds = np.linspace(0, total_queries, args.procs + 1).astype(int)

    config = {
        "collection": args.collection,
        "queries_path": str(Path(args.queries_path).resolve()),
        "output_dir": str(output_dir.resolve()),
        "endpoints": endpoints,
        "top_k": args.top_k,
        "hnsw_ef": args.hnsw_ef,
        "batch": args.batch,
        "repeat": args.repeat,
        "inflight_per_worker": args.inflight // args.procs,
        "warmup_per_worker": args.warmup // args.procs,
    }

    context = mp.get_context("spawn")
    ready = context.Queue()
    results = context.Queue()
    start_signal = context.Event()
    core_groups = [list(client_cores[index :: args.procs]) for index in range(args.procs)]
    processes = [
        context.Process(
            target=worker_main,
            args=(
                index,
                core_groups[index],
                config,
                (int(bounds[index]), int(bounds[index + 1])),
                ready,
                start_signal,
                results,
            ),
            daemon=True,
        )
        for index in range(args.procs)
    ]
    for process in processes:
        process.start()

    client_pids: list[int] = []
    deadline = time.monotonic() + WORKER_STARTED_TIMEOUT_S
    while len(client_pids) < args.procs:
        if time.monotonic() > deadline:
            raise TestbedError("client processes failed to become ready")
        if any(not process.is_alive() for process in processes):
            drain_worker_errors(results)
            raise TestbedError("a client process exited before becoming ready")
        try:
            _, pid = ready.get(timeout=1.0)
        except Exception:  # noqa: BLE001 - empty queue while polling liveness
            continue
        client_pids.append(pid)

    peer_before = {peer.name: peer.cpu_usage_usec() for peer in peers}
    client_before = process_cpu_usec([os.getpid(), *client_pids])
    host_before = host_busy_usec()
    started = time.perf_counter()
    start_signal.set()

    summaries = [results.get() for _ in processes]
    elapsed_s = time.perf_counter() - started
    failed = [summary for summary in summaries if "error" in summary]
    if failed:
        for summary in failed:
            print(f"worker {summary['worker_id']} failed:\n{summary['traceback']}")
        raise TestbedError(f"{len(failed)} client process(es) failed")
    host_after = host_busy_usec()
    client_after = process_cpu_usec([os.getpid(), *client_pids])
    peer_after = {peer.name: peer.cpu_usage_usec() for peer in peers}
    for process in processes:
        process.join(timeout=30)

    latencies = np.concatenate(
        [np.asarray(summary["latencies_us"], dtype=np.float64) for summary in summaries]
    )
    measured_queries = sum(summary["queries"] for summary in summaries)
    elapsed_us = elapsed_s * 1_000_000
    gates, per_peer_pct = evaluate_gates(
        peers=peers,
        peer_usage_delta_usec={
            name: peer_after[name] - peer_before[name] for name in peer_before
        },
        client_usage_delta_usec=client_after - client_before,
        host_busy_delta_usec=host_after - host_before,
        elapsed_us=elapsed_us,
        client_core_count=len(client_cores),
    )

    summary = {
        "tag": args.tag,
        "collection": args.collection,
        "queries": measured_queries,
        "elapsed_s": elapsed_s,
        "qps": measured_queries / elapsed_s,
        "batch": args.batch,
        "repeat": args.repeat,
        "inflight": args.inflight,
        "procs": args.procs,
        "hnsw_ef": args.hnsw_ef,
        "top_k": args.top_k,
        "coordinator": args.coordinator,
        "client_cores": list(client_cores),
        "peers": {peer.name: list(peer.cores) for peer in peers},
        "request_latency_us": {
            "p50": float(np.percentile(latencies, 50)),
            "p95": float(np.percentile(latencies, 95)),
            "p99": float(np.percentile(latencies, 99)),
        },
        "per_peer_cpu_pct": per_peer_pct,
        "gates": asdict(gates),
        "gates_passed": gates.passed,
        "note": "recall is scored offline by score_recall.py; G3 needs two runs",
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    if not gates.passed:
        print("\nGATES FAILED: this run must not be used for any performance claim.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
