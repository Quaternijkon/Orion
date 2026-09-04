#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shlex
import statistics
import subprocess
import time
from concurrent.futures import (
    FIRST_COMPLETED,
    Future,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    wait,
)
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import h5py
import numpy as np

from c1_benchmark import (
    DATASETS,
    HTTP,
    PER_QUERY_FIELDS,
    Node,
    current_cpu_affinity,
    discover_nodes,
    execute_query,
    load_partition,
    load_topology,
    oracle_minimum_fanout,
    preprocess_rows,
    round_robin_placement,
    summarize_rows,
    validate_benchmark_affinity,
)
from c1_protocol import TARGET_RECALL, TOP_K, sha256_path, utc_timestamp


E1_PER_QUERY_FIELDS = ("repetition", "sequence_number", *PER_QUERY_FIELDS)
DEFAULT_CONCURRENCY_GRID = (1, 2, 4, 8, 16, 32, 64, 128)


_PROCESS_QUERY_CONFIG: dict[str, Any] | None = None
_PROCESS_REQUEST_EXECUTOR: ThreadPoolExecutor | None = None


@dataclass(frozen=True)
class NodeSnapshot:
    timestamp: float
    qdrant_pid: int
    qdrant_cpu_ticks: int
    network_rx_bytes: int
    network_tx_bytes: int
    network_speed_mbps: int


def initialize_process_query_worker(
    spec: Any,
    method: str,
    logical_shards: int,
    fanout: int,
    ef_search: int,
    collection: str,
    placement: dict[int, Node],
    centroids: np.ndarray,
    request_workers: int,
) -> None:
    global _PROCESS_QUERY_CONFIG, _PROCESS_REQUEST_EXECUTOR
    _PROCESS_QUERY_CONFIG = {
        "spec": spec,
        "method": method,
        "logical_shards": logical_shards,
        "fanout": fanout,
        "ef_search": ef_search,
        "collection": collection,
        "placement": placement,
        "centroids": centroids,
    }
    _PROCESS_REQUEST_EXECUTOR = ThreadPoolExecutor(max_workers=request_workers)


def execute_query_in_process(
    query_id: int,
    query: np.ndarray,
    ground_truth: np.ndarray,
    oracle_fanout: int,
) -> dict[str, Any]:
    if _PROCESS_QUERY_CONFIG is None or _PROCESS_REQUEST_EXECUTOR is None:
        raise RuntimeError("process query worker was not initialized")
    config = _PROCESS_QUERY_CONFIG
    return execute_query(
        query_id,
        query,
        ground_truth,
        config["spec"],
        config["method"],
        config["logical_shards"],
        config["fanout"],
        config["ef_search"],
        config["collection"],
        config["placement"],
        config["centroids"],
        oracle_fanout,
        _PROCESS_REQUEST_EXECUTOR,
        require_worker_cpu_time=True,
    )


def execute_query_batch_in_process(
    batch: Sequence[tuple[int, int, np.ndarray, np.ndarray, int]],
) -> list[tuple[int, dict[str, Any]]]:
    return [
        (
            sequence_number,
            execute_query_in_process(query_id, query, ground_truth, oracle_fanout),
        )
        for sequence_number, query_id, query, ground_truth, oracle_fanout in batch
    ]


def process_worker_pid() -> int:
    return os.getpid()


def process_executor_pids(executor: ProcessPoolExecutor, expected: int) -> list[int]:
    executor.submit(process_worker_pid).result()
    processes = getattr(executor, "_processes", None)
    if not isinstance(processes, dict):
        raise RuntimeError("cannot inspect process-pool workers")
    pids = sorted(int(pid) for pid in processes)
    if len(pids) != expected:
        raise RuntimeError(f"expected {expected} client processes, got {pids}")
    return pids


def read_process_cpu_ticks(pids: Sequence[int]) -> dict[int, int]:
    result: dict[int, int] = {}
    for pid in pids:
        stat = Path(f"/proc/{pid}/stat")
        if not stat.is_file():
            raise RuntimeError(f"client process disappeared during measurement: {pid}")
        fields = stat.read_text(encoding="utf-8").split()
        result[int(pid)] = int(fields[13]) + int(fields[14])
    return result


def atomic_write_text(path: str | Path, text: str) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, destination)
    return destination


def write_json_atomic(path: str | Path, payload: Any) -> Path:
    return atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_e1_rows(path: str | Path, repetition: int, rows: Sequence[dict[str, Any]]) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=E1_PER_QUERY_FIELDS)
        writer.writeheader()
        for sequence_number, row in enumerate(rows):
            encoded = dict(row)
            encoded["repetition"] = repetition
            encoded["sequence_number"] = sequence_number
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
            writer.writerow({field: encoded[field] for field in E1_PER_QUERY_FIELDS})
    os.replace(temporary, destination)
    return destination


def parse_int_grid(value: str) -> list[int]:
    result = sorted(set(int(item) for item in value.split(",") if item.strip()))
    if not result or result[0] <= 0:
        raise argparse.ArgumentTypeError("concurrency grid must contain positive integers")
    return result


def _local_qdrant_pid() -> int:
    output = subprocess.check_output(["pgrep", "-x", "qdrant"], text=True).splitlines()
    if len(output) != 1:
        raise RuntimeError(f"expected one local qdrant process, got {output}")
    return int(output[0])


def _read_local_snapshot(interface: str) -> NodeSnapshot:
    pid = _local_qdrant_pid()
    fields = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split()
    cpu_ticks = int(fields[13]) + int(fields[14])
    root = Path("/sys/class/net") / interface
    return NodeSnapshot(
        timestamp=time.monotonic(),
        qdrant_pid=pid,
        qdrant_cpu_ticks=cpu_ticks,
        network_rx_bytes=int((root / "statistics/rx_bytes").read_text()),
        network_tx_bytes=int((root / "statistics/tx_bytes").read_text()),
        network_speed_mbps=int((root / "speed").read_text()),
    )


def _read_remote_snapshot(node: Node, interface: str) -> NodeSnapshot:
    command = (
        "pid=$(pgrep -x qdrant); "
        "test -n \"$pid\"; "
        "set -- $(cat /proc/$pid/stat); "
        "echo $pid; "
        "echo $((${14}+${15})); "
        f"cat /sys/class/net/{interface}/statistics/rx_bytes; "
        f"cat /sys/class/net/{interface}/statistics/tx_bytes; "
        f"cat /sys/class/net/{interface}/speed"
    )
    output = subprocess.check_output(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=5",
            node.ssh_host,
            "bash -lc " + shlex.quote(command),
        ],
        text=True,
        timeout=15,
    ).splitlines()
    if len(output) != 5:
        raise RuntimeError(f"invalid resource snapshot from {node.private_ip}: {output}")
    return NodeSnapshot(
        timestamp=time.monotonic(),
        qdrant_pid=int(output[0]),
        qdrant_cpu_ticks=int(output[1]),
        network_rx_bytes=int(output[2]),
        network_tx_bytes=int(output[3]),
        network_speed_mbps=int(output[4]),
    )


def read_node_snapshot(node: Node, interface: str) -> NodeSnapshot:
    if node.ssh_host in {"localhost", "127.0.0.1", "::1"}:
        return _read_local_snapshot(interface)
    return _read_remote_snapshot(node, interface)


def read_node_snapshots(nodes: Sequence[Node], interface: str) -> dict[str, NodeSnapshot]:
    with ThreadPoolExecutor(max_workers=len(nodes)) as executor:
        futures = {
            executor.submit(read_node_snapshot, node, interface): node.private_ip
            for node in nodes
        }
        return {future_ip: future.result() for future, future_ip in futures.items()}


def resource_deltas(
    before: dict[str, NodeSnapshot],
    after: dict[str, NodeSnapshot],
    *,
    worker_cpu_count: int,
) -> dict[str, Any]:
    clock_ticks = int(os.sysconf("SC_CLK_TCK"))
    per_host: dict[str, Any] = {}
    for host in sorted(before):
        start = before[host]
        stop = after[host]
        if start.qdrant_pid != stop.qdrant_pid:
            raise RuntimeError(f"qdrant restarted during measurement on {host}")
        elapsed = stop.timestamp - start.timestamp
        if elapsed <= 0:
            raise RuntimeError(f"non-positive resource interval on {host}")
        cpu_seconds = (stop.qdrant_cpu_ticks - start.qdrant_cpu_ticks) / clock_ticks
        rx_bytes = stop.network_rx_bytes - start.network_rx_bytes
        tx_bytes = stop.network_tx_bytes - start.network_tx_bytes
        if cpu_seconds < 0 or rx_bytes < 0 or tx_bytes < 0:
            raise RuntimeError(f"resource counter decreased on {host}")
        speed_bps = start.network_speed_mbps * 1_000_000
        network_utilization = (
            max(rx_bytes, tx_bytes) * 8 / elapsed / speed_bps * 100.0
            if speed_bps > 0
            else None
        )
        per_host[host] = {
            "elapsed_seconds": elapsed,
            "qdrant_pid": start.qdrant_pid,
            "qdrant_cpu_seconds": cpu_seconds,
            "qdrant_cpu_utilization_pct_of_reserved_cores": (
                cpu_seconds / elapsed / worker_cpu_count * 100.0
            ),
            "network_rx_bytes": rx_bytes,
            "network_tx_bytes": tx_bytes,
            "network_speed_mbps": start.network_speed_mbps,
            "network_utilization_pct_max_direction": network_utilization,
        }
    network_values = [
        row["network_utilization_pct_max_direction"]
        for row in per_host.values()
        if row["network_utilization_pct_max_direction"] is not None
    ]
    return {
        "per_host": per_host,
        "max_worker_cpu_utilization_pct": max(
            row["qdrant_cpu_utilization_pct_of_reserved_cores"]
            for row in per_host.values()
        ),
        "max_network_utilization_pct": max(network_values) if network_values else None,
    }


def run_query_phase(
    *,
    phase_name: str,
    total_queries: int,
    concurrency: int,
    query_id_start: int,
    queries: np.ndarray,
    ground_truth: np.ndarray,
    oracle: np.ndarray,
    spec: Any,
    method: str,
    logical_shards: int,
    fanout: int,
    ef_search: int,
    collection: str,
    placement: dict[int, Node],
    centroids: np.ndarray,
    query_executor: ProcessPoolExecutor,
    aggregator_process_pids: Sequence[int],
    used_nodes: Sequence[Node],
    network_interface: str,
    benchmark_cpu_count: int,
    worker_cpu_count: int,
    progress_every: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if total_queries <= 0 or concurrency <= 0:
        raise ValueError("phase query count and concurrency must be positive")
    if not len(queries) == len(ground_truth) == len(oracle):
        raise ValueError("query, truth, and oracle arrays must align")
    query_worker_count = len(aggregator_process_pids) - 1
    if concurrency > query_worker_count:
        raise ValueError(
            f"concurrency {concurrency} exceeds {query_worker_count} client query processes"
        )
    batches: list[list[tuple[int, int, np.ndarray, np.ndarray, int]]] = [
        [] for _ in range(min(concurrency, total_queries))
    ]
    for sequence_number in range(total_queries):
        offset = sequence_number % len(queries)
        batches[sequence_number % len(batches)].append(
            (
                sequence_number,
                query_id_start + offset,
                queries[offset],
                ground_truth[offset],
                int(oracle[offset]),
            )
        )
    before = read_node_snapshots(used_nodes, network_interface)
    aggregator_cpu_before = read_process_cpu_ticks(aggregator_process_pids)
    started = time.perf_counter()
    rows: list[dict[str, Any] | None] = [None] * total_queries
    pending: set[Future[list[tuple[int, dict[str, Any]]]]] = {
        query_executor.submit(execute_query_batch_in_process, batch) for batch in batches
    }
    completed = 0
    last_reported = 0
    try:
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                pending.remove(future)
                batch_results = future.result()
                for sequence_number, row in batch_results:
                    rows[sequence_number] = row
                completed += len(batch_results)
                if progress_every > 0 and (
                    completed - last_reported >= progress_every
                    or completed == total_queries
                ):
                    print(
                        f"{phase_name} c={concurrency}: {completed}/{total_queries}",
                        flush=True,
                    )
                    last_reported = completed
    except BaseException:
        for future in pending:
            future.cancel()
        raise
    ended = time.perf_counter()
    aggregator_cpu_after = read_process_cpu_ticks(aggregator_process_pids)
    after = read_node_snapshots(used_nodes, network_interface)
    elapsed = ended - started
    typed_rows = [row for row in rows if row is not None]
    if len(typed_rows) != total_queries:
        raise RuntimeError("one or more closed-loop queries were not recorded")
    summary = summarize_rows(typed_rows, elapsed)
    resources = resource_deltas(
        before, after, worker_cpu_count=worker_cpu_count
    )
    clock_ticks = int(os.sysconf("SC_CLK_TCK"))
    aggregator_cpu_seconds_per_process = {
        str(pid): (aggregator_cpu_after[pid] - aggregator_cpu_before[pid]) / clock_ticks
        for pid in aggregator_process_pids
    }
    if any(value < 0 for value in aggregator_cpu_seconds_per_process.values()):
        raise RuntimeError("client process CPU counter decreased during measurement")
    aggregator_cpu_seconds = sum(aggregator_cpu_seconds_per_process.values())
    aggregator_max_single_process_cpu_utilization = max(
        aggregator_cpu_seconds_per_process.values(), default=0.0
    ) / elapsed * 100.0
    routing_cpu_seconds = sum(
        float(row["routing_cpu_time_us"]) for row in typed_rows
    ) / 1_000_000.0
    worker_cpu_seconds = sum(
        sum(map(int, row["worker_cpu_time_us_per_shard"])) for row in typed_rows
    ) / 1_000_000.0
    routing_share = (
        routing_cpu_seconds / (routing_cpu_seconds + worker_cpu_seconds) * 100.0
        if routing_cpu_seconds + worker_cpu_seconds > 0
        else 0.0
    )
    summary.update(
        {
            "phase_name": phase_name,
            "concurrency": concurrency,
            "offered_qps": total_queries / elapsed,
            "completed_qps": total_queries / elapsed,
            "elapsed_seconds": elapsed,
            "aggregator_cpu_seconds": aggregator_cpu_seconds,
            "aggregator_cpu_seconds_per_process": aggregator_cpu_seconds_per_process,
            "aggregator_process_count": len(aggregator_process_pids),
            "aggregator_query_worker_process_count": query_worker_count,
            "aggregator_cpu_utilization_pct_of_reserved_cores": (
                aggregator_cpu_seconds / elapsed / benchmark_cpu_count * 100.0
            ),
            "aggregator_max_single_process_cpu_utilization_pct": (
                aggregator_max_single_process_cpu_utilization
            ),
            "aggregator_coordinator_cpu_utilization_pct": (
                aggregator_cpu_seconds_per_process[str(os.getpid())] / elapsed * 100.0
            ),
            "routing_cpu_seconds": routing_cpu_seconds,
            "worker_cpu_seconds_from_responses": worker_cpu_seconds,
            "routing_share_of_routing_plus_worker_cpu_pct": routing_share,
            "aggregator_waiting_queue_depth_start": 0,
            "aggregator_waiting_queue_depth_end": 0,
            "aggregator_waiting_queue_depth_max": 0,
            "persistent_aggregator_queue_growth": False,
            "resource_utilization": resources,
        }
    )
    return typed_rows, summary


def select_saturation_concurrency(
    rows: Sequence[dict[str, Any]],
    *,
    improvement_threshold: float = 0.05,
    p99_multiplier: float = 5.0,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("saturation selection requires sweep rows")
    minimum_p99 = math.inf
    consecutive_low_improvements = 0
    stop_index: int | None = None
    stop_reason = "maximum_concurrency_reached_without_knee"
    for index, row in enumerate(rows):
        p99 = float(row["p99_latency_us"])
        minimum_p99 = min(minimum_p99, p99)
        if index > 0:
            previous_qps = float(rows[index - 1]["completed_qps"])
            improvement = float(row["completed_qps"]) / previous_qps - 1.0
            row["qps_improvement_from_previous"] = improvement
            consecutive_low_improvements = (
                consecutive_low_improvements + 1
                if improvement < improvement_threshold
                else 0
            )
        else:
            row["qps_improvement_from_previous"] = None
        if p99 > p99_multiplier * minimum_p99:
            stop_index = index
            stop_reason = "p99_exceeded_multiplier"
            break
        if consecutive_low_improvements >= 2:
            stop_index = index
            stop_reason = "two_consecutive_subthreshold_qps_improvements"
            break
    if stop_index is None:
        eligible = list(rows)
    elif stop_reason == "p99_exceeded_multiplier" and stop_index > 0:
        eligible = list(rows[:stop_index])
    else:
        eligible = list(rows[: stop_index + 1])
    if not eligible:
        eligible = [rows[0]]
    selected = max(
        eligible,
        key=lambda row: (float(row["completed_qps"]), -int(row["concurrency"])),
    )
    return {
        "selected_concurrency": int(selected["concurrency"]),
        "selected_sweep_qps": float(selected["completed_qps"]),
        "stop_reason": stop_reason,
        "knee_observed": stop_index is not None,
        "stop_concurrency": (
            int(rows[stop_index]["concurrency"]) if stop_index is not None else None
        ),
        "minimum_observed_p99_latency_us": minimum_p99,
    }


def apply_formal_concurrency_override(
    selection: dict[str, Any],
    sweep_rows: Sequence[dict[str, Any]],
    *,
    override: int | None,
    reason: str | None,
) -> dict[str, Any]:
    if override is None:
        return selection
    automatic_concurrency = int(selection["selected_concurrency"])
    measured = {int(row["concurrency"]): row for row in sweep_rows}
    if override not in measured:
        raise ValueError("formal concurrency override was not measured in the sweep")
    if override >= automatic_concurrency:
        raise ValueError(
            "formal concurrency override must be below the automatic knee selection"
        )
    return {
        **selection,
        "automatic_selected_concurrency": automatic_concurrency,
        "automatic_selected_sweep_qps": selection["selected_sweep_qps"],
        "selected_concurrency": override,
        "selected_sweep_qps": float(measured[override]["completed_qps"]),
        "formal_concurrency_override_reason": reason,
    }


def sample_coefficient_of_variation(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = statistics.fmean(values)
    return statistics.stdev(values) / mean if mean else math.inf


def confidence_interval_95(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("confidence interval requires values")
    mean = statistics.fmean(values)
    if len(values) == 1:
        return {"n": 1, "mean": mean, "sample_std": 0.0, "lower": mean, "upper": mean}
    sample_std = statistics.stdev(values)
    critical = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}.get(len(values), 1.96)
    half_width = critical * sample_std / math.sqrt(len(values))
    return {
        "n": len(values),
        "mean": mean,
        "sample_std": sample_std,
        "lower": mean - half_width,
        "upper": mean + half_width,
    }


def phase_validity(summary: dict[str, Any]) -> tuple[str, list[str]]:
    flags: list[str] = []
    if float(summary["achieved_recall"]) < TARGET_RECALL:
        flags.append("INVALID_RECALL")
    if (
        int(summary.get("aggregator_process_count") or 1) == 1
        and float(summary.get("aggregator_max_single_process_cpu_utilization_pct") or 0.0)
        > 85.0
    ):
        flags.append("AGGREGATOR_SINGLE_PROCESS_LIMITED")
    if (
        int(summary.get("aggregator_process_count") or 1) > 1
        and float(summary.get("aggregator_coordinator_cpu_utilization_pct") or 0.0) > 85.0
    ):
        flags.append("AGGREGATOR_COORDINATOR_LIMITED")
    if float(summary["aggregator_cpu_utilization_pct_of_reserved_cores"]) > 85.0:
        flags.append("AGGREGATOR_CPU_LIMITED")
    if bool(summary["persistent_aggregator_queue_growth"]):
        flags.append("AGGREGATOR_QUEUE_GROWTH")
    if float(summary["routing_share_of_routing_plus_worker_cpu_pct"]) > 25.0:
        flags.append("AGGREGATOR_ROUTING_LIMITED")
    network = summary["resource_utilization"]["max_network_utilization_pct"]
    if network is not None and float(network) > 85.0:
        flags.append("NETWORK_LIMITED")
    return ("VALID_E1" if not flags else "INVALID_E1"), flags


def execute(args: argparse.Namespace) -> dict[str, Any]:
    benchmark_affinity = validate_benchmark_affinity(args.topology)
    topology = load_topology(args.topology)
    nodes = discover_nodes(topology)
    spec = DATASETS[args.dataset]
    partition = load_partition(args.partition_artifact, args.method, args.logical_shards)
    placement = round_robin_placement(args.logical_shards, nodes)
    used_nodes = list(dict.fromkeys(placement.values()))
    collection_payload, _ = HTTP.request(
        nodes[0].base_url,
        "GET",
        f"/collections/{args.collection}",
        timeout=30.0,
    )
    info = collection_payload.get("result") or {}
    if (
        info.get("status") != "green"
        or int(info.get("points_count") or 0) != len(partition.assignments)
        or int(info.get("indexed_vectors_count") or 0) != len(partition.assignments)
    ):
        raise RuntimeError(f"collection is not fully indexed and green: {info}")
    with h5py.File(args.hdf5_path, "r") as handle:
        warm_queries = preprocess_rows(handle["test"][: args.warmup_query_count], spec)
        warm_truth = np.asarray(
            handle["neighbors"][: args.warmup_query_count, :TOP_K], dtype=np.int32
        )
        measurement_queries = preprocess_rows(handle["test"][1_000:10_000], spec)
        measurement_truth = np.asarray(
            handle["neighbors"][1_000:10_000, :TOP_K], dtype=np.int32
        )
    warm_oracle = oracle_minimum_fanout(
        partition.assignments, warm_truth, args.logical_shards
    )
    measurement_oracle = oracle_minimum_fanout(
        partition.assignments, measurement_truth, args.logical_shards
    )
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    client_processes = args.client_processes or len(benchmark_affinity)
    concurrency_grid = [
        concurrency
        for concurrency in args.concurrency_grid
        if concurrency <= client_processes
    ]
    if not concurrency_grid:
        raise ValueError("client process count is below the minimum concurrency")
    request_workers_per_process = max(args.request_workers, args.fanout)
    worker_cpu_count = len(
        {
            cpu
            for part in str(nodes[0].cpuset).split(",")
            for cpu in (
                range(int(part.split("-")[0]), int(part.split("-")[1]) + 1)
                if "-" in part
                else [int(part)]
            )
        }
    )
    with ProcessPoolExecutor(
        max_workers=client_processes,
        initializer=initialize_process_query_worker,
        initargs=(
            spec,
            args.method,
            args.logical_shards,
            args.fanout,
            args.ef_search,
            args.collection,
            placement,
            partition.centroids,
            request_workers_per_process,
        ),
    ) as query_executor:
        aggregator_process_pids = [
            os.getpid(),
            *process_executor_pids(query_executor, client_processes),
        ]
        _, warmup = run_query_phase(
            phase_name=f"warmup {args.experiment_id}",
            total_queries=args.warmup_query_count,
            concurrency=min(args.warmup_concurrency, max(concurrency_grid)),
            query_id_start=0,
            queries=warm_queries,
            ground_truth=warm_truth,
            oracle=warm_oracle,
            spec=spec,
            method=args.method,
            logical_shards=args.logical_shards,
            fanout=args.fanout,
            ef_search=args.ef_search,
            collection=args.collection,
            placement=placement,
            centroids=partition.centroids,
            query_executor=query_executor,
            aggregator_process_pids=aggregator_process_pids,
            used_nodes=used_nodes,
            network_interface=args.network_interface,
            benchmark_cpu_count=len(benchmark_affinity),
            worker_cpu_count=worker_cpu_count,
            progress_every=args.progress_every,
        )
        sweep_rows: list[dict[str, Any]] = []
        for concurrency in concurrency_grid:
            _, sweep = run_query_phase(
                phase_name=f"sweep {args.experiment_id}",
                total_queries=args.sweep_query_count,
                concurrency=concurrency,
                query_id_start=1_000,
                queries=measurement_queries,
                ground_truth=measurement_truth,
                oracle=measurement_oracle,
                spec=spec,
                method=args.method,
                logical_shards=args.logical_shards,
                fanout=args.fanout,
                ef_search=args.ef_search,
                collection=args.collection,
                placement=placement,
                centroids=partition.centroids,
                query_executor=query_executor,
                aggregator_process_pids=aggregator_process_pids,
                used_nodes=used_nodes,
                network_interface=args.network_interface,
                benchmark_cpu_count=len(benchmark_affinity),
                worker_cpu_count=worker_cpu_count,
                progress_every=args.progress_every,
            )
            sweep_rows.append(sweep)
            selection = select_saturation_concurrency(sweep_rows)
            if selection["knee_observed"]:
                break
        selection = select_saturation_concurrency(sweep_rows)
        if not selection["knee_observed"]:
            raise RuntimeError(
                "saturation knee not observed; extend --concurrency-grid before formal measurement"
            )
        selection = apply_formal_concurrency_override(
            selection,
            sweep_rows,
            override=args.formal_concurrency_override,
            reason=args.formal_concurrency_override_reason,
        )
        selected_concurrency = int(selection["selected_concurrency"])
        repetitions: list[dict[str, Any]] = []
        target_repetitions = args.minimum_repetitions
        repetition = 1
        while repetition <= target_repetitions:
            _, rep_warmup = run_query_phase(
                phase_name=f"rep{repetition}-warmup {args.experiment_id}",
                total_queries=args.warmup_query_count,
                concurrency=selected_concurrency,
                query_id_start=0,
                queries=warm_queries,
                ground_truth=warm_truth,
                oracle=warm_oracle,
                spec=spec,
                method=args.method,
                logical_shards=args.logical_shards,
                fanout=args.fanout,
                ef_search=args.ef_search,
                collection=args.collection,
                placement=placement,
                centroids=partition.centroids,
                query_executor=query_executor,
                aggregator_process_pids=aggregator_process_pids,
                used_nodes=used_nodes,
                network_interface=args.network_interface,
                benchmark_cpu_count=len(benchmark_affinity),
                worker_cpu_count=worker_cpu_count,
                progress_every=args.progress_every,
            )
            rows, summary = run_query_phase(
                phase_name=f"rep{repetition} {args.experiment_id}",
                total_queries=args.measurement_query_count,
                concurrency=selected_concurrency,
                query_id_start=1_000,
                queries=measurement_queries,
                ground_truth=measurement_truth,
                oracle=measurement_oracle,
                spec=spec,
                method=args.method,
                logical_shards=args.logical_shards,
                fanout=args.fanout,
                ef_search=args.ef_search,
                collection=args.collection,
                placement=placement,
                centroids=partition.centroids,
                query_executor=query_executor,
                aggregator_process_pids=aggregator_process_pids,
                used_nodes=used_nodes,
                network_interface=args.network_interface,
                benchmark_cpu_count=len(benchmark_affinity),
                worker_cpu_count=worker_cpu_count,
                progress_every=args.progress_every,
            )
            raw_path = output_dir / f"rep{repetition}-per-query.csv"
            write_e1_rows(raw_path, repetition, rows)
            status, flags = phase_validity(summary)
            summary.update(
                {
                    "repetition": repetition,
                    "warmup_query_count": args.warmup_query_count,
                    "measurement_query_count": args.measurement_query_count,
                    "warmup_summary": rep_warmup,
                    "raw_per_query": str(raw_path),
                    "raw_per_query_sha256": sha256_path(raw_path),
                    "run_status": status,
                    "bottleneck_flags": flags,
                }
            )
            write_json_atomic(output_dir / f"rep{repetition}-summary.json", summary)
            repetitions.append(summary)
            if repetition == args.minimum_repetitions:
                cv = sample_coefficient_of_variation(
                    [float(row["completed_qps"]) for row in repetitions]
                )
                if cv > args.cv_threshold:
                    target_repetitions = args.maximum_repetitions
            repetition += 1

    qps_values = [float(row["completed_qps"]) for row in repetitions]
    aggregate_status = (
        "VALID_E1"
        if all(row["run_status"] == "VALID_E1" for row in repetitions)
        else "INVALID_E1"
    )
    result = {
        "timestamp": utc_timestamp(),
        "record_type": "e1_physical_scaleout_configuration",
        "experiment_id": args.experiment_id,
        "dataset": args.dataset,
        "dataset_sha256": sha256_path(args.hdf5_path),
        "partition_method": args.method,
        "partition_artifact": partition.path,
        "partition_artifact_sha256": partition.sha256,
        "logical_shard_count": args.logical_shards,
        "physical_machine_count": args.logical_shards,
        "collection": args.collection,
        "fanout": args.fanout,
        "ef_search": args.ef_search,
        "target_recall": TARGET_RECALL,
        "benchmark_cpu_affinity": benchmark_affinity,
        "worker_cpu_affinity": {node.private_ip: node.cpuset for node in used_nodes},
        "logical_to_physical_mapping": {
            str(shard): placement[shard].private_ip for shard in placement
        },
        "network_interface": args.network_interface,
        "client_processes": client_processes,
        "concurrency_grid": concurrency_grid,
        "request_workers_per_process": request_workers_per_process,
        "request_workers": client_processes * request_workers_per_process,
        "warmup": warmup,
        "sweep": sweep_rows,
        "saturation_selection": selection,
        "selected_concurrency": selected_concurrency,
        "repetition_count": len(repetitions),
        "qps_coefficient_of_variation": sample_coefficient_of_variation(qps_values),
        "qps_confidence_interval_95": confidence_interval_95(qps_values),
        "repetitions": repetitions,
        "status": aggregate_status,
    }
    write_json_atomic(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run C1 E1 closed-loop physical scale-out measurement"
    )
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--topology", default="experiments/c1/topology-amd-4node.json")
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--hdf5-path", required=True)
    parser.add_argument("--method", choices=("random", "kmeans"), required=True)
    parser.add_argument("--logical-shards", type=int, choices=(1, 2, 4), required=True)
    parser.add_argument("--partition-artifact", required=True)
    parser.add_argument("--collection", required=True)
    parser.add_argument("--fanout", type=int, required=True)
    parser.add_argument("--ef-search", type=int, required=True)
    parser.add_argument("--network-interface", default="enp65s0f0np0")
    parser.add_argument("--warmup-query-count", type=int, default=1_000)
    parser.add_argument("--warmup-concurrency", type=int, default=16)
    parser.add_argument("--sweep-query-count", type=int, default=2_000)
    parser.add_argument("--measurement-query-count", type=int, default=18_000)
    parser.add_argument("--concurrency-grid", type=parse_int_grid, default=list(DEFAULT_CONCURRENCY_GRID))
    parser.add_argument("--formal-concurrency-override", type=int)
    parser.add_argument("--formal-concurrency-override-reason")
    parser.add_argument(
        "--client-processes",
        type=int,
        default=0,
        help="query worker processes; zero uses every reserved benchmark CPU",
    )
    parser.add_argument("--request-workers", type=int, default=4)
    parser.add_argument("--minimum-repetitions", type=int, default=3)
    parser.add_argument("--maximum-repetitions", type=int, default=5)
    parser.add_argument("--cv-threshold", type=float, default=0.05)
    parser.add_argument("--progress-every", type=int, default=1_000)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    positive = (
        "warmup_query_count",
        "warmup_concurrency",
        "sweep_query_count",
        "measurement_query_count",
        "request_workers",
        "minimum_repetitions",
        "maximum_repetitions",
    )
    if any(getattr(args, name) <= 0 for name in positive):
        parser.error("query, concurrency, worker, and repetition arguments must be positive")
    if args.client_processes < 0:
        parser.error("client process count must be non-negative")
    if args.maximum_repetitions < args.minimum_repetitions:
        parser.error("maximum repetitions must be at least the minimum")
    if bool(args.formal_concurrency_override) != bool(
        args.formal_concurrency_override_reason
    ):
        parser.error(
            "formal concurrency override and its reason must be provided together"
        )
    if (
        args.formal_concurrency_override is not None
        and args.formal_concurrency_override <= 0
    ):
        parser.error("formal concurrency override must be positive")
    if args.measurement_query_count < 10_000:
        parser.error("formal E1 measurement requires at least 10,000 completed queries")
    if args.method == "random" and args.fanout != args.logical_shards:
        parser.error("Random E1 must broadcast to every shard")
    if not 1 <= args.fanout <= args.logical_shards:
        parser.error("fan-out must be within the logical shard count")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    execute(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
