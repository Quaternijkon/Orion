#!/usr/bin/env python3
"""Run C6 E5 compact-MultiEP physical validation on the four-machine cluster."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from c6_capture import encode_entry_point, request_json  # noqa: E402
from c6_native_cluster import load_topology, node_root, nodes, run_shell  # noqa: E402
from c6_protocol import (  # noqa: E402
    distance_computations_from_usage,
    policy_config_from_dict,
    policy_efs,
    policy_shards,
    recall_at_k,
    route_queries_from_trace,
    utc_timestamp,
    write_csv_atomic,
    write_json_atomic,
)


def parse_ints(value: str) -> tuple[int, ...]:
    try:
        values = tuple(sorted(set(int(item) for item in value.split(",") if item)))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not values or values[0] <= 0:
        raise argparse.ArgumentTypeError("values must be positive")
    return values


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("sift1m", "glove-200-angular"), required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--collection", required=True)
    parser.add_argument("--shard-key-map", required=True)
    parser.add_argument("--tuning-route-trace", required=True)
    parser.add_argument("--measurement-route-trace", required=True)
    parser.add_argument("--hdf5-path", required=True)
    parser.add_argument("--tuning-selection", required=True)
    parser.add_argument("--algorithmic-completion", required=True)
    parser.add_argument("--topology", required=True)
    parser.add_argument("--cluster-run-id", required=True)
    parser.add_argument("--runtime-root", default="/users/dry/orion-c6-runtime")
    parser.add_argument("--concurrency", type=parse_ints, default=(1, 4, 8, 16, 32, 64))
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--measurement-cycles", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--source-id-dedup-block-size", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", required=True)
    return parser.parse_args(argv)


def load_dataset(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    import h5py

    with h5py.File(path, "r") as handle:
        tuning_queries = handle["test"][:1000].astype(np.float32, copy=True)
        tuning_truth = handle["neighbors"][:1000, :10].astype(np.int64, copy=True)
        measurement_queries = handle["test"][1000:].astype(np.float32, copy=True)
        measurement_truth = handle["neighbors"][1000:, :10].astype(np.int64, copy=True)
    return tuning_queries, tuning_truth, measurement_queries, measurement_truth


def selected_configs(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    selected = payload.get("selected") if isinstance(payload, dict) else None
    if not isinstance(selected, dict):
        raise ValueError("tuning selection is missing selected policies")
    result = {}
    for policy in ("P0", "P1", "P2", "P3"):
        config = policy_config_from_dict(selected[policy]["config"])
        config.validate(4)
        result[policy] = config
    return result


def build_bodies(
    routes: Sequence[Any],
    queries: np.ndarray,
    config: Any,
    shard_keys: Mapping[int, str],
    *,
    top_k: int,
    block_size: int,
) -> list[dict[str, Any]]:
    bodies = []
    for route, query in zip(routes, queries, strict=True):
        selected = policy_shards(route, config)
        efs = policy_efs(route, selected, config)
        evidence = route.evidence_by_shard()
        keys = [shard_keys[shard_id] for shard_id in selected]
        entry_points = {
            shard_keys[shard_id]: [
                encode_entry_point(
                    int(point_id),
                    shard_id,
                    mode="copy_block",
                    block_size=block_size,
                )
                for point_id in evidence[shard_id].entry_points
            ]
            for shard_id in selected
            if shard_id in evidence
        }
        body: dict[str, Any] = {
            "vector": query.tolist(),
            "limit": top_k,
            "with_payload": ["source_id"],
            "with_vector": False,
            "shard_key": keys,
            "hnsw_ef_by_shard": {
                shard_keys[shard_id]: ef_search
                for shard_id, ef_search in zip(selected, efs, strict=True)
            },
            "params": {"hnsw_ef": max(efs), "exact": False},
            "source_id_dedup_block_size": block_size,
        }
        if entry_points:
            body["hnsw_entry_points_by_shard"] = entry_points
        bodies.append(body)
    return bodies


def run_one(
    base_url: str,
    collection: str,
    body: Mapping[str, Any],
    truth: Sequence[int],
    dimension: int,
) -> dict[str, Any]:
    started = time.perf_counter_ns()
    payload, response_bytes = request_json(
        base_url,
        "POST",
        f"/collections/{urllib.parse.quote(collection, safe='')}/points/search",
        body,
        timeout=300.0,
    )
    latency_us = (time.perf_counter_ns() - started) / 1_000.0
    results = payload.get("result") or []
    result_ids = [int(item["payload"]["source_id"]) for item in results]
    usage = payload.get("usage") or {}
    hardware = usage.get("hardware") or {}
    required = {"cpu", "cpu_time_us", "cpu_wall_time_us", "graph_nodes_visited"}
    missing = sorted(required - set(hardware))
    if missing:
        raise RuntimeError(f"physical search response is missing hardware fields: {missing}")
    return {
        "recall_at_10": recall_at_k(result_ids, truth),
        "latency_us": latency_us,
        "response_bytes": response_bytes,
        "distance_computations": distance_computations_from_usage(
            int(hardware["cpu"]), dimension
        ),
        "nodes_visited": int(hardware["graph_nodes_visited"]),
        "worker_cpu_time_us": int(hardware["cpu_time_us"]),
        "worker_cpu_wall_time_us": int(hardware["cpu_wall_time_us"]),
    }


def resource_snapshot(topology: dict[str, Any], runtime_root: str, cluster_run_id: str) -> dict[str, Any]:
    result = {}
    for node in nodes(topology):
        root = node_root(runtime_root, cluster_run_id, node)
        command = (
            f"pid=$(cat {root / 'qdrant.pid'}); "
            "ticks=$(awk '{print $14+$15}' /proc/$pid/stat); "
            "hz=$(getconf CLK_TCK); "
            "bytes=$(awk -F'[: ]+' 'NR>2 && $1 != \"lo\" {sum += $3 + $11} END {print sum+0}' /proc/net/dev); "
            "printf '%s %s %s\\n' \"$ticks\" \"$hz\" \"$bytes\""
        )
        fields = run_shell(node, command).stdout.split()
        result[str(node["role"])] = {
            "ticks": int(fields[0]),
            "hz": int(fields[1]),
            "network_bytes": int(fields[2]),
            "allocated_cpus": len(
                [
                    cpu
                    for part in str(node["cpuset"]).split(",")
                    for cpu in (
                        range(int(part.split("-")[0]), int(part.split("-")[1]) + 1)
                        if "-" in part
                        else [int(part)]
                    )
                ]
            ),
        }
    return result


def resource_delta(
    before: Mapping[str, Any], after: Mapping[str, Any], wall_s: float, query_count: int
) -> dict[str, Any]:
    cpu = {}
    network_delta = 0
    for role in before:
        start = before[role]
        end = after[role]
        cpu_seconds = (end["ticks"] - start["ticks"]) / start["hz"]
        cpu[role] = 100.0 * cpu_seconds / wall_s / start["allocated_cpus"]
        network_delta += end["network_bytes"] - start["network_bytes"]
    workers = [value for role, value in cpu.items() if role != "controller"]
    return {
        "controller_combined_cpu_utilization_pct": cpu["controller"],
        "mean_worker_cpu_utilization_pct": statistics.fmean(workers),
        "max_worker_cpu_utilization_pct": max(workers),
        "cluster_network_bytes_per_query": network_delta / query_count,
    }


def execute(
    args: argparse.Namespace,
    topology: dict[str, Any],
    bodies: Sequence[Mapping[str, Any]],
    truth: np.ndarray,
    concurrency: int,
    *,
    collect_resources: bool,
    executor: ThreadPoolExecutor | None = None,
) -> dict[str, Any]:
    before = (
        resource_snapshot(topology, args.runtime_root, args.cluster_run_id)
        if collect_resources
        else None
    )
    started = time.perf_counter()
    client_cpu_started = time.process_time()
    def run(active_executor: ThreadPoolExecutor) -> list[dict[str, Any]]:
        return list(
            active_executor.map(
                lambda item: run_one(
                    args.base_url,
                    args.collection,
                    item[0],
                    item[1],
                    len(item[0]["vector"]),
                ),
                zip(bodies, truth, strict=True),
            )
        )

    if executor is None:
        with ThreadPoolExecutor(max_workers=concurrency) as temporary_executor:
            rows = run(temporary_executor)
    else:
        rows = run(executor)
    wall_s = time.perf_counter() - started
    client_cpu_seconds = time.process_time() - client_cpu_started
    after = (
        resource_snapshot(topology, args.runtime_root, args.cluster_run_id)
        if collect_resources
        else None
    )
    latencies = [row["latency_us"] for row in rows]
    summary = {
        "query_count": len(rows),
        "concurrency": concurrency,
        "wall_seconds": wall_s,
        "qps": len(rows) / wall_s,
        "client_cpu_utilization_pct": (
            100.0
            * client_cpu_seconds
            / wall_s
            / max(len(os.sched_getaffinity(0)), 1)
        ),
        "recall_at_10": statistics.fmean(row["recall_at_10"] for row in rows),
        "mean_latency_us": statistics.fmean(latencies),
        "p50_latency_us": float(np.percentile(latencies, 50)),
        "p95_latency_us": float(np.percentile(latencies, 95)),
        "p99_latency_us": float(np.percentile(latencies, 99)),
        "response_bytes_per_query": statistics.fmean(
            row["response_bytes"] for row in rows
        ),
        "response_distance_computations_per_query": statistics.fmean(
            row["distance_computations"] for row in rows
        ),
        "response_nodes_visited_per_query": statistics.fmean(
            row["nodes_visited"] for row in rows
        ),
        "response_worker_cpu_time_us_per_query": statistics.fmean(
            row["worker_cpu_time_us"] for row in rows
        ),
    }
    if before is not None and after is not None:
        summary.update(resource_delta(before, after, wall_s, len(rows)))
    return summary


def coefficient_of_variation(values: Sequence[float]) -> float:
    return statistics.pstdev(values) / statistics.fmean(values) if len(values) > 1 else 0.0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.repetitions < 3:
        raise ValueError("physical validation requires at least three repetitions")
    if args.measurement_cycles <= 0:
        raise ValueError("measurement-cycles must be positive")
    topology = load_topology(args.topology)
    configs = selected_configs(args.tuning_selection)
    shard_keys = {
        int(key): value
        for key, value in json.loads(Path(args.shard_key_map).read_text()).items()
    }
    tuning_routes = route_queries_from_trace(
        json.loads(Path(args.tuning_route_trace).read_text())
    )
    measurement_routes = route_queries_from_trace(
        json.loads(Path(args.measurement_route_trace).read_text())
    )
    tuning_queries, tuning_truth, measurement_queries, measurement_truth = load_dataset(
        args.hdf5_path
    )
    algorithmic = json.loads(Path(args.algorithmic_completion).read_text())
    algorithmic_by_policy = {row["policy"]: row for row in algorithmic["summaries"]}
    sweep_rows = []
    repetition_rows = []
    final_rows = []
    for policy in ("P0", "P1", "P2", "P3"):
        tuning_bodies = build_bodies(
            tuning_routes,
            tuning_queries,
            configs[policy],
            shard_keys,
            top_k=args.top_k,
            block_size=args.source_id_dedup_block_size,
        )
        measurement_bodies = build_bodies(
            measurement_routes,
            measurement_queries,
            configs[policy],
            shard_keys,
            top_k=args.top_k,
            block_size=args.source_id_dedup_block_size,
        )
        cycled_measurement_bodies = measurement_bodies * args.measurement_cycles
        cycled_measurement_truth = np.tile(
            measurement_truth, (args.measurement_cycles, 1)
        )
        policy_sweep = []
        for concurrency in args.concurrency:
            summary = execute(
                args,
                topology,
                tuning_bodies,
                tuning_truth,
                concurrency,
                collect_resources=False,
            )
            row = {"policy": policy, **summary}
            sweep_rows.append(row)
            policy_sweep.append(row)
        feasible = [row for row in policy_sweep if row["recall_at_10"] >= 0.90]
        if not feasible:
            raise RuntimeError(f"physical concurrency sweep has no feasible {policy} point")
        peak_qps = max(row["qps"] for row in feasible)
        near_peak = [row for row in feasible if row["qps"] >= peak_qps * 0.98]
        selected_concurrency = min(
            near_peak, key=lambda row: row["concurrency"]
        )["concurrency"]
        ordered_sweep = sorted(feasible, key=lambda row: row["concurrency"])
        prior_peak = max(
            (row["qps"] for row in ordered_sweep[:-1]),
            default=ordered_sweep[-1]["qps"],
        )
        saturation_boundary_reached = (
            selected_concurrency < max(args.concurrency)
            or ordered_sweep[-1]["qps"] <= prior_peak * 1.02
        )
        policy_repetitions = []
        target_repetitions = args.repetitions
        with ThreadPoolExecutor(max_workers=selected_concurrency) as persistent_executor:
            while len(policy_repetitions) < target_repetitions:
                execute(
                    args,
                    topology,
                    tuning_bodies,
                    tuning_truth,
                    selected_concurrency,
                    collect_resources=False,
                    executor=persistent_executor,
                )
                summary = execute(
                    args,
                    topology,
                    cycled_measurement_bodies,
                    cycled_measurement_truth,
                    selected_concurrency,
                    collect_resources=True,
                    executor=persistent_executor,
                )
                row = {
                    "policy": policy,
                    "repetition": len(policy_repetitions) + 1,
                    **summary,
                }
                repetition_rows.append(row)
                policy_repetitions.append(row)
                if len(policy_repetitions) == args.repetitions and coefficient_of_variation(
                    [item["qps"] for item in policy_repetitions]
                ) > 0.05:
                    target_repetitions = 5
        qps_values = [row["qps"] for row in policy_repetitions]
        qps_cv = coefficient_of_variation(qps_values)
        alg = algorithmic_by_policy[policy]
        final_rows.append(
            {
                "dataset": args.dataset,
                "logical_shards": 4,
                "policy": policy,
                "selected_concurrency": selected_concurrency,
                "concurrency_selection_rule": (
                    "lowest concurrency within 2% of peak tuning-set QPS"
                ),
                "repetitions": len(policy_repetitions),
                "recall_at_10": statistics.fmean(
                    row["recall_at_10"] for row in policy_repetitions
                ),
                "qps_mean": statistics.fmean(qps_values),
                "qps_cv": qps_cv,
                "mean_latency_us": statistics.fmean(
                    row["mean_latency_us"] for row in policy_repetitions
                ),
                "p50_latency_us": statistics.fmean(
                    row["p50_latency_us"] for row in policy_repetitions
                ),
                "p95_latency_us": statistics.fmean(
                    row["p95_latency_us"] for row in policy_repetitions
                ),
                "p99_latency_us": statistics.fmean(
                    row["p99_latency_us"] for row in policy_repetitions
                ),
                "mean_shards_per_query": alg["mean_shards_per_query"],
                "aggregate_distance_computations_per_query": alg[
                    "aggregate_distance_computations_per_query"
                ],
                "p95_max_shard_distance_computations": alg[
                    "p95_max_shard_distance_computations"
                ],
                "controller_combined_cpu_utilization_pct": statistics.fmean(
                    row["controller_combined_cpu_utilization_pct"]
                    for row in policy_repetitions
                ),
                "mean_worker_cpu_utilization_pct": statistics.fmean(
                    row["mean_worker_cpu_utilization_pct"]
                    for row in policy_repetitions
                ),
                "client_aggregator_cpu_utilization_pct": statistics.fmean(
                    row["client_cpu_utilization_pct"] for row in policy_repetitions
                ),
                "cluster_network_bytes_per_query": statistics.fmean(
                    row["cluster_network_bytes_per_query"]
                    for row in policy_repetitions
                ),
                "response_bytes_per_query": statistics.fmean(
                    row["response_bytes_per_query"] for row in policy_repetitions
                ),
                "official_measurement_query_count": len(measurement_routes),
                "measurement_cycles": args.measurement_cycles,
                "executed_measurement_requests_per_repetition": len(
                    cycled_measurement_bodies
                ),
                "protocol_query_count_exception": (
                    "official dataset has 10000 queries; 1000 tuning queries are excluded, "
                    "all 9000 remaining unique queries are measured in identical repeated cycles"
                ),
                "controller_cpu_includes_colocated_local_shard": True,
                "saturation_boundary_reached": saturation_boundary_reached,
                "stable_repetition_cv_gate": qps_cv <= 0.05,
                "physical_run_valid": (
                    saturation_boundary_reached
                    and qps_cv <= 0.05
                    and all(row["recall_at_10"] >= 0.90 for row in policy_repetitions)
                ),
            }
        )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_csv_atomic(output / f"{args.run_id}-concurrency-sweep.csv", sweep_rows)
    write_csv_atomic(output / f"{args.run_id}-repetitions.csv", repetition_rows)
    write_csv_atomic(output / "c6_physical_validation_summary.csv", final_rows)
    record = {
        "record_type": "c6_e5_physical_validation",
        "format_version": 1,
        "timestamp": utc_timestamp(),
        "run_id": args.run_id,
        "cluster_run_id": args.cluster_run_id,
        "policies": final_rows,
    }
    write_json_atomic(output / f"{args.run_id}-summary.json", record)
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
