#!/usr/bin/env python3
"""Measure selected Claim E execution modes with Docker CPU/network counters.

This is a container-level overhead supplement. It samples `docker stats` while
running selected Claim E query batches and records controller CPU%, controller
network byte deltas, aggregate Qdrant-cluster network byte deltas, and
container-level memory usage snapshots. It also samples process RSS through
Docker's `/containers/{name}/top` API.

It is not a physical NIC capture and does not measure HTTP framing separately.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import re
import subprocess
import threading
import time
import urllib.parse
from pathlib import Path
from typing import Any


DEFAULT_QDRANT_CONTAINERS = [
    "qdrant-controller-qdrant_controller-1",
    "qdrant-controller-qdrant_shard_1-1",
    "qdrant-controller-qdrant_shard_2-1",
    "qdrant-controller-qdrant_shard_3-1",
]

DEFAULT_CONTROLLER_CONTAINER = "qdrant-controller-qdrant_controller-1"

CAVEAT = (
    "docker_stats_and_top_container_level_counters_not_physical_nic_or_internal_attribution"
)


def load_module(path: str | Path, module_name: str) -> Any:
    module_path = Path(path)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_docker_bytes(raw: str) -> float:
    text = raw.strip()
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)\s*([A-Za-z]+)", text)
    if not match:
        raise ValueError(f"unsupported docker byte value: {raw!r}")
    value = float(match.group(1))
    unit = match.group(2)
    multipliers = {
        "B": 1,
        "kB": 1_000,
        "KB": 1_000,
        "MB": 1_000_000,
        "GB": 1_000_000_000,
        "TB": 1_000_000_000_000,
        "KiB": 1024,
        "MiB": 1024**2,
        "GiB": 1024**3,
        "TiB": 1024**4,
    }
    if unit not in multipliers:
        raise ValueError(f"unsupported docker byte unit: {unit!r}")
    return value * multipliers[unit]


def parse_net_io(raw: str) -> tuple[float, float]:
    parts = [part.strip() for part in raw.split("/")]
    if len(parts) != 2:
        raise ValueError(f"unsupported docker NetIO value: {raw!r}")
    return parse_docker_bytes(parts[0]), parse_docker_bytes(parts[1])


def parse_mem_usage(raw_usage: str, raw_pct: str) -> tuple[float, float, float]:
    parts = [part.strip() for part in raw_usage.split("/")]
    if len(parts) != 2:
        raise ValueError(f"unsupported docker MemUsage value: {raw_usage!r}")
    return parse_docker_bytes(parts[0]), parse_docker_bytes(parts[1]), parse_cpu_pct(raw_pct)


def parse_cpu_pct(raw: str) -> float:
    return float(raw.strip().rstrip("%"))


def parse_docker_top_process_rss(data: dict[str, Any]) -> dict[str, float]:
    titles = list(data.get("Titles") or [])
    try:
        rss_index = titles.index("RSS")
    except ValueError as exc:
        raise ValueError(f"docker top output missing RSS column: {titles!r}") from exc

    total_rss_kib = 0.0
    process_count = 0
    for process in data.get("Processes") or []:
        if rss_index >= len(process):
            continue
        total_rss_kib += float(process[rss_index])
        process_count += 1
    return {
        "process_count": process_count,
        "process_rss_bytes": total_rss_kib * 1024.0,
    }


def docker_top_process_rss_for_container(container_name: str) -> dict[str, float]:
    url_name = urllib.parse.quote(container_name, safe="")
    completed = subprocess.run(
        [
            "curl",
            "--silent",
            "--show-error",
            "--unix-socket",
            "/var/run/docker.sock",
            f"http://localhost/containers/{url_name}/top?ps_args=-eo%20pid,ppid,rss,comm,args",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return parse_docker_top_process_rss(json.loads(completed.stdout))


def raw_docker_stats_for_container(container_name: str) -> dict[str, float]:
    url_name = urllib.parse.quote(container_name, safe="")
    completed = subprocess.run(
        [
            "curl",
            "--silent",
            "--show-error",
            "--unix-socket",
            "/var/run/docker.sock",
            f"http://localhost/containers/{url_name}/stats?stream=false",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    data = json.loads(completed.stdout)
    cpu_stats = data.get("cpu_stats") or {}
    precpu_stats = data.get("precpu_stats") or {}
    cpu_usage = cpu_stats.get("cpu_usage") or {}
    precpu_usage = precpu_stats.get("cpu_usage") or {}
    cpu_delta = float(cpu_usage.get("total_usage", 0)) - float(precpu_usage.get("total_usage", 0))
    system_delta = float(cpu_stats.get("system_cpu_usage", 0)) - float(precpu_stats.get("system_cpu_usage", 0))
    online_cpus = cpu_stats.get("online_cpus") or len(cpu_usage.get("percpu_usage") or []) or 1
    cpu_pct = (cpu_delta / system_delta * float(online_cpus) * 100.0) if system_delta > 0 else 0.0
    networks = data.get("networks") or {}
    rx_bytes = sum(float(values.get("rx_bytes", 0)) for values in networks.values())
    tx_bytes = sum(float(values.get("tx_bytes", 0)) for values in networks.values())
    memory_stats = data.get("memory_stats") or {}
    mem_usage_bytes = float(memory_stats.get("usage", 0))
    mem_limit_bytes = float(memory_stats.get("limit", 0))
    mem_pct = (mem_usage_bytes / mem_limit_bytes * 100.0) if mem_limit_bytes > 0 else 0.0
    return {
        "cpu_pct": cpu_pct,
        "net_rx_bytes": rx_bytes,
        "net_tx_bytes": tx_bytes,
        "mem_usage_bytes": mem_usage_bytes,
        "mem_limit_bytes": mem_limit_bytes,
        "mem_pct": mem_pct,
    }


def docker_stats_snapshot_from_cli(container_names: list[str]) -> dict[str, dict[str, float]]:
    cmd = ["docker", "stats", "--no-stream", "--format", "{{json .}}", *container_names]
    completed = subprocess.run(cmd, check=True, capture_output=True, text=True)
    snapshot: dict[str, dict[str, float]] = {}
    for line in completed.stdout.splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        name = row["Name"]
        rx_bytes, tx_bytes = parse_net_io(row["NetIO"])
        mem_usage_bytes, mem_limit_bytes, mem_pct = parse_mem_usage(
            row["MemUsage"],
            row["MemPerc"],
        )
        snapshot[name] = {
            "cpu_pct": parse_cpu_pct(row["CPUPerc"]),
            "net_rx_bytes": rx_bytes,
            "net_tx_bytes": tx_bytes,
            "mem_usage_bytes": mem_usage_bytes,
            "mem_limit_bytes": mem_limit_bytes,
            "mem_pct": mem_pct,
        }
    return snapshot


def docker_stats_snapshot(container_names: list[str]) -> dict[str, dict[str, float]]:
    try:
        snapshot = {name: raw_docker_stats_for_container(name) for name in container_names}
    except Exception:
        snapshot = docker_stats_snapshot_from_cli(container_names)
    for name, values in snapshot.items():
        values.update(docker_top_process_rss_for_container(name))
    return snapshot


def summarize_container_samples(
    samples: list[dict[str, dict[str, float]]],
    *,
    controller_name: str,
    container_names: list[str],
) -> dict[str, Any]:
    if not samples:
        return {
            "sample_count": 0,
            "controller_cpu_pct_avg": 0.0,
            "controller_cpu_pct_max": 0.0,
            "cluster_cpu_pct_avg_sum": 0.0,
            "cluster_cpu_pct_max_sum": 0.0,
            "controller_net_rx_bytes_delta": 0.0,
            "controller_net_tx_bytes_delta": 0.0,
            "cluster_net_rx_bytes_delta": 0.0,
            "cluster_net_tx_bytes_delta": 0.0,
            "controller_mem_usage_bytes_avg": 0.0,
            "controller_mem_usage_bytes_max": 0.0,
            "controller_mem_pct_avg": 0.0,
            "controller_mem_pct_max": 0.0,
            "cluster_mem_usage_bytes_avg_sum": 0.0,
            "cluster_mem_usage_bytes_max_sum": 0.0,
            "cluster_mem_limit_bytes_max_sum": 0.0,
            "cluster_mem_pct_avg_sum": 0.0,
            "controller_process_rss_bytes_avg": 0.0,
            "controller_process_rss_bytes_max": 0.0,
            "cluster_process_rss_bytes_avg_sum": 0.0,
            "cluster_process_rss_bytes_max_sum": 0.0,
            "cluster_process_count_max_sum": 0.0,
        }

    controller_cpu = [
        float(sample.get(controller_name, {}).get("cpu_pct", 0.0))
        for sample in samples
    ]
    cluster_cpu_sums = [
        sum(float(sample.get(name, {}).get("cpu_pct", 0.0)) for name in container_names)
        for sample in samples
    ]
    controller_mem_usage = [
        float(sample.get(controller_name, {}).get("mem_usage_bytes", 0.0))
        for sample in samples
    ]
    controller_mem_pct = [
        float(sample.get(controller_name, {}).get("mem_pct", 0.0))
        for sample in samples
    ]
    cluster_mem_usage_sums = [
        sum(float(sample.get(name, {}).get("mem_usage_bytes", 0.0)) for name in container_names)
        for sample in samples
    ]
    cluster_mem_pct_sums = [
        sum(float(sample.get(name, {}).get("mem_pct", 0.0)) for name in container_names)
        for sample in samples
    ]
    cluster_mem_limit_sums = [
        sum(float(sample.get(name, {}).get("mem_limit_bytes", 0.0)) for name in container_names)
        for sample in samples
    ]
    controller_process_rss = [
        float(sample.get(controller_name, {}).get("process_rss_bytes", 0.0))
        for sample in samples
    ]
    cluster_process_rss_sums = [
        sum(float(sample.get(name, {}).get("process_rss_bytes", 0.0)) for name in container_names)
        for sample in samples
    ]
    cluster_process_count_sums = [
        sum(float(sample.get(name, {}).get("process_count", 0.0)) for name in container_names)
        for sample in samples
    ]
    first = samples[0]
    last = samples[-1]

    def delta(name: str, field: str) -> float:
        return float(last.get(name, {}).get(field, 0.0)) - float(first.get(name, {}).get(field, 0.0))

    return {
        "sample_count": len(samples),
        "controller_cpu_pct_avg": sum(controller_cpu) / len(controller_cpu),
        "controller_cpu_pct_max": max(controller_cpu),
        "cluster_cpu_pct_avg_sum": sum(cluster_cpu_sums) / len(cluster_cpu_sums),
        "cluster_cpu_pct_max_sum": max(cluster_cpu_sums),
        "controller_net_rx_bytes_delta": delta(controller_name, "net_rx_bytes"),
        "controller_net_tx_bytes_delta": delta(controller_name, "net_tx_bytes"),
        "cluster_net_rx_bytes_delta": sum(delta(name, "net_rx_bytes") for name in container_names),
        "cluster_net_tx_bytes_delta": sum(delta(name, "net_tx_bytes") for name in container_names),
        "controller_mem_usage_bytes_avg": sum(controller_mem_usage) / len(controller_mem_usage),
        "controller_mem_usage_bytes_max": max(controller_mem_usage),
        "controller_mem_pct_avg": sum(controller_mem_pct) / len(controller_mem_pct),
        "controller_mem_pct_max": max(controller_mem_pct),
        "cluster_mem_usage_bytes_avg_sum": sum(cluster_mem_usage_sums) / len(cluster_mem_usage_sums),
        "cluster_mem_usage_bytes_max_sum": max(cluster_mem_usage_sums),
        "cluster_mem_limit_bytes_max_sum": max(cluster_mem_limit_sums),
        "cluster_mem_pct_avg_sum": sum(cluster_mem_pct_sums) / len(cluster_mem_pct_sums),
        "controller_process_rss_bytes_avg": sum(controller_process_rss) / len(controller_process_rss),
        "controller_process_rss_bytes_max": max(controller_process_rss),
        "cluster_process_rss_bytes_avg_sum": sum(cluster_process_rss_sums) / len(cluster_process_rss_sums),
        "cluster_process_rss_bytes_max_sum": max(cluster_process_rss_sums),
        "cluster_process_count_max_sum": max(cluster_process_count_sums),
    }


class DockerStatsSampler:
    def __init__(self, container_names: list[str], interval_s: float) -> None:
        self.container_names = container_names
        self.interval_s = interval_s
        self.samples: list[dict[str, dict[str, float]]] = []
        self.sample_times: list[float] = []
        self.errors: list[str] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def collect_once(self) -> None:
        try:
            self.samples.append(docker_stats_snapshot(self.container_names))
            self.sample_times.append(time.time())
        except Exception as exc:  # pragma: no cover - surfaced in runtime CSV
            self.errors.append(f"{type(exc).__name__}: {exc}")

    def start(self) -> None:
        self.collect_once()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.wait(self.interval_s):
            self.collect_once()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval_s * 4))
        self.collect_once()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def flatten_samples(
    *,
    variant: str,
    repeat: int,
    batch_size: int,
    sampler: DockerStatsSampler,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample_index, sample in enumerate(sampler.samples):
        sample_ts = sampler.sample_times[sample_index] if sample_index < len(sampler.sample_times) else 0.0
        for container_name, values in sample.items():
            rows.append(
                {
                    "variant": variant,
                    "repeat": repeat,
                    "batch_size": batch_size,
                    "sample_index": sample_index,
                    "sample_unix_s": sample_ts,
                    "container": container_name,
                    "cpu_pct": values["cpu_pct"],
                    "net_rx_bytes": values["net_rx_bytes"],
                    "net_tx_bytes": values["net_tx_bytes"],
                    "mem_usage_bytes": values.get("mem_usage_bytes", 0.0),
                    "mem_limit_bytes": values.get("mem_limit_bytes", 0.0),
                    "mem_pct": values.get("mem_pct", 0.0),
                    "process_rss_bytes": values.get("process_rss_bytes", 0.0),
                    "process_count": values.get("process_count", 0.0),
                }
            )
    return rows


def run_variant_with_overhead(
    *,
    args: argparse.Namespace,
    q2l: Any,
    claim_e: Any,
    spec: dict[str, str],
    plans: list[dict[str, Any]],
    neighbors: Any,
    repeat: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    warmup = args.warmup_query_count
    if warmup:
        q2l.execute_query_plans_once(
            args.base_url,
            args.collection,
            plans[:warmup],
            neighbors[:warmup],
            args.top_k,
            lower_execution_order=spec["lower_execution_order"],
        )

    measured_plans = plans[warmup:]
    measured_neighbors = neighbors[warmup:]
    sampler = DockerStatsSampler(args.container, args.sample_interval_s)
    sampler.start()
    batch_rows: list[dict[str, Any]] = []
    try:
        for batch_index, start_idx in enumerate(range(0, len(measured_plans), args.batch_size)):
            end_idx = min(start_idx + args.batch_size, len(measured_plans))
            started = time.perf_counter()
            result = q2l.execute_query_plans_once(
                args.base_url,
                args.collection,
                measured_plans[start_idx:end_idx],
                measured_neighbors[start_idx:end_idx],
                args.top_k,
                lower_execution_order=spec["lower_execution_order"],
            )
            wall_s = time.perf_counter() - started
            batch_rows.append(
                {
                    "variant": spec["variant"],
                    "description": spec["description"],
                    "routed_execution_mode": spec["routed_execution_mode"],
                    "lower_execution_order": spec["lower_execution_order"],
                    "collection": args.collection,
                    "repeat": repeat,
                    "batch_size": args.batch_size,
                    "batch_index": batch_index,
                    "query_start": start_idx,
                    "query_end": end_idx,
                    "query_count": int(result["query_count"]),
                    "hits": int(result["hits"]),
                    "wall_s": wall_s,
                    "batch_latency_ms": wall_s * 1000.0,
                    "visited_shards": int(result["visited_shards"]),
                    "assigned_ef_sum": int(result["assigned_ef_sum"]),
                    "search_batch_calls": int(result["search_batch_calls"]),
                    "search_request_count": int(result.get("search_request_count", 0)),
                    "candidate_group_count": int(result.get("candidate_group_count", 0)),
                    "returned_candidate_count": int(result.get("returned_candidate_count", 0)),
                }
            )
    finally:
        sampler.stop()

    summary = claim_e.summarize_variant_rows(spec["variant"], batch_rows, args.top_k)
    summary.update(
        {
            "status": "ok",
            "description": spec["description"],
            "routed_execution_mode": spec["routed_execution_mode"],
            "lower_execution_order": spec["lower_execution_order"],
            "collection": args.collection,
            "repeat": repeat,
            "batch_size": args.batch_size,
            "upper_k": args.upper_k,
            "base_ef": args.base_ef,
            "factor": args.factor,
            "warmup_query_count": warmup,
            **summarize_container_samples(
                sampler.samples,
                controller_name=args.controller_container,
                container_names=args.container,
            ),
            "docker_stats_error_count": len(sampler.errors),
            "docker_stats_errors": "; ".join(sampler.errors[:3]),
            "caveat": CAVEAT,
        }
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return summary, batch_rows, flatten_samples(
        variant=spec["variant"],
        repeat=repeat,
        batch_size=args.batch_size,
        sampler=sampler,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:6833")
    parser.add_argument("--collection", default="qdrant_controller_idea_method4map_full_20260601")
    parser.add_argument("--routing-source-collection", default=None)
    parser.add_argument("--hdf5-path", default="/home/taig/dry/faiss/datasets/glove-200-angular.hdf5")
    parser.add_argument("--num-shards", type=int, default=46)
    parser.add_argument("--upper-k", type=int, default=160)
    parser.add_argument("--base-ef", type=int, default=80)
    parser.add_argument("--factor", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--eval-query-count", type=int, default=3000)
    parser.add_argument("--warmup-query-count", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--variant",
        action="append",
        default=None,
        choices=["grouped_by_ef_materialized", "compact_current", "client_shard_major_expanded"],
    )
    parser.add_argument("--sample-denominator", type=int, default=32)
    parser.add_argument("--upper-sample-seed", type=int, default=100)
    parser.add_argument("--upper-m", type=int, default=32)
    parser.add_argument("--upper-ef-construction", type=int, default=100)
    parser.add_argument("--upper-search-ef", type=int, default=200)
    parser.add_argument("--scroll-page-size", type=int, default=20000)
    parser.add_argument("--sample-interval-s", type=float, default=0.5)
    parser.add_argument("--controller-container", default=DEFAULT_CONTROLLER_CONTAINER)
    parser.add_argument("--container", action="append", default=None)
    parser.add_argument("--output-root", default="results/method4_claim_e_container_overhead_20260705")
    parser.add_argument("--qdrant-tool", default="tools/qdrant_two_level_routing_experiment.py")
    parser.add_argument("--claim-e-tool", default="tools/method4_claim_e_execution_mode_latency.py")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.container = args.container or DEFAULT_QDRANT_CONTAINERS
    if args.controller_container not in args.container:
        args.container = [args.controller_container, *args.container]

    q2l = load_module(args.qdrant_tool, "qdrant_two_level_routing_experiment")
    claim_e = load_module(args.claim_e_tool, "method4_claim_e_execution_mode_latency")
    specs = claim_e.variant_specs(args.variant)
    output_dir = Path(args.output_root) / time.strftime("analysis_%Y%m%d_%H%M%S", time.gmtime())
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"output_dir {output_dir}", flush=True)

    queries, upper_vectors, neighbors, train_count, dim = claim_e.load_queries_and_neighbors(args, q2l)
    upper_indices = q2l.global_upper_indices(train_count, args.sample_denominator, args.upper_sample_seed)
    upper_index = q2l.build_upper_index(
        upper_vectors,
        upper_indices.astype("int64", copy=False),
        dim,
        args.upper_m,
        args.upper_ef_construction,
        max(args.upper_search_ef, args.upper_k),
    )
    routing_source = args.routing_source_collection or args.collection
    point_to_shards = claim_e.recover_upper_membership(args, q2l, routing_source, upper_indices, train_count)

    plans_by_mode: dict[str, list[dict[str, Any]]] = {}
    for spec in specs:
        mode = spec["routed_execution_mode"]
        if mode not in plans_by_mode:
            plans_by_mode[mode] = claim_e.build_plans_for_execution_mode(
                q2l,
                queries,
                upper_index,
                point_to_shards,
                args.num_shards,
                args.top_k,
                args.upper_k,
                args.base_ef,
                args.factor,
                mode,
                train_count + 1,
            )

    summary_rows: list[dict[str, Any]] = []
    batch_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    for repeat in range(1, args.repeats + 1):
        ordered_specs = specs if repeat % 2 == 1 else list(reversed(specs))
        for spec in ordered_specs:
            summary, batches, samples = run_variant_with_overhead(
                args=args,
                q2l=q2l,
                claim_e=claim_e,
                spec=spec,
                plans=plans_by_mode[spec["routed_execution_mode"]],
                neighbors=neighbors,
                repeat=repeat,
            )
            summary_rows.append(summary)
            batch_rows.extend(batches)
            sample_rows.extend(samples)

    write_csv(output_dir / "claim_e_container_overhead_summary.csv", summary_rows)
    write_csv(output_dir / "claim_e_container_overhead_batches.csv", batch_rows)
    write_csv(output_dir / "claim_e_container_overhead_samples.csv", sample_rows)
    metadata = {
        "analysis_kind": "claim_e_container_cpu_network_overhead_supplement",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "base_url": args.base_url,
        "collection": args.collection,
        "routing_source_collection": routing_source,
        "hdf5_path": args.hdf5_path,
        "num_points": train_count,
        "num_shards": args.num_shards,
        "upper_k": args.upper_k,
        "base_ef": args.base_ef,
        "factor": args.factor,
        "top_k": args.top_k,
        "eval_query_count": args.eval_query_count,
        "warmup_query_count": args.warmup_query_count,
        "batch_size": args.batch_size,
        "repeats": args.repeats,
        "variants": specs,
        "controller_container": args.controller_container,
        "containers": args.container,
        "sample_interval_s": args.sample_interval_s,
        "notes": [
            CAVEAT,
            "CPU values are docker stats CPUPerc samples collected while measured batches execute.",
            "Network byte values are deltas of docker stats cumulative per-container NetIO counters.",
            "Memory values are docker stats container memory usage snapshots; they are not process RSS attribution.",
            "Process RSS values are sampled through Docker top and are not Qdrant subsystem attribution.",
            "This is a selected batch-size supplement, not a physical NIC packet capture.",
        ],
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"wrote {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
