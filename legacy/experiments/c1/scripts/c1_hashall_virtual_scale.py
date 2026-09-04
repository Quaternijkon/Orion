#!/usr/bin/env python3
"""Run native HashAll scaling on four hosts with M=1..32 and linear CPU quotas.

The serving CPU budget is defined relative to the M=32 point.  Four Qdrant
containers each receive 16 logical CPUs at M=32 (64 CPUs total).  At logical
shard count M the sum of all four container quotas is exactly 2*M CPUs.  Peers
without data shards retain a tiny control-plane floor; that floor is deducted
from the data-bearing peers so the total quota remains exact.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import importlib.util
import json
import os
import shlex
import statistics
import subprocess
import time
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import h5py
import numpy as np


def load_fixed_recall_module():
    source = Path(__file__).resolve().with_name("c1_hashall_fixed_recall.py")
    spec = importlib.util.spec_from_file_location("c1_hashall_fixed_recall_vscale", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fixed = load_fixed_recall_module()


LOGICAL_SHARD_COUNTS = (1, 2, 4, 8, 16, 32)
PHYSICAL_MACHINE_COUNT = 4
FULL_CORES_PER_NODE = 16.0
FULL_CLUSTER_CORES = PHYSICAL_MACHINE_COUNT * FULL_CORES_PER_NODE
CORES_PER_LOGICAL_SHARD = FULL_CLUSTER_CORES / max(LOGICAL_SHARD_COUNTS)
CONTROL_PLANE_FLOOR_CORES = 0.05
QDRANT_CPUSET = "0-7,16-23"
BENCHMARK_CPUSET = "8-15,24-31"
EXPECTED_IMAGE = "orion-c1:20260821-deterministic-seed"
EXPECTED_IMAGE_ID = "sha256:54819a6648dc706f654f3e541351c68814c0c1798312149c5e3ee099c183bfc8"
MAX_SEGMENT_SIZE_KB = 2_000_000
EXPECTED_SEGMENTS_PER_SHARD = 2


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    display_name: str
    hdf5_path: Path
    train_count: int
    vector_size: int
    hdf5_distance: str
    qdrant_distance: str
    collection_prefix: str
    output_directory: str
    ef_candidates: tuple[int, ...]
    max_ef: int


DATASET_SPECS = {
    "sift1m": DatasetSpec(
        key="sift1m",
        display_name="SIFT1M",
        hdf5_path=Path("/users/dry/orion-distributed/datasets/sift-128-euclidean.hdf5"),
        train_count=1_000_000,
        vector_size=128,
        hdf5_distance="euclidean",
        qdrant_distance="Euclid",
        collection_prefix="hashall_vscale_linear_cpu",
        output_directory="hashall-virtual-linear-cpu-20260824",
        ef_candidates=(10, 12, 14, 16, 18, 20, 22, 24, 28, 32, 40, 48, 64),
        max_ef=64,
    ),
    "glove-200-angular": DatasetSpec(
        key="glove-200-angular",
        display_name="GloVe-200-angular",
        hdf5_path=Path("/users/dry/orion-distributed/datasets/glove-200-angular.hdf5"),
        train_count=1_183_514,
        vector_size=200,
        hdf5_distance="angular",
        qdrant_distance="Cosine",
        collection_prefix="hashall_vscale_linear_cpu_glove",
        output_directory="hashall-virtual-linear-cpu-glove-20260825",
        ef_candidates=(
            10,
            16,
            24,
            32,
            48,
            64,
            80,
            96,
            128,
            160,
            192,
            256,
            320,
            384,
            512,
        ),
        max_ef=512,
    ),
}


@dataclass(frozen=True)
class Node:
    ssh_host: str
    private_ip: str
    container: str


NODES = (
    Node("localhost", "10.10.1.1", "orion-dist-c1-20260821-v2-controller"),
    Node("10.10.1.2", "10.10.1.2", "orion-dist-c1-20260821-v2-qdrant_shard_1"),
    Node("10.10.1.3", "10.10.1.3", "orion-dist-c1-20260821-v2-qdrant_shard_2"),
    Node("10.10.1.4", "10.10.1.4", "orion-dist-c1-20260821-v2-qdrant_shard_3"),
)


def utc_timestamp() -> str:
    import datetime

    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def load_experiment_module(repo_root: Path):
    source = repo_root / "tools" / "qdrant_two_level_routing_experiment.py"
    spec = importlib.util.spec_from_file_location("qdrant_experiment_vscale", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_int_csv(value: str) -> list[int]:
    result = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not result:
        raise ValueError("expected at least one integer")
    return result


def shard_counts(logical_shards: int) -> list[int]:
    if logical_shards not in LOGICAL_SHARD_COUNTS:
        raise ValueError(f"unsupported logical shard count: {logical_shards}")
    return [
        logical_shards // PHYSICAL_MACHINE_COUNT
        + (1 if index < logical_shards % PHYSICAL_MACHINE_COUNT else 0)
        for index in range(PHYSICAL_MACHINE_COUNT)
    ]


def cpuset_cpu_count(value: str) -> int:
    cpus: set[int] = set()
    for part in value.split(","):
        token = part.strip()
        if not token:
            continue
        if "-" in token:
            start_text, stop_text = token.split("-", 1)
            cpus.update(range(int(start_text), int(stop_text) + 1))
        else:
            cpus.add(int(token))
    if not cpus:
        raise ValueError(f"empty CPU set: {value!r}")
    return len(cpus)


def quota_schedule(
    logical_shards: int,
    *,
    control_plane_floor: float = CONTROL_PLANE_FLOOR_CORES,
    shards_per_node: Sequence[int] | None = None,
) -> list[float]:
    """Return exact per-container CPU quotas whose sum is 2*M."""
    counts = (
        shard_counts(logical_shards)
        if shards_per_node is None
        else [int(value) for value in shards_per_node]
    )
    if len(counts) != PHYSICAL_MACHINE_COUNT:
        raise ValueError(
            f"shards_per_node must contain {PHYSICAL_MACHINE_COUNT} values"
        )
    if any(value < 0 for value in counts) or sum(counts) != logical_shards:
        raise ValueError(
            f"shards_per_node must be non-negative and sum to {logical_shards}: {counts}"
        )
    total = CORES_PER_LOGICAL_SHARD * logical_shards
    floor_total = control_plane_floor * PHYSICAL_MACHINE_COUNT
    if control_plane_floor <= 0 or floor_total >= total:
        raise ValueError("control-plane floor is incompatible with the total CPU budget")
    distributable = total - floor_total
    quotas = [
        round(control_plane_floor + distributable * count / logical_shards, 6)
        for count in counts
    ]
    if not np.isclose(sum(quotas), total, rtol=0.0, atol=1e-12):
        raise AssertionError("quota schedule does not preserve the exact total")
    if logical_shards == 32 and any(
        not np.isclose(value, FULL_CORES_PER_NODE, rtol=0.0, atol=1e-12)
        for value in quotas
    ):
        raise AssertionError("M=32 must expose the full per-node compute budget")
    return quotas


def remote_command(
    node: Node,
    command: str,
    *,
    timeout: float = 120.0,
    capture: bool = True,
) -> str:
    if node.ssh_host in {"localhost", "127.0.0.1", "::1"}:
        argv = ["bash", "-lc", command]
    else:
        argv = [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=5",
            node.ssh_host,
            "bash -lc " + shlex.quote(command),
        ]
    completed = subprocess.run(
        argv,
        check=True,
        capture_output=capture,
        text=True,
        timeout=timeout,
    )
    return completed.stdout.strip() if capture else ""


def inspect_container(node: Node) -> dict[str, Any]:
    raw = remote_command(
        node,
        "sudo -n docker inspect " + shlex.quote(node.container),
    )
    payload = json.loads(raw)[0]
    host = payload["HostConfig"]
    cpu_max = remote_command(
        node,
        "sudo -n docker exec "
        + shlex.quote(node.container)
        + " cat /sys/fs/cgroup/cpu.max",
    )
    affinity = remote_command(
        node,
        "sudo -n docker exec "
        + shlex.quote(node.container)
        + " awk '/Cpus_allowed_list:/ {print $2}' /proc/1/status",
    )
    return {
        "ssh_host": node.ssh_host,
        "private_ip": node.private_ip,
        "container": node.container,
        "container_id": payload["Id"],
        "image": payload["Config"]["Image"],
        "image_id": payload["Image"],
        "cpuset": host["CpusetCpus"],
        "nano_cpus": int(host["NanoCpus"]),
        "cpu_quota": int(host["CpuQuota"]),
        "cpu_period": int(host["CpuPeriod"]),
        "cpu_max": cpu_max,
        "runtime_affinity": affinity,
        "mounts": payload["Mounts"],
        "environment": payload["Config"]["Env"],
    }


def inspect_all_containers() -> list[dict[str, Any]]:
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        return list(executor.map(inspect_container, NODES))


def update_container_resources(node: Node, *, cpuset: str, cpus: float) -> None:
    if cpus <= 0:
        raise ValueError("CPU quota must be positive")
    command = (
        "set -euo pipefail; sudo -n docker update --cpuset-cpus "
        + shlex.quote(cpuset)
        + " --cpus "
        + shlex.quote(f"{cpus:.6f}")
        + " "
        + shlex.quote(node.container)
        + " >/dev/null"
    )
    remote_command(node, command)


def apply_resource_schedule(
    logical_shards: int | None,
    *,
    output: Path,
    shards_per_node: Sequence[int] | None = None,
) -> dict[str, Any]:
    if logical_shards is None and shards_per_node is not None:
        raise ValueError("shards_per_node is only valid for a measurement budget")
    resolved_shard_counts = (
        None
        if logical_shards is None
        else (
            shard_counts(logical_shards)
            if shards_per_node is None
            else [int(value) for value in shards_per_node]
        )
    )
    quotas = (
        [FULL_CORES_PER_NODE] * PHYSICAL_MACHINE_COUNT
        if logical_shards is None
        else quota_schedule(
            logical_shards,
            shards_per_node=resolved_shard_counts,
        )
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(
                update_container_resources,
                node,
                cpuset=QDRANT_CPUSET,
                cpus=quota,
            )
            for node, quota in zip(NODES, quotas, strict=True)
        ]
        for future in futures:
            future.result()
    snapshots = inspect_all_containers()
    for snapshot, expected in zip(snapshots, quotas, strict=True):
        actual = snapshot["nano_cpus"] / 1_000_000_000.0
        if snapshot["cpuset"] != QDRANT_CPUSET:
            raise RuntimeError(f"cpuset mismatch: {snapshot}")
        if snapshot["runtime_affinity"] != QDRANT_CPUSET:
            raise RuntimeError(f"runtime affinity mismatch: {snapshot}")
        if not np.isclose(actual, expected, rtol=0.0, atol=1e-6):
            raise RuntimeError(
                f"CPU quota mismatch on {snapshot['private_ip']}: {actual} != {expected}"
            )
    record = {
        "timestamp": utc_timestamp(),
        "record_type": "hashall_virtual_linear_cpu_resource_schedule",
        "mode": "full_build_budget" if logical_shards is None else "measurement_budget",
        "logical_shards": logical_shards,
        "physical_machines": PHYSICAL_MACHINE_COUNT,
        "qdrant_cpuset": QDRANT_CPUSET,
        "benchmark_cpuset": BENCHMARK_CPUSET,
        "full_cluster_cores": FULL_CLUSTER_CORES,
        "cores_per_logical_shard": CORES_PER_LOGICAL_SHARD,
        "control_plane_floor_cores_per_peer": CONTROL_PLANE_FLOOR_CORES,
        "shards_per_node": resolved_shard_counts,
        "shard_allocation_source": (
            "not_applicable_full_build_budget"
            if logical_shards is None
            else (
                "nominal_round_robin"
                if shards_per_node is None
                else "live_collection_placement"
            )
        ),
        "quota_cores_per_node": quotas,
        "quota_cores_total": round(sum(quotas), 6),
        "expected_total": FULL_CLUSTER_CORES
        if logical_shards is None
        else CORES_PER_LOGICAL_SHARD * logical_shards,
        "containers": snapshots,
        "status": "PASS",
    }
    write_json(output, record)
    return record


def restore_resource_state(original: Sequence[dict[str, Any]], output: Path) -> dict[str, Any]:
    failures: list[str] = []
    restored: list[dict[str, Any]] = []
    for node, before in zip(NODES, original, strict=True):
        try:
            nano = int(before["nano_cpus"])
            if nano > 0:
                cpus = nano / 1_000_000_000.0
                update_container_resources(node, cpuset=str(before["cpuset"]), cpus=cpus)
            else:
                # Docker's update API does not clear an existing NanoCPUs value
                # when --cpus=0 is supplied.  Restore the original *effective*
                # unlimited state by setting a quota larger than the restored
                # cpuset can consume.  The saved pre-experiment JSON retains the
                # exact original metadata for audit.
                cpus = float(os.cpu_count() or 32)
                update_container_resources(
                    node,
                    cpuset=str(before["cpuset"]),
                    cpus=cpus,
                )
            after = inspect_container(node)
            if nano == 0:
                quota_value, period_value = after["cpu_max"].split()
                effective_unlimited = quota_value == "max" or (
                    float(quota_value) / float(period_value)
                    >= cpuset_cpu_count(str(before["cpuset"]))
                )
                exact_metadata = after["nano_cpus"] == 0
                ok = after["cpuset"] == before["cpuset"] and effective_unlimited
            else:
                effective_unlimited = False
                exact_metadata = after["nano_cpus"] == nano
                ok = after["cpuset"] == before["cpuset"] and exact_metadata
            if not ok:
                failures.append(node.private_ip)
            restored.append(
                {
                    "before": before,
                    "after": after,
                    "exact_docker_metadata_restored": exact_metadata,
                    "effective_unlimited_restored": effective_unlimited,
                    "valid": ok,
                }
            )
        except Exception as error:
            failures.append(node.private_ip)
            restored.append({"before": before, "valid": False, "error": repr(error)})
    record = {
        "timestamp": utc_timestamp(),
        "record_type": "hashall_virtual_linear_cpu_resource_restore",
        "status": "PASS" if not failures else "FAIL",
        "nodes": restored,
    }
    write_json(output, record)
    if failures:
        raise RuntimeError(f"failed to restore resources on {failures}")
    return record


def cpu_stat(node: Node) -> dict[str, int]:
    raw = remote_command(
        node,
        "sudo -n docker exec "
        + shlex.quote(node.container)
        + " cat /sys/fs/cgroup/cpu.stat",
        timeout=30.0,
    )
    result: dict[str, int] = {}
    for line in raw.splitlines():
        fields = line.split()
        if len(fields) == 2:
            result[fields[0]] = int(fields[1])
    if "usage_usec" not in result:
        raise RuntimeError(f"usage_usec missing for {node.private_ip}")
    return result


def cpu_stat_snapshot() -> dict[str, dict[str, int]]:
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        values = list(executor.map(cpu_stat, NODES))
    return {node.private_ip: value for node, value in zip(NODES, values, strict=True)}


def stat_delta(
    before: dict[str, dict[str, int]], after: dict[str, dict[str, int]]
) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for host in before:
        keys = set(before[host]) | set(after[host])
        result[host] = {
            key: int(after[host].get(key, 0) - before[host].get(key, 0))
            for key in sorted(keys)
        }
    return result


def timed_run(
    host: str,
    port: int,
    path: str,
    bodies: list[bytes],
    *,
    concurrency: int,
    duration_s: float,
    batch_size: int,
) -> dict[str, Any]:
    import http.client
    import threading

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
                started = time.perf_counter()
                connection.request(
                    "POST",
                    path,
                    body=body,
                    headers={"Content-Type": "application/json"},
                )
                response = connection.getresponse()
                payload = response.read()
                latencies_ms.append((time.perf_counter() - started) * 1000.0)
                if response.status != 200:
                    raise RuntimeError(f"HTTP {response.status}: {payload[:1000]!r}")
                completed += 1
                response_bytes += len(payload)
        finally:
            connection.close()
        return completed, response_bytes, latencies_ms

    cpu_before = cpu_stat_snapshot()
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(worker, worker_id) for worker_id in range(concurrency)]
        barrier.wait()
        rows = [future.result() for future in futures]
    wall_s = time.perf_counter() - started
    cpu_after = cpu_stat_snapshot()
    completed_batches = sum(row[0] for row in rows)
    query_count = completed_batches * batch_size
    latency_values = [value for row in rows for value in row[2]]
    deltas = stat_delta(cpu_before, cpu_after)
    cpu_cores = {
        host_name: values["usage_usec"] / (wall_s * 1_000_000.0)
        for host_name, values in deltas.items()
    }
    return {
        "wall_s": wall_s,
        "concurrency": concurrency,
        "completed_batches": completed_batches,
        "query_count": query_count,
        "qps": query_count / wall_s,
        "response_bytes": sum(row[1] for row in rows),
        "batch_latency_ms": fixed.percentiles(latency_values),
        "cpu_stat_before": cpu_before,
        "cpu_stat_after": cpu_after,
        "cpu_stat_delta": deltas,
        "cpu_average_cores": cpu_cores,
        "cpu_average_cores_total": sum(cpu_cores.values()),
    }


def cluster_snapshot(exp, base_url: str) -> dict[str, Any]:
    return exp.request_json(base_url, "GET", "/cluster")["result"]


def wait_cluster_ready(exp, base_url: str, timeout: float = 180.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    last: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        last = cluster_snapshot(exp, base_url)
        peer_uris = {
            str(row["uri"]).rstrip("/") for row in (last.get("peers") or {}).values()
        }
        expected = {f"http://10.10.1.{index}:6335" for index in range(1, 5)}
        if (
            peer_uris == expected
            and int((last.get("raft_info") or {}).get("pending_operations") or 0) == 0
        ):
            return last
        time.sleep(1.0)
    raise TimeoutError(f"cluster did not stabilize: {last}")


def collection_body(
    logical_shards: int, dataset: DatasetSpec | None = None
) -> dict[str, Any]:
    dataset = dataset or DATASET_SPECS["sift1m"]
    return {
        "vectors": {
            "size": dataset.vector_size,
            "distance": dataset.qdrant_distance,
        },
        "shard_number": logical_shards,
        "sharding_method": "auto",
        "replication_factor": 1,
        "write_consistency_factor": 1,
        "on_disk_payload": True,
        "hnsw_config": {
            "m": 32,
            "ef_construct": 200,
            "full_scan_threshold": 10,
            "max_indexing_threads": 1,
            "on_disk": False,
        },
        "optimizers_config": {
            "default_segment_number": 1,
            "indexing_threshold": 10,
            "max_optimization_threads": 1,
            "max_segment_size_kb": MAX_SEGMENT_SIZE_KB,
        },
    }


def ordered_peer_ids(exp, base_url: str) -> tuple[list[int], dict[int, str]]:
    _, peers, _ = exp.cluster_peer_map(base_url)
    peer_by_ip = {
        str(uri).split("//", 1)[1].split(":", 1)[0]: int(peer_id)
        for peer_id, uri in peers.items()
    }
    ordered = [peer_by_ip[node.private_ip] for node in NODES]
    return ordered, {int(peer_id): str(uri) for peer_id, uri in peers.items()}


def per_node_collection_cluster(exp, collection: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for node in NODES:
        base_url = f"http://{node.private_ip}:6333"
        result[node.private_ip] = exp.request_json(
            base_url,
            "GET",
            f"/collections/{urllib.parse.quote(collection, safe='')}/cluster",
        )["result"]
    return result


def audit_local_shards(
    per_node: dict[str, Any], logical_shards: int
) -> dict[str, Any]:
    observed_counts: list[int] = []
    point_counts: dict[str, dict[str, int]] = {}
    all_ids: list[int] = []
    for node in NODES:
        rows = per_node[node.private_ip].get("local_shards") or []
        observed_counts.append(len(rows))
        point_counts[node.private_ip] = {
            str(int(row["shard_id"])): int(row.get("points_count") or 0) for row in rows
        }
        all_ids.extend(int(row["shard_id"]) for row in rows)
    expected_counts = shard_counts(logical_shards)
    valid = observed_counts == expected_counts and sorted(all_ids) == list(
        range(logical_shards)
    )
    return {
        "status": "PASS" if valid else "FAIL",
        "logical_shards": logical_shards,
        "expected_shards_per_node": expected_counts,
        "observed_shards_per_node": observed_counts,
        "local_point_counts": point_counts,
        "all_shard_ids": sorted(all_ids),
    }


def prepare_collection(
    args: argparse.Namespace,
    exp,
    *,
    logical_shards: int,
    collection: str,
) -> dict[str, Any]:
    base_url = args.base_url
    exp.delete_collection_if_exists(base_url, collection)
    created = exp.request_json(
        base_url,
        "PUT",
        f"/collections/{urllib.parse.quote(collection, safe='')}",
        body=collection_body(logical_shards, args.dataset_spec),
    )
    peer_ids, peers = ordered_peer_ids(exp, base_url)
    target = {
        shard_id: peer_ids[shard_id % PHYSICAL_MACHINE_COUNT]
        for shard_id in range(logical_shards)
    }
    move = exp.move_numeric_shards_explicit(
        base_url,
        collection,
        peer_ids,
        target,
        expected_shard_count=logical_shards,
        include_controller=True,
        transfer_method="snapshot",
    )
    with h5py.File(args.hdf5_path, "r") as handle:
        train = np.asarray(handle["train"], dtype=np.float32)
    upload_started = time.monotonic()
    upload = exp.upsert_numeric_auto_points(
        base_url,
        collection,
        train,
        batch_size=args.upload_batch_size,
        timeout=900.0,
    )
    upload["seconds"] = time.monotonic() - upload_started
    info = exp.wait_collection_indexed(
        base_url, collection, len(train), timeout_sec=args.index_timeout
    )
    cluster = exp.collection_cluster_info(base_url, collection)
    placement = exp.discover_numeric_shard_placement(
        base_url, collection, expected_shard_count=logical_shards
    )
    per_node = per_node_collection_cluster(exp, collection)
    shard_audit = audit_local_shards(per_node, logical_shards)
    expected_segments = EXPECTED_SEGMENTS_PER_SHARD * logical_shards
    observed_segments = int(info.get("segments_count") or 0)
    segment_gate = {
        "status": "PASS" if observed_segments == expected_segments else "FAIL",
        "expected_total_segments": expected_segments,
        "observed_total_segments": observed_segments,
        "expected_nonempty_hnsw_segments": logical_shards,
        "expected_empty_appendable_segments": logical_shards,
        "expected_segments_per_shard": EXPECTED_SEGMENTS_PER_SHARD,
    }
    if placement != target:
        raise RuntimeError(f"placement mismatch: expected={target}, actual={placement}")
    if shard_audit["status"] != "PASS":
        raise RuntimeError(f"local shard audit failed: {shard_audit}")
    if segment_gate["status"] != "PASS":
        raise RuntimeError(f"single-HNSW-per-shard gate failed: {segment_gate}")
    record = {
        "timestamp": utc_timestamp(),
        "record_type": "hashall_virtual_linear_cpu_prepare",
        "dataset": args.dataset_spec.display_name,
        "dataset_key": args.dataset_spec.key,
        "logical_shards": logical_shards,
        "physical_machines": PHYSICAL_MACHINE_COUNT,
        "collection": collection,
        "created": created,
        "collection_body": collection_body(logical_shards, args.dataset_spec),
        "target_placement": target,
        "actual_placement": placement,
        "peer_uris": peers,
        "move": move,
        "upload": upload,
        "collection_info": info,
        "collection_cluster": cluster,
        "per_node_collection_cluster": per_node,
        "shard_layout_gate": shard_audit,
        "single_hnsw_segment_gate": segment_gate,
        "status": "PASS",
    }
    return record


def select_saturation(sweep: list[dict[str, Any]]) -> dict[str, Any]:
    if not sweep:
        raise ValueError("empty concurrency sweep")
    best = max(float(row["qps"]) for row in sweep)
    eligible = [row for row in sweep if float(row["qps"]) >= best * 0.98]
    selected = min(eligible, key=lambda row: int(row["concurrency"]))
    max_candidate = max(int(row["concurrency"]) for row in sweep)
    max_row = next(row for row in sweep if int(row["concurrency"]) == max_candidate)
    return {
        "selection_rule": "minimum concurrency within 98% of best observed sweep QPS",
        "best_sweep_qps": best,
        "selected_concurrency": int(selected["concurrency"]),
        "selected_qps": float(selected["qps"]),
        "max_candidate": max_candidate,
        "max_candidate_qps": float(max_row["qps"]),
        "knee_observed": int(selected["concurrency"]) < max_candidate,
    }


def run_benchmark(
    args: argparse.Namespace,
    exp,
    *,
    logical_shards: int,
    collection: str,
    resource_record: dict[str, Any],
) -> dict[str, Any]:
    parsed = urllib.parse.urlparse(args.base_url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 6333
    endpoint = (
        f"/collections/{urllib.parse.quote(collection, safe='')}/points/search/batch"
    )
    queries, neighbors = fixed.load_dataset(args.hdf5_path, args.top_k)
    tuning_queries, tuning_neighbors = queries[:1000], neighbors[:1000]
    heldout_queries, heldout_neighbors = queries[1000:10_000], neighbors[1000:10_000]

    tuning_sweep: list[dict[str, Any]] = []
    selected_ef: int | None = None
    tuning_target = args.target_recall + args.tuning_margin
    for hnsw_ef in args.ef_candidates:
        row = fixed.recall_probe(
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

    selected_index = args.ef_candidates.index(selected_ef)
    heldout: dict[str, Any] | None = None
    selection_reason = "smallest ef reaching tuning target with safety margin"
    while selected_index < len(args.ef_candidates):
        selected_ef = args.ef_candidates[selected_index]
        heldout = fixed.recall_probe(
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
        selection_reason = "advanced because the disjoint held-out recall missed target"
        selected_index += 1
    if heldout is None or heldout["recall_at_10"] < args.target_recall:
        raise RuntimeError("bounded ef grid did not pass held-out recall")

    bodies = fixed.make_bodies(
        heldout_queries,
        batch_size=args.batch_size,
        top_k=args.top_k,
        hnsw_ef=selected_ef,
    )
    concurrency_sweep = [
        timed_run(
            host,
            port,
            endpoint,
            bodies,
            concurrency=concurrency,
            duration_s=args.sweep_seconds,
            batch_size=args.batch_size,
        )
        for concurrency in args.concurrency_candidates
    ]
    selection = select_saturation(concurrency_sweep)
    if not selection["knee_observed"]:
        raise RuntimeError(
            "saturation knee not observed; extend concurrency grid before formal measurement"
        )
    selected_concurrency = int(selection["selected_concurrency"])
    warmup = timed_run(
        host,
        port,
        endpoint,
        bodies,
        concurrency=selected_concurrency,
        duration_s=args.warmup_seconds,
        batch_size=args.batch_size,
    )
    repeats: list[dict[str, Any]] = []
    while len(repeats) < args.min_repeats:
        repeats.append(
            timed_run(
                host,
                port,
                endpoint,
                bodies,
                concurrency=selected_concurrency,
                duration_s=args.measure_seconds,
                batch_size=args.batch_size,
            )
        )
    while len(repeats) < args.max_repeats:
        qps_now = [float(row["qps"]) for row in repeats]
        cv_now = statistics.stdev(qps_now) / statistics.fmean(qps_now)
        if cv_now <= args.max_cv:
            break
        repeats.append(
            timed_run(
                host,
                port,
                endpoint,
                bodies,
                concurrency=selected_concurrency,
                duration_s=args.measure_seconds,
                batch_size=args.batch_size,
            )
        )
    qps_values = [float(row["qps"]) for row in repeats]
    qps_mean = statistics.fmean(qps_values)
    qps_stdev = statistics.stdev(qps_values) if len(qps_values) > 1 else 0.0
    qps_cv = qps_stdev / qps_mean
    if qps_cv > args.max_cv:
        raise RuntimeError(f"formal QPS CV remains too high: {qps_cv:.4f}")

    per_node_cluster = per_node_collection_cluster(exp, collection)
    return {
        "timestamp": utc_timestamp(),
        "record_type": "native_hashall_virtual_linear_cpu_fixed_recall",
        "logical_shards": logical_shards,
        "physical_machines": PHYSICAL_MACHINE_COUNT,
        "base_url": args.base_url,
        "collection": collection,
        "request_contract": {
            "sharding_method": "auto",
            "standard_coordinator_request": True,
            "shard_selector_present": False,
            "client_side_fanout": False,
            "endpoint": endpoint,
            "hashall_fanout": logical_shards,
        },
        "resource_contract": resource_record,
        "protocol": {
            "dataset": args.dataset_spec.display_name,
            "dataset_key": args.dataset_spec.key,
            "vector_size": args.dataset_spec.vector_size,
            "distance": args.dataset_spec.qdrant_distance,
            "tuning_query_range": [0, 1000],
            "heldout_query_range": [1000, 10000],
            "target_recall_at_10": args.target_recall,
            "tuning_margin": args.tuning_margin,
            "ef_lower_bound": args.top_k,
            "ef_upper_bound": max(args.ef_candidates),
            "ef_selection_rule": selection_reason,
        },
        "parameters": {
            "selected_hnsw_ef": selected_ef,
            "top_k": args.top_k,
            "batch_size": args.batch_size,
            "selected_concurrency": selected_concurrency,
            "sweep_seconds": args.sweep_seconds,
            "warmup_seconds": args.warmup_seconds,
            "measure_seconds": args.measure_seconds,
            "repeat_count": len(repeats),
        },
        "server": fixed.request_json(args.base_url + "/"),
        "collection_info": fixed.request_json(
            args.base_url + f"/collections/{urllib.parse.quote(collection, safe='')}"
        )["result"],
        "collection_cluster": fixed.request_json(
            args.base_url
            + f"/collections/{urllib.parse.quote(collection, safe='')}/cluster"
        )["result"],
        "per_node_collection_cluster": per_node_cluster,
        "tuning_sweep": tuning_sweep,
        "heldout_recall": heldout,
        "concurrency_sweep": concurrency_sweep,
        "saturation_selection": selection,
        "warmup": warmup,
        "repeats": repeats,
        "qps_mean": qps_mean,
        "qps_stdev": qps_stdev,
        "qps_cv": qps_cv,
        "status": "PASS",
    }


def verify_client_affinity() -> dict[str, Any]:
    actual = sorted(os.sched_getaffinity(0))
    expected = sorted(
        set(range(8, 16)).union(range(24, 32))
    )
    valid = actual == expected
    record = {
        "expected": expected,
        "actual": actual,
        "expected_cpuset": BENCHMARK_CPUSET,
        "valid": valid,
    }
    if not valid:
        raise RuntimeError(
            "benchmark client affinity mismatch; run this script with "
            f"taskset -c {BENCHMARK_CPUSET}"
        )
    return record


def preflight(args: argparse.Namespace, exp) -> dict[str, Any]:
    client_affinity = verify_client_affinity()
    containers = inspect_all_containers()
    for snapshot in containers:
        if snapshot["image"] != EXPECTED_IMAGE or snapshot["image_id"] != EXPECTED_IMAGE_ID:
            raise RuntimeError(f"unexpected image identity: {snapshot}")
        environment = set(snapshot["environment"])
        required = {
            "QDRANT__STORAGE__PERFORMANCE__MAX_SEARCH_THREADS=16",
            "QDRANT__SERVICE__HARDWARE_REPORTING=true",
            "QDRANT_HNSW_GRAPH_BUILD_SEED=20260821",
        }
        if not required.issubset(environment):
            raise RuntimeError(f"required environment missing: {snapshot}")
    cluster = wait_cluster_ready(exp, args.base_url)
    collections = exp.request_json(args.base_url, "GET", "/collections")["result"][
        "collections"
    ]
    if collections:
        raise RuntimeError(f"preflight requires an empty cluster, found: {collections}")
    with h5py.File(args.hdf5_path, "r") as handle:
        shapes = {key: list(handle[key].shape) for key in ("train", "test", "neighbors")}
        hdf5_distance = str(handle.attrs.get("distance", ""))
    expected_train_shape = [
        args.dataset_spec.train_count,
        args.dataset_spec.vector_size,
    ]
    if shapes["train"] != expected_train_shape or shapes["test"][:1] != [10_000]:
        raise RuntimeError(
            f"unexpected {args.dataset_spec.display_name} dataset shapes: {shapes}"
        )
    if hdf5_distance != args.dataset_spec.hdf5_distance:
        raise RuntimeError(
            f"unexpected dataset distance: {hdf5_distance!r} != "
            f"{args.dataset_spec.hdf5_distance!r}"
        )
    schedule = {
        str(m): {
            "shards_per_node": shard_counts(m),
            "quota_cores_per_node": quota_schedule(m),
            "quota_cores_total": round(sum(quota_schedule(m)), 6),
        }
        for m in LOGICAL_SHARD_COUNTS
    }
    return {
        "timestamp": utc_timestamp(),
        "record_type": "hashall_virtual_linear_cpu_preflight",
        "status": "PASS",
        "client_affinity": client_affinity,
        "containers": containers,
        "cluster": cluster,
        "collections": collections,
        "dataset": {
            "key": args.dataset_spec.key,
            "name": args.dataset_spec.display_name,
            "path": str(args.hdf5_path),
            "shapes": shapes,
            "hdf5_distance": hdf5_distance,
            "qdrant_distance": args.dataset_spec.qdrant_distance,
        },
        "resource_schedule": schedule,
    }


def collection_name(args: argparse.Namespace, logical_shards: int) -> str:
    return f"{args.dataset_spec.collection_prefix}_m{logical_shards}_20260825"


def execute(args: argparse.Namespace) -> int:
    repo_root = Path(__file__).resolve().parents[3]
    exp = load_experiment_module(repo_root)
    args.output_root.mkdir(parents=True, exist_ok=True)

    original_path = args.output_root / "original-resource-state.json"
    if args.restore_only:
        original = json.loads(original_path.read_text(encoding="utf-8"))["containers"]
        restore_resource_state(original, args.output_root / "resource-restored.json")
        return 0

    original = inspect_all_containers()
    if not original_path.exists():
        write_json(
            original_path,
            {
                "timestamp": utc_timestamp(),
                "record_type": "hashall_virtual_linear_cpu_original_resources",
                "containers": original,
            },
        )
    preflight_record = preflight(args, exp)
    write_json(args.output_root / "preflight.json", preflight_record)
    if args.preflight_only:
        return 0

    # A resumed point must not leave a stale top-level terminal record from an
    # earlier failed or partial attempt in the same dataset-specific directory.
    (args.output_root / "execution-failed.json").unlink(missing_ok=True)
    (args.output_root / "execution-complete.json").unlink(missing_ok=True)

    completed: list[int] = []
    body_error: BaseException | None = None
    try:
        for logical_shards in args.logical_shards:
            point_root = args.output_root / f"m{logical_shards}"
            collection = collection_name(args, logical_shards)
            print(
                f"[hashall-vscale] prepare M={logical_shards} with full build budget",
                flush=True,
            )
            build_resources = apply_resource_schedule(
                None, output=point_root / "resources-build.json"
            )
            wait_cluster_ready(exp, args.base_url)
            prepare = prepare_collection(
                args,
                exp,
                logical_shards=logical_shards,
                collection=collection,
            )
            prepare["build_resource_contract"] = build_resources
            write_json(point_root / "prepare.json", prepare)

            print(
                f"[hashall-vscale] apply exact measurement quota for M={logical_shards}",
                flush=True,
            )
            resources = apply_resource_schedule(
                logical_shards, output=point_root / "resources-measure.json"
            )
            wait_cluster_ready(exp, args.base_url)
            print(f"[hashall-vscale] benchmark M={logical_shards}", flush=True)
            benchmark = run_benchmark(
                args,
                exp,
                logical_shards=logical_shards,
                collection=collection,
                resource_record=resources,
            )
            write_json(point_root / "benchmark.json", benchmark)
            completed.append(logical_shards)

            apply_resource_schedule(None, output=point_root / "resources-cleanup.json")
            exp.delete_collection_if_exists(args.base_url, collection)
            remaining = exp.request_json(args.base_url, "GET", "/collections")["result"][
                "collections"
            ]
            if remaining:
                raise RuntimeError(f"collection cleanup failed: {remaining}")
            write_json(
                point_root / "cleanup.json",
                {
                    "timestamp": utc_timestamp(),
                    "record_type": "hashall_virtual_linear_cpu_cleanup",
                    "dataset": args.dataset_spec.display_name,
                    "dataset_key": args.dataset_spec.key,
                    "logical_shards": logical_shards,
                    "deleted_collection": collection,
                    "remaining_collections": remaining,
                    "status": "PASS",
                },
            )
            print(f"[hashall-vscale] completed M={logical_shards}", flush=True)
        write_json(
            args.output_root / "execution-complete.json",
            {
                "timestamp": utc_timestamp(),
                "record_type": "hashall_virtual_linear_cpu_execution",
                "status": "MEASUREMENTS_COMPLETE",
                "dataset": args.dataset_spec.display_name,
                "dataset_key": args.dataset_spec.key,
                "logical_shards": completed,
                "physical_machines": PHYSICAL_MACHINE_COUNT,
                "full_cluster_cores": FULL_CLUSTER_CORES,
                "cores_per_logical_shard": CORES_PER_LOGICAL_SHARD,
            },
        )
        return 0
    except BaseException as error:
        body_error = error
        write_json(
            args.output_root / "execution-failed.json",
            {
                "timestamp": utc_timestamp(),
                "record_type": "hashall_virtual_linear_cpu_execution",
                "status": "FAILED",
                "dataset": args.dataset_spec.display_name,
                "dataset_key": args.dataset_spec.key,
                "completed_logical_shards": completed,
                "error": repr(error),
            },
        )
        raise
    finally:
        for logical_shards in args.logical_shards:
            try:
                exp.delete_collection_if_exists(
                    args.base_url, collection_name(args, logical_shards)
                )
            except Exception:
                pass
        saved_original = json.loads(original_path.read_text(encoding="utf-8"))["containers"]
        try:
            restore_resource_state(
                saved_original, args.output_root / "resource-restored.json"
            )
        except Exception:
            if body_error is None:
                raise
            print("[hashall-vscale] resource restoration also failed", flush=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASET_SPECS),
        default="sift1m",
    )
    parser.add_argument("--base-url", default="http://10.10.1.1:6333")
    parser.add_argument(
        "--hdf5-path",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
    )
    parser.add_argument("--logical-shards", default="1,2,4,8,16,32")
    parser.add_argument("--upload-batch-size", type=int, default=2000)
    parser.add_argument("--index-timeout", type=float, default=10_800.0)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--target-recall", type=float, default=0.90)
    parser.add_argument("--tuning-margin", type=float, default=0.002)
    parser.add_argument("--ef-candidates", default=None)
    parser.add_argument("--concurrency-candidates", default="1,2,4,8,16,32,64")
    parser.add_argument("--sweep-seconds", type=float, default=8.0)
    parser.add_argument("--warmup-seconds", type=float, default=10.0)
    parser.add_argument("--measure-seconds", type=float, default=20.0)
    parser.add_argument("--min-repeats", type=int, default=5)
    parser.add_argument("--max-repeats", type=int, default=7)
    parser.add_argument("--max-cv", type=float, default=0.05)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--restore-only", action="store_true")
    args = parser.parse_args(argv)
    args.dataset_spec = DATASET_SPECS[args.dataset]
    if args.hdf5_path is None:
        args.hdf5_path = args.dataset_spec.hdf5_path
    if args.output_root is None:
        args.output_root = Path(
            "/proj/intelisys-PG0/exp/orion-distributed"
        ) / args.dataset_spec.output_directory
    if args.ef_candidates is None:
        args.ef_candidates = ",".join(map(str, args.dataset_spec.ef_candidates))
    args.hdf5_path = args.hdf5_path.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    args.logical_shards = parse_int_csv(args.logical_shards)
    if any(value not in LOGICAL_SHARD_COUNTS for value in args.logical_shards):
        raise ValueError(f"logical shards must be a subset of {LOGICAL_SHARD_COUNTS}")
    args.ef_candidates = parse_int_csv(args.ef_candidates)
    if (
        min(args.ef_candidates) < args.top_k
        or max(args.ef_candidates) > args.dataset_spec.max_ef
    ):
        raise ValueError(
            "ef candidates must stay within "
            f"[top_k, {args.dataset_spec.max_ef}] for {args.dataset_spec.key}"
        )
    args.concurrency_candidates = parse_int_csv(args.concurrency_candidates)
    if args.concurrency_candidates != sorted(set(args.concurrency_candidates)):
        raise ValueError("concurrency candidates must be sorted and unique")
    if args.min_repeats < 2 or args.max_repeats < args.min_repeats:
        raise ValueError("invalid repeat bounds")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    return execute(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
