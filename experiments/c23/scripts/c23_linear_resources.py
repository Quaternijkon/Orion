#!/usr/bin/env python3
"""Manage the four-host, one-container-per-shard C23 resource envelope.

The module keeps infrastructure mutation narrow and reversible: existing
containers are stopped but never removed, experiment containers have an
unambiguous label/name prefix, and cleanup verifies restoration by container
ID. Formal experiment runners should use :func:`isolated_shard_cluster` so a
Python exception still triggers cleanup.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import shlex
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence


PROTOCOL_VERSION = "c23-linear-20260824-v1"
CONTAINER_PREFIX = "c23lr-"
ALLOWED_LOGICAL_SHARDS = (1, 2, 4, 8, 16, 32)
DEFAULT_TOPOLOGY = (
    Path(__file__).resolve().parents[1]
    / "retests"
    / "stage13-linear-resource-virtualized"
    / "topology.json"
)


@dataclass(frozen=True)
class Node:
    index: int
    ssh_host: str
    private_ip: str


@dataclass(frozen=True)
class ShardPlacement:
    shard_id: int
    node_index: int
    ssh_host: str
    private_ip: str
    host_slot: int
    physical_core: int
    cpuset: str
    http_port: int
    grpc_port: int

    @property
    def base_url(self) -> str:
        return f"http://{self.private_ip}:{self.http_port}"


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def write_json_atomic(path: str | Path, payload: Any) -> None:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary)
        raise


def load_topology(path: str | Path = DEFAULT_TOPOLOGY) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError(f"unexpected protocol version in {source}")
    nodes = payload.get("nodes")
    if not isinstance(nodes, list) or len(nodes) != 4:
        raise ValueError("linear-resource topology requires exactly four nodes")
    indices = [int(node["index"]) for node in nodes]
    if indices != list(range(4)):
        raise ValueError(f"node indices must be [0,1,2,3], got {indices}")
    private_ips = [str(node["private_ip"]) for node in nodes]
    if len(set(private_ips)) != 4:
        raise ValueError("node private IPs must be unique")
    logical_shards = tuple(int(value) for value in payload["logical_shards"])
    if logical_shards != ALLOWED_LOGICAL_SHARDS:
        raise ValueError(
            f"logical-shard matrix must be {ALLOWED_LOGICAL_SHARDS}, got {logical_shards}"
        )
    cores = [int(value) for value in payload["server_physical_cores_per_host"]]
    if len(cores) != 8 or len(set(cores)) != 8 or min(cores) < 0:
        raise ValueError("server CPU pool must contain eight unique physical cores")
    if int(payload["smt_sibling_offset"]) <= max(cores):
        raise ValueError("SMT sibling offset overlaps the physical-core CPU IDs")
    memory = int(payload["memory_bytes_per_shard"])
    memory_swap = int(payload["memory_swap_bytes_per_shard"])
    if memory <= 0 or memory_swap != memory:
        raise ValueError("memory and memory+swap cap must be equal and positive")
    if not str(payload["image"]["id"]).startswith("sha256:"):
        raise ValueError("topology must pin a Docker image ID")
    return payload


def nodes_from_topology(topology: dict[str, Any]) -> list[Node]:
    return [
        Node(
            index=int(row["index"]),
            ssh_host=str(row["ssh_host"]),
            private_ip=str(row["private_ip"]),
        )
        for row in topology["nodes"]
    ]


def shard_placements(
    topology: dict[str, Any], logical_shards: int
) -> list[ShardPlacement]:
    if logical_shards not in ALLOWED_LOGICAL_SHARDS:
        raise ValueError(
            f"logical_shards must be one of {ALLOWED_LOGICAL_SHARDS}, got {logical_shards}"
        )
    nodes = nodes_from_topology(topology)
    cores = [int(value) for value in topology["server_physical_cores_per_host"]]
    sibling = int(topology["smt_sibling_offset"])
    http_base = int(topology["http_base_port"])
    grpc_base = int(topology["grpc_base_port"])
    result: list[ShardPlacement] = []
    for shard_id in range(logical_shards):
        node = nodes[shard_id % len(nodes)]
        slot = shard_id // len(nodes)
        physical_core = cores[slot]
        result.append(
            ShardPlacement(
                shard_id=shard_id,
                node_index=node.index,
                ssh_host=node.ssh_host,
                private_ip=node.private_ip,
                host_slot=slot,
                physical_core=physical_core,
                cpuset=f"{physical_core},{physical_core + sibling}",
                http_port=http_base + slot,
                grpc_port=grpc_base + slot,
            )
        )
    return result


def resource_contract(topology: dict[str, Any], logical_shards: int) -> dict[str, Any]:
    placements = shard_placements(topology, logical_shards)
    per_host = {
        str(index): sum(row.node_index == index for row in placements)
        for index in range(4)
    }
    memory_per_shard = int(topology["memory_bytes_per_shard"])
    return {
        "logical_shards": logical_shards,
        "physical_hosts_with_shards": sum(value > 0 for value in per_host.values()),
        "physical_core_equivalents": logical_shards,
        "m32_capacity_fraction": logical_shards / 32.0,
        "memory_bytes_per_shard": memory_per_shard,
        "total_memory_capacity_bytes": memory_per_shard * logical_shards,
        "shards_per_host": per_host,
        "placement": [asdict(row) | {"base_url": row.base_url} for row in placements],
        "checks": {
            "one_physical_core_per_logical_shard": "PASS",
            "total_cpu_capacity_equals_M": "PASS"
            if len(placements) == logical_shards
            else "FAIL",
            "m32_uses_all_declared_server_cores": "PASS"
            if logical_shards != 32
            or all(value == 8 for value in per_host.values())
            else "FAIL",
            "linear_memory_capacity": "PASS",
        },
    }


def _remote_argv(ssh_host: str, command: str) -> list[str]:
    if ssh_host in {"localhost", "127.0.0.1", "::1"}:
        return ["bash", "-lc", command]
    return [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=5",
        ssh_host,
        "bash -lc " + shlex.quote(command),
    ]


def remote_command(ssh_host: str, command: str, *, timeout: float = 60.0) -> str:
    completed = subprocess.run(
        _remote_argv(ssh_host, command),
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if completed.returncode:
        raise RuntimeError(
            f"remote command failed on {ssh_host} with rc={completed.returncode}: "
            f"{completed.stderr.strip() or completed.stdout.strip()}"
        )
    return completed.stdout.strip()


def list_containers(node: Node) -> list[dict[str, str]]:
    template = "{{.ID}}\\t{{.Names}}\\t{{.State}}\\t{{.Status}}\\t{{.Image}}"
    raw = remote_command(
        node.ssh_host,
        "sudo -n docker ps -a --no-trunc --format " + shlex.quote(template),
    )
    rows: list[dict[str, str]] = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t", 4)
        if len(parts) != 5:
            raise RuntimeError(f"unexpected docker ps row on {node.ssh_host}: {line!r}")
        rows.append(
            {
                "id": parts[0],
                "name": parts[1],
                "state": parts[2],
                "status": parts[3],
                "image": parts[4],
            }
        )
    return rows


def host_preflight(node: Node, topology: dict[str, Any]) -> dict[str, Any]:
    hostname = remote_command(node.ssh_host, "hostname")
    cpu_online = remote_command(
        node.ssh_host, "cat /sys/devices/system/cpu/online"
    )
    logical_cpu_count = int(remote_command(node.ssh_host, "nproc"))
    physical_core_count = int(
        remote_command(
            node.ssh_host,
            "lscpu -p=CORE,SOCKET | sed '/^#/d' | sort -u | wc -l",
        )
    )
    memory_kib = int(
        remote_command(node.ssh_host, "awk '/MemTotal:/ {print $2}' /proc/meminfo")
    )
    cgroup = remote_command(node.ssh_host, "stat -fc %T /sys/fs/cgroup")
    docker_version = remote_command(
        node.ssh_host, "sudo -n docker version --format '{{.Server.Version}}'"
    )
    image_id = remote_command(
        node.ssh_host,
        "sudo -n docker image inspect --format '{{.Id}}' "
        + shlex.quote(str(topology["image"]["tag"])),
    )
    core_rows = remote_command(
        node.ssh_host,
        "lscpu -p=CPU,CORE,SOCKET | sed '/^#/d'",
    )
    sibling_offset = int(topology["smt_sibling_offset"])
    server_cores = [int(value) for value in topology["server_physical_cores_per_host"]]
    mapping: dict[int, list[int]] = {}
    for raw in core_rows.splitlines():
        cpu, core, _socket = map(int, raw.split(","))
        mapping.setdefault(core, []).append(cpu)
    expected_siblings = {
        core: [core, core + sibling_offset] for core in server_cores
    }
    actual_siblings = {core: sorted(mapping.get(core, [])) for core in server_cores}
    checks = {
        "logical_cpu_count_is_32": logical_cpu_count == 32,
        "physical_core_count_is_16": physical_core_count == 16,
        "cgroup_v2": cgroup == "cgroup2fs",
        "image_id_matches": image_id == str(topology["image"]["id"]),
        "server_smt_pairs_match": actual_siblings == expected_siblings,
        "memory_supports_m32_host_cap": memory_kib * 1024
        > int(topology["memory_bytes_per_shard"]) * 8,
    }
    return {
        "node": asdict(node),
        "hostname": hostname,
        "cpu_online": cpu_online,
        "logical_cpu_count": logical_cpu_count,
        "physical_core_count": physical_core_count,
        "memory_kib": memory_kib,
        "cgroup_filesystem": cgroup,
        "docker_version": docker_version,
        "image_id": image_id,
        "server_core_siblings": actual_siblings,
        "containers": list_containers(node),
        "checks": {key: "PASS" if value else "FAIL" for key, value in checks.items()},
        "status": "PASS" if all(checks.values()) else "FAIL",
    }


def preflight(topology: dict[str, Any]) -> dict[str, Any]:
    nodes = nodes_from_topology(topology)
    hosts = [host_preflight(node, topology) for node in nodes]
    stale_experiment = [
        {"node": host["node"], "container": row}
        for host in hosts
        for row in host["containers"]
        if row["name"].startswith(CONTAINER_PREFIX)
    ]
    checks = {
        "four_hosts_pass": all(host["status"] == "PASS" for host in hosts),
        "no_stale_experiment_containers": not stale_experiment,
        "all_scale_contracts_pass": all(
            all(value == "PASS" for value in resource_contract(topology, m)["checks"].values())
            for m in ALLOWED_LOGICAL_SHARDS
        ),
    }
    return {
        "protocol_version": PROTOCOL_VERSION,
        "timestamp": utc_timestamp(),
        "hosts": hosts,
        "resource_contracts": [
            resource_contract(topology, value) for value in ALLOWED_LOGICAL_SHARDS
        ],
        "stale_experiment_containers": stale_experiment,
        "checks": {key: "PASS" if value else "FAIL" for key, value in checks.items()},
        "status": "PASS" if all(checks.values()) else "FAIL",
    }


def sanitize_run_id(value: str) -> str:
    normalized = value.strip().lower()
    if not re.fullmatch(r"[a-z0-9][a-z0-9_.-]{0,47}", normalized):
        raise ValueError("run ID must be 1-48 lowercase letters, digits, dot, underscore, or dash")
    return normalized


def container_name(run_id: str, shard_id: int) -> str:
    return f"{CONTAINER_PREFIX}{sanitize_run_id(run_id)}-s{shard_id:02d}"


def suspend_existing_containers(topology: dict[str, Any], run_id: str) -> dict[str, Any]:
    expected_prefix = f"{CONTAINER_PREFIX}{sanitize_run_id(run_id)}-"
    records: list[dict[str, Any]] = []
    for node in nodes_from_topology(topology):
        containers = list_containers(node)
        unexpected_experiment = [
            row
            for row in containers
            if row["name"].startswith(CONTAINER_PREFIX)
            and not row["name"].startswith(expected_prefix)
        ]
        if unexpected_experiment:
            raise RuntimeError(
                f"stale experiment containers on {node.ssh_host}: {unexpected_experiment}"
            )
        running = [
            row
            for row in containers
            if row["state"] == "running" and not row["name"].startswith(CONTAINER_PREFIX)
        ]
        if running:
            ids = [row["id"] for row in running]
            remote_command(
                node.ssh_host,
                "sudo -n docker stop " + " ".join(shlex.quote(value) for value in ids),
                timeout=180.0,
            )
        records.append(
            {
                "node": asdict(node),
                "running_before": running,
                "containers_before": containers,
            }
        )
    return {
        "protocol_version": PROTOCOL_VERSION,
        "timestamp": utc_timestamp(),
        "run_id": sanitize_run_id(run_id),
        "nodes": records,
        "status": "PASS",
    }


def _assert_ports_free(placement: ShardPlacement) -> None:
    for port in (placement.http_port, placement.grpc_port):
        command = (
            "if ss -ltnH 'sport = :"
            + str(port)
            + "' | grep -q .; then exit 17; fi"
        )
        try:
            remote_command(placement.ssh_host, command)
        except RuntimeError as exc:
            raise RuntimeError(
                f"port {port} is already in use on {placement.ssh_host}"
            ) from exc


def start_shard_containers(
    topology: dict[str, Any], run_id: str, logical_shards: int
) -> list[dict[str, Any]]:
    run_id = sanitize_run_id(run_id)
    image = str(topology["image"]["tag"])
    memory = int(topology["memory_bytes_per_shard"])
    memory_swap = int(topology["memory_swap_bytes_per_shard"])
    tmpfs_size = min(memory * 3 // 4, 6 * 1024**3)
    graph_seed = int(topology["hnsw"]["graph_seed"])
    started: list[dict[str, Any]] = []
    try:
        for placement in shard_placements(topology, logical_shards):
            name = container_name(run_id, placement.shard_id)
            existing = [
                row for row in list_containers(Node(placement.node_index, placement.ssh_host, placement.private_ip))
                if row["name"] == name
            ]
            if existing:
                raise RuntimeError(f"container already exists: {placement.ssh_host}:{name}")
            _assert_ports_free(placement)
            command = [
                "sudo",
                "-n",
                "docker",
                "run",
                "-d",
                "--name",
                name,
                "--network",
                "host",
                "--cpuset-cpus",
                placement.cpuset,
                "--memory",
                str(memory),
                "--memory-swap",
                str(memory_swap),
                "--pids-limit",
                "512",
                "--tmpfs",
                f"/qdrant/storage:rw,nosuid,nodev,size={tmpfs_size}",
                "--label",
                f"orion.c23.protocol={PROTOCOL_VERSION}",
                "--label",
                f"orion.c23.run_id={run_id}",
                "--label",
                f"orion.c23.shard_id={placement.shard_id}",
                "-e",
                "QDRANT__CLUSTER__ENABLED=false",
                "-e",
                f"QDRANT__SERVICE__HTTP_PORT={placement.http_port}",
                "-e",
                f"QDRANT__SERVICE__GRPC_PORT={placement.grpc_port}",
                "-e",
                "QDRANT__SERVICE__HARDWARE_REPORTING=true",
                "-e",
                "QDRANT__STORAGE__PERFORMANCE__MAX_SEARCH_THREADS=2",
                "-e",
                "QDRANT__STORAGE__PERFORMANCE__OPTIMIZER_CPU_BUDGET=1",
                "-e",
                f"QDRANT_HNSW_GRAPH_BUILD_SEED={graph_seed}",
                image,
            ]
            container_id = remote_command(
                placement.ssh_host,
                " ".join(shlex.quote(value) for value in command),
                timeout=180.0,
            )
            started.append(
                {
                    "container_id": container_id,
                    "container_name": name,
                    "placement": asdict(placement) | {"base_url": placement.base_url},
                }
            )
        wait_for_health([row["placement"] for row in started], timeout=120.0)
        return started
    except BaseException:
        with contextlib.suppress(BaseException):
            remove_experiment_containers(topology, run_id)
        raise


def wait_for_health(placements: Sequence[dict[str, Any]], *, timeout: float) -> None:
    pending = {str(row["base_url"]): None for row in placements}
    deadline = time.monotonic() + timeout
    while pending and time.monotonic() < deadline:
        for base_url in list(pending):
            try:
                with urllib.request.urlopen(base_url + "/healthz", timeout=2.0) as response:
                    if response.status == 200:
                        pending.pop(base_url, None)
            except (urllib.error.URLError, TimeoutError, OSError):
                pass
        if pending:
            time.sleep(0.5)
    if pending:
        raise TimeoutError(f"Qdrant health timeout: {sorted(pending)}")


def _container_runtime(
    topology: dict[str, Any], placement: ShardPlacement, run_id: str
) -> dict[str, Any]:
    name = container_name(run_id, placement.shard_id)
    raw = remote_command(
        placement.ssh_host,
        "sudo -n docker inspect " + shlex.quote(name),
    )
    inspect = json.loads(raw)[0]
    pid = int(inspect["State"]["Pid"])
    runtime_cpuset = remote_command(
        placement.ssh_host,
        "awk '/Cpus_allowed_list:/ {print $2}' " + shlex.quote(f"/proc/{pid}/status"),
    )
    cgroup_path = remote_command(
        placement.ssh_host,
        "awk -F: '$1 == 0 {print $3}' " + shlex.quote(f"/proc/{pid}/cgroup"),
    )
    if not cgroup_path.startswith("/") or ".." in cgroup_path.split("/"):
        raise RuntimeError(f"unsafe cgroup path for {name}: {cgroup_path!r}")
    cgroup_root = "/sys/fs/cgroup" + cgroup_path

    def read_cgroup(filename: str) -> str:
        return remote_command(
            placement.ssh_host,
            "cat " + shlex.quote(f"{cgroup_root}/{filename}"),
        )

    host_config = inspect["HostConfig"]
    config = inspect["Config"]
    labels = config.get("Labels") or {}
    return {
        "container_id": str(inspect["Id"]),
        "container_name": str(inspect["Name"]).removeprefix("/"),
        "image_id": str(inspect["Image"]),
        "running": bool(inspect["State"]["Running"]),
        "pid": pid,
        "docker_cpuset": str(host_config.get("CpusetCpus") or ""),
        "runtime_cpuset": runtime_cpuset,
        "cgroup_cpuset_effective": read_cgroup("cpuset.cpus.effective"),
        "docker_memory": int(host_config.get("Memory") or 0),
        "docker_memory_swap": int(host_config.get("MemorySwap") or 0),
        "cgroup_memory_max": read_cgroup("memory.max"),
        "cgroup_memory_swap_max": read_cgroup("memory.swap.max"),
        "cgroup_memory_current": int(read_cgroup("memory.current")),
        "cgroup_cpu_max": read_cgroup("cpu.max"),
        "cgroup_cpu_stat": read_cgroup("cpu.stat"),
        "oom_killed": bool(inspect["State"].get("OOMKilled")),
        "labels": labels,
        "health_url": placement.base_url + "/healthz",
    }


def verify_cluster(
    topology: dict[str, Any], run_id: str, logical_shards: int
) -> dict[str, Any]:
    run_id = sanitize_run_id(run_id)
    placements = shard_placements(topology, logical_shards)
    expected_names = {
        container_name(run_id, placement.shard_id) for placement in placements
    }
    running_by_host: dict[str, list[dict[str, str]]] = {}
    for node in nodes_from_topology(topology):
        running_by_host[node.ssh_host] = [
            row for row in list_containers(node) if row["state"] == "running"
        ]
    unexpected_running = [
        {"ssh_host": host, "container": row}
        for host, rows in running_by_host.items()
        for row in rows
        if row["name"] not in expected_names
    ]
    runtime = [
        {
            "placement": asdict(placement) | {"base_url": placement.base_url},
            "runtime": _container_runtime(topology, placement, run_id),
        }
        for placement in placements
    ]
    memory = int(topology["memory_bytes_per_shard"])
    memory_swap = int(topology["memory_swap_bytes_per_shard"])
    image_id = str(topology["image"]["id"])
    per_container_checks: list[dict[str, Any]] = []
    for row in runtime:
        placement = row["placement"]
        actual = row["runtime"]
        checks = {
            "running": actual["running"],
            "image_id": actual["image_id"] == image_id,
            "docker_cpuset": actual["docker_cpuset"] == placement["cpuset"],
            "runtime_cpuset": actual["runtime_cpuset"] == placement["cpuset"],
            "cgroup_cpuset": set(actual["cgroup_cpuset_effective"].split(","))
            == set(placement["cpuset"].split(",")),
            "docker_memory": actual["docker_memory"] == memory,
            "docker_memory_swap": actual["docker_memory_swap"] == memory_swap,
            "cgroup_memory_max": actual["cgroup_memory_max"] == str(memory),
            "no_swap": actual["cgroup_memory_swap_max"] == "0",
            "not_oom_killed": not actual["oom_killed"],
            "protocol_label": actual["labels"].get("orion.c23.protocol")
            == PROTOCOL_VERSION,
            "run_label": actual["labels"].get("orion.c23.run_id") == run_id,
            "shard_label": actual["labels"].get("orion.c23.shard_id")
            == str(placement["shard_id"]),
        }
        per_container_checks.append(
            {
                "container_name": actual["container_name"],
                "checks": {key: "PASS" if value else "FAIL" for key, value in checks.items()},
                "status": "PASS" if all(checks.values()) else "FAIL",
            }
        )
    cpusets_by_host: dict[int, list[str]] = {index: [] for index in range(4)}
    for placement in placements:
        cpusets_by_host[placement.node_index].append(placement.cpuset)
    disjoint = all(
        len({cpu for value in values for cpu in value.split(",")}) == 2 * len(values)
        for values in cpusets_by_host.values()
    )
    checks = {
        "exact_container_count": len(runtime) == logical_shards,
        "all_containers_pass": all(row["status"] == "PASS" for row in per_container_checks),
        "pairwise_disjoint_cpusets": disjoint,
        "no_foreign_running_containers": not unexpected_running,
        "resource_contract": all(
            value == "PASS"
            for value in resource_contract(topology, logical_shards)["checks"].values()
        ),
    }
    return {
        "protocol_version": PROTOCOL_VERSION,
        "timestamp": utc_timestamp(),
        "run_id": run_id,
        "logical_shards": logical_shards,
        "resource_contract": resource_contract(topology, logical_shards),
        "runtime": runtime,
        "per_container_checks": per_container_checks,
        "unexpected_running_containers": unexpected_running,
        "checks": {key: "PASS" if value else "FAIL" for key, value in checks.items()},
        "status": "PASS" if all(checks.values()) else "FAIL",
    }


def remove_experiment_containers(topology: dict[str, Any], run_id: str) -> list[dict[str, Any]]:
    run_id = sanitize_run_id(run_id)
    prefix = f"{CONTAINER_PREFIX}{run_id}-"
    removed: list[dict[str, Any]] = []
    for node in nodes_from_topology(topology):
        matches = [row for row in list_containers(node) if row["name"].startswith(prefix)]
        for row in matches:
            remote_command(
                node.ssh_host,
                "sudo -n docker rm -f " + shlex.quote(row["id"]),
                timeout=180.0,
            )
            removed.append({"node": asdict(node), "container": row})
    return removed


def restore_existing_containers(
    topology: dict[str, Any], suspended: dict[str, Any]
) -> dict[str, Any]:
    restored: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    node_by_index = {node.index: node for node in nodes_from_topology(topology)}
    for record in suspended.get("nodes") or []:
        node = node_by_index[int(record["node"]["index"])]
        for before in record.get("running_before") or []:
            try:
                remote_command(
                    node.ssh_host,
                    "sudo -n docker start " + shlex.quote(str(before["id"])),
                    timeout=180.0,
                )
                current = {row["id"]: row for row in list_containers(node)}.get(before["id"])
                valid = bool(current and current["state"] == "running" and current["name"] == before["name"])
                row = {
                    "node": asdict(node),
                    "before": before,
                    "after": current,
                    "valid": valid,
                }
                restored.append(row)
                if not valid:
                    failures.append(row)
            except BaseException as error:
                row = {
                    "node": asdict(node),
                    "before": before,
                    "valid": False,
                    "error": repr(error),
                }
                restored.append(row)
                failures.append(row)
    return {
        "protocol_version": PROTOCOL_VERSION,
        "timestamp": utc_timestamp(),
        "restored": restored,
        "failures": failures,
        "status": "PASS" if not failures else "FAIL",
    }


@contextlib.contextmanager
def isolated_shard_cluster(
    topology: dict[str, Any],
    run_id: str,
    logical_shards: int,
    *,
    state_output: str | Path | None = None,
) -> Iterator[dict[str, Any]]:
    run_id = sanitize_run_id(run_id)
    suspended = suspend_existing_containers(topology, run_id)
    if state_output is not None:
        write_json_atomic(
            state_output,
            {
                "protocol_version": PROTOCOL_VERSION,
                "timestamp": utc_timestamp(),
                "run_id": run_id,
                "logical_shards": logical_shards,
                "phase": "foreign_containers_suspended",
                "suspended": suspended,
            },
        )
    try:
        started = start_shard_containers(topology, run_id, logical_shards)
        verification = verify_cluster(topology, run_id, logical_shards)
        if verification["status"] != "PASS":
            raise RuntimeError(f"resource verification failed: {verification['checks']}")
        if state_output is not None:
            write_json_atomic(
                state_output,
                {
                    "protocol_version": PROTOCOL_VERSION,
                    "timestamp": utc_timestamp(),
                    "run_id": run_id,
                    "logical_shards": logical_shards,
                    "phase": "experiment_cluster_verified",
                    "suspended": suspended,
                    "started": started,
                    "verification": verification,
                },
            )
        yield {
            "run_id": run_id,
            "logical_shards": logical_shards,
            "suspended": suspended,
            "started": started,
            "verification": verification,
        }
    finally:
        removed = remove_experiment_containers(topology, run_id)
        restored = restore_existing_containers(topology, suspended)
        if state_output is not None:
            write_json_atomic(
                state_output,
                {
                    "protocol_version": PROTOCOL_VERSION,
                    "timestamp": utc_timestamp(),
                    "run_id": run_id,
                    "logical_shards": logical_shards,
                    "phase": "cleanup_complete",
                    "suspended": suspended,
                    "removed": removed,
                    "restored": restored,
                },
            )
        if restored["status"] != "PASS":
            raise RuntimeError(f"failed to restore pre-existing containers: {restored['failures']}")


def run_smoke(
    topology: dict[str, Any], run_id: str, logical_shards: int, output: str | Path
) -> dict[str, Any]:
    pre = preflight(topology)
    if pre["status"] != "PASS":
        raise RuntimeError(f"preflight failed: {pre['checks']}")
    output_path = Path(output).expanduser().resolve()
    state_path = output_path.with_name(output_path.stem + "-state.json")
    active: dict[str, Any] | None = None
    with isolated_shard_cluster(
        topology,
        run_id,
        logical_shards,
        state_output=state_path,
    ) as cluster:
        active = cluster["verification"]
    cleanup = json.loads(state_path.read_text(encoding="utf-8"))
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "timestamp": utc_timestamp(),
        "record_type": "c23_linear_resource_isolation_smoke",
        "run_id": sanitize_run_id(run_id),
        "logical_shards": logical_shards,
        "preflight": pre,
        "active_verification": active,
        "cleanup": cleanup,
        "checks": {
            "preflight": "PASS",
            "active_cluster": "PASS" if active and active["status"] == "PASS" else "FAIL",
            "cleanup": "PASS"
            if cleanup.get("phase") == "cleanup_complete"
            and cleanup.get("restored", {}).get("status") == "PASS"
            else "FAIL",
        },
    }
    payload["status"] = (
        "PASS" if all(value == "PASS" for value in payload["checks"].values()) else "FAIL"
    )
    write_json_atomic(output_path, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topology", default=str(DEFAULT_TOPOLOGY))
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan_parser = subparsers.add_parser("plan")
    plan_parser.add_argument("--logical-shards", type=int, required=True)

    preflight_parser = subparsers.add_parser("preflight")
    preflight_parser.add_argument("--output", required=True)

    smoke_parser = subparsers.add_parser("smoke")
    smoke_parser.add_argument("--run-id", required=True)
    smoke_parser.add_argument("--logical-shards", type=int, default=32)
    smoke_parser.add_argument("--output", required=True)

    cleanup_parser = subparsers.add_parser("cleanup")
    cleanup_parser.add_argument("--run-id", required=True)
    cleanup_parser.add_argument("--state", required=True)
    cleanup_parser.add_argument("--output", required=True)

    args = parser.parse_args()
    topology = load_topology(args.topology)
    if args.command == "plan":
        print(json.dumps(resource_contract(topology, args.logical_shards), indent=2))
        return 0
    if args.command == "preflight":
        payload = preflight(topology)
        write_json_atomic(args.output, payload)
        print(json.dumps({"status": payload["status"], "output": str(Path(args.output).resolve())}))
        return 0 if payload["status"] == "PASS" else 1
    if args.command == "smoke":
        payload = run_smoke(topology, args.run_id, args.logical_shards, args.output)
        print(json.dumps({"status": payload["status"], "output": str(Path(args.output).resolve())}))
        return 0 if payload["status"] == "PASS" else 1
    if args.command == "cleanup":
        state = json.loads(Path(args.state).read_text(encoding="utf-8"))
        removed = remove_experiment_containers(topology, args.run_id)
        restored = restore_existing_containers(topology, state["suspended"])
        payload = {
            "protocol_version": PROTOCOL_VERSION,
            "timestamp": utc_timestamp(),
            "removed": removed,
            "restored": restored,
            "status": restored["status"],
        }
        write_json_atomic(args.output, payload)
        return 0 if payload["status"] == "PASS" else 1
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
