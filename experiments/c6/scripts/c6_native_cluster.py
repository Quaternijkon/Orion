#!/usr/bin/env python3
"""Deploy and inspect a run-scoped native four-node Qdrant cluster for C6."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from c6_protocol import sha256_path, utc_timestamp, write_json_atomic  # noqa: E402


SSH_OPTIONS = [
    "-o",
    "BatchMode=yes",
    "-o",
    "StrictHostKeyChecking=no",
    "-o",
    "UserKnownHostsFile=/dev/null",
    "-o",
    "ConnectTimeout=10",
]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("deploy", "status", "stop"):
        subparser = subparsers.add_parser(name)
        subparser.add_argument("--topology", required=True)
        subparser.add_argument("--run-id", required=True)
        subparser.add_argument(
            "--runtime-root", default="/users/dry/orion-c6-runtime"
        )
        subparser.add_argument("--output")
        if name == "deploy":
            subparser.add_argument("--binary", required=True)
            subparser.add_argument("--config", required=True)
            subparser.add_argument("--readiness-timeout", type=float, default=120.0)
    return parser.parse_args(argv)


def validate_run_id(value: str) -> str:
    if not value or Path(value).name != value or not all(
        character.isalnum() or character in "-_" for character in value
    ):
        raise ValueError("run-id must contain only letters, digits, '-' and '_'")
    return value


def load_topology(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    nodes = [payload.get("controller"), *(payload.get("workers") or [])]
    if len(nodes) != 4 or not all(isinstance(node, dict) for node in nodes):
        raise ValueError("C6 native cluster requires exactly one controller and three workers")
    for node in nodes:
        for key in ("role", "ssh_host", "private_ip", "cpuset"):
            if not node.get(key):
                raise ValueError(f"topology node is missing {key}")
    ports = payload.get("ports") or {}
    if set(ports) < {"http", "grpc", "p2p"}:
        raise ValueError("topology is missing http/grpc/p2p ports")
    return payload


def nodes(topology: dict[str, Any]) -> list[dict[str, Any]]:
    return [topology["controller"], *topology["workers"]]


def is_local(node: dict[str, Any]) -> bool:
    return str(node["ssh_host"]) in {"localhost", "127.0.0.1"}


def run_shell(node: dict[str, Any], command: str, *, check: bool = True) -> subprocess.CompletedProcess[str]:
    argv = ["bash", "-lc", command] if is_local(node) else [
        "ssh",
        *SSH_OPTIONS,
        str(node["ssh_host"]),
        command,
    ]
    return subprocess.run(argv, text=True, capture_output=True, check=check)


def copy_file(node: dict[str, Any], source: Path, destination: Path) -> None:
    if is_local(node):
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".tmp")
        temporary.write_bytes(source.read_bytes())
        os.replace(temporary, destination)
        return
    parent = shlex.quote(str(destination.parent))
    run_shell(node, f"mkdir -p {parent}")
    temporary = f"{destination}.tmp"
    subprocess.run(
        ["scp", *SSH_OPTIONS, str(source), f"{node['ssh_host']}:{temporary}"],
        check=True,
        text=True,
        capture_output=True,
    )
    run_shell(
        node,
        f"mv {shlex.quote(temporary)} {shlex.quote(str(destination))}",
    )


def node_root(runtime_root: str | Path, run_id: str, node: dict[str, Any]) -> Path:
    return Path(runtime_root) / run_id / str(node["role"])


def environment(
    topology: dict[str, Any], node: dict[str, Any], root: Path
) -> dict[str, str]:
    return {
        "QDRANT__CLUSTER__ENABLED": "true",
        "QDRANT__CLUSTER__P2P__PORT": str(topology["ports"]["p2p"]),
        "QDRANT__SERVICE__HOST": "0.0.0.0",
        "QDRANT__SERVICE__HTTP_PORT": str(topology["ports"]["http"]),
        "QDRANT__SERVICE__GRPC_PORT": str(topology["ports"]["grpc"]),
        "QDRANT__SERVICE__HARDWARE_REPORTING": "true",
        "QDRANT__STORAGE__STORAGE_PATH": str(root / "storage"),
        "QDRANT__STORAGE__SNAPSHOTS_PATH": str(root / "snapshots"),
        "QDRANT__STORAGE__TEMP_PATH": str(root / "tmp"),
        "QDRANT__STORAGE__PERFORMANCE__MAX_SEARCH_THREADS": str(
            node.get("max_search_threads", 16)
        ),
        "QDRANT__STORAGE__PERFORMANCE__OPTIMIZER_CPU_BUDGET": str(
            node.get("optimizer_cpu_budget", 4)
        ),
        "QDRANT__LOG_LEVEL": "INFO",
    }


def start_command(
    topology: dict[str, Any],
    node: dict[str, Any],
    root: Path,
    binary: Path,
    config: Path,
) -> str:
    env = " ".join(
        f"{key}={shlex.quote(value)}"
        for key, value in environment(topology, node, root).items()
    )
    args = [
        str(binary),
        "--config-path",
        str(config),
        "--disable-telemetry",
    ]
    if str(node["role"]) != "controller":
        args.extend(
            [
                "--bootstrap",
                f"http://{topology['controller']['private_ip']}:{topology['ports']['p2p']}",
            ]
        )
    args.extend(
        ["--uri", f"http://{node['private_ip']}:{topology['ports']['p2p']}"]
    )
    executable = " ".join(shlex.quote(value) for value in args)
    return (
        f"{env} nohup taskset -c {shlex.quote(str(node['cpuset']))} {executable} "
        f">{shlex.quote(str(root / 'stdout.log'))} "
        f"2>{shlex.quote(str(root / 'stderr.log'))} </dev/null & echo $!"
    )


def systemd_unit_name(run_id: str, node: dict[str, Any]) -> str:
    raw = f"orion-c6-{run_id}-{node['role']}"
    return "".join(character if character.isalnum() or character == "-" else "-" for character in raw)


def start_local_systemd(
    topology: dict[str, Any],
    node: dict[str, Any],
    root: Path,
    binary: Path,
    config: Path,
    run_id: str,
) -> tuple[int, str]:
    unit = systemd_unit_name(run_id, node)
    argv = [
        "sudo",
        "-n",
        "systemd-run",
        f"--unit={unit}",
        f"--uid={os.getuid()}",
        f"--gid={os.getgid()}",
        "--collect",
        f"--property=CPUAffinity={node['cpuset']}",
        f"--property=StandardOutput=append:{root / 'stdout.log'}",
        f"--property=StandardError=append:{root / 'stderr.log'}",
    ]
    argv.extend(
        f"--setenv={key}={value}"
        for key, value in environment(topology, node, root).items()
    )
    argv.extend(
        [
            str(binary),
            "--config-path",
            str(config),
            "--disable-telemetry",
            "--uri",
            f"http://{node['private_ip']}:{topology['ports']['p2p']}",
        ]
    )
    subprocess.run(argv, check=True, text=True, capture_output=True)
    pid = int(
        subprocess.check_output(
            ["systemctl", "show", "-p", "MainPID", "--value", f"{unit}.service"],
            text=True,
        ).strip()
    )
    if pid <= 0:
        raise RuntimeError(f"systemd did not report a live PID for {unit}")
    return pid, unit


def http_json(url: str, timeout: float = 10.0) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read())


def wait_ready(topology: dict[str, Any], timeout: float) -> None:
    deadline = time.monotonic() + timeout
    pending = {str(node["private_ip"]) for node in nodes(topology)}
    while pending and time.monotonic() < deadline:
        for host in list(pending):
            try:
                with urllib.request.urlopen(
                    f"http://{host}:{topology['ports']['http']}/readyz", timeout=2
                ) as response:
                    if response.status == 200:
                        pending.remove(host)
            except Exception:
                pass
        if pending:
            time.sleep(1)
    if pending:
        raise TimeoutError(f"Qdrant nodes did not become ready: {sorted(pending)}")


def deploy(args: argparse.Namespace, topology: dict[str, Any]) -> dict[str, Any]:
    binary_source = Path(args.binary).expanduser().resolve()
    config_source = Path(args.config).expanduser().resolve()
    if not binary_source.is_file() or not os.access(binary_source, os.X_OK):
        raise ValueError("--binary must be an executable file")
    if not config_source.is_file():
        raise FileNotFoundError(config_source)
    binary_sha256 = sha256_path(binary_source)
    records = []
    for node in nodes(topology):
        root = node_root(args.runtime_root, args.run_id, node)
        exists = run_shell(node, f"test -e {shlex.quote(str(root))}", check=False)
        if exists.returncode == 0:
            raise FileExistsError(f"run-scoped node root already exists: {node['role']}:{root}")
        run_shell(
            node,
            "mkdir -p "
            + " ".join(
                shlex.quote(str(root / name))
                for name in ("storage", "snapshots", "tmp", "bin", "config")
            ),
        )
        binary = root / "bin" / "qdrant"
        config = root / "config" / "config.yaml"
        copy_file(node, binary_source, binary)
        copy_file(node, config_source, config)
        run_shell(node, f"chmod 0555 {shlex.quote(str(binary))}")
        remote_sha = run_shell(
            node, f"sha256sum {shlex.quote(str(binary))} | awk '{{print $1}}'"
        ).stdout.strip()
        if remote_sha != binary_sha256:
            raise RuntimeError(f"deployed binary checksum mismatch on {node['role']}")
        if is_local(node):
            pid, systemd_unit = start_local_systemd(
                topology, node, root, binary, config, args.run_id
            )
        else:
            pid_text = run_shell(
                node, start_command(topology, node, root, binary, config)
            ).stdout.strip()
            pid = int(pid_text.splitlines()[-1])
            systemd_unit = None
        run_shell(node, f"printf '%s\n' {pid} > {shlex.quote(str(root / 'qdrant.pid'))}")
        records.append(
            {
                "role": node["role"],
                "private_ip": node["private_ip"],
                "root": str(root),
                "pid": pid,
                "systemd_unit": systemd_unit,
                "binary_sha256": remote_sha,
            }
        )
        if str(node["role"]) == "controller":
            time.sleep(2)
    wait_ready(topology, args.readiness_timeout)
    cluster = http_json(
        f"http://{topology['controller']['private_ip']}:{topology['ports']['http']}/cluster"
    )
    peers = ((cluster.get("result") or {}).get("peers") or {})
    if len(peers) != 4:
        raise RuntimeError(f"cluster peer count differs: expected=4, actual={len(peers)}")
    return {
        "record_type": "c6_native_cluster_deploy",
        "timestamp": utc_timestamp(),
        "run_id": args.run_id,
        "binary": str(binary_source),
        "binary_sha256": binary_sha256,
        "config_sha256": sha256_path(config_source),
        "nodes": records,
        "cluster": cluster,
    }


def status(args: argparse.Namespace, topology: dict[str, Any]) -> dict[str, Any]:
    node_records = []
    for node in nodes(topology):
        root = node_root(args.runtime_root, args.run_id, node)
        if is_local(node):
            process = subprocess.run(
                [
                    "systemctl",
                    "is-active",
                    "--quiet",
                    f"{systemd_unit_name(args.run_id, node)}.service",
                ],
                check=False,
            )
        else:
            process = run_shell(
                node,
                f"test -f {shlex.quote(str(root / 'qdrant.pid'))} && "
                f"pid=$(cat {shlex.quote(str(root / 'qdrant.pid'))}) && kill -0 \"$pid\"",
                check=False,
            )
        node_records.append(
            {"role": node["role"], "root": str(root), "process_live": process.returncode == 0}
        )
    cluster = None
    try:
        cluster = http_json(
            f"http://{topology['controller']['private_ip']}:{topology['ports']['http']}/cluster"
        )
    except Exception:
        pass
    return {
        "record_type": "c6_native_cluster_status",
        "timestamp": utc_timestamp(),
        "run_id": args.run_id,
        "nodes": node_records,
        "cluster": cluster,
    }


def stop(args: argparse.Namespace, topology: dict[str, Any]) -> dict[str, Any]:
    stopped = []
    for node in reversed(nodes(topology)):
        root = node_root(args.runtime_root, args.run_id, node)
        if is_local(node):
            unit = f"{systemd_unit_name(args.run_id, node)}.service"
            result = subprocess.run(
                ["sudo", "-n", "systemctl", "stop", unit],
                text=True,
                capture_output=True,
                check=False,
            )
            if result.returncode != 0:
                raise RuntimeError(f"failed to stop {node['role']}: {result.stderr}")
            stopped.append(str(node["role"]))
            continue
        command = (
            f"pid_file={shlex.quote(str(root / 'qdrant.pid'))}; "
            "test -f \"$pid_file\" || exit 0; pid=$(cat \"$pid_file\"); "
            "cmd=$(tr '\\0' ' ' </proc/$pid/cmdline 2>/dev/null || true); "
            f"case \"$cmd\" in *{shlex.quote(str(root / 'bin' / 'qdrant'))}*) kill -TERM \"$pid\" ;; "
            "*) echo 'PID identity mismatch' >&2; exit 3 ;; esac"
        )
        result = run_shell(node, command, check=False)
        if result.returncode not in {0}:
            raise RuntimeError(f"failed to stop {node['role']}: {result.stderr}")
        stopped.append(str(node["role"]))
    return {
        "record_type": "c6_native_cluster_stop",
        "timestamp": utc_timestamp(),
        "run_id": args.run_id,
        "stopped": stopped,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.run_id = validate_run_id(args.run_id)
    topology = load_topology(args.topology)
    if args.command == "deploy":
        record = deploy(args, topology)
    elif args.command == "status":
        record = status(args, topology)
    else:
        record = stop(args, topology)
    if args.output:
        write_json_atomic(args.output, record)
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
