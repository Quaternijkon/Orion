#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_TOPOLOGY = Path(__file__).with_name("distributed") / "cloudlab_orion_4node.json"
RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$")
RESOURCE_PREFIX = "orion-dist-"
EXPECTED_COMMIT = "1a5ac4c47237b9224ae3e4ca28c2cefb2b514352"
PEER_PREMERGE_DISABLE_ENV = "QDRANT_DISABLE_SHARD_MAJOR_PEER_PREMERGE"
PEER_PREMERGE_MODE_LABEL = "orion.distributed.peer_premerge"
CONTROLLER_FINGERPRINT_LABEL = "orion.distributed.controller_fingerprint"
NOFILE_LABEL = "orion.distributed.nofile"
CONTAINER_NOFILE_SOFT = 65536
CONTAINER_NOFILE_HARD = 65536


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and operate the four-host Orion/Qdrant CloudLab cluster."
    )
    parser.add_argument(
        "--topology",
        default=str(DEFAULT_TOPOLOGY),
        help="Four-node topology JSON.",
    )
    parser.add_argument("--run-id", required=True, help="Experiment-scoped resource id.")
    parser.add_argument("--repo", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--ssh-user", default=None)
    parser.add_argument("--ssh-option", action="append", default=[])
    parser.add_argument("--image-tag", default=None)
    parser.add_argument("--expected-commit", default=EXPECTED_COMMIT)
    parser.add_argument("--dry-run", action="store_true")

    subparsers = parser.add_subparsers(dest="command", required=True)
    bootstrap = subparsers.add_parser("bootstrap", help="Install idempotent node dependencies.")
    bootstrap.add_argument("--skip-dataset", action="store_true")
    bootstrap.add_argument("--skip-python-venv", action="store_true")

    build = subparsers.add_parser("build", help="Build and export the release image.")
    build.add_argument("--allow-dirty", action="store_true")
    build.add_argument("--force", action="store_true")

    deploy = subparsers.add_parser("deploy", help="Load and start all four cluster nodes.")
    deploy.add_argument("--wait-timeout", type=float, default=180.0)
    deploy.add_argument(
        "--disable-peer-premerge",
        action="store_true",
        help="Disable native shard-major peer pre-merge on the controller for an A/B run.",
    )

    status = subparsers.add_parser(
        "status", help="Inspect containers, endpoints, peers, and placement."
    )
    status.add_argument(
        "--disable-peer-premerge",
        action="store_true",
        help="Validate/report status against a controller with peer pre-merge disabled.",
    )
    subparsers.add_parser("down", help="Stop this run's containers and preserve storage.")
    clean = subparsers.add_parser("clean", help="Delete this run's containers and local storage.")
    clean.add_argument(
        "--yes-delete-storage",
        action="store_true",
        help="Required acknowledgement for deleting run-scoped storage.",
    )
    manifest = subparsers.add_parser("manifest", help="Write and print the deployment manifest.")
    manifest.add_argument(
        "--disable-peer-premerge",
        action="store_true",
        help="Record/validate a controller with peer pre-merge disabled.",
    )
    return parser.parse_args()


def load_topology(path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_topology(data)
    return data


def validate_topology(topology: dict[str, Any]) -> None:
    controller = topology.get("controller")
    workers = topology.get("workers")
    if not isinstance(controller, dict) or controller.get("role") != "controller":
        raise ValueError("topology requires exactly one controller with role=controller")
    if not isinstance(workers, list) or len(workers) != 3:
        raise ValueError("topology requires exactly three workers")
    nodes = [controller, *workers]
    roles = [str(node.get("role") or "") for node in nodes]
    ips = [str(node.get("private_ip") or "") for node in nodes]
    ssh_hosts = [str(node.get("ssh_host") or "") for node in nodes]
    if len(set(roles)) != 4 or any(not role for role in roles):
        raise ValueError("controller and worker roles must be non-empty and unique")
    if len(set(ips)) != 4 or any(not ip.startswith("10.10.1.") for ip in ips):
        raise ValueError("all four qdrant-lan private IPs must be unique in 10.10.1.0/24")
    if len(set(ssh_hosts)) != 4:
        raise ValueError("all four ssh_host entries must be unique")
    if any(not node.get("cpuset") for node in nodes):
        raise ValueError("every node requires an explicit cpuset")
    if topology.get("benchmark_client_cpuset") == controller.get("cpuset"):
        raise ValueError("benchmark client and controller cpusets must differ")


def validate_run_id(run_id: str) -> str:
    if not RUN_ID_RE.fullmatch(run_id):
        raise ValueError(
            "run-id must start with an alphanumeric character and contain only "
            "letters, digits, dot, underscore, or dash"
        )
    return run_id


def safe_run_token(run_id: str) -> str:
    return re.sub(r"[^a-z0-9_.-]+", "-", validate_run_id(run_id).lower())


def all_nodes(topology: dict[str, Any]) -> list[dict[str, Any]]:
    return [topology["controller"], *topology["workers"]]


def controller_uri(topology: dict[str, Any]) -> str:
    return f"http://{topology['controller']['private_ip']}:{topology['ports']['p2p']}"


def advertised_uri(node: dict[str, Any], topology: dict[str, Any]) -> str:
    return f"http://{node['private_ip']}:{topology['ports']['p2p']}"


def expected_peer_premerge_mode(
    node: dict[str, Any], disable_peer_premerge: bool = False
) -> str:
    if str(node.get("role")) != "controller":
        return "not_applicable"
    return "disabled" if disable_peer_premerge else "enabled"


def controller_config_fingerprint(
    node: dict[str, Any],
    run_id: str,
    image_id: str,
    disable_peer_premerge: bool = False,
) -> str:
    """Fingerprint controller settings that must not drift across an A/B run."""
    if str(node.get("role")) != "controller":
        raise ValueError("controller fingerprint requested for a non-controller node")
    payload = {
        "schema_version": 2,
        "run_id": validate_run_id(run_id),
        "role": str(node["role"]),
        "private_ip": str(node["private_ip"]),
        "cpuset": str(node["cpuset"]),
        "image_id": str(image_id),
        "peer_premerge": expected_peer_premerge_mode(node, disable_peer_premerge),
        "nofile": {
            "soft": CONTAINER_NOFILE_SOFT,
            "hard": CONTAINER_NOFILE_HARD,
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def expected_nofile_label() -> str:
    return f"{CONTAINER_NOFILE_SOFT}:{CONTAINER_NOFILE_HARD}"


def inspected_nofile_limits(inspected: dict[str, Any] | None) -> dict[str, int] | None:
    if not inspected:
        return None
    for item in (inspected.get("HostConfig") or {}).get("Ulimits") or []:
        if str((item or {}).get("Name") or "").lower() != "nofile":
            continue
        return {
            "soft": int((item or {}).get("Soft") or 0),
            "hard": int((item or {}).get("Hard") or 0),
        }
    return None


def inspected_peer_premerge_mode(
    inspected: dict[str, Any] | None, node: dict[str, Any]
) -> str | None:
    if str(node.get("role")) != "controller":
        return "not_applicable"
    if not inspected:
        return None
    config = inspected.get("Config") or {}
    labels = config.get("Labels") or {}
    label_mode = labels.get(PEER_PREMERGE_MODE_LABEL)
    disable_value: str | None = None
    for item in config.get("Env") or []:
        key, separator, value = str(item).partition("=")
        if separator and key == PEER_PREMERGE_DISABLE_ENV:
            disable_value = value
            break
    env_mode = (
        "disabled"
        if str(disable_value or "").strip().lower() in {"1", "true", "yes", "on"}
        else "enabled"
    )
    if label_mode in {"enabled", "disabled"} and label_mode != env_mode:
        return "inconsistent"
    return str(label_mode) if label_mode in {"enabled", "disabled"} else env_mode


def peer_premerge_summary(
    disable_peer_premerge: bool,
    current_mode: str | None = None,
) -> dict[str, Any]:
    requested_mode = "disabled" if disable_peer_premerge else "enabled"
    return {
        "scope": "controller",
        "requested_mode": requested_mode,
        "current_mode": current_mode,
        "matches_requested": (
            current_mode == requested_mode if current_mode is not None else None
        ),
        "disable_environment_variable": PEER_PREMERGE_DISABLE_ENV,
    }


def normalize_peer_uri(uri: str) -> str:
    """Normalize the harmless trailing slash Qdrant adds to advertised URIs."""
    return str(uri).rstrip("/")


def http_url(node: dict[str, Any], topology: dict[str, Any]) -> str:
    return f"http://{node['private_ip']}:{topology['ports']['http']}"


def image_tag_for_commit(commit: str) -> str:
    return f"orion-method4:{commit[:12]}"


def container_name(run_id: str, role: str) -> str:
    role_token = re.sub(r"[^a-z0-9_.-]+", "-", role.lower())
    return f"{RESOURCE_PREFIX}{safe_run_token(run_id)}-{role_token}"


def shared_run_dir(topology: dict[str, Any], run_id: str) -> Path:
    return Path(topology["shared_root"]) / validate_run_id(run_id)


def local_role_root(topology: dict[str, Any], run_id: str, role: str) -> Path:
    base = Path(topology["local_storage_root"]).resolve()
    candidate = (base / validate_run_id(run_id) / role).resolve()
    if base not in candidate.parents:
        raise ValueError(f"unsafe storage path outside {base}: {candidate}")
    return candidate


def manifest_path(topology: dict[str, Any], run_id: str) -> Path:
    return shared_run_dir(topology, run_id) / "manifest.json"


def shell_join(command: list[str]) -> str:
    return shlex.join([str(value) for value in command])


def ssh_target(node: dict[str, Any], ssh_user: str | None) -> str:
    host = str(node["ssh_host"])
    if host in {"localhost", "127.0.0.1", socket.gethostname(), socket.getfqdn()}:
        return "localhost"
    return f"{ssh_user}@{host}" if ssh_user else host


def wrap_node_command(
    node: dict[str, Any],
    command: list[str] | str,
    ssh_user: str | None = None,
    ssh_options: list[str] | None = None,
) -> list[str]:
    command_text = command if isinstance(command, str) else shell_join(command)
    target = ssh_target(node, ssh_user)
    if target == "localhost":
        return ["bash", "-lc", command_text]
    wrapped = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]
    for option in ssh_options or []:
        wrapped.extend(["-o", option])
    wrapped.extend([target, "bash", "-lc", shlex.quote(command_text)])
    return wrapped


def run_command(
    command: list[str],
    *,
    dry_run: bool = False,
    capture: bool = False,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    print(f"[cmd] {shell_join(command)}", flush=True)
    if dry_run:
        return subprocess.CompletedProcess(command, 0, "", "")
    return subprocess.run(
        command,
        check=check,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
    )


def run_on_node(
    node: dict[str, Any],
    command: list[str] | str,
    args: argparse.Namespace,
    *,
    capture: bool = False,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    return run_command(
        wrap_node_command(node, command, args.ssh_user, args.ssh_option),
        dry_run=args.dry_run,
        capture=capture,
        check=check,
    )


def bootstrap_command() -> str:
    return """
set -euo pipefail
if ! command -v docker >/dev/null 2>&1; then
  sudo -n apt-get update
  sudo -n env DEBIAN_FRONTEND=noninteractive apt-get install -y docker.io
fi
sudo -n docker version >/dev/null
if ! sudo -n docker buildx version >/dev/null 2>&1; then
  sudo -n apt-get update
  sudo -n env DEBIAN_FRONTEND=noninteractive apt-get install -y docker-buildx
fi
if ! command -v python3 >/dev/null 2>&1; then
  sudo -n apt-get update
  sudo -n env DEBIAN_FRONTEND=noninteractive apt-get install -y python3
fi
if ! dpkg-query -W -f='${Status}' python3-venv 2>/dev/null | grep -q 'install ok installed'; then
  sudo -n apt-get update
  sudo -n env DEBIAN_FRONTEND=noninteractive apt-get install -y python3-venv
fi
""".strip()


def docker_run_command(
    topology: dict[str, Any],
    node: dict[str, Any],
    run_id: str,
    image_tag: str,
    image_id: str,
    disable_peer_premerge: bool = False,
) -> list[str]:
    role = str(node["role"])
    name = container_name(run_id, role)
    storage = local_role_root(topology, run_id, role) / "storage"
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
        str(node["cpuset"]),
        "--ulimit",
        f"nofile={expected_nofile_label()}",
        "--restart",
        "unless-stopped",
        "--label",
        f"orion.distributed.run_id={run_id}",
        "--label",
        f"orion.distributed.role={role}",
        "--label",
        f"orion.distributed.private_ip={node['private_ip']}",
        "--label",
        f"orion.distributed.image_id={image_id}",
        "--label",
        f"{NOFILE_LABEL}={expected_nofile_label()}",
    ]
    if role == "controller":
        command.extend(
            [
                "--label",
                f"{PEER_PREMERGE_MODE_LABEL}="
                f"{expected_peer_premerge_mode(node, disable_peer_premerge)}",
                "--label",
                f"{CONTROLLER_FINGERPRINT_LABEL}="
                f"{controller_config_fingerprint(node, run_id, image_id, disable_peer_premerge)}",
            ]
        )
    command.extend(
        [
            "-e",
            "QDRANT__CLUSTER__ENABLED=true",
            "-e",
            f"QDRANT__CLUSTER__P2P__PORT={topology['ports']['p2p']}",
            "-e",
            f"QDRANT__SERVICE__HTTP_PORT={topology['ports']['http']}",
            "-e",
            f"QDRANT__SERVICE__GRPC_PORT={topology['ports']['grpc']}",
            "-e",
            f"QDRANT__STORAGE__PERFORMANCE__MAX_SEARCH_THREADS={node['max_search_threads']}",
            "-e",
            f"QDRANT__STORAGE__PERFORMANCE__OPTIMIZER_CPU_BUDGET={node['optimizer_cpu_budget']}",
        ]
    )
    if role == "controller" and disable_peer_premerge:
        command.extend(["-e", f"{PEER_PREMERGE_DISABLE_ENV}=1"])
    command.extend(
        [
            "-v",
            f"{storage}:/qdrant/storage",
            image_tag,
            "./qdrant",
        ]
    )
    if role != "controller":
        command.extend(["--bootstrap", controller_uri(topology)])
    command.extend(["--uri", advertised_uri(node, topology)])
    return command


def clean_commands(
    topology: dict[str, Any],
    node: dict[str, Any],
    run_id: str,
) -> list[list[str]]:
    role = str(node["role"])
    name = container_name(run_id, role)
    role_root = local_role_root(topology, run_id, role)
    expected = Path(topology["local_storage_root"]).resolve() / validate_run_id(run_id) / role
    if role_root != expected.resolve():
        raise ValueError(f"refusing unsafe clean path: {role_root}")
    return [
        ["sudo", "-n", "docker", "rm", "-f", name],
        ["rm", "-rf", "--", str(role_root)],
    ]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_manifest_image_archive(
    tar_path: Path,
    image_tag: str,
    manifest: dict[str, Any],
) -> dict[str, str]:
    """Verify that a manifest still describes the exact image archive on disk."""
    image = manifest.get("image") or {}
    expected_tag = str(image.get("tag") or "")
    if expected_tag != image_tag:
        raise RuntimeError(
            "image tar manifest tag mismatch: "
            f"manifest={expected_tag or '<missing>'}, requested={image_tag}; "
            "rerun build --force"
        )

    manifest_tar_path = str(image.get("tar_path") or "")
    if not manifest_tar_path:
        raise RuntimeError(
            "existing image tar has no manifest tar_path; rerun build --force"
        )
    if Path(manifest_tar_path).resolve() != tar_path.resolve():
        raise RuntimeError(
            "image tar path mismatch: "
            f"manifest={manifest_tar_path}, actual={tar_path}; rerun build --force"
        )

    expected_sha256 = str(image.get("tar_sha256") or "")
    if not expected_sha256:
        raise RuntimeError(
            "existing image tar has no manifest SHA-256; rerun build --force"
        )
    actual_sha256 = sha256_file(tar_path)
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            "image tar SHA-256 mismatch: "
            f"manifest={expected_sha256}, actual={actual_sha256}; rerun build --force"
        )

    expected_image_id = str(image.get("id") or "")
    if not expected_image_id:
        raise RuntimeError(
            "existing image tar manifest has no image id; rerun build --force"
        )
    return {
        "image_id": expected_image_id,
        "tar_sha256": actual_sha256,
    }


def validate_reusable_image_archive(
    tar_path: Path,
    image_tag: str,
    manifest: dict[str, Any],
    current_image_id: str,
) -> dict[str, str]:
    """Close the tar -> manifest -> local tag identity loop before reuse."""
    verified = validate_manifest_image_archive(tar_path, image_tag, manifest)
    if current_image_id != verified["image_id"]:
        raise RuntimeError(
            "existing image tar manifest id does not match current tag id: "
            f"manifest={verified['image_id']}, current={current_image_id or '<missing>'}; "
            "rerun build --force"
        )
    return verified


def git_state(repo: Path) -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True, text=True, capture_output=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=repo, check=True, text=True, capture_output=True
    ).stdout
    dirty_paths = [line[3:].strip() for line in status.splitlines() if len(line) >= 4]
    return {
        "commit": commit,
        "short_commit": commit[:12],
        "dirty": bool(dirty_paths),
        "dirty_paths": dirty_paths,
    }


def read_manifest(topology: dict[str, Any], run_id: str) -> dict[str, Any]:
    path = manifest_path(topology, run_id)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_manifest(topology: dict[str, Any], run_id: str, data: dict[str, Any]) -> Path:
    path = manifest_path(topology, run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    temp.replace(path)
    return path


def node_facts(node: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    script = """
set -euo pipefail
python3 - <<'PY'
import json, os, platform
data = {
  "hostname": platform.node(),
  "kernel": platform.release(),
  "machine": platform.machine(),
  "cpu_count": os.cpu_count(),
}
try:
  data["memory_bytes"] = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
except (ValueError, OSError, AttributeError):
  data["memory_bytes"] = None
print(json.dumps(data))
PY
""".strip()
    result = run_on_node(node, script, args, capture=True)
    return json.loads(result.stdout.strip() or "{}") if not args.dry_run else {}


def dataset_manifest(topology: dict[str, Any]) -> dict[str, Any] | None:
    dataset = topology.get("dataset") or {}
    path = Path(str(dataset.get("path") or ""))
    if not path.is_file():
        return None
    validator = Path(topology["local_storage_root"]) / "venv/bin/python"
    validation_script = (
        "import h5py,json,sys; "
        "f=h5py.File(sys.argv[1],'r'); "
        "print(json.dumps({k:list(f[k].shape) for k in ('train','test','neighbors') if k in f})); "
        "f.close()"
    )
    try:
        if validator.is_file():
            result = subprocess.run(
                [str(validator), "-c", validation_script, str(path)],
                check=True,
                text=True,
                capture_output=True,
            )
            shape = json.loads(result.stdout)
        else:
            import h5py

            with h5py.File(path, "r") as handle:
                shape = {
                    key: list(handle[key].shape)
                    for key in ("train", "test", "neighbors")
                    if key in handle
                }
    except Exception as exc:  # recorded as evidence; bootstrap reports hard failures
        shape = {"validation_error": str(exc)}
    return {
        "url": dataset.get("url"),
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "hdf5_shapes": shape,
    }


def build_manifest_data(
    topology: dict[str, Any],
    run_id: str,
    repo: Path,
    image_tag: str,
    image_id: str | None,
    nodes: list[dict[str, Any]] | None = None,
    disable_peer_premerge: bool = False,
    current_peer_premerge_mode: str | None = None,
) -> dict[str, Any]:
    state = git_state(repo)
    return {
        "schema_version": 1,
        "run_id": run_id,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "repository": {"path": str(repo), **state},
        "image": {"tag": image_tag, "id": image_id},
        "topology": topology,
        "controller_url": http_url(topology["controller"], topology),
        "controller_uri": controller_uri(topology),
        "worker_uris": [advertised_uri(node, topology) for node in topology["workers"]],
        "peer_premerge": peer_premerge_summary(
            disable_peer_premerge, current_peer_premerge_mode
        ),
        "container_ulimits": {
            "nofile": {
                "soft": CONTAINER_NOFILE_SOFT,
                "hard": CONTAINER_NOFILE_HARD,
            }
        },
        "dataset": dataset_manifest(topology),
        "nodes": nodes or [],
    }


def image_id_local(image_tag: str, dry_run: bool = False) -> str:
    result = run_command(
        ["sudo", "-n", "docker", "image", "inspect", image_tag, "--format", "{{.Id}}"],
        dry_run=dry_run,
        capture=True,
    )
    return result.stdout.strip() if not dry_run else "dry-run-image-id"


def command_bootstrap(args: argparse.Namespace, topology: dict[str, Any]) -> int:
    for node in all_nodes(topology):
        run_on_node(node, bootstrap_command(), args)
    controller = topology["controller"]
    if not args.skip_python_venv:
        venv = Path(topology["local_storage_root"]) / "venv"
        command = (
            "if ! dpkg-query -W -f='${Status}' python3-dev 2>/dev/null | grep -q 'install ok installed'; "
            "then sudo -n apt-get update && sudo -n env DEBIAN_FRONTEND=noninteractive "
            "apt-get install -y python3-dev build-essential; fi; "
            f"if [ ! -x {shlex.quote(str(venv / 'bin/python'))} ]; then "
            f"python3 -m venv {shlex.quote(str(venv))}; fi; "
            f"if ! {shlex.quote(str(venv / 'bin/python'))} -c "
            "'import h5py, numpy, hnswlib, matplotlib, pytest' >/dev/null 2>&1; then "
            f"{shlex.quote(str(venv / 'bin/pip'))} install --upgrade pip && "
            f"{shlex.quote(str(venv / 'bin/pip'))} install h5py numpy hnswlib matplotlib pytest; fi"
        )
        run_on_node(controller, command, args)
    if not args.skip_dataset:
        dataset = topology["dataset"]
        path = Path(dataset["path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            run_command(
                ["curl", "-fL", "--retry", "5", "--output", str(path) + ".part", dataset["url"]],
                dry_run=args.dry_run,
            )
            if not args.dry_run:
                Path(str(path) + ".part").replace(path)
        if not args.dry_run:
            manifest = dataset_manifest(topology)
            if not manifest or "validation_error" in manifest.get("hdf5_shapes", {}):
                raise RuntimeError(f"dataset failed HDF5 validation: {manifest}")
            print(json.dumps(manifest, indent=2), flush=True)
    return 0


def command_build(args: argparse.Namespace, topology: dict[str, Any]) -> int:
    repo = Path(args.repo).resolve()
    state = git_state(repo)
    if state["commit"] != args.expected_commit:
        raise RuntimeError(
            f"expected commit {args.expected_commit}, found {state['commit']}; refusing mixed-image experiment"
        )
    binary_affecting_dirty = [
        path
        for path in state.get("dirty_paths", [])
        if not path.startswith(("tools/", "tests/", "docs/", "results/"))
    ]
    if binary_affecting_dirty and not args.allow_dirty:
        raise RuntimeError(
            "working tree has Qdrant-image-affecting changes; use --allow-dirty only for an "
            f"explicitly non-canonical build: {binary_affecting_dirty}"
        )
    image_tag = args.image_tag or image_tag_for_commit(state["commit"])
    run_dir = shared_run_dir(topology, args.run_id)
    tar_path = run_dir / f"{image_tag.replace(':', '_')}.tar"
    run_dir.mkdir(parents=True, exist_ok=True)
    stored = read_manifest(topology, args.run_id)
    if tar_path.is_file() and not args.force:
        if args.dry_run:
            print(
                f"[dry-run] would validate and reuse image archive {tar_path}",
                flush=True,
            )
            return 0
        try:
            current_image_id = image_id_local(image_tag)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"existing image tar cannot be matched to local tag {image_tag}; "
                "rerun build --force"
            ) from exc
        verified = validate_reusable_image_archive(
            tar_path,
            image_tag,
            stored,
            current_image_id,
        )
        run_command(["sudo", "-n", "chmod", "0644", str(tar_path)])
        print(
            "Reusing verified image archive "
            f"{tar_path} ({verified['image_id']}, sha256={verified['tar_sha256']})",
            flush=True,
        )
        return 0
    if args.force or not tar_path.exists():
        run_command(
            [
                "sudo",
                "-n",
                "docker",
                "buildx",
                "build",
                "--load",
                "--build-arg",
                f"GIT_COMMIT_ID={state['commit']}",
                "--tag",
                image_tag,
                ".",
            ],
            dry_run=args.dry_run,
        )
        run_command(
            ["sudo", "-n", "docker", "run", "--rm", image_tag, "./qdrant", "--version"],
            dry_run=args.dry_run,
        )
        run_command(
            ["sudo", "-n", "docker", "save", "--output", str(tar_path), image_tag],
            dry_run=args.dry_run,
        )
    if tar_path.exists() or args.dry_run:
        run_command(
            ["sudo", "-n", "chmod", "0644", str(tar_path)],
            dry_run=args.dry_run,
        )
    image_id = image_id_local(image_tag, args.dry_run)
    data = build_manifest_data(topology, args.run_id, repo, image_tag, image_id)
    data["image"]["tar_path"] = str(tar_path)
    if tar_path.exists():
        data["image"]["tar_size_bytes"] = tar_path.stat().st_size
        data["image"]["tar_sha256"] = sha256_file(tar_path)
    if not args.dry_run:
        path = write_manifest(topology, args.run_id, data)
        print(f"Wrote {path}")
    return 0


def inspect_container(node: dict[str, Any], name: str, args: argparse.Namespace) -> dict[str, Any] | None:
    result = run_on_node(
        node,
        ["sudo", "-n", "docker", "inspect", name],
        args,
        capture=True,
        check=False,
    )
    if result.returncode != 0 or args.dry_run:
        return None
    payload = json.loads(result.stdout)
    return payload[0] if payload else None


def verify_container_reusable(
    inspected: dict[str, Any],
    node: dict[str, Any],
    run_id: str,
    image_id: str,
    disable_peer_premerge: bool = False,
) -> bool:
    labels = (inspected.get("Config") or {}).get("Labels") or {}
    expected = {
        "orion.distributed.run_id": run_id,
        "orion.distributed.role": str(node["role"]),
        "orion.distributed.private_ip": str(node["private_ip"]),
        "orion.distributed.image_id": image_id,
        NOFILE_LABEL: expected_nofile_label(),
    }
    mismatches = {key: (labels.get(key), value) for key, value in expected.items() if labels.get(key) != value}
    host_config = inspected.get("HostConfig") or {}
    if str(host_config.get("CpusetCpus") or "") != str(node["cpuset"]):
        mismatches["cpuset"] = (host_config.get("CpusetCpus"), node["cpuset"])
    expected_nofile = {
        "soft": CONTAINER_NOFILE_SOFT,
        "hard": CONTAINER_NOFILE_HARD,
    }
    actual_nofile = inspected_nofile_limits(inspected)
    if actual_nofile != expected_nofile:
        mismatches["nofile"] = (actual_nofile, expected_nofile)
    if str(inspected.get("Image") or "") != image_id:
        mismatches["image"] = (inspected.get("Image"), image_id)
    if str(node.get("role")) == "controller":
        expected_mode = expected_peer_premerge_mode(node, disable_peer_premerge)
        expected_fingerprint = controller_config_fingerprint(
            node, run_id, image_id, disable_peer_premerge
        )
        if labels.get(PEER_PREMERGE_MODE_LABEL) != expected_mode:
            mismatches[PEER_PREMERGE_MODE_LABEL] = (
                labels.get(PEER_PREMERGE_MODE_LABEL),
                expected_mode,
            )
        if labels.get(CONTROLLER_FINGERPRINT_LABEL) != expected_fingerprint:
            mismatches[CONTROLLER_FINGERPRINT_LABEL] = (
                labels.get(CONTROLLER_FINGERPRINT_LABEL),
                expected_fingerprint,
            )
        env = (inspected.get("Config") or {}).get("Env") or []
        disable_values = [
            str(item).partition("=")[2]
            for item in env
            if str(item).partition("=")[:2] == (PEER_PREMERGE_DISABLE_ENV, "=")
        ]
        expected_disable_values = ["1"] if disable_peer_premerge else []
        if disable_values != expected_disable_values:
            mismatches[PEER_PREMERGE_DISABLE_ENV] = (
                disable_values,
                expected_disable_values,
            )
    if mismatches:
        raise RuntimeError(
            f"container {inspected.get('Name')} exists with incompatible configuration: {mismatches}"
        )
    return bool((inspected.get("State") or {}).get("Running"))


def wait_http_ready(url: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{url.rstrip('/')}/readyz", timeout=3.0) as response:
                if 200 <= response.status < 300:
                    return
        except (OSError, urllib.error.URLError) as exc:
            last_error = exc
        time.sleep(1.0)
    raise TimeoutError(f"timed out waiting for {url}/readyz: {last_error}")


def cluster_snapshot(topology: dict[str, Any]) -> dict[str, Any]:
    url = f"{http_url(topology['controller'], topology)}/cluster"
    with urllib.request.urlopen(url, timeout=10.0) as response:
        return json.loads(response.read().decode("utf-8"))


def http_json(url: str) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=10.0) as response:
        return json.loads(response.read().decode("utf-8"))


def run_collection_placements(topology: dict[str, Any], run_id: str) -> dict[str, Any]:
    base = http_url(topology["controller"], topology).rstrip("/")
    payload = http_json(f"{base}/collections")
    prefix = f"dist_{run_id}_"
    names = [
        str(item.get("name"))
        for item in (payload.get("result") or {}).get("collections") or []
        if str(item.get("name") or "").startswith(prefix)
    ]
    placements: dict[str, Any] = {}
    for name in names:
        quoted = urllib.parse.quote(name, safe="")
        placements[name] = {
            "info": http_json(f"{base}/collections/{quoted}").get("result"),
            "cluster": http_json(f"{base}/collections/{quoted}/cluster").get("result"),
        }
    return placements


def cluster_validation_errors(
    topology: dict[str, Any], snapshot: dict[str, Any]
) -> list[str]:
    errors: list[str] = []
    if not isinstance(snapshot, dict):
        return ["cluster endpoint returned a non-object payload"]
    result = snapshot.get("result") or {}
    if not isinstance(result, dict):
        return ["cluster result payload is not an object"]
    peers = result.get("peers") or {}
    if not isinstance(peers, dict):
        return ["cluster peers payload is not an object"]
    if len(peers) != 4:
        errors.append(f"cluster peer count is {len(peers)}, expected 4")
    actual_uris = {
        normalize_peer_uri(str((value or {}).get("uri") or ""))
        for value in peers.values()
    }
    expected_uris = {
        normalize_peer_uri(advertised_uri(node, topology))
        for node in all_nodes(topology)
    }
    if actual_uris != expected_uris:
        errors.append(
            "cluster peer URI mismatch: "
            f"expected {sorted(expected_uris)}, got {sorted(actual_uris)}"
        )
    controller_uri_value = normalize_peer_uri(
        advertised_uri(topology["controller"], topology)
    )
    controller_peer_ids = {
        str(peer_id)
        for peer_id, value in peers.items()
        if normalize_peer_uri(str((value or {}).get("uri") or ""))
        == controller_uri_value
    }
    reported_peer_id = str(result.get("peer_id") or "")
    if controller_peer_ids and reported_peer_id not in controller_peer_ids:
        errors.append(
            "controller endpoint peer id mismatch: "
            f"URI {controller_uri_value} belongs to {sorted(controller_peer_ids)}, "
            f"endpoint reports {reported_peer_id or '<missing>'}"
        )
    return errors


def collection_validation_errors(
    topology: dict[str, Any],
    snapshot: dict[str, Any],
    collections: dict[str, Any],
) -> list[str]:
    """Validate existing run-scoped collections from the controller's view."""
    errors: list[str] = []
    if not isinstance(snapshot, dict):
        return ["cannot validate collections without an object cluster snapshot"]
    if not isinstance(collections, dict):
        return ["run-scoped collections payload is not an object"]
    peers = ((snapshot.get("result") or {}).get("peers") or {})
    peer_ids_by_uri = {
        normalize_peer_uri(str((value or {}).get("uri") or "")): str(peer_id)
        for peer_id, value in peers.items()
    }
    controller_peer_id = peer_ids_by_uri.get(
        normalize_peer_uri(advertised_uri(topology["controller"], topology))
    )
    worker_peer_ids = {
        peer_ids_by_uri.get(normalize_peer_uri(advertised_uri(node, topology)))
        for node in topology["workers"]
    }
    worker_peer_ids.discard(None)

    for name, payload in sorted(collections.items()):
        if not isinstance(payload, dict):
            errors.append(f"collection {name} payload is not an object")
            continue
        info = payload.get("info") or {}
        cluster = payload.get("cluster") or {}
        if not isinstance(info, dict):
            errors.append(f"collection {name} info payload is not an object")
            info = {}
        if not isinstance(cluster, dict) or not cluster:
            errors.append(f"collection {name} has no cluster placement payload")
            continue
        if str(info.get("status") or "").lower() != "green":
            errors.append(
                f"collection {name} status is {info.get('status')!r}, expected 'green'"
            )
        reported_peer_id = str(cluster.get("peer_id") or "")
        if controller_peer_id and reported_peer_id != controller_peer_id:
            errors.append(
                f"collection {name} was not inspected from controller peer "
                f"{controller_peer_id}: reported {reported_peer_id or '<missing>'}"
            )
        local_shards = cluster.get("local_shards") or []
        remote_shards = cluster.get("remote_shards") or []
        shard_transfers = cluster.get("shard_transfers") or []
        if not isinstance(local_shards, list):
            errors.append(f"collection {name} local_shards payload is not a list")
            local_shards = []
        if not isinstance(remote_shards, list):
            errors.append(f"collection {name} remote_shards payload is not a list")
            remote_shards = []
        if not isinstance(shard_transfers, list):
            errors.append(f"collection {name} shard_transfers payload is not a list")
            shard_transfers = []
        if local_shards:
            errors.append(
                f"collection {name} has {len(local_shards)} lower shard(s) on controller"
            )
        if shard_transfers:
            errors.append(
                f"collection {name} has {len(shard_transfers)} shard transfer(s)"
            )
        if not remote_shards:
            errors.append(f"collection {name} has no remote lower-shard placement")
        for shard in [*local_shards, *remote_shards]:
            if str((shard or {}).get("state") or "") != "Active":
                errors.append(
                    f"collection {name} shard {(shard or {}).get('shard_id')} "
                    f"on peer {(shard or {}).get('peer_id', reported_peer_id)} is "
                    f"{(shard or {}).get('state')!r}, expected 'Active'"
                )
        unexpected_peer_ids = sorted(
            {
                str((shard or {}).get("peer_id") or "")
                for shard in remote_shards
                if str((shard or {}).get("peer_id") or "") not in worker_peer_ids
            }
        )
        if unexpected_peer_ids:
            errors.append(
                f"collection {name} has lower shards outside worker peers "
                f"{sorted(worker_peer_ids)}: {unexpected_peer_ids}"
            )
    return errors


def command_deploy(args: argparse.Namespace, topology: dict[str, Any]) -> int:
    repo = Path(args.repo).resolve()
    stored = read_manifest(topology, args.run_id)
    state = git_state(repo)
    image_tag = args.image_tag or stored.get("image", {}).get("tag") or image_tag_for_commit(state["commit"])
    tar_path = Path(
        stored.get("image", {}).get("tar_path")
        or shared_run_dir(topology, args.run_id) / f"{image_tag.replace(':', '_')}.tar"
    )
    if not args.dry_run and not tar_path.is_file():
        raise FileNotFoundError(f"image tar not found; run build first: {tar_path}")
    if not args.dry_run:
        archive = validate_manifest_image_archive(tar_path, image_tag, stored)
        desired_image_id = archive["image_id"]
    else:
        desired_image_id = str(stored.get("image", {}).get("id") or "")
    if args.dry_run:
        desired_image_id = "dry-run-image-id"

    deployment_nodes: list[dict[str, Any]] = []
    for index, node in enumerate(all_nodes(topology)):
        remote_id_result = run_on_node(
            node,
            ["sudo", "-n", "docker", "image", "inspect", image_tag, "--format", "{{.Id}}"],
            args,
            capture=True,
            check=False,
        )
        remote_id = remote_id_result.stdout.strip() if remote_id_result.returncode == 0 else ""
        if not args.dry_run and remote_id and remote_id != desired_image_id:
            raise RuntimeError(
                f"{node['ssh_host']} has tag {image_tag} with {remote_id}, expected {desired_image_id}"
            )
        if remote_id != desired_image_id:
            run_on_node(
                node,
                ["sudo", "-n", "docker", "load", "--input", str(tar_path)],
                args,
            )
        actual_id_result = run_on_node(
            node,
            ["sudo", "-n", "docker", "image", "inspect", image_tag, "--format", "{{.Id}}"],
            args,
            capture=True,
            check=False,
        )
        if args.dry_run:
            actual_image_id = desired_image_id
        elif actual_id_result.returncode == 0:
            actual_image_id = actual_id_result.stdout.strip()
        else:
            actual_image_id = ""
        if not args.dry_run and actual_image_id != desired_image_id:
            raise RuntimeError(
                f"{node['ssh_host']} tag {image_tag} resolved to "
                f"{actual_image_id or '<missing>'} after load/verification, "
                f"expected {desired_image_id}"
            )
        role_root = local_role_root(topology, args.run_id, str(node["role"]))
        run_on_node(node, ["mkdir", "-p", str(role_root / "storage")], args)
        name = container_name(args.run_id, str(node["role"]))
        inspected = inspect_container(node, name, args)
        if inspected is None:
            run_on_node(
                node,
                docker_run_command(
                    topology,
                    node,
                    args.run_id,
                    image_tag,
                    desired_image_id,
                    args.disable_peer_premerge,
                ),
                args,
            )
        elif not verify_container_reusable(
            inspected,
            node,
            args.run_id,
            desired_image_id,
            args.disable_peer_premerge,
        ):
            run_on_node(node, ["sudo", "-n", "docker", "start", name], args)
        if not args.dry_run:
            wait_http_ready(http_url(node, topology), args.wait_timeout)
        deployment_nodes.append(
            {
                **node,
                "container_name": name,
                "advertised_uri": advertised_uri(node, topology),
                "http_url": http_url(node, topology),
                "storage_path": str(role_root / "storage"),
                "image_id": actual_image_id,
                "peer_premerge_mode": expected_peer_premerge_mode(
                    node, args.disable_peer_premerge
                ),
                "nofile": {
                    "soft": CONTAINER_NOFILE_SOFT,
                    "hard": CONTAINER_NOFILE_HARD,
                },
                "facts": node_facts(node, args),
            }
        )
        if index == 0 and not args.dry_run:
            time.sleep(2.0)

    if not args.dry_run:
        snapshot = cluster_snapshot(topology)
        cluster_errors = cluster_validation_errors(topology, snapshot)
        if cluster_errors:
            raise RuntimeError("; ".join(cluster_errors))
        data = build_manifest_data(
            topology,
            args.run_id,
            repo,
            image_tag,
            desired_image_id,
            deployment_nodes,
            args.disable_peer_premerge,
            expected_peer_premerge_mode(
                topology["controller"], args.disable_peer_premerge
            ),
        )
        data["image"].update(stored.get("image") or {})
        data["cluster_snapshot"] = snapshot
        path = write_manifest(topology, args.run_id, data)
        print(f"Cluster ready; manifest: {path}")
    return 0


def status_for_node(
    topology: dict[str, Any],
    node: dict[str, Any],
    run_id: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    name = container_name(run_id, str(node["role"]))
    inspected = inspect_container(node, name, args)
    status: dict[str, Any] = {
        "role": node["role"],
        "ssh_host": node["ssh_host"],
        "private_ip": node["private_ip"],
        "container_name": name,
        "exists": inspected is not None,
        "peer_premerge_mode": inspected_peer_premerge_mode(inspected, node),
    }
    if inspected:
        status.update(
            {
                "running": bool((inspected.get("State") or {}).get("Running")),
                "health": (inspected.get("State") or {}).get("Health"),
                "image_id": inspected.get("Image"),
                "cpuset": (inspected.get("HostConfig") or {}).get("CpusetCpus"),
                "nofile": inspected_nofile_limits(inspected),
                "labels": (inspected.get("Config") or {}).get("Labels") or {},
            }
        )
    return status


def command_status(args: argparse.Namespace, topology: dict[str, Any]) -> int:
    node_statuses = [
        status_for_node(topology, node, args.run_id, args) for node in all_nodes(topology)
    ]
    controller_mode = next(
        (
            node.get("peer_premerge_mode")
            for node in node_statuses
            if node.get("role") == "controller"
        ),
        None,
    )
    status = {
        "run_id": args.run_id,
        "peer_premerge": peer_premerge_summary(
            args.disable_peer_premerge, controller_mode
        ),
        "nodes": node_statuses,
    }
    validation_errors = [
        f"container {node['container_name']} is not running"
        for node in node_statuses
        if not node.get("running")
    ]
    if not status["peer_premerge"]["matches_requested"]:
        validation_errors.append(
            "controller peer-premerge mode mismatch: "
            f"requested={status['peer_premerge']['requested_mode']}, "
            f"current={status['peer_premerge']['current_mode']}"
        )
    if not args.dry_run:
        try:
            status["cluster"] = cluster_snapshot(topology)
        except Exception as exc:
            status["cluster_error"] = str(exc)
            validation_errors.append(f"cluster endpoint is not accessible: {exc}")
        else:
            validation_errors.extend(
                cluster_validation_errors(topology, status["cluster"])
            )
            try:
                status["collections"] = run_collection_placements(
                    topology, args.run_id
                )
            except Exception as exc:
                status["collections_error"] = str(exc)
                validation_errors.append(
                    f"run-scoped collection placement is not accessible: {exc}"
                )
            else:
                validation_errors.extend(
                    collection_validation_errors(
                        topology,
                        status["cluster"],
                        status["collections"],
                    )
                )
    status["validation"] = {
        "ok": not validation_errors,
        "errors": validation_errors,
    }
    print(json.dumps(status, indent=2), flush=True)
    return 0 if status["validation"]["ok"] else 1


def command_down(args: argparse.Namespace, topology: dict[str, Any]) -> int:
    for node in all_nodes(topology):
        name = container_name(args.run_id, str(node["role"]))
        run_on_node(
            node,
            ["sudo", "-n", "docker", "stop", name],
            args,
            check=False,
        )
    return 0


def command_clean(args: argparse.Namespace, topology: dict[str, Any]) -> int:
    if not args.yes_delete_storage:
        raise RuntimeError("clean requires --yes-delete-storage; down preserves storage")
    for node in all_nodes(topology):
        for command in clean_commands(topology, node, args.run_id):
            run_on_node(node, command, args, check=False)
    run_dir = shared_run_dir(topology, args.run_id).resolve()
    shared_base = Path(topology["shared_root"]).resolve()
    if shared_base not in run_dir.parents:
        raise RuntimeError(f"refusing unsafe shared cleanup path: {run_dir}")
    if run_dir.exists() and not args.dry_run:
        shutil.rmtree(run_dir)
    return 0


def command_manifest(args: argparse.Namespace, topology: dict[str, Any]) -> int:
    repo = Path(args.repo).resolve()
    stored = read_manifest(topology, args.run_id)
    state = git_state(repo)
    image_tag = args.image_tag or stored.get("image", {}).get("tag") or image_tag_for_commit(state["commit"])
    image_id = stored.get("image", {}).get("id")
    nodes = []
    for node in all_nodes(topology):
        inspected = inspect_container(
            node, container_name(args.run_id, str(node["role"])), args
        )
        nodes.append(
            {
                **node,
                "container_name": container_name(args.run_id, str(node["role"])),
                "advertised_uri": advertised_uri(node, topology),
                "http_url": http_url(node, topology),
                "storage_path": str(local_role_root(topology, args.run_id, str(node["role"])) / "storage"),
                "peer_premerge_mode": inspected_peer_premerge_mode(inspected, node),
                "nofile": inspected_nofile_limits(inspected),
                "facts": node_facts(node, args),
            }
        )
    controller_mode = next(
        (
            node.get("peer_premerge_mode")
            for node in nodes
            if node.get("role") == "controller"
        ),
        None,
    )
    data = build_manifest_data(
        topology,
        args.run_id,
        repo,
        image_tag,
        image_id,
        nodes,
        args.disable_peer_premerge,
        controller_mode,
    )
    data["image"].update(stored.get("image") or {})
    if not args.dry_run:
        try:
            data["cluster_snapshot"] = cluster_snapshot(topology)
        except Exception as exc:
            data["cluster_snapshot_error"] = str(exc)
        path = write_manifest(topology, args.run_id, data)
        print(f"Wrote {path}")
    print(json.dumps(data, indent=2), flush=True)
    return 0


def main() -> int:
    args = parse_args()
    validate_run_id(args.run_id)
    topology = load_topology(args.topology)
    handlers = {
        "bootstrap": command_bootstrap,
        "build": command_build,
        "deploy": command_deploy,
        "status": command_status,
        "down": command_down,
        "clean": command_clean,
        "manifest": command_manifest,
    }
    return handlers[args.command](args, topology)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, FileNotFoundError, subprocess.CalledProcessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
