#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import math
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
COLLECTION_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,254}$")
SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
RESOURCE_PREFIX = "orion-dist-"
EXPECTED_COMMIT = "1a5ac4c47237b9224ae3e4ca28c2cefb2b514352"
PEER_PREMERGE_DISABLE_ENV = "QDRANT_DISABLE_SHARD_MAJOR_PEER_PREMERGE"
PEER_PREMERGE_MODE_LABEL = "orion.distributed.peer_premerge"
PEER_PREMERGE_SHARDS_PER_RPC_ENV = "QDRANT_ORION_PEER_PREMERGE_SHARDS_PER_RPC"
PEER_PREMERGE_SHARDS_PER_RPC_LABEL = (
    "orion.distributed.peer_premerge_shards_per_rpc"
)
CONTROLLER_FINGERPRINT_LABEL = "orion.distributed.controller_fingerprint"
NOFILE_LABEL = "orion.distributed.nofile"
CONTAINER_NOFILE_SOFT = 65536
CONTAINER_NOFILE_HARD = 65536
ORION_ARTIFACT_FORMAT_VERSION = 1
SIMPLE_KMEANS_ARTIFACT_FORMAT_VERSION = 1


def normalize_peer_premerge_shards_per_rpc(value: Any) -> str:
    """Return the canonical controller chunk setting: ``all`` or a decimal N."""
    if value is None:
        return "all"
    if isinstance(value, bool):
        raise ValueError(
            "peer-premerge shards-per-rpc must be all, 0, or a positive integer"
        )
    normalized = str(value).strip().lower()
    if normalized == "all":
        return "all"
    if not re.fullmatch(r"[0-9]+", normalized):
        raise ValueError(
            "peer-premerge shards-per-rpc must be all, 0, or a positive integer"
        )
    parsed = int(normalized, 10)
    if parsed == 0:
        return "all"
    if parsed > (2**63 - 1):
        raise ValueError("peer-premerge shards-per-rpc is too large for the runtime")
    return str(parsed)


def add_peer_premerge_chunk_argument(
    parser: argparse.ArgumentParser,
    *,
    option: str = "--peer-premerge-shards-per-rpc",
    default: str | None = "all",
) -> None:
    parser.add_argument(
        option,
        type=normalize_peer_premerge_shards_per_rpc,
        default=default,
        metavar="{all|0|N}",
        help=(
            "Controller-only maximum routed logical shards per peer-premerge RPC. "
            "all and 0 preserve one RPC per worker; N enables whole-shard chunking."
        ),
    )


def add_install_artifact_parser(
    subparsers: argparse._SubParsersAction,
    command: str,
    description: str,
) -> None:
    install_artifact = subparsers.add_parser(command, help=description)
    install_artifact.add_argument("--collection", required=True)
    install_artifact.add_argument("--generation", required=True, type=int)
    install_artifact.add_argument("--artifact", required=True)
    install_artifact.add_argument("--expected-sha256", required=True)
    install_artifact.add_argument(
        "--restart",
        nargs="?",
        const="workers-first",
        choices=("workers-first", "controller-first"),
        help=(
            "Activate and verify the already-configured policy by safely restarting "
            "only this run's containers. With no value, workers-first is used."
        ),
    )
    install_artifact.add_argument("--wait-timeout", type=float, default=180.0)


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
    add_peer_premerge_chunk_argument(deploy)

    status = subparsers.add_parser(
        "status", help="Inspect containers, endpoints, peers, and placement."
    )
    status.add_argument(
        "--disable-peer-premerge",
        action="store_true",
        help="Validate/report status against a controller with peer pre-merge disabled.",
    )
    add_peer_premerge_chunk_argument(status)
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
    add_peer_premerge_chunk_argument(manifest)
    set_peer_premerge = subparsers.add_parser(
        "set-peer-premerge",
        help=(
            "Safely recreate only this run's controller to change native "
            "shard-major peer pre-merge mode and/or RPC shard chunking."
        ),
    )
    set_peer_premerge.add_argument(
        "--mode",
        choices=("enabled", "disabled"),
        default=None,
    )
    add_peer_premerge_chunk_argument(
        set_peer_premerge,
        option="--shards-per-rpc",
        default=None,
    )
    set_peer_premerge.add_argument("--wait-timeout", type=float, default=180.0)
    add_install_artifact_parser(
        subparsers,
        "install-orion-artifact",
        (
            "Atomically install one canonical native Orion routing artifact on all "
            "four run-scoped storage volumes."
        ),
    )
    add_install_artifact_parser(
        subparsers,
        "install-simple-kmeans-artifact",
        (
            "Atomically install one canonical native Simple KMeans routing artifact "
            "on all four run-scoped storage volumes."
        ),
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


def validate_collection_name(collection: str) -> str:
    if not COLLECTION_NAME_RE.fullmatch(collection):
        raise ValueError(
            "collection must be one safe storage path component starting with an "
            "alphanumeric character and containing only letters, digits, dot, "
            "underscore, or dash"
        )
    return collection


def normalize_sha256(value: str) -> str:
    normalized = str(value).strip()
    if normalized.lower().startswith("sha256:"):
        normalized = normalized.split(":", 1)[1]
    if not SHA256_RE.fullmatch(normalized):
        raise ValueError("expected SHA-256 must be exactly 64 hexadecimal characters")
    return normalized.lower()


def validate_generation(generation: int) -> int:
    if (
        isinstance(generation, bool)
        or not isinstance(generation, int)
        or generation <= 0
    ):
        raise ValueError("routing artifact generation must be a positive integer")
    return generation


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


def validate_peer_premerge_mode(mode: str) -> str:
    normalized = str(mode).strip().lower()
    if normalized not in {"enabled", "disabled"}:
        raise ValueError(
            "peer-premerge mode must be exactly 'enabled' or 'disabled'"
        )
    return normalized


def peer_premerge_disabled(mode: str) -> bool:
    return validate_peer_premerge_mode(mode) == "disabled"


def expected_peer_premerge_shards_per_rpc(
    node: dict[str, Any], shards_per_rpc: Any = "all"
) -> str:
    if str(node.get("role")) != "controller":
        return "not_applicable"
    return normalize_peer_premerge_shards_per_rpc(shards_per_rpc)


def peer_premerge_shards_per_rpc_env_value(shards_per_rpc: Any) -> str | None:
    normalized = normalize_peer_premerge_shards_per_rpc(shards_per_rpc)
    return None if normalized == "all" else normalized


def controller_config_fingerprint(
    node: dict[str, Any],
    run_id: str,
    image_id: str,
    disable_peer_premerge: bool = False,
    peer_premerge_shards_per_rpc: Any = "all",
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
    normalized_shards_per_rpc = normalize_peer_premerge_shards_per_rpc(
        peer_premerge_shards_per_rpc
    )
    if normalized_shards_per_rpc != "all":
        payload["peer_premerge_shards_per_rpc"] = normalized_shards_per_rpc
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


def inspected_peer_premerge_shards_per_rpc(
    inspected: dict[str, Any] | None, node: dict[str, Any]
) -> str | None:
    if str(node.get("role")) != "controller":
        return "not_applicable"
    if not inspected:
        return None
    config = inspected.get("Config") or {}
    labels = config.get("Labels") or {}
    label_value = labels.get(PEER_PREMERGE_SHARDS_PER_RPC_LABEL)
    env_values = [
        str(item).partition("=")[2]
        for item in config.get("Env") or []
        if str(item).partition("=")[:2]
        == (PEER_PREMERGE_SHARDS_PER_RPC_ENV, "=")
    ]
    if label_value is None and not env_values:
        return "all"
    if label_value is None or len(env_values) != 1:
        return "inconsistent"
    try:
        label_normalized = normalize_peer_premerge_shards_per_rpc(label_value)
        env_normalized = normalize_peer_premerge_shards_per_rpc(env_values[0])
    except ValueError:
        return "inconsistent"
    if (
        label_normalized == "all"
        or env_normalized == "all"
        or label_normalized != env_normalized
    ):
        return "inconsistent"
    return label_normalized


def peer_premerge_summary(
    disable_peer_premerge: bool,
    current_mode: str | None = None,
    requested_shards_per_rpc: Any = "all",
    current_shards_per_rpc: str | None = "all",
) -> dict[str, Any]:
    requested_mode = "disabled" if disable_peer_premerge else "enabled"
    requested_shards_per_rpc = normalize_peer_premerge_shards_per_rpc(
        requested_shards_per_rpc
    )
    mode_matches = (
        current_mode == requested_mode if current_mode is not None else None
    )
    shards_match = (
        current_shards_per_rpc == requested_shards_per_rpc
        if current_shards_per_rpc is not None
        else None
    )
    return {
        "scope": "controller",
        "requested_mode": requested_mode,
        "current_mode": current_mode,
        "mode_matches_requested": mode_matches,
        "requested_shards_per_rpc": requested_shards_per_rpc,
        "current_shards_per_rpc": current_shards_per_rpc,
        "shards_per_rpc_matches_requested": shards_match,
        "matches_requested": (
            mode_matches and shards_match
            if mode_matches is not None and shards_match is not None
            else None
        ),
        "disable_environment_variable": PEER_PREMERGE_DISABLE_ENV,
        "shards_per_rpc_environment_variable": PEER_PREMERGE_SHARDS_PER_RPC_ENV,
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


def routing_artifact_destination(
    topology: dict[str, Any],
    run_id: str,
    role: str,
    collection: str,
    generation: int,
    *,
    router_directory: str,
    policy_label: str,
) -> Path:
    collection = validate_collection_name(collection)
    generation = validate_generation(generation)
    if router_directory not in {"orion_router", "simple_kmeans_router"}:
        raise ValueError(f"unsupported routing artifact directory: {router_directory!r}")
    storage = (local_role_root(topology, run_id, role) / "storage").resolve()
    destination = (
        storage
        / "collections"
        / collection
        / router_directory
        / f"generation-{generation}.json"
    ).resolve()
    if storage not in destination.parents:
        raise ValueError(
            f"unsafe {policy_label} artifact path outside {storage}: {destination}"
        )
    return destination


def orion_artifact_destination(
    topology: dict[str, Any],
    run_id: str,
    role: str,
    collection: str,
    generation: int,
) -> Path:
    return routing_artifact_destination(
        topology,
        run_id,
        role,
        collection,
        generation,
        router_directory="orion_router",
        policy_label="Orion",
    )


def simple_kmeans_artifact_destination(
    topology: dict[str, Any],
    run_id: str,
    role: str,
    collection: str,
    generation: int,
) -> Path:
    return routing_artifact_destination(
        topology,
        run_id,
        role,
        collection,
        generation,
        router_directory="simple_kmeans_router",
        policy_label="Simple KMeans",
    )


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
    # Do not use a remote login shell here. CloudLab's stock .bash_logout can
    # turn a successful `set -e` script into SSH status 1 while the login shell
    # exits, and orchestration commands must not depend on user profile files.
    wrapped.extend([target, "bash", "-c", shlex.quote(command_text)])
    return wrapped


def copy_to_node_command(
    node: dict[str, Any],
    source: str | Path,
    destination: str | Path,
    ssh_user: str | None = None,
    ssh_options: list[str] | None = None,
) -> list[str]:
    source_path = str(Path(source))
    destination_path = str(Path(destination))
    target = ssh_target(node, ssh_user)
    if target == "localhost":
        return ["cp", "--", source_path, destination_path]
    command = ["scp", "-q", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]
    for option in ssh_options or []:
        command.extend(["-o", option])
    command.extend(["--", source_path, f"{target}:{destination_path}"])
    return command


def _artifact_path_guard_script(storage_root: Path, destination: Path) -> str:
    storage_text = shlex.quote(str(storage_root))
    destination_text = shlex.quote(str(destination))
    return f"""
storage_root={storage_text}
destination={destination_text}
storage_real=$(realpath -m -- "$storage_root")
destination_real=$(realpath -m -- "$destination")
case "$destination_real" in
  "$storage_real"/*) ;;
  *) echo "unsafe artifact destination outside run storage: $destination_real" >&2; exit 70 ;;
esac
""".strip()


def artifact_probe_command(
    storage_root: Path,
    destination: Path,
    expected_sha256: str,
) -> str:
    expected_sha256 = normalize_sha256(expected_sha256)
    return f"""
set -euo pipefail
{_artifact_path_guard_script(storage_root, destination)}
if [ ! -e "$destination" ]; then
  printf 'missing\\n'
  exit 0
fi
if [ ! -f "$destination" ]; then
  echo "artifact destination exists but is not a regular file: $destination" >&2
  exit 71
fi
actual=$(sha256sum -- "$destination" | awk '{{print $1}}')
if [ "$actual" != {shlex.quote(expected_sha256)} ]; then
  echo "artifact generation already exists with a different SHA-256: $actual" >&2
  exit 72
fi
printf 'match %s\\n' "$actual"
""".strip()


def artifact_prepare_command(storage_root: Path, temporary: Path) -> str:
    return f"""
set -euo pipefail
{_artifact_path_guard_script(storage_root, temporary)}
mkdir -p -- "$(dirname -- "$destination")"
if [ -e "$destination" ] && [ ! -f "$destination" ]; then
  echo "artifact temporary path exists but is not a regular file: $destination" >&2
  exit 73
fi
""".strip()


def artifact_finalize_command(
    storage_root: Path,
    staged: Path,
    temporary: Path,
    destination: Path,
    expected_sha256: str,
) -> str:
    expected_sha256 = normalize_sha256(expected_sha256)
    staged_text = shlex.quote(str(staged))
    temporary_text = shlex.quote(str(temporary))
    destination_text = shlex.quote(str(destination))
    return f"""
set -euo pipefail
{_artifact_path_guard_script(storage_root, destination)}
staged={staged_text}
temporary={temporary_text}
final_destination={destination_text}
staged_real=$(realpath -m -- "$staged")
temporary_real=$(realpath -m -- "$temporary")
case "$staged_real" in
  "$storage_real"/*) ;;
  *) echo "unsafe staged artifact path outside run storage: $staged_real" >&2; exit 74 ;;
esac
case "$temporary_real" in
  "$storage_real"/*) ;;
  *) echo "unsafe artifact temporary path outside run storage: $temporary_real" >&2; exit 75 ;;
esac
if [ ! -f "$staged" ]; then
  echo "staged artifact file is missing: $staged" >&2
  exit 76
fi
actual=$(sha256sum -- "$staged" | awk '{{print $1}}')
if [ "$actual" != {shlex.quote(expected_sha256)} ]; then
  rm -f -- "$staged"
  echo "copied artifact SHA-256 mismatch: $actual" >&2
  exit 77
fi
sudo -n mkdir -p -- "$(dirname -- "$final_destination")"
sudo -n cp -- "$staged" "$temporary"
temporary_sha=$(sudo -n sha256sum -- "$temporary" | awk '{{print $1}}')
if [ "$temporary_sha" != {shlex.quote(expected_sha256)} ]; then
  sudo -n rm -f -- "$temporary"
  echo "same-directory temporary artifact SHA-256 mismatch: $temporary_sha" >&2
  exit 78
fi
sudo -n chmod 0644 -- "$temporary"
sudo -n mv --no-clobber -- "$temporary" "$final_destination"
if sudo -n test -e "$temporary"; then
  existing=$(sudo -n sha256sum -- "$final_destination" | awk '{{print $1}}')
  if [ "$existing" != {shlex.quote(expected_sha256)} ]; then
    sudo -n rm -f -- "$temporary"
    echo "artifact appeared concurrently with a different SHA-256: $existing" >&2
    exit 79
  fi
  sudo -n rm -f -- "$temporary"
fi
rm -f -- "$staged"
final_sha=$(sudo -n sha256sum -- "$final_destination" | awk '{{print $1}}')
if [ "$final_sha" != {shlex.quote(expected_sha256)} ]; then
  echo "installed artifact SHA-256 mismatch: $final_sha" >&2
  exit 80
fi
printf 'installed %s\\n' "$final_sha"
""".strip()


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
    peer_premerge_shards_per_rpc: Any = "all",
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
        normalized_shards_per_rpc = normalize_peer_premerge_shards_per_rpc(
            peer_premerge_shards_per_rpc
        )
        controller_fingerprint = controller_config_fingerprint(
            node,
            run_id,
            image_id,
            disable_peer_premerge,
            normalized_shards_per_rpc,
        )
        command.extend(
            [
                "--label",
                f"{PEER_PREMERGE_MODE_LABEL}="
                f"{expected_peer_premerge_mode(node, disable_peer_premerge)}",
                "--label",
                f"{CONTROLLER_FINGERPRINT_LABEL}={controller_fingerprint}",
            ]
        )
        if normalized_shards_per_rpc != "all":
            command.extend(
                [
                    "--label",
                    f"{PEER_PREMERGE_SHARDS_PER_RPC_LABEL}={normalized_shards_per_rpc}",
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
    if role == "controller":
        shards_per_rpc_env = peer_premerge_shards_per_rpc_env_value(
            peer_premerge_shards_per_rpc
        )
        if shards_per_rpc_env is not None:
            command.extend(
                ["-e", f"{PEER_PREMERGE_SHARDS_PER_RPC_ENV}={shards_per_rpc_env}"]
            )
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


def read_local_routing_artifact(
    artifact_path: str | Path,
    expected_sha256: str,
    *,
    policy_label: str,
    builder_name: str,
) -> tuple[Path, str, dict[str, Any]]:
    """Read the exact canonical artifact bytes that will be copied to every node."""
    expected_sha256 = normalize_sha256(expected_sha256)
    path = Path(artifact_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{policy_label} artifact not found: {path}")
    actual_file_sha256 = sha256_file(path)
    if actual_file_sha256 != expected_sha256:
        raise RuntimeError(
            f"{policy_label} artifact file SHA-256 does not match the expected canonical "
            f"SHA-256: expected={expected_sha256}, file={actual_file_sha256}. "
            f"Install the exact canonical JSON emitted by {builder_name}."
        )
    try:
        payload = json.loads(
            path.read_bytes(),
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON number {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"invalid {policy_label} artifact JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{policy_label} artifact root must be a JSON object")
    return path, actual_file_sha256, payload


def validate_local_orion_artifact(
    artifact_path: str | Path,
    generation: int,
    expected_sha256: str,
) -> dict[str, Any]:
    """Validate the canonical-builder-output/file-checksum contract before copying."""
    generation = validate_generation(generation)
    expected_sha256 = normalize_sha256(expected_sha256)
    path, actual_file_sha256, payload = read_local_routing_artifact(
        artifact_path,
        expected_sha256,
        policy_label="Orion",
        builder_name="orion_build_artifact",
    )
    format_version = payload.get("format_version")
    if (
        isinstance(format_version, bool)
        or not isinstance(format_version, int)
        or format_version != ORION_ARTIFACT_FORMAT_VERSION
    ):
        raise RuntimeError(
            "unsupported Orion artifact format version: "
            f"expected={ORION_ARTIFACT_FORMAT_VERSION}, actual={format_version!r}"
        )
    actual_generation = payload.get("generation")
    if (
        isinstance(actual_generation, bool)
        or not isinstance(actual_generation, int)
        or actual_generation != generation
    ):
        raise RuntimeError(
            "Orion artifact generation mismatch: "
            f"requested={generation}, artifact={actual_generation!r}"
        )
    if not isinstance(payload.get("upper_graph"), dict):
        raise RuntimeError(
            "Orion artifact is graphless; production installation requires upper_graph"
        )
    shard_count = payload.get("shard_count")
    if isinstance(shard_count, bool) or not isinstance(shard_count, int) or shard_count <= 0:
        raise ValueError("Orion artifact shard_count must be a positive integer")
    layout_sha256 = normalize_sha256(str(payload.get("layout_sha256") or ""))
    logical_point_count = payload.get("logical_point_count")
    physical_point_count = payload.get("physical_point_count")
    if (
        isinstance(logical_point_count, bool)
        or not isinstance(logical_point_count, int)
        or logical_point_count <= 0
    ):
        raise ValueError("Orion artifact logical_point_count must be a positive integer")
    if (
        isinstance(physical_point_count, bool)
        or not isinstance(physical_point_count, int)
        or physical_point_count < logical_point_count
    ):
        raise ValueError(
            "Orion artifact physical_point_count must be at least logical_point_count"
        )
    vector_schema = payload.get("vector_schema")
    if not isinstance(vector_schema, dict):
        raise ValueError("Orion artifact vector_schema must be a JSON object")
    return {
        "path": str(path),
        "generation": generation,
        "canonical_sha256": expected_sha256,
        "file_sha256": actual_file_sha256,
        "size_bytes": path.stat().st_size,
        "format_version": format_version,
        "shard_count": shard_count,
        "layout_sha256": layout_sha256,
        "logical_point_count": logical_point_count,
        "physical_point_count": physical_point_count,
        "vector_schema": vector_schema,
    }


def validate_local_simple_kmeans_artifact(
    artifact_path: str | Path,
    generation: int,
    expected_sha256: str,
) -> dict[str, Any]:
    """Validate a production static Simple KMeans routing artifact before copying."""
    generation = validate_generation(generation)
    expected_sha256 = normalize_sha256(expected_sha256)
    path, actual_file_sha256, payload = read_local_routing_artifact(
        artifact_path,
        expected_sha256,
        policy_label="Simple KMeans",
        builder_name="the Simple KMeans artifact builder",
    )
    format_version = payload.get("format_version")
    if (
        isinstance(format_version, bool)
        or not isinstance(format_version, int)
        or format_version != SIMPLE_KMEANS_ARTIFACT_FORMAT_VERSION
    ):
        raise RuntimeError(
            "unsupported Simple KMeans artifact format version: "
            f"expected={SIMPLE_KMEANS_ARTIFACT_FORMAT_VERSION}, actual={format_version!r}"
        )
    actual_generation = payload.get("generation")
    if (
        isinstance(actual_generation, bool)
        or not isinstance(actual_generation, int)
        or actual_generation != generation
    ):
        raise RuntimeError(
            "Simple KMeans artifact generation mismatch: "
            f"requested={generation}, artifact={actual_generation!r}"
        )
    if "upper_graph" in payload:
        raise RuntimeError("Simple KMeans artifact must not contain upper_graph")

    shard_count = payload.get("shard_count")
    if isinstance(shard_count, bool) or not isinstance(shard_count, int) or shard_count <= 0:
        raise ValueError("Simple KMeans artifact shard_count must be a positive integer")
    layout_sha256 = normalize_sha256(str(payload.get("layout_sha256") or ""))
    logical_point_count = payload.get("logical_point_count")
    physical_point_count = payload.get("physical_point_count")
    if (
        isinstance(logical_point_count, bool)
        or not isinstance(logical_point_count, int)
        or logical_point_count <= 0
    ):
        raise ValueError(
            "Simple KMeans artifact logical_point_count must be a positive integer"
        )
    if (
        isinstance(physical_point_count, bool)
        or not isinstance(physical_point_count, int)
        or physical_point_count != logical_point_count
    ):
        raise ValueError(
            "Simple KMeans artifact physical_point_count must equal logical_point_count"
        )

    vector_schema = payload.get("vector_schema")
    if not isinstance(vector_schema, dict):
        raise ValueError("Simple KMeans artifact vector_schema must be a JSON object")
    expected_schema_fields = {"vector_name", "dimension", "distance", "datatype"}
    if set(vector_schema) != expected_schema_fields:
        raise ValueError(
            "Simple KMeans artifact vector_schema fields must be exactly "
            f"{sorted(expected_schema_fields)}"
        )
    if not isinstance(vector_schema.get("vector_name"), str):
        raise ValueError("Simple KMeans artifact vector_name must be a string")
    dimension = vector_schema.get("dimension")
    if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0:
        raise ValueError("Simple KMeans artifact vector dimension must be positive")
    if vector_schema.get("distance") not in {"Cosine", "Dot", "Euclid", "Manhattan"}:
        raise ValueError("Simple KMeans artifact vector distance is unsupported")
    if vector_schema.get("datatype") not in {"float32", "float16", "uint8"}:
        raise ValueError("Simple KMeans artifact vector datatype is unsupported")
    if payload.get("routing_distance") != "squared_l2":
        raise ValueError(
            "Simple KMeans artifact routing_distance must be 'squared_l2'"
        )

    nprobe = payload.get("nprobe")
    if (
        isinstance(nprobe, bool)
        or not isinstance(nprobe, int)
        or nprobe <= 0
        or nprobe > shard_count
    ):
        raise ValueError(
            "Simple KMeans artifact nprobe must be positive and no larger than shard_count"
        )
    lower_hnsw_ef = payload.get("lower_hnsw_ef")
    if (
        isinstance(lower_hnsw_ef, bool)
        or not isinstance(lower_hnsw_ef, int)
        or lower_hnsw_ef <= 0
    ):
        raise ValueError("Simple KMeans artifact lower_hnsw_ef must be positive")

    centroids = payload.get("centroids")
    if not isinstance(centroids, list) or len(centroids) != shard_count:
        raise ValueError(
            "Simple KMeans artifact must contain exactly one centroid per shard"
        )
    seen_shards: set[int] = set()
    for centroid in centroids:
        if not isinstance(centroid, dict) or set(centroid) != {"shard_id", "vector"}:
            raise ValueError(
                "Simple KMeans artifact centroid entries require shard_id and vector"
            )
        shard_id = centroid.get("shard_id")
        if (
            isinstance(shard_id, bool)
            or not isinstance(shard_id, int)
            or not 0 <= shard_id < shard_count
        ):
            raise ValueError(
                f"Simple KMeans artifact centroid shard_id is out of range: {shard_id!r}"
            )
        if shard_id in seen_shards:
            raise ValueError(
                f"Simple KMeans artifact repeats centroid for shard {shard_id}"
            )
        seen_shards.add(shard_id)
        vector = centroid.get("vector")
        if not isinstance(vector, list) or len(vector) != dimension:
            raise ValueError(
                f"Simple KMeans centroid for shard {shard_id} must have dimension {dimension}"
            )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in vector
        ):
            raise ValueError(
                f"Simple KMeans centroid for shard {shard_id} contains a non-finite value"
            )
    if seen_shards != set(range(shard_count)):
        raise ValueError(
            "Simple KMeans artifact must contain exactly one centroid for every shard ID"
        )
    return {
        "path": str(path),
        "generation": generation,
        "canonical_sha256": expected_sha256,
        "file_sha256": actual_file_sha256,
        "size_bytes": path.stat().st_size,
        "format_version": format_version,
        "shard_count": shard_count,
        "layout_sha256": layout_sha256,
        "logical_point_count": logical_point_count,
        "physical_point_count": physical_point_count,
        "vector_schema": vector_schema,
        "nprobe": nprobe,
        "lower_hnsw_ef": lower_hnsw_ef,
        "routing_distance": "squared_l2",
    }


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
    tracked_status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=repo,
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    dirty_paths = [line[3:].strip() for line in status.splitlines() if len(line) >= 4]
    tracked_dirty_paths = [
        line[3:].strip() for line in tracked_status.splitlines() if len(line) >= 4
    ]
    return {
        "commit": commit,
        "short_commit": commit[:12],
        "dirty": bool(dirty_paths),
        "dirty_paths": dirty_paths,
        "tracked_dirty": bool(tracked_dirty_paths),
        "tracked_dirty_paths": tracked_dirty_paths,
        "untracked_entry_count": max(0, len(dirty_paths) - len(tracked_dirty_paths)),
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


def preserve_run_manifest_metadata(
    data: dict[str, Any], stored: dict[str, Any]
) -> dict[str, Any]:
    for key in (
        "orion_artifacts",
        "simple_kmeans_artifacts",
        "peer_premerge_transitions",
    ):
        if key in stored:
            data[key] = stored[key]
    return data


def upsert_routing_artifact_manifest(
    manifest: dict[str, Any],
    entry: dict[str, Any],
    *,
    policy_kind: str,
    policy_label: str,
    manifest_key: str,
) -> dict[str, Any]:
    collection = validate_collection_name(str(entry["collection"]))
    generation = validate_generation(entry["generation"])
    checksum = normalize_sha256(str(entry["canonical_sha256"]))
    artifacts = list(manifest.get(manifest_key) or [])
    replacement_index: int | None = None
    for index, existing in enumerate(artifacts):
        if not isinstance(existing, dict):
            raise RuntimeError(
                f"run manifest contains a malformed {policy_label} artifact entry"
            )
        existing_policy_kind = str(existing.get("policy_kind") or policy_kind)
        if existing_policy_kind != policy_kind:
            raise RuntimeError(
                f"run manifest {manifest_key} contains policy kind "
                f"{existing_policy_kind!r}, expected {policy_kind!r}"
            )
        existing_generation = existing.get("generation")
        if isinstance(existing_generation, bool) or not isinstance(
            existing_generation, int
        ):
            raise RuntimeError(
                f"run manifest contains an invalid {policy_label} generation"
            )
        if (
            str(existing.get("collection")) == collection
            and existing_generation == generation
        ):
            existing_checksum = normalize_sha256(
                str(existing.get("canonical_sha256") or "")
            )
            if existing_checksum != checksum:
                raise RuntimeError(
                    f"run manifest already records this {policy_label} collection/generation "
                    f"with a different checksum: {existing_checksum}"
                )
            replacement_index = index
            if existing.get("installed_at"):
                entry["installed_at"] = existing["installed_at"]
            break
    entry = {
        **entry,
        "policy_kind": policy_kind,
        "collection": collection,
        "generation": generation,
    }
    if replacement_index is None:
        artifacts.append(entry)
    else:
        artifacts[replacement_index] = entry
    artifacts.sort(key=lambda item: (str(item["collection"]), int(item["generation"])))
    manifest[manifest_key] = artifacts
    return manifest


def upsert_orion_artifact_manifest(
    manifest: dict[str, Any], entry: dict[str, Any]
) -> dict[str, Any]:
    return upsert_routing_artifact_manifest(
        manifest,
        entry,
        policy_kind="orion",
        policy_label="Orion",
        manifest_key="orion_artifacts",
    )


def upsert_simple_kmeans_artifact_manifest(
    manifest: dict[str, Any], entry: dict[str, Any]
) -> dict[str, Any]:
    return upsert_routing_artifact_manifest(
        manifest,
        entry,
        policy_kind="simple_kmeans",
        policy_label="Simple KMeans",
        manifest_key="simple_kmeans_artifacts",
    )


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
    peer_premerge_shards_per_rpc: Any = "all",
    current_peer_premerge_shards_per_rpc: str | None = None,
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
            disable_peer_premerge,
            current_peer_premerge_mode,
            peer_premerge_shards_per_rpc,
            current_peer_premerge_shards_per_rpc,
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
    preserve_run_manifest_metadata(data, stored)
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
    peer_premerge_shards_per_rpc: Any = "all",
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
        expected_shards_per_rpc = normalize_peer_premerge_shards_per_rpc(
            peer_premerge_shards_per_rpc
        )
        expected_fingerprint = controller_config_fingerprint(
            node,
            run_id,
            image_id,
            disable_peer_premerge,
            expected_shards_per_rpc,
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
        expected_chunk_label = (
            None if expected_shards_per_rpc == "all" else expected_shards_per_rpc
        )
        if labels.get(PEER_PREMERGE_SHARDS_PER_RPC_LABEL) != expected_chunk_label:
            mismatches[PEER_PREMERGE_SHARDS_PER_RPC_LABEL] = (
                labels.get(PEER_PREMERGE_SHARDS_PER_RPC_LABEL),
                expected_chunk_label,
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
        chunk_values = [
            str(item).partition("=")[2]
            for item in env
            if str(item).partition("=")[:2]
            == (PEER_PREMERGE_SHARDS_PER_RPC_ENV, "=")
        ]
        expected_chunk_values = (
            [] if expected_shards_per_rpc == "all" else [expected_shards_per_rpc]
        )
        if chunk_values != expected_chunk_values:
            mismatches[PEER_PREMERGE_SHARDS_PER_RPC_ENV] = (
                chunk_values,
                expected_chunk_values,
            )
    else:
        env = inspected_environment(inspected)
        if PEER_PREMERGE_SHARDS_PER_RPC_ENV in env:
            mismatches[PEER_PREMERGE_SHARDS_PER_RPC_ENV] = (
                env[PEER_PREMERGE_SHARDS_PER_RPC_ENV],
                None,
            )
        if PEER_PREMERGE_SHARDS_PER_RPC_LABEL in labels:
            mismatches[PEER_PREMERGE_SHARDS_PER_RPC_LABEL] = (
                labels[PEER_PREMERGE_SHARDS_PER_RPC_LABEL],
                None,
            )
    if mismatches:
        raise RuntimeError(
            f"container {inspected.get('Name')} exists with incompatible configuration: {mismatches}"
        )
    return bool((inspected.get("State") or {}).get("Running"))


def verify_run_container_identity(
    inspected: dict[str, Any] | None,
    node: dict[str, Any],
    run_id: str,
) -> None:
    name = container_name(run_id, str(node["role"]))
    if inspected is None:
        raise RuntimeError(f"run-scoped container does not exist: {name}")
    labels = (inspected.get("Config") or {}).get("Labels") or {}
    expected = {
        "orion.distributed.run_id": validate_run_id(run_id),
        "orion.distributed.role": str(node["role"]),
        "orion.distributed.private_ip": str(node["private_ip"]),
    }
    mismatches = {
        key: (labels.get(key), value)
        for key, value in expected.items()
        if labels.get(key) != value
    }
    if mismatches:
        raise RuntimeError(
            f"refusing to restart container {name} without exact run ownership: "
            f"{mismatches}"
        )


def inspected_environment(inspected: dict[str, Any]) -> dict[str, str]:
    environment: dict[str, str] = {}
    for item in (inspected.get("Config") or {}).get("Env") or []:
        key, separator, value = str(item).partition("=")
        if separator:
            environment[key] = value
    return environment


def expected_qdrant_command(
    topology: dict[str, Any], node: dict[str, Any]
) -> list[str]:
    command = ["./qdrant"]
    if str(node.get("role")) != "controller":
        command.extend(["--bootstrap", controller_uri(topology)])
    command.extend(["--uri", advertised_uri(node, topology)])
    return command


def verify_peer_premerge_transition_identity(
    topology: dict[str, Any],
    inspected: dict[str, Any] | None,
    node: dict[str, Any],
    run_id: str,
    image_tag: str,
    image_id: str,
    *,
    current_mode: str | None = None,
    current_shards_per_rpc: str | None = None,
    require_running: bool = True,
) -> bool:
    """Fail closed before a run-scoped controller replacement.

    This is intentionally stricter than normal deploy reuse.  A peer-premerge
    transition removes a container, so the complete immutable runtime identity
    must match the topology and run manifest before any destructive Docker
    command is issued.
    """
    role = str(node.get("role"))
    if inspected is None:
        raise RuntimeError(
            "run-scoped container does not exist: "
            f"{container_name(run_id, role)}"
        )
    if role == "controller":
        if current_mode is None:
            current_mode = inspected_peer_premerge_mode(inspected, node)
        if current_mode not in {"enabled", "disabled"}:
            raise RuntimeError(
                "controller peer-premerge mode is not internally consistent: "
                f"{current_mode!r}"
            )
        disable_peer_premerge = peer_premerge_disabled(current_mode)
        if current_shards_per_rpc is None:
            current_shards_per_rpc = inspected_peer_premerge_shards_per_rpc(
                inspected, node
            )
        if current_shards_per_rpc in {None, "inconsistent", "not_applicable"}:
            raise RuntimeError(
                "controller peer-premerge shards-per-rpc is not internally "
                f"consistent: {current_shards_per_rpc!r}"
            )
        current_shards_per_rpc = normalize_peer_premerge_shards_per_rpc(
            current_shards_per_rpc
        )
    else:
        disable_peer_premerge = False
        current_shards_per_rpc = "all"

    running = verify_container_reusable(
        inspected,
        node,
        run_id,
        image_id,
        disable_peer_premerge,
        current_shards_per_rpc,
    )
    mismatches: dict[str, tuple[Any, Any]] = {}
    expected_name = container_name(run_id, role)
    actual_name = str(inspected.get("Name") or "").lstrip("/")
    if actual_name != expected_name:
        mismatches["name"] = (actual_name, expected_name)

    config = inspected.get("Config") or {}
    if str(config.get("Image") or "") != image_tag:
        mismatches["image_tag"] = (config.get("Image"), image_tag)
    expected_command = expected_qdrant_command(topology, node)
    actual_command = [str(value) for value in config.get("Cmd") or []]
    if actual_command != expected_command:
        mismatches["command"] = (actual_command, expected_command)

    expected_environment = {
        "QDRANT__CLUSTER__ENABLED": "true",
        "QDRANT__CLUSTER__P2P__PORT": str(topology["ports"]["p2p"]),
        "QDRANT__SERVICE__HTTP_PORT": str(topology["ports"]["http"]),
        "QDRANT__SERVICE__GRPC_PORT": str(topology["ports"]["grpc"]),
        "QDRANT__STORAGE__PERFORMANCE__MAX_SEARCH_THREADS": str(
            node["max_search_threads"]
        ),
        "QDRANT__STORAGE__PERFORMANCE__OPTIMIZER_CPU_BUDGET": str(
            node["optimizer_cpu_budget"]
        ),
    }
    actual_environment = inspected_environment(inspected)
    for key, expected_value in expected_environment.items():
        if actual_environment.get(key) != expected_value:
            mismatches[f"environment:{key}"] = (
                actual_environment.get(key),
                expected_value,
            )
    if role != "controller" and PEER_PREMERGE_DISABLE_ENV in actual_environment:
        mismatches[f"environment:{PEER_PREMERGE_DISABLE_ENV}"] = (
            actual_environment[PEER_PREMERGE_DISABLE_ENV],
            None,
        )
    if (
        role != "controller"
        and PEER_PREMERGE_SHARDS_PER_RPC_ENV in actual_environment
    ):
        mismatches[f"environment:{PEER_PREMERGE_SHARDS_PER_RPC_ENV}"] = (
            actual_environment[PEER_PREMERGE_SHARDS_PER_RPC_ENV],
            None,
        )

    host_config = inspected.get("HostConfig") or {}
    if str(host_config.get("NetworkMode") or "") != "host":
        mismatches["network_mode"] = (host_config.get("NetworkMode"), "host")
    restart_policy = (host_config.get("RestartPolicy") or {}).get("Name")
    if str(restart_policy or "") != "unless-stopped":
        mismatches["restart_policy"] = (restart_policy, "unless-stopped")

    expected_storage = str(
        (local_role_root(topology, run_id, role) / "storage").resolve()
    )
    storage_mounts = [
        mount
        for mount in inspected.get("Mounts") or []
        if str((mount or {}).get("Destination") or "") == "/qdrant/storage"
    ]
    if len(storage_mounts) != 1:
        mismatches["storage_mount_count"] = (len(storage_mounts), 1)
    else:
        mount = storage_mounts[0] or {}
        actual_mount = {
            "type": str(mount.get("Type") or ""),
            "source": str(mount.get("Source") or ""),
            "rw": bool(mount.get("RW")),
        }
        expected_mount = {
            "type": "bind",
            "source": expected_storage,
            "rw": True,
        }
        if actual_mount != expected_mount:
            mismatches["storage_mount"] = (actual_mount, expected_mount)

    if require_running and not running:
        mismatches["running"] = (False, True)
    if mismatches:
        raise RuntimeError(
            f"container /{expected_name} is not safe for a peer-premerge "
            f"transition: {mismatches}"
        )
    return running


def router_log_command(
    run_id: str,
    node: dict[str, Any],
    since_epoch: int,
) -> list[str]:
    return [
        "sudo",
        "-n",
        "docker",
        "logs",
        "--since",
        str(since_epoch),
        container_name(run_id, str(node["role"])),
    ]


def orion_router_log_command(
    run_id: str,
    node: dict[str, Any],
    since_epoch: int,
) -> list[str]:
    return router_log_command(run_id, node, since_epoch)


def verify_routing_artifact_loaded_logs(
    node: dict[str, Any],
    run_id: str,
    collection: str,
    generation: int,
    since_epoch: int,
    args: argparse.Namespace,
    *,
    policy_label: str,
) -> dict[str, Any]:
    result = run_on_node(
        node,
        router_log_command(run_id, node, since_epoch),
        args,
        capture=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"could not read {node['role']} logs after {policy_label} restart: "
            f"{(result.stderr or result.stdout).strip()}"
        )
    logs = f"{result.stdout}\n{result.stderr}"
    loaded_marker = (
        f"Loaded {policy_label} routing generation {generation} for collection {collection}"
    )
    fallback_lines = [
        line
        for line in logs.splitlines()
        if policy_label.lower() in line.lower()
        and collection in line
        and re.search(r"\b(?:unavailable|fallback|falling back)\b", line, re.IGNORECASE)
    ]
    if fallback_lines:
        raise RuntimeError(
            f"{node['role']} reported {policy_label} fallback after restart; "
            f"markers={fallback_lines[:5]}"
        )
    if loaded_marker not in logs:
        raise RuntimeError(
            f"{node['role']} did not report loaded {policy_label} generation {generation} "
            f"for collection {collection} after restart"
        )
    return {
        "role": str(node["role"]),
        "container_name": container_name(run_id, str(node["role"])),
        "since_epoch": since_epoch,
        "loaded_marker": loaded_marker,
    }


def verify_orion_router_loaded_logs(
    node: dict[str, Any],
    run_id: str,
    collection: str,
    generation: int,
    since_epoch: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    return verify_routing_artifact_loaded_logs(
        node,
        run_id,
        collection,
        generation,
        since_epoch,
        args,
        policy_label="Orion",
    )


def verify_simple_kmeans_router_loaded_logs(
    node: dict[str, Any],
    run_id: str,
    collection: str,
    generation: int,
    since_epoch: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    return verify_routing_artifact_loaded_logs(
        node,
        run_id,
        collection,
        generation,
        since_epoch,
        args,
        policy_label="Simple KMeans",
    )


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


def wait_cluster_healthy(
    topology: dict[str, Any], timeout: float
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    last_errors: list[str] = ["cluster endpoint has not responded"]
    while time.monotonic() < deadline:
        try:
            snapshot = cluster_snapshot(topology)
            last_errors = cluster_validation_errors(topology, snapshot)
            if not last_errors:
                return snapshot
        except (OSError, urllib.error.URLError, ValueError, json.JSONDecodeError) as exc:
            last_errors = [str(exc)]
        time.sleep(1.0)
    raise TimeoutError(
        "timed out waiting for the four-peer cluster after artifact activation: "
        + "; ".join(last_errors)
    )


def cluster_snapshot(topology: dict[str, Any]) -> dict[str, Any]:
    url = f"{http_url(topology['controller'], topology)}/cluster"
    with urllib.request.urlopen(url, timeout=10.0) as response:
        return json.loads(response.read().decode("utf-8"))


def http_json(url: str) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=10.0) as response:
        return json.loads(response.read().decode("utf-8"))


def validate_collection_routing_policy(
    topology: dict[str, Any],
    collection: str,
    generation: int,
    expected_sha256: str,
    artifact: dict[str, Any] | None = None,
    *,
    policy_kind: str,
    policy_label: str,
) -> dict[str, Any]:
    collection = validate_collection_name(collection)
    generation = validate_generation(generation)
    expected_sha256 = normalize_sha256(expected_sha256)
    encoded_collection = urllib.parse.quote(collection, safe="")
    payload = http_json(
        f"{http_url(topology['controller'], topology).rstrip('/')}"
        f"/collections/{encoded_collection}"
    )
    result = payload.get("result") or {}
    config = result.get("config") or {}
    policy = config.get("auto_shard_policy")
    expected_policy = {
        "type": policy_kind,
        "generation": generation,
        "artifact_sha256": expected_sha256,
    }
    if not isinstance(policy, dict):
        raise RuntimeError(
            f"collection {collection} does not have an active {policy_label} "
            "auto-shard policy"
        )
    actual_type = str(policy.get("type") or "").lower()
    actual_generation = policy.get("generation")
    if isinstance(actual_generation, bool) or not isinstance(actual_generation, int):
        raise RuntimeError(
            f"collection {collection} has invalid {policy_label} policy generation: "
            f"{actual_generation!r}"
        )
    try:
        actual_sha256 = normalize_sha256(str(policy.get("artifact_sha256") or ""))
    except ValueError as exc:
        raise RuntimeError(
            f"collection {collection} has malformed {policy_label} policy metadata: "
            f"{policy}"
        ) from exc
    actual_policy = {
        "type": actual_type,
        "generation": actual_generation,
        "artifact_sha256": actual_sha256,
    }
    if actual_policy != expected_policy:
        raise RuntimeError(
            f"collection {collection} {policy_label} policy mismatch: "
            f"expected={expected_policy}, actual={actual_policy}"
        )
    if artifact is None:
        return {"policy": policy}

    status = str(result.get("status") or "").lower()
    if status != "green":
        raise RuntimeError(
            f"collection {collection} status is {status or '<missing>'}, expected green"
        )
    optimizer_status = result.get("optimizer_status")
    if str(optimizer_status or "").lower() != "ok":
        raise RuntimeError(
            f"collection {collection} optimizer status is {optimizer_status!r}, expected 'ok'"
        )
    params = config.get("params") or {}
    if str(params.get("sharding_method") or "auto").lower() != "auto":
        raise RuntimeError(
            f"collection {collection} is not using automatic numeric sharding"
        )
    actual_shard_count = params.get("shard_number")
    expected_shard_count = artifact.get("shard_count")
    if (
        isinstance(actual_shard_count, bool)
        or not isinstance(actual_shard_count, int)
        or actual_shard_count != expected_shard_count
    ):
        raise RuntimeError(
            f"collection {collection} shard count does not match the {policy_label} artifact: "
            f"collection={actual_shard_count!r}, artifact={expected_shard_count!r}"
        )

    artifact_schema = artifact.get("vector_schema") or {}
    vector_name = str(artifact_schema.get("vector_name") or "")
    vectors = params.get("vectors") or {}
    if vector_name:
        vector_params = vectors.get(vector_name) if isinstance(vectors, dict) else None
    else:
        vector_params = vectors if isinstance(vectors, dict) and "size" in vectors else None
    if not isinstance(vector_params, dict):
        raise RuntimeError(
            f"collection {collection} does not contain {policy_label} routing vector "
            f"{vector_name!r}"
        )
    collection_schema = {
        "vector_name": vector_name,
        "dimension": vector_params.get("size"),
        "distance": str(vector_params.get("distance") or "").lower(),
        "datatype": str(vector_params.get("datatype") or "float32").lower(),
    }
    expected_schema = {
        "vector_name": vector_name,
        "dimension": artifact_schema.get("dimension"),
        "distance": str(artifact_schema.get("distance") or "").lower(),
        "datatype": str(artifact_schema.get("datatype") or "float32").lower(),
    }
    if collection_schema != expected_schema:
        raise RuntimeError(
            f"collection {collection} vector schema does not match the {policy_label} "
            "artifact: "
            f"collection={collection_schema}, artifact={expected_schema}"
        )
    if vector_params.get("multivector_config") is not None:
        raise RuntimeError(
            f"collection {collection} {policy_label} routing vector must not be multivector"
        )
    actual_points_count = result.get("points_count")
    if actual_points_count != artifact.get("physical_point_count"):
        raise RuntimeError(
            f"collection {collection} physical point count does not match the "
            f"{policy_label} artifact: "
            f"collection={actual_points_count!r}, artifact={artifact.get('physical_point_count')!r}"
        )

    cluster_payload = http_json(
        f"{http_url(topology['controller'], topology).rstrip('/')}"
        f"/collections/{encoded_collection}/cluster"
    )
    cluster = cluster_payload.get("result") or {}
    transfers = cluster.get("shard_transfers") or []
    if transfers:
        raise RuntimeError(
            f"collection {collection} has {len(transfers)} shard transfer(s) in progress"
        )
    shards = [
        *(cluster.get("local_shards") or []),
        *(cluster.get("remote_shards") or []),
    ]
    logical_shards = {
        shard.get("shard_id") for shard in shards if isinstance(shard, dict)
    }
    if len(logical_shards) != expected_shard_count:
        raise RuntimeError(
            f"collection {collection} placement exposes {len(logical_shards)} logical "
            f"shards, expected {expected_shard_count}"
        )
    inactive = [
        shard
        for shard in shards
        if str((shard or {}).get("state") or "") != "Active"
    ]
    if inactive:
        raise RuntimeError(
            f"collection {collection} has non-Active shard replicas: {inactive}"
        )
    return {
        "policy_kind": policy_kind,
        "policy": policy,
        "status": status,
        "optimizer_status": optimizer_status,
        "shard_count": actual_shard_count,
        "vector_schema": collection_schema,
        "cluster": cluster,
    }


def validate_collection_orion_policy(
    topology: dict[str, Any],
    collection: str,
    generation: int,
    expected_sha256: str,
    artifact: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return validate_collection_routing_policy(
        topology,
        collection,
        generation,
        expected_sha256,
        artifact,
        policy_kind="orion",
        policy_label="Orion",
    )


def validate_collection_simple_kmeans_policy(
    topology: dict[str, Any],
    collection: str,
    generation: int,
    expected_sha256: str,
    artifact: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return validate_collection_routing_policy(
        topology,
        collection,
        generation,
        expected_sha256,
        artifact,
        policy_kind="simple_kmeans",
        policy_label="Simple KMeans",
    )


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
    consensus = result.get("consensus_thread_status")
    consensus_status = (
        consensus.get("consensus_thread_status")
        if isinstance(consensus, dict)
        else consensus
    )
    if consensus_status != "working":
        errors.append(
            f"consensus thread status is {consensus_status!r}, expected 'working'"
        )
    raft_info = result.get("raft_info") or {}
    pending_operations = (
        raft_info.get("pending_operations") if isinstance(raft_info, dict) else None
    )
    if pending_operations != 0:
        errors.append(
            f"raft pending operations is {pending_operations!r}, expected 0"
        )
    message_send_failures = result.get("message_send_failures")
    if message_send_failures != {}:
        errors.append(
            "cluster message_send_failures is not empty: "
            f"{message_send_failures!r}"
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
        if str(info.get("optimizer_status") or "").lower() != "ok":
            errors.append(
                f"collection {name} optimizer_status is "
                f"{info.get('optimizer_status')!r}, expected 'ok'"
            )
        update_queue = info.get("update_queue") or {}
        update_queue_length = (
            update_queue.get("length") if isinstance(update_queue, dict) else None
        )
        if update_queue_length != 0:
            errors.append(
                f"collection {name} update queue length is "
                f"{update_queue_length!r}, expected 0"
            )
        indexed_vectors_count = info.get("indexed_vectors_count")
        points_count = info.get("points_count")
        if (
            isinstance(indexed_vectors_count, int)
            and not isinstance(indexed_vectors_count, bool)
            and isinstance(points_count, int)
            and not isinstance(points_count, bool)
            and indexed_vectors_count != points_count
        ):
            errors.append(
                f"collection {name} indexed_vectors_count={indexed_vectors_count} "
                f"does not match points_count={points_count}"
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
        params = (((info.get("config") or {}).get("params") or {}))
        expected_shard_count = params.get("shard_number")
        if isinstance(expected_shard_count, bool) or not isinstance(
            expected_shard_count, int
        ):
            errors.append(
                f"collection {name} has invalid shard_number {expected_shard_count!r}"
            )
        elif len(remote_shards) != expected_shard_count:
            errors.append(
                f"collection {name} has {len(remote_shards)} remote shards, "
                f"expected {expected_shard_count}"
            )
        replication_factor = params.get("replication_factor")
        if replication_factor != 1:
            errors.append(
                f"collection {name} replication_factor is "
                f"{replication_factor!r}, expected 1"
            )
        shard_ids = [str((shard or {}).get("shard_id")) for shard in remote_shards]
        if len(set(shard_ids)) != len(shard_ids):
            errors.append(f"collection {name} has duplicate lower-shard replicas")
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
        worker_counts = {
            peer_id: sum(
                1
                for shard in remote_shards
                if str((shard or {}).get("peer_id") or "") == peer_id
            )
            for peer_id in worker_peer_ids
        }
        if worker_counts and max(worker_counts.values()) - min(worker_counts.values()) > 1:
            errors.append(
                f"collection {name} worker shard placement is imbalanced: {worker_counts}"
            )
    return errors


def peer_premerge_collection_validation_errors(
    topology: dict[str, Any],
    snapshot: dict[str, Any],
    collections: dict[str, Any],
    run_id: str,
) -> list[str]:
    """Require a real run workload before changing peer-premerge runtime state."""
    errors = collection_validation_errors(topology, snapshot, collections)
    if isinstance(collections, dict) and not collections:
        errors.insert(
            0,
            f"run {validate_run_id(run_id)!r} has no run-scoped collections",
        )
    return errors


def topology_runtime_identity(topology: dict[str, Any]) -> dict[str, Any]:
    return {
        "nodes": [
            {
                key: node.get(key)
                for key in (
                    "role",
                    "ssh_host",
                    "private_ip",
                    "cpuset",
                    "max_search_threads",
                    "optimizer_cpu_budget",
                )
            }
            for node in all_nodes(topology)
        ],
        "ports": dict(topology.get("ports") or {}),
        "local_storage_root": str(topology.get("local_storage_root") or ""),
    }


def validate_peer_premerge_transition_manifest(
    topology: dict[str, Any],
    run_id: str,
    manifest: dict[str, Any],
    image_tag_override: str | None = None,
    expected_deployment_commit: str | None = None,
) -> tuple[str, str, str, str, str]:
    run_id = validate_run_id(run_id)
    if not manifest:
        raise RuntimeError(
            f"run manifest does not exist for {run_id}; deploy the run first"
        )
    if str(manifest.get("run_id") or "") != run_id:
        raise RuntimeError(
            "run manifest identity mismatch: "
            f"expected={run_id}, actual={manifest.get('run_id')!r}"
        )
    stored_topology = manifest.get("topology")
    if not isinstance(stored_topology, dict):
        raise RuntimeError("run manifest does not contain the deployed topology")
    try:
        validate_topology(stored_topology)
        stored_runtime_identity = topology_runtime_identity(stored_topology)
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("run manifest contains a malformed topology") from exc
    if stored_runtime_identity != topology_runtime_identity(topology):
        raise RuntimeError(
            "run manifest topology does not match the requested controller/worker runtime"
        )

    repository = manifest.get("repository") or {}
    if not isinstance(repository, dict):
        raise RuntimeError("run manifest repository payload is malformed")
    deployment_commit = str(repository.get("commit") or "")
    if not deployment_commit:
        raise RuntimeError("run manifest does not contain the deployment commit")
    if (
        expected_deployment_commit
        and str(expected_deployment_commit) != deployment_commit
    ):
        raise RuntimeError(
            "run manifest deployment commit does not match --expected-commit: "
            f"manifest={deployment_commit}, expected={expected_deployment_commit}"
        )

    image = manifest.get("image") or {}
    if not isinstance(image, dict):
        raise RuntimeError("run manifest image payload is malformed")
    image_tag = str(image.get("tag") or "")
    image_id = str(image.get("id") or "")
    if not image_tag or not image_id:
        raise RuntimeError("run manifest does not contain an exact image tag and id")
    if image_tag_override and image_tag_override != image_tag:
        raise RuntimeError(
            "set-peer-premerge must reuse the deployed image tag: "
            f"manifest={image_tag}, override={image_tag_override}"
        )

    peer_premerge = manifest.get("peer_premerge") or {}
    if not isinstance(peer_premerge, dict):
        raise RuntimeError("run manifest peer-premerge payload is malformed")
    manifest_mode = peer_premerge.get("current_mode")
    if manifest_mode not in {"enabled", "disabled"}:
        raise RuntimeError(
            "run manifest does not contain a definitive controller peer-premerge mode"
        )
    try:
        manifest_shards_per_rpc = normalize_peer_premerge_shards_per_rpc(
            peer_premerge.get("current_shards_per_rpc", "all")
        )
    except ValueError as exc:
        raise RuntimeError(
            "run manifest contains an invalid controller peer-premerge "
            "shards-per-rpc value"
        ) from exc
    transitions = manifest.get("peer_premerge_transitions") or []
    if not isinstance(transitions, list) or any(
        not isinstance(item, dict) for item in transitions
    ):
        raise RuntimeError("run manifest contains malformed peer-premerge transitions")
    nodes = manifest.get("nodes") or []
    if not isinstance(nodes, list):
        raise RuntimeError("run manifest nodes payload is malformed")
    controller_nodes = [
        node
        for node in nodes
        if isinstance(node, dict) and str(node.get("role")) == "controller"
    ]
    if len(controller_nodes) != 1:
        raise RuntimeError(
            "run manifest must contain exactly one deployed controller node"
        )
    return (
        image_tag,
        image_id,
        str(manifest_mode),
        manifest_shards_per_rpc,
        deployment_commit,
    )


def transition_cluster_proof(snapshot: dict[str, Any] | None) -> dict[str, Any] | None:
    if snapshot is None:
        return None
    result = snapshot.get("result") or {}
    peers = result.get("peers") or {}
    raft_info = result.get("raft_info") or {}
    return {
        "peer_id": result.get("peer_id"),
        "peer_count": len(peers) if isinstance(peers, dict) else None,
        "peer_uris": sorted(
            normalize_peer_uri(str((peer or {}).get("uri") or ""))
            for peer in peers.values()
        )
        if isinstance(peers, dict)
        else [],
        "consensus_thread_status": result.get("consensus_thread_status"),
        "pending_operations": raft_info.get("pending_operations"),
        "message_send_failures": result.get("message_send_failures"),
    }


def transition_controller_proof(
    topology: dict[str, Any],
    run_id: str,
    inspected: dict[str, Any],
) -> dict[str, Any]:
    controller = topology["controller"]
    labels = (inspected.get("Config") or {}).get("Labels") or {}
    storage_mount = next(
        (
            mount
            for mount in inspected.get("Mounts") or []
            if str((mount or {}).get("Destination") or "") == "/qdrant/storage"
        ),
        {},
    )
    return {
        "container_id": str(inspected.get("Id") or ""),
        "container_name": container_name(run_id, "controller"),
        "private_ip": str(controller["private_ip"]),
        "image_id": str(inspected.get("Image") or ""),
        "image_tag": str((inspected.get("Config") or {}).get("Image") or ""),
        "cpuset": str((inspected.get("HostConfig") or {}).get("CpusetCpus") or ""),
        "nofile": inspected_nofile_limits(inspected),
        "network_mode": str(
            (inspected.get("HostConfig") or {}).get("NetworkMode") or ""
        ),
        "storage": {
            "type": storage_mount.get("Type"),
            "source": storage_mount.get("Source"),
            "destination": storage_mount.get("Destination"),
            "rw": storage_mount.get("RW"),
        },
        "peer_premerge_mode": inspected_peer_premerge_mode(inspected, controller),
        "peer_premerge_shards_per_rpc": inspected_peer_premerge_shards_per_rpc(
            inspected, controller
        ),
        "controller_fingerprint": labels.get(CONTROLLER_FINGERPRINT_LABEL),
    }


def transition_worker_proof(
    topology: dict[str, Any],
    run_id: str,
    node: dict[str, Any],
    inspected: dict[str, Any],
) -> dict[str, Any]:
    role = str(node["role"])
    storage_mount = next(
        (
            mount
            for mount in inspected.get("Mounts") or []
            if str((mount or {}).get("Destination") or "") == "/qdrant/storage"
        ),
        {},
    )
    return {
        "role": role,
        "container_id": str(inspected.get("Id") or ""),
        "container_name": container_name(run_id, role),
        "private_ip": str(node["private_ip"]),
        "image_id": str(inspected.get("Image") or ""),
        "image_tag": str((inspected.get("Config") or {}).get("Image") or ""),
        "cpuset": str((inspected.get("HostConfig") or {}).get("CpusetCpus") or ""),
        "nofile": inspected_nofile_limits(inspected),
        "network_mode": str(
            (inspected.get("HostConfig") or {}).get("NetworkMode") or ""
        ),
        "storage": {
            "type": storage_mount.get("Type"),
            "source": storage_mount.get("Source"),
            "destination": storage_mount.get("Destination"),
            "rw": storage_mount.get("RW"),
        },
    }


def transition_collections_proof(collections: dict[str, Any]) -> dict[str, Any]:
    proof: dict[str, Any] = {}
    for name, payload in sorted(collections.items()):
        info = (payload or {}).get("info") or {}
        cluster_payload = (payload or {}).get("cluster") or {}
        proof[name] = {
            "status": info.get("status"),
            "optimizer_status": info.get("optimizer_status"),
            "points_count": info.get("points_count"),
            "indexed_vectors_count": info.get("indexed_vectors_count"),
            "update_queue": info.get("update_queue"),
            "local_shards": cluster_payload.get("local_shards") or [],
            "remote_shards": cluster_payload.get("remote_shards") or [],
            "shard_transfers": cluster_payload.get("shard_transfers") or [],
        }
    return proof


def inspect_peer_premerge_transition_workers(
    topology: dict[str, Any],
    run_id: str,
    image_tag: str,
    image_id: str,
    args: argparse.Namespace,
) -> dict[str, dict[str, Any]]:
    inspected_workers: dict[str, dict[str, Any]] = {}
    for node in topology["workers"]:
        role = str(node["role"])
        inspected = inspect_container(node, container_name(run_id, role), args)
        verify_peer_premerge_transition_identity(
            topology,
            inspected,
            node,
            run_id,
            image_tag,
            image_id,
            require_running=True,
        )
        inspected_workers[role] = inspected
    return inspected_workers


def verify_peer_premerge_transition_proof_identity(
    *,
    action: str,
    before_controller: dict[str, Any],
    after_controller: dict[str, Any],
    workers_before: list[dict[str, Any]],
    workers_after: list[dict[str, Any]],
    cluster_before: dict[str, Any],
    cluster_after: dict[str, Any],
) -> None:
    before_controller_id = str(before_controller.get("container_id") or "")
    after_controller_id = str(after_controller.get("container_id") or "")
    if action == "recreated" and before_controller_id == after_controller_id:
        raise RuntimeError("controller recreation did not change the container ID")
    if action == "reused" and before_controller_id != after_controller_id:
        raise RuntimeError("reused controller changed the container ID")
    if workers_before != workers_after:
        raise RuntimeError("worker runtime identity changed during controller transition")
    if cluster_before.get("peer_id") != cluster_after.get("peer_id"):
        raise RuntimeError("controller peer ID changed during controller transition")


def update_peer_premerge_transition_manifest(
    topology: dict[str, Any],
    run_id: str,
    stored: dict[str, Any],
    requested_mode: str,
    final_mode: str | None,
    requested_shards_per_rpc: Any,
    final_shards_per_rpc: str | None,
    transition: dict[str, Any],
    cluster_snapshot_value: dict[str, Any] | None,
    controller_inspected: dict[str, Any] | None,
) -> dict[str, Any]:
    """Update operational state without dropping artifact or archive metadata."""
    requested_mode = validate_peer_premerge_mode(requested_mode)
    if final_mode is not None:
        final_mode = validate_peer_premerge_mode(final_mode)
    requested_shards_per_rpc = normalize_peer_premerge_shards_per_rpc(
        requested_shards_per_rpc
    )
    if final_shards_per_rpc is not None:
        final_shards_per_rpc = normalize_peer_premerge_shards_per_rpc(
            final_shards_per_rpc
        )
    data = json.loads(json.dumps(stored))
    transitions = list(data.get("peer_premerge_transitions") or [])
    transitions.append(transition)
    data["peer_premerge_transitions"] = transitions
    data["peer_premerge"] = peer_premerge_summary(
        peer_premerge_disabled(requested_mode),
        final_mode,
        requested_shards_per_rpc,
        final_shards_per_rpc,
    )
    data["generated_at"] = time.strftime(
        "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
    )
    data["last_peer_premerge_transition"] = transition.get("transition_id")
    if cluster_snapshot_value is not None:
        data["cluster_snapshot"] = cluster_snapshot_value
        data.pop("cluster_snapshot_error", None)
    if controller_inspected is not None:
        for node in data.get("nodes") or []:
            if isinstance(node, dict) and str(node.get("role")) == "controller":
                node["peer_premerge_mode"] = final_mode
                node["peer_premerge_shards_per_rpc"] = final_shards_per_rpc
                node["image_id"] = str(controller_inspected.get("Image") or "")
                node["cpuset"] = str(
                    (controller_inspected.get("HostConfig") or {}).get(
                        "CpusetCpus"
                    )
                    or ""
                )
                node["nofile"] = inspected_nofile_limits(controller_inspected)
                node["storage_path"] = str(
                    local_role_root(topology, run_id, "controller") / "storage"
                )
    return data


def command_deploy(args: argparse.Namespace, topology: dict[str, Any]) -> int:
    repo = Path(args.repo).resolve()
    requested_shards_per_rpc = normalize_peer_premerge_shards_per_rpc(
        getattr(args, "peer_premerge_shards_per_rpc", "all")
    )
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
                    requested_shards_per_rpc,
                ),
                args,
            )
        elif not verify_container_reusable(
            inspected,
            node,
            args.run_id,
            desired_image_id,
            args.disable_peer_premerge,
            requested_shards_per_rpc,
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
                "peer_premerge_shards_per_rpc": expected_peer_premerge_shards_per_rpc(
                    node, requested_shards_per_rpc
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
            requested_shards_per_rpc,
            expected_peer_premerge_shards_per_rpc(
                topology["controller"], requested_shards_per_rpc
            ),
        )
        preserve_run_manifest_metadata(data, stored)
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
        "peer_premerge_shards_per_rpc": inspected_peer_premerge_shards_per_rpc(
            inspected, node
        ),
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
    requested_shards_per_rpc = normalize_peer_premerge_shards_per_rpc(
        getattr(args, "peer_premerge_shards_per_rpc", "all")
    )
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
    controller_shards_per_rpc = next(
        (
            node.get("peer_premerge_shards_per_rpc")
            for node in node_statuses
            if node.get("role") == "controller"
        ),
        None,
    )
    status = {
        "run_id": args.run_id,
        "peer_premerge": peer_premerge_summary(
            args.disable_peer_premerge,
            controller_mode,
            requested_shards_per_rpc,
            controller_shards_per_rpc,
        ),
        "nodes": node_statuses,
    }
    validation_errors = [
        f"container {node['container_name']} is not running"
        for node in node_statuses
        if not node.get("running")
    ]
    if status["peer_premerge"]["mode_matches_requested"] is not True:
        validation_errors.append(
            "controller peer-premerge mode mismatch: "
            f"requested={status['peer_premerge']['requested_mode']}, "
            f"current={status['peer_premerge']['current_mode']}"
        )
    if status["peer_premerge"]["shards_per_rpc_matches_requested"] is not True:
        validation_errors.append(
            "controller peer-premerge shards-per-rpc mismatch: "
            f"requested={status['peer_premerge']['requested_shards_per_rpc']}, "
            f"current={status['peer_premerge']['current_shards_per_rpc']}"
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


def wait_controller_and_cluster_healthy(
    topology: dict[str, Any], timeout: float, run_id: str | None = None
) -> dict[str, Any]:
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("wait-timeout must be a positive finite number")
    started = time.monotonic()
    wait_http_ready(http_url(topology["controller"], topology), timeout)
    remaining = timeout - (time.monotonic() - started)
    if remaining <= 0:
        raise TimeoutError(
            "controller became HTTP-ready but exhausted the peer-premerge wait timeout"
        )
    if run_id is None:
        return wait_cluster_healthy(topology, remaining)

    run_id = validate_run_id(run_id)
    deadline = time.monotonic() + remaining
    last_errors: list[str] = ["run-scoped collections have not become healthy"]
    while time.monotonic() < deadline:
        try:
            snapshot = cluster_snapshot(topology)
            last_errors = cluster_validation_errors(topology, snapshot)
            collections = run_collection_placements(topology, run_id)
            last_errors.extend(
                collection_validation_errors(topology, snapshot, collections)
            )
            if not last_errors:
                return snapshot
        except (OSError, urllib.error.URLError, ValueError, json.JSONDecodeError) as exc:
            last_errors = [str(exc)]
        time.sleep(1.0)
    raise TimeoutError(
        "timed out waiting for the controller, cluster, and run-scoped collections: "
        + "; ".join(last_errors)
    )


def inspect_peer_premerge_transition_runtime(
    topology: dict[str, Any],
    run_id: str,
    image_tag: str,
    image_id: str,
    manifest_mode: str,
    manifest_shards_per_rpc: str,
    args: argparse.Namespace,
) -> tuple[dict[str, dict[str, Any]], str, str]:
    inspected_nodes: dict[str, dict[str, Any]] = {}
    controller_mode: str | None = None
    controller_shards_per_rpc: str | None = None
    for node in all_nodes(topology):
        role = str(node["role"])
        inspected = inspect_container(node, container_name(run_id, role), args)
        if inspected is None:
            raise RuntimeError(
                f"run-scoped container does not exist: {container_name(run_id, role)}"
            )
        if role == "controller":
            controller_mode = inspected_peer_premerge_mode(inspected, node)
            controller_shards_per_rpc = inspected_peer_premerge_shards_per_rpc(
                inspected, node
            )
        verify_peer_premerge_transition_identity(
            topology,
            inspected,
            node,
            run_id,
            image_tag,
            image_id,
            current_mode=controller_mode if role == "controller" else None,
            current_shards_per_rpc=(
                controller_shards_per_rpc if role == "controller" else None
            ),
            require_running=True,
        )
        inspected_nodes[role] = inspected
    if controller_mode != manifest_mode:
        raise RuntimeError(
            "controller peer-premerge mode disagrees with the run manifest: "
            f"container={controller_mode!r}, manifest={manifest_mode!r}"
        )
    if controller_shards_per_rpc != manifest_shards_per_rpc:
        raise RuntimeError(
            "controller peer-premerge shards-per-rpc disagrees with the run "
            "manifest: "
            f"container={controller_shards_per_rpc!r}, "
            f"manifest={manifest_shards_per_rpc!r}"
        )

    image_result = run_on_node(
        topology["controller"],
        [
            "sudo",
            "-n",
            "docker",
            "image",
            "inspect",
            image_tag,
            "--format",
            "{{.Id}}",
        ],
        args,
        capture=True,
        check=False,
    )
    resolved_image_id = image_result.stdout.strip() if image_result.returncode == 0 else ""
    if resolved_image_id != image_id:
        raise RuntimeError(
            "controller image tag no longer resolves to the deployed image id: "
            f"tag={image_tag}, expected={image_id}, "
            f"actual={resolved_image_id or '<missing>'}"
        )
    return inspected_nodes, str(controller_mode), str(controller_shards_per_rpc)


def replace_peer_premerge_controller(
    topology: dict[str, Any],
    run_id: str,
    image_tag: str,
    image_id: str,
    mode: str,
    shards_per_rpc: Any,
    args: argparse.Namespace,
) -> None:
    controller = topology["controller"]
    name = container_name(run_id, "controller")
    run_on_node(
        controller,
        ["sudo", "-n", "docker", "stop", name],
        args,
    )
    run_on_node(
        controller,
        ["sudo", "-n", "docker", "rm", name],
        args,
    )
    run_on_node(
        controller,
        docker_run_command(
            topology,
            controller,
            run_id,
            image_tag,
            image_id,
            peer_premerge_disabled(mode),
            shards_per_rpc,
        ),
        args,
    )


def rollback_peer_premerge_controller(
    topology: dict[str, Any],
    run_id: str,
    image_tag: str,
    image_id: str,
    original_mode: str,
    original_shards_per_rpc: str,
    wait_timeout: float,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Restore the exact original controller config without touching workers."""
    controller = topology["controller"]
    name = container_name(run_id, "controller")
    inspected = inspect_container(controller, name, args)
    if inspected is not None:
        candidate_mode = inspected_peer_premerge_mode(inspected, controller)
        candidate_shards_per_rpc = inspected_peer_premerge_shards_per_rpc(
            inspected, controller
        )
        verify_peer_premerge_transition_identity(
            topology,
            inspected,
            controller,
            run_id,
            image_tag,
            image_id,
            current_mode=candidate_mode,
            current_shards_per_rpc=candidate_shards_per_rpc,
            require_running=False,
        )
        if (
            candidate_mode == original_mode
            and candidate_shards_per_rpc == original_shards_per_rpc
        ):
            if not bool((inspected.get("State") or {}).get("Running")):
                run_on_node(
                    controller,
                    ["sudo", "-n", "docker", "start", name],
                    args,
                )
        else:
            run_on_node(
                controller,
                ["sudo", "-n", "docker", "stop", name],
                args,
                check=False,
            )
            run_on_node(
                controller,
                ["sudo", "-n", "docker", "rm", name],
                args,
            )
            inspected = None
    if inspected is None:
        run_on_node(
            controller,
            docker_run_command(
                topology,
                controller,
                run_id,
                image_tag,
                image_id,
                peer_premerge_disabled(original_mode),
                original_shards_per_rpc,
            ),
            args,
        )

    snapshot = wait_controller_and_cluster_healthy(topology, wait_timeout, run_id)
    restored = inspect_container(controller, name, args)
    verify_peer_premerge_transition_identity(
        topology,
        restored,
        controller,
        run_id,
        image_tag,
        image_id,
        current_mode=original_mode,
        current_shards_per_rpc=original_shards_per_rpc,
        require_running=True,
    )
    return snapshot, restored


def command_set_peer_premerge(
    args: argparse.Namespace, topology: dict[str, Any]
) -> int:
    run_id = validate_run_id(args.run_id)
    requested_mode_arg = getattr(args, "mode", None)
    requested_shards_per_rpc_arg = getattr(args, "shards_per_rpc", None)
    if requested_mode_arg is None and requested_shards_per_rpc_arg is None:
        raise ValueError(
            "set-peer-premerge requires --mode and/or --shards-per-rpc"
        )
    wait_timeout = float(args.wait_timeout)
    if not math.isfinite(wait_timeout) or wait_timeout <= 0:
        raise ValueError("wait-timeout must be a positive finite number")

    stored = read_manifest(topology, run_id)
    (
        image_tag,
        image_id,
        manifest_mode,
        manifest_shards_per_rpc,
        deployment_commit,
    ) = (
        validate_peer_premerge_transition_manifest(
            topology,
            run_id,
            stored,
            args.image_tag,
            args.expected_commit,
        )
    )
    requested_mode = (
        manifest_mode
        if requested_mode_arg is None
        else validate_peer_premerge_mode(requested_mode_arg)
    )
    requested_shards_per_rpc = (
        manifest_shards_per_rpc
        if requested_shards_per_rpc_arg is None
        else normalize_peer_premerge_shards_per_rpc(
            requested_shards_per_rpc_arg
        )
    )
    controller = topology["controller"]
    name = container_name(run_id, "controller")

    if args.dry_run:
        action = (
            "reused"
            if manifest_mode == requested_mode
            and manifest_shards_per_rpc == requested_shards_per_rpc
            else "recreated"
        )
        if action == "recreated":
            replace_peer_premerge_controller(
                topology,
                run_id,
                image_tag,
                image_id,
                requested_mode,
                requested_shards_per_rpc,
                args,
            )
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "run_id": run_id,
                    "controller": name,
                    "from_mode": manifest_mode,
                    "from_shards_per_rpc": manifest_shards_per_rpc,
                    "requested_mode": requested_mode,
                    "requested_shards_per_rpc": requested_shards_per_rpc,
                    "action": action,
                    "workers_restarted": False,
                    "storage_preserved": True,
                    "manifest_written": False,
                },
                indent=2,
            ),
            flush=True,
        )
        return 0

    inspected_nodes, current_mode, current_shards_per_rpc = (
        inspect_peer_premerge_transition_runtime(
            topology,
            run_id,
            image_tag,
            image_id,
            manifest_mode,
            manifest_shards_per_rpc,
            args,
        )
    )
    before_snapshot = cluster_snapshot(topology)
    before_errors = cluster_validation_errors(topology, before_snapshot)
    before_collections = run_collection_placements(topology, run_id)
    before_errors.extend(
        peer_premerge_collection_validation_errors(
            topology, before_snapshot, before_collections, run_id
        )
    )
    if before_errors:
        raise RuntimeError(
            "refusing peer-premerge transition from an unhealthy cluster: "
            + "; ".join(before_errors)
        )

    previous_transitions = list(stored.get("peer_premerge_transitions") or [])
    transition_id = f"{run_id}-peer-premerge-{len(previous_transitions) + 1:04d}"
    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    before_controller = inspected_nodes["controller"]
    workers_before = [
        transition_worker_proof(
            topology,
            run_id,
            node,
            inspected_nodes[str(node["role"])],
        )
        for node in topology["workers"]
    ]
    tooling_state = git_state(Path(args.repo).resolve())
    base_proof: dict[str, Any] = {
        "schema_version": 1,
        "transition_id": transition_id,
        "run_id": run_id,
        "started_at": started_at,
        "deployment_commit": deployment_commit,
        "tooling_repository": {
            "path": str(Path(args.repo).resolve()),
            "commit": tooling_state["commit"],
            "tracked_dirty": tooling_state["tracked_dirty"],
            "tracked_dirty_paths": tooling_state["tracked_dirty_paths"],
            "untracked_entry_count": tooling_state["untracked_entry_count"],
        },
        "from_mode": current_mode,
        "from_shards_per_rpc": current_shards_per_rpc,
        "requested_mode": requested_mode,
        "requested_shards_per_rpc": requested_shards_per_rpc,
        "workers_restarted": False,
        "storage_preserved": True,
        "controller_before": transition_controller_proof(
            topology, run_id, before_controller
        ),
        "validated_workers": workers_before,
        "workers_before": workers_before,
        "cluster_before": transition_cluster_proof(before_snapshot),
        "collections_before": transition_collections_proof(before_collections),
    }

    if (
        current_mode == requested_mode
        and current_shards_per_rpc == requested_shards_per_rpc
    ):
        after_snapshot = wait_controller_and_cluster_healthy(
            topology, wait_timeout, run_id
        )
        after_controller = inspect_container(controller, name, args)
        verify_peer_premerge_transition_identity(
            topology,
            after_controller,
            controller,
            run_id,
            image_tag,
            image_id,
            current_mode=requested_mode,
            current_shards_per_rpc=requested_shards_per_rpc,
            require_running=True,
        )
        inspected_workers_after = inspect_peer_premerge_transition_workers(
            topology, run_id, image_tag, image_id, args
        )
        workers_after = [
            transition_worker_proof(
                topology,
                run_id,
                node,
                inspected_workers_after[str(node["role"])],
            )
            for node in topology["workers"]
        ]
        after_collections = run_collection_placements(topology, run_id)
        after_errors = peer_premerge_collection_validation_errors(
            topology, after_snapshot, after_collections, run_id
        )
        if after_errors:
            raise RuntimeError(
                "peer-premerge reuse left unhealthy collections: "
                + "; ".join(after_errors)
            )
        controller_after_proof = transition_controller_proof(
            topology, run_id, after_controller
        )
        cluster_after_proof = transition_cluster_proof(after_snapshot)
        verify_peer_premerge_transition_proof_identity(
            action="reused",
            before_controller=base_proof["controller_before"],
            after_controller=controller_after_proof,
            workers_before=workers_before,
            workers_after=workers_after,
            cluster_before=base_proof["cluster_before"],
            cluster_after=cluster_after_proof,
        )
        proof = {
            **base_proof,
            "finished_at": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
            ),
            "action": "reused",
            "outcome": "success",
            "final_mode": requested_mode,
            "final_shards_per_rpc": requested_shards_per_rpc,
            "rollback": {"attempted": False},
            "controller_after": controller_after_proof,
            "workers_after": workers_after,
            "cluster_after": cluster_after_proof,
            "collections_after": transition_collections_proof(after_collections),
        }
        updated = update_peer_premerge_transition_manifest(
            topology,
            run_id,
            stored,
            requested_mode,
            requested_mode,
            requested_shards_per_rpc,
            requested_shards_per_rpc,
            proof,
            after_snapshot,
            after_controller,
        )
        path = write_manifest(topology, run_id, updated)
        print(
            json.dumps(
                {
                    "run_id": run_id,
                    "action": "reused",
                    "mode": requested_mode,
                    "shards_per_rpc": requested_shards_per_rpc,
                    "transition_id": transition_id,
                    "manifest": str(path),
                },
                indent=2,
            ),
            flush=True,
        )
        return 0

    mutation_started = False
    try:
        mutation_started = True
        replace_peer_premerge_controller(
            topology,
            run_id,
            image_tag,
            image_id,
            requested_mode,
            requested_shards_per_rpc,
            args,
        )
        after_controller = inspect_container(controller, name, args)
        verify_peer_premerge_transition_identity(
            topology,
            after_controller,
            controller,
            run_id,
            image_tag,
            image_id,
            current_mode=requested_mode,
            current_shards_per_rpc=requested_shards_per_rpc,
            require_running=True,
        )
        after_snapshot = wait_controller_and_cluster_healthy(
            topology, wait_timeout, run_id
        )
        after_controller = inspect_container(controller, name, args)
        verify_peer_premerge_transition_identity(
            topology,
            after_controller,
            controller,
            run_id,
            image_tag,
            image_id,
            current_mode=requested_mode,
            current_shards_per_rpc=requested_shards_per_rpc,
            require_running=True,
        )
        inspected_workers_after = inspect_peer_premerge_transition_workers(
            topology, run_id, image_tag, image_id, args
        )
        workers_after = [
            transition_worker_proof(
                topology,
                run_id,
                node,
                inspected_workers_after[str(node["role"])],
            )
            for node in topology["workers"]
        ]
        after_collections = run_collection_placements(topology, run_id)
        after_errors = peer_premerge_collection_validation_errors(
            topology, after_snapshot, after_collections, run_id
        )
        if after_errors:
            raise RuntimeError(
                "peer-premerge recreation left unhealthy collections: "
                + "; ".join(after_errors)
            )
        controller_after_proof = transition_controller_proof(
            topology, run_id, after_controller
        )
        cluster_after_proof = transition_cluster_proof(after_snapshot)
        verify_peer_premerge_transition_proof_identity(
            action="recreated",
            before_controller=base_proof["controller_before"],
            after_controller=controller_after_proof,
            workers_before=workers_before,
            workers_after=workers_after,
            cluster_before=base_proof["cluster_before"],
            cluster_after=cluster_after_proof,
        )
        proof = {
            **base_proof,
            "finished_at": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
            ),
            "action": "recreated",
            "outcome": "success",
            "final_mode": requested_mode,
            "final_shards_per_rpc": requested_shards_per_rpc,
            "rollback": {"attempted": False},
            "controller_after": controller_after_proof,
            "workers_after": workers_after,
            "cluster_after": cluster_after_proof,
            "collections_after": transition_collections_proof(after_collections),
        }
        updated = update_peer_premerge_transition_manifest(
            topology,
            run_id,
            stored,
            requested_mode,
            requested_mode,
            requested_shards_per_rpc,
            requested_shards_per_rpc,
            proof,
            after_snapshot,
            after_controller,
        )
        path = write_manifest(topology, run_id, updated)
    except Exception as transition_error:
        if not mutation_started:
            raise
        rollback_snapshot: dict[str, Any] | None = None
        rollback_controller: dict[str, Any] | None = None
        rollback_error: Exception | None = None
        try:
            rollback_snapshot, rollback_controller = rollback_peer_premerge_controller(
                topology,
                run_id,
                image_tag,
                image_id,
                current_mode,
                current_shards_per_rpc,
                wait_timeout,
                args,
            )
        except Exception as exc:
            rollback_error = exc

        proof = {
            **base_proof,
            "finished_at": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
            ),
            "action": "recreated",
            "outcome": (
                "failed_rolled_back"
                if rollback_error is None
                else "failed_rollback_failed"
            ),
            "final_mode": current_mode if rollback_error is None else None,
            "final_shards_per_rpc": (
                current_shards_per_rpc if rollback_error is None else None
            ),
            "forward_error": str(transition_error),
            "rollback": {
                "attempted": True,
                "succeeded": rollback_error is None,
                "error": str(rollback_error) if rollback_error else None,
            },
            "controller_after": transition_controller_proof(
                topology, run_id, rollback_controller
            )
            if rollback_controller is not None
            else None,
            "cluster_after": transition_cluster_proof(rollback_snapshot),
        }
        proof_write_error: Exception | None = None
        try:
            updated = update_peer_premerge_transition_manifest(
                topology,
                run_id,
                stored,
                requested_mode,
                current_mode if rollback_error is None else None,
                requested_shards_per_rpc,
                current_shards_per_rpc if rollback_error is None else None,
                proof,
                rollback_snapshot,
                rollback_controller,
            )
            write_manifest(topology, run_id, updated)
        except Exception as exc:
            proof_write_error = exc
        details = [f"controller transition failed: {transition_error}"]
        if rollback_error is None:
            details.append(
                f"controller rolled back to {current_mode} with "
                f"shards_per_rpc={current_shards_per_rpc}"
            )
        else:
            details.append(f"controller rollback failed: {rollback_error}")
        if proof_write_error is not None:
            details.append(f"transition proof write failed: {proof_write_error}")
        raise RuntimeError("; ".join(details)) from transition_error

    print(
        json.dumps(
            {
                "run_id": run_id,
                "action": "recreated",
                "from_mode": current_mode,
                "from_shards_per_rpc": current_shards_per_rpc,
                "mode": requested_mode,
                "shards_per_rpc": requested_shards_per_rpc,
                "transition_id": transition_id,
                "manifest": str(path),
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


def artifact_restart_nodes(
    topology: dict[str, Any], restart_order: str
) -> list[dict[str, Any]]:
    if restart_order == "controller-first":
        return [topology["controller"], *topology["workers"]]
    if restart_order == "workers-first":
        return [*topology["workers"], topology["controller"]]
    raise ValueError(f"unsupported artifact restart order: {restart_order}")


def command_install_routing_artifact(
    args: argparse.Namespace,
    topology: dict[str, Any],
    *,
    policy_kind: str,
    policy_label: str,
    manifest_key: str,
    staging_directory: str,
    validate_local_artifact: Any,
    validate_collection_policy: Any,
    artifact_destination: Any,
    upsert_artifact_manifest: Any,
    verify_router_loaded_logs: Any,
) -> int:
    run_id = validate_run_id(args.run_id)
    collection = validate_collection_name(args.collection)
    generation = validate_generation(args.generation)
    expected_sha256 = normalize_sha256(args.expected_sha256)
    if args.wait_timeout <= 0:
        raise ValueError("wait-timeout must be greater than zero")
    artifact = validate_local_artifact(args.artifact, generation, expected_sha256)
    source = Path(artifact["path"])
    stored = read_manifest(topology, run_id)
    if not stored:
        raise RuntimeError(
            f"run manifest does not exist for {run_id}; deploy the run before installing artifacts"
        )
    if str(stored.get("run_id") or "") != run_id:
        raise RuntimeError(
            "run manifest identity mismatch: "
            f"expected={run_id}, actual={stored.get('run_id')!r}"
        )
    preinstall_collection_proof = None
    if not args.dry_run:
        preinstall_collection_proof = validate_collection_policy(
            topology, collection, generation, expected_sha256, artifact
        )

    installed_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    base_entry: dict[str, Any] = {
        "policy_kind": policy_kind,
        "collection": collection,
        "generation": generation,
        "canonical_sha256": expected_sha256,
        "file_sha256": artifact["file_sha256"],
        "source_path": str(source),
        "size_bytes": artifact["size_bytes"],
        "format_version": artifact["format_version"],
        "shard_count": artifact["shard_count"],
        "layout_sha256": artifact["layout_sha256"],
        "logical_point_count": artifact["logical_point_count"],
        "physical_point_count": artifact["physical_point_count"],
        "vector_schema": artifact["vector_schema"],
        "installed_at": installed_at,
        "last_verified_at": installed_at,
        "preinstall_collection_proof": preinstall_collection_proof,
        "activation": {
            "status": "restart_in_progress" if args.restart else "not_requested",
            "restart_order": args.restart,
        },
        "nodes": [],
    }
    for optional_field in ("routing_distance", "nprobe", "lower_hnsw_ef"):
        if optional_field in artifact:
            base_entry[optional_field] = artifact[optional_field]
    prospective_manifest = {
        **stored,
        manifest_key: list(stored.get(manifest_key) or []),
    }
    upsert_artifact_manifest(prospective_manifest, dict(base_entry))

    plans: list[dict[str, Any]] = []
    for node in all_nodes(topology):
        role = str(node["role"])
        storage_root = (local_role_root(topology, run_id, role) / "storage").resolve()
        destination = artifact_destination(
            topology, run_id, role, collection, generation
        )
        staged = (
            storage_root
            / staging_directory
            / f"{collection}-generation-{generation}-{expected_sha256[:16]}.json"
        ).resolve()
        temporary = destination.with_name(
            f".{destination.name}.tmp-{expected_sha256[:16]}"
        )
        probe = run_on_node(
            node,
            artifact_probe_command(storage_root, destination, expected_sha256),
            args,
            capture=True,
            check=False,
        )
        if probe.returncode != 0:
            detail = (probe.stderr or probe.stdout or "probe failed").strip()
            raise RuntimeError(
                f"{node['ssh_host']} refused {policy_label} artifact installation: {detail}"
            )
        state = "reused" if probe.stdout.strip().startswith("match ") else "missing"
        plans.append(
            {
                "node": node,
                "role": role,
                "storage_root": storage_root,
                "destination": destination,
                "staged": staged,
                "temporary": temporary,
                "action": state,
            }
        )

    for plan in plans:
        if plan["action"] == "reused":
            continue
        node = plan["node"]
        run_on_node(
            node,
            artifact_prepare_command(plan["storage_root"], plan["staged"]),
            args,
        )
        run_command(
            copy_to_node_command(
                node,
                source,
                plan["staged"],
                args.ssh_user,
                args.ssh_option,
            ),
            dry_run=args.dry_run,
        )
        finalized = run_on_node(
            node,
            artifact_finalize_command(
                plan["storage_root"],
                plan["staged"],
                plan["temporary"],
                plan["destination"],
                expected_sha256,
            ),
            args,
            capture=True,
        )
        if not args.dry_run and expected_sha256 not in finalized.stdout:
            raise RuntimeError(
                f"{node['ssh_host']} did not confirm the installed artifact checksum"
            )
        plan["action"] = "installed"

    node_records: list[dict[str, Any]] = []
    for plan in plans:
        verified = run_on_node(
            plan["node"],
            artifact_probe_command(
                plan["storage_root"], plan["destination"], expected_sha256
            ),
            args,
            capture=True,
            check=False,
        )
        if not args.dry_run and (
            verified.returncode != 0
            or verified.stdout.strip() != f"match {expected_sha256}"
        ):
            detail = (verified.stderr or verified.stdout or "verification failed").strip()
            raise RuntimeError(
                f"{plan['node']['ssh_host']} failed final artifact verification: {detail}"
            )
        node_records.append(
            {
                "role": plan["role"],
                "ssh_host": plan["node"]["ssh_host"],
                "destination_path": str(plan["destination"]),
                "sha256": expected_sha256,
                "action": plan["action"],
            }
        )

    base_entry["nodes"] = node_records
    manifest_data = {
        **stored,
        manifest_key: list(stored.get(manifest_key) or []),
    }
    upsert_artifact_manifest(manifest_data, base_entry)
    if not args.dry_run:
        write_manifest(topology, run_id, manifest_data)

    if args.restart:
        ordered_nodes = artifact_restart_nodes(topology, args.restart)
        try:
            restart_started_at = int(time.time()) - 1
            if not args.dry_run:
                validate_collection_policy(
                    topology, collection, generation, expected_sha256, artifact
                )
                for node in ordered_nodes:
                    inspected = inspect_container(
                        node, container_name(run_id, str(node["role"])), args
                    )
                    verify_run_container_identity(inspected, node, run_id)
            for node in ordered_nodes:
                run_on_node(
                    node,
                    [
                        "sudo",
                        "-n",
                        "docker",
                        "restart",
                        "--time",
                        "30",
                        container_name(run_id, str(node["role"])),
                    ],
                    args,
                )
                if not args.dry_run:
                    wait_http_ready(http_url(node, topology), args.wait_timeout)
            cluster = None if args.dry_run else wait_cluster_healthy(
                topology, args.wait_timeout
            )
            if not args.dry_run:
                collection_proof = validate_collection_policy(
                    topology, collection, generation, expected_sha256, artifact
                )
                router_log_proof = [
                    verify_router_loaded_logs(
                        node,
                        run_id,
                        collection,
                        generation,
                        restart_started_at,
                        args,
                    )
                    for node in all_nodes(topology)
                ]
            else:
                collection_proof = None
                router_log_proof = []
            base_entry["activation"] = {
                "status": (
                    "dry_run_restart_planned"
                    if args.dry_run
                    else "activated_after_restart"
                ),
                "restart_order": args.restart,
            }
            if not args.dry_run:
                base_entry["activation"]["activated_at"] = time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                )
            if cluster is not None:
                base_entry["activation"]["cluster_snapshot"] = cluster
            if collection_proof is not None:
                base_entry["activation"]["collection_proof"] = collection_proof
                base_entry["activation"]["router_log_proof"] = router_log_proof
        except Exception as exc:
            base_entry["activation"] = {
                "status": "restart_failed",
                "restart_order": args.restart,
                "error": str(exc),
            }
            if not args.dry_run:
                upsert_artifact_manifest(manifest_data, base_entry)
                write_manifest(topology, run_id, manifest_data)
            raise
        if not args.dry_run:
            upsert_artifact_manifest(manifest_data, base_entry)
            write_manifest(topology, run_id, manifest_data)

    summary = {
        "run_id": run_id,
        "policy_kind": policy_kind,
        "collection": collection,
        "generation": generation,
        "sha256": expected_sha256,
        "nodes": node_records,
        "activation": base_entry["activation"],
        "manifest": str(manifest_path(topology, run_id)),
    }
    print(json.dumps(summary, indent=2), flush=True)
    return 0


def command_install_orion_artifact(
    args: argparse.Namespace, topology: dict[str, Any]
) -> int:
    return command_install_routing_artifact(
        args,
        topology,
        policy_kind="orion",
        policy_label="Orion",
        manifest_key="orion_artifacts",
        staging_directory=".orion-artifact-staging",
        validate_local_artifact=validate_local_orion_artifact,
        validate_collection_policy=validate_collection_orion_policy,
        artifact_destination=orion_artifact_destination,
        upsert_artifact_manifest=upsert_orion_artifact_manifest,
        verify_router_loaded_logs=verify_orion_router_loaded_logs,
    )


def command_install_simple_kmeans_artifact(
    args: argparse.Namespace, topology: dict[str, Any]
) -> int:
    return command_install_routing_artifact(
        args,
        topology,
        policy_kind="simple_kmeans",
        policy_label="Simple KMeans",
        manifest_key="simple_kmeans_artifacts",
        staging_directory=".simple-kmeans-artifact-staging",
        validate_local_artifact=validate_local_simple_kmeans_artifact,
        validate_collection_policy=validate_collection_simple_kmeans_policy,
        artifact_destination=simple_kmeans_artifact_destination,
        upsert_artifact_manifest=upsert_simple_kmeans_artifact_manifest,
        verify_router_loaded_logs=verify_simple_kmeans_router_loaded_logs,
    )


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
    requested_shards_per_rpc = normalize_peer_premerge_shards_per_rpc(
        getattr(args, "peer_premerge_shards_per_rpc", "all")
    )
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
                "peer_premerge_shards_per_rpc": inspected_peer_premerge_shards_per_rpc(
                    inspected, node
                ),
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
    controller_shards_per_rpc = next(
        (
            node.get("peer_premerge_shards_per_rpc")
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
        requested_shards_per_rpc,
        controller_shards_per_rpc,
    )
    preserve_run_manifest_metadata(data, stored)
    if isinstance(stored.get("repository"), dict) and stored["repository"].get(
        "commit"
    ):
        data["repository"] = json.loads(json.dumps(stored["repository"]))
        data["tooling_repository"] = {
            "path": str(repo),
            **state,
        }
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
        "set-peer-premerge": command_set_peer_premerge,
        "install-orion-artifact": command_install_orion_artifact,
        "install-simple-kmeans-artifact": command_install_simple_kmeans_artifact,
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
