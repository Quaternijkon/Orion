from __future__ import annotations

import copy
import importlib.util
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def load_module():
    path = REPO_ROOT / "tools/method4_distributed_cluster.py"
    spec = importlib.util.spec_from_file_location("method4_distributed_cluster", path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def topology(module):
    return module.load_topology(
        REPO_ROOT / "tools/distributed/cloudlab_orion_4node.json"
    )


def isolated_topology(module, tmp_path):
    value = copy.deepcopy(topology(module))
    value["shared_root"] = str(tmp_path / "shared")
    value["local_storage_root"] = str(tmp_path / "local")
    value["dataset"]["path"] = str(tmp_path / "missing-dataset.hdf5")
    return value


def healthy_cluster_snapshot(module, value):
    return {
        "result": {
            "peer_id": 100,
            "raft_info": {"pending_operations": 0},
            "consensus_thread_status": {
                "consensus_thread_status": "working"
            },
            "message_send_failures": {},
            "peers": {
                str(100 + index): {"uri": module.advertised_uri(node, value)}
                for index, node in enumerate(module.all_nodes(value))
            },
        }
    }


def healthy_run_collections():
    return {
        "dist_run-1_orion_s3": {
            "info": {
                "status": "green",
                "optimizer_status": "ok",
                "points_count": 300,
                "indexed_vectors_count": 300,
                "update_queue": {"length": 0},
                "config": {
                    "params": {
                        "shard_number": 3,
                        "replication_factor": 1,
                    }
                },
            },
            "cluster": {
                "peer_id": 100,
                "local_shards": [],
                "remote_shards": [
                    {"shard_id": index, "peer_id": 101 + index, "state": "Active"}
                    for index in range(3)
                ],
                "shard_transfers": [],
            },
        }
    }


def option_values(command, option):
    return [
        command[index + 1]
        for index, item in enumerate(command[:-1])
        if item == option
    ]


def matching_node_inspect(
    module,
    value,
    node,
    disable_peer_premerge=False,
    shards_per_rpc="all",
    container_id=None,
    image_tag="image:test",
    image_id="sha256:abc",
    running=True,
    wire_version="1",
    wire_max_version="2",
):
    mode = module.expected_peer_premerge_mode(node, disable_peer_premerge)
    normalized_shards_per_rpc = (
        module.normalize_peer_premerge_shards_per_rpc(shards_per_rpc)
        if node["role"] == "controller"
        else "all"
    )
    normalized_wire_version = module.normalize_orion_compact_wire_version(
        wire_version
    )
    labels = {
        "orion.distributed.run_id": "run-1",
        "orion.distributed.role": node["role"],
        "orion.distributed.private_ip": node["private_ip"],
        "orion.distributed.image_id": image_id,
        module.NOFILE_LABEL: module.expected_nofile_label(),
    }
    if wire_max_version is not None:
        labels[module.ORION_COMPACT_WIRE_MAX_VERSION_LABEL] = str(
            wire_max_version
        )
    if node["role"] == "controller":
        labels.update(
            {
                module.PEER_PREMERGE_MODE_LABEL: mode,
                module.CONTROLLER_FINGERPRINT_LABEL: (
                    module.controller_config_fingerprint(
                        node,
                        "run-1",
                        image_id,
                        disable_peer_premerge,
                        normalized_shards_per_rpc,
                        normalized_wire_version,
                    )
                ),
            }
        )
        if normalized_shards_per_rpc != "all":
            labels[module.PEER_PREMERGE_SHARDS_PER_RPC_LABEL] = (
                normalized_shards_per_rpc
            )
        if normalized_wire_version == "2":
            labels[module.ORION_COMPACT_WIRE_VERSION_LABEL] = "2"
    env = [
        "QDRANT__CLUSTER__ENABLED=true",
        f"QDRANT__CLUSTER__P2P__PORT={value['ports']['p2p']}",
        f"QDRANT__SERVICE__HTTP_PORT={value['ports']['http']}",
        f"QDRANT__SERVICE__GRPC_PORT={value['ports']['grpc']}",
        (
            "QDRANT__STORAGE__PERFORMANCE__MAX_SEARCH_THREADS="
            f"{node['max_search_threads']}"
        ),
        (
            "QDRANT__STORAGE__PERFORMANCE__OPTIMIZER_CPU_BUDGET="
            f"{node['optimizer_cpu_budget']}"
        ),
    ]
    if node["role"] == "controller" and disable_peer_premerge:
        env.append(f"{module.PEER_PREMERGE_DISABLE_ENV}=1")
    if node["role"] == "controller" and normalized_shards_per_rpc != "all":
        env.append(
            f"{module.PEER_PREMERGE_SHARDS_PER_RPC_ENV}="
            f"{normalized_shards_per_rpc}"
        )
    if node["role"] == "controller" and normalized_wire_version == "2":
        env.append(f"{module.ORION_COMPACT_WIRE_VERSION_ENV}=2")
    return {
        "Id": container_id or f"container-{node['role']}",
        "Name": f"/{module.container_name('run-1', node['role'])}",
        "Image": image_id,
        "Config": {
            "Image": image_tag,
            "Labels": labels,
            "Env": env,
            "Cmd": module.expected_qdrant_command(value, node),
        },
        "HostConfig": {
            "CpusetCpus": node["cpuset"],
            "NetworkMode": "host",
            "RestartPolicy": {"Name": "unless-stopped"},
            "Ulimits": [
                {
                    "Name": "nofile",
                    "Soft": module.CONTAINER_NOFILE_SOFT,
                    "Hard": module.CONTAINER_NOFILE_HARD,
                }
            ],
        },
        "Mounts": [
            {
                "Type": "bind",
                "Source": str(
                    (
                        module.local_role_root(value, "run-1", node["role"])
                        / "storage"
                    ).resolve()
                ),
                "Destination": "/qdrant/storage",
                "RW": True,
            }
        ],
        "State": {"Running": running},
    }


def matching_controller_inspect(
    module,
    value,
    disable_peer_premerge,
    shards_per_rpc="all",
    wire_version="1",
):
    return matching_node_inspect(
        module,
        value,
        value["controller"],
        disable_peer_premerge,
        shards_per_rpc,
        wire_version=wire_version,
    )


def peer_premerge_manifest(
    value,
    mode="enabled",
    shards_per_rpc=None,
    wire_version="1",
):
    peer_premerge = {
        "requested_mode": mode,
        "current_mode": mode,
        "matches_requested": True,
    }
    if shards_per_rpc is not None:
        normalized_shards_per_rpc = str(shards_per_rpc)
        peer_premerge.update(
            {
                "requested_shards_per_rpc": normalized_shards_per_rpc,
                "current_shards_per_rpc": normalized_shards_per_rpc,
                "shards_per_rpc_matches_requested": True,
            }
        )
    return {
        "schema_version": 1,
        "run_id": "run-1",
        "image": {
            "tag": "image:test",
            "id": "sha256:abc",
            "tar_path": "/shared/image.tar",
            "tar_sha256": "a" * 64,
            "capabilities": {
                "orion_compact_wire_max_version": "2",
            },
        },
        "repository": {"commit": "commit-1"},
        "topology": copy.deepcopy(value),
        "nodes": [
            {
                **copy.deepcopy(node),
                "orion_compact_wire_version": (
                    str(wire_version)
                    if node["role"] == "controller"
                    else "not_applicable"
                ),
            }
            for node in [value["controller"], *value["workers"]]
        ],
        "peer_premerge": peer_premerge,
        "orion_compact_wire": {
            "requested_version": str(
                value["controller"].get("orion_compact_wire_version", 1)
            ),
            "current_version": str(wire_version),
            "matches_requested": str(wire_version)
            == str(value["controller"].get("orion_compact_wire_version", 1)),
        },
        "orion_artifacts": [{"collection": "orion", "generation": 7}],
        "simple_kmeans_artifacts": [{"collection": "simple", "generation": 9}],
    }


def image_transition_manifest(
    module, value, image_id="sha256:abc", wire_version="1"
):
    stored = peer_premerge_manifest(
        value, "enabled", "all", wire_version=wire_version
    )
    archive = module.shared_run_dir(value, "run-1") / "active-image.tar"
    archive.parent.mkdir(parents=True, exist_ok=True)
    archive.write_bytes(b"active image archive")
    stored["image"].update(
        {
            "id": image_id,
            "tar_path": str(archive),
            "tar_size_bytes": archive.stat().st_size,
            "tar_sha256": module.sha256_file(archive),
        }
    )
    stored["image_transitions"] = []
    return stored


def write_image_transition_candidate(
    module,
    value,
    *,
    image_tag="image:candidate",
    image_id="sha256:def",
    archive_bytes=b"candidate image archive",
    dirty=False,
    wire_version=None,
):
    if wire_version is None:
        wire_version = module.requested_orion_compact_wire_version(value)
    wire_version = module.normalize_orion_compact_wire_version(wire_version)
    fingerprint = "b" * 64
    tar_path, manifest_path = module.image_candidate_paths(
        value, "run-1", image_tag, wire_version
    )
    tar_path.parent.mkdir(parents=True, exist_ok=True)
    tar_path.write_bytes(archive_bytes)
    payload = {
        "schema_version": module.IMAGE_CANDIDATE_SCHEMA_VERSION,
        "kind": "orion-image-transition-candidate",
        "run_id": "run-1",
        "orion_compact_wire_version": wire_version,
        "generated_at": "2026-07-21T00:00:00Z",
        "topology_runtime_identity": module.topology_runtime_identity(value),
        "source_fingerprint": fingerprint,
        "repository": {
            "path": str(REPO_ROOT),
            "commit": "commit-2",
            "source_fingerprint": fingerprint,
            "image_affecting_dirty_paths": ["src/main.rs"] if dirty else [],
        },
        "image": {
            "tag": image_tag,
            "id": image_id,
            "tar_path": str(tar_path),
            "tar_size_bytes": tar_path.stat().st_size,
            "tar_sha256": module.sha256_file(tar_path),
            "capabilities": {
                "orion_compact_wire_max_version": (
                    module.CURRENT_ORION_COMPACT_WIRE_MAX_VERSION
                )
            },
        },
    }
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    return manifest_path, payload


def transition_image_args(candidate_manifest, *, dry_run=False):
    return SimpleNamespace(
        run_id="run-1",
        candidate_manifest=str(candidate_manifest),
        expected_current_image_id="sha256:abc",
        strategy="offline",
        wait_timeout=10.0,
        image_tag=None,
        expected_commit="commit-1",
        repo=str(REPO_ROOT),
        dry_run=dry_run,
        ssh_user=None,
        ssh_option=[],
    )


def install_transition_runtime_fakes(
    monkeypatch,
    module,
    value,
    stored,
    candidate,
    *,
    collection_values=None,
    wait_values=None,
    inspect_mutator=None,
):
    snapshot = healthy_cluster_snapshot(module, value)
    roles = [str(node["role"]) for node in module.all_nodes(value)]
    active_wire_version = module.manifest_current_orion_compact_wire_version(stored)
    state = {
        role: {
            "exists": True,
            "running": True,
            "image_tag": stored["image"]["tag"],
            "image_id": stored["image"]["id"],
            "generation": 0,
            "wire_version": active_wire_version,
        }
        for role in roles
    }
    remote_images = {
        role: {stored["image"]["tag"]: stored["image"]["id"]}
        for role in roles
    }
    commands = []
    writes = []
    collections = list(collection_values or [healthy_run_collections()])
    waits = list(wait_values or [snapshot])

    def fake_inspect(node, _name, _args):
        role = str(node["role"])
        current = state[role]
        if not current["exists"]:
            return None
        inspected = matching_node_inspect(
            module,
            value,
            node,
            False,
            "all",
            container_id=f"container-{role}-{current['generation']}",
            image_tag=current["image_tag"],
            image_id=current["image_id"],
            running=current["running"],
            wire_version=current["wire_version"],
        )
        if inspect_mutator is not None:
            inspect_mutator(role, inspected)
        return inspected

    def fake_run_on_node(node, command, _args, **_kwargs):
        role = str(node["role"])
        commands.append((role, command))
        if isinstance(command, list) and command[3:5] == ["image", "inspect"]:
            image_tag = command[5]
            image_id = remote_images[role].get(image_tag, "")
            return subprocess.CompletedProcess(
                command, 0 if image_id else 1, f"{image_id}\n" if image_id else "", ""
            )
        if isinstance(command, list) and command[3:4] == ["load"]:
            archive = str(command[-1])
            if archive == str(candidate["image"]["tar_path"]):
                remote_images[role][candidate["image"]["tag"]] = candidate["image"][
                    "id"
                ]
            elif archive == str(stored["image"]["tar_path"]):
                remote_images[role][stored["image"]["tag"]] = stored["image"]["id"]
            return subprocess.CompletedProcess(command, 0, "", "")
        if isinstance(command, list) and command[3:4] == ["stop"]:
            state[role]["running"] = False
        elif isinstance(command, list) and command[3:4] == ["rm"]:
            state[role]["exists"] = False
        elif isinstance(command, list) and command[3:4] == ["run"]:
            image_tag = command[command.index("./qdrant") - 1]
            image_labels = [
                item.partition("=")[2]
                for item in option_values(command, "--label")
                if item.startswith("orion.distributed.image_id=")
            ]
            assert len(image_labels) == 1
            wire_values = [
                item.partition("=")[2]
                for item in option_values(command, "-e")
                if item.startswith(f"{module.ORION_COMPACT_WIRE_VERSION_ENV}=")
            ]
            wire_version = "2" if wire_values == ["2"] else "1"
            state[role].update(
                {
                    "exists": True,
                    "running": True,
                    "image_tag": image_tag,
                    "image_id": image_labels[0],
                    "generation": state[role]["generation"] + 1,
                    "wire_version": wire_version,
                }
            )
        return subprocess.CompletedProcess(command, 0, "", "")

    def fake_collections(*_args):
        if len(collections) > 1:
            return copy.deepcopy(collections.pop(0))
        return copy.deepcopy(collections[0])

    def fake_wait(*_args):
        if len(waits) > 1:
            value_or_error = waits.pop(0)
        else:
            value_or_error = waits[0]
        if isinstance(value_or_error, Exception):
            raise value_or_error
        return copy.deepcopy(value_or_error)

    monkeypatch.setattr(module, "read_manifest", lambda *_args: copy.deepcopy(stored))
    monkeypatch.setattr(module, "inspect_container", fake_inspect)
    monkeypatch.setattr(module, "run_on_node", fake_run_on_node)
    monkeypatch.setattr(module, "cluster_snapshot", lambda _topology: snapshot)
    monkeypatch.setattr(module, "run_collection_placements", fake_collections)
    monkeypatch.setattr(module, "wait_controller_and_cluster_healthy", fake_wait)
    monkeypatch.setattr(
        module,
        "write_manifest",
        lambda _topology, _run_id, data: writes.append(copy.deepcopy(data))
        or Path(value["shared_root"]) / "manifest.json",
    )
    return state, remote_images, commands, writes


def write_orion_artifact(module, tmp_path, generation=7):
    payload = {
        "format_version": 1,
        "generation": generation,
        "vector_schema": {
            "vector_name": "",
            "dimension": 2,
            "distance": "Cosine",
            "datatype": "float32",
        },
        "shard_count": 4,
        "layout_sha256": "b" * 64,
        "logical_point_count": 1,
        "physical_point_count": 1,
        "upper_k": 1,
        "upper_ef_search": 8,
        "dynamic_ef_base": 20,
        "dynamic_ef_factor": 4,
        "upper_nodes": [
            {"label": 1, "vector": [1.0, 0.0], "shard_membership": [0]}
        ],
        "upper_graph": {
            "entry_point": 1,
            "max_level": 0,
            "nodes": [{"label": 1, "neighbors_by_level": [[]]}],
        },
    }
    artifact = tmp_path / "artifact.json"
    artifact.write_text(
        json.dumps(payload, separators=(",", ":")), encoding="utf-8"
    )
    return artifact, module.sha256_file(artifact)


def write_simple_kmeans_artifact(module, tmp_path, generation=7):
    payload = {
        "format_version": 1,
        "generation": generation,
        "vector_schema": {
            "vector_name": "",
            "dimension": 2,
            "distance": "Cosine",
            "datatype": "float32",
        },
        "shard_count": 4,
        "layout_sha256": "c" * 64,
        "logical_point_count": 4,
        "physical_point_count": 4,
        "routing_distance": "squared_l2",
        "nprobe": 2,
        "lower_hnsw_ef": 48,
        "centroids": [
            {"shard_id": shard_id, "vector": [float(shard_id), 1.0]}
            for shard_id in range(4)
        ],
    }
    artifact = tmp_path / "simple-kmeans-artifact.json"
    artifact.write_text(
        json.dumps(payload, separators=(",", ":")), encoding="utf-8"
    )
    return artifact, module.sha256_file(artifact)


def test_cloudlab_topology_has_one_controller_three_unique_workers():
    module = load_module()
    value = topology(module)

    nodes = module.all_nodes(value)

    assert len(nodes) == 4
    assert nodes[0]["role"] == "controller"
    assert len({node["role"] for node in nodes}) == 4
    assert len({node["private_ip"] for node in nodes}) == 4
    assert [node["private_ip"] for node in value["workers"]] == [
        "10.10.1.2",
        "10.10.1.3",
        "10.10.1.4",
    ]
    assert value["controller"]["orion_compact_wire_version"] == 2
    assert all(
        "orion_compact_wire_version" not in node for node in value["workers"]
    )


@pytest.mark.parametrize(
    ("value", "expected"),
    [(None, "1"), (1, "1"), ("1", "1"), (2, "2"), ("2", "2")],
)
def test_orion_compact_wire_version_normalization(value, expected):
    module = load_module()

    assert module.normalize_orion_compact_wire_version(value) == expected


@pytest.mark.parametrize("value", ["", 0, 3, "02", True, False, "v2"])
def test_orion_compact_wire_version_rejects_unsupported_values(value):
    module = load_module()

    with pytest.raises(ValueError, match="wire version"):
        module.normalize_orion_compact_wire_version(value)


def test_topology_rejects_worker_compact_wire_metadata():
    module = load_module()
    value = topology(module)
    value["workers"][0]["orion_compact_wire_version"] = 2

    with pytest.raises(ValueError, match="controller-only"):
        module.validate_topology(value)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, "all"),
        ("all", "all"),
        ("ALL", "all"),
        (0, "all"),
        ("0", "all"),
        ("00", "all"),
        (1, "1"),
        ("4", "4"),
        ("008", "8"),
        (2**63 - 1, str(2**63 - 1)),
    ],
)
def test_peer_premerge_shards_per_rpc_normalization(value, expected):
    module = load_module()

    assert module.normalize_peer_premerge_shards_per_rpc(value) == expected


@pytest.mark.parametrize(
    "value",
    ["", "-1", -1, "1.0", 1.0, "many", True, False, 2**63],
)
def test_peer_premerge_shards_per_rpc_rejects_invalid_values(value):
    module = load_module()

    with pytest.raises(ValueError, match="shards-per-rpc"):
        module.normalize_peer_premerge_shards_per_rpc(value)


def test_qdrant_commands_advertise_private_uri_and_workers_bootstrap_controller():
    module = load_module()
    value = topology(module)

    controller = module.docker_run_command(
        value, value["controller"], "run-1", "image:test", "sha256:abc"
    )
    worker = module.docker_run_command(
        value, value["workers"][0], "run-1", "image:test", "sha256:abc"
    )

    assert "--network" in controller and "host" in controller
    assert "--cpuset-cpus" in controller and "0-7" in controller
    assert option_values(controller, "--ulimit") == ["nofile=65536:65536"]
    assert option_values(worker, "--ulimit") == ["nofile=65536:65536"]
    assert f"{module.NOFILE_LABEL}=65536:65536" in option_values(
        controller, "--label"
    )
    assert "--bootstrap" not in controller
    assert controller[-2:] == ["--uri", "http://10.10.1.1:6335"]
    assert worker[-4:] == [
        "--bootstrap",
        "http://10.10.1.1:6335",
        "--uri",
        "http://10.10.1.2:6335",
    ]
    assert "0-19" in worker


def test_peer_premerge_is_enabled_by_default_and_disabled_only_on_controller():
    module = load_module()
    value = topology(module)

    enabled = module.docker_run_command(
        value, value["controller"], "run-1", "image:test", "sha256:abc"
    )
    disabled = module.docker_run_command(
        value,
        value["controller"],
        "run-1",
        "image:test",
        "sha256:abc",
        True,
    )
    disabled_worker = module.docker_run_command(
        value,
        value["workers"][0],
        "run-1",
        "image:test",
        "sha256:abc",
        True,
    )

    enabled_env = option_values(enabled, "-e")
    disabled_env = option_values(disabled, "-e")
    worker_env = option_values(disabled_worker, "-e")
    enabled_labels = option_values(enabled, "--label")
    disabled_labels = option_values(disabled, "--label")

    assert f"{module.PEER_PREMERGE_DISABLE_ENV}=1" not in enabled_env
    assert f"{module.PEER_PREMERGE_DISABLE_ENV}=1" in disabled_env
    assert f"{module.PEER_PREMERGE_DISABLE_ENV}=1" not in worker_env
    assert f"{module.PEER_PREMERGE_MODE_LABEL}=enabled" in enabled_labels
    assert f"{module.PEER_PREMERGE_MODE_LABEL}=disabled" in disabled_labels
    assert not any(
        label.startswith(f"{module.PEER_PREMERGE_MODE_LABEL}=")
        for label in option_values(disabled_worker, "--label")
    )
    enabled_fingerprint = next(
        label
        for label in enabled_labels
        if label.startswith(module.CONTROLLER_FINGERPRINT_LABEL)
    )
    disabled_fingerprint = next(
        label
        for label in disabled_labels
        if label.startswith(module.CONTROLLER_FINGERPRINT_LABEL)
    )
    assert enabled_fingerprint != disabled_fingerprint


def test_controller_fingerprint_preserves_v3_all_compatibility():
    module = load_module()
    value = topology(module)
    controller = value["controller"]
    legacy_enabled = (
        "sha256:a5bc3071b4e5c222d873ab914743776ac1c9d1da7b900c7f8a89a9ae590fc9d7"
    )
    legacy_disabled = (
        "sha256:b12bc0bface87b51f3c019247ec7d737aeeb89dfe25ba4883c7351d40e57a3c0"
    )

    assert module.controller_config_fingerprint(
        controller, "run-1", "sha256:abc"
    ) == legacy_enabled
    assert module.controller_config_fingerprint(
        controller, "run-1", "sha256:abc", False, "all"
    ) == legacy_enabled
    assert module.controller_config_fingerprint(
        controller, "run-1", "sha256:abc", False, 0
    ) == legacy_enabled
    assert module.controller_config_fingerprint(
        controller, "run-1", "sha256:abc", True, "all"
    ) == legacy_disabled
    assert module.controller_config_fingerprint(
        controller, "run-1", "sha256:abc", False, 4
    ) != legacy_enabled
    assert module.controller_config_fingerprint(
        controller, "run-1", "sha256:abc", False, "all", "2"
    ) != legacy_enabled


def test_compact_wire_v1_is_implicit_and_v2_is_controller_only():
    module = load_module()
    value = topology(module)
    controller_v1 = module.docker_run_command(
        value,
        value["controller"],
        "run-1",
        "image:test",
        "sha256:abc",
        False,
        "all",
        "1",
    )
    controller_v2 = module.docker_run_command(
        value,
        value["controller"],
        "run-1",
        "image:test",
        "sha256:abc",
        False,
        "all",
        "2",
    )
    worker_v2 = module.docker_run_command(
        value,
        value["workers"][0],
        "run-1",
        "image:test",
        "sha256:abc",
        False,
        "all",
        "2",
    )
    prefix_env = f"{module.ORION_COMPACT_WIRE_VERSION_ENV}="
    prefix_label = f"{module.ORION_COMPACT_WIRE_VERSION_LABEL}="

    assert not any(
        item.startswith(prefix_env) for item in option_values(controller_v1, "-e")
    )
    assert not any(
        item.startswith(prefix_label)
        for item in option_values(controller_v1, "--label")
    )
    assert f"{prefix_env}2" in option_values(controller_v2, "-e")
    assert f"{prefix_label}2" in option_values(controller_v2, "--label")
    assert not any(
        item.startswith(prefix_env) for item in option_values(worker_v2, "-e")
    )
    assert not any(
        item.startswith(prefix_label)
        for item in option_values(worker_v2, "--label")
    )


def test_compact_wire_inspection_and_reuse_are_fail_closed():
    module = load_module()
    value = topology(module)
    controller = value["controller"]
    worker = value["workers"][0]
    v1 = matching_controller_inspect(module, value, False, wire_version="1")
    v2 = matching_controller_inspect(module, value, False, wire_version="2")

    assert module.inspected_orion_compact_wire_version(v1, controller) == "1"
    assert module.inspected_orion_compact_wire_version(v2, controller) == "2"
    assert module.verify_container_reusable(
        v2, controller, "run-1", "sha256:abc", False, "all", "2"
    )
    with pytest.raises(RuntimeError, match="compact_wire"):
        module.verify_container_reusable(
            v1, controller, "run-1", "sha256:abc", False, "all", "2"
        )

    malformed = []
    missing_env = copy.deepcopy(v2)
    missing_env["Config"]["Env"] = [
        item
        for item in missing_env["Config"]["Env"]
        if not item.startswith(f"{module.ORION_COMPACT_WIRE_VERSION_ENV}=")
    ]
    malformed.append(missing_env)
    missing_label = copy.deepcopy(v2)
    del missing_label["Config"]["Labels"][module.ORION_COMPACT_WIRE_VERSION_LABEL]
    malformed.append(missing_label)
    mismatched = copy.deepcopy(v2)
    mismatched["Config"]["Env"][-1] = (
        f"{module.ORION_COMPACT_WIRE_VERSION_ENV}=1"
    )
    malformed.append(mismatched)
    duplicate = copy.deepcopy(v2)
    duplicate["Config"]["Env"].append(
        f"{module.ORION_COMPACT_WIRE_VERSION_ENV}=2"
    )
    malformed.append(duplicate)
    invalid = copy.deepcopy(v2)
    invalid["Config"]["Labels"][module.ORION_COMPACT_WIRE_VERSION_LABEL] = "3"
    malformed.append(invalid)
    for inspected in malformed:
        assert (
            module.inspected_orion_compact_wire_version(inspected, controller)
            == "inconsistent"
        )

    worker_leak = matching_node_inspect(module, value, worker)
    worker_leak["Config"]["Env"].append(
        f"{module.ORION_COMPACT_WIRE_VERSION_ENV}=2"
    )
    assert (
        module.inspected_orion_compact_wire_version(worker_leak, worker)
        == "inconsistent"
    )
    with pytest.raises(RuntimeError, match="compact_wire"):
        module.verify_container_reusable(
            worker_leak, worker, "run-1", "sha256:abc", False, "all", "2"
        )


def test_compact_wire_v2_requires_image_bound_capability_on_every_role():
    module = load_module()
    value = topology(module)
    for node in module.all_nodes(value):
        missing = matching_node_inspect(
            module,
            value,
            node,
            wire_version="2",
            wire_max_version=None,
        )
        with pytest.raises(RuntimeError, match="max_version"):
            module.verify_container_reusable(
                missing,
                node,
                "run-1",
                "sha256:abc",
                False,
                "all",
                "2",
            )
        implicit_v1 = matching_node_inspect(
            module,
            value,
            node,
            wire_version="1",
            wire_max_version=None,
        )
        assert module.verify_container_reusable(
            implicit_v1,
            node,
            "run-1",
            "sha256:abc",
            False,
            "all",
            "1",
        )


def test_docker_image_declares_compact_wire_v2_capability():
    module = load_module()
    dockerfile = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert (
        f'LABEL {module.ORION_COMPACT_WIRE_MAX_VERSION_LABEL}="2"'
        in dockerfile
    )


@pytest.mark.parametrize(
    ("labels", "required", "expected", "error"),
    [
        ({"org.qdrant.orion.compact_wire.max_version": "2"}, "2", "2", None),
        ({}, "1", "1", None),
        (None, "1", "1", None),
        ({}, "2", None, "does not support requested"),
        ({"org.qdrant.orion.compact_wire.max_version": "3"}, "2", None, "invalid"),
    ],
)
def test_local_image_compact_wire_capability_is_verified(
    monkeypatch, labels, required, expected, error
):
    module = load_module()
    monkeypatch.setattr(
        module,
        "run_command",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            [], 0, json.dumps(labels), ""
        ),
    )
    if error is None:
        assert (
            module.image_compact_wire_max_version_local("image:test", required)
            == expected
        )
    else:
        with pytest.raises(RuntimeError, match=error):
            module.image_compact_wire_max_version_local("image:test", required)


def test_peer_premerge_rpc_chunking_is_controller_only_and_all_is_implicit():
    module = load_module()
    value = topology(module)
    controller_all = module.docker_run_command(
        value,
        value["controller"],
        "run-1",
        "image:test",
        "sha256:abc",
        False,
        "all",
    )
    controller_bounded = module.docker_run_command(
        value,
        value["controller"],
        "run-1",
        "image:test",
        "sha256:abc",
        False,
        "008",
    )
    worker_bounded = module.docker_run_command(
        value,
        value["workers"][0],
        "run-1",
        "image:test",
        "sha256:abc",
        False,
        8,
    )

    chunk_env_prefix = f"{module.PEER_PREMERGE_SHARDS_PER_RPC_ENV}="
    chunk_label_prefix = f"{module.PEER_PREMERGE_SHARDS_PER_RPC_LABEL}="
    assert not any(
        item.startswith(chunk_env_prefix)
        for item in option_values(controller_all, "-e")
    )
    assert not any(
        item.startswith(chunk_label_prefix)
        for item in option_values(controller_all, "--label")
    )
    assert f"{chunk_env_prefix}8" in option_values(controller_bounded, "-e")
    assert f"{chunk_label_prefix}8" in option_values(
        controller_bounded, "--label"
    )
    assert not any(
        item.startswith(chunk_env_prefix)
        for item in option_values(worker_bounded, "-e")
    )
    assert not any(
        item.startswith(chunk_label_prefix)
        for item in option_values(worker_bounded, "--label")
    )


@pytest.mark.parametrize("command", ["deploy", "status", "manifest"])
def test_peer_premerge_cli_defaults_enabled_and_accepts_explicit_disable(
    monkeypatch, command
):
    module = load_module()

    monkeypatch.setattr(module.sys, "argv", ["cluster", "--run-id", "run-1", command])
    default_args = module.parse_args()
    assert default_args.disable_peer_premerge is False
    assert default_args.peer_premerge_shards_per_rpc == "all"

    monkeypatch.setattr(
        module.sys,
        "argv",
        [
            "cluster",
            "--run-id",
            "run-1",
            command,
            "--disable-peer-premerge",
            "--peer-premerge-shards-per-rpc",
            "008",
        ],
    )
    explicit_args = module.parse_args()
    assert explicit_args.disable_peer_premerge is True
    assert explicit_args.peer_premerge_shards_per_rpc == "8"

    monkeypatch.setattr(
        module.sys,
        "argv",
        [
            "cluster",
            "--run-id",
            "run-1",
            command,
            "--peer-premerge-shards-per-rpc",
            "0",
        ],
    )
    assert module.parse_args().peer_premerge_shards_per_rpc == "all"


@pytest.mark.parametrize(
    ("options", "expected_mode", "expected_shards_per_rpc"),
    [
        (["--mode", "disabled"], "disabled", None),
        (["--shards-per-rpc", "4"], None, "4"),
        (
            ["--mode", "enabled", "--shards-per-rpc", "0"],
            "enabled",
            "all",
        ),
    ],
)
def test_set_peer_premerge_cli_accepts_mode_chunk_or_both(
    monkeypatch, options, expected_mode, expected_shards_per_rpc
):
    module = load_module()
    monkeypatch.setattr(
        module.sys,
        "argv",
        [
            "cluster",
            "--run-id",
            "run-1",
            "set-peer-premerge",
            *options,
            "--wait-timeout",
            "75",
        ],
    )

    args = module.parse_args()

    assert args.command == "set-peer-premerge"
    assert args.mode == expected_mode
    assert args.shards_per_rpc == expected_shards_per_rpc
    assert args.wait_timeout == 75.0


def test_build_cli_accepts_stage_for_transition(monkeypatch):
    module = load_module()
    monkeypatch.setattr(
        module.sys,
        "argv",
        [
            "cluster",
            "--run-id",
            "run-1",
            "build",
            "--allow-dirty",
            "--stage-for-transition",
        ],
    )

    args = module.parse_args()

    assert args.command == "build"
    assert args.allow_dirty is True
    assert args.stage_for_transition is True


def test_transition_image_cli_accepts_offline_candidate(monkeypatch):
    module = load_module()
    monkeypatch.setattr(
        module.sys,
        "argv",
        [
            "cluster",
            "--run-id",
            "run-1",
            "transition-image",
            "--candidate-manifest",
            "/shared/candidate.json",
            "--expected-current-image-id",
            "sha256:abc",
            "--strategy",
            "offline",
            "--wait-timeout",
            "75",
        ],
    )

    args = module.parse_args()

    assert args.command == "transition-image"
    assert args.candidate_manifest == "/shared/candidate.json"
    assert args.expected_current_image_id == "sha256:abc"
    assert args.strategy == "offline"
    assert args.wait_timeout == 75.0


def test_set_peer_premerge_rejects_invocation_without_mode_or_chunk():
    module = load_module()
    args = SimpleNamespace(run_id="run-1", mode=None, shards_per_rpc=None)

    with pytest.raises(ValueError, match="requires --mode and/or --shards-per-rpc"):
        module.command_set_peer_premerge(args, topology(module))


def test_peer_uri_normalization_accepts_qdrant_trailing_slash():
    module = load_module()

    assert module.normalize_peer_uri("http://10.10.1.2:6335/") == (
        "http://10.10.1.2:6335"
    )
    assert module.normalize_peer_uri("http://10.10.1.2:6335") == (
        "http://10.10.1.2:6335"
    )


def test_remote_command_uses_batch_mode_ssh_and_local_command_does_not():
    module = load_module()
    value = topology(module)

    local = module.wrap_node_command(value["controller"], ["docker", "ps"])
    remote = module.wrap_node_command(value["workers"][0], ["docker", "ps"])

    assert local[:2] == ["bash", "-lc"]
    assert remote[:5] == ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]
    assert "hp052.utah.cloudlab.us" in remote
    assert remote[-3:-1] == ["bash", "-c"]
    assert "-lc" not in remote


def test_clean_is_scoped_to_exact_run_role_path():
    module = load_module()
    value = topology(module)
    node = value["workers"][0]

    commands = module.clean_commands(value, node, "abc-123")

    assert commands[0] == [
        "sudo",
        "-n",
        "docker",
        "rm",
        "-f",
        "orion-dist-abc-123-qdrant_shard_1",
    ]
    assert commands[1] == [
        "rm",
        "-rf",
        "--",
        "/users/dry/orion-distributed/abc-123/qdrant_shard_1",
    ]


@pytest.mark.parametrize("run_id", ["../bad", "/tmp/bad", "bad id", "", "a" * 81])
def test_run_id_rejects_path_traversal_and_unbounded_names(run_id):
    module = load_module()

    with pytest.raises(ValueError):
        module.validate_run_id(run_id)


def test_existing_container_with_wrong_image_or_cpuset_is_rejected():
    module = load_module()
    value = topology(module)
    node = value["workers"][0]
    inspected = {
        "Name": "/orion-dist-run-qdrant_shard_1",
        "Image": "sha256:wrong",
        "Config": {
            "Labels": {
                "orion.distributed.run_id": "run",
                "orion.distributed.role": node["role"],
                "orion.distributed.private_ip": node["private_ip"],
                "orion.distributed.image_id": "sha256:wrong",
            }
        },
        "HostConfig": {"CpusetCpus": "0-7"},
        "State": {"Running": True},
    }

    with pytest.raises(RuntimeError, match="incompatible configuration"):
        module.verify_container_reusable(inspected, node, "run", "sha256:expected")


def test_existing_container_without_expected_nofile_limit_is_rejected():
    module = load_module()
    value = topology(module)
    inspected = matching_controller_inspect(module, value, False)
    inspected["HostConfig"]["Ulimits"] = [
        {"Name": "nofile", "Soft": 1024, "Hard": 524288}
    ]

    with pytest.raises(RuntimeError, match="nofile"):
        module.verify_container_reusable(
            inspected, value["controller"], "run-1", "sha256:abc"
        )


@pytest.mark.parametrize("disable_peer_premerge", [False, True])
def test_matching_controller_peer_premerge_container_is_reusable(disable_peer_premerge):
    module = load_module()
    value = topology(module)
    inspected = matching_controller_inspect(module, value, disable_peer_premerge)

    assert module.verify_container_reusable(
        inspected,
        value["controller"],
        "run-1",
        "sha256:abc",
        disable_peer_premerge,
    )
    assert module.inspected_peer_premerge_mode(inspected, value["controller"]) == (
        "disabled" if disable_peer_premerge else "enabled"
    )


def test_matching_bounded_peer_premerge_container_is_reusable():
    module = load_module()
    value = topology(module)
    inspected = matching_controller_inspect(module, value, False, "008")

    assert module.verify_container_reusable(
        inspected,
        value["controller"],
        "run-1",
        "sha256:abc",
        False,
        8,
    )
    assert module.inspected_peer_premerge_shards_per_rpc(
        inspected, value["controller"]
    ) == "8"


def test_inspected_peer_premerge_shards_per_rpc_is_fail_closed():
    module = load_module()
    value = topology(module)
    controller = value["controller"]
    worker = value["workers"][0]
    implicit_all = matching_controller_inspect(module, value, False, "all")
    bounded = matching_controller_inspect(module, value, False, 4)

    assert module.inspected_peer_premerge_shards_per_rpc(
        implicit_all, controller
    ) == "all"
    assert module.inspected_peer_premerge_shards_per_rpc(bounded, controller) == "4"
    assert module.inspected_peer_premerge_shards_per_rpc(
        matching_node_inspect(module, value, worker, False, 4), worker
    ) == "not_applicable"

    missing_env = copy.deepcopy(bounded)
    missing_env["Config"]["Env"] = [
        item
        for item in missing_env["Config"]["Env"]
        if not item.startswith(f"{module.PEER_PREMERGE_SHARDS_PER_RPC_ENV}=")
    ]
    missing_label = copy.deepcopy(bounded)
    del missing_label["Config"]["Labels"][
        module.PEER_PREMERGE_SHARDS_PER_RPC_LABEL
    ]
    mismatched = copy.deepcopy(bounded)
    mismatched["Config"]["Env"][-1] = (
        f"{module.PEER_PREMERGE_SHARDS_PER_RPC_ENV}=5"
    )
    explicit_all = copy.deepcopy(implicit_all)
    explicit_all["Config"]["Labels"][
        module.PEER_PREMERGE_SHARDS_PER_RPC_LABEL
    ] = "all"
    explicit_all["Config"]["Env"].append(
        f"{module.PEER_PREMERGE_SHARDS_PER_RPC_ENV}=0"
    )
    duplicate_env = copy.deepcopy(bounded)
    duplicate_env["Config"]["Env"].append(
        f"{module.PEER_PREMERGE_SHARDS_PER_RPC_ENV}=4"
    )

    for inspected in (
        missing_env,
        missing_label,
        mismatched,
        explicit_all,
        duplicate_env,
    ):
        assert module.inspected_peer_premerge_shards_per_rpc(
            inspected, controller
        ) == "inconsistent"


def test_controller_reuse_rejects_bounded_chunk_drift():
    module = load_module()
    value = topology(module)
    controller = value["controller"]
    bounded = matching_controller_inspect(module, value, False, 4)

    missing_env = copy.deepcopy(bounded)
    missing_env["Config"]["Env"] = [
        item
        for item in missing_env["Config"]["Env"]
        if not item.startswith(f"{module.PEER_PREMERGE_SHARDS_PER_RPC_ENV}=")
    ]
    missing_label = copy.deepcopy(bounded)
    del missing_label["Config"]["Labels"][
        module.PEER_PREMERGE_SHARDS_PER_RPC_LABEL
    ]
    wrong_env = copy.deepcopy(bounded)
    wrong_env["Config"]["Env"][-1] = (
        f"{module.PEER_PREMERGE_SHARDS_PER_RPC_ENV}=5"
    )
    wrong_label = copy.deepcopy(bounded)
    wrong_label["Config"]["Labels"][
        module.PEER_PREMERGE_SHARDS_PER_RPC_LABEL
    ] = "5"
    wrong_fingerprint = copy.deepcopy(bounded)
    wrong_fingerprint["Config"]["Labels"][module.CONTROLLER_FINGERPRINT_LABEL] = (
        module.controller_config_fingerprint(
            controller, "run-1", "sha256:abc", False, "all"
        )
    )

    for inspected in (
        missing_env,
        missing_label,
        wrong_env,
        wrong_label,
        wrong_fingerprint,
    ):
        with pytest.raises(RuntimeError, match="incompatible configuration"):
            module.verify_container_reusable(
                inspected,
                controller,
                "run-1",
                "sha256:abc",
                False,
                4,
            )


@pytest.mark.parametrize("leak_kind", ["environment", "label"])
def test_worker_reuse_rejects_peer_premerge_chunk_metadata(leak_kind):
    module = load_module()
    value = topology(module)
    worker = value["workers"][0]
    inspected = matching_node_inspect(module, value, worker)
    if leak_kind == "environment":
        inspected["Config"]["Env"].append(
            f"{module.PEER_PREMERGE_SHARDS_PER_RPC_ENV}=4"
        )
    else:
        inspected["Config"]["Labels"][
            module.PEER_PREMERGE_SHARDS_PER_RPC_LABEL
        ] = "4"

    with pytest.raises(RuntimeError, match="incompatible configuration"):
        module.verify_container_reusable(
            inspected, worker, "run-1", "sha256:abc", False, 4
        )


@pytest.mark.parametrize(
    ("container_disabled", "requested_disabled"), [(False, True), (True, False)]
)
def test_controller_reuse_rejects_peer_premerge_mode_mismatch(
    container_disabled, requested_disabled
):
    module = load_module()
    value = topology(module)
    inspected = matching_controller_inspect(module, value, container_disabled)

    with pytest.raises(RuntimeError, match="peer_premerge"):
        module.verify_container_reusable(
            inspected,
            value["controller"],
            "run-1",
            "sha256:abc",
            requested_disabled,
        )


def test_controller_reuse_rejects_missing_fingerprint():
    module = load_module()
    value = topology(module)
    inspected = matching_controller_inspect(module, value, False)
    del inspected["Config"]["Labels"][module.CONTROLLER_FINGERPRINT_LABEL]

    with pytest.raises(RuntimeError, match="controller_fingerprint"):
        module.verify_container_reusable(
            inspected, value["controller"], "run-1", "sha256:abc"
        )


def test_peer_premerge_transition_identity_requires_exact_bind_mount_and_runtime():
    module = load_module()
    value = topology(module)
    inspected = matching_controller_inspect(module, value, False)

    assert module.verify_peer_premerge_transition_identity(
        value,
        inspected,
        value["controller"],
        "run-1",
        "image:test",
        "sha256:abc",
        current_mode="enabled",
    )

    inspected["Mounts"][0]["Source"] = "/tmp/not-this-run"
    with pytest.raises(RuntimeError, match="storage_mount"):
        module.verify_peer_premerge_transition_identity(
            value,
            inspected,
            value["controller"],
            "run-1",
            "image:test",
            "sha256:abc",
            current_mode="enabled",
        )


def test_peer_premerge_transition_manifest_binds_deployment_commit(tmp_path):
    module = load_module()
    value = isolated_topology(module, tmp_path)
    stored = peer_premerge_manifest(value)

    assert module.validate_peer_premerge_transition_manifest(
        value, "run-1", stored, None, "commit-1"
    ) == ("image:test", "sha256:abc", "enabled", "all", "1", "commit-1")
    with pytest.raises(RuntimeError, match="deployment commit"):
        module.validate_peer_premerge_transition_manifest(
            value, "run-1", stored, None, "other-commit"
        )


def test_peer_premerge_transition_manifest_reads_bounded_chunk_setting(tmp_path):
    module = load_module()
    value = isolated_topology(module, tmp_path)
    stored = peer_premerge_manifest(value, "disabled", "008")

    assert module.validate_peer_premerge_transition_manifest(
        value, "run-1", stored, None, "commit-1"
    ) == ("image:test", "sha256:abc", "disabled", "8", "1", "commit-1")

    stored["peer_premerge"]["current_shards_per_rpc"] = "bad"
    with pytest.raises(RuntimeError, match="invalid.*shards-per-rpc"):
        module.validate_peer_premerge_transition_manifest(
            value, "run-1", stored, None, "commit-1"
        )


def test_transition_manifest_cross_checks_top_level_and_node_wire_identity(
    tmp_path,
):
    module = load_module()
    value = isolated_topology(module, tmp_path)
    stored = peer_premerge_manifest(value, wire_version="2")

    assert module.validate_peer_premerge_transition_manifest(
        value, "run-1", stored, None, "commit-1"
    )[4] == "2"

    inconsistent = copy.deepcopy(stored)
    controller = next(
        node for node in inconsistent["nodes"] if node["role"] == "controller"
    )
    controller["orion_compact_wire_version"] = "1"
    with pytest.raises(RuntimeError, match="disagrees between top-level and node"):
        module.validate_peer_premerge_transition_manifest(
            value, "run-1", inconsistent, None, "commit-1"
        )

    partial = copy.deepcopy(stored)
    del partial["orion_compact_wire"]
    with pytest.raises(RuntimeError, match="fully legacy"):
        module.validate_peer_premerge_transition_manifest(
            value, "run-1", partial, None, "commit-1"
        )

    unsupported = copy.deepcopy(stored)
    del unsupported["image"]["capabilities"]
    with pytest.raises(RuntimeError, match="does not support its active"):
        module.validate_peer_premerge_transition_manifest(
            value, "run-1", unsupported, None, "commit-1"
        )

    legacy = copy.deepcopy(stored)
    del legacy["orion_compact_wire"]
    for node in legacy["nodes"]:
        del node["orion_compact_wire_version"]
    assert module.validate_peer_premerge_transition_manifest(
        value, "run-1", legacy, None, "commit-1"
    )[4] == "1"


def test_dirty_candidate_tag_uses_deterministic_image_source_fingerprint(tmp_path):
    module = load_module()
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / "docs").mkdir()
    (repo / "src/main.rs").write_text("fn main() {}\n", encoding="utf-8")
    (repo / "docs/note.md").write_text("baseline\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(
        ["git", "config", "user.email", "orion-test@example.invalid"],
        cwd=repo,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Orion Test"], cwd=repo, check=True
    )
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "baseline"], cwd=repo, check=True)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    (repo / "src/main.rs").write_text("fn main() { work(); }\n", encoding="utf-8")

    first = module.image_source_fingerprint(repo, commit)
    second = module.image_source_fingerprint(repo, commit)
    (repo / "docs/note.md").write_text("ignored documentation edit\n", encoding="utf-8")
    after_docs = module.image_source_fingerprint(repo, commit)
    tag = module.transition_candidate_image_tag(
        module.image_tag_for_commit(commit), commit, first["sha256"], True
    )

    assert first == second == after_docs
    assert first["dirty_paths"] == ["src/main.rs"]
    assert tag.endswith(f"-dirty-{first['sha256'][:12]}")


def test_staged_build_writes_candidate_without_replacing_active_manifest(
    monkeypatch, tmp_path
):
    module = load_module()
    value = isolated_topology(module, tmp_path)
    active_path = module.manifest_path(value, "run-1")
    active_path.parent.mkdir(parents=True)
    active_path.write_text('{"active": true}\n', encoding="utf-8")
    source = {
        "sha256": "c" * 64,
        "tracked_paths": [],
        "untracked_paths": [],
        "dirty_paths": [],
    }
    state = {
        "commit": "commit-1",
        "short_commit": "commit-1",
        "dirty": False,
        "dirty_paths": [],
        "tracked_dirty": False,
        "tracked_dirty_paths": [],
        "untracked_entry_count": 0,
    }
    commands = []

    def fake_run_command(command, **_kwargs):
        commands.append(command)
        if "save" in command:
            output = Path(command[command.index("--output") + 1])
            output.write_bytes(b"candidate archive")
        elif command[2:3] == ["mv"]:
            Path(command[-2]).replace(command[-1])
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(module, "git_state", lambda _repo: copy.deepcopy(state))
    monkeypatch.setattr(
        module,
        "image_source_fingerprint",
        lambda *_args: copy.deepcopy(source),
    )
    monkeypatch.setattr(module, "run_command", fake_run_command)
    monkeypatch.setattr(
        module, "image_id_local", lambda *_args, **_kwargs: "sha256:candidate"
    )
    monkeypatch.setattr(
        module,
        "image_compact_wire_max_version_local",
        lambda *_args, **_kwargs: "2",
    )
    monkeypatch.setattr(
        module,
        "write_manifest",
        lambda *_args: pytest.fail("staged build must not write active manifest"),
    )
    args = SimpleNamespace(
        repo=str(tmp_path),
        expected_commit="commit-1",
        allow_dirty=False,
        image_tag="image:base",
        run_id="run-1",
        force=False,
        dry_run=False,
        stage_for_transition=True,
    )

    assert module.command_build(args, value) == 0

    candidate_tag = f"image:base-source-{source['sha256'][:12]}"
    _tar_path, candidate_path = module.image_candidate_paths(
        value, "run-1", candidate_tag
    )
    candidate = module.validate_image_candidate_manifest(
        value, "run-1", candidate_path
    )
    assert candidate["image_id"] == "sha256:candidate"
    assert candidate["source_fingerprint"] == source["sha256"]
    assert candidate["orion_compact_wire_version"] == "2"
    assert active_path.read_text(encoding="utf-8") == '{"active": true}\n'
    assert any("buildx" in command for command in commands)


@pytest.mark.parametrize("mutation", ["missing", "invalid", "mismatch"])
def test_candidate_manifest_requires_exact_requested_compact_wire_version(
    tmp_path, mutation
):
    module = load_module()
    value = isolated_topology(module, tmp_path)
    candidate_path, payload = write_image_transition_candidate(module, value)
    mutated = copy.deepcopy(payload)
    if mutation == "missing":
        del mutated["orion_compact_wire_version"]
    elif mutation == "invalid":
        mutated["orion_compact_wire_version"] = "3"
    else:
        mutated["orion_compact_wire_version"] = "1"
    candidate_path.write_text(json.dumps(mutated), encoding="utf-8")

    with pytest.raises(RuntimeError, match="compact wire version"):
        module.validate_image_candidate_manifest(value, "run-1", candidate_path)


def test_candidate_manifest_and_archive_paths_are_canonical_for_wire_identity(
    tmp_path,
):
    module = load_module()
    value = isolated_topology(module, tmp_path)
    candidate_path, payload = write_image_transition_candidate(module, value)

    moved_manifest = candidate_path.with_name("copied-candidate.json")
    moved_manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="manifest path is not canonical"):
        module.validate_image_candidate_manifest(value, "run-1", moved_manifest)

    moved_archive = candidate_path.with_name("copied-candidate.tar")
    moved_archive.write_bytes(Path(payload["image"]["tar_path"]).read_bytes())
    mutated = copy.deepcopy(payload)
    mutated["image"]["tar_path"] = str(moved_archive)
    mutated["image"]["tar_sha256"] = module.sha256_file(moved_archive)
    mutated["image"]["tar_size_bytes"] = moved_archive.stat().st_size
    candidate_path.write_text(json.dumps(mutated), encoding="utf-8")
    with pytest.raises(RuntimeError, match="archive path is not canonical"):
        module.validate_image_candidate_manifest(value, "run-1", candidate_path)


def test_candidate_manifest_requires_image_compact_wire_capability(tmp_path):
    module = load_module()
    value = isolated_topology(module, tmp_path)
    candidate_path, payload = write_image_transition_candidate(module, value)
    mutated = copy.deepcopy(payload)
    del mutated["image"]["capabilities"]
    candidate_path.write_text(json.dumps(mutated), encoding="utf-8")

    with pytest.raises(RuntimeError, match="compact-wire capability"):
        module.validate_image_candidate_manifest(value, "run-1", candidate_path)


def test_build_reuses_tar_only_after_manifest_sha_and_tag_id_close_loop(
    monkeypatch, tmp_path
):
    module = load_module()
    value = isolated_topology(module, tmp_path)
    image_tag = "orion-method4:expected"
    image_id = "sha256:expected"
    tar_path = module.shared_run_dir(value, "run-1") / "orion-method4_expected.tar"
    tar_path.parent.mkdir(parents=True)
    tar_path.write_bytes(b"verified archive")
    stored = {
        "image": {
            "tag": image_tag,
            "id": image_id,
            "tar_path": str(tar_path),
            "tar_sha256": module.sha256_file(tar_path),
            "capabilities": {
                "orion_compact_wire_max_version": "2",
            },
        }
    }
    commands = []
    monkeypatch.setattr(
        module,
        "git_state",
        lambda _repo: {"commit": "commit-1", "dirty_paths": []},
    )
    monkeypatch.setattr(module, "read_manifest", lambda *_args: stored)
    monkeypatch.setattr(module, "image_id_local", lambda *_args, **_kwargs: image_id)
    monkeypatch.setattr(
        module,
        "image_compact_wire_max_version_local",
        lambda *_args, **_kwargs: "2",
    )
    monkeypatch.setattr(
        module,
        "run_command",
        lambda command, **_kwargs: commands.append(command)
        or subprocess.CompletedProcess(command, 0, "", ""),
    )
    monkeypatch.setattr(
        module,
        "write_manifest",
        lambda *_args, **_kwargs: pytest.fail("verified reuse must not rewrite manifest"),
    )
    args = SimpleNamespace(
        repo=str(tmp_path),
        expected_commit="commit-1",
        allow_dirty=False,
        image_tag=image_tag,
        run_id="run-1",
        force=False,
        dry_run=False,
    )

    assert module.command_build(args, value) == 0
    assert commands == [["sudo", "-n", "chmod", "0644", str(tar_path)]]


def test_existing_tar_reuse_rejects_manifest_sha_mismatch(tmp_path):
    module = load_module()
    tar_path = tmp_path / "image.tar"
    tar_path.write_bytes(b"actual archive")
    manifest = {
        "image": {
            "tag": "image:test",
            "id": "sha256:expected",
            "tar_path": str(tar_path),
            "tar_sha256": "0" * 64,
        }
    }

    with pytest.raises(RuntimeError, match="tar SHA-256 mismatch"):
        module.validate_reusable_image_archive(
            tar_path, "image:test", manifest, "sha256:expected"
        )


def test_existing_tar_reuse_rejects_current_tag_image_id_mismatch(tmp_path):
    module = load_module()
    tar_path = tmp_path / "image.tar"
    tar_path.write_bytes(b"actual archive")
    manifest = {
        "image": {
            "tag": "image:test",
            "id": "sha256:expected",
            "tar_path": str(tar_path),
            "tar_sha256": module.sha256_file(tar_path),
        }
    }

    with pytest.raises(RuntimeError, match="current tag id"):
        module.validate_reusable_image_archive(
            tar_path, "image:test", manifest, "sha256:wrong"
        )


def test_transition_image_rejects_candidate_archive_mismatch_before_stop(
    monkeypatch, tmp_path
):
    module = load_module()
    value = isolated_topology(module, tmp_path)
    stored = image_transition_manifest(module, value)
    candidate_path, candidate = write_image_transition_candidate(module, value)
    Path(candidate["image"]["tar_path"]).write_bytes(b"tampered archive")
    monkeypatch.setattr(module, "read_manifest", lambda *_args: stored)
    monkeypatch.setattr(
        module,
        "inspect_container",
        lambda *_args: pytest.fail("candidate mismatch must fail before inspection"),
    )
    monkeypatch.setattr(
        module,
        "write_manifest",
        lambda *_args: pytest.fail("candidate mismatch must not write manifest"),
    )

    with pytest.raises(RuntimeError, match="tar SHA-256 mismatch"):
        module.command_transition_image(
            transition_image_args(candidate_path), value
        )


def test_transition_image_rejects_active_image_id_mismatch_before_stop(
    monkeypatch, tmp_path
):
    module = load_module()
    value = isolated_topology(module, tmp_path)
    stored = image_transition_manifest(module, value)
    candidate_path, _candidate = write_image_transition_candidate(module, value)
    monkeypatch.setattr(module, "read_manifest", lambda *_args: stored)
    monkeypatch.setattr(
        module,
        "inspect_container",
        lambda *_args: pytest.fail("active id mismatch must fail before inspection"),
    )
    args = transition_image_args(candidate_path)
    args.expected_current_image_id = "sha256:not-active"

    with pytest.raises(RuntimeError, match="expected-current-image-id"):
        module.command_transition_image(args, value)


def test_transition_image_rejects_wrong_storage_mount_before_lifecycle(
    monkeypatch, tmp_path
):
    module = load_module()
    value = isolated_topology(module, tmp_path)
    stored = image_transition_manifest(module, value)
    candidate_path, candidate = write_image_transition_candidate(module, value)

    def wrong_mount(role, inspected):
        if role == "qdrant_shard_2":
            inspected["Mounts"][0]["Source"] = "/tmp/not-this-run"

    _state, _images, commands, writes = install_transition_runtime_fakes(
        monkeypatch,
        module,
        value,
        stored,
        candidate,
        inspect_mutator=wrong_mount,
    )

    with pytest.raises(RuntimeError, match="storage_mount"):
        module.command_transition_image(
            transition_image_args(candidate_path), value
        )

    lifecycle = [
        command[3]
        for _role, command in commands
        if isinstance(command, list)
        and len(command) > 3
        and command[3] in {"stop", "rm", "run", "start"}
    ]
    assert lifecycle == []
    assert writes == []


def test_transition_image_rejects_unhealthy_cluster_before_lifecycle(
    monkeypatch, tmp_path
):
    module = load_module()
    value = isolated_topology(module, tmp_path)
    stored = image_transition_manifest(module, value)
    candidate_path, candidate = write_image_transition_candidate(module, value)
    _state, _images, commands, writes = install_transition_runtime_fakes(
        monkeypatch, module, value, stored, candidate
    )
    unhealthy = healthy_cluster_snapshot(module, value)
    unhealthy["result"]["raft_info"]["pending_operations"] = 2
    monkeypatch.setattr(module, "cluster_snapshot", lambda _topology: unhealthy)

    with pytest.raises(RuntimeError, match="unhealthy cluster"):
        module.command_transition_image(
            transition_image_args(candidate_path), value
        )

    assert not any(
        isinstance(command, list)
        and len(command) > 3
        and command[3] in {"stop", "rm", "run", "start"}
        for _role, command in commands
    )
    assert writes == []


def test_transition_image_same_image_id_is_idempotent_noop(
    monkeypatch, tmp_path
):
    module = load_module()
    value = isolated_topology(module, tmp_path)
    stored = image_transition_manifest(module, value, wire_version="2")
    candidate_path, candidate = write_image_transition_candidate(
        module, value, image_id="sha256:abc"
    )
    _state, _images, commands, writes = install_transition_runtime_fakes(
        monkeypatch, module, value, stored, candidate
    )

    assert module.command_transition_image(
        transition_image_args(candidate_path), value
    ) == 0

    assert not any(
        isinstance(command, list)
        and len(command) > 3
        and command[3] in {"load", "stop", "rm", "run", "start"}
        for _role, command in commands
    )
    assert writes == []


def test_transition_image_same_image_different_wire_is_not_a_noop(
    monkeypatch, tmp_path
):
    module = load_module()
    value = isolated_topology(module, tmp_path)
    stored = image_transition_manifest(module, value, wire_version="1")
    candidate_path, candidate = write_image_transition_candidate(
        module, value, image_id="sha256:abc"
    )
    state, _images, commands, writes = install_transition_runtime_fakes(
        monkeypatch, module, value, stored, candidate
    )

    assert module.command_transition_image(
        transition_image_args(candidate_path), value
    ) == 0

    run_commands = [
        (role, command)
        for role, command in commands
        if isinstance(command, list)
        and len(command) > 3
        and command[3] == "run"
    ]
    assert len(run_commands) == 4
    controller_run = next(
        command for role, command in run_commands if role == "controller"
    )
    assert f"{module.ORION_COMPACT_WIRE_VERSION_ENV}=2" in option_values(
        controller_run, "-e"
    )
    assert state["controller"]["wire_version"] == "2"
    assert writes[-1]["orion_compact_wire"] == module.orion_compact_wire_summary(
        "2", "2"
    )
    proof = writes[-1]["image_transitions"][-1]
    assert proof["old_image"]["orion_compact_wire_version"] == "1"
    assert proof["candidate_image"]["orion_compact_wire_version"] == "2"


def test_transition_image_dry_run_has_no_live_or_manifest_mutation(
    monkeypatch, tmp_path
):
    module = load_module()
    value = isolated_topology(module, tmp_path)
    stored = image_transition_manifest(module, value)
    candidate_path, _candidate = write_image_transition_candidate(module, value)
    monkeypatch.setattr(module, "read_manifest", lambda *_args: stored)
    monkeypatch.setattr(
        module,
        "inspect_container",
        lambda *_args: pytest.fail("dry-run must not inspect live containers"),
    )
    monkeypatch.setattr(
        module,
        "run_on_node",
        lambda *_args, **_kwargs: pytest.fail("dry-run must not run node commands"),
    )
    monkeypatch.setattr(
        module,
        "write_manifest",
        lambda *_args: pytest.fail("dry-run must not write the active manifest"),
    )

    assert module.command_transition_image(
        transition_image_args(candidate_path, dry_run=True), value
    ) == 0


def test_transition_image_recreates_exact_four_nodes_and_preserves_storage(
    monkeypatch, tmp_path
):
    module = load_module()
    value = isolated_topology(module, tmp_path)
    stored = image_transition_manifest(module, value)
    candidate_path, candidate = write_image_transition_candidate(module, value)
    state, remote_images, commands, writes = install_transition_runtime_fakes(
        monkeypatch,
        module,
        value,
        stored,
        candidate,
        collection_values=[healthy_run_collections(), healthy_run_collections()],
    )

    assert module.command_transition_image(
        transition_image_args(candidate_path), value
    ) == 0

    lifecycle = [
        (role, command)
        for role, command in commands
        if isinstance(command, list)
        and len(command) > 3
        and command[3] in {"stop", "rm", "run", "start"}
    ]
    stop_order = [node["role"] for node in value["workers"]] + ["controller"]
    start_order = ["controller"] + [node["role"] for node in value["workers"]]
    assert [role for role, command in lifecycle if command[3] == "stop"] == stop_order
    assert [role for role, command in lifecycle if command[3] == "rm"] == stop_order
    assert [role for role, command in lifecycle if command[3] == "run"] == start_order
    assert sum(command[3] == "stop" for _role, command in lifecycle) == 4
    assert sum(command[3] == "rm" for _role, command in lifecycle) == 4
    assert sum(command[3] == "run" for _role, command in lifecycle) == 4
    assert all("rm -rf" not in " ".join(command) for _role, command in commands)
    for role, command in lifecycle:
        if command[3] != "run":
            continue
        storage = (
            module.local_role_root(value, "run-1", role) / "storage"
        ).resolve()
        assert f"{storage}:/qdrant/storage" in option_values(command, "-v")
    assert all(
        current["image_id"] == candidate["image"]["id"]
        and current["image_tag"] == candidate["image"]["tag"]
        and current["generation"] == 1
        for current in state.values()
    )
    assert all(
        images[candidate["image"]["tag"]] == candidate["image"]["id"]
        for images in remote_images.values()
    )
    assert len(writes) == 1
    updated = writes[0]
    assert updated["image"]["id"] == candidate["image"]["id"]
    assert updated["orion_artifacts"] == stored["orion_artifacts"]
    assert updated["simple_kmeans_artifacts"] == stored["simple_kmeans_artifacts"]
    proof = updated["image_transitions"][-1]
    assert proof["outcome"] == "success"
    assert proof["storage_preserved"] is True
    assert len(proof["containers_before"]) == 4
    assert len(proof["containers_after"]) == 4
    assert proof["cluster_before"] == proof["cluster_after"]
    assert proof["collections_before"] == proof["collections_after"]


def test_transition_image_failure_restores_old_image_and_records_rollback(
    monkeypatch, tmp_path
):
    module = load_module()
    value = isolated_topology(module, tmp_path)
    stored = image_transition_manifest(module, value)
    candidate_path, candidate = write_image_transition_candidate(module, value)
    snapshot = healthy_cluster_snapshot(module, value)
    state, _remote_images, commands, writes = install_transition_runtime_fakes(
        monkeypatch,
        module,
        value,
        stored,
        candidate,
        collection_values=[healthy_run_collections(), healthy_run_collections()],
        wait_values=[TimeoutError("injected candidate timeout"), snapshot],
    )

    with pytest.raises(RuntimeError, match="rolled back"):
        module.command_transition_image(
            transition_image_args(candidate_path), value
        )

    assert all(
        current["image_tag"] == stored["image"]["tag"]
        and current["image_id"] == stored["image"]["id"]
        and current["running"] is True
        for current in state.values()
    )
    lifecycle = [
        command[3]
        for _role, command in commands
        if isinstance(command, list)
        and len(command) > 3
        and command[3] in {"stop", "rm", "run", "start"}
    ]
    assert lifecycle.count("stop") == 8
    assert lifecycle.count("rm") == 8
    assert lifecycle.count("run") == 8
    assert not any("rm -rf" in " ".join(command) for _role, command in commands)
    assert writes[-1]["image"] == stored["image"]
    proof = writes[-1]["image_transitions"][-1]
    assert proof["outcome"] == "rolled_back"
    assert proof["rollback"]["succeeded"] is True
    assert len(proof["rollback"]["containers"]) == 4
    assert proof["rollback"]["orion_compact_wire_version"] == "1"
    assert writes[-1]["orion_compact_wire"] == module.orion_compact_wire_summary(
        "2", "1"
    )
    controller_runs = [
        command
        for role, command in commands
        if role == "controller"
        and isinstance(command, list)
        and len(command) > 3
        and command[3] == "run"
    ]
    assert len(controller_runs) == 2
    assert f"{module.ORION_COMPACT_WIRE_VERSION_ENV}=2" in option_values(
        controller_runs[0], "-e"
    )
    assert not any(
        item.startswith(f"{module.ORION_COMPACT_WIRE_VERSION_ENV}=")
        for item in option_values(controller_runs[1], "-e")
    )


def test_transition_image_postflight_placement_mismatch_triggers_rollback(
    monkeypatch, tmp_path
):
    module = load_module()
    value = isolated_topology(module, tmp_path)
    stored = image_transition_manifest(module, value)
    candidate_path, candidate = write_image_transition_candidate(module, value)
    before = healthy_run_collections()
    changed = healthy_run_collections()
    changed["dist_run-1_orion_s3"]["cluster"]["remote_shards"][0][
        "peer_id"
    ] = 102
    changed["dist_run-1_orion_s3"]["cluster"]["remote_shards"][1][
        "peer_id"
    ] = 101
    state, _images, _commands, writes = install_transition_runtime_fakes(
        monkeypatch,
        module,
        value,
        stored,
        candidate,
        collection_values=[before, changed, before],
        wait_values=[
            healthy_cluster_snapshot(module, value),
            healthy_cluster_snapshot(module, value),
        ],
    )

    with pytest.raises(RuntimeError, match="changed run-scoped collection"):
        module.command_transition_image(
            transition_image_args(candidate_path), value
        )

    assert all(current["image_id"] == "sha256:abc" for current in state.values())
    assert writes[-1]["image"] == stored["image"]
    assert writes[-1]["image_transitions"][-1]["outcome"] == "rolled_back"


def test_transition_image_remote_candidate_conflict_fails_before_stop(
    monkeypatch, tmp_path
):
    module = load_module()
    value = isolated_topology(module, tmp_path)
    stored = image_transition_manifest(module, value)
    candidate_path, candidate = write_image_transition_candidate(module, value)
    _state, remote_images, commands, writes = install_transition_runtime_fakes(
        monkeypatch, module, value, stored, candidate
    )
    remote_images["qdrant_shard_3"][candidate["image"]["tag"]] = "sha256:wrong"

    with pytest.raises(RuntimeError, match="conflicting remote tag"):
        module.command_transition_image(
            transition_image_args(candidate_path), value
        )

    assert not any(
        isinstance(command, list)
        and len(command) > 3
        and command[3] in {"stop", "rm", "run", "start"}
        for _role, command in commands
    )
    assert writes == []


def test_image_and_peer_premerge_transitions_share_run_lifecycle_lock(
    monkeypatch, tmp_path
):
    module = load_module()
    value = isolated_topology(module, tmp_path)
    monkeypatch.setattr(
        module,
        "_command_transition_image_unlocked",
        lambda *_args: pytest.fail("locked image transition must not enter command"),
    )
    monkeypatch.setattr(
        module,
        "_command_set_peer_premerge_unlocked",
        lambda *_args: pytest.fail("locked peer transition must not enter command"),
    )
    image_args = SimpleNamespace(run_id="run-1", dry_run=False)
    peer_args = SimpleNamespace(
        run_id="run-1", dry_run=False, mode="enabled", shards_per_rpc=None
    )

    with module.run_lifecycle_lock(value, "run-1"):
        with pytest.raises(RuntimeError, match="already holds the run lock"):
            module.command_transition_image(image_args, value)
        with pytest.raises(RuntimeError, match="already holds the run lock"):
            module.command_set_peer_premerge(peer_args, value)


def test_deploy_reinspects_loaded_tag_and_records_actual_image_id(
    monkeypatch, tmp_path
):
    module = load_module()
    value = isolated_topology(module, tmp_path)
    node = value["controller"]
    image_tag = "image:test"
    image_id = "sha256:expected"
    tar_path = tmp_path / "image.tar"
    tar_path.write_bytes(b"archive")
    stored = {
        "image": {
            "tag": image_tag,
            "id": image_id,
            "tar_path": str(tar_path),
            "tar_sha256": module.sha256_file(tar_path),
        }
    }
    image_inspects = 0
    container_inspects = 0
    commands = []

    def fake_run_on_node(_node, command, _args, **_kwargs):
        nonlocal image_inspects
        commands.append(command)
        if isinstance(command, list) and command[3:6] == ["image", "inspect", image_tag]:
            image_inspects += 1
            if image_inspects == 1:
                return subprocess.CompletedProcess(command, 1, "", "not found")
            return subprocess.CompletedProcess(command, 0, image_id + "\n", "")
        return subprocess.CompletedProcess(command, 0, "", "")

    written = {}
    monkeypatch.setattr(module, "all_nodes", lambda _topology: [node])
    monkeypatch.setattr(module, "read_manifest", lambda *_args: stored)
    monkeypatch.setattr(
        module,
        "git_state",
        lambda _repo: {"commit": "commit-1", "dirty_paths": []},
    )
    monkeypatch.setattr(module, "run_on_node", fake_run_on_node)
    def fake_inspect_container(_node, _name, _args):
        nonlocal container_inspects
        container_inspects += 1
        if container_inspects == 1:
            return None
        return matching_node_inspect(
            module,
            value,
            node,
            False,
            "4",
            image_tag=image_tag,
            image_id=image_id,
            wire_version="2",
        )

    monkeypatch.setattr(module, "inspect_container", fake_inspect_container)
    monkeypatch.setattr(module, "wait_http_ready", lambda *_args: None)
    monkeypatch.setattr(module, "node_facts", lambda *_args: {})
    monkeypatch.setattr(module.time, "sleep", lambda *_args: None)
    monkeypatch.setattr(module, "cluster_snapshot", lambda _topology: {"result": {}})
    monkeypatch.setattr(module, "cluster_validation_errors", lambda *_args: [])

    def fake_manifest(
        _topology,
        _run_id,
        _repo,
        tag,
        manifest_image_id,
        nodes,
        *_args,
    ):
        return {"image": {"tag": tag, "id": manifest_image_id}, "nodes": nodes}

    monkeypatch.setattr(module, "build_manifest_data", fake_manifest)
    monkeypatch.setattr(
        module,
        "write_manifest",
        lambda _topology, _run_id, data: written.update(data) or tmp_path / "manifest.json",
    )
    args = SimpleNamespace(
        repo=str(tmp_path),
        run_id="run-1",
        image_tag=None,
        dry_run=False,
        disable_peer_premerge=False,
        peer_premerge_shards_per_rpc="4",
        wait_timeout=1.0,
        ssh_user=None,
        ssh_option=[],
    )

    assert module.command_deploy(args, value) == 0
    assert image_inspects == 2
    assert container_inspects == 2
    assert ["sudo", "-n", "docker", "load", "--input", str(tar_path)] in commands
    assert written["nodes"][0]["image_id"] == image_id
    assert written["nodes"][0]["peer_premerge_shards_per_rpc"] == "4"
    assert written["nodes"][0]["orion_compact_wire_version"] == "2"
    assert written["nodes"][0]["orion_compact_wire_max_version"] == "2"
    run_command = next(
        command
        for command in commands
        if isinstance(command, list) and command[3:4] == ["run"]
    )
    assert f"{module.PEER_PREMERGE_SHARDS_PER_RPC_ENV}=4" in option_values(
        run_command, "-e"
    )
    assert f"{module.PEER_PREMERGE_SHARDS_PER_RPC_LABEL}=4" in option_values(
        run_command, "--label"
    )


def test_deploy_rejects_tag_id_mismatch_after_load(monkeypatch, tmp_path):
    module = load_module()
    value = isolated_topology(module, tmp_path)
    node = value["controller"]
    tar_path = tmp_path / "image.tar"
    tar_path.write_bytes(b"archive")
    stored = {
        "image": {
            "tag": "image:test",
            "id": "sha256:expected",
            "tar_path": str(tar_path),
            "tar_sha256": module.sha256_file(tar_path),
        }
    }
    image_inspects = 0

    def fake_run_on_node(_node, command, _args, **_kwargs):
        nonlocal image_inspects
        if isinstance(command, list) and command[3:6] == ["image", "inspect", "image:test"]:
            image_inspects += 1
            if image_inspects == 1:
                return subprocess.CompletedProcess(command, 1, "", "not found")
            return subprocess.CompletedProcess(command, 0, "sha256:wrong\n", "")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(module, "all_nodes", lambda _topology: [node])
    monkeypatch.setattr(module, "read_manifest", lambda *_args: stored)
    monkeypatch.setattr(
        module,
        "git_state",
        lambda _repo: {"commit": "commit-1", "dirty_paths": []},
    )
    monkeypatch.setattr(module, "run_on_node", fake_run_on_node)
    args = SimpleNamespace(
        repo=str(tmp_path),
        run_id="run-1",
        image_tag=None,
        dry_run=False,
        disable_peer_premerge=False,
        wait_timeout=1.0,
        ssh_user=None,
        ssh_option=[],
    )

    with pytest.raises(RuntimeError, match="after load/verification"):
        module.command_deploy(args, value)


def test_status_requires_cluster_peer_uris_premerge_and_active_worker_placement(
    monkeypatch, capsys
):
    module = load_module()
    value = topology(module)
    snapshot = healthy_cluster_snapshot(module, value)
    collections = healthy_run_collections()

    def fake_node_status(_topology, node, run_id, _args):
        return {
            "role": node["role"],
            "container_name": module.container_name(run_id, node["role"]),
            "running": True,
            "peer_premerge_mode": (
                "enabled" if node["role"] == "controller" else "not_applicable"
            ),
            "peer_premerge_shards_per_rpc": (
                "all" if node["role"] == "controller" else "not_applicable"
            ),
            "orion_compact_wire_version": (
                "2" if node["role"] == "controller" else "not_applicable"
            ),
            "orion_compact_wire_max_version": "2",
        }

    monkeypatch.setattr(module, "status_for_node", fake_node_status)
    monkeypatch.setattr(module, "cluster_snapshot", lambda _topology: snapshot)
    monkeypatch.setattr(
        module,
        "run_collection_placements",
        lambda _topology, _run_id: collections,
    )
    args = SimpleNamespace(run_id="run-1", disable_peer_premerge=False, dry_run=False)

    assert module.command_status(args, value) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["validation"] == {"ok": True, "errors": []}


def test_cluster_proof_and_validation_read_top_level_message_send_failures():
    module = load_module()
    value = topology(module)
    snapshot = healthy_cluster_snapshot(module, value)
    failures = {"103": {"count": 2, "latest_error": "transport closed"}}
    snapshot["result"]["message_send_failures"] = failures
    snapshot["result"]["raft_info"]["message_send_failures"] = {}

    errors = module.cluster_validation_errors(value, snapshot)
    proof = module.transition_cluster_proof(snapshot)

    assert any("message_send_failures" in error for error in errors)
    assert proof["message_send_failures"] == failures


@pytest.mark.parametrize(
    ("requested", "current", "expected_return"),
    [("4", "4", 0), ("8", "4", 1)],
)
def test_status_reports_and_validates_peer_premerge_chunk_setting(
    monkeypatch, capsys, requested, current, expected_return
):
    module = load_module()
    value = topology(module)
    snapshot = healthy_cluster_snapshot(module, value)
    collections = healthy_run_collections()

    def fake_node_status(_topology, node, run_id, _args):
        return {
            "role": node["role"],
            "container_name": module.container_name(run_id, node["role"]),
            "running": True,
            "peer_premerge_mode": (
                "enabled" if node["role"] == "controller" else "not_applicable"
            ),
            "peer_premerge_shards_per_rpc": (
                current if node["role"] == "controller" else "not_applicable"
            ),
            "orion_compact_wire_version": (
                "2" if node["role"] == "controller" else "not_applicable"
            ),
            "orion_compact_wire_max_version": "2",
        }

    monkeypatch.setattr(module, "status_for_node", fake_node_status)
    monkeypatch.setattr(module, "cluster_snapshot", lambda _topology: snapshot)
    monkeypatch.setattr(
        module,
        "run_collection_placements",
        lambda _topology, _run_id: collections,
    )
    args = SimpleNamespace(
        run_id="run-1",
        disable_peer_premerge=False,
        peer_premerge_shards_per_rpc=requested,
        dry_run=False,
    )

    assert module.command_status(args, value) == expected_return
    payload = json.loads(capsys.readouterr().out)
    assert payload["peer_premerge"]["requested_shards_per_rpc"] == requested
    assert payload["peer_premerge"]["current_shards_per_rpc"] == current
    assert payload["peer_premerge"]["shards_per_rpc_matches_requested"] is (
        expected_return == 0
    )
    if expected_return:
        assert "shards-per-rpc mismatch" in "\n".join(
            payload["validation"]["errors"]
        )


@pytest.mark.parametrize(
    ("fault", "expected_error"),
    [
        ("peer_uri", "cluster peer URI mismatch"),
        ("premerge", "peer-premerge mode mismatch"),
        ("inactive", "expected 'Active'"),
        ("controller_local", "lower shard(s) on controller"),
    ],
)
def test_status_returns_nonzero_for_cluster_or_placement_mismatch(
    monkeypatch, capsys, fault, expected_error
):
    module = load_module()
    value = topology(module)
    snapshot = healthy_cluster_snapshot(module, value)
    collections = healthy_run_collections()
    if fault == "peer_uri":
        snapshot["result"]["peers"]["103"]["uri"] = "http://10.10.1.99:6335"
    if fault == "inactive":
        collections["dist_run-1_orion_s3"]["cluster"]["remote_shards"][0][
            "state"
        ] = "Dead"
    if fault == "controller_local":
        collections["dist_run-1_orion_s3"]["cluster"]["local_shards"] = [
            {"shard_id": 0, "state": "Active"}
        ]

    def fake_node_status(_topology, node, run_id, _args):
        return {
            "role": node["role"],
            "container_name": module.container_name(run_id, node["role"]),
            "running": True,
            "peer_premerge_mode": (
                "disabled"
                if fault == "premerge" and node["role"] == "controller"
                else "enabled"
                if node["role"] == "controller"
                else "not_applicable"
            ),
            "peer_premerge_shards_per_rpc": (
                "all" if node["role"] == "controller" else "not_applicable"
            ),
            "orion_compact_wire_version": (
                "2" if node["role"] == "controller" else "not_applicable"
            ),
            "orion_compact_wire_max_version": "2",
        }

    monkeypatch.setattr(module, "status_for_node", fake_node_status)
    monkeypatch.setattr(module, "cluster_snapshot", lambda _topology: snapshot)
    monkeypatch.setattr(
        module,
        "run_collection_placements",
        lambda _topology, _run_id: collections,
    )
    args = SimpleNamespace(run_id="run-1", disable_peer_premerge=False, dry_run=False)

    assert module.command_status(args, value) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["validation"]["ok"] is False
    assert expected_error in "\n".join(payload["validation"]["errors"])


def test_status_returns_nonzero_when_cluster_endpoint_is_unreachable(
    monkeypatch, capsys
):
    module = load_module()
    value = topology(module)

    def fake_node_status(_topology, node, run_id, _args):
        return {
            "role": node["role"],
            "container_name": module.container_name(run_id, node["role"]),
            "running": True,
            "peer_premerge_mode": (
                "enabled" if node["role"] == "controller" else "not_applicable"
            ),
            "peer_premerge_shards_per_rpc": (
                "all" if node["role"] == "controller" else "not_applicable"
            ),
            "orion_compact_wire_version": (
                "2" if node["role"] == "controller" else "not_applicable"
            ),
            "orion_compact_wire_max_version": "2",
        }

    monkeypatch.setattr(module, "status_for_node", fake_node_status)
    monkeypatch.setattr(
        module,
        "cluster_snapshot",
        lambda _topology: (_ for _ in ()).throw(OSError("connection refused")),
    )
    args = SimpleNamespace(run_id="run-1", disable_peer_premerge=False, dry_run=False)

    assert module.command_status(args, value) == 1
    payload = json.loads(capsys.readouterr().out)
    assert "cluster endpoint is not accessible" in "\n".join(
        payload["validation"]["errors"]
    )


def test_status_summary_exposes_requested_and_current_peer_premerge_mode(monkeypatch):
    module = load_module()
    value = topology(module)
    inspected = matching_controller_inspect(module, value, True)
    monkeypatch.setattr(module, "inspect_container", lambda *args, **kwargs: inspected)
    args = type("Args", (), {})()

    node_status = module.status_for_node(
        value, value["controller"], "run-1", args
    )
    summary = module.peer_premerge_summary(True, node_status["peer_premerge_mode"])

    assert node_status["peer_premerge_mode"] == "disabled"
    assert node_status["nofile"] == {"soft": 65536, "hard": 65536}
    assert summary["requested_mode"] == "disabled"
    assert summary["current_mode"] == "disabled"
    assert summary["matches_requested"] is True


def test_manifest_reports_requested_and_current_peer_premerge_chunk_setting(
    monkeypatch, tmp_path, capsys
):
    module = load_module()
    value = isolated_topology(module, tmp_path)
    stored = peer_premerge_manifest(value, "enabled", "4")
    monkeypatch.setattr(module, "read_manifest", lambda *_args: stored)
    monkeypatch.setattr(
        module,
        "git_state",
        lambda _repo: {
            "commit": "commit-1",
            "dirty": False,
            "dirty_paths": [],
            "tracked_dirty": False,
            "tracked_dirty_paths": [],
            "untracked_entry_count": 0,
        },
    )
    monkeypatch.setattr(
        module,
        "inspect_container",
        lambda node, _name, _args: matching_node_inspect(
            module, value, node, False, 4
        ),
    )
    monkeypatch.setattr(module, "node_facts", lambda *_args: {})
    args = SimpleNamespace(
        repo=str(tmp_path),
        run_id="run-1",
        image_tag=None,
        disable_peer_premerge=False,
        peer_premerge_shards_per_rpc="4",
        dry_run=True,
        ssh_user=None,
        ssh_option=[],
    )

    assert module.command_manifest(args, value) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["peer_premerge"]["requested_shards_per_rpc"] == "4"
    assert payload["peer_premerge"]["current_shards_per_rpc"] == "4"
    assert payload["peer_premerge"]["matches_requested"] is True
    controller = next(
        node for node in payload["nodes"] if node["role"] == "controller"
    )
    assert controller["peer_premerge_shards_per_rpc"] == "4"
    assert all(
        node["peer_premerge_shards_per_rpc"] == "not_applicable"
        for node in payload["nodes"]
        if node["role"] != "controller"
    )


def test_manifest_preserves_deployment_commit_and_records_current_tooling_repo(
    monkeypatch, tmp_path, capsys
):
    module = load_module()
    value = isolated_topology(module, tmp_path)
    stored = peer_premerge_manifest(value, "enabled", "all")
    stored["repository"] = {
        "path": "/deployment/repository",
        "commit": "old-deployment-commit",
        "short_commit": "old-deploy",
        "dirty": False,
        "dirty_paths": [],
    }
    stored["last_peer_premerge_transition"] = "run-1-peer-premerge-0002"
    stored["last_image_transition"] = "run-1-image-0003"
    tooling_state = {
        "commit": "new-tooling-commit",
        "short_commit": "new-tooling",
        "dirty": True,
        "dirty_paths": ["tools/method4_distributed_cluster.py"],
        "tracked_dirty": True,
        "tracked_dirty_paths": ["tools/method4_distributed_cluster.py"],
        "untracked_entry_count": 0,
    }
    monkeypatch.setattr(module, "read_manifest", lambda *_args: stored)
    monkeypatch.setattr(module, "git_state", lambda _repo: tooling_state)
    monkeypatch.setattr(
        module,
        "inspect_container",
        lambda node, _name, _args: matching_node_inspect(
            module, value, node, False, "all"
        ),
    )
    monkeypatch.setattr(module, "node_facts", lambda *_args: {})
    args = SimpleNamespace(
        repo=str(tmp_path),
        run_id="run-1",
        image_tag=None,
        disable_peer_premerge=False,
        peer_premerge_shards_per_rpc="all",
        dry_run=True,
        ssh_user=None,
        ssh_option=[],
    )

    assert module.command_manifest(args, value) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["repository"] == stored["repository"]
    assert payload["repository"]["commit"] == "old-deployment-commit"
    assert payload["last_peer_premerge_transition"] == "run-1-peer-premerge-0002"
    assert payload["last_image_transition"] == "run-1-image-0003"
    assert payload["tooling_repository"] == {
        "path": str(tmp_path.resolve()),
        **tooling_state,
    }
    assert payload["orion_compact_wire"] == module.orion_compact_wire_summary(
        "2", "1"
    )


def test_manifest_recovers_last_transition_markers_from_preserved_history(
    monkeypatch, tmp_path, capsys
):
    module = load_module()
    value = isolated_topology(module, tmp_path)
    stored = peer_premerge_manifest(value, "enabled", "all")
    stored["peer_premerge_transitions"] = [
        {"transition_id": "run-1-peer-premerge-0002"}
    ]
    stored["image_transitions"] = [{"transition_id": "run-1-image-0003"}]
    monkeypatch.setattr(module, "read_manifest", lambda *_args: stored)
    monkeypatch.setattr(
        module,
        "git_state",
        lambda _repo: {
            "commit": "tooling-commit",
            "short_commit": "tooling",
            "dirty": False,
            "dirty_paths": [],
            "tracked_dirty": False,
            "tracked_dirty_paths": [],
            "untracked_entry_count": 0,
        },
    )
    monkeypatch.setattr(
        module,
        "inspect_container",
        lambda node, _name, _args: matching_node_inspect(
            module, value, node, False, "all"
        ),
    )
    monkeypatch.setattr(module, "node_facts", lambda *_args: {})
    args = SimpleNamespace(
        repo=str(tmp_path),
        run_id="run-1",
        image_tag=None,
        disable_peer_premerge=False,
        peer_premerge_shards_per_rpc="all",
        dry_run=True,
        ssh_user=None,
        ssh_option=[],
    )

    assert module.command_manifest(args, value) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["last_peer_premerge_transition"] == (
        "run-1-peer-premerge-0002"
    )
    assert payload["last_image_transition"] == "run-1-image-0003"


def test_set_peer_premerge_same_mode_reuses_controller_and_appends_proof(
    monkeypatch, tmp_path
):
    module = load_module()
    value = isolated_topology(module, tmp_path)
    stored = peer_premerge_manifest(value, "enabled")
    snapshot = healthy_cluster_snapshot(module, value)
    writes = []
    commands = []

    def fake_inspect(_node, _name, _args):
        return matching_node_inspect(module, value, _node, False)

    def fake_run_on_node(node, command, _args, **_kwargs):
        commands.append((node["role"], command))
        if isinstance(command, list) and command[3:6] == [
            "image",
            "inspect",
            "image:test",
        ]:
            return subprocess.CompletedProcess(command, 0, "sha256:abc\n", "")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(module, "read_manifest", lambda *_args: stored)
    monkeypatch.setattr(module, "inspect_container", fake_inspect)
    monkeypatch.setattr(module, "run_on_node", fake_run_on_node)
    monkeypatch.setattr(module, "cluster_snapshot", lambda _topology: snapshot)
    monkeypatch.setattr(
        module,
        "run_collection_placements",
        lambda *_args: healthy_run_collections(),
    )
    monkeypatch.setattr(
        module,
        "wait_controller_and_cluster_healthy",
        lambda *_args: snapshot,
    )
    monkeypatch.setattr(
        module,
        "write_manifest",
        lambda _topology, _run_id, data: writes.append(copy.deepcopy(data))
        or tmp_path / "manifest.json",
    )
    args = SimpleNamespace(
        run_id="run-1",
        mode="enabled",
        wait_timeout=10.0,
        image_tag=None,
        expected_commit="commit-1",
        repo=str(REPO_ROOT),
        dry_run=False,
        ssh_user=None,
        ssh_option=[],
    )

    assert module.command_set_peer_premerge(args, value) == 0

    lifecycle = [
        command
        for _role, command in commands
        if not (isinstance(command, list) and "image" in command)
    ]
    assert lifecycle == []
    assert writes[-1]["orion_artifacts"] == stored["orion_artifacts"]
    assert writes[-1]["simple_kmeans_artifacts"] == stored[
        "simple_kmeans_artifacts"
    ]
    proof = writes[-1]["peer_premerge_transitions"][-1]
    assert proof["action"] == "reused"
    assert proof["outcome"] == "success"
    assert proof["workers_restarted"] is False
    assert set(proof["tooling_repository"]) == {
        "path",
        "commit",
        "tracked_dirty",
        "tracked_dirty_paths",
        "untracked_entry_count",
    }
    assert [worker["container_id"] for worker in proof["validated_workers"]] == [
        "container-qdrant_shard_1",
        "container-qdrant_shard_2",
        "container-qdrant_shard_3",
    ]


def test_set_peer_premerge_reuse_rejects_exact_collection_change(
    monkeypatch, tmp_path
):
    module = load_module()
    value = isolated_topology(module, tmp_path)
    stored = peer_premerge_manifest(value, "enabled")
    snapshot = healthy_cluster_snapshot(module, value)
    before = healthy_run_collections()
    changed = healthy_run_collections()
    changed["dist_run-1_orion_s3"]["info"]["points_count"] = 301
    changed["dist_run-1_orion_s3"]["info"]["indexed_vectors_count"] = 301
    collection_values = [before, changed]
    commands = []

    def fake_run_on_node(node, command, _args, **_kwargs):
        commands.append((node["role"], command))
        if isinstance(command, list) and command[3:6] == [
            "image",
            "inspect",
            "image:test",
        ]:
            return subprocess.CompletedProcess(command, 0, "sha256:abc\n", "")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(module, "read_manifest", lambda *_args: stored)
    monkeypatch.setattr(
        module,
        "inspect_container",
        lambda node, _name, _args: matching_node_inspect(
            module, value, node, False
        ),
    )
    monkeypatch.setattr(module, "run_on_node", fake_run_on_node)
    monkeypatch.setattr(module, "cluster_snapshot", lambda _topology: snapshot)
    monkeypatch.setattr(
        module,
        "run_collection_placements",
        lambda *_args: copy.deepcopy(collection_values.pop(0)),
    )
    monkeypatch.setattr(
        module,
        "wait_controller_and_cluster_healthy",
        lambda *_args: snapshot,
    )
    monkeypatch.setattr(
        module,
        "write_manifest",
        lambda *_args: pytest.fail("failed reuse equality must not write manifest"),
    )
    args = SimpleNamespace(
        run_id="run-1",
        mode="enabled",
        wait_timeout=10.0,
        image_tag=None,
        expected_commit="commit-1",
        repo=str(REPO_ROOT),
        dry_run=False,
        ssh_user=None,
        ssh_option=[],
    )

    with pytest.raises(RuntimeError, match="peer-premerge reuse changed"):
        module.command_set_peer_premerge(args, value)

    assert not any(
        isinstance(command, list)
        and len(command) > 3
        and command[3] in {"stop", "rm", "run", "start"}
        for _role, command in commands
    )


def test_set_peer_premerge_collection_change_after_recreate_rolls_back(
    monkeypatch, tmp_path
):
    module = load_module()
    value = isolated_topology(module, tmp_path)
    stored = peer_premerge_manifest(value, "enabled")
    snapshot = healthy_cluster_snapshot(module, value)
    before = healthy_run_collections()
    changed = healthy_run_collections()
    changed["dist_run-1_orion_s3"]["cluster"]["remote_shards"][0][
        "peer_id"
    ] = 102
    changed["dist_run-1_orion_s3"]["cluster"]["remote_shards"][1][
        "peer_id"
    ] = 101
    collection_values = [before, changed]
    state = {"mode": "enabled"}
    commands = []
    writes = []

    def fake_inspect(node, _name, _args):
        return matching_node_inspect(
            module,
            value,
            node,
            state["mode"] == "disabled" if node["role"] == "controller" else False,
        )

    def fake_run_on_node(node, command, _args, **_kwargs):
        commands.append((node["role"], command))
        if isinstance(command, list) and command[3:6] == [
            "image",
            "inspect",
            "image:test",
        ]:
            return subprocess.CompletedProcess(command, 0, "sha256:abc\n", "")
        if isinstance(command, list) and command[3:4] == ["run"]:
            state["mode"] = (
                "disabled"
                if f"{module.PEER_PREMERGE_DISABLE_ENV}=1"
                in option_values(command, "-e")
                else "enabled"
            )
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(module, "read_manifest", lambda *_args: stored)
    monkeypatch.setattr(module, "inspect_container", fake_inspect)
    monkeypatch.setattr(module, "run_on_node", fake_run_on_node)
    monkeypatch.setattr(module, "cluster_snapshot", lambda _topology: snapshot)
    monkeypatch.setattr(
        module,
        "run_collection_placements",
        lambda *_args: copy.deepcopy(collection_values.pop(0)),
    )
    monkeypatch.setattr(
        module,
        "wait_controller_and_cluster_healthy",
        lambda *_args: snapshot,
    )
    monkeypatch.setattr(
        module,
        "write_manifest",
        lambda _topology, _run_id, data: writes.append(copy.deepcopy(data))
        or tmp_path / "manifest.json",
    )
    args = SimpleNamespace(
        run_id="run-1",
        mode="disabled",
        wait_timeout=10.0,
        image_tag=None,
        expected_commit="commit-1",
        repo=str(REPO_ROOT),
        dry_run=False,
        ssh_user=None,
        ssh_option=[],
    )

    with pytest.raises(RuntimeError, match="changed run-scoped collection"):
        module.command_set_peer_premerge(args, value)

    assert state["mode"] == "enabled"
    lifecycle = [
        command[3]
        for _role, command in commands
        if isinstance(command, list)
        and len(command) > 3
        and command[3] in {"stop", "rm", "run", "start"}
    ]
    assert lifecycle == ["stop", "rm", "run", "stop", "rm", "run"]
    assert writes[-1]["peer_premerge_transitions"][-1]["outcome"] == (
        "failed_rolled_back"
    )


def test_set_peer_premerge_rejects_run_without_scoped_collections(
    monkeypatch, tmp_path
):
    module = load_module()
    value = isolated_topology(module, tmp_path)
    stored = peer_premerge_manifest(value, "enabled")
    snapshot = healthy_cluster_snapshot(module, value)
    commands = []

    def fake_run_on_node(node, command, _args, **_kwargs):
        commands.append((node["role"], command))
        if isinstance(command, list) and command[3:6] == [
            "image",
            "inspect",
            "image:test",
        ]:
            return subprocess.CompletedProcess(command, 0, "sha256:abc\n", "")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(module, "read_manifest", lambda *_args: stored)
    monkeypatch.setattr(
        module,
        "inspect_container",
        lambda node, _name, _args: matching_node_inspect(
            module, value, node, False
        ),
    )
    monkeypatch.setattr(module, "run_on_node", fake_run_on_node)
    monkeypatch.setattr(module, "cluster_snapshot", lambda _topology: snapshot)
    monkeypatch.setattr(module, "run_collection_placements", lambda *_args: {})
    monkeypatch.setattr(
        module,
        "write_manifest",
        lambda *_args: pytest.fail("rejected transition must not write a manifest"),
    )
    args = SimpleNamespace(
        run_id="run-1",
        mode="enabled",
        wait_timeout=10.0,
        image_tag=None,
        expected_commit="commit-1",
        repo=str(REPO_ROOT),
        dry_run=False,
        ssh_user=None,
        ssh_option=[],
    )

    with pytest.raises(RuntimeError, match="no run-scoped collections"):
        module.command_set_peer_premerge(args, value)

    lifecycle = [
        command[3]
        for _role, command in commands
        if isinstance(command, list)
        and len(command) > 3
        and command[3] in {"stop", "rm", "run", "start"}
    ]
    assert lifecycle == []


def test_set_peer_premerge_recreates_only_controller_and_preserves_storage(
    monkeypatch, tmp_path
):
    module = load_module()
    value = isolated_topology(module, tmp_path)
    stored = peer_premerge_manifest(value, "enabled")
    snapshot = healthy_cluster_snapshot(module, value)
    state = {"mode": "enabled", "controller_generation": 0}
    commands = []
    writes = []

    def fake_inspect(node, _name, _args):
        return matching_node_inspect(
            module,
            value,
            node,
            state["mode"] == "disabled" if node["role"] == "controller" else False,
            container_id=(
                f"container-controller-{state['controller_generation']}"
                if node["role"] == "controller"
                else None
            ),
        )

    def fake_run_on_node(node, command, _args, **_kwargs):
        commands.append((node["role"], command))
        if isinstance(command, list) and command[3:6] == [
            "image",
            "inspect",
            "image:test",
        ]:
            return subprocess.CompletedProcess(command, 0, "sha256:abc\n", "")
        if isinstance(command, list) and command[3:4] == ["run"]:
            state["controller_generation"] += 1
            state["mode"] = (
                "disabled"
                if f"{module.PEER_PREMERGE_DISABLE_ENV}=1"
                in option_values(command, "-e")
                else "enabled"
            )
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(module, "read_manifest", lambda *_args: stored)
    monkeypatch.setattr(module, "inspect_container", fake_inspect)
    monkeypatch.setattr(module, "run_on_node", fake_run_on_node)
    monkeypatch.setattr(module, "cluster_snapshot", lambda _topology: snapshot)
    monkeypatch.setattr(
        module,
        "run_collection_placements",
        lambda *_args: healthy_run_collections(),
    )
    monkeypatch.setattr(
        module,
        "wait_controller_and_cluster_healthy",
        lambda *_args: snapshot,
    )
    monkeypatch.setattr(
        module,
        "write_manifest",
        lambda _topology, _run_id, data: writes.append(copy.deepcopy(data))
        or tmp_path / "manifest.json",
    )
    args = SimpleNamespace(
        run_id="run-1",
        mode="disabled",
        wait_timeout=10.0,
        image_tag=None,
        expected_commit="commit-1",
        repo=str(REPO_ROOT),
        dry_run=False,
        ssh_user=None,
        ssh_option=[],
    )

    assert module.command_set_peer_premerge(args, value) == 0

    lifecycle = [
        (role, command)
        for role, command in commands
        if not (isinstance(command, list) and "image" in command)
    ]
    assert [role for role, _command in lifecycle] == ["controller"] * 3
    assert [command[3] for _role, command in lifecycle] == ["stop", "rm", "run"]
    run_command = lifecycle[-1][1]
    storage = (
        module.local_role_root(value, "run-1", "controller") / "storage"
    ).resolve()
    assert f"{storage}:/qdrant/storage" in option_values(run_command, "-v")
    assert f"{module.PEER_PREMERGE_DISABLE_ENV}=1" in option_values(
        run_command, "-e"
    )
    assert not any(
        item.startswith(f"{module.ORION_COMPACT_WIRE_VERSION_ENV}=")
        for item in option_values(run_command, "-e")
    )
    assert not any("rm" == item and str(storage) in run_command for item in run_command)
    assert writes[-1]["orion_artifacts"] == stored["orion_artifacts"]
    assert writes[-1]["peer_premerge"]["current_mode"] == "disabled"
    assert writes[-1]["orion_compact_wire"] == module.orion_compact_wire_summary(
        "2", "1"
    )
    proof = writes[-1]["peer_premerge_transitions"][-1]
    assert proof["outcome"] == "success"
    assert proof["from_mode"] == "enabled"
    assert proof["final_mode"] == "disabled"
    assert proof["final_orion_compact_wire_version"] == "1"
    assert proof["controller_before"]["container_id"] == "container-controller-0"
    assert proof["controller_after"]["container_id"] == "container-controller-1"


@pytest.mark.parametrize(
    ("from_chunk", "target_chunk", "expected_action"),
    [
        ("all", "4", "recreated"),
        ("4", "all", "recreated"),
        ("4", "4", "reused"),
    ],
)
def test_set_peer_premerge_chunk_only_transition_is_controller_scoped(
    monkeypatch,
    tmp_path,
    from_chunk,
    target_chunk,
    expected_action,
):
    module = load_module()
    value = isolated_topology(module, tmp_path)
    stored = peer_premerge_manifest(value, "enabled", from_chunk)
    snapshot = healthy_cluster_snapshot(module, value)
    state = {
        "mode": "enabled",
        "shards_per_rpc": from_chunk,
        "controller_generation": 0,
    }
    commands = []
    writes = []

    def fake_inspect(node, _name, _args):
        return matching_node_inspect(
            module,
            value,
            node,
            state["mode"] == "disabled" if node["role"] == "controller" else False,
            state["shards_per_rpc"],
            (
                f"container-controller-{state['controller_generation']}"
                if node["role"] == "controller"
                else None
            ),
        )

    def fake_run_on_node(node, command, _args, **_kwargs):
        commands.append((node["role"], command))
        if isinstance(command, list) and command[3:6] == [
            "image",
            "inspect",
            "image:test",
        ]:
            return subprocess.CompletedProcess(command, 0, "sha256:abc\n", "")
        if isinstance(command, list) and command[3:4] == ["run"]:
            state["controller_generation"] += 1
            env = option_values(command, "-e")
            state["mode"] = (
                "disabled"
                if f"{module.PEER_PREMERGE_DISABLE_ENV}=1" in env
                else "enabled"
            )
            chunk_values = [
                item.partition("=")[2]
                for item in env
                if item.startswith(f"{module.PEER_PREMERGE_SHARDS_PER_RPC_ENV}=")
            ]
            state["shards_per_rpc"] = chunk_values[0] if chunk_values else "all"
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(module, "read_manifest", lambda *_args: stored)
    monkeypatch.setattr(module, "inspect_container", fake_inspect)
    monkeypatch.setattr(module, "run_on_node", fake_run_on_node)
    monkeypatch.setattr(module, "cluster_snapshot", lambda _topology: snapshot)
    monkeypatch.setattr(
        module,
        "run_collection_placements",
        lambda *_args: healthy_run_collections(),
    )
    monkeypatch.setattr(
        module,
        "wait_controller_and_cluster_healthy",
        lambda *_args: snapshot,
    )
    monkeypatch.setattr(
        module,
        "write_manifest",
        lambda _topology, _run_id, data: writes.append(copy.deepcopy(data))
        or tmp_path / "manifest.json",
    )
    args = SimpleNamespace(
        run_id="run-1",
        mode=None,
        shards_per_rpc=target_chunk,
        wait_timeout=10.0,
        image_tag=None,
        expected_commit="commit-1",
        repo=str(REPO_ROOT),
        dry_run=False,
        ssh_user=None,
        ssh_option=[],
    )

    assert module.command_set_peer_premerge(args, value) == 0
    lifecycle = [
        (role, command)
        for role, command in commands
        if not (isinstance(command, list) and "image" in command)
    ]
    if expected_action == "recreated":
        assert [role for role, _command in lifecycle] == ["controller"] * 3
        assert [command[3] for _role, command in lifecycle] == [
            "stop",
            "rm",
            "run",
        ]
        run_command = lifecycle[-1][1]
        chunk_env = option_values(run_command, "-e")
        chunk_labels = option_values(run_command, "--label")
        if target_chunk == "all":
            assert not any(
                item.startswith(f"{module.PEER_PREMERGE_SHARDS_PER_RPC_ENV}=")
                for item in chunk_env
            )
            assert not any(
                item.startswith(f"{module.PEER_PREMERGE_SHARDS_PER_RPC_LABEL}=")
                for item in chunk_labels
            )
        else:
            assert (
                f"{module.PEER_PREMERGE_SHARDS_PER_RPC_ENV}={target_chunk}"
                in chunk_env
            )
            assert (
                f"{module.PEER_PREMERGE_SHARDS_PER_RPC_LABEL}={target_chunk}"
                in chunk_labels
            )
        assert f"{module.PEER_PREMERGE_DISABLE_ENV}=1" not in chunk_env
    else:
        assert lifecycle == []

    assert state["mode"] == "enabled"
    assert state["shards_per_rpc"] == target_chunk
    assert state["controller_generation"] == (1 if expected_action == "recreated" else 0)
    assert writes[-1]["peer_premerge"]["current_mode"] == "enabled"
    assert writes[-1]["peer_premerge"]["current_shards_per_rpc"] == target_chunk
    controller_node = next(
        node for node in writes[-1]["nodes"] if node["role"] == "controller"
    )
    assert controller_node["peer_premerge_shards_per_rpc"] == target_chunk
    proof = writes[-1]["peer_premerge_transitions"][-1]
    assert proof["action"] == expected_action
    assert proof["from_mode"] == "enabled"
    assert proof["requested_mode"] == "enabled"
    assert proof["final_mode"] == "enabled"
    assert proof["from_shards_per_rpc"] == from_chunk
    assert proof["requested_shards_per_rpc"] == target_chunk
    assert proof["final_shards_per_rpc"] == target_chunk
    assert proof["workers_restarted"] is False
    assert proof["controller_after"]["peer_premerge_shards_per_rpc"] == target_chunk
    if expected_action == "recreated":
        assert proof["controller_before"]["container_id"] != proof["controller_after"][
            "container_id"
        ]
    else:
        assert proof["controller_before"]["container_id"] == proof["controller_after"][
            "container_id"
        ]
    assert proof["controller_after"]["controller_fingerprint"] == (
        module.controller_config_fingerprint(
            value["controller"],
            "run-1",
            "sha256:abc",
            False,
            target_chunk,
        )
    )


def test_set_peer_premerge_failure_rolls_controller_back_and_records_proof(
    monkeypatch, tmp_path
):
    module = load_module()
    value = isolated_topology(module, tmp_path)
    stored = peer_premerge_manifest(value, "enabled")
    snapshot = healthy_cluster_snapshot(module, value)
    state = {"mode": "enabled"}
    commands = []
    writes = []
    waits = 0

    def fake_inspect(node, _name, _args):
        return matching_node_inspect(
            module,
            value,
            node,
            state["mode"] == "disabled" if node["role"] == "controller" else False,
        )

    def fake_run_on_node(node, command, _args, **_kwargs):
        commands.append((node["role"], command))
        if isinstance(command, list) and command[3:6] == [
            "image",
            "inspect",
            "image:test",
        ]:
            return subprocess.CompletedProcess(command, 0, "sha256:abc\n", "")
        if isinstance(command, list) and command[3:4] == ["run"]:
            state["mode"] = (
                "disabled"
                if f"{module.PEER_PREMERGE_DISABLE_ENV}=1"
                in option_values(command, "-e")
                else "enabled"
            )
        return subprocess.CompletedProcess(command, 0, "", "")

    def fake_wait(*_args):
        nonlocal waits
        waits += 1
        if waits == 1:
            raise TimeoutError("injected four-peer timeout")
        return snapshot

    monkeypatch.setattr(module, "read_manifest", lambda *_args: stored)
    monkeypatch.setattr(module, "inspect_container", fake_inspect)
    monkeypatch.setattr(module, "run_on_node", fake_run_on_node)
    monkeypatch.setattr(module, "cluster_snapshot", lambda _topology: snapshot)
    monkeypatch.setattr(
        module,
        "run_collection_placements",
        lambda *_args: healthy_run_collections(),
    )
    monkeypatch.setattr(module, "wait_controller_and_cluster_healthy", fake_wait)
    monkeypatch.setattr(
        module,
        "write_manifest",
        lambda _topology, _run_id, data: writes.append(copy.deepcopy(data))
        or tmp_path / "manifest.json",
    )
    args = SimpleNamespace(
        run_id="run-1",
        mode="disabled",
        wait_timeout=10.0,
        image_tag=None,
        expected_commit="commit-1",
        repo=str(REPO_ROOT),
        dry_run=False,
        ssh_user=None,
        ssh_option=[],
    )

    with pytest.raises(RuntimeError, match="rolled back to enabled"):
        module.command_set_peer_premerge(args, value)

    assert state["mode"] == "enabled"
    lifecycle = [
        command[3]
        for _role, command in commands
        if isinstance(command, list)
        and len(command) > 3
        and command[3] in {"stop", "rm", "run", "start"}
    ]
    assert lifecycle == ["stop", "rm", "run", "stop", "rm", "run"]
    proof = writes[-1]["peer_premerge_transitions"][-1]
    assert proof["outcome"] == "failed_rolled_back"
    assert proof["rollback"] == {
        "attempted": True,
        "succeeded": True,
        "error": None,
    }
    assert writes[-1]["peer_premerge"]["current_mode"] == "enabled"
    assert writes[-1]["orion_compact_wire"] == module.orion_compact_wire_summary(
        "2", "1"
    )
    assert writes[-1]["orion_artifacts"] == stored["orion_artifacts"]
    controller_runs = [
        command
        for role, command in commands
        if role == "controller"
        and isinstance(command, list)
        and len(command) > 3
        and command[3] == "run"
    ]
    assert len(controller_runs) == 2
    assert all(
        not any(
            item.startswith(f"{module.ORION_COMPACT_WIRE_VERSION_ENV}=")
            for item in option_values(command, "-e")
        )
        for command in controller_runs
    )


def test_set_peer_premerge_chunk_failure_rolls_back_exact_original_chunk(
    monkeypatch, tmp_path
):
    module = load_module()
    value = isolated_topology(module, tmp_path)
    stored = peer_premerge_manifest(value, "enabled", "3")
    snapshot = healthy_cluster_snapshot(module, value)
    state = {"mode": "enabled", "shards_per_rpc": "3"}
    commands = []
    writes = []
    waits = 0

    def fake_inspect(node, _name, _args):
        return matching_node_inspect(
            module,
            value,
            node,
            state["mode"] == "disabled" if node["role"] == "controller" else False,
            state["shards_per_rpc"],
        )

    def fake_run_on_node(node, command, _args, **_kwargs):
        commands.append((node["role"], command))
        if isinstance(command, list) and command[3:6] == [
            "image",
            "inspect",
            "image:test",
        ]:
            return subprocess.CompletedProcess(command, 0, "sha256:abc\n", "")
        if isinstance(command, list) and command[3:4] == ["run"]:
            env = option_values(command, "-e")
            chunk_values = [
                item.partition("=")[2]
                for item in env
                if item.startswith(f"{module.PEER_PREMERGE_SHARDS_PER_RPC_ENV}=")
            ]
            state["mode"] = (
                "disabled"
                if f"{module.PEER_PREMERGE_DISABLE_ENV}=1" in env
                else "enabled"
            )
            state["shards_per_rpc"] = chunk_values[0] if chunk_values else "all"
        return subprocess.CompletedProcess(command, 0, "", "")

    def fake_wait(*_args):
        nonlocal waits
        waits += 1
        if waits == 1:
            raise TimeoutError("injected chunk transition timeout")
        return snapshot

    monkeypatch.setattr(module, "read_manifest", lambda *_args: stored)
    monkeypatch.setattr(module, "inspect_container", fake_inspect)
    monkeypatch.setattr(module, "run_on_node", fake_run_on_node)
    monkeypatch.setattr(module, "cluster_snapshot", lambda _topology: snapshot)
    monkeypatch.setattr(
        module,
        "run_collection_placements",
        lambda *_args: healthy_run_collections(),
    )
    monkeypatch.setattr(module, "wait_controller_and_cluster_healthy", fake_wait)
    monkeypatch.setattr(
        module,
        "write_manifest",
        lambda _topology, _run_id, data: writes.append(copy.deepcopy(data))
        or tmp_path / "manifest.json",
    )
    args = SimpleNamespace(
        run_id="run-1",
        mode=None,
        shards_per_rpc="5",
        wait_timeout=10.0,
        image_tag=None,
        expected_commit="commit-1",
        repo=str(REPO_ROOT),
        dry_run=False,
        ssh_user=None,
        ssh_option=[],
    )

    with pytest.raises(RuntimeError, match="shards_per_rpc=3"):
        module.command_set_peer_premerge(args, value)

    assert state == {"mode": "enabled", "shards_per_rpc": "3"}
    lifecycle = [
        (role, command[3])
        for role, command in commands
        if isinstance(command, list)
        and len(command) > 3
        and command[3] in {"stop", "rm", "run", "start"}
    ]
    assert lifecycle == [
        ("controller", "stop"),
        ("controller", "rm"),
        ("controller", "run"),
        ("controller", "stop"),
        ("controller", "rm"),
        ("controller", "run"),
    ]
    proof = writes[-1]["peer_premerge_transitions"][-1]
    assert proof["outcome"] == "failed_rolled_back"
    assert proof["from_shards_per_rpc"] == "3"
    assert proof["requested_shards_per_rpc"] == "5"
    assert proof["final_shards_per_rpc"] == "3"
    assert proof["rollback"] == {
        "attempted": True,
        "succeeded": True,
        "error": None,
    }
    assert writes[-1]["peer_premerge"]["current_mode"] == "enabled"
    assert writes[-1]["peer_premerge"]["current_shards_per_rpc"] == "3"


def test_set_peer_premerge_dry_run_plans_only_controller_and_writes_nothing(
    monkeypatch, tmp_path
):
    module = load_module()
    value = isolated_topology(module, tmp_path)
    stored = peer_premerge_manifest(value, "enabled")
    commands = []
    monkeypatch.setattr(module, "read_manifest", lambda *_args: stored)
    monkeypatch.setattr(
        module,
        "run_on_node",
        lambda node, command, _args, **_kwargs: commands.append(
            (node["role"], command)
        )
        or subprocess.CompletedProcess(command, 0, "", ""),
    )
    monkeypatch.setattr(
        module,
        "write_manifest",
        lambda *_args: pytest.fail("dry-run must not write the manifest"),
    )
    monkeypatch.setattr(
        module,
        "inspect_container",
        lambda *_args: pytest.fail("dry-run must not inspect live containers"),
    )
    args = SimpleNamespace(
        run_id="run-1",
        mode="disabled",
        shards_per_rpc="4",
        wait_timeout=10.0,
        image_tag=None,
        expected_commit="commit-1",
        repo=str(REPO_ROOT),
        dry_run=True,
        ssh_user=None,
        ssh_option=[],
    )

    assert module.command_set_peer_premerge(args, value) == 0
    assert [role for role, _command in commands] == ["controller"] * 3
    assert [command[3] for _role, command in commands] == ["stop", "rm", "run"]
    run_command = commands[-1][1]
    assert f"{module.PEER_PREMERGE_DISABLE_ENV}=1" in option_values(
        run_command, "-e"
    )
    assert f"{module.PEER_PREMERGE_SHARDS_PER_RPC_ENV}=4" in option_values(
        run_command, "-e"
    )
    assert f"{module.PEER_PREMERGE_SHARDS_PER_RPC_LABEL}=4" in option_values(
        run_command, "--label"
    )
    assert all("clean" not in command for _role, command in commands)


def test_install_orion_artifact_cli_accepts_optional_restart_order(monkeypatch):
    module = load_module()
    checksum = "a" * 64

    monkeypatch.setattr(
        module.sys,
        "argv",
        [
            "cluster",
            "--run-id",
            "run-1",
            "install-orion-artifact",
            "--collection",
            "native_orion",
            "--generation",
            "7",
            "--artifact",
            "/tmp/artifact.json",
            "--expected-sha256",
            checksum,
            "--restart",
            "controller-first",
        ],
    )
    args = module.parse_args()

    assert args.command == "install-orion-artifact"
    assert args.collection == "native_orion"
    assert args.generation == 7
    assert args.restart == "controller-first"

    monkeypatch.setattr(
        module.sys,
        "argv",
        [
            "cluster",
            "--run-id",
            "run-1",
            "install-orion-artifact",
            "--collection",
            "native_orion",
            "--generation",
            "7",
            "--artifact",
            "/tmp/artifact.json",
            "--expected-sha256",
            checksum,
            "--restart",
        ],
    )
    assert module.parse_args().restart == "workers-first"


def test_install_simple_kmeans_artifact_cli_and_destination(monkeypatch, tmp_path):
    module = load_module()
    checksum = "b" * 64
    monkeypatch.setattr(
        module.sys,
        "argv",
        [
            "cluster",
            "--run-id",
            "run-1",
            "install-simple-kmeans-artifact",
            "--collection",
            "native_simple",
            "--generation",
            "9",
            "--artifact",
            "/tmp/simple.json",
            "--expected-sha256",
            checksum,
            "--restart",
        ],
    )

    args = module.parse_args()
    assert args.command == "install-simple-kmeans-artifact"
    assert args.collection == "native_simple"
    assert args.generation == 9
    assert args.restart == "workers-first"

    value = isolated_topology(module, tmp_path)
    destination = module.simple_kmeans_artifact_destination(
        value, "run-1", "qdrant_shard_1", "native_simple", 9
    )
    assert destination == (
        tmp_path
        / "local/run-1/qdrant_shard_1/storage/collections/native_simple"
        / "simple_kmeans_router/generation-9.json"
    ).resolve()


def test_orion_artifact_destination_is_run_role_scoped_and_rejects_escape(tmp_path):
    module = load_module()
    value = isolated_topology(module, tmp_path)

    destination = module.orion_artifact_destination(
        value, "run-1", "qdrant_shard_1", "native_orion", 7
    )

    assert destination == (
        tmp_path
        / "local/run-1/qdrant_shard_1/storage/collections/native_orion"
        / "orion_router/generation-7.json"
    ).resolve()
    for collection in ("../escape", "/tmp/escape", "nested/name", "", "."):
        with pytest.raises(ValueError, match="collection"):
            module.orion_artifact_destination(
                value, "run-1", "qdrant_shard_1", collection, 7
            )


def test_local_orion_artifact_enforces_canonical_file_checksum_and_generation(
    tmp_path,
):
    module = load_module()
    artifact, checksum = write_orion_artifact(module, tmp_path, generation=7)

    metadata = module.validate_local_orion_artifact(
        artifact, 7, f"sha256:{checksum.upper()}"
    )

    assert metadata["canonical_sha256"] == checksum
    assert metadata["file_sha256"] == checksum
    assert metadata["generation"] == 7
    pretty = tmp_path / "pretty-artifact.json"
    pretty.write_text(
        json.dumps(json.loads(artifact.read_text(encoding="utf-8")), indent=2),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="canonical"):
        module.validate_local_orion_artifact(pretty, 7, checksum)
    with pytest.raises(RuntimeError, match="file SHA-256"):
        module.validate_local_orion_artifact(artifact, 7, "0" * 64)
    with pytest.raises(RuntimeError, match="generation mismatch"):
        module.validate_local_orion_artifact(artifact, 8, checksum)


def test_local_orion_artifact_rejects_graphless_production_input(tmp_path):
    module = load_module()
    payload = {"format_version": 1, "generation": 7}
    artifact = tmp_path / "graphless.json"
    artifact.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")

    with pytest.raises(RuntimeError, match="graphless"):
        module.validate_local_orion_artifact(
            artifact, 7, module.sha256_file(artifact)
        )


def test_local_simple_kmeans_artifact_enforces_static_baseline_contract(tmp_path):
    module = load_module()
    artifact, checksum = write_simple_kmeans_artifact(module, tmp_path)

    metadata = module.validate_local_simple_kmeans_artifact(artifact, 7, checksum)
    assert metadata["shard_count"] == 4
    assert metadata["logical_point_count"] == metadata["physical_point_count"] == 4
    assert metadata["nprobe"] == 2
    assert metadata["lower_hnsw_ef"] == 48
    assert metadata["routing_distance"] == "squared_l2"

    payload = json.loads(artifact.read_text(encoding="utf-8"))
    payload["upper_graph"] = None
    artifact.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    with pytest.raises(RuntimeError, match="must not contain upper_graph"):
        module.validate_local_simple_kmeans_artifact(
            artifact, 7, module.sha256_file(artifact)
        )


@pytest.mark.parametrize(
    "mutate,match",
    [
        (
            lambda payload: payload.update(physical_point_count=5),
            "must equal logical_point_count",
        ),
        (
            lambda payload: payload["centroids"].__setitem__(
                3, {"shard_id": 2, "vector": [2.0, 1.0]}
            ),
            "repeats centroid",
        ),
        (
            lambda payload: payload.update(nprobe=5),
            "no larger than shard_count",
        ),
    ],
)
def test_local_simple_kmeans_artifact_rejects_invalid_layout(
    tmp_path, mutate, match
):
    module = load_module()
    artifact, _checksum = write_simple_kmeans_artifact(module, tmp_path)
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    mutate(payload)
    artifact.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        module.validate_local_simple_kmeans_artifact(
            artifact, 7, module.sha256_file(artifact)
        )


def test_artifact_copy_and_atomic_install_commands_are_node_and_path_scoped(
    tmp_path,
):
    module = load_module()
    value = isolated_topology(module, tmp_path)
    source = tmp_path / "artifact.json"
    destination = module.orion_artifact_destination(
        value, "run-1", "qdrant_shard_1", "native_orion", 7
    )
    staged = destination.parents[3] / ".orion-artifact-staging/artifact.json"
    temporary = destination.with_name(".generation-7.json.tmp-deadbeef")
    storage = destination.parents[3]

    local_copy = module.copy_to_node_command(
        value["controller"], source, staged
    )
    remote_copy = module.copy_to_node_command(
        value["workers"][0], source, staged, "dry", ["StrictHostKeyChecking=yes"]
    )
    finalize = module.artifact_finalize_command(
        storage, staged, temporary, destination, "a" * 64
    )

    assert local_copy == ["cp", "--", str(source), str(staged)]
    assert remote_copy[:6] == [
        "scp",
        "-q",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=10",
    ]
    assert remote_copy[-1] == f"dry@hp052.utah.cloudlab.us:{staged}"
    assert str(destination) in finalize
    assert str(staged) in finalize
    assert str(temporary) in finalize
    assert "mv --no-clobber" in finalize
    assert "sha256sum" in finalize


def test_install_orion_artifact_is_idempotent_and_upserts_one_manifest_entry(
    monkeypatch, tmp_path
):
    module = load_module()
    value = isolated_topology(module, tmp_path)
    artifact, checksum = write_orion_artifact(module, tmp_path)
    current_manifest = {"schema_version": 1, "run_id": "run-1", "image": {}}
    writes = []
    copies = []

    def fake_read_manifest(*_args):
        return current_manifest

    def fake_write_manifest(_topology, _run_id, data):
        current_manifest.clear()
        current_manifest.update(json.loads(json.dumps(data)))
        writes.append(json.loads(json.dumps(data)))
        return tmp_path / "manifest.json"

    def fake_run_on_node(_node, command, _args, **_kwargs):
        assert "artifact generation already exists" in command
        return subprocess.CompletedProcess(command, 0, f"match {checksum}\n", "")

    monkeypatch.setattr(module, "read_manifest", fake_read_manifest)
    monkeypatch.setattr(module, "write_manifest", fake_write_manifest)
    monkeypatch.setattr(
        module,
        "validate_collection_orion_policy",
        lambda *_args: {"policy": {"type": "orion"}, "status": "green"},
    )
    monkeypatch.setattr(module, "run_on_node", fake_run_on_node)
    monkeypatch.setattr(
        module,
        "run_command",
        lambda command, **_kwargs: copies.append(command)
        or subprocess.CompletedProcess(command, 0, "", ""),
    )
    args = SimpleNamespace(
        run_id="run-1",
        collection="native_orion",
        generation=7,
        artifact=str(artifact),
        expected_sha256=checksum,
        restart=None,
        wait_timeout=10.0,
        ssh_user=None,
        ssh_option=[],
        dry_run=False,
    )

    assert module.command_install_orion_artifact(args, value) == 0
    assert module.command_install_orion_artifact(args, value) == 0

    assert copies == []
    assert len(current_manifest["orion_artifacts"]) == 1
    entry = current_manifest["orion_artifacts"][0]
    assert entry["canonical_sha256"] == checksum
    assert [node["action"] for node in entry["nodes"]] == ["reused"] * 4
    assert len(writes) == 2


def test_install_simple_kmeans_artifact_is_idempotent_and_records_policy_kind(
    monkeypatch, tmp_path
):
    module = load_module()
    value = isolated_topology(module, tmp_path)
    artifact, checksum = write_simple_kmeans_artifact(module, tmp_path)
    current_manifest = {"schema_version": 1, "run_id": "run-1", "image": {}}
    copies = []

    def fake_write_manifest(_topology, _run_id, data):
        current_manifest.clear()
        current_manifest.update(json.loads(json.dumps(data)))
        return tmp_path / "manifest.json"

    monkeypatch.setattr(module, "read_manifest", lambda *_args: current_manifest)
    monkeypatch.setattr(module, "write_manifest", fake_write_manifest)
    monkeypatch.setattr(
        module,
        "validate_collection_simple_kmeans_policy",
        lambda *_args: {
            "policy_kind": "simple_kmeans",
            "policy": {"type": "simple_kmeans"},
            "status": "green",
        },
    )
    monkeypatch.setattr(
        module,
        "run_on_node",
        lambda _node, command, _args, **_kwargs: subprocess.CompletedProcess(
            command, 0, f"match {checksum}\n", ""
        ),
    )
    monkeypatch.setattr(
        module,
        "run_command",
        lambda command, **_kwargs: copies.append(command),
    )
    args = SimpleNamespace(
        run_id="run-1",
        collection="native_simple",
        generation=7,
        artifact=str(artifact),
        expected_sha256=checksum,
        restart=None,
        wait_timeout=10.0,
        ssh_user=None,
        ssh_option=[],
        dry_run=False,
    )

    assert module.command_install_simple_kmeans_artifact(args, value) == 0
    assert module.command_install_simple_kmeans_artifact(args, value) == 0
    assert copies == []
    assert "orion_artifacts" not in current_manifest
    assert len(current_manifest["simple_kmeans_artifacts"]) == 1
    entry = current_manifest["simple_kmeans_artifacts"][0]
    assert entry["policy_kind"] == "simple_kmeans"
    assert entry["nprobe"] == 2
    assert entry["lower_hnsw_ef"] == 48
    assert all(
        "/simple_kmeans_router/generation-7.json" in node["destination_path"]
        for node in entry["nodes"]
    )


def test_install_orion_artifact_copies_missing_nodes_then_verifies_remote_sha(
    monkeypatch, tmp_path
):
    module = load_module()
    value = isolated_topology(module, tmp_path)
    artifact, checksum = write_orion_artifact(module, tmp_path)
    copied_commands = []
    probe_count = 0

    def fake_run_on_node(_node, command, _args, **_kwargs):
        nonlocal probe_count
        if "artifact generation already exists" in command:
            probe_count += 1
            if probe_count <= 4:
                return subprocess.CompletedProcess(command, 0, "missing\n", "")
            return subprocess.CompletedProcess(command, 0, f"match {checksum}\n", "")
        if "installed artifact SHA-256 mismatch" in command:
            return subprocess.CompletedProcess(command, 0, f"installed {checksum}\n", "")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(
        module,
        "read_manifest",
        lambda *_args: {"schema_version": 1, "run_id": "run-1", "image": {}},
    )
    monkeypatch.setattr(
        module,
        "write_manifest",
        lambda *_args: tmp_path / "manifest.json",
    )
    monkeypatch.setattr(
        module,
        "validate_collection_orion_policy",
        lambda *_args: {"policy": {"type": "orion"}, "status": "green"},
    )
    monkeypatch.setattr(module, "run_on_node", fake_run_on_node)
    monkeypatch.setattr(
        module,
        "run_command",
        lambda command, **_kwargs: copied_commands.append(command)
        or subprocess.CompletedProcess(command, 0, "", ""),
    )
    args = SimpleNamespace(
        run_id="run-1",
        collection="native_orion",
        generation=7,
        artifact=str(artifact),
        expected_sha256=checksum,
        restart=None,
        wait_timeout=10.0,
        ssh_user=None,
        ssh_option=[],
        dry_run=False,
    )

    assert module.command_install_orion_artifact(args, value) == 0

    assert probe_count == 8
    assert len(copied_commands) == 4
    assert copied_commands[0][0] == "cp"
    assert [command[0] for command in copied_commands[1:]] == ["scp"] * 3
    assert all("run-1" in command[-1] for command in copied_commands)


def test_install_orion_artifact_rejects_existing_checksum_drift_before_copy(
    monkeypatch, tmp_path
):
    module = load_module()
    value = isolated_topology(module, tmp_path)
    artifact, checksum = write_orion_artifact(module, tmp_path)
    copies = []

    monkeypatch.setattr(
        module,
        "read_manifest",
        lambda *_args: {"schema_version": 1, "run_id": "run-1", "image": {}},
    )
    monkeypatch.setattr(
        module,
        "validate_collection_orion_policy",
        lambda *_args: {"policy": {"type": "orion"}, "status": "green"},
    )
    monkeypatch.setattr(
        module,
        "run_on_node",
        lambda _node, command, _args, **_kwargs: subprocess.CompletedProcess(
            command, 72, "", "artifact generation already exists with a different SHA-256"
        ),
    )
    monkeypatch.setattr(
        module,
        "run_command",
        lambda command, **_kwargs: copies.append(command),
    )
    args = SimpleNamespace(
        run_id="run-1",
        collection="native_orion",
        generation=7,
        artifact=str(artifact),
        expected_sha256=checksum,
        restart=None,
        wait_timeout=10.0,
        ssh_user=None,
        ssh_option=[],
        dry_run=False,
    )

    with pytest.raises(RuntimeError, match="different SHA-256"):
        module.command_install_orion_artifact(args, value)
    assert copies == []


def test_artifact_restart_order_is_explicit_and_run_container_identity_is_checked():
    module = load_module()
    value = topology(module)

    assert [
        node["role"]
        for node in module.artifact_restart_nodes(value, "workers-first")
    ] == ["qdrant_shard_1", "qdrant_shard_2", "qdrant_shard_3", "controller"]
    assert [
        node["role"]
        for node in module.artifact_restart_nodes(value, "controller-first")
    ] == ["controller", "qdrant_shard_1", "qdrant_shard_2", "qdrant_shard_3"]

    inspected = matching_controller_inspect(module, value, False)
    module.verify_run_container_identity(
        inspected, value["controller"], "run-1"
    )
    inspected["Config"]["Labels"]["orion.distributed.run_id"] = "other-run"
    with pytest.raises(RuntimeError, match="exact run ownership"):
        module.verify_run_container_identity(
            inspected, value["controller"], "run-1"
        )


def test_artifact_activation_requires_matching_collection_policy(monkeypatch):
    module = load_module()
    value = topology(module)
    checksum = "a" * 64
    payload = {
        "result": {
            "config": {
                "auto_shard_policy": {
                    "type": "orion",
                    "generation": 7,
                    "artifact_sha256": checksum,
                }
            }
        }
    }
    monkeypatch.setattr(module, "http_json", lambda _url: payload)

    proof = module.validate_collection_orion_policy(
        value, "native_orion", 7, checksum
    )
    assert proof["policy"]["generation"] == 7

    payload["result"]["config"]["auto_shard_policy"]["generation"] = 8
    with pytest.raises(RuntimeError, match="policy mismatch"):
        module.validate_collection_orion_policy(
            value, "native_orion", 7, checksum
        )


def test_artifact_activation_validates_schema_shards_and_runtime_load_logs(
    monkeypatch,
):
    module = load_module()
    value = topology(module)
    checksum = "a" * 64
    info = {
        "result": {
            "status": "green",
            "optimizer_status": "ok",
            "points_count": 3,
            "config": {
                "params": {
                    "shard_number": 2,
                    "sharding_method": "auto",
                    "vectors": {
                        "size": 2,
                        "distance": "Cosine",
                        "datatype": "float32",
                    },
                },
                "auto_shard_policy": {
                    "type": "orion",
                    "generation": 7,
                    "artifact_sha256": checksum,
                },
            },
        }
    }
    cluster = {
        "result": {
            "local_shards": [],
            "remote_shards": [
                {"shard_id": 0, "peer_id": 101, "state": "Active"},
                {"shard_id": 1, "peer_id": 102, "state": "Active"},
            ],
            "shard_transfers": [],
        }
    }
    monkeypatch.setattr(
        module,
        "http_json",
        lambda url: cluster if url.endswith("/cluster") else info,
    )
    artifact = {
        "shard_count": 2,
        "physical_point_count": 3,
        "vector_schema": {
            "vector_name": "",
            "dimension": 2,
            "distance": "Cosine",
            "datatype": "float32",
        },
    }

    proof = module.validate_collection_orion_policy(
        value, "native_orion", 7, checksum, artifact
    )
    assert proof["shard_count"] == 2

    node = value["workers"][0]
    marker = "Loaded Orion routing generation 7 for collection native_orion"
    monkeypatch.setattr(
        module,
        "run_on_node",
        lambda *_args, **_kwargs: subprocess.CompletedProcess([], 0, "", marker),
    )
    log_proof = module.verify_orion_router_loaded_logs(
        node,
        "run-1",
        "native_orion",
        7,
        123,
        SimpleNamespace(),
    )
    assert log_proof["role"] == "qdrant_shard_1"

    fallback = "Orion routing is unavailable for collection native_orion"
    monkeypatch.setattr(
        module,
        "run_on_node",
        lambda *_args, **_kwargs: subprocess.CompletedProcess([], 0, "", fallback),
    )
    with pytest.raises(RuntimeError, match="fallback"):
        module.verify_orion_router_loaded_logs(
            node,
            "run-1",
            "native_orion",
            7,
            123,
            SimpleNamespace(),
        )

    info["result"]["config"]["params"]["vectors"]["size"] = 3
    with pytest.raises(RuntimeError, match="vector schema"):
        module.validate_collection_orion_policy(
            value, "native_orion", 7, checksum, artifact
        )


def test_simple_kmeans_activation_validates_policy_collection_and_logs(monkeypatch):
    module = load_module()
    value = topology(module)
    checksum = "d" * 64
    info = {
        "result": {
            "status": "green",
            "optimizer_status": "ok",
            "points_count": 4,
            "config": {
                "params": {
                    "shard_number": 4,
                    "sharding_method": "auto",
                    "vectors": {
                        "size": 2,
                        "distance": "Cosine",
                        "datatype": "float32",
                    },
                },
                "auto_shard_policy": {
                    "type": "simple_kmeans",
                    "generation": 7,
                    "artifact_sha256": checksum,
                },
            },
        }
    }
    cluster = {
        "result": {
            "local_shards": [],
            "remote_shards": [
                {
                    "shard_id": shard_id,
                    "peer_id": 101 + (shard_id % 3),
                    "state": "Active",
                }
                for shard_id in range(4)
            ],
            "shard_transfers": [],
        }
    }
    monkeypatch.setattr(
        module,
        "http_json",
        lambda url: cluster if url.endswith("/cluster") else info,
    )
    artifact = {
        "shard_count": 4,
        "physical_point_count": 4,
        "vector_schema": {
            "vector_name": "",
            "dimension": 2,
            "distance": "Cosine",
            "datatype": "float32",
        },
    }

    proof = module.validate_collection_simple_kmeans_policy(
        value, "native_simple", 7, checksum, artifact
    )
    assert proof["policy_kind"] == "simple_kmeans"
    assert proof["shard_count"] == 4

    node = value["workers"][0]
    loaded = "Loaded Simple KMeans routing generation 7 for collection native_simple"
    monkeypatch.setattr(
        module,
        "run_on_node",
        lambda *_args, **_kwargs: subprocess.CompletedProcess([], 0, loaded, ""),
    )
    log_proof = module.verify_simple_kmeans_router_loaded_logs(
        node, "run-1", "native_simple", 7, 123, SimpleNamespace()
    )
    assert log_proof["loaded_marker"] == loaded

    fallback = (
        "Simple KMeans routing generation 7 failed for collection native_simple; "
        "falling back to all shards"
    )
    monkeypatch.setattr(
        module,
        "run_on_node",
        lambda *_args, **_kwargs: subprocess.CompletedProcess([], 0, fallback, ""),
    )
    with pytest.raises(RuntimeError, match="Simple KMeans fallback"):
        module.verify_simple_kmeans_router_loaded_logs(
            node, "run-1", "native_simple", 7, 123, SimpleNamespace()
        )

    info["result"]["config"]["auto_shard_policy"]["type"] = "orion"
    with pytest.raises(RuntimeError, match="Simple KMeans policy mismatch"):
        module.validate_collection_simple_kmeans_policy(
            value, "native_simple", 7, checksum, artifact
        )


def test_install_orion_artifact_restart_verifies_all_router_load_markers(
    monkeypatch, tmp_path
):
    module = load_module()
    value = isolated_topology(module, tmp_path)
    artifact, checksum = write_orion_artifact(module, tmp_path)
    restart_roles = []
    written = {}

    def fake_run_on_node(node, command, _args, **_kwargs):
        if isinstance(command, str):
            return subprocess.CompletedProcess(command, 0, f"match {checksum}\n", "")
        if command[3] == "restart":
            restart_roles.append(node["role"])
            return subprocess.CompletedProcess(command, 0, "", "")
        if command[3] == "logs":
            marker = "Loaded Orion routing generation 7 for collection native_orion"
            return subprocess.CompletedProcess(command, 0, "", marker)
        raise AssertionError(command)

    def fake_inspect(_node, _name, _args):
        node = _node
        return {
            "Config": {
                "Labels": {
                    "orion.distributed.run_id": "run-1",
                    "orion.distributed.role": node["role"],
                    "orion.distributed.private_ip": node["private_ip"],
                }
            }
        }

    monkeypatch.setattr(
        module,
        "read_manifest",
        lambda *_args: {"schema_version": 1, "run_id": "run-1", "image": {}},
    )
    monkeypatch.setattr(
        module,
        "write_manifest",
        lambda _topology, _run_id, data: written.update(json.loads(json.dumps(data)))
        or tmp_path / "manifest.json",
    )
    monkeypatch.setattr(module, "run_on_node", fake_run_on_node)
    monkeypatch.setattr(module, "inspect_container", fake_inspect)
    monkeypatch.setattr(module, "wait_http_ready", lambda *_args: None)
    monkeypatch.setattr(
        module,
        "wait_cluster_healthy",
        lambda *_args: healthy_cluster_snapshot(module, value),
    )
    monkeypatch.setattr(
        module,
        "validate_collection_orion_policy",
        lambda *_args: {"policy": {"type": "orion"}, "status": "green"},
    )
    args = SimpleNamespace(
        run_id="run-1",
        collection="native_orion",
        generation=7,
        artifact=str(artifact),
        expected_sha256=checksum,
        restart="workers-first",
        wait_timeout=10.0,
        ssh_user=None,
        ssh_option=[],
        dry_run=False,
    )

    assert module.command_install_orion_artifact(args, value) == 0

    assert restart_roles == [
        "qdrant_shard_1",
        "qdrant_shard_2",
        "qdrant_shard_3",
        "controller",
    ]
    activation = written["orion_artifacts"][0]["activation"]
    assert activation["status"] == "activated_after_restart"
    assert len(activation["router_log_proof"]) == 4


def test_install_simple_kmeans_artifact_restart_verifies_all_router_load_markers(
    monkeypatch, tmp_path
):
    module = load_module()
    value = isolated_topology(module, tmp_path)
    artifact, checksum = write_simple_kmeans_artifact(module, tmp_path)
    restart_roles = []
    written = {}

    def fake_run_on_node(node, command, _args, **_kwargs):
        if isinstance(command, str):
            return subprocess.CompletedProcess(command, 0, f"match {checksum}\n", "")
        if command[3] == "restart":
            restart_roles.append(node["role"])
            return subprocess.CompletedProcess(command, 0, "", "")
        if command[3] == "logs":
            marker = (
                "Loaded Simple KMeans routing generation 7 for collection native_simple"
            )
            return subprocess.CompletedProcess(command, 0, marker, "")
        raise AssertionError(command)

    def fake_inspect(node, _name, _args):
        return {
            "Config": {
                "Labels": {
                    "orion.distributed.run_id": "run-1",
                    "orion.distributed.role": node["role"],
                    "orion.distributed.private_ip": node["private_ip"],
                }
            }
        }

    monkeypatch.setattr(
        module,
        "read_manifest",
        lambda *_args: {"schema_version": 1, "run_id": "run-1", "image": {}},
    )
    monkeypatch.setattr(
        module,
        "write_manifest",
        lambda _topology, _run_id, data: written.update(json.loads(json.dumps(data)))
        or tmp_path / "manifest.json",
    )
    monkeypatch.setattr(module, "run_on_node", fake_run_on_node)
    monkeypatch.setattr(module, "inspect_container", fake_inspect)
    monkeypatch.setattr(module, "wait_http_ready", lambda *_args: None)
    monkeypatch.setattr(
        module,
        "wait_cluster_healthy",
        lambda *_args: healthy_cluster_snapshot(module, value),
    )
    monkeypatch.setattr(
        module,
        "validate_collection_simple_kmeans_policy",
        lambda *_args: {
            "policy_kind": "simple_kmeans",
            "policy": {"type": "simple_kmeans"},
            "status": "green",
        },
    )
    args = SimpleNamespace(
        run_id="run-1",
        collection="native_simple",
        generation=7,
        artifact=str(artifact),
        expected_sha256=checksum,
        restart="workers-first",
        wait_timeout=10.0,
        ssh_user=None,
        ssh_option=[],
        dry_run=False,
    )

    assert module.command_install_simple_kmeans_artifact(args, value) == 0
    assert restart_roles == [
        "qdrant_shard_1",
        "qdrant_shard_2",
        "qdrant_shard_3",
        "controller",
    ]
    entry = written["simple_kmeans_artifacts"][0]
    assert entry["policy_kind"] == "simple_kmeans"
    assert entry["activation"]["status"] == "activated_after_restart"
    assert len(entry["activation"]["router_log_proof"]) == 4
