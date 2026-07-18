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
            "peers": {
                str(100 + index): {"uri": module.advertised_uri(node, value)}
                for index, node in enumerate(module.all_nodes(value))
            },
        }
    }


def healthy_run_collections():
    return {
        "dist_run-1_orion_s3": {
            "info": {"status": "green"},
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


def matching_controller_inspect(module, value, disable_peer_premerge):
    node = value["controller"]
    image_id = "sha256:abc"
    mode = module.expected_peer_premerge_mode(node, disable_peer_premerge)
    labels = {
        "orion.distributed.run_id": "run-1",
        "orion.distributed.role": node["role"],
        "orion.distributed.private_ip": node["private_ip"],
        "orion.distributed.image_id": image_id,
        module.NOFILE_LABEL: module.expected_nofile_label(),
        module.PEER_PREMERGE_MODE_LABEL: mode,
        module.CONTROLLER_FINGERPRINT_LABEL: module.controller_config_fingerprint(
            node, "run-1", image_id, disable_peer_premerge
        ),
    }
    env = (
        [f"{module.PEER_PREMERGE_DISABLE_ENV}=1"]
        if disable_peer_premerge
        else []
    )
    return {
        "Name": "/orion-dist-run-1-controller",
        "Image": image_id,
        "Config": {"Labels": labels, "Env": env},
        "HostConfig": {
            "CpusetCpus": node["cpuset"],
            "Ulimits": [
                {
                    "Name": "nofile",
                    "Soft": module.CONTAINER_NOFILE_SOFT,
                    "Hard": module.CONTAINER_NOFILE_HARD,
                }
            ],
        },
        "State": {"Running": True},
    }


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


@pytest.mark.parametrize("command", ["deploy", "status", "manifest"])
def test_peer_premerge_cli_defaults_enabled_and_accepts_explicit_disable(
    monkeypatch, command
):
    module = load_module()

    monkeypatch.setattr(module.sys, "argv", ["cluster", "--run-id", "run-1", command])
    assert module.parse_args().disable_peer_premerge is False

    monkeypatch.setattr(
        module.sys,
        "argv",
        ["cluster", "--run-id", "run-1", command, "--disable-peer-premerge"],
    )
    assert module.parse_args().disable_peer_premerge is True


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
    monkeypatch.setattr(module, "inspect_container", lambda *_args: None)
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
        wait_timeout=1.0,
        ssh_user=None,
        ssh_option=[],
    )

    assert module.command_deploy(args, value) == 0
    assert image_inspects == 2
    assert ["sudo", "-n", "docker", "load", "--input", str(tar_path)] in commands
    assert written["nodes"][0]["image_id"] == image_id


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
