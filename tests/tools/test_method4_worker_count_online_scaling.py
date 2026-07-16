from __future__ import annotations

import importlib.util
from pathlib import Path


def load_module():
    script_path = Path("/home/taig/dry/qdrant/tools/method4_worker_count_online_scaling.py")
    spec = importlib.util.spec_from_file_location("method4_worker_count_online_scaling", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_render_compose_uses_home_bind_mounts_and_requested_worker_count(tmp_path):
    module = load_module()

    compose = module.render_worker_count_compose(
        worker_count=2,
        image="qdrant/qdrant:method4-peer-premerge",
        host_http_port=6933,
        host_grpc_port=6934,
        shard_http_start=6943,
        shard_grpc_start=6944,
        storage_dir=tmp_path / "storage",
        cpu_base=24,
        controller_cpus=4,
        worker_cpus=4,
    )

    assert "qdrant_controller:" in compose
    assert "qdrant_shard_1:" in compose
    assert "qdrant_shard_2:" in compose
    assert "qdrant_shard_3:" not in compose
    assert str(tmp_path / "storage" / "controller") in compose
    assert str(tmp_path / "storage" / "shard_1") in compose
    assert str(tmp_path / "storage" / "shard_2") in compose
    assert "qdrant_controller_storage:" not in compose
    assert "volumes:\n  qdrant_controller_storage" not in compose


def test_worker_ports_are_stable_and_non_overlapping():
    module = load_module()

    assert module.ports_for_worker_count(1) == {
        "controller_http": 6933,
        "controller_grpc": 6934,
        "shard_http_start": 6943,
        "shard_grpc_start": 6944,
    }
    assert module.ports_for_worker_count(4) == {
        "controller_http": 7233,
        "controller_grpc": 7234,
        "shard_http_start": 7243,
        "shard_grpc_start": 7244,
    }


def test_root_owned_storage_cleanup_uses_temporary_container(tmp_path):
    module = load_module()

    command = module.root_owned_storage_cleanup_command(
        tmp_path / "docker_storage",
        image="qdrant/qdrant:method4-peer-premerge",
    )

    assert command[:4] == ["docker", "run", "--rm", "-v"]
    assert command[4] == f"{tmp_path / 'docker_storage'}:/cleanup"
    assert command[5:] == [
        "qdrant/qdrant:method4-peer-premerge",
        "bash",
        "-lc",
        "rm -rf /cleanup/* /cleanup/.[!.]* /cleanup/..?*",
    ]


def test_build_cluster_metadata_is_preserved_for_derived_tables():
    module = load_module()

    metadata = module.extract_build_cluster_metadata(
        {
            "num_shards": 46,
            "build": {"points_count": 1_183_514},
            "collection_cluster": {
                "cluster_peer_count": 2,
                "cluster_shard_count": 46,
                "cluster_active_shards": 46,
            },
        }
    )

    assert metadata == {
        "build_effective_num_shards": 46,
        "build_points_count": 1_183_514,
        "cluster_peer_count": 2,
        "cluster_shard_count": 46,
        "cluster_active_shards": 46,
    }
