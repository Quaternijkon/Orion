import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest


def load_module():
    path = (
        Path(__file__).resolve().parents[2]
        / "experiments/c1/scripts/c1_simple_kmeans_virtual_scale.py"
    )
    name = "c1_simple_kmeans_virtual_scale_test"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_plain_requests_leave_routing_and_ef_to_server():
    module = load_module()
    bodies = module.make_simple_bodies(
        np.zeros((200, 200), dtype=np.float32), batch_size=200, top_k=10
    )
    payload = json.loads(bodies[0])
    assert len(payload["searches"]) == 200
    assert "params" not in payload["searches"][0]
    assert "shard_key" not in payload["searches"][0]


def test_profiles_are_plain_single_assignment_simple_kmeans():
    module = load_module()
    for logical_shards, profile in module.PROFILES.items():
        audit = module.audit_plain_simple_kmeans(profile)
        assert audit["status"] == "PASS"
        assert audit["logical_point_count"] == audit["physical_point_count"]
        assert audit["expansion_ratio"] == 1.0
        assert audit["nprobe"] <= logical_shards


def test_resource_schedule_matches_hashall_contract():
    module = load_module()
    assert module.hashall.quota_schedule(1) == [1.85, 0.05, 0.05, 0.05]
    assert module.hashall.quota_schedule(2) == [1.95, 1.95, 0.05, 0.05]
    assert module.hashall.quota_schedule(32) == [16.0, 16.0, 16.0, 16.0]


def test_parallel_prepare_and_measure_phases_are_explicit_and_exclusive():
    module = load_module()
    prepare = module.parse_args(
        ["--logical-shards", "2,4,8", "--prepare-only", "--parallel-prepares", "3"]
    )
    assert prepare.prepare_only is True
    assert prepare.measure_only is False
    assert prepare.parallel_prepares == 3

    measure = module.parse_args(["--logical-shards", "2,4,8", "--measure-only"])
    assert measure.prepare_only is False
    assert measure.measure_only is True

    with pytest.raises(SystemExit):
        module.parse_args(["--prepare-only", "--measure-only"])


def test_artifacts_can_be_installed_without_restart_then_activated():
    module = load_module()
    args = module.parse_args(["--logical-shards", "1"])
    install = module.artifact_installer_command(
        args, module.PROFILES[1], restart=False
    )
    activate = module.artifact_installer_command(
        args, module.PROFILES[1], restart=True
    )
    assert "install-simple-kmeans-artifact" in install
    assert "--restart" not in install
    assert activate[-2:] == ["--restart", "workers-first"]


def test_live_resource_placement_uses_actual_shard_owners(monkeypatch):
    module = load_module()
    per_node = {
        "10.10.1.1": {"local_shards": [{"shard_id": 0}]},
        "10.10.1.2": {"local_shards": []},
        "10.10.1.3": {"local_shards": [{"shard_id": 1}]},
        "10.10.1.4": {"local_shards": []},
    }
    monkeypatch.setattr(
        module.hashall,
        "per_node_collection_cluster",
        lambda *_args: per_node,
    )
    proof = module.live_numeric_shard_placement(object(), "example", 2)
    assert proof["shards_per_node"] == [1, 0, 1, 0]
    assert proof["shard_ids_per_node"]["10.10.1.3"] == [1]
