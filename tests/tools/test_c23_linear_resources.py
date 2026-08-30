from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "experiments"
    / "c23"
    / "scripts"
    / "c23_linear_resources.py"
)
SPEC = importlib.util.spec_from_file_location("c23_linear_resources", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
c23_linear_resources = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = c23_linear_resources
SPEC.loader.exec_module(c23_linear_resources)


@pytest.fixture()
def topology():
    return c23_linear_resources.load_topology()


@pytest.mark.parametrize("logical_shards", (1, 2, 4, 8, 16, 32))
def test_linear_resource_contract(topology, logical_shards):
    contract = c23_linear_resources.resource_contract(topology, logical_shards)

    assert contract["physical_core_equivalents"] == logical_shards
    assert contract["m32_capacity_fraction"] == logical_shards / 32
    assert contract["total_memory_capacity_bytes"] == (
        logical_shards * topology["memory_bytes_per_shard"]
    )
    assert set(contract["checks"].values()) == {"PASS"}


def test_m32_uses_eight_disjoint_cores_per_host(topology):
    placements = c23_linear_resources.shard_placements(topology, 32)

    for node_index in range(4):
        node_rows = [row for row in placements if row.node_index == node_index]
        assert len(node_rows) == 8
        assert {row.host_slot for row in node_rows} == set(range(8))
        assert {row.physical_core for row in node_rows} == set(range(8))
        cpus = [cpu for row in node_rows for cpu in row.cpuset.split(",")]
        assert len(cpus) == len(set(cpus)) == 16


def test_round_robin_mapping_and_ports_are_host_local(topology):
    placements = c23_linear_resources.shard_placements(topology, 16)

    for row in placements:
        assert row.node_index == row.shard_id % 4
        assert row.host_slot == row.shard_id // 4
        assert row.http_port == topology["http_base_port"] + row.host_slot
        assert row.grpc_port == topology["grpc_base_port"] + row.host_slot


def test_rejects_non_protocol_scale(topology):
    with pytest.raises(ValueError, match="logical_shards"):
        c23_linear_resources.shard_placements(topology, 3)


@pytest.mark.parametrize(
    ("value", "expected"),
    (
        ("stage13-smoke", "stage13-smoke"),
        ("abc_1.2", "abc_1.2"),
        ("UPPER", "upper"),
    ),
)
def test_sanitize_run_id(value, expected):
    assert c23_linear_resources.sanitize_run_id(value) == expected


@pytest.mark.parametrize("value", ("", "space here", "../escape"))
def test_rejects_unsafe_run_id(value):
    with pytest.raises(ValueError):
        c23_linear_resources.sanitize_run_id(value)
