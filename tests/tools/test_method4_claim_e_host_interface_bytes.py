import importlib.util
from pathlib import Path

import pytest


def load_module():
    module_path = Path("tools/method4_claim_e_host_interface_bytes.py")
    spec = importlib.util.spec_from_file_location("method4_claim_e_host_interface_bytes", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_classify_interface_distinguishes_physical_docker_and_loopback():
    module = load_module()

    assert module.classify_interface("lo", has_device=False) == "loopback"
    assert module.classify_interface("ens6f0np0", has_device=True) == "physical_nic"
    assert module.classify_interface("docker0", has_device=False) == "docker_bridge"
    assert module.classify_interface("br-b9aac8010880", has_device=False) == "docker_bridge"
    assert module.classify_interface("veth58cae6d", has_device=False) == "docker_veth"
    assert module.classify_interface("netmaker", has_device=False) == "virtual_other"


def test_delta_rows_compute_bytes_per_query_and_preserve_roles():
    module = load_module()
    before = {
        "ens6f0np0": {
            "rx_bytes": 1000,
            "tx_bytes": 2000,
            "operstate": "up",
            "role": "physical_nic",
            "has_device": True,
        },
        "br-test": {
            "rx_bytes": 500,
            "tx_bytes": 700,
            "operstate": "up",
            "role": "docker_bridge",
            "has_device": False,
        },
    }
    after = {
        "ens6f0np0": {
            "rx_bytes": 1500,
            "tx_bytes": 2600,
            "operstate": "up",
            "role": "physical_nic",
            "has_device": True,
        },
        "br-test": {
            "rx_bytes": 900,
            "tx_bytes": 1500,
            "operstate": "up",
            "role": "docker_bridge",
            "has_device": False,
        },
    }

    rows = module.build_interface_delta_rows(
        before=before,
        after=after,
        context={
            "variant": "compact_current",
            "repeat": 1,
            "batch_size": 200,
            "query_count": 100,
            "wall_s": 2.0,
        },
    )

    by_iface = {row["interface"]: row for row in rows}
    assert by_iface["ens6f0np0"]["role"] == "physical_nic"
    assert by_iface["ens6f0np0"]["rx_delta_bytes"] == 500
    assert by_iface["ens6f0np0"]["tx_delta_bytes"] == 600
    assert by_iface["ens6f0np0"]["total_delta_bytes"] == 1100
    assert by_iface["ens6f0np0"]["total_bytes_per_query"] == pytest.approx(11.0)
    assert by_iface["br-test"]["role"] == "docker_bridge"
    assert by_iface["br-test"]["rx_delta_bytes"] == 400
    assert by_iface["br-test"]["tx_delta_bytes"] == 800
    assert by_iface["br-test"]["total_bytes_per_query"] == pytest.approx(12.0)


def test_summarize_role_deltas_groups_by_variant_and_role():
    module = load_module()
    rows = [
        {
            "variant": "compact_current",
            "repeat": 1,
            "batch_size": 200,
            "query_count": 100,
            "wall_s": 2.0,
            "interface": "ens6f0np0",
            "role": "physical_nic",
            "rx_delta_bytes": 10,
            "tx_delta_bytes": 20,
            "total_delta_bytes": 30,
            "total_bytes_per_query": 0.3,
        },
        {
            "variant": "compact_current",
            "repeat": 1,
            "batch_size": 200,
            "query_count": 100,
            "wall_s": 2.0,
            "interface": "br-test",
            "role": "docker_bridge",
            "rx_delta_bytes": 100,
            "tx_delta_bytes": 300,
            "total_delta_bytes": 400,
            "total_bytes_per_query": 4.0,
        },
        {
            "variant": "grouped_by_ef_materialized",
            "repeat": 1,
            "batch_size": 200,
            "query_count": 100,
            "wall_s": 4.0,
            "interface": "br-test",
            "role": "docker_bridge",
            "rx_delta_bytes": 200,
            "tx_delta_bytes": 600,
            "total_delta_bytes": 800,
            "total_bytes_per_query": 8.0,
        },
    ]

    summary = module.summarize_role_deltas(rows)

    by_key = {(row["variant"], row["role"]): row for row in summary}
    compact_bridge = by_key[("compact_current", "docker_bridge")]
    assert compact_bridge["interface_count"] == 1
    assert compact_bridge["total_delta_bytes"] == 400
    assert compact_bridge["total_bytes_per_query"] == pytest.approx(4.0)
    grouped_bridge = by_key[("grouped_by_ef_materialized", "docker_bridge")]
    assert grouped_bridge["total_bytes_per_query"] == pytest.approx(8.0)
    compact_physical = by_key[("compact_current", "physical_nic")]
    assert compact_physical["total_delta_bytes"] == 30
    assert compact_physical["total_bytes_per_query"] == pytest.approx(0.3)
