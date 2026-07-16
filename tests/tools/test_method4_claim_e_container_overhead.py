import importlib.util
from pathlib import Path

import pytest


def load_tool():
    path = Path(__file__).resolve().parents[2] / "tools" / "method4_claim_e_container_overhead.py"
    spec = importlib.util.spec_from_file_location("method4_claim_e_container_overhead", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_parse_docker_bytes_handles_common_units():
    tool = load_tool()

    assert tool.parse_docker_bytes("0B") == 0
    assert tool.parse_docker_bytes("126B") == 126
    assert tool.parse_docker_bytes("10.2kB") == pytest.approx(10_200)
    assert tool.parse_docker_bytes("1.5MB") == pytest.approx(1_500_000)
    assert tool.parse_docker_bytes("2GiB") == pytest.approx(2 * 1024**3)


def test_parse_net_io_splits_rx_and_tx_bytes():
    tool = load_tool()

    assert tool.parse_net_io("252GB / 181GB") == (252_000_000_000, 181_000_000_000)
    assert tool.parse_net_io("521kB / 126B") == (521_000, 126)


def test_parse_mem_usage_splits_usage_limit_and_percent():
    tool = load_tool()

    usage, limit, pct = tool.parse_mem_usage("1.072GiB / 503.5GiB", "0.21%")

    assert usage == pytest.approx(1.072 * 1024**3)
    assert limit == pytest.approx(503.5 * 1024**3)
    assert pct == pytest.approx(0.21)


def test_parse_docker_top_process_rss_sums_kib_values():
    tool = load_tool()
    top = {
        "Titles": ["PID", "PPID", "RSS", "COMMAND", "COMMAND"],
        "Processes": [
            ["10", "1", "1024", "qdrant", "./qdrant"],
            ["11", "10", "256", "helper", "helper --flag"],
        ],
    }

    parsed = tool.parse_docker_top_process_rss(top)

    assert parsed["process_count"] == 2
    assert parsed["process_rss_bytes"] == pytest.approx((1024 + 256) * 1024)


def test_summarize_container_samples_reports_controller_and_cluster_deltas():
    tool = load_tool()
    samples = [
        {
            "controller": {
                "cpu_pct": 10.0,
                "net_rx_bytes": 1000,
                "net_tx_bytes": 2000,
                "mem_usage_bytes": 100,
                "mem_limit_bytes": 1000,
                "mem_pct": 10.0,
                "process_rss_bytes": 80,
                "process_count": 1,
            },
            "shard1": {
                "cpu_pct": 20.0,
                "net_rx_bytes": 3000,
                "net_tx_bytes": 4000,
                "mem_usage_bytes": 200,
                "mem_limit_bytes": 1000,
                "mem_pct": 20.0,
                "process_rss_bytes": 160,
                "process_count": 2,
            },
        },
        {
            "controller": {
                "cpu_pct": 20.0,
                "net_rx_bytes": 1600,
                "net_tx_bytes": 2600,
                "mem_usage_bytes": 120,
                "mem_limit_bytes": 1000,
                "mem_pct": 12.0,
                "process_rss_bytes": 90,
                "process_count": 1,
            },
            "shard1": {
                "cpu_pct": 40.0,
                "net_rx_bytes": 4500,
                "net_tx_bytes": 6500,
                "mem_usage_bytes": 240,
                "mem_limit_bytes": 1000,
                "mem_pct": 24.0,
                "process_rss_bytes": 180,
                "process_count": 2,
            },
        },
    ]

    row = tool.summarize_container_samples(
        samples,
        controller_name="controller",
        container_names=["controller", "shard1"],
    )

    assert row["sample_count"] == 2
    assert row["controller_cpu_pct_avg"] == pytest.approx(15.0)
    assert row["controller_cpu_pct_max"] == pytest.approx(20.0)
    assert row["cluster_cpu_pct_avg_sum"] == pytest.approx(45.0)
    assert row["cluster_cpu_pct_max_sum"] == pytest.approx(60.0)
    assert row["controller_net_rx_bytes_delta"] == 600
    assert row["controller_net_tx_bytes_delta"] == 600
    assert row["cluster_net_rx_bytes_delta"] == 2100
    assert row["cluster_net_tx_bytes_delta"] == 3100
    assert row["controller_mem_usage_bytes_avg"] == pytest.approx(110)
    assert row["controller_mem_usage_bytes_max"] == pytest.approx(120)
    assert row["controller_mem_pct_avg"] == pytest.approx(11.0)
    assert row["cluster_mem_usage_bytes_avg_sum"] == pytest.approx(330)
    assert row["cluster_mem_usage_bytes_max_sum"] == pytest.approx(360)
    assert row["cluster_mem_limit_bytes_max_sum"] == pytest.approx(2000)
    assert row["controller_process_rss_bytes_avg"] == pytest.approx(85)
    assert row["controller_process_rss_bytes_max"] == pytest.approx(90)
    assert row["cluster_process_rss_bytes_avg_sum"] == pytest.approx(255)
    assert row["cluster_process_rss_bytes_max_sum"] == pytest.approx(270)
    assert row["cluster_process_count_max_sum"] == 3
