import importlib.util
import socket
import struct
from pathlib import Path

import pytest


def load_tool():
    path = Path(__file__).resolve().parents[2] / "tools" / "method4_claim_e_packet_capture_summary.py"
    spec = importlib.util.spec_from_file_location("method4_claim_e_packet_capture_summary", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def ethernet_ipv4_tcp_frame(src_ip: str, dst_ip: str, payload_len: int) -> bytes:
    ethernet = bytes.fromhex("00112233445566778899aabb0800")
    version_ihl = 0x45
    tos = 0
    total_length = 20 + 20 + payload_len
    identification = 0
    flags_fragment = 0
    ttl = 64
    protocol = 6
    checksum = 0
    ip_header = struct.pack(
        "!BBHHHBBH4s4s",
        version_ihl,
        tos,
        total_length,
        identification,
        flags_fragment,
        ttl,
        protocol,
        checksum,
        socket.inet_aton(src_ip),
        socket.inet_aton(dst_ip),
    )
    tcp_header = struct.pack(
        "!HHIIHHHH",
        12345,
        6335,
        1,
        1,
        5 << 12,
        1024,
        0,
        0,
    )
    return ethernet + ip_header + tcp_header + (b"x" * payload_len)


def write_pcap(path: Path, frames: list[bytes]) -> None:
    with path.open("wb") as handle:
        handle.write(struct.pack("<IHHIIII", 0xA1B2C3D4, 2, 4, 0, 0, 262144, 1))
        for index, frame in enumerate(frames):
            handle.write(struct.pack("<IIII", 1000 + index, index, len(frame), len(frame)))
            handle.write(frame)


def test_summarize_pcap_groups_controller_worker_directions(tmp_path):
    tool = load_tool()
    pcap = tmp_path / "claim_e.pcap"
    write_pcap(
        pcap,
        [
            ethernet_ipv4_tcp_frame("172.24.0.2", "172.24.0.3", payload_len=10),
            ethernet_ipv4_tcp_frame("172.24.0.3", "172.24.0.2", payload_len=20),
        ],
    )

    rows = tool.summarize_pcap(
        pcap,
        controller_ip="172.24.0.2",
        worker_ips={"172.24.0.3", "172.24.0.4", "172.24.0.5"},
        context={"variant": "compact_current", "repeat": 1, "batch_size": 200, "query_count": 100},
        bridge_interface="br-test",
    )

    by_role = {row["direction_role"]: row for row in rows}
    controller_to_worker = by_role["controller_to_worker"]
    assert controller_to_worker["packet_count"] == 1
    assert controller_to_worker["frame_bytes"] == 64
    assert controller_to_worker["ip_bytes"] == 50
    assert controller_to_worker["tcp_payload_bytes"] == 10
    assert controller_to_worker["frame_bytes_per_query"] == pytest.approx(0.64)

    worker_to_controller = by_role["worker_to_controller"]
    assert worker_to_controller["packet_count"] == 1
    assert worker_to_controller["frame_bytes"] == 74
    assert worker_to_controller["ip_bytes"] == 60
    assert worker_to_controller["tcp_payload_bytes"] == 20
    assert worker_to_controller["frame_bytes_per_query"] == pytest.approx(0.74)
    assert all(row["capture_scope"] == "docker_bridge_packet_capture" for row in rows)


def test_summarize_empty_physical_nic_capture_emits_zero_negative_control_row(tmp_path):
    tool = load_tool()
    pcap = tmp_path / "empty_physical_nic.pcap"
    write_pcap(pcap, [])

    rows = tool.summarize_pcap(
        pcap,
        controller_ip="172.24.0.2",
        worker_ips={"172.24.0.3", "172.24.0.4", "172.24.0.5"},
        context={"variant": "compact_current", "repeat": "1-3", "batch_size": 200, "query_count": 9000},
        bridge_interface="ens6f0np0",
        capture_scope="physical_nic_packet_capture",
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["capture_scope"] == "physical_nic_packet_capture"
    assert row["direction_role"] == "no_matching_packets"
    assert row["packet_count"] == 0
    assert row["frame_bytes"] == 0
    assert row["tcp_payload_bytes"] == 0
    assert row["frame_bytes_per_query"] == 0.0
    assert "physical_nic" in row["caveat"]
