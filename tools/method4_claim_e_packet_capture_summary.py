#!/usr/bin/env python3
"""Summarize Claim E packet-capture pcaps.

The tool reads classic pcap files captured on the Qdrant Docker bridge and
summarizes Ethernet frame bytes, IP bytes, and TCP payload bytes by traffic
direction. It is intentionally limited to Ethernet / IPv4 / TCP captures, which
matches the Claim E bridge capture command used in this workspace.
"""

from __future__ import annotations

import argparse
import csv
import socket
import struct
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


CAVEATS = {
    "docker_bridge_packet_capture": (
        "packet_capture_on_single_host_docker_bridge_not_qdrant_internal_or_physical_nic_attribution"
    ),
    "physical_nic_packet_capture": (
        "physical_nic_packet_capture_negative_control_for_docker_subnet_not_qdrant_internal_attribution"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Claim E packet-capture pcaps.")
    parser.add_argument(
        "--capture",
        action="append",
        required=True,
        help="Variant capture in the form variant=/path/to/file.pcap.",
    )
    parser.add_argument("--controller-ip", default="172.24.0.2")
    parser.add_argument("--worker-ip", action="append", default=["172.24.0.3", "172.24.0.4", "172.24.0.5"])
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--query-count", type=int, default=3000)
    parser.add_argument("--repeat", default="1")
    parser.add_argument("--bridge-interface", default="br-b9aac8010880")
    parser.add_argument("--capture-scope", default="docker_bridge_packet_capture")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def pcap_byte_order(global_header: bytes) -> tuple[str, int]:
    if len(global_header) != 24:
        raise ValueError("pcap global header must be 24 bytes")
    magic = global_header[:4]
    if magic == b"\xd4\xc3\xb2\xa1":
        endian = "<"
    elif magic == b"\xa1\xb2\xc3\xd4":
        endian = ">"
    else:
        raise ValueError(f"unsupported pcap magic: {magic.hex()}")
    _magic, _version_major, _version_minor, _thiszone, _sigfigs, _snaplen, network = struct.unpack(
        endian + "IHHIIII", global_header
    )
    return endian, int(network)


def ipv4_text(raw: bytes) -> str:
    return socket.inet_ntoa(raw)


def parse_ethernet_ipv4_tcp(frame: bytes, orig_len: int) -> dict[str, Any] | None:
    if len(frame) < 14:
        return None
    ethertype = struct.unpack("!H", frame[12:14])[0]
    if ethertype != 0x0800:
        return None

    ip_start = 14
    if len(frame) < ip_start + 20:
        return None
    version_ihl = frame[ip_start]
    version = version_ihl >> 4
    ihl = (version_ihl & 0x0F) * 4
    if version != 4 or ihl < 20:
        return None
    if len(frame) < ip_start + ihl:
        return None

    total_length = struct.unpack("!H", frame[ip_start + 2 : ip_start + 4])[0]
    protocol = frame[ip_start + 9]
    if protocol != 6:
        return None
    src_ip = ipv4_text(frame[ip_start + 12 : ip_start + 16])
    dst_ip = ipv4_text(frame[ip_start + 16 : ip_start + 20])

    tcp_start = ip_start + ihl
    tcp_payload_bytes = 0
    if len(frame) >= tcp_start + 20:
        data_offset = (frame[tcp_start + 12] >> 4) * 4
        if data_offset >= 20:
            tcp_payload_bytes = max(0, total_length - ihl - data_offset)

    return {
        "src_ip": src_ip,
        "dst_ip": dst_ip,
        "frame_bytes": int(orig_len),
        "ip_bytes": int(total_length),
        "tcp_payload_bytes": int(tcp_payload_bytes),
    }


def iter_pcap_packets(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("rb") as handle:
        endian, network = pcap_byte_order(handle.read(24))
        if network != 1:
            raise ValueError(f"unsupported pcap link type {network}; expected Ethernet link type 1")
        while True:
            record_header = handle.read(16)
            if not record_header:
                break
            if len(record_header) != 16:
                raise ValueError(f"truncated pcap record header in {path}")
            _ts_sec, _ts_usec, incl_len, orig_len = struct.unpack(endian + "IIII", record_header)
            frame = handle.read(incl_len)
            if len(frame) != incl_len:
                raise ValueError(f"truncated pcap record body in {path}")
            parsed = parse_ethernet_ipv4_tcp(frame, orig_len)
            if parsed is not None:
                yield parsed


def endpoint_group(ip: str, controller_ip: str, worker_ips: set[str]) -> str:
    if ip == controller_ip:
        return "controller"
    if ip in worker_ips:
        return "worker"
    return "other"


def direction_role(src_group: str, dst_group: str) -> str:
    if src_group == "controller" and dst_group == "worker":
        return "controller_to_worker"
    if src_group == "worker" and dst_group == "controller":
        return "worker_to_controller"
    if src_group == "worker" and dst_group == "worker":
        return "worker_to_worker"
    return f"{src_group}_to_{dst_group}"


def caveat_for_scope(capture_scope: str) -> str:
    return CAVEATS.get(capture_scope, f"{capture_scope}_not_qdrant_internal_attribution")


def summarize_pcap(
    path: Path,
    *,
    controller_ip: str,
    worker_ips: set[str],
    context: dict[str, Any],
    bridge_interface: str,
    capture_scope: str = "docker_bridge_packet_capture",
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], dict[str, Any]] = defaultdict(
        lambda: {
            "packet_count": 0,
            "frame_bytes": 0,
            "ip_bytes": 0,
            "tcp_payload_bytes": 0,
        }
    )
    for packet in iter_pcap_packets(path):
        src_group = endpoint_group(str(packet["src_ip"]), controller_ip, worker_ips)
        dst_group = endpoint_group(str(packet["dst_ip"]), controller_ip, worker_ips)
        role = direction_role(src_group, dst_group)
        row = grouped[(role, src_group, dst_group)]
        row["packet_count"] += 1
        row["frame_bytes"] += int(packet["frame_bytes"])
        row["ip_bytes"] += int(packet["ip_bytes"])
        row["tcp_payload_bytes"] += int(packet["tcp_payload_bytes"])

    query_count = int(context.get("query_count") or 0)
    rows: list[dict[str, Any]] = []
    caveat = caveat_for_scope(capture_scope)
    for (role, src_group, dst_group), values in sorted(grouped.items()):
        frame_bytes = int(values["frame_bytes"])
        ip_bytes = int(values["ip_bytes"])
        tcp_payload_bytes = int(values["tcp_payload_bytes"])
        rows.append(
            {
                "variant": context.get("variant", ""),
                "repeat": context.get("repeat", ""),
                "batch_size": context.get("batch_size", ""),
                "query_count": query_count,
                "capture_path": str(path),
                "bridge_interface": bridge_interface,
                "capture_scope": capture_scope,
                "direction_role": role,
                "src_group": src_group,
                "dst_group": dst_group,
                "packet_count": int(values["packet_count"]),
                "frame_bytes": frame_bytes,
                "ip_bytes": ip_bytes,
                "tcp_payload_bytes": tcp_payload_bytes,
                "frame_bytes_per_query": frame_bytes / query_count if query_count else 0.0,
                "ip_bytes_per_query": ip_bytes / query_count if query_count else 0.0,
                "tcp_payload_bytes_per_query": tcp_payload_bytes / query_count if query_count else 0.0,
                "caveat": caveat,
            }
        )
    if not rows:
        rows.append(
            {
                "variant": context.get("variant", ""),
                "repeat": context.get("repeat", ""),
                "batch_size": context.get("batch_size", ""),
                "query_count": query_count,
                "capture_path": str(path),
                "bridge_interface": bridge_interface,
                "capture_scope": capture_scope,
                "direction_role": "no_matching_packets",
                "src_group": "none",
                "dst_group": "none",
                "packet_count": 0,
                "frame_bytes": 0,
                "ip_bytes": 0,
                "tcp_payload_bytes": 0,
                "frame_bytes_per_query": 0.0,
                "ip_bytes_per_query": 0.0,
                "tcp_payload_bytes_per_query": 0.0,
                "caveat": caveat,
            }
        )
    return rows


def parse_capture(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"--capture must be variant=path, got {value!r}")
    variant, path = value.split("=", 1)
    if not variant:
        raise ValueError(f"--capture variant is empty in {value!r}")
    return variant, Path(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    rows: list[dict[str, Any]] = []
    worker_ips = set(args.worker_ip)
    for capture in args.capture:
        variant, path = parse_capture(capture)
        rows.extend(
            summarize_pcap(
                path,
                controller_ip=args.controller_ip,
                worker_ips=worker_ips,
                context={
                    "variant": variant,
                    "repeat": args.repeat,
                    "batch_size": args.batch_size,
                    "query_count": args.query_count,
                },
                bridge_interface=args.bridge_interface,
                capture_scope=args.capture_scope,
            )
        )
    write_csv(Path(args.output), rows)
    print(args.output)


if __name__ == "__main__":
    main()
