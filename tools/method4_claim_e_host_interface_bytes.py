#!/usr/bin/env python3
"""Measure Claim E execution modes with host interface byte counters.

This is a host-interface supplement for Claim E. It reads Linux
`/sys/class/net/*/statistics/{rx,tx}_bytes` before and after each measured
variant run and groups deltas by interface role.

On the current single-host Docker deployment, Qdrant inter-container traffic is
expected to appear mainly on Docker bridge/veth interfaces. Physical NIC deltas
therefore should be interpreted only as host external-interface counters, not as
Qdrant-internal network attribution.
"""

from __future__ import annotations

import argparse
import copy
import csv
import importlib.util
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


CAVEAT = "host_interface_counters_not_qdrant_internal_or_packet_capture"


def load_module(path: str | Path, module_name: str) -> Any:
    module_path = Path(path)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def classify_interface(name: str, *, has_device: bool) -> str:
    if name == "lo":
        return "loopback"
    if name == "docker0" or name.startswith("br-"):
        return "docker_bridge"
    if name.startswith("veth"):
        return "docker_veth"
    if has_device:
        return "physical_nic"
    return "virtual_other"


def read_int(path: Path, default: int = 0) -> int:
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except FileNotFoundError:
        return default


def read_text(path: Path, default: str = "") -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return default


def read_interface_counters(
    sys_class_net: Path = Path("/sys/class/net"),
    interface_names: list[str] | None = None,
) -> dict[str, dict[str, Any]]:
    names = interface_names or sorted(path.name for path in sys_class_net.iterdir() if path.is_dir())
    counters: dict[str, dict[str, Any]] = {}
    for name in names:
        interface_dir = sys_class_net / name
        has_device = (interface_dir / "device").exists()
        operstate = read_text(interface_dir / "operstate", "unknown")
        counters[name] = {
            "rx_bytes": read_int(interface_dir / "statistics" / "rx_bytes"),
            "tx_bytes": read_int(interface_dir / "statistics" / "tx_bytes"),
            "operstate": operstate,
            "role": classify_interface(name, has_device=has_device),
            "has_device": has_device,
        }
    return counters


def build_interface_delta_rows(
    *,
    before: dict[str, dict[str, Any]],
    after: dict[str, dict[str, Any]],
    context: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    query_count = int(context.get("query_count") or 0)
    for name in sorted(set(before) | set(after)):
        start = before.get(name, {})
        end = after.get(name, {})
        rx_delta = max(0, int(end.get("rx_bytes", 0)) - int(start.get("rx_bytes", 0)))
        tx_delta = max(0, int(end.get("tx_bytes", 0)) - int(start.get("tx_bytes", 0)))
        total_delta = rx_delta + tx_delta
        role = str(end.get("role") or start.get("role") or "unknown")
        rows.append(
            {
                "variant": context.get("variant", ""),
                "repeat": context.get("repeat", ""),
                "batch_size": context.get("batch_size", ""),
                "query_count": query_count,
                "wall_s": float(context.get("wall_s") or 0.0),
                "interface": name,
                "role": role,
                "operstate_before": start.get("operstate", ""),
                "operstate_after": end.get("operstate", ""),
                "has_device": bool(end.get("has_device", start.get("has_device", False))),
                "rx_before_bytes": int(start.get("rx_bytes", 0)),
                "rx_after_bytes": int(end.get("rx_bytes", 0)),
                "rx_delta_bytes": rx_delta,
                "tx_before_bytes": int(start.get("tx_bytes", 0)),
                "tx_after_bytes": int(end.get("tx_bytes", 0)),
                "tx_delta_bytes": tx_delta,
                "total_delta_bytes": total_delta,
                "total_bytes_per_query": total_delta / query_count if query_count else 0.0,
                "caveat": CAVEAT,
            }
        )
    return rows


def summarize_role_deltas(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, Any, Any, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (row["variant"], row["repeat"], row["batch_size"], row["role"])
        grouped[key].append(row)

    summary_rows: list[dict[str, Any]] = []
    for (variant, repeat, batch_size, role), group_rows in sorted(grouped.items()):
        query_count = max(int(row.get("query_count") or 0) for row in group_rows)
        wall_s = max(float(row.get("wall_s") or 0.0) for row in group_rows)
        rx_delta = sum(int(row["rx_delta_bytes"]) for row in group_rows)
        tx_delta = sum(int(row["tx_delta_bytes"]) for row in group_rows)
        total_delta = rx_delta + tx_delta
        summary_rows.append(
            {
                "variant": variant,
                "repeat": repeat,
                "batch_size": batch_size,
                "role": role,
                "interface_count": len(group_rows),
                "interfaces": ";".join(str(row["interface"]) for row in group_rows),
                "query_count": query_count,
                "wall_s": wall_s,
                "rx_delta_bytes": rx_delta,
                "tx_delta_bytes": tx_delta,
                "total_delta_bytes": total_delta,
                "total_bytes_per_query": total_delta / query_count if query_count else 0.0,
                "bytes_per_second": total_delta / wall_s if wall_s > 0 else 0.0,
                "caveat": CAVEAT,
            }
        )
    return summary_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:6833")
    parser.add_argument("--collection", default="qdrant_controller_idea_method4map_full_20260601")
    parser.add_argument("--routing-source-collection", default=None)
    parser.add_argument("--hdf5-path", default="/home/taig/dry/faiss/datasets/glove-200-angular.hdf5")
    parser.add_argument("--num-shards", type=int, default=46)
    parser.add_argument("--upper-k", type=int, default=160)
    parser.add_argument("--base-ef", type=int, default=80)
    parser.add_argument("--factor", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--eval-query-count", type=int, default=3000)
    parser.add_argument("--warmup-query-count", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--variant",
        action="append",
        default=None,
        choices=["grouped_by_ef_materialized", "compact_current", "client_shard_major_expanded"],
    )
    parser.add_argument("--sample-denominator", type=int, default=32)
    parser.add_argument("--upper-sample-seed", type=int, default=100)
    parser.add_argument("--upper-m", type=int, default=32)
    parser.add_argument("--upper-ef-construction", type=int, default=100)
    parser.add_argument("--upper-search-ef", type=int, default=200)
    parser.add_argument("--scroll-page-size", type=int, default=20000)
    parser.add_argument("--interface", action="append", default=None)
    parser.add_argument("--sys-class-net", default="/sys/class/net")
    parser.add_argument("--output-root", default="results/method4_claim_e_host_interface_bytes_20260705")
    parser.add_argument("--qdrant-tool", default="tools/qdrant_two_level_routing_experiment.py")
    parser.add_argument("--claim-e-tool", default="tools/method4_claim_e_execution_mode_latency.py")
    return parser.parse_args()


def run_variant_with_interface_counters(
    *,
    args: argparse.Namespace,
    q2l: Any,
    claim_e: Any,
    spec: dict[str, str],
    plans: list[dict[str, Any]],
    neighbors: Any,
    repeat: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    warmup = int(args.warmup_query_count)
    if warmup:
        q2l.execute_query_plans_once(
            args.base_url,
            args.collection,
            plans[:warmup],
            neighbors[:warmup],
            args.top_k,
            lower_execution_order=spec["lower_execution_order"],
        )

    measured_args = copy.copy(args)
    measured_args.warmup_query_count = 0
    before = read_interface_counters(Path(args.sys_class_net), args.interface)
    summary, batch_rows = claim_e.run_variant_batches(
        measured_args,
        q2l,
        spec,
        args.batch_size,
        repeat,
        plans[warmup:],
        neighbors[warmup:],
    )
    after = read_interface_counters(Path(args.sys_class_net), args.interface)
    summary["warmup_query_count"] = warmup
    interface_rows = build_interface_delta_rows(
        before=before,
        after=after,
        context={
            "variant": spec["variant"],
            "repeat": repeat,
            "batch_size": args.batch_size,
            "query_count": summary["query_count"],
            "wall_s": summary["wall_s"],
        },
    )
    return summary, batch_rows, interface_rows


def main() -> int:
    args = parse_args()
    q2l = load_module(args.qdrant_tool, "qdrant_two_level_routing_experiment")
    claim_e = load_module(args.claim_e_tool, "method4_claim_e_execution_mode_latency")
    specs = claim_e.variant_specs(args.variant)
    output_dir = Path(args.output_root) / time.strftime("analysis_%Y%m%d_%H%M%S", time.gmtime())
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"output_dir {output_dir}", flush=True)

    queries, upper_vectors, neighbors, train_count, dim = claim_e.load_queries_and_neighbors(args, q2l)
    upper_indices = q2l.global_upper_indices(train_count, args.sample_denominator, args.upper_sample_seed)
    upper_index = q2l.build_upper_index(
        upper_vectors,
        upper_indices.astype("int64", copy=False),
        dim,
        args.upper_m,
        args.upper_ef_construction,
        max(args.upper_search_ef, args.upper_k),
    )
    routing_source = args.routing_source_collection or args.collection
    point_to_shards = claim_e.recover_upper_membership(args, q2l, routing_source, upper_indices, train_count)

    plans_by_mode: dict[str, list[dict[str, Any]]] = {}
    for spec in specs:
        mode = spec["routed_execution_mode"]
        if mode not in plans_by_mode:
            plans_by_mode[mode] = claim_e.build_plans_for_execution_mode(
                q2l,
                queries,
                upper_index,
                point_to_shards,
                args.num_shards,
                args.top_k,
                args.upper_k,
                args.base_ef,
                args.factor,
                mode,
                train_count + 1,
            )

    latency_rows: list[dict[str, Any]] = []
    batch_rows: list[dict[str, Any]] = []
    interface_rows: list[dict[str, Any]] = []
    for repeat in range(1, args.repeats + 1):
        ordered_specs = specs if repeat % 2 == 1 else list(reversed(specs))
        for spec in ordered_specs:
            summary, batches, interfaces = run_variant_with_interface_counters(
                args=args,
                q2l=q2l,
                claim_e=claim_e,
                spec=spec,
                plans=plans_by_mode[spec["routed_execution_mode"]],
                neighbors=neighbors,
                repeat=repeat,
            )
            latency_rows.append(summary)
            batch_rows.extend(batches)
            interface_rows.extend(interfaces)

    role_summary_rows = summarize_role_deltas(interface_rows)
    write_csv(output_dir / "claim_e_host_interface_latency_summary.csv", latency_rows)
    write_csv(output_dir / "claim_e_host_interface_latency_batches.csv", batch_rows)
    write_csv(output_dir / "claim_e_host_interface_bytes_interfaces.csv", interface_rows)
    write_csv(output_dir / "claim_e_host_interface_bytes_summary.csv", role_summary_rows)
    metadata = {
        "analysis_kind": "claim_e_host_interface_byte_counter_supplement",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "base_url": args.base_url,
        "collection": args.collection,
        "routing_source_collection": routing_source,
        "hdf5_path": args.hdf5_path,
        "num_points": train_count,
        "num_shards": args.num_shards,
        "upper_k": args.upper_k,
        "base_ef": args.base_ef,
        "factor": args.factor,
        "top_k": args.top_k,
        "eval_query_count": args.eval_query_count,
        "warmup_query_count": args.warmup_query_count,
        "batch_size": args.batch_size,
        "repeats": args.repeats,
        "variants": specs,
        "interfaces": args.interface or "all",
        "sys_class_net": args.sys_class_net,
        "notes": [
            CAVEAT,
            "Counters are Linux host interface byte counters from /sys/class/net.",
            "Single-host Docker traffic is expected to appear mainly on docker_bridge and docker_veth roles.",
            "physical_nic rows represent host external NIC counters, not Qdrant-internal remote-worker traffic attribution.",
        ],
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"wrote {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
