#!/usr/bin/env python3
"""Estimate serialized request payload bytes for Claim E execution modes.

This is a request-size accounting supplement for the Claim E execution-mode
matrix. It reconstructs the same routed search plans used by the latency runner
and measures the UTF-8 byte size of the JSON body that would be sent to
`/points/search/batch`.

It does not measure network framing, response bytes, controller CPU, or server
execution time.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


DEFAULT_VARIANTS = [
    "grouped_by_ef_materialized",
    "compact_current",
    "client_shard_major_expanded",
]


BATCH_FIELDNAMES = [
    "variant",
    "batch_size",
    "batch_index",
    "query_start",
    "query_end",
    "query_count",
    "lower_execution_order",
    "search_batch_calls",
    "search_request_count",
    "avg_search_requests_per_query",
    "request_payload_bytes",
    "request_payload_bytes_per_query",
    "visited_shards",
    "assigned_ef_sum",
]


SUMMARY_FIELDNAMES = [
    "variant",
    "batch_size",
    "batch_count",
    "query_count",
    "lower_execution_order",
    "search_batch_calls",
    "search_request_count",
    "avg_search_requests_per_query",
    "request_payload_bytes",
    "request_payload_bytes_per_query",
    "request_payload_bytes_per_batch_mean",
    "request_payload_bytes_per_batch_p95",
    "request_payload_bytes_per_batch_max",
    "request_count_reduction_vs_baseline_pct",
    "request_payload_bytes_reduction_vs_baseline_pct",
    "baseline_variant",
    "caveat",
]


CAVEAT = "serialized_json_request_body_only_not_network_or_response_bytes"


def load_module(path: str | Path, module_name: str) -> Any:
    module_path = Path(path)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def json_payload_bytes(payload: Any) -> int:
    return len(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    )


def shard_major_searches_for_query(search: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    shard_keys = [str(shard_key) for shard_key in search.get("shard_key") or []]
    if not shard_keys:
        return [("", dict(search))]

    entry_points_by_shard = search.get("hnsw_entry_points_by_shard")
    ef_by_shard = search.get("hnsw_ef_by_shard")
    expanded: list[tuple[str, dict[str, Any]]] = []
    for shard_key in shard_keys:
        single = dict(search)
        single["shard_key"] = [shard_key]

        if entry_points_by_shard is not None:
            single.pop("hnsw_entry_points_by_shard", None)
            entry_points = entry_points_by_shard.get(shard_key)
            if entry_points:
                single["hnsw_entry_points"] = list(entry_points)
            else:
                single.pop("hnsw_entry_points", None)

        if ef_by_shard is not None:
            single.pop("hnsw_ef_by_shard", None)
            if shard_key in ef_by_shard:
                params = dict(single.get("params") or {})
                params["hnsw_ef"] = int(ef_by_shard[shard_key])
                single["params"] = params

        expanded.append((shard_key, single))
    return expanded


def flatten_searches(
    plans: list[dict[str, Any]],
    lower_execution_order: str,
) -> list[dict[str, Any]]:
    if lower_execution_order not in {"query_major", "shard_major"}:
        raise ValueError(f"unsupported lower execution order: {lower_execution_order}")
    flat_searches: list[dict[str, Any]] = []
    for plan in plans:
        for search in plan["searches"]:
            if lower_execution_order == "shard_major":
                flat_searches.extend(
                    single_search
                    for _shard_key, single_search in shard_major_searches_for_query(search)
                )
            else:
                flat_searches.append(search)
    return flat_searches


def payload_batch_rows(
    plans: list[dict[str, Any]],
    *,
    variant: str,
    batch_size: int,
    lower_execution_order: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for batch_index, start_idx in enumerate(range(0, len(plans), batch_size)):
        end_idx = min(start_idx + batch_size, len(plans))
        batch_plans = plans[start_idx:end_idx]
        searches = flatten_searches(batch_plans, lower_execution_order)
        payload_bytes = json_payload_bytes({"searches": searches}) if searches else 0
        query_count = len(batch_plans)
        search_request_count = len(searches)
        rows.append(
            {
                "variant": variant,
                "batch_size": int(batch_size),
                "batch_index": int(batch_index),
                "query_start": int(start_idx),
                "query_end": int(end_idx),
                "query_count": int(query_count),
                "lower_execution_order": lower_execution_order,
                "search_batch_calls": 1 if searches else 0,
                "search_request_count": int(search_request_count),
                "avg_search_requests_per_query": (
                    search_request_count / query_count if query_count else 0.0
                ),
                "request_payload_bytes": int(payload_bytes),
                "request_payload_bytes_per_query": (
                    payload_bytes / query_count if query_count else 0.0
                ),
                "visited_shards": sum(int(plan.get("visited_shards", 0)) for plan in batch_plans),
                "assigned_ef_sum": sum(int(plan.get("assigned_ef_sum", 0)) for plan in batch_plans),
            }
        )
    return rows


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * pct / 100.0
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    if low == high:
        return ordered[low]
    weight = rank - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def reduction_pct(value: float, baseline: float | None) -> float | str:
    if baseline in (None, 0):
        return ""
    return (float(baseline) - float(value)) / float(baseline) * 100.0


def summarize_payload_rows(
    rows: list[dict[str, Any]],
    *,
    baseline_variant: str,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["batch_size"]), str(row["variant"]))].append(row)

    summaries: list[dict[str, Any]] = []
    baseline_by_batch: dict[int, dict[str, Any]] = {}
    for (batch_size, variant), group_rows in sorted(grouped.items()):
        query_count = sum(int(row["query_count"]) for row in group_rows)
        search_request_count = sum(int(row["search_request_count"]) for row in group_rows)
        payload_bytes = sum(int(row["request_payload_bytes"]) for row in group_rows)
        batch_bytes = [float(row["request_payload_bytes"]) for row in group_rows]
        summary = {
            "variant": variant,
            "batch_size": batch_size,
            "batch_count": len(group_rows),
            "query_count": query_count,
            "lower_execution_order": group_rows[0].get("lower_execution_order", ""),
            "search_batch_calls": sum(int(row.get("search_batch_calls", 1)) for row in group_rows),
            "search_request_count": search_request_count,
            "avg_search_requests_per_query": (
                search_request_count / query_count if query_count else 0.0
            ),
            "request_payload_bytes": payload_bytes,
            "request_payload_bytes_per_query": (
                payload_bytes / query_count if query_count else 0.0
            ),
            "request_payload_bytes_per_batch_mean": mean(batch_bytes) if batch_bytes else 0.0,
            "request_payload_bytes_per_batch_p95": percentile(batch_bytes, 95),
            "request_payload_bytes_per_batch_max": max(batch_bytes) if batch_bytes else 0.0,
            "request_count_reduction_vs_baseline_pct": "",
            "request_payload_bytes_reduction_vs_baseline_pct": "",
            "baseline_variant": baseline_variant,
            "caveat": CAVEAT,
        }
        if variant == baseline_variant:
            baseline_by_batch[batch_size] = summary
        summaries.append(summary)

    for summary in summaries:
        baseline = baseline_by_batch.get(int(summary["batch_size"]))
        summary["request_count_reduction_vs_baseline_pct"] = reduction_pct(
            float(summary["search_request_count"]),
            float(baseline["search_request_count"]) if baseline else None,
        )
        summary["request_payload_bytes_reduction_vs_baseline_pct"] = reduction_pct(
            float(summary["request_payload_bytes"]),
            float(baseline["request_payload_bytes"]) if baseline else None,
        )
    return summaries


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
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
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[50, 100, 200, 400])
    parser.add_argument("--variant", action="append", default=None, choices=DEFAULT_VARIANTS)
    parser.add_argument("--sample-denominator", type=int, default=32)
    parser.add_argument("--upper-sample-seed", type=int, default=100)
    parser.add_argument("--upper-m", type=int, default=32)
    parser.add_argument("--upper-ef-construction", type=int, default=100)
    parser.add_argument("--upper-search-ef", type=int, default=200)
    parser.add_argument("--scroll-page-size", type=int, default=20000)
    parser.add_argument("--baseline-variant", default="client_shard_major_expanded")
    parser.add_argument("--output-root", default="results/method4_claim_e_payload_bytes_20260705")
    parser.add_argument("--qdrant-tool", default="tools/qdrant_two_level_routing_experiment.py")
    parser.add_argument("--claim-e-tool", default="tools/method4_claim_e_execution_mode_latency.py")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    q2l = load_module(args.qdrant_tool, "qdrant_two_level_routing_experiment")
    claim_e = load_module(args.claim_e_tool, "method4_claim_e_execution_mode_latency")
    specs = claim_e.variant_specs(args.variant)
    output_dir = Path(args.output_root) / time.strftime("analysis_%Y%m%d_%H%M%S", time.gmtime())
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"output_dir {output_dir}", flush=True)

    queries, upper_vectors, _neighbors, train_count, dim = claim_e.load_queries_and_neighbors(args, q2l)
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
        if mode in plans_by_mode:
            continue
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
        )[args.warmup_query_count :]

    batch_rows: list[dict[str, Any]] = []
    for batch_size in args.batch_sizes:
        for spec in specs:
            batch_rows.extend(
                payload_batch_rows(
                    plans_by_mode[spec["routed_execution_mode"]],
                    variant=spec["variant"],
                    batch_size=batch_size,
                    lower_execution_order=spec["lower_execution_order"],
                )
            )

    summary_rows = summarize_payload_rows(batch_rows, baseline_variant=args.baseline_variant)
    write_csv(output_dir / "claim_e_payload_bytes_batches.csv", batch_rows, BATCH_FIELDNAMES)
    write_csv(output_dir / "claim_e_payload_bytes_summary.csv", summary_rows, SUMMARY_FIELDNAMES)
    metadata = {
        "analysis_kind": "claim_e_serialized_request_payload_bytes",
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
        "warmup_query_count_excluded_from_payload_summary": args.warmup_query_count,
        "batch_sizes": args.batch_sizes,
        "variants": specs,
        "baseline_variant": args.baseline_variant,
        "notes": [
            CAVEAT,
            "The script reconstructs request plans and serializes JSON payload bodies only; it does not send lower-search requests.",
            "Response bytes, HTTP framing, compression, controller CPU, and network interface counters are not measured.",
        ],
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"wrote {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
