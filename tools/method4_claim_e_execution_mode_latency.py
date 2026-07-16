#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import statistics
import time
import urllib.parse
from pathlib import Path
from typing import Any

import h5py
import numpy as np


DEFAULT_VARIANTS = [
    "grouped_by_ef_materialized",
    "compact_current",
    "client_shard_major_expanded",
]

VARIANT_DEFINITIONS: dict[str, dict[str, str]] = {
    "grouped_by_ef_materialized": {
        "routed_execution_mode": "grouped_by_ef",
        "lower_execution_order": "query_major",
        "description": "materialized JSON route plan grouped by EF",
    },
    "compact_current": {
        "routed_execution_mode": "compact_multi_ep",
        "lower_execution_order": "query_major",
        "description": "compact MultiEP request on the current server path",
    },
    "client_shard_major_expanded": {
        "routed_execution_mode": "compact_multi_ep",
        "lower_execution_order": "shard_major",
        "description": "client-expanded single-shard negative control",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Claim E current-runtime execution-mode batch-latency matrix. "
            "Compares materialized grouped-by-EF JSON route plans, compact "
            "current server execution, and client-expanded shard-major negative control."
        )
    )
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
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--variant", action="append", default=None, choices=DEFAULT_VARIANTS)
    parser.add_argument("--sample-denominator", type=int, default=32)
    parser.add_argument("--upper-sample-seed", type=int, default=100)
    parser.add_argument("--upper-m", type=int, default=32)
    parser.add_argument("--upper-ef-construction", type=int, default=100)
    parser.add_argument("--upper-search-ef", type=int, default=200)
    parser.add_argument("--scroll-page-size", type=int, default=20000)
    parser.add_argument("--output-root", default="results/method4_claim_e_execution_mode_latency_20260704")
    parser.add_argument("--qdrant-tool", default="tools/qdrant_two_level_routing_experiment.py")
    return parser.parse_args()


def load_qdrant_tool(path: str | Path) -> Any:
    module_path = Path(path)
    spec = importlib.util.spec_from_file_location("qdrant_two_level_routing_experiment", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def variant_specs(values: list[str] | None) -> list[dict[str, str]]:
    selected = values or DEFAULT_VARIANTS
    specs: list[dict[str, str]] = []
    for variant in selected:
        if variant not in VARIANT_DEFINITIONS:
            raise ValueError(f"unsupported variant: {variant}")
        specs.append({"variant": variant, **VARIANT_DEFINITIONS[variant]})
    return specs


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * (float(pct) / 100.0)
    low = int(np.floor(rank))
    high = int(np.ceil(rank))
    if low == high:
        return ordered[low]
    weight = rank - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize_variant_rows(
    variant: str,
    rows: list[dict[str, Any]],
    top_k: int,
) -> dict[str, Any]:
    query_count = sum(int(row["query_count"]) for row in rows)
    hits = sum(int(row["hits"]) for row in rows)
    wall_s = sum(float(row["wall_s"]) for row in rows)
    latencies = [float(row["batch_latency_ms"]) for row in rows]
    visited = sum(int(row["visited_shards"]) for row in rows)
    ef_sum = sum(int(row["assigned_ef_sum"]) for row in rows)
    search_requests = sum(int(row.get("search_request_count", 0)) for row in rows)
    candidate_groups = sum(int(row.get("candidate_group_count", 0)) for row in rows)
    returned_candidates = sum(int(row.get("returned_candidate_count", 0)) for row in rows)
    search_batch_calls = sum(int(row.get("search_batch_calls", 0)) for row in rows)
    return {
        "variant": variant,
        "query_count": query_count,
        "top_k": top_k,
        "recall_at_10": hits / (query_count * top_k) if query_count else 0.0,
        "qps": query_count / wall_s if wall_s > 0.0 else 0.0,
        "wall_s": wall_s,
        "avg_visited_shards": visited / query_count if query_count else 0.0,
        "avg_assigned_ef_sum": ef_sum / query_count if query_count else 0.0,
        "avg_search_requests_per_query": search_requests / query_count if query_count else 0.0,
        "avg_candidate_groups_per_query": candidate_groups / query_count if query_count else 0.0,
        "avg_returned_candidates_per_query": returned_candidates / query_count if query_count else 0.0,
        "search_batch_calls": search_batch_calls,
        "batch_count": len(rows),
        "batch_latency_mean_ms": statistics.mean(latencies) if latencies else 0.0,
        "batch_latency_p50_ms": percentile(latencies, 50),
        "batch_latency_p95_ms": percentile(latencies, 95),
        "batch_latency_p99_ms": percentile(latencies, 99),
        "batch_latency_max_ms": max(latencies) if latencies else 0.0,
    }


def failure_summary_row(
    spec: dict[str, str],
    batch_size: int,
    repeat: int,
    query_count: int,
    top_k: int,
    collection: str,
    upper_k: int,
    base_ef: int,
    factor: int,
    warmup_query_count: int,
    error: BaseException,
) -> dict[str, Any]:
    return {
        "variant": spec["variant"],
        "query_count": query_count,
        "top_k": top_k,
        "recall_at_10": "",
        "qps": "",
        "wall_s": "",
        "avg_visited_shards": "",
        "avg_assigned_ef_sum": "",
        "avg_search_requests_per_query": "",
        "avg_candidate_groups_per_query": "",
        "avg_returned_candidates_per_query": "",
        "search_batch_calls": "",
        "batch_count": "",
        "batch_latency_mean_ms": "",
        "batch_latency_p50_ms": "",
        "batch_latency_p95_ms": "",
        "batch_latency_p99_ms": "",
        "batch_latency_max_ms": "",
        "description": spec["description"],
        "routed_execution_mode": spec["routed_execution_mode"],
        "lower_execution_order": spec["lower_execution_order"],
        "collection": collection,
        "repeat": repeat,
        "batch_size": batch_size,
        "upper_k": upper_k,
        "base_ef": base_ef,
        "factor": factor,
        "warmup_query_count": warmup_query_count,
        "status": "error",
        "error_type": type(error).__name__,
        "error_message": str(error),
    }


def load_queries_and_neighbors(args: argparse.Namespace, q2l: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    total_queries = args.warmup_query_count + args.eval_query_count
    with h5py.File(args.hdf5_path, "r") as handle:
        train_count = int(handle["train"].shape[0])
        dim = int(handle["train"].shape[1])
        queries = handle["test"][:total_queries].astype(np.float32, copy=True)
        neighbors = handle["neighbors"][:total_queries, : args.top_k].astype(np.int64, copy=True)
        upper_indices = q2l.global_upper_indices(train_count, args.sample_denominator, args.upper_sample_seed)
        order = np.argsort(upper_indices)
        sorted_upper = upper_indices[order]
        upper_vectors_sorted = handle["train"][sorted_upper].astype(np.float32, copy=True)
        inverse = np.empty_like(order)
        inverse[order] = np.arange(len(order))
        upper_vectors = upper_vectors_sorted[inverse]
    return q2l.normalize_rows(queries), q2l.normalize_rows(upper_vectors), neighbors, train_count, dim


def recover_upper_membership(
    args: argparse.Namespace,
    q2l: Any,
    routing_source_collection: str,
    upper_indices: np.ndarray,
    train_count: int,
) -> list[list[int]]:
    upper_set = {int(point_id) for point_id in upper_indices.tolist()}
    point_to_shards: list[list[int]] = [[] for _ in range(train_count)]
    for shard_id in range(args.num_shards):
        shard_key = q2l.shard_key_for_id(shard_id)
        offset: Any = None
        scanned = 0
        while True:
            body: dict[str, Any] = {
                "limit": args.scroll_page_size,
                "with_payload": ["source_id"],
                "with_vector": False,
                "shard_key": shard_key,
            }
            if offset is not None:
                body["offset"] = offset
            result = q2l.request_json(
                args.base_url,
                "POST",
                f"/collections/{urllib.parse.quote(routing_source_collection, safe='')}/points/scroll",
                body=body,
                timeout=300.0,
            )["result"]
            points = result.get("points") or []
            scanned += len(points)
            for point in points:
                source_id = q2l.source_id_from_scrolled_point(point, train_count)
                if source_id in upper_set:
                    point_to_shards[int(source_id)].append(int(shard_id))
            offset = result.get("next_page_offset")
            if offset is None:
                break
        print(f"recovered {routing_source_collection} shard {shard_id}: scanned {scanned}", flush=True)
    missing = [int(point_id) for point_id in upper_indices.tolist() if not point_to_shards[int(point_id)]]
    if missing:
        preview = ", ".join(str(point_id) for point_id in missing[:10])
        raise RuntimeError(f"missing upper shard membership for {len(missing)} points; first: {preview}")
    return point_to_shards


def build_plans_for_execution_mode(
    q2l: Any,
    queries: np.ndarray,
    upper_index: Any,
    point_to_shards: list[list[int]],
    num_shards: int,
    top_k: int,
    upper_k: int,
    base_ef: int,
    factor: int,
    routed_execution_mode: str,
    source_id_dedup_block_size: int,
) -> list[dict[str, Any]]:
    upper_labels = q2l.compute_upper_labels(upper_index, queries, upper_k)
    return q2l.build_routed_search_plans(
        queries,
        upper_labels.astype(np.int64, copy=False),
        point_to_shards,
        num_shards,
        top_k,
        base_ef,
        factor,
        False,
        True,
        routed_execution_mode=routed_execution_mode,
        compact_ef_mode="max",
        routed_result_limit_mode="top_k",
        routed_result_limit_multiplier=1,
        source_id_dedup_block_size=source_id_dedup_block_size,
    )


def run_variant_batches(
    args: argparse.Namespace,
    q2l: Any,
    spec: dict[str, str],
    batch_size: int,
    repeat: int,
    query_plans: list[dict[str, Any]],
    neighbors: np.ndarray,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    warmup = args.warmup_query_count
    if warmup:
        q2l.execute_query_plans_once(
            args.base_url,
            args.collection,
            query_plans[:warmup],
            neighbors[:warmup],
            args.top_k,
            lower_execution_order=spec["lower_execution_order"],
        )

    measured_plans = query_plans[warmup:]
    measured_neighbors = neighbors[warmup:]
    batch_rows: list[dict[str, Any]] = []
    for batch_index, start_idx in enumerate(range(0, len(measured_plans), batch_size)):
        end_idx = min(start_idx + batch_size, len(measured_plans))
        started = time.perf_counter()
        result = q2l.execute_query_plans_once(
            args.base_url,
            args.collection,
            measured_plans[start_idx:end_idx],
            measured_neighbors[start_idx:end_idx],
            args.top_k,
            lower_execution_order=spec["lower_execution_order"],
        )
        wall_s = time.perf_counter() - started
        batch_rows.append(
            {
                "variant": spec["variant"],
                "description": spec["description"],
                "routed_execution_mode": spec["routed_execution_mode"],
                "lower_execution_order": spec["lower_execution_order"],
                "collection": args.collection,
                "repeat": repeat,
                "batch_size": batch_size,
                "batch_index": batch_index,
                "query_start": start_idx,
                "query_end": end_idx,
                "query_count": int(result["query_count"]),
                "hits": int(result["hits"]),
                "wall_s": wall_s,
                "batch_latency_ms": wall_s * 1000.0,
                "visited_shards": int(result["visited_shards"]),
                "assigned_ef_sum": int(result["assigned_ef_sum"]),
                "search_batch_calls": int(result["search_batch_calls"]),
                "search_request_count": int(result.get("search_request_count", 0)),
                "candidate_group_count": int(result.get("candidate_group_count", 0)),
                "returned_candidate_count": int(result.get("returned_candidate_count", 0)),
            }
        )

    summary = summarize_variant_rows(spec["variant"], batch_rows, args.top_k)
    summary.update(
        {
            "status": "ok",
            "error_type": "",
            "error_message": "",
            "description": spec["description"],
            "routed_execution_mode": spec["routed_execution_mode"],
            "lower_execution_order": spec["lower_execution_order"],
            "collection": args.collection,
            "repeat": repeat,
            "batch_size": batch_size,
            "upper_k": args.upper_k,
            "base_ef": args.base_ef,
            "factor": args.factor,
            "warmup_query_count": warmup,
        }
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return summary, batch_rows


def main() -> int:
    args = parse_args()
    q2l = load_qdrant_tool(args.qdrant_tool)
    specs = variant_specs(args.variant)
    output_dir = Path(args.output_root) / time.strftime("analysis_%Y%m%d_%H%M%S", time.gmtime())
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"output_dir {output_dir}", flush=True)

    queries, upper_vectors, neighbors, train_count, dim = load_queries_and_neighbors(args, q2l)
    upper_indices = q2l.global_upper_indices(train_count, args.sample_denominator, args.upper_sample_seed)
    upper_index = q2l.build_upper_index(
        upper_vectors,
        upper_indices.astype(np.int64, copy=False),
        dim,
        args.upper_m,
        args.upper_ef_construction,
        max(args.upper_search_ef, args.upper_k),
    )
    routing_source = args.routing_source_collection or args.collection
    point_to_shards = recover_upper_membership(args, q2l, routing_source, upper_indices, train_count)

    plans_by_mode: dict[str, list[dict[str, Any]]] = {}
    for spec in specs:
        mode = spec["routed_execution_mode"]
        if mode in plans_by_mode:
            continue
        plans_by_mode[mode] = build_plans_for_execution_mode(
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

    summary_rows: list[dict[str, Any]] = []
    batch_rows: list[dict[str, Any]] = []
    for batch_size in args.batch_sizes:
        for repeat in range(1, args.repeats + 1):
            ordered_specs = specs if repeat % 2 == 1 else list(reversed(specs))
            for spec in ordered_specs:
                try:
                    summary, rows = run_variant_batches(
                        args,
                        q2l,
                        spec,
                        batch_size,
                        repeat,
                        plans_by_mode[spec["routed_execution_mode"]],
                        neighbors,
                    )
                    batch_rows.extend(rows)
                except Exception as exc:
                    summary = failure_summary_row(
                        spec,
                        batch_size=batch_size,
                        repeat=repeat,
                        query_count=args.eval_query_count,
                        top_k=args.top_k,
                        collection=args.collection,
                        upper_k=args.upper_k,
                        base_ef=args.base_ef,
                        factor=args.factor,
                        warmup_query_count=args.warmup_query_count,
                        error=exc,
                    )
                    print(json.dumps(summary, ensure_ascii=False), flush=True)
                summary_rows.append(summary)

    write_csv(output_dir / "claim_e_execution_mode_latency_summary.csv", summary_rows)
    write_csv(output_dir / "claim_e_execution_mode_latency_batches.csv", batch_rows)
    metadata = {
        "analysis_kind": "claim_e_current_runtime_execution_mode_batch_latency_matrix",
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
        "batch_sizes": args.batch_sizes,
        "repeats": args.repeats,
        "variants": specs,
        "notes": [
            "This is a current-runtime matrix. It does not recreate the pre-shard-major Qdrant binary.",
            "compact_current uses compact MultiEP requests on the currently running server path.",
            "client_shard_major_expanded is the client-side negative control that expands compact requests into per-logical-shard searches before sending the REST batch.",
            "grouped_by_ef_materialized is a materialized JSON route-plan baseline grouped by equal EF; it is not the historical old binary.",
        ],
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"wrote {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
