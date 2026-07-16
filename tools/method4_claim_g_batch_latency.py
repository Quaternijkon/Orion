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


DEFAULT_COLLECTIONS = [
    "round_robin=qdrant_controller_idea_full_20260521",
    "size_balanced=qdrant_controller_idea_matched_sizebalanced_20260704",
    "method4_aware=qdrant_controller_idea_matched_method4map_20260704",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Claim G online batch-size latency matrix for matched physical layouts."
        )
    )
    parser.add_argument("--base-url", default="http://localhost:6833")
    parser.add_argument("--routing-source-collection", default="qdrant_controller_idea_full_20260521")
    parser.add_argument("--collection", action="append", default=None, help="label=collection_name; repeatable")
    parser.add_argument("--hdf5-path", default="/home/taig/dry/faiss/datasets/glove-200-angular.hdf5")
    parser.add_argument("--num-shards", type=int, default=46)
    parser.add_argument("--upper-k", type=int, default=160)
    parser.add_argument("--base-ef", type=int, default=80)
    parser.add_argument("--factor", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--eval-query-count", type=int, default=3000)
    parser.add_argument("--warmup-query-count", type=int, default=100)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[100, 200])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--sample-denominator", type=int, default=32)
    parser.add_argument("--upper-sample-seed", type=int, default=100)
    parser.add_argument("--upper-m", type=int, default=32)
    parser.add_argument("--upper-ef-construction", type=int, default=100)
    parser.add_argument("--upper-search-ef", type=int, default=160)
    parser.add_argument("--scroll-page-size", type=int, default=20000)
    parser.add_argument("--output-root", default="results/method4_claim_g_batch_latency_20260704")
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


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_collections(values: list[str] | None) -> list[tuple[str, str]]:
    specs = values or DEFAULT_COLLECTIONS
    parsed: list[tuple[str, str]] = []
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"collection spec must be label=name, got {spec!r}")
        label, collection = spec.split("=", 1)
        if not label or not collection:
            raise ValueError(f"collection spec must be label=name, got {spec!r}")
        parsed.append((label, collection))
    return parsed


def load_upper_eval_and_neighbors(
    args: argparse.Namespace,
    q2l: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    total_queries = args.warmup_query_count + args.eval_query_count
    with h5py.File(args.hdf5_path, "r") as handle:
        train_ds = handle["train"]
        num_points = int(train_ds.shape[0])
        dim = int(train_ds.shape[1])
        upper_indices = q2l.global_upper_indices(num_points, args.sample_denominator, args.upper_sample_seed)

        order = np.argsort(upper_indices)
        sorted_upper = upper_indices[order]
        upper_vectors_sorted = train_ds[sorted_upper].astype(np.float32, copy=True)
        inverse = np.empty_like(order)
        inverse[order] = np.arange(len(order))
        upper_vectors = upper_vectors_sorted[inverse]

        queries = handle["test"][:total_queries].astype(np.float32, copy=True)
        neighbors = handle["neighbors"][:total_queries, : args.top_k].astype(np.int64, copy=True)

    return q2l.normalize_rows(upper_vectors), q2l.normalize_rows(queries), neighbors, num_points, dim


def recover_upper_membership(
    args: argparse.Namespace,
    q2l: Any,
    upper_indices: np.ndarray,
    num_points: int,
) -> list[list[int]]:
    upper_set = {int(point_id) for point_id in upper_indices.tolist()}
    point_to_shards: list[list[int]] = [[] for _ in range(num_points)]
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
                f"/collections/{urllib.parse.quote(args.routing_source_collection, safe='')}/points/scroll",
                body=body,
                timeout=300.0,
            )["result"]
            points = result.get("points") or []
            scanned += len(points)
            for point in points:
                source_id = q2l.source_id_from_scrolled_point(point, num_points)
                if source_id in upper_set:
                    point_to_shards[int(source_id)].append(int(shard_id))
            offset = result.get("next_page_offset")
            if offset is None:
                break
        print(f"recovered routing shard {shard_id}: scanned {scanned}", flush=True)
    missing = [int(point_id) for point_id in upper_indices.tolist() if not point_to_shards[int(point_id)]]
    if missing:
        preview = ", ".join(str(point_id) for point_id in missing[:10])
        raise RuntimeError(f"missing upper shard membership for {len(missing)} points; first: {preview}")
    return point_to_shards


def summarize_latencies(latencies_ms: list[float], q2l: Any) -> dict[str, float]:
    return {
        "batch_latency_mean_ms": statistics.mean(latencies_ms) if latencies_ms else 0.0,
        "batch_latency_p50_ms": q2l.percentile(latencies_ms, 50),
        "batch_latency_p95_ms": q2l.percentile(latencies_ms, 95),
        "batch_latency_p99_ms": q2l.percentile(latencies_ms, 99),
        "batch_latency_max_ms": max(latencies_ms) if latencies_ms else 0.0,
    }


def run_matrix_cell(
    args: argparse.Namespace,
    q2l: Any,
    label: str,
    collection: str,
    batch_size: int,
    repeat: int,
    query_plans: list[dict[str, Any]],
    neighbors: np.ndarray,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    warmup = args.warmup_query_count
    if warmup:
        q2l.execute_query_plans_once(
            args.base_url,
            collection,
            query_plans[:warmup],
            neighbors[:warmup],
            args.top_k,
        )

    measured_plans = query_plans[warmup:]
    measured_neighbors = neighbors[warmup:]
    batch_rows: list[dict[str, Any]] = []
    batch_latencies_ms: list[float] = []
    hits = 0
    visited_shards = 0
    assigned_ef_sum = 0
    search_batch_calls = 0
    started = time.perf_counter()
    for batch_index, start_idx in enumerate(range(0, len(measured_plans), batch_size)):
        end_idx = min(start_idx + batch_size, len(measured_plans))
        batch_start = time.perf_counter()
        result = q2l.execute_query_plans_once(
            args.base_url,
            collection,
            measured_plans[start_idx:end_idx],
            measured_neighbors[start_idx:end_idx],
            args.top_k,
        )
        latency_ms = (time.perf_counter() - batch_start) * 1000.0
        batch_latencies_ms.append(latency_ms)
        hits += int(result["hits"])
        visited_shards += int(result["visited_shards"])
        assigned_ef_sum += int(result["assigned_ef_sum"])
        search_batch_calls += int(result["search_batch_calls"])
        batch_rows.append(
            {
                "placement": label,
                "collection": collection,
                "repeat": repeat,
                "batch_size": batch_size,
                "batch_index": batch_index,
                "query_start": start_idx,
                "query_end": end_idx,
                "query_count": end_idx - start_idx,
                "batch_latency_ms": latency_ms,
                "recall_at_k": int(result["hits"]) / ((end_idx - start_idx) * args.top_k),
                "visited_shards": int(result["visited_shards"]),
                "assigned_ef_sum": int(result["assigned_ef_sum"]),
            }
        )
    wall_s = time.perf_counter() - started
    query_count = len(measured_plans)
    summary: dict[str, Any] = {
        "placement": label,
        "collection": collection,
        "repeat": repeat,
        "batch_size": batch_size,
        "query_count": query_count,
        "warmup_query_count": warmup,
        "batch_count": len(batch_latencies_ms),
        "recall_at_k": hits / (query_count * args.top_k),
        "qps": query_count / wall_s if wall_s > 0 else 0.0,
        "wall_s": wall_s,
        "avg_visited_shards": visited_shards / query_count if query_count else 0.0,
        "avg_assigned_ef_sum": assigned_ef_sum / query_count if query_count else 0.0,
        "search_batch_calls": search_batch_calls,
    }
    summary.update(summarize_latencies(batch_latencies_ms, q2l))
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return summary, batch_rows


def main() -> int:
    args = parse_args()
    q2l = load_qdrant_tool(args.qdrant_tool)
    collections = parse_collections(args.collection)
    output_dir = Path(args.output_root) / time.strftime("analysis_%Y%m%d_%H%M%S", time.gmtime())
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"output_dir {output_dir}", flush=True)

    upper_vectors, queries, neighbors, num_points, dim = load_upper_eval_and_neighbors(args, q2l)
    upper_indices = q2l.global_upper_indices(num_points, args.sample_denominator, args.upper_sample_seed)
    point_to_shards = recover_upper_membership(args, q2l, upper_indices, num_points)
    upper_index = q2l.build_upper_index(
        upper_vectors,
        upper_indices.astype(np.int64, copy=False),
        dim,
        args.upper_m,
        args.upper_ef_construction,
        max(args.upper_search_ef, args.upper_k),
    )
    upper_labels, _distances = upper_index.knn_query(queries, k=args.upper_k)
    query_plans = q2l.build_routed_search_plans(
        queries,
        upper_labels.astype(np.int64, copy=False),
        point_to_shards,
        args.num_shards,
        args.top_k,
        args.base_ef,
        args.factor,
        False,
        True,
        routed_execution_mode="compact_multi_ep",
        compact_ef_mode="max",
        routed_result_limit_mode="top_k",
        routed_result_limit_multiplier=1,
        source_id_dedup_block_size=num_points + 1,
    )

    summary_rows: list[dict[str, Any]] = []
    batch_rows: list[dict[str, Any]] = []
    for batch_size in args.batch_sizes:
        for repeat in range(1, args.repeats + 1):
            ordered = collections if repeat % 2 == 1 else list(reversed(collections))
            for label, collection in ordered:
                summary, rows = run_matrix_cell(
                    args,
                    q2l,
                    label,
                    collection,
                    batch_size,
                    repeat,
                    query_plans,
                    neighbors,
                )
                summary_rows.append(summary)
                batch_rows.extend(rows)

    write_csv(output_dir / "batch_latency_runs.csv", summary_rows)
    write_csv(output_dir / "batch_latency_per_batch.csv", batch_rows)
    metadata = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "analysis_kind": "claim_g_matched_physical_layout_batch_latency_matrix",
        "base_url": args.base_url,
        "routing_source_collection": args.routing_source_collection,
        "collections": [{"placement": label, "collection": collection} for label, collection in collections],
        "hdf5_path": args.hdf5_path,
        "num_points": num_points,
        "source_id_dedup_block_size": num_points + 1,
        "num_shards": args.num_shards,
        "upper_k": args.upper_k,
        "base_ef": args.base_ef,
        "factor": args.factor,
        "eval_query_count": args.eval_query_count,
        "warmup_query_count": args.warmup_query_count,
        "batch_sizes": args.batch_sizes,
        "repeats": args.repeats,
        "notes": [
            "All placements use route plans recovered from routing_source_collection.",
            "The size_balanced and method4_aware target collections were deployed by cloning source logical point-to-shard membership and changing only physical placement.",
            "Latency metrics are client-observed batch endpoint wall time per batch, not per-query latency inside the server.",
        ],
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"wrote {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
