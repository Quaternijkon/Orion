#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import statistics
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import h5py
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Claim G online single-query latency A/B for round-robin and "
            "Method4-aware physical placement collections."
        )
    )
    parser.add_argument("--base-url", default="http://localhost:6833")
    parser.add_argument(
        "--routing-source-collection",
        default=None,
        help="Optional compatibility override: use one routing source for both collections.",
    )
    parser.add_argument("--round-robin-collection", default="qdrant_controller_idea_full_20260521")
    parser.add_argument("--method4-aware-collection", default="qdrant_controller_idea_method4map_full_20260601")
    parser.add_argument("--hdf5-path", default="/home/taig/dry/faiss/datasets/glove-200-angular.hdf5")
    parser.add_argument("--num-shards", type=int, default=46)
    parser.add_argument("--upper-k", type=int, default=160)
    parser.add_argument("--base-ef", type=int, default=80)
    parser.add_argument("--factor", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--eval-query-count", type=int, default=1000)
    parser.add_argument("--warmup-query-count", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--sample-denominator", type=int, default=32)
    parser.add_argument("--upper-sample-seed", type=int, default=100)
    parser.add_argument("--upper-m", type=int, default=32)
    parser.add_argument("--upper-ef-construction", type=int, default=100)
    parser.add_argument("--upper-search-ef", type=int, default=160)
    parser.add_argument("--scroll-page-size", type=int, default=20000)
    parser.add_argument("--output-root", default="results/method4_claim_g_online_latency_20260704")
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

        eval_queries = handle["test"][:total_queries].astype(np.float32, copy=True)
        eval_neighbors = handle["neighbors"][:total_queries, : args.top_k].astype(np.int64, copy=True)

    return (
        q2l.normalize_rows(upper_vectors),
        q2l.normalize_rows(eval_queries),
        eval_neighbors,
        num_points,
        dim,
    )


def recover_upper_membership(
    args: argparse.Namespace,
    q2l: Any,
    routing_source_collection: str,
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
                f"/collections/{urllib.parse.quote(routing_source_collection, safe='')}/points/scroll",
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
        print(f"recovered {routing_source_collection} routing shard {shard_id}: scanned {scanned}", flush=True)

    missing = [int(point_id) for point_id in upper_indices.tolist() if not point_to_shards[int(point_id)]]
    if missing:
        preview = ", ".join(str(point_id) for point_id in missing[:10])
        raise RuntimeError(f"missing upper shard membership for {len(missing)} points; first: {preview}")
    return point_to_shards


def build_query_plans_for_collection(
    args: argparse.Namespace,
    q2l: Any,
    routing_source_collection: str,
    upper_indices: np.ndarray,
    upper_labels: np.ndarray,
    eval_queries: np.ndarray,
    num_points: int,
) -> list[dict[str, Any]]:
    point_to_shards = recover_upper_membership(args, q2l, routing_source_collection, upper_indices, num_points)
    return q2l.build_routed_search_plans(
        eval_queries,
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


def summarize_latencies(latencies_ms: list[float], q2l: Any) -> dict[str, float]:
    return {
        "latency_mean_ms": statistics.mean(latencies_ms) if latencies_ms else 0.0,
        "latency_p50_ms": q2l.percentile(latencies_ms, 50),
        "latency_p95_ms": q2l.percentile(latencies_ms, 95),
        "latency_p99_ms": q2l.percentile(latencies_ms, 99),
        "latency_max_ms": max(latencies_ms) if latencies_ms else 0.0,
    }


def run_collection_once(
    args: argparse.Namespace,
    q2l: Any,
    collection_label: str,
    collection: str,
    query_plans: list[dict[str, Any]],
    neighbors: np.ndarray,
    repeat: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    warmup = args.warmup_query_count
    measured_plans = query_plans[warmup:]
    measured_neighbors = neighbors[warmup:]

    for query_idx in range(warmup):
        q2l.execute_query_plans_once(
            args.base_url,
            collection,
            [query_plans[query_idx]],
            neighbors[query_idx : query_idx + 1],
            args.top_k,
        )

    query_rows: list[dict[str, Any]] = []
    if args.concurrency <= 0:
        raise ValueError("--concurrency must be positive")

    def execute_one(local_idx: int, plan: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        query_start = time.perf_counter()
        result = q2l.execute_query_plans_once(
            args.base_url,
            collection,
            [plan],
            measured_neighbors[local_idx : local_idx + 1],
            args.top_k,
        )
        latency_ms = (time.perf_counter() - query_start) * 1000.0
        row = {
            "collection_label": collection_label,
            "collection": collection,
            "repeat": repeat,
            "query_index": local_idx,
            "latency_ms": latency_ms,
            "hits_at_k": int(result["hits"]),
            "recall_at_k": int(result["hits"]) / args.top_k,
            "visited_shards": int(result["visited_shards"]),
            "assigned_ef_sum": int(result["assigned_ef_sum"]),
        }
        return row, result

    started = time.perf_counter()
    if args.concurrency == 1:
        executed = [execute_one(local_idx, plan) for local_idx, plan in enumerate(measured_plans)]
    else:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = [
                pool.submit(execute_one, local_idx, plan)
                for local_idx, plan in enumerate(measured_plans)
            ]
            executed = [future.result() for future in futures]
    wall_s = time.perf_counter() - started

    latencies_ms: list[float] = []
    hits = 0
    visited_shards = 0
    assigned_ef_sum = 0
    for row, result in sorted(executed, key=lambda item: int(item[0]["query_index"])):
        query_rows.append(row)
        latencies_ms.append(float(row["latency_ms"]))
        hits += int(result["hits"])
        visited_shards += int(result["visited_shards"])
        assigned_ef_sum += int(result["assigned_ef_sum"])

    summary: dict[str, Any] = {
        "collection_label": collection_label,
        "collection": collection,
        "repeat": repeat,
        "query_count": len(measured_plans),
        "warmup_query_count": warmup,
        "concurrency": args.concurrency,
        "recall_at_k": hits / (len(measured_plans) * args.top_k),
        "qps": len(measured_plans) / wall_s if wall_s > 0 else 0.0,
        "wall_s": wall_s,
        "avg_visited_shards": visited_shards / len(measured_plans) if measured_plans else 0.0,
        "avg_assigned_ef_sum": assigned_ef_sum / len(measured_plans) if measured_plans else 0.0,
    }
    summary.update(summarize_latencies(latencies_ms, q2l))
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return summary, query_rows


def main() -> int:
    args = parse_args()
    q2l = load_qdrant_tool(args.qdrant_tool)
    output_dir = Path(args.output_root) / time.strftime("analysis_%Y%m%d_%H%M%S", time.gmtime())
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"output_dir {output_dir}", flush=True)

    upper_vectors, eval_queries, eval_neighbors, num_points, dim = load_upper_eval_and_neighbors(args, q2l)
    upper_indices = q2l.global_upper_indices(num_points, args.sample_denominator, args.upper_sample_seed)
    upper_index = q2l.build_upper_index(
        upper_vectors,
        upper_indices.astype(np.int64, copy=False),
        dim,
        args.upper_m,
        args.upper_ef_construction,
        max(args.upper_search_ef, args.upper_k),
    )
    upper_labels, _distances = upper_index.knn_query(eval_queries, k=args.upper_k)

    collections = [
        (
            "round_robin",
            args.round_robin_collection,
            args.routing_source_collection or args.round_robin_collection,
        ),
        (
            "method4_aware",
            args.method4_aware_collection,
            args.routing_source_collection or args.method4_aware_collection,
        ),
    ]
    query_plans_by_label: dict[str, list[dict[str, Any]]] = {}
    route_source_by_label: dict[str, str] = {}
    for collection_label, _collection, routing_source_collection in collections:
        route_source_by_label[collection_label] = routing_source_collection
        query_plans_by_label[collection_label] = build_query_plans_for_collection(
            args,
            q2l,
            routing_source_collection,
            upper_indices,
            upper_labels,
            eval_queries,
            num_points,
        )

    summary_rows: list[dict[str, Any]] = []
    query_rows: list[dict[str, Any]] = []
    for repeat in range(1, args.repeats + 1):
        ordered = collections if repeat % 2 == 1 else list(reversed(collections))
        for collection_label, collection, _routing_source_collection in ordered:
            summary, rows = run_collection_once(
                args,
                q2l,
                collection_label,
                collection,
                query_plans_by_label[collection_label],
                eval_neighbors,
                repeat,
            )
            summary_rows.append(summary)
            query_rows.extend(rows)

    write_csv(output_dir / "online_single_query_latency_summary.csv", summary_rows)
    write_csv(output_dir / "online_single_query_latency_per_query.csv", query_rows)
    metadata = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "analysis_kind": "claim_g_online_single_query_latency_ab",
        "base_url": args.base_url,
        "routing_source_collection": args.routing_source_collection,
        "route_source_by_label": route_source_by_label,
        "round_robin_collection": args.round_robin_collection,
        "method4_aware_collection": args.method4_aware_collection,
        "hdf5_path": args.hdf5_path,
        "num_points": num_points,
        "source_id_dedup_block_size": num_points + 1,
        "num_shards": args.num_shards,
        "upper_k": args.upper_k,
        "base_ef": args.base_ef,
        "factor": args.factor,
        "eval_query_count": args.eval_query_count,
        "warmup_query_count": args.warmup_query_count,
        "repeats": args.repeats,
        "concurrency": args.concurrency,
        "notes": [
            "This measures client-observed single-query latency by executing one preplanned Method4 compact MultiEP query plan per request.",
            "This is an online tail-latency supplement, not the batch_size 100/200 online matrix from the Claim G planning document.",
            "Both collections use the same reconstructed route plans from the round-robin routing source collection.",
        ],
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"wrote {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
