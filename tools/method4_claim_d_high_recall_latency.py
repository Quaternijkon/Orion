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


DEFAULT_METHOD4_CONFIGS = [
    "method4_r097=200,60,20",
    "method4_r099=360,100,20",
]

DEFAULT_NAIVE_EFS = [
    "naive_ef128=128",
    "naive_ef160=160",
    "naive_ef200=200",
    "naive_ef240=240",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Claim D high-recall Method4-vs-naive batch latency experiment."
    )
    parser.add_argument("--base-url", default="http://localhost:6833")
    parser.add_argument("--method4-collection", default="qdrant_controller_idea_method4map_full_20260601")
    parser.add_argument("--method4-routing-source-collection", default=None)
    parser.add_argument("--naive-collection", default="qdrant_controller_naive46_full_20260521")
    parser.add_argument("--kmeans-collection", default="bench095_cpp_kmeans_s46")
    parser.add_argument("--simple-kmeans-collection", default="l2_sift100k_simple_s16_20260705")
    parser.add_argument("--hdf5-path", default="/home/taig/dry/faiss/datasets/glove-200-angular.hdf5")
    parser.add_argument("--vector-distance", choices=["cosine", "euclid", "l2"], default="cosine")
    parser.add_argument("--num-shards", type=int, default=46)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--eval-query-count", type=int, default=3000)
    parser.add_argument("--warmup-query-count", type=int, default=100)
    parser.add_argument("--query-start-offset", type=int, default=0)
    parser.add_argument("--query-order", choices=["original", "shuffled"], default="original")
    parser.add_argument("--query-shuffle-seed", type=int, default=20260705)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--sample-denominator", type=int, default=32)
    parser.add_argument("--upper-sample-seed", type=int, default=100)
    parser.add_argument("--upper-m", type=int, default=32)
    parser.add_argument("--upper-ef-construction", type=int, default=100)
    parser.add_argument("--upper-search-ef", type=int, default=200)
    parser.add_argument("--cpp-kmeans-train-size", type=int, default=10000)
    parser.add_argument("--kmeans-iters", type=int, default=10)
    parser.add_argument("--scroll-page-size", type=int, default=20000)
    parser.add_argument("--method4-config", action="append", default=None, help="label=upper_k,base_ef,factor")
    parser.add_argument("--naive-ef", action="append", default=None, help="label=ef")
    parser.add_argument("--kmeans-config", action="append", default=None, help="label=upper_k,base_ef,factor")
    parser.add_argument("--simple-kmeans-config", action="append", default=None, help="label=nprobe,hnsw_ef")
    parser.add_argument("--skip-method4", action="store_true", help="Run only naive controls and skip Method4 configs.")
    parser.add_argument("--skip-naive", action="store_true", help="Run only Method4 configs and skip naive controls.")
    parser.add_argument("--skip-kmeans", action="store_true", help="Skip KMeans routed controls.")
    parser.add_argument("--skip-simple-kmeans", action="store_true", help="Skip simple KMeans nprobe controls.")
    parser.add_argument("--output-root", default="results/method4_claim_d_high_recall_latency_20260704")
    parser.add_argument("--qdrant-tool", default="tools/qdrant_two_level_routing_experiment.py")
    return parser.parse_args()


def vector_distance_config(vector_distance: str) -> dict[str, Any]:
    normalized = vector_distance.lower()
    if normalized == "cosine":
        return {
            "name": "cosine",
            "hnsw_space": "cosine",
            "normalize_vectors": True,
            "score_higher_is_better": True,
        }
    if normalized in {"euclid", "l2"}:
        return {
            "name": "euclid",
            "hnsw_space": "l2",
            "normalize_vectors": False,
            "score_higher_is_better": False,
        }
    raise ValueError(f"unsupported vector distance: {vector_distance}")


def prepare_vectors_for_distance(q2l: Any, arr: np.ndarray, vector_distance: str) -> np.ndarray:
    config = vector_distance_config(vector_distance)
    if config["normalize_vectors"]:
        return q2l.normalize_rows(arr)
    return arr


def load_qdrant_tool(path: str | Path) -> Any:
    module_path = Path(path)
    spec = importlib.util.spec_from_file_location("qdrant_two_level_routing_experiment", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def parse_method4_configs(values: list[str] | None, skip_method4: bool = False) -> list[tuple[str, int, int, int]]:
    if skip_method4:
        return []
    specs = values or DEFAULT_METHOD4_CONFIGS
    parsed: list[tuple[str, int, int, int]] = []
    for spec in specs:
        label, raw = spec.split("=", 1)
        upper_k, base_ef, factor = [int(part) for part in raw.split(",")]
        parsed.append((label, upper_k, base_ef, factor))
    return parsed


def parse_naive_efs(values: list[str] | None, skip_naive: bool = False) -> list[tuple[str, int]]:
    if skip_naive:
        return []
    specs = values or DEFAULT_NAIVE_EFS
    parsed: list[tuple[str, int]] = []
    for spec in specs:
        label, raw = spec.split("=", 1)
        parsed.append((label, int(raw)))
    return parsed


def parse_kmeans_configs(values: list[str] | None, skip_kmeans: bool = False) -> list[tuple[str, int, int, int]]:
    if skip_kmeans:
        return []
    if not values:
        return []
    parsed: list[tuple[str, int, int, int]] = []
    for spec in values:
        label, raw = spec.split("=", 1)
        upper_k, base_ef, factor = [int(part) for part in raw.split(",")]
        parsed.append((label, upper_k, base_ef, factor))
    return parsed


def parse_simple_kmeans_configs(values: list[str] | None, skip_simple_kmeans: bool = False) -> list[tuple[str, int, int]]:
    if skip_simple_kmeans:
        return []
    if not values:
        return []
    parsed: list[tuple[str, int, int]] = []
    for spec in values:
        label, raw = spec.split("=", 1)
        nprobe, hnsw_ef = [int(part) for part in raw.split(",")]
        parsed.append((label, nprobe, hnsw_ef))
    return parsed


def load_queries_and_neighbors(args: argparse.Namespace, q2l: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    total_queries = args.warmup_query_count + args.eval_query_count
    start = int(getattr(args, "query_start_offset", 0))
    if start < 0:
        raise ValueError(f"query_start_offset must be non-negative, got {start}")
    end = start + total_queries
    with h5py.File(args.hdf5_path, "r") as handle:
        train_count = int(handle["train"].shape[0])
        dim = int(handle["train"].shape[1])
        queries = handle["test"][start:end].astype(np.float32, copy=True)
        neighbors = handle["neighbors"][start:end, : args.top_k].astype(np.int64, copy=True)
        upper_indices = q2l.global_upper_indices(train_count, args.sample_denominator, args.upper_sample_seed)
        order = np.argsort(upper_indices)
        sorted_upper = upper_indices[order]
        upper_vectors_sorted = handle["train"][sorted_upper].astype(np.float32, copy=True)
        inverse = np.empty_like(order)
        inverse[order] = np.arange(len(order))
        upper_vectors = upper_vectors_sorted[inverse]
    return (
        prepare_vectors_for_distance(q2l, queries, args.vector_distance),
        prepare_vectors_for_distance(q2l, upper_vectors, args.vector_distance),
        neighbors,
        train_count,
        dim,
    )


def load_training_vectors(args: argparse.Namespace, q2l: Any) -> np.ndarray:
    with h5py.File(args.hdf5_path, "r") as handle:
        train = handle["train"][:].astype(np.float32, copy=True)
    return prepare_vectors_for_distance(q2l, train, args.vector_distance)


def query_order_indices(
    total_queries: int,
    warmup_query_count: int,
    query_order: str,
    query_shuffle_seed: int,
) -> list[int]:
    if total_queries < 0:
        raise ValueError(f"total_queries must be non-negative, got {total_queries}")
    if warmup_query_count < 0:
        raise ValueError(f"warmup_query_count must be non-negative, got {warmup_query_count}")
    if warmup_query_count > total_queries:
        raise ValueError(
            f"warmup_query_count ({warmup_query_count}) cannot exceed total_queries ({total_queries})"
        )

    warmup_indices = list(range(warmup_query_count))
    measured_indices = list(range(warmup_query_count, total_queries))
    if query_order == "original":
        return warmup_indices + measured_indices
    if query_order == "shuffled":
        rng = np.random.default_rng(int(query_shuffle_seed))
        shuffled_measured = rng.permutation(np.array(measured_indices, dtype=np.int64)).tolist()
        return warmup_indices + [int(index) for index in shuffled_measured]
    raise ValueError(f"unsupported query_order: {query_order}")


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


def build_method4_plans(
    q2l: Any,
    queries: np.ndarray,
    upper_index: Any,
    point_to_shards: list[list[int]],
    num_shards: int,
    top_k: int,
    upper_k: int,
    base_ef: int,
    factor: int,
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
        routed_execution_mode="compact_multi_ep",
        compact_ef_mode="max",
        routed_result_limit_mode="top_k",
        routed_result_limit_multiplier=1,
        source_id_dedup_block_size=source_id_dedup_block_size,
    )


def build_naive_plans(
    q2l: Any,
    queries: np.ndarray,
    top_k: int,
    num_shards: int,
    hnsw_ef: int,
    use_payload_source_id: bool,
    source_id_dedup_block_size: int,
) -> list[dict[str, Any]]:
    return q2l.build_all_shard_search_plans(
        queries,
        top_k,
        num_shards,
        hnsw_ef,
        use_payload_source_id,
        source_id_dedup_block_size,
    )


def build_cpp_kmeans_plans(
    q2l: Any,
    train: np.ndarray,
    queries: np.ndarray,
    num_shards: int,
    top_k: int,
    upper_k: int,
    base_ef: int,
    factor: int,
    cpp_kmeans_train_size: int,
    kmeans_iters: int,
    upper_sample_seed: int,
    sample_denominator: int,
    upper_m: int,
    upper_ef_construction: int,
    upper_search_ef: int,
    hnsw_space: str,
    source_id_dedup_block_size: int,
) -> list[dict[str, Any]]:
    assignments, _centroids = q2l.build_cpp_kmeans_baseline_assignments(
        train,
        num_shards,
        cpp_kmeans_train_size,
        kmeans_iters,
        upper_sample_seed,
    )
    upper_vectors, upper_ids, label_to_shard, _sample_rows = q2l.sample_cpp_kmeans_upper_points(
        train,
        assignments,
        num_shards,
        sample_denominator,
        upper_sample_seed,
    )
    if len(upper_ids) == 0:
        raise ValueError(
            "cpp_kmeans_baseline sampled no upper points; decrease --sample-denominator "
            "or increase the training set size"
        )
    upper_index = q2l.build_upper_index(
        upper_vectors,
        upper_ids.astype(np.int64, copy=False),
        train.shape[1],
        upper_m,
        upper_ef_construction,
        max(upper_search_ef, upper_k),
        hnsw_space=hnsw_space,
    )
    upper_labels = q2l.compute_upper_labels(upper_index, queries, upper_k)
    # cpp_kmeans_baseline collections store a single point copy with id=source_id+1.
    # Do not encode entry points with Method4's multi-copy shard block scheme here.
    kmeans_source_id_dedup_block_size = None
    return [
        q2l.legacy_routed_search_plan(
            query.tolist(),
            labels_row,
            label_to_shard,
            top_k,
            base_ef,
            factor,
            False,
            "compact_multi_ep",
            "max",
            "top_k",
            1,
            kmeans_source_id_dedup_block_size,
        )
        for query, labels_row in zip(queries, upper_labels)
    ]


def build_simple_kmeans_plans(
    q2l: Any,
    train: np.ndarray,
    queries: np.ndarray,
    num_shards: int,
    top_k: int,
    nprobe: int,
    hnsw_ef: int,
    cpp_kmeans_train_size: int,
    kmeans_iters: int,
    upper_sample_seed: int,
) -> list[dict[str, Any]]:
    _assignments, centroids = q2l.build_cpp_kmeans_baseline_assignments(
        train,
        num_shards,
        cpp_kmeans_train_size,
        kmeans_iters,
        upper_sample_seed,
    )
    return q2l.build_kmeans_simple_nprobe_search_plans(
        queries,
        centroids,
        nprobe,
        top_k,
        hnsw_ef,
        False,
        None,
    )


def summarize_batch_rows(
    method: str,
    config_label: str,
    rows: list[dict[str, Any]],
    top_k: int,
) -> dict[str, Any]:
    query_count = sum(int(row["query_count"]) for row in rows)
    hits = sum(int(row["hits"]) for row in rows)
    wall_s = sum(float(row["wall_s"]) for row in rows)
    latencies = [float(row["batch_latency_ms"]) for row in rows]
    visited = sum(int(row["visited_shards"]) for row in rows)
    ef_sum = sum(int(row["assigned_ef_sum"]) for row in rows)
    search_batch_calls = sum(int(row["search_batch_calls"]) for row in rows)
    return {
        "method": method,
        "config_label": config_label,
        "query_count": query_count,
        "top_k": top_k,
        "recall_at_10": hits / (query_count * top_k) if query_count else 0.0,
        "qps": query_count / wall_s if wall_s > 0 else 0.0,
        "wall_s": wall_s,
        "avg_visited_shards": visited / query_count if query_count else 0.0,
        "avg_assigned_ef_sum": ef_sum / query_count if query_count else 0.0,
        "search_batch_calls": search_batch_calls,
        "batch_count": len(rows),
        "batch_latency_mean_ms": statistics.mean(latencies) if latencies else 0.0,
        "batch_latency_p50_ms": percentile(latencies, 50),
        "batch_latency_p95_ms": percentile(latencies, 95),
        "batch_latency_p99_ms": percentile(latencies, 99),
        "batch_latency_max_ms": max(latencies) if latencies else 0.0,
    }


def run_plan_batches(
    args: argparse.Namespace,
    q2l: Any,
    method: str,
    config_label: str,
    collection: str,
    query_plans: list[dict[str, Any]],
    neighbors: np.ndarray,
    repeat: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    warmup = args.warmup_query_count
    if warmup:
        q2l.execute_query_plans_once(
            args.base_url,
            collection,
            query_plans[:warmup],
            neighbors[:warmup],
            args.top_k,
            score_higher_is_better=args.score_higher_is_better,
        )
    measured_plans = query_plans[warmup:]
    measured_neighbors = neighbors[warmup:]
    batch_rows: list[dict[str, Any]] = []
    for batch_index, start_idx in enumerate(range(0, len(measured_plans), args.batch_size)):
        end_idx = min(start_idx + args.batch_size, len(measured_plans))
        started = time.perf_counter()
        result = q2l.execute_query_plans_once(
            args.base_url,
            collection,
            measured_plans[start_idx:end_idx],
            measured_neighbors[start_idx:end_idx],
            args.top_k,
            score_higher_is_better=args.score_higher_is_better,
        )
        wall_s = time.perf_counter() - started
        batch_rows.append(
            {
                "method": method,
                "config_label": config_label,
                "collection": collection,
                "repeat": repeat,
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
            }
        )
    summary = summarize_batch_rows(method, config_label, batch_rows, args.top_k)
    summary.update(
        {
            "collection": collection,
            "repeat": repeat,
            "batch_size": args.batch_size,
            "query_start_offset": getattr(args, "query_start_offset", 0),
            "query_order": getattr(args, "query_order", "original"),
            "query_shuffle_seed": getattr(args, "query_shuffle_seed", 20260705),
        }
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return summary, batch_rows


def main() -> int:
    args = parse_args()
    q2l = load_qdrant_tool(args.qdrant_tool)
    distance_config = vector_distance_config(args.vector_distance)
    args.score_higher_is_better = distance_config["score_higher_is_better"]
    output_dir = Path(args.output_root) / time.strftime("analysis_%Y%m%d_%H%M%S", time.gmtime())
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"output_dir {output_dir}", flush=True)

    queries, upper_vectors, neighbors, train_count, dim = load_queries_and_neighbors(args, q2l)
    upper_indices = q2l.global_upper_indices(train_count, args.sample_denominator, args.upper_sample_seed)
    method4_configs = parse_method4_configs(args.method4_config, args.skip_method4)
    kmeans_configs = parse_kmeans_configs(args.kmeans_config, args.skip_kmeans)
    simple_kmeans_configs = parse_simple_kmeans_configs(args.simple_kmeans_config, args.skip_simple_kmeans)
    max_upper_k = max((upper_k for _label, upper_k, _base_ef, _factor in method4_configs), default=0)
    routing_source = args.method4_routing_source_collection or args.method4_collection
    upper_index = None
    point_to_shards: list[list[int]] | None = None
    if method4_configs:
        upper_index = q2l.build_upper_index(
            upper_vectors,
            upper_indices.astype(np.int64, copy=False),
            dim,
            args.upper_m,
            args.upper_ef_construction,
            max(args.upper_search_ef, max_upper_k),
            hnsw_space=distance_config["hnsw_space"],
        )
        point_to_shards = recover_upper_membership(args, q2l, routing_source, upper_indices, train_count)
    source_id_dedup_block_size = train_count + 1
    order_indices = query_order_indices(
        len(neighbors),
        args.warmup_query_count,
        args.query_order,
        args.query_shuffle_seed,
    )
    ordered_neighbors = neighbors[order_indices]

    plan_specs: list[tuple[str, str, str, list[dict[str, Any]], dict[str, Any]]] = []
    for label, upper_k, base_ef, factor in method4_configs:
        if upper_index is None or point_to_shards is None:
            raise RuntimeError("Method4 configs require upper index and point_to_shards")
        plans = build_method4_plans(
            q2l,
            queries,
            upper_index,
            point_to_shards,
            args.num_shards,
            args.top_k,
            upper_k,
            base_ef,
            factor,
            source_id_dedup_block_size,
        )
        plan_specs.append(
            (
                "Method4",
                label,
                args.method4_collection,
                [plans[index] for index in order_indices],
                {"upper_k": upper_k, "base_ef": base_ef, "factor": factor},
            )
        )
    train: np.ndarray | None = None
    if kmeans_configs or simple_kmeans_configs:
        train = load_training_vectors(args, q2l)
    for label, upper_k, base_ef, factor in kmeans_configs:
        if train is None:
            raise RuntimeError("KMeans configs require training vectors")
        plans = build_cpp_kmeans_plans(
            q2l,
            train,
            queries,
            args.num_shards,
            args.top_k,
            upper_k,
            base_ef,
            factor,
            args.cpp_kmeans_train_size,
            args.kmeans_iters,
            args.upper_sample_seed,
            args.sample_denominator,
            args.upper_m,
            args.upper_ef_construction,
            args.upper_search_ef,
            distance_config["hnsw_space"],
            source_id_dedup_block_size,
        )
        plan_specs.append(
            (
                "KMeans",
                label,
                args.kmeans_collection,
                [plans[index] for index in order_indices],
                {"upper_k": upper_k, "base_ef": base_ef, "factor": factor},
            )
        )
    for label, nprobe, hnsw_ef in simple_kmeans_configs:
        if train is None:
            raise RuntimeError("Simple KMeans configs require training vectors")
        plans = build_simple_kmeans_plans(
            q2l,
            train,
            queries,
            args.num_shards,
            args.top_k,
            nprobe,
            hnsw_ef,
            args.cpp_kmeans_train_size,
            args.kmeans_iters,
            args.upper_sample_seed,
        )
        plan_specs.append(
            (
                "SimpleKMeans",
                label,
                args.simple_kmeans_collection,
                [plans[index] for index in order_indices],
                {"upper_k": nprobe, "base_ef": hnsw_ef, "factor": 0},
            )
        )
    for label, hnsw_ef in parse_naive_efs(args.naive_ef, args.skip_naive):
        plans = build_naive_plans(
            q2l,
            queries,
            args.top_k,
            args.num_shards,
            hnsw_ef,
            True,
            source_id_dedup_block_size,
        )
        plan_specs.append(
            (
                "Naive",
                label,
                args.naive_collection,
                [plans[index] for index in order_indices],
                {"upper_k": 0, "base_ef": hnsw_ef, "factor": 0},
            )
        )

    summary_rows: list[dict[str, Any]] = []
    batch_rows: list[dict[str, Any]] = []
    for method, label, collection, plans, params in plan_specs:
        for repeat in range(1, args.repeats + 1):
            summary, rows = run_plan_batches(args, q2l, method, label, collection, plans, ordered_neighbors, repeat)
            summary.update(params)
            summary_rows.append(summary)
            batch_rows.extend(rows)

    write_csv(output_dir / "claim_d_high_recall_latency_summary.csv", summary_rows)
    write_csv(output_dir / "claim_d_high_recall_latency_batches.csv", batch_rows)
    metadata = {
        "analysis_kind": "claim_d_high_recall_method4_vs_naive_batch_latency",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "base_url": args.base_url,
        "method4_collection": args.method4_collection,
        "method4_routing_source_collection": routing_source,
        "naive_collection": args.naive_collection,
        "kmeans_collection": args.kmeans_collection,
        "simple_kmeans_collection": args.simple_kmeans_collection,
        "hdf5_path": args.hdf5_path,
        "vector_distance": distance_config["name"],
        "upper_hnsw_space": distance_config["hnsw_space"],
        "normalize_vectors": distance_config["normalize_vectors"],
        "score_higher_is_better": distance_config["score_higher_is_better"],
        "num_shards": args.num_shards,
        "top_k": args.top_k,
        "eval_query_count": args.eval_query_count,
        "warmup_query_count": args.warmup_query_count,
        "query_start_offset": args.query_start_offset,
        "query_order": args.query_order,
        "query_shuffle_seed": args.query_shuffle_seed,
        "batch_size": args.batch_size,
        "repeats": args.repeats,
        "cpp_kmeans_train_size": args.cpp_kmeans_train_size,
        "kmeans_iters": args.kmeans_iters,
        "method4_configs": [] if args.skip_method4 else (args.method4_config or DEFAULT_METHOD4_CONFIGS),
        "naive_efs": [] if args.skip_naive else (args.naive_ef or DEFAULT_NAIVE_EFS),
        "kmeans_configs": [] if args.skip_kmeans else (args.kmeans_config or []),
        "simple_kmeans_configs": [] if args.skip_simple_kmeans else (args.simple_kmeans_config or []),
        "notes": [
            "Latency metrics are client-observed batch endpoint wall time per batch.",
            "The script precomputes Method4, KMeans, Simple KMeans, and naive lower-search plans before timing each batch.",
        ],
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
