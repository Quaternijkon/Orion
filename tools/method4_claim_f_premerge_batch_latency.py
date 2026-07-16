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


DEFAULT_DIRECT_PEER_URLS = [
    "6015626418395790=http://localhost:6843",
    "2980601005324529=http://localhost:6853",
    "6846760844865837=http://localhost:6863",
]

DEFAULT_VARIANTS = [
    "coordinator_compact_current",
    "direct_peer_no_premerge",
    "direct_peer_local_premerge",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Claim F worker-local pre-merge batch-latency and fan-in matrix. "
            "Runs the same Method4 route plans through coordinator compact "
            "execution and direct-peer shard-major variants."
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
    parser.add_argument("--direct-peer-http-urls", nargs="+", default=DEFAULT_DIRECT_PEER_URLS)
    parser.add_argument("--sample-denominator", type=int, default=32)
    parser.add_argument("--upper-sample-seed", type=int, default=100)
    parser.add_argument("--upper-m", type=int, default=32)
    parser.add_argument("--upper-ef-construction", type=int, default=100)
    parser.add_argument("--upper-search-ef", type=int, default=200)
    parser.add_argument("--scroll-page-size", type=int, default=20000)
    parser.add_argument("--output-root", default="results/method4_claim_f_premerge_batch_latency_20260704")
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


def parse_direct_peer_urls(values: list[str]) -> dict[int, str]:
    parsed: dict[int, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"direct peer URL spec must be PEER_ID=URL, got {value!r}")
        peer_id_raw, url = value.split("=", 1)
        if not peer_id_raw or not url:
            raise ValueError(f"direct peer URL spec must be PEER_ID=URL, got {value!r}")
        try:
            peer_id = int(peer_id_raw)
        except ValueError as exc:
            raise ValueError(f"direct peer URL spec must be PEER_ID=URL, got {value!r}") from exc
        parsed[peer_id] = url
    return parsed


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


def reduction_pct(baseline: float | None, observed: float) -> float:
    if baseline is None or baseline <= 0.0:
        return 0.0
    return (baseline - observed) / baseline * 100.0


def summarize_variant_rows(
    variant: str,
    rows: list[dict[str, Any]],
    top_k: int,
    baseline_candidate_groups_per_query: float | None = None,
    baseline_returned_candidates_per_query: float | None = None,
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

    avg_candidate_groups = candidate_groups / query_count if query_count else 0.0
    avg_returned_candidates = returned_candidates / query_count if query_count else 0.0
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
        "avg_candidate_groups_per_query": avg_candidate_groups,
        "avg_returned_candidates_per_query": avg_returned_candidates,
        "candidate_group_reduction_vs_baseline_pct": reduction_pct(
            baseline_candidate_groups_per_query,
            avg_candidate_groups,
        ),
        "returned_candidate_reduction_vs_baseline_pct": reduction_pct(
            baseline_returned_candidates_per_query,
            avg_returned_candidates,
        ),
        "search_batch_calls": search_batch_calls,
        "batch_count": len(rows),
        "batch_latency_mean_ms": statistics.mean(latencies) if latencies else 0.0,
        "batch_latency_p50_ms": percentile(latencies, 50),
        "batch_latency_p95_ms": percentile(latencies, 95),
        "batch_latency_p99_ms": percentile(latencies, 99),
        "batch_latency_max_ms": max(latencies) if latencies else 0.0,
    }


def apply_direct_peer_baselines(rows: list[dict[str, Any]]) -> None:
    baselines: dict[tuple[int, int], tuple[float, float]] = {}
    for row in rows:
        if row.get("variant") != "direct_peer_no_premerge":
            continue
        key = (int(row["batch_size"]), int(row["repeat"]))
        baselines[key] = (
            float(row["avg_candidate_groups_per_query"]),
            float(row["avg_returned_candidates_per_query"]),
        )
        row["candidate_group_reduction_vs_baseline_pct"] = 0.0
        row["returned_candidate_reduction_vs_baseline_pct"] = 0.0

    for row in rows:
        if row.get("variant") != "direct_peer_local_premerge":
            continue
        key = (int(row["batch_size"]), int(row["repeat"]))
        if key not in baselines:
            continue
        baseline_candidate_groups, baseline_returned_candidates = baselines[key]
        row["candidate_group_reduction_vs_baseline_pct"] = reduction_pct(
            baseline_candidate_groups,
            float(row["avg_candidate_groups_per_query"]),
        )
        row["returned_candidate_reduction_vs_baseline_pct"] = reduction_pct(
            baseline_returned_candidates,
            float(row["avg_returned_candidates_per_query"]),
        )


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


def execute_variant_once(
    args: argparse.Namespace,
    q2l: Any,
    variant: str,
    query_plans: list[dict[str, Any]],
    neighbors: np.ndarray,
    direct_peer_urls: dict[int, str],
    shard_key_to_peer: dict[str, int],
) -> dict[str, Any]:
    if variant == "coordinator_compact_current":
        return q2l.execute_query_plans_once(
            args.base_url,
            args.collection,
            query_plans,
            neighbors,
            args.top_k,
            lower_execution_order="query_major",
        )
    if variant == "direct_peer_no_premerge":
        return q2l.execute_query_plans_once(
            args.base_url,
            args.collection,
            query_plans,
            neighbors,
            args.top_k,
            direct_peer_urls=direct_peer_urls,
            shard_key_to_peer=shard_key_to_peer,
            lower_execution_order="shard_major",
            direct_peer_local_premerge=False,
        )
    if variant == "direct_peer_local_premerge":
        return q2l.execute_query_plans_once(
            args.base_url,
            args.collection,
            query_plans,
            neighbors,
            args.top_k,
            direct_peer_urls=direct_peer_urls,
            shard_key_to_peer=shard_key_to_peer,
            lower_execution_order="shard_major",
            direct_peer_local_premerge=True,
        )
    raise ValueError(f"unsupported variant: {variant}")


def run_variant_batches(
    args: argparse.Namespace,
    q2l: Any,
    variant: str,
    batch_size: int,
    repeat: int,
    query_plans: list[dict[str, Any]],
    neighbors: np.ndarray,
    direct_peer_urls: dict[int, str],
    shard_key_to_peer: dict[str, int],
    baseline_candidate_groups_per_query: float | None,
    baseline_returned_candidates_per_query: float | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    warmup = args.warmup_query_count
    if warmup:
        execute_variant_once(
            args,
            q2l,
            variant,
            query_plans[:warmup],
            neighbors[:warmup],
            direct_peer_urls,
            shard_key_to_peer,
        )

    measured_plans = query_plans[warmup:]
    measured_neighbors = neighbors[warmup:]
    batch_rows: list[dict[str, Any]] = []
    for batch_index, start_idx in enumerate(range(0, len(measured_plans), batch_size)):
        end_idx = min(start_idx + batch_size, len(measured_plans))
        started = time.perf_counter()
        result = execute_variant_once(
            args,
            q2l,
            variant,
            measured_plans[start_idx:end_idx],
            measured_neighbors[start_idx:end_idx],
            direct_peer_urls,
            shard_key_to_peer,
        )
        wall_s = time.perf_counter() - started
        batch_rows.append(
            {
                "variant": variant,
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

    summary = summarize_variant_rows(
        variant,
        batch_rows,
        args.top_k,
        baseline_candidate_groups_per_query=baseline_candidate_groups_per_query,
        baseline_returned_candidates_per_query=baseline_returned_candidates_per_query,
    )
    summary.update(
        {
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


def validate_direct_peer_mapping(
    direct_peer_urls: dict[int, str],
    shard_key_to_peer: dict[str, int],
) -> None:
    peers = {int(peer_id) for peer_id in shard_key_to_peer.values()}
    missing = sorted(peer_id for peer_id in peers if peer_id not in direct_peer_urls)
    if missing:
        raise RuntimeError(f"missing direct peer HTTP URLs for peers: {missing}")


def main() -> int:
    args = parse_args()
    q2l = load_qdrant_tool(args.qdrant_tool)
    variants = args.variant or DEFAULT_VARIANTS
    direct_peer_urls = parse_direct_peer_urls(args.direct_peer_http_urls)
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
    query_plans = build_method4_plans(
        q2l,
        queries,
        upper_index,
        point_to_shards,
        args.num_shards,
        args.top_k,
        args.upper_k,
        args.base_ef,
        args.factor,
        train_count + 1,
    )

    shard_key_to_peer = q2l.collection_shard_key_to_peer(args.base_url, args.collection)
    if any(variant.startswith("direct_peer") for variant in variants):
        validate_direct_peer_mapping(direct_peer_urls, shard_key_to_peer)

    summary_rows: list[dict[str, Any]] = []
    batch_rows: list[dict[str, Any]] = []
    for batch_size in args.batch_sizes:
        for repeat in range(1, args.repeats + 1):
            ordered_variants = variants if repeat % 2 == 1 else list(reversed(variants))
            for variant in ordered_variants:
                summary, rows = run_variant_batches(
                    args,
                    q2l,
                    variant,
                    batch_size,
                    repeat,
                    query_plans,
                    neighbors,
                    direct_peer_urls,
                    shard_key_to_peer,
                    None,
                    None,
                )
                summary_rows.append(summary)
                batch_rows.extend(rows)

    apply_direct_peer_baselines(summary_rows)
    write_csv(output_dir / "claim_f_premerge_batch_latency_summary.csv", summary_rows)
    write_csv(output_dir / "claim_f_premerge_batch_latency_batches.csv", batch_rows)
    metadata = {
        "analysis_kind": "claim_f_worker_local_premerge_batch_latency_and_fanin_matrix",
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
        "variants": variants,
        "direct_peer_urls": {str(key): value for key, value in sorted(direct_peer_urls.items())},
        "shard_key_to_peer_count": len(shard_key_to_peer),
        "notes": [
            "Latency metrics are client-observed batch endpoint wall time per batch.",
            "The same Method4 compact MultiEP query plans are reused for all variants.",
            "direct_peer_local_premerge is a Python-side architecture simulation: it reduces final candidate groups after direct-peer logical-shard responses have already crossed the process boundary.",
            "coordinator_compact_current measures the currently running server path and cannot by itself toggle QDRANT_DISABLE_SHARD_MAJOR_PEER_PREMERGE.",
        ],
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"wrote {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
