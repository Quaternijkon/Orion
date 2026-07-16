#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import time
import urllib.parse
from collections import defaultdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Claim G offline physical-placement analysis. Reconstruct Method4 "
            "routing traces and compare round-robin, size-balanced, and "
            "Method4-aware shard placement for worker-count sweeps."
        )
    )
    parser.add_argument("--base-url", default="http://localhost:6833")
    parser.add_argument("--collection", default="qdrant_controller_idea_full_20260521")
    parser.add_argument("--hdf5-path", default="/home/taig/dry/faiss/datasets/glove-200-angular.hdf5")
    parser.add_argument("--num-shards", type=int, default=46)
    parser.add_argument("--worker-counts", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--upper-k", type=int, default=160)
    parser.add_argument("--base-ef", type=int, default=80)
    parser.add_argument("--factor", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--eval-query-count", type=int, default=3000)
    parser.add_argument("--sample-denominator", type=int, default=32)
    parser.add_argument("--upper-sample-seed", type=int, default=100)
    parser.add_argument("--upper-m", type=int, default=32)
    parser.add_argument("--upper-ef-construction", type=int, default=100)
    parser.add_argument("--upper-search-ef", type=int, default=160)
    parser.add_argument("--scroll-page-size", type=int, default=20000)
    parser.add_argument("--output-root", default="results/method4_claim_g_physical_layout_20260704")
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


def cv_pct(values: list[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    if mean <= 0:
        return 0.0
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return (variance ** 0.5) / mean * 100.0


def load_upper_and_eval_data(args: argparse.Namespace, q2l: Any) -> tuple[np.ndarray, np.ndarray, int, int]:
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

        eval_queries = handle["test"][: args.eval_query_count].astype(np.float32, copy=True)

    return q2l.normalize_rows(upper_vectors), q2l.normalize_rows(eval_queries), num_points, dim


def recover_upper_membership_and_counts(
    args: argparse.Namespace,
    q2l: Any,
    upper_indices: np.ndarray,
    num_points: int,
) -> tuple[list[list[int]], list[int]]:
    upper_set = {int(point_id) for point_id in upper_indices.tolist()}
    point_to_shards: list[list[int]] = [[] for _ in range(num_points)]
    shard_copy_counts: list[int] = []

    for shard_id in range(args.num_shards):
        shard_key = q2l.shard_key_for_id(shard_id)
        offset: Any = None
        shard_count = 0
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
                f"/collections/{urllib.parse.quote(args.collection, safe='')}/points/scroll",
                body=body,
                timeout=300.0,
            )["result"]
            points = result.get("points") or []
            shard_count += len(points)
            for point in points:
                source_id = q2l.source_id_from_scrolled_point(point, num_points)
                if source_id in upper_set:
                    point_to_shards[int(source_id)].append(int(shard_id))
            offset = result.get("next_page_offset")
            if offset is None:
                break
        shard_copy_counts.append(shard_count)
        print(f"recovered shard {shard_id}: {shard_count} copies", flush=True)

    missing = [int(point_id) for point_id in upper_indices.tolist() if not point_to_shards[int(point_id)]]
    if missing:
        preview = ", ".join(str(point_id) for point_id in missing[:10])
        raise RuntimeError(f"missing upper shard membership for {len(missing)} points; first: {preview}")
    return point_to_shards, shard_copy_counts


def size_balanced_placement(shard_keys: list[str], shard_copy_counts: list[int], peer_count: int, q2l: Any) -> dict[str, int]:
    if peer_count <= 0:
        raise ValueError("peer_count must be positive")
    sizes = {
        q2l.shard_key_for_id(shard_id): int(count)
        for shard_id, count in enumerate(shard_copy_counts)
    }
    peer_sizes = [0 for _ in range(peer_count)]
    placement: dict[str, int] = {}
    for shard_key in sorted(shard_keys, key=lambda key: (-sizes.get(key, 0), key)):
        peer_id = min(range(peer_count), key=lambda idx: (peer_sizes[idx], idx))
        placement[shard_key] = peer_id
        peer_sizes[peer_id] += sizes.get(shard_key, 0)
    return dict(sorted(placement.items(), key=lambda item: q2l.shard_key_sort_key(item[0])))


def placement_storage_summary(
    placement: dict[str, int],
    shard_copy_counts: list[int],
    peer_count: int,
    q2l: Any,
) -> dict[str, Any]:
    shard_counts = [0 for _ in range(peer_count)]
    copy_counts = [0 for _ in range(peer_count)]
    for shard_key, peer_id in placement.items():
        shard_id = int(shard_key.rsplit("_", 1)[1])
        shard_counts[int(peer_id)] += 1
        copy_counts[int(peer_id)] += int(shard_copy_counts[shard_id])
    return {
        "shards_per_worker": " ".join(str(value) for value in shard_counts),
        "copies_per_worker": " ".join(str(value) for value in copy_counts),
        "storage_copy_cv_pct": cv_pct([float(value) for value in copy_counts]),
        "storage_max_copy_count": max(copy_counts) if copy_counts else 0,
        "storage_min_copy_count": min(copy_counts) if copy_counts else 0,
    }


def query_distribution_metrics(
    traces: list[dict[str, float]],
    placement: dict[str, int],
    peer_count: int,
    q2l: Any,
) -> dict[str, Any]:
    max_shard_counts: list[float] = []
    active_peer_counts: list[float] = []
    peer_query_participation = [0 for _ in range(peer_count)]

    for trace in traces:
        peer_shard_counts = [0 for _ in range(peer_count)]
        for shard_key in trace:
            peer_id = int(placement[shard_key])
            peer_shard_counts[peer_id] += 1
        active = [count for count in peer_shard_counts if count > 0]
        max_shard_counts.append(float(max(active) if active else 0))
        active_peer_counts.append(float(len(active)))
        for peer_id, count in enumerate(peer_shard_counts):
            if count > 0:
                peer_query_participation[peer_id] += 1

    return {
        "avg_query_max_peer_shard_count": sum(max_shard_counts) / len(max_shard_counts) if max_shard_counts else 0.0,
        "p95_query_max_peer_shard_count": q2l.percentile(max_shard_counts, 95),
        "max_query_max_peer_shard_count": max(max_shard_counts) if max_shard_counts else 0.0,
        "avg_active_workers_per_query": sum(active_peer_counts) / len(active_peer_counts) if active_peer_counts else 0.0,
        "p95_active_workers_per_query": q2l.percentile(active_peer_counts, 95),
        "worker_query_participation_cv_pct": cv_pct([float(value) for value in peer_query_participation]),
        "queries_touching_worker": " ".join(str(value) for value in peer_query_participation),
    }


def co_routed_cut_metrics(
    traces: list[dict[str, float]],
    placement: dict[str, int],
    q2l: Any,
) -> dict[str, Any]:
    total_pairs = 0
    cross_pairs = 0
    weighted_total = 0.0
    weighted_cross = 0.0
    per_query_cross_pct: list[float] = []

    for trace in traces:
        items = list(trace.items())
        if len(items) < 2:
            continue
        query_pairs = 0
        query_cross = 0
        for left_idx in range(len(items)):
            left_key, left_cost = items[left_idx]
            for right_key, right_cost in items[left_idx + 1 :]:
                total_pairs += 1
                query_pairs += 1
                weight = min(float(left_cost), float(right_cost))
                weighted_total += weight
                if placement[left_key] != placement[right_key]:
                    cross_pairs += 1
                    query_cross += 1
                    weighted_cross += weight
        if query_pairs:
            per_query_cross_pct.append(query_cross / query_pairs * 100.0)

    return {
        "co_routed_pair_count": total_pairs,
        "co_routed_pair_cross_worker_pct": cross_pairs / total_pairs * 100.0 if total_pairs else 0.0,
        "co_routed_pair_same_worker_pct": (total_pairs - cross_pairs) / total_pairs * 100.0 if total_pairs else 0.0,
        "weighted_co_routed_pair_cross_worker_pct": (
            weighted_cross / weighted_total * 100.0 if weighted_total > 0 else 0.0
        ),
        "avg_query_co_routed_pair_cross_worker_pct": (
            sum(per_query_cross_pct) / len(per_query_cross_pct) if per_query_cross_pct else 0.0
        ),
        "p95_query_co_routed_pair_cross_worker_pct": q2l.percentile(per_query_cross_pct, 95),
    }


def summarize_placement(
    traces: list[dict[str, float]],
    placement: dict[str, int],
    placement_name: str,
    worker_count: int,
    shard_copy_counts: list[int],
    args: argparse.Namespace,
    q2l: Any,
) -> dict[str, Any]:
    load_metrics = q2l.evaluate_query_peer_loads(traces, placement)
    row: dict[str, Any] = {
        "worker_count": worker_count,
        "placement": placement_name,
        "query_count": len(traces),
        "shard_count": len(placement),
        "upper_k": args.upper_k,
        "base_ef": args.base_ef,
        "factor": args.factor,
    }
    row.update(load_metrics)
    row.update(query_distribution_metrics(traces, placement, worker_count, q2l))
    row.update(co_routed_cut_metrics(traces, placement, q2l))
    row.update(placement_storage_summary(placement, shard_copy_counts, worker_count, q2l))
    return row


def improvement_pct(before: float, after: float) -> float:
    if before == 0:
        return 0.0
    return (before - after) / before * 100.0


def build_delta_rows(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_worker: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in summary_rows:
        by_worker[int(row["worker_count"])][str(row["placement"])] = row

    lower_is_better = [
        "avg_query_max_peer_load",
        "p95_query_max_peer_load",
        "max_query_max_peer_load",
        "peer_load_cv_pct",
        "avg_query_max_peer_shard_count",
        "p95_query_max_peer_shard_count",
        "worker_query_participation_cv_pct",
        "storage_copy_cv_pct",
    ]
    delta_rows: list[dict[str, Any]] = []
    for worker_count, rows in sorted(by_worker.items()):
        baseline = rows.get("round_robin")
        if baseline is None:
            continue
        for placement_name, row in sorted(rows.items()):
            if placement_name == "round_robin":
                continue
            delta: dict[str, Any] = {
                "worker_count": worker_count,
                "placement": placement_name,
                "baseline": "round_robin",
            }
            for metric in lower_is_better:
                delta[f"{metric}_improvement_pct"] = improvement_pct(float(baseline[metric]), float(row[metric]))
            delta["co_routed_pair_cross_worker_delta_pp"] = (
                float(row["co_routed_pair_cross_worker_pct"])
                - float(baseline["co_routed_pair_cross_worker_pct"])
            )
            delta["weighted_co_routed_pair_cross_worker_delta_pp"] = (
                float(row["weighted_co_routed_pair_cross_worker_pct"])
                - float(baseline["weighted_co_routed_pair_cross_worker_pct"])
            )
            delta_rows.append(delta)
    return delta_rows


def main() -> int:
    args = parse_args()
    q2l = load_qdrant_tool(args.qdrant_tool)
    output_dir = Path(args.output_root) / time.strftime("analysis_%Y%m%d_%H%M%S", time.gmtime())
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"output_dir {output_dir}", flush=True)

    upper_vectors, eval_queries, num_points, dim = load_upper_and_eval_data(args, q2l)
    upper_indices = q2l.global_upper_indices(num_points, args.sample_denominator, args.upper_sample_seed)
    point_to_shards, shard_copy_counts = recover_upper_membership_and_counts(args, q2l, upper_indices, num_points)

    upper_index = q2l.build_upper_index(
        upper_vectors,
        upper_indices.astype(np.int64, copy=False),
        dim,
        args.upper_m,
        args.upper_ef_construction,
        max(args.upper_search_ef, args.upper_k),
    )
    upper_labels, _distances = upper_index.knn_query(eval_queries, k=args.upper_k)
    query_plans = q2l.build_routed_search_plans(
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
        source_id_dedup_block_size=None,
    )
    traces = q2l.query_shard_cost_traces(query_plans)
    shard_keys = sorted({shard_key for trace in traces for shard_key in trace}, key=q2l.shard_key_sort_key)

    summary_rows: list[dict[str, Any]] = []
    placement_maps: dict[str, Any] = {}
    for worker_count in args.worker_counts:
        round_robin = q2l.round_robin_simulated_placement(shard_keys, worker_count)
        size_balanced = size_balanced_placement(shard_keys, shard_copy_counts, worker_count, q2l)
        method4_aware = q2l.greedy_method4_aware_placement(
            traces,
            worker_count,
            initial_placement=round_robin,
        )
        placements = {
            "round_robin": round_robin,
            "size_balanced": size_balanced,
            "method4_aware": method4_aware,
        }
        placement_maps[str(worker_count)] = placements
        for placement_name, placement in placements.items():
            row = summarize_placement(
                traces,
                placement,
                placement_name,
                worker_count,
                shard_copy_counts,
                args,
                q2l,
            )
            summary_rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)

    delta_rows = build_delta_rows(summary_rows)
    write_csv(output_dir / "placement_offline_summary.csv", summary_rows)
    write_csv(output_dir / "placement_offline_deltas.csv", delta_rows)
    write_csv(
        output_dir / "collection_shard_counts.csv",
        [
            {
                "shard_id": shard_id,
                "shard_key": q2l.shard_key_for_id(shard_id),
                "scrolled_points": count,
            }
            for shard_id, count in enumerate(shard_copy_counts)
        ],
    )
    (output_dir / "placement_maps.json").write_text(
        json.dumps(placement_maps, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    metadata = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "analysis_kind": "claim_g_method4_aware_physical_layout_offline_matrix",
        "base_url": args.base_url,
        "collection": args.collection,
        "hdf5_path": args.hdf5_path,
        "num_points": num_points,
        "num_shards": args.num_shards,
        "worker_counts": args.worker_counts,
        "upper_k": args.upper_k,
        "base_ef": args.base_ef,
        "factor": args.factor,
        "eval_query_count": args.eval_query_count,
        "sample_denominator": args.sample_denominator,
        "upper_sample_seed": args.upper_sample_seed,
        "upper_count": int(len(upper_indices)),
        "query_count": len(traces),
        "shard_count_in_traces": len(shard_keys),
        "placements": ["round_robin", "size_balanced", "method4_aware"],
        "notes": [
            "Costs are per-query per-shard dynamic EF values extracted from Method4 compact MultiEP query plans.",
            "P95 query max peer load is an offline tail-load proxy, not measured online latency.",
            "Size-balanced placement greedily balances scrolled point-copy counts per worker.",
            "Method4-aware placement reuses greedy_method4_aware_placement from qdrant_two_level_routing_experiment.py.",
        ],
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"wrote {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
