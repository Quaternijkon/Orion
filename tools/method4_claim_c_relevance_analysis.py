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


DEFAULT_CONFIGS = [
    {
        "label": "fixed_width_claim_c_095",
        "upper_k": 120,
        "base_ef": 80,
        "factor": 10,
        "fixed_ef_reference": 240,
    },
    {
        "label": "frontier_095_dynamic",
        "upper_k": 80,
        "base_ef": 80,
        "factor": 20,
        "fixed_ef_reference": 280,
    },
    {
        "label": "frontier_097_dynamic",
        "upper_k": 200,
        "base_ef": 60,
        "factor": 20,
        "fixed_ef_reference": 480,
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Offline Claim C relevance analysis: test whether routed entry-point "
            "count predicts shards containing ground-truth top-k copies, and "
            "whether Dynamic EF allocates more budget to those shards than a "
            "fixed-EF baseline."
        )
    )
    parser.add_argument("--base-url", default="http://localhost:6833")
    parser.add_argument("--collection", default="bench095_rr_orion_s31")
    parser.add_argument("--hdf5-path", default="/home/taig/dry/faiss/datasets/glove-200-angular.hdf5")
    parser.add_argument("--num-shards", type=int, default=46)
    parser.add_argument("--sample-denominator", type=int, default=32)
    parser.add_argument("--upper-sample-seed", type=int, default=100)
    parser.add_argument("--upper-m", type=int, default=32)
    parser.add_argument("--upper-ef-construction", type=int, default=100)
    parser.add_argument("--upper-search-ef", type=int, default=400)
    parser.add_argument("--eval-query-count", type=int, default=3000)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--scroll-page-size", type=int, default=20000)
    parser.add_argument(
        "--output-root",
        default="results/method4_claim_c_dynamic_ef_relevance_20260704",
    )
    parser.add_argument(
        "--qdrant-tool",
        default="tools/qdrant_two_level_routing_experiment.py",
        help="Path to the Method4 experiment module to reuse routing helpers.",
    )
    return parser.parse_args()


def load_qdrant_tool(path: str | Path) -> Any:
    module_path = Path(path)
    spec = importlib.util.spec_from_file_location("qdrant_two_level_routing_experiment", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_upper_and_eval_data(args: argparse.Namespace, q2l: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    with h5py.File(args.hdf5_path, "r") as handle:
        train_ds = handle["train"]
        num_points = int(train_ds.shape[0])
        dim = int(train_ds.shape[1])
        upper_indices = q2l.global_upper_indices(num_points, args.sample_denominator, args.upper_sample_seed)

        # h5py fancy indexing requires increasing indices; restore the original
        # sampled order before building the upper HNSW labels.
        order = np.argsort(upper_indices)
        sorted_upper = upper_indices[order]
        upper_vectors_sorted = train_ds[sorted_upper].astype(np.float32, copy=True)
        inverse = np.empty_like(order)
        inverse[order] = np.arange(len(order))
        upper_vectors = upper_vectors_sorted[inverse]

        eval_queries = handle["test"][: args.eval_query_count].astype(np.float32, copy=True)
        eval_neighbors = handle["neighbors"][: args.eval_query_count, : args.top_k].astype(np.int64, copy=True)

    return q2l.normalize_rows(upper_vectors), q2l.normalize_rows(eval_queries), eval_neighbors, num_points, dim


def recover_point_to_shards(args: argparse.Namespace, q2l: Any, num_points: int) -> tuple[list[list[int]], list[int]]:
    point_to_shards: list[list[int]] = [[] for _ in range(num_points)]
    per_shard_counts: list[int] = []
    for shard_id in range(args.num_shards):
        shard_key = q2l.shard_key_for_id(shard_id)
        offset: Any = None
        count = 0
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
            count += len(points)
            for point in points:
                source_id = q2l.source_id_from_scrolled_point(point, num_points)
                if 0 <= source_id < num_points:
                    point_to_shards[source_id].append(int(shard_id))
            offset = result.get("next_page_offset")
            if offset is None:
                break
        per_shard_counts.append(count)
        print(f"recovered shard {shard_id}: {count} point copies", flush=True)
    return point_to_shards, per_shard_counts


def bucket_for_count(ep_count: int) -> str:
    return str(ep_count) if ep_count <= 8 else "9+"


def analyze_config(
    q2l: Any,
    labels: np.ndarray,
    eval_neighbors: np.ndarray,
    point_to_shards: list[list[int]],
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    upper_k = int(config["upper_k"])
    base_ef = int(config["base_ef"])
    factor = int(config["factor"])
    fixed_ef_reference = int(config["fixed_ef_reference"])

    buckets: dict[str, dict[str, float]] = defaultdict(
        lambda: {
            "visited_shards": 0,
            "queries_with_bucket": 0,
            "gt_hit_shards": 0,
            "gt_copy_count": 0,
            "dynamic_ef_sum": 0,
            "dynamic_ef_on_gt_hit_shards": 0,
            "fixed_ef_sum": 0,
            "fixed_ef_on_gt_hit_shards": 0,
        }
    )
    query_bucket_seen: dict[str, set[int]] = defaultdict(set)
    total: dict[str, float] = defaultdict(float)
    total["queries"] = len(eval_neighbors)

    for query_idx in range(len(eval_neighbors)):
        shard_to_eps = q2l.route_upper_labels_to_shard_eps(
            labels[query_idx, :upper_k].tolist(),
            point_to_shards,
        )
        gt_shard_copy_counts: dict[int, int] = defaultdict(int)
        for gt_id in eval_neighbors[query_idx].tolist():
            for shard_id in point_to_shards[int(gt_id)]:
                gt_shard_copy_counts[int(shard_id)] += 1
        total["total_gt_copy_count"] += sum(gt_shard_copy_counts.values())

        query_has_gt_visited = False
        for shard_id, eps in shard_to_eps.items():
            ep_count = len(eps)
            if ep_count <= 0:
                continue
            bucket = bucket_for_count(ep_count)
            dynamic_ef = base_ef + factor * ep_count
            fixed_ef = fixed_ef_reference
            gt_copies = int(gt_shard_copy_counts.get(int(shard_id), 0))
            gt_hit = gt_copies > 0

            query_bucket_seen[bucket].add(query_idx)
            bucket_row = buckets[bucket]
            bucket_row["visited_shards"] += 1
            bucket_row["gt_hit_shards"] += int(gt_hit)
            bucket_row["gt_copy_count"] += gt_copies
            bucket_row["dynamic_ef_sum"] += dynamic_ef
            bucket_row["fixed_ef_sum"] += fixed_ef

            if gt_hit:
                bucket_row["dynamic_ef_on_gt_hit_shards"] += dynamic_ef
                bucket_row["fixed_ef_on_gt_hit_shards"] += fixed_ef
                total["routed_ep_count_hit_sum"] += ep_count
                total["hit_obs"] += 1
                query_has_gt_visited = True
            else:
                total["routed_ep_count_nonhit_sum"] += ep_count
                total["nonhit_obs"] += 1

            total["visited_shards"] += 1
            total["gt_hit_shards"] += int(gt_hit)
            total["gt_copy_count"] += gt_copies
            total["visited_gt_copy_count"] += gt_copies
            total["dynamic_ef_sum"] += dynamic_ef
            total["fixed_ef_sum"] += fixed_ef
            if gt_hit:
                total["dynamic_ef_on_gt_hit_shards"] += dynamic_ef
                total["fixed_ef_on_gt_hit_shards"] += fixed_ef
        total["queries_with_any_gt_visited"] += int(query_has_gt_visited)

    for bucket, query_indices in query_bucket_seen.items():
        buckets[bucket]["queries_with_bucket"] = len(query_indices)

    bucket_rows: list[dict[str, Any]] = []
    for bucket in sorted(buckets, key=lambda value: 99 if value == "9+" else int(value)):
        row = buckets[bucket]
        visited = row["visited_shards"]
        bucket_rows.append(
            {
                "config_label": config["label"],
                "upper_k": upper_k,
                "base_ef": base_ef,
                "factor": factor,
                "fixed_ef_reference": fixed_ef_reference,
                "routed_ep_count_bucket": bucket,
                "visited_shard_observations": int(visited),
                "queries_with_bucket": int(row["queries_with_bucket"]),
                "gt_hit_shards": int(row["gt_hit_shards"]),
                "gt_hit_rate": row["gt_hit_shards"] / visited if visited else 0.0,
                "avg_gt_copies_per_visited_shard": row["gt_copy_count"] / visited if visited else 0.0,
                "avg_dynamic_ef": row["dynamic_ef_sum"] / visited if visited else 0.0,
                "dynamic_ef_sum": int(row["dynamic_ef_sum"]),
                "dynamic_ef_on_gt_hit_shards": int(row["dynamic_ef_on_gt_hit_shards"]),
                "fixed_ef_sum": int(row["fixed_ef_sum"]),
                "fixed_ef_on_gt_hit_shards": int(row["fixed_ef_on_gt_hit_shards"]),
            }
        )

    dynamic_gt_share = total["dynamic_ef_on_gt_hit_shards"] / total["dynamic_ef_sum"]
    fixed_gt_share = total["fixed_ef_on_gt_hit_shards"] / total["fixed_ef_sum"]
    hit_mean_ep = total["routed_ep_count_hit_sum"] / total["hit_obs"]
    nonhit_mean_ep = total["routed_ep_count_nonhit_sum"] / total["nonhit_obs"]
    summary_row = {
        "config_label": config["label"],
        "upper_k": upper_k,
        "base_ef": base_ef,
        "factor": factor,
        "fixed_ef_reference": fixed_ef_reference,
        "query_count": int(total["queries"]),
        "visited_shard_observations": int(total["visited_shards"]),
        "avg_visited_shards": total["visited_shards"] / total["queries"],
        "gt_hit_shards": int(total["gt_hit_shards"]),
        "gt_hit_rate": total["gt_hit_shards"] / total["visited_shards"],
        "queries_with_any_gt_visited": int(total["queries_with_any_gt_visited"]),
        "query_gt_visited_rate": total["queries_with_any_gt_visited"] / total["queries"],
        "avg_routed_ep_count_on_gt_hit_shards": hit_mean_ep,
        "avg_routed_ep_count_on_nonhit_shards": nonhit_mean_ep,
        "routed_ep_count_hit_minus_nonhit": hit_mean_ep - nonhit_mean_ep,
        "dynamic_ef_sum": int(total["dynamic_ef_sum"]),
        "fixed_ef_sum": int(total["fixed_ef_sum"]),
        "dynamic_ef_on_gt_hit_shards": int(total["dynamic_ef_on_gt_hit_shards"]),
        "fixed_ef_on_gt_hit_shards": int(total["fixed_ef_on_gt_hit_shards"]),
        "dynamic_gt_budget_share": dynamic_gt_share,
        "fixed_gt_budget_share": fixed_gt_share,
        "dynamic_minus_fixed_gt_budget_share": dynamic_gt_share - fixed_gt_share,
        "total_gt_copy_count": int(total["total_gt_copy_count"]),
        "visited_gt_copy_count": int(total["visited_gt_copy_count"]),
        "visited_gt_copy_coverage": (
            total["visited_gt_copy_count"] / total["total_gt_copy_count"]
            if total["total_gt_copy_count"]
            else 0.0
        ),
    }
    return bucket_rows, summary_row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    q2l = load_qdrant_tool(args.qdrant_tool)
    output_dir = Path(args.output_root) / time.strftime("analysis_%Y%m%d_%H%M%S", time.gmtime())
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"output_dir {output_dir}", flush=True)

    upper_vectors, eval_queries, eval_neighbors, num_points, dim = load_upper_and_eval_data(args, q2l)
    upper_indices = q2l.global_upper_indices(num_points, args.sample_denominator, args.upper_sample_seed)
    max_upper_k = max(int(config["upper_k"]) for config in DEFAULT_CONFIGS)
    upper_index = q2l.build_upper_index(
        upper_vectors,
        upper_indices.astype(np.int64, copy=False),
        dim,
        args.upper_m,
        args.upper_ef_construction,
        args.upper_search_ef,
    )
    labels, _distances = upper_index.knn_query(eval_queries, k=max_upper_k)
    labels = labels.astype(np.int64, copy=False)

    point_to_shards, per_shard_counts = recover_point_to_shards(args, q2l, num_points)
    missing = sum(1 for shards in point_to_shards if not shards)
    assigned_copies = sum(len(shards) for shards in point_to_shards)
    print(f"recovered assigned copies {assigned_copies}; missing logical points {missing}", flush=True)

    bucket_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for config in DEFAULT_CONFIGS:
        config_bucket_rows, config_summary = analyze_config(q2l, labels, eval_neighbors, point_to_shards, config)
        bucket_rows.extend(config_bucket_rows)
        summary_rows.append(config_summary)

    write_csv(output_dir / "routed_ep_relevance_by_count.csv", bucket_rows)
    write_csv(output_dir / "budget_alignment_summary.csv", summary_rows)
    write_csv(
        output_dir / "collection_shard_counts.csv",
        [
            {
                "shard_id": shard_id,
                "shard_key": q2l.shard_key_for_id(shard_id),
                "scrolled_points": count,
            }
            for shard_id, count in enumerate(per_shard_counts)
        ],
    )
    metadata = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "analysis_kind": "offline_routed_ep_count_vs_ground_truth_shard_relevance",
        "base_url": args.base_url,
        "collection": args.collection,
        "hdf5_path": args.hdf5_path,
        "num_points": num_points,
        "num_shards": args.num_shards,
        "assigned_copies_recovered": assigned_copies,
        "missing_logical_points": missing,
        "sample_denominator": args.sample_denominator,
        "upper_sample_seed": args.upper_sample_seed,
        "upper_count": int(len(upper_indices)),
        "upper_search_ef": args.upper_search_ef,
        "eval_query_count": args.eval_query_count,
        "top_k": args.top_k,
        "configs": DEFAULT_CONFIGS,
        "notes": [
            "A shard is GT-hit if it contains at least one ground-truth top-k source_id copy for the query.",
            "routed_ep_count is the number of routed upper entry points assigned to that shard for the query.",
            "Dynamic EF is base_ef + factor * routed_ep_count. Fixed EF reference is constant per visited shard.",
            "This is an offline relevance-proxy analysis and does not execute lower HNSW search.",
        ],
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("wrote", output_dir, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
