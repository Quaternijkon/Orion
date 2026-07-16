#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np


DEFAULT_STRATEGIES = [
    {
        "strategy": "orion_single",
        "use_multi_assign": False,
        "multi_assign_min_max_vote": 2,
        "multi_assign_vote_delta": 0,
        "multi_assign_max_shards": 0,
    },
    {
        "strategy": "orion_default",
        "use_multi_assign": True,
        "multi_assign_min_max_vote": 2,
        "multi_assign_vote_delta": 0,
        "multi_assign_max_shards": 0,
    },
    {
        "strategy": "orion_w2c2",
        "use_multi_assign": True,
        "multi_assign_min_max_vote": 2,
        "multi_assign_vote_delta": 2,
        "multi_assign_max_shards": 2,
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Claim B offline oracle analysis. Compare Orion single assignment "
            "with voting multi-assignment by measuring GT top-k shard coverage "
            "under the same upper routing labels."
        )
    )
    parser.add_argument("--hdf5-path", default="/home/taig/dry/faiss/datasets/glove-200-angular.hdf5")
    parser.add_argument("--num-shards", type=int, default=31)
    parser.add_argument("--upper-ks", type=int, nargs="+", default=[80, 120, 160])
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--eval-query-count", type=int, default=3000)
    parser.add_argument("--sample-denominator", type=int, default=32)
    parser.add_argument("--upper-sample-seed", type=int, default=100)
    parser.add_argument("--upper-m", type=int, default=32)
    parser.add_argument("--upper-ef-construction", type=int, default=100)
    parser.add_argument("--upper-search-ef", type=int, default=400)
    parser.add_argument("--k-overlap", type=int, default=10)
    parser.add_argument("--topology-iters", type=int, default=50)
    parser.add_argument("--upper-build-batch-size", type=int, default=10000)
    parser.add_argument("--kmeans-iters", type=int, default=10)
    parser.add_argument("--kmeans-rand-seed", type=int, default=1)
    parser.add_argument("--disable-fission", action="store_true")
    parser.add_argument("--output-root", default="results/method4_claim_b_oracle_miss_20260704")
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


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), pct))


def route_shards_for_labels(labels: np.ndarray, point_to_shards: list[list[int]], upper_k: int) -> set[int]:
    routed: set[int] = set()
    for label in labels[:upper_k].tolist():
        point_id = int(label)
        if 0 <= point_id < len(point_to_shards):
            routed.update(int(shard_id) for shard_id in point_to_shards[point_id])
    return routed


def analyze_oracle(
    strategy: str,
    labels: np.ndarray,
    neighbors: np.ndarray,
    point_to_shards: list[list[int]],
    upper_ks: list[int],
    expansion_ratio: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    query_count = int(len(neighbors))
    top_k = int(neighbors.shape[1]) if neighbors.ndim == 2 else 0

    for upper_k in upper_ks:
        covered_items = 0
        total_items = 0
        covered_copies = 0
        total_copies = 0
        all_covered_queries = 0
        any_covered_queries = 0
        routed_shard_counts: list[float] = []
        min_shards_for_gt: list[float] = []

        for query_idx in range(query_count):
            routed_shards = route_shards_for_labels(labels[query_idx], point_to_shards, int(upper_k))
            routed_shard_counts.append(float(len(routed_shards)))
            query_covered_items = 0
            query_gt_shards: set[int] = set()

            for gt_id in neighbors[query_idx].tolist():
                gt_shards = {
                    int(shard_id)
                    for shard_id in point_to_shards[int(gt_id)]
                    if 0 <= int(shard_id)
                }
                query_gt_shards.update(gt_shards)
                total_items += 1
                total_copies += len(gt_shards)
                copy_hits = len(gt_shards & routed_shards)
                covered_copies += copy_hits
                if copy_hits > 0:
                    covered_items += 1
                    query_covered_items += 1

            min_shards_for_gt.append(float(len(query_gt_shards)))
            if query_covered_items > 0:
                any_covered_queries += 1
            if query_covered_items == top_k:
                all_covered_queries += 1

        coverage = covered_items / total_items if total_items else 0.0
        copy_coverage = covered_copies / total_copies if total_copies else 0.0
        row = {
            "strategy": strategy,
            "upper_k": int(upper_k),
            "query_count": query_count,
            "top_k": top_k,
            f"oracle_gt_coverage_at_{top_k}": coverage,
            f"oracle_gt_miss_at_{top_k}": 1.0 - coverage,
            f"oracle_gt_copy_coverage_at_{top_k}": copy_coverage,
            "query_any_gt_covered_rate": any_covered_queries / query_count if query_count else 0.0,
            "query_all_gt_covered_rate": all_covered_queries / query_count if query_count else 0.0,
            "avg_routed_shards": sum(routed_shard_counts) / query_count if query_count else 0.0,
            "p95_routed_shards": percentile(routed_shard_counts, 95),
            "avg_min_shards_for_gt": sum(min_shards_for_gt) / query_count if query_count else 0.0,
            "p95_min_shards_for_gt": percentile(min_shards_for_gt, 95),
            "index_expansion_ratio": float(expansion_ratio),
        }
        rows.append(row)

    return rows


def load_data(args: argparse.Namespace, q2l: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    with h5py.File(args.hdf5_path, "r") as handle:
        train = handle["train"][:].astype(np.float32, copy=True)
        queries = handle["test"][: args.eval_query_count].astype(np.float32, copy=True)
        neighbors = handle["neighbors"][: args.eval_query_count, : args.top_k].astype(np.int64, copy=True)
    train = q2l.normalize_rows(train)
    queries = q2l.normalize_rows(queries)
    upper_indices = q2l.global_upper_indices(len(train), args.sample_denominator, args.upper_sample_seed)
    return train, queries, neighbors, upper_indices, int(train.shape[1])


def build_routing_strategy(
    q2l: Any,
    train: np.ndarray,
    upper_indices: np.ndarray,
    point_to_l1s: list[list[int]],
    args: argparse.Namespace,
    config: dict[str, Any],
) -> Any:
    return q2l.build_original_routing_state(
        train,
        upper_indices,
        point_to_l1s,
        args.num_shards,
        args.kmeans_iters,
        args.kmeans_rand_seed,
        args.topology_iters,
        bool(config["use_multi_assign"]),
        not args.disable_fission,
        int(config["multi_assign_min_max_vote"]),
        int(config["multi_assign_vote_delta"]),
        int(config["multi_assign_max_shards"]),
    )


def main() -> int:
    args = parse_args()
    q2l = load_qdrant_tool(args.qdrant_tool)
    output_dir = Path(args.output_root) / time.strftime("analysis_%Y%m%d_%H%M%S", time.gmtime())
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"output_dir {output_dir}", flush=True)

    train, queries, neighbors, upper_indices, dim = load_data(args, q2l)
    upper_index = q2l.build_upper_index(
        train[upper_indices],
        upper_indices.astype(np.int64, copy=False),
        dim,
        args.upper_m,
        args.upper_ef_construction,
        args.upper_search_ef,
    )
    max_upper_k = max(int(value) for value in args.upper_ks)
    labels = q2l.compute_upper_labels(upper_index, queries, max_upper_k).astype(np.int64, copy=False)
    print("computing point_to_l1s", flush=True)
    point_to_l1s = q2l.compute_point_to_l1s(
        upper_index,
        train,
        args.k_overlap,
        args.upper_build_batch_size,
    )

    oracle_rows: list[dict[str, Any]] = []
    strategy_rows: list[dict[str, Any]] = []
    for config in DEFAULT_STRATEGIES:
        strategy = str(config["strategy"])
        print(f"building routing strategy {strategy}", flush=True)
        routing_state = build_routing_strategy(q2l, train, upper_indices, point_to_l1s, args, config)
        strategy_rows.append(
            {
                "strategy": strategy,
                "initial_num_shards": routing_state.initial_num_shards,
                "effective_num_shards": routing_state.num_shards,
                "total_assigned": routing_state.total_assigned,
                "index_expansion_ratio": routing_state.expansion_ratio,
                "topology_iterations": routing_state.topology_iterations,
                "fission_event_count": len(routing_state.fission_events),
                "use_multi_assign": config["use_multi_assign"],
                "multi_assign_min_max_vote": config["multi_assign_min_max_vote"],
                "multi_assign_vote_delta": config["multi_assign_vote_delta"],
                "multi_assign_max_shards": config["multi_assign_max_shards"],
            }
        )
        oracle_rows.extend(
            analyze_oracle(
                strategy,
                labels,
                neighbors,
                routing_state.point_to_shards,
                [int(value) for value in args.upper_ks],
                routing_state.expansion_ratio,
            )
        )

    write_csv(output_dir / "oracle_gt_miss_summary.csv", oracle_rows)
    write_csv(output_dir / "strategy_build_summary.csv", strategy_rows)
    metadata = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "analysis_kind": "method4_claim_b_orion_multiassign_oracle_gt_miss",
        "hdf5_path": args.hdf5_path,
        "num_points": int(len(train)),
        "initial_num_shards": args.num_shards,
        "upper_ks": [int(value) for value in args.upper_ks],
        "top_k": args.top_k,
        "eval_query_count": args.eval_query_count,
        "sample_denominator": args.sample_denominator,
        "upper_sample_seed": args.upper_sample_seed,
        "upper_count": int(len(upper_indices)),
        "upper_search_ef": args.upper_search_ef,
        "k_overlap": args.k_overlap,
        "topology_iters": args.topology_iters,
        "disable_fission": bool(args.disable_fission),
        "strategies": DEFAULT_STRATEGIES,
        "oracle_summary_csv": str(output_dir / "oracle_gt_miss_summary.csv"),
        "strategy_build_summary_csv": str(output_dir / "strategy_build_summary.csv"),
        "notes": [
            "oracle_gt_coverage counts each GT top-k source as covered if any copy is in the routed shard set.",
            "oracle_gt_miss is 1 - oracle_gt_coverage.",
            "This is an offline routing oracle analysis and does not execute lower HNSW search.",
        ],
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"wrote {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
