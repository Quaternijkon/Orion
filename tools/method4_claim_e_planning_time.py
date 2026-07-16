#!/usr/bin/env python3
"""Measure Claim E client-side route-planning time.

The timing reported by this tool excludes dataset loading, upper-index build,
and Qdrant scroll/recovery. It measures:

1. upper HNSW routing label computation for the measured query slice;
2. variant-specific routed search plan materialization.

It does not execute lower searches.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import time
from pathlib import Path
from typing import Any


DEFAULT_VARIANTS = [
    "grouped_by_ef_materialized",
    "compact_current",
    "client_shard_major_expanded",
]


FIELDNAMES = [
    "variant",
    "query_count",
    "lower_execution_order",
    "upper_label_time_s",
    "plan_build_time_s",
    "total_planning_time_s",
    "upper_label_ms_per_query",
    "plan_build_ms_per_query",
    "total_planning_ms_per_query",
    "search_request_count",
    "avg_search_requests_per_query",
    "candidate_group_count",
    "avg_candidate_groups_per_query",
    "returned_candidate_count",
    "avg_returned_candidates_per_query",
    "visited_shards",
    "avg_visited_shards",
    "assigned_ef_sum",
    "avg_assigned_ef_sum",
    "caveat",
]


CAVEAT = "client_side_upper_label_and_plan_materialization_only_no_lower_search"


def load_module(path: str | Path, module_name: str) -> Any:
    module_path = Path(path)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def shard_major_search_count(search: dict[str, Any]) -> int:
    shard_keys = search.get("shard_key") or []
    return len(shard_keys) if shard_keys else 1


def executed_request_count(plans: list[dict[str, Any]], *, lower_execution_order: str) -> int:
    if lower_execution_order not in {"query_major", "shard_major"}:
        raise ValueError(f"unsupported lower execution order: {lower_execution_order}")
    total = 0
    for plan in plans:
        for search in plan.get("searches", []):
            total += shard_major_search_count(search) if lower_execution_order == "shard_major" else 1
    return total


def candidate_group_count(plans: list[dict[str, Any]], *, lower_execution_order: str) -> int:
    return executed_request_count(plans, lower_execution_order=lower_execution_order)


def returned_candidate_count(plans: list[dict[str, Any]], *, lower_execution_order: str) -> int:
    total = 0
    if lower_execution_order == "shard_major":
        for plan in plans:
            for search in plan.get("searches", []):
                limit = int(search.get("limit", 0))
                total += shard_major_search_count(search) * limit
        return total
    for plan in plans:
        for search in plan.get("searches", []):
            total += int(search.get("limit", 0))
    return total


def planning_summary_row(
    *,
    variant: str,
    lower_execution_order: str,
    query_count: int,
    upper_label_time_s: float,
    plan_build_time_s: float,
    search_request_count: int,
    candidate_group_count: int,
    returned_candidate_count: int,
    visited_shards: int = 0,
    assigned_ef_sum: int = 0,
) -> dict[str, Any]:
    total = upper_label_time_s + plan_build_time_s
    return {
        "variant": variant,
        "query_count": query_count,
        "lower_execution_order": lower_execution_order,
        "upper_label_time_s": upper_label_time_s,
        "plan_build_time_s": plan_build_time_s,
        "total_planning_time_s": total,
        "upper_label_ms_per_query": upper_label_time_s * 1000.0 / query_count if query_count else 0.0,
        "plan_build_ms_per_query": plan_build_time_s * 1000.0 / query_count if query_count else 0.0,
        "total_planning_ms_per_query": total * 1000.0 / query_count if query_count else 0.0,
        "search_request_count": search_request_count,
        "avg_search_requests_per_query": search_request_count / query_count if query_count else 0.0,
        "candidate_group_count": candidate_group_count,
        "avg_candidate_groups_per_query": candidate_group_count / query_count if query_count else 0.0,
        "returned_candidate_count": returned_candidate_count,
        "avg_returned_candidates_per_query": returned_candidate_count / query_count if query_count else 0.0,
        "visited_shards": visited_shards,
        "avg_visited_shards": visited_shards / query_count if query_count else 0.0,
        "assigned_ef_sum": assigned_ef_sum,
        "avg_assigned_ef_sum": assigned_ef_sum / query_count if query_count else 0.0,
        "caveat": CAVEAT,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
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
    parser.add_argument("--variant", action="append", default=None, choices=DEFAULT_VARIANTS)
    parser.add_argument("--sample-denominator", type=int, default=32)
    parser.add_argument("--upper-sample-seed", type=int, default=100)
    parser.add_argument("--upper-m", type=int, default=32)
    parser.add_argument("--upper-ef-construction", type=int, default=100)
    parser.add_argument("--upper-search-ef", type=int, default=200)
    parser.add_argument("--scroll-page-size", type=int, default=20000)
    parser.add_argument("--output-root", default="results/method4_claim_e_planning_time_20260705")
    parser.add_argument("--qdrant-tool", default="tools/qdrant_two_level_routing_experiment.py")
    parser.add_argument("--claim-e-tool", default="tools/method4_claim_e_execution_mode_latency.py")
    return parser.parse_args()


def build_plans_from_labels(
    *,
    q2l: Any,
    queries: Any,
    upper_labels: Any,
    point_to_shards: list[list[int]],
    num_shards: int,
    top_k: int,
    base_ef: int,
    factor: int,
    routed_execution_mode: str,
    source_id_dedup_block_size: int,
) -> list[dict[str, Any]]:
    return q2l.build_routed_search_plans(
        queries,
        upper_labels.astype("int64", copy=False),
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


def main() -> int:
    args = parse_args()
    q2l = load_module(args.qdrant_tool, "qdrant_two_level_routing_experiment")
    claim_e = load_module(args.claim_e_tool, "method4_claim_e_execution_mode_latency")
    specs = claim_e.variant_specs(args.variant)
    output_dir = Path(args.output_root) / time.strftime("analysis_%Y%m%d_%H%M%S", time.gmtime())
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"output_dir {output_dir}", flush=True)

    setup_started = time.perf_counter()
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
    setup_wall_s = time.perf_counter() - setup_started

    measured_queries = queries[args.warmup_query_count :]
    upper_started = time.perf_counter()
    upper_labels = q2l.compute_upper_labels(upper_index, measured_queries, args.upper_k)
    upper_label_time_s = time.perf_counter() - upper_started

    rows: list[dict[str, Any]] = []
    for spec in specs:
        build_started = time.perf_counter()
        plans = build_plans_from_labels(
            q2l=q2l,
            queries=measured_queries,
            upper_labels=upper_labels,
            point_to_shards=point_to_shards,
            num_shards=args.num_shards,
            top_k=args.top_k,
            base_ef=args.base_ef,
            factor=args.factor,
            routed_execution_mode=spec["routed_execution_mode"],
            source_id_dedup_block_size=train_count + 1,
        )
        plan_build_time_s = time.perf_counter() - build_started
        row = planning_summary_row(
            variant=spec["variant"],
            lower_execution_order=spec["lower_execution_order"],
            query_count=len(measured_queries),
            upper_label_time_s=upper_label_time_s,
            plan_build_time_s=plan_build_time_s,
            search_request_count=executed_request_count(
                plans,
                lower_execution_order=spec["lower_execution_order"],
            ),
            candidate_group_count=candidate_group_count(
                plans,
                lower_execution_order=spec["lower_execution_order"],
            ),
            returned_candidate_count=returned_candidate_count(
                plans,
                lower_execution_order=spec["lower_execution_order"],
            ),
            visited_shards=sum(int(plan.get("visited_shards", 0)) for plan in plans),
            assigned_ef_sum=sum(int(plan.get("assigned_ef_sum", 0)) for plan in plans),
        )
        print(json.dumps(row, ensure_ascii=False), flush=True)
        rows.append(row)

    write_csv(output_dir / "claim_e_planning_time_summary.csv", rows)
    metadata = {
        "analysis_kind": "claim_e_client_side_route_planning_time",
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
        "warmup_query_count_excluded_from_timing": args.warmup_query_count,
        "setup_wall_s_excluded_from_timing": setup_wall_s,
        "variants": specs,
        "notes": [
            CAVEAT,
            "Timing excludes dataset load, upper-index build, and Qdrant scroll/recovery.",
            "upper_label_time_s is shared across variants; plan_build_time_s is variant-specific route-plan materialization.",
            "No lower search requests are executed.",
        ],
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"wrote {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
