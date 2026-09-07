#!/usr/bin/env python3
"""Compute C6 E1 oracle shard and local-budget difficulty distributions."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from c6_analyze import load_ground_truth  # noqa: E402
from c6_protocol import (  # noqa: E402
    EF_GRID,
    full_oracle_query,
    local_trace_lookup,
    oracle_as_dict,
    oracle_prefix_query,
    read_local_search_jsonl,
    route_queries_from_trace,
    sha256_path,
    utc_timestamp,
    write_csv_atomic,
    write_json_atomic,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("sift1m", "glove-200-angular"), required=True)
    parser.add_argument("--logical-shards", type=int, choices=(4, 8, 16, 32), required=True)
    parser.add_argument("--route-trace", required=True)
    parser.add_argument("--local-search-traces", required=True)
    parser.add_argument("--hdf5-path", required=True)
    parser.add_argument("--query-start", type=int, required=True)
    parser.add_argument("--query-count", type=int, required=True)
    parser.add_argument("--oracle-high-ef", type=int, default=max(EF_GRID))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--score-higher-is-better", action="store_true")
    return parser.parse_args(argv)


def distribution(values: Sequence[float]) -> dict[str, float]:
    if not values:
        raise ValueError("cannot summarize an empty distribution")
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p75": float(np.percentile(array, 75)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
        "standard_deviation": float(np.std(array)),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    trace = json.loads(Path(args.route_trace).read_text(encoding="utf-8"))
    routes = route_queries_from_trace(trace)
    if len(routes) != args.query_count:
        raise ValueError("route trace query count differs from E1 query count")
    truth, _total = load_ground_truth(args.hdf5_path, args.query_start, args.query_count)
    lookup = local_trace_lookup(read_local_search_jsonl(args.local_search_traces))
    p4 = []
    p6 = []
    per_shard = []
    for route in routes:
        ground_truth = truth[route.query_id]
        prefix = oracle_prefix_query(
            route,
            lookup,
            ground_truth,
            high_ef_search=args.oracle_high_ef,
            score_higher_is_better=args.score_higher_is_better,
        )
        full = full_oracle_query(
            route,
            lookup,
            ground_truth,
            ef_grid=EF_GRID,
            score_higher_is_better=args.score_higher_is_better,
        )
        p4.append(prefix)
        p6.append(full)
        for index, shard_id in enumerate(full.selected_shards):
            per_shard.append(
                {
                    "query_id": route.query_id,
                    "shard_id": shard_id,
                    "route_rank": route.fixed_policy_shard_order().index(shard_id) + 1,
                    "entry_point_count": (
                        route.evidence_by_shard()[shard_id].entry_point_count
                        if shard_id in route.evidence_by_shard()
                        else 0
                    ),
                    "oracle_ef_search": full.assigned_efs[index],
                    "distance_computations": full.distance_computations_per_shard[index],
                    "nodes_visited": full.nodes_visited_per_shard[index],
                }
            )
    p4_rows = [oracle_as_dict(row) for row in p4]
    p6_rows = [oracle_as_dict(row) for row in p6]
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_csv_atomic(output / f"{args.run_id}-p4.csv", p4_rows)
    write_csv_atomic(output / f"{args.run_id}-p6.csv", p6_rows)
    write_csv_atomic(output / f"{args.run_id}-p6-per-shard.csv", per_shard)
    prefix_counts = [len(row.selected_shards) for row in p4]
    p6_mean_ef = [statistics.fmean(row.assigned_efs) for row in p6]
    p6_max_ef = [max(row.assigned_efs) for row in p6]
    summary = {
        "record_type": "c6_e1_query_difficulty",
        "format_version": 1,
        "timestamp": utc_timestamp(),
        "dataset": args.dataset,
        "logical_shards": args.logical_shards,
        "query_count": len(routes),
        "oracle_prefix_shards": distribution(prefix_counts),
        "oracle_prefix_fraction": distribution(
            [value / args.logical_shards for value in prefix_counts]
        ),
        "oracle_total_work": distribution(
            [row.aggregate_distance_computations for row in p6]
        ),
        "oracle_max_shard_work": distribution(
            [row.max_shard_distance_computations for row in p6]
        ),
        "oracle_mean_local_ef": distribution(p6_mean_ef),
        "oracle_max_local_ef": distribution(p6_max_ef),
        "target_reached_fraction": statistics.fmean(
            1.0 if row.reached_target else 0.0 for row in p6
        ),
        "sources": {
            "route_trace_sha256": sha256_path(args.route_trace),
            "local_search_traces_sha256": sha256_path(args.local_search_traces),
            "dataset_sha256": sha256_path(args.hdf5_path),
        },
    }
    write_json_atomic(output / f"{args.run_id}-summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
