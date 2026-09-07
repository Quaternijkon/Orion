#!/usr/bin/env python3
"""Run the Stage 1 P0-P3 smoke audit on an arbitrary small query slice."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from c6_analyze import load_policy_matrix  # noqa: E402
from c6_protocol import (  # noqa: E402
    evaluate_policy_query,
    local_trace_lookup,
    read_local_search_jsonl,
    route_queries_from_trace,
    sha256_path,
    summarize_query_rows,
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
    parser.add_argument("--policy-matrix", required=True)
    parser.add_argument("--frozen-collection", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--per-query-output", required=True)
    parser.add_argument("--score-higher-is-better", action="store_true")
    return parser.parse_args(argv)


def load_truth(path: str | Path, start: int, count: int) -> np.ndarray:
    import h5py

    with h5py.File(path, "r") as handle:
        return handle["neighbors"][start : start + count, :10].astype(
            np.int64, copy=True
        )


def smoke_audit(
    rows_by_policy: dict[str, list[dict[str, Any]]],
    *,
    frozen_collection: dict[str, Any],
) -> dict[str, Any]:
    policies = ("P0", "P1", "P2", "P3")
    if set(rows_by_policy) != set(policies):
        raise ValueError("smoke audit requires exactly P0-P3")
    query_count = len(rows_by_policy["P0"])
    if query_count <= 0 or any(len(rows_by_policy[policy]) != query_count for policy in policies):
        raise ValueError("P0-P3 smoke rows must be non-empty and aligned")
    route_identity_pass = True
    fixed_set_pass = True
    adaptive_set_pass = True
    counters_pass = True
    for index in range(query_count):
        rows = [rows_by_policy[policy][index] for policy in policies]
        if [int(row["query_id"]) for row in rows] != [index] * len(policies):
            raise ValueError("smoke query IDs are not aligned")
        route_identity_pass &= len(
            {
                json.dumps(row["ranked_candidate_shards"], separators=(",", ":"))
                for row in rows
            }
        ) == 1
        fixed_set_pass &= (
            rows_by_policy["P0"][index]["selected_shard_ids"]
            == rows_by_policy["P2"][index]["selected_shard_ids"]
        )
        adaptive_set_pass &= (
            rows_by_policy["P1"][index]["selected_shard_ids"]
            == rows_by_policy["P3"][index]["selected_shard_ids"]
        )
        counters_pass &= all(
            row["aggregate_distance_computations"] > 0
            and row["aggregate_nodes_visited"] > 0
            and row["aggregate_worker_cpu_time_us"] > 0
            for row in rows
        )
    identity = frozen_collection.get("index_identity")
    return {
        "query_count": query_count,
        "same_ranked_routing_candidates_across_policies": route_identity_pass,
        "p0_p2_fixed_shard_sets_identical": fixed_set_pass,
        "p1_p3_adaptive_shard_sets_identical": adaptive_set_pass,
        "positive_per_query_hardware_counters": counters_pass,
        "frozen_collection_identity_present": isinstance(identity, dict),
        "policy_switching_rebuild_required": frozen_collection.get(
            "policy_switching_rebuild_required"
        ),
        "all_stage1_gates_pass": all(
            (
                route_identity_pass,
                fixed_set_pass,
                adaptive_set_pass,
                counters_pass,
                isinstance(identity, dict),
                frozen_collection.get("policy_switching_rebuild_required") is False,
            )
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.query_start < 0 or args.query_count <= 0:
        raise ValueError("query range must be valid")
    route_trace = json.loads(Path(args.route_trace).read_text(encoding="utf-8"))
    routes = route_queries_from_trace(route_trace)
    if len(routes) != args.query_count:
        raise ValueError("route trace query count differs from smoke query count")
    truth = load_truth(args.hdf5_path, args.query_start, args.query_count)
    lookup = local_trace_lookup(read_local_search_jsonl(args.local_search_traces))
    matrix = load_policy_matrix(args.policy_matrix, args.logical_shards)
    if any(len(matrix[policy]) != 1 for policy in matrix):
        raise ValueError("smoke policy matrix must contain one configuration per policy")
    rows_by_policy = {}
    summaries = {}
    flattened = []
    for policy in ("P0", "P1", "P2", "P3"):
        config = matrix[policy][0]
        rows = [
            evaluate_policy_query(
                route,
                config,
                lookup,
                truth[route.query_id],
                dataset=args.dataset,
                logical_shards=args.logical_shards,
                score_higher_is_better=args.score_higher_is_better,
            )
            for route in routes
        ]
        rows_by_policy[policy] = rows
        summaries[policy] = summarize_query_rows(rows)
        flattened.extend(rows)
    frozen_collection = json.loads(Path(args.frozen_collection).read_text(encoding="utf-8"))
    audit = smoke_audit(rows_by_policy, frozen_collection=frozen_collection)
    record = {
        "record_type": "c6_stage1_smoke",
        "format_version": 1,
        "timestamp": utc_timestamp(),
        "dataset": args.dataset,
        "logical_shards": args.logical_shards,
        "query_start": args.query_start,
        "query_count": args.query_count,
        "sources": {
            "route_trace_sha256": sha256_path(args.route_trace),
            "local_search_traces_sha256": sha256_path(args.local_search_traces),
            "dataset_sha256": sha256_path(args.hdf5_path),
            "policy_matrix_sha256": sha256_path(args.policy_matrix),
            "frozen_collection_sha256": sha256_path(args.frozen_collection),
        },
        "summaries": summaries,
        "audit": audit,
    }
    write_csv_atomic(args.per_query_output, flattened)
    write_json_atomic(args.output, record)
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0 if audit["all_stage1_gates_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
