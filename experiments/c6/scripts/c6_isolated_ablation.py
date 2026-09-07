#!/usr/bin/env python3
"""Produce the protocol-isolated E2 and E3 comparisons from frozen C6 traces."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from c6_analyze import (  # noqa: E402
    assignment_memberships_for_truth,
    calibration_rows,
    load_ground_truth,
)
from c6_protocol import (  # noqa: E402
    EF_GRID,
    TARGET_RECALL,
    evaluate_policy_query,
    local_trace_lookup,
    oracle_local_budget_query,
    oracle_prefix_query,
    oracle_query_row,
    paired_bootstrap_relative_reduction,
    policy_config_from_dict,
    read_local_search_jsonl,
    route_queries_from_trace,
    summarize_query_rows,
    utc_timestamp,
    write_csv_atomic,
    write_json_atomic,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("sift1m", "glove-200-angular"), required=True)
    parser.add_argument("--logical-shards", type=int, choices=(4, 8, 16, 32), required=True)
    parser.add_argument("--tuning-csv", required=True)
    parser.add_argument("--route-trace", required=True)
    parser.add_argument("--local-search-traces", action="append", required=True)
    parser.add_argument("--hdf5-path", required=True)
    parser.add_argument("--query-start", type=int, required=True)
    parser.add_argument("--query-count", type=int, required=True)
    parser.add_argument("--assignments-jsonl", required=True)
    parser.add_argument("--oracle-high-ef", type=int, default=max(EF_GRID))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--score-higher-is-better", action="store_true")
    return parser.parse_args(argv)


def load_tuning_rows(path: str | Path) -> list[dict[str, Any]]:
    result = []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            decoded = dict(row)
            decoded["config"] = json.loads(decoded["config"])
            for key in (
                "recall_at_10",
                "aggregate_distance_computations_per_query",
                "p99_max_shard_distance_computations",
            ):
                decoded[key] = float(decoded[key])
            result.append(decoded)
    return result


def select_e2(rows: Sequence[Mapping[str, Any]], logical_shards: int) -> dict[str, Any]:
    full_fanout = sorted(
        (
            row
            for row in rows
            if row["policy"] == "P0"
            and int(row["config"].get("fixed_p", 0)) == logical_shards
            and float(row["recall_at_10"]) >= TARGET_RECALL
        ),
        key=lambda row: int(row["config"]["uniform_ef_search"]),
    )
    if not full_fanout:
        raise RuntimeError("E2 could not find a full-fanout EF reaching target recall")
    common_ef = int(full_fanout[0]["config"]["uniform_ef_search"])
    static = sorted(
        (
            row
            for row in rows
            if row["policy"] == "P0"
            and int(row["config"].get("uniform_ef_search", 0)) == common_ef
            and float(row["recall_at_10"]) >= TARGET_RECALL
        ),
        key=lambda row: int(row["config"]["fixed_p"]),
    )
    if not static:
        raise RuntimeError("E2 could not find a static P at the common EF")
    adaptive = next(
        (
            row
            for row in rows
            if row["policy"] == "P1"
            and int(row["config"].get("uniform_ef_search", 0)) == common_ef
        ),
        None,
    )
    if adaptive is None:
        raise RuntimeError("E2 tuning CSV is missing P1 at the common EF")
    return {
        "common_ef_search": common_ef,
        "static_p": int(static[0]["config"]["fixed_p"]),
        "p0_tuning": dict(static[0]),
        "p1_tuning": dict(adaptive),
    }


def select_e3(
    rows: Sequence[Mapping[str, Any]], *, fixed_p: int
) -> dict[str, Any]:
    uniform = sorted(
        (
            row
            for row in rows
            if row["policy"] == "P0"
            and int(row["config"].get("fixed_p", 0)) == fixed_p
            and float(row["recall_at_10"]) >= TARGET_RECALL
        ),
        key=lambda row: int(row["config"]["uniform_ef_search"]),
    )
    if not uniform:
        raise RuntimeError("E3 uniform sweep did not reach target recall")
    adaptive = [
        row
        for row in rows
        if row["policy"] == "P2"
        and int(row["config"].get("fixed_p", 0)) == fixed_p
        and float(row["recall_at_10"]) >= TARGET_RECALL
    ]
    if not adaptive:
        raise RuntimeError("E3 adaptive-EF sweep did not reach target recall")
    best_work = min(
        float(row["aggregate_distance_computations_per_query"])
        for row in adaptive
    )
    near_best = [
        row
        for row in adaptive
        if float(row["aggregate_distance_computations_per_query"]) <= best_work * 1.02
    ]
    selected_adaptive = min(
        near_best,
        key=lambda row: (
            float(row["p99_max_shard_distance_computations"]),
            float(row["aggregate_distance_computations_per_query"]),
        ),
    )
    return {
        "fixed_p": fixed_p,
        "uniform": dict(uniform[0]),
        "adaptive": dict(selected_adaptive),
    }


def evaluate(
    routes: Sequence[Any],
    truth: Any,
    lookup: Any,
    config_payload: Mapping[str, Any],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    config = policy_config_from_dict(config_payload)
    config.validate(args.logical_shards)
    return [
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


def paired_e2(p0: Sequence[Mapping[str, Any]], p1: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    delta_shards = [
        int(left["selected_shard_count"]) - int(right["selected_shard_count"])
        for left, right in zip(p0, p1, strict=True)
    ]
    delta_work = [
        int(left["aggregate_distance_computations"])
        - int(right["aggregate_distance_computations"])
        for left, right in zip(p0, p1, strict=True)
    ]
    return {
        "fraction_fewer_shards": sum(value > 0 for value in delta_shards) / len(delta_shards),
        "fraction_equal_shards": sum(value == 0 for value in delta_shards) / len(delta_shards),
        "fraction_more_shards": sum(value < 0 for value in delta_shards) / len(delta_shards),
        "mean_shards_saved": statistics.fmean(delta_shards),
        "mean_work_saved": statistics.fmean(delta_work),
        "paired_work_relative_reduction": paired_bootstrap_relative_reduction(
            [float(row["aggregate_distance_computations"]) for row in p0],
            [float(row["aggregate_distance_computations"]) for row in p1],
        ),
    }


def average_ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    result = [0.0] * len(values)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and values[order[stop]] == values[order[start]]:
            stop += 1
        rank = (start + 1 + stop) / 2.0
        for position in order[start:stop]:
            result[position] = rank
        start = stop
    return result


def spearman(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_ranks = average_ranks(left)
    right_ranks = average_ranks(right)
    left_mean = statistics.fmean(left_ranks)
    right_mean = statistics.fmean(right_ranks)
    numerator = sum(
        (x - left_mean) * (y - right_mean)
        for x, y in zip(left_ranks, right_ranks, strict=True)
    )
    left_scale = math.sqrt(sum((x - left_mean) ** 2 for x in left_ranks))
    right_scale = math.sqrt(sum((y - right_mean) ** 2 for y in right_ranks))
    if left_scale == 0 or right_scale == 0:
        return None
    return numerator / (left_scale * right_scale)


def summarize_calibration(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    groups = []
    for name in ("1", "2", "3-4", "5-8", ">8"):
        selected = [row for row in rows if row["entry_point_group"] == name]
        groups.append(
            {
                "entry_point_group": name,
                "sample_count": len(selected),
                "probability_contributes_ground_truth": (
                    statistics.fmean(
                        1.0 if row["contributes_ground_truth"] else 0.0
                        for row in selected
                    )
                    if selected
                    else None
                ),
                "mean_ground_truth_contribution": (
                    statistics.fmean(
                        float(row["ground_truth_contribution"]) for row in selected
                    )
                    if selected
                    else None
                ),
                "mean_oracle_local_ef_search": (
                    statistics.fmean(
                        float(row["oracle_local_ef_search"]) for row in selected
                    )
                    if selected
                    else None
                ),
                "mean_oracle_local_distance_computations": (
                    statistics.fmean(
                        float(row["oracle_local_distance_computations"])
                        for row in selected
                    )
                    if selected
                    else None
                ),
            }
        )
    entry_points = [float(row["entry_point_count"]) for row in rows]
    return {
        "sample_count": len(rows),
        "groups": groups,
        "spearman_entry_points_vs_oracle_local_work": spearman(
            entry_points,
            [float(row["oracle_local_distance_computations"]) for row in rows],
        ),
        "spearman_entry_points_vs_ground_truth_contribution": spearman(
            entry_points,
            [float(row["ground_truth_contribution"]) for row in rows],
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    tuning_rows = load_tuning_rows(args.tuning_csv)
    e2 = select_e2(tuning_rows, args.logical_shards)
    e3 = select_e3(tuning_rows, fixed_p=e2["static_p"])
    route_trace = json.loads(Path(args.route_trace).read_text(encoding="utf-8"))
    routes = route_queries_from_trace(route_trace)
    truth, _total = load_ground_truth(args.hdf5_path, args.query_start, args.query_count)
    lookup = local_trace_lookup(
        [
            trace
            for path in args.local_search_traces
            for trace in read_local_search_jsonl(path)
        ]
    )

    e2_p0 = evaluate(routes, truth, lookup, e2["p0_tuning"]["config"], args)
    e2_p1 = evaluate(routes, truth, lookup, e2["p1_tuning"]["config"], args)
    p4_results = [
        oracle_prefix_query(
            route,
            lookup,
            truth[route.query_id],
            high_ef_search=args.oracle_high_ef,
            score_higher_is_better=args.score_higher_is_better,
        )
        for route in routes
    ]
    e2_p4 = [
        oracle_query_row(
            route, result, dataset=args.dataset, logical_shards=args.logical_shards
        )
        for route, result in zip(routes, p4_results, strict=True)
    ]

    e3_p0 = evaluate(routes, truth, lookup, e3["uniform"]["config"], args)
    e3_p2 = evaluate(routes, truth, lookup, e3["adaptive"]["config"], args)
    p5_results = [
        oracle_local_budget_query(
            route,
            route.fixed_policy_shard_order()[: e3["fixed_p"]],
            lookup,
            truth[route.query_id],
            ef_grid=EF_GRID,
            score_higher_is_better=args.score_higher_is_better,
        )
        for route in routes
    ]
    e3_p5 = [
        oracle_query_row(
            route, result, dataset=args.dataset, logical_shards=args.logical_shards
        )
        for route, result in zip(routes, p5_results, strict=True)
    ]

    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_csv_atomic(output / f"{args.run_id}-e2-p0.csv", e2_p0)
    write_csv_atomic(output / f"{args.run_id}-e2-p1.csv", e2_p1)
    write_csv_atomic(output / f"{args.run_id}-e2-p4.csv", e2_p4)
    write_csv_atomic(output / f"{args.run_id}-e3-p0.csv", e3_p0)
    write_csv_atomic(output / f"{args.run_id}-e3-p2.csv", e3_p2)
    write_csv_atomic(output / f"{args.run_id}-e3-p5.csv", e3_p5)

    memberships = assignment_memberships_for_truth(args.assignments_jsonl, truth)
    calibration = calibration_rows(routes, truth, e3_p5, memberships)
    write_csv_atomic(output / f"{args.run_id}-e3-calibration.csv", calibration)
    calibration_summary = summarize_calibration(calibration)
    write_json_atomic(
        output / f"{args.run_id}-e3-calibration-summary.json",
        calibration_summary,
    )
    record = {
        "record_type": "c6_e2_e3_isolated",
        "format_version": 1,
        "timestamp": utc_timestamp(),
        "dataset": args.dataset,
        "logical_shards": args.logical_shards,
        "query_count": len(routes),
        "E2": {
            "selection": e2,
            "P0": summarize_query_rows(e2_p0),
            "P1": summarize_query_rows(e2_p1),
            "P4": summarize_query_rows(e2_p4),
            "paired": paired_e2(e2_p0, e2_p1),
        },
        "E3": {
            "selection": e3,
            "P0": summarize_query_rows(e3_p0),
            "P2": summarize_query_rows(e3_p2),
            "P5": summarize_query_rows(e3_p5),
            "paired_work_relative_reduction": paired_bootstrap_relative_reduction(
                [float(row["aggregate_distance_computations"]) for row in e3_p0],
                [float(row["aggregate_distance_computations"]) for row in e3_p2],
            ),
            "entry_point_calibration": calibration_summary,
        },
    }
    write_json_atomic(output / f"{args.run_id}-e2-e3-summary.json", record)
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
