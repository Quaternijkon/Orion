#!/usr/bin/env python3
"""Tune or measure C6 policies from frozen routing and local-search traces."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from c6_protocol import (  # noqa: E402
    EF_GRID,
    TARGET_RECALL,
    TUNING_QUERY_COUNT,
    append_jsonl,
    classify_static_waste,
    evaluate_policy_query,
    full_oracle_query,
    local_trace_lookup,
    oracle_as_dict,
    oracle_local_budget_query,
    oracle_prefix_query,
    oracle_query_row,
    paired_bootstrap_relative_reduction,
    policy_config_from_dict,
    read_local_search_jsonl,
    route_queries_from_trace,
    select_tuned_candidate,
    sha256_path,
    summarize_query_rows,
    utc_timestamp,
    write_csv_atomic,
    write_json_atomic,
)


def parse_int_grid(value: str) -> tuple[int, ...]:
    try:
        grid = tuple(sorted(set(int(item) for item in value.split(",") if item)))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("EF grid must contain integers") from exc
    if not grid or grid[0] <= 0:
        raise argparse.ArgumentTypeError("EF grid must contain positive integers")
    return grid


def common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", choices=("sift1m", "glove-200-angular"), required=True)
    parser.add_argument("--logical-shards", type=int, choices=(4, 8, 16, 32), required=True)
    parser.add_argument("--route-trace", required=True)
    parser.add_argument("--local-search-traces", action="append", required=True)
    parser.add_argument("--hdf5-path", required=True)
    parser.add_argument("--query-start", type=int, required=True)
    parser.add_argument("--query-count", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--score-higher-is-better", action="store_true")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    tuning = subparsers.add_parser("tune", help="Evaluate and select P0-P3 on 1,000 tuning queries.")
    common_arguments(tuning)
    tuning.add_argument("--policy-matrix", required=True)

    measurement = subparsers.add_parser(
        "measure", help="Evaluate frozen P0-P3 selections and P4-P6 on held-out queries."
    )
    common_arguments(measurement)
    measurement.add_argument("--tuning-selection", required=True)
    measurement.add_argument("--oracle-high-ef", type=int, default=max(EF_GRID))
    measurement.add_argument("--oracle-ef-grid", type=parse_int_grid, default=EF_GRID)
    measurement.add_argument("--assignments-jsonl")
    return parser.parse_args(argv)


def validate_common(args: argparse.Namespace, total_query_count: int) -> None:
    if args.logical_shards <= 0 or args.query_start < 0 or args.query_count <= 0:
        raise ValueError("logical shards and query range must be valid")
    if not args.run_id or Path(args.run_id).name != args.run_id:
        raise ValueError("run-id must be one safe path component")
    if args.query_start + args.query_count > total_query_count:
        raise ValueError("query range exceeds the official query set")
    if args.dataset == "glove-200-angular" and not args.score_higher_is_better:
        raise ValueError("glove-200-angular requires --score-higher-is-better")
    if args.dataset == "sift1m" and args.score_higher_is_better:
        raise ValueError("SIFT1M uses lower-is-better L2 scores")
    if args.command == "tune":
        if args.query_start != 0 or args.query_count != TUNING_QUERY_COUNT:
            raise ValueError("tuning must use exactly official queries [0, 1000)")
    elif args.query_start != TUNING_QUERY_COUNT or args.query_count != (
        total_query_count - TUNING_QUERY_COUNT
    ):
        raise ValueError("measurement must use every official query after the tuning prefix")


def load_ground_truth(
    path: str | Path, query_start: int, query_count: int
) -> tuple[np.ndarray, int]:
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("c6_analyze requires h5py") from exc
    with h5py.File(path, "r") as handle:
        if "neighbors" not in handle:
            raise ValueError("HDF5 dataset is missing neighbors")
        total = len(handle["neighbors"])
        truth = handle["neighbors"][
            query_start : query_start + query_count, :10
        ].astype(np.int64, copy=True)
    return truth, total


def load_policy_matrix(path: str | Path, logical_shards: int) -> dict[str, list[Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("policy matrix must be a JSON object")
    result: dict[str, list[Any]] = {}
    for policy in ("P0", "P1", "P2", "P3"):
        raw_rows = payload.get(policy)
        if not isinstance(raw_rows, list) or not raw_rows:
            raise ValueError(f"policy matrix must contain candidates for {policy}")
        configs = []
        for raw in raw_rows:
            if not isinstance(raw, dict):
                raise ValueError(f"{policy} candidate must be an object")
            normalized = dict(raw)
            if normalized.get("policy", policy) != policy:
                raise ValueError(f"candidate stored under wrong policy key {policy}")
            normalized["policy"] = policy
            config = policy_config_from_dict(normalized)
            config.validate(logical_shards)
            configs.append(config)
        result[policy] = configs
    return result


def load_selected_configs(path: str | Path, logical_shards: int) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    selected = payload.get("selected") if isinstance(payload, dict) else None
    if not isinstance(selected, dict):
        raise ValueError("tuning selection is missing selected policies")
    result = {}
    for policy in ("P0", "P1", "P2", "P3"):
        raw = selected.get(policy)
        config_payload = raw.get("config") if isinstance(raw, dict) else None
        if not isinstance(config_payload, dict):
            raise ValueError(f"tuning selection is missing {policy} config")
        config = policy_config_from_dict(config_payload)
        config.validate(logical_shards)
        if config.policy != policy:
            raise ValueError(f"selected {policy} config has the wrong policy name")
        result[policy] = config
    return result


def evaluate_config(
    routes: Sequence[Any],
    truth: np.ndarray,
    lookup: Mapping[tuple[int, int, int], Any],
    config: Any,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
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
    return rows, summarize_query_rows(rows)


def candidate_record(config: Any, summary: Mapping[str, Any]) -> dict[str, Any]:
    config_payload = {
        key: value
        for key, value in config.__dict__.items()
        if value is not None
    }
    return {
        "policy": config.policy,
        "config_id": canonical_config_id(config_payload),
        "config": config_payload,
        **summary,
    }


def canonical_config_id(config: Mapping[str, Any]) -> str:
    from c6_protocol import canonical_json_sha256

    return canonical_json_sha256(config)[:16]


def tuning(args: argparse.Namespace, routes: Sequence[Any], truth: np.ndarray, lookup: Any) -> int:
    matrix = load_policy_matrix(args.policy_matrix, args.logical_shards)
    candidates: list[dict[str, Any]] = []
    selected: dict[str, Any] = {}
    for policy, configs in matrix.items():
        policy_candidates = []
        for config in configs:
            _rows, summary = evaluate_config(routes, truth, lookup, config, args)
            record = candidate_record(config, summary)
            candidates.append(record)
            policy_candidates.append(record)
        choice = select_tuned_candidate(policy_candidates)
        selected[policy] = {
            "config_id": choice["config_id"],
            "config": choice["config"],
            "summary": {
                key: value
                for key, value in choice.items()
                if key not in {"config_id", "config", "policy"}
            },
        }
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    selection = {
        "format_version": 1,
        "timestamp": utc_timestamp(),
        "run_id": args.run_id,
        "dataset": args.dataset,
        "logical_shards": args.logical_shards,
        "query_start": args.query_start,
        "query_count": args.query_count,
        "target_recall": TARGET_RECALL,
        "sources": source_records(args),
        "selected": selected,
    }
    write_json_atomic(output / f"{args.run_id}-selection.json", selection)
    write_csv_atomic(output / f"{args.run_id}-tuning.csv", candidates)
    append_jsonl(output / "manifest.jsonl", {"record_type": "c6_tuning", **selection})
    print(json.dumps(selection, sort_keys=True))
    return 0


def source_records(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "route_trace_sha256": sha256_path(args.route_trace),
        "local_search_traces": [
            {"path": str(Path(path).resolve()), "sha256": sha256_path(path)}
            for path in args.local_search_traces
        ],
        "dataset_sha256": sha256_path(args.hdf5_path),
    }


def explode_per_shard(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in rows:
        shard_ids = row["selected_shard_ids"]
        entry_points = row["entry_point_count_per_selected_shard"]
        efs = row["assigned_efSearch_per_shard"]
        distances = row["distance_computations_per_shard"]
        nodes = row["nodes_visited_per_shard"]
        cpu = row["worker_cpu_time_per_shard"]
        if not (
            len(shard_ids)
            == len(entry_points)
            == len(efs)
            == len(distances)
            == len(nodes)
        ):
            raise ValueError("per-query shard arrays are not aligned")
        represented_order = list(row["ranked_candidate_shards"])
        represented = set(represented_order)
        fixed_order = represented_order + [
            shard_id
            for shard_id in range(int(row["logical_shards"]))
            if shard_id not in represented
        ]
        for index, shard_id in enumerate(shard_ids):
            result.append(
                {
                    "query_id": row["query_id"],
                    "dataset": row["dataset"],
                    "logical_shards": row["logical_shards"],
                    "policy": row["policy"],
                    "shard_id": shard_id,
                    "route_rank": fixed_order.index(shard_id) + 1,
                    "entry_point_count": entry_points[index],
                    "ef_search": efs[index],
                    "distance_computations": distances[index],
                    "nodes_visited": nodes[index],
                    "worker_cpu_time_us": cpu[index] if index < len(cpu) else None,
                }
            )
    return result


def assignment_memberships_for_truth(
    path: str | Path, truth: np.ndarray
) -> dict[int, tuple[int, ...]]:
    needed = set(int(value) for value in truth.reshape(-1))
    result: dict[int, tuple[int, ...]] = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
                point_id = int(row["id"])
                shards = tuple(int(value) for value in row["shards"])
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid assignment row at line {line_number}") from exc
            if point_id in needed:
                if not shards or len(set(shards)) != len(shards):
                    raise ValueError(f"invalid shard memberships for point {point_id}")
                result[point_id] = shards
    missing = sorted(needed - set(result))
    if missing:
        raise ValueError(f"assignments are missing ground-truth point {missing[0]}")
    return result


def calibration_rows(
    routes: Sequence[Any],
    truth: np.ndarray,
    p5_rows: Sequence[Mapping[str, Any]],
    memberships: Mapping[int, tuple[int, ...]],
) -> list[dict[str, Any]]:
    by_query = {int(row["query_id"]): row for row in p5_rows}
    result = []
    for route in routes:
        row = by_query[route.query_id]
        evidence = route.evidence_by_shard()
        truth_row = [int(value) for value in truth[route.query_id]]
        for index, shard_id in enumerate(row["selected_shard_ids"]):
            contribution = sum(
                1 for point_id in truth_row if shard_id in memberships[point_id]
            )
            ep_count = (
                evidence[shard_id].entry_point_count if shard_id in evidence else 0
            )
            result.append(
                {
                    "query_id": route.query_id,
                    "shard_id": shard_id,
                    "entry_point_count": ep_count,
                    "entry_point_group": (
                        "1"
                        if ep_count == 1
                        else "2"
                        if ep_count == 2
                        else "3-4"
                        if ep_count <= 4
                        else "5-8"
                        if ep_count <= 8
                        else ">8"
                    ),
                    "ground_truth_contribution": contribution,
                    "contributes_ground_truth": contribution > 0,
                    "oracle_local_ef_search": row["assigned_efSearch_per_shard"][index],
                    "oracle_local_distance_computations": row[
                        "distance_computations_per_shard"
                    ][index],
                }
            )
    return result


def measure(args: argparse.Namespace, routes: Sequence[Any], truth: np.ndarray, lookup: Any) -> int:
    selected = load_selected_configs(args.tuning_selection, args.logical_shards)
    output = Path(args.output_dir).expanduser().resolve()
    per_query_dir = output / "per_query"
    per_shard_dir = output / "per_shard"
    oracle_dir = output / "oracle"
    for directory in (output, per_query_dir, per_shard_dir, oracle_dir):
        directory.mkdir(parents=True, exist_ok=True)

    rows_by_policy: dict[str, list[dict[str, Any]]] = {}
    summaries = []
    for policy, config in selected.items():
        rows, summary = evaluate_config(routes, truth, lookup, config, args)
        rows_by_policy[policy] = rows
        summary_row = candidate_record(config, summary)
        summaries.append(summary_row)
        write_csv_atomic(per_query_dir / f"{args.run_id}-{policy}.csv", rows)
        write_csv_atomic(
            per_shard_dir / f"{args.run_id}-{policy}.csv", explode_per_shard(rows)
        )

    p0 = selected["P0"]
    assert p0.fixed_p is not None
    p4_results = []
    p5_results = []
    p6_results = []
    for route in routes:
        ground_truth = truth[route.query_id]
        p4_results.append(
            oracle_prefix_query(
                route,
                lookup,
                ground_truth,
                high_ef_search=args.oracle_high_ef,
                score_higher_is_better=args.score_higher_is_better,
            )
        )
        p5_results.append(
            oracle_local_budget_query(
                route,
                route.fixed_policy_shard_order()[: p0.fixed_p],
                lookup,
                ground_truth,
                ef_grid=args.oracle_ef_grid,
                score_higher_is_better=args.score_higher_is_better,
            )
        )
        p6_results.append(
            full_oracle_query(
                route,
                lookup,
                ground_truth,
                ef_grid=args.oracle_ef_grid,
                score_higher_is_better=args.score_higher_is_better,
            )
        )
    oracle_results = {"P4": p4_results, "P5": p5_results, "P6": p6_results}
    for policy, results in oracle_results.items():
        rows = [
            oracle_query_row(
                route, result, dataset=args.dataset, logical_shards=args.logical_shards
            )
            for route, result in zip(routes, results, strict=True)
        ]
        rows_by_policy[policy] = rows
        summary = summarize_query_rows(rows)
        summaries.append(
            {
                "policy": policy,
                "config_id": "offline_oracle",
                "config": {
                    "high_ef_search": args.oracle_high_ef if policy == "P4" else None,
                    "ef_grid": list(args.oracle_ef_grid) if policy != "P4" else None,
                },
                **summary,
            }
        )
        write_csv_atomic(oracle_dir / f"{args.run_id}-{policy}.csv", rows)
        write_csv_atomic(
            per_shard_dir / f"{args.run_id}-{policy}.csv", explode_per_shard(rows)
        )

    write_csv_atomic(output / f"{args.run_id}-summary.csv", summaries)
    waste = classify_static_waste(rows_by_policy["P0"], p6_results)
    write_json_atomic(output / f"{args.run_id}-static-waste.json", waste)
    paired = {}
    baseline_work = [row["aggregate_distance_computations"] for row in rows_by_policy["P0"]]
    for policy in ("P1", "P2", "P3", "P6"):
        paired[policy] = paired_bootstrap_relative_reduction(
            baseline_work,
            [row["aggregate_distance_computations"] for row in rows_by_policy[policy]],
        )
    write_json_atomic(output / f"{args.run_id}-paired-bootstrap.json", paired)

    if args.assignments_jsonl:
        memberships = assignment_memberships_for_truth(args.assignments_jsonl, truth)
        calibration = calibration_rows(routes, truth, rows_by_policy["P5"], memberships)
        write_csv_atomic(output / f"{args.run_id}-entry-point-calibration.csv", calibration)

    completion = {
        "record_type": "c6_measurement",
        "format_version": 1,
        "timestamp": utc_timestamp(),
        "run_id": args.run_id,
        "dataset": args.dataset,
        "logical_shards": args.logical_shards,
        "query_start": args.query_start,
        "query_count": args.query_count,
        "sources": {
            **source_records(args),
            "tuning_selection_sha256": sha256_path(args.tuning_selection),
            "assignments_sha256": (
                sha256_path(args.assignments_jsonl) if args.assignments_jsonl else None
            ),
        },
        "summaries": summaries,
        "static_waste": waste,
        "paired_bootstrap": paired,
    }
    write_json_atomic(output / f"{args.run_id}-completion.json", completion)
    append_jsonl(output / "manifest.jsonl", completion)
    print(json.dumps(completion, sort_keys=True))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    truth, total_query_count = load_ground_truth(
        args.hdf5_path, args.query_start, args.query_count
    )
    validate_common(args, total_query_count)
    route_trace = json.loads(Path(args.route_trace).read_text(encoding="utf-8"))
    routes = route_queries_from_trace(route_trace)
    if len(routes) != args.query_count:
        raise ValueError("route trace query count differs from requested analysis range")
    artifact_shards = int((route_trace.get("artifact") or {}).get("shard_count") or 0)
    if artifact_shards != args.logical_shards:
        raise ValueError("route trace shard count differs from --logical-shards")
    traces = [
        trace
        for path in args.local_search_traces
        for trace in read_local_search_jsonl(path)
    ]
    lookup = local_trace_lookup(traces)
    if args.command == "tune":
        return tuning(args, routes, truth, lookup)
    if args.oracle_high_ef <= 0:
        raise ValueError("oracle-high-ef must be positive")
    return measure(args, routes, truth, lookup)


if __name__ == "__main__":
    raise SystemExit(main())
