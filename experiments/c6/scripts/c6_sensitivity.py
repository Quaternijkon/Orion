#!/usr/bin/env python3
"""Build and analyze exact C6 E6 P3 sensitivity configurations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from c6_analyze import load_ground_truth  # noqa: E402
from c6_protocol import (  # noqa: E402
    evaluate_policy_query,
    local_trace_lookup,
    policy_config_from_dict,
    read_local_search_jsonl,
    route_queries_from_trace,
    sha256_path,
    summarize_query_rows,
    utc_timestamp,
    write_csv_atomic,
    write_json_atomic,
)


def scaled(value: int, factor: float) -> int:
    return max(1, int(round(value * factor)))


def selected_p3(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    config = dict(payload["selected"]["P3"]["config"])
    config["policy"] = "P3"
    return config


def parameter_configs(base: Mapping[str, Any]) -> list[dict[str, Any]]:
    result = []
    seen = set()
    for field in ("alpha", "beta"):
        default = int(base[field])
        for factor in (0.5, 1.0, 2.0):
            config = dict(base)
            config[field] = scaled(default, factor)
            key = json.dumps(config, sort_keys=True, separators=(",", ":"))
            if key not in seen:
                seen.add(key)
                result.append(config)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    matrix = subparsers.add_parser("matrix")
    matrix.add_argument("--tuning-selection", required=True)
    matrix.add_argument("--output", required=True)
    matrix.add_argument("--default-only", action="store_true")

    analyze = subparsers.add_parser("analyze")
    analyze.add_argument("--dataset", choices=("sift1m", "glove-200-angular"), required=True)
    analyze.add_argument("--logical-shards", type=int, choices=(4, 8, 16, 32), required=True)
    analyze.add_argument("--tuning-selection", required=True)
    analyze.add_argument("--default-route-trace", required=True)
    analyze.add_argument("--default-local-search-traces", action="append", required=True)
    analyze.add_argument("--parameter-local-search-traces", action="append", required=True)
    analyze.add_argument("--navigation-k-default", type=int, required=True)
    analyze.add_argument("--low-k-route-trace", required=True)
    analyze.add_argument("--low-k-local-search-traces", action="append", required=True)
    analyze.add_argument("--high-k-route-trace", required=True)
    analyze.add_argument("--high-k-local-search-traces", action="append", required=True)
    analyze.add_argument("--hdf5-path", required=True)
    analyze.add_argument("--query-start", type=int, default=1000)
    analyze.add_argument("--query-count", type=int, default=9000)
    analyze.add_argument("--output-dir", required=True)
    analyze.add_argument("--run-id", required=True)
    analyze.add_argument("--score-higher-is-better", action="store_true")
    return parser.parse_args(argv)


def lookup(paths: Sequence[str]) -> dict[Any, Any]:
    return local_trace_lookup(
        [trace for path in paths for trace in read_local_search_jsonl(path)]
    )


def evaluate_loaded(
    routes: Sequence[Any],
    traces: Mapping[Any, Any],
    truth: Any,
    config_payload: Mapping[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    config = policy_config_from_dict(config_payload)
    config.validate(args.logical_shards)
    rows = [
        evaluate_policy_query(
            route,
            config,
            traces,
            truth[route.query_id],
            dataset=args.dataset,
            logical_shards=args.logical_shards,
            score_higher_is_better=args.score_higher_is_better,
        )
        for route in routes
    ]
    return summarize_query_rows(rows)


def summary_row(
    args: argparse.Namespace,
    *,
    parameter: str,
    factor: float | None,
    value: int,
    config: Mapping[str, Any],
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "dataset": args.dataset,
        "logical_shards": args.logical_shards,
        "sensitivity_parameter": parameter,
        "factor": factor,
        "value": value,
        "alpha": config["alpha"],
        "beta": config["beta"],
        "navigation_k": (
            value if parameter == "navigation_k" else args.navigation_k_default
        ),
        "recall_at_10": summary["recall_at_10"],
        "recall_gate_valid": summary["recall_at_10"] >= 0.90,
        "mean_shards_per_query": summary["mean_shards_per_query"],
        "p95_shards_per_query": summary["p95_shards_per_query"],
        "aggregate_distance_computations_per_query": summary[
            "aggregate_distance_computations_per_query"
        ],
        "p95_max_shard_distance_computations": summary[
            "p95_max_shard_distance_computations"
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "matrix":
        base = selected_p3(args.tuning_selection)
        configs = [base] if args.default_only else parameter_configs(base)
        write_json_atomic(args.output, {"P3": configs})
        print(json.dumps({"output": str(Path(args.output).resolve()), "configs": len(configs)}))
        return 0

    if args.dataset == "glove-200-angular" and not args.score_higher_is_better:
        raise ValueError("GloVe sensitivity requires higher-is-better scores")
    if args.dataset == "sift1m" and args.score_higher_is_better:
        raise ValueError("SIFT sensitivity uses lower-is-better scores")
    truth, _total = load_ground_truth(args.hdf5_path, args.query_start, args.query_count)
    base = selected_p3(args.tuning_selection)
    default_paths = [*args.default_local_search_traces, *args.parameter_local_search_traces]
    default_routes = route_queries_from_trace(
        json.loads(Path(args.default_route_trace).read_text(encoding="utf-8"))
    )
    default_lookup = lookup(default_paths)
    low_k_routes = route_queries_from_trace(
        json.loads(Path(args.low_k_route_trace).read_text(encoding="utf-8"))
    )
    low_k_lookup = lookup(args.low_k_local_search_traces)
    high_k_routes = route_queries_from_trace(
        json.loads(Path(args.high_k_route_trace).read_text(encoding="utf-8"))
    )
    high_k_lookup = lookup(args.high_k_local_search_traces)
    base_summary = evaluate_loaded(default_routes, default_lookup, truth, base, args)
    rows = []
    for field in ("alpha", "beta"):
        default = int(base[field])
        for factor in (0.5, 1.0, 2.0):
            config = dict(base)
            config[field] = scaled(default, factor)
            summary = (
                base_summary
                if factor == 1.0
                else evaluate_loaded(default_routes, default_lookup, truth, config, args)
            )
            rows.append(
                summary_row(
                    args,
                    parameter=field,
                    factor=factor,
                    value=int(config[field]),
                    config=config,
                    summary=summary,
                )
            )
    low_k = scaled(args.navigation_k_default, 0.5)
    high_k = scaled(args.navigation_k_default, 2.0)
    for navigation_k, factor, routes, traces in (
        (low_k, 0.5, low_k_routes, low_k_lookup),
        (args.navigation_k_default, 1.0, default_routes, default_lookup),
        (high_k, 2.0, high_k_routes, high_k_lookup),
    ):
        summary = base_summary if navigation_k == 48 else evaluate_loaded(
            routes, traces, truth, base, args
        )
        rows.append(
            summary_row(
                args,
                parameter="navigation_k",
                factor=factor,
                value=navigation_k,
                config=base,
                summary=summary,
            )
        )
    rows.append(
        summary_row(
            args,
            parameter="logical_shards",
            factor=None,
            value=args.logical_shards,
            config=base,
            summary=base_summary,
        )
    )
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / f"{args.run_id}-sensitivity.csv"
    write_csv_atomic(csv_path, rows)
    record = {
        "record_type": "c6_e6_sensitivity",
        "timestamp": utc_timestamp(),
        "dataset": args.dataset,
        "logical_shards": args.logical_shards,
        "run_id": args.run_id,
        "base_config": base,
        "sources": {
            "tuning_selection_sha256": sha256_path(args.tuning_selection),
            "default_route_trace_sha256": sha256_path(args.default_route_trace),
            "low_k_route_trace_sha256": sha256_path(args.low_k_route_trace),
            "high_k_route_trace_sha256": sha256_path(args.high_k_route_trace),
        },
        "rows": rows,
    }
    write_json_atomic(output / f"{args.run_id}-sensitivity.json", record)
    print(json.dumps({"output": str(csv_path), "rows": len(rows)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
