#!/usr/bin/env python3
"""Measure selected Claim E modes with Qdrant /metrics deltas.

This supplement reuses the current Claim E routed execution harness, but samples
Qdrant's Prometheus `/metrics` endpoint immediately before and after each
measured variant repeat. It records REST search/batch counters and duration
deltas, collection hardware CPU deltas when exposed, and process/memory gauge
snapshots from Qdrant itself.

The metrics are Qdrant-exposed process/server counters. They are not packet
capture, not build-stage resource accounting, and not subsystem-level CPU
attribution inside Qdrant.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import re
import time
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_VARIANTS = [
    "grouped_by_ef_materialized",
    "compact_current",
    "client_shard_major_expanded",
]

SEARCH_BATCH_LABELS = {
    "method": "POST",
    "endpoint": "/collections/{collection_name}/points/search/batch",
    "status": "200",
}

CAVEAT = (
    "qdrant_prometheus_metrics_not_packet_capture_or_subsystem_internal_attribution"
)


def load_module(path: str | Path, module_name: str) -> Any:
    module_path = Path(path)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_prometheus_labels(raw: str) -> dict[str, str]:
    labels: dict[str, str] = {}
    if not raw:
        return labels
    pattern = re.compile(r'([A-Za-z_][A-Za-z0-9_]*)="((?:\\.|[^"\\])*)"')
    for match in pattern.finditer(raw):
        value = (
            match.group(2)
            .replace(r"\\", "\\")
            .replace(r"\"", '"')
            .replace(r"\n", "\n")
        )
        labels[match.group(1)] = value
    return labels


def split_prometheus_sample(line: str) -> tuple[str, str | None, str] | None:
    if "{" not in line:
        parts = line.split()
        if len(parts) < 2:
            return None
        return parts[0], None, parts[1]

    brace_start = line.find("{")
    name = line[:brace_start]
    if not re.match(r"^[A-Za-z_:][A-Za-z0-9_:]*$", name):
        return None

    in_quote = False
    escaped = False
    brace_end: int | None = None
    for index in range(brace_start + 1, len(line)):
        char = line[index]
        if escaped:
            escaped = False
            continue
        if char == "\\" and in_quote:
            escaped = True
            continue
        if char == '"':
            in_quote = not in_quote
            continue
        if char == "}" and not in_quote:
            brace_end = index
            break
    if brace_end is None:
        return None

    rest = line[brace_end + 1 :].split()
    if not rest:
        return None
    return name, line[brace_start + 1 : brace_end], rest[0]


def parse_prometheus_metrics(text: str) -> dict[tuple[str, tuple[tuple[str, str], ...]], float]:
    samples: dict[tuple[str, tuple[tuple[str, str], ...]], float] = {}
    value_re = re.compile(
        r"^[-+]?(?:[0-9]*\.[0-9]+|[0-9]+)(?:[eE][-+]?[0-9]+)?$|^[-+]?Inf$|^NaN$"
    )
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parsed = split_prometheus_sample(line)
        if parsed is None:
            continue
        name, raw_labels, raw_value = parsed
        if not value_re.match(raw_value):
            continue
        if raw_value in {"Inf", "+Inf"}:
            value = math.inf
        elif raw_value == "-Inf":
            value = -math.inf
        elif raw_value == "NaN":
            value = math.nan
        else:
            value = float(raw_value)
        labels = tuple(sorted(parse_prometheus_labels(raw_labels or "").items()))
        samples[(name, labels)] = value
    return samples


def metric_value(
    samples: dict[tuple[str, tuple[tuple[str, str], ...]], float],
    name: str,
    labels: dict[str, str] | None = None,
    default: float = 0.0,
) -> float:
    key = (name, tuple(sorted((labels or {}).items())))
    return float(samples.get(key, default))


def scrape_qdrant_metrics(base_url: str) -> dict[tuple[str, tuple[tuple[str, str], ...]], float]:
    url = base_url.rstrip("/") + "/metrics"
    with urllib.request.urlopen(url, timeout=30.0) as response:
        text = response.read().decode("utf-8")
    return parse_prometheus_metrics(text)


def metric_delta(
    before: dict[tuple[str, tuple[tuple[str, str], ...]], float],
    after: dict[tuple[str, tuple[tuple[str, str], ...]], float],
    name: str,
    labels: dict[str, str] | None = None,
) -> float:
    return metric_value(after, name, labels) - metric_value(before, name, labels)


def qdrant_metric_summary(
    before: dict[tuple[str, tuple[tuple[str, str], ...]], float],
    after: dict[tuple[str, tuple[tuple[str, str], ...]], float],
    *,
    collection: str,
) -> dict[str, float]:
    search_count_delta = metric_delta(
        before,
        after,
        "rest_responses_total",
        SEARCH_BATCH_LABELS,
    )
    search_duration_delta = metric_delta(
        before,
        after,
        "rest_responses_duration_seconds_sum",
        SEARCH_BATCH_LABELS,
    )
    return {
        "qdrant_search_batch_count_delta": search_count_delta,
        "qdrant_search_batch_duration_s_delta": search_duration_delta,
        "qdrant_search_batch_avg_duration_ms_delta_window": (
            search_duration_delta / search_count_delta * 1000.0
            if search_count_delta > 0
            else 0.0
        ),
        "qdrant_search_batch_max_duration_s_after": metric_value(
            after,
            "rest_responses_max_duration_seconds",
            SEARCH_BATCH_LABELS,
        ),
        "qdrant_collection_cpu_delta": metric_delta(
            before,
            after,
            "collection_hardware_metric_cpu",
            {"id": collection},
        ),
        "qdrant_memory_resident_bytes_before": metric_value(before, "memory_resident_bytes"),
        "qdrant_memory_resident_bytes_after": metric_value(after, "memory_resident_bytes"),
        "qdrant_memory_resident_bytes_delta": metric_delta(
            before,
            after,
            "memory_resident_bytes",
        ),
        "qdrant_memory_allocated_bytes_before": metric_value(before, "memory_allocated_bytes"),
        "qdrant_memory_allocated_bytes_after": metric_value(after, "memory_allocated_bytes"),
        "qdrant_memory_allocated_bytes_delta": metric_delta(
            before,
            after,
            "memory_allocated_bytes",
        ),
        "qdrant_process_threads_before": metric_value(before, "process_threads"),
        "qdrant_process_threads_after": metric_value(after, "process_threads"),
        "qdrant_process_open_fds_before": metric_value(before, "process_open_fds"),
        "qdrant_process_open_fds_after": metric_value(after, "process_open_fds"),
        "qdrant_process_minor_page_faults_delta": metric_delta(
            before,
            after,
            "process_minor_page_faults_total",
        ),
        "qdrant_process_major_page_faults_delta": metric_delta(
            before,
            after,
            "process_major_page_faults_total",
        ),
    }


def flatten_metric_snapshot(
    samples: dict[tuple[str, tuple[tuple[str, str], ...]], float],
    *,
    variant: str,
    repeat: int,
    phase: str,
) -> list[dict[str, Any]]:
    wanted = {
        "memory_resident_bytes",
        "memory_allocated_bytes",
        "memory_active_bytes",
        "memory_retained_bytes",
        "process_threads",
        "process_open_fds",
        "process_minor_page_faults_total",
        "process_major_page_faults_total",
        "rest_responses_total",
        "rest_responses_duration_seconds_sum",
        "rest_responses_duration_seconds_count",
        "rest_responses_max_duration_seconds",
        "collection_hardware_metric_cpu",
    }
    rows: list[dict[str, Any]] = []
    for (name, labels), value in sorted(samples.items()):
        if name not in wanted:
            continue
        rows.append(
            {
                "variant": variant,
                "repeat": repeat,
                "phase": phase,
                "metric": name,
                "labels_json": json.dumps(dict(labels), sort_keys=True),
                "value": value,
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_variant_with_qdrant_metrics(
    *,
    args: argparse.Namespace,
    q2l: Any,
    claim_e: Any,
    spec: dict[str, str],
    plans: list[dict[str, Any]],
    neighbors: Any,
    repeat: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    warmup = args.warmup_query_count
    if warmup:
        q2l.execute_query_plans_once(
            args.base_url,
            args.collection,
            plans[:warmup],
            neighbors[:warmup],
            args.top_k,
            lower_execution_order=spec["lower_execution_order"],
        )

    before = scrape_qdrant_metrics(args.base_url)
    measured_plans = plans[warmup:]
    measured_neighbors = neighbors[warmup:]
    batch_rows: list[dict[str, Any]] = []
    try:
        for batch_index, start_idx in enumerate(range(0, len(measured_plans), args.batch_size)):
            end_idx = min(start_idx + args.batch_size, len(measured_plans))
            started = time.perf_counter()
            result = q2l.execute_query_plans_once(
                args.base_url,
                args.collection,
                measured_plans[start_idx:end_idx],
                measured_neighbors[start_idx:end_idx],
                args.top_k,
                lower_execution_order=spec["lower_execution_order"],
            )
            wall_s = time.perf_counter() - started
            batch_rows.append(
                {
                    "variant": spec["variant"],
                    "description": spec["description"],
                    "routed_execution_mode": spec["routed_execution_mode"],
                    "lower_execution_order": spec["lower_execution_order"],
                    "collection": args.collection,
                    "repeat": repeat,
                    "batch_size": args.batch_size,
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
    finally:
        after = scrape_qdrant_metrics(args.base_url)

    summary = claim_e.summarize_variant_rows(spec["variant"], batch_rows, args.top_k)
    summary.update(
        {
            "status": "ok",
            "description": spec["description"],
            "routed_execution_mode": spec["routed_execution_mode"],
            "lower_execution_order": spec["lower_execution_order"],
            "collection": args.collection,
            "repeat": repeat,
            "batch_size": args.batch_size,
            "upper_k": args.upper_k,
            "base_ef": args.base_ef,
            "factor": args.factor,
            "warmup_query_count": warmup,
            **qdrant_metric_summary(before, after, collection=args.collection),
            "caveat": CAVEAT,
        }
    )
    metric_rows = flatten_metric_snapshot(before, variant=spec["variant"], repeat=repeat, phase="before")
    metric_rows.extend(flatten_metric_snapshot(after, variant=spec["variant"], repeat=repeat, phase="after"))
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return summary, batch_rows, metric_rows


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
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--variant", action="append", default=None, choices=DEFAULT_VARIANTS)
    parser.add_argument("--sample-denominator", type=int, default=32)
    parser.add_argument("--upper-sample-seed", type=int, default=100)
    parser.add_argument("--upper-m", type=int, default=32)
    parser.add_argument("--upper-ef-construction", type=int, default=100)
    parser.add_argument("--upper-search-ef", type=int, default=200)
    parser.add_argument("--scroll-page-size", type=int, default=20000)
    parser.add_argument("--output-root", default="results/method4_claim_e_qdrant_metrics_20260705")
    parser.add_argument("--qdrant-tool", default="tools/qdrant_two_level_routing_experiment.py")
    parser.add_argument("--claim-e-tool", default="tools/method4_claim_e_execution_mode_latency.py")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    q2l = load_module(args.qdrant_tool, "qdrant_two_level_routing_experiment")
    claim_e = load_module(args.claim_e_tool, "method4_claim_e_execution_mode_latency")
    specs = claim_e.variant_specs(args.variant)
    output_dir = Path(args.output_root) / time.strftime("analysis_%Y%m%d_%H%M%S", time.gmtime())
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"output_dir {output_dir}", flush=True)

    queries, upper_vectors, neighbors, train_count, dim = claim_e.load_queries_and_neighbors(args, q2l)
    upper_indices = q2l.global_upper_indices(
        train_count,
        args.sample_denominator,
        args.upper_sample_seed,
    )
    upper_index = q2l.build_upper_index(
        upper_vectors,
        upper_indices.astype("int64", copy=False),
        dim,
        args.upper_m,
        args.upper_ef_construction,
        max(args.upper_search_ef, args.upper_k),
    )
    routing_source = args.routing_source_collection or args.collection
    point_to_shards = claim_e.recover_upper_membership(
        args,
        q2l,
        routing_source,
        upper_indices,
        train_count,
    )

    plans_by_mode: dict[str, list[dict[str, Any]]] = {}
    for spec in specs:
        mode = spec["routed_execution_mode"]
        if mode in plans_by_mode:
            continue
        plans_by_mode[mode] = claim_e.build_plans_for_execution_mode(
            q2l,
            queries,
            upper_index,
            point_to_shards,
            args.num_shards,
            args.top_k,
            args.upper_k,
            args.base_ef,
            args.factor,
            mode,
            train_count + 1,
        )

    summary_rows: list[dict[str, Any]] = []
    batch_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    for repeat in range(1, args.repeats + 1):
        ordered_specs = specs if repeat % 2 == 1 else list(reversed(specs))
        for spec in ordered_specs:
            summary, rows, metrics = run_variant_with_qdrant_metrics(
                args=args,
                q2l=q2l,
                claim_e=claim_e,
                spec=spec,
                plans=plans_by_mode[spec["routed_execution_mode"]],
                neighbors=neighbors,
                repeat=repeat,
            )
            summary_rows.append(summary)
            batch_rows.extend(rows)
            metric_rows.extend(metrics)

    write_csv(output_dir / "claim_e_qdrant_metrics_summary.csv", summary_rows)
    write_csv(output_dir / "claim_e_qdrant_metrics_batches.csv", batch_rows)
    write_csv(output_dir / "claim_e_qdrant_metrics_samples.csv", metric_rows)
    metadata = {
        "analysis_kind": "claim_e_qdrant_prometheus_metrics_supplement",
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
        "batch_size": args.batch_size,
        "repeats": args.repeats,
        "variants": specs,
        "metrics_caveat": CAVEAT,
        "notes": [
            "Metrics are scraped from Qdrant /metrics before and after each measured variant repeat.",
            "Warmup queries run before the before-metrics snapshot.",
            "The route-plan construction and scroll-based upper membership recovery happen before measured snapshots.",
            "Metrics are Qdrant-exposed server/process counters, not packet capture, build-stage RSS, or subsystem-level CPU attribution.",
        ],
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"wrote {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
