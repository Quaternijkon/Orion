#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import re
import shlex
import statistics
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, NamedTuple


class BenchmarkRun(NamedTuple):
    run_id: str
    case_name: str
    preset: str
    args: dict[str, Any]
    output_dir: Path
    tags: dict[str, Any]


class CollectionCleanupTarget(NamedTuple):
    base_url: str
    collection: str


PRESET_ARGS: dict[str, dict[str, Any]] = {
    "orion": {
        "routing_mode": "faithful_original_rest",
        "routed_execution_mode": "compact_multi_ep",
        "routed_planning_mode": "materialized",
        "routed_result_limit_mode": "top_k",
        "search_dispatch_mode": "coordinator",
    },
    "naive": {
        "routing_mode": "naive_hash_all_shards",
        "search_dispatch_mode": "coordinator",
    },
    # KMeans-only ablation inside the Method4 routing framework: keep upper
    # routing / voting assignment path, but disable topology convergence and
    # fission so the L1 shard map remains the balanced KMeans initialization.
    "kmeans": {
        "routing_mode": "faithful_original_rest",
        "topology_iters": 0,
        "disable_fission": True,
        "routed_execution_mode": "compact_multi_ep",
        "routed_planning_mode": "materialized",
        "routed_result_limit_mode": "top_k",
        "search_dispatch_mode": "coordinator",
    },
    "method4_kmeans_ablation": {
        "routing_mode": "faithful_original_rest",
        "topology_iters": 0,
        "disable_fission": True,
        "routed_execution_mode": "compact_multi_ep",
        "routed_planning_mode": "materialized",
        "routed_result_limit_mode": "top_k",
        "search_dispatch_mode": "coordinator",
    },
    "kmeans_independent": {
        "routing_mode": "cpp_kmeans_baseline",
        "routed_execution_mode": "compact_multi_ep",
        "routed_planning_mode": "materialized",
        "routed_result_limit_mode": "top_k",
        "search_dispatch_mode": "coordinator",
    },
    "kmeans_simple_nprobe": {
        "routing_mode": "kmeans_simple_nprobe",
        "search_dispatch_mode": "coordinator",
    },
    # Legacy centroid routing is intentionally separate from the KMeans-only
    # Method4 ablation because it uses a different older routing pipeline.
    "legacy_centroid": {
        "routing_mode": "legacy_centroid",
        "routed_execution_mode": "compact_multi_ep",
        "search_dispatch_mode": "coordinator",
    },
}

PRESET_TAGS: dict[str, dict[str, Any]] = {
    "orion": {"partition_family": "orion"},
    "naive": {"partition_family": "naive_all_shards"},
    "kmeans": {"partition_family": "method4_kmeans_ablation"},
    "method4_kmeans_ablation": {"partition_family": "method4_kmeans_ablation"},
    "kmeans_independent": {"partition_family": "kmeans_independent"},
    "kmeans_simple_nprobe": {"partition_family": "kmeans_simple_nprobe"},
    "legacy_centroid": {"partition_family": "legacy_centroid"},
}


SAFE_EXPERIMENT_COLLECTION_PREFIXES = ("dist_", "bench_")
SAFE_MATRIX_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{2,127}$")
SAFE_COLLECTION_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,254}$")

SUMMARY_FIELDS = [
    "run_id",
    "case_name",
    "preset",
    "partition_family",
    "routing_mode",
    "target_recall",
    "requested_target_recall",
    "recall_match_status",
    "confirmed_recall_match_status",
    "recall_delta_to_target",
    "num_shards",
    "effective_num_shards",
    "upper_k",
    "nprobe",
    "base_ef",
    "factor",
    "batch_size",
    "recall",
    "recall_stdev",
    "qps",
    "qps_stdev",
    "final_qps",
    "avg_visited_shards",
    "avg_ef_per_visited_shard",
    "estimated_ef_sum_per_query",
    "avg_physical_peers_per_query",
    "latency_p50_ms",
    "latency_p50_ms_stdev",
    "latency_p95_ms",
    "latency_p95_ms_stdev",
    "latency_p99_ms",
    "latency_p99_ms_stdev",
    "cluster_peer_count",
    "image_digest",
    "placement_valid",
    "runtime_health_valid",
    "search_batch_calls",
    "index_expansion_ratio",
    "worker_shard_counts",
    "worker_point_counts",
    "summary_path",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Expand and run Orion/naive/KMeans benchmark matrices on top of "
            "tools/qdrant_two_level_routing_experiment.py."
        )
    )
    parser.add_argument("--config", required=True, help="JSON benchmark matrix config.")
    parser.add_argument(
        "--output-root",
        default=None,
        help=(
            "Root for generated run directories. Defaults to config output_root or "
            "results/method4_benchmark_matrix."
        ),
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional matrix run id. Defaults to current UTC-like timestamp.",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Actually execute commands. Without this flag the tool only writes commands and manifest.",
    )
    parser.add_argument(
        "--collect-only",
        action="store_true",
        help="Do not execute; collect summary.json files from the expanded output dirs.",
    )
    parser.add_argument(
        "--python",
        default=None,
        help="Python executable. Defaults to config python or python3.",
    )
    parser.add_argument(
        "--harness",
        default=None,
        help=(
            "Experiment harness path. Defaults to config harness or "
            "tools/qdrant_two_level_routing_experiment.py."
        ),
    )
    return parser.parse_args()


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"benchmark config must be a JSON object: {path}")
    return data


def normalize_key(key: str) -> str:
    return key.strip().lstrip("-").replace("-", "_")


def normalize_args(raw: dict[str, Any] | None) -> dict[str, Any]:
    if not raw:
        return {}
    return {normalize_key(str(key)): value for key, value in raw.items()}


def merge_args(*parts: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for part in parts:
        for key, value in part.items():
            merged[normalize_key(key)] = value
    return merged


def values_for_matrix(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    return [value]


def expand_matrix(matrix: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not matrix:
        return [{}]
    keys = [normalize_key(key) for key in matrix]
    value_lists = [values_for_matrix(value) for value in matrix.values()]
    rows: list[dict[str, Any]] = []
    for values in itertools.product(*value_lists):
        rows.append(dict(zip(keys, values)))
    return rows


def id_value(value: Any) -> str:
    if isinstance(value, float):
        text = f"{value:g}".replace(".", "p")
    else:
        text = str(value)
    allowed = []
    for char in text:
        if char.isalnum():
            allowed.append(char)
        else:
            allowed.append("_")
    return "".join(allowed).strip("_") or "x"


def build_run_id(case_name: str, matrix_row: dict[str, Any]) -> str:
    if not matrix_row:
        return id_value(case_name)
    suffix = "__".join(f"{key}-{id_value(value)}" for key, value in matrix_row.items())
    return f"{id_value(case_name)}__{suffix}"


def format_collection(
    template: str,
    case_name: str,
    preset: str,
    run_id: str,
    args: dict[str, Any],
    tags: dict[str, Any],
    matrix_run_id: str | None = None,
) -> str:
    context = {
        "case": case_name,
        "preset": preset,
        "run_id": run_id,
        "matrix_run_id": matrix_run_id or run_id,
        **tags,
        **args,
    }
    return template.format(**context)


def format_matrix_run_values(args: dict[str, Any], matrix_run_id: str | None) -> dict[str, Any]:
    if not matrix_run_id:
        return args
    context = {"matrix_run_id": matrix_run_id}
    return {
        key: value.format(**context) if isinstance(value, str) and "{matrix_run_id}" in value else value
        for key, value in args.items()
    }


def expand_runs(
    spec: dict[str, Any],
    output_root: Path,
    matrix_run_id: str | None = None,
) -> list[BenchmarkRun]:
    defaults = normalize_args(spec.get("defaults"))
    cases = spec.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("benchmark config requires a non-empty cases list")

    runs: list[BenchmarkRun] = []
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("each case must be a JSON object")
        case_name = str(case.get("name") or case.get("preset") or f"case_{len(runs)}")
        preset = str(case.get("preset") or "orion")
        if preset not in PRESET_ARGS:
            raise ValueError(f"unknown preset {preset!r}; expected one of {sorted(PRESET_ARGS)}")

        preset_args = dict(PRESET_ARGS[preset])
        case_args = normalize_args(case.get("args"))
        case_tags = {
            **PRESET_TAGS.get(preset, {}),
            **normalize_args(case.get("tags")),
        }
        collection_template = case.get("collection_template")
        matrix_rows = expand_matrix(case.get("matrix"))

        for matrix_row in matrix_rows:
            run_id = build_run_id(case_name, matrix_row)
            args = merge_args(defaults, preset_args, case_args, matrix_row)
            args = format_matrix_run_values(args, matrix_run_id)
            if collection_template and "collection" not in args:
                args["collection"] = format_collection(
                    str(collection_template),
                    case_name,
                    preset,
                    run_id,
                    args,
                    case_tags,
                    matrix_run_id,
                )
            output_dir = output_root / run_id
            runs.append(
                BenchmarkRun(
                    run_id=run_id,
                    case_name=case_name,
                    preset=preset,
                    args=args,
                    output_dir=output_dir,
                    tags=case_tags,
                )
            )
    return runs


def validated_collection_cleanup_targets(
    spec: dict[str, Any],
    runs: list[BenchmarkRun],
    matrix_run_id: str,
) -> list[CollectionCleanupTarget]:
    """Build an all-or-nothing cleanup plan for this exact matrix invocation.

    Cleanup is deliberately stricter than collection creation.  Every target
    must come from a case-level ``collection_template`` that explicitly embeds
    ``{matrix_run_id}``; fixed/default/matrix-provided collection names are not
    accepted.  The complete plan is validated before any DELETE can be sent.
    """

    if not SAFE_MATRIX_RUN_ID_RE.fullmatch(matrix_run_id):
        raise ValueError(
            "delete_collections_after_run requires a validated matrix run id "
            "(3-128 ASCII letters, digits, dot, underscore, or hyphen)"
        )

    defaults = normalize_args(spec.get("defaults"))
    cases = spec.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("benchmark config requires a non-empty cases list")

    sources: list[tuple[str, str, str, str]] = []
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("each case must be a JSON object")
        case_name = str(case.get("name") or case.get("preset") or f"case_{len(sources)}")
        preset = str(case.get("preset") or "orion")
        if preset not in PRESET_ARGS:
            raise ValueError(f"unknown preset {preset!r}; expected one of {sorted(PRESET_ARGS)}")
        collection_template = case.get("collection_template")
        if not isinstance(collection_template, str) or (
            "{matrix_run_id}" not in collection_template
        ):
            raise ValueError(
                "delete_collections_after_run refuses case "
                f"{case_name!r}: collection_template must explicitly contain "
                "{matrix_run_id}"
            )
        case_args = normalize_args(case.get("args"))
        case_tags = {
            **PRESET_TAGS.get(preset, {}),
            **normalize_args(case.get("tags")),
        }
        for matrix_row in expand_matrix(case.get("matrix")):
            run_id = build_run_id(case_name, matrix_row)
            raw_run_args = merge_args(
                defaults,
                PRESET_ARGS[preset],
                case_args,
                matrix_row,
            )
            if "collection" in raw_run_args:
                raise ValueError(
                    "delete_collections_after_run refuses case "
                    f"{case_name!r}: collection is explicitly supplied instead "
                    "of being generated by collection_template"
                )
            formatted_args = format_matrix_run_values(raw_run_args, matrix_run_id)
            expected_collection = format_collection(
                collection_template,
                case_name,
                preset,
                run_id,
                formatted_args,
                case_tags,
                matrix_run_id,
            )
            sources.append((run_id, case_name, preset, expected_collection))

    if len(sources) != len(runs):
        raise ValueError(
            "delete_collections_after_run cleanup provenance does not match "
            "the expanded matrix runs"
        )

    targets: list[CollectionCleanupTarget] = []
    seen: set[CollectionCleanupTarget] = set()
    for run, (run_id, case_name, preset, expected_collection) in zip(runs, sources):
        if (run.run_id, run.case_name, run.preset) != (run_id, case_name, preset):
            raise ValueError(
                "delete_collections_after_run cleanup provenance does not match "
                f"expanded run {run.run_id!r}"
            )
        collection = str(run.args.get("collection") or "")
        if collection != expected_collection:
            raise ValueError(
                "delete_collections_after_run refuses collection "
                f"{collection!r}: it was not generated by this matrix run's "
                "collection_template"
            )
        if not SAFE_COLLECTION_NAME_RE.fullmatch(collection):
            raise ValueError(
                "delete_collections_after_run refuses unsafe collection name "
                f"{collection!r}"
            )
        safe_prefix = next(
            (
                prefix
                for prefix in SAFE_EXPERIMENT_COLLECTION_PREFIXES
                if collection.startswith(prefix)
            ),
            None,
        )
        if safe_prefix is None:
            raise ValueError(
                "delete_collections_after_run refuses collection "
                f"{collection!r}: expected an experimental prefix from "
                f"{SAFE_EXPERIMENT_COLLECTION_PREFIXES}"
            )
        run_scoped_prefix = f"{safe_prefix}{matrix_run_id}"
        if collection != run_scoped_prefix and not collection.startswith(
            f"{run_scoped_prefix}_"
        ):
            raise ValueError(
                "delete_collections_after_run refuses collection "
                f"{collection!r}: validated matrix run token must immediately "
                "follow the experimental prefix"
            )
        base_url = str(run.args.get("base_url") or "").rstrip("/")
        parsed_base_url = urllib.parse.urlsplit(base_url)
        if parsed_base_url.scheme not in {"http", "https"} or not parsed_base_url.netloc:
            raise ValueError(
                "delete_collections_after_run requires an absolute HTTP(S) "
                f"base_url for run {run.run_id!r}"
            )
        target = CollectionCleanupTarget(base_url, collection)
        if target not in seen:
            seen.add(target)
            targets.append(target)
    return targets


def cli_key(key: str) -> str:
    return "--" + key.replace("_", "-")


def str_value(value: Any) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    return str(value)


def render_command(
    run: BenchmarkRun,
    python_executable: str,
    harness: str,
    command_prefix: list[str] | None = None,
) -> list[str]:
    command = [*(command_prefix or []), python_executable, harness]
    for key, value in run.args.items():
        if value is None or value is False:
            continue
        if key == "output_dir":
            continue
        flag = cli_key(key)
        if value is True:
            command.append(flag)
        elif isinstance(value, list):
            command.append(flag)
            command.extend(str_value(item) for item in value)
        else:
            command.extend([flag, str_value(value)])
    command.extend(["--output-dir", str(run.output_dir)])
    return command


def latest_summary_path(output_dir: Path) -> Path | None:
    direct = output_dir / "summary.json"
    if direct.exists():
        return direct
    candidates = sorted(
        output_dir.glob("*/summary.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def first_number(*values: Any) -> Any:
    for value in values:
        if isinstance(value, (int, float)):
            return value
    return ""


def stability_metric_stats(summary_path: Path, field: str) -> tuple[Any, Any]:
    """Return the repeated-run mean and sample stdev for one metric.

    ``summary.json`` intentionally keeps only recall/QPS stability aggregates,
    while the latency percentiles for every repeat live in the sibling
    ``stability_runs.csv``.  Ignore blank/non-finite values so an older or
    partially populated CSV can still fall back to ``final_row`` per metric.
    """

    stability_path = summary_path.with_name("stability_runs.csv")
    if not stability_path.exists():
        return "", ""
    values: list[float] = []
    with stability_path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            raw_value = row.get(field)
            if raw_value in (None, ""):
                continue
            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                values.append(value)
    if not values:
        return "", ""
    return (
        statistics.fmean(values),
        statistics.stdev(values) if len(values) > 1 else 0.0,
    )


def confirmed_recall_match_status(
    recall: Any,
    requested_target_recall: Any,
    window: float = 0.003,
) -> tuple[str, Any]:
    if not isinstance(recall, (int, float)) or not isinstance(
        requested_target_recall, (int, float)
    ):
        return "", ""
    delta = float(recall) - float(requested_target_recall)
    status = "strict" if abs(delta) <= window + 1e-12 else "nearest"
    return status, delta


def collect_summary_row(run: BenchmarkRun, summary_path: Path) -> dict[str, Any]:
    data = load_json(summary_path)
    final_row = data.get("final_row") if isinstance(data.get("final_row"), dict) else {}
    best_row = data.get("best_tuning_row") if isinstance(data.get("best_tuning_row"), dict) else {}
    stability = data.get("stability_summary") if isinstance(data.get("stability_summary"), dict) else {}
    original_routing = data.get("original_routing") if isinstance(data.get("original_routing"), dict) else {}
    simple_kmeans = data.get("kmeans_simple_nprobe") if isinstance(data.get("kmeans_simple_nprobe"), dict) else {}
    cluster = data.get("collection_cluster") if isinstance(data.get("collection_cluster"), dict) else {}
    worker_shard_points = (
        data.get("worker_shard_points")
        if isinstance(data.get("worker_shard_points"), dict)
        else {}
    )
    requested_target_recall = run.tags.get(
        "requested_target_recall", run.args.get("target_recall", "")
    )
    confirmed_recall = first_number(
        stability.get("recall_mean"), final_row.get("recall_at_k"), final_row.get("recall")
    )
    confirmation_target = (
        requested_target_recall
        if isinstance(requested_target_recall, (int, float))
        and float(requested_target_recall) > 0.0
        and int(run.args.get("stability_repeats") or 0) > 0
        else None
    )
    confirmed_status, recall_delta = confirmed_recall_match_status(
        confirmed_recall,
        confirmation_target,
    )
    latency_p50_mean, latency_p50_stdev = stability_metric_stats(
        summary_path, "latency_p50_ms"
    )
    latency_p95_mean, latency_p95_stdev = stability_metric_stats(
        summary_path, "latency_p95_ms"
    )
    latency_p99_mean, latency_p99_stdev = stability_metric_stats(
        summary_path, "latency_p99_ms"
    )

    row = {
        "run_id": run.run_id,
        "case_name": run.case_name,
        "preset": run.preset,
        "partition_family": run.tags.get("partition_family", ""),
        "routing_mode": data.get("routing_mode", run.args.get("routing_mode", "")),
        "target_recall": run.args.get("target_recall", ""),
        "requested_target_recall": requested_target_recall,
        "recall_match_status": run.tags.get("recall_match_status", ""),
        "confirmed_recall_match_status": confirmed_status,
        "recall_delta_to_target": recall_delta,
        "num_shards": data.get("initial_num_shards", run.args.get("num_shards", "")),
        "effective_num_shards": data.get("num_shards", ""),
        "upper_k": first_number(final_row.get("upper_k"), best_row.get("upper_k")),
        "nprobe": first_number(
            final_row.get("nprobe"),
            best_row.get("nprobe"),
            final_row.get("upper_k") if data.get("routing_mode") == "kmeans_simple_nprobe" else None,
        ),
        "base_ef": first_number(final_row.get("base_ef"), best_row.get("base_ef")),
        "factor": first_number(final_row.get("factor"), best_row.get("factor")),
        "batch_size": run.args.get("batch_size", ""),
        "recall": confirmed_recall,
        "recall_stdev": first_number(stability.get("recall_stdev")),
        "qps": first_number(stability.get("qps_mean"), final_row.get("qps")),
        "qps_stdev": first_number(stability.get("qps_stdev")),
        "final_qps": first_number(final_row.get("qps")),
        "avg_visited_shards": first_number(final_row.get("avg_visited_shards")),
        "avg_ef_per_visited_shard": first_number(
            final_row.get("avg_assigned_ef_per_visited_shard"),
            final_row.get("avg_ef_per_visited_shard"),
        ),
        "estimated_ef_sum_per_query": first_number(
            final_row.get("estimated_ef_sum_per_query")
        ),
        "avg_physical_peers_per_query": first_number(
            final_row.get("avg_physical_peers_per_query"),
            (data.get("physical_execution_trace") or {}).get("avg_physical_peers_per_query")
            if isinstance(data.get("physical_execution_trace"), dict)
            else None,
        ),
        # Keep the established latency field names for CSV compatibility, but
        # report the repeated-run mean whenever stability data is available.
        "latency_p50_ms": first_number(
            latency_p50_mean, final_row.get("latency_p50_ms")
        ),
        "latency_p50_ms_stdev": latency_p50_stdev,
        "latency_p95_ms": first_number(
            latency_p95_mean, final_row.get("latency_p95_ms")
        ),
        "latency_p95_ms_stdev": latency_p95_stdev,
        "latency_p99_ms": first_number(
            latency_p99_mean, final_row.get("latency_p99_ms")
        ),
        "latency_p99_ms_stdev": latency_p99_stdev,
        "cluster_peer_count": first_number(
            (data.get("cluster_preflight") or {}).get("peer_count")
            if isinstance(data.get("cluster_preflight"), dict)
            else None,
            cluster.get("cluster_peer_count"),
        ),
        "image_digest": data.get("image_digest", ""),
        "placement_valid": first_number(data.get("placement_valid"), cluster.get("cluster_placement_valid")),
        "runtime_health_valid": first_number(
            (data.get("runtime_health_audit") or {}).get("valid")
            if isinstance(data.get("runtime_health_audit"), dict)
            else None
        ),
        "search_batch_calls": first_number(final_row.get("search_batch_calls")),
        "index_expansion_ratio": first_number(
            original_routing.get("expansion_ratio"),
            simple_kmeans.get("expansion_ratio"),
            1.0 if data.get("routing_mode") == "naive_hash_all_shards" else None,
        ),
        "worker_shard_counts": json.dumps(
            {
                str(peer_id): int((value or {}).get("shard_count") or 0)
                for peer_id, value in sorted(worker_shard_points.items())
            },
            sort_keys=True,
        ),
        "worker_point_counts": json.dumps(
            {
                str(peer_id): int((value or {}).get("points_count") or 0)
                for peer_id, value in sorted(worker_shard_points.items())
            },
            sort_keys=True,
        ),
        "summary_path": str(summary_path),
    }
    return row


def collect_rows(runs: list[BenchmarkRun]) -> list[dict[str, Any]]:
    rows = []
    for run in runs:
        summary_path = latest_summary_path(run.output_dir)
        if summary_path is None:
            rows.append(
                {
                    "run_id": run.run_id,
                    "case_name": run.case_name,
                    "preset": run.preset,
                    "partition_family": run.tags.get("partition_family", ""),
                    "summary_path": "",
                }
            )
            continue
        rows.append(collect_summary_row(run, summary_path))
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] = SUMMARY_FIELDS) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_manifest(
    matrix_dir: Path,
    spec: dict[str, Any],
    runs: list[BenchmarkRun],
    commands: list[list[str]],
) -> None:
    matrix_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "matrix_name": spec.get("matrix_name") or spec.get("name") or "method4_matrix",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runs": [
            {
                "run_id": run.run_id,
                "case_name": run.case_name,
                "preset": run.preset,
                "args": run.args,
                "output_dir": str(run.output_dir),
                "tags": run.tags,
                "command": command,
            }
            for run, command in zip(runs, commands)
        ],
    }
    (matrix_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    with (matrix_dir / "commands.sh").open("w", encoding="utf-8") as handle:
        handle.write("#!/usr/bin/env bash\nset -euo pipefail\n\n")
        for command in commands:
            handle.write(shlex.join(command))
            handle.write("\n")


def parse_csv_value(value: str) -> Any:
    text = value.strip()
    if text == "":
        return ""
    try:
        number = float(text)
        return int(number) if number.is_integer() and not any(char in text.lower() for char in ".e") else number
    except ValueError:
        return value


def collect_recall_qps_points(runs: list[BenchmarkRun]) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for run in runs:
        summary_path = latest_summary_path(run.output_dir)
        if summary_path is None:
            continue
        summary = load_json(summary_path)
        tuning_path = summary_path.with_name("routing_tuning.csv")
        if not tuning_path.exists():
            continue
        with tuning_path.open("r", newline="", encoding="utf-8") as handle:
            for raw in csv.DictReader(handle):
                row = {key: parse_csv_value(value or "") for key, value in raw.items()}
                row.update(
                    {
                        "run_id": run.run_id,
                        "method": run.preset,
                        "case_name": run.case_name,
                        "partition_family": run.tags.get("partition_family", ""),
                        "routing_mode": summary.get("routing_mode", run.args.get("routing_mode", "")),
                        "collection": summary.get("collection", run.args.get("collection", "")),
                        "image_digest": summary.get("image_digest", ""),
                        "cluster_peer_count": (summary.get("cluster_preflight") or {}).get(
                            "peer_count", (summary.get("collection_cluster") or {}).get("cluster_peer_count", "")
                        ),
                        "placement_valid": summary.get(
                            "placement_valid",
                            (summary.get("collection_cluster") or {}).get("cluster_placement_valid", ""),
                        ),
                        "index_expansion_ratio": first_number(
                            (summary.get("original_routing") or {}).get("expansion_ratio"),
                            (summary.get("kmeans_simple_nprobe") or {}).get("expansion_ratio"),
                            1.0 if summary.get("routing_mode") == "naive_hash_all_shards" else None,
                        ),
                        "summary_path": str(summary_path),
                    }
                )
                if summary.get("routing_mode") == "kmeans_simple_nprobe":
                    row["nprobe"] = first_number(row.get("nprobe"), row.get("upper_k"))
                points.append(row)
    return points


def pareto_frontier(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frontier: list[dict[str, Any]] = []
    for point in points:
        recall = float(point.get("recall_at_k") or point.get("recall") or 0.0)
        qps = float(point.get("qps") or 0.0)
        dominated = False
        for other in points:
            if other is point:
                continue
            other_recall = float(other.get("recall_at_k") or other.get("recall") or 0.0)
            other_qps = float(other.get("qps") or 0.0)
            if other_recall >= recall and other_qps >= qps and (
                other_recall > recall or other_qps > qps
            ):
                dominated = True
                break
        if not dominated:
            frontier.append(dict(point))
    return sorted(frontier, key=lambda row: (float(row.get("recall_at_k") or 0), -float(row.get("qps") or 0)))


def select_same_recall_point(
    points: list[dict[str, Any]],
    target_recall: float,
    window: float = 0.003,
) -> dict[str, Any]:
    if not points:
        raise ValueError("cannot select a same-recall point from an empty method frontier")
    strict = [
        point
        for point in points
        if abs(
            float(point.get("recall_at_k") or point.get("recall") or 0.0)
            - target_recall
        )
        <= window + 1e-12
    ]
    if strict:
        selected = max(strict, key=lambda row: float(row.get("qps") or 0.0))
        status = "strict"
    else:
        selected = min(
            points,
            key=lambda row: (
                abs(float(row.get("recall_at_k") or row.get("recall") or 0.0) - target_recall),
                -float(row.get("qps") or 0.0),
            ),
        )
        status = "nearest"
    return {**selected, "target_recall": target_recall, "recall_match_status": status}


def write_curve_plots(matrix_dir: Path, points: list[dict[str, Any]]) -> None:
    if not points:
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    plot_specs = [
        ("qps", "QPS", "recall_qps.png"),
        ("latency_p95_ms", "P95 latency (ms)", "recall_p95.png"),
        ("latency_p99_ms", "P99 latency (ms)", "recall_p99.png"),
        ("avg_visited_shards", "Visited logical shards/query", "recall_visited_shards.png"),
        ("estimated_ef_sum_per_query", "Estimated EF sum/query", "recall_estimated_ef_sum.png"),
    ]
    methods = sorted({str(row.get("method") or "unknown") for row in points})
    for y_field, y_label, filename in plot_specs:
        fig, axis = plt.subplots(figsize=(7.2, 4.8))
        plotted = False
        for method in methods:
            rows = [row for row in points if str(row.get("method")) == method]
            xy = [
                (
                    float(row.get("recall_at_k") or row.get("recall") or 0.0),
                    float(row[y_field]),
                )
                for row in rows
                if isinstance(row.get(y_field), (int, float)) and math.isfinite(float(row[y_field]))
            ]
            if not xy:
                continue
            xy.sort()
            axis.plot([x for x, _ in xy], [y for _, y in xy], marker="o", label=method)
            plotted = True
        if plotted:
            axis.set_xlabel("Recall@10")
            axis.set_ylabel(y_label)
            axis.grid(True, alpha=0.3)
            axis.legend()
            fig.tight_layout()
            fig.savefig(matrix_dir / filename, dpi=160)
        plt.close(fig)


def write_analysis_artifacts(
    matrix_dir: Path,
    spec: dict[str, Any],
    runs: list[BenchmarkRun],
) -> None:
    points = collect_recall_qps_points(runs)
    if not points:
        return
    point_fields = sorted({key for row in points for key in row})
    write_csv(matrix_dir / "recall_qps_points.csv", points, point_fields)
    frontiers: list[dict[str, Any]] = []
    selections: list[dict[str, Any]] = []
    for method in sorted({str(row["method"]) for row in points}):
        method_frontier = pareto_frontier([row for row in points if str(row["method"]) == method])
        frontiers.extend(method_frontier)
        for target in spec.get("confirmation_targets", [0.90, 0.95]):
            selections.append(select_same_recall_point(method_frontier, float(target)))
    write_csv(
        matrix_dir / "pareto_frontier.csv",
        frontiers,
        sorted({key for row in frontiers for key in row}),
    )
    write_csv(
        matrix_dir / "same_recall_selection.csv",
        selections,
        sorted({key for row in selections for key in row}),
    )
    write_curve_plots(matrix_dir, points)
    write_confirmation_config(matrix_dir, spec, runs, selections)


def write_confirmation_config(
    matrix_dir: Path,
    spec: dict[str, Any],
    runs: list[BenchmarkRun],
    selections: list[dict[str, Any]],
) -> None:
    runs_by_id = {run.run_id: run for run in runs}
    cases: list[dict[str, Any]] = []
    for selected in selections:
        source = runs_by_id.get(str(selected.get("run_id") or ""))
        if source is None:
            continue
        args = dict(source.args)
        args.update(
            {
                "reuse_existing": True,
                "tuning_query_count": 500,
                "eval_query_count": 3000,
                "warmup_query_count": 500,
                "stability_repeats": 3,
                # Preserve the source experiment's batching so the generated
                # same-recall confirmation remains directly comparable to the
                # screening run.  Falling back to 100 keeps compatibility with
                # older configs that did not declare a batch size.
                "batch_size": int(source.args.get("batch_size") or 100),
                "target_recall": (
                    float(selected.get("recall_at_k") or selected.get("recall") or 0.0)
                    if selected["recall_match_status"] == "nearest"
                    else float(selected["target_recall"])
                ),
                "write_per_query_metrics": True,
            }
        )
        upper_k = int(float(selected.get("upper_k") or 0))
        base_ef = int(float(selected.get("base_ef") or 0))
        factor = int(float(selected.get("factor") or 0))
        args["upper_k_candidates"] = [upper_k]
        args["base_ef_candidates"] = [base_ef]
        args["factor_candidates"] = [factor]
        if source.preset == "kmeans_simple_nprobe":
            args["nprobe_candidates"] = [int(float(selected.get("nprobe") or upper_k))]
        if source.preset == "orion":
            args["recover_routing_from_collection"] = True
        cases.append(
            {
                "name": f"{source.case_name}_r{str(selected['target_recall']).replace('.', 'p')}",
                "preset": source.preset,
                "tags": {
                    **source.tags,
                    "recall_match_status": selected["recall_match_status"],
                    "requested_target_recall": selected["target_recall"],
                    "selected_tuning_recall": selected.get("recall_at_k"),
                },
                "args": args,
            }
        )
    confirmation = {
        "matrix_name": f"{spec.get('matrix_name', 'method4')}_confirmation",
        "harness": spec.get("harness", "tools/qdrant_two_level_routing_experiment.py"),
        "python": spec.get("python", "python3"),
        "output_root": str(matrix_dir / "confirmation"),
        "client_cpuset": spec.get("client_cpuset"),
        "cases": cases,
    }
    (matrix_dir / "confirmation_matrix.json").write_text(
        json.dumps(confirmation, indent=2), encoding="utf-8"
    )


def execute_runs(runs: list[BenchmarkRun], commands: list[list[str]]) -> None:
    for run, command in zip(runs, commands):
        print(f"[run] {run.run_id}: {shlex.join(command)}", flush=True)
        subprocess.run(command, check=True)


def delete_run_collections(targets: list[CollectionCleanupTarget]) -> None:
    for base_url, collection in targets:
        url = f"{base_url}/collections/{urllib.parse.quote(collection, safe='')}"
        request = urllib.request.Request(url, method="DELETE")
        try:
            with urllib.request.urlopen(request, timeout=120.0):
                pass
        except urllib.error.HTTPError as exc:
            if exc.code != 404:
                raise


def main() -> int:
    args = parse_args()
    spec = load_json(args.config)
    matrix_name = str(spec.get("matrix_name") or spec.get("name") or Path(args.config).stem)
    output_root = Path(args.output_root or spec.get("output_root") or "results/method4_benchmark_matrix")
    run_id = args.run_id or time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    matrix_dir = output_root / matrix_name / run_id
    python_executable = args.python or spec.get("python") or "python3"
    harness = args.harness or spec.get("harness") or "tools/qdrant_two_level_routing_experiment.py"

    runs = expand_runs(spec, matrix_dir, matrix_run_id=run_id)
    command_prefix = (
        ["taskset", "-c", str(spec["client_cpuset"])]
        if spec.get("client_cpuset")
        else []
    )
    commands = [
        render_command(run, python_executable, harness, command_prefix=command_prefix)
        for run in runs
    ]
    write_manifest(matrix_dir, spec, runs, commands)

    cleanup_targets: list[CollectionCleanupTarget] = []
    if args.run and not args.collect_only and spec.get("delete_collections_after_run"):
        # Preflight the complete deletion plan before running the benchmark so
        # a fixed/reused production collection can never be partially deleted.
        cleanup_targets = validated_collection_cleanup_targets(spec, runs, run_id)

    if args.run and not args.collect_only:
        execute_runs(runs, commands)

    rows = collect_rows(runs)
    write_csv(matrix_dir / "matrix_summary.csv", rows)
    confirmation_rows = [
        row for run, row in zip(runs, rows) if "requested_target_recall" in run.tags
    ]
    if confirmation_rows:
        write_csv(matrix_dir / "same_recall_confirmation.csv", confirmation_rows)
    write_analysis_artifacts(matrix_dir, spec, runs)
    if cleanup_targets:
        delete_run_collections(cleanup_targets)
    print(f"Wrote matrix files to: {matrix_dir}")
    print(f"Runs: {len(runs)}")
    if not args.run and not args.collect_only:
        print("Dry run only. Re-run with --run to execute commands.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
