#!/usr/bin/env python3
"""Audit one completed HashAll/Orion/Simple-KMeans policy-parity matrix.

The audit is intentionally independent of the benchmark collector.  It reads the
frozen matrix and per-case manifests, verifies the execution/provenance contract,
and recomputes repeat-level QPS comparisons from ``stability_runs.csv``.

The Welch intervals are descriptive: matrix cases are executed in method blocks,
not as paired or interleaved observations, and the formal configuration currently
contains only three repeats per case.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any


METHODS = ("hash_all", "orion", "simple_kmeans")
EXECUTION_IDENTITY = {
    "hash_all": {
        "plan_kind": "all_shards",
        "planner": "explicit_all_shards",
        "executor": "qdrant_all_shards_replica_set",
    },
    "orion": {
        "plan_kind": "selected_by_shard",
        "planner": "orion_upper_hnsw",
        "executor": "common_selected_shard_executor",
    },
    "simple_kmeans": {
        "plan_kind": "selected_by_shard",
        "planner": "simple_kmeans_nprobe",
        "executor": "common_selected_shard_executor",
    },
}
COMMON_EXECUTION_IDENTITY = {
    "cluster_architecture": "symmetric_qdrant_peers_request_scoped_coordinator",
    "logical_shard_plan": "collection_logical_shard_plan_v1",
    "replica_read_path": "ShardReplicaSet::core_search",
    "global_merge_path": "Collection::merge_from_shards",
    "transport_mode": "ordinary_per_shard_replica_set",
    "peer_premerge_mode": "disabled",
    "policy_isolation_transport": True,
}

CASE_FIELDS = [
    "target_recall",
    "method",
    "case_name",
    "recall_at_k",
    "qps_mean",
    "qps_stdev",
    "latency_p50_batch_ms",
    "latency_p95_batch_ms",
    "latency_p99_batch_ms",
    "visited_shards",
    "ef_sum_per_query",
    "planner",
    "executor",
    "logical_points",
    "physical_points",
    "index_expansion_ratio",
    "stability_repeats",
]
REPEAT_FIELDS = [
    "target_recall",
    "method",
    "case_name",
    "repeat",
    "recall_at_k",
    "qps",
    "latency_p50_batch_ms",
    "latency_p95_batch_ms",
    "latency_p99_batch_ms",
]
COMPARISON_FIELDS = [
    "target_recall",
    "method_a",
    "method_b",
    "n_a",
    "n_b",
    "qps_mean_a",
    "qps_mean_b",
    "difference_qps_a_minus_b",
    "difference_pct_vs_b",
    "welch_standard_error",
    "welch_degrees_of_freedom",
    "welch_t_statistic",
    "welch_two_sided_p_value",
    "welch_95_ci_low_qps",
    "welch_95_ci_high_qps",
    "design_note",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("matrix_dir", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--expected-repeats", type=int, default=3)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Write audit_report.json and compact CSV evidence to this directory.",
    )
    return parser.parse_args(argv)


def load_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite_float(value: Any, *, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric, got {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite, got {value!r}")
    return result


def _beta_continued_fraction(a: float, b: float, x: float) -> float:
    maximum_iterations = 300
    epsilon = 3.0e-14
    minimum = 1.0e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < minimum:
        d = minimum
    d = 1.0 / d
    result = d
    for iteration in range(1, maximum_iterations + 1):
        doubled = 2 * iteration
        coefficient = (
            iteration
            * (b - iteration)
            * x
            / ((qam + doubled) * (a + doubled))
        )
        d = 1.0 + coefficient * d
        if abs(d) < minimum:
            d = minimum
        c = 1.0 + coefficient / c
        if abs(c) < minimum:
            c = minimum
        d = 1.0 / d
        result *= d * c

        coefficient = -(
            (a + iteration)
            * (qab + iteration)
            * x
            / ((a + doubled) * (qap + doubled))
        )
        d = 1.0 + coefficient * d
        if abs(d) < minimum:
            d = minimum
        c = 1.0 + coefficient / c
        if abs(c) < minimum:
            c = minimum
        d = 1.0 / d
        delta = d * c
        result *= delta
        if abs(delta - 1.0) <= epsilon:
            return result
    raise ArithmeticError("incomplete-beta continued fraction did not converge")


def regularized_incomplete_beta(x: float, a: float, b: float) -> float:
    if not 0.0 <= x <= 1.0 or a <= 0.0 or b <= 0.0:
        raise ValueError("invalid regularized incomplete beta arguments")
    if x == 0.0:
        return 0.0
    if x == 1.0:
        return 1.0
    front = math.exp(
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _beta_continued_fraction(a, b, x) / a
    return 1.0 - front * _beta_continued_fraction(b, a, 1.0 - x) / b


def student_t_cdf(value: float, degrees_of_freedom: float) -> float:
    if degrees_of_freedom <= 0.0 or not math.isfinite(degrees_of_freedom):
        raise ValueError("degrees_of_freedom must be positive and finite")
    if value == 0.0:
        return 0.5
    x = degrees_of_freedom / (degrees_of_freedom + value * value)
    tail = 0.5 * regularized_incomplete_beta(
        x, degrees_of_freedom / 2.0, 0.5
    )
    return 1.0 - tail if value > 0.0 else tail


def student_t_ppf(probability: float, degrees_of_freedom: float) -> float:
    if not 0.0 < probability < 1.0:
        raise ValueError("probability must be strictly between zero and one")
    if probability == 0.5:
        return 0.0
    if probability < 0.5:
        return -student_t_ppf(1.0 - probability, degrees_of_freedom)
    lower = 0.0
    upper = 1.0
    while student_t_cdf(upper, degrees_of_freedom) < probability:
        upper *= 2.0
    for _ in range(100):
        midpoint = (lower + upper) / 2.0
        if student_t_cdf(midpoint, degrees_of_freedom) < probability:
            lower = midpoint
        else:
            upper = midpoint
    return (lower + upper) / 2.0


def welch_comparison(
    values_a: list[float],
    values_b: list[float],
    *,
    target_recall: float,
    method_a: str,
    method_b: str,
) -> dict[str, Any]:
    if len(values_a) < 2 or len(values_b) < 2:
        raise ValueError("Welch comparison requires at least two observations per method")
    mean_a = statistics.fmean(values_a)
    mean_b = statistics.fmean(values_b)
    variance_a = statistics.variance(values_a)
    variance_b = statistics.variance(values_b)
    component_a = variance_a / len(values_a)
    component_b = variance_b / len(values_b)
    standard_error = math.sqrt(component_a + component_b)
    if standard_error == 0.0:
        raise ValueError("Welch comparison has zero standard error")
    degrees_of_freedom = (component_a + component_b) ** 2 / (
        component_a**2 / (len(values_a) - 1)
        + component_b**2 / (len(values_b) - 1)
    )
    difference = mean_a - mean_b
    statistic = difference / standard_error
    p_value = 2.0 * (1.0 - student_t_cdf(abs(statistic), degrees_of_freedom))
    critical = student_t_ppf(0.975, degrees_of_freedom)
    margin = critical * standard_error
    return {
        "target_recall": target_recall,
        "method_a": method_a,
        "method_b": method_b,
        "n_a": len(values_a),
        "n_b": len(values_b),
        "qps_mean_a": mean_a,
        "qps_mean_b": mean_b,
        "difference_qps_a_minus_b": difference,
        "difference_pct_vs_b": difference / mean_b * 100.0,
        "welch_standard_error": standard_error,
        "welch_degrees_of_freedom": degrees_of_freedom,
        "welch_t_statistic": statistic,
        "welch_two_sided_p_value": p_value,
        "welch_95_ci_low_qps": difference - margin,
        "welch_95_ci_high_qps": difference + margin,
        "design_note": (
            "descriptive unpaired Welch interval; n=3 repeat runs per case; "
            "cases were block-ordered, not interleaved"
        ),
    }


def require(errors: list[str], condition: bool, message: str) -> None:
    if not condition:
        errors.append(message)


def require_equal(
    errors: list[str], actual: Any, expected: Any, label: str
) -> None:
    require(errors, actual == expected, f"{label}: expected {expected!r}, got {actual!r}")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def audit(args: argparse.Namespace) -> dict[str, Any]:
    matrix_dir = args.matrix_dir.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    manifest_path = matrix_dir / "run_manifest.json"
    matrix = load_json_object(manifest_path)
    config = load_json_object(config_path)
    errors: list[str] = []

    require_equal(
        errors,
        matrix.get("config_sha256"),
        sha256_file(config_path),
        "matrix config SHA-256",
    )
    require_equal(errors, matrix.get("require_strict_same_recall"), True, "strict gate")
    require_equal(
        errors,
        matrix.get("required_peer_premerge_mode"),
        "disabled",
        "required peer-premerge mode",
    )
    shared = matrix.get("shared") or {}
    shared_provenance = matrix.get("shared_provenance") or {}
    require_equal(errors, shared.get("stability_repeats"), args.expected_repeats, "shared repeats")
    require_equal(errors, shared.get("api"), "search", "shared API")
    require_equal(
        errors,
        shared_provenance.get("benchmark_commit"),
        args.expected_commit,
        "shared benchmark commit",
    )
    require_equal(
        errors,
        shared_provenance.get("deployment_commit"),
        args.expected_commit,
        "shared deployment commit",
    )
    require_equal(
        errors,
        shared_provenance.get("benchmark_tracked_dirty"),
        False,
        "shared benchmark tracked-dirty state",
    )
    require_equal(
        errors,
        shared_provenance.get("deployment_tracked_dirty"),
        False,
        "shared deployment tracked-dirty state",
    )
    require_equal(errors, shared_provenance.get("peer_premerge_mode"), "disabled", "peer premerge")
    require_equal(
        errors,
        shared_provenance.get("numeric_shard_count"),
        46,
        "numeric shard count",
    )
    require_equal(errors, shared_provenance.get("replication_factor"), 1, "replication factor")

    confirmations = matrix.get("same_recall_confirmation")
    require(errors, isinstance(confirmations, list), "same_recall_confirmation must be a list")
    confirmations = confirmations if isinstance(confirmations, list) else []
    configured_targets = config.get("same_recall_targets") or []
    require_equal(errors, len(confirmations), len(configured_targets), "confirmation count")
    case_targets: dict[str, float] = {}
    strict_rows: list[dict[str, Any]] = []
    for row in confirmations:
        target = finite_float(row.get("target_recall"), field="target_recall")
        row_errors: list[str] = []
        require_equal(row_errors, row.get("strict_same_recall"), True, "strict_same_recall")
        require_equal(row_errors, row.get("all_methods_strict"), True, "all_methods_strict")
        require_equal(row_errors, row.get("pairwise_within_window"), True, "pairwise window")
        require_equal(row_errors, row.get("confirmation_status"), "strict", "confirmation status")
        for method in METHODS:
            case_name = str(row.get(f"{method}_case_name") or "")
            require(row_errors, bool(case_name), f"missing {method} case name")
            if case_name:
                case_targets[case_name] = target
        if row_errors:
            errors.extend(f"target {target}: {message}" for message in row_errors)
        strict_rows.append(
            {
                "target_recall": target,
                "strict_same_recall": row.get("strict_same_recall"),
                "pairwise_recall_spread": row.get("pairwise_recall_spread"),
                "pairwise_recall_window": row.get("pairwise_recall_window"),
                "confirmation_status": row.get("confirmation_status"),
            }
        )

    case_records = matrix.get("cases")
    require(errors, isinstance(case_records, list), "matrix cases must be a list")
    case_records = case_records if isinstance(case_records, list) else []
    config_cases = config.get("cases") if isinstance(config.get("cases"), list) else []
    expected_names = {str(case.get("name")) for case in config_cases}
    actual_names = {str(case.get("name")) for case in case_records}
    require_equal(errors, actual_names, expected_names, "matrix/config case-name set")
    require_equal(
        errors,
        len(case_records),
        len(METHODS) * len(configured_targets),
        "formal case count",
    )

    top_lock = matrix.get("benchmark_lock") or {}
    expected_lock_token = top_lock.get("token_sha256")
    expected_transport_identity = shared_provenance.get("transport_identity_sha256")
    expected_image_identity = shared_provenance.get("image_identity")
    logical_points = int(
        (shared_provenance.get("dataset_shapes") or {}).get("train", [0])[0]
    )

    case_audits: list[dict[str, Any]] = []
    case_metrics: list[dict[str, Any]] = []
    repeat_metrics: list[dict[str, Any]] = []
    qps_by_target_method: dict[tuple[float, str], list[float]] = {}

    for record in sorted(case_records, key=lambda item: str(item.get("name"))):
        name = str(record.get("name") or "")
        method = str(record.get("method") or "")
        case_errors: list[str] = []
        require(case_errors, name in case_targets, "case is not selected by a strict target")
        require(case_errors, method in METHODS, f"unsupported method {method!r}")
        require_equal(case_errors, record.get("returncode"), 0, "benchmark return code")
        target = case_targets.get(name, float("nan"))
        case_dir = matrix_dir / "cases" / name
        summary_path = case_dir / "summary.json"
        case_manifest_path = case_dir / "run_manifest.json"
        stability_path = case_dir / "stability_runs.csv"
        stderr_path = matrix_dir / "logs" / f"{name}.stderr.log"
        for path in (summary_path, case_manifest_path, stability_path, stderr_path):
            require(case_errors, path.is_file(), f"missing file {path}")
        if case_errors:
            errors.extend(f"{name}: {message}" for message in case_errors)
            case_audits.append(
                {
                    "case_name": name,
                    "method": method,
                    "ok": False,
                    "errors": case_errors,
                }
            )
            continue

        summary = load_json_object(summary_path)
        case_manifest = load_json_object(case_manifest_path)
        rows = load_csv(stability_path)
        require_equal(case_errors, stderr_path.stat().st_size, 0, "stderr size")
        require_equal(case_errors, len(rows), args.expected_repeats, "stability row count")
        require_equal(
            case_errors,
            summary.get("stability_runs"),
            args.expected_repeats,
            "summary repeats",
        )
        require_equal(case_errors, summary.get("method"), method, "summary method")
        require_equal(case_errors, summary.get("placement_valid"), True, "summary placement")
        require_equal(case_errors, case_manifest.get("method"), method, "case-manifest method")

        execution = summary.get("distributed_index_execution") or {}
        for key, expected in COMMON_EXECUTION_IDENTITY.items():
            require_equal(case_errors, execution.get(key), expected, f"execution.{key}")
        for key, expected in EXECUTION_IDENTITY.get(method, {}).items():
            require_equal(case_errors, execution.get(key), expected, f"execution.{key}")

        transport = summary.get("deployment_transport") or {}
        require_equal(
            case_errors,
            transport.get("transport_identity_sha256"),
            expected_transport_identity,
            "transport identity",
        )
        peer_premerge = transport.get("peer_premerge") or {}
        require_equal(
            case_errors,
            peer_premerge.get("current_mode"),
            "disabled",
            "live peer premerge",
        )
        require_equal(
            case_errors,
            peer_premerge.get("matches_requested"),
            True,
            "peer-premerge match",
        )

        request_contract = case_manifest.get("request_contract") or {}
        require_equal(
            case_errors,
            request_contract.get("standard_coordinator_request"),
            True,
            "standard request",
        )
        for field in (
            "entry_point_hints",
            "per_shard_ef",
            "shard_selector",
            "source_id_hint",
        ):
            require_equal(
                case_errors,
                request_contract.get(field),
                False,
                f"request_contract.{field}",
            )

        placement = case_manifest.get("placement_proof") or {}
        indexing = case_manifest.get("indexing_readiness") or {}
        require_equal(case_errors, placement.get("valid"), True, "placement proof")
        require_equal(
            case_errors,
            placement.get("shard_transfers"),
            [],
            "placement shard transfers",
        )
        require_equal(case_errors, indexing.get("fully_indexed"), True, "fully indexed")
        require_equal(
            case_errors,
            indexing.get("completion_mode"),
            "fully_indexed",
            "index completion",
        )
        require_equal(case_errors, indexing.get("status"), "green", "collection status")
        require_equal(case_errors, indexing.get("optimizer_status"), "ok", "optimizer status")
        require_equal(case_errors, indexing.get("shard_transfers"), [], "index shard transfers")

        binding = case_manifest.get("repository_binding") or {}
        require_equal(
            case_errors,
            binding.get("benchmark_commit"),
            args.expected_commit,
            "benchmark commit",
        )
        require_equal(
            case_errors,
            binding.get("deployment_commit"),
            args.expected_commit,
            "deployment commit",
        )
        require_equal(case_errors, binding.get("tracked_dirty"), False, "tracked-dirty state")
        end_proof = binding.get("end_proof") or {}
        require_equal(case_errors, end_proof.get("head_unchanged"), True, "HEAD unchanged")
        require_equal(
            case_errors,
            end_proof.get("end_tracked_dirty"),
            False,
            "end tracked-dirty state",
        )

        lock = case_manifest.get("benchmark_lock") or {}
        require_equal(
            case_errors,
            lock.get("token_sha256"),
            expected_lock_token,
            "benchmark lock token",
        )
        deployment = case_manifest.get("deployment") or {}
        deployment_image = deployment.get("image") or {}
        require_equal(
            case_errors,
            deployment_image.get("id"),
            expected_image_identity,
            "image identity",
        )

        final = summary.get("final_metrics") or {}
        repeat_qps: list[float] = []
        for row in rows:
            repeat_qps.append(finite_float(row.get("qps"), field=f"{name}.qps"))
            require_equal(case_errors, row.get("method"), method, "repeat method")
            require_equal(
                case_errors,
                row.get("execution_mode"),
                EXECUTION_IDENTITY[method]["executor"],
                "repeat execution mode",
            )
            require_equal(
                case_errors,
                row.get("transport_mode"),
                COMMON_EXECUTION_IDENTITY["transport_mode"],
                "repeat transport mode",
            )
            repeat_metrics.append(
                {
                    "target_recall": target,
                    "method": method,
                    "case_name": name,
                    "repeat": row.get("run"),
                    "recall_at_k": row.get("recall_at_k"),
                    "qps": row.get("qps"),
                    "latency_p50_batch_ms": row.get("latency_p50_ms"),
                    "latency_p95_batch_ms": row.get("latency_p95_ms"),
                    "latency_p99_batch_ms": row.get("latency_p99_ms"),
                }
            )
        qps_by_target_method[(target, method)] = repeat_qps
        require(
            case_errors,
            math.isclose(
                statistics.fmean(repeat_qps),
                finite_float(final.get("qps"), field=f"{name}.final.qps"),
                rel_tol=1.0e-12,
                abs_tol=1.0e-12,
            ),
            "summary QPS does not equal repeat mean",
        )

        physical_points = int(indexing.get("points_count") or 0)
        case_metrics.append(
            {
                "target_recall": target,
                "method": method,
                "case_name": name,
                "recall_at_k": final.get("recall_at_k"),
                "qps_mean": final.get("qps"),
                "qps_stdev": final.get("qps_stdev"),
                "latency_p50_batch_ms": final.get("latency_p50_ms"),
                "latency_p95_batch_ms": final.get("latency_p95_ms"),
                "latency_p99_batch_ms": final.get("latency_p99_ms"),
                "visited_shards": final.get("visited_shards"),
                "ef_sum_per_query": final.get("ef_sum_per_query"),
                "planner": EXECUTION_IDENTITY[method]["planner"],
                "executor": EXECUTION_IDENTITY[method]["executor"],
                "logical_points": logical_points,
                "physical_points": physical_points,
                "index_expansion_ratio": physical_points / logical_points,
                "stability_repeats": len(rows),
            }
        )
        if case_errors:
            errors.extend(f"{name}: {message}" for message in case_errors)
        case_audits.append(
            {
                "case_name": name,
                "method": method,
                "target_recall": target,
                "ok": not case_errors,
                "errors": case_errors,
                "stderr_size_bytes": stderr_path.stat().st_size,
                "stability_rows": len(rows),
                "benchmark_lock_token_sha256": lock.get("token_sha256"),
                "transport_identity_sha256": transport.get("transport_identity_sha256"),
                "image_identity": deployment_image.get("id"),
            }
        )

    comparisons: list[dict[str, Any]] = []
    for target in sorted(float(value) for value in configured_targets):
        for baseline in ("hash_all", "simple_kmeans"):
            key_a = (target, "orion")
            key_b = (target, baseline)
            if key_a not in qps_by_target_method or key_b not in qps_by_target_method:
                errors.append(f"target {target}: missing QPS repeats for Orion vs {baseline}")
                continue
            comparisons.append(
                welch_comparison(
                    qps_by_target_method[key_a],
                    qps_by_target_method[key_b],
                    target_recall=target,
                    method_a="orion",
                    method_b=baseline,
                )
            )

    case_metrics.sort(
        key=lambda row: (
            float(row["target_recall"]),
            METHODS.index(str(row["method"])),
        )
    )
    repeat_metrics.sort(
        key=lambda row: (
            float(row["target_recall"]),
            METHODS.index(str(row["method"])),
            int(str(row["repeat"])),
        )
    )
    report = {
        "schema_version": 1,
        "ok": not errors,
        "errors": errors,
        "matrix": {
            "run_id": matrix.get("run_id"),
            "path": str(matrix_dir),
            "run_manifest_sha256": sha256_file(manifest_path),
            "config_path": str(config_path),
            "config_sha256": sha256_file(config_path),
            "expected_commit": args.expected_commit,
            "expected_repeats": args.expected_repeats,
        },
        "shared_identity": {
            "benchmark_commit": shared_provenance.get("benchmark_commit"),
            "deployment_commit": shared_provenance.get("deployment_commit"),
            "image_identity": expected_image_identity,
            "image_tag": shared_provenance.get("image_tag"),
            "image_source_fingerprint": shared_provenance.get("image_source_fingerprint"),
            "transport_identity_sha256": expected_transport_identity,
            "benchmark_lock_token_sha256": expected_lock_token,
            "cluster_architecture": COMMON_EXECUTION_IDENTITY["cluster_architecture"],
            "replica_read_path": COMMON_EXECUTION_IDENTITY["replica_read_path"],
            "global_merge_path": COMMON_EXECUTION_IDENTITY["global_merge_path"],
            "transport_mode": COMMON_EXECUTION_IDENTITY["transport_mode"],
            "peer_premerge_mode": COMMON_EXECUTION_IDENTITY["peer_premerge_mode"],
        },
        "strict_same_recall": strict_rows,
        "case_audits": case_audits,
        "case_metrics": case_metrics,
        "repeat_level_qps_comparisons": comparisons,
        "statistical_scope": {
            "unit": "repeat run",
            "repeats_per_case": args.expected_repeats,
            "case_order": "block ordered, not interleaved",
            "comparison": "unpaired Welch descriptive interval",
            "warning": (
                "Small n and environmental drift make especially small differences "
                "non-decisive; timed queries within a repeat are not independent units."
            ),
        },
    }

    if args.output_dir is not None:
        output_dir = args.output_dir.expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "audit_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        write_csv(output_dir / "case_metrics.csv", case_metrics, CASE_FIELDS)
        write_csv(output_dir / "repeat_metrics.csv", repeat_metrics, REPEAT_FIELDS)
        write_csv(output_dir / "welch_qps_comparisons.csv", comparisons, COMPARISON_FIELDS)
    return report


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.expected_repeats < 2:
        raise ValueError("--expected-repeats must be at least 2")
    report = audit(args)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
