#!/usr/bin/env python3
"""Build Claim C Dynamic-vs-fixed EF latency tables from a latency rerun."""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path
from typing import Any


SUMMARY_NAME = "claim_d_high_recall_latency_summary.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate Claim C Dynamic-vs-fixed EF latency rerun.")
    parser.add_argument(
        "--input-dir",
        default="results/method4_claim_c_dynamic_vs_fixed_latency_20260704/analysis_20260704_235925",
    )
    parser.add_argument(
        "--output-dir",
        default="results/method4_claim_coverage_20260704/derived_claim_tables",
    )
    parser.add_argument("--comparison-label", default="latency_rerun_u80_095_b200")
    return parser.parse_args()


def float_value(value: Any) -> float:
    if value in ("", None):
        return 0.0
    return float(value)


def int_value(value: Any) -> int:
    if value in ("", None):
        return 0
    return int(float(value))


def read_summary(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def stdev(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def pct_delta(value: float, baseline: float) -> float:
    if baseline == 0.0:
        return 0.0
    return (value - baseline) / baseline * 100.0


def method_name(row: dict[str, str]) -> str:
    return "Fixed EF" if int_value(row.get("factor")) == 0 else "Dynamic EF"


def params_label(row: dict[str, str]) -> str:
    upper_k = int_value(row.get("upper_k"))
    base_ef = int_value(row.get("base_ef"))
    factor = int_value(row.get("factor"))
    if factor == 0:
        return f"upper_k={upper_k}; fixed_ef={base_ef}"
    return f"upper_k={upper_k}; base_ef={base_ef}; factor={factor}"


def aggregate_config(rows: list[dict[str, str]], comparison_label: str, source_dir: Path) -> dict[str, Any]:
    first = rows[0]
    fields = {
        "recall_at_10": [float_value(row.get("recall_at_10")) for row in rows],
        "qps": [float_value(row.get("qps")) for row in rows],
        "avg_visited_shards": [float_value(row.get("avg_visited_shards")) for row in rows],
        "avg_assigned_ef_sum": [float_value(row.get("avg_assigned_ef_sum")) for row in rows],
        "batch_latency_p50_ms": [float_value(row.get("batch_latency_p50_ms")) for row in rows],
        "batch_latency_p95_ms": [float_value(row.get("batch_latency_p95_ms")) for row in rows],
        "batch_latency_p99_ms": [float_value(row.get("batch_latency_p99_ms")) for row in rows],
    }
    return {
        "comparison": comparison_label,
        "method": method_name(first),
        "config_label": first.get("config_label", ""),
        "params": params_label(first),
        "collection": first.get("collection", ""),
        "batch_size": int_value(first.get("batch_size")),
        "repeats": len(rows),
        "recall_at_10_mean": mean(fields["recall_at_10"]),
        "recall_at_10_stdev": stdev(fields["recall_at_10"]),
        "qps_mean": mean(fields["qps"]),
        "qps_stdev": stdev(fields["qps"]),
        "qps_median": median(fields["qps"]),
        "avg_visited_shards_mean": mean(fields["avg_visited_shards"]),
        "avg_assigned_ef_sum_mean": mean(fields["avg_assigned_ef_sum"]),
        "batch_latency_p50_ms_mean": mean(fields["batch_latency_p50_ms"]),
        "batch_latency_p95_ms_mean": mean(fields["batch_latency_p95_ms"]),
        "batch_latency_p95_ms_median": median(fields["batch_latency_p95_ms"]),
        "batch_latency_p99_ms_mean": mean(fields["batch_latency_p99_ms"]),
        "batch_latency_p99_ms_median": median(fields["batch_latency_p99_ms"]),
        "source_summary_csv": str(source_dir / SUMMARY_NAME),
        "notes": "client-observed batch latency rerun using Claim D latency runner; same live Orion collection",
    }


def build_latency_tables(input_dir: Path, comparison_label: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summary_path = input_dir / SUMMARY_NAME
    rows = read_summary(summary_path)
    by_config: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_config.setdefault(row.get("config_label", ""), []).append(row)
    matrix = [
        aggregate_config(config_rows, comparison_label, input_dir)
        for _label, config_rows in sorted(by_config.items())
    ]
    fixed = next(row for row in matrix if row["method"] == "Fixed EF")
    dynamic = next(row for row in matrix if row["method"] == "Dynamic EF")
    deltas = [
        {
            "comparison": comparison_label,
            "recall_delta_dynamic_minus_fixed": dynamic["recall_at_10_mean"] - fixed["recall_at_10_mean"],
            "qps_delta_dynamic_minus_fixed": dynamic["qps_mean"] - fixed["qps_mean"],
            "qps_delta_pct": pct_delta(dynamic["qps_mean"], fixed["qps_mean"]),
            "qps_delta_pct_median": pct_delta(dynamic["qps_median"], fixed["qps_median"]),
            "visited_shards_delta": dynamic["avg_visited_shards_mean"] - fixed["avg_visited_shards_mean"],
            "visited_shards_delta_pct": pct_delta(
                dynamic["avg_visited_shards_mean"], fixed["avg_visited_shards_mean"]
            ),
            "ef_sum_delta": dynamic["avg_assigned_ef_sum_mean"] - fixed["avg_assigned_ef_sum_mean"],
            "ef_sum_delta_pct": pct_delta(dynamic["avg_assigned_ef_sum_mean"], fixed["avg_assigned_ef_sum_mean"]),
            "p95_latency_delta_ms": dynamic["batch_latency_p95_ms_mean"] - fixed["batch_latency_p95_ms_mean"],
            "p95_latency_delta_pct": pct_delta(
                dynamic["batch_latency_p95_ms_mean"], fixed["batch_latency_p95_ms_mean"]
            ),
            "p95_latency_delta_pct_median": pct_delta(
                dynamic["batch_latency_p95_ms_median"], fixed["batch_latency_p95_ms_median"]
            ),
            "p99_latency_delta_ms": dynamic["batch_latency_p99_ms_mean"] - fixed["batch_latency_p99_ms_mean"],
            "p99_latency_delta_pct": pct_delta(
                dynamic["batch_latency_p99_ms_mean"], fixed["batch_latency_p99_ms_mean"]
            ),
            "p99_latency_delta_pct_median": pct_delta(
                dynamic["batch_latency_p99_ms_median"], fixed["batch_latency_p99_ms_median"]
            ),
            "fixed_repeats": fixed["repeats"],
            "dynamic_repeats": dynamic["repeats"],
            "source_summary_csv": str(summary_path),
            "notes": "Latency rerun; use as P95/P99 supplement to the prior Claim C same-recall QPS and EF-sum evidence.",
        }
    ]
    return matrix, deltas


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"no rows to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    matrix, deltas = build_latency_tables(input_dir, args.comparison_label)
    matrix_path = output_dir / "claim_c_dynamic_vs_fixed_latency_matrix.csv"
    deltas_path = output_dir / "claim_c_dynamic_vs_fixed_latency_deltas.csv"
    write_csv(matrix_path, matrix)
    write_csv(deltas_path, deltas)
    print(matrix_path)
    print(deltas_path)


if __name__ == "__main__":
    main()
