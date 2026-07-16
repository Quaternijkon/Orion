#!/usr/bin/env python3
"""Summarize request/candidate pressure proxy metrics from Claim E/F runs."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


CAVEAT = "counts_only_not_serialized_bytes_or_network_payload"


FIELDNAMES = [
    "claim_id",
    "evidence_kind",
    "comparison_scope",
    "baseline_variant",
    "variant",
    "batch_size",
    "repeat_count",
    "recall_at_10_mean",
    "qps_mean",
    "p95_batch_ms_mean",
    "p99_batch_ms_mean",
    "avg_search_requests_per_query_mean",
    "avg_candidate_groups_per_query_mean",
    "avg_returned_candidates_per_query_mean",
    "search_request_reduction_vs_baseline_pct",
    "candidate_group_reduction_vs_baseline_pct",
    "returned_candidate_reduction_vs_baseline_pct",
    "source_summary_csv",
    "caveat",
]


CLAIM_SPECS = {
    "E": {
        "evidence_kind": "request_object_pressure_proxy",
        "comparison_scope": "compact_request_execution_modes",
        "baseline_variant": "client_shard_major_expanded",
    },
    "F": {
        "evidence_kind": "candidate_merge_pressure_proxy",
        "comparison_scope": "worker_local_premerge_modes",
        "baseline_variant": "direct_peer_no_premerge",
    },
}


def float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def read_success_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return [row for row in csv.DictReader(handle) if row.get("status", "ok") == "ok"]


def safe_mean(rows: list[dict[str, str]], field: str) -> float | str:
    values = [value for row in rows if (value := float_or_none(row.get(field))) is not None]
    return mean(values) if values else ""


def reduction_pct(value: float | str, baseline: float | str) -> float | str:
    if value == "" or baseline in ("", 0):
        return ""
    return (float(baseline) - float(value)) / float(baseline) * 100.0


def aggregate_claim_rows(
    *,
    claim_id: str,
    summary_path: Path,
    rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    spec = CLAIM_SPECS[claim_id]
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        batch_size = row.get("batch_size", "")
        variant = row.get("variant", "")
        if batch_size and variant:
            grouped[(batch_size, variant)].append(row)

    baseline_by_batch: dict[str, dict[str, Any]] = {}
    aggregates: list[dict[str, Any]] = []
    for (batch_size, variant), group_rows in sorted(grouped.items(), key=lambda item: (int(item[0][0]), item[0][1])):
        aggregate = {
            "claim_id": claim_id,
            "evidence_kind": spec["evidence_kind"],
            "comparison_scope": spec["comparison_scope"],
            "baseline_variant": spec["baseline_variant"],
            "variant": variant,
            "batch_size": int(batch_size),
            "repeat_count": len(group_rows),
            "recall_at_10_mean": safe_mean(group_rows, "recall_at_10"),
            "qps_mean": safe_mean(group_rows, "qps"),
            "p95_batch_ms_mean": safe_mean(group_rows, "batch_latency_p95_ms"),
            "p99_batch_ms_mean": safe_mean(group_rows, "batch_latency_p99_ms"),
            "avg_search_requests_per_query_mean": safe_mean(group_rows, "avg_search_requests_per_query"),
            "avg_candidate_groups_per_query_mean": safe_mean(group_rows, "avg_candidate_groups_per_query"),
            "avg_returned_candidates_per_query_mean": safe_mean(group_rows, "avg_returned_candidates_per_query"),
            "source_summary_csv": str(summary_path),
            "caveat": CAVEAT,
        }
        if variant == spec["baseline_variant"]:
            baseline_by_batch[batch_size] = aggregate
        aggregates.append(aggregate)

    for aggregate in aggregates:
        baseline = baseline_by_batch.get(str(aggregate["batch_size"]))
        aggregate["search_request_reduction_vs_baseline_pct"] = reduction_pct(
            aggregate["avg_search_requests_per_query_mean"],
            baseline["avg_search_requests_per_query_mean"] if baseline else "",
        )
        aggregate["candidate_group_reduction_vs_baseline_pct"] = reduction_pct(
            aggregate["avg_candidate_groups_per_query_mean"],
            baseline["avg_candidate_groups_per_query_mean"] if baseline else "",
        )
        aggregate["returned_candidate_reduction_vs_baseline_pct"] = reduction_pct(
            aggregate["avg_returned_candidates_per_query_mean"],
            baseline["avg_returned_candidates_per_query_mean"] if baseline else "",
        )
    return aggregates


def build_proxy_rows(
    *,
    claim_e_summary: Path,
    claim_f_summary: Path,
    output_path: Path | None,
) -> list[dict[str, Any]]:
    rows = []
    rows.extend(
        aggregate_claim_rows(
            claim_id="E",
            summary_path=claim_e_summary,
            rows=read_success_rows(claim_e_summary),
        )
    )
    rows.extend(
        aggregate_claim_rows(
            claim_id="F",
            summary_path=claim_f_summary,
            rows=read_success_rows(claim_f_summary),
        )
    )
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
            writer.writeheader()
            writer.writerows(rows)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--claim-e-summary",
        type=Path,
        default=Path(
            "results/method4_claim_e_execution_mode_latency_20260704/"
            "analysis_20260704_225847/claim_e_execution_mode_latency_summary.csv"
        ),
    )
    parser.add_argument(
        "--claim-f-summary",
        type=Path,
        default=Path(
            "results/method4_claim_f_premerge_batch_latency_20260704/"
            "analysis_20260704_222700/claim_f_premerge_batch_latency_summary.csv"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/method4_claim_coverage_20260704/derived_claim_tables/"
            "crosscut_request_candidate_pressure_proxy.csv"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_proxy_rows(
        claim_e_summary=args.claim_e_summary,
        claim_f_summary=args.claim_f_summary,
        output_path=args.output,
    )
    print(args.output)


if __name__ == "__main__":
    main()
