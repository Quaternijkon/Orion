#!/usr/bin/env python3
"""Aggregate completed C6 isolated/full summaries into required CSV tables."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from c6_protocol import write_csv_atomic, write_json_atomic  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isolated", action="append", required=True)
    parser.add_argument("--measurement", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--verdict-output", required=True)
    parser.add_argument("--physical")
    parser.add_argument("--query-difficulty", action="append", default=[])
    parser.add_argument("--latex-output")
    return parser.parse_args(argv)


def load(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    shard_rows = []
    local_rows = []
    for path in args.isolated:
        payload = load(path)
        e2 = payload["E2"]
        e3 = payload["E3"]
        paired_e2 = e2["paired"]["paired_work_relative_reduction"]
        paired_e3 = e3["paired_work_relative_reduction"]
        shard_rows.append(
            {
                "dataset": payload["dataset"],
                "logical_shards": payload["logical_shards"],
                "common_ef_search": e2["selection"]["common_ef_search"],
                "static_p": e2["selection"]["static_p"],
                "p0_recall_at_10": e2["P0"]["recall_at_10"],
                "p1_recall_at_10": e2["P1"]["recall_at_10"],
                "recall_gate_valid": (
                    e2["P0"]["recall_at_10"] >= 0.90
                    and e2["P1"]["recall_at_10"] >= 0.90
                ),
                "p0_mean_shards": e2["P0"]["mean_shards_per_query"],
                "p1_mean_shards": e2["P1"]["mean_shards_per_query"],
                "p4_mean_shards": e2["P4"]["mean_shards_per_query"],
                "mean_shards_saved": e2["paired"]["mean_shards_saved"],
                "p0_work": e2["P0"]["aggregate_distance_computations_per_query"],
                "p1_work": e2["P1"]["aggregate_distance_computations_per_query"],
                "paired_work_relative_reduction": paired_e2[
                    "mean_relative_reduction"
                ],
                "paired_work_ci95_low": paired_e2["ci95_low"],
                "paired_work_ci95_high": paired_e2["ci95_high"],
                "fraction_fewer_shards": e2["paired"]["fraction_fewer_shards"],
                "fraction_equal_shards": e2["paired"]["fraction_equal_shards"],
                "fraction_more_shards": e2["paired"]["fraction_more_shards"],
            }
        )
        calibration = e3["entry_point_calibration"]
        local_rows.append(
            {
                "dataset": payload["dataset"],
                "logical_shards": payload["logical_shards"],
                "fixed_p": e3["selection"]["fixed_p"],
                "uniform_ef_search": e3["selection"]["uniform"]["config"][
                    "uniform_ef_search"
                ],
                "adaptive_alpha": e3["selection"]["adaptive"]["config"]["alpha"],
                "adaptive_beta": e3["selection"]["adaptive"]["config"]["beta"],
                "p0_recall_at_10": e3["P0"]["recall_at_10"],
                "p2_recall_at_10": e3["P2"]["recall_at_10"],
                "p5_recall_at_10": e3["P5"]["recall_at_10"],
                "recall_gate_valid": (
                    e3["P0"]["recall_at_10"] >= 0.90
                    and e3["P2"]["recall_at_10"] >= 0.90
                ),
                "p0_work": e3["P0"]["aggregate_distance_computations_per_query"],
                "p2_work": e3["P2"]["aggregate_distance_computations_per_query"],
                "p5_work": e3["P5"]["aggregate_distance_computations_per_query"],
                "paired_work_relative_reduction": paired_e3[
                    "mean_relative_reduction"
                ],
                "paired_work_ci95_low": paired_e3["ci95_low"],
                "paired_work_ci95_high": paired_e3["ci95_high"],
                "p0_p95_max_shard_work": e3["P0"][
                    "p95_max_shard_distance_computations"
                ],
                "p2_p95_max_shard_work": e3["P2"][
                    "p95_max_shard_distance_computations"
                ],
                "p5_p95_max_shard_work": e3["P5"][
                    "p95_max_shard_distance_computations"
                ],
                "spearman_ep_vs_gt_contribution": calibration[
                    "spearman_entry_points_vs_ground_truth_contribution"
                ],
                "spearman_ep_vs_oracle_work": calibration[
                    "spearman_entry_points_vs_oracle_local_work"
                ],
            }
        )

    full_rows = []
    oracle_rows = []
    for path in args.measurement:
        payload = load(path)
        summaries = {row["policy"]: row for row in payload["summaries"]}
        p0_work = summaries["P0"]["aggregate_distance_computations_per_query"]
        for policy in ("P0", "P1", "P2", "P3", "P4", "P5", "P6"):
            summary = summaries[policy]
            full_rows.append(
                {
                    "dataset": payload["dataset"],
                    "logical_shards": payload["logical_shards"],
                    "run_id": payload["run_id"],
                    "fallback_used": "fallback" in str(payload["run_id"]),
                    "policy": policy,
                    "recall_at_10": summary["recall_at_10"],
                    "recall_gate_valid": summary["recall_at_10"] >= 0.90,
                    "mean_shards_per_query": summary["mean_shards_per_query"],
                    "aggregate_distance_computations": summary[
                        "aggregate_distance_computations_per_query"
                    ],
                    "p95_max_shard_work": summary[
                        "p95_max_shard_distance_computations"
                    ],
                    "relative_work_vs_p0": summary[
                        "aggregate_distance_computations_per_query"
                    ]
                    / p0_work,
                }
            )
        oracle_rows.append(
            {
                "dataset": payload["dataset"],
                "logical_shards": payload["logical_shards"],
                "run_id": payload["run_id"],
                "fallback_used": "fallback" in str(payload["run_id"]),
                "p0_work": p0_work,
                "p3_work": summaries["P3"][
                    "aggregate_distance_computations_per_query"
                ],
                "p6_work": summaries["P6"][
                    "aggregate_distance_computations_per_query"
                ],
                "p3_minus_p6": summaries["P3"][
                    "aggregate_distance_computations_per_query"
                ]
                - summaries["P6"]["aggregate_distance_computations_per_query"],
                "p3_over_p6": summaries["P3"][
                    "aggregate_distance_computations_per_query"
                ]
                / summaries["P6"]["aggregate_distance_computations_per_query"],
                "oracle_headroom_relative_to_p0": (
                    summaries["P3"]["aggregate_distance_computations_per_query"]
                    - summaries["P6"]["aggregate_distance_computations_per_query"]
                )
                / p0_work,
            }
        )

    shard_rows.sort(key=lambda row: (row["dataset"], row["logical_shards"]))
    local_rows.sort(key=lambda row: (row["dataset"], row["logical_shards"]))
    full_rows.sort(
        key=lambda row: (row["dataset"], row["logical_shards"], row["policy"])
    )
    oracle_rows.sort(key=lambda row: (row["dataset"], row["logical_shards"]))
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_csv_atomic(output / "c6_shard_adaptation_summary.csv", shard_rows)
    write_csv_atomic(output / "c6_local_budget_summary.csv", local_rows)
    write_csv_atomic(output / "c6_full_ablation_summary.csv", full_rows)
    write_csv_atomic(output / "c6_oracle_gap_summary.csv", oracle_rows)
    datasets = sorted({row["dataset"] for row in full_rows})
    valid_b = [row for row in shard_rows if row["recall_gate_valid"]]
    valid_c = [row for row in local_rows if row["recall_gate_valid"]]
    b_improvements = [
        row["p1_work"] < row["p0_work"] or row["p1_mean_shards"] < row["p0_mean_shards"]
        for row in valid_b
    ]
    c_improvements = [
        row["p2_work"] < row["p0_work"]
        or row["p2_p95_max_shard_work"] < row["p0_p95_max_shard_work"]
        for row in valid_c
    ]
    by_measurement = {}
    for row in full_rows:
        by_measurement.setdefault((row["dataset"], row["logical_shards"]), {})[
            row["policy"]
        ] = row
    valid_d = [
        rows
        for rows in by_measurement.values()
        if rows["P0"]["recall_gate_valid"] and rows["P3"]["recall_gate_valid"]
    ]
    d_improvements = [
        rows["P3"]["aggregate_distance_computations"]
        < rows["P0"]["aggregate_distance_computations"]
        for rows in valid_d
    ]
    d_near_oracle = [
        rows["P3"]["aggregate_distance_computations"]
        <= 1.25 * rows["P6"]["aggregate_distance_computations"]
        for rows in valid_d
    ]
    difficulty = [load(path) for path in args.query_difficulty]
    difficulty_supported = bool(difficulty) and all(
        float(row["oracle_prefix_shards"]["standard_deviation"]) > 0
        and float(row["oracle_max_local_ef"]["standard_deviation"]) > 0
        for row in difficulty
    )

    def mixed_verdict(values: list[bool], *, supported: str, contradicted: str) -> str:
        if values and all(values):
            return supported
        if any(values):
            return "CONTRADICTED_AS_GENERAL_CLAIM_DATASET_OR_SCALE_DEPENDENT"
        return contradicted

    verdict = {
        "scope": f"{', '.join(datasets)} M=4,8,16,32 algorithmic evidence",
        "C6-a": "SUPPORTED" if difficulty_supported else "INSUFFICIENT",
        "C6-b": mixed_verdict(
            b_improvements,
            supported="SUPPORTED",
            contradicted="CONTRADICTED_CURRENT_ADAPTIVE_SHARD_POLICY",
        ),
        "C6-c": mixed_verdict(
            c_improvements,
            supported="SUPPORTED",
            contradicted="CONTRADICTED_CURRENT_LINEAR_EP_POLICY",
        ),
        "C6-d": (
            "SUPPORTED"
            if d_improvements and all(d_improvements) and all(d_near_oracle)
            else "CONTRADICTED_NEAR_ORACLE_OR_GENERAL_FULL_POLICY_CLAIM"
        ),
        "notes": {
            "recall_invalid_E2_rows": [
                [row["dataset"], row["logical_shards"]]
                for row in shard_rows
                if not row["recall_gate_valid"]
            ],
            "recall_invalid_E3_rows": [
                [row["dataset"], row["logical_shards"]]
                for row in local_rows
                if not row["recall_gate_valid"]
            ],
            "C6-b_improvement_by_valid_row": b_improvements,
            "C6-c_improvement_by_valid_row": c_improvements,
            "C6-d_static_improvement_by_valid_row": d_improvements,
            "C6-d_near_oracle_by_valid_row": d_near_oracle,
        },
    }
    if args.physical:
        physical_rows = list(csv.DictReader(Path(args.physical).open(encoding="utf-8")))
        valid = {
            f"{row['dataset']}:{row['policy']}": (
                str(row["physical_run_valid"]).lower() == "true"
            )
            for row in physical_rows
        }
        verdict["physical_validation"] = {
            "status": (
                "SUPPORTED_ALL_POLICIES_STABLE"
                if valid and all(valid.values())
                else "INSUFFICIENT_UNSTABLE_POLICY_REPETITIONS"
            ),
            "valid_by_dataset_policy": valid,
            "source": str(Path(args.physical).resolve()),
        }
    write_json_atomic(args.verdict_output, verdict)

    physical_lookup = {
        (row["dataset"], int(row["logical_shards"]), row["policy"]): row
        for row in (physical_rows if args.physical else [])
    }
    latex_lines = [
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r"Dataset & $M$ & Policy & Recall@10 & Shards/query & Dist. comps. & P95 max-shard & QPS / P99 $\mu$s \\",
        r"\midrule",
    ]
    for row in full_rows:
        physical = physical_lookup.get(
            (row["dataset"], int(row["logical_shards"]), row["policy"])
        )
        qps_latency = (
            f"{float(physical['qps_mean']):.1f} / {float(physical['p99_latency_us']):.0f}"
            if physical
            else "--"
        )
        latex_lines.append(
            f"{row['dataset']} & {row['logical_shards']} & {row['policy']} & "
            f"{row['recall_at_10']:.4f} & {row['mean_shards_per_query']:.3f} & "
            f"{row['aggregate_distance_computations']:.1f} & "
            f"{row['p95_max_shard_work']:.1f} & {qps_latency} \\\\"
        )
    latex_lines.extend([r"\bottomrule", r"\end{tabular}"])
    latex_path = Path(args.latex_output) if args.latex_output else output / "c6_latex_table.tex"
    latex_path = latex_path.expanduser().resolve()
    latex_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = latex_path.with_suffix(latex_path.suffix + ".tmp")
    temporary.write_text("\n".join(latex_lines) + "\n", encoding="utf-8")
    os.replace(temporary, latex_path)
    print(json.dumps({"output_dir": str(output.resolve()), "verdict": verdict}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
