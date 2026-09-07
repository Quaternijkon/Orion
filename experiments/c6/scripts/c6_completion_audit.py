#!/usr/bin/env python3
"""Audit final C6 artifacts against the explicit protocol deliverables."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from c6_protocol import utc_timestamp, write_json_atomic  # noqa: E402


FIGURES = {
    "c6_fig1_oracle_shard_difficulty_cdf.pdf",
    "c6_fig2_oracle_local_budget_cdf.pdf",
    "c6_fig3_static_waste_breakdown.pdf",
    "c6_fig4_static_vs_adaptive_fanout.pdf",
    "c6_fig5_adaptive_vs_oracle_fanout.pdf",
    "c6_fig6_per_query_shard_savings_cdf.pdf",
    "c6_fig7_recall_work_frontier_shard_adaptation.pdf",
    "c6_fig8_uniform_vs_adaptive_ef.pdf",
    "c6_fig9_local_budget_vs_oracle.pdf",
    "c6_fig10_max_shard_work.pdf",
    "c6_fig11_entry_point_count_calibration.pdf",
    "c6_fig12_physical_tail_latency.pdf",
    "c6_fig13_full_ablation.pdf",
    "c6_fig14_full_recall_work_frontier.pdf",
    "c6_fig15_oracle_gap.pdf",
    "c6_fig16_physical_end_to_end.pdf",
    "c6_fig17_physical_qps.pdf",
    "c6_fig18_physical_p99_latency.pdf",
    "c6_fig19_physical_work_vs_latency.pdf",
    "c6_combined_adaptivity.pdf",
}

TABLES = {
    "c6_query_difficulty_summary.csv": 8,
    "c6_shard_adaptation_summary.csv": 8,
    "c6_local_budget_summary.csv": 8,
    "c6_full_ablation_summary.csv": 56,
    "c6_oracle_gap_summary.csv": 8,
    "c6_physical_validation_summary.csv": 8,
    "c6_sensitivity_summary.csv": 80,
}

FINAL_MEASUREMENTS = {
    ("sift1m", 4): "sift1m-m4-measurement-completion.json",
    ("sift1m", 8): "sift1m-m8-measurement-completion.json",
    ("sift1m", 16): "sift1m-m16-measurement-completion.json",
    ("sift1m", 32): "sift1m-m32-measurement-completion.json",
    ("glove-200-angular", 4): "glove-m4-measurement-completion.json",
    ("glove-200-angular", 8): "glove-m8-measurement-fallback-p2-rank2-completion.json",
    ("glove-200-angular", 16): "glove-m16-measurement-completion.json",
    ("glove-200-angular", 32): "glove-m32-measurement-fallback-p3-rank1-completion.json",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=str(SCRIPT_DIR.parents[2]))
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(args.repo_root).expanduser().resolve()
    c6 = root / "experiments/c6"
    runs = c6 / "runs"
    checks: list[dict[str, Any]] = []

    def check(name: str, condition: bool, evidence: Any) -> None:
        checks.append({"name": name, "passed": bool(condition), "evidence": evidence})

    protocol = root / "plan/Orion_C6_Experimental_Protocol.md"
    copied = c6 / "PLAN.md"
    check(
        "protocol copied exactly",
        protocol.is_file() and copied.is_file() and sha256(protocol) == sha256(copied),
        {"source": str(protocol), "copy": str(copied)},
    )
    for directory in (runs, runs / "per_query", runs / "per_shard", runs / "oracle", c6 / "figures", c6 / "scripts", c6 / "logs"):
        check(f"directory {directory.name}", directory.is_dir(), str(directory))

    for (dataset, logical_shards), filename in FINAL_MEASUREMENTS.items():
        measurement_path = runs / filename
        measurement = json.loads(measurement_path.read_text(encoding="utf-8"))
        summaries = {row["policy"]: row for row in measurement["summaries"]}
        check(
            f"final measurement {dataset} M={logical_shards}",
            measurement["query_count"] == 9000
            and all(summaries[policy]["recall_at_10"] >= 0.90 for policy in ("P0", "P1", "P2", "P3")),
            {policy: summaries[policy]["recall_at_10"] for policy in ("P0", "P1", "P2", "P3")},
        )
        prefix = "sift1m" if dataset == "sift1m" else "glove"
        check(
            f"E1 {dataset} M={logical_shards}",
            (runs / "e1" / f"{prefix}-m{logical_shards}-e1-summary.json").is_file(),
            logical_shards,
        )
        check(
            f"isolated E2/E3 {dataset} M={logical_shards}",
            (runs / f"{prefix}-m{logical_shards}-isolated-e2-e3-summary.json").is_file(),
            logical_shards,
        )

    def policy_recall(filename: str, policy: str) -> float:
        payload = json.loads((runs / filename).read_text(encoding="utf-8"))
        return float(next(row for row in payload["summaries"] if row["policy"] == policy)["recall_at_10"])

    fallback_evidence = {
        "m8_primary_p2": policy_recall("glove-m8-measurement-completion.json", "P2"),
        "m8_rank1_p2": policy_recall("glove-m8-measurement-fallback-p2-rank1-completion.json", "P2"),
        "m8_rank2_p2": policy_recall("glove-m8-measurement-fallback-p2-rank2-completion.json", "P2"),
        "m32_primary_p3": policy_recall("glove-m32-measurement-completion.json", "P3"),
        "m32_rank1_p3": policy_recall("glove-m32-measurement-fallback-p3-rank1-completion.json", "P3"),
    }
    check(
        "invalid primaries and valid preordered fallbacks retained",
        fallback_evidence["m8_primary_p2"] < 0.90
        and fallback_evidence["m8_rank1_p2"] < 0.90
        and fallback_evidence["m8_rank2_p2"] >= 0.90
        and fallback_evidence["m32_primary_p3"] < 0.90
        and fallback_evidence["m32_rank1_p3"] >= 0.90,
        fallback_evidence,
    )

    for filename, expected_rows in TABLES.items():
        path = runs / filename
        actual = len(rows(path)) if path.is_file() else -1
        check(f"table {filename}", actual == expected_rows, {"expected": expected_rows, "actual": actual})
    latex = runs / "c6_latex_table.tex"
    check("LaTeX table", latex.is_file() and latex.stat().st_size > 0, str(latex))

    actual_figures = {path.name for path in (c6 / "figures").glob("c6_*.pdf") if path.stat().st_size > 0}
    check("20 required PDF figures", actual_figures == FIGURES, sorted(actual_figures))

    physical = rows(runs / "c6_physical_validation_summary.csv")
    physical_status = {
        f"{row['dataset']}:{row['policy']}": row["physical_run_valid"].lower() == "true"
        for row in physical
    }
    expected_physical = {
        "glove-200-angular:P0": True,
        "glove-200-angular:P1": True,
        "glove-200-angular:P2": True,
        "glove-200-angular:P3": True,
        "sift1m:P0": True,
        "sift1m:P1": True,
        "sift1m:P2": False,
        "sift1m:P3": True,
    }
    check("physical E5 retained validity", physical_status == expected_physical, physical_status)

    sensitivity = rows(runs / "c6_sensitivity_summary.csv")
    k_values = {
        dataset: sorted(
            {
                int(row["value"])
                for row in sensitivity
                if row["dataset"] == dataset and row["sensitivity_parameter"] == "navigation_k"
            }
        )
        for dataset in ("sift1m", "glove-200-angular")
    }
    check(
        "dataset-specific E6 navigation K",
        k_values == {"sift1m": [50, 100, 200], "glove-200-angular": [24, 48, 96]},
        k_values,
    )

    restored = json.loads((c6 / "logs/e5-glove-worker-affinity-restored.json").read_text(encoding="utf-8"))
    check(
        "worker affinities restored",
        set(restored["applied"].values()) == {"0-19"},
        restored["applied"],
    )
    results_text = (c6 / "RESULTS.md").read_text(encoding="utf-8")
    status_text = (c6 / "STATUS.md").read_text(encoding="utf-8")
    check("final verdict appended", "Stages 9-10 and final C6 verdict" in results_text, "RESULTS.md")
    check(
        "status records completion",
        "Current stage: Complete" in status_text,
        "STATUS.md",
    )

    record = {
        "record_type": "c6_completion_audit",
        "timestamp": utc_timestamp(),
        "passed": all(item["passed"] for item in checks),
        "checks": checks,
    }
    write_json_atomic(args.output, record)
    print(json.dumps({"passed": record["passed"], "check_count": len(checks)}, sort_keys=True))
    return 0 if record["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
