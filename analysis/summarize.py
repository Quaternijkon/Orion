"""Tabulate fan-out headroom reports produced by fanout_headroom.py."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", nargs="+", required=True)
    parser.add_argument("--target", default="0.95")
    args = parser.parse_args()

    rows = []
    for path in args.reports:
        report = json.loads(Path(path).read_text(encoding="utf-8"))
        oracle = report["orderings"]["oracle"]["targets"][args.target]
        centroid = report["orderings"]["centroid"]["targets"][args.target]
        rows.append(
            {
                "dataset": report["dataset"].split("-")[0],
                "method": report["layout_method"],
                "shards": report["shards"],
                "fanout_mean": report["required_fanout"]["mean"],
                "fanout_p95": report["required_fanout"]["p95"],
                "balance": report["layout_balance"]["max_over_mean"],
                "expansion": report.get("expansion_ratio", 1.0),
                "oracle_fixed": oracle["fixed_budget"],
                "oracle_adaptive": oracle["adaptive_mean_shards"],
                "centroid_fixed": centroid["fixed_budget"],
                "centroid_adaptive": centroid["adaptive_mean_shards"],
                "h_ordering": report["headroom_ordering"][args.target],
                "h_adaptive_centroid": centroid["headroom_adaptive"],
                "h_total": centroid["fixed_budget"] / oracle["adaptive_mean_shards"],
            }
        )
    rows.sort(key=lambda row: (row["dataset"], row["shards"], row["method"]))

    print(f"Recall target R = {args.target}, budgets in shards searched per query\n")
    header = (
        f"{'dataset':6} {'P':>3} {'layout':18} "
        f"{'fanout':>7} {'p95':>4} {'bal':>5} {'exp':>5} | "
        f"{'orc.fix':>7} {'orc.ad':>7} {'cen.fix':>7} {'cen.ad':>7} | "
        f"{'H_ord':>6} {'H_adp':>6} {'H_tot':>6}"
    )
    print(header)
    print("-" * len(header))
    previous = None
    for row in rows:
        if previous is not None and (row["dataset"], row["shards"]) != previous:
            print()
        previous = (row["dataset"], row["shards"])
        print(
            f"{row['dataset']:6} {row['shards']:3d} {row['method']:18} "
            f"{row['fanout_mean']:7.2f} {row['fanout_p95']:4.0f} "
            f"{row['balance']:5.2f} {row['expansion']:5.3f} | "
            f"{row['oracle_fixed']:7d} {row['oracle_adaptive']:7.2f} "
            f"{row['centroid_fixed']:7d} {row['centroid_adaptive']:7.2f} | "
            f"{row['h_ordering']:5.2f}x {row['h_adaptive_centroid']:5.2f}x "
            f"{row['h_total']:5.2f}x"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
