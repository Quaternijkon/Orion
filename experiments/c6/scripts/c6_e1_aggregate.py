#!/usr/bin/env python3
"""Aggregate per-M C6 E1 summaries into the required query-difficulty table."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from c6_protocol import write_csv_atomic  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", action="append", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def row_from_summary(path: str | Path) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    prefix = payload["oracle_prefix_shards"]
    fraction = payload["oracle_prefix_fraction"]
    total = payload["oracle_total_work"]
    maximum = payload["oracle_max_shard_work"]
    mean_ef = payload["oracle_mean_local_ef"]
    max_ef = payload["oracle_max_local_ef"]
    return {
        "dataset": payload["dataset"],
        "logical_shards": payload["logical_shards"],
        "query_count": payload["query_count"],
        "oracle_prefix_mean": prefix["mean"],
        "oracle_prefix_median": prefix["median"],
        "oracle_prefix_p75": prefix["p75"],
        "oracle_prefix_p90": prefix["p90"],
        "oracle_prefix_p95": prefix["p95"],
        "oracle_prefix_p99": prefix["p99"],
        "oracle_prefix_max": prefix["maximum"],
        "oracle_fraction_mean": fraction["mean"],
        "oracle_total_work_mean": total["mean"],
        "oracle_total_work_p95": total["p95"],
        "oracle_max_shard_work_mean": maximum["mean"],
        "oracle_max_shard_work_p95": maximum["p95"],
        "oracle_mean_local_ef_median": mean_ef["median"],
        "oracle_mean_local_ef_p95": mean_ef["p95"],
        "oracle_max_local_ef_median": max_ef["median"],
        "oracle_max_local_ef_p95": max_ef["p95"],
        "target_reached_fraction": payload["target_reached_fraction"],
        "summary_path": str(Path(path).resolve()),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    rows = sorted(
        (row_from_summary(path) for path in args.summary),
        key=lambda row: (str(row["dataset"]), int(row["logical_shards"])),
    )
    write_csv_atomic(args.output, rows)
    print(json.dumps({"output": str(Path(args.output).resolve()), "rows": len(rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
