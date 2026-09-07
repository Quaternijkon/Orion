#!/usr/bin/env python3
"""Combine dataset-specific C6 physical summaries into the final table."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from c6_protocol import write_csv_atomic  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    rows: list[dict[str, Any]] = []
    seen = set()
    for path in args.input:
        with Path(path).open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                key = (row["dataset"], int(row["logical_shards"]), row["policy"])
                if key in seen:
                    raise ValueError(f"duplicate physical row {key}")
                seen.add(key)
                rows.append(dict(row))
    rows.sort(key=lambda row: (row["dataset"], int(row["logical_shards"]), row["policy"]))
    write_csv_atomic(args.output, rows)
    print(f"combined {len(rows)} physical rows into {Path(args.output).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
