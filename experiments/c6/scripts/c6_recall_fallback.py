#!/usr/bin/env python3
"""Preorder recall-safe measurement fallbacks using tuning evidence only."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from c6_protocol import (  # noqa: E402
    TARGET_RECALL,
    sha256_path,
    utc_timestamp,
    write_json_atomic,
)


POLICIES = ("P0", "P1", "P2", "P3")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tuning-selection", required=True)
    parser.add_argument("--tuning-csv", required=True)
    parser.add_argument("--policy", choices=POLICIES, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--fallback-rank", type=int)
    return parser.parse_args(argv)


def load_tuning_candidates(path: str | Path, policy: str) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("policy") != policy:
                continue
            decoded = dict(row)
            decoded["config"] = json.loads(decoded["config"])
            for key in (
                "recall_at_10",
                "aggregate_distance_computations_per_query",
                "p99_max_shard_distance_computations",
            ):
                decoded[key] = float(decoded[key])
            candidates.append(decoded)
    if not candidates:
        raise ValueError(f"tuning CSV has no candidates for {policy}")
    return candidates


def preorder_candidates(
    candidates: Sequence[Mapping[str, Any]],
    *,
    selected_config_id: str,
    target_recall: float = TARGET_RECALL,
) -> list[dict[str, Any]]:
    feasible = [
        dict(row)
        for row in candidates
        if float(row["recall_at_10"]) >= target_recall
    ]
    by_id = {str(row["config_id"]): row for row in feasible}
    if len(by_id) != len(feasible):
        raise ValueError("tuning candidates contain duplicate config IDs")
    try:
        primary = by_id.pop(selected_config_id)
    except KeyError as exc:
        raise ValueError(
            "selected tuning config is absent or below the recall target"
        ) from exc
    remainder = sorted(
        by_id.values(),
        key=lambda row: (
            float(row["aggregate_distance_computations_per_query"]),
            float(row["p99_max_shard_distance_computations"]),
            str(row["config_id"]),
        ),
    )
    return [primary, *remainder]


def build_fallback_selection(
    tuning_selection: Mapping[str, Any],
    *,
    policy: str,
    candidate: Mapping[str, Any],
    fallback_rank: int,
    order_sha256: str | None = None,
) -> dict[str, Any]:
    selected = tuning_selection.get("selected")
    if not isinstance(selected, dict) or any(name not in selected for name in POLICIES):
        raise ValueError("tuning selection must contain P0-P3")
    result = json.loads(json.dumps(tuning_selection))
    result["record_type"] = "c6_tuning_fallback_selection"
    result["timestamp"] = utc_timestamp()
    result["fallback"] = {
        "policy": policy,
        "rank": fallback_rank,
        "selection_basis": "preordered_tuning_only",
        "measurement_metrics_consulted_for_ordering": False,
        "order_sha256": order_sha256,
    }
    result["selected"][policy] = {
        "config_id": candidate["config_id"],
        "config": candidate["config"],
        "summary": {
            key: value
            for key, value in candidate.items()
            if key not in {"policy", "config_id", "config"}
        },
    }
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.run_id or Path(args.run_id).name != args.run_id:
        raise ValueError("run-id must be one safe path component")
    if args.fallback_rank is not None and args.fallback_rank < 1:
        raise ValueError("fallback-rank must be at least 1; rank 0 is the primary")

    selection_path = Path(args.tuning_selection).expanduser().resolve()
    csv_path = Path(args.tuning_csv).expanduser().resolve()
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selected = selection.get("selected")
    if not isinstance(selected, dict) or not isinstance(selected.get(args.policy), dict):
        raise ValueError(f"tuning selection is missing {args.policy}")
    selected_config_id = str(selected[args.policy].get("config_id", ""))
    ordered = preorder_candidates(
        load_tuning_candidates(csv_path, args.policy),
        selected_config_id=selected_config_id,
    )

    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    order_path = output / f"{args.run_id}-fallback-order.json"
    order_record = {
        "record_type": "c6_tuning_fallback_order",
        "format_version": 1,
        "timestamp": utc_timestamp(),
        "run_id": args.run_id,
        "dataset": selection.get("dataset"),
        "logical_shards": selection.get("logical_shards"),
        "policy": args.policy,
        "target_recall": TARGET_RECALL,
        "ordering_rule": [
            "frozen_primary_first",
            "aggregate_distance_computations_per_query_ascending",
            "p99_max_shard_distance_computations_ascending",
            "config_id_ascending",
        ],
        "measurement_metrics_consulted_for_ordering": False,
        "sources": {
            "tuning_selection": {
                "path": str(selection_path),
                "sha256": sha256_path(selection_path),
            },
            "tuning_csv": {
                "path": str(csv_path),
                "sha256": sha256_path(csv_path),
            },
        },
        "candidates": [
            {"rank": rank, **candidate}
            for rank, candidate in enumerate(ordered)
        ],
    }
    write_json_atomic(order_path, order_record)

    result: dict[str, Any] = {"order_path": str(order_path), "candidate_count": len(ordered)}
    if args.fallback_rank is not None:
        if args.fallback_rank >= len(ordered):
            raise ValueError(
                f"fallback-rank {args.fallback_rank} exceeds {len(ordered) - 1}"
            )
        selection_record = build_fallback_selection(
            selection,
            policy=args.policy,
            candidate=ordered[args.fallback_rank],
            fallback_rank=args.fallback_rank,
            order_sha256=sha256_path(order_path),
        )
        fallback_path = output / f"{args.run_id}-fallback-{args.fallback_rank}-selection.json"
        write_json_atomic(fallback_path, selection_record)
        result["fallback_selection_path"] = str(fallback_path)
        result["fallback_config_id"] = ordered[args.fallback_rank]["config_id"]
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
