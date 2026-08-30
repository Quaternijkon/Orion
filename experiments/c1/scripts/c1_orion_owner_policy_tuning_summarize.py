#!/usr/bin/env python3
"""Freeze the nine-candidate owner-policy tuning ranking before held-out use."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CANDIDATES = {
    "c-cnbr-current-all": "C_CNBR+current_all_max",
    "c-cnbr-single-rank": "C_CNBR+single_rank",
    "c-cnbr-bmr10": "C_CNBR+BMR_10",
    "historical-current-all": "HISTORICAL_PRIMARY+current_all_max",
    "historical-single-rank": "HISTORICAL_PRIMARY+single_rank",
    "historical-bmr10": "HISTORICAL_PRIMARY+BMR_10",
    "ccnb-current-all": "CCNB+current_all_max",
    "ccnb-single-rank": "CCNB+single_rank",
    "ccnb-bmr10": "CCNB+BMR_10",
}


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screens-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    screens_dir = args.screens_dir.expanduser().resolve(strict=True)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    observed = {path.name for path in screens_dir.iterdir() if path.is_dir()}
    if observed != set(CANDIDATES):
        raise RuntimeError(
            f"candidate directory mismatch: observed={sorted(observed)}, "
            f"expected={sorted(CANDIDATES)}"
        )

    rows: list[dict[str, Any]] = []
    for slug, expected_combination in CANDIDATES.items():
        screen_path = screens_dir / slug / "screen.json"
        completion_path = screens_dir / slug / "completion-audit.json"
        screen = load_object(screen_path)
        completion = load_object(completion_path)
        protocol = screen.get("protocol") or {}
        qps = screen.get("qps") or {}
        checks = screen.get("checks") or {}
        source_transition = screen.get("repository_measurement_transition") or {}
        if screen.get("status") != "PASS" or completion.get("status") != "PASS":
            raise RuntimeError(f"screen did not pass: {expected_combination}")
        if screen.get("combination") != expected_combination:
            raise RuntimeError(f"combination mismatch: {slug}")
        recall = float(screen["tuning_recall_at_10"])
        cv = float(qps["cv"])
        if not (0.90 <= recall < 0.93 and screen.get("recall_band_pass") is True):
            raise RuntimeError(f"recall band failed: {expected_combination}={recall}")
        if not (cv <= 0.05 and all(checks.values())):
            raise RuntimeError(f"screen audit failed: {expected_combination}")
        if protocol.get("query_range") != [0, 1000] or protocol.get(
            "heldout_queries_issued"
        ) is not False:
            raise RuntimeError(f"held-out isolation failed: {expected_combination}")
        if source_transition.get("status") != "PASS":
            raise RuntimeError(f"source transition failed: {expected_combination}")
        rows.append(
            {
                "combination": expected_combination,
                "collection": screen["collection"],
                "tuning_recall_at_10": recall,
                "selected_concurrency": int(screen["selected_concurrency"]),
                "qps_mean": float(qps["mean"]),
                "qps_stdev": float(qps["stdev"]),
                "qps_cv": cv,
                "physical_point_count": int(
                    (screen.get("assignment") or {})["physical_point_count"]
                ),
                "expansion_ratio": float(
                    (screen.get("assignment") or {})["expansion_ratio"]
                ),
                "screen_path": str(screen_path),
                "screen_sha256": sha256_path(screen_path),
                "completion_audit_sha256": sha256_path(completion_path),
                "measurement_source_path_count": int(
                    source_transition["source_path_count"]
                ),
            }
        )

    rows.sort(key=lambda row: (-row["qps_mean"], row["combination"]))
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    source_counts = {row["measurement_source_path_count"] for row in rows}
    if len(source_counts) != 1:
        raise RuntimeError(f"measurement source counts differ: {source_counts}")

    created_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    ranking = {
        "created_at": created_at,
        "record_type": "c1_orion_owner_policy_tuning_ranking",
        "status": "PASS",
        "protocol": {
            "dataset": "GloVe-200-angular",
            "distance": "Cosine",
            "top_k": 10,
            "physical_hosts": 4,
            "logical_shards": 32,
            "server_cpu_total": 64,
            "recall_band": [0.90, 0.93],
            "query_range": [0, 1000],
            "heldout_queries_issued": False,
            "ranking_metric": "qps_mean",
        },
        "candidate_count": len(rows),
        "rows": rows,
    }
    ranking_path = output_dir / "tuning-ranking.json"
    write_json(ranking_path, ranking)
    with (output_dir / "tuning-ranking.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    finalists = {
        "created_at": created_at,
        "record_type": "c1_orion_owner_policy_frozen_finalists",
        "status": "FROZEN_BEFORE_HELDOUT",
        "selection_rule": (
            "top two tuning mean QPS among nine PASS candidates with "
            "Recall@10 in [0.90,0.93) and QPS CV <= 5%"
        ),
        "heldout_queries_issued_before_freeze": False,
        "ranking_sha256": sha256_path(ranking_path),
        "finalists": rows[:2],
    }
    finalists_path = output_dir / "frozen-finalists.json"
    write_json(finalists_path, finalists)
    completion = {
        "created_at": created_at,
        "status": "PASS",
        "checks": {
            "nine_candidates": len(rows) == 9,
            "all_screens_pass": True,
            "all_recall_in_band": True,
            "all_cv_at_most_5pct": True,
            "all_tuning_only": True,
            "common_measurement_source_path_count": len(source_counts) == 1,
            "two_finalists_frozen": len(finalists["finalists"]) == 2,
        },
        "artifacts": {
            "tuning-ranking.json": sha256_path(ranking_path),
            "tuning-ranking.csv": sha256_path(output_dir / "tuning-ranking.csv"),
            "frozen-finalists.json": sha256_path(finalists_path),
        },
    }
    write_json(output_dir / "completion-audit.json", completion)
    print(json.dumps({"output_dir": str(output_dir), "finalists": rows[:2]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
