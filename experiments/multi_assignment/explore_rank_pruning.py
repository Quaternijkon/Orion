#!/usr/bin/env python3
"""Explore evidence-strength pruning after the frozen cap-2 rank policy.

This is explicitly exploratory on SIFT and the 1,000-query GloVe tuning split.
It never reads the disjoint 9,000-query online measurement split separately and
never changes the frozen C_CNBR owner.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any, Callable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.l1_balance.run_offline_screen import (  # noqa: E402
    checked_binary_matrix,
    load_upper,
    local_hits,
    query_metrics,
    sha256_path,
)
from experiments.multi_assignment.screen_fixed_owner import (  # noqa: E402
    MAX_COVERAGE_DROP,
    MAX_FULL_COVERAGE_DROP,
    MAX_ROUTE_WORK_RATIO,
    MIN_EXCESS_REDUCTION,
    _copy_histogram,
    _gate_results,
    _load_json,
    _load_metrics,
    build_vote_evidence,
    membership_for_policy,
    POLICIES,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--phase-b-screen", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--query-count", type=int, default=0)
    return parser.parse_args()


def _selected(values: np.ndarray, shards: np.ndarray) -> np.ndarray:
    rows = np.arange(len(shards), dtype=np.int64)
    safe = np.maximum(shards, 0)
    result = values[rows, safe]
    return np.where(shards >= 0, result, 0)


def _support_evidence(
    hit_owner: np.ndarray,
    num_partitions: int,
    rank_top: np.ndarray,
) -> dict[str, np.ndarray]:
    rows, width = hit_owner.shape
    row_ids = np.arange(rows, dtype=np.int64)
    borda = np.zeros((rows, num_partitions), dtype=np.uint16)
    reciprocal = np.zeros((rows, num_partitions), dtype=np.float32)
    rank_sum = np.zeros((rows, num_partitions), dtype=np.uint16)
    last_rank = np.zeros((rows, num_partitions), dtype=np.uint8)
    second_rank = np.full((rows, num_partitions), width, dtype=np.uint8)
    seen = np.zeros((rows, num_partitions), dtype=np.uint8)
    for rank in range(width):
        shard = hit_owner[:, rank]
        borda[row_ids, shard] += width - rank
        reciprocal[row_ids, shard] += 1.0 / float(rank + 1)
        rank_sum[row_ids, shard] += rank
        last_rank[row_ids, shard] = rank
        already = seen[row_ids, shard]
        is_second = already == 1
        second_rank[row_ids[is_second], shard[is_second]] = rank
        seen[row_ids, shard] += 1

    primary = rank_top[:, 0].astype(np.int32, copy=False)
    secondary = rank_top[:, 1].astype(np.int32, copy=False)
    return {
        "primary": primary,
        "secondary": secondary,
        "primary_borda": _selected(borda, primary),
        "secondary_borda": _selected(borda, secondary),
        "primary_reciprocal": _selected(reciprocal, primary),
        "secondary_reciprocal": _selected(reciprocal, secondary),
        "primary_rank_sum": _selected(rank_sum, primary),
        "secondary_rank_sum": _selected(rank_sum, secondary),
        "primary_last_rank": _selected(last_rank, primary),
        "secondary_last_rank": _selected(last_rank, secondary),
        "primary_second_rank": _selected(second_rank, primary),
        "secondary_second_rank": _selected(second_rank, secondary),
    }


def _policy_family() -> list[tuple[str, Callable[[dict[str, np.ndarray]], np.ndarray]]]:
    family: list[tuple[str, Callable[[dict[str, np.ndarray]], np.ndarray]]] = []
    for percent in range(50, 100, 5):
        family.append(
            (
                f"two_tie_borda_ratio_{percent}",
                lambda e, p=percent: e["secondary_borda"] * 100
                >= e["primary_borda"] * p,
            )
        )
        family.append(
            (
                f"two_tie_rr_ratio_{percent}",
                lambda e, p=percent: e["secondary_reciprocal"] * 100.0
                >= e["primary_reciprocal"] * float(p),
            )
        )
    for gap in range(1, 9):
        family.append(
            (
                f"two_tie_first_gap_le_{gap}",
                lambda e, g=gap: e["secondary_first_rank"]
                - e["primary_first_rank"]
                <= g,
            )
        )
    for slack in range(0, 6):
        family.append(
            (
                f"two_tie_secondary_before_primary_second_plus_{slack}",
                lambda e, s=slack: e["secondary_first_rank"]
                <= e["primary_second_rank"] + s,
            )
        )
        family.append(
            (
                f"two_tie_secondary_before_primary_last_plus_{slack}",
                lambda e, s=slack: e["secondary_first_rank"]
                <= e["primary_last_rank"] + s,
            )
        )
    return family


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output_dir}")

    phase_b_path = Path(args.phase_b_screen).expanduser().resolve()
    phase_b = _load_json(phase_b_path)
    phase_a_path = Path(phase_b["phase_a_manifest"]).expanduser().resolve()
    if sha256_path(phase_a_path) != phase_b["phase_a_manifest_sha256"]:
        raise ValueError("Phase-A checksum drifted")
    phase_a = _load_json(phase_a_path)
    construction = phase_a["construction_inputs"]
    artifact_record = construction["artifact"]
    artifact_path = Path(artifact_record["path"]).expanduser().resolve()
    if sha256_path(artifact_path) != artifact_record["sha256"]:
        raise ValueError("upper artifact checksum drifted")
    owner_record = phase_a["owners"]["C_CNBR"]["owner"]
    owner_path = (phase_a_path.parent / owner_record["path"]).resolve()
    if sha256_path(owner_path) != owner_record["sha256"]:
        raise ValueError("C_CNBR owner checksum drifted")
    owner = checked_binary_matrix(
        owner_path, rows=int(owner_record["row_count"]), width=1, dtype="<i4"
    ).reshape(-1)

    inputs = phase_b["post_freeze_evaluation_inputs"]
    artifact, _adj, _vectors, labels, _entry, _left, _right, navigator_sha = load_upper(
        artifact_path
    )
    num_partitions = int(artifact["shard_count"])
    attachment_manifest = _load_json(Path(inputs["attachments_manifest"]))
    row_count = int(attachment_manifest["row_count"])
    attachment_k = int(attachment_manifest["top_k"])
    attachment_hits = checked_binary_matrix(
        Path(inputs["attachments"]), rows=row_count, width=attachment_k, dtype="<u8"
    )
    attachment_local = local_hits(attachment_hits, labels, row_count)
    hit_owner = owner[attachment_local]
    votes, maximum, first_rank, rank_top, id_top = build_vote_evidence(
        hit_owner, num_partitions
    )
    tie_count = ((votes == maximum[:, None]) & (maximum[:, None] >= 2)).sum(
        axis=1, dtype=np.int16
    )
    evidence = _support_evidence(hit_owner, num_partitions, rank_top)
    evidence["primary_first_rank"] = _selected(first_rank, evidence["primary"])
    evidence["secondary_first_rank"] = _selected(first_rank, evidence["secondary"])

    query_manifest = _load_json(Path(inputs["query_hits_manifest"]))
    query_total = int(query_manifest["row_count"])
    query_count = int(args.query_count) if int(args.query_count) > 0 else query_total
    if query_count <= 0 or query_count > query_total:
        raise ValueError("query-count is outside the bound query corpus")
    query_hits = checked_binary_matrix(
        Path(inputs["query_hits"]),
        rows=query_total,
        width=int(query_manifest["top_k"]),
        dtype="<u8",
    )
    query_local = local_hits(query_hits[:query_count], labels, query_count)
    gt_manifest = _load_json(Path(inputs["ground_truth_manifest"]))
    ground_truth = checked_binary_matrix(
        Path(inputs["ground_truth"]),
        rows=query_total,
        width=int(gt_manifest["width"]),
        dtype="<u4",
    )[:query_count]

    current_policy = next(p for p in POLICIES if p.name == "current_all_max")
    cap2_policy = next(p for p in POLICIES if p.name == "cap2_rank")
    current = membership_for_policy(
        hit_owner=hit_owner,
        votes=votes,
        maximum=maximum,
        first_rank=first_rank,
        rank_top=rank_top,
        id_top=id_top,
        policy=current_policy,
    )
    cap2 = membership_for_policy(
        hit_owner=hit_owner,
        votes=votes,
        maximum=maximum,
        first_rank=first_rank,
        rank_top=rank_top,
        id_top=id_top,
        policy=cap2_policy,
    )

    def metrics(name: str, membership: np.ndarray) -> dict[str, Any]:
        copy_count = membership.sum(axis=1, dtype=np.int16)
        loads = membership.sum(axis=0, dtype=np.int64)
        row: dict[str, Any] = {
            "policy": name,
            "logical_point_count": row_count,
            "physical_point_count": int(np.sum(copy_count)),
            "expansion_ratio": float(np.mean(copy_count)),
            "extra_copy_fraction": float(np.mean(copy_count) - 1.0),
            "copy_count_histogram": _copy_histogram(copy_count),
            **_load_metrics(loads),
        }
        row.update(
            query_metrics(
                owner,
                query_local,
                membership[labels],
                membership,
                ground_truth,
                int(artifact["dynamic_ef_base"]),
                int(artifact["dynamic_ef_factor"]),
            )
        )
        return row

    baseline = metrics("current_all_max", current)
    cap2_row = metrics("cap2_rank", cap2)
    rows = [baseline, cap2_row]
    row_ids = np.arange(row_count, dtype=np.int64)
    secondary = evidence["secondary"].astype(np.int32, copy=False)
    has_secondary = secondary >= 0
    two_tie = tie_count == 2
    for name, predicate in _policy_family():
        keep = np.asarray(predicate(evidence), dtype=bool)
        if keep.shape != (row_count,):
            raise AssertionError(f"policy {name} returned the wrong shape")
        membership = cap2.copy()
        remove = has_secondary & two_tie & ~keep
        membership[row_ids[remove], secondary[remove]] = False
        row = metrics(name, membership)
        row["removed_two_tie_secondary_count"] = int(np.count_nonzero(remove))
        rows.append(row)

    for row in rows:
        if row["policy"] == "current_all_max":
            row["gate_results"] = {}
            row["all_offline_gates_pass"] = True
            row["excess_expansion_reduction_fraction"] = 0.0
        else:
            row["gate_results"] = _gate_results(row, baseline, query_count)
            row["all_offline_gates_pass"] = all(
                bool(gate["pass"]) for gate in row["gate_results"].values()
            )
            base_excess = float(baseline["expansion_ratio"]) - 1.0
            row["excess_expansion_reduction_fraction"] = 1.0 - (
                float(row["expansion_ratio"]) - 1.0
            ) / base_excess

    output_dir.mkdir(parents=True)
    result = {
        "format_version": 1,
        "record_type": "fixed_cnbr_multi_assignment_rank_pruning_exploration",
        "status": "EXPLORATORY_NOT_ADOPTION_EVIDENCE",
        "dataset": args.dataset,
        "query_count": query_count,
        "query_scope": "full" if query_count == query_total else "prefix_tuning_only",
        "owner_sha256": owner_record["sha256"],
        "upper_artifact_sha256": artifact_record["sha256"],
        "navigator_sha256": navigator_sha,
        "gate_contract": {
            "minimum_excess_expansion_reduction": MIN_EXCESS_REDUCTION,
            "maximum_gt_coverage_drop": MAX_COVERAGE_DROP,
            "maximum_full_coverage_drop": MAX_FULL_COVERAGE_DROP,
            "maximum_route_work_ratio": MAX_ROUTE_WORK_RATIO,
        },
        "rows": rows,
    }
    (output_dir / "exploration-manifest.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    fields = sorted(
        key for key in rows[0] if not isinstance(rows[0][key], (dict, list))
    )
    with (output_dir / "summary.csv").open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fields} for row in rows)
    passing = sorted(
        (row for row in rows[1:] if row["all_offline_gates_pass"]),
        key=lambda row: (float(row["expansion_ratio"]), row["policy"]),
    )
    for row in passing[:12]:
        print(
            f"PASS {row['policy']} expansion={row['expansion_ratio']:.6f} "
            f"coverage={row['gt_routing_coverage_mean']:.6f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
