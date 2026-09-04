#!/usr/bin/env python3
"""Explore globally budgeted, navigation-importance-ranked replication."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

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
    POLICIES,
    _copy_histogram,
    _gate_results,
    _load_json,
    _load_metrics,
    build_vote_evidence,
    membership_for_policy,
)


BUDGETS = (0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10)
SCORE_NAMES = (
    "secondary_borda",
    "secondary_reciprocal_rank",
    "secondary_proxy_mass_sum",
    "secondary_proxy_mass_borda",
    "secondary_proxy_mass_reciprocal_rank",
    "secondary_proxy_mass_max",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--phase-b-screen", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--query-count", type=int, default=0)
    return parser.parse_args()


def _score_secondary(
    *,
    hit_owner: np.ndarray,
    attachment_local: np.ndarray,
    secondary: np.ndarray,
    proxy_mass: np.ndarray,
) -> dict[str, np.ndarray]:
    rows, width = hit_owner.shape
    row_ids = np.arange(rows, dtype=np.int64)
    safe_secondary = np.maximum(secondary, 0)
    mask = hit_owner == safe_secondary[:, None]
    ranks = np.arange(width, dtype=np.float64)[None, :]
    borda_weights = width - ranks
    reciprocal_weights = 1.0 / (ranks + 1.0)
    hit_mass = proxy_mass[attachment_local].astype(np.float64, copy=False)
    valid = secondary >= 0

    scores = {
        "secondary_borda": np.sum(mask * borda_weights, axis=1),
        "secondary_reciprocal_rank": np.sum(mask * reciprocal_weights, axis=1),
        "secondary_proxy_mass_sum": np.sum(mask * hit_mass, axis=1),
        "secondary_proxy_mass_borda": np.sum(
            mask * hit_mass * borda_weights, axis=1
        ),
        "secondary_proxy_mass_reciprocal_rank": np.sum(
            mask * hit_mass * reciprocal_weights, axis=1
        ),
        "secondary_proxy_mass_max": np.max(
            np.where(mask, hit_mass, 0.0), axis=1
        ),
    }
    for values in scores.values():
        values[~valid] = -np.inf
    return scores


def _budget_membership(
    *,
    primary: np.ndarray,
    secondary: np.ndarray,
    score: np.ndarray,
    budget: float,
    num_partitions: int,
) -> tuple[np.ndarray, int]:
    rows = len(primary)
    row_ids = np.arange(rows, dtype=np.int64)
    membership = np.zeros((rows, num_partitions), dtype=bool)
    membership[row_ids, primary] = True
    eligible = np.flatnonzero(secondary >= 0)
    requested = int(np.floor(float(budget) * rows))
    keep_count = min(requested, len(eligible))
    if keep_count:
        order = np.lexsort((eligible, -score[eligible]))
        kept = eligible[order[:keep_count]]
        membership[kept, secondary[kept]] = True
    return membership, keep_count


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
    mass_record = phase_a["mass"]["values"]
    mass_path = (phase_a_path.parent / mass_record["path"]).resolve()
    if sha256_path(mass_path) != mass_record["sha256"]:
        raise ValueError("upper proxy-mass checksum drifted")
    proxy_mass = checked_binary_matrix(
        mass_path, rows=int(mass_record["row_count"]), width=1, dtype="<u8"
    ).reshape(-1)

    inputs = phase_b["post_freeze_evaluation_inputs"]
    artifact, _adj, _vectors, labels, _entry, _left, _right, navigator_sha = load_upper(
        artifact_path
    )
    num_partitions = int(artifact["shard_count"])
    attachment_manifest = _load_json(Path(inputs["attachments_manifest"]))
    row_count = int(attachment_manifest["row_count"])
    attachment_hits = checked_binary_matrix(
        Path(inputs["attachments"]),
        rows=row_count,
        width=int(attachment_manifest["top_k"]),
        dtype="<u8",
    )
    attachment_local = local_hits(attachment_hits, labels, row_count)
    hit_owner = owner[attachment_local]
    votes, maximum, first_rank, rank_top, id_top = build_vote_evidence(
        hit_owner, num_partitions
    )
    fallback = maximum < 2
    primary = rank_top[:, 0].astype(np.int32, copy=True)
    primary[fallback] = hit_owner[fallback, 0]
    secondary = rank_top[:, 1].astype(np.int32, copy=False)
    scores = _score_secondary(
        hit_owner=hit_owner,
        attachment_local=attachment_local,
        secondary=secondary,
        proxy_mass=proxy_mass,
    )

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
    current = membership_for_policy(
        hit_owner=hit_owner,
        votes=votes,
        maximum=maximum,
        first_rank=first_rank,
        rank_top=rank_top,
        id_top=id_top,
        policy=current_policy,
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
    rows = [baseline]
    for score_name in SCORE_NAMES:
        for budget in BUDGETS:
            membership, kept = _budget_membership(
                primary=primary,
                secondary=secondary,
                score=scores[score_name],
                budget=budget,
                num_partitions=num_partitions,
            )
            row = metrics(f"budget_{budget:.2f}_{score_name}", membership)
            row.update(
                {
                    "budget": budget,
                    "score": score_name,
                    "eligible_secondary_count": int(np.count_nonzero(secondary >= 0)),
                    "kept_secondary_count": kept,
                }
            )
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
        "record_type": "fixed_cnbr_budgeted_multi_assignment_exploration",
        "status": "EXPLORATORY_NOT_ADOPTION_EVIDENCE",
        "dataset": args.dataset,
        "query_count": query_count,
        "query_scope": "full" if query_count == query_total else "prefix_tuning_only",
        "owner_sha256": owner_record["sha256"],
        "proxy_mass_sha256": mass_record["sha256"],
        "upper_artifact_sha256": artifact_record["sha256"],
        "navigator_sha256": navigator_sha,
        "budgets": BUDGETS,
        "scores": SCORE_NAMES,
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
    for row in passing:
        print(
            f"PASS {row['policy']} expansion={row['expansion_ratio']:.6f} "
            f"coverage={row['gt_routing_coverage_mean']:.6f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
