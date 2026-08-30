#!/usr/bin/env python3
"""Screen L0 multi-assignment rules while freezing the C_CNBR L1 owner."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, NamedTuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.l1_balance.run_offline_screen import (
    checked_binary_matrix,
    distribution,
    load_upper,
    local_hits,
    query_metrics,
    sha256_path,
)


class Policy(NamedTuple):
    name: str
    cap: int
    order: str
    second_prefix: int | None = None


POLICIES: tuple[Policy, ...] = (
    Policy("current_all_max", 0, "all"),
    Policy("single_rank", 1, "rank"),
    Policy("cap2_rank", 2, "rank"),
    Policy("cap3_rank", 3, "rank"),
    Policy("cap2_rank_prefix4", 2, "rank", 4),
    Policy("cap2_rank_prefix6", 2, "rank", 6),
    Policy("cap2_rank_prefix8", 2, "rank", 8),
    Policy("cap2_shard_id", 2, "shard_id"),
)

MIN_EXCESS_REDUCTION = 0.20
MAX_COVERAGE_DROP = 0.002
MAX_FULL_COVERAGE_DROP = 0.02
MAX_ROUTE_WORK_RATIO = 1.05


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def build_vote_evidence(
    hit_owner: np.ndarray, num_partitions: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return votes, maximum, first rank, rank top-3, and shard-id top-2."""

    hit_owner = np.asarray(hit_owner, dtype=np.int32)
    rows, width = hit_owner.shape
    row_ids = np.arange(rows, dtype=np.int64)
    votes = np.zeros((rows, num_partitions), dtype=np.uint8)
    first_rank = np.full((rows, num_partitions), width, dtype=np.uint8)
    for rank in range(width):
        shard = hit_owner[:, rank]
        np.add.at(votes, (row_ids, shard), 1)
        first_rank[row_ids, shard] = np.minimum(first_rank[row_ids, shard], rank)

    maximum = votes.max(axis=1)
    eligible = (votes == maximum[:, None]) & (maximum[:, None] >= 2)
    shard_ids = np.arange(num_partitions, dtype=np.int16)[None, :]

    rank_score = np.where(
        eligible,
        first_rank.astype(np.int16) * num_partitions + shard_ids,
        np.iinfo(np.int16).max,
    )
    rank_top = np.full((rows, 3), -1, dtype=np.int16)
    for column in range(3):
        chosen = np.argmin(rank_score, axis=1)
        valid = rank_score[row_ids, chosen] != np.iinfo(np.int16).max
        rank_top[valid, column] = chosen[valid].astype(np.int16)
        rank_score[row_ids[valid], chosen[valid]] = np.iinfo(np.int16).max

    id_score = np.where(eligible, shard_ids, np.iinfo(np.int16).max).astype(
        np.int16, copy=False
    )
    id_top = np.full((rows, 2), -1, dtype=np.int16)
    for column in range(2):
        chosen = np.argmin(id_score, axis=1)
        valid = id_score[row_ids, chosen] != np.iinfo(np.int16).max
        id_top[valid, column] = chosen[valid].astype(np.int16)
        id_score[row_ids[valid], chosen[valid]] = np.iinfo(np.int16).max
    return votes, maximum, first_rank, rank_top, id_top


def membership_for_policy(
    *,
    hit_owner: np.ndarray,
    votes: np.ndarray,
    maximum: np.ndarray,
    first_rank: np.ndarray,
    rank_top: np.ndarray,
    id_top: np.ndarray,
    policy: Policy,
) -> np.ndarray:
    """Apply one fixed policy and return a point-by-shard membership matrix."""

    rows, num_partitions = votes.shape
    row_ids = np.arange(rows, dtype=np.int64)
    fallback = maximum < 2
    if policy.order == "all":
        membership = votes == maximum[:, None]
        membership[fallback] = False
        membership[row_ids[fallback], hit_owner[fallback, 0]] = True
        return membership

    membership = np.zeros((rows, num_partitions), dtype=bool)
    chosen = rank_top if policy.order == "rank" else id_top
    primary = chosen[:, 0].astype(np.int32, copy=True)
    primary[fallback] = hit_owner[fallback, 0]
    if np.any(primary < 0):
        raise AssertionError("policy failed to select a primary shard")
    membership[row_ids, primary] = True

    for column in range(1, min(policy.cap, chosen.shape[1])):
        shard = chosen[:, column].astype(np.int32, copy=False)
        include = shard >= 0
        if column == 1 and policy.second_prefix is not None:
            safe_shard = np.maximum(shard, 0)
            include &= first_rank[row_ids, safe_shard] < policy.second_prefix
        membership[row_ids[include], shard[include]] = True
    return membership


def _load_metrics(loads: np.ndarray) -> dict[str, float | int]:
    mean = float(np.mean(loads))
    return {
        "physical_copy_load_min": int(np.min(loads)),
        "physical_copy_load_max": int(np.max(loads)),
        "physical_copy_load_mean": mean,
        "physical_copy_load_cv": float(np.std(loads) / mean),
        "physical_copy_load_max_over_mean": float(np.max(loads) / mean),
        "physical_copy_load_min_over_mean": float(np.min(loads) / mean),
        "physical_copy_load_empty_shards": int(np.count_nonzero(loads == 0)),
    }


def _semantic_sha256(membership: np.ndarray) -> str:
    packed = np.packbits(membership, axis=1, bitorder="little")
    return hashlib.sha256(np.ascontiguousarray(packed).tobytes()).hexdigest()


def _copy_histogram(copy_count: np.ndarray) -> dict[str, int]:
    values, counts = np.unique(copy_count, return_counts=True)
    return {
        str(int(value)): int(count)
        for value, count in zip(values.tolist(), counts.tolist(), strict=True)
    }


def _gate_results(
    observed: dict[str, Any], baseline: dict[str, Any], query_count: int
) -> dict[str, dict[str, Any]]:
    excess_limit = (float(baseline["expansion_ratio"]) - 1.0) * (
        1.0 - MIN_EXCESS_REDUCTION
    )
    zero_limit = min(
        int(baseline["gt_queries_zero_coverage"]) + 2,
        int(np.floor(query_count * 0.001)),
    )
    specs = {
        "excess_expansion": (
            float(observed["expansion_ratio"]) - 1.0,
            "<=",
            excess_limit,
        ),
        "gt_routing_coverage_mean": (
            float(observed["gt_routing_coverage_mean"]),
            ">=",
            float(baseline["gt_routing_coverage_mean"]) - MAX_COVERAGE_DROP,
        ),
        "gt_queries_full_coverage_fraction": (
            float(observed["gt_queries_full_coverage_fraction"]),
            ">=",
            float(baseline["gt_queries_full_coverage_fraction"])
            - MAX_FULL_COVERAGE_DROP,
        ),
        "gt_queries_zero_coverage": (
            int(observed["gt_queries_zero_coverage"]),
            "<=",
            zero_limit,
        ),
        "routed_shards_mean": (
            float(observed["routed_shards_mean"]),
            "<=",
            float(baseline["routed_shards_mean"]) * MAX_ROUTE_WORK_RATIO,
        ),
        "route_ef_sum_mean": (
            float(observed["route_ef_sum_mean"]),
            "<=",
            float(baseline["route_ef_sum_mean"]) * MAX_ROUTE_WORK_RATIO,
        ),
        "physical_copy_load_max": (
            int(observed["physical_copy_load_max"]),
            "<=",
            int(baseline["physical_copy_load_max"]),
        ),
    }
    return {
        name: {
            "observed": observed_value,
            "reference": baseline.get(name),
            "operator": operator,
            "threshold": threshold,
            "pass": observed_value <= threshold
            if operator == "<="
            else observed_value >= threshold,
        }
        for name, (observed_value, operator, threshold) in specs.items()
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--phase-b-screen", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--query-count", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output_dir}")

    phase_b_path = Path(args.phase_b_screen).expanduser().resolve()
    phase_b = _load_json(phase_b_path)
    phase_a_path = Path(phase_b["phase_a_manifest"]).expanduser().resolve()
    if sha256_path(phase_a_path) != phase_b["phase_a_manifest_sha256"]:
        raise ValueError("Phase-A manifest checksum drifted")
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
    for key in (
        "attachments",
        "attachments_manifest",
        "query_hits",
        "query_hits_manifest",
        "ground_truth",
        "ground_truth_manifest",
        "dataset_manifest",
        "source_build_manifest",
    ):
        path = Path(inputs[key]).expanduser().resolve()
        expected = inputs.get(f"{key}_sha256")
        if expected is not None and sha256_path(path) != expected:
            raise ValueError(f"bound input checksum drifted: {key}")

    artifact, _adjacency, _vectors, labels, _entry, _left, _right, navigator_sha = (
        load_upper(artifact_path)
    )
    num_partitions = int(artifact["shard_count"])
    if num_partitions != 32 or np.any(owner < 0) or np.any(owner >= num_partitions):
        raise ValueError("frozen owner/partition contract drifted")
    if len(owner) != len(labels):
        raise ValueError("owner length differs from ordered upper labels")

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
    ground_truth_manifest = _load_json(Path(inputs["ground_truth_manifest"]))
    gt_width = int(ground_truth_manifest["width"])
    ground_truth = checked_binary_matrix(
        Path(inputs["ground_truth"]), rows=query_total, width=gt_width, dtype="<u4"
    )[:query_count]

    rows: list[dict[str, Any]] = []
    memberships: dict[str, np.ndarray] = {}
    for policy in POLICIES:
        membership = membership_for_policy(
            hit_owner=hit_owner,
            votes=votes,
            maximum=maximum,
            first_rank=first_rank,
            rank_top=rank_top,
            id_top=id_top,
            policy=policy,
        )
        copy_count = membership.sum(axis=1, dtype=np.int16)
        loads = membership.sum(axis=0, dtype=np.int64)
        metrics: dict[str, Any] = {
            "policy": policy.name,
            "logical_point_count": row_count,
            "physical_point_count": int(np.sum(copy_count)),
            "expansion_ratio": float(np.mean(copy_count)),
            "extra_copy_fraction": float(np.mean(copy_count) - 1.0),
            "membership_semantic_sha256": _semantic_sha256(membership),
            "copy_count_histogram": _copy_histogram(copy_count),
            **_load_metrics(loads),
        }
        metrics.update(
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
        rows.append(metrics)
        memberships[policy.name] = membership
        print(
            f"{args.dataset} {policy.name} expansion={metrics['expansion_ratio']:.6f} "
            f"coverage={metrics['gt_routing_coverage_mean']:.6f}",
            flush=True,
        )

    baseline = rows[0]
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
            baseline_excess = float(baseline["expansion_ratio"]) - 1.0
            row["excess_expansion_reduction_fraction"] = float(
                1.0 - (float(row["expansion_ratio"]) - 1.0) / baseline_excess
            )

    output_dir.mkdir(parents=True)
    summary = {
        "format_version": 1,
        "record_type": "fixed_cnbr_multi_assignment_screen",
        "dataset": args.dataset,
        "query_count": query_count,
        "query_scope": "full" if query_count == query_total else "prefix_tuning_only",
        "causal_boundary": {
            "load_balance_owner": "C_CNBR",
            "owner_path": str(owner_path),
            "owner_sha256": owner_record["sha256"],
            "upper_artifact": str(artifact_path),
            "upper_artifact_sha256": artifact_record["sha256"],
            "navigator_sha256": navigator_sha,
            "owner_changed_between_policies": False,
            "multi_assignment_only": True,
        },
        "bound_inputs": {
            key: inputs[key]
            for key in sorted(inputs)
            if key.endswith("sha256") or key in {
                "attachments",
                "query_hits",
                "ground_truth",
                "dataset_manifest",
                "source_build_manifest",
            }
        },
        "policy_order": [policy.name for policy in POLICIES],
        "gate_contract": {
            "minimum_excess_expansion_reduction": MIN_EXCESS_REDUCTION,
            "maximum_gt_coverage_drop": MAX_COVERAGE_DROP,
            "maximum_full_coverage_drop": MAX_FULL_COVERAGE_DROP,
            "maximum_route_work_ratio": MAX_ROUTE_WORK_RATIO,
        },
        "rows": rows,
    }
    (output_dir / "screen-manifest.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    scalar_fields = sorted(
        key
        for key in rows[0]
        if not isinstance(rows[0][key], (dict, list))
    )
    with (output_dir / "summary.csv").open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=scalar_fields)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in scalar_fields} for row in rows)


if __name__ == "__main__":
    main()
