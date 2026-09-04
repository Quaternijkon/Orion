#!/usr/bin/env python3
"""Formal offline evaluation of BMR_10 on a frozen C_CNBR owner."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
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
from experiments.multi_assignment import budgeted_policy  # noqa: E402
from experiments.multi_assignment.screen_fixed_owner import (  # noqa: E402
    MAX_COVERAGE_DROP,
    MAX_FULL_COVERAGE_DROP,
    MAX_ROUTE_WORK_RATIO,
    POLICIES,
    _copy_histogram,
    _load_json,
    _load_metrics,
    build_vote_evidence,
    membership_for_policy,
)


MAX_EXPANSION = 1.10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--phase-b-screen", required=True)
    parser.add_argument("--exploration-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--query-count", type=int, default=0)
    return parser.parse_args()


def assignment_jsonl_sha256(membership: np.ndarray) -> str:
    digest = hashlib.sha256()
    for point_id, row in enumerate(membership):
        shards = np.flatnonzero(row)
        digest.update(
            (
                f'{{"id":{point_id},"shards":['
                + ",".join(str(int(shard)) for shard in shards)
                + "]}\n"
            ).encode("ascii")
        )
    return digest.hexdigest()


def metrics(
    *,
    name: str,
    membership: np.ndarray,
    owner: np.ndarray,
    labels: np.ndarray,
    query_local: np.ndarray,
    ground_truth: np.ndarray,
    artifact: dict[str, Any],
) -> dict[str, Any]:
    copy_count = membership.sum(axis=1, dtype=np.int16)
    loads = membership.sum(axis=0, dtype=np.int64)
    packed = np.packbits(membership, axis=1, bitorder="little")
    result: dict[str, Any] = {
        "policy": name,
        "logical_point_count": int(len(membership)),
        "physical_point_count": int(np.sum(copy_count)),
        "expansion_ratio": float(np.mean(copy_count)),
        "extra_copy_fraction": float(np.mean(copy_count) - 1.0),
        "copy_count_histogram": _copy_histogram(copy_count),
        "membership_semantic_sha256": hashlib.sha256(
            np.ascontiguousarray(packed).tobytes()
        ).hexdigest(),
        "assignment_jsonl_sha256": assignment_jsonl_sha256(membership),
        **_load_metrics(loads),
    }
    result.update(
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
    return result


def gate_results(
    candidate: dict[str, Any], baseline: dict[str, Any], query_count: int
) -> dict[str, dict[str, Any]]:
    zero_limit = min(
        int(baseline["gt_queries_zero_coverage"]) + 2,
        int(np.floor(query_count * 0.001)),
    )
    specs = {
        "expansion_ratio": (
            float(candidate["expansion_ratio"]),
            "<=",
            MAX_EXPANSION,
        ),
        "expansion_improves": (
            float(candidate["expansion_ratio"]),
            "<",
            float(baseline["expansion_ratio"]),
        ),
        "gt_routing_coverage_mean": (
            float(candidate["gt_routing_coverage_mean"]),
            ">=",
            float(baseline["gt_routing_coverage_mean"]) - MAX_COVERAGE_DROP,
        ),
        "gt_queries_full_coverage_fraction": (
            float(candidate["gt_queries_full_coverage_fraction"]),
            ">=",
            float(baseline["gt_queries_full_coverage_fraction"])
            - MAX_FULL_COVERAGE_DROP,
        ),
        "gt_queries_zero_coverage": (
            int(candidate["gt_queries_zero_coverage"]),
            "<=",
            zero_limit,
        ),
        "routed_shards_mean": (
            float(candidate["routed_shards_mean"]),
            "<=",
            float(baseline["routed_shards_mean"]) * MAX_ROUTE_WORK_RATIO,
        ),
        "route_ef_sum_mean": (
            float(candidate["route_ef_sum_mean"]),
            "<=",
            float(baseline["route_ef_sum_mean"]) * MAX_ROUTE_WORK_RATIO,
        ),
        "physical_copy_load_max": (
            int(candidate["physical_copy_load_max"]),
            "<=",
            int(baseline["physical_copy_load_max"]),
        ),
    }
    gates = {}
    for name, (observed, operator, threshold) in specs.items():
        if operator == "<=":
            passed = observed <= threshold
        elif operator == "<":
            passed = observed < threshold
        else:
            passed = observed >= threshold
        gates[name] = {
            "observed": observed,
            "reference": baseline.get(name),
            "operator": operator,
            "threshold": threshold,
            "pass": passed,
        }
    return gates


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output_dir}")

    phase_b_path = Path(args.phase_b_screen).expanduser().resolve()
    phase_b_sha256 = sha256_path(phase_b_path)
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
        raise ValueError("proxy-mass checksum drifted")
    proxy_mass = checked_binary_matrix(
        mass_path, rows=int(mass_record["row_count"]), width=1, dtype="<u8"
    ).reshape(-1)

    inputs = phase_b["post_freeze_evaluation_inputs"]
    for key in (
        "attachments",
        "attachments_manifest",
        "query_hits",
        "query_hits_manifest",
        "ground_truth",
        "ground_truth_manifest",
    ):
        expected = inputs.get(f"{key}_sha256")
        path = Path(inputs[key]).expanduser().resolve()
        if expected is not None and sha256_path(path) != expected:
            raise ValueError(f"bound input checksum drifted: {key}")

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
    candidate_assignment = budgeted_policy.build_budgeted_assignment(
        owner=owner,
        attachment_local=attachment_local,
        proxy_mass=proxy_mass,
        num_partitions=num_partitions,
    )
    candidate = candidate_assignment.membership
    if np.any(candidate.sum(axis=1) < 1) or np.any(candidate.sum(axis=1) > 2):
        raise AssertionError("BMR_10 copy-count invariant failed")
    if np.any(candidate & ~(votes > 0)):
        raise AssertionError("BMR_10 emitted a copy without navigation evidence")

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

    baseline_metrics = metrics(
        name="current_all_max",
        membership=current,
        owner=owner,
        labels=labels,
        query_local=query_local,
        ground_truth=ground_truth,
        artifact=artifact,
    )
    candidate_metrics = metrics(
        name=budgeted_policy.CANDIDATE_ID,
        membership=candidate,
        owner=owner,
        labels=labels,
        query_local=query_local,
        ground_truth=ground_truth,
        artifact=artifact,
    )
    if (
        candidate_metrics["membership_semantic_sha256"]
        != candidate_assignment.membership_semantic_sha256
    ):
        raise AssertionError("BMR_10 membership semantic hash drifted")
    gates = gate_results(candidate_metrics, baseline_metrics, query_count)

    exploration_path = Path(args.exploration_manifest).expanduser().resolve()
    exploration_sha256 = sha256_path(exploration_path)
    exploration = _load_json(exploration_path)
    exploration_rows = {row["policy"]: row for row in exploration["rows"]}
    exploratory_name = "budget_0.10_secondary_proxy_mass_max"
    exploratory_row = exploration_rows.get(exploratory_name)
    if not isinstance(exploratory_row, dict):
        raise ValueError("exploration lacks the frozen BMR_10 source row")
    parity_fields = (
        "logical_point_count",
        "physical_point_count",
        "expansion_ratio",
        "copy_count_histogram",
        "physical_copy_load_min",
        "physical_copy_load_max",
        "physical_copy_load_mean",
        "physical_copy_load_cv",
        "physical_copy_load_max_over_mean",
        "gt_routing_coverage_mean",
        "gt_queries_full_coverage_fraction",
        "gt_queries_zero_coverage",
        "routed_shards_mean",
        "route_entry_points_mean",
        "route_ef_sum_mean",
    )
    drift = {
        field: {
            "formal": candidate_metrics.get(field),
            "exploratory": exploratory_row.get(field),
        }
        for field in parity_fields
        if candidate_metrics.get(field) != exploratory_row.get(field)
    }
    if drift:
        raise ValueError(f"formal BMR_10 differs from exploration: {drift}")

    source_files = {
        "evaluator": Path(__file__).resolve(),
        "policy": Path(budgeted_policy.__file__).resolve(),
        "protocol": REPO_ROOT / "experiments/multi_assignment/PROTOCOL.md",
        "addendum": REPO_ROOT
        / "experiments/multi_assignment/BUDGETED_REPLICATION_ADDENDUM.md",
    }
    result = {
        "format_version": 1,
        "record_type": "fixed_cnbr_bmr10_formal_offline_evaluation",
        "status": "PASS" if all(bool(gate["pass"]) for gate in gates.values()) else "FAIL",
        "dataset": args.dataset,
        "query_count": query_count,
        "query_scope": "full" if query_count == query_total else "prefix_tuning_only",
        "causal_boundary": {
            "load_balance_owner": "C_CNBR",
            "owner_sha256": owner_record["sha256"],
            "owner_changed_between_arms": False,
            "upper_artifact_sha256": artifact_record["sha256"],
            "navigator_sha256": navigator_sha,
            "proxy_mass_sha256": mass_record["sha256"],
            "multi_assignment_only": True,
        },
        "policy": {
            "candidate_id": budgeted_policy.CANDIDATE_ID,
            "version": budgeted_policy.POLICY_VERSION,
            "extra_copy_budget_numerator": budgeted_policy.EXTRA_COPY_BUDGET_NUMERATOR,
            "extra_copy_budget_denominator": budgeted_policy.EXTRA_COPY_BUDGET_DENOMINATOR,
            "score": budgeted_policy.SCORE,
            "eligible_secondary_count": candidate_assignment.eligible_secondary_count,
            "kept_secondary_count": candidate_assignment.kept_secondary_count,
            "at_most_two_copies": True,
            "navigation_evidence_only": True,
        },
        "baseline": baseline_metrics,
        "candidate": candidate_metrics,
        "gate_contract": {
            "maximum_expansion": MAX_EXPANSION,
            "maximum_gt_coverage_drop": MAX_COVERAGE_DROP,
            "maximum_full_coverage_drop": MAX_FULL_COVERAGE_DROP,
            "maximum_route_work_ratio": MAX_ROUTE_WORK_RATIO,
        },
        "gates": gates,
        "all_gates_pass": all(bool(gate["pass"]) for gate in gates.values()),
        "inputs": {
            "phase_a_manifest": str(phase_a_path),
            "phase_a_manifest_sha256": phase_b["phase_a_manifest_sha256"],
            "phase_b_screen": str(phase_b_path),
            "phase_b_screen_sha256": phase_b_sha256,
            "exploration_manifest": str(exploration_path),
            "exploration_manifest_sha256": exploration_sha256,
            "attachments": inputs["attachments"],
            "attachments_sha256": inputs["attachments_sha256"],
            "query_hits": inputs["query_hits"],
            "query_hits_sha256": inputs["query_hits_sha256"],
            "ground_truth": inputs["ground_truth"],
            "ground_truth_sha256": inputs["ground_truth_sha256"],
        },
        "source_code": {
            name: {
                "path": str(path.resolve()),
                "sha256": sha256_path(path.resolve()),
                "size_bytes": path.resolve().stat().st_size,
            }
            for name, path in source_files.items()
        },
    }
    output_dir.mkdir(parents=True)
    manifest_path = output_dir / "formal-evaluation.json"
    manifest_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    digest = sha256_path(manifest_path)
    sidecar = manifest_path.with_name(manifest_path.name + ".sha256")
    sidecar.write_text(digest + "\n", encoding="ascii")
    os.chmod(manifest_path, 0o444)
    os.chmod(sidecar, 0o444)
    print(json.dumps({"status": result["status"], "manifest": str(manifest_path), "sha256": digest}, indent=2))


if __name__ == "__main__":
    main()
