#!/usr/bin/env python3
"""Offline screen of frozen C_CNBR+BMR_10 versus BMR_10_CAP35."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
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
from experiments.multi_assignment import capacity_bounded_bmr10 as cap  # noqa: E402
from experiments.multi_assignment.screen_fixed_owner import (  # noqa: E402
    MAX_COVERAGE_DROP,
    MAX_FULL_COVERAGE_DROP,
    MAX_ROUTE_WORK_RATIO,
    _copy_histogram,
    _load_json,
)


TOOL_PATH = "experiments/multi_assignment/screen_capacity_bounded_bmr10.py"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--phase-b-screen", type=Path, required=True)
    parser.add_argument("--query-count", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": sha256_path(path),
        "size_bytes": path.stat().st_size,
    }


def membership_metrics(
    *,
    name: str,
    membership: np.ndarray,
    owner: np.ndarray,
    labels: np.ndarray,
    query_local: np.ndarray,
    ground_truth: np.ndarray,
    dynamic_ef_base: int,
    dynamic_ef_factor: int,
    semantic_sha256: str,
) -> dict[str, Any]:
    copies = membership.sum(axis=1, dtype=np.int16)
    loads = membership.sum(axis=0, dtype=np.int64)
    mean = float(loads.mean())
    result = {
        "name": name,
        "logical_point_count": len(membership),
        "physical_point_count": int(copies.sum()),
        "expansion_ratio": float(copies.mean()),
        "copy_count_histogram": _copy_histogram(copies),
        "membership_semantic_sha256": semantic_sha256,
        "physical_copy_load_min": int(loads.min()),
        "physical_copy_load_max": int(loads.max()),
        "physical_copy_load_mean": mean,
        "physical_copy_load_cv": float(loads.std() / mean),
        "physical_copy_load_min_over_mean": float(loads.min() / mean),
        "physical_copy_load_max_over_mean": float(loads.max() / mean),
    }
    result.update(
        query_metrics(
            owner,
            query_local,
            membership[labels],
            membership,
            ground_truth,
            dynamic_ef_base,
            dynamic_ef_factor,
        )
    )
    return result


def comparison_gates(
    candidate: dict[str, Any],
    baseline: dict[str, Any],
    diagnostics: dict[str, Any],
    query_count: int,
) -> dict[str, dict[str, Any]]:
    zero_limit = min(
        int(baseline["gt_queries_zero_coverage"]) + 2,
        max(1, int(query_count * 0.001)),
    )
    specs = {
        "same_physical_point_count": (
            candidate["physical_point_count"],
            "==",
            baseline["physical_point_count"],
        ),
        "same_copy_count_histogram": (
            candidate["copy_count_histogram"],
            "==",
            baseline["copy_count_histogram"],
        ),
        "capacity_bounds_satisfied": (
            diagnostics.get("bounds_satisfied"),
            "==",
            True,
        ),
        "physical_copy_load_min": (
            candidate["physical_copy_load_min"],
            ">=",
            diagnostics["lower_load"],
        ),
        "physical_copy_load_max": (
            candidate["physical_copy_load_max"],
            "<=",
            baseline["physical_copy_load_max"],
        ),
        "physical_copy_load_cv": (
            candidate["physical_copy_load_cv"],
            "<=",
            baseline["physical_copy_load_cv"],
        ),
        "gt_routing_coverage_mean": (
            candidate["gt_routing_coverage_mean"],
            ">=",
            baseline["gt_routing_coverage_mean"] - MAX_COVERAGE_DROP,
        ),
        "gt_queries_full_coverage_fraction": (
            candidate["gt_queries_full_coverage_fraction"],
            ">=",
            baseline["gt_queries_full_coverage_fraction"]
            - MAX_FULL_COVERAGE_DROP,
        ),
        "gt_queries_zero_coverage": (
            candidate["gt_queries_zero_coverage"],
            "<=",
            zero_limit,
        ),
        "routed_shards_mean": (
            candidate["routed_shards_mean"],
            "<=",
            baseline["routed_shards_mean"] * MAX_ROUTE_WORK_RATIO,
        ),
        "route_ef_sum_mean": (
            candidate["route_ef_sum_mean"],
            "<=",
            baseline["route_ef_sum_mean"] * MAX_ROUTE_WORK_RATIO,
        ),
    }
    gates: dict[str, dict[str, Any]] = {}
    for name, (observed, operator, threshold) in specs.items():
        if operator == "==":
            passed = observed == threshold
        elif operator == "<=":
            passed = observed <= threshold
        else:
            passed = observed >= threshold
        gates[name] = {
            "observed": observed,
            "operator": operator,
            "threshold": threshold,
            "pass": bool(passed),
        }
    return gates


def run(args: argparse.Namespace) -> Path:
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output}")
    phase_b_path = args.phase_b_screen.expanduser().resolve(strict=True)
    phase_b = _load_json(phase_b_path)
    phase_a_path = Path(phase_b["phase_a_manifest"]).expanduser().resolve(strict=True)
    if sha256_path(phase_a_path) != phase_b["phase_a_manifest_sha256"]:
        raise ValueError("Phase-A manifest checksum drifted")
    phase_a = _load_json(phase_a_path)
    artifact_path = Path(
        phase_a["construction_inputs"]["artifact"]["path"]
    ).expanduser().resolve(strict=True)
    if sha256_path(artifact_path) != phase_a["construction_inputs"]["artifact"][
        "sha256"
    ]:
        raise ValueError("upper artifact checksum drifted")
    artifact, _adj, _vectors, labels, _entry, _left, _right, navigator_sha = (
        load_upper(artifact_path)
    )
    partitions = int(artifact["shard_count"])
    owner_record = phase_a["owners"]["C_CNBR"]["owner"]
    owner_path = (phase_a_path.parent / owner_record["path"]).resolve()
    if sha256_path(owner_path) != owner_record["sha256"]:
        raise ValueError("C_CNBR owner checksum drifted")
    owner = checked_binary_matrix(
        owner_path, rows=len(labels), width=1, dtype="<i4"
    ).reshape(-1)
    proxy_record = phase_a["mass"]["values"]
    proxy_path = (phase_a_path.parent / proxy_record["path"]).resolve()
    if sha256_path(proxy_path) != proxy_record["sha256"]:
        raise ValueError("proxy mass checksum drifted")
    proxy_mass = checked_binary_matrix(
        proxy_path, rows=len(labels), width=1, dtype="<u8"
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
        path = Path(inputs[key]).expanduser().resolve(strict=True)
        expected = inputs.get(f"{key}_sha256")
        if expected is not None and sha256_path(path) != expected:
            raise ValueError(f"bound input checksum drifted: {key}")
    attachment_manifest = _load_json(Path(inputs["attachments_manifest"]))
    logical_count = int(attachment_manifest["row_count"])
    attachment_hits = checked_binary_matrix(
        Path(inputs["attachments"]),
        rows=logical_count,
        width=int(attachment_manifest["top_k"]),
        dtype="<u8",
    )
    attachment_local = local_hits(attachment_hits, labels, logical_count)

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

    started = time.perf_counter()
    baseline_assignment = budgeted_policy.build_budgeted_assignment(
        owner=owner,
        attachment_local=attachment_local,
        proxy_mass=proxy_mass,
        num_partitions=partitions,
    )
    candidate_assignment = cap.build_capacity_bounded_assignment(
        owner=owner,
        attachment_local=attachment_local,
        proxy_mass=proxy_mass,
        num_partitions=partitions,
    )
    base = membership_metrics(
        name=budgeted_policy.CANDIDATE_ID,
        membership=baseline_assignment.membership,
        owner=owner,
        labels=labels,
        query_local=query_local,
        ground_truth=ground_truth,
        dynamic_ef_base=int(artifact["dynamic_ef_base"]),
        dynamic_ef_factor=int(artifact["dynamic_ef_factor"]),
        semantic_sha256=baseline_assignment.membership_semantic_sha256,
    )
    candidate = membership_metrics(
        name=cap.CANDIDATE_ID,
        membership=candidate_assignment.membership,
        owner=owner,
        labels=labels,
        query_local=query_local,
        ground_truth=ground_truth,
        dynamic_ef_base=int(artifact["dynamic_ef_base"]),
        dynamic_ef_factor=int(artifact["dynamic_ef_factor"]),
        semantic_sha256=candidate_assignment.membership_semantic_sha256,
    )
    gates = comparison_gates(
        candidate, base, candidate_assignment.diagnostics, query_count
    )
    result = {
        "format_version": 1,
        "record_type": "capacity_bounded_bmr10_offline_screen",
        "status": "PASS" if all(gate["pass"] for gate in gates.values()) else "FAIL",
        "dataset": args.dataset,
        "query_count": query_count,
        "tool": TOOL_PATH,
        "parameters": {
            "min_load_ratio": cap.MIN_LOAD_RATIO,
            "max_load_ratio": cap.MAX_LOAD_RATIO,
            "max_vote_loss": cap.MAX_VOTE_LOSS,
            "max_passes": cap.MAX_PASSES,
        },
        "identity": {
            "phase_b_screen": file_record(phase_b_path),
            "phase_a_manifest": file_record(phase_a_path),
            "source_upper_artifact": file_record(artifact_path),
            "owner": file_record(owner_path),
            "upper_navigator_sha256": navigator_sha,
            "owner_unchanged": True,
        },
        "baseline": base,
        "candidate": candidate,
        "capacity_diagnostics": candidate_assignment.diagnostics,
        "gates": gates,
        "elapsed_seconds": time.perf_counter() - started,
        "interpretation_boundary": {
            "online_qps_inferred": False,
            "materialization_allowed_only_after_dual_dataset_pass": True,
        },
    }
    output.mkdir(parents=True)
    path = output / "screen.json"
    path.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path


def main(argv: list[str] | None = None) -> int:
    path = run(parse_args(argv))
    print(json.dumps({"screen": str(path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
