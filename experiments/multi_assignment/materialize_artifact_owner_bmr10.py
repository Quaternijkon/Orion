#!/usr/bin/env python3
"""Freeze an artifact-derived primary owner and materialize owner+BMR_10.

This is a research-only path for comparing a previously built Orion upper
owner with the frozen BMR_10 multi-assignment policy.  For upper nodes with
multiple memberships, the first membership is frozen as the primary owner.
The resulting bundle is not claimed to reproduce the historical full layout.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.c1.scripts import c1_orion_bmr10_scaling_materialize as scaling  # noqa: E402
from experiments.l1_balance import evaluate_cnbr_frozen_owners as phase_b  # noqa: E402
from experiments.l1_balance import materialize_native_cnbr_candidate as native  # noqa: E402
from experiments.l1_balance.run_offline_screen import (  # noqa: E402
    load_upper,
    topology_metrics,
)
from experiments.multi_assignment import budgeted_policy  # noqa: E402
from experiments.multi_assignment import screen_owner_policy_matrix as matrix  # noqa: E402


TOOL_PATH = "experiments/multi_assignment/materialize_artifact_owner_bmr10.py"
FORMAT_VERSION = 1
DEFAULT_PHASE_B = scaling.DEFAULT_PHASE_B
DEFAULT_CNBR_SELECTION = scaling.DEFAULT_CNBR_SELECTION
DEFAULT_COST_AUDIT = scaling.DEFAULT_COST_AUDIT


def load_json(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return value


def require_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} is missing: {resolved}")
    return resolved


def file_record(path: Path, known_sha256: str | None = None) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": known_sha256 or scaling.sha256(path),
        "size_bytes": path.stat().st_size,
    }


def freeze_owner(args: argparse.Namespace) -> dict[str, Any]:
    owner_name = str(args.owner_name)
    if not owner_name or owner_name in {"N_native", "C_CNBR"}:
        raise ValueError("owner-name must be a non-reserved identifier")
    phase_b_path = require_file(args.phase_b_screen, "Phase-B screen")
    screen = load_json(phase_b_path, "Phase-B screen")
    phase_a_path = require_file(Path(screen["phase_a_manifest"]), "Phase-A manifest")
    if scaling.sha256(phase_a_path) != screen["phase_a_manifest_sha256"]:
        raise ValueError("Phase-A checksum drifted")
    phase_a_manifest = load_json(phase_a_path, "Phase-A manifest")
    source_record = phase_a_manifest["construction_inputs"]["artifact"]
    source_path = require_file(Path(source_record["path"]), "source upper artifact")
    if scaling.sha256(source_path) != source_record["sha256"]:
        raise ValueError("source upper artifact checksum drifted")
    (
        source_artifact,
        _adj,
        _vectors,
        labels,
        _entry,
        edge_left,
        edge_right,
        navigator_sha,
    ) = load_upper(source_path)
    partitions = int(source_artifact["shard_count"])
    if partitions != 32:
        raise ValueError("artifact-owner BMR_10 research path requires P=32")

    owner_artifact_path = require_file(args.owner_artifact, "owner artifact")
    owner, source = matrix.artifact_owner(
        owner_artifact_path,
        reference_labels=labels,
        reference_navigator_sha256=navigator_sha,
        num_partitions=partitions,
    )
    semantic_sha = matrix.owner_semantic_sha256(labels, owner)

    matrix_path = require_file(args.matrix_screen, "owner-policy matrix")
    matrix_screen = load_json(matrix_path, "owner-policy matrix")
    if (
        matrix_screen.get("record_type")
        != "orion_owner_multi_assignment_matrix_screen"
        or matrix_screen.get("upper_navigator_sha256") != navigator_sha
        or matrix_screen.get("logical_shard_count") != partitions
    ):
        raise ValueError("owner-policy matrix identity drifted")
    owner_matrix_record = (matrix_screen.get("owner_records") or {}).get(owner_name)
    if not isinstance(owner_matrix_record, dict):
        raise ValueError(f"matrix lacks owner {owner_name}")
    if owner_matrix_record.get("owner_semantic_sha256") != semantic_sha:
        raise ValueError("matrix owner semantic checksum drifted")
    expected_combination = f"{owner_name}+{budgeted_policy.CANDIDATE_ID}"
    selected = next(
        (
            row
            for row in matrix_screen.get("rows", [])
            if row.get("combination") == expected_combination
        ),
        None,
    )
    if not isinstance(selected, dict) or selected.get("all_offline_gates_pass") is not True:
        raise ValueError(f"matrix candidate is absent or failed: {expected_combination}")

    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite owner directory: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()
    owner_path = output / f"{owner_name}.owner.i32le"
    scaling.write_new(owner_path, np.asarray(owner, dtype="<i4").tobytes(order="C"))
    owner_sha = scaling.sha256(owner_path)
    upper_loads = np.bincount(owner, minlength=partitions)
    topology = topology_metrics(
        owner, edge_left, edge_right, len(owner), partitions
    )
    record_path = output / f"{owner_name}.owner-record.json"
    record = {
        "format_version": FORMAT_VERSION,
        "record_type": "artifact_first_membership_primary_owner",
        "role": "research_only_artifact_primary_owner",
        "method": owner_name,
        "num_partitions": partitions,
        "owner": file_record(owner_path, owner_sha),
        "owner_semantic_sha256": semantic_sha,
        "partition_sizes": upper_loads.astype(int).tolist(),
        "topology_metrics": topology,
        "source": source,
        "interpretation_boundary": {
            "first_membership_frozen_as_primary": True,
            "historical_full_layout_reproduction_claim": False,
            "online_matched_recall_qps_required": True,
        },
    }
    scaling.write_json_new(record_path, record)

    policy_path = Path(budgeted_policy.__file__).resolve()
    materializer_path = Path(__file__).resolve()
    manifest_path = output / "owner-manifest.json"
    manifest = {
        "format_version": FORMAT_VERSION,
        "record_type": "artifact_owner_bmr10_owner_manifest",
        "status": "PASS",
        "created_at": scaling.utc_now(),
        "owner_name": owner_name,
        "partitions": partitions,
        "causal_boundary": {
            "owner_frozen_before_full_l0_attachment_access": True,
            "full_l0_attachments_opened_by_freeze_step": False,
            "queries_opened_by_freeze_step": False,
            "ground_truth_opened_by_freeze_step": False,
            "owner_source_uses_upper_memberships_only": True,
        },
        "construction_inputs": {
            "phase_b_screen": file_record(phase_b_path),
            "phase_a_manifest": file_record(phase_a_path),
            "source_upper_artifact": file_record(source_path),
            "owner_artifact": file_record(owner_artifact_path),
            "owner_policy_matrix": file_record(matrix_path),
            "upper_navigator_sha256": navigator_sha,
        },
        "source_code": {
            "budgeted_policy": file_record(policy_path),
            "matrix_screen": file_record(Path(matrix.__file__).resolve()),
            "materializer": file_record(materializer_path),
        },
        "owner": {
            "path": owner_path.name,
            "sha256": owner_sha,
            "semantic_sha256": semantic_sha,
            "record": record_path.name,
            "record_sha256": scaling.sha256(record_path),
        },
        "selected_matrix_row": selected,
    }
    scaling.write_json_new(manifest_path, manifest)
    scaling.freeze_directory(output, (owner_path, record_path, manifest_path))
    return {
        "status": "PASS",
        "owner_dir": str(output),
        "owner_manifest": str(manifest_path),
        "owner_manifest_sha256": scaling.sha256(manifest_path),
        "owner_name": owner_name,
        "owner_sha256": owner_sha,
        "owner_semantic_sha256": semantic_sha,
    }


def load_frozen_owner(owner_dir: Path) -> dict[str, Any]:
    directory = owner_dir.expanduser().resolve()
    checksums = scaling.verify_checksum_listing(directory)
    manifest_path = directory / "owner-manifest.json"
    manifest = load_json(manifest_path, "artifact-owner manifest")
    if (
        manifest.get("status") != "PASS"
        or manifest.get("record_type") != "artifact_owner_bmr10_owner_manifest"
        or manifest.get("partitions") != 32
    ):
        raise ValueError("artifact-owner manifest identity drifted")
    for name, current in (
        ("budgeted_policy", Path(budgeted_policy.__file__).resolve()),
        ("matrix_screen", Path(matrix.__file__).resolve()),
        ("materializer", Path(__file__).resolve()),
    ):
        record = (manifest.get("source_code") or {}).get(name)
        if not isinstance(record, dict):
            raise ValueError(f"owner manifest lacks source binding {name}")
        scaling.verify_record(current, record, name)
    owner_record = manifest.get("owner")
    if not isinstance(owner_record, dict):
        raise ValueError("owner manifest lacks owner record")
    owner_path = directory / str(owner_record.get("path") or "")
    record_path = directory / str(owner_record.get("record") or "")
    if (
        checksums.get(owner_path.name) != owner_record.get("sha256")
        or checksums.get(record_path.name) != owner_record.get("record_sha256")
        or checksums.get(manifest_path.name) != scaling.sha256(manifest_path)
    ):
        raise ValueError("artifact-owner checksum listing drifted")
    return {
        "dir": directory,
        "manifest": manifest,
        "manifest_path": manifest_path,
        "manifest_sha256": checksums[manifest_path.name],
        "owner_name": manifest["owner_name"],
        "owner_path": owner_path,
        "owner_sha256": owner_record["sha256"],
        "owner_record_path": record_path,
        "owner_record_sha256": owner_record["record_sha256"],
        "owner_record": load_json(record_path, "artifact-owner record"),
    }


def require_matrix_metric_parity(
    metrics: Mapping[str, Any], selected: Mapping[str, Any]
) -> None:
    fields = (
        "logical_point_count",
        "physical_point_count",
        "expansion_ratio",
        "copy_count_histogram",
        "physical_copy_load_min",
        "physical_copy_load_max",
        "physical_copy_load_mean",
        "physical_copy_load_cv",
        "physical_copy_load_max_over_mean",
        "membership_semantic_sha256",
    )
    drift = {
        key: {"materialized": metrics.get(key), "matrix": selected.get(key)}
        for key in fields
        if metrics.get(key) != selected.get(key)
    }
    if drift:
        raise ValueError(f"owner+BMR_10 matrix parity drifted: {drift}")


def materialize_bundle(args: argparse.Namespace) -> dict[str, Any]:
    frozen_owner = load_frozen_owner(args.owner_dir)
    base = scaling.frozen_base_binding(args)
    vectors_source = require_file(args.vectors_source, "local vector source")
    if (
        scaling.sha256(vectors_source) != base.vectors_sha256
        or vectors_source.stat().st_size != base.vectors_path.stat().st_size
    ):
        raise ValueError("local vector source differs from the bound canonical rows")
    base = replace(base, vectors_path=vectors_source)
    manifest = frozen_owner["manifest"]
    construction = manifest["construction_inputs"]
    scaling.verify_record(
        base.source_artifact_path,
        construction["source_upper_artifact"],
        "source upper artifact",
    )
    if construction["phase_b_screen"]["sha256"] != scaling.sha256(
        require_file(args.phase_b_screen, "Phase-B screen")
    ):
        raise ValueError("owner and bundle Phase-B screens differ")
    upper_count = len(base.frozen.labels)
    values = np.memmap(
        frozen_owner["owner_path"], dtype="<i4", mode="r", shape=(upper_count,)
    )
    owner = phase_b.FrozenOwner(
        frozen_owner["owner_name"],
        "research_only_artifact_primary_owner",
        dict(frozen_owner["owner_record"]),
        frozen_owner["owner_path"],
        frozen_owner["owner_sha256"],
        values,
        frozen_owner["owner_record_path"],
        frozen_owner["owner_record_sha256"],
        dict(frozen_owner["owner_record"]),
    )
    owner_set = dict(base.frozen.owners)
    owner_set[owner.name] = owner
    minimal_phase_a = {
        "format_version": FORMAT_VERSION,
        "record_type": "artifact_owner_bmr10_owner_manifest",
        "fixed_parameters": {
            "num_partitions": 32,
            "upper_subset_denominator": 32,
        },
        "construction_inputs": {
            "upper_only_projection": base.frozen.manifest["construction_inputs"][
                "upper_only_projection"
            ]
        },
    }
    frozen = replace(
        base.frozen,
        manifest_path=frozen_owner["manifest_path"],
        manifest_sha256=frozen_owner["manifest_sha256"],
        manifest=minimal_phase_a,
        owners=owner_set,
    )
    assignment = budgeted_policy.build_budgeted_assignment(
        owner=owner.values,
        attachment_local=base.inputs.attachment_local,
        proxy_mass=frozen.mass_values,
        num_partitions=32,
    )
    metrics, histogram, shard_counts = scaling.assignment_metrics(
        assignment,
        logical_count=int(frozen.artifact["logical_point_count"]),
        partitions=32,
    )
    selected = manifest["selected_matrix_row"]
    require_matrix_metric_parity(metrics, selected)
    layout_sha = scaling.assignment_digest(assignment.membership)
    metrics["assignment_jsonl_sha256"] = layout_sha
    arm_name = f"{owner.name}_{budgeted_policy.CANDIDATE_ID}"
    arm = native.ArmContract(
        arm=arm_name,
        balance_variant=owner.name.lower(),
        phase_a_role="artifact_first_membership_primary_research",
        phase_b_role="offline_matrix_pass_online_recall_validation_required",
        selected_adoption_candidate=False,
    )
    record = {
        "identity_all_pass": True,
        "topology_all_pass": True,
        "topology_metrics": frozen_owner["owner_record"]["topology_metrics"],
        "graph_topology_gates": {},
        "query_topology_gates": selected["gate_results"],
        "physical_copy_load_improves_over_reference": False,
        "materialization_parity": {"assignment_bytes_sha256": layout_sha},
    }
    binding = replace(
        base,
        arm=arm,
        screen_path=frozen_owner["manifest_path"],
        screen_sha256=frozen_owner["manifest_sha256"],
        screen=manifest,
        record=record,
        frozen=frozen,
        owner=owner,
        selection_path=None,
        selection_sha256=None,
        selection_source_code_record_sha256=None,
        construction_cost_path=None,
        construction_cost_sha256=None,
        construction_cost_gate=None,
    )
    output = args.output_dir.expanduser().resolve()
    result = native.materialize_bundle(
        binding,
        generation=int(args.generation),
        rebind_binary=require_file(args.rebind_binary, "rebind binary"),
        replay_verifier=require_file(args.replay_verifier, "replay verifier"),
        output_dir=output,
        chunk_size=100_000,
        tool_path=TOOL_PATH,
        parameter_overrides={
            "l1_partitioner": owner.name,
            "balance_mode": "artifact_first_membership_primary_research",
            "initial_num_shards": 32,
            "multi_assign_max_shards": 2,
            "multi_assign_extra_copy_budget_numerator": 1,
            "multi_assign_extra_copy_budget_denominator": 10,
            "multi_assign_score": budgeted_policy.SCORE,
            "multi_assignment_policy": budgeted_policy.CANDIDATE_ID,
            "multi_assignment_policy_version": budgeted_policy.POLICY_VERSION,
        },
        extra_diagnostics={
            "research_candidate": "artifact_first_membership_primary_plus_bmr10",
            "historical_full_layout_reproduction_claim": False,
            "owner_manifest_sha256": frozen_owner["manifest_sha256"],
            "owner_semantic_sha256": frozen_owner["owner_record"][
                "owner_semantic_sha256"
            ],
            "assignment_metrics": metrics,
            "multi_assignment_contract": {
                "candidate_id": budgeted_policy.CANDIDATE_ID,
                "version": budgeted_policy.POLICY_VERSION,
                "extra_copy_budget_numerator": 1,
                "extra_copy_budget_denominator": 10,
                "maximum_copies_per_point": 2,
                "score": budgeted_policy.SCORE,
                "load_balance_owner_unchanged": True,
                "canonical_default_eligible": False,
            },
        },
        extra_provenance={
            "artifact_owner_manifest": str(frozen_owner["manifest_path"]),
            "artifact_owner_manifest_sha256": frozen_owner["manifest_sha256"],
            "owner_policy_matrix": construction["owner_policy_matrix"]["path"],
            "owner_policy_matrix_sha256": construction["owner_policy_matrix"][
                "sha256"
            ],
            "bmr10_policy_source": str(Path(budgeted_policy.__file__).resolve()),
            "bmr10_policy_source_sha256": scaling.sha256(
                Path(budgeted_policy.__file__).resolve()
            ),
            "local_vectors_source": str(vectors_source),
            "local_vectors_source_sha256": base.vectors_sha256,
            "local_vectors_source_reason": "project_filesystem_full_same_bytes_hardlink",
        },
        assignment_builder=scaling.assignment_builder(
            assignment,
            expected_digest=layout_sha,
            histogram=histogram,
            shard_counts=shard_counts,
        ),
    )
    return {
        **result,
        "status": "PASS",
        "owner_name": owner.name,
        "assignment_metrics": metrics,
        "interpretation_boundary": {
            "historical_full_layout_reproduction_claim": False,
            "online_matched_recall_qps_required": True,
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    owner = subparsers.add_parser("owner")
    owner.add_argument("--owner-name", required=True)
    owner.add_argument("--owner-artifact", type=Path, required=True)
    owner.add_argument("--matrix-screen", type=Path, required=True)
    owner.add_argument("--phase-b-screen", type=Path, default=DEFAULT_PHASE_B)
    owner.add_argument("--output-dir", type=Path, required=True)

    bundle = subparsers.add_parser("bundle")
    bundle.add_argument("--owner-dir", type=Path, required=True)
    bundle.add_argument("--phase-b-screen", type=Path, default=DEFAULT_PHASE_B)
    bundle.add_argument(
        "--cnbr-selection-manifest", type=Path, default=DEFAULT_CNBR_SELECTION
    )
    bundle.add_argument(
        "--construction-cost-audit", type=Path, default=DEFAULT_COST_AUDIT
    )
    bundle.add_argument("--generation", type=int, required=True)
    bundle.add_argument("--rebind-binary", type=Path, required=True)
    bundle.add_argument("--replay-verifier", type=Path, required=True)
    bundle.add_argument("--vectors-source", type=Path, required=True)
    bundle.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = freeze_owner(args) if args.command == "owner" else materialize_bundle(args)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
