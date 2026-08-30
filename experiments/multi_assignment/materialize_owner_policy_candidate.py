#!/usr/bin/env python3
"""Materialize one frozen owner x multi-assignment tournament candidate.

The owner is frozen from the checksum-bound owner-policy matrix before full L0
attachments are opened.  Bundle construction then applies exactly one of the
three already-screened post-owner policies and requires byte/metric parity with
the corresponding matrix row.  Every output is research-only until the online
matched-recall tournament is complete.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import hashlib
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
from experiments.multi_assignment import budgeted_policy  # noqa: E402
from experiments.multi_assignment import screen_fixed_owner as fixed  # noqa: E402
from experiments.multi_assignment import screen_owner_policy_matrix as matrix  # noqa: E402


TOOL_PATH = "experiments/multi_assignment/materialize_owner_policy_candidate.py"
FORMAT_VERSION = 1
OWNER_NAMES = ("C_CNBR", "HISTORICAL_PRIMARY", "CCNB")
POLICY_NAMES = ("current_all_max", "single_rank", budgeted_policy.CANDIDATE_ID)
DEFAULT_MATRIX = Path(
    "/proj/intelisys-PG0/exp/orion-distributed/owner-policy-matrix-20260828/"
    "glove-p32-v1/screen-manifest.json"
)
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


def load_matrix(path: Path) -> tuple[Path, dict[str, Any]]:
    resolved = require_file(path, "owner-policy matrix")
    manifest = load_json(resolved, "owner-policy matrix")
    if (
        manifest.get("record_type")
        != "orion_owner_multi_assignment_matrix_screen"
        or manifest.get("logical_shard_count") != 32
        or tuple(manifest.get("owner_order") or ()) != OWNER_NAMES
        or tuple(manifest.get("policy_order") or ()) != POLICY_NAMES
    ):
        raise ValueError("owner-policy matrix identity drifted")
    return resolved, manifest


def selected_row(
    manifest: Mapping[str, Any], owner_name: str, policy_name: str
) -> dict[str, Any]:
    combination = f"{owner_name}+{policy_name}"
    rows = [
        row
        for row in manifest.get("rows", [])
        if isinstance(row, dict) and row.get("combination") == combination
    ]
    if len(rows) != 1:
        raise ValueError(f"matrix must contain exactly one row for {combination}")
    return dict(rows[0])


def freeze_owner(args: argparse.Namespace) -> dict[str, Any]:
    owner_name = str(args.owner_name)
    if owner_name not in OWNER_NAMES:
        raise ValueError(f"owner-name must be one of {OWNER_NAMES}")
    matrix_path, screen = load_matrix(args.matrix_screen)
    phase_b_record = screen.get("phase_b_screen")
    if not isinstance(phase_b_record, dict):
        raise ValueError("matrix lacks its Phase-B binding")
    phase_b_path = require_file(Path(str(phase_b_record.get("path") or "")), "Phase-B screen")
    if scaling.sha256(phase_b_path) != phase_b_record.get("sha256"):
        raise ValueError("matrix Phase-B checksum drifted")
    if args.phase_b_screen is not None:
        declared = require_file(args.phase_b_screen, "declared Phase-B screen")
        if declared != phase_b_path:
            raise ValueError("declared Phase-B screen differs from the matrix")

    phase_b_screen = load_json(phase_b_path, "Phase-B screen")
    phase_a_path = require_file(Path(phase_b_screen["phase_a_manifest"]), "Phase-A manifest")
    if scaling.sha256(phase_a_path) != phase_b_screen["phase_a_manifest_sha256"]:
        raise ValueError("Phase-A checksum drifted")
    phase_a_manifest = load_json(phase_a_path, "Phase-A manifest")
    source_record = phase_a_manifest["construction_inputs"]["artifact"]
    source_artifact = require_file(Path(source_record["path"]), "source upper artifact")
    if scaling.sha256(source_artifact) != source_record["sha256"]:
        raise ValueError("source upper artifact checksum drifted")
    (
        _artifact,
        _adjacency,
        _vectors,
        labels,
        _entry,
        _edge_left,
        _edge_right,
        navigator_sha,
    ) = matrix.load_upper(source_artifact)
    if navigator_sha != screen.get("upper_navigator_sha256"):
        raise ValueError("matrix upper navigator checksum drifted")

    owner_record = (screen.get("owner_records") or {}).get(owner_name)
    if not isinstance(owner_record, dict):
        raise ValueError(f"matrix lacks owner record {owner_name}")
    if owner_name == "C_CNBR":
        binary_record = owner_record.get("owner_binary")
        if not isinstance(binary_record, dict):
            raise ValueError("C_CNBR matrix owner lacks binary binding")
        source_owner = require_file(Path(binary_record["path"]), "C_CNBR owner")
        if scaling.sha256(source_owner) != binary_record.get("sha256"):
            raise ValueError("C_CNBR owner checksum drifted")
        owner = np.fromfile(source_owner, dtype="<i4")
        source = {
            "source_kind": "phase_a_frozen_binary_owner",
            "owner_binary": file_record(source_owner),
            "historical_layout_reproduction_claim": True,
        }
    else:
        artifact_record = owner_record.get("artifact")
        if not isinstance(artifact_record, dict):
            raise ValueError(f"{owner_name} matrix owner lacks artifact binding")
        artifact_path = require_file(Path(artifact_record["path"]), f"{owner_name} owner artifact")
        if scaling.sha256(artifact_path) != artifact_record.get("sha256"):
            raise ValueError(f"{owner_name} owner artifact checksum drifted")
        owner, source = matrix.artifact_owner(
            artifact_path,
            reference_labels=labels,
            reference_navigator_sha256=navigator_sha,
            num_partitions=32,
        )
    if len(owner) != len(labels):
        raise ValueError("owner row count differs from upper labels")
    semantic_sha = matrix.owner_semantic_sha256(labels, owner)
    if semantic_sha != owner_record.get("owner_semantic_sha256"):
        raise ValueError(f"{owner_name} semantic checksum differs from matrix")

    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite owner directory: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()
    owner_path = output / f"{owner_name}.owner.i32le"
    scaling.write_new(owner_path, np.asarray(owner, dtype="<i4").tobytes(order="C"))
    owner_sha = scaling.sha256(owner_path)
    partition_sizes = np.bincount(owner, minlength=32).astype(int).tolist()
    record_path = output / f"{owner_name}.owner-record.json"
    record = {
        "format_version": FORMAT_VERSION,
        "record_type": "owner_policy_tournament_frozen_owner",
        "role": "research_only_pre_assignment_owner",
        "method": owner_name,
        "num_partitions": 32,
        "owner": file_record(owner_path, owner_sha),
        "owner_semantic_sha256": semantic_sha,
        "partition_sizes": partition_sizes,
        "topology_metrics": owner_record["topology_metrics"],
        "source": source,
        "interpretation_boundary": {
            "owner_frozen_before_full_l0_attachment_access": True,
            "historical_full_layout_reproduction_claim": bool(
                owner_record.get("historical_layout_reproduction_claim")
            ),
            "online_matched_recall_qps_required": True,
        },
    }
    scaling.write_json_new(record_path, record)
    manifest_path = output / "owner-manifest.json"
    manifest = {
        "format_version": FORMAT_VERSION,
        "record_type": "owner_policy_tournament_owner_manifest",
        "status": "PASS",
        "created_at": scaling.utc_now(),
        "owner_name": owner_name,
        "partitions": 32,
        "causal_boundary": {
            "owner_frozen_before_full_l0_attachment_access": True,
            "full_l0_attachments_opened_by_freeze_step": False,
            "queries_opened_by_freeze_step": False,
            "ground_truth_opened_by_freeze_step": False,
        },
        "construction_inputs": {
            "owner_policy_matrix": file_record(matrix_path),
            "phase_b_screen": file_record(phase_b_path),
            "phase_a_manifest": file_record(phase_a_path),
            "source_upper_artifact": file_record(source_artifact),
            "upper_navigator_sha256": navigator_sha,
        },
        "source_code": {
            "matrix_screen": file_record(Path(matrix.__file__).resolve()),
            "materializer": file_record(Path(__file__).resolve()),
        },
        "owner": {
            "path": owner_path.name,
            "sha256": owner_sha,
            "semantic_sha256": semantic_sha,
            "record": record_path.name,
            "record_sha256": scaling.sha256(record_path),
        },
        "matrix_owner_record": owner_record,
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
    manifest = load_json(manifest_path, "tournament owner manifest")
    if (
        manifest.get("status") != "PASS"
        or manifest.get("record_type") != "owner_policy_tournament_owner_manifest"
        or manifest.get("owner_name") not in OWNER_NAMES
        or manifest.get("partitions") != 32
    ):
        raise ValueError("tournament owner manifest identity drifted")
    for name, current in (
        ("matrix_screen", Path(matrix.__file__).resolve()),
        ("materializer", Path(__file__).resolve()),
    ):
        record = (manifest.get("source_code") or {}).get(name)
        if not isinstance(record, dict):
            raise ValueError(f"owner manifest lacks source binding {name}")
        scaling.verify_record(current, record, name)
    owner_record = manifest.get("owner")
    if not isinstance(owner_record, dict):
        raise ValueError("tournament owner manifest lacks owner record")
    owner_path = directory / str(owner_record.get("path") or "")
    record_path = directory / str(owner_record.get("record") or "")
    if (
        checksums.get(owner_path.name) != owner_record.get("sha256")
        or checksums.get(record_path.name) != owner_record.get("record_sha256")
        or checksums.get(manifest_path.name) != scaling.sha256(manifest_path)
    ):
        raise ValueError("tournament owner checksum listing drifted")
    return {
        "dir": directory,
        "manifest": manifest,
        "manifest_path": manifest_path,
        "manifest_sha256": checksums[manifest_path.name],
        "owner_name": manifest["owner_name"],
        "owner_path": owner_path,
        "owner_sha256": owner_record["sha256"],
        "owner_semantic_sha256": owner_record["semantic_sha256"],
        "owner_record_path": record_path,
        "owner_record_sha256": owner_record["record_sha256"],
        "owner_record": load_json(record_path, "tournament owner record"),
    }


def build_membership(
    owner: np.ndarray,
    attachment_local: np.ndarray,
    proxy_mass: np.ndarray,
    policy_name: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    if policy_name == budgeted_policy.CANDIDATE_ID:
        assignment = budgeted_policy.build_budgeted_assignment(
            owner=owner,
            attachment_local=attachment_local,
            proxy_mass=proxy_mass,
            num_partitions=32,
        )
        return assignment.membership, {
            "eligible_secondary_count": assignment.eligible_secondary_count,
            "kept_secondary_count": assignment.kept_secondary_count,
            "extra_copy_budget_cap": len(attachment_local) // 10,
        }
    policy = next((row for row in fixed.POLICIES if row.name == policy_name), None)
    if policy is None or policy_name not in {"current_all_max", "single_rank"}:
        raise ValueError(f"unsupported tournament policy: {policy_name}")
    hit_owner = np.asarray(owner[attachment_local], dtype=np.int32)
    votes, maximum, first_rank, rank_top, id_top = fixed.build_vote_evidence(
        hit_owner, 32
    )
    membership = fixed.membership_for_policy(
        hit_owner=hit_owner,
        votes=votes,
        maximum=maximum,
        first_rank=first_rank,
        rank_top=rank_top,
        id_top=id_top,
        policy=policy,
    )
    return membership, {}


def membership_metrics(
    membership: np.ndarray, policy_name: str, extra: Mapping[str, Any]
) -> tuple[dict[str, Any], Counter[int], np.ndarray]:
    copies = membership.sum(axis=1, dtype=np.int16)
    loads = membership.sum(axis=0, dtype=np.int64)
    values, counts = np.unique(copies, return_counts=True)
    histogram = Counter(
        {
            int(value): int(count)
            for value, count in zip(values.tolist(), counts.tolist(), strict=True)
        }
    )
    logical = len(membership)
    physical = int(copies.sum())
    mean = float(loads.mean())
    metrics: dict[str, Any] = {
        "policy": policy_name,
        "membership_semantic_sha256": fixed._semantic_sha256(membership),
        "logical_point_count": logical,
        "physical_point_count": physical,
        "expansion_ratio": physical / logical,
        "extra_copy_fraction": physical / logical - 1.0,
        "copy_count_histogram": {
            str(key): int(value) for key, value in sorted(histogram.items())
        },
        "physical_copy_load_min": int(loads.min()),
        "physical_copy_load_max": int(loads.max()),
        "physical_copy_load_mean": mean,
        "physical_copy_load_cv": float(loads.std() / mean),
        "physical_copy_load_max_over_mean": float(loads.max() / mean),
        "physical_copy_load_min_over_mean": float(loads.min() / mean),
        "physical_copy_load_empty_shards": int((loads == 0).sum()),
        **dict(extra),
    }
    return metrics, histogram, loads


def assignment_digest(membership: np.ndarray) -> str:
    digest = hashlib.sha256()
    for point_id, row in enumerate(membership):
        digest.update(native.assignment_bytes(point_id, np.flatnonzero(row)))
    return digest.hexdigest()


def assignment_builder(
    membership: np.ndarray,
    *,
    expected_digest: str,
    histogram: Counter[int],
    shard_counts: np.ndarray,
) -> Any:
    def build(
        binding: native.MaterializationBinding,
        output_dir: Path,
        chunk_size: int,
    ) -> native.AssignmentResult:
        del chunk_size
        row_count = int(binding.frozen.artifact["logical_point_count"])
        upper_labels = np.asarray(binding.frozen.labels, dtype=np.int64)
        upper_position = np.full(row_count, -1, dtype=np.int32)
        upper_position[upper_labels] = np.arange(len(upper_labels), dtype=np.int32)
        upper_memberships: list[list[int] | None] = [None] * len(upper_labels)
        path = output_dir / "orion_numeric_import.assignments.jsonl"
        temporary = path.with_name(path.name + ".tmp")
        digest = hashlib.sha256()
        primary_digest = hashlib.sha256()
        primary_loads = np.zeros(32, dtype=np.int64)
        try:
            with temporary.open("xb") as handle:
                for point_id, row in enumerate(membership):
                    shards = np.flatnonzero(row).astype(int).tolist()
                    if not shards:
                        raise RuntimeError(f"point {point_id} has no shard assignment")
                    encoded = native.assignment_bytes(point_id, shards)
                    handle.write(encoded)
                    digest.update(encoded)
                    primary_digest.update(np.asarray([shards[0]], dtype="<i4").tobytes())
                    primary_loads[shards[0]] += 1
                    upper_index = int(upper_position[point_id])
                    if upper_index >= 0:
                        upper_memberships[upper_index] = shards
            if digest.hexdigest() != expected_digest:
                raise RuntimeError("tournament assignment digest changed while publishing")
            os.replace(temporary, path)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        if any(row is None for row in upper_memberships):
            raise RuntimeError("tournament assignment did not bind every upper node")
        return native.AssignmentResult(
            path=path,
            layout_sha256=expected_digest,
            physical_point_count=int(membership.sum()),
            shard_counts=shard_counts,
            copy_histogram=histogram,
            primary_shards_sha256=primary_digest.hexdigest(),
            primary_shard_loads=primary_loads,
            upper_memberships=[list(row) for row in upper_memberships if row is not None],
        )

    return build


def require_matrix_metric_parity(
    observed: Mapping[str, Any], selected: Mapping[str, Any]
) -> None:
    fields = (
        "logical_point_count",
        "physical_point_count",
        "expansion_ratio",
        "extra_copy_fraction",
        "copy_count_histogram",
        "membership_semantic_sha256",
        "physical_copy_load_min",
        "physical_copy_load_max",
        "physical_copy_load_mean",
        "physical_copy_load_cv",
        "physical_copy_load_max_over_mean",
        "physical_copy_load_min_over_mean",
        "physical_copy_load_empty_shards",
        "eligible_secondary_count",
        "kept_secondary_count",
        "extra_copy_budget_cap",
    )
    drift = {
        key: {"materialized": observed.get(key), "matrix": selected.get(key)}
        for key in fields
        if (key in observed or key in selected) and observed.get(key) != selected.get(key)
    }
    if drift:
        raise ValueError(f"owner-policy matrix parity drifted: {drift}")


def policy_parameter_overrides(owner_name: str, policy_name: str) -> dict[str, Any]:
    common: dict[str, Any] = {
        "l1_partitioner": owner_name,
        "balance_mode": "owner_policy_tournament_research",
        "initial_num_shards": 32,
        "enable_fission": False,
        "enable_topology_refinement": False,
        "l0_repair": False,
        "multi_assign_min_max_vote": 2,
        "multi_assign_vote_delta": 0,
        "multi_assignment_policy": policy_name,
        "multi_assignment_policy_version": 1,
    }
    if policy_name == "current_all_max":
        common.update({"use_multi_assign": True, "multi_assign_max_shards": 0})
    elif policy_name == "single_rank":
        common.update({"use_multi_assign": False, "multi_assign_max_shards": 1})
    elif policy_name == budgeted_policy.CANDIDATE_ID:
        common.update(
            {
                "use_multi_assign": True,
                "multi_assign_max_shards": 2,
                "multi_assign_extra_copy_budget_numerator": 1,
                "multi_assign_extra_copy_budget_denominator": 10,
                "multi_assign_score": budgeted_policy.SCORE,
            }
        )
    else:
        raise ValueError(f"unsupported policy: {policy_name}")
    return common


def materialize_bundle(args: argparse.Namespace) -> dict[str, Any]:
    owner_info = load_frozen_owner(args.owner_dir)
    owner_name = owner_info["owner_name"]
    policy_name = str(args.policy)
    if policy_name not in POLICY_NAMES:
        raise ValueError(f"policy must be one of {POLICY_NAMES}")
    matrix_path, screen = load_matrix(args.matrix_screen)
    owner_manifest_matrix = owner_info["manifest"]["construction_inputs"][
        "owner_policy_matrix"
    ]
    if (
        owner_manifest_matrix.get("path") != str(matrix_path)
        or owner_manifest_matrix.get("sha256") != scaling.sha256(matrix_path)
    ):
        raise ValueError("owner and bundle matrices differ")
    selected = selected_row(screen, owner_name, policy_name)
    if selected.get("owner_semantic_sha256") != owner_info["owner_semantic_sha256"]:
        raise ValueError("selected matrix row owner checksum drifted")

    base = scaling.frozen_base_binding(args)
    vectors_source = require_file(args.vectors_source, "local vector source")
    if (
        scaling.sha256(vectors_source) != base.vectors_sha256
        or vectors_source.stat().st_size != base.vectors_path.stat().st_size
    ):
        raise ValueError("local vector source differs from bound canonical rows")
    base = replace(base, vectors_path=vectors_source)
    for key, current in (
        ("phase_b_screen", base.screen_path),
        ("source_upper_artifact", base.source_artifact_path),
    ):
        record = owner_info["manifest"]["construction_inputs"][key]
        scaling.verify_record(current, record, key)

    upper_count = len(base.frozen.labels)
    owner_values = np.memmap(
        owner_info["owner_path"], dtype="<i4", mode="r", shape=(upper_count,)
    )
    owner = phase_b.FrozenOwner(
        owner_name,
        "research_only_pre_assignment_owner",
        dict(owner_info["owner_record"]),
        owner_info["owner_path"],
        owner_info["owner_sha256"],
        owner_values,
        owner_info["owner_record_path"],
        owner_info["owner_record_sha256"],
        dict(owner_info["owner_record"]),
    )
    owner_set = dict(base.frozen.owners)
    owner_set[owner_name] = owner
    frozen = replace(
        base.frozen,
        manifest_path=owner_info["manifest_path"],
        manifest_sha256=owner_info["manifest_sha256"],
        manifest={
            "format_version": FORMAT_VERSION,
            "record_type": "owner_policy_tournament_owner_manifest",
            "fixed_parameters": {
                "num_partitions": 32,
                "upper_subset_denominator": 32,
            },
            "construction_inputs": {
                "upper_only_projection": base.frozen.manifest["construction_inputs"][
                    "upper_only_projection"
                ]
            },
        },
        owners=owner_set,
    )
    membership, extra = build_membership(
        owner.values,
        base.inputs.attachment_local,
        frozen.mass_values,
        policy_name,
    )
    metrics, histogram, shard_counts = membership_metrics(
        membership, policy_name, extra
    )
    require_matrix_metric_parity(metrics, selected)
    layout_sha = assignment_digest(membership)
    metrics["assignment_jsonl_sha256"] = layout_sha

    arm_name = f"OWNER_POLICY_{owner_name}_{policy_name}"
    arm = native.ArmContract(
        arm=arm_name,
        balance_variant=owner_name.lower(),
        phase_a_role="owner_policy_tournament_frozen_owner",
        phase_b_role="online_tuning_and_heldout_validation_required",
        selected_adoption_candidate=False,
    )
    record = {
        "identity_all_pass": True,
        "topology_all_pass": True,
        "topology_metrics": owner_info["owner_record"]["topology_metrics"],
        "graph_topology_gates": {},
        "query_topology_gates": selected.get("gate_results") or {},
        "physical_copy_load_improves_over_reference": False,
        "materialization_parity": {"assignment_bytes_sha256": layout_sha},
    }
    binding = replace(
        base,
        arm=arm,
        screen_path=owner_info["manifest_path"],
        screen_sha256=owner_info["manifest_sha256"],
        screen=owner_info["manifest"],
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
        replay_verifier=require_file(args.replay_verifier, "upper replay verifier"),
        output_dir=output,
        chunk_size=100_000,
        tool_path=TOOL_PATH,
        parameter_overrides=policy_parameter_overrides(owner_name, policy_name),
        extra_diagnostics={
            "research_candidate": "owner_policy_tournament",
            "combination": f"{owner_name}+{policy_name}",
            "owner_name": owner_name,
            "owner_semantic_sha256": owner_info["owner_semantic_sha256"],
            "owner_manifest_sha256": owner_info["manifest_sha256"],
            "multi_assignment_policy": policy_name,
            "assignment_metrics": metrics,
            "matrix_row_all_offline_gates_pass": selected.get(
                "all_offline_gates_pass"
            ),
            "online_matched_recall_qps_required": True,
            "canonical_default_eligible": False,
        },
        extra_provenance={
            "owner_policy_matrix": str(matrix_path),
            "owner_policy_matrix_sha256": scaling.sha256(matrix_path),
            "owner_evidence_manifest": str(owner_info["manifest_path"]),
            "owner_evidence_manifest_sha256": owner_info["manifest_sha256"],
            "tournament_materializer_source": str(Path(__file__).resolve()),
            "tournament_materializer_source_sha256": scaling.sha256(
                Path(__file__).resolve()
            ),
            "matrix_screen_source": str(Path(matrix.__file__).resolve()),
            "matrix_screen_source_sha256": scaling.sha256(
                Path(matrix.__file__).resolve()
            ),
            "local_vectors_source": str(vectors_source),
            "local_vectors_source_sha256": base.vectors_sha256,
            "local_vectors_source_reason": "project_filesystem_full_same_bytes_hardlink",
        },
        assignment_builder=assignment_builder(
            membership,
            expected_digest=layout_sha,
            histogram=histogram,
            shard_counts=shard_counts,
        ),
    )
    return {
        **result,
        "status": "PASS",
        "combination": f"{owner_name}+{policy_name}",
        "assignment_metrics": metrics,
        "matrix_row_all_offline_gates_pass": selected.get(
            "all_offline_gates_pass"
        ),
        "interpretation_boundary": {
            "online_matched_recall_qps_required": True,
            "offline_metrics_do_not_establish_qps": True,
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    owner = subparsers.add_parser("owner")
    owner.add_argument("--owner-name", choices=OWNER_NAMES, required=True)
    owner.add_argument("--matrix-screen", type=Path, default=DEFAULT_MATRIX)
    owner.add_argument("--phase-b-screen", type=Path)
    owner.add_argument("--output-dir", type=Path, required=True)

    bundle = subparsers.add_parser("bundle")
    bundle.add_argument("--owner-dir", type=Path, required=True)
    bundle.add_argument("--policy", choices=POLICY_NAMES, required=True)
    bundle.add_argument("--matrix-screen", type=Path, default=DEFAULT_MATRIX)
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
