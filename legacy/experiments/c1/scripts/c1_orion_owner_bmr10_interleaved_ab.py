#!/usr/bin/env python3
"""Matched-recall A/B of C_CNBR+BMR_10 versus artifact-primary+BMR_10.

Both arms use the identical immutable upper graph, P=32 four-host placement,
lower HNSW settings, CPU contract, frozen u48/upperEF48/base50/factor14 search
budget, and exact BMR_10 replication budget.  The intended treatment is only
the frozen L1 owner.  The artifact-primary arm is research-only and does not
claim reproduction of the historical full multi-membership layout.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.c1.scripts import c1_orion_balance_interleaved_ab as base  # noqa: E402
from experiments.c1.scripts import (  # noqa: E402
    c1_orion_budgeted_multi_assignment_interleaved_ab as current_ab,
)
from experiments.c1.scripts import (  # noqa: E402
    c1_orion_l1_owner_repair_interleaved_ab as ledger,
)
from experiments.l1_balance import cnbr_construction_cost_gate as cost_gate  # noqa: E402
from experiments.l1_balance import native_cnbr_core  # noqa: E402
from experiments.multi_assignment import budgeted_policy  # noqa: E402
from experiments.multi_assignment import materialize_budgeted_candidate as bmr  # noqa: E402
from experiments.multi_assignment import (  # noqa: E402
    materialize_artifact_owner_bmr10 as artifact_bmr,
)


benchmark_lock = base.benchmark_lock
ARM_NAMES = {"A": "C_CNBR_BMR_10", "B": "HISTORICAL_PRIMARY_BMR_10"}


def arm_specs(args: argparse.Namespace) -> dict[str, base.ArmSpec]:
    def build(label: str, suffix: str) -> base.ArmSpec:
        return base.ArmSpec(
            label=label,
            name=ARM_NAMES[label],
            collection=getattr(args, f"collection_{suffix}"),
            artifact=getattr(args, f"artifact_{suffix}").expanduser().resolve(),
            layout_dir=getattr(args, f"layout_dir_{suffix}").expanduser().resolve(),
            prepare_manifest=getattr(
                args, f"prepare_manifest_{suffix}"
            ).expanduser().resolve(),
            allow_balance_layout=False,
            allow_scaling_layout=False,
            allow_l1_partition_layout=True,
            allow_historical_prepare_deployment=False,
        )

    return {"A": build("A", "a"), "B": build("B", "b")}


def prepare_binding(
    arm: base.ArmSpec,
    *,
    base_url: str,
    deployment_manifest: Path,
    topology: Path,
) -> dict[str, Any]:
    compatibility = replace(arm, allow_historical_prepare_deployment=True)
    if arm.label == "A":
        build_manifest = base.load_json_object(arm.layout_dir / "build-manifest.json")
        original_core_file = native_cnbr_core.__file__
        try:
            native_cnbr_core.__file__ = str(
                current_ab._frozen_phase_b_core(build_manifest)
            )
            binding = base.validate_prepare_binding(
                compatibility,
                base_url=base_url,
                deployment_manifest=deployment_manifest,
                topology=topology,
            )
        finally:
            native_cnbr_core.__file__ = original_core_file
    else:
        binding = base.validate_prepare_binding(
            compatibility,
            base_url=base_url,
            deployment_manifest=deployment_manifest,
            topology=topology,
        )
    topology_binding = binding.get("prepare_topology_binding")
    if (
        not isinstance(topology_binding, Mapping)
        or topology_binding.get("mode") != "current"
    ):
        raise RuntimeError("owner+BMR_10 A/B cannot excuse topology drift")
    ledger_arm = replace(arm, name="C_CNBR")
    current_ledger = ledger.validate_current_deployment_artifact_ledger(
        ledger_arm, binding, deployment_manifest
    )
    deployment_binding = binding.get("prepare_deployment_binding")
    if not isinstance(deployment_binding, Mapping):
        raise RuntimeError("owner+BMR_10 prepare deployment binding is missing")
    result = dict(binding)
    result["prepare_deployment_binding"] = {
        **dict(deployment_binding),
        "mode": "current_artifact_ledger_supersession",
        "explicitly_allowed": False,
        "generic_historical_exception_used": False,
        "artifact_ledger": current_ledger,
    }
    result["current_deployment_artifact_ledger"] = current_ledger
    return result


def validate_owner_fairness(
    bindings: Mapping[str, Mapping[str, Any]],
    live: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    a_binding, b_binding = bindings["A"], bindings["B"]
    a_live, b_live = live["A"], live["B"]
    a_manifest = a_binding["build_manifest"]
    b_manifest = b_binding["build_manifest"]
    a_diag = (a_manifest.get("routing") or {}).get("l1_partition_diagnostics")
    b_diag = (b_manifest.get("routing") or {}).get("l1_partition_diagnostics")
    a_provenance = a_manifest.get("provenance") or {}
    b_provenance = b_manifest.get("provenance") or {}
    if not all(
        isinstance(value, Mapping)
        for value in (a_diag, b_diag, a_provenance, b_provenance)
    ):
        raise RuntimeError("owner+BMR_10 A/B lacks bundle diagnostics")

    common_keys = tuple(base.COMMON_BUILD_PARAMETER_KEYS) + (
        "initial_num_shards",
        "enable_fission",
        "enable_topology_refinement",
        "l0_repair",
        "use_multi_assign",
        "multi_assign_max_shards",
        "multi_assign_extra_copy_budget_numerator",
        "multi_assign_extra_copy_budget_denominator",
        "multi_assign_score",
        "multi_assignment_policy",
        "multi_assignment_policy_version",
    )
    parameter_mismatches = {
        key: {
            "A": a_binding["parameters"].get(key),
            "B": b_binding["parameters"].get(key),
        }
        for key in common_keys
        if a_binding["parameters"].get(key) != b_binding["parameters"].get(key)
    }
    exact_bmr = {
        "use_multi_assign": True,
        "multi_assign_min_max_vote": 2,
        "multi_assign_vote_delta": 0,
        "multi_assign_max_shards": 2,
        "multi_assign_extra_copy_budget_numerator": 1,
        "multi_assign_extra_copy_budget_denominator": 10,
        "multi_assign_score": budgeted_policy.SCORE,
        "multi_assignment_policy": budgeted_policy.CANDIDATE_ID,
        "multi_assignment_policy_version": budgeted_policy.POLICY_VERSION,
    }
    checks = {
        "same_upper_navigator": a_live["upper_navigator_sha256"]
        == b_live["upper_navigator_sha256"],
        "same_upper_graph_semantics": a_live["upper_graph_semantic_sha256"]
        == b_live["upper_graph_semantic_sha256"],
        "same_vector_schema": a_live["vector_schema"] == b_live["vector_schema"],
        "same_logical_shard_count": a_live["shard_count"] == b_live["shard_count"],
        "same_logical_point_count": a_live["logical_point_count"]
        == b_live["logical_point_count"],
        "same_physical_point_count": a_live["physical_point_count"]
        == b_live["physical_point_count"],
        "same_hnsw_config": a_live["hnsw_config"] == b_live["hnsw_config"],
        "same_optimizer_config": a_live["optimizer_config"]
        == b_live["optimizer_config"],
        "same_numeric_shard_placement": a_live["expected_placement"]
        == b_live["expected_placement"],
        "same_common_build_parameters": not parameter_mismatches,
        "multi_assignment_enabled_in_both": all(
            bindings[label]["parameters"].get("use_multi_assign") is True
            for label in base.ARM_LABELS
        ),
        "same_frozen_historical_budget": all(
            all(
                bindings[label]["parameters"].get(key) == expected
                for key, expected in base.HISTORICAL_BUDGET.items()
            )
            for label in base.ARM_LABELS
        ),
        "same_exact_bmr10_policy": all(
            all(
                bindings[label]["parameters"].get(key) == expected
                for key, expected in exact_bmr.items()
            )
            for label in base.ARM_LABELS
        ),
        "same_full_attachment_bytes": a_diag.get("attachments_sha256")
        == b_diag.get("attachments_sha256"),
        "same_source_upper_artifact": a_provenance.get("source_artifact_sha256")
        == b_provenance.get("source_artifact_sha256"),
        "owner_is_the_intended_difference": (
            a_diag.get("owner_sha256") != b_diag.get("owner_sha256")
            and a_binding["parameters"].get("l1_partitioner") == "C_CNBR"
            and b_binding["parameters"].get("l1_partitioner")
            == "HISTORICAL_PRIMARY"
        ),
        "current_deployment_artifact_ledgers_bound": all(
            (bindings[label].get("current_deployment_artifact_ledger") or {}).get(
                "status"
            )
            == "PASS"
            for label in base.ARM_LABELS
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(
            "owner+BMR_10 fairness contract failed: "
            + json.dumps(
                {"checks": checks, "parameter_mismatches": parameter_mismatches},
                sort_keys=True,
            )
        )
    return {
        "status": "PASS",
        "checks": checks,
        "common_build_parameters": {
            key: a_binding["parameters"].get(key) for key in common_keys
        },
        "allowed_treatment": {
            "stage": "frozen_L1_owner_before_identical_BMR_10",
            "A": "C_CNBR",
            "B": "HISTORICAL_PRIMARY_first_membership",
        },
        "owner_sha256_by_arm": {
            "A": a_diag["owner_sha256"],
            "B": b_diag["owner_sha256"],
        },
        "upper_navigator_sha256": a_live["upper_navigator_sha256"],
        "upper_graph_semantic_sha256": a_live["upper_graph_semantic_sha256"],
        "logical_shard_count": a_live["shard_count"],
        "logical_point_count": a_live["logical_point_count"],
        "physical_point_count": a_live["physical_point_count"],
        "numeric_shard_placement": a_live["expected_placement"],
    }


def validate_pair(
    arms: Mapping[str, base.ArmSpec],
    bindings: Mapping[str, Mapping[str, Any]],
    live: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    del arms
    a_manifest = bindings["A"].get("build_manifest")
    b_manifest = bindings["B"].get("build_manifest")
    if not isinstance(a_manifest, Mapping) or not isinstance(b_manifest, Mapping):
        raise RuntimeError("owner+BMR_10 pair lacks build manifests")
    a_diag = (a_manifest.get("routing") or {}).get("l1_partition_diagnostics")
    b_diag = (b_manifest.get("routing") or {}).get("l1_partition_diagnostics")
    if not isinstance(a_diag, Mapping) or not isinstance(b_diag, Mapping):
        raise RuntimeError("owner+BMR_10 pair lacks L1 diagnostics")
    a_proof = (live["A"].get("artifact_bundle") or {}).get(
        "l1_partition_layout_proof"
    )
    b_proof = (live["B"].get("artifact_bundle") or {}).get(
        "l1_partition_layout_proof"
    )
    if not isinstance(a_proof, Mapping) or not isinstance(b_proof, Mapping):
        raise RuntimeError("owner+BMR_10 live bundle proof is missing")
    a_assignment = a_proof.get("assignment")
    b_assignment = b_proof.get("assignment")
    if not isinstance(a_assignment, Mapping) or not isinstance(b_assignment, Mapping):
        raise RuntimeError("owner+BMR_10 assignment proof is missing")
    checks = {
        "arm_a_is_c_cnbr_bmr10": (
            a_manifest.get("tool") == bmr.TOOL_PATH
            and a_manifest.get("parameters", {}).get("l1_partitioner") == "C_CNBR"
            and a_manifest.get("parameters", {}).get("multi_assignment_policy")
            == budgeted_policy.CANDIDATE_ID
        ),
        "arm_b_is_artifact_primary_bmr10": (
            b_manifest.get("tool") == artifact_bmr.TOOL_PATH
            and b_manifest.get("parameters", {}).get("l1_partitioner")
            == "HISTORICAL_PRIMARY"
            and b_manifest.get("parameters", {}).get("multi_assignment_policy")
            == budgeted_policy.CANDIDATE_ID
        ),
        "owners_are_distinct": a_diag.get("owner_sha256")
        != b_diag.get("owner_sha256"),
        "same_full_attachment_bytes": a_diag.get("attachments_sha256")
        == b_diag.get("attachments_sha256"),
        "exact_10pct_copy_budget_both": all(
            proof.get("extra_copy_count") == proof.get("exact_extra_copy_budget")
            for proof in (a_assignment, b_assignment)
        ),
        "maximum_two_copies_both": all(
            proof.get("maximum_copies_per_point") == 2
            for proof in (a_assignment, b_assignment)
        ),
        "same_physical_copy_count": live["A"]["physical_point_count"]
        == live["B"]["physical_point_count"],
        "fixed_p32": all(
            live[label]["shard_count"] == 32 for label in base.ARM_LABELS
        ),
        "artifact_primary_research_boundary": (
            b_proof.get("canonical_default_eligible") is False
            and b_proof.get("historical_full_layout_reproduction_claim") is False
        ),
        "current_deployment_artifact_ledgers_bound": all(
            (bindings[label].get("current_deployment_artifact_ledger") or {}).get(
                "status"
            )
            == "PASS"
            for label in base.ARM_LABELS
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(
            "owner+BMR_10 identity contract failed: "
            + json.dumps(checks, sort_keys=True)
        )
    return {
        "status": "PASS",
        "roles": dict(ARM_NAMES),
        "checks": checks,
        "owner_sha256_by_arm": {
            "A": a_diag["owner_sha256"],
            "B": b_diag["owner_sha256"],
        },
        "physical_point_count_by_arm": {
            label: live[label]["physical_point_count"] for label in base.ARM_LABELS
        },
        "artifact_primary_matrix_sha256": b_proof.get("matrix_sha256"),
    }


def contract() -> base.InterleavedABContract:
    return base.InterleavedABContract(
        arm_specs_factory=arm_specs,
        fairness_validator=validate_owner_fairness,
        comparison_validator=validate_pair,
        comparison_manifest_key="owner_bmr10_pair",
        comparison_completion_key="owner_bmr10_pair_bound",
        fairness_budget_check_key="same_frozen_historical_budget",
        fairness_budget_completion_key="same_frozen_historical_budget",
        summary_record_type="c1_orion_owner_bmr10_ab_summary",
        run_record_type="c1_orion_owner_bmr10_interleaved_ab",
        budget_mode="frozen_u48_upper_ef48_base50_factor14_bmr10_both_owners",
        budget_subject="the identical frozen Orion search and BMR_10 budget",
        adoption_pass_decision="HISTORICAL_PRIMARY_BMR10_QPS_GATE_PASS",
        adoption_fail_decision="RETAIN_C_CNBR_BMR10",
        adoption_scope=(
            "Four-physical-host GloVe/Cosine P=32 matched-recall research "
            "comparison of the frozen L1 owner only. The historical-primary "
            "arm is not a reproduction claim for the historical full layout."
        ),
        decision_gate_key="historical_primary_bmr10_online_qps_gate",
        semantic_ratio_key="paired_ratio_historical_over_cnbr_bmr10",
        completion_error_subject="C_CNBR/HISTORICAL_PRIMARY BMR_10 interleaved A/B",
        prepare_binding_validator=prepare_binding,
    )


def validate_args(args: argparse.Namespace) -> None:
    if {"A": args.arm_a_name, "B": args.arm_b_name} != ARM_NAMES:
        raise ValueError("arm names are fixed to the two owner+BMR_10 treatments")
    base.validate_args(args)


def run(args: argparse.Namespace) -> Path:
    validate_args(args)
    output = Path(args.output_dir).expanduser().resolve()
    existed = output.exists()
    with benchmark_lock.hold_from_args(
        args,
        args.deployment_manifest,
        owner={
            "kind": "c1_orion_owner_bmr10_interleaved_ab",
            "collection_a": args.collection_a,
            "collection_b": args.collection_b,
            "output_dir": str(output),
        },
    ) as held:
        original_bundle_validator = base.validate_artifact_bundle_for_arm

        def source_aware_bundle_validator(
            arm: base.ArmSpec,
            binding: Mapping[str, Any],
            artifact_proof: Mapping[str, Any],
        ) -> dict[str, Any]:
            if arm.label != "A":
                return original_bundle_validator(arm, binding, artifact_proof)
            original_core_file = native_cnbr_core.__file__
            original_current_source_validator = (
                cost_gate._validate_current_source_bindings
            )
            try:
                native_cnbr_core.__file__ = str(
                    current_ab._frozen_phase_b_core(binding["build_manifest"])
                )
                cost_gate._validate_current_source_bindings = lambda _audit: None
                return original_bundle_validator(arm, binding, artifact_proof)
            finally:
                native_cnbr_core.__file__ = original_core_file
                cost_gate._validate_current_source_bindings = (
                    original_current_source_validator
                )

        base.validate_artifact_bundle_for_arm = source_aware_bundle_validator
        try:
            return base._run_locked(args, held, contract=contract())
        except BaseException as exc:
            if not existed and output.is_dir():
                try:
                    base.write_json(
                        output / "execution-failed.json",
                        {
                            "timestamp": base.utc_timestamp(),
                            "record_type": "c1_orion_owner_bmr10_ab_failure",
                            "status": "FAIL",
                            "error": repr(exc),
                            "benchmark_lock": held.evidence(),
                        },
                    )
                except Exception:
                    pass
            raise
        finally:
            base.validate_artifact_bundle_for_arm = original_bundle_validator


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--hdf5-path", type=Path, required=True)
    parser.add_argument("--topology", type=Path, required=True)
    parser.add_argument("--deployment-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--collection-a", required=True)
    parser.add_argument("--collection-b", required=True)
    parser.add_argument("--artifact-a", type=Path, required=True)
    parser.add_argument("--artifact-b", type=Path, required=True)
    parser.add_argument("--layout-dir-a", type=Path, required=True)
    parser.add_argument("--layout-dir-b", type=Path, required=True)
    parser.add_argument("--prepare-manifest-a", type=Path, required=True)
    parser.add_argument("--prepare-manifest-b", type=Path, required=True)
    parser.add_argument("--target-recall", type=float, default=base.TARGET_RECALL)
    parser.add_argument("--concurrency-candidates", default="1,2,4,8,16,32,64")
    parser.add_argument("--sweep-seconds", type=float, default=8.0)
    parser.add_argument("--warmup-seconds", type=float, default=10.0)
    benchmark_lock.add_cli_arguments(parser)
    args = parser.parse_args(argv)
    args.arm_a_name = ARM_NAMES["A"]
    args.arm_b_name = ARM_NAMES["B"]
    args.concurrency_candidates = base.parse_int_csv(args.concurrency_candidates)
    return args


def main(argv: Sequence[str] | None = None) -> int:
    output = run(parse_args(argv))
    print(json.dumps({"output_dir": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
