#!/usr/bin/env python3
"""Read-only A/B of current C_CNBR multi-assignment versus frozen BMR_10.

Both arms keep the same C_CNBR owner, immutable upper graph, P=32 round-robin
placement, lower HNSW settings, CPU contract, and u48/upperEF48/base50/factor14
search budget.  The only intended treatment is the L0 multi-assignment rule
applied after that owner is frozen: current all-max replication versus BMR_10.
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
from experiments.c1.scripts import c1_orion_l1_owner_repair_interleaved_ab as ledger  # noqa: E402
from experiments.l1_balance import cnbr_construction_cost_gate as cost_gate  # noqa: E402
from experiments.l1_balance import native_cnbr_core  # noqa: E402
from experiments.multi_assignment import budgeted_policy  # noqa: E402
from experiments.multi_assignment import materialize_budgeted_candidate as bmr  # noqa: E402


benchmark_lock = base.benchmark_lock
ARM_NAMES = {"A": "C_CNBR_current", "B": "C_CNBR_BMR_10"}
CURRENT_TOOL = "experiments/l1_balance/materialize_native_cnbr_candidate.py"


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


def _frozen_phase_b_core(build_manifest: Mapping[str, Any]) -> Path:
    provenance = build_manifest.get("provenance")
    if not isinstance(provenance, Mapping):
        raise RuntimeError("C_CNBR/BMR bundle lacks provenance")
    formal_screen = Path(str(provenance.get("phase_b_screen") or "")).resolve()
    frozen_core = formal_screen.parent / "source" / "native_cnbr_core.py"
    if not frozen_core.is_file():
        raise FileNotFoundError(frozen_core)
    return frozen_core


def prepare_binding(
    arm: base.ArmSpec,
    *,
    base_url: str,
    deployment_manifest: Path,
    topology: Path,
) -> dict[str, Any]:
    """Accept current topology plus exact current artifact-ledger identity."""
    compatibility = replace(arm, allow_historical_prepare_deployment=True)
    build_manifest = base.load_json_object(arm.layout_dir / "build-manifest.json")
    original_core_file = native_cnbr_core.__file__
    try:
        native_cnbr_core.__file__ = str(_frozen_phase_b_core(build_manifest))
        binding = base.validate_prepare_binding(
            compatibility,
            base_url=base_url,
            deployment_manifest=deployment_manifest,
            topology=topology,
        )
    finally:
        native_cnbr_core.__file__ = original_core_file
    topology_binding = binding.get("prepare_topology_binding")
    if (
        not isinstance(topology_binding, Mapping)
        or topology_binding.get("mode") != "current"
    ):
        raise RuntimeError("BMR_10 A/B cannot excuse topology drift")
    # The ledger checker predates this orthogonal multi-assignment arm name.
    ledger_arm = replace(arm, name="C_CNBR")
    current_ledger = ledger.validate_current_deployment_artifact_ledger(
        ledger_arm, binding, deployment_manifest
    )
    deployment_binding = binding.get("prepare_deployment_binding")
    if not isinstance(deployment_binding, Mapping):
        raise RuntimeError("BMR_10 prepare deployment binding is missing")
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


def validate_multi_assignment_fairness(
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
        raise RuntimeError("multi-assignment A/B lacks frozen bundle diagnostics")

    common_keys = tuple(
        key
        for key in base.COMMON_BUILD_PARAMETER_KEYS
        if key != "multi_assign_max_shards"
    ) + (
        "initial_num_shards",
        "enable_fission",
        "enable_topology_refinement",
        "l0_repair",
        "l1_partitioner",
        "balance_mode",
    )
    parameter_mismatches = {
        key: {
            "A": a_binding["parameters"].get(key),
            "B": b_binding["parameters"].get(key),
        }
        for key in common_keys
        if a_binding["parameters"].get(key)
        != b_binding["parameters"].get(key)
    }
    b_parameters = b_binding["parameters"]
    exact_bmr_policy = {
        "multi_assign_max_shards": 2,
        "multi_assign_extra_copy_budget_numerator": 1,
        "multi_assign_extra_copy_budget_denominator": 10,
        "multi_assign_score": budgeted_policy.SCORE,
        "multi_assignment_policy": budgeted_policy.CANDIDATE_ID,
        "multi_assignment_policy_version": budgeted_policy.POLICY_VERSION,
    }
    equality_checks = {
        "same_upper_navigator": a_live["upper_navigator_sha256"]
        == b_live["upper_navigator_sha256"],
        "same_upper_graph_semantics": a_live["upper_graph_semantic_sha256"]
        == b_live["upper_graph_semantic_sha256"],
        "same_vector_schema": a_live["vector_schema"] == b_live["vector_schema"],
        "same_logical_shard_count": a_live["shard_count"] == b_live["shard_count"],
        "same_logical_point_count": a_live["logical_point_count"]
        == b_live["logical_point_count"],
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
        "same_frozen_cnbr_owner": (
            len(str(a_diag.get("owner_sha256") or "")) == 64
            and a_diag.get("owner_sha256") == b_diag.get("owner_sha256")
        ),
        "same_phase_a_manifest": a_diag.get("phase_a_manifest_sha256")
        == b_diag.get("phase_a_manifest_sha256"),
        "same_full_attachment_bytes": a_diag.get("attachments_sha256")
        == b_diag.get("attachments_sha256"),
        "same_source_upper_artifact": a_provenance.get("source_artifact_sha256")
        == b_provenance.get("source_artifact_sha256"),
        "same_l1_owner_sizes": a_diag.get("l1_sizes") == b_diag.get("l1_sizes"),
        "only_multi_assignment_policy_differs": (
            a_binding["parameters"].get("multi_assign_max_shards") == 0
            and all(b_parameters.get(key) == value for key, value in exact_bmr_policy.items())
        ),
        "bmr_reduces_physical_copies": b_live["physical_point_count"]
        < a_live["physical_point_count"],
    }
    if not all(equality_checks.values()):
        raise RuntimeError(
            "C_CNBR current/BMR_10 fairness contract failed: "
            + json.dumps(
                {
                    "checks": equality_checks,
                    "parameter_mismatches": parameter_mismatches,
                },
                sort_keys=True,
            )
        )
    return {
        "status": "PASS",
        "checks": equality_checks,
        "common_build_parameters": {
            key: a_binding["parameters"].get(key) for key in common_keys
        },
        "allowed_treatment": {
            "stage": "L0_multi_assignment_after_frozen_C_CNBR_owner",
            "A": "current_all_max",
            "B": budgeted_policy.CANDIDATE_ID,
        },
        "owner_sha256": a_diag["owner_sha256"],
        "upper_navigator_sha256": a_live["upper_navigator_sha256"],
        "upper_graph_semantic_sha256": a_live["upper_graph_semantic_sha256"],
        "logical_shard_count": a_live["shard_count"],
        "logical_point_count": a_live["logical_point_count"],
        "physical_point_count_by_arm": {
            label: live[label]["physical_point_count"] for label in base.ARM_LABELS
        },
        "numeric_shard_placement": a_live["expected_placement"],
    }


def validate_pair(
    arms: Mapping[str, base.ArmSpec],
    bindings: Mapping[str, Mapping[str, Any]],
    live: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    a_manifest = bindings["A"].get("build_manifest")
    b_manifest = bindings["B"].get("build_manifest")
    if not isinstance(a_manifest, Mapping) or not isinstance(b_manifest, Mapping):
        raise RuntimeError("BMR_10 pair lacks build manifests")
    a_diag = (a_manifest.get("routing") or {}).get("l1_partition_diagnostics")
    b_diag = (b_manifest.get("routing") or {}).get("l1_partition_diagnostics")
    if not isinstance(a_diag, Mapping) or not isinstance(b_diag, Mapping):
        raise RuntimeError("BMR_10 pair lacks L1 diagnostics")
    b_provenance = b_manifest.get("provenance") or {}
    if not isinstance(b_provenance, Mapping):
        raise RuntimeError("BMR_10 pair lacks provenance")
    selection_path = Path(
        str(b_provenance.get("bmr10_selection_manifest") or "")
    ).resolve()
    selection = base.load_json_object(selection_path)
    b_proof = (live["B"].get("artifact_bundle") or {}).get(
        "l1_partition_layout_proof"
    )
    if not isinstance(b_proof, Mapping):
        raise RuntimeError("BMR_10 live bundle proof is missing")
    assignment = b_proof.get("assignment")
    if not isinstance(assignment, Mapping):
        raise RuntimeError("BMR_10 assignment proof is missing")
    checks = {
        "arm_a_is_current_cnbr": (
            a_manifest.get("tool") == CURRENT_TOOL
            and a_manifest.get("parameters", {}).get("l1_partitioner") == "C_CNBR"
            and a_manifest.get("parameters", {}).get("multi_assign_max_shards") == 0
        ),
        "arm_b_is_frozen_bmr10": (
            b_manifest.get("tool") == bmr.TOOL_PATH
            and b_manifest.get("parameters", {}).get("l1_partitioner") == "C_CNBR"
            and b_manifest.get("parameters", {}).get("multi_assignment_policy")
            == budgeted_policy.CANDIDATE_ID
            and b_diag.get("multi_assignment_contract", {}).get(
                "canonical_default_eligible"
            )
            is False
        ),
        "same_owner": a_diag.get("owner_sha256") == b_diag.get("owner_sha256"),
        "same_source_phase_a": a_diag.get("phase_a_manifest_sha256")
        == b_diag.get("phase_a_manifest_sha256"),
        "same_full_attachment_bytes": a_diag.get("attachments_sha256")
        == b_diag.get("attachments_sha256"),
        "selection_is_frozen_without_measurement_queries": (
            base.sha256_path(selection_path)
            == b_provenance.get("bmr10_selection_manifest_sha256")
            and selection.get("status") == "PASS"
            and selection.get("selected_candidate") == budgeted_policy.CANDIDATE_ID
            and selection.get("glove_online_measurement_queries_seen") is False
            and selection.get("strict_glove_holdout_claim") is False
            and selection.get("no_fallback_retuning") is True
        ),
        "exact_10pct_copy_budget": assignment.get("extra_copy_count")
        == assignment.get("exact_extra_copy_budget"),
        "maximum_two_copies": assignment.get("maximum_copies_per_point") == 2,
        "lower_index_expansion": live["B"]["physical_point_count"]
        < live["A"]["physical_point_count"],
        "current_deployment_artifact_ledgers_bound": all(
            (bindings[label].get("current_deployment_artifact_ledger") or {}).get(
                "status"
            )
            == "PASS"
            for label in base.ARM_LABELS
        ),
        "fixed_p32": all(
            live[label]["shard_count"] == 32 for label in base.ARM_LABELS
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(
            "C_CNBR current/BMR_10 identity contract failed: "
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
        "selection_sha256": b_provenance["bmr10_selection_manifest_sha256"],
    }


def contract() -> base.InterleavedABContract:
    return base.InterleavedABContract(
        arm_specs_factory=arm_specs,
        fairness_validator=validate_multi_assignment_fairness,
        comparison_validator=validate_pair,
        comparison_manifest_key="cnbr_current_bmr10_pair",
        comparison_completion_key="cnbr_current_bmr10_pair_bound",
        fairness_budget_check_key="same_frozen_historical_budget",
        fairness_budget_completion_key="same_frozen_historical_budget",
        summary_record_type="c1_orion_budgeted_multi_assignment_ab_summary",
        run_record_type="c1_orion_budgeted_multi_assignment_interleaved_ab",
        budget_mode="frozen_u48_upper_ef48_base50_factor14_current_vs_bmr10",
        budget_subject="the identical frozen Orion search budget",
        adoption_pass_decision="BMR10_QPS_GATE_PASS",
        adoption_fail_decision="RETAIN_CURRENT_CNBR_MULTI_ASSIGNMENT",
        adoption_scope=(
            "Four-physical-host GloVe/Cosine P=32 matched-recall evidence for "
            "the post-owner BMR_10 multi-assignment treatment only."
        ),
        decision_gate_key="bmr10_online_qps_gate",
        semantic_ratio_key="paired_ratio_bmr10_over_current",
        completion_error_subject="C_CNBR current/BMR_10 interleaved A/B",
        prepare_binding_validator=prepare_binding,
    )


def validate_args(args: argparse.Namespace) -> None:
    if {"A": args.arm_a_name, "B": args.arm_b_name} != ARM_NAMES:
        raise ValueError("arm names are fixed to current C_CNBR and C_CNBR BMR_10")
    base.validate_args(args)


def run(args: argparse.Namespace) -> Path:
    validate_args(args)
    output = Path(args.output_dir).expanduser().resolve()
    existed = output.exists()
    with benchmark_lock.hold_from_args(
        args,
        args.deployment_manifest,
        owner={
            "kind": "c1_orion_budgeted_multi_assignment_interleaved_ab",
            "collection_a": args.collection_a,
            "collection_b": args.collection_b,
            "output_dir": str(output),
        },
    ) as held:
        original_bundle_validator = base.validate_artifact_bundle_for_arm

        def frozen_source_bundle_validator(
            arm: base.ArmSpec,
            binding: Mapping[str, Any],
            artifact_proof: Mapping[str, Any],
        ) -> dict[str, Any]:
            original_core_file = native_cnbr_core.__file__
            original_current_source_validator = (
                cost_gate._validate_current_source_bindings
            )
            try:
                native_cnbr_core.__file__ = str(
                    _frozen_phase_b_core(binding["build_manifest"])
                )
                # The historical audit and frozen source snapshot remain bound;
                # only comparison against today's additive BMR working tree is
                # inapplicable to the already-frozen C_CNBR owner decision.
                cost_gate._validate_current_source_bindings = lambda _audit: None
                return original_bundle_validator(arm, binding, artifact_proof)
            finally:
                native_cnbr_core.__file__ = original_core_file
                cost_gate._validate_current_source_bindings = (
                    original_current_source_validator
                )

        base.validate_artifact_bundle_for_arm = frozen_source_bundle_validator
        try:
            return base._run_locked(args, held, contract=contract())
        except BaseException as exc:
            if not existed and output.is_dir():
                try:
                    base.write_json(
                        output / "execution-failed.json",
                        {
                            "timestamp": base.utc_timestamp(),
                            "record_type": (
                                "c1_orion_budgeted_multi_assignment_ab_failure"
                            ),
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
