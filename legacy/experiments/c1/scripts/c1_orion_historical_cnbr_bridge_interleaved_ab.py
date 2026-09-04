#!/usr/bin/env python3
"""Read-only historical H versus C_CNBR deployment-performance bridge.

Arm A is the exact historical generation-3248141 P24-to-fission32 Orion
deployment anchor (H). Arm B is the canonical fixed-P32 C_CNBR layout. This
comparison is descriptive only: it measures how much historical deployment
performance C_CNBR recovers, but it cannot establish CNBR's causal effect,
select a fallback, or make H eligible for adoption.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.c1.scripts import (  # noqa: E402
    c1_orion_balance_interleaved_ab as base,
)
from experiments.c1.scripts import (  # noqa: E402
    c1_orion_l1_owner_repair_interleaved_ab as native_cnbr,
)


benchmark_lock = base.benchmark_lock
ARM_LABELS = base.ARM_LABELS
FORMAL_LOGICAL_SHARDS = base.FORMAL_LOGICAL_SHARDS
HISTORICAL_BUDGET = base.HISTORICAL_BUDGET
TARGET_RECALL = base.TARGET_RECALL
utc_timestamp = base.utc_timestamp
load_json_object = base.load_json_object


def historical_cnbr_arm_specs(
    args: argparse.Namespace,
) -> dict[str, base.ArmSpec]:
    return {
        "A": base.ArmSpec(
            label="A",
            name=args.arm_a_name,
            collection=args.collection_a,
            artifact=args.artifact_a.expanduser().resolve(),
            layout_dir=args.layout_dir_a.expanduser().resolve(),
            prepare_manifest=args.prepare_manifest_a.expanduser().resolve(),
            allow_balance_layout=False,
            allow_scaling_layout=True,
            allow_l1_partition_layout=False,
            allow_historical_prepare_deployment=True,
        ),
        "B": base.ArmSpec(
            label="B",
            name=args.arm_b_name,
            collection=args.collection_b,
            artifact=args.artifact_b.expanduser().resolve(),
            layout_dir=args.layout_dir_b.expanduser().resolve(),
            prepare_manifest=args.prepare_manifest_b.expanduser().resolve(),
            allow_balance_layout=False,
            allow_scaling_layout=False,
            allow_l1_partition_layout=True,
            allow_historical_prepare_deployment=False,
        ),
    }


def validate_historical_cnbr_fairness(
    bindings: Mapping[str, Mapping[str, Any]],
    live: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    proof = base.validate_cross_arm_fairness(bindings, live)
    exact_multi_assignment = all(
        bindings[label]["parameters"].get("use_multi_assign") is True
        and bindings[label]["parameters"].get("multi_assign_min_max_vote") == 2
        and bindings[label]["parameters"].get("multi_assign_vote_delta") == 0
        and bindings[label]["parameters"].get("multi_assign_max_shards") == 0
        for label in ARM_LABELS
    )
    if not exact_multi_assignment:
        raise RuntimeError(
            "historical H/C_CNBR bridge requires identical frozen "
            "multi-assignment parameters"
        )
    proof["checks"]["exact_multi_assignment_in_both"] = True
    return proof


def validate_historical_cnbr_pair(
    arms: Mapping[str, base.ArmSpec],
    bindings: Mapping[str, Mapping[str, Any]],
    live: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    historical = base.validate_adjustment_before_baseline(
        arms["A"], bindings["A"], live["A"]
    )
    diagnostics = native_cnbr.l1_partition_diagnostics(bindings["B"], "B")
    variant = native_cnbr.balance_variant(diagnostics, "B")
    phase_a_manifest = native_cnbr.diagnostic_sha256(
        diagnostics, "phase_a_manifest_sha256", "B"
    )
    phase_a_owner = native_cnbr.diagnostic_sha256(
        diagnostics, "phase_a_owner_sha256", "B"
    )
    owner = native_cnbr.diagnostic_sha256(diagnostics, "owner_sha256", "B")
    attachments = native_cnbr.diagnostic_sha256(
        diagnostics, "attachments_sha256", "B"
    )
    parent_owner = native_cnbr.cnbr_parent_owner_sha256(diagnostics)
    parent_manifest = native_cnbr.cnbr_parent_manifest_sha256(diagnostics)
    forbidden = native_cnbr.phase_a_forbidden_invocations(diagnostics, "B")
    parameters = bindings["B"]["parameters"]
    routing = bindings["B"]["routing"]

    checks = {
        "arm_a_is_exact_historical_h": historical["status"] == "PASS",
        "arm_b_is_l1_partition_layout": arms["B"].allow_l1_partition_layout,
        "arm_b_has_no_historical_prepare_exception": not arms[
            "B"
        ].allow_historical_prepare_deployment,
        "arm_b_is_c_cnbr": variant == native_cnbr.CNBR_BALANCE_VARIANT,
        "arm_b_phase_a_owner_matches_final_owner": phase_a_owner == owner,
        "arm_b_parent_manifest_matches_phase_a": parent_manifest
        == phase_a_manifest,
        "arm_b_parent_n_owner_is_distinct_checksum": parent_owner != owner,
        "arm_b_fixed_initial_and_final_p32": parameters.get(
            "initial_num_shards"
        )
        == FORMAL_LOGICAL_SHARDS
        and routing.get("effective_num_shards") == FORMAL_LOGICAL_SHARDS
        and live["B"]["shard_count"] == FORMAL_LOGICAL_SHARDS,
        "arm_b_fission_disabled": parameters.get("enable_fission") is False
        and routing.get("fission_events") == [],
        "arm_b_topology_refinement_disabled": parameters.get(
            "enable_topology_refinement"
        )
        is False,
        "arm_b_l0_repair_disabled": parameters.get("l0_repair") is False,
        "arm_b_forbidden_phase_a_invocations_zero": all(
            forbidden[key] == 0
            for key in native_cnbr.PHASE_A_FORBIDDEN_INVOCATION_KEYS
        ),
        "arm_b_multi_assignment_after_owner_freeze": diagnostics.get(
            "multi_assignment_after_owner_freeze"
        )
        is True,
    }
    if not all(checks.values()):
        raise RuntimeError(
            "historical H/C_CNBR bridge identity contract failed: "
            + json.dumps(checks, sort_keys=True)
        )
    return {
        "status": "PASS",
        "roles": {"A": "H", "B": "C_CNBR"},
        "causal_interpretation_allowed": False,
        "adoption_or_fallback_decision_allowed": False,
        "checks": checks,
        "historical_h": historical,
        "c_cnbr": {
            "balance_variant": variant,
            "phase_a_manifest_sha256": phase_a_manifest,
            "phase_a_owner_sha256": phase_a_owner,
            "owner_sha256": owner,
            "parent_n_owner_sha256": parent_owner,
            "attachments_sha256": attachments,
            "forbidden_stage_invocations": forbidden,
        },
    }


def validate_bridge_construction_cost(
    args: argparse.Namespace,
    bindings: Mapping[str, Mapping[str, Any]],
    comparison_validation: Mapping[str, Any],
) -> dict[str, Any]:
    """Require the same cost-qualified C bundle for the descriptive bridge."""

    candidate = native_cnbr.validate_candidate_construction_cost_binding(
        args, bindings["B"]
    )
    role_check = comparison_validation.get("roles") == {
        "A": "H",
        "B": "C_CNBR",
    }
    if not role_check:
        raise RuntimeError("historical H/C_CNBR cost binding role drifted")
    return {
        "status": "PASS",
        "record_type": "historical_h_cnbr_bridge_prerequisites",
        "checks": {
            **candidate["checks"],
            "fixed_bridge_roles_reconfirmed": role_check,
        },
        "construction_cost_v4": candidate["construction_cost_v4"],
    }


def historical_cnbr_contract() -> base.InterleavedABContract:
    return base.InterleavedABContract(
        arm_specs_factory=historical_cnbr_arm_specs,
        fairness_validator=validate_historical_cnbr_fairness,
        comparison_validator=validate_historical_cnbr_pair,
        comparison_manifest_key="historical_h_cnbr_bridge",
        comparison_completion_key="historical_h_cnbr_bridge_bound",
        fairness_budget_check_key="same_frozen_historical_budget",
        fairness_budget_completion_key="same_frozen_historical_budget",
        summary_record_type="c1_orion_historical_cnbr_bridge_summary",
        run_record_type="c1_orion_historical_cnbr_bridge",
        budget_mode="frozen_u48_upper_ef48_base50_factor14_h_cnbr_bridge",
        budget_subject="the frozen symmetric H/C_CNBR search budget",
        adoption_pass_decision="C_CNBR_SIGNIFICANTLY_EXCEEDS_H_QPS",
        adoption_fail_decision="C_CNBR_DOES_NOT_SIGNIFICANTLY_EXCEED_H_QPS",
        adoption_scope=(
            "Historical deployment-performance bridge only. This result cannot "
            "establish CNBR causality, select a compliant fallback, alter the "
            "N_native/C_CNBR adoption decision, or make H eligible."
        ),
        decision_gate_key="historical_c_over_h_recovery_gate",
        semantic_ratio_key="paired_ratio_c_over_h",
        completion_error_subject="historical H/C_CNBR interleaved bridge",
        adoption_prerequisite_validator=validate_bridge_construction_cost,
    )


def validate_args(args: argparse.Namespace) -> None:
    if getattr(args, "construction_cost_audit", None) in (None, ""):
        raise ValueError("H/C_CNBR bridge requires --construction-cost-audit")
    base.validate_args(args)


def run(args: argparse.Namespace) -> Path:
    validate_args(args)
    native_cnbr.cost_gate.validate_construction_cost_v4_pass(
        args.construction_cost_audit
    )
    output = Path(args.output_dir).expanduser().resolve()
    output_preexisted = output.exists()
    with benchmark_lock.hold_from_args(
        args,
        args.deployment_manifest,
        owner={
            "kind": "c1_orion_historical_cnbr_bridge_interleaved_ab",
            "collection_a": args.collection_a,
            "collection_b": args.collection_b,
            "output_dir": str(output),
        },
    ) as held_lock:
        try:
            return base._run_locked(
                args,
                held_lock,
                contract=historical_cnbr_contract(),
            )
        except BaseException as exc:
            if not output_preexisted and output.is_dir():
                try:
                    base.write_json(
                        output / "execution-failed.json",
                        {
                            "timestamp": base.utc_timestamp(),
                            "record_type": (
                                "c1_orion_historical_cnbr_bridge_failure"
                            ),
                            "status": "FAIL",
                            "error": repr(exc),
                            "benchmark_lock": held_lock.evidence(),
                        },
                    )
                except Exception:
                    pass
            raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--hdf5-path", type=Path, required=True)
    parser.add_argument("--topology", type=Path, required=True)
    parser.add_argument("--deployment-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--arm-a-name", default="H_historical")
    parser.add_argument("--arm-b-name", default="C_CNBR")
    parser.add_argument("--collection-a", required=True)
    parser.add_argument("--collection-b", required=True)
    parser.add_argument("--artifact-a", type=Path, required=True)
    parser.add_argument("--artifact-b", type=Path, required=True)
    parser.add_argument("--layout-dir-a", type=Path, required=True)
    parser.add_argument("--layout-dir-b", type=Path, required=True)
    parser.add_argument("--prepare-manifest-a", type=Path, required=True)
    parser.add_argument("--prepare-manifest-b", type=Path, required=True)
    parser.add_argument("--construction-cost-audit", type=Path, required=True)
    parser.add_argument("--target-recall", type=float, default=TARGET_RECALL)
    parser.add_argument(
        "--concurrency-candidates", default="1,2,4,8,16,32,64"
    )
    parser.add_argument("--sweep-seconds", type=float, default=8.0)
    parser.add_argument("--warmup-seconds", type=float, default=10.0)
    benchmark_lock.add_cli_arguments(parser)
    args = parser.parse_args(argv)
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
