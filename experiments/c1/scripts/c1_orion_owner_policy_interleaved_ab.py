#!/usr/bin/env python3
"""Held-out interleaved A/B for two frozen owner-policy finalists."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.c1.scripts import c1_orion_balance_interleaved_ab as base  # noqa: E402
from experiments.c1.scripts import (  # noqa: E402
    c1_orion_owner_policy_tuning_screen as tuning_screen,
)
from experiments.multi_assignment import materialize_owner_policy_candidate as candidate  # noqa: E402


benchmark_lock = base.benchmark_lock


def arm_specs(args: argparse.Namespace) -> dict[str, base.ArmSpec]:
    def build(label: str, suffix: str) -> base.ArmSpec:
        return base.ArmSpec(
            label=label,
            name=getattr(args, f"arm_{suffix}_name"),
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
            allow_single_assignment_layout=True,
        )

    return {"A": build("A", "a"), "B": build("B", "b")}


def validate_fairness(
    bindings: Mapping[str, Mapping[str, Any]],
    live: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    a_binding, b_binding = bindings["A"], bindings["B"]
    a_live, b_live = live["A"], live["B"]
    parameter_keys = (
        "upper_k",
        "upper_search_ef",
        "dynamic_ef_base",
        "dynamic_ef_factor",
        "sample_denominator",
        "upper_sample_seed",
        "upper_m",
        "upper_ef_construction",
        "upper_graph_seed",
        "attachment_search_ef",
        "k_overlap",
        "initial_num_shards",
        "enable_fission",
        "enable_topology_refinement",
        "l0_repair",
    )
    parameter_mismatches = {
        key: {
            "A": a_binding["parameters"].get(key),
            "B": b_binding["parameters"].get(key),
        }
        for key in parameter_keys
        if a_binding["parameters"].get(key) != b_binding["parameters"].get(key)
    }
    checks = {
        "same_upper_navigator": a_live["upper_navigator_sha256"]
        == b_live["upper_navigator_sha256"],
        "same_upper_graph_semantics": a_live["upper_graph_semantic_sha256"]
        == b_live["upper_graph_semantic_sha256"],
        "same_vector_schema": a_live["vector_schema"] == b_live["vector_schema"],
        "same_logical_shard_count": a_live["shard_count"] == b_live["shard_count"] == 32,
        "same_logical_point_count": a_live["logical_point_count"]
        == b_live["logical_point_count"],
        "same_hnsw_config": a_live["hnsw_config"] == b_live["hnsw_config"],
        "same_optimizer_config": a_live["optimizer_config"]
        == b_live["optimizer_config"],
        "same_numeric_shard_placement": a_live["expected_placement"]
        == b_live["expected_placement"],
        "same_common_build_parameters": not parameter_mismatches,
        "physical_point_count_is_allowed_treatment": True,
    }
    if not all(checks.values()):
        raise RuntimeError(
            "owner-policy finalist fairness failed: "
            + json.dumps(
                {"checks": checks, "parameter_mismatches": parameter_mismatches},
                sort_keys=True,
            )
        )
    return {
        "status": "PASS",
        "checks": checks,
        "common_build_parameters": {
            key: a_binding["parameters"].get(key) for key in parameter_keys
        },
        "allowed_treatments": [
            "frozen_L1_owner",
            "post_owner_multi_assignment_policy",
            "resulting_physical_copy_count_and_distribution",
        ],
        "upper_navigator_sha256": a_live["upper_navigator_sha256"],
        "upper_graph_semantic_sha256": a_live["upper_graph_semantic_sha256"],
        "logical_point_count": a_live["logical_point_count"],
        "physical_point_count_by_arm": {
            label: live[label]["physical_point_count"] for label in base.ARM_LABELS
        },
    }


def validate_pair(
    arms: Mapping[str, base.ArmSpec],
    bindings: Mapping[str, Mapping[str, Any]],
    live: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    proofs: dict[str, Mapping[str, Any]] = {}
    for label in base.ARM_LABELS:
        proof = (live[label].get("artifact_bundle") or {}).get(
            "l1_partition_layout_proof"
        )
        if not isinstance(proof, Mapping):
            raise RuntimeError(f"finalist {label} lacks owner-policy proof")
        proofs[label] = proof
    combinations = {label: proofs[label].get("combination") for label in base.ARM_LABELS}
    matrix_sha = {label: proofs[label].get("matrix_sha256") for label in base.ARM_LABELS}
    checks = {
        "both_owner_policy_tournament_bundles": all(
            proofs[label].get("mode") == "owner_policy_tournament"
            for label in base.ARM_LABELS
        ),
        "declared_names_match_combinations": all(
            arms[label].name == combinations[label] for label in base.ARM_LABELS
        ),
        "distinct_combinations": combinations["A"] != combinations["B"],
        "same_frozen_matrix": matrix_sha["A"] == matrix_sha["B"],
        "both_research_only_before_gate": all(
            proofs[label].get("canonical_default_eligible") is False
            for label in base.ARM_LABELS
        ),
        "same_search_budget": all(
            bindings["A"]["parameters"].get(key)
            == bindings["B"]["parameters"].get(key)
            for key in ("upper_k", "upper_search_ef", "dynamic_ef_base", "dynamic_ef_factor")
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(
            "owner-policy finalist identity failed: " + json.dumps(checks, sort_keys=True)
        )
    return {
        "status": "PASS",
        "roles": combinations,
        "checks": checks,
        "matrix_sha256": matrix_sha["A"],
        "assignment_by_arm": {
            label: proofs[label].get("assignment") for label in base.ARM_LABELS
        },
    }


def contract() -> base.InterleavedABContract:
    return base.InterleavedABContract(
        arm_specs_factory=arm_specs,
        fairness_validator=validate_fairness,
        comparison_validator=validate_pair,
        comparison_manifest_key="owner_policy_finalist_pair",
        comparison_completion_key="owner_policy_finalist_pair_bound",
        fairness_budget_check_key="same_common_build_parameters",
        fairness_budget_completion_key="same_common_build_parameters",
        summary_record_type="c1_orion_owner_policy_finalist_ab_summary",
        run_record_type="c1_orion_owner_policy_finalist_interleaved_ab",
        budget_mode="frozen_u48_upper_ef48_base50_factor14_owner_policy_finalists",
        budget_subject="the frozen owner-policy tournament search budget",
        adoption_pass_decision="ARM_B_OWNER_POLICY_QPS_GATE_PASS",
        adoption_fail_decision="NO_ARM_B_WIN_ON_PAIRED_EVIDENCE",
        adoption_scope=(
            "GloVe/Cosine, four physical hosts, P=32, 64 server CPUs and the "
            "two tuning-selected owner-policy finalists only."
        ),
        decision_gate_key="owner_policy_finalist_online_qps_gate",
        semantic_ratio_key="paired_ratio_arm_b_over_arm_a",
        completion_error_subject="owner-policy finalist interleaved A/B",
        prepare_binding_validator=tuning_screen.prepare_binding,
        multi_assignment_retained_check_key=None,
    )


def validate_args(args: argparse.Namespace) -> None:
    for value in (args.arm_a_name, args.arm_b_name):
        owner, separator, policy = value.partition("+")
        if not separator or owner not in candidate.OWNER_NAMES or policy not in candidate.POLICY_NAMES:
            raise ValueError(f"invalid frozen owner-policy combination: {value}")
    base.validate_args(args)


def run(args: argparse.Namespace) -> Path:
    validate_args(args)
    output = Path(args.output_dir).expanduser().resolve()
    existed = output.exists()
    with benchmark_lock.hold_from_args(
        args,
        args.deployment_manifest,
        owner={
            "kind": "c1_orion_owner_policy_finalist_interleaved_ab",
            "collection_a": args.collection_a,
            "collection_b": args.collection_b,
            "output_dir": str(output),
        },
    ) as held:
        try:
            return base._run_locked(args, held, contract=contract())
        except BaseException as exc:
            if not existed and output.is_dir():
                try:
                    base.write_json(
                        output / "execution-failed.json",
                        {
                            "timestamp": base.utc_timestamp(),
                            "record_type": "c1_orion_owner_policy_finalist_ab_failure",
                            "status": "FAIL",
                            "error": repr(exc),
                            "benchmark_lock": held.evidence(),
                        },
                    )
                except Exception:
                    pass
            raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return base.parse_args(argv)


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
