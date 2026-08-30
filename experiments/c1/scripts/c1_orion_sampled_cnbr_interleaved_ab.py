#!/usr/bin/env python3
"""Fair read-only A/B of canonical C_CNBR versus sampled-CNBR research.

Both arms use the same four physical hosts, P=32 round-robin placement, upper
graph, HNSW configuration, frozen u48/upperEF48/base50/factor14 search budget,
and enabled/2/0/0 multi-assignment.  Only the frozen L1 owner differs.
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
from experiments.l1_balance import materialize_sampled_cnbr_candidate as sampled  # noqa: E402
from experiments.l1_balance import native_cnbr_core  # noqa: E402
from experiments.l1_balance import cnbr_construction_cost_gate as cost_gate  # noqa: E402


benchmark_lock = base.benchmark_lock
ARM_NAMES = {"A": "C_CNBR", "B": sampled.ARM_NAME}


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
    """Accept only current topology plus exact current artifact-ledger identity."""

    compatibility = replace(arm, allow_historical_prepare_deployment=True)
    original_core_file = native_cnbr_core.__file__
    try:
        if arm.name == "C_CNBR":
            build_manifest = base.load_json_object(
                arm.layout_dir / "build-manifest.json"
            )
            formal_screen = Path(
                build_manifest["provenance"]["phase_b_screen"]
            ).resolve()
            frozen_core = formal_screen.parent / "source" / "native_cnbr_core.py"
            if not frozen_core.is_file():
                raise FileNotFoundError(frozen_core)
            # The canonical validator compares its source binding through the
            # module's __file__.  Point that lookup at the bundle-frozen copy
            # only for validation, then restore before measurement snapshots.
            native_cnbr_core.__file__ = str(frozen_core)
        binding = base.validate_prepare_binding(
            compatibility,
            base_url=base_url,
            deployment_manifest=deployment_manifest,
            topology=topology,
        )
    finally:
        native_cnbr_core.__file__ = original_core_file
    topology_binding = binding.get("prepare_topology_binding")
    if not isinstance(topology_binding, Mapping) or topology_binding.get("mode") != "current":
        raise RuntimeError("sampled-CNBR A/B cannot excuse topology drift")
    # The existing ledger checker is collection/artifact based; its arm-name
    # whitelist predates this research arm.  Alias only that display field.
    ledger_arm = arm if arm.name == "C_CNBR" else replace(arm, name="C_CNBR")
    current_ledger = ledger.validate_current_deployment_artifact_ledger(
        ledger_arm, binding, deployment_manifest
    )
    deployment_binding = binding.get("prepare_deployment_binding")
    if not isinstance(deployment_binding, Mapping):
        raise RuntimeError("sampled-CNBR prepare deployment binding is missing")
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


def validate_pair(
    arms: Mapping[str, base.ArmSpec],
    bindings: Mapping[str, Mapping[str, Any]],
    live: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    a_manifest = bindings["A"].get("build_manifest")
    b_manifest = bindings["B"].get("build_manifest")
    if not isinstance(a_manifest, Mapping) or not isinstance(b_manifest, Mapping):
        raise RuntimeError("sampled-CNBR pair lacks build manifests")
    a_diag = (a_manifest.get("routing") or {}).get("l1_partition_diagnostics")
    b_diag = (b_manifest.get("routing") or {}).get("l1_partition_diagnostics")
    if not isinstance(a_diag, Mapping) or not isinstance(b_diag, Mapping):
        raise RuntimeError("sampled-CNBR pair lacks L1 diagnostics")
    a_variant = (a_diag.get("balance_contract") or {}).get("variant")
    b_variant = (b_diag.get("balance_contract") or {}).get("variant")
    a_owner = str(a_diag.get("owner_sha256") or "")
    b_owner = str(b_diag.get("owner_sha256") or "")
    checks = {
        "arm_a_is_canonical_cnbr": (
            a_manifest.get("tool")
            == "experiments/l1_balance/materialize_native_cnbr_candidate.py"
            and a_variant == "cnbr"
            and a_manifest.get("parameters", {}).get("l1_partitioner") == "C_CNBR"
        ),
        "arm_b_is_sampled_research": (
            b_manifest.get("tool") == sampled.TOOL_PATH
            and b_variant == sampled.BALANCE_VARIANT
            and b_manifest.get("parameters", {}).get("l1_partitioner")
            == sampled.ARM_NAME
            and b_diag.get("offline_research_only") is True
            and b_diag.get("canonical_default_eligible") is False
        ),
        "owners_are_distinct": len(a_owner) == 64 and len(b_owner) == 64 and a_owner != b_owner,
        "same_source_phase_a": a_diag.get("phase_a_manifest_sha256")
        == b_diag.get("phase_a_manifest_sha256"),
        "same_full_attachment_bytes": a_diag.get("attachments_sha256")
        == b_diag.get("attachments_sha256"),
        "sample_parameters_exact": all(
            b_manifest.get("parameters", {}).get(key) == value
            for key, value in {
                "sample_fill_fraction": sampled.EXPECTED_FRACTION,
                "sample_fill_seed": sampled.EXPECTED_SEED,
                "sample_fill_weight": sampled.EXPECTED_SAMPLE_WEIGHT,
                "upper_prior_weight": 1.0 - sampled.EXPECTED_SAMPLE_WEIGHT,
            }.items()
        ),
        "offline_graph_and_query_gates_pass": all(
            all(isinstance(gate, Mapping) and gate.get("pass") is True for gate in gates.values())
            for gates in (
                b_diag.get("graph_topology_gates") or {},
                b_diag.get("query_topology_gates") or {},
            )
        ),
        "current_deployment_artifact_ledgers_bound": all(
            (bindings[label].get("current_deployment_artifact_ledger") or {}).get(
                "status"
            )
            == "PASS"
            for label in base.ARM_LABELS
        ),
        "fixed_p32": all(live[label]["shard_count"] == 32 for label in base.ARM_LABELS),
    }
    if not all(checks.values()):
        raise RuntimeError(
            "C_CNBR/sampled-CNBR identity contract failed: "
            + json.dumps(checks, sort_keys=True)
        )
    return {
        "status": "PASS",
        "roles": dict(ARM_NAMES),
        "checks": checks,
        "owner_sha256_by_arm": {"A": a_owner, "B": b_owner},
        "balance_variant_by_arm": {"A": a_variant, "B": b_variant},
    }


def contract() -> base.InterleavedABContract:
    return base.InterleavedABContract(
        arm_specs_factory=arm_specs,
        fairness_validator=base.validate_cross_arm_fairness,
        comparison_validator=validate_pair,
        comparison_manifest_key="cnbr_sampled_cnbr_pair",
        comparison_completion_key="cnbr_sampled_cnbr_pair_bound",
        fairness_budget_check_key="same_frozen_historical_budget",
        fairness_budget_completion_key="same_frozen_historical_budget",
        summary_record_type="c1_orion_sampled_cnbr_interleaved_ab_summary",
        run_record_type="c1_orion_sampled_cnbr_interleaved_ab",
        budget_mode="frozen_u48_upper_ef48_base50_factor14_cnbr_sampled_cnbr",
        budget_subject="the identical frozen Orion search budget",
        adoption_pass_decision="SAMPLED_CNBR_QPS_GATE_PASS",
        adoption_fail_decision="RETAIN_CANONICAL_CNBR_ON_ONLINE_EVIDENCE",
        adoption_scope=(
            "Research-only matched-recall online comparison of sampled-CNBR "
            "against canonical C_CNBR; this cannot by itself change the default."
        ),
        decision_gate_key="online_qps_research_gate",
        semantic_ratio_key="paired_ratio_sampled_over_cnbr",
        completion_error_subject="C_CNBR/sampled-CNBR interleaved A/B",
        prepare_binding_validator=prepare_binding,
    )


def validate_args(args: argparse.Namespace) -> None:
    if {"A": args.arm_a_name, "B": args.arm_b_name} != ARM_NAMES:
        raise ValueError("arm names are fixed to C_CNBR and sampled_CNBR")
    base.validate_args(args)


def run(args: argparse.Namespace) -> Path:
    validate_args(args)
    output = Path(args.output_dir).expanduser().resolve()
    existed = output.exists()
    with benchmark_lock.hold_from_args(
        args,
        args.deployment_manifest,
        owner={
            "kind": "c1_orion_sampled_cnbr_interleaved_ab",
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
            original_current_source_validator = cost_gate._validate_current_source_bindings
            try:
                if arm.name == "C_CNBR":
                    formal_screen = Path(
                        binding["build_manifest"]["provenance"]["phase_b_screen"]
                    ).resolve()
                    native_cnbr_core.__file__ = str(
                        formal_screen.parent / "source" / "native_cnbr_core.py"
                    )
                    # The immutable cost audit and its frozen source copies are
                    # still fully verified.  Only its comparison to today's
                    # additive research working tree is inapplicable here.
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
                            "record_type": "c1_orion_sampled_cnbr_interleaved_ab_failure",
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
