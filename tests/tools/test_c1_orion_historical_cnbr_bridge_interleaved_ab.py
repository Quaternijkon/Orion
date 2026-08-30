from __future__ import annotations

import argparse
import ast
import importlib.util
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = (
    REPO_ROOT
    / "experiments/c1/scripts/c1_orion_historical_cnbr_bridge_interleaved_ab.py"
)


def load_module():
    name = "c1_orion_historical_cnbr_bridge_interleaved_ab_test"
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def args() -> argparse.Namespace:
    return argparse.Namespace(
        arm_a_name="H",
        arm_b_name="C_CNBR",
        collection_a="historical",
        collection_b="cnbr",
        artifact_a=Path("historical/generation.json"),
        artifact_b=Path("cnbr/generation.json"),
        layout_dir_a=Path("historical"),
        layout_dir_b=Path("cnbr"),
        prepare_manifest_a=Path("historical/prepare.json"),
        prepare_manifest_b=Path("cnbr/prepare.json"),
    )


def c_diagnostics(module) -> dict[str, object]:
    return {
        "balance_contract": {"variant": "cnbr"},
        "phase_a_manifest_sha256": "a" * 64,
        "phase_a_owner_sha256": "b" * 64,
        "owner_sha256": "b" * 64,
        "attachments_sha256": "c" * 64,
        "parent_n_owner_sha256": "d" * 64,
        "parent_n_manifest_sha256": "a" * 64,
        "forbidden_stage_invocations": {
            key: 0 for key in module.native_cnbr.PHASE_A_FORBIDDEN_INVOCATION_KEYS
        },
        "multi_assignment_after_owner_freeze": True,
    }


def c_binding(module) -> dict[str, object]:
    return {
        "parameters": {
            "initial_num_shards": 32,
            "enable_fission": False,
            "enable_topology_refinement": False,
            "l0_repair": False,
        },
        "routing": {
            "effective_num_shards": 32,
            "fission_events": [],
            "l1_partition_diagnostics": c_diagnostics(module),
        },
    }


def test_bridge_arm_specs_fix_historical_h_and_current_cnbr_roles():
    module = load_module()
    specs = module.historical_cnbr_arm_specs(args())
    assert specs["A"].allow_scaling_layout is True
    assert specs["A"].allow_historical_prepare_deployment is True
    assert specs["A"].allow_l1_partition_layout is False
    assert specs["B"].allow_l1_partition_layout is True
    assert specs["B"].allow_historical_prepare_deployment is False
    assert specs["B"].allow_scaling_layout is False


def test_bridge_validates_exact_h_and_c_cnbr_contract(monkeypatch):
    module = load_module()
    monkeypatch.setattr(
        module.base,
        "validate_adjustment_before_baseline",
        lambda *_args, **_kwargs: {"status": "PASS", "checks": {"exact_h": True}},
    )
    specs = module.historical_cnbr_arm_specs(args())
    bindings = {"A": {}, "B": c_binding(module)}
    live = {"A": {}, "B": {"shard_count": 32}}
    proof = module.validate_historical_cnbr_pair(specs, bindings, live)
    assert proof["roles"] == {"A": "H", "B": "C_CNBR"}
    assert proof["causal_interpretation_allowed"] is False
    assert proof["adoption_or_fallback_decision_allowed"] is False
    assert proof["checks"]["arm_b_forbidden_phase_a_invocations_zero"] is True


def test_bridge_rejects_noncanonical_cnbr_alias(monkeypatch):
    module = load_module()
    monkeypatch.setattr(
        module.base,
        "validate_adjustment_before_baseline",
        lambda *_args, **_kwargs: {"status": "PASS"},
    )
    specs = module.historical_cnbr_arm_specs(args())
    bindings = {"A": {}, "B": c_binding(module)}
    bindings["B"]["routing"]["l1_partition_diagnostics"]["balance_contract"][
        "variant"
    ] = "cnbr_upper_only_boundary_repair"
    with pytest.raises(RuntimeError, match="non-canonical balance variant"):
        module.validate_historical_cnbr_pair(
            specs, bindings, {"A": {}, "B": {"shard_count": 32}}
        )


def test_bridge_rejects_cnbr_fission(monkeypatch):
    module = load_module()
    monkeypatch.setattr(
        module.base,
        "validate_adjustment_before_baseline",
        lambda *_args, **_kwargs: {"status": "PASS"},
    )
    specs = module.historical_cnbr_arm_specs(args())
    bindings = {"A": {}, "B": c_binding(module)}
    bindings["B"]["parameters"]["enable_fission"] = True
    with pytest.raises(RuntimeError, match="identity contract failed"):
        module.validate_historical_cnbr_pair(
            specs, bindings, {"A": {}, "B": {"shard_count": 32}}
        )


def test_bridge_fairness_requires_exact_multi_assignment(monkeypatch):
    module = load_module()
    monkeypatch.setattr(
        module.base,
        "validate_cross_arm_fairness",
        lambda *_args, **_kwargs: {
            "status": "PASS",
            "checks": {"multi_assignment_enabled_in_both": True},
        },
    )
    exact = {
        "use_multi_assign": True,
        "multi_assign_min_max_vote": 2,
        "multi_assign_vote_delta": 0,
        "multi_assign_max_shards": 0,
    }
    proof = module.validate_historical_cnbr_fairness(
        {"A": {"parameters": exact}, "B": {"parameters": exact}}, {}
    )
    assert proof["checks"]["exact_multi_assignment_in_both"] is True
    with pytest.raises(RuntimeError, match="multi-assignment parameters"):
        module.validate_historical_cnbr_fairness(
            {
                "A": {"parameters": exact},
                "B": {"parameters": {**exact, "multi_assign_vote_delta": 1}},
            },
            {},
        )


def test_bridge_contract_is_descriptive_and_has_no_adoption_fallback_decision():
    module = load_module()
    contract = module.historical_cnbr_contract()
    assert contract.semantic_ratio_key == "paired_ratio_c_over_h"
    assert contract.decision_gate_key == "historical_c_over_h_recovery_gate"
    assert contract.comparison_manifest_key == "historical_h_cnbr_bridge"
    assert "RETAIN_ORIGINAL" not in contract.adoption_fail_decision
    assert "ADOPT" not in contract.adoption_pass_decision
    assert "cannot" in contract.adoption_scope
    assert (
        contract.adoption_prerequisite_validator
        is module.validate_bridge_construction_cost
    )


def test_bridge_runner_source_contains_no_cluster_or_resource_mutator_calls():
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    forbidden = {
        "apply_resource_schedule",
        "update_container_resources",
        "delete_collection_if_exists",
        "move_numeric_shards_explicit",
        "install_profile_artifact",
        "run_prepare",
        "prepare_collection",
        "activate_artifact",
    }
    called = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    called.update(
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    )
    assert called.isdisjoint(forbidden)
