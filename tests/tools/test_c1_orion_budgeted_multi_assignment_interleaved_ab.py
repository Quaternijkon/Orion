from __future__ import annotations

from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def load_module():
    path = (
        REPO_ROOT
        / "experiments/c1/scripts/c1_orion_budgeted_multi_assignment_interleaved_ab.py"
    )
    spec = importlib.util.spec_from_file_location(
        "c1_orion_budgeted_multi_assignment_interleaved_ab_for_test", path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def fairness_fixture(module):
    common = {
        "sample_denominator": 32,
        "upper_sample_seed": 100,
        "upper_m": 32,
        "upper_ef_construction": 100,
        "attachment_search_ef": 100,
        "upper_graph_seed": 100,
        "k_overlap": 10,
        "use_multi_assign": True,
        "multi_assign_min_max_vote": 2,
        "multi_assign_vote_delta": 0,
        **module.base.HISTORICAL_BUDGET,
        "initial_num_shards": 32,
        "enable_fission": False,
        "enable_topology_refinement": False,
        "l0_repair": False,
        "l1_partitioner": "C_CNBR",
        "balance_mode": "cnbr",
    }
    owner = "a" * 64
    diagnostics = {
        "owner_sha256": owner,
        "phase_a_manifest_sha256": "b" * 64,
        "attachments_sha256": "c" * 64,
        "l1_sizes": [2, 2],
    }
    a_parameters = {**common, "multi_assign_max_shards": 0}
    b_parameters = {
        **common,
        "multi_assign_max_shards": 2,
        "multi_assign_extra_copy_budget_numerator": 1,
        "multi_assign_extra_copy_budget_denominator": 10,
        "multi_assign_score": module.budgeted_policy.SCORE,
        "multi_assignment_policy": module.budgeted_policy.CANDIDATE_ID,
        "multi_assignment_policy_version": module.budgeted_policy.POLICY_VERSION,
    }
    bindings = {
        "A": {
            "parameters": a_parameters,
            "build_manifest": {
                "routing": {"l1_partition_diagnostics": diagnostics},
                "provenance": {"source_artifact_sha256": "d" * 64},
            },
        },
        "B": {
            "parameters": b_parameters,
            "build_manifest": {
                "routing": {
                    "l1_partition_diagnostics": deepcopy(diagnostics)
                },
                "provenance": {"source_artifact_sha256": "d" * 64},
            },
        },
    }
    live_common = {
        "upper_navigator_sha256": "e" * 64,
        "upper_graph_semantic_sha256": "f" * 64,
        "vector_schema": {"dimension": 200, "distance": "cosine"},
        "shard_count": 32,
        "logical_point_count": 100,
        "hnsw_config": {"m": 32, "ef_construct": 200},
        "optimizer_config": {"indexing_threshold": 10},
        "expected_placement": {str(index): index % 4 for index in range(32)},
    }
    live = {
        "A": {**live_common, "physical_point_count": 114},
        "B": {**live_common, "physical_point_count": 110},
    }
    return bindings, live


def test_multi_assignment_fairness_accepts_only_post_owner_policy_change():
    module = load_module()
    bindings, live = fairness_fixture(module)

    proof = module.validate_multi_assignment_fairness(bindings, live)

    assert proof["status"] == "PASS"
    assert proof["checks"]["same_frozen_cnbr_owner"] is True
    assert proof["checks"]["only_multi_assignment_policy_differs"] is True
    assert proof["physical_point_count_by_arm"] == {"A": 114, "B": 110}

    drifted = deepcopy(bindings)
    drifted["B"]["build_manifest"]["routing"]["l1_partition_diagnostics"][
        "owner_sha256"
    ] = "9" * 64
    with pytest.raises(RuntimeError, match="fairness contract failed"):
        module.validate_multi_assignment_fairness(drifted, live)


def test_budgeted_ab_contract_reports_bmr_over_current_ratio():
    module = load_module()
    contract = module.contract()

    assert contract.semantic_ratio_key == "paired_ratio_bmr10_over_current"
    assert contract.adoption_pass_decision == "BMR10_QPS_GATE_PASS"
    assert contract.prepare_binding_validator is module.prepare_binding
