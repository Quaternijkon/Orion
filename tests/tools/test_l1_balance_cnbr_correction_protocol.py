from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = (
    REPO_ROOT / "experiments/l1_balance/CNBR_CORRECTION_ADDENDUM.json"
)
DOC_PATH = REPO_ROOT / "experiments/l1_balance/CNBR_CORRECTION_ADDENDUM.md"


def load_spec() -> dict:
    return json.loads(SPEC_PATH.read_text(encoding="utf-8"))


def test_role_correction_removes_historical_balance_from_candidate_set() -> None:
    spec = load_spec()

    assert spec["scope"]["frozen_old_protocol_unchanged"] is True
    assert spec["scope"]["supersession_limited_to"] == [
        "role_mapping",
        "comparison_interpretation",
        "historical_fallback_semantics",
    ]
    assert spec["role_mapping"] == {
        "C_CNBR": "only_balance_replacement_candidate",
        "H": "historical_l0_informed_p24_fission32_performance_anchor_only",
        "N_native": "balance_free_orion_causal_baseline",
        "R_v2": "frozen_failed_historical_candidate",
    }
    assert spec["fallback"] == {
        "allowed": "N_native",
        "historical_H_forbidden": True,
    }
    assert spec["historical_h"] == {
        "adoption_veto": False,
        "candidate": False,
        "fallback": False,
        "performance_anchor_only": True,
    }


def test_phase_a_requires_neutral_upper_only_projection() -> None:
    projection = load_spec()["upper_only_projection"]

    assert projection == {
        "direct_historical_production_artifact_input_to_phase_a_forbidden": True,
        "historical_layout_metadata_removed": True,
        "historical_shard_membership_removed": True,
        "phase_a_artifact_input": "neutral_upper_only_projection_only",
        "preserve_ordered_upper_labels_and_vector_bits_exact": True,
        "preserve_production_upper_graph_exact": True,
        "source_build_manifest_parameters_whitelisted": [
            "upper_sample_seed",
            "upper_m",
            "upper_ef_construction",
            "upper_graph_seed",
        ],
    }


def test_native_baseline_is_plain_unconstrained_upper_kmeans() -> None:
    native = load_spec()["n_native"]

    assert native["partitions"] == 32
    assert native["seed"] == 1
    assert native["max_iter"] == 10
    assert native["assignment"] == "plain_squared_l2_argmin"
    assert native["empty_cluster_or_unequal_size_repair"] is False
    assert set(native["prohibited_features"]) == {
        "weights",
        "capacity",
        "quota",
        "topology_refinement",
        "fission",
        "l0_inspection",
        "owner_repair",
    }


def test_cnbr_grid_and_selected_contract_are_exact() -> None:
    spec = load_spec()
    family = spec["candidate_family"]
    cnbr = spec["cnbr"]

    assert [
        row["display_decimal"] for row in family["fixed_sift_pre_glove_grid"]
    ] == [
        "2.25",
        "2.0",
        "1.75",
        "1.6",
        "1.5",
        "1.4",
    ]
    assert family["selected_ratio"] == {
        "denominator": 4,
        "display_decimal": "2.25",
        "numerator": 9,
    }
    for ratio in [
        *family["fixed_sift_pre_glove_grid"],
        family["selected_ratio"],
        cnbr["overload_trigger_ratio"],
    ]:
        assert "decimal" not in ratio
        assert isinstance(ratio["numerator"], int)
        assert isinstance(ratio["denominator"], int)
    assert family["glove_is_strictly_held_out_for_family_selection"] is False
    assert cnbr["max_rounds"] == 8
    assert cnbr["node_move_limit"] == 1
    assert cnbr["requires_cross_owner_graph_neighbor"] is False
    assert cnbr["self_navigation_only_target_allowed"] is True
    assert cnbr["load_predicates"] == [
        "source_load > trigger",
        "target_load < trigger",
        "source_load - target_load > node_mass",
        "target_load + node_mass <= trigger",
    ]
    assert cnbr["integer_load_predicates_selected_ratio"] == [
        "4 * P * source_load > 9 * total_mass",
        "4 * P * target_load < 9 * total_mass",
        "source_load - target_load > node_mass",
        "4 * P * (target_load + node_mass) <= 9 * total_mass",
    ]
    assert cnbr["target_key"] == [
        "max(edge_delta,0)",
        "max(nav_loss,0)",
        "edge_delta",
        "nav_loss",
        "target_load",
        "negative_node_mass",
        "target_id",
    ]
    proxy = spec["proxy_mass"]
    assert proxy["proxy_mass_contract_id"] == (
        "cnbr-upper-self-navigation-raw-occurrence-v1"
    )
    assert proxy["source"] == "frozen_production_upper_self_navigation_top10"
    assert proxy["transform"] == "raw_occurrence_count"
    assert proxy["estimator_version"] == 1
    assert proxy["not_r_v2_mass_balanced_kmeans"] is True
    trace = cnbr["trace_contract"]
    assert trace["commit_decision_for_each_formed_proposal"] is True
    assert trace["explicit_skip_reason_for_rejected_commit"] is True
    assert trace["nodes_without_a_formed_proposal_required"] is False
    assert trace["rejected_target_candidates_required"] is False


def test_stage_boundary_multi_assignment_build_and_online_gates_are_frozen() -> None:
    spec = load_spec()

    assert spec["construction_boundary"]["owner_checksum_freeze_precedes_l0_access"]
    assert spec["multi_assignment"] == {
        "applied_after_owner_freeze": True,
        "enabled": True,
        "max_shards": 0,
        "min_max_vote": 2,
        "orthogonal_to_balance_patch": True,
        "vote_delta": 0,
    }
    assert spec["construction_cost"]["additional_l0_pass_allowed"] is False
    assert (
        spec["construction_cost"][
            "balance_stage_wall_over_normal_full_attachment_wall_max"
        ]
        == 0.05
    )
    construction = spec["construction_cost"]
    assert construction["balance_stage_wall_time_components"] == [
        "upper_self_navigation",
        "required_input_validation_and_proxy_mass_replay",
        "cnbr",
    ]
    assert construction["incremental_timing_contract_id"] == (
        "cnbr-integrated-incremental-cost-v1"
    )
    assert construction["required_validation_boundary"] == {
        "excluded_shared_prefix": (
            "upper artifact deserialization and graph normalization shared by "
            "N_native and C_CNBR"
        ),
        "includes": [
            "projected_artifact_sha256",
            "upper_only_projection_provenance_and_redaction",
            "upper_input_manifest_and_exact_byte_parity",
            "self_navigation_manifest_and_row_validation",
            "proxy_mass_replay_and_duplicate_tie_proof",
        ],
        "production_projection_pass_allowed_without_cost": False,
    }
    assert construction["required_validation_and_mass_replay_must_be_counted"] is True
    assert construction["strict_wall_ratio_formula"] == (
        "(external_self_navigation_wall_seconds + "
        "required_input_validation_and_mass_replay_wall_seconds + "
        "formal_cnbr_wall_seconds) / external_full_attachment_wall_seconds"
    )
    online = spec["online_confirmation"]
    assert online["primary_causal_gate"]["comparison"] == "N_native_vs_C_CNBR"
    assert online["primary_causal_gate"]["recall_at_10_min_each_arm"] == 0.9
    assert online["historical_bridge"]["comparison"] == "H_vs_C_CNBR"
    assert online["historical_bridge"]["mandatory"] is True
    assert online["historical_bridge"]["adoption_veto"] is False


def test_human_addendum_states_non_destructive_correction() -> None:
    text = DOC_PATH.read_text(encoding="utf-8")

    assert "without editing or erasing" in text
    assert "the compliant fallback is `N_native`, never `H`" in text
    assert "GloVe\nis **not** claimed to be a strictly untouched held-out dataset" in text
    assert "CNBR does not remove multi-assignment" in text
    assert "Omitting it produces only a lower bound" in text
    assert "integrated-construction boundary" in text
    assert "experiment-only adapter" in text
    assert "The neutral upper-only projection is the **only** artifact" in text
    assert "`H` is `candidate=false`, `adoption_veto=false`, and" in text
