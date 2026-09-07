from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.multi_assignment import budgeted_policy
from experiments.multi_assignment import capacity_bounded_bmr10 as cap
from experiments.multi_assignment import screen_capacity_bounded_bmr10 as screen
from experiments.multi_assignment import synthesize_capacity_bounded_selection as synth


def test_capacity_bounded_bmr10_preserves_copy_counts_and_guarantees_floor():
    owner = np.asarray([0, 0, 0, 1, 2, 2], dtype=np.int32)
    attachment_local = np.tile(
        np.asarray([[0, 1, 2, 3, 4, 5]], dtype=np.int32), (30, 1)
    )
    proxy_mass = np.ones(len(owner), dtype=np.uint64)
    base = budgeted_policy.build_budgeted_assignment(
        owner=owner,
        attachment_local=attachment_local,
        proxy_mass=proxy_mass,
        num_partitions=3,
    )

    result = cap.build_capacity_bounded_assignment(
        owner=owner,
        attachment_local=attachment_local,
        proxy_mass=proxy_mass,
        num_partitions=3,
    )

    assert np.array_equal(
        result.membership.sum(axis=1), base.membership.sum(axis=1)
    )
    assert int(result.membership.sum()) == int(base.membership.sum())
    assert result.diagnostics["bounds_satisfied"] is True
    assert result.diagnostics["copy_count_preserved"] is True
    assert result.diagnostics["all_assignments_have_navigation_evidence"] is True
    loads = result.membership.sum(axis=0)
    assert int(loads.min()) >= int(result.diagnostics["lower_load"])
    assert int(loads.max()) <= int(result.diagnostics["upper_load"])
    assert result.membership_semantic_sha256 != result.base_membership_semantic_sha256


def test_capacity_screen_gates_require_floor_and_preserve_quality():
    baseline = {
        "physical_point_count": 100,
        "copy_count_histogram": {"1": 90, "2": 5},
        "physical_copy_load_min": 2,
        "physical_copy_load_max": 20,
        "physical_copy_load_cv": 0.6,
        "gt_routing_coverage_mean": 0.97,
        "gt_queries_full_coverage_fraction": 0.80,
        "gt_queries_zero_coverage": 0,
        "routed_shards_mean": 8.0,
        "route_ef_sum_mean": 1000.0,
    }
    candidate = {
        **baseline,
        "physical_copy_load_min": 4,
        "physical_copy_load_max": 19,
        "physical_copy_load_cv": 0.55,
        "gt_routing_coverage_mean": 0.969,
        "routed_shards_mean": 8.2,
        "route_ef_sum_mean": 1010.0,
    }
    diagnostics = {"bounds_satisfied": True, "lower_load": 4}

    gates = screen.comparison_gates(candidate, baseline, diagnostics, 1000)

    assert all(gate["pass"] for gate in gates.values())


def test_dual_dataset_selection_freezes_two_passing_screens(tmp_path: Path):
    paths = {}
    for dataset in ("glove-200-angular", "sift1m"):
        path = tmp_path / f"{dataset}.json"
        value = {
            "record_type": "capacity_bounded_bmr10_offline_screen",
            "dataset": dataset,
            "status": "PASS",
            "parameters": {
                "min_load_ratio": cap.MIN_LOAD_RATIO,
                "max_load_ratio": cap.MAX_LOAD_RATIO,
                "max_vote_loss": cap.MAX_VOTE_LOSS,
                "max_passes": cap.MAX_PASSES,
            },
            "gates": {"all": {"pass": True}},
            "identity": {"owner_unchanged": True},
            "baseline": {"physical_point_count": 100},
            "candidate": {"physical_point_count": 100},
            "capacity_diagnostics": {"bounds_satisfied": True},
        }
        path.write_text(json.dumps(value), encoding="utf-8")
        paths[dataset] = path

    output = tmp_path / "selection"
    result = synth.run(
        argparse.Namespace(
            glove_screen=paths["glove-200-angular"],
            sift_screen=paths["sift1m"],
            output_dir=output,
        )
    )
    selection = json.loads(result.read_text(encoding="utf-8"))

    assert selection["selected_candidate"] == cap.CANDIDATE_ID
    assert selection["status"] == "FROZEN_OFFLINE_PASS_ONLINE_PENDING"
    assert set(selection["rows"]) == {"glove-200-angular", "sift1m"}
