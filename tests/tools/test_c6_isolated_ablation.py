from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "experiments/c6/scripts/c6_isolated_ablation.py"


def load_module():
    spec = importlib.util.spec_from_file_location("c6_isolated_ablation", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_spearman_and_calibration_group_summary():
    module = load_module()
    rows = [
        {
            "entry_point_group": "1",
            "entry_point_count": 1,
            "contributes_ground_truth": False,
            "ground_truth_contribution": 0,
            "oracle_local_ef_search": 8,
            "oracle_local_distance_computations": 10,
        },
        {
            "entry_point_group": "2",
            "entry_point_count": 2,
            "contributes_ground_truth": True,
            "ground_truth_contribution": 1,
            "oracle_local_ef_search": 16,
            "oracle_local_distance_computations": 20,
        },
        {
            "entry_point_group": "3-4",
            "entry_point_count": 4,
            "contributes_ground_truth": True,
            "ground_truth_contribution": 2,
            "oracle_local_ef_search": 32,
            "oracle_local_distance_computations": 30,
        },
    ]
    summary = module.summarize_calibration(rows)
    assert summary["spearman_entry_points_vs_oracle_local_work"] == pytest.approx(1.0)
    assert summary["spearman_entry_points_vs_ground_truth_contribution"] == pytest.approx(1.0)
    assert summary["groups"][0]["probability_contributes_ground_truth"] == 0.0
    assert summary["groups"][1]["probability_contributes_ground_truth"] == 1.0
