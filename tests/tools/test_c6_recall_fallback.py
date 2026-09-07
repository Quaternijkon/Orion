from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "experiments/c6/scripts/c6_recall_fallback.py"


def load_module():
    spec = importlib.util.spec_from_file_location("c6_recall_fallback", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def candidate(config_id, recall, work, p99):
    return {
        "policy": "P2",
        "config_id": config_id,
        "config": {"policy": "P2", "fixed_p": 2},
        "recall_at_10": recall,
        "aggregate_distance_computations_per_query": work,
        "p99_max_shard_distance_computations": p99,
    }


def test_preorder_is_tuning_only_and_keeps_frozen_primary_first():
    module = load_module()
    rows = [
        candidate("primary", 0.901, 100.0, 50.0),
        candidate("tail-b", 0.910, 110.0, 40.0),
        candidate("tail-a", 0.920, 110.0, 30.0),
        candidate("infeasible", 0.899, 1.0, 1.0),
    ]
    ordered = module.preorder_candidates(rows, selected_config_id="primary")
    assert [row["config_id"] for row in ordered] == ["primary", "tail-a", "tail-b"]


def test_fallback_selection_replaces_only_requested_policy():
    module = load_module()
    selection = {
        "dataset": "glove-200-angular",
        "selected": {
            name: {
                "config_id": f"{name}-original",
                "config": {"policy": name},
                "summary": {},
            }
            for name in module.POLICIES
        },
    }
    replacement = candidate("P2-fallback", 0.91, 120.0, 45.0)
    result = module.build_fallback_selection(
        selection,
        policy="P2",
        candidate=replacement,
        fallback_rank=1,
    )
    assert result["selected"]["P2"]["config_id"] == "P2-fallback"
    assert result["selected"]["P1"]["config_id"] == "P1-original"
    assert result["fallback"]["measurement_metrics_consulted_for_ordering"] is False
