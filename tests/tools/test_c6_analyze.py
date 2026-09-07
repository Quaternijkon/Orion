from __future__ import annotations

import importlib.util
import json
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "experiments/c6/scripts/c6_analyze.py"


def load_module():
    spec = importlib.util.spec_from_file_location("c6_analyze", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_split_validation_enforces_exact_tuning_and_all_remaining_measurement():
    module = load_module()
    tune = Namespace(
        logical_shards=4,
        query_start=0,
        query_count=1000,
        run_id="tune",
        dataset="sift1m",
        score_higher_is_better=False,
        command="tune",
    )
    module.validate_common(tune, 10_000)
    tune.query_count = 999
    with pytest.raises(ValueError, match="exactly"):
        module.validate_common(tune, 10_000)

    measure = Namespace(
        logical_shards=4,
        query_start=1000,
        query_count=9000,
        run_id="measure",
        dataset="glove-200-angular",
        score_higher_is_better=True,
        command="measure",
    )
    module.validate_common(measure, 10_000)
    measure.query_count = 8999
    with pytest.raises(ValueError, match="every official query"):
        module.validate_common(measure, 10_000)


def test_policy_matrix_is_complete_and_validated(tmp_path):
    module = load_module()
    payload = {
        "P0": [{"fixed_p": 2, "uniform_ef_search": 16}],
        "P1": [{"uniform_ef_search": 16}],
        "P2": [{"fixed_p": 2, "alpha": 4, "beta": 8}],
        "P3": [{"alpha": 4, "beta": 8}],
    }
    path = tmp_path / "matrix.json"
    path.write_text(json.dumps(payload))
    matrix = module.load_policy_matrix(path, 4)
    assert [config.policy for config in matrix["P3"]] == ["P3"]
    del payload["P2"]
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="P2"):
        module.load_policy_matrix(path, 4)


def test_assignment_memberships_stream_only_required_ground_truth(tmp_path):
    module = load_module()
    path = tmp_path / "assignments.jsonl"
    path.write_text(
        "\n".join(
            json.dumps({"id": point_id, "shards": [point_id % 2]})
            for point_id in range(20)
        )
        + "\n"
    )
    truth = np.asarray([[2, 3, 4], [4, 5, 6]])
    memberships = module.assignment_memberships_for_truth(path, truth)
    assert memberships == {2: (0,), 3: (1,), 4: (0,), 5: (1,), 6: (0,)}


def test_explode_per_shard_preserves_aligned_query_arrays():
    module = load_module()
    rows = [
        {
            "query_id": 0,
            "dataset": "sift1m",
            "logical_shards": 4,
            "policy": "P2",
            "ranked_candidate_shards": [2, 0, 1],
            "selected_shard_ids": [2, 0],
            "entry_point_count_per_selected_shard": [4, 2],
            "assigned_efSearch_per_shard": [24, 16],
            "distance_computations_per_shard": [100, 60],
            "nodes_visited_per_shard": [10, 7],
            "worker_cpu_time_per_shard": [30, 20],
        }
    ]
    exploded = module.explode_per_shard(rows)
    assert exploded[0]["route_rank"] == 1
    assert exploded[1]["route_rank"] == 2
    assert exploded[1]["distance_computations"] == 60
