from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def load_module():
    script_path = Path("/home/taig/dry/qdrant/tools/qdrant_partition_scheme_comparison.py")
    spec = importlib.util.spec_from_file_location("partition_cmp", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_stable_hash_bucket_is_deterministic_and_in_range():
    module = load_module()

    buckets_a = [module.stable_hash_bucket(point_id, 44) for point_id in range(1, 20)]
    buckets_b = [module.stable_hash_bucket(point_id, 44) for point_id in range(1, 20)]

    assert buckets_a == buckets_b
    assert all(0 <= bucket < 44 for bucket in buckets_a)


def test_balanced_assign_respects_capacities():
    module = load_module()

    order = np.array(
        [
            [0, 1, 2],
            [0, 1, 2],
            [0, 1, 2],
            [1, 0, 2],
            [1, 0, 2],
            [2, 1, 0],
        ],
        dtype=np.uint8,
    )
    priorities = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4], dtype=np.float32)
    capacities = [2, 2, 2]

    assignments = module.balanced_assign_from_rankings(order, priorities, capacities)

    counts = np.bincount(assignments, minlength=3).tolist()
    assert counts == [2, 2, 2]


def test_choose_best_matched_recall_prefers_highest_qps_among_valid_rows():
    module = load_module()

    rows = [
        {"scheme": "a", "recall_at_k": 0.87, "qps": 200.0},
        {"scheme": "b", "recall_at_k": 0.88, "qps": 180.0},
        {"scheme": "c", "recall_at_k": 0.90, "qps": 220.0},
    ]

    chosen = module.choose_best_matched_recall(rows, target_recall=0.88)

    assert chosen["scheme"] == "c"
