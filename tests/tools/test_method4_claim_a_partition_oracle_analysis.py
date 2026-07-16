from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


def load_module():
    script_path = Path("/home/taig/dry/qdrant/tools/method4_claim_a_partition_oracle_analysis.py")
    spec = importlib.util.spec_from_file_location("method4_claim_a_partition_oracle_analysis", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_min_shards_for_gt_uses_exact_set_cover():
    module = load_module()

    assert module.min_shards_for_gt([{0}, {1}, {2}]) == 3
    assert module.min_shards_for_gt([{0, 1}, {1}, {2}]) == 2
    assert module.min_shards_for_gt([{0, 1}, {1, 2}, {0, 2}]) == 2


def test_analyze_partition_oracle_counts_coverage_waste_and_entropy():
    module = load_module()
    labels = np.array(
        [
            [0, 1],
            [5, 4],
        ],
        dtype=np.int64,
    )
    neighbors = np.array(
        [
            [2, 3],
            [0, 1],
        ],
        dtype=np.int64,
    )
    point_to_shards = [[0], [1], [2], [1], [0], [1]]

    rows = module.analyze_partition_oracle(
        partition="toy",
        labels=labels,
        neighbors=neighbors,
        point_to_shards=point_to_shards,
        upper_ks=[1, 2],
        index_expansion_ratio=1.0,
        topology_edge_cut=0.25,
    )

    by_upper_k = {row["upper_k"]: row for row in rows}
    assert by_upper_k[1]["oracle_gt_coverage_at_2"] == pytest.approx(1 / 4)
    assert by_upper_k[1]["oracle_gt_miss_at_2"] == pytest.approx(3 / 4)
    assert by_upper_k[1]["avg_routed_shards"] == pytest.approx(1.0)
    assert by_upper_k[1]["routed_waste_ratio"] == pytest.approx(0.5)
    assert by_upper_k[1]["avg_min_shards_for_gt_at_2"] == pytest.approx(2.0)
    assert by_upper_k[1]["topology_edge_cut"] == pytest.approx(0.25)

    assert by_upper_k[2]["oracle_gt_coverage_at_2"] == pytest.approx(3 / 4)
    assert by_upper_k[2]["oracle_gt_miss_at_2"] == pytest.approx(1 / 4)
    assert by_upper_k[2]["avg_routed_shards"] == pytest.approx(2.0)
    assert by_upper_k[2]["routed_waste_ratio"] == pytest.approx(0.25)
    assert by_upper_k[2]["avg_min_shards_for_gt_at_2"] == pytest.approx(2.0)
