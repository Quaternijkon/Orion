from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


def load_module():
    script_path = Path("/home/taig/dry/qdrant/tools/method4_claim_b_oracle_miss_analysis.py")
    spec = importlib.util.spec_from_file_location("method4_claim_b_oracle_miss_analysis", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_analyze_oracle_counts_gt_item_coverage_and_miss_rate():
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
            [3, 4, 5],
            [0, 1, 2],
        ],
        dtype=np.int64,
    )
    point_to_shards = [[0], [1], [2], [1, 2], [0], [2]]

    rows = module.analyze_oracle(
        strategy="toy",
        labels=labels,
        neighbors=neighbors,
        point_to_shards=point_to_shards,
        upper_ks=[1, 2],
        expansion_ratio=1.0,
    )

    by_upper_k = {row["upper_k"]: row for row in rows}
    assert by_upper_k[1]["oracle_gt_coverage_at_3"] == pytest.approx(2 / 6)
    assert by_upper_k[1]["oracle_gt_miss_at_3"] == pytest.approx(4 / 6)
    assert by_upper_k[1]["avg_routed_shards"] == 1.0
    assert by_upper_k[1]["query_all_gt_covered_rate"] == 0.0

    assert by_upper_k[2]["oracle_gt_coverage_at_3"] == pytest.approx(4 / 6)
    assert by_upper_k[2]["oracle_gt_miss_at_3"] == pytest.approx(2 / 6)
    assert by_upper_k[2]["avg_routed_shards"] == 2.0
    assert by_upper_k[2]["query_all_gt_covered_rate"] == 0.0
