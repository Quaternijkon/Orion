from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


SOURCE = (
    Path(__file__).resolve().parents[2]
    / "experiments"
    / "c1"
    / "scripts"
    / "c1_hashall_virtual_scale.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("c1_hashall_virtual_scale", SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_linear_cpu_quota_schedule_is_exact():
    module = load_module()
    for logical_shards in module.LOGICAL_SHARD_COUNTS:
        quotas = module.quota_schedule(logical_shards)
        assert sum(quotas) == pytest.approx(2.0 * logical_shards)
        assert len(quotas) == 4
        assert all(quota > 0 for quota in quotas)
    assert module.quota_schedule(32) == pytest.approx([16.0] * 4)


def test_round_robin_shard_counts_match_four_physical_hosts():
    module = load_module()
    assert module.shard_counts(1) == [1, 0, 0, 0]
    assert module.shard_counts(2) == [1, 1, 0, 0]
    assert module.shard_counts(4) == [1, 1, 1, 1]
    assert module.shard_counts(8) == [2, 2, 2, 2]
    assert module.shard_counts(16) == [4, 4, 4, 4]
    assert module.shard_counts(32) == [8, 8, 8, 8]


def test_cpuset_cpu_count_handles_ranges_and_singletons():
    module = load_module()
    assert module.cpuset_cpu_count("0-7,16-23") == 16
    assert module.cpuset_cpu_count("0-19") == 20
    assert module.cpuset_cpu_count("1,3,5-6") == 4


def test_low_m_control_plane_floor_is_deducted_from_total():
    module = load_module()
    assert module.quota_schedule(1) == pytest.approx([1.85, 0.05, 0.05, 0.05])
    assert module.quota_schedule(2) == pytest.approx([1.95, 1.95, 0.05, 0.05])


def test_quota_schedule_can_follow_live_permuted_placement():
    module = load_module()
    quotas = module.quota_schedule(2, shards_per_node=[1, 0, 1, 0])
    assert quotas == pytest.approx([1.95, 0.05, 1.95, 0.05])
    assert sum(quotas) == pytest.approx(4.0)
    with pytest.raises(ValueError, match="sum to 2"):
        module.quota_schedule(2, shards_per_node=[1, 1, 1, 0])


def test_saturation_selection_uses_smallest_point_within_two_percent_of_best():
    module = load_module()
    rows = [
        {"concurrency": 1, "qps": 100.0},
        {"concurrency": 2, "qps": 150.0},
        {"concurrency": 4, "qps": 198.0},
        {"concurrency": 8, "qps": 200.0},
        {"concurrency": 16, "qps": 197.0},
    ]
    selected = module.select_saturation(rows)
    assert selected["selected_concurrency"] == 4
    assert selected["knee_observed"] is True


def test_glove_dataset_configuration_uses_cosine_and_bounded_ef_grid():
    module = load_module()
    spec = module.DATASET_SPECS["glove-200-angular"]
    assert spec.train_count == 1_183_514
    assert spec.vector_size == 200
    assert spec.hdf5_distance == "angular"
    assert spec.qdrant_distance == "Cosine"
    assert spec.ef_candidates[-1] == 512
    assert tuple(sorted(set(spec.ef_candidates))) == spec.ef_candidates


def test_glove_collection_body_and_default_arguments_are_dataset_specific():
    module = load_module()
    args = module.parse_args(["--dataset", "glove-200-angular", "--logical-shards", "1"])
    body = module.collection_body(1, args.dataset_spec)
    assert body["vectors"] == {"size": 200, "distance": "Cosine"}
    assert args.hdf5_path.name == "glove-200-angular.hdf5"
    assert args.output_root.name == "hashall-virtual-linear-cpu-glove-20260825"
    assert args.ef_candidates[0] == 10
    assert args.ef_candidates[-1] == 512
