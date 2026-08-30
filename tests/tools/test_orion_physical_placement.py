from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def load_script(relative: str, module_name: str):
    path = REPO_ROOT / relative
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


plan = load_script(
    "experiments/load_balance/orion_physical_placement_plan.py",
    "orion_physical_placement_plan_test",
)
online = load_script(
    "experiments/load_balance/orion_physical_placement_online.py",
    "orion_physical_placement_online_test",
)


def test_percentile_matches_linear_interpolation() -> None:
    assert plan.percentile([0.0, 10.0], 0.95) == pytest.approx(9.5)
    assert plan.percentile([3.0], 0.99) == 3.0


def test_controller_tail_bins_cover_every_shard_once() -> None:
    flattened = [shard for group in plan.CONTROLLER_TAIL_BINS for shard in group]
    assert sorted(flattened) == list(range(32))
    assert [len(group) for group in plan.CONTROLLER_TAIL_BINS] == [8, 8, 8, 8]


def test_assign_bins_keeps_controller_bin_fixed_and_minimizes_moves() -> None:
    bins = [tuple(range(index, 32, 4)) for index in range(4)]
    peers = [10, 20, 30, 40]
    baseline = {shard: peers[shard % 4] for shard in range(32)}
    weights = [shard + 1 for shard in range(32)]
    result = plan.assign_bins_min_movement(
        bins,
        peers,
        baseline,
        weights,
        fixed_first_peer=10,
    )
    assert result == baseline


def test_aggregate_repeat_metrics() -> None:
    repeats = [
        {
            "qps": 100.0,
            "cpu_average_cores": {"n1": 3.0, "n2": 4.0},
            "batch_latency_ms": {"p50": 10.0, "p95": 20.0, "p99": 30.0, "max": 40.0},
        },
        {
            "qps": 110.0,
            "cpu_average_cores": {"n1": 5.0, "n2": 6.0},
            "batch_latency_ms": {"p50": 12.0, "p95": 22.0, "p99": 32.0, "max": 42.0},
        },
    ]
    result = online.aggregate_repeat_metrics(repeats)
    assert result["qps_mean"] == 105.0
    assert result["cpu_average_cores_mean_across_repeats"] == {"n1": 4.0, "n2": 5.0}
    assert result["batch_latency_ms_mean_across_repeats"]["p95"] == 21.0


class FakeExperiment:
    def __init__(self, indexed_vectors_count: int):
        self.indexed_vectors_count = indexed_vectors_count

    def collection_info(self, _base_url: str, _collection: str):
        return {
            "status": "green",
            "optimizer_status": "ok",
            "points_count": online.EXPECTED_POINT_COUNT,
            "indexed_vectors_count": self.indexed_vectors_count,
            "config": {
                "params": {
                    "shard_number": online.EXPECTED_SHARD_COUNT,
                    "replication_factor": 1,
                },
                "hnsw_config": {"m": 32, "ef_construct": 200},
                "auto_shard_policy": {
                    "type": "orion",
                    "generation": online.EXPECTED_GENERATION,
                    "artifact_sha256": online.EXPECTED_ARTIFACT_SHA256,
                },
            },
        }


def test_collection_contract_accepts_full_scan_tail_below_threshold() -> None:
    args = SimpleNamespace(base_url="http://controller", collection="test")
    result = online.verify_collection_contract(
        FakeExperiment(online.EXPECTED_POINT_COUNT - 9), args
    )
    assert result["full_scan_tail_vector_count"] == 9


def test_collection_contract_rejects_tail_at_threshold() -> None:
    args = SimpleNamespace(base_url="http://controller", collection="test")
    with pytest.raises(RuntimeError, match="unindexed_tail=10"):
        online.verify_collection_contract(
            FakeExperiment(online.EXPECTED_POINT_COUNT - 10), args
        )
