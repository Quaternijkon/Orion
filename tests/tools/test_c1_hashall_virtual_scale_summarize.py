from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest


SOURCE = (
    Path(__file__).resolve().parents[2]
    / "experiments"
    / "c1"
    / "scripts"
    / "c1_hashall_virtual_scale_summarize.py"
)
LIVE_ROOT = Path(
    "/proj/intelisys-PG0/exp/orion-distributed/"
    "hashall-virtual-linear-cpu-20260824"
)


def load_module():
    spec = importlib.util.spec_from_file_location(
        "c1_hashall_virtual_scale_summarize", SOURCE
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_expected_shard_distribution_covers_virtual_scales():
    module = load_module()
    assert module.expected_shards_per_node(1) == [1, 0, 0, 0]
    assert module.expected_shards_per_node(2) == [1, 1, 0, 0]
    assert module.expected_shards_per_node(4) == [1, 1, 1, 1]
    assert module.expected_shards_per_node(8) == [2, 2, 2, 2]
    assert module.expected_shards_per_node(16) == [4, 4, 4, 4]
    assert module.expected_shards_per_node(32) == [8, 8, 8, 8]


def test_anchored_log_fit_recovers_qps_per_doubling():
    module = load_module()
    machines = np.asarray([1, 2, 4, 8, 16, 32], dtype=float)
    observed = 7_000.0 + 2_100.0 * np.log2(machines)
    slope, predicted, metrics = module.fit_anchored_log2(machines, observed)
    assert slope == pytest.approx(2_100.0)
    assert predicted == pytest.approx(observed)
    assert metrics["r_squared"] == pytest.approx(1.0)


def test_ef_aware_theory_reduces_local_distance_proxy():
    module = load_module()
    machines = np.asarray([1, 2, 4, 8, 16, 32], dtype=float)
    ef = np.asarray([24, 18, 14, 10, 10, 10], dtype=float)
    distance, qps = module.theoretical_qps(machines, ef, 7_183.0)
    assert np.all(np.diff(distance) < 0)
    assert np.all(np.diff(qps) > 0)
    assert qps[-1] / qps[0] == pytest.approx(2.2347616724)


def test_glove_metadata_and_theory_use_artifact_dataset_size(tmp_path):
    module = load_module()
    (tmp_path / "m1").mkdir()
    (tmp_path / "preflight.json").write_text(
        json.dumps(
            {
                "dataset": {
                    "key": "glove-200-angular",
                    "name": "GloVe-200-angular",
                    "path": "/datasets/glove.hdf5",
                    "hdf5_distance": "angular",
                    "qdrant_distance": "Cosine",
                    "shapes": {
                        "train": [1_183_514, 200],
                        "test": [10_000, 200],
                        "neighbors": [10_000, 100],
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "m1" / "benchmark.json").write_text(
        json.dumps(
            {
                "protocol": {
                    "dataset": "GloVe-200-angular",
                    "dataset_key": "glove-200-angular",
                    "distance": "Cosine",
                    "target_recall_at_10": 0.9,
                    "ef_lower_bound": 10,
                    "ef_upper_bound": 512,
                },
                "parameters": {"top_k": 10, "selected_hnsw_ef": 384},
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "m1" / "prepare.json").write_text(
        json.dumps(
            {
                "collection_body": {
                    "vectors": {"size": 200, "distance": "Cosine"},
                    "hnsw_config": {"m": 32, "ef_construct": 200},
                }
            }
        ),
        encoding="utf-8",
    )

    metadata = module.experiment_metadata(tmp_path)
    assert metadata["dataset_name"] == "GloVe-200-angular"
    assert metadata["train_count"] == 1_183_514
    assert metadata["vector_size"] == 200
    assert metadata["qdrant_distance"] == "Cosine"
    machines = np.asarray([1, 2, 4], dtype=float)
    ef = np.asarray([384, 256, 160], dtype=float)
    distance, _ = module.theoretical_qps(
        machines,
        ef,
        433.0,
        dataset_size=metadata["train_count"],
        hnsw_m=metadata["hnsw_m"],
    )
    default_distance, _ = module.theoretical_qps(machines, ef, 433.0)
    assert not np.array_equal(distance, default_distance)


@pytest.mark.skipif(
    not (LIVE_ROOT / "m32" / "benchmark.json").is_file(),
    reason="authoritative live experiment artifacts are not present",
)
def test_authoritative_six_point_run_passes_completion_audit():
    module = load_module()
    rows, checks = module.audit_and_collect(LIVE_ROOT)
    model = module.add_derived_columns(rows)
    assert [row["logical_shards"] for row in rows] == [1, 2, 4, 8, 16, 32]
    assert all(check["status"] == "PASS" for check in checks)
    assert rows[-1]["allocated_cpu_cores"] == pytest.approx(64.0)
    assert rows[-1]["heldout_recall_at_10"] >= 0.90
    assert model["fit_metrics"]["r_squared"] > 0.97
    assert model["theory_metrics"]["r_squared"] > 0.90
