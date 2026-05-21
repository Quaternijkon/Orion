from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def load_module():
    script_path = Path("/home/taig/dry/qdrant/tools/hnsw_experiment.py")
    spec = importlib.util.spec_from_file_location("hnsw_exp", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_wait_completion_accepts_green_collection_with_no_optimizer_work():
    module = load_module()

    assert module.indexing_wait_is_complete(
        points_count=10000,
        expected_points=10000,
        indexed_vectors_count=9809,
        running=[],
        queued=[],
    )


def test_wait_completion_rejects_incomplete_points():
    module = load_module()

    assert not module.indexing_wait_is_complete(
        points_count=9999,
        expected_points=10000,
        indexed_vectors_count=9999,
        running=[],
        queued=[],
    )


def test_wait_completion_rejects_running_optimizer_work():
    module = load_module()

    assert not module.indexing_wait_is_complete(
        points_count=10000,
        expected_points=10000,
        indexed_vectors_count=9809,
        running=[{"optimizer": "indexing"}],
        queued=[],
    )
