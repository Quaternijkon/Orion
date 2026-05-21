from __future__ import annotations

import importlib.util
from pathlib import Path


def load_module():
    script_path = Path("/home/taig/dry/qdrant/tools/qdrant_custom_shard_adaptive_ef_experiment.py")
    spec = importlib.util.spec_from_file_location("adaptive_exp", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_rank_schedule_assigns_non_increasing_ef_by_rank():
    module = load_module()

    schedule = [
        (4, 128),
        (12, 96),
        (24, 64),
        (44, 32),
    ]

    ef_values = module.ef_schedule_for_nprobe(schedule, nprobe=10)

    assert ef_values == [128, 128, 128, 128, 96, 96, 96, 96, 96, 96]


def test_group_selected_keys_by_ef_collapses_same_ef_buckets():
    module = load_module()

    keys = [f"s{i}" for i in range(1, 9)]
    ef_values = [128, 128, 96, 96, 96, 64, 64, 32]

    grouped = module.group_selected_keys_by_ef(keys, ef_values)

    assert grouped == {
        128: ["s1", "s2"],
        96: ["s3", "s4", "s5"],
        64: ["s6", "s7"],
        32: ["s8"],
    }


def test_merge_topk_deduplicates_ids_and_keeps_highest_score():
    module = load_module()

    merged = module.merge_topk_candidates(
        [
            [(0.95, 10), (0.80, 20), (0.70, 30)],
            [(0.92, 20), (0.85, 40), (0.60, 50)],
            [(0.91, 10), (0.89, 60)],
        ],
        top_k=4,
    )

    assert merged == [10, 20, 60, 40]
