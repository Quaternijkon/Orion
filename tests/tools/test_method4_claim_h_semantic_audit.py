from __future__ import annotations

import importlib.util
from pathlib import Path


def load_module():
    script_path = Path("/home/taig/dry/qdrant/tools/method4_claim_h_semantic_audit.py")
    spec = importlib.util.spec_from_file_location("method4_claim_h_semantic_audit", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def compact_plan():
    return {
        "query_index": 0,
        "upper_labels": [1, 2, 3],
        "visited_shards": 3,
        "upper_hits": 3,
        "assigned_ef_sum": 84,
        "assigned_ef_count": 3,
        "searches": [
            {
                "vector": [0.1, 0.2],
                "limit": 10,
                "params": {"hnsw_ef": 28},
                "with_payload": False,
                "with_vector": False,
                "shard_key": ["centroid_00", "centroid_01", "centroid_02"],
                "hnsw_entry_points_by_shard": {
                    "centroid_00": [2],
                    "centroid_01": [1004],
                    "centroid_02": [2002, 2004],
                },
                "hnsw_ef_by_shard": {
                    "centroid_00": 24,
                    "centroid_01": 24,
                    "centroid_02": 36,
                },
                "source_id_dedup_block_size": 1001,
            }
        ],
    }


def test_audit_query_plans_passes_all_external_method4_invariants():
    module = load_module()

    rows, samples = module.audit_query_plans(
        [compact_plan()],
        point_to_shards=[[], [0, 2], [1], [2]],
        source_id_dedup_block_size=1001,
        base_ef=20,
        factor=4,
        top_k=10,
    )

    assert {row["status"] for row in rows} == {"pass"}
    assert {row["invariant"] for row in rows} == {
        "upper routing unchanged",
        "point_to_shards unchanged",
        "multi-assignment preserved",
        "per-shard entry points preserved",
        "dynamic EF formula preserved",
        "lower search remains per logical shard",
        "no adaptive shard pruning",
        "source-id dedup preserved",
        "final global merge preserved",
    }
    assert samples[0]["routed_shards"] == ["centroid_00", "centroid_01", "centroid_02"]
    assert samples[0]["upper_labels"] == "1 2 3"


def test_audit_query_plans_fails_when_dynamic_ef_formula_is_wrong():
    module = load_module()
    plan = compact_plan()
    plan["searches"][0]["hnsw_ef_by_shard"]["centroid_02"] = 28

    rows, _samples = module.audit_query_plans(
        [plan],
        point_to_shards=[[], [0, 2], [1], [2]],
        source_id_dedup_block_size=1001,
        base_ef=20,
        factor=4,
        top_k=10,
    )

    by_name = {row["invariant"]: row for row in rows}
    assert by_name["dynamic EF formula preserved"]["status"] == "fail"
    assert "centroid_02 expected 36 got 28" in by_name["dynamic EF formula preserved"]["detail"]
