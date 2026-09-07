from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "experiments/c6/scripts/c6_protocol.py"


def load_module():
    spec = importlib.util.spec_from_file_location("c6_protocol", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def route(module):
    evidence = (
        module.ShardEvidence(0, 4, 1, 0.1, 0.25, (10, 11, 12, 13)),
        module.ShardEvidence(1, 3, 2, 0.2, 0.30, (20, 21, 22)),
        module.ShardEvidence(2, 1, 8, 0.8, 0.80, (30,)),
    )
    return module.RouteQuery(
        query_id=0,
        navigation_candidate_count=8,
        candidate_shard_count=3,
        ranked_candidate_shards=(0, 1, 2),
        adaptive_selected_shards=(0, 1, 2),
        evidence=evidence,
    )


def trace(module, shard_id, ef_search, ids, work, nodes=None):
    return module.LocalSearchTrace(
        query_id=0,
        shard_id=shard_id,
        ef_search=ef_search,
        result_ids=tuple(ids),
        result_scores=tuple(float(point_id) for point_id in ids),
        distance_computations=work,
        nodes_visited=nodes if nodes is not None else work // 2,
        worker_cpu_time_us=work + 1,
        worker_cpu_wall_time_us=work + 2,
        latency_us=float(work + 3),
    )


def oracle_cube(module):
    rows = [
        trace(module, 0, 8, [0, 1, 2, 3], 5),
        trace(module, 0, 16, [0, 1, 2, 3, 4], 7),
        trace(module, 1, 8, [5, 6, 7, 8], 6),
        trace(module, 1, 16, [5, 6, 7, 8, 9], 20),
        trace(module, 2, 8, [90, 91], 1),
        trace(module, 2, 16, [90, 91, 92], 2),
    ]
    return module.local_trace_lookup(rows)


def test_deterministic_shard_order_uses_count_rank_then_id():
    module = load_module()
    rows = [
        module.ShardEvidence(3, 2, 1, 0.1, 0.2, (1, 2)),
        module.ShardEvidence(1, 3, 4, 0.1, 0.2, (3, 4, 5)),
        module.ShardEvidence(0, 3, 2, 0.1, 0.2, (6, 7, 8)),
        module.ShardEvidence(2, 2, 1, 0.1, 0.2, (9, 10)),
    ]
    assert module.deterministic_shard_order(rows) == (0, 1, 2, 3)


def test_route_trace_loader_requires_c6_evidence_and_replays_ranking():
    module = load_module()
    payload = {
        "format_version": 1,
        "artifact": {"shard_count": 2},
        "per_query": [
            {
                "query_index": 0,
                "navigation_candidate_count": 3,
                "candidate_shard_count": 2,
                "candidates": [
                    {"rank": 1, "label": 10, "distance": 0.1, "shard_ids": [0]},
                    {"rank": 2, "label": 20, "distance": 0.2, "shard_ids": [1]},
                    {"rank": 3, "label": 30, "distance": 0.3, "shard_ids": [1]},
                ],
                "ranked_candidate_shards": [1, 0],
                "shard_evidence": [
                    {
                        "shard_id": 0,
                        "candidate_count": 1,
                        "best_candidate_rank": 1,
                        "minimum_candidate_distance": 0.1,
                        "mean_candidate_distance": 0.1,
                        "entry_points": [10],
                    },
                    {
                        "shard_id": 1,
                        "candidate_count": 2,
                        "best_candidate_rank": 2,
                        "minimum_candidate_distance": 0.2,
                        "mean_candidate_distance": 0.25,
                        "entry_points": [20, 30],
                    },
                ],
                "targets": [
                    {"shard_id": 0, "ef": 12, "entry_points": [10]},
                    {"shard_id": 1, "ef": 16, "entry_points": [20, 30]},
                ],
            }
        ],
    }
    loaded = module.route_queries_from_trace(payload)
    assert loaded[0].ranked_candidate_shards == (1, 0)
    assert loaded[0].adaptive_selected_shards == (1, 0)

    payload["per_query"][0]["ranked_candidate_shards"] = [0, 1]
    with pytest.raises(ValueError, match="deterministic ranking"):
        module.route_queries_from_trace(payload)


def test_p0_p1_p2_p3_share_routes_and_apply_expected_ef_policy():
    module = load_module()
    route_row = route(module)
    traces = []
    for shard_id in range(3):
        for ef_search in (8, 16, 20, 24):
            traces.append(
                trace(
                    module,
                    shard_id,
                    ef_search,
                    [shard_id * 3 + offset for offset in range(3)],
                    ef_search + shard_id,
                )
            )
    lookup = module.local_trace_lookup(traces)
    truth = list(range(10))

    p0 = module.evaluate_policy_query(
        route_row,
        module.PolicyConfig("P0", fixed_p=2, uniform_ef_search=16),
        lookup,
        truth,
        dataset="sift1m",
        logical_shards=4,
        score_higher_is_better=False,
    )
    assert p0["selected_shard_ids"] == [0, 1]
    assert p0["assigned_efSearch_per_shard"] == [16, 16]

    p1 = module.evaluate_policy_query(
        route_row,
        module.PolicyConfig("P1", uniform_ef_search=16),
        lookup,
        truth,
        dataset="sift1m",
        logical_shards=4,
        score_higher_is_better=False,
    )
    assert p1["selected_shard_ids"] == [0, 1, 2]

    p2 = module.evaluate_policy_query(
        route_row,
        module.PolicyConfig("P2", fixed_p=2, alpha=4, beta=4),
        lookup,
        truth,
        dataset="sift1m",
        logical_shards=4,
        score_higher_is_better=False,
    )
    assert p2["assigned_efSearch_per_shard"] == [20, 16]

    p3 = module.evaluate_policy_query(
        route_row,
        module.PolicyConfig("P3", alpha=4, beta=4),
        lookup,
        truth,
        dataset="sift1m",
        logical_shards=4,
        score_higher_is_better=False,
    )
    assert p3["assigned_efSearch_per_shard"] == [20, 16, 8]


def test_fixed_p_appends_zero_evidence_shards_but_adaptive_does_not():
    module = load_module()
    route_row = module.RouteQuery(
        query_id=0,
        navigation_candidate_count=2,
        candidate_shard_count=1,
        ranked_candidate_shards=(2,),
        adaptive_selected_shards=(2,),
        evidence=(module.ShardEvidence(2, 2, 1, 0.1, 0.2, (10, 11)),),
        logical_shards=4,
    )
    p2 = module.PolicyConfig("P2", fixed_p=4, alpha=4, beta=8)
    shards = module.policy_shards(route_row, p2)
    assert shards == (2, 0, 1, 3)
    assert module.policy_efs(route_row, shards, p2) == (16, 8, 8, 8)
    assert module.policy_shards(
        route_row, module.PolicyConfig("P1", uniform_ef_search=16)
    ) == (2,)


def test_merge_deduplicates_copies_and_obeys_metric_direction():
    module = load_module()
    left = module.LocalSearchTrace(0, 0, 8, (7, 8), (0.8, 0.2), 1, 1, 1, 1, 1.0)
    right = module.LocalSearchTrace(0, 1, 8, (7, 9), (0.9, 0.3), 1, 1, 1, 1, 1.0)
    assert module.merge_search_results(
        [left, right], score_higher_is_better=True, top_k=3
    ) == (7, 9, 8)
    assert module.merge_search_results(
        [left, right], score_higher_is_better=False, top_k=3
    ) == (8, 9, 7)


def test_p4_oracle_prefix_uses_deployed_ranking_and_high_fixed_ef():
    module = load_module()
    result = module.oracle_prefix_query(
        route(module),
        oracle_cube(module),
        list(range(10)),
        high_ef_search=16,
        score_higher_is_better=False,
    )
    assert result.reached_target is True
    assert result.selected_shards == (0, 1)
    assert result.assigned_efs == (16, 16)
    assert result.recall_at_10 == 1.0


def test_p5_dynamic_programming_finds_exact_minimum_work_allocation():
    module = load_module()
    result = module.oracle_local_budget_query(
        route(module),
        (0, 1, 2),
        oracle_cube(module),
        list(range(10)),
        ef_grid=(8, 16),
        score_higher_is_better=False,
    )
    assert result.reached_target is True
    assert result.assigned_efs == (16, 8, 8)
    assert result.aggregate_distance_computations == 14
    assert result.recall_at_10 == 0.9


def test_p6_combines_prefix_and_budget_oracles():
    module = load_module()
    result = module.full_oracle_query(
        route(module),
        oracle_cube(module),
        list(range(10)),
        ef_grid=(8, 16),
        score_higher_is_better=False,
    )
    assert result.reached_target is True
    assert result.selected_shards == (0, 1)
    assert result.assigned_efs == (16, 8)
    assert result.aggregate_distance_computations == 13


def test_tuning_selection_applies_two_percent_tail_tiebreak():
    module = load_module()
    candidates = [
        {
            "name": "minimum-mean",
            "recall_at_10": 0.91,
            "aggregate_distance_computations_per_query": 100.0,
            "p99_max_shard_distance_computations": 90.0,
        },
        {
            "name": "lower-tail",
            "recall_at_10": 0.90,
            "aggregate_distance_computations_per_query": 101.5,
            "p99_max_shard_distance_computations": 70.0,
        },
        {
            "name": "outside-band",
            "recall_at_10": 0.95,
            "aggregate_distance_computations_per_query": 103.0,
            "p99_max_shard_distance_computations": 10.0,
        },
    ]
    assert module.select_tuned_candidate(candidates)["name"] == "lower-tail"


def test_static_waste_classification_is_paired_and_exhaustive():
    module = load_module()
    static = [
        {"query_id": 0, "recall_at_10": 0.8, "aggregate_distance_computations": 50},
        {"query_id": 1, "recall_at_10": 0.9, "aggregate_distance_computations": 105},
        {"query_id": 2, "recall_at_10": 1.0, "aggregate_distance_computations": 150},
    ]
    oracle = [
        module.OracleResult("P6", index, True, (0,), (8,), 0.9, 100, 100, 1, 1, ())
        for index in range(3)
    ]
    summary = module.classify_static_waste(static, oracle)
    assert summary["under_searched"] == pytest.approx(1 / 3)
    assert summary["near_sufficient"] == pytest.approx(1 / 3)
    assert summary["over_searched"] == pytest.approx(1 / 3)


def test_bootstrap_is_deterministic_and_persistence_is_atomic(tmp_path):
    module = load_module()
    first = module.paired_bootstrap_relative_reduction(
        [100, 200, 300], [80, 160, 240], repetitions=200, seed=7
    )
    second = module.paired_bootstrap_relative_reduction(
        [100, 200, 300], [80, 160, 240], repetitions=200, seed=7
    )
    assert first == second
    assert first["mean_relative_reduction"] == pytest.approx(0.2)

    json_path = module.write_json_atomic(tmp_path / "summary.json", {"ok": True})
    assert json.loads(json_path.read_text()) == {"ok": True}
    module.append_jsonl(tmp_path / "manifest.jsonl", {"stage": 0})
    assert json.loads((tmp_path / "manifest.jsonl").read_text()) == {"stage": 0}

    batched = tmp_path / "batched.jsonl"
    with module.DurableJsonlBatchWriter(batched, fsync_interval=2) as writer:
        writer.append({"row": 1})
        writer.append({"row": 2})
        writer.append({"row": 3})
    assert [json.loads(line) for line in batched.read_text().splitlines()] == [
        {"row": 1},
        {"row": 2},
        {"row": 3},
    ]
