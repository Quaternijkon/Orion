import importlib.util
from pathlib import Path

import pytest


def load_module():
    module_path = Path("tools/method4_claim_f_premerge_batch_latency.py")
    spec = importlib.util.spec_from_file_location("method4_claim_f_premerge_batch_latency", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_parse_direct_peer_urls_requires_peer_equals_url():
    module = load_module()

    assert module.parse_direct_peer_urls(["7=http://localhost:7007", "9=http://localhost:7009"]) == {
        7: "http://localhost:7007",
        9: "http://localhost:7009",
    }

    with pytest.raises(ValueError, match="PEER_ID=URL"):
        module.parse_direct_peer_urls(["http://localhost:7007"])


def test_summarize_variant_rows_reports_stream_reduction_and_latency():
    module = load_module()
    rows = [
        {
            "hits": 80,
            "query_count": 10,
            "wall_s": 0.2,
            "batch_latency_ms": 200.0,
            "visited_shards": 230,
            "assigned_ef_sum": 32000,
            "search_request_count": 230,
            "candidate_group_count": 230,
            "returned_candidate_count": 2300,
            "search_batch_calls": 3,
        },
        {
            "hits": 82,
            "query_count": 10,
            "wall_s": 0.3,
            "batch_latency_ms": 300.0,
            "visited_shards": 230,
            "assigned_ef_sum": 33000,
            "search_request_count": 230,
            "candidate_group_count": 30,
            "returned_candidate_count": 300,
            "search_batch_calls": 3,
        },
    ]

    summary = module.summarize_variant_rows(
        "direct_peer_premerge",
        rows,
        top_k=10,
        baseline_candidate_groups_per_query=23.0,
    )

    assert summary["variant"] == "direct_peer_premerge"
    assert summary["query_count"] == 20
    assert summary["recall_at_10"] == pytest.approx(0.81)
    assert summary["qps"] == pytest.approx(40.0)
    assert summary["avg_candidate_groups_per_query"] == pytest.approx(13.0)
    assert summary["candidate_group_reduction_vs_baseline_pct"] == pytest.approx(43.47826086956522)
    assert summary["batch_latency_p50_ms"] == pytest.approx(250.0)
    assert summary["batch_latency_p95_ms"] == pytest.approx(295.0)


def test_apply_direct_peer_baselines_is_order_independent():
    module = load_module()
    rows = [
        {
            "variant": "direct_peer_local_premerge",
            "batch_size": 50,
            "repeat": 2,
            "avg_candidate_groups_per_query": 3.0,
            "avg_returned_candidates_per_query": 30.0,
        },
        {
            "variant": "direct_peer_no_premerge",
            "batch_size": 50,
            "repeat": 2,
            "avg_candidate_groups_per_query": 24.0,
            "avg_returned_candidates_per_query": 240.0,
        },
    ]

    module.apply_direct_peer_baselines(rows)

    assert rows[0]["candidate_group_reduction_vs_baseline_pct"] == pytest.approx(87.5)
    assert rows[0]["returned_candidate_reduction_vs_baseline_pct"] == pytest.approx(87.5)
    assert rows[1]["candidate_group_reduction_vs_baseline_pct"] == 0.0
    assert rows[1]["returned_candidate_reduction_vs_baseline_pct"] == 0.0
