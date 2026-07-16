import importlib.util
from pathlib import Path

import pytest


def load_module():
    module_path = Path("tools/method4_claim_e_execution_mode_latency.py")
    spec = importlib.util.spec_from_file_location("method4_claim_e_execution_mode_latency", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_variant_specs_define_execution_modes_and_lower_order():
    module = load_module()

    specs = module.variant_specs(["grouped_by_ef_materialized", "compact_current", "client_shard_major_expanded"])

    assert specs == [
        {
            "variant": "grouped_by_ef_materialized",
            "routed_execution_mode": "grouped_by_ef",
            "lower_execution_order": "query_major",
            "description": "materialized JSON route plan grouped by EF",
        },
        {
            "variant": "compact_current",
            "routed_execution_mode": "compact_multi_ep",
            "lower_execution_order": "query_major",
            "description": "compact MultiEP request on the current server path",
        },
        {
            "variant": "client_shard_major_expanded",
            "routed_execution_mode": "compact_multi_ep",
            "lower_execution_order": "shard_major",
            "description": "client-expanded single-shard negative control",
        },
    ]

    with pytest.raises(ValueError, match="unsupported variant"):
        module.variant_specs(["unknown"])


def test_summarize_variant_rows_reports_request_and_latency_metrics():
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
            "search_batch_calls": 1,
        },
        {
            "hits": 82,
            "query_count": 10,
            "wall_s": 0.3,
            "batch_latency_ms": 300.0,
            "visited_shards": 230,
            "assigned_ef_sum": 33000,
            "search_request_count": 20,
            "candidate_group_count": 20,
            "returned_candidate_count": 200,
            "search_batch_calls": 1,
        },
    ]

    summary = module.summarize_variant_rows(
        "client_shard_major_expanded",
        rows,
        top_k=10,
    )

    assert summary["variant"] == "client_shard_major_expanded"
    assert summary["query_count"] == 20
    assert summary["recall_at_10"] == pytest.approx(0.81)
    assert summary["qps"] == pytest.approx(40.0)
    assert summary["avg_search_requests_per_query"] == pytest.approx(12.5)
    assert summary["avg_candidate_groups_per_query"] == pytest.approx(12.5)
    assert summary["avg_returned_candidates_per_query"] == pytest.approx(125.0)
    assert summary["batch_latency_p50_ms"] == pytest.approx(250.0)
    assert summary["batch_latency_p95_ms"] == pytest.approx(295.0)


def test_failure_summary_row_preserves_failed_cell_metadata():
    module = load_module()

    row = module.failure_summary_row(
        {
            "variant": "client_shard_major_expanded",
            "description": "client-expanded single-shard negative control",
            "routed_execution_mode": "compact_multi_ep",
            "lower_execution_order": "shard_major",
        },
        batch_size=400,
        repeat=2,
        query_count=3000,
        top_k=10,
        collection="c",
        upper_k=160,
        base_ef=80,
        factor=8,
        warmup_query_count=100,
        error=RuntimeError("broken pipe"),
    )

    assert row["status"] == "error"
    assert row["variant"] == "client_shard_major_expanded"
    assert row["batch_size"] == 400
    assert row["repeat"] == 2
    assert row["query_count"] == 3000
    assert row["recall_at_10"] == ""
    assert row["qps"] == ""
    assert row["error_type"] == "RuntimeError"
    assert row["error_message"] == "broken pipe"
