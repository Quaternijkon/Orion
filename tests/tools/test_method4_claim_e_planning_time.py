import importlib.util
from pathlib import Path


def load_tool():
    path = Path(__file__).resolve().parents[2] / "tools" / "method4_claim_e_planning_time.py"
    spec = importlib.util.spec_from_file_location("method4_claim_e_planning_time", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_executed_request_count_uses_shard_major_expansion():
    tool = load_tool()
    plans = [
        {
            "searches": [
                {"shard_key": ["shard_0", "shard_1"]},
                {"shard_key": ["shard_2"]},
            ]
        },
        {"searches": [{"shard_key": ["shard_3", "shard_4", "shard_5"]}]},
    ]

    assert tool.executed_request_count(plans, lower_execution_order="query_major") == 3
    assert tool.executed_request_count(plans, lower_execution_order="shard_major") == 6


def test_planning_summary_includes_upper_and_variant_time_per_query():
    tool = load_tool()

    row = tool.planning_summary_row(
        variant="compact_current",
        lower_execution_order="query_major",
        query_count=100,
        upper_label_time_s=0.25,
        plan_build_time_s=0.75,
        search_request_count=100,
        candidate_group_count=100,
        returned_candidate_count=1000,
    )

    assert row["upper_label_ms_per_query"] == 2.5
    assert row["plan_build_ms_per_query"] == 7.5
    assert row["total_planning_ms_per_query"] == 10.0
    assert row["avg_search_requests_per_query"] == 1.0
    assert row["avg_returned_candidates_per_query"] == 10.0
