import importlib.util
from pathlib import Path


def load_tool():
    path = Path(__file__).resolve().parents[2] / "tools" / "method4_claim_e_payload_bytes.py"
    spec = importlib.util.spec_from_file_location("method4_claim_e_payload_bytes", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_payload_batch_rows_expand_shard_major_requests():
    tool = load_tool()
    plans = [
        {
            "searches": [
                {
                    "vector": [0.1, 0.2],
                    "limit": 10,
                    "params": {"hnsw_ef": 80},
                    "with_payload": ["source_id"],
                    "with_vector": False,
                    "shard_key": ["shard_0", "shard_1"],
                    "hnsw_entry_points_by_shard": {
                        "shard_0": [11, 12],
                        "shard_1": [21],
                    },
                    "hnsw_ef_by_shard": {"shard_0": 96, "shard_1": 80},
                }
            ],
            "visited_shards": 2,
            "assigned_ef_sum": 176,
        },
        {
            "searches": [
                {
                    "vector": [0.3, 0.4],
                    "limit": 10,
                    "params": {"hnsw_ef": 72},
                    "with_payload": ["source_id"],
                    "with_vector": False,
                    "shard_key": ["shard_2"],
                }
            ],
            "visited_shards": 1,
            "assigned_ef_sum": 72,
        },
    ]

    compact_rows = tool.payload_batch_rows(
        plans,
        variant="compact_current",
        batch_size=2,
        lower_execution_order="query_major",
    )
    expanded_rows = tool.payload_batch_rows(
        plans,
        variant="client_shard_major_expanded",
        batch_size=2,
        lower_execution_order="shard_major",
    )

    assert compact_rows[0]["search_request_count"] == 2
    assert expanded_rows[0]["search_request_count"] == 3
    assert expanded_rows[0]["request_payload_bytes"] > compact_rows[0]["request_payload_bytes"]
    assert expanded_rows[0]["request_payload_bytes_per_query"] > compact_rows[0]["request_payload_bytes_per_query"]


def test_summarize_payload_rows_reports_reduction_against_baseline():
    tool = load_tool()
    rows = [
        {
            "variant": "client_shard_major_expanded",
            "batch_size": 2,
            "query_count": 2,
            "search_request_count": 6,
            "request_payload_bytes": 600,
            "request_payload_bytes_per_query": 300,
        },
        {
            "variant": "compact_current",
            "batch_size": 2,
            "query_count": 2,
            "search_request_count": 2,
            "request_payload_bytes": 240,
            "request_payload_bytes_per_query": 120,
        },
    ]

    summary = tool.summarize_payload_rows(
        rows,
        baseline_variant="client_shard_major_expanded",
    )

    compact = next(row for row in summary if row["variant"] == "compact_current")
    assert compact["request_count_reduction_vs_baseline_pct"] == 66.66666666666666
    assert compact["request_payload_bytes_reduction_vs_baseline_pct"] == 60.0
