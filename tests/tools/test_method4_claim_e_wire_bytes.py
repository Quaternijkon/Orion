import importlib.util
import json
from pathlib import Path


def load_tool():
    path = Path(__file__).resolve().parents[2] / "tools" / "method4_claim_e_wire_bytes.py"
    spec = importlib.util.spec_from_file_location("method4_claim_e_wire_bytes", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_flatten_with_positions_expands_shard_major_searches():
    tool = load_tool()
    plans = [
        {
            "searches": [
                {
                    "vector": [0.1, 0.2],
                    "limit": 10,
                    "params": {"hnsw_ef": 80},
                    "shard_key": ["shard_0", "shard_1"],
                    "hnsw_entry_points_by_shard": {
                        "shard_0": [11],
                        "shard_1": [21, 22],
                    },
                    "hnsw_ef_by_shard": {"shard_0": 80, "shard_1": 96},
                }
            ]
        }
    ]

    positions, searches = tool.flatten_searches_with_positions(
        plans,
        lower_execution_order="shard_major",
    )

    assert positions == [0, 0]
    assert [search["shard_key"] for search in searches] == [["shard_0"], ["shard_1"]]
    assert searches[0]["hnsw_entry_points"] == [11]
    assert searches[1]["hnsw_entry_points"] == [21, 22]
    assert searches[1]["params"]["hnsw_ef"] == 96
    assert "hnsw_entry_points_by_shard" not in searches[0]


def test_parse_search_batch_response_counts_raw_bytes_and_source_ids():
    tool = load_tool()
    body = json.dumps(
        {
            "result": [
                [{"id": 12, "score": 0.9, "payload": {"source_id": 7}}],
                [{"id": 13, "score": 0.8}],
            ]
        }
    ).encode("utf-8")

    parsed = tool.parse_search_batch_response(body)

    assert parsed.response_body_bytes == len(body)
    assert parsed.results == [[(0.9, 7)], [(0.8, 12)]]


def test_summarize_wire_rows_reports_request_and_response_reductions():
    tool = load_tool()
    rows = [
        {
            "variant": "client_shard_major_expanded",
            "batch_size": 200,
            "status": "ok",
            "query_count": 200,
            "search_request_count": 4000,
            "request_body_bytes": 20_000,
            "response_body_bytes": 40_000,
            "total_body_bytes": 60_000,
            "wall_s": 2.0,
        },
        {
            "variant": "compact_current",
            "batch_size": 200,
            "status": "ok",
            "query_count": 200,
            "search_request_count": 200,
            "request_body_bytes": 2_000,
            "response_body_bytes": 4_000,
            "total_body_bytes": 6_000,
            "wall_s": 1.0,
        },
    ]

    summary = tool.summarize_wire_rows(
        rows,
        baseline_variant="client_shard_major_expanded",
    )

    compact = next(row for row in summary if row["variant"] == "compact_current")
    assert compact["request_body_bytes_reduction_vs_baseline_pct"] == 90.0
    assert compact["response_body_bytes_reduction_vs_baseline_pct"] == 90.0
    assert compact["total_body_bytes_reduction_vs_baseline_pct"] == 90.0
    assert compact["response_body_bytes_per_query"] == 20.0
