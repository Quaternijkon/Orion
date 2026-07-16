import csv
import importlib.util
from pathlib import Path


def load_module():
    module_path = Path("tools/method4_claim_cost_proxy_metrics.py")
    spec = importlib.util.spec_from_file_location("method4_claim_cost_proxy_metrics", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_aggregates_proxy_counts_and_reductions(tmp_path):
    proxy = load_module()
    claim_e = tmp_path / "claim_e.csv"
    claim_f = tmp_path / "claim_f.csv"
    output = tmp_path / "proxy.csv"

    common = {
        "query_count": "3000",
        "top_k": "10",
        "recall_at_10": "0.955",
        "qps": "100",
        "batch_latency_p95_ms": "10",
        "batch_latency_p99_ms": "11",
        "status": "ok",
        "collection": "c",
        "repeat": "1",
        "batch_size": "200",
    }
    write_csv(
        claim_e,
        [
            {
                **common,
                "variant": "client_shard_major_expanded",
                "avg_search_requests_per_query": "20",
                "avg_candidate_groups_per_query": "20",
                "avg_returned_candidates_per_query": "200",
            },
            {
                **common,
                "variant": "client_shard_major_expanded",
                "qps": "120",
                "repeat": "2",
                "avg_search_requests_per_query": "20",
                "avg_candidate_groups_per_query": "20",
                "avg_returned_candidates_per_query": "200",
            },
            {
                **common,
                "variant": "compact_current",
                "qps": "300",
                "avg_search_requests_per_query": "1",
                "avg_candidate_groups_per_query": "1",
                "avg_returned_candidates_per_query": "10",
            },
        ],
    )
    write_csv(
        claim_f,
        [
            {
                **common,
                "variant": "direct_peer_no_premerge",
                "avg_search_requests_per_query": "24",
                "avg_candidate_groups_per_query": "24",
                "avg_returned_candidates_per_query": "240",
            },
            {
                **common,
                "variant": "direct_peer_local_premerge",
                "avg_search_requests_per_query": "24",
                "avg_candidate_groups_per_query": "3",
                "avg_returned_candidates_per_query": "30",
            },
        ],
    )

    rows = proxy.build_proxy_rows(
        claim_e_summary=claim_e,
        claim_f_summary=claim_f,
        output_path=output,
    )

    compact = next(row for row in rows if row["claim_id"] == "E" and row["variant"] == "compact_current")
    assert compact["repeat_count"] == 1
    assert compact["qps_mean"] == 300.0
    assert compact["avg_search_requests_per_query_mean"] == 1.0
    assert compact["search_request_reduction_vs_baseline_pct"] == 95.0
    assert compact["candidate_group_reduction_vs_baseline_pct"] == 95.0
    assert compact["returned_candidate_reduction_vs_baseline_pct"] == 95.0

    premerge = next(row for row in rows if row["claim_id"] == "F" and row["variant"] == "direct_peer_local_premerge")
    assert premerge["search_request_reduction_vs_baseline_pct"] == 0.0
    assert premerge["candidate_group_reduction_vs_baseline_pct"] == 87.5
    assert premerge["returned_candidate_reduction_vs_baseline_pct"] == 87.5

    written = list(csv.DictReader(output.open()))
    assert len(written) == len(rows)
    assert written[0]["caveat"] == "counts_only_not_serialized_bytes_or_network_payload"


def test_missing_successful_baseline_leaves_reductions_blank(tmp_path):
    proxy = load_module()
    claim_e = tmp_path / "claim_e.csv"
    claim_f = tmp_path / "claim_f.csv"

    write_csv(
        claim_e,
        [
            {
                "variant": "client_shard_major_expanded",
                "query_count": "3000",
                "top_k": "10",
                "recall_at_10": "",
                "qps": "",
                "batch_latency_p95_ms": "",
                "batch_latency_p99_ms": "",
                "avg_search_requests_per_query": "",
                "avg_candidate_groups_per_query": "",
                "avg_returned_candidates_per_query": "",
                "status": "error",
                "collection": "c",
                "repeat": "1",
                "batch_size": "400",
            },
            {
                "variant": "compact_current",
                "query_count": "3000",
                "top_k": "10",
                "recall_at_10": "0.955",
                "qps": "300",
                "batch_latency_p95_ms": "10",
                "batch_latency_p99_ms": "11",
                "avg_search_requests_per_query": "1",
                "avg_candidate_groups_per_query": "1",
                "avg_returned_candidates_per_query": "10",
                "status": "ok",
                "collection": "c",
                "repeat": "1",
                "batch_size": "400",
            },
        ],
    )
    write_csv(
        claim_f,
        [
            {
                "variant": "direct_peer_local_premerge",
                "query_count": "3000",
                "top_k": "10",
                "recall_at_10": "0.955",
                "qps": "150",
                "batch_latency_p95_ms": "20",
                "batch_latency_p99_ms": "21",
                "avg_search_requests_per_query": "24",
                "avg_candidate_groups_per_query": "3",
                "avg_returned_candidates_per_query": "30",
                "status": "ok",
                "collection": "c",
                "repeat": "1",
                "batch_size": "200",
            }
        ],
    )

    rows = proxy.build_proxy_rows(claim_e_summary=claim_e, claim_f_summary=claim_f, output_path=None)

    compact = next(row for row in rows if row["claim_id"] == "E" and row["variant"] == "compact_current")
    assert compact["search_request_reduction_vs_baseline_pct"] == ""
    assert compact["candidate_group_reduction_vs_baseline_pct"] == ""
    assert compact["returned_candidate_reduction_vs_baseline_pct"] == ""
