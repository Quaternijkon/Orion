import importlib.util
from pathlib import Path

import pytest


def load_tool():
    path = Path(__file__).resolve().parents[2] / "tools" / "method4_claim_e_qdrant_metrics.py"
    spec = importlib.util.spec_from_file_location("method4_claim_e_qdrant_metrics", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_parse_prometheus_metrics_handles_labels_and_histogram_suffixes():
    tool = load_tool()
    text = """
# HELP rest_responses_total number of REST responses
# TYPE rest_responses_total counter
rest_responses_total{method="POST",endpoint="/collections/{collection_name}/points/search/batch",status="200"} 10
rest_responses_duration_seconds_sum{method="POST",endpoint="/collections/{collection_name}/points/search/batch",status="200"} 2.5
memory_resident_bytes 1024
"""

    samples = tool.parse_prometheus_metrics(text)

    labels = {
        "method": "POST",
        "endpoint": "/collections/{collection_name}/points/search/batch",
        "status": "200",
    }
    assert tool.metric_value(samples, "rest_responses_total", labels) == pytest.approx(10)
    assert tool.metric_value(samples, "rest_responses_duration_seconds_sum", labels) == pytest.approx(2.5)
    assert tool.metric_value(samples, "memory_resident_bytes") == pytest.approx(1024)


def test_qdrant_metric_summary_computes_search_and_memory_deltas():
    tool = load_tool()
    labels = tool.SEARCH_BATCH_LABELS
    before = {
        ("rest_responses_total", tuple(sorted(labels.items()))): 10.0,
        ("rest_responses_duration_seconds_sum", tuple(sorted(labels.items()))): 2.0,
        ("rest_responses_max_duration_seconds", tuple(sorted(labels.items()))): 0.8,
        ("collection_hardware_metric_cpu", (("id", "collection_a"),)): 100.0,
        ("memory_resident_bytes", ()): 1000.0,
        ("memory_allocated_bytes", ()): 500.0,
        ("process_threads", ()): 12.0,
        ("process_open_fds", ()): 30.0,
        ("process_minor_page_faults_total", ()): 1000.0,
        ("process_major_page_faults_total", ()): 2.0,
    }
    after = {
        ("rest_responses_total", tuple(sorted(labels.items()))): 14.0,
        ("rest_responses_duration_seconds_sum", tuple(sorted(labels.items()))): 3.2,
        ("rest_responses_max_duration_seconds", tuple(sorted(labels.items()))): 1.1,
        ("collection_hardware_metric_cpu", (("id", "collection_a"),)): 175.0,
        ("memory_resident_bytes", ()): 1500.0,
        ("memory_allocated_bytes", ()): 650.0,
        ("process_threads", ()): 13.0,
        ("process_open_fds", ()): 31.0,
        ("process_minor_page_faults_total", ()): 1400.0,
        ("process_major_page_faults_total", ()): 3.0,
    }

    summary = tool.qdrant_metric_summary(before, after, collection="collection_a")

    assert summary["qdrant_search_batch_count_delta"] == pytest.approx(4)
    assert summary["qdrant_search_batch_duration_s_delta"] == pytest.approx(1.2)
    assert summary["qdrant_search_batch_avg_duration_ms_delta_window"] == pytest.approx(300)
    assert summary["qdrant_search_batch_max_duration_s_after"] == pytest.approx(1.1)
    assert summary["qdrant_collection_cpu_delta"] == pytest.approx(75)
    assert summary["qdrant_memory_resident_bytes_delta"] == pytest.approx(500)
    assert summary["qdrant_memory_allocated_bytes_delta"] == pytest.approx(150)
    assert summary["qdrant_process_threads_before"] == pytest.approx(12)
    assert summary["qdrant_process_threads_after"] == pytest.approx(13)
    assert summary["qdrant_process_open_fds_before"] == pytest.approx(30)
    assert summary["qdrant_process_open_fds_after"] == pytest.approx(31)
    assert summary["qdrant_process_minor_page_faults_delta"] == pytest.approx(400)
    assert summary["qdrant_process_major_page_faults_delta"] == pytest.approx(1)


def test_flatten_metric_snapshot_keeps_selected_metrics_and_labels():
    tool = load_tool()
    samples = {
        ("memory_resident_bytes", ()): 1024.0,
        ("collections_total", ()): 25.0,
        ("collection_hardware_metric_cpu", (("id", "collection_a"),)): 7.0,
    }

    rows = tool.flatten_metric_snapshot(samples, variant="compact_current", repeat=1, phase="before")

    by_metric = {(row["metric"], row["labels_json"]): row for row in rows}
    assert ("memory_resident_bytes", "{}") in by_metric
    assert ("collection_hardware_metric_cpu", '{"id": "collection_a"}') in by_metric
    assert all(row["variant"] == "compact_current" for row in rows)
    assert all(row["repeat"] == 1 for row in rows)
    assert all(row["phase"] == "before" for row in rows)
    assert not any(row["metric"] == "collections_total" for row in rows)
