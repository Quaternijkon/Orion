import csv
import importlib.util
from pathlib import Path

import pytest


def load_module():
    module_path = Path("tools/method4_claim_c_latency_summary.py")
    spec = importlib.util.spec_from_file_location("method4_claim_c_latency_summary", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_summary(path: Path) -> None:
    rows = [
        {
            "method": "Method4",
            "config_label": "fixed_u80_ef280",
            "query_count": 3000,
            "top_k": 10,
            "recall_at_10": 0.950,
            "qps": 400.0,
            "wall_s": 7.5,
            "avg_visited_shards": 16.0,
            "avg_assigned_ef_sum": 4480.0,
            "search_batch_calls": 15,
            "batch_count": 15,
            "batch_latency_mean_ms": 500.0,
            "batch_latency_p50_ms": 490.0,
            "batch_latency_p95_ms": 520.0,
            "batch_latency_p99_ms": 530.0,
            "batch_latency_max_ms": 540.0,
            "collection": "orion",
            "repeat": 1,
            "batch_size": 200,
            "upper_k": 80,
            "base_ef": 280,
            "factor": 0,
        },
        {
            "method": "Method4",
            "config_label": "fixed_u80_ef280",
            "query_count": 3000,
            "top_k": 10,
            "recall_at_10": 0.950,
            "qps": 410.0,
            "wall_s": 7.3,
            "avg_visited_shards": 16.0,
            "avg_assigned_ef_sum": 4480.0,
            "search_batch_calls": 15,
            "batch_count": 15,
            "batch_latency_mean_ms": 490.0,
            "batch_latency_p50_ms": 480.0,
            "batch_latency_p95_ms": 510.0,
            "batch_latency_p99_ms": 520.0,
            "batch_latency_max_ms": 530.0,
            "collection": "orion",
            "repeat": 2,
            "batch_size": 200,
            "upper_k": 80,
            "base_ef": 280,
            "factor": 0,
        },
        {
            "method": "Method4",
            "config_label": "dynamic_u80_b80_f20",
            "query_count": 3000,
            "top_k": 10,
            "recall_at_10": 0.951,
            "qps": 520.0,
            "wall_s": 5.8,
            "avg_visited_shards": 16.0,
            "avg_assigned_ef_sum": 3020.0,
            "search_batch_calls": 15,
            "batch_count": 15,
            "batch_latency_mean_ms": 385.0,
            "batch_latency_p50_ms": 380.0,
            "batch_latency_p95_ms": 410.0,
            "batch_latency_p99_ms": 420.0,
            "batch_latency_max_ms": 430.0,
            "collection": "orion",
            "repeat": 1,
            "batch_size": 200,
            "upper_k": 80,
            "base_ef": 80,
            "factor": 20,
        },
        {
            "method": "Method4",
            "config_label": "dynamic_u80_b80_f20",
            "query_count": 3000,
            "top_k": 10,
            "recall_at_10": 0.951,
            "qps": 530.0,
            "wall_s": 5.7,
            "avg_visited_shards": 16.0,
            "avg_assigned_ef_sum": 3020.0,
            "search_batch_calls": 15,
            "batch_count": 15,
            "batch_latency_mean_ms": 375.0,
            "batch_latency_p50_ms": 370.0,
            "batch_latency_p95_ms": 400.0,
            "batch_latency_p99_ms": 410.0,
            "batch_latency_max_ms": 420.0,
            "collection": "orion",
            "repeat": 2,
            "batch_size": 200,
            "upper_k": 80,
            "base_ef": 80,
            "factor": 20,
        },
    ]
    path.mkdir(parents=True, exist_ok=True)
    with (path / "claim_d_high_recall_latency_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_build_latency_tables_groups_fixed_and_dynamic(tmp_path):
    module = load_module()
    write_summary(tmp_path)

    matrix, deltas = module.build_latency_tables(tmp_path, comparison_label="latency_rerun")

    assert len(matrix) == 2
    fixed = next(row for row in matrix if row["method"] == "Fixed EF")
    dynamic = next(row for row in matrix if row["method"] == "Dynamic EF")
    assert fixed["qps_mean"] == pytest.approx(405.0)
    assert dynamic["qps_mean"] == pytest.approx(525.0)
    assert fixed["params"] == "upper_k=80; fixed_ef=280"
    assert dynamic["params"] == "upper_k=80; base_ef=80; factor=20"
    assert fixed["repeats"] == 2

    assert len(deltas) == 1
    delta = deltas[0]
    assert delta["comparison"] == "latency_rerun"
    assert delta["recall_delta_dynamic_minus_fixed"] == pytest.approx(0.001)
    assert delta["qps_delta_pct"] == pytest.approx((525.0 - 405.0) / 405.0 * 100.0)
    assert delta["ef_sum_delta_pct"] == pytest.approx((3020.0 - 4480.0) / 4480.0 * 100.0)
    assert delta["p95_latency_delta_pct"] == pytest.approx((405.0 - 515.0) / 515.0 * 100.0)
