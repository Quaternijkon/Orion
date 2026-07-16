import csv
import importlib.util
from pathlib import Path

import pytest


def load_module():
    module_path = Path("tools/method4_claim_a_online_latency_summary.py")
    spec = importlib.util.spec_from_file_location("method4_claim_a_online_latency_summary", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    out = path / "claim_d_high_recall_latency_summary.csv"
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summary_row(method: str, label: str, qps: float, p95: float, *, batch_size: int = 200) -> dict[str, object]:
    return {
        "method": method,
        "config_label": label,
        "query_count": 3000,
        "top_k": 10,
        "recall_at_10": 0.952,
        "qps": qps,
        "wall_s": 3000 / qps,
        "avg_visited_shards": 21.5 if method == "Method4" else 46.0,
        "avg_assigned_ef_sum": 3138.0 if method == "Method4" else 3496.0,
        "search_batch_calls": 15,
        "batch_count": 15,
        "batch_latency_mean_ms": p95 * 0.8,
        "batch_latency_p50_ms": p95 * 0.75,
        "batch_latency_p95_ms": p95,
        "batch_latency_p99_ms": p95 * 1.05,
        "batch_latency_max_ms": p95 * 1.10,
        "collection": "method_collection" if method == "Method4" else "naive_collection",
        "repeat": "",
        "batch_size": batch_size,
    }


def test_aggregate_group_combines_repeats_and_flags_high_variance(tmp_path):
    module = load_module()
    run1 = tmp_path / "run1"
    run2 = tmp_path / "run2"
    write_summary(
        run1,
        [
            summary_row("Naive", "naive_ef76", 240.0, 1000.0),
            summary_row("Naive", "naive_ef76", 100.0, 3000.0),
        ],
    )
    write_summary(
        run2,
        [
            summary_row("Naive", "naive_ef76", 250.0, 950.0),
            summary_row("Naive", "naive_ef76", 140.0, 2400.0),
        ],
    )

    row = module.aggregate_group(
        root=tmp_path,
        group=module.RunGroup(
            comparison_group="kmeans_vs_naive",
            partition_label="naive_reference_for_kmeans",
            method_label="naive_all_shards",
            method="Naive",
            config_label="naive_ef76",
            collection="naive_collection",
            routing_source_collection="",
            batch_size=200,
            source_dirs=("run1", "run2"),
            base_note="",
        ),
    )

    assert row["repeats"] == 4
    assert row["qps_mean"] == pytest.approx(182.5)
    assert row["qps_median"] == pytest.approx(190.0)
    assert row["batch_latency_p95_ms_mean"] == pytest.approx(1837.5)
    assert row["qps_cv"] > 0.30
    assert "high_qps_variance" in row["notes"]
    assert row["source_dirs"] == "run1;run2"


def test_build_delta_rows_reports_mean_and_median_changes():
    module = load_module()
    method = {
        "comparison_group": "full_fission_vs_naive",
        "method_label": "full_fission_existing",
        "batch_size": 100,
        "recall_at_10_mean": 0.953,
        "qps_mean": 420.0,
        "qps_median": 421.0,
        "batch_latency_p95_ms_mean": 250.0,
        "batch_latency_p95_ms_median": 251.0,
        "batch_latency_p99_ms_mean": 260.0,
        "batch_latency_p99_ms_median": 261.0,
        "avg_visited_shards_mean": 22.7,
        "avg_assigned_ef_sum_mean": 3214.0,
        "notes": "",
    }
    baseline = {
        "comparison_group": "full_fission_vs_naive",
        "method_label": "naive_all_shards",
        "batch_size": 100,
        "recall_at_10_mean": 0.952,
        "qps_mean": 280.0,
        "qps_median": 279.0,
        "batch_latency_p95_ms_mean": 400.0,
        "batch_latency_p95_ms_median": 399.0,
        "batch_latency_p99_ms_mean": 430.0,
        "batch_latency_p99_ms_median": 431.0,
        "avg_visited_shards_mean": 46.0,
        "avg_assigned_ef_sum_mean": 3496.0,
        "notes": "",
    }

    delta = module.build_delta_rows([method, baseline], [("full_vs_naive", "full_fission_existing", "naive_all_shards")])[0]

    assert delta["comparison_label"] == "full_vs_naive"
    assert delta["batch_size"] == 100
    assert delta["recall_delta_mean"] == pytest.approx(0.001)
    assert delta["qps_delta_pct_mean"] == pytest.approx(50.0)
    assert delta["qps_delta_pct_median"] == pytest.approx((421.0 - 279.0) / 279.0 * 100.0)
    assert delta["visited_shards_delta_pct_mean"] == pytest.approx((22.7 - 46.0) / 46.0 * 100.0)
    assert delta["p95_latency_delta_pct_mean"] == pytest.approx(-37.5)


def test_default_groups_include_topology_no_fission_selected_cell():
    module = load_module()

    groups = {(g.comparison_group, g.method_label): g for g in module.DEFAULT_GROUPS}

    topo = groups[("topology_no_fission_vs_full_fission", "topology_no_fission")]
    baseline = groups[("topology_no_fission_vs_full_fission", "full_fission_existing")]

    assert topo.collection == "bench095_rr_topology_no_fission_s46_20260705"
    assert topo.config_label == "topology_no_fission_u160_b80_f10"
    assert topo.batch_size == 200
    assert topo.source_dirs == ("topology_no_fission_selected/analysis_20260705_012559",)
    assert "topology_iters=50" in topo.base_note
    assert baseline.config_label == "full_fission_u160_b80_f8"

    assert (
        "topology_no_fission_vs_full_fission",
        "topology_no_fission",
        "full_fission_existing",
    ) in module.DEFAULT_DELTAS


def test_run_group_can_read_from_explicit_source_root(tmp_path):
    module = load_module()
    primary_root = tmp_path / "primary"
    supplemental_root = tmp_path / "supplemental"
    write_summary(
        supplemental_root / "random_balanced_46_ef400build" / "analysis_20260709_092159",
        [
            {
                **summary_row("Method4", "random_balanced_u160_b80_f8", 320.0, 640.0),
                "collection": "random_balanced_46",
                "upper_k": 160,
                "base_ef": 80,
                "factor": 8,
            }
        ],
    )

    row = module.aggregate_group(
        root=primary_root,
        group=module.RunGroup(
            comparison_group="claim_a_partition_family_current_rebuilds_vs_naive",
            partition_label="random_balanced_46",
            method_label="random_balanced_46",
            method="Method4",
            config_label="random_balanced_u160_b80_f8",
            collection="random_balanced_46",
            routing_source_collection="random_balanced_46",
            batch_size=200,
            source_dirs=("random_balanced_46_ef400build/analysis_20260709_092159",),
            source_root=str(supplemental_root),
        ),
    )

    assert row["qps_mean"] == pytest.approx(320.0)
    assert row["source_dirs"] == str(
        supplemental_root / "random_balanced_46_ef400build" / "analysis_20260709_092159"
    )


def test_default_groups_include_current_harness_claim_a_family_rebuilds():
    module = load_module()

    groups = {(g.comparison_group, g.method_label): g for g in module.DEFAULT_GROUPS}

    random_group = groups[("random_balanced_46_current_rebuild_vs_naive", "random_balanced_46")]
    random_naive = groups[("random_balanced_46_current_rebuild_vs_naive", "naive_all_shards")]
    topology_group = groups[("kmeans_topology_46_current_rebuild_vs_naive", "kmeans_topology_46")]
    topology_naive = groups[("kmeans_topology_46_current_rebuild_vs_naive", "naive_all_shards")]
    recal_group = groups[
        ("kmeans_topology_load_recalibrated_46_current_rebuild_vs_naive", "kmeans_topology_load_recalibrated_46")
    ]
    recal_naive = groups[
        ("kmeans_topology_load_recalibrated_46_current_rebuild_vs_naive", "naive_all_shards")
    ]

    assert random_group.source_root == "results/method4_claim_a_partition_online_latency_20260709"
    assert random_group.source_dirs == ("random_balanced_46_ef400build/analysis_20260709_092159",)
    assert random_group.config_label == "random_balanced_u160_b80_f8"
    assert topology_group.config_label == "kmeans_topology_u160_b80_f10"
    assert recal_group.config_label == "kmeans_topology_load_recalibrated_u160_b80_f10"
    assert random_naive.source_dirs == random_group.source_dirs
    assert topology_naive.source_dirs == topology_group.source_dirs
    assert recal_naive.source_dirs == recal_group.source_dirs

    assert (
        "random_balanced_46_vs_naive_current_rebuild",
        "random_balanced_46",
        "naive_all_shards",
    ) in module.DEFAULT_DELTAS
    assert (
        "kmeans_topology_46_vs_naive_current_rebuild",
        "kmeans_topology_46",
        "naive_all_shards",
    ) in module.DEFAULT_DELTAS
    assert (
        "kmeans_topology_load_recalibrated_46_vs_naive_current_rebuild",
        "kmeans_topology_load_recalibrated_46",
        "naive_all_shards",
    ) in module.DEFAULT_DELTAS


def test_build_topology_no_fission_sensitivity_rows_compares_candidates(tmp_path):
    module = load_module()
    run_dir = tmp_path / "topology_no_fission_selected" / "analysis_20260705_012559"
    rows = [
        {
            **summary_row("Method4", "topology_no_fission_u160_b80_f10", 430.0, 480.0),
            "recall_at_10": 0.9521,
            "avg_visited_shards": 18.94,
            "avg_assigned_ef_sum": 3228.0,
            "batch_latency_p99_ms": 484.0,
            "collection": "bench095_rr_topology_no_fission_s46_20260705",
            "upper_k": 160,
            "base_ef": 80,
            "factor": 10,
        },
        {
            **summary_row("Method4", "topology_no_fission_u160_b80_f10", 432.0, 478.0),
            "recall_at_10": 0.9521,
            "avg_visited_shards": 18.94,
            "avg_assigned_ef_sum": 3228.0,
            "batch_latency_p99_ms": 482.0,
            "collection": "bench095_rr_topology_no_fission_s46_20260705",
            "upper_k": 160,
            "base_ef": 80,
            "factor": 10,
        },
        {
            **summary_row("Method4", "topology_no_fission_u160_b120_f8", 415.0, 500.0),
            "recall_at_10": 0.9515,
            "avg_visited_shards": 18.94,
            "avg_assigned_ef_sum": 3643.0,
            "batch_latency_p99_ms": 506.0,
            "collection": "bench095_rr_topology_no_fission_s46_20260705",
            "upper_k": 160,
            "base_ef": 120,
            "factor": 8,
        },
        {
            **summary_row("Method4", "topology_no_fission_u160_b120_f8", 417.0, 498.0),
            "recall_at_10": 0.9515,
            "avg_visited_shards": 18.94,
            "avg_assigned_ef_sum": 3643.0,
            "batch_latency_p99_ms": 504.0,
            "collection": "bench095_rr_topology_no_fission_s46_20260705",
            "upper_k": 160,
            "base_ef": 120,
            "factor": 8,
        },
    ]
    write_summary(run_dir, rows)

    sensitivity_rows = module.build_topology_no_fission_sensitivity_rows(tmp_path)

    assert [row["config_label"] for row in sensitivity_rows] == [
        "topology_no_fission_u160_b80_f10",
        "topology_no_fission_u160_b120_f8",
    ]
    selected, heavier = sensitivity_rows
    assert selected["selection_role"] == "selected_main_cell"
    assert selected["qps_delta_pct_vs_selected"] == pytest.approx(0.0)
    assert heavier["selection_role"] == "slower_sensitivity_candidate"
    assert heavier["base_ef"] == 120
    assert heavier["factor"] == 8
    assert heavier["recall_at_10_mean"] == pytest.approx(0.9515)
    assert heavier["qps_delta_pct_vs_selected"] == pytest.approx((416.0 - 431.0) / 431.0 * 100.0)
    assert heavier["ef_sum_delta_pct_vs_selected"] == pytest.approx((3643.0 - 3228.0) / 3228.0 * 100.0)
    assert heavier["p95_latency_delta_pct_vs_selected"] == pytest.approx((499.0 - 479.0) / 479.0 * 100.0)
    assert heavier["p99_latency_delta_pct_vs_selected"] == pytest.approx((505.0 - 483.0) / 483.0 * 100.0)
