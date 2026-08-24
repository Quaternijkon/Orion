from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "experiments"
    / "c1"
    / "scripts"
    / "c1_protocol.py"
)
BENCHMARK_MODULE_PATH = MODULE_PATH.with_name("c1_benchmark.py")
E2_MODULE_PATH = MODULE_PATH.with_name("c1_e2_fanout.py")
E1_MODULE_PATH = MODULE_PATH.with_name("c1_e1_scaleout.py")
E1_AGGREGATE_MODULE_PATH = MODULE_PATH.with_name("c1_e1_aggregate.py")
E4_MODULE_PATH = MODULE_PATH.with_name("c1_e4_decomposition.py")
FINAL_MODULE_PATH = MODULE_PATH.with_name("c1_finalize.py")
AUDIT_MODULE_PATH = MODULE_PATH.with_name("c1_completion_audit.py")
DOWNSTREAM_MODULE_PATH = MODULE_PATH.with_name("c1_deterministic_downstream.py")


def load_module():
    spec = importlib.util.spec_from_file_location("c1_protocol", MODULE_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_benchmark_module():
    scripts_dir = str(BENCHMARK_MODULE_PATH.parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location("c1_benchmark", BENCHMARK_MODULE_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_e2_module():
    scripts_dir = str(E2_MODULE_PATH.parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location("c1_e2_fanout", E2_MODULE_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_e1_module():
    scripts_dir = str(E1_MODULE_PATH.parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location("c1_e1_scaleout", E1_MODULE_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_e1_aggregate_module():
    scripts_dir = str(E1_AGGREGATE_MODULE_PATH.parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location(
        "c1_e1_aggregate", E1_AGGREGATE_MODULE_PATH
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_e4_module():
    scripts_dir = str(E4_MODULE_PATH.parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location("c1_e4_decomposition", E4_MODULE_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_final_module():
    scripts_dir = str(FINAL_MODULE_PATH.parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location("c1_finalize", FINAL_MODULE_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_audit_module():
    scripts_dir = str(AUDIT_MODULE_PATH.parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location("c1_completion_audit", AUDIT_MODULE_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_downstream_module():
    scripts_dir = str(DOWNSTREAM_MODULE_PATH.parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location(
        "c1_deterministic_downstream", DOWNSTREAM_MODULE_PATH
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_random_assignments_are_seeded_uniform_draws():
    module = load_module()
    first = module.deterministic_random_assignments(10_000, 8, seed=17)
    second = module.deterministic_random_assignments(10_000, 8, seed=17)
    different = module.deterministic_random_assignments(10_000, 8, seed=18)

    assert np.array_equal(first, second)
    assert not np.array_equal(first, different)
    assert int(first.min()) == 0
    assert int(first.max()) == 7
    counts = np.bincount(first, minlength=8)
    assert counts.max() - counts.min() < 200


def test_fit_kmeans_separates_euclidean_clusters_deterministically():
    module = load_module()
    rng = np.random.default_rng(4)
    left = rng.normal(loc=(-5.0, 0.0), scale=0.05, size=(100, 2)).astype(np.float32)
    right = rng.normal(loc=(5.0, 0.0), scale=0.05, size=(100, 2)).astype(np.float32)
    rows = np.vstack([left, right])

    first_assignments, first_centroids, history = module.fit_kmeans(
        rows, 2, module.DATASETS["sift1m"], seed=9
    )
    second_assignments, second_centroids, _ = module.fit_kmeans(
        rows, 2, module.DATASETS["sift1m"], seed=9
    )

    assert np.array_equal(first_assignments, second_assignments)
    assert np.allclose(first_centroids, second_centroids)
    assert len(set(first_assignments[:100])) == 1
    assert len(set(first_assignments[100:])) == 1
    assert first_assignments[0] != first_assignments[-1]
    assert history[-1]["empty_clusters"] == 0


def test_cosine_centroid_assignment_uses_angular_similarity():
    module = load_module()
    rows = np.asarray([[10.0, 0.0], [0.0, 3.0], [-2.0, 0.0]], dtype=np.float32)
    centroids = np.asarray([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=np.float32)
    assigned = module.centroid_assignments(
        rows, centroids, module.DATASETS["glove-200-angular"]
    )
    assert assigned.tolist() == [0, 1, 2]


def test_oracle_minimum_fanout_counts_ground_truth_contributions():
    module = load_module()
    assignments = np.asarray([0, 0, 0, 0, 1, 1, 1, 2, 2, 3, 3, 3], dtype=np.uint8)
    neighbors = np.asarray(
        [
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            [0, 4, 7, 9, 1, 5, 8, 10, 2, 11],
        ],
        dtype=np.int32,
    )
    oracle = module.oracle_minimum_fanout(assignments, neighbors, 4)
    assert oracle.tolist() == [3, 4]


def test_e2_distribution_summary_preserves_full_discrete_distribution():
    module = load_e2_module()
    summary = module.distribution_summary([1, 1, 2, 4])
    assert summary["mean"] == 2.0
    assert summary["median"] == 1.5
    assert summary["p95"] == pytest.approx(3.7)
    assert summary["distribution_counts"] == {"1": 2, "2": 1, "4": 1}


def test_e2_event_aggregation_reconstructs_actual_fanout_and_recall(tmp_path):
    module = load_e2_module()
    path = tmp_path / "e3.csv"
    fieldnames = [
        "query_id",
        "dataset",
        "partition_method",
        "logical_shards",
        "selected_fanout",
        "ef_search",
        "shard_id",
        "route_rank",
        "ground_truth_points_in_shard",
        "ground_truth_hits",
    ]
    rows = [
        [10, "sift1m", "kmeans", 4, 2, 32, 3, 2, 3, 3],
        [11, "sift1m", "kmeans", 4, 2, 32, 0, 1, 5, 4],
        [10, "sift1m", "kmeans", 4, 2, 32, 1, 1, 7, 6],
        [11, "sift1m", "kmeans", 4, 2, 32, 2, 2, 4, 4],
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(fieldnames)
        writer.writerows(rows)
    actual, selected = module.load_e3_actual_events(
        path,
        dataset="sift1m",
        method="kmeans",
        logical_shards=4,
        query_start=10,
        query_stop=12,
    )
    assert selected == {"selected_fanout": 2, "ef_search": 32}
    assert actual[10]["selected_shard_ids"] == [1, 3]
    assert actual[10]["achieved_recall"] == 0.9
    assert actual[11]["ground_truth_shard_coverage"] == 0.9


def test_e2_tuning_source_stems_support_dataset_specific_stage_names(tmp_path):
    module = load_e2_module()
    glove = tmp_path / "stage1-glove-200-angular-kmeans-m32-tuning-pinned.json"
    glove.write_text("{}\n", encoding="utf-8")
    selected = module.tuning_path(
        tmp_path,
        "glove-200-angular",
        "kmeans",
        32,
        large_source_stem="stage1-{dataset}",
    )
    assert selected == glove.resolve()


def test_e2_source_stem_rejects_paths():
    module = load_e2_module()
    with pytest.raises(ValueError, match="one filename component"):
        module.source_stem("../stage1-{dataset}", "sift1m")


def test_e2_sensitivity_checks_reject_reversed_kmeans_trend():
    module = load_e2_module()
    summaries = []
    for method in module.METHODS:
        for shards in module.LOGICAL_SHARD_COUNTS:
            sensitivity = {}
            for threshold in module.SENSITIVITY_TARGETS:
                selected = shards if method == "random" else min(3, shards)
                sensitivity[f"{threshold:.2f}"] = {
                    "globally_selected_fanout": selected,
                    "oracle_fanout": {"mean": float(min(shards, 3))},
                }
            summaries.append(
                {
                    "partition_method": method,
                    "logical_shard_count": shards,
                    "sensitivity": sensitivity,
                }
            )
    checks = module.qualitative_sensitivity_checks(summaries)
    assert checks["all_thresholds_passed"] is True
    summaries[-1]["sensitivity"]["0.90"]["globally_selected_fanout"] = 1
    with pytest.raises(ValueError, match="qualitatively reverse"):
        module.qualitative_sensitivity_checks(summaries)


def test_e2_sensitivity_checks_retain_substantial_non_monotonic_kmeans_dip():
    module = load_e2_module()
    summaries = []
    for method in module.METHODS:
        for shards in module.LOGICAL_SHARD_COUNTS:
            sensitivity = {}
            for threshold in module.SENSITIVITY_TARGETS:
                selected = shards if method == "random" else min(3, shards)
                if method == "kmeans" and threshold == 0.90:
                    if shards == 16:
                        selected = 9
                    elif shards == 32:
                        selected = 8
                sensitivity[f"{threshold:.2f}"] = {
                    "globally_selected_fanout": selected,
                    "oracle_fanout": {"mean": float(min(shards, 3))},
                }
            summaries.append(
                {
                    "partition_method": method,
                    "logical_shard_count": shards,
                    "sensitivity": sensitivity,
                }
            )

    checks = module.qualitative_sensitivity_checks(summaries)

    assert checks["all_thresholds_passed"] is True
    assert checks["non_monotonic_kmeans_targets"] == ["0.90"]
    assert checks["0.90"]["kmeans_actual_series"][-2:] == [9, 8]


def test_e1_saturation_selection_uses_highest_stable_qps_before_latency_knee():
    module = load_e1_module()
    rows = [
        {"concurrency": 1, "completed_qps": 100.0, "p99_latency_us": 1000.0},
        {"concurrency": 2, "completed_qps": 180.0, "p99_latency_us": 1200.0},
        {"concurrency": 4, "completed_qps": 188.0, "p99_latency_us": 1800.0},
        {"concurrency": 8, "completed_qps": 189.0, "p99_latency_us": 2500.0},
    ]
    selected = module.select_saturation_concurrency(rows)
    assert selected["knee_observed"] is True
    assert selected["stop_reason"] == "two_consecutive_subthreshold_qps_improvements"
    assert selected["selected_concurrency"] == 8


def test_e1_saturation_selection_excludes_latency_explosion():
    module = load_e1_module()
    rows = [
        {"concurrency": 1, "completed_qps": 100.0, "p99_latency_us": 1000.0},
        {"concurrency": 2, "completed_qps": 170.0, "p99_latency_us": 1500.0},
        {"concurrency": 4, "completed_qps": 180.0, "p99_latency_us": 6000.0},
    ]
    selected = module.select_saturation_concurrency(rows)
    assert selected["stop_reason"] == "p99_exceeded_multiplier"
    assert selected["selected_concurrency"] == 2


def test_e1_formal_concurrency_override_preserves_automatic_selection():
    module = load_e1_module()
    rows = [
        {"concurrency": 8, "completed_qps": 200.0},
        {"concurrency": 16, "completed_qps": 300.0},
        {"concurrency": 32, "completed_qps": 320.0},
    ]
    automatic = {
        "selected_concurrency": 32,
        "selected_sweep_qps": 320.0,
        "knee_observed": True,
    }
    selected = module.apply_formal_concurrency_override(
        automatic,
        rows,
        override=16,
        reason="formal repetitions exceeded the client CPU validity gate",
    )
    assert selected["selected_concurrency"] == 16
    assert selected["automatic_selected_concurrency"] == 32
    assert selected["automatic_selected_sweep_qps"] == 320.0
    assert selected["formal_concurrency_override_reason"].startswith("formal repetitions")


def test_e1_repetition_statistics_use_sample_variation():
    module = load_e1_module()
    values = [100.0, 102.0, 98.0]
    assert module.sample_coefficient_of_variation(values) == pytest.approx(0.02)
    interval = module.confidence_interval_95(values)
    assert interval["n"] == 3
    assert interval["mean"] == 100.0
    assert interval["lower"] < 100.0 < interval["upper"]


def test_e1_aggregate_uses_dataset_specific_fixed_configurations():
    module = load_e1_aggregate_module()
    assert module.EXPECTED_CONFIGURATIONS["sift1m"][("kmeans", 4)] == (1, 48)
    assert module.EXPECTED_CONFIGURATIONS["glove-200-angular"] == {
        ("random", 1): (1, 384),
        ("random", 2): (2, 256),
        ("random", 4): (4, 128),
        ("kmeans", 1): (1, 384),
        ("kmeans", 2): (2, 256),
        ("kmeans", 4): (3, 256),
    }


def test_e4_source_stems_reject_paths():
    module = load_e4_module()
    assert module.source_component("stage2-e3", "E3") == "stage2-e3"
    with pytest.raises(ValueError, match="one filename component"):
        module.source_component("../stage2-e3", "E3")


def test_e4_loads_physical_e1_json_summary(tmp_path):
    module = load_e4_module()
    configurations = []
    for method in module.METHODS:
        for workers in module.PHYSICAL_MACHINE_COUNTS:
            configurations.append(
                {
                    "dataset": "sift1m",
                    "partition_method": method,
                    "physical_machine_count": workers,
                    "experiment_id": (
                        f"stage6-e1-corrected-sift1m-{method}-m{workers}"
                    ),
                    "status": "VALID_E1",
                    "bottleneck_flags": "",
                    "normalized_qps_to_m1": 1.0,
                }
            )
    path = tmp_path / "physical-summary.json"
    path.write_text(
        json.dumps(
            {
                "record_type": "e1_physical_scaleout_analysis",
                "dataset": "sift1m",
                "status": "VALID_E1",
                "configurations": configurations,
            }
        ),
        encoding="utf-8",
    )

    rows = module.load_physical_e1(
        path, "sift1m", source_stem="stage6-e1-corrected"
    )

    assert set(rows) == {
        (method, workers)
        for method in module.METHODS
        for workers in module.PHYSICAL_MACHINE_COUNTS
    }


def test_e4_external_projection_gate_withholds_cross_dataset_release(tmp_path):
    module = load_e4_module()
    record = tmp_path / "sift-e4-record.json"
    record.write_text(
        json.dumps(
            {
                "record_type": "e4_aggregate_work_decomposition",
                "experiment_id": "sift-e4",
                "dataset": "sift1m",
                "m_gt_4_projection_status": (
                    "WITHHELD_PHYSICAL_SANITY_CHECK_FAILED"
                ),
            }
        ),
        encoding="utf-8",
    )
    gates, passed = module.load_external_projection_gates(
        [record], current_dataset="glove-200-angular"
    )
    assert passed is False
    assert gates[0]["dataset"] == "sift1m"
    assert gates[0]["experiment_id"] == "sift-e4"
    assert gates[0]["source_record_sha256"] == module.sha256_path(record)


def test_e4_external_projection_gate_rejects_current_dataset(tmp_path):
    module = load_e4_module()
    record = tmp_path / "glove-e4-record.json"
    record.write_text(
        json.dumps(
            {
                "record_type": "e4_aggregate_work_decomposition",
                "experiment_id": "glove-e4",
                "dataset": "glove-200-angular",
                "m_gt_4_projection_status": "AVAILABLE",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="must use another dataset"):
        module.load_external_projection_gates(
            [record], current_dataset="glove-200-angular"
        )


def test_c1_final_sensitivity_candidate_matches_fanout_and_ef():
    module = load_final_module()
    tuning = {
        "candidates": [
            {"fanout": 2, "ef_search": 32, "recall_at_10": 0.89},
            {"fanout": 3, "ef_search": 32, "recall_at_10": 0.91},
        ]
    }
    selected = module.select_sensitivity_candidate(tuning, fanout=3, ef_search=32)
    assert selected["recall_at_10"] == 0.91
    with pytest.raises(ValueError, match="expected one"):
        module.select_sensitivity_candidate(tuning, fanout=4, ef_search=32)


def test_c1_final_requires_completed_section31_proof(tmp_path):
    module = load_final_module()
    proof = {
        "graph_build_seed": 20260821,
        "image_id": "sha256:test",
        "max_indexing_threads": 1,
        "max_optimization_threads": 1,
        "identical_build_count": 2,
        "graph_content_sha256": "graph-sha256",
        "source_commit": "source-commit",
        "deterministic_construction_verified": True,
        "authoritative_e2_e4_rerun_complete": False,
    }
    proof_path = tmp_path / "section31.json"
    proof_path.write_text(json.dumps(proof), encoding="utf-8")
    with pytest.raises(ValueError, match="not complete"):
        module.validate_section31_proof(proof_path)

    proof["authoritative_e2_e4_rerun_complete"] = True
    proof_path.write_text(json.dumps(proof), encoding="utf-8")
    assert module.validate_section31_proof(proof_path)["graph_build_seed"] == 20260821


def test_c1_final_defaults_to_stage12_deterministic_sources():
    module = load_final_module()
    args = module.parse_args([])
    assert args.record_stem == "stage12-c1-deterministic-final"
    assert args.e2_source_stem == "stage12-e2-deterministic"
    assert args.e3_source_stem == "stage12-e3-deterministic"
    with pytest.raises(ValueError, match="one filename component"):
        module.source_component("../stage12", "E3")


def test_c1_final_derives_mechanism_boundaries_from_stage12_rows():
    module = load_final_module()
    fanout = []
    local = []
    for dataset in module.DATASETS:
        for method in module.METHODS:
            for shards in module.LOGICAL_SHARDS:
                actual_fanout = shards
                if dataset == "sift1m" and method == "kmeans":
                    actual_fanout = {1: 1, 2: 1, 4: 1, 8: 2, 16: 3, 32: 3}[shards]
                fanout.append(
                    {
                        "dataset": dataset,
                        "partition_method": method,
                        "logical_shard_count": str(shards),
                        "actual_fanout_mean": str(actual_fanout),
                    }
                )
                local.append(
                    {
                        "dataset": dataset,
                        "partition_method": method,
                        "logical_shard_count": shards,
                        "mean_distance_computations_per_searched_shard": 1000
                        / shards**0.5,
                        "mean_nodes_visited_per_searched_shard": 100 / shards**0.5,
                    }
                )
    e4_records = {
        dataset: {"physical_attribution_status": "INSUFFICIENT"}
        for dataset in module.DATASETS
    }
    checks = module.evaluate_final_mechanisms(fanout, local, e4_records)
    assert checks["slow_local_work_supported"] is True
    assert checks["generalized_high_fanout_contradicted"] is True
    assert checks["physical_attribution_insufficient"] is True

    next(
        row
        for row in fanout
        if row["dataset"] == "sift1m"
        and row["partition_method"] == "kmeans"
        and row["logical_shard_count"] == "32"
    )["actual_fanout_mean"] = "8"
    with pytest.raises(ValueError, match="small-fan-out contradiction"):
        module.evaluate_final_mechanisms(fanout, local, e4_records)


def test_c1_audit_accepts_complete_e3_with_contradictory_local_work():
    module = load_audit_module()
    rows = []
    for dataset in module.DATASETS:
        for method in module.METHODS:
            for shards in module.LOGICAL_SHARDS:
                distance = 1000 / shards**0.5
                nodes = 100 / shards**0.5
                if dataset == "glove-200-angular" and method == "random" and shards == 2:
                    distance = 350
                rows.append(
                    {
                        "dataset": dataset,
                        "partition_method": method,
                        "logical_shard_count": shards,
                        "status": "VALID_E3",
                        "mean_distance_computations_per_searched_shard": distance,
                        "mean_nodes_visited_per_searched_shard": nodes,
                    }
                )

    evidence = module.evaluate_local_work_evidence(rows)
    assert evidence["configuration_count"] == 24
    assert evidence["comparison_count"] == 40
    assert evidence["at_or_below_ideal_count"] == 1
    assert evidence["slow_local_work_supported"] is False
    assert evidence["c1_c_status"] == "CONTRADICTED"


def test_c1_audit_accepts_complete_e5_with_reported_trend_reversal():
    module = load_audit_module()
    rows = []
    for dataset in module.DATASETS:
        for method in module.METHODS:
            for shards in (4, 16):
                for target in (0.89, 0.90, 0.91):
                    local = 100.0 if shards == 4 else 30.0
                    if (
                        dataset == "glove-200-angular"
                        and method == "random"
                        and shards == 16
                        and target == 0.90
                    ):
                        local = 20.0
                    fanout = shards
                    rows.append(
                        {
                            "dataset": dataset,
                            "partition_method": method,
                            "logical_shard_count": shards,
                            "sensitivity_target_recall": target,
                            "selected_fanout": fanout,
                            "achieved_tuning_recall": target + 0.01,
                            "mean_local_distance_computations": local,
                            "mean_aggregate_distance_computations": local * fanout,
                        }
                    )

    evidence = module.evaluate_sensitivity_evidence(rows)

    assert evidence["configuration_count"] == 24
    assert evidence["all_trends_preserved"] is False
    assert evidence["contradicted_checks"] == [
        "glove-200-angular/random/0.90"
    ]
    assert evidence["status"] == "CONTRADICTED_QUALITATIVE_TREND_REVERSAL"


def test_c1_audit_replays_protocol_tuning_selection():
    module = load_audit_module()
    random_candidates = [
        {
            "fanout": 4,
            "ef_search": 16,
            "recall_at_10": 0.89,
            "mean_aggregate_distance_computations": 100.0,
        },
        {
            "fanout": 4,
            "ef_search": 24,
            "recall_at_10": 0.91,
            "mean_aggregate_distance_computations": 120.0,
        },
        {
            "fanout": 4,
            "ef_search": 32,
            "recall_at_10": 0.95,
            "mean_aggregate_distance_computations": 140.0,
        },
    ]
    assert module.select_protocol_tuning_candidate(
        random_candidates, "random"
    ) == random_candidates[1]

    kmeans_candidates = [
        {
            "fanout": 1,
            "ef_search": 128,
            "recall_at_10": 0.91,
            "mean_aggregate_distance_computations": 101.5,
        },
        {
            "fanout": 2,
            "ef_search": 32,
            "recall_at_10": 0.91,
            "mean_aggregate_distance_computations": 100.0,
        },
    ]
    assert module.select_protocol_tuning_candidate(
        kmeans_candidates, "kmeans"
    ) == kmeans_candidates[0]


def test_c1_audit_enforces_deterministic_round_robin_placement():
    module = load_audit_module()
    assert module.expected_round_robin_mapping(2) == {
        "0": "10.10.1.1",
        "1": "10.10.1.2",
    }
    mapping = module.expected_round_robin_mapping(8)
    assert mapping == {
        str(shard): f"10.10.1.{shard % 4 + 1}" for shard in range(8)
    }


def test_c1_downstream_validates_c1_c_against_mechanism_evidence(tmp_path):
    module = load_downstream_module()
    proof_path = tmp_path / "section31.json"
    proof_path.write_text("{}\n", encoding="utf-8")
    record = {
        "c1_status": "INSUFFICIENT",
        "subclaim_status": {
            "c1_b_high_logical_shard_fanout": "CONTRADICTED",
            "c1_c_slow_local_work_decrease": "CONTRADICTED",
        },
        "observed_mechanism_status": {
            "slow_local_work_decrease": "CONTRADICTED",
        },
        "mechanism_checks": {"slow_local_work_supported": False},
        "section31_index_construction_status": (
            "VERIFIED_DETERMINISTIC_SINGLE_BUILD_AUTHORITATIVE_RERUN"
        ),
        "graph_build_seed": 20260821,
        "section31_proof_sha256": module.sha256_path(proof_path),
        "authoritative_source_stems": {
            "e2": module.E2_STEM,
            "e3": module.E3_STEM,
        },
    }
    record_path = tmp_path / "final.json"
    record_path.write_text(json.dumps(record), encoding="utf-8")
    assert module.valid_final_record(record_path, 20260821, proof_path)

    record["subclaim_status"]["c1_c_slow_local_work_decrease"] = "SUPPORTED"
    record_path.write_text(json.dumps(record), encoding="utf-8")
    assert not module.valid_final_record(record_path, 20260821, proof_path)


def test_e1_single_process_client_saturation_is_invalid():
    module = load_e1_module()
    summary = {
        "achieved_recall": 0.91,
        "aggregator_process_count": 1,
        "aggregator_max_single_process_cpu_utilization_pct": 99.0,
        "aggregator_cpu_utilization_pct_of_reserved_cores": 8.25,
        "persistent_aggregator_queue_growth": False,
        "routing_share_of_routing_plus_worker_cpu_pct": 10.0,
        "resource_utilization": {"max_network_utilization_pct": 1.0},
    }
    status, flags = module.phase_validity(summary)
    assert status == "INVALID_E1"
    assert flags == ["AGGREGATOR_SINGLE_PROCESS_LIMITED"]


def test_e1_multiprocess_client_uses_total_reserved_core_gate():
    module = load_e1_module()
    summary = {
        "achieved_recall": 0.91,
        "aggregator_process_count": 13,
        "aggregator_max_single_process_cpu_utilization_pct": 99.0,
        "aggregator_cpu_utilization_pct_of_reserved_cores": 70.0,
        "persistent_aggregator_queue_growth": False,
        "routing_share_of_routing_plus_worker_cpu_pct": 10.0,
        "resource_utilization": {"max_network_utilization_pct": 1.0},
    }
    status, flags = module.phase_validity(summary)
    assert status == "VALID_E1"
    assert flags == []


def test_recall_and_candidate_selection_follow_protocol_rules():
    module = load_module()
    truth = np.asarray([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]], dtype=np.int32)
    recall = module.recall_per_query([[0, 1, 2, 3, 4, 5, 6, 7, 8, 99]], truth)
    assert recall.tolist() == [0.9]

    random_choice = module.select_random_tuning_candidate(
        [
            {"ef_search": 32, "recall_at_10": 0.89},
            {"ef_search": 48, "recall_at_10": 0.91},
            {"ef_search": 64, "recall_at_10": 0.95},
        ]
    )
    assert random_choice["ef_search"] == 48

    kmeans_choice = module.select_kmeans_tuning_candidate(
        [
            {
                "fanout": 4,
                "ef_search": 64,
                "recall_at_10": 0.91,
                "mean_aggregate_distance_computations": 100.0,
            },
            {
                "fanout": 3,
                "ef_search": 80,
                "recall_at_10": 0.90,
                "mean_aggregate_distance_computations": 101.5,
            },
            {
                "fanout": 2,
                "ef_search": 96,
                "recall_at_10": 0.89,
                "mean_aggregate_distance_computations": 90.0,
            },
        ]
    )
    assert kmeans_choice["fanout"] == 3


def test_dense_usage_converts_exactly_to_distance_computations():
    module = load_module()
    assert module.distance_computations_from_usage(200 * 4 * 123, 200) == 123
    with pytest.raises(ValueError, match="not divisible"):
        module.distance_computations_from_usage(801, 200)


def test_dataset_audit_enforces_disjoint_tuning_and_measurement_split(tmp_path):
    module = load_module()
    path = tmp_path / "tiny-angular.hdf5"
    with h5py.File(path, "w") as handle:
        handle.attrs["distance"] = "angular"
        handle.create_dataset("train", data=np.zeros((20, 3), dtype=np.float32))
        handle.create_dataset("test", data=np.zeros((1_010, 3), dtype=np.float32))
        handle.create_dataset(
            "neighbors",
            data=np.zeros((1_010, 10), dtype=np.int32),
        )

    audit = module.dataset_audit(path, module.DATASETS["glove-200-angular"])
    assert audit["tuning_query_range"] == [0, 1_000]
    assert audit["measurement_query_range"] == [1_000, 1_010]
    assert audit["measurement_query_count"] == 10
    assert audit["query_sets_disjoint"] is True


def test_c1_round_robin_placement_preserves_physical_and_logical_rules():
    module = load_benchmark_module()
    assert module.HNSW_MAX_INDEXING_THREADS == 1
    assert module.MAX_OPTIMIZATION_THREADS == 1
    nodes = [
        module.Node(
            role=f"node_{index}",
            ssh_host=f"node-{index}",
            private_ip=f"10.0.0.{index + 1}",
            cpuset="0-19",
            peer_id=index + 100,
            base_url=f"http://10.0.0.{index + 1}:6333",
        )
        for index in range(4)
    ]

    physical = module.round_robin_placement(4, nodes)
    assert [physical[index].peer_id for index in range(4)] == [100, 101, 102, 103]

    logical = module.round_robin_placement(8, nodes)
    assert [logical[index].peer_id for index in range(8)] == [
        100,
        101,
        102,
        103,
        100,
        101,
        102,
        103,
    ]


def test_c1_ranking_and_merge_follow_dataset_distance_semantics():
    module = load_benchmark_module()
    euclidean = module.DATASETS["sift1m"]
    cosine = module.DATASETS["glove-200-angular"]
    centroids = np.asarray([[0.0, 0.0], [3.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    assert module.rank_kmeans_shards(
        np.asarray([1.1, 0.0], dtype=np.float32), centroids, euclidean
    ).tolist() == [2, 0, 1]

    cosine_centroids = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    assert module.rank_kmeans_shards(
        np.asarray([4.0, 1.0], dtype=np.float32), cosine_centroids, cosine
    ).tolist() == [0, 1]

    shards = [
        module.ShardSearch(0, 1.0, 1, 1, 1, 1, 1, [(10, 0.4), (11, 0.2)]),
        module.ShardSearch(1, 1.0, 1, 1, 1, 1, 1, [(12, 0.3), (10, 0.1)]),
    ]
    assert module.merge_shard_results(shards, euclidean, top_k=3) == [10, 11, 12]
    assert module.merge_shard_results(shards, cosine, top_k=3) == [10, 12, 11]


def test_c1_summary_aggregates_per_shard_work():
    module = load_benchmark_module()
    row = {
        "achieved_recall": 0.9,
        "queried_shards": 2,
        "end_to_end_latency_us": 100.0,
        "distance_computations_per_shard": [10, 20],
        "nodes_visited_per_shard": [3, 4],
        "worker_cpu_time_us_per_shard": [5, 7],
        "routing_latency_us": 2.0,
        "routing_cpu_time_us": 1.0,
        "response_bytes_per_shard": [50, 60],
        "oracle_minimum_fanout": 1,
    }
    summary = module.summarize_rows([row, row], elapsed_seconds=0.5)
    assert summary["achieved_recall"] == 0.9
    assert summary["mean_distance_computations_per_query"] == 30
    assert summary["mean_nodes_visited_per_query"] == 7
    assert summary["mean_worker_cpu_time_us_per_query"] == 12
    assert summary["network_bytes_per_query"] == 110
    assert summary["completed_qps"] == 4


def test_c1_e3_route_plan_broadcasts_random_and_uses_centroid_prefix():
    module = load_benchmark_module()
    queries = np.asarray([[0.1, 0.0], [9.9, 0.0]], dtype=np.float32)
    random_plan = module.build_e3_route_plan(
        queries,
        module.DATASETS["sift1m"],
        "random",
        4,
        4,
        np.empty((0, 2), dtype=np.float32),
    )
    assert random_plan == {
        0: [(0, 1), (1, 1)],
        1: [(0, 2), (1, 2)],
        2: [(0, 3), (1, 3)],
        3: [(0, 4), (1, 4)],
    }

    centroids = np.asarray(
        [[0.0, 0.0], [3.0, 0.0], [7.0, 0.0], [10.0, 0.0]], dtype=np.float32
    )
    kmeans_plan = module.build_e3_route_plan(
        queries,
        module.DATASETS["sift1m"],
        "kmeans",
        4,
        2,
        centroids,
    )
    assert kmeans_plan == {
        0: [(0, 1)],
        1: [(0, 2)],
        2: [(1, 2)],
        3: [(1, 1)],
    }


def test_c1_e3_summary_weights_searched_shard_events():
    module = load_benchmark_module()
    rows = [
        {
            "shard_id": 0,
            "shard_point_count": 60,
            "distance_computations": 10,
            "nodes_visited": 3,
            "worker_cpu_time_us": 5,
            "worker_cpu_wall_time_us": 6,
            "local_search_latency_us": 7.0,
            "response_bytes": 50,
        },
        {
            "shard_id": 0,
            "shard_point_count": 60,
            "distance_computations": 20,
            "nodes_visited": 5,
            "worker_cpu_time_us": 7,
            "worker_cpu_wall_time_us": 8,
            "local_search_latency_us": 9.0,
            "response_bytes": 60,
        },
        {
            "shard_id": 1,
            "shard_point_count": 40,
            "distance_computations": 30,
            "nodes_visited": 7,
            "worker_cpu_time_us": 9,
            "worker_cpu_wall_time_us": 10,
            "local_search_latency_us": 11.0,
            "response_bytes": 70,
        },
    ]
    summary = module.summarize_e3_search_rows(rows, query_count=2, logical_shards=2)
    assert summary["searched_shard_events"] == 3
    assert summary["mean_shards_per_query"] == 1.5
    assert summary["mean_distance_computations_per_searched_shard"] == 20
    assert summary["mean_nodes_visited_per_searched_shard"] == 5
    assert summary["shards"][0]["selection_frequency"] == 1.0
    assert summary["shards"][1]["selection_frequency"] == 0.5
    assert summary["shards"][0]["mean_distance_computations"] == 15
    assert summary["shards"][1]["mean_distance_computations"] == 30


def test_c1_e3_measurement_recall_sums_routed_shard_contributions():
    module = load_benchmark_module()
    rows = [
        {"query_id": 100, "recall_contribution": 0.4},
        {"query_id": 100, "recall_contribution": 0.5},
        {"query_id": 101, "recall_contribution": 1.0},
    ]
    assert module.e3_measurement_recall(
        rows, query_start=100, query_count=2
    ) == pytest.approx(0.95)


def test_c1_e3_measurement_recall_rejects_missing_queries():
    module = load_benchmark_module()
    with pytest.raises(ValueError, match="missing 1 measurement queries"):
        module.e3_measurement_recall(
            [{"query_id": 100, "recall_contribution": 0.9}],
            query_start=100,
            query_count=2,
        )


def test_c1_kmeans_prefix_reuse_matches_every_ranked_prefix():
    module = load_benchmark_module()
    spec = module.DATASETS["sift1m"]
    ground_truth = np.asarray([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    ranked = [
        module.ShardSearch(
            2,
            11.0,
            100,
            10,
            7,
            6,
            50,
            [(10, 0.1), (20, 0.2), (999, 0.3)],
        ),
        module.ShardSearch(
            0,
            12.0,
            200,
            20,
            8,
            7,
            60,
            [(30, 0.1), (40, 0.2), (998, 0.3)],
        ),
        module.ShardSearch(
            1,
            13.0,
            300,
            30,
            9,
            8,
            70,
            [(50, 0.1), (60, 0.2), (997, 0.3)],
        ),
    ]

    prefixes = module.prefix_metrics_from_ranked_searches(ranked, ground_truth, spec)
    assert [row["fanout"] for row in prefixes] == [1, 2, 3]
    assert [row["recall_at_10"] for row in prefixes] == [0.2, 0.4, 0.6]
    assert [row["aggregate_distance_computations"] for row in prefixes] == [
        100,
        300,
        600,
    ]
    assert [row["aggregate_nodes_visited"] for row in prefixes] == [10, 30, 60]
    assert [row["aggregate_worker_cpu_time_us"] for row in prefixes] == [7, 15, 24]
    for fanout, row in enumerate(prefixes, start=1):
        expected = module.merge_shard_results(ranked[:fanout], spec)
        expected_recall = module.recall_per_query(
            [expected], ground_truth[None, :]
        )[0]
        assert row["recall_at_10"] == expected_recall


def test_c1_e3_resource_helpers_measure_llc_and_graph_files(tmp_path):
    module = load_benchmark_module()
    llc = tmp_path / "llc-size"
    llc.write_text("16M\n", encoding="utf-8")
    assert module.local_llc_size_bytes(llc) == 16 * 1024 * 1024

    index = (
        tmp_path
        / "collections"
        / "example"
        / "7"
        / "segments"
        / "segment-a"
        / "vector_index"
    )
    index.mkdir(parents=True)
    (index / "graph.bin").write_bytes(b"a" * 11)
    (index / "links_compressed.bin").write_bytes(b"b" * 13)
    assert module.local_graph_index_size_bytes(tmp_path, "example", 7) == 24
