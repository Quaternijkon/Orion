from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def load_module():
    script_path = Path("/home/taig/dry/qdrant/tools/method4_benchmark_matrix.py")
    spec = importlib.util.spec_from_file_location("method4_benchmark_matrix", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_expand_runs_applies_orion_naive_and_kmeans_presets():
    module = load_module()
    spec = {
        "matrix_name": "smoke",
        "defaults": {
            "base_url": "http://localhost:6833",
            "hdf5_path": "/data/glove.hdf5",
            "eval_query_count": 3000,
            "tuning_query_count": 500,
            "upper_k_candidates": [120, 160],
            "base_ef_candidates": [60, 80],
            "factor_candidates": [6, 8],
        },
        "cases": [
            {
                "name": "orion",
                "preset": "orion",
                "collection_template": "bench_{case}_{target_recall}",
                "matrix": {"target_recall": [0.95]},
            },
            {
                "name": "naive",
                "preset": "naive",
                "collection_template": "bench_{case}_{target_recall}",
                "matrix": {"target_recall": [0.95]},
            },
            {
                "name": "kmeans",
                "preset": "kmeans",
                "collection_template": "bench_{case}_{target_recall}",
                "matrix": {"target_recall": [0.95]},
            },
        ],
    }

    runs = module.expand_runs(spec, Path("results/matrix"))

    assert [run.case_name for run in runs] == ["orion", "naive", "kmeans"]
    assert runs[0].args["routing_mode"] == "faithful_original_rest"
    assert runs[0].args["routed_execution_mode"] == "compact_multi_ep"
    assert runs[1].args["routing_mode"] == "naive_hash_all_shards"
    assert runs[2].args["routing_mode"] == "faithful_original_rest"
    assert runs[2].args["routed_execution_mode"] == "compact_multi_ep"
    assert runs[2].args["routed_planning_mode"] == "materialized"
    assert runs[2].args["topology_iters"] == 0
    assert runs[2].args["disable_fission"] is True
    assert runs[2].tags["partition_family"] == "method4_kmeans_ablation"
    assert all(run.args["collection"].startswith("bench_") for run in runs)


def test_kmeans_independent_preset_keeps_cpp_baseline_available():
    module = load_module()
    spec = {
        "matrix_name": "smoke",
        "defaults": {
            "base_url": "http://localhost:6833",
            "hdf5_path": "/data/glove.hdf5",
        },
        "cases": [
            {
                "name": "independent",
                "preset": "kmeans_independent",
            },
        ],
    }

    [run] = module.expand_runs(spec, Path("results/matrix"))

    assert run.args["routing_mode"] == "cpp_kmeans_baseline"
    assert run.args["routed_execution_mode"] == "compact_multi_ep"
    assert "topology_iters" not in run.args
    assert "disable_fission" not in run.args
    assert run.tags["partition_family"] == "kmeans_independent"


def test_kmeans_simple_nprobe_preset_uses_plain_centroid_probe_mode():
    module = load_module()
    spec = {
        "matrix_name": "smoke",
        "defaults": {
            "base_url": "http://localhost:6833",
            "hdf5_path": "/data/glove.hdf5",
        },
        "cases": [
            {
                "name": "simple",
                "preset": "kmeans_simple_nprobe",
            },
        ],
    }

    [run] = module.expand_runs(spec, Path("results/matrix"))

    assert run.args["routing_mode"] == "kmeans_simple_nprobe"
    assert run.args["search_dispatch_mode"] == "coordinator"
    assert "topology_iters" not in run.args
    assert "disable_fission" not in run.args
    assert "routed_execution_mode" not in run.args
    assert "routed_planning_mode" not in run.args
    assert run.tags["partition_family"] == "kmeans_simple_nprobe"


def test_method4_kmeans_ablation_alias_matches_default_kmeans_preset():
    module = load_module()
    spec = {
        "matrix_name": "smoke",
        "defaults": {
            "base_url": "http://localhost:6833",
            "hdf5_path": "/data/glove.hdf5",
        },
        "cases": [
            {
                "name": "default_kmeans",
                "preset": "kmeans",
            },
            {
                "name": "explicit_ablation",
                "preset": "method4_kmeans_ablation",
            },
        ],
    }

    default_run, alias_run = module.expand_runs(spec, Path("results/matrix"))

    assert default_run.args == alias_run.args
    assert default_run.tags == alias_run.tags


def test_render_command_handles_lists_flags_and_snake_case_args():
    module = load_module()
    run = module.BenchmarkRun(
        run_id="orion__target_recall-0p95",
        case_name="orion",
        preset="orion",
        args={
            "base_url": "http://localhost:6833",
            "upper_k_candidates": [120, 160],
            "reuse_existing": True,
            "disable_fission": False,
            "train_limit": None,
        },
        output_dir=Path("results/matrix/orion"),
        tags={},
    )

    command = module.render_command(run, python_executable="python3", harness="tools/qdrant_two_level_routing_experiment.py")

    assert command == [
        "python3",
        "tools/qdrant_two_level_routing_experiment.py",
        "--base-url",
        "http://localhost:6833",
        "--upper-k-candidates",
        "120",
        "160",
        "--reuse-existing",
        "--output-dir",
        "results/matrix/orion",
    ]


def test_collect_summary_extracts_stability_and_final_rows(tmp_path):
    module = load_module()
    result_dir = tmp_path / "run" / "20260617_120000"
    result_dir.mkdir(parents=True)
    (result_dir / "summary.json").write_text(
        json.dumps(
            {
                "routing_mode": "faithful_original_rest",
                "num_shards": 46,
                "best_tuning_row": {
                    "upper_k": 160,
                    "base_ef": 80,
                    "factor": 8,
                },
                "final_row": {
                    "recall": 0.955,
                    "qps": 396.7,
                    "avg_visited_shards": 23.2,
                    "avg_assigned_ef": 141.0,
                },
                "stability_summary": {
                    "runs": 3,
                    "qps_mean": 390.0,
                    "qps_stdev": 2.5,
                    "recall_mean": 0.954,
                    "recall_stdev": 0.001,
                },
                "original_routing": {
                    "expansion_ratio": 1.18,
                },
            }
        ),
        encoding="utf-8",
    )

    row = module.collect_summary_row(
        run=module.BenchmarkRun(
            run_id="orion",
            case_name="orion",
            preset="orion",
            args={"target_recall": 0.955},
            output_dir=tmp_path / "run",
            tags={"partition_family": "orion"},
        ),
        summary_path=result_dir / "summary.json",
    )

    assert row["run_id"] == "orion"
    assert row["partition_family"] == "orion"
    assert row["routing_mode"] == "faithful_original_rest"
    assert row["target_recall"] == 0.955
    assert row["recall"] == 0.954
    assert row["qps"] == 390.0
    assert row["qps_stdev"] == 2.5
    assert row["avg_visited_shards"] == 23.2
    assert row["index_expansion_ratio"] == 1.18


def test_collect_summary_extracts_simple_kmeans_expansion_ratio(tmp_path):
    module = load_module()
    result_dir = tmp_path / "run" / "20260629_120000"
    result_dir.mkdir(parents=True)
    (result_dir / "summary.json").write_text(
        json.dumps(
            {
                "routing_mode": "kmeans_simple_nprobe",
                "num_shards": 46,
                "best_tuning_row": {
                    "upper_k": 32,
                    "base_ef": 160,
                    "factor": 0,
                },
                "final_row": {
                    "recall_at_k": 0.954,
                    "qps": 210.0,
                    "avg_visited_shards": 32.0,
                    "avg_assigned_ef_per_visited_shard": 160.0,
                },
                "kmeans_simple_nprobe": {
                    "expansion_ratio": 1.42,
                },
            }
        ),
        encoding="utf-8",
    )

    row = module.collect_summary_row(
        run=module.BenchmarkRun(
            run_id="simple",
            case_name="simple",
            preset="kmeans_simple_nprobe",
            args={"target_recall": 0.95},
            output_dir=tmp_path / "run",
            tags={"partition_family": "kmeans_simple_nprobe"},
        ),
        summary_path=result_dir / "summary.json",
    )

    assert row["routing_mode"] == "kmeans_simple_nprobe"
    assert row["recall"] == 0.954
    assert row["qps"] == 210.0
    assert row["index_expansion_ratio"] == 1.42
