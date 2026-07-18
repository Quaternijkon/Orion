from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def load_module():
    script_path = Path(__file__).resolve().parents[2] / "tools/method4_benchmark_matrix.py"
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


def test_collect_summary_extracts_distributed_latency_nprobe_and_image(tmp_path):
    module = load_module()
    result_dir = tmp_path / "run" / "20260717_120000"
    result_dir.mkdir(parents=True)
    (result_dir / "summary.json").write_text(
        json.dumps(
            {
                "routing_mode": "kmeans_simple_nprobe",
                "num_shards": 46,
                "image_digest": "sha256:abc",
                "placement_valid": True,
                "runtime_health_audit": {"valid": True},
                "cluster_preflight": {"peer_count": 4},
                "worker_shard_points": {
                    "22": {"shard_count": 15, "points_count": 500},
                    "11": {"shard_count": 16, "points_count": 510},
                },
                "best_tuning_row": {"upper_k": 24, "base_ef": 80, "factor": 0},
                "final_row": {
                    "recall_at_k": 0.951,
                    "qps": 220.0,
                    "nprobe": 24,
                    "avg_visited_shards": 24.0,
                    "avg_physical_peers_per_query": 3.0,
                    "latency_p50_ms": 40.0,
                    "latency_p95_ms": 55.0,
                    "latency_p99_ms": 61.0,
                    "estimated_ef_sum_per_query": 1920.0,
                },
            }
        ),
        encoding="utf-8",
    )
    row = module.collect_summary_row(
        module.BenchmarkRun(
            "simple", "simple", "kmeans_simple_nprobe", {}, tmp_path / "run", {}
        ),
        result_dir / "summary.json",
    )

    assert row["nprobe"] == 24
    assert row["latency_p95_ms"] == 55.0
    assert row["latency_p99_ms"] == 61.0
    assert row["latency_p50_ms_stdev"] == ""
    assert row["latency_p95_ms_stdev"] == ""
    assert row["latency_p99_ms_stdev"] == ""
    assert row["avg_physical_peers_per_query"] == 3.0
    assert row["cluster_peer_count"] == 4
    assert row["image_digest"] == "sha256:abc"
    assert row["placement_valid"] is True
    assert row["runtime_health_valid"] is True
    assert json.loads(row["worker_shard_counts"]) == {"11": 16, "22": 15}
    assert json.loads(row["worker_point_counts"]) == {"11": 510, "22": 500}


def test_collect_summary_uses_stability_latency_mean_and_stdev(tmp_path):
    module = load_module()
    result_dir = tmp_path / "run" / "20260718_120000"
    result_dir.mkdir(parents=True)
    (result_dir / "summary.json").write_text(
        json.dumps(
            {
                "routing_mode": "faithful_original_rest",
                "num_shards": 46,
                "final_row": {
                    "recall_at_k": 0.951,
                    "qps": 800.0,
                    "latency_p50_ms": 999.0,
                    "latency_p95_ms": 999.0,
                    "latency_p99_ms": 999.0,
                },
            }
        ),
        encoding="utf-8",
    )
    (result_dir / "stability_runs.csv").write_text(
        "run,latency_p50_ms,latency_p95_ms,latency_p99_ms\n"
        "0,100,200,300\n"
        "1,110,220,330\n"
        "2,120,240,360\n",
        encoding="utf-8",
    )

    row = module.collect_summary_row(
        module.BenchmarkRun(
            "orion", "orion", "orion", {}, tmp_path / "run", {}
        ),
        result_dir / "summary.json",
    )

    assert row["latency_p50_ms"] == pytest.approx(110.0)
    assert row["latency_p50_ms_stdev"] == pytest.approx(10.0)
    assert row["latency_p95_ms"] == pytest.approx(220.0)
    assert row["latency_p95_ms_stdev"] == pytest.approx(20.0)
    assert row["latency_p99_ms"] == pytest.approx(330.0)
    assert row["latency_p99_ms_stdev"] == pytest.approx(30.0)


@pytest.mark.parametrize(
    ("recall", "target", "expected_status", "expected_delta"),
    [
        (0.901, 0.9, "strict", 0.001),
        (0.897, 0.9, "strict", -0.003),
        (0.896, 0.9, "nearest", -0.004),
        (0.904, 0.9, "nearest", 0.004),
    ],
)
def test_confirmed_recall_status_uses_actual_eval_result(
    recall, target, expected_status, expected_delta
):
    module = load_module()

    status, delta = module.confirmed_recall_match_status(recall, target)

    assert status == expected_status
    assert delta == pytest.approx(expected_delta)


def test_collect_summary_marks_manual_stability_confirmation(tmp_path):
    module = load_module()
    result_dir = tmp_path / "run" / "20260717_000000"
    result_dir.mkdir(parents=True)
    (result_dir / "summary.json").write_text(
        json.dumps(
            {
                "routing_mode": "faithful_original_rest",
                "num_shards": 46,
                "final_row": {
                    "upper_k": 36,
                    "base_ef": 48,
                    "factor": 15,
                    "recall_at_k": 0.8998,
                    "qps": 1000.0,
                },
                "stability_summary": {
                    "recall_mean": 0.8998,
                    "recall_stdev": 0.0,
                    "qps_mean": 1000.0,
                    "qps_stdev": 10.0,
                },
            }
        ),
        encoding="utf-8",
    )
    run = module.BenchmarkRun(
        "orion",
        "orion",
        "orion",
        {"target_recall": 0.9, "stability_repeats": 3},
        tmp_path / "run",
        {},
    )

    row = module.collect_summary_row(run, result_dir / "summary.json")

    assert row["confirmed_recall_match_status"] == "strict"
    assert row["recall_delta_to_target"] == pytest.approx(-0.0002)


def test_same_recall_selection_uses_strict_window_then_nearest():
    module = load_module()
    points = [
        {"recall_at_k": 0.899, "qps": 300.0},
        {"recall_at_k": 0.901, "qps": 280.0},
        {"recall_at_k": 0.910, "qps": 400.0},
    ]

    strict = module.select_same_recall_point(points, 0.9)
    nearest = module.select_same_recall_point(points, 0.95)

    assert strict["recall_at_k"] == 0.899
    assert strict["recall_match_status"] == "strict"
    assert nearest["recall_at_k"] == 0.910
    assert nearest["recall_match_status"] == "nearest"


def test_pareto_frontier_removes_points_worse_in_both_recall_and_qps():
    module = load_module()
    points = [
        {"name": "dominated", "recall_at_k": 0.90, "qps": 100.0},
        {"name": "fast", "recall_at_k": 0.90, "qps": 120.0},
        {"name": "accurate", "recall_at_k": 0.95, "qps": 90.0},
    ]

    frontier = module.pareto_frontier(points)

    assert {row["name"] for row in frontier} == {"fast", "accurate"}


def test_confirmation_config_reuses_collection_and_sets_confirmation_budget(tmp_path):
    module = load_module()
    run = module.BenchmarkRun(
        run_id="orion",
        case_name="orion",
        preset="orion",
        args={
            "collection": "dist_run_orion_s46",
            "base_url": "http://10.10.1.1:6333",
            "num_shards": 31,
            "batch_size": 200,
        },
        output_dir=tmp_path / "orion",
        tags={"partition_family": "orion"},
    )
    selections = [
        {
            "run_id": "orion",
            "target_recall": 0.95,
            "recall_at_k": 0.951,
            "recall_match_status": "strict",
            "upper_k": 120,
            "base_ef": 80,
            "factor": 8,
        }
    ]

    module.write_confirmation_config(
        tmp_path,
        {"matrix_name": "initial", "client_cpuset": "8-19"},
        [run],
        selections,
    )
    config = json.loads((tmp_path / "confirmation_matrix.json").read_text())
    args = config["cases"][0]["args"]

    assert args["collection"] == "dist_run_orion_s46"
    assert args["reuse_existing"] is True
    assert args["recover_routing_from_collection"] is True
    assert args["eval_query_count"] == 3000
    assert args["warmup_query_count"] == 500
    assert args["stability_repeats"] == 3
    assert args["batch_size"] == 200
    assert args["upper_k_candidates"] == [120]


def test_delete_collections_only_uses_validated_run_scoped_templates(monkeypatch):
    module = load_module()
    matrix_run_id = "smoke-20260718"
    spec = {
        "defaults": {"base_url": "http://10.10.1.1:6333"},
        "cases": [
            {
                "name": "orion",
                "preset": "orion",
                "collection_template": "dist_{matrix_run_id}_smoke_orion",
            },
            {
                "name": "simple",
                "preset": "kmeans_simple_nprobe",
                "collection_template": "bench_{matrix_run_id}_smoke_simple",
            },
        ],
    }
    runs = module.expand_runs(
        spec,
        Path("results/matrix"),
        matrix_run_id=matrix_run_id,
    )

    targets = module.validated_collection_cleanup_targets(
        spec, runs, matrix_run_id
    )

    assert targets == [
        module.CollectionCleanupTarget(
            "http://10.10.1.1:6333", "dist_smoke-20260718_smoke_orion"
        ),
        module.CollectionCleanupTarget(
            "http://10.10.1.1:6333", "bench_smoke-20260718_smoke_simple"
        ),
    ]

    requests = []

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

    def fake_urlopen(request, timeout):
        requests.append((request.full_url, request.get_method(), timeout))
        return FakeResponse()

    monkeypatch.setattr(module.urllib.request, "urlopen", fake_urlopen)
    module.delete_run_collections(targets)

    assert requests == [
        (
            "http://10.10.1.1:6333/collections/dist_smoke-20260718_smoke_orion",
            "DELETE",
            120.0,
        ),
        (
            "http://10.10.1.1:6333/collections/bench_smoke-20260718_smoke_simple",
            "DELETE",
            120.0,
        ),
    ]


def test_delete_collections_rejects_fixed_reused_collection_before_delete(monkeypatch):
    module = load_module()
    matrix_run_id = "confirmation-20260718"
    spec = {
        "defaults": {"base_url": "http://10.10.1.1:6333"},
        "cases": [
            {
                "name": "orion",
                "preset": "orion",
                "collection_template": "dist_{matrix_run_id}_orion",
                "args": {
                    "collection": "dist_dist-20260717-initial_orion_s46",
                    "reuse_existing": True,
                },
            }
        ],
    }
    runs = module.expand_runs(
        spec,
        Path("results/matrix"),
        matrix_run_id=matrix_run_id,
    )
    delete_calls = []
    monkeypatch.setattr(
        module.urllib.request,
        "urlopen",
        lambda *args, **kwargs: delete_calls.append((args, kwargs)),
    )

    with pytest.raises(ValueError, match="collection is explicitly supplied"):
        targets = module.validated_collection_cleanup_targets(
            spec, runs, matrix_run_id
        )
        module.delete_run_collections(targets)

    assert delete_calls == []


@pytest.mark.parametrize(
    ("collection_template", "match"),
    [
        ("dist_smoke_orion", "must explicitly contain"),
        ("prod_{matrix_run_id}_orion", "experimental prefix"),
        (
            "dist_smoke_{matrix_run_id}_orion",
            "run token must immediately follow",
        ),
    ],
)
def test_delete_collections_rejects_unscoped_or_unsafe_templates(
    collection_template, match
):
    module = load_module()
    matrix_run_id = "smoke-20260718"
    spec = {
        "defaults": {"base_url": "http://10.10.1.1:6333"},
        "cases": [
            {
                "name": "orion",
                "preset": "orion",
                "collection_template": collection_template,
            }
        ],
    }
    runs = module.expand_runs(
        spec,
        Path("results/matrix"),
        matrix_run_id=matrix_run_id,
    )

    with pytest.raises(ValueError, match=match):
        module.validated_collection_cleanup_targets(spec, runs, matrix_run_id)


def test_delete_collections_rejects_invalid_matrix_run_token():
    module = load_module()
    spec = {
        "defaults": {"base_url": "http://10.10.1.1:6333"},
        "cases": [
            {
                "name": "orion",
                "preset": "orion",
                "collection_template": "dist_{matrix_run_id}_orion",
            }
        ],
    }
    runs = module.expand_runs(
        spec,
        Path("results/matrix"),
        matrix_run_id="../unsafe",
    )

    with pytest.raises(ValueError, match="validated matrix run id"):
        module.validated_collection_cleanup_targets(spec, runs, "../unsafe")
