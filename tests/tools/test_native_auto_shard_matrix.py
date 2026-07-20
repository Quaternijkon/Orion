from __future__ import annotations

import csv
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def load_module():
    path = REPO_ROOT / "tools/native_auto_shard_matrix.py"
    spec = importlib.util.spec_from_file_location("native_auto_shard_matrix", path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def base_config() -> dict:
    return {
        "shared": {
            "base_url": "http://10.10.1.1:6333",
            "hdf5_path": "/data/glove.hdf5",
            "topology": "tools/distributed/cloudlab_orion_4node.json",
            "deployment_manifest": "/runs/manifest.json",
            "warmup_query_count": 10,
            "eval_query_count": 100,
            "top_k": 10,
            "batch_size": 20,
            "stability_repeats": 3,
            "api": "search",
            "vector_distance": "cosine",
            "vector_name": "",
            "cargo_runner": "tools/cargo_in_docker.sh",
            "cargo_target_dir": "/external/cargo-target",
        },
        "cases": [
            {
                "name": "hash40",
                "method": "hash_all",
                "collection": "native_hash",
                "hnsw_ef": 40,
            },
            {
                "name": "orion90",
                "method": "orion",
                "collection": "native_orion",
                "artifact": "/artifacts/orion.json",
                "orion_route_trace": True,
            },
            {
                "name": "simple90",
                "method": "simple_kmeans",
                "collection": "native_simple",
                "artifact": "/artifacts/simple.json",
            },
        ],
    }


def write_config(tmp_path: Path, value: dict | None = None) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "matrix.json"
    path.write_text(json.dumps(value or base_config()), encoding="utf-8")
    return path


def shared_manifest(method: str, image_id: str = "sha256:same") -> dict:
    return {
        "method": method,
        "api": "search",
        "dataset": {
            "sha256": "d" * 64,
            "train_shape": [1000, 200],
            "test_shape": [100, 200],
            "neighbors_shape": [100, 10],
        },
        "deployment": {
            "image": {"id": image_id, "tag": "native:test"},
            "repository": {"commit": "commit-1"},
        },
        "repository": {"commit": "benchmark-commit", "dirty": False},
        "process_affinity": list(range(8, 20)),
        "topology": {
            "controller": {"private_ip": "10.10.1.1"},
            "workers": [
                {"private_ip": "10.10.1.2"},
                {"private_ip": "10.10.1.3"},
                {"private_ip": "10.10.1.4"},
            ],
        },
        "cluster_preflight": {
            "peers": {
                "101": "http://10.10.1.1:6335",
                "202": "http://10.10.1.2:6335",
                "303": "http://10.10.1.3:6335",
                "404": "http://10.10.1.4:6335",
            }
        },
        "parameters": {
            "vector_distance": "cosine",
            "vector_name": "",
            "eval_query_count": 100,
            "top_k": 10,
            "batch_size": 20,
        },
    }


def write_case_result(
    matrix_dir: Path,
    case: dict,
    recall: float,
    qps: float,
    *,
    visited,
    ef_sum,
    image_id: str = "sha256:same",
) -> None:
    case_dir = matrix_dir / "cases" / case["name"]
    case_dir.mkdir(parents=True)
    manifest = shared_manifest(case["method"], image_id=image_id)
    manifest["collection"] = case["collection"]
    if case.get("orion_route_trace") is True:
        manifest["orion_route_trace"] = {
            "status": "verified",
            "source": "exact_offline_production_router_trace",
        }
    (case_dir / "run_manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    fields = [
        "method",
        "recall_at_k",
        "qps",
        "latency_p50_ms",
        "latency_p95_ms",
        "latency_p99_ms",
        "visited_shards",
        "visited_shards_source",
        "ef_sum_per_query",
        "ef_sum_source",
    ]
    with (case_dir / "final_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerow(
            {
                "method": case["method"],
                "recall_at_k": recall,
                "qps": qps,
                "latency_p50_ms": 2.0,
                "latency_p95_ms": 3.0,
                "latency_p99_ms": 4.0,
                "visited_shards": visited,
                "visited_shards_source": (
                    "unknown_without_server_trace"
                    if visited is None
                    else (
                        "exact_offline_production_router_trace"
                        if case.get("orion_route_trace") is True
                        else "derived"
                    )
                ),
                "ef_sum_per_query": ef_sum,
                "ef_sum_source": (
                    "unknown_without_server_trace"
                    if ef_sum is None
                    else (
                        "exact_offline_production_router_trace"
                        if case.get("orion_route_trace") is True
                        else "derived"
                    )
                ),
            }
        )


def test_config_validation_and_taskset_command(tmp_path):
    module = load_module()
    config = module.load_config(write_config(tmp_path))
    command = module.benchmark_command(
        config["shared"],
        config["cases"][0],
        tmp_path / "case-output",
        "8-19",
    )

    assert command[:3] == ["taskset", "-c", "8-19"]
    assert "native_auto_shard_benchmark.py" in command[4]
    assert command[command.index("--method") + 1] == "hash_all"
    assert command[command.index("--hnsw-ef") + 1] == "40"
    assert command[command.index("--output-dir") + 1] == str(tmp_path / "case-output")
    assert command[command.index("--cargo-runner") + 1] == "tools/cargo_in_docker.sh"
    assert command[command.index("--cargo-target-dir") + 1] == "/external/cargo-target"

    orion_command = module.benchmark_command(
        config["shared"],
        config["cases"][1],
        tmp_path / "orion-output",
        None,
    )
    assert "--orion-route-trace" in orion_command

    invalid = base_config()
    invalid["cases"] = invalid["cases"][:-1]
    with pytest.raises(ValueError, match="all three methods"):
        module.load_config(write_config(tmp_path / "invalid", invalid))

    invalid_trace = base_config()
    invalid_trace["cases"][0]["orion_route_trace"] = True
    with pytest.raises(ValueError, match="must not provide orion_route_trace"):
        module.load_config(write_config(tmp_path / "invalid-trace", invalid_trace))


def test_matrix_output_must_be_new_and_outside_repository(tmp_path):
    module = load_module()
    with pytest.raises(ValueError, match="outside the repository"):
        module.matrix_directory(REPO_ROOT / "results", "native-run", must_exist=False)

    matrix_dir = module.matrix_directory(tmp_path, "native-run", must_exist=False)
    assert matrix_dir.is_dir()
    with pytest.raises(FileExistsError):
        module.matrix_directory(tmp_path, "native-run", must_exist=False)
    assert module.matrix_directory(tmp_path, "native-run", must_exist=True) == matrix_dir


def test_execute_cases_preserves_raw_case_directories_and_process_logs(
    monkeypatch, tmp_path
):
    module = load_module()
    config = module.load_config(write_config(tmp_path))
    matrix_dir = tmp_path / "matrix"
    matrix_dir.mkdir()
    commands = []

    def fake_run(command, **_kwargs):
        commands.append(command)
        output = Path(command[command.index("--output-dir") + 1])
        output.mkdir()
        return subprocess.CompletedProcess(command, 0, "raw stdout", "raw stderr")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    records = module.execute_cases(matrix_dir, config, "8-19")

    assert len(records) == 3
    assert len(commands) == 3
    assert all(command[:3] == ["taskset", "-c", "8-19"] for command in commands)
    assert (matrix_dir / "cases/hash40").is_dir()
    assert (matrix_dir / "logs/hash40.stdout.log").read_text() == "raw stdout"
    assert (matrix_dir / "logs/hash40.stderr.log").read_text() == "raw stderr"


def test_collect_rejects_configured_orion_trace_without_exact_proof(tmp_path):
    module = load_module()
    config = module.load_config(write_config(tmp_path))
    matrix_dir = tmp_path / "matrix"
    matrix_dir.mkdir()
    case = next(case for case in config["cases"] if case["method"] == "orion")
    write_case_result(
        matrix_dir,
        case,
        0.9,
        100.0,
        visited=1.5,
        ef_sum=54,
    )
    manifest_path = matrix_dir / "cases" / case["name"] / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("orion_route_trace")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RuntimeError, match="requires a verified Orion route trace"):
        module.load_case_result(matrix_dir, case)


def test_collect_aggregates_tables_and_preserves_null_orion_costs(monkeypatch, tmp_path):
    module = load_module()
    value = base_config()
    value["same_recall_targets"] = [0.90, 0.95]
    value["same_recall_window"] = 0.003
    value["cases"] = [
        {"name": "hash90", "method": "hash_all", "collection": "h", "hnsw_ef": 40},
        {"name": "hash95", "method": "hash_all", "collection": "h", "hnsw_ef": 76},
        {"name": "orion90", "method": "orion", "collection": "o90", "artifact": "/o90"},
        {"name": "orion95", "method": "orion", "collection": "o95", "artifact": "/o95"},
        {"name": "simple90", "method": "simple_kmeans", "collection": "s90", "artifact": "/s90"},
        {"name": "simple95", "method": "simple_kmeans", "collection": "s95", "artifact": "/s95"},
    ]
    config = module.load_config(write_config(tmp_path, value))
    matrix_dir = tmp_path / "matrix"
    matrix_dir.mkdir()
    observations = [
        ("hash90", 0.901, 300.0, 3, 120),
        ("hash95", 0.951, 210.0, 3, 228),
        ("orion90", 0.902, 500.0, None, None),
        ("orion95", 0.949, 360.0, None, None),
        ("simple90", 0.900, 420.0, 2, 96),
        ("simple95", 0.952, 250.0, 3, 240),
    ]
    cases = {case["name"]: case for case in config["cases"]}
    for name, recall, qps, visited, ef_sum in observations:
        write_case_result(
            matrix_dir,
            cases[name],
            recall,
            qps,
            visited=visited,
            ef_sum=ef_sum,
        )

    plotted_points = []

    def fake_plots(_matrix_dir, points):
        plotted_points.extend(points)
        return ["recall_qps.png"]

    monkeypatch.setattr(module, "write_plots", fake_plots)
    manifest = module.collect_results(
        matrix_dir,
        config,
        case_records=None,
        run_id="native-run",
        mode="collect_only",
        taskset_cpus=None,
    )

    assert manifest["shared_provenance"]["image_identity"] == "sha256:same"
    assert (matrix_dir / "recall_qps_points.csv").is_file()
    assert (matrix_dir / "pareto_frontier.csv").is_file()
    assert (matrix_dir / "same_recall_selection.csv").is_file()
    assert (matrix_dir / "run_manifest.json").is_file()
    orion_points = [point for point in plotted_points if point["method"] == "orion"]
    assert all(point["visited_shards"] is None for point in orion_points)
    assert all(point["ef_sum_per_query"] is None for point in orion_points)

    with (matrix_dir / "recall_qps_points.csv").open(
        "r", newline="", encoding="utf-8"
    ) as handle:
        raw_points = list(csv.DictReader(handle))
    orion_raw = [row for row in raw_points if row["method"] == "orion"]
    assert all(row["visited_shards"] == "" for row in orion_raw)
    assert all(row["ef_sum_per_query"] == "" for row in orion_raw)

    with (matrix_dir / "same_recall_selection.csv").open(
        "r", newline="", encoding="utf-8"
    ) as handle:
        selections = list(csv.DictReader(handle))
    assert len(selections) == 6
    assert {row["recall_match_status"] for row in selections} == {"strict"}


def test_collect_rejects_image_dataset_or_topology_provenance_drift(tmp_path):
    module = load_module()
    config = module.load_config(write_config(tmp_path))
    matrix_dir = tmp_path / "matrix"
    matrix_dir.mkdir()
    for index, case in enumerate(config["cases"]):
        write_case_result(
            matrix_dir,
            case,
            0.90,
            100.0,
            visited=1.5 if case.get("orion_route_trace") is True else 3,
            ef_sum=54 if case.get("orion_route_trace") is True else 120,
            image_id="sha256:drift" if index == 2 else "sha256:same",
        )
    results = [module.load_case_result(matrix_dir, case) for case in config["cases"]]

    with pytest.raises(RuntimeError, match="shared provenance mismatch"):
        module.validate_shared_provenance(results)


def test_plot_writer_generates_all_curves_without_coercing_null_orion_costs(tmp_path):
    module = load_module()
    points = [
        {
            "method": "hash_all",
            "recall_at_k": 0.90,
            "qps": 300.0,
            "latency_p95_ms": 3.0,
            "visited_shards": 3,
            "ef_sum_per_query": 120,
        },
        {
            "method": "orion",
            "recall_at_k": 0.91,
            "qps": 500.0,
            "latency_p95_ms": 2.0,
            "visited_shards": None,
            "ef_sum_per_query": None,
        },
        {
            "method": "simple_kmeans",
            "recall_at_k": 0.90,
            "qps": 420.0,
            "latency_p95_ms": 2.5,
            "visited_shards": 2,
            "ef_sum_per_query": 96,
        },
    ]

    written = module.write_plots(tmp_path, points)

    assert set(written) == {
        "recall_qps.png",
        "recall_latency.png",
        "recall_visited_shards.png",
        "recall_ef_sum.png",
    }
    assert all((tmp_path / filename).is_file() for filename in written)


def test_example_config_is_valid_and_contains_only_placeholders():
    module = load_module()
    path = (
        REPO_ROOT
        / "tools/benchmark_configs/native_auto_shard_glove200_initial.example.json"
    )
    config = module.load_config(path)

    assert {case["method"] for case in config["cases"]} == set(module.METHODS)
    assert "results/" not in path.read_text(encoding="utf-8")
    assert all("REPLACE" in case["collection"] for case in config["cases"])
    assert config["shared"]["cargo_runner"] == "tools/cargo_in_docker.sh"
    assert all(
        case.get("orion_route_trace") is True
        for case in config["cases"]
        if case["method"] == "orion"
    )
    assert all(
        "orion_route_trace" not in case
        for case in config["cases"]
        if case["method"] != "orion"
    )
