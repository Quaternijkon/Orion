from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def load_module():
    path = REPO_ROOT / "tools/native_auto_shard_benchmark.py"
    spec = importlib.util.spec_from_file_location("native_auto_shard_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_dataset(module, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with module.experiment.h5py.File(path, "w") as handle:
        handle.create_dataset(
            "train",
            data=module.experiment.np.array(
                [[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]],
                dtype=module.experiment.np.float32,
            ),
        )
        handle.create_dataset(
            "test",
            data=module.experiment.np.array(
                [[1.0, 0.0], [0.0, 1.0]], dtype=module.experiment.np.float32
            ),
        )
        handle.create_dataset(
            "neighbors",
            data=module.experiment.np.array(
                [[0, 2], [1, 2]], dtype=module.experiment.np.int64
            ),
        )


def write_orion_artifact(module, path: Path) -> tuple[dict, str]:
    payload = {
        "format_version": 1,
        "generation": 7,
        "vector_schema": {
            "vector_name": "",
            "dimension": 2,
            "distance": "Cosine",
            "datatype": "float32",
        },
        "shard_count": 3,
        "layout_sha256": "b" * 64,
        "logical_point_count": 3,
        "physical_point_count": 4,
        "upper_k": 2,
        "upper_ef_search": 8,
        "dynamic_ef_base": 20,
        "dynamic_ef_factor": 4,
        "upper_nodes": [
            {"label": 0, "vector": [1.0, 0.0], "shard_membership": [0, 1]},
            {"label": 1, "vector": [0.0, 1.0], "shard_membership": [1, 2]},
        ],
        "upper_graph": {
            "entry_point": 0,
            "max_level": 0,
            "nodes": [
                {"label": 0, "neighbors_by_level": [[1]]},
                {"label": 1, "neighbors_by_level": [[0]]},
            ],
        },
    }
    path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    return payload, module.cluster_tool.sha256_file(path)


def write_simple_kmeans_artifact(module, path: Path) -> tuple[dict, str]:
    payload = {
        "format_version": 1,
        "generation": 9,
        "vector_schema": {
            "vector_name": "",
            "dimension": 2,
            "distance": "Cosine",
            "datatype": "float32",
        },
        "shard_count": 3,
        "layout_sha256": "c" * 64,
        "logical_point_count": 3,
        "physical_point_count": 3,
        "routing_distance": "squared_l2",
        "nprobe": 2,
        "lower_hnsw_ef": 48,
        "centroids": [
            {"shard_id": 0, "vector": [1.0, 0.0]},
            {"shard_id": 1, "vector": [0.0, 1.0]},
            {"shard_id": 2, "vector": [0.5, 0.5]},
        ],
    }
    path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    return payload, module.cluster_tool.sha256_file(path)


def args_for(module, tmp_path, *extra):
    dataset = tmp_path / "dataset.hdf5"
    write_dataset(module, dataset)
    return module.parse_args(
        [
            "--method",
            "hash_all",
            "--base-url",
            "http://10.10.1.1:6333",
            "--collection",
            "native_hash",
            "--hdf5-path",
            str(dataset),
            "--topology",
            str(REPO_ROOT / "tools/distributed/cloudlab_orion_4node.json"),
            "--output-dir",
            str(tmp_path / "output"),
            *extra,
        ]
    )


def route_trace_payload(
    *,
    artifact_sha256: str,
    query_sha256: str,
    query_count: int = 2,
    dimension: int = 2,
) -> dict:
    return {
        "format_version": 1,
        "artifact": {
            "path": "/artifact.json",
            "generation": 7,
            "layout_sha256": "b" * 64,
            "sha256": artifact_sha256,
            "file_sha256": artifact_sha256,
            "vector_schema": {
                "vector_name": "",
                "dimension": 2,
                "distance": "Cosine",
                "datatype": "float32",
            },
            "shard_count": 3,
            "upper_k": 2,
            "upper_ef_search": 8,
            "dynamic_ef_base": 20,
            "dynamic_ef_factor": 4,
        },
        "queries": {
            "path": "/queries.f32le",
            "sha256": query_sha256,
            "query_count": query_count,
            "dimension": dimension,
        },
        "aggregate": {
            "query_count": query_count,
            "visited_shards": {
                "average": 1.5,
                "min": 1,
                "max": 2,
                "p50": 1,
                "p95": 2,
                "p99": 2,
            },
            "entry_point_count": {
                "average": 2.0,
                "min": 1,
                "max": 3,
                "p50": 1,
                "p95": 3,
                "p99": 3,
            },
            "ef_sum_per_query": {
                "average": 54.0,
                "min": 24,
                "max": 84,
                "p50": 24,
                "p95": 84,
                "p99": 84,
            },
            "per_shard": [],
        },
    }


def test_argument_guards_restrict_hnsw_ef_and_repository_output(tmp_path):
    module = load_module()
    missing_ef = args_for(module, tmp_path / "missing")
    with pytest.raises(ValueError, match="requires an explicit"):
        module.validate_args(missing_ef)

    args = args_for(module, tmp_path, "--hnsw-ef", "40")
    module.validate_args(args)

    args.method = "orion"
    with pytest.raises(ValueError, match="only valid"):
        module.validate_args(args)

    args.hnsw_ef = None
    args.method = "simple_kmeans"
    args.orion_route_trace = True
    with pytest.raises(ValueError, match="only valid for --method orion"):
        module.validate_args(args)

    with pytest.raises(ValueError, match="outside the repository"):
        module.create_output_directory(REPO_ROOT / "results/native")

    output = module.create_output_directory(tmp_path / "new-output")
    assert output.is_dir()
    with pytest.raises(FileExistsError):
        module.create_output_directory(output)


def test_expected_placement_map_accepts_raw_and_transition_wrapped_json(tmp_path):
    module = load_module()
    expected = {0: 303, 1: 202, 2: 404}

    raw_path = tmp_path / "raw-placement.json"
    raw_path.write_text(json.dumps(expected), encoding="utf-8")
    raw, raw_proof = module.load_expected_placement_map(raw_path, 3)
    assert raw == expected
    assert raw_proof["selected_key"] is None
    assert raw_proof["sha256"] == module.experiment.sha256_path(raw_path)

    wrapped_path = tmp_path / "weighted-move.json"
    wrapped_path.write_text(
        json.dumps(
            {
                "baseline_placement": {0: 202, 1: 303, 2: 404},
                "target_placement": expected,
                "final_placement": expected,
            }
        ),
        encoding="utf-8",
    )
    wrapped, wrapped_proof = module.load_expected_placement_map(wrapped_path, 3)
    assert wrapped == expected
    assert wrapped_proof["selected_key"] == "target_placement"
    assert wrapped_proof["matching_keys"] == [
        "target_placement",
        "final_placement",
    ]
    assert wrapped_proof["placement_sha256"] == raw_proof["placement_sha256"]

    conflicting_path = tmp_path / "conflicting-move.json"
    conflicting_path.write_text(
        json.dumps(
            {
                "target_placement": expected,
                "final_placement": {0: 202, 1: 303, 2: 404},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="conflicting wrapped placement fields"):
        module.load_expected_placement_map(conflicting_path, 3)

    incomplete_path = tmp_path / "incomplete.json"
    incomplete_path.write_text(json.dumps({0: 303, 1: 202}), encoding="utf-8")
    with pytest.raises(ValueError, match="contiguous shard range"):
        module.load_expected_placement_map(incomplete_path, 3)


def test_live_placement_defaults_to_round_robin_and_explicit_map_is_opt_in(
    monkeypatch, tmp_path
):
    module = load_module()
    args = args_for(module, tmp_path, "--hnsw-ef", "40")
    calls = []

    def fake_round_robin(*call_args):
        calls.append(("round_robin", call_args))
        return {"valid": True, "placement_mode": "round_robin"}

    def fake_explicit(*call_args):
        calls.append(("explicit", call_args))
        return {"valid": True, "placement_mode": "explicit"}

    monkeypatch.setattr(
        module.experiment,
        "validate_numeric_shard_round_robin_placement",
        fake_round_robin,
    )
    monkeypatch.setattr(
        module.experiment,
        "validate_numeric_shard_explicit_placement",
        fake_explicit,
    )

    default_proof = module.validate_live_numeric_placement(
        args,
        {"info": True},
        {"cluster": True},
        [202, 303, 404],
        3,
    )
    assert default_proof["placement_mode"] == "round_robin"
    assert [call[0] for call in calls] == ["round_robin"]

    placement_path = tmp_path / "placement.json"
    placement_path.write_text(
        json.dumps({"target_placement": {0: 303, 1: 202, 2: 404}}),
        encoding="utf-8",
    )
    args.expected_placement_map = str(placement_path)
    explicit_proof = module.validate_live_numeric_placement(
        args,
        {"info": True},
        {"cluster": True},
        [202, 303, 404],
        3,
    )
    assert explicit_proof["placement_mode"] == "explicit"
    assert explicit_proof["expected_placement_source"]["selected_key"] == (
        "target_placement"
    )
    assert [call[0] for call in calls] == ["round_robin", "explicit"]
    assert calls[-1][1][-1] == {0: 303, 1: 202, 2: 404}


def test_repository_binding_requires_same_clean_tracked_commit():
    module = load_module()
    deployment = {"repository": {"commit": "abc123"}}

    proof = module.validate_repository_binding(
        {
            "commit": "abc123",
            "dirty": True,
            "tracked_dirty": False,
            "untracked_entry_count": 7,
        },
        deployment,
    )

    assert proof == {
        "deployment_commit": "abc123",
        "benchmark_commit": "abc123",
        "tracked_dirty": False,
        "untracked_entry_count": 7,
    }

    with pytest.raises(RuntimeError, match="tracked changes"):
        module.validate_repository_binding(
            {"commit": "abc123", "tracked_dirty": True}, deployment
        )
    with pytest.raises(RuntimeError, match="commit mismatch"):
        module.validate_repository_binding(
            {"commit": "different", "tracked_dirty": False}, deployment
        )


def test_hash_all_scalar_hnsw_ef_uses_standard_request_field_only():
    module = load_module()
    request = module.experiment.standard_dense_vector_request(
        [0.1, 0.2], 10, api="search", hnsw_ef=64
    )
    assert request["params"] == {"hnsw_ef": 64}
    assert {
        "shard_key",
        "hnsw_entry_points",
        "hnsw_entry_points_by_shard",
        "hnsw_ef_by_shard",
        "source_id_dedup_block_size",
    }.isdisjoint(request)


def test_route_reporting_never_claims_orion_actual_visited_shards():
    module = load_module()
    orion = {
        "layout_sha256": "a" * 64,
        "logical_point_count": 3,
        "physical_point_count": 4,
        "shard_count": 3,
        "upper_k": 2,
        "upper_ef_search": 8,
        "dynamic_ef_base": 20,
        "dynamic_ef_factor": 4,
    }
    report = module.route_reporting("orion", 3, orion, None)
    assert report["visited_shards"] is None
    assert report["visited_shards_source"] == "unknown_without_server_trace"
    assert report["ef_sum_per_query"] is None

    simple = {
        "layout_sha256": "b" * 64,
        "logical_point_count": 3,
        "physical_point_count": 3,
        "shard_count": 3,
        "nprobe": 2,
        "lower_hnsw_ef": 48,
    }
    simple_report = module.route_reporting("simple_kmeans", 3, simple, None)
    assert simple_report["visited_shards"] == 2
    assert simple_report["ef_sum_per_query"] == 96
    assert simple_report["visited_shards_source"].startswith("artifact_derived")

    hash_report = module.route_reporting("hash_all", 3, None, 40)
    assert hash_report["visited_shards"] == 3
    assert hash_report["ef_sum_per_query"] == 120


def test_exact_orion_route_trace_exports_queries_validates_provenance_and_reports_costs(
    monkeypatch, tmp_path
):
    module = load_module()
    artifact_path = tmp_path / "generation-7.json"
    _artifact, artifact_sha256 = write_orion_artifact(module, artifact_path)
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    args = args_for(module, tmp_path / "args", "--hnsw-ef", "40")
    args.method = "orion"
    args.hnsw_ef = None
    args.orion_route_trace = True
    args.cargo_runner = str(tmp_path / "cargo-runner")
    args.cargo_target_dir = str(tmp_path / "cargo-target")
    queries = module.experiment.np.array(
        [[0.6, 0.8], [0.0, 1.0]], dtype=module.experiment.np.float32
    )
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["env"] = kwargs["env"]
        separator = command.index("--")
        query_path = Path(command[separator + 2])
        output_path = Path(command[separator + 5])
        query_bytes = query_path.read_bytes()
        captured["query_path"] = query_path
        captured["query_bytes"] = query_bytes
        output_path.write_text(
            json.dumps(
                route_trace_payload(
                    artifact_sha256=artifact_sha256,
                    query_sha256=hashlib.sha256(query_bytes).hexdigest(),
                )
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, "trace stdout", "trace stderr")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    proof = module.run_orion_route_trace(
        args,
        artifact_path=artifact_path,
        artifact_proof={
            "sha256": artifact_sha256,
            "generation": 7,
            "layout_sha256": "b" * 64,
            "shard_count": 3,
        },
        policy={"artifact_sha256": artifact_sha256},
        eval_queries=queries,
        output_dir=output_dir,
    )

    assert captured["command"][1:7] == [
        "run",
        "--release",
        "-p",
        "collection",
        "--example",
        "orion_route_trace",
    ]
    assert captured["env"]["CARGO_TARGET_DIR"] == str(
        (tmp_path / "cargo-target").resolve()
    )
    assert captured["query_bytes"] == module.experiment.np.ascontiguousarray(
        queries, dtype=module.experiment.np.dtype("<f4")
    ).tobytes(order="C")
    assert not captured["query_path"].exists()
    assert proof["temporary_query_file_removed"] is True
    assert proof["included_in_timed_benchmark"] is False
    assert proof["metrics"]["visited_shards"] == 1.5
    assert proof["metrics"]["ef_sum_per_query"] == 54.0
    assert (output_dir / "orion_route_trace.json").is_file()
    assert (output_dir / "orion_route_trace.stdout.log").read_text() == "trace stdout"
    report = module.route_reporting(
        "orion",
        3,
        {
            "layout_sha256": "b" * 64,
            "logical_point_count": 3,
            "physical_point_count": 4,
            "shard_count": 3,
            "upper_k": 2,
            "upper_ef_search": 8,
            "dynamic_ef_base": 20,
            "dynamic_ef_factor": 4,
        },
        None,
        proof["metrics"],
    )
    assert report["visited_shards_source"] == "exact_offline_production_router_trace"
    assert report["ef_sum_source"] == "exact_offline_production_router_trace"


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("artifact_file_checksum", "artifact provenance mismatch"),
        ("artifact_canonical_checksum", "artifact provenance mismatch"),
        ("query_checksum", "query provenance mismatch"),
        ("query_count", "query provenance mismatch"),
    ],
)
def test_orion_route_trace_rejects_checksum_or_count_mismatch(mutation, message):
    module = load_module()
    checksum = "a" * 64
    query_checksum = "b" * 64
    trace = route_trace_payload(
        artifact_sha256=checksum,
        query_sha256=query_checksum,
    )
    if mutation == "artifact_file_checksum":
        trace["artifact"]["file_sha256"] = "c" * 64
    elif mutation == "artifact_canonical_checksum":
        trace["artifact"]["sha256"] = "c" * 64
    elif mutation == "query_checksum":
        trace["queries"]["sha256"] = "c" * 64
    else:
        trace["queries"]["query_count"] = 1

    with pytest.raises(RuntimeError, match=message):
        module.validate_orion_route_trace_output(
            trace,
            artifact_proof={
                "sha256": checksum,
                "generation": 7,
                "layout_sha256": "b" * 64,
                "shard_count": 3,
            },
            expected_canonical_sha256=checksum,
            expected_query_count=2,
            expected_dimension=2,
            expected_query_sha256=query_checksum,
        )


def test_orion_route_trace_subprocess_failure_preserves_logs_and_removes_queries(
    monkeypatch, tmp_path
):
    module = load_module()
    artifact_path = tmp_path / "generation-7.json"
    _artifact, artifact_sha256 = write_orion_artifact(module, artifact_path)
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    args = args_for(module, tmp_path / "args", "--hnsw-ef", "40")
    args.method = "orion"
    args.hnsw_ef = None
    args.orion_route_trace = True
    captured = {}

    def fake_run(command, **_kwargs):
        separator = command.index("--")
        captured["query_path"] = Path(command[separator + 2])
        return subprocess.CompletedProcess(command, 17, "partial stdout", "route failed")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="exit code 17"):
        module.run_orion_route_trace(
            args,
            artifact_path=artifact_path,
            artifact_proof={
                "sha256": artifact_sha256,
                "generation": 7,
                "layout_sha256": "b" * 64,
                "shard_count": 3,
            },
            policy={"artifact_sha256": artifact_sha256},
            eval_queries=module.experiment.np.array(
                [[1.0, 0.0], [0.0, 1.0]], dtype=module.experiment.np.float32
            ),
            output_dir=output_dir,
        )

    assert not captured["query_path"].exists()
    assert (output_dir / "orion_route_trace.stdout.log").read_text() == "partial stdout"
    assert (output_dir / "orion_route_trace.stderr.log").read_text() == "route failed"


@pytest.mark.parametrize("method", ["orion", "simple_kmeans"])
def test_routed_artifact_validation_binds_policy_schema_shards_and_counts(
    tmp_path, method
):
    module = load_module()
    artifact_path = tmp_path / f"{method}.json"
    if method == "orion":
        payload, checksum = write_orion_artifact(module, artifact_path)
        points_count = 4
        generation = 7
    else:
        payload, checksum = write_simple_kmeans_artifact(module, artifact_path)
        points_count = 3
        generation = 9
    policy = {
        "type": method,
        "generation": generation,
        "artifact_sha256": checksum,
    }
    loaded, proof = module.validate_artifact(
        method,
        artifact_path,
        policy,
        {
            "vector_name": "",
            "dimension": 2,
            "distance": "Cosine",
            "datatype": "float32",
        },
        shard_count=3,
        points_count=points_count,
        train_count=3,
    )
    assert loaded == payload
    assert proof["status"] == "verified"
    assert proof["sha256"] == checksum
    assert len(proof["routing_structure_sha256"]) == 64

    with pytest.raises(RuntimeError, match="shard_count"):
        module.validate_artifact(
            method,
            artifact_path,
            policy,
            {
                "vector_name": "",
                "dimension": 2,
                "distance": "Cosine",
                "datatype": "float32",
            },
            shard_count=4,
            points_count=points_count,
            train_count=3,
        )


def test_artifact_bundle_fingerprint_excludes_runtime_only_parameters(
    monkeypatch, tmp_path
):
    module = load_module()
    artifact_path = tmp_path / "generation-1.json"
    artifact_path.write_text("{}\n", encoding="utf-8")
    build_manifest_path = tmp_path / "build-manifest.json"
    import_manifest_path = tmp_path / "numeric.manifest.json"
    build_manifest = {
        "parameters": {
            "initial_num_shards": 31,
            "sample_denominator": 32,
            "upper_k": 36,
            "upper_search_ef": 36,
            "dynamic_ef_base": 48,
            "dynamic_ef_factor": 15,
            "generation": 1,
            "cargo_target_dir": "/external/target",
        },
        "dataset": {"sha256": "a" * 64, "dimension": 200},
        "routing": {"effective_num_shards": 46, "shard_counts": [10, 11]},
    }
    import_manifest = {
        "vectors_sha256": "b" * 64,
        "assignments_sha256": "c" * 64,
    }
    build_manifest_path.write_text(json.dumps(build_manifest), encoding="utf-8")
    import_manifest_path.write_text(json.dumps(import_manifest), encoding="utf-8")
    monkeypatch.setattr(
        module.prepare,
        "load_routed_layout",
        lambda *_args: {
            "artifact_path": str(artifact_path),
            "build_manifest_path": str(build_manifest_path),
            "build_manifest_sha256": "d" * 64,
            "import_manifest_path": str(import_manifest_path),
            "import_manifest_sha256": "e" * 64,
        },
    )
    artifact_proof = {
        "layout_sha256": "c" * 64,
        "routing_structure_sha256": "f" * 64,
        "logical_point_count": 1000,
        "physical_point_count": 1200,
        "shard_count": 46,
        "vector_schema": {
            "vector_name": "",
            "dimension": 200,
            "distance": "Cosine",
            "datatype": "float32",
        },
    }

    first = module.validate_artifact_bundle("orion", artifact_path, artifact_proof)
    build_manifest["parameters"].update(
        {
            "upper_k": 96,
            "upper_search_ef": 96,
            "dynamic_ef_base": 64,
            "generation": 2,
            "cargo_target_dir": "/another/target",
        }
    )
    build_manifest_path.write_text(json.dumps(build_manifest), encoding="utf-8")
    second = module.validate_artifact_bundle("orion", artifact_path, artifact_proof)

    assert first["offline_layout_fingerprint"] == second["offline_layout_fingerprint"]
    assert first["formal_evidence_eligible"] is True
    assert first["vectors_sha256"] == "b" * 64
    assert first["assignments_sha256"] == "c" * 64


def test_hash_all_mocked_run_writes_reproducible_outputs(monkeypatch, tmp_path):
    module = load_module()
    deployment = tmp_path / "deployment.json"
    deployment.write_text(
        json.dumps(
            {
                "repository": {"commit": "current-commit"},
                "image": {"tag": "image:tag", "id": "sha256:image"},
                "nodes": [{"role": "controller", "cpuset": "0-7"}],
            }
        ),
        encoding="utf-8",
    )
    args = args_for(
        module,
        tmp_path,
        "--deployment-manifest",
        str(deployment),
        "--warmup-query-count",
        "1",
        "--eval-query-count",
        "2",
        "--top-k",
        "2",
        "--batch-size",
        "2",
        "--stability-repeats",
        "2",
        "--hnsw-ef",
        "40",
        "--write-per-query-metrics",
    )
    collection_info = {
        "status": "green",
        "optimizer_status": "ok",
        "points_count": 3,
        "config": {
            "params": {
                "vectors": {"size": 2, "distance": "Cosine"},
                "sharding_method": "auto",
                "shard_number": 3,
                "replication_factor": 1,
            }
        },
    }
    collection_cluster = {
        "peer_id": 101,
        "shard_count": 3,
        "local_shards": [],
        "remote_shards": [],
        "shard_transfers": [],
    }
    preflight = {
        "worker_peer_ids": [202, 303, 404],
        "raw": {"peer_id": 101, "peers": {}},
    }
    evaluation_calls = []

    def fake_evaluate(
        _base_url,
        _collection,
        queries,
        _neighbors,
        top_k,
        batch_size,
        *,
        api,
        vector_name,
        hnsw_ef,
        include_per_query_metrics,
    ):
        evaluation_calls.append(
            (
                len(queries),
                top_k,
                batch_size,
                api,
                vector_name,
                hnsw_ef,
                include_per_query_metrics,
            )
        )
        assert hnsw_ef == 40
        rows = [
            {
                "query_index": index,
                "hits_at_k": top_k,
                "recall_at_k": 1.0,
                "retrieved_ids": "0 1",
                "ground_truth_ids": "0 1",
            }
            for index in range(len(queries))
        ]
        return {
            "query_count": len(queries),
            "recall_at_k": 1.0,
            "qps": 100.0 + len(evaluation_calls),
            "wall_s": 0.02,
            "latency_p50_ms": 2.0,
            "latency_p95_ms": 3.0,
            "latency_p99_ms": 4.0,
            "per_query_rows": rows if include_per_query_metrics else [],
        }

    monkeypatch.setattr(
        module.experiment, "validate_cluster_preflight", lambda *_args: preflight
    )
    monkeypatch.setattr(
        module.experiment, "collection_info", lambda *_args: collection_info
    )
    monkeypatch.setattr(
        module.experiment,
        "collection_cluster_info",
        lambda *_args: collection_cluster,
    )
    monkeypatch.setattr(
        module.experiment,
        "wait_collection_indexed",
        lambda *_args, **_kwargs: collection_info,
    )
    monkeypatch.setattr(
        module.experiment,
        "validate_numeric_shard_round_robin_placement",
        lambda *_args: {"valid": True, "shards_per_worker": {202: 1, 303: 1, 404: 1}},
    )
    monkeypatch.setattr(
        module.experiment, "evaluate_standard_dense_vector_batches", fake_evaluate
    )
    monkeypatch.setattr(
        module.experiment,
        "repository_provenance",
        lambda *_args: {
            "commit": "current-commit",
            "dirty": True,
            "tracked_dirty": False,
            "untracked_entry_count": 4,
        },
    )

    output = module.run(args)
    assert output == tmp_path / "output"
    assert [call[0] for call in evaluation_calls] == [1, 2, 1, 2]
    assert (output / "run_manifest.json").is_file()
    assert (output / "stability_runs.csv").is_file()
    assert (output / "final_metrics.csv").is_file()
    assert (output / "per_query_metrics.csv").is_file()
    assert (output / "summary.json").is_file()

    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["request_contract"] == {
        "standard_coordinator_request": True,
        "shard_selector": False,
        "entry_point_hints": False,
        "per_shard_ef": False,
        "source_id_hint": False,
        "scalar_hnsw_ef": 40,
    }
    assert manifest["route_reporting"]["visited_shards"] == 3
    assert manifest["route_reporting"]["ef_sum_per_query"] == 120
    assert manifest["deployment"]["image"]["id"] == "sha256:image"
    assert manifest["repository"]["commit"] == "current-commit"
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["final_metrics"]["recall_at_k"] == 1.0
    assert summary["placement_valid"] is True
