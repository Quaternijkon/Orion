from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import re
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


class FakeHeldLock:
    fd = 77
    token = "matrix-lock-token"

    def inheritance_cli_arguments(self):
        return [
            "--benchmark-lock-fd",
            str(self.fd),
            "--benchmark-lock-token",
            self.token,
        ]

    def inheritance_pass_fds(self):
        return (self.fd,)

    def evidence(self):
        return {
            "path": "/runs/benchmark.lock",
            "mode": "acquired",
            "token_sha256": "a" * 64,
        }


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
    wire = {
        "scope": "controller",
        "requested_version": "2",
        "current_version": "2",
        "matches_requested": True,
    }
    peer_premerge = {
        "scope": "controller",
        "requested_mode": "enabled",
        "current_mode": "enabled",
        "requested_shards_per_rpc": "4",
        "current_shards_per_rpc": "4",
        "matches_requested": True,
    }
    transport_identity = {
        "schema_version": 1,
        "image": {
            "id": image_id,
            "tag": "native:test",
            "source_fingerprint": "a" * 64,
            "tar_sha256": "b" * 64,
            "orion_compact_wire_max_version": "2",
        },
        "orion_compact_wire": wire,
        "peer_premerge": peer_premerge,
        "nodes": [],
    }
    transport_identity_sha256 = hashlib.sha256(
        json.dumps(
            transport_identity,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
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
            "manifest_sha256": "c" * 64,
            "image": {
                "id": image_id,
                "tag": "native:test",
                "source_fingerprint": "a" * 64,
                "tar_sha256": "b" * 64,
                "capabilities": {"orion_compact_wire_max_version": "2"},
            },
            "repository": {"commit": "commit-1", "dirty": False},
            "orion_compact_wire": wire,
            "peer_premerge": peer_premerge,
            "transport_identity": transport_identity,
            "transport_identity_sha256": transport_identity_sha256,
        },
        "repository": {"commit": "commit-1", "dirty": False},
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
        "collection_info": {
            "status": "green",
            "optimizer_status": "ok",
            "update_queue": {"length": 0},
            "config": {
                "params": {
                    "sharding_method": "auto",
                    "shard_number": 3,
                    "replication_factor": 1,
                    "write_consistency_factor": 1,
                },
                "hnsw_config": {
                    "m": 32,
                    "ef_construct": 100,
                    "full_scan_threshold": 10000,
                },
                "optimizer_config": {
                    "indexing_threshold": 10000,
                    "default_segment_number": 0,
                },
            },
        },
        "collection_cluster": {
            "shard_count": 3,
            "shard_transfers": [],
        },
        "indexing_readiness": {
            "status": "green",
            "optimizer_status": "ok",
            "points_count": 1000,
            "indexed_vectors_count": 1000,
            "fully_indexed": True,
            "completion_mode": "fully_indexed",
            "update_queue": {"length": 0},
            "shard_transfers": [],
        },
        "placement_proof": {
            "valid": True,
            "shard_count": 3,
            "replication_factor": 1,
            "placement": {0: 202, 1: 303, 2: 404},
            "shard_transfers": [],
        },
        "parameters": {
            "vector_distance": "cosine",
            "vector_name": "",
            "eval_query_count": 100,
            "top_k": 10,
            "batch_size": 20,
        },
    }


def write_orion_route_trace(
    case_dir: Path,
    *,
    query_count: int,
    shard_count: int,
    visited_average: float,
    ef_sum_average: float,
    artifact_file_sha256: str,
    artifact_canonical_sha256: str,
    artifact_generation: int,
    artifact_layout_sha256: str,
    dimension: int,
    query_sha256: str,
) -> tuple[Path, dict, dict, dict]:
    lower_visited = int(visited_average)
    high_query_count = round((float(visited_average) - lower_visited) * query_count)
    if not math.isclose(
        lower_visited + high_query_count / query_count,
        float(visited_average),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("test route trace visited average is not exactly representable")
    visited_counts = [
        lower_visited + (1 if query_index < high_query_count else 0)
        for query_index in range(query_count)
    ]
    if not visited_counts or min(visited_counts) <= 0 or max(visited_counts) > shard_count:
        raise ValueError("test route trace visited count is outside shard range")
    if not float(ef_sum_average).is_integer() or ef_sum_average <= 0:
        raise ValueError("test route trace EF sum must be a positive integer")
    ef_sum = int(ef_sum_average)

    per_query = []
    for query_index, visited_shards in enumerate(visited_counts):
        base_ef, remainder = divmod(ef_sum, visited_shards)
        if base_ef <= 0:
            raise ValueError("test route trace per-shard EF must be positive")
        targets = [
            {
                "shard_id": shard_id,
                "ef": base_ef + (1 if shard_id < remainder else 0),
                "entry_points": [query_index * 10 + shard_id + 1],
            }
            for shard_id in range(visited_shards)
        ]
        per_query.append(
            {
                "query_index": query_index,
                "visited_shards": visited_shards,
                "entry_point_count": visited_shards,
                "ef_sum": ef_sum,
                "targets": targets,
            }
        )
    route_sha256 = hashlib.sha256(
        json.dumps(
            per_query,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    aggregate = {
        "query_count": query_count,
        "visited_shards": {"average": float(visited_average)},
        "entry_point_count": {"average": float(visited_average)},
        "ef_sum_per_query": {"average": float(ef_sum_average)},
    }
    per_query_fields = [
        "query_index",
        "visited_shards",
        "entry_point_count",
        "ef_sum",
        "targets[].shard_id",
        "targets[].ef",
        "targets[].entry_points",
    ]
    trace = {
        "format_version": 1,
        "artifact": {
            "path": "/artifacts/orion.json",
            "generation": artifact_generation,
            "layout_sha256": artifact_layout_sha256,
            "sha256": artifact_canonical_sha256,
            "file_sha256": artifact_file_sha256,
            "vector_schema": {
                "vector_name": "",
                "dimension": dimension,
                "distance": "Cosine",
                "datatype": "float32",
            },
            "shard_count": shard_count,
        },
        "queries": {
            "path": "/queries.f32le",
            "sha256": query_sha256,
            "query_count": query_count,
            "dimension": dimension,
        },
        "aggregate": aggregate,
        "per_query": per_query,
    }
    trace_path = case_dir / "orion_route_trace.json"
    trace_path.write_text(json.dumps(trace), encoding="utf-8")
    per_query_metrics = {
        "query_count": query_count,
        "canonical_sha256": route_sha256,
        "ordered_targets": True,
        "fields": per_query_fields,
        "visited_shards_average": float(visited_average),
        "entry_point_count_average": float(visited_average),
        "ef_sum_average": float(ef_sum_average),
    }
    metrics = {
        "visited_shards": float(visited_average),
        "ef_sum_per_query": float(ef_sum_average),
        "query_count": query_count,
        "dimension": dimension,
        "query_sha256": query_sha256,
        "artifact_generation": artifact_generation,
        "artifact_layout_sha256": artifact_layout_sha256,
        "artifact_file_sha256": artifact_file_sha256,
        "artifact_canonical_sha256": artifact_canonical_sha256,
        "per_query": per_query_metrics,
        "per_query_canonical_sha256": route_sha256,
    }
    manifest_per_query = {
        "included": True,
        "query_count": query_count,
        "canonical_sha256": route_sha256,
        "ordered_targets": True,
        "fields": per_query_fields,
        "source": "orion_route_trace.json#per_query",
    }
    return trace_path, aggregate, metrics, manifest_per_query


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
    if case["method"] in {"orion", "simple_kmeans"}:
        layout_sha256 = (
            "a" * 64 if case["method"] == "orion" else "b" * 64
        )
        structure_sha256 = (
            "c" * 64 if case["method"] == "orion" else "d" * 64
        )
        manifest["artifact"] = {
            "status": "verified",
            "sha256": "9" * 64,
            "generation": 7,
            "layout_sha256": layout_sha256,
            "routing_structure_sha256": structure_sha256,
            "logical_point_count": 1000,
            "physical_point_count": 1200 if case["method"] == "orion" else 1000,
            "shard_count": 3,
            "vector_schema": {
                "vector_name": "",
                "dimension": 200,
                "distance": "Cosine",
                "datatype": "float32",
            },
        }
        manifest["artifact_bundle"] = {
            "status": "verified",
            "formal_evidence_eligible": True,
            "offline_layout_fingerprint": structure_sha256,
            "vectors_sha256": "e" * 64,
            "assignments_sha256": layout_sha256,
        }
    if case.get("orion_route_trace") is True:
        artifact_canonical_sha256 = "8" * 64
        query_sha256 = "7" * 64
        manifest["live_policy"] = {
            "artifact_sha256": artifact_canonical_sha256,
        }
        trace_path, aggregate, metrics, per_query_proof = write_orion_route_trace(
            case_dir,
            query_count=manifest["parameters"]["eval_query_count"],
            shard_count=manifest["artifact"]["shard_count"],
            visited_average=float(visited),
            ef_sum_average=float(ef_sum),
            artifact_file_sha256=manifest["artifact"]["sha256"],
            artifact_canonical_sha256=artifact_canonical_sha256,
            artifact_generation=manifest["artifact"]["generation"],
            artifact_layout_sha256=manifest["artifact"]["layout_sha256"],
            dimension=manifest["artifact"]["vector_schema"]["dimension"],
            query_sha256=query_sha256,
        )
        manifest["orion_route_trace"] = {
            "status": "verified",
            "source": "exact_offline_production_router_trace",
            "trace_path": str(trace_path.resolve()),
            "metrics": metrics,
            "aggregate": aggregate,
            "per_query": per_query_proof,
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
    assert config["require_strict_same_recall"] is False
    assert config["same_recall_pairwise_window"] == config["same_recall_window"]
    command = module.benchmark_command(
        config["shared"],
        config["cases"][0],
        tmp_path / "case-output",
        "8-19",
        FakeHeldLock(),
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
        FakeHeldLock(),
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

    invalid_strict = base_config()
    invalid_strict["require_strict_same_recall"] = "false"
    with pytest.raises(ValueError, match="must be a boolean"):
        module.load_config(write_config(tmp_path / "invalid-strict", invalid_strict))

    invalid_pairwise = base_config()
    invalid_pairwise["same_recall_pairwise_window"] = -0.001
    with pytest.raises(ValueError, match="pairwise_window must be non-negative"):
        module.load_config(
            write_config(tmp_path / "invalid-pairwise", invalid_pairwise)
        )

    missing_deployment = base_config()
    del missing_deployment["shared"]["deployment_manifest"]
    with pytest.raises(ValueError, match="deployment_manifest"):
        module.load_config(
            write_config(tmp_path / "missing-deployment", missing_deployment)
        )


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

    run_kwargs = []

    def fake_run(command, **kwargs):
        commands.append(command)
        run_kwargs.append(kwargs)
        output = Path(command[command.index("--output-dir") + 1])
        output.mkdir()
        return subprocess.CompletedProcess(command, 0, "raw stdout", "raw stderr")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    records = module.execute_cases(matrix_dir, config, "8-19", FakeHeldLock())

    assert len(records) == 3
    assert len(commands) == 3
    assert all(command[:3] == ["taskset", "-c", "8-19"] for command in commands)
    assert all(kwargs["pass_fds"] == (77,) for kwargs in run_kwargs)
    assert all("--benchmark-lock-fd" in command for command in commands)
    assert all("--benchmark-lock-fd" not in record["command"] for record in records)
    assert (matrix_dir / "cases/hash40").is_dir()
    assert (matrix_dir / "logs/hash40.stdout.log").read_text() == "raw stdout"
    assert (matrix_dir / "logs/hash40.stderr.log").read_text() == "raw stderr"


def test_matrix_run_holds_one_lock_across_all_cases_and_collection(
    monkeypatch, tmp_path
):
    module = load_module()
    config = module.load_config(write_config(tmp_path / "config"))
    manifest = tmp_path / "deployment" / "manifest.json"
    manifest.parent.mkdir()
    manifest.write_text("{}\n", encoding="utf-8")
    config["shared"]["deployment_manifest"] = str(manifest)
    matrix_dir = tmp_path / "matrix"
    events = []

    class LockContext:
        def __enter__(self):
            events.append("lock-enter")
            return FakeHeldLock()

        def __exit__(self, *_exc):
            events.append("lock-exit")

    monkeypatch.setattr(module, "load_config", lambda _path: config)
    monkeypatch.setattr(
        module.benchmark_lock,
        "hold_from_args",
        lambda *_args, **_kwargs: LockContext(),
    )
    monkeypatch.setattr(
        module,
        "matrix_directory",
        lambda *_args, **_kwargs: matrix_dir,
    )

    def fake_execute(*_args):
        events.append("execute")
        return [{"name": "cases"}]

    def fake_collect(*_args):
        events.append("collect")
        return {}

    monkeypatch.setattr(module, "execute_cases", fake_execute)
    monkeypatch.setattr(module, "collect_results", fake_collect)
    args = SimpleNamespace(
        config="config.json",
        run_id="matrix-run",
        output_root=str(tmp_path),
        run=True,
        collect_only=False,
        taskset_cpus="8-19",
        benchmark_lock_fd=None,
        benchmark_lock_token=None,
    )

    assert module.run(args) == matrix_dir
    assert events == ["lock-enter", "execute", "collect", "lock-exit"]


def test_collect_only_does_not_acquire_or_claim_a_new_benchmark_lock(
    monkeypatch, tmp_path
):
    module = load_module()
    config = module.load_config(write_config(tmp_path / "config"))
    matrix_dir = tmp_path / "existing-matrix"
    matrix_dir.mkdir()
    observed = []

    monkeypatch.setattr(module, "load_config", lambda _path: config)
    monkeypatch.setattr(
        module,
        "matrix_directory",
        lambda *_args, **kwargs: (
            matrix_dir
            if kwargs["must_exist"] is True
            else pytest.fail("collect-only attempted to create a matrix directory")
        ),
    )
    monkeypatch.setattr(
        module.benchmark_lock,
        "hold_from_args",
        lambda *_args, **_kwargs: pytest.fail(
            "collect-only must not acquire a measurement lock"
        ),
    )

    def fake_collect(*args):
        observed.append(args)
        return {}

    monkeypatch.setattr(module, "collect_results", fake_collect)
    args = SimpleNamespace(
        config="config.json",
        run_id="matrix-run",
        output_root=str(tmp_path),
        run=False,
        collect_only=True,
        taskset_cpus=None,
        benchmark_lock_fd=None,
        benchmark_lock_token=None,
    )

    assert module.run(args) == matrix_dir
    assert observed[0][-1] is None


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


def test_collect_rejects_orion_trace_without_matching_per_query_fingerprint(tmp_path):
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
    manifest["orion_route_trace"]["metrics"][
        "per_query_canonical_sha256"
    ] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RuntimeError, match="fingerprint mismatch"):
        module.load_case_result(matrix_dir, case)


def test_collect_revalidates_orion_route_trace_source_file(tmp_path):
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

    result = module.load_case_result(matrix_dir, case)

    trace_path = matrix_dir / "cases" / case["name"] / "orion_route_trace.json"
    assert result["point"]["orion_route_trace_file_sha256"] == hashlib.sha256(
        trace_path.read_bytes()
    ).hexdigest()
    assert result["point"]["orion_route_trace_per_query_sha256"] == result[
        "manifest"
    ]["orion_route_trace"]["per_query"]["canonical_sha256"]
    assert re.fullmatch(
        r"[0-9a-f]{64}",
        result["point"]["orion_route_trace_aggregate_sha256"],
    )


def test_collect_rejects_missing_orion_route_trace_source_file(tmp_path):
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
    trace_path = matrix_dir / "cases" / case["name"] / "orion_route_trace.json"
    trace_path.unlink()

    with pytest.raises(FileNotFoundError, match="route trace file is missing"):
        module.load_case_result(matrix_dir, case)


def test_collect_rejects_invalid_orion_route_trace_json(tmp_path):
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
    trace_path = matrix_dir / "cases" / case["name"] / "orion_route_trace.json"
    trace_path.write_text("{invalid", encoding="utf-8")

    with pytest.raises(RuntimeError, match="invalid JSON"):
        module.load_case_result(matrix_dir, case)


@pytest.mark.parametrize("mutation", ["per_query", "aggregate"])
def test_collect_rejects_tampered_orion_route_trace_source_file(
    tmp_path, mutation
):
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
    trace_path = matrix_dir / "cases" / case["name"] / "orion_route_trace.json"
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    if mutation == "per_query":
        trace["per_query"][0]["targets"][0]["entry_points"][0] += 10000
        expected = "metrics do not match"
    else:
        trace["aggregate"]["tampered"] = True
        expected = "aggregate does not match"
    trace_path.write_text(json.dumps(trace), encoding="utf-8")

    with pytest.raises(RuntimeError, match=expected):
        module.load_case_result(matrix_dir, case)


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ("format", "unsupported Orion route trace format_version"),
        ("artifact", "artifact provenance mismatch"),
        ("query", "query provenance mismatch"),
    ],
)
def test_collect_rejects_orion_route_trace_provenance_tampering(
    tmp_path, mutation, expected
):
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
    trace_path = matrix_dir / "cases" / case["name"] / "orion_route_trace.json"
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    if mutation == "format":
        trace["format_version"] = 2
    elif mutation == "artifact":
        trace["artifact"]["layout_sha256"] = "0" * 64
    else:
        trace["queries"]["sha256"] = "0" * 64
    trace_path.write_text(json.dumps(trace), encoding="utf-8")

    with pytest.raises(RuntimeError, match=expected):
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
    assert manifest["shared_provenance"]["deployment_manifest_sha256"] == "c" * 64
    assert manifest["shared_provenance"]["orion_compact_wire_version"] == "2"
    assert manifest["shared_provenance"]["image_compact_wire_max_version"] == "2"
    assert manifest["shared_provenance"]["peer_premerge_mode"] == "enabled"
    assert manifest["shared_provenance"]["peer_premerge_shards_per_rpc"] == "4"
    assert re.fullmatch(
        r"[0-9a-f]{64}",
        manifest["shared_provenance"]["transport_identity_sha256"],
    )
    assert manifest["shared_provenance"]["deployment_commit"] == "commit-1"
    assert manifest["shared_provenance"]["benchmark_commit"] == "commit-1"
    assert manifest["shared_provenance"]["deployment_tracked_dirty"] is False
    assert manifest["shared_provenance"]["benchmark_tracked_dirty"] is False
    assert manifest["shared_provenance"]["sharding_method"] == "auto"
    assert manifest["shared_provenance"]["numeric_shard_count"] == 3
    assert manifest["shared_provenance"]["replication_factor"] == 1
    assert manifest["shared_provenance"]["write_consistency_factor"] == 1
    assert manifest["shared_provenance"]["hnsw"] == {
        "m": 32,
        "ef_construct": 100,
        "full_scan_threshold": 10000,
    }
    assert manifest["shared_provenance"]["optimizer"] == {
        "indexing_threshold": 10000,
        "default_segment_number": 0,
    }
    assert manifest["shared_provenance"]["exact_placement"] == {
        "0": 202,
        "1": 303,
        "2": 404,
    }
    assert (matrix_dir / "recall_qps_points.csv").is_file()
    assert (matrix_dir / "pareto_frontier.csv").is_file()
    assert (matrix_dir / "same_recall_selection.csv").is_file()
    assert (matrix_dir / "same_recall_confirmation.csv").is_file()
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

    with (matrix_dir / "same_recall_confirmation.csv").open(
        "r", newline="", encoding="utf-8"
    ) as handle:
        confirmations = list(csv.DictReader(handle))
    assert len(confirmations) == 2
    assert {row["confirmation_status"] for row in confirmations} == {"strict"}
    assert {row["strict_same_recall"] for row in confirmations} == {"True"}
    assert float(confirmations[0]["pairwise_recall_spread"]) == pytest.approx(
        0.002
    )
    assert manifest["require_strict_same_recall"] is False
    assert manifest["same_recall_pairwise_window"] == pytest.approx(0.003)
    assert all(
        row["strict_same_recall"] is True
        for row in manifest["same_recall_confirmation"]
    )


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


def test_provenance_requires_same_clean_tracked_commit_with_dirty_fallback():
    module = load_module()
    manifest = shared_manifest("hash_all")

    fingerprint = module.provenance_fingerprint(manifest)

    assert fingerprint["deployment_commit"] == "commit-1"
    assert fingerprint["benchmark_commit"] == "commit-1"
    assert fingerprint["deployment_tracked_dirty"] is False
    assert fingerprint["benchmark_tracked_dirty"] is False

    mismatched = json.loads(json.dumps(manifest))
    mismatched["repository"]["commit"] = "different"
    with pytest.raises(RuntimeError, match="deployment/benchmark commit mismatch"):
        module.provenance_fingerprint(mismatched)

    dirty = json.loads(json.dumps(manifest))
    dirty["repository"]["dirty"] = True
    with pytest.raises(RuntimeError, match="tracked_dirty=false"):
        module.provenance_fingerprint(dirty)

    dirty_deployment = json.loads(json.dumps(manifest))
    dirty_deployment["deployment"]["repository"]["dirty"] = True
    with pytest.raises(
        RuntimeError, match="deployment repository.*tracked_dirty=false"
    ):
        module.provenance_fingerprint(dirty_deployment)

    explicitly_clean = json.loads(json.dumps(manifest))
    explicitly_clean["repository"]["tracked_dirty"] = False
    explicitly_clean["repository"]["dirty"] = True
    assert (
        module.provenance_fingerprint(explicitly_clean)["benchmark_tracked_dirty"]
        is False
    )


def test_provenance_binds_compact_wire_peer_premerge_and_transport_hash():
    module = load_module()
    manifest = shared_manifest("hash_all")

    fingerprint = module.provenance_fingerprint(manifest)

    assert fingerprint["orion_compact_wire_version"] == "2"
    assert fingerprint["peer_premerge_mode"] == "enabled"
    assert fingerprint["peer_premerge_shards_per_rpc"] == "4"
    assert fingerprint["transport_identity_sha256"] == (
        manifest["deployment"]["transport_identity_sha256"]
    )

    corrupted = json.loads(json.dumps(manifest))
    corrupted["deployment"]["transport_identity_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="transport identity SHA-256 mismatch"):
        module.provenance_fingerprint(corrupted)

    wire_v1 = json.loads(json.dumps(manifest))
    wire = wire_v1["deployment"]["transport_identity"]["orion_compact_wire"]
    wire.update(
        {
            "requested_version": "1",
            "current_version": "1",
            "matches_requested": True,
        }
    )
    wire_v1["deployment"]["orion_compact_wire"] = json.loads(json.dumps(wire))
    wire_v1["deployment"]["transport_identity_sha256"] = module.canonical_sha256(
        wire_v1["deployment"]["transport_identity"]
    )
    assert module.provenance_fingerprint(wire_v1)[
        "transport_identity_sha256"
    ] != fingerprint["transport_identity_sha256"]

    peer_disabled = json.loads(json.dumps(manifest))
    peer = peer_disabled["deployment"]["transport_identity"]["peer_premerge"]
    peer.update(
        {
            "requested_mode": "disabled",
            "current_mode": "disabled",
            "matches_requested": True,
        }
    )
    peer_disabled["deployment"]["peer_premerge"] = json.loads(json.dumps(peer))
    peer_disabled["deployment"]["transport_identity_sha256"] = (
        module.canonical_sha256(
            peer_disabled["deployment"]["transport_identity"]
        )
    )
    assert module.provenance_fingerprint(peer_disabled)[
        "transport_identity_sha256"
    ] != fingerprint["transport_identity_sha256"]


def test_shared_provenance_rejects_same_image_with_mixed_compact_wire():
    module = load_module()
    results = []
    for method in module.METHODS:
        manifest = shared_manifest(method)
        if method == "orion":
            wire = manifest["deployment"]["transport_identity"][
                "orion_compact_wire"
            ]
            wire.update(
                {
                    "requested_version": "1",
                    "current_version": "1",
                    "matches_requested": True,
                }
            )
            manifest["deployment"]["orion_compact_wire"] = json.loads(
                json.dumps(wire)
            )
            manifest["deployment"]["transport_identity_sha256"] = (
                module.canonical_sha256(
                    manifest["deployment"]["transport_identity"]
                )
            )
        results.append(
            {
                "manifest": manifest,
                "point": {"method": method, "case_name": method},
            }
        )

    with pytest.raises(RuntimeError, match="shared provenance mismatch"):
        module.validate_shared_provenance(results)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("status", "yellow", "not green"),
        ("optimizer_status", {"error": "backlog"}, "optimizer is not ready"),
        ("update_queue", {"length": 1}, "update queue is not empty"),
        ("shard_transfers", [{"shard_id": 0}], "active or invalid shard transfers"),
    ],
)
def test_provenance_rejects_non_ready_collection(field, value, message):
    module = load_module()
    manifest = shared_manifest("hash_all")
    manifest["indexing_readiness"][field] = value

    with pytest.raises(RuntimeError, match=message):
        module.provenance_fingerprint(manifest)


def test_provenance_rejects_stable_but_not_fully_indexed_collection():
    module = load_module()
    manifest = shared_manifest("hash_all")
    manifest["indexing_readiness"].update(
        {
            "indexed_vectors_count": 990,
            "fully_indexed": False,
            "completion_mode": "stable_small_segment_full_scan_exception",
        }
    )

    with pytest.raises(RuntimeError, match="requires fully indexed collections"):
        module.provenance_fingerprint(manifest)


def test_routed_profile_family_guard_rejects_offline_structure_drift(tmp_path):
    module = load_module()
    config = module.load_config(write_config(tmp_path / "config"))
    matrix_dir = module.matrix_directory(tmp_path, "family-guard", must_exist=False)
    for case in config["cases"]:
        write_case_result(
            matrix_dir,
            case,
            recall=0.9,
            qps=100.0,
            visited=2 if case["method"] == "orion" else 3,
            ef_sum=80,
        )
    results = [module.load_case_result(matrix_dir, case) for case in config["cases"]]
    families = module.validate_routed_profile_families(results)
    assert families["orion"]["layout_sha256"] == "a" * 64
    assert families["simple_kmeans"]["routing_structure_sha256"] == "d" * 64

    diagnostic = json.loads(json.dumps(results[1]))
    diagnostic["manifest"]["artifact_bundle"]["formal_evidence_eligible"] = False
    with pytest.raises(RuntimeError, match="not eligible for formal evidence"):
        module.validate_routed_profile_families(
            [results[0], diagnostic, results[2]]
        )

    second_orion = dict(config["cases"][1], name="orion95")
    write_case_result(
        matrix_dir,
        second_orion,
        recall=0.95,
        qps=80.0,
        visited=3,
        ef_sum=160,
    )
    drifted = module.load_case_result(matrix_dir, second_orion)
    drifted["manifest"]["artifact"]["routing_structure_sha256"] = "e" * 64

    with pytest.raises(RuntimeError, match="do not share one offline routing structure"):
        module.validate_routed_profile_families([*results, drifted])


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("params", "sharding_method", "custom"),
        ("params", "shard_number", 4),
        ("params", "replication_factor", 2),
        ("params", "write_consistency_factor", 2),
        ("hnsw_config", "m", 16),
        ("hnsw_config", "ef_construct", 200),
        ("hnsw_config", "full_scan_threshold", 5000),
        ("optimizer_config", "indexing_threshold", 20000),
        ("optimizer_config", "default_segment_number", 2),
        ("placement", "2", 202),
    ],
)
def test_shared_fingerprint_rejects_collection_or_placement_drift(
    section, field, value
):
    module = load_module()
    results = []
    for index, method in enumerate(module.METHODS):
        manifest = shared_manifest(method)
        if index == 2:
            if section == "placement":
                manifest["placement_proof"]["placement"][int(field)] = value
            else:
                manifest["collection_info"]["config"][section][field] = value
            if section == "params" and field == "shard_number":
                manifest["collection_cluster"]["shard_count"] = value
                manifest["placement_proof"]["shard_count"] = value
                manifest["placement_proof"]["placement"][3] = 202
            if section == "params" and field == "replication_factor":
                manifest["placement_proof"]["replication_factor"] = value
        results.append(
            {
                "point": {"method": method, "case_name": f"case-{method}"},
                "manifest": manifest,
            }
        )

    expected_error = (
        "not auto"
        if section == "params" and field == "sharding_method"
        else "shared provenance mismatch"
    )
    with pytest.raises(RuntimeError, match=expected_error):
        module.validate_shared_provenance(results)


@pytest.mark.parametrize(
    ("target", "selection_window", "pairwise_window", "recalls", "status"),
    [
        (0.95, 0.003, 0.003, [0.899, 0.900, 0.901], "nearest_selection"),
        (
            0.90,
            0.010,
            0.003,
            [0.894, 0.900, 0.906],
            "pairwise_spread_exceeded",
        ),
    ],
)
def test_collect_strict_same_recall_rejects_nearest_or_pairwise_spread(
    tmp_path, target, selection_window, pairwise_window, recalls, status
):
    module = load_module()
    value = base_config()
    value["same_recall_targets"] = [target]
    value["same_recall_window"] = selection_window
    value["same_recall_pairwise_window"] = pairwise_window
    value["require_strict_same_recall"] = True
    config = module.load_config(write_config(tmp_path, value))
    matrix_dir = tmp_path / "matrix"
    matrix_dir.mkdir()
    for case, recall in zip(config["cases"], recalls):
        write_case_result(
            matrix_dir,
            case,
            recall,
            100.0,
            visited=1.5 if case["method"] == "orion" else 3,
            ef_sum=54 if case["method"] == "orion" else 120,
        )

    with pytest.raises(RuntimeError, match="strict same-recall confirmation failed"):
        module.collect_results(
            matrix_dir,
            config,
            case_records=None,
            run_id="native-run",
            mode="collect_only",
            taskset_cpus=None,
        )

    with (matrix_dir / "same_recall_confirmation.csv").open(
        "r", newline="", encoding="utf-8"
    ) as handle:
        confirmation = next(csv.DictReader(handle))
    assert confirmation["strict_same_recall"] == "False"
    assert confirmation["confirmation_status"] == status
    assert not (matrix_dir / "run_manifest.json").exists()


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
    assert config["require_strict_same_recall"] is False
    assert config["same_recall_pairwise_window"] == pytest.approx(0.003)
