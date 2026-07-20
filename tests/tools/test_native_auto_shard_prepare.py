from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
TOPOLOGY = REPO_ROOT / "tools/distributed/cloudlab_orion_4node.json"


def load_module():
    path = REPO_ROOT / "tools/native_auto_shard_prepare.py"
    spec = importlib.util.spec_from_file_location("native_auto_shard_prepare", path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_simple_runtime_profile_module():
    path = REPO_ROOT / "tools/simple_kmeans_native_runtime_profile.py"
    spec = importlib.util.spec_from_file_location(
        "simple_kmeans_native_runtime_profile_for_prepare_test", path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def args_for(module, tmp_path, method, *extra):
    values = [
        "--method",
        method,
        "--topology",
        str(TOPOLOGY),
        "--run-id",
        "prepare-test",
        "--collection",
        f"native_{method}",
        "--base-url",
        "http://10.10.1.1:6333",
        "--output-dir",
        str(tmp_path / f"proof-{method}"),
        "--transfer-poll-interval-secs",
        "0",
    ]
    if method == "hash_all":
        values.extend(
            [
                "--hdf5-path",
                str(tmp_path / "glove.hdf5"),
                "--p",
                "4",
            ]
        )
    else:
        values.extend(["--layout-dir", str(tmp_path / f"layout-{method}")])
    values.extend(extra)
    return module.parse_args(values)


def collection_info(method, points_count, policy=None, metadata=None):
    return {
        "status": "green",
        "optimizer_status": "ok",
        "points_count": points_count,
        "config": {
            "params": {
                "vectors": {
                    "size": 2,
                    "distance": "Cosine",
                    "datatype": "float32",
                },
                "sharding_method": "auto",
                "shard_number": 4,
                "replication_factor": 1,
                "write_consistency_factor": 1,
            },
            "hnsw_config": {
                "m": 32,
                "ef_construct": 100,
                "full_scan_threshold": 10,
            },
            "optimizer_config": {"indexing_threshold": 10},
            **({"auto_shard_policy": policy} if policy is not None else {}),
            **({"metadata": metadata} if metadata is not None else {}),
        },
    }


def test_optional_collection_info_treats_http_404_as_absent(monkeypatch):
    module = load_module()

    def missing_collection(*_args):
        raise RuntimeError(
            "GET http://10.10.1.1:6333/collections/missing "
            "failed (HTTP 404): collection does not exist"
        )

    monkeypatch.setattr(module.experiment, "collection_info", missing_collection)

    assert module.optional_collection_info("http://10.10.1.1:6333", "missing") is None


def test_optional_collection_info_does_not_hide_non_404_errors(monkeypatch):
    module = load_module()

    def unavailable_collection(*_args):
        raise RuntimeError(
            "GET http://10.10.1.1:6333/collections/example "
            "failed (HTTP 503): upstream payload mentioned 404 but is unavailable"
        )

    monkeypatch.setattr(module.experiment, "collection_info", unavailable_collection)

    with pytest.raises(RuntimeError, match="HTTP 503"):
        module.optional_collection_info("http://10.10.1.1:6333", "example")


def test_faithful_orion_build_parameters_bind_offline_and_runtime_semantics():
    module = load_module()
    parameters = {
        "initial_num_shards": 31,
        "sample_denominator": 32,
        "upper_sample_seed": 100,
        "upper_m": 32,
        "upper_ef_construction": 100,
        "attachment_search_ef": 100,
        "upper_k": 36,
        "upper_search_ef": 36,
        "dynamic_ef_base": 48,
        "dynamic_ef_factor": 15,
        "k_overlap": 10,
        "kmeans_iters": 10,
        "kmeans_seed": 1,
        "topology_iters": 50,
        "use_multi_assign": True,
        "multi_assign_min_max_vote": 2,
        "multi_assign_vote_delta": 0,
        "multi_assign_max_shards": 0,
        "enable_fission": True,
        "upper_graph_seed": 100,
        "allow_decoupled_runtime_upper_search": False,
    }
    artifact = {
        "upper_k": 36,
        "upper_ef_search": 36,
        "dynamic_ef_base": 48,
        "dynamic_ef_factor": 15,
    }

    assert module.validate_faithful_orion_build_parameters(parameters, artifact) == 100

    invalid_attachment = dict(parameters, attachment_search_ef=99)
    with pytest.raises(RuntimeError, match="attachment_search_ef must be 100"):
        module.validate_faithful_orion_build_parameters(
            invalid_attachment, artifact
        )

    decoupled_runtime = dict(parameters, upper_search_ef=100)
    decoupled_artifact = dict(artifact, upper_ef_search=100)
    with pytest.raises(RuntimeError, match="must equal upper_k"):
        module.validate_faithful_orion_build_parameters(
            decoupled_runtime, decoupled_artifact
        )

    mismatched_artifact = dict(artifact, dynamic_ef_factor=16)
    with pytest.raises(RuntimeError, match="runtime artifact mismatch"):
        module.validate_faithful_orion_build_parameters(
            parameters, mismatched_artifact
        )

    for field, non_faithful_value in (
        ("initial_num_shards", 46),
        ("sample_denominator", 16),
        ("upper_sample_seed", 99),
        ("upper_m", 16),
        ("upper_ef_construction", 200),
        ("k_overlap", 8),
        ("kmeans_iters", 20),
        ("kmeans_seed", 7),
        ("topology_iters", 25),
        ("use_multi_assign", False),
        ("multi_assign_min_max_vote", 3),
        ("multi_assign_vote_delta", 1),
        ("multi_assign_max_shards", 2),
        ("enable_fission", False),
        ("upper_graph_seed", 101),
        ("allow_decoupled_runtime_upper_search", True),
    ):
        with pytest.raises(RuntimeError, match="main-idea parameter drift"):
            module.validate_faithful_orion_build_parameters(
                dict(parameters, **{field: non_faithful_value}), artifact
            )


def patch_common_cluster(module, monkeypatch, tmp_path):
    deployment_path = tmp_path / "deployment-manifest.json"
    deployment_path.write_text('{"run_id":"prepare-test"}\n', encoding="utf-8")
    monkeypatch.setattr(
        module.cluster_tool,
        "read_manifest",
        lambda *_args: {"run_id": "prepare-test", "image": {"id": "sha256:image"}},
    )
    monkeypatch.setattr(
        module.cluster_tool,
        "manifest_path",
        lambda *_args: deployment_path,
    )
    monkeypatch.setattr(
        module.experiment,
        "validate_cluster_preflight",
        lambda *_args: {
            "peer_id": 101,
            "peer_count": 4,
            "controller_peer_id": 101,
            "worker_peer_ids": [202, 303, 404],
            "peers": {
                "101": "http://10.10.1.1:6335",
                "202": "http://10.10.1.2:6335",
                "303": "http://10.10.1.3:6335",
                "404": "http://10.10.1.4:6335",
            },
            "raw": {"not": "persisted"},
        },
    )
    monkeypatch.setattr(
        module.experiment,
        "move_numeric_shards_round_robin",
        lambda *_args, **_kwargs: {
            "valid": True,
            "placement": {0: 202, 1: 303, 2: 404, 3: 202},
            "moves": [],
        },
    )
    monkeypatch.setattr(
        module.experiment,
        "validate_numeric_shard_round_robin_placement",
        lambda *_args, **_kwargs: {
            "valid": True,
            "placement": {0: 202, 1: 303, 2: 404, 3: 202},
        },
    )
    monkeypatch.setattr(
        module.experiment,
        "collection_cluster_info",
        lambda *_args: {"peer_id": 101, "shard_count": 4},
    )
    monkeypatch.setattr(
        module.experiment,
        "wait_collection_indexed",
        lambda *_args, **_kwargs: {
            "status": "green",
            "optimizer_status": "ok",
            "points_count": 0,
            "indexed_vectors_count": 0,
            "segments_count": 0,
        },
    )
    monkeypatch.setattr(
        module,
        "run_standard_api_smoke",
        lambda *_args, **_kwargs: {
            "standard_request_contract": True,
            "search": {"external_ids": [0], "external_ids_unique": True},
            "query": {"external_ids": [0], "external_ids_unique": True},
        },
    )


def test_hash_all_prepare_creates_places_and_publicly_upserts(monkeypatch, tmp_path):
    module = load_module()
    args = args_for(module, tmp_path, "hash_all", "--batch-size", "2")
    patch_common_cluster(module, monkeypatch, tmp_path)
    train = module.experiment.np.asarray(
        [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]],
        dtype=module.experiment.np.float32,
    )
    monkeypatch.setattr(
        module.layout_common,
        "load_train_vectors",
        lambda *_args: (
            train,
            {"path": "/dataset.hdf5", "sha256": "d" * 64, "dimension": 2},
        ),
    )
    monkeypatch.setattr(module, "optional_collection_info", lambda *_args: None)
    readiness_waits = []
    monkeypatch.setattr(
        module.experiment,
        "wait_collection_indexed",
        lambda _base_url, _collection, expected_points: readiness_waits.append(
            expected_points
        )
        or {
            "status": "green",
            "optimizer_status": "ok",
            "points_count": expected_points,
            "indexed_vectors_count": expected_points,
            "segments_count": 1 if expected_points else 0,
        },
    )
    created = []
    monkeypatch.setattr(
        module.experiment,
        "create_numeric_auto_shard_collection",
        lambda *call_args, **kwargs: created.append((call_args, kwargs)) or {"result": True},
    )
    provenance = module.build_provenance_metadata(
        method="hash_all",
        schema={
            "vector_name": "",
            "dimension": 2,
            "distance": "Cosine",
            "datatype": "float32",
        },
        shard_count=4,
        logical_point_count=3,
        physical_point_count=3,
        dataset_proof={"sha256": "d" * 64},
        layout=None,
    )
    infos = iter(
        [
            collection_info("hash_all", 0, metadata=provenance),
            collection_info("hash_all", 3, metadata=provenance),
            collection_info("hash_all", 3, metadata=provenance),
        ]
    )
    monkeypatch.setattr(module.experiment, "collection_info", lambda *_args: next(infos))
    upserts = []
    monkeypatch.setattr(
        module.experiment,
        "upsert_numeric_auto_points",
        lambda *call_args, **kwargs: upserts.append((call_args, kwargs))
        or {"point_count": 3, "batch_count": 2, "uses_shard_key": False},
    )
    monkeypatch.setattr(
        module,
        "run_command",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("HashAll must not run Cargo or artifact installer")
        ),
    )

    manifest_path = module.prepare(args)

    assert created[0][1]["auto_shard_policy"] is None
    assert created[0][1]["replication_factor"] == 1
    assert created[0][1]["write_consistency_factor"] == 1
    assert created[0][1]["metadata"] == provenance
    assert upserts[0][1]["batch_size"] == 2
    assert upserts[0][1]["vector_name"] == ""
    assert readiness_waits == [0, 3]
    manifest = json.loads(manifest_path.read_text())
    assert manifest["method"] == "hash_all"
    assert manifest["created_collection"] is True
    assert manifest["physical_point_count"] == 3
    assert manifest["commands"][0]["kind"] == "public_hash_all_upsert"
    assert manifest["commands"][0]["proof"]["uses_shard_key"] is False
    assert manifest["checksums"]["dataset_sha256"] == "d" * 64
    assert manifest["initial_readiness"]["points_count"] == 0
    assert manifest["indexing_readiness"]["indexed_vectors_count"] == 3
    assert manifest["standard_api_smoke"]["standard_request_contract"] is True
    assert manifest["provenance_metadata"] == provenance
    assert "raw" not in manifest["cluster_preflight"]


@pytest.mark.parametrize(
    ("method", "installer"),
    [
        ("orion", "install-orion-artifact"),
        ("simple_kmeans", "install-simple-kmeans-artifact"),
    ],
)
def test_routed_prepare_runs_importer_and_matching_installer(
    monkeypatch,
    tmp_path,
    method,
    installer,
):
    module = load_module()
    args = args_for(module, tmp_path, method, "--resume")
    patch_common_cluster(module, monkeypatch, tmp_path)
    layout_dir = Path(args.layout_dir)
    layout_dir.mkdir()
    artifact = layout_dir / "generation-7.json"
    import_manifest = layout_dir / "numeric.manifest.json"
    artifact.write_text("{}", encoding="utf-8")
    import_manifest.write_text("{}", encoding="utf-8")
    layout = {
        "layout_dir": str(layout_dir),
        "artifact_path": str(artifact),
        "artifact_sha256": "a" * 64,
        "artifact": {},
        "import_manifest_path": str(import_manifest),
        "generation": 7,
        "vector_schema": {
            "vector_name": "",
            "dimension": 2,
            "distance": "Cosine",
            "datatype": "float32",
        },
        "shard_count": 4,
        "logical_point_count": 3,
        "physical_point_count": 3,
        "attachment_search_ef": 100,
        "smoke_vector": [1.0, 0.0],
        "checksums": {"generation-7.json": "a" * 64},
    }
    layout["artifact"] = {"layout_sha256": "b" * 64}
    monkeypatch.setattr(module, "load_routed_layout", lambda *_args: layout)
    monkeypatch.setattr(module, "optional_collection_info", lambda *_args: None)
    policy = {
        "type": method,
        "generation": 7,
        "artifact_sha256": "a" * 64,
    }
    created = []
    monkeypatch.setattr(
        module.experiment,
        "create_numeric_auto_shard_collection",
        lambda *call_args, **kwargs: created.append(kwargs) or {"result": True},
    )
    provenance = module.build_provenance_metadata(
        method=method,
        schema=layout["vector_schema"],
        shard_count=4,
        logical_point_count=3,
        physical_point_count=3,
        dataset_proof=None,
        layout=layout,
    )
    infos = iter(
        [
            collection_info(method, 0, policy, provenance),
            collection_info(method, 3, policy, provenance),
            collection_info(method, 3, policy, provenance),
        ]
    )
    monkeypatch.setattr(module.experiment, "collection_info", lambda *_args: next(infos))
    commands = []

    def fake_run(command, env=None):
        commands.append((command, env))
        return {"command": command, "returncode": 0, "stdout": "{}", "stderr": ""}

    monkeypatch.setattr(module, "run_command", fake_run)

    manifest_path = module.prepare(args)

    assert created[0]["auto_shard_policy"] == policy
    importer_command = commands[0][0]
    assert importer_command[0].endswith("tools/cargo_in_docker.sh")
    assert "orion_numeric_shard_import" in importer_command
    assert "--resume" in importer_command
    installer_command = commands[1][0]
    assert installer in installer_command
    assert installer_command[-2:] == ["--restart", "workers-first"]
    manifest = json.loads(manifest_path.read_text())
    assert manifest["method"] == method
    assert len(manifest["commands"]) == 2
    assert manifest["checksums"]["routing_artifact_sha256"] == "a" * 64
    assert manifest["final_collection_proof"]["points_count"] == 3
    assert manifest["provenance_metadata"] == provenance
    envelope = provenance[module.PROVENANCE_METADATA_KEY]
    assert envelope["schema_version"] == 2
    if method == "orion":
        assert envelope["provenance"]["routing"]["attachment_search_ef"] == 100


def test_validate_collection_configuration_rejects_hnsw_policy_and_count_drift():
    module = load_module()
    policy = {"type": "orion", "generation": 1, "artifact_sha256": "a" * 64}
    provenance = {
        module.PROVENANCE_METADATA_KEY: {
            "schema_version": 1,
            "provenance": {"test": True},
            "provenance_sha256": "a" * 64,
        }
    }
    info = collection_info("orion", 2, policy, provenance)
    info["config"]["hnsw_config"]["m"] = 16

    with pytest.raises(RuntimeError, match="refusing to reuse collection") as error:
        module.validate_collection_configuration(
            info,
            method="orion",
            expected_schema={
                "vector_name": "",
                "dimension": 2,
                "distance": "Cosine",
                "datatype": "float32",
            },
            expected_shard_count=4,
            expected_policy=policy,
            expected_metadata=provenance,
            hnsw_m=32,
            ef_construct=100,
            full_scan_threshold=10,
            indexing_threshold=10,
            expected_point_count=3,
            allow_empty=True,
            allow_partial=False,
        )

    assert "hnsw.m=16" in str(error.value)
    assert "points_count=2" in str(error.value)


def test_existing_collection_without_exact_provenance_is_rejected():
    module = load_module()
    policy = {"type": "orion", "generation": 1, "artifact_sha256": "a" * 64}
    expected = {
        module.PROVENANCE_METADATA_KEY: {
            "schema_version": 1,
            "provenance": {"method": "orion"},
            "provenance_sha256": "b" * 64,
        }
    }
    info = collection_info("orion", 3, policy, metadata=None)

    with pytest.raises(RuntimeError, match="provenance metadata is missing"):
        module.validate_collection_configuration(
            info,
            method="orion",
            expected_schema={
                "vector_name": "",
                "dimension": 2,
                "distance": "Cosine",
                "datatype": "float32",
            },
            expected_shard_count=4,
            expected_policy=policy,
            expected_metadata=expected,
            hnsw_m=32,
            ef_construct=100,
            full_scan_threshold=10,
            indexing_threshold=10,
            expected_point_count=3,
            allow_empty=False,
            allow_partial=False,
        )


def test_existing_partial_routed_collection_can_resume_and_converge_placement(
    monkeypatch,
    tmp_path,
):
    module = load_module()
    args = args_for(module, tmp_path, "orion", "--resume")
    patch_common_cluster(module, monkeypatch, tmp_path)
    layout_dir = Path(args.layout_dir)
    layout_dir.mkdir()
    artifact = layout_dir / "generation-3.json"
    import_manifest = layout_dir / "numeric.manifest.json"
    artifact.write_text("{}", encoding="utf-8")
    import_manifest.write_text("{}", encoding="utf-8")
    policy = {"type": "orion", "generation": 3, "artifact_sha256": "c" * 64}
    layout = {
        "layout_dir": str(layout_dir),
        "artifact_path": str(artifact),
        "artifact_sha256": "c" * 64,
        "artifact": {},
        "import_manifest_path": str(import_manifest),
        "generation": 3,
        "vector_schema": {
            "vector_name": "",
            "dimension": 2,
            "distance": "Cosine",
            "datatype": "float32",
        },
        "shard_count": 4,
        "logical_point_count": 3,
        "physical_point_count": 3,
        "attachment_search_ef": 100,
        "smoke_vector": [1.0, 0.0],
        "checksums": {},
    }
    layout["artifact"] = {"layout_sha256": "e" * 64}
    monkeypatch.setattr(module, "load_routed_layout", lambda *_args: layout)
    provenance = module.build_provenance_metadata(
        method="orion",
        schema=layout["vector_schema"],
        shard_count=4,
        logical_point_count=3,
        physical_point_count=3,
        dataset_proof=None,
        layout=layout,
    )
    monkeypatch.setattr(
        module,
        "optional_collection_info",
        lambda *_args: collection_info("orion", 1, policy, provenance),
    )
    existing_cluster = {
        "peer_id": 101,
        "shard_count": 4,
        "local_shards": [{"shard_id": 0, "state": "Active"}],
        "remote_shards": [
            {"shard_id": 1, "peer_id": 202, "state": "Active"},
            {"shard_id": 2, "peer_id": 202, "state": "Active"},
            {"shard_id": 3, "peer_id": 404, "state": "Active"},
        ],
        "shard_transfers": [],
    }
    cluster_calls = iter(
        [
            existing_cluster,
            {"peer_id": 101, "shard_count": 4},
        ]
    )
    monkeypatch.setattr(
        module.experiment,
        "collection_cluster_info",
        lambda *_args: next(cluster_calls),
    )
    infos = iter(
        [
            collection_info("orion", 3, policy, provenance),
            collection_info("orion", 3, policy, provenance),
        ]
    )
    monkeypatch.setattr(module.experiment, "collection_info", lambda *_args: next(infos))
    monkeypatch.setattr(
        module.experiment,
        "create_numeric_auto_shard_collection",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("existing collection must not be recreated")
        ),
    )
    commands = []
    monkeypatch.setattr(
        module,
        "run_command",
        lambda command, env=None: commands.append(command)
        or {"command": command, "returncode": 0, "stdout": "", "stderr": ""},
    )

    manifest_path = module.prepare(args)

    assert "--resume" in commands[0]
    manifest = json.loads(manifest_path.read_text())
    assert manifest["created_collection"] is False
    assert manifest["initial_collection_proof"]["points_count"] == 1
    assert manifest["initial_placement_proof"]["placement"]["0"] == 101


def test_load_simple_layout_validates_build_artifact_and_import_binding(tmp_path):
    module = load_module()
    layout_dir = tmp_path / "layout"
    layout_dir.mkdir()
    train = module.experiment.np.asarray(
        [[1.0, 0.0], [0.0, 1.0]], dtype=module.experiment.np.float32
    )
    point_to_shards = [[0], [1]]
    artifact = module.experiment.write_simple_kmeans_graphless_artifact(
        train,
        train.copy(),
        point_to_shards,
        2,
        layout_dir / "generation-2.json",
        generation=2,
        vector_distance="cosine",
        nprobe=1,
        lower_hnsw_ef=32,
    )
    Path(f"{artifact}.sha256").write_text(
        module.layout_common.sha256_path(artifact) + "\n", encoding="utf-8"
    )
    import_manifest = module.experiment.write_numeric_shard_import_bundle_v2(
        train,
        point_to_shards,
        2,
        layout_dir,
        routing_policy="simple_kmeans",
        routing_generation=2,
        routing_artifact_path=artifact,
        prefix="numeric",
    )
    build_manifest = {
        "format_version": 1,
        "tool": "tools/simple_kmeans_native_layout.py",
        "mode": "production_bundle",
        "outputs": {
            "production_artifact": artifact.name,
            "import_manifest": import_manifest.name,
            "files": {
                artifact.name: {
                    "sha256": module.layout_common.sha256_path(artifact),
                    "size_bytes": artifact.stat().st_size,
                },
                import_manifest.name: {
                    "sha256": module.layout_common.sha256_path(import_manifest),
                    "size_bytes": import_manifest.stat().st_size,
                },
            },
        },
    }
    module.layout_common.write_json_new(
        layout_dir / module.layout_common.BUILD_MANIFEST_NAME,
        build_manifest,
    )
    module.layout_common.write_checksums(layout_dir)

    proof = module.load_routed_layout("simple_kmeans", layout_dir)

    assert proof["generation"] == 2
    assert proof["shard_count"] == 2
    assert proof["logical_point_count"] == proof["physical_point_count"] == 2
    assert proof["artifact_sha256"] == module.layout_common.sha256_path(artifact)
    assert proof["smoke_vector"] == [1.0, 0.0]


def build_simple_runtime_profile_bundle(module, monkeypatch, tmp_path):
    runtime = load_simple_runtime_profile_module()
    source_dir = tmp_path / "simple-source"
    source_dir.mkdir()
    train = module.experiment.np.asarray(
        [[1.0, 0.0], [0.0, 1.0]], dtype=module.experiment.np.float32
    )
    point_to_shards = [[0], [1]]
    graphless = module.experiment.write_simple_kmeans_graphless_artifact(
        train,
        train.copy(),
        point_to_shards,
        2,
        source_dir / "simple-kmeans-graphless.json",
        generation=1,
        vector_distance="cosine",
        nprobe=1,
        lower_hnsw_ef=32,
    )
    artifact = source_dir / "generation-1.json"
    artifact.write_bytes(graphless.read_bytes())
    import_manifest = module.experiment.write_numeric_shard_import_bundle_v2(
        train,
        point_to_shards,
        2,
        source_dir,
        routing_policy="simple_kmeans",
        routing_generation=1,
        routing_artifact_path=artifact,
        prefix="numeric",
    )
    artifact_payload = json.loads(artifact.read_text())
    parameters = {
        "generation": 1,
        "num_shards": 2,
        "vector_distance": "cosine",
        "vector_name": "",
        "routing_distance": "squared_l2",
        "nprobe": 1,
        "lower_hnsw_ef": 32,
        "kmeans_train_size": 2,
        "kmeans_iters": 3,
        "kmeans_seed": 7,
        "cargo_target_dir": None,
    }
    routing = {
        "policy": "simple_kmeans",
        "logical_point_count": 2,
        "physical_point_count": 2,
        "expansion_ratio": 1.0,
        "shard_counts": [1, 1],
    }
    source_files = module.layout_common.relative_file_records(
        source_dir,
        {
            module.layout_common.BUILD_MANIFEST_NAME,
            module.layout_common.CHECKSUMS_NAME,
        },
    )
    source_build_manifest = {
        "format_version": 1,
        "tool": "tools/simple_kmeans_native_layout.py",
        "mode": "production_bundle",
        "dataset": {
            "path": "/data/test.hdf5",
            "size_bytes": 1,
            "sha256": "d" * 64,
            "train_rows_total": 2,
            "train_rows_used": 2,
            "dimension": 2,
        },
        "parameters": parameters,
        "artifact_binding": {
            "format_version": 1,
            "generation": 1,
            "shard_count": 2,
            "logical_point_count": 2,
            "physical_point_count": 2,
            "routing_distance": "squared_l2",
            "nprobe": 1,
            "lower_hnsw_ef": 32,
            "layout_sha256": artifact_payload["layout_sha256"],
        },
        "routing": routing,
        "outputs": {
            "graphless_artifact": graphless.name,
            "production_artifact": artifact.name,
            "import_manifest": import_manifest.name,
            "files": source_files,
        },
    }
    module.layout_common.write_json_new(
        source_dir / module.layout_common.BUILD_MANIFEST_NAME,
        source_build_manifest,
    )
    module.layout_common.write_checksums(source_dir)

    def fake_builder(_args, graphless_path, production_path):
        production_path.write_bytes(graphless_path.read_bytes())
        Path(f"{production_path}.sha256").write_text(
            module.layout_common.sha256_path(production_path) + "\n",
            encoding="utf-8",
        )
        return ["mock-cargo", "simple_kmeans_build_artifact"]

    monkeypatch.setattr(runtime.simple_layout, "run_rust_builder", fake_builder)
    derived_dir = tmp_path / "simple-derived"
    runtime.build(
        runtime.parse_args(
            [
                "--source-layout-dir",
                str(source_dir),
                "--output-dir",
                str(derived_dir),
                "--generation",
                "2",
                "--nprobe",
                "2",
                "--lower-hnsw-ef",
                "96",
            ]
        )
    )
    return derived_dir


def rewrite_layout_checksums(module, layout_dir):
    (layout_dir / module.layout_common.CHECKSUMS_NAME).unlink()
    module.layout_common.write_checksums(layout_dir)


def test_load_simple_runtime_profile_accepts_one_offline_layout(
    monkeypatch, tmp_path
):
    module = load_module()
    derived_dir = build_simple_runtime_profile_bundle(module, monkeypatch, tmp_path)

    proof = module.load_routed_layout("simple_kmeans", derived_dir)

    assert proof["generation"] == 2
    assert len(proof["artifact"]["layout_sha256"]) == 64
    artifact = json.loads(Path(proof["artifact_path"]).read_text())
    assert artifact["nprobe"] == 2
    assert artifact["lower_hnsw_ef"] == 96
    build_manifest = json.loads(
        (derived_dir / module.layout_common.BUILD_MANIFEST_NAME).read_text()
    )
    assert build_manifest["derivation"]["formal_evidence_eligible"] is True


def test_load_simple_runtime_profile_rejects_offline_parameter_drift(
    monkeypatch, tmp_path
):
    module = load_module()
    derived_dir = build_simple_runtime_profile_bundle(module, monkeypatch, tmp_path)
    manifest_path = derived_dir / module.layout_common.BUILD_MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text())
    manifest["parameters"]["kmeans_seed"] = 99
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    rewrite_layout_checksums(module, derived_dir)

    with pytest.raises(RuntimeError, match="offline KMeans parameters"):
        module.load_routed_layout("simple_kmeans", derived_dir)


def test_load_simple_runtime_profile_rejects_formal_eligibility_lie(
    monkeypatch, tmp_path
):
    module = load_module()
    derived_dir = build_simple_runtime_profile_bundle(module, monkeypatch, tmp_path)
    manifest_path = derived_dir / module.layout_common.BUILD_MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text())
    manifest["derivation"]["formal_evidence_eligible"] = False
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    rewrite_layout_checksums(module, derived_dir)

    with pytest.raises(RuntimeError, match="formal evidence eligibility"):
        module.load_routed_layout("simple_kmeans", derived_dir)


def test_simple_runtime_profile_rejects_centroid_or_payload_proof_drift(
    monkeypatch, tmp_path
):
    module = load_module()
    derived_dir = build_simple_runtime_profile_bundle(module, monkeypatch, tmp_path)
    manifest_path = derived_dir / module.layout_common.BUILD_MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text())
    artifact_path = derived_dir / manifest["outputs"]["production_artifact"]
    artifact = json.loads(artifact_path.read_text())
    artifact["centroids"][0]["vector"][0] = 0.5

    with pytest.raises(RuntimeError, match="centroid/offline artifact"):
        module.validate_simple_kmeans_runtime_profile_derivation(
            manifest,
            manifest["parameters"],
            artifact,
        )

    manifest["derivation"]["reused_payloads"]["vectors"]["sha256"] = "e" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    rewrite_layout_checksums(module, derived_dir)
    with pytest.raises(RuntimeError, match="vectors reuse proof mismatch"):
        module.load_routed_layout("simple_kmeans", derived_dir)


def test_standard_search_and_query_smoke_uses_no_routing_hints(monkeypatch):
    module = load_module()
    calls = []

    def fake_request_json(base_url, method, path, body=None, timeout=300.0):
        calls.append(
            {
                "base_url": base_url,
                "method": method,
                "path": path,
                "body": body,
                "timeout": timeout,
            }
        )
        if path.endswith("/points/search"):
            return {"result": [{"id": 7, "score": 1.0}, {"id": 8, "score": 0.9}]}
        return {
            "result": {
                "points": [{"id": 7, "score": 1.0}, {"id": 9, "score": 0.8}]
            }
        }

    monkeypatch.setattr(module.experiment, "request_json", fake_request_json)

    proof = module.run_standard_api_smoke(
        "http://10.10.1.1:6333",
        "native simple",
        [1.0, 0.0],
        vector_name="embedding",
        limit=2,
        timeout=30.0,
    )

    assert [call["method"] for call in calls] == ["POST", "POST"]
    assert calls[0]["path"] == "/collections/native%20simple/points/search"
    assert calls[1]["path"] == "/collections/native%20simple/points/query"
    assert calls[0]["body"]["vector"] == {
        "name": "embedding",
        "vector": [1.0, 0.0],
    }
    assert calls[1]["body"]["query"] == [1.0, 0.0]
    assert calls[1]["body"]["using"] == "embedding"
    forbidden = set(proof["forbidden_request_fields"])
    assert all(forbidden.isdisjoint(call["body"]) for call in calls)
    assert proof["search"]["external_ids"] == [7, 8]
    assert proof["query"]["external_ids"] == [7, 9]
    assert proof["search"]["external_ids_unique"] is True
    assert proof["query"]["external_ids_unique"] is True


def test_standard_api_smoke_rejects_duplicate_external_ids(monkeypatch):
    module = load_module()
    monkeypatch.setattr(
        module.experiment,
        "request_json",
        lambda *_args, **_kwargs: {
            "result": [{"id": 1, "score": 1.0}, {"id": 1, "score": 0.9}]
        },
    )

    with pytest.raises(RuntimeError, match="duplicate external IDs"):
        module.run_standard_api_smoke(
            "http://10.10.1.1:6333",
            "native",
            [1.0, 0.0],
            vector_name="",
            limit=2,
            timeout=30.0,
        )


def test_preparation_output_must_be_new_and_outside_repository(tmp_path):
    module = load_module()
    existing = tmp_path / "existing"
    existing.mkdir()

    with pytest.raises(FileExistsError):
        module.create_output_directory(existing)
    with pytest.raises(ValueError, match="outside the repository"):
        module.create_output_directory(REPO_ROOT / "results" / "prepare-proof")
