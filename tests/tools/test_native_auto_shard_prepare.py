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


def test_prebuilt_importer_command_skips_cargo(tmp_path):
    module = load_module()
    importer = tmp_path / "orion_numeric_shard_import"
    importer.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    importer.chmod(0o755)
    args = args_for(
        module,
        tmp_path,
        "orion",
        "--importer-binary",
        str(importer),
        "--resume",
    )
    module.validate_args(args)
    command = module.importer_command(
        args,
        {"ports": {"p2p": 6335}, "controller": {"private_ip": "10.10.1.1"}},
        tmp_path / "import.manifest.json",
    )
    assert command[0] == str(importer.resolve())
    assert "cargo" not in " ".join(command)
    assert command[-1] == "--resume"


def test_checkpoint_preservation_rejects_resume_and_hashall(tmp_path):
    module = load_module()
    routed = args_for(
        module,
        tmp_path,
        "orion",
        "--preserve-import-checkpoint",
        "--resume",
    )
    with pytest.raises(ValueError, match="cannot be combined"):
        module.validate_args(routed)

    hashall = args_for(
        module,
        tmp_path,
        "hash_all",
        "--preserve-import-checkpoint",
    )
    with pytest.raises(ValueError, match="only valid for routed methods"):
        module.validate_args(hashall)


def test_snapshot_transfer_method_is_explicitly_selectable(tmp_path):
    module = load_module()
    args = args_for(
        module,
        tmp_path,
        "orion",
        "--transfer-method",
        "snapshot",
    )
    module.validate_args(args)
    assert args.transfer_method == "snapshot"


def test_layout_size_balanced_placement_is_routed_only(tmp_path):
    module = load_module()

    hash_args = args_for(
        module,
        tmp_path,
        "hash_all",
        "--placement-strategy",
        "layout_size_balanced",
    )
    with pytest.raises(ValueError, match="only valid for routed methods"):
        module.validate_args(hash_args)

    routed_args = args_for(
        module,
        tmp_path,
        "orion",
        "--placement-strategy",
        "layout_size_balanced",
    )
    module.validate_args(routed_args)


def test_l1_partition_layout_authorization_is_orion_only_and_not_ccnb(tmp_path):
    module = load_module()

    hash_args = args_for(
        module,
        tmp_path,
        "hash_all",
        "--allow-orion-l1-partition-layout",
    )
    with pytest.raises(ValueError, match="only valid for --method orion"):
        module.validate_args(hash_args)

    combined = args_for(
        module,
        tmp_path,
        "orion",
        "--allow-orion-balance-layout",
        "--allow-orion-l1-partition-layout",
    )
    with pytest.raises(ValueError, match="different layout families"):
        module.validate_args(combined)

    l1_args = args_for(
        module,
        tmp_path,
        "orion",
        "--allow-orion-l1-partition-layout",
    )
    module.validate_args(l1_args)


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

    scaled_parameters = dict(parameters, initial_num_shards=13)
    assert (
        module.validate_faithful_orion_build_parameters(
            scaled_parameters,
            artifact,
            allow_scaling_initial_num_shards=True,
        )
        == 100
    )

    balanced_parameters = dict(
        parameters,
        initial_num_shards=32,
        use_multi_assign=False,
        enable_fission=False,
        balance_mode="capacity_constrained",
    )
    with pytest.raises(RuntimeError, match="main-idea parameter drift"):
        module.validate_faithful_orion_build_parameters(
            balanced_parameters,
            artifact,
            allow_scaling_initial_num_shards=True,
        )
    assert (
        module.validate_faithful_orion_build_parameters(
            balanced_parameters,
            artifact,
            allow_balance_layout=True,
        )
        == 100
    )

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


def test_orion_balance_layout_requires_complete_l0_proof():
    module = load_module()
    parameters = {
        "balance_mode": "capacity_constrained",
    }
    artifact = {
        "shard_count": 2,
        "physical_point_count": 10,
    }
    manifest = {
        "routing": {
            "initial_num_shards": 2,
            "shard_counts": [5, 5],
            "balance_diagnostics": {
                "mode": "capacity_constrained",
                "fixed_num_shards": True,
                "fission_applied": False,
                "l1_topology": {
                    "bounds_satisfied": True,
                    "over_upper_shards": [],
                    "under_lower_shards": [],
                },
                "l0_physical_copies": {
                    "bounds_satisfied": True,
                    "copy_count_preserved": True,
                    "all_assignments_have_navigation_evidence": True,
                    "non_evidence_assignment_count": 0,
                    "no_evidence_points": 0,
                    "over_upper_shards": [],
                    "under_lower_shards": [],
                    "requested_total_copies": 10,
                    "final": {"loads": [5, 5]},
                },
            },
        }
    }

    proof = module.validate_orion_balance_layout(manifest, parameters, artifact)

    assert proof == {
        "mode": "capacity_constrained",
        "bounds_satisfied": True,
        "l1_bounds_satisfied": True,
        "l0_bounds_satisfied": True,
        "copy_count_preserved": True,
        "all_assignments_have_navigation_evidence": True,
        "min_load": 5,
        "max_load": 5,
        "physical_point_count": 10,
    }

    manifest["routing"]["balance_diagnostics"]["l0_physical_copies"][
        "all_assignments_have_navigation_evidence"
    ] = False
    with pytest.raises(RuntimeError, match="failed required balance proofs"):
        module.validate_orion_balance_layout(manifest, parameters, artifact)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("balance_diagnostics", "l1_max_vote_loss"), 2, "L1 vote-loss contract"),
        (("balance_diagnostics", "l0_max_vote_loss"), 1, "L0 vote-loss contract"),
        (("balance_diagnostics", "l1_topology", "max_vote_loss"), 2, "L1 topology"),
        (
            (
                "balance_diagnostics",
                "l0_physical_copies",
                "configured_max_vote_loss",
            ),
            1,
            "L0 placement",
        ),
    ],
)
def test_orion_balance_layout_binds_stage_specific_vote_loss(
    path, value, message
):
    module = load_module()
    parameters = {
        "balance_mode": "capacity_constrained",
        "balance_max_vote_loss": 2,
        "balance_l1_max_vote_loss": 1,
        "balance_l0_max_vote_loss": 2,
    }
    artifact = {"shard_count": 2, "physical_point_count": 10}
    manifest = {
        "routing": {
            "initial_num_shards": 2,
            "shard_counts": [5, 5],
            "balance_diagnostics": {
                "mode": "capacity_constrained",
                "fixed_num_shards": True,
                "fission_applied": False,
                "l1_max_vote_loss": 1,
                "l0_max_vote_loss": 2,
                "l1_topology": {
                    "max_vote_loss": 1,
                    "bounds_satisfied": True,
                    "over_upper_shards": [],
                    "under_lower_shards": [],
                },
                "l0_physical_copies": {
                    "configured_max_vote_loss": 2,
                    "bounds_satisfied": True,
                    "copy_count_preserved": True,
                    "all_assignments_have_navigation_evidence": True,
                    "non_evidence_assignment_count": 0,
                    "no_evidence_points": 0,
                    "over_upper_shards": [],
                    "under_lower_shards": [],
                    "requested_total_copies": 10,
                    "final": {"loads": [5, 5]},
                },
            },
        }
    }

    proof = module.validate_orion_balance_layout(manifest, parameters, artifact)
    assert proof["l1_max_vote_loss"] == 1
    assert proof["l0_max_vote_loss"] == 2

    target = manifest["routing"]
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(RuntimeError, match=message):
        module.validate_orion_balance_layout(manifest, parameters, artifact)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("bounds_satisfied", False, "L1 topology failed required balance proofs"),
        ("over_upper_shards", [0], "L1 topology reports residual capacity violations"),
        ("under_lower_shards", [1], "L1 topology reports residual capacity violations"),
    ],
)
def test_orion_balance_layout_rejects_incomplete_l1_proof(field, value, message):
    module = load_module()
    parameters = {"balance_mode": "capacity_constrained"}
    artifact = {"shard_count": 2, "physical_point_count": 10}
    manifest = {
        "routing": {
            "initial_num_shards": 2,
            "shard_counts": [5, 5],
            "balance_diagnostics": {
                "mode": "capacity_constrained",
                "fixed_num_shards": True,
                "fission_applied": False,
                "l1_topology": {
                    "bounds_satisfied": True,
                    "over_upper_shards": [],
                    "under_lower_shards": [],
                },
                "l0_physical_copies": {
                    "bounds_satisfied": True,
                    "copy_count_preserved": True,
                    "all_assignments_have_navigation_evidence": True,
                    "non_evidence_assignment_count": 0,
                    "no_evidence_points": 0,
                    "over_upper_shards": [],
                    "under_lower_shards": [],
                    "requested_total_copies": 10,
                    "final": {"loads": [5, 5]},
                },
            },
        }
    }
    manifest["routing"]["balance_diagnostics"]["l1_topology"][field] = value

    with pytest.raises(RuntimeError, match=message):
        module.validate_orion_balance_layout(manifest, parameters, artifact)


def add_l1_upper_replay_contract(
    module,
    manifest,
    artifact,
    layout_dir,
    checksums,
):
    np = module.experiment.np
    artifact_path = layout_dir / "generation-2.json"
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
    artifact_sha256 = module.layout_common.sha256_path(artifact_path)
    queries_path = layout_dir / "upper-replay-queries.f32le"
    np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype="<f4").tofile(queries_path)
    queries_sha256 = module.layout_common.sha256_path(queries_path)
    verifier_path = layout_dir / "orion_verify_upper_replay"
    verifier_path.write_bytes(b"test verifier binary")
    verifier_sha256 = module.layout_common.sha256_path(verifier_path)
    replay_path = layout_dir / "upper-replay.bin"
    replay_path.write_bytes(b"ORION_UPPER_REPLAY_V1-test")
    replay_sha256 = module.layout_common.sha256_path(replay_path)
    source_path = Path(manifest["provenance"]["source_artifact"])
    source_sha256 = manifest["provenance"]["source_artifact_sha256"]
    canonical_graph_sha256 = "4" * 64
    upper_identity_sha256 = "5" * 64
    replay_manifest_path = layout_dir / "upper-replay-manifest.json"
    replay_manifest = {
        "format_version": 1,
        "verdict": "PASS",
        "source": {
            "path": str(source_path),
            "file_sha256": source_sha256,
            "canonical_artifact_sha256": "6" * 64,
            "generation": 1,
            "layout_sha256": "7" * 64,
            "shard_count": artifact["shard_count"],
            "physical_point_count": artifact["physical_point_count"],
            "canonical_upper_graph_sha256": canonical_graph_sha256,
            "canonical_upper_graph_size_bytes": 100,
            "ordered_upper_nodes_identity_sha256": upper_identity_sha256,
        },
        "rebound": {
            "path": str(artifact_path),
            "file_sha256": artifact_sha256,
            "canonical_artifact_sha256": "8" * 64,
            "generation": artifact["generation"],
            "layout_sha256": artifact["layout_sha256"],
            "shard_count": artifact["shard_count"],
            "physical_point_count": artifact["physical_point_count"],
            "canonical_upper_graph_sha256": canonical_graph_sha256,
            "canonical_upper_graph_size_bytes": 100,
            "ordered_upper_nodes_identity_sha256": upper_identity_sha256,
        },
        "query_corpus": {
            "path": str(queries_path),
            "sha256": queries_sha256,
            "row_count": 2,
            "dimension": 2,
            "size_bytes": queries_path.stat().st_size,
        },
        "replay": {
            "path": str(replay_path),
            "encoding": "ORION_UPPER_REPLAY_V1 test",
            "sha256": replay_sha256,
            "size_bytes": replay_path.stat().st_size,
            "upper_k": artifact["upper_k"],
            "compared_hit_count": 2 * artifact["upper_k"],
            "ordered_label_and_distance_bits_sha256": replay_sha256,
        },
        "gates": {
            "generation_advanced": True,
            "immutable_metadata_equal": True,
            "vector_schema_equal": True,
            "upper_search_contract_equal": True,
            "canonical_upper_graph_bytes_equal": True,
            "ordered_upper_labels_and_vector_bits_equal": True,
            "ordered_hit_labels_equal": True,
            "ordered_distance_bits_equal": True,
            "production_router_replay_complete": True,
        },
        "elapsed_seconds": 0.1,
    }
    replay_manifest_path.write_text(
        json.dumps(replay_manifest),
        encoding="utf-8",
    )
    replay_manifest_sha256 = module.layout_common.sha256_path(
        replay_manifest_path
    )
    diagnostics = manifest["routing"]["l1_partition_diagnostics"]
    diagnostics["upper_replay_manifest_sha256"] = replay_manifest_sha256
    manifest["provenance"].update(
        {
            "upper_replay": str(replay_path),
            "upper_replay_manifest": str(replay_manifest_path),
            "upper_replay_verifier": str(verifier_path),
            "upper_replay_verifier_sha256": verifier_sha256,
            "upper_replay_queries": str(queries_path),
            "upper_replay_queries_sha256": queries_sha256,
        }
    )
    manifest["outputs"].update(
        {
            "production_artifact": artifact_path.name,
            "upper_replay": replay_path.name,
            "upper_replay_manifest": replay_manifest_path.name,
        }
    )
    for path in (artifact_path, replay_path, replay_manifest_path):
        sha256 = module.layout_common.sha256_path(path)
        manifest["outputs"]["files"][path.name] = {
            "sha256": sha256,
            "size_bytes": path.stat().st_size,
        }
        checksums[path.name] = sha256


def l1_partition_contract_fixture(module, tmp_path, mode):
    layout_dir = tmp_path / f"l1-partition-{mode}"
    layout_dir.mkdir()
    is_candidate = mode == "l1_graph_prepartition"
    owner_values = [0, 1, 0, 1] if is_candidate else [0, 0, 0, 1]
    owner_path = layout_dir / "l1-owner.i32le"
    owner_path.write_bytes(
        b"".join(
            int(value).to_bytes(4, "little", signed=True)
            for value in owner_values
        )
    )
    owner_sha256 = module.layout_common.sha256_path(owner_path)
    l1_sizes = [owner_values.count(0), owner_values.count(1)]
    source_artifact_path = layout_dir / "source-generation-1.json"
    source_artifact_path.write_text('{"source":true}\n', encoding="utf-8")
    source_artifact_sha256 = module.layout_common.sha256_path(
        source_artifact_path
    )
    screen_manifest_path = layout_dir / "screen-manifest.json"
    screen_manifest = {"format_version": 1}
    if is_candidate:
        screen_manifest["contract"] = {
            "partition_input_scope": "production_upper_graph_only",
            "partitioner_reads_full_attachments": False,
            "partitioner_reads_l0_load": False,
            "partitioner_reads_multi_assignment_state": False,
            "l0_assignment_after_owner_freeze": True,
            "l0_repair_or_flow": False,
            "multi_assignment": {
                "enabled": True,
                "min_max_vote": 2,
                "vote_delta": 0,
                "max_shards": 0,
            },
        }
    screen_manifest_path.write_text(
        json.dumps(screen_manifest),
        encoding="utf-8",
    )
    screen_manifest_sha256 = module.layout_common.sha256_path(
        screen_manifest_path
    )
    artifact = {
        "generation": 2,
        "layout_sha256": "a" * 64,
        "logical_point_count": 6,
        "physical_point_count": 8,
        "shard_count": 2,
        "upper_k": 36,
        "upper_ef_search": 36,
        "dynamic_ef_base": 48,
        "dynamic_ef_factor": 15,
        "vector_schema": {"dimension": 2},
    }
    parameters = {
        "balance_mode": mode,
        "l1_partitioner": "ldg1" if is_candidate else "original_fixed_p",
        "initial_num_shards": 2,
        "enable_fission": False,
        "enable_topology_refinement": False,
        "use_multi_assign": True,
        "multi_assign_min_max_vote": 2,
        "multi_assign_vote_delta": 0,
        "multi_assign_max_shards": 0,
        "sample_denominator": 32,
        "upper_sample_seed": 100,
        "upper_m": 32,
        "upper_ef_construction": 100,
        "upper_graph_seed": 100,
        "attachment_search_ef": 100,
        "k_overlap": 10,
        "upper_k": 36,
        "upper_search_ef": 36,
        "dynamic_ef_base": 48,
        "dynamic_ef_factor": 15,
        "allow_decoupled_runtime_upper_search": False,
        "generation": 2,
        "l0_repair": False,
    }
    diagnostics = {
        "balance_contract": (
            module.ORION_L1_UNIT_BALANCE_CONTRACT
            if is_candidate
            else module.ORION_L1_REFERENCE_BALANCE_CONTRACT
        ),
        "input_scope": (
            "production_upper_graph_only"
            if is_candidate
            else "legacy_l0_informed_reference"
        ),
        "method": parameters["l1_partitioner"],
        "new_algorithm_eligible": is_candidate,
        "exact_quota": max(l1_sizes) - min(l1_sizes) <= 1,
        "l1_sizes": l1_sizes,
        "owner_sha256": owner_sha256,
        "source_artifact_sha256": source_artifact_sha256,
        "attachments_sha256": "c" * 64,
        "screen_manifest_sha256": screen_manifest_sha256,
        "l0_repair": False,
        "multi_assignment_after_owner_freeze": True,
    }
    manifest = {
        "parameters": parameters,
        "artifact_binding": {
            "generation": 2,
            "layout_sha256": "a" * 64,
            "logical_point_count": 6,
            "physical_point_count": 8,
            "shard_count": 2,
        },
        "routing": {
            "initial_num_shards": 2,
            "effective_num_shards": 2,
            "logical_point_count": 6,
            "physical_point_count": 8,
            "shard_counts": [4, 4],
            "upper_point_count": 4,
            "fission_events": [],
            "l1_partition_diagnostics": diagnostics,
            "copy_count_histogram": {"1": 4, "2": 2},
        },
        "provenance": {
            "source_artifact": str(source_artifact_path),
            "source_artifact_sha256": source_artifact_sha256,
            "screen_manifest": str(screen_manifest_path),
        },
        "outputs": {
            "owner": owner_path.name,
            "files": {
                owner_path.name: {
                    "sha256": owner_sha256,
                    "size_bytes": owner_path.stat().st_size,
                }
            },
        },
    }
    checksums = {owner_path.name: owner_sha256}
    add_l1_upper_replay_contract(
        module,
        manifest,
        artifact,
        layout_dir,
        checksums,
    )
    return manifest, parameters, artifact, layout_dir, checksums


def l1_navigation_mass_contract_fixture(module, tmp_path):
    np = module.experiment.np
    layout_dir = tmp_path / "l1-partition-navigation-mass"
    layout_dir.mkdir()
    labels = np.arange(100, 112, dtype=np.uint64)
    rows = []
    for query in range(len(labels)):
        local = [query]
        for node in range(9):
            if node not in local:
                local.append(node)
        fallback = 9
        while len(local) < module.ORION_L1_MASS_TOP_K:
            if fallback not in local:
                local.append(fallback)
            fallback += 1
        rows.append([int(labels[node]) for node in local])
    hits = np.asarray(rows, dtype="<u8")
    hits_path = layout_dir / "l1-self-hits-top10.u64le"
    hits.tofile(hits_path)
    hits_sha256 = module.layout_common.sha256_path(hits_path)

    source_artifact_path = layout_dir / "source-generation-1.json"
    source_artifact_path.write_text('{"source":true}\n', encoding="utf-8")
    source_artifact_sha256 = module.layout_common.sha256_path(
        source_artifact_path
    )
    hits_manifest_path = layout_dir / "l1-self-hits-top10.manifest.json"
    hits_manifest = {
        "format_version": 1,
        "artifact_sha256": source_artifact_sha256,
        "row_count": len(labels),
        "top_k": module.ORION_L1_MASS_TOP_K,
        "search_ef": module.ORION_L1_MASS_SEARCH_EF,
        "hits_path": str(hits_path),
        "hits_sha256": hits_sha256,
        "hits_size_bytes": hits_path.stat().st_size,
    }
    hits_manifest_path.write_text(json.dumps(hits_manifest), encoding="utf-8")
    hits_manifest_sha256 = module.layout_common.sha256_path(hits_manifest_path)

    local_hits = hits.astype(np.int64) - int(labels[0])
    raw_counts = np.bincount(local_hits.reshape(-1), minlength=len(labels))
    mass = raw_counts.astype(np.int64)
    mass_digest = module.hashlib.sha256()
    mass_digest.update(module.ORION_L1_MASS_SOURCE.encode("ascii"))
    mass_digest.update(b"\0")
    mass_digest.update(module.ORION_L1_MASS_TRANSFORM.encode("ascii"))
    mass_digest.update(
        module.struct.pack(
            "<QQ",
            len(labels),
            module.ORION_L1_MASS_TOP_K,
        )
    )
    mass_digest.update(mass.astype("<u8").tobytes())
    mass_sha256 = mass_digest.hexdigest()
    mass_values_path = layout_dir / "navigation-mass.u64le"
    mass.astype("<u8").tofile(mass_values_path)
    mass_values_sha256 = module.layout_common.sha256_path(mass_values_path)

    navigator_sha256 = "d" * 64
    upper_graph_sha256 = "c" * 64
    labels_sha256 = "e" * 64
    vectors_sha256 = "f" * 64
    upper_input_manifest_sha256 = "1" * 64
    partitioner_source_sha256 = "2" * 64
    amendment_path = module.ORION_L1_PROTOCOL_AMENDMENT_PATH.resolve()
    protocol_path = module.ORION_L1_PROTOCOL_PATH.resolve()
    protocol_amendment = {
        **json.loads(amendment_path.read_text(encoding="utf-8")),
        "amendment_path": str(amendment_path),
        "amendment_sha256": module.layout_common.sha256_path(amendment_path),
        "protocol_path": str(protocol_path),
        "protocol_sha256": module.layout_common.sha256_path(protocol_path),
    }
    mass_manifest_path = layout_dir / "navigation-mass-manifest.json"
    mass_manifest = {
        "format_version": 1,
        "mass_mode": module.ORION_L1_MASS_MODE,
        "source": module.ORION_L1_MASS_SOURCE,
        "transform": module.ORION_L1_MASS_TRANSFORM,
        "estimator_version": module.ORION_L1_MASS_ESTIMATOR_VERSION,
        "semantic_sha256": mass_sha256,
        "values_file": str(mass_values_path),
        "values_sha256": mass_values_sha256,
        "values_size_bytes": mass_values_path.stat().st_size,
        "total_mass": int(mass.sum()),
        "source_artifact": str(source_artifact_path),
        "source_artifact_sha256": source_artifact_sha256,
        "navigator_sha256": navigator_sha256,
        "upper_graph_sha256": upper_graph_sha256,
        "ordered_labels_sha256": labels_sha256,
        "ordered_vectors_sha256": vectors_sha256,
        "upper_navigation_hits": str(hits_path),
        "upper_navigation_hits_sha256": hits_sha256,
        "upper_navigation_manifest": str(hits_manifest_path),
        "upper_navigation_manifest_sha256": hits_manifest_sha256,
        "upper_node_count": len(labels),
        "dimension": 2,
        "top_k": module.ORION_L1_MASS_TOP_K,
        "search_ef": module.ORION_L1_MASS_SEARCH_EF,
        "self_present_count": len(labels),
        "self_first_count": len(labels),
        "self_first_fraction": 1.0,
        "duplicate_tie_exception_count": 0,
        "duplicate_tie_proof_sha256": module.canonical_json_sha256([]),
        "partitioner_source": "/experiment/l1_mass_partitioner.py",
        "partitioner_source_sha256": partitioner_source_sha256,
        "protocol_amendment": protocol_amendment,
    }
    mass_manifest_path.write_text(json.dumps(mass_manifest), encoding="utf-8")
    mass_manifest_sha256 = module.layout_common.sha256_path(mass_manifest_path)

    owner_values = [0] * 5 + [1] * 7
    owner_path = layout_dir / "l1-owner-mass-balanced-kmeans.i32le"
    owner_path.write_bytes(
        b"".join(
            int(value).to_bytes(4, "little", signed=True)
            for value in owner_values
        )
    )
    owner_sha256 = module.layout_common.sha256_path(owner_path)
    raw_owner_path = layout_dir / "raw-mass-balanced-kmeans.owner.i32le"
    raw_owner_path.write_bytes(owner_path.read_bytes())
    raw_reference_path = layout_dir / "raw-candidate-manifest.json"
    raw_reference = {
        "format_version": 1,
        "stage": "upper_only_candidates_frozen",
        "parameters": {"mass_mode": "raw"},
        "candidates": [
            {
                "method": module.ORION_L1_MASS_METHOD,
                "mass_mode": "raw",
                "navigation_mass_source": module.ORION_L1_RAW_MASS_SOURCE,
                "navigation_mass_transform": module.ORION_L1_RAW_MASS_TRANSFORM,
                "source_artifact_sha256": source_artifact_sha256,
                "owner_path": str(raw_owner_path),
                "owner_sha256": owner_sha256,
            }
        ],
    }
    raw_reference_path.write_text(json.dumps(raw_reference), encoding="utf-8")
    raw_reference_sha256 = module.layout_common.sha256_path(raw_reference_path)
    raw_reference_path.with_name(raw_reference_path.name + ".sha256").write_text(
        raw_reference_sha256 + "\n",
        encoding="ascii",
    )
    owner_parity = {
        "reference_mass_mode": "raw",
        "reference_candidate_manifest_path": str(raw_reference_path),
        "reference_candidate_manifest_sha256": raw_reference_sha256,
        "reference_owner_path": str(raw_owner_path),
        "reference_owner_sha256": owner_sha256,
        "owner_bytes_identical": True,
    }
    owner_sizes = [5, 7]
    estimated_masses = [
        int(mass[:5].sum()),
        int(mass[5:].sum()),
    ]
    mass_target = float(mass.sum() / 2)
    mass_limit = module.math.ceil(mass_target) + int(mass.max()) - 1

    construction_inputs = {
        "artifact": str(source_artifact_path),
        "artifact_sha256": source_artifact_sha256,
        "navigator_sha256": navigator_sha256,
        "upper_graph_sha256": upper_graph_sha256,
        "logical_point_count": 120,
        "upper_node_count": len(labels),
        "dimension": 2,
        "ordered_labels_sha256": labels_sha256,
        "ordered_vectors_sha256": vectors_sha256,
        "upper_input_manifest": "/experiment/upper-input.json",
        "upper_input_manifest_sha256": upper_input_manifest_sha256,
        "upper_navigation_hits": str(hits_path),
        "upper_navigation_hits_sha256": hits_sha256,
        "upper_navigation_manifest": str(hits_manifest_path),
        "upper_navigation_manifest_sha256": hits_manifest_sha256,
        "upper_navigation_top_k": module.ORION_L1_MASS_TOP_K,
        "upper_navigation_search_ef": module.ORION_L1_MASS_SEARCH_EF,
        "partitioner_source": "/experiment/l1_mass_partitioner.py",
        "partitioner_source_sha256": partitioner_source_sha256,
    }
    mass_estimator = {
        "format_version": 1,
        "source": module.ORION_L1_MASS_SOURCE,
        "transform": module.ORION_L1_MASS_TRANSFORM,
        "estimator_version": module.ORION_L1_MASS_ESTIMATOR_VERSION,
        "mass_mode": module.ORION_L1_MASS_MODE,
        "sha256": mass_sha256,
        "manifest_path": str(mass_manifest_path),
        "manifest_sha256": mass_manifest_sha256,
        "query_count": len(labels),
        "top_k": module.ORION_L1_MASS_TOP_K,
        "total_mass": int(mass.sum()),
        "self_present_count": len(labels),
        "self_first_count": len(labels),
        "self_first_fraction": 1.0,
        "duplicate_tie_exception_count": 0,
        "duplicate_tie_proof_sha256": module.canonical_json_sha256([]),
        "vertex_mass_min": int(mass.min()),
        "vertex_mass_max": int(mass.max()),
        "vertex_mass_mean": float(mass.mean()),
        "vertex_mass_cv": float(mass.std() / mass.mean()),
        "rank_weighting": False,
        "eligible_as_finalist": True,
    }
    frozen_contract = {
        "partition_input_scope": (
            "production_upper_graph_and_upper_navigation_mass_only"
        ),
        "construction_process_accepts_l0_inputs": False,
        "construction_process_accepts_query_ground_truth": False,
        "partitioner_reads_full_attachments": False,
        "partitioner_reads_l0_load": False,
        "partitioner_reads_multi_assignment_state": False,
        "separate_evaluator_process_required": True,
        "l0_repair_or_flow": False,
    }
    frozen_candidate = {
        "method": module.ORION_L1_MASS_METHOD,
        "balance_contract": module.ORION_L1_MASS_BALANCE_CONTRACT,
        "source_artifact_sha256": source_artifact_sha256,
        "owner_path": str(owner_path),
        "owner_sha256": owner_sha256,
        "partitioner_source_sha256": partitioner_source_sha256,
        "navigation_mass_manifest_path": str(mass_manifest_path),
        "navigation_mass_manifest_sha256": mass_manifest_sha256,
        "navigation_mass_source": module.ORION_L1_MASS_SOURCE,
        "navigation_mass_transform": module.ORION_L1_MASS_TRANSFORM,
        "navigation_mass_sha256": mass_sha256,
        "navigation_mass_estimator_version": (
            module.ORION_L1_MASS_ESTIMATOR_VERSION
        ),
        "mass_mode": module.ORION_L1_MASS_MODE,
        "protocol_amendment": protocol_amendment,
        "owner_parity": owner_parity,
        "partition_sizes": owner_sizes,
        "estimated_partition_masses": estimated_masses,
        "estimated_mass_target": mass_target,
        "estimated_mass_limit": mass_limit,
        "graph_topology_all_pass": True,
        "topology_all_pass": False,
        "eligibility_pending_post_freeze_query_gates": True,
        "materialization_eligible": False,
    }
    frozen_manifest_path = layout_dir / "candidate-manifest.json"
    frozen_manifest = {
        "format_version": 1,
        "stage": "upper_only_candidates_frozen",
        "contract": frozen_contract,
        "mass_estimator": mass_estimator,
        "construction_inputs": construction_inputs,
        "candidates": [frozen_candidate],
    }
    frozen_manifest_path.write_text(json.dumps(frozen_manifest), encoding="utf-8")
    frozen_manifest_sha256 = module.layout_common.sha256_path(frozen_manifest_path)
    frozen_manifest_path.with_name(frozen_manifest_path.name + ".sha256").write_text(
        frozen_manifest_sha256 + "\n",
        encoding="ascii",
    )

    layout_sha256 = "a" * 64
    copy_histogram = {"1": 100, "2": 20}
    final_candidate = {
        **frozen_candidate,
        "attachments_sha256": "c" * 64,
        "upper_navigation_hits_sha256": hits_sha256,
        "upper_navigation_manifest_sha256": hits_manifest_sha256,
        "identity_all_pass": True,
        "topology_gate_inputs_complete": True,
        "topology_all_pass": True,
        "owner_parity_all_pass": True,
        "eligibility_pending_post_freeze_query_gates": False,
        "materialization_eligible": True,
        "topology_gates": {
            "upper_edge_cut_ratio": {
                "observed": 0.5,
                "operator": "<=",
                "threshold": 0.6,
                "pass": True,
            }
        },
        "materialization_parity": {
            "canonical_format": "orion_numeric_import.assignments.jsonl-v1",
            "assignment_bytes_sha256": layout_sha256,
            "logical_point_count": 120,
            "physical_point_count": 140,
            "shard_loads": [70, 70],
            "copy_count_histogram": copy_histogram,
        },
    }
    screen_manifest_path = layout_dir / "screen-manifest.json"
    screen_manifest = {
        "format_version": 1,
        "stage": "post_freeze_evaluation",
        "contract": {
            **frozen_contract,
            "evaluator_runs_in_separate_process": True,
            "owners_changed_after_freeze": False,
            "l0_assignment_after_owner_freeze": True,
            "multi_assignment": {
                "enabled": True,
                "min_max_vote": 2,
                "vote_delta": 0,
                "max_shards": 0,
            },
        },
        "frozen_candidate_manifest": str(frozen_manifest_path),
        "frozen_candidate_manifest_sha256": frozen_manifest_sha256,
        "mass_estimator": mass_estimator,
        "identity_all_pass": True,
        "owner_parity_all_pass": True,
        "construction_inputs": construction_inputs,
        "candidates": [final_candidate],
    }
    screen_manifest_path.write_text(json.dumps(screen_manifest), encoding="utf-8")
    screen_manifest_sha256 = module.layout_common.sha256_path(screen_manifest_path)
    screen_manifest_path.with_name(screen_manifest_path.name + ".sha256").write_text(
        screen_manifest_sha256 + "\n",
        encoding="ascii",
    )

    v1_failure_screen_path = layout_dir / "v1-floor1-failed-screen.json"
    v1_failure_screen_path.write_text(
        json.dumps({"topology_all_pass": False}),
        encoding="utf-8",
    )
    dataset_confirmation = {
        "graph_topology_all_pass": True,
        "identity_all_pass": True,
        "logical_point_count": 120,
        "logical_shards": 32,
        "materialization_eligible": True,
        "materialization_parity": final_candidate["materialization_parity"],
        "owner_bytes_identical": True,
        "owner_parity_all_pass": True,
        "owner_path": str(owner_path),
        "owner_sha256": owner_sha256,
        "phase_a_candidate_manifest_path": str(frozen_manifest_path),
        "phase_a_candidate_manifest_sha256": frozen_manifest_sha256,
        "phase_b_screen_manifest_path": str(screen_manifest_path),
        "phase_b_screen_manifest_sha256": screen_manifest_sha256,
        "raw_reference_candidate_manifest_path": str(raw_reference_path),
        "raw_reference_candidate_manifest_sha256": raw_reference_sha256,
        "raw_reference_owner_path": str(raw_owner_path),
        "raw_reference_owner_sha256": owner_sha256,
        "source_artifact_path": str(source_artifact_path),
        "source_artifact_sha256": source_artifact_sha256,
        "topology_all_pass": True,
        "topology_gates": final_candidate["topology_gates"],
    }
    confirmation_builder_path = (
        module.ORION_L1_V2_CONFIRMATION_BUILDER_PATH.resolve()
    )
    cross_dataset_confirmation_path = layout_dir / "confirmation-manifest.json"
    cross_dataset_confirmation = {
        "format_version": 1,
        "stage": "raw_regularized_v2_cross_dataset_offline_confirmation",
        "status": "materialization_eligible_online_qps_pending",
        "builder": {
            "path": str(confirmation_builder_path),
            "sha256": module.layout_common.sha256_path(confirmation_builder_path),
        },
        "method_contract": {
            "mass_mode": module.ORION_L1_MASS_MODE,
            "source": module.ORION_L1_MASS_SOURCE,
            "transform": module.ORION_L1_MASS_TRANSFORM,
            "estimator_version": module.ORION_L1_MASS_ESTIMATOR_VERSION,
            "method": module.ORION_L1_MASS_METHOD,
            "upper_only_phase_a": True,
            "l0_evaluation_only_after_owner_freeze": True,
            "multi_assignment_unchanged": True,
        },
        "confirmation_contract": {
            "alternate_method_or_parameter_scan": False,
            "old_raw_owner_byte_parity_required": True,
            "online_cluster_entered_by_this_stage": False,
            "online_qps_confirmation_completed": False,
            "online_qps_confirmation_required": True,
            "online_qps_only_confirmation": True,
            "post_exploratory_not_preregistered": True,
            "topology_thresholds_unchanged": True,
            "v1_heldout_failure_retained": True,
        },
        "protocol_amendment": protocol_amendment,
        "datasets": {
            "sift": {**dataset_confirmation, "dataset": "sift"},
            "glove": {**dataset_confirmation, "dataset": "glove"},
        },
        "cross_dataset_identity_all_pass": True,
        "cross_dataset_topology_all_pass": True,
        "owner_parity_all_pass": True,
        "materialization_eligible": True,
        "online_qps_confirmation_required": True,
        "v1_heldout_failure": {
            "dataset": "glove",
            "mass_mode": "self-debiased-floor1",
            "method": module.ORION_L1_MASS_METHOD,
            "topology_all_pass": False,
            "materialization_eligible": False,
            "screen_manifest_path": str(v1_failure_screen_path),
            "screen_manifest_sha256": module.layout_common.sha256_path(
                v1_failure_screen_path
            ),
            "failed_gates": {
                "routed_shards_mean": {
                    "observed": 10.1,
                    "operator": "<=",
                    "threshold": 10.0,
                    "pass": False,
                }
            },
        },
    }
    cross_dataset_confirmation_path.write_text(
        json.dumps(cross_dataset_confirmation),
        encoding="utf-8",
    )
    cross_dataset_confirmation_sha256 = module.layout_common.sha256_path(
        cross_dataset_confirmation_path
    )
    cross_dataset_confirmation_path.with_name(
        cross_dataset_confirmation_path.name + ".sha256"
    ).write_text(
        cross_dataset_confirmation_sha256 + "\n",
        encoding="ascii",
    )

    artifact = {
        "generation": 2,
        "layout_sha256": layout_sha256,
        "logical_point_count": 120,
        "physical_point_count": 140,
        "shard_count": 2,
        "upper_k": 36,
        "upper_ef_search": 36,
        "dynamic_ef_base": 48,
        "dynamic_ef_factor": 15,
        "vector_schema": {"dimension": 2},
        "upper_nodes": [
            {"label": int(label), "vector": [float(index), 0.0]}
            for index, label in enumerate(labels.tolist())
        ],
    }
    parameters = {
        "balance_mode": "l1_graph_prepartition",
        "l1_partitioner": module.ORION_L1_MASS_METHOD,
        "initial_num_shards": 2,
        "enable_fission": False,
        "enable_topology_refinement": False,
        "use_multi_assign": True,
        "multi_assign_min_max_vote": 2,
        "multi_assign_vote_delta": 0,
        "multi_assign_max_shards": 0,
        "sample_denominator": 32,
        "upper_sample_seed": 100,
        "upper_m": 32,
        "upper_ef_construction": 100,
        "upper_graph_seed": 100,
        "attachment_search_ef": 100,
        "k_overlap": 10,
        "upper_k": 36,
        "upper_search_ef": 36,
        "dynamic_ef_base": 48,
        "dynamic_ef_factor": 15,
        "allow_decoupled_runtime_upper_search": False,
        "generation": 2,
        "l0_repair": False,
    }
    mass_balance = {
        "method": module.ORION_L1_MASS_METHOD,
        "owner_sha256": owner_sha256,
        "partition_sizes": owner_sizes,
        "estimated_partition_masses": estimated_masses,
        "estimated_mass_target": mass_target,
        "estimated_mass_limit": mass_limit,
        "navigation_mass_source": module.ORION_L1_MASS_SOURCE,
        "navigation_mass_transform": module.ORION_L1_MASS_TRANSFORM,
        "navigation_mass_estimator_version": (
            module.ORION_L1_MASS_ESTIMATOR_VERSION
        ),
        "mass_mode": module.ORION_L1_MASS_MODE,
        "navigation_mass_sha256": mass_sha256,
        "navigation_mass_manifest_sha256": mass_manifest_sha256,
        "upper_navigation_hits_sha256": hits_sha256,
        "upper_navigation_manifest_sha256": hits_manifest_sha256,
        "materialization_eligible": True,
    }
    diagnostics = {
        "balance_contract": module.ORION_L1_MASS_BALANCE_CONTRACT,
        "input_scope": (
            "production_upper_graph_and_upper_navigation_mass_only"
        ),
        "method": module.ORION_L1_MASS_METHOD,
        "new_algorithm_eligible": True,
        "exact_quota": False,
        "l1_sizes": owner_sizes,
        "owner_sha256": owner_sha256,
        "source_artifact_sha256": source_artifact_sha256,
        "attachments_sha256": "c" * 64,
        "screen_manifest_sha256": screen_manifest_sha256,
        "protocol_amendment_sha256": protocol_amendment["amendment_sha256"],
        "cross_dataset_confirmation_sha256": (
            cross_dataset_confirmation_sha256
        ),
        "owner_parity": owner_parity,
        "l0_repair": False,
        "multi_assignment_after_owner_freeze": True,
        "mass_balance": mass_balance,
    }
    manifest = {
        "parameters": parameters,
        "artifact_binding": {
            "generation": 2,
            "layout_sha256": layout_sha256,
            "logical_point_count": 120,
            "physical_point_count": 140,
            "shard_count": 2,
        },
        "routing": {
            "initial_num_shards": 2,
            "effective_num_shards": 2,
            "logical_point_count": 120,
            "physical_point_count": 140,
            "shard_counts": [70, 70],
            "upper_point_count": len(labels),
            "fission_events": [],
            "l1_partition_diagnostics": diagnostics,
            "copy_count_histogram": copy_histogram,
        },
        "provenance": {
            "source_artifact": str(source_artifact_path),
            "source_artifact_sha256": source_artifact_sha256,
            "screen_manifest": str(screen_manifest_path),
            "upper_navigation_hits": str(hits_path),
            "upper_navigation_manifest": str(hits_manifest_path),
            "navigation_mass_manifest": str(mass_manifest_path),
            "protocol_amendment": str(amendment_path),
            "protocol": str(protocol_path),
            "cross_dataset_confirmation": str(cross_dataset_confirmation_path),
        },
        "outputs": {
            "owner": owner_path.name,
            "files": {
                owner_path.name: {
                    "sha256": owner_sha256,
                    "size_bytes": owner_path.stat().st_size,
                }
            },
        },
    }
    checksums = {owner_path.name: owner_sha256}
    add_l1_upper_replay_contract(
        module,
        manifest,
        artifact,
        layout_dir,
        checksums,
    )
    return manifest, parameters, artifact, layout_dir, checksums


def test_l1_partition_layout_requires_explicit_loader_authorization(tmp_path):
    module = load_module()
    layout_dir = tmp_path / "authorization"
    layout_dir.mkdir()
    manifest_path = layout_dir / module.layout_common.BUILD_MANIFEST_NAME
    manifest_path.write_text(
        json.dumps(
            {
                "format_version": 1,
                "tool": module.ORION_L1_PARTITION_LAYOUT_TOOL,
                "mode": "production_bundle",
            }
        ),
        encoding="utf-8",
    )
    (layout_dir / module.layout_common.CHECKSUMS_NAME).write_text(
        f"{module.layout_common.sha256_path(manifest_path)}  {manifest_path.name}\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="allow-orion-l1-partition-layout"):
        module.load_routed_layout("orion", layout_dir)


def test_l1_partition_tool_allowlist_includes_explicit_cnbr_tools() -> None:
    module = load_module()
    assert module.ORION_L1_PARTITION_LAYOUT_TOOLS == {
        module.ORION_L1_PARTITION_LAYOUT_TOOL,
        module.ORION_NATIVE_CNBR_LAYOUT_TOOL,
        module.ORION_SAMPLED_CNBR_LAYOUT_TOOL,
        module.ORION_BUDGETED_MULTI_ASSIGNMENT_LAYOUT_TOOL,
        module.ORION_BMR10_SCALING_LAYOUT_TOOL,
        module.ORION_ARTIFACT_OWNER_BMR10_LAYOUT_TOOL,
        module.ORION_OWNER_POLICY_TOURNAMENT_LAYOUT_TOOL,
    }
    assert module.ORION_NATIVE_CNBR_LAYOUT_TOOL in module.ORION_LAYOUT_TOOLS
    assert (
        module.ORION_BUDGETED_MULTI_ASSIGNMENT_LAYOUT_TOOL
        in module.ORION_LAYOUT_TOOLS
    )
    assert module.ORION_OWNER_POLICY_TOURNAMENT_LAYOUT_TOOL in module.ORION_LAYOUT_TOOLS


def test_budgeted_multi_assignment_replays_copy_cap_budget_and_semantic_hash(
    tmp_path,
):
    module = load_module()
    assignment = tmp_path / "bmr-assignments.jsonl"
    rows = [
        {"id": point_id, "shards": [0, 1] if point_id == 0 else [point_id // 5]}
        for point_id in range(10)
    ]
    assignment.write_bytes(
        b"".join(
            (json.dumps(row, separators=(",", ":")) + "\n").encode("utf-8")
            for row in rows
        )
    )
    membership = module.experiment.np.zeros((10, 2), dtype=bool)
    for row in rows:
        membership[row["id"], row["shards"]] = True
    packed = module.experiment.np.packbits(membership, axis=1, bitorder="little")
    loads = membership.sum(axis=0, dtype=module.experiment.np.int64)
    mean = float(loads.mean())
    candidate = {
        "policy": "BMR_10",
        "assignment_jsonl_sha256": module.layout_common.sha256_path(assignment),
        "membership_semantic_sha256": module.hashlib.sha256(
            module.experiment.np.ascontiguousarray(packed).tobytes()
        ).hexdigest(),
        "logical_point_count": 10,
        "physical_point_count": 11,
        "expansion_ratio": 1.1,
        "extra_copy_fraction": 1.1 - 1.0,
        "copy_count_histogram": {"1": 9, "2": 1},
        "physical_copy_load_min": int(loads.min()),
        "physical_copy_load_max": int(loads.max()),
        "physical_copy_load_mean": mean,
        "physical_copy_load_cv": float(loads.std() / mean),
        "physical_copy_load_max_over_mean": float(loads.max() / mean),
        "physical_copy_load_min_over_mean": float(loads.min() / mean),
        "physical_copy_load_empty_shards": 0,
        "extra_copy_budget_numerator": 1,
        "extra_copy_budget_denominator": 10,
    }

    proof = module._validate_budgeted_multi_assignment_bytes(
        assignment,
        candidate=candidate,
        shard_count=2,
    )
    assert proof["extra_copy_count"] == proof["exact_extra_copy_budget"] == 1
    assert proof["maximum_copies_per_point"] == 2
    assert proof["physical_copy_shard_loads"] == [5, 6]

    too_many = tmp_path / "too-many.jsonl"
    too_many_rows = [dict(row) for row in rows]
    too_many_rows[0] = {"id": 0, "shards": [0, 1, 2]}
    too_many.write_bytes(
        b"".join(
            (json.dumps(row, separators=(",", ":")) + "\n").encode("utf-8")
            for row in too_many_rows
        )
    )
    with pytest.raises(RuntimeError, match="one-or-two-copy"):
        module._validate_budgeted_multi_assignment_bytes(
            too_many,
            candidate={
                "assignment_jsonl_sha256": module.layout_common.sha256_path(
                    too_many
                )
            },
            shard_count=3,
        )

    over_budget = tmp_path / "over-budget.jsonl"
    over_budget_rows = [dict(row) for row in rows]
    over_budget_rows[1] = {"id": 1, "shards": [0, 1]}
    over_budget.write_bytes(
        b"".join(
            (json.dumps(row, separators=(",", ":")) + "\n").encode("utf-8")
            for row in over_budget_rows
        )
    )
    with pytest.raises(RuntimeError, match="exact frozen 10% budget"):
        module._validate_budgeted_multi_assignment_bytes(
            over_budget,
            candidate={
                "assignment_jsonl_sha256": module.layout_common.sha256_path(
                    over_budget
                ),
                "extra_copy_budget_numerator": 1,
                "extra_copy_budget_denominator": 10,
            },
            shard_count=2,
        )


@pytest.mark.parametrize(
    ("policy_name", "rows"),
    [
        (
            "current_all_max",
            [
                {"id": 0, "shards": [0, 1, 2]},
                {"id": 1, "shards": [1]},
                {"id": 2, "shards": [2]},
            ],
        ),
        (
            "single_rank",
            [
                {"id": 0, "shards": [0]},
                {"id": 1, "shards": [1]},
                {"id": 2, "shards": [2]},
            ],
        ),
        (
            "BMR_10",
            [
                {"id": point_id, "shards": [0, 1] if point_id == 0 else [point_id % 3]}
                for point_id in range(10)
            ],
        ),
    ],
)
def test_owner_policy_tournament_assignment_replays_exact_metrics(
    tmp_path, policy_name, rows
):
    module = load_module()
    assignment = tmp_path / f"{policy_name}.jsonl"
    assignment.write_bytes(
        b"".join(
            (json.dumps(row, separators=(",", ":")) + "\n").encode("utf-8")
            for row in rows
        )
    )
    membership = module.experiment.np.zeros((len(rows), 3), dtype=bool)
    for row in rows:
        membership[row["id"], row["shards"]] = True
    packed = module.experiment.np.packbits(membership, axis=1, bitorder="little")
    loads = membership.sum(axis=0, dtype=module.experiment.np.int64)
    copies = membership.sum(axis=1, dtype=module.experiment.np.int64)
    values, counts = module.experiment.np.unique(copies, return_counts=True)
    mean = float(loads.mean())
    candidate = {
        "policy": policy_name,
        "assignment_jsonl_sha256": module.layout_common.sha256_path(assignment),
        "membership_semantic_sha256": module.hashlib.sha256(
            module.experiment.np.ascontiguousarray(packed).tobytes()
        ).hexdigest(),
        "logical_point_count": len(rows),
        "physical_point_count": int(copies.sum()),
        "expansion_ratio": float(copies.sum() / len(rows)),
        "extra_copy_fraction": float(copies.sum() / len(rows) - 1.0),
        "copy_count_histogram": {
            str(int(value)): int(count)
            for value, count in zip(values.tolist(), counts.tolist(), strict=True)
        },
        "physical_copy_load_min": int(loads.min()),
        "physical_copy_load_max": int(loads.max()),
        "physical_copy_load_mean": mean,
        "physical_copy_load_cv": float(loads.std() / mean),
        "physical_copy_load_max_over_mean": float(loads.max() / mean),
        "physical_copy_load_min_over_mean": float(loads.min() / mean),
        "physical_copy_load_empty_shards": int((loads == 0).sum()),
    }
    proof = module._validate_owner_policy_tournament_assignment_bytes(
        assignment,
        candidate=candidate,
        shard_count=3,
        policy_name=policy_name,
    )
    assert proof["physical_point_count"] == int(copies.sum())
    assert proof["maximum_copies_per_point"] == int(copies.max())


def test_native_cnbr_assignment_golden_replays_every_parity_field(tmp_path):
    module = load_module()
    assignment = tmp_path / "assignments.jsonl"
    assignment.write_bytes(
        b'{"id":0,"shards":[0]}\n'
        b'{"id":1,"shards":[0,1]}\n'
        b'{"id":2,"shards":[1]}\n'
    )
    primary = module.hashlib.sha256()
    for value in (0, 0, 1):
        primary.update(module.struct.pack("<i", value))
    golden = {
        "canonical_format": "orion_numeric_import.assignments.jsonl-v1",
        "assignment_bytes_sha256": module.layout_common.sha256_path(assignment),
        "logical_point_count": 3,
        "physical_point_count": 4,
        "primary_shards_sha256": primary.hexdigest(),
        "primary_shard_loads": [2, 1],
        "physical_copy_shard_loads": [2, 2],
        "copy_count_histogram": {"1": 2, "2": 1},
    }

    assert module._validate_native_cnbr_assignment_golden(
        assignment,
        golden=golden,
        shard_count=2,
    ) == golden

    drifted = dict(golden, primary_shard_loads=[1, 2])
    with pytest.raises(RuntimeError, match="assignment/golden parity mismatch"):
        module._validate_native_cnbr_assignment_golden(
            assignment,
            golden=drifted,
            shard_count=2,
        )


def _native_cnbr_prebinding_fixture(module, tmp_path, arm_name):
    phase_b = tmp_path / f"{arm_name}-screen.json"
    selection = tmp_path / f"{arm_name}-selection.json"
    cost = tmp_path / f"{arm_name}-construction-cost-audit.json"
    for path in (phase_b, selection, cost):
        path.write_text("{}\n", encoding="utf-8")
    artifact = {
        "generation": 2,
        "shard_count": 2,
        "logical_point_count": 4,
        "physical_point_count": 4,
        "layout_sha256": "a" * 64,
        "upper_k": 2,
        "upper_ef_search": 4,
        "dynamic_ef_base": 5,
        "dynamic_ef_factor": 2,
    }
    variant = "natural_orion" if arm_name == "N_native" else "cnbr"
    parameters = {
        "generation": 2,
        "initial_num_shards": 2,
        "sample_denominator": 32,
        "upper_sample_seed": 100,
        "upper_m": 32,
        "upper_ef_construction": 100,
        "upper_graph_seed": 100,
        "attachment_search_ef": 100,
        "upper_k": 2,
        "upper_search_ef": 4,
        "dynamic_ef_base": 5,
        "dynamic_ef_factor": 2,
        "k_overlap": 10,
        "use_multi_assign": True,
        "multi_assign_min_max_vote": 2,
        "multi_assign_vote_delta": 0,
        "multi_assign_max_shards": 0,
        "enable_fission": False,
        "enable_topology_refinement": False,
        "l0_repair": False,
        "allow_decoupled_runtime_upper_search": False,
        "l1_partitioner": arm_name,
        "balance_mode": variant,
    }
    diagnostics = {}
    provenance = {
        "phase_b_screen": str(phase_b.resolve()),
        "phase_b_screen_sha256": module.layout_common.sha256_path(phase_b),
    }
    if arm_name == "C_CNBR":
        cost_sha256 = module.layout_common.sha256_path(cost)
        provenance.update(
            {
                "selection_manifest": str(selection.resolve()),
                "selection_manifest_sha256": module.layout_common.sha256_path(
                    selection
                ),
                "construction_cost_v4_audit": str(cost.resolve()),
                "construction_cost_v4_audit_sha256": cost_sha256,
            }
        )
        diagnostics.update(
            {
                "construction_cost_v4_gate": "PASS",
                "construction_cost_v4_audit_sha256": cost_sha256,
            }
        )
    manifest = {
        "artifact_binding": {
            "generation": 2,
            "layout_sha256": "a" * 64,
            "logical_point_count": 4,
            "physical_point_count": 4,
            "shard_count": 2,
        },
        "routing": {
            "initial_num_shards": 2,
            "effective_num_shards": 2,
            "logical_point_count": 4,
            "physical_point_count": 4,
            "fission_events": [],
            "shard_counts": [2, 2],
            "l1_partition_diagnostics": diagnostics,
        },
        "provenance": provenance,
        "outputs": {},
    }
    return manifest, parameters, artifact, cost


@pytest.mark.parametrize("arm_name", ["N_native", "C_CNBR"])
def test_native_cnbr_loader_forwards_arm_specific_cost_binding(
    tmp_path, monkeypatch, arm_name
):
    module = load_module()
    from experiments.l1_balance import (
        materialize_native_cnbr_candidate as native_cnbr,
    )

    manifest, parameters, artifact, cost = _native_cnbr_prebinding_fixture(
        module, tmp_path, arm_name
    )
    captured = {}

    class Probe(Exception):
        pass

    def probe(phase_b, arm, selection, construction_cost):
        captured.update(
            {
                "phase_b": phase_b,
                "arm": arm,
                "selection": selection,
                "construction_cost": construction_cost,
            }
        )
        raise Probe

    monkeypatch.setattr(native_cnbr, "validate_phase_b_screen", probe)
    with pytest.raises(Probe):
        module.validate_orion_native_cnbr_layout(
            manifest, parameters, artifact, tmp_path, {}
        )
    assert captured["arm"] == arm_name
    if arm_name == "N_native":
        assert captured["selection"] is None
        assert captured["construction_cost"] is None
    else:
        assert captured["selection"] == tmp_path / "C_CNBR-selection.json"
        assert captured["construction_cost"] == cost.resolve()


def test_native_cnbr_loader_rejects_candidate_cost_sha_drift(tmp_path) -> None:
    module = load_module()
    manifest, _parameters, _artifact, _cost = _native_cnbr_prebinding_fixture(
        module, tmp_path, "C_CNBR"
    )
    manifest["routing"]["l1_partition_diagnostics"][
        "construction_cost_v4_audit_sha256"
    ] = "f" * 64
    with pytest.raises(RuntimeError, match="different construction-cost audit"):
        module._resolve_native_cnbr_construction_cost_audit(
            manifest["provenance"],
            manifest["routing"]["l1_partition_diagnostics"],
            "C_CNBR",
        )


def test_native_cnbr_loader_rejects_cost_claim_on_native(tmp_path) -> None:
    module = load_module()
    manifest, _parameters, _artifact, cost = _native_cnbr_prebinding_fixture(
        module, tmp_path, "N_native"
    )
    manifest["provenance"]["construction_cost_v4_audit"] = str(cost.resolve())
    with pytest.raises(RuntimeError, match="must not claim"):
        module._resolve_native_cnbr_construction_cost_audit(
            manifest["provenance"],
            manifest["routing"]["l1_partition_diagnostics"],
            "N_native",
        )


@pytest.mark.parametrize(
    ("mode", "eligible", "scope", "exact_quota"),
    [
        (
            "l1_graph_prepartition",
            True,
            "production_upper_graph_only",
            True,
        ),
        (
            "matched_l0_informed_reference",
            False,
            "legacy_l0_informed_reference",
            False,
        ),
    ],
)
def test_l1_partition_layout_accepts_candidate_and_explicit_baseline_reference(
    tmp_path,
    mode,
    eligible,
    scope,
    exact_quota,
):
    module = load_module()
    manifest, parameters, artifact, layout_dir, checksums = (
        l1_partition_contract_fixture(module, tmp_path, mode)
    )

    proof = module.validate_orion_l1_partition_layout(
        manifest,
        parameters,
        artifact,
        layout_dir,
        checksums,
    )

    assert proof["mode"] == mode
    assert proof["new_algorithm_eligible"] is eligible
    assert proof["baseline_only"] is (not eligible)
    assert proof["input_scope"] == scope
    assert proof["exact_quota"] is exact_quota
    assert proof["multi_assignment"] == {
        "enabled": True,
        "min_max_vote": 2,
        "vote_delta": 0,
        "max_shards": 0,
    }
    assert proof["l0_repair"] is False
    assert proof["fission"] is False
    assert proof["upper_replay"]["verdict"] == "PASS"


def test_l1_partition_accepts_bounded_navigation_mass_with_unequal_node_counts(
    tmp_path,
):
    module = load_module()
    manifest, parameters, artifact, layout_dir, checksums = (
        l1_navigation_mass_contract_fixture(module, tmp_path)
    )

    proof = module.validate_orion_l1_partition_layout(
        manifest,
        parameters,
        artifact,
        layout_dir,
        checksums,
    )

    assert proof["balance_contract"] == module.ORION_L1_MASS_BALANCE_CONTRACT
    assert proof["exact_quota"] is False
    assert proof["l1_sizes"] == [5, 7]
    assert proof["navigation_mass"]["source"] == module.ORION_L1_MASS_SOURCE
    assert proof["navigation_mass"]["transform"] == (
        module.ORION_L1_MASS_TRANSFORM
    )
    assert proof["navigation_mass"]["partition_masses"] == [60, 60]
    assert proof["navigation_mass"]["mass_limit"] == 71
    assert proof["upper_replay"]["verdict"] == "PASS"


def test_l1_partition_requires_upper_replay_pass_manifest(tmp_path):
    module = load_module()
    manifest, parameters, artifact, layout_dir, checksums = (
        l1_partition_contract_fixture(
            module,
            tmp_path,
            "l1_graph_prepartition",
        )
    )
    replay_manifest_path = Path(
        manifest["provenance"]["upper_replay_manifest"]
    )
    replay_manifest = json.loads(replay_manifest_path.read_text())
    replay_manifest["gates"]["ordered_distance_bits_equal"] = False
    replay_manifest_path.write_text(
        json.dumps(replay_manifest),
        encoding="utf-8",
    )
    replay_manifest_sha256 = module.layout_common.sha256_path(
        replay_manifest_path
    )
    relative = replay_manifest_path.name
    checksums[relative] = replay_manifest_sha256
    manifest["outputs"]["files"][relative] = {
        "sha256": replay_manifest_sha256,
        "size_bytes": replay_manifest_path.stat().st_size,
    }
    manifest["routing"]["l1_partition_diagnostics"][
        "upper_replay_manifest_sha256"
    ] = replay_manifest_sha256

    with pytest.raises(RuntimeError, match="did not pass every required gate"):
        module.validate_orion_l1_partition_layout(
            manifest,
            parameters,
            artifact,
            layout_dir,
            checksums,
        )


def test_l1_partition_rejects_upper_replay_query_checksum_drift(tmp_path):
    module = load_module()
    manifest, parameters, artifact, layout_dir, checksums = (
        l1_partition_contract_fixture(
            module,
            tmp_path,
            "l1_graph_prepartition",
        )
    )
    manifest["provenance"]["upper_replay_queries_sha256"] = "0" * 64

    with pytest.raises(RuntimeError, match="query checksum mismatch"):
        module.validate_orion_l1_partition_layout(
            manifest,
            parameters,
            artifact,
            layout_dir,
            checksums,
        )


def test_l1_navigation_mass_rejects_replayed_load_mismatch(tmp_path):
    module = load_module()
    manifest, parameters, artifact, layout_dir, checksums = (
        l1_navigation_mass_contract_fixture(module, tmp_path)
    )
    manifest["routing"]["l1_partition_diagnostics"]["mass_balance"][
        "estimated_partition_masses"
    ] = [59, 61]

    with pytest.raises(RuntimeError, match="loads differ from self-hit replay"):
        module.validate_orion_l1_partition_layout(
            manifest,
            parameters,
            artifact,
            layout_dir,
            checksums,
        )


def test_l1_navigation_mass_keeps_phase_a_ineligible_until_phase_b(tmp_path):
    module = load_module()
    manifest, parameters, artifact, layout_dir, checksums = (
        l1_navigation_mass_contract_fixture(module, tmp_path)
    )
    screen_path = Path(manifest["provenance"]["screen_manifest"])
    screen = json.loads(screen_path.read_text())
    frozen_path = Path(screen["frozen_candidate_manifest"])
    frozen = json.loads(frozen_path.read_text())
    frozen_candidate = frozen["candidates"][0]
    frozen_candidate["eligibility_pending_post_freeze_query_gates"] = False
    frozen_candidate["topology_all_pass"] = True
    frozen_candidate["materialization_eligible"] = True
    frozen_path.write_text(json.dumps(frozen), encoding="utf-8")
    screen["frozen_candidate_manifest_sha256"] = (
        module.layout_common.sha256_path(frozen_path)
    )
    screen_path.write_text(json.dumps(screen), encoding="utf-8")
    manifest["routing"]["l1_partition_diagnostics"][
        "screen_manifest_sha256"
    ] = module.layout_common.sha256_path(screen_path)

    with pytest.raises(RuntimeError, match="frozen candidate record mismatch"):
        module.validate_orion_l1_partition_layout(
            manifest,
            parameters,
            artifact,
            layout_dir,
            checksums,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("upper_navigation_top_k", 9),
        ("upper_navigation_search_ef", 99),
    ],
)
def test_l1_navigation_mass_binds_fixed_top10_ef100(tmp_path, field, value):
    module = load_module()
    manifest, parameters, artifact, layout_dir, checksums = (
        l1_navigation_mass_contract_fixture(module, tmp_path)
    )
    screen_path = Path(manifest["provenance"]["screen_manifest"])
    screen = json.loads(screen_path.read_text())
    frozen_path = Path(screen["frozen_candidate_manifest"])
    frozen = json.loads(frozen_path.read_text())
    screen["construction_inputs"][field] = value
    frozen["construction_inputs"][field] = value
    frozen_path.write_text(json.dumps(frozen), encoding="utf-8")
    screen["frozen_candidate_manifest_sha256"] = (
        module.layout_common.sha256_path(frozen_path)
    )
    screen_path.write_text(json.dumps(screen), encoding="utf-8")
    manifest["routing"]["l1_partition_diagnostics"][
        "screen_manifest_sha256"
    ] = module.layout_common.sha256_path(screen_path)

    with pytest.raises(RuntimeError, match="construction input binding mismatch"):
        module.validate_orion_l1_partition_layout(
            manifest,
            parameters,
            artifact,
            layout_dir,
            checksums,
        )


def test_l1_navigation_mass_rejects_l0_load_as_partition_input(tmp_path):
    module = load_module()
    manifest, parameters, artifact, layout_dir, checksums = (
        l1_navigation_mass_contract_fixture(module, tmp_path)
    )
    screen_path = Path(manifest["provenance"]["screen_manifest"])
    screen = json.loads(screen_path.read_text())
    screen["contract"]["partitioner_reads_l0_load"] = True
    screen_path.write_text(json.dumps(screen), encoding="utf-8")
    manifest["routing"]["l1_partition_diagnostics"][
        "screen_manifest_sha256"
    ] = module.layout_common.sha256_path(screen_path)

    with pytest.raises(RuntimeError, match="input-isolation contract mismatch"):
        module.validate_orion_l1_partition_layout(
            manifest,
            parameters,
            artifact,
            layout_dir,
            checksums,
        )


@pytest.mark.parametrize(
    ("source", "transform", "mass_mode", "estimator_version"),
    [
        (
            "production_upper_navigation_top10_raw_hit_frequency",
            "raw_hit_count",
            "raw",
            1,
        ),
        (
            "production_upper_navigation_top10_self_debiased_floor1",
            "max(1, raw_hit_count - 1)",
            "self-debiased-floor1",
            1,
        ),
        (
            "production_upper_navigation_top10_hit_frequency_regularized_v2",
            "raw_count_with_unit_l1_prior",
            "raw-regularized-v2",
            1,
        ),
    ],
)
def test_l1_navigation_mass_rejects_legacy_or_version_drift(
    tmp_path,
    source,
    transform,
    mass_mode,
    estimator_version,
):
    module = load_module()
    manifest, parameters, artifact, layout_dir, checksums = (
        l1_navigation_mass_contract_fixture(module, tmp_path)
    )
    mass_balance = manifest["routing"]["l1_partition_diagnostics"][
        "mass_balance"
    ]
    mass_balance["navigation_mass_source"] = source
    mass_balance["navigation_mass_transform"] = transform
    mass_balance["mass_mode"] = mass_mode
    mass_balance["navigation_mass_estimator_version"] = estimator_version

    with pytest.raises(RuntimeError, match="fixed contract mismatch"):
        module.validate_orion_l1_partition_layout(
            manifest,
            parameters,
            artifact,
            layout_dir,
            checksums,
        )


def test_l1_navigation_mass_rejects_non_confirmation_method(tmp_path):
    module = load_module()
    manifest, parameters, artifact, layout_dir, checksums = (
        l1_navigation_mass_contract_fixture(module, tmp_path)
    )
    parameters["l1_partitioner"] = "mass-ldg"
    diagnostics = manifest["routing"]["l1_partition_diagnostics"]
    diagnostics["method"] = "mass-ldg"
    diagnostics["mass_balance"]["method"] = "mass-ldg"

    with pytest.raises(RuntimeError, match="locked v2 confirmation method"):
        module.validate_orion_l1_partition_layout(
            manifest,
            parameters,
            artifact,
            layout_dir,
            checksums,
        )


def test_l1_navigation_mass_rejects_amendment_sha_drift(tmp_path):
    module = load_module()
    manifest, parameters, artifact, layout_dir, checksums = (
        l1_navigation_mass_contract_fixture(module, tmp_path)
    )
    manifest["routing"]["l1_partition_diagnostics"][
        "protocol_amendment_sha256"
    ] = "0" * 64

    with pytest.raises(RuntimeError, match="checksum differs from diagnostics"):
        module.validate_orion_l1_partition_layout(
            manifest,
            parameters,
            artifact,
            layout_dir,
            checksums,
        )


def test_l1_navigation_mass_rejects_owner_parity_drift(tmp_path):
    module = load_module()
    manifest, parameters, artifact, layout_dir, checksums = (
        l1_navigation_mass_contract_fixture(module, tmp_path)
    )
    manifest["routing"]["l1_partition_diagnostics"]["owner_parity"] = {
        **manifest["routing"]["l1_partition_diagnostics"]["owner_parity"],
        "owner_bytes_identical": False,
    }

    with pytest.raises(RuntimeError, match="differs between candidate and diagnostics"):
        module.validate_orion_l1_partition_layout(
            manifest,
            parameters,
            artifact,
            layout_dir,
            checksums,
        )


def test_l1_navigation_mass_rejects_cross_dataset_topology_failure(tmp_path):
    module = load_module()
    manifest, parameters, artifact, layout_dir, checksums = (
        l1_navigation_mass_contract_fixture(module, tmp_path)
    )
    confirmation_path = Path(
        manifest["provenance"]["cross_dataset_confirmation"]
    )
    confirmation = json.loads(confirmation_path.read_text())
    confirmation["datasets"]["glove"]["topology_all_pass"] = False
    confirmation_path.write_text(json.dumps(confirmation), encoding="utf-8")
    manifest["routing"]["l1_partition_diagnostics"][
        "cross_dataset_confirmation_sha256"
    ] = module.layout_common.sha256_path(confirmation_path)
    confirmation_path.with_name(confirmation_path.name + ".sha256").write_text(
        module.layout_common.sha256_path(confirmation_path) + "\n",
        encoding="ascii",
    )

    with pytest.raises(RuntimeError, match="glove confirmation verdict mismatch"):
        module.validate_orion_l1_partition_layout(
            manifest,
            parameters,
            artifact,
            layout_dir,
            checksums,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("use_multi_assign", False),
        ("multi_assign_min_max_vote", 3),
        ("multi_assign_vote_delta", 1),
        ("multi_assign_max_shards", 2),
        ("enable_fission", True),
        ("l0_repair", True),
    ],
)
def test_l1_partition_candidate_rejects_main_idea_or_lightweight_drift(
    tmp_path, field, value
):
    module = load_module()
    manifest, parameters, artifact, layout_dir, checksums = (
        l1_partition_contract_fixture(
            module,
            tmp_path,
            "l1_graph_prepartition",
        )
    )
    parameters[field] = value

    with pytest.raises(RuntimeError, match="lightweight/multi-assignment contract"):
        module.validate_orion_l1_partition_layout(
            manifest,
            parameters,
            artifact,
            layout_dir,
            checksums,
        )


@pytest.mark.parametrize(
    "field",
    [
        "owner_sha256",
        "source_artifact_sha256",
        "attachments_sha256",
        "screen_manifest_sha256",
    ],
)
def test_l1_partition_candidate_requires_canonical_bound_checksums(tmp_path, field):
    module = load_module()
    manifest, parameters, artifact, layout_dir, checksums = (
        l1_partition_contract_fixture(
            module,
            tmp_path,
            "l1_graph_prepartition",
        )
    )
    manifest["routing"]["l1_partition_diagnostics"][field] = "not-a-sha256"

    with pytest.raises(RuntimeError, match=f"{field} is not a valid SHA-256"):
        module.validate_orion_l1_partition_layout(
            manifest,
            parameters,
            artifact,
            layout_dir,
            checksums,
        )


def test_l1_partition_candidate_rejects_ineligible_or_non_graph_scope(tmp_path):
    module = load_module()
    manifest, parameters, artifact, layout_dir, checksums = (
        l1_partition_contract_fixture(
            module,
            tmp_path,
            "l1_graph_prepartition",
        )
    )
    diagnostics = manifest["routing"]["l1_partition_diagnostics"]
    diagnostics["new_algorithm_eligible"] = False
    diagnostics["input_scope"] = "legacy_l0_informed_reference"

    with pytest.raises(RuntimeError, match="eligibility/input-scope contract"):
        module.validate_orion_l1_partition_layout(
            manifest,
            parameters,
            artifact,
            layout_dir,
            checksums,
        )


def test_l1_partition_reference_must_remain_explicitly_ineligible(tmp_path):
    module = load_module()
    manifest, parameters, artifact, layout_dir, checksums = (
        l1_partition_contract_fixture(
            module,
            tmp_path,
            "matched_l0_informed_reference",
        )
    )
    diagnostics = manifest["routing"]["l1_partition_diagnostics"]
    diagnostics["new_algorithm_eligible"] = True
    diagnostics["input_scope"] = "production_upper_graph_only"

    with pytest.raises(RuntimeError, match="eligibility/input-scope contract"):
        module.validate_orion_l1_partition_layout(
            manifest,
            parameters,
            artifact,
            layout_dir,
            checksums,
        )


def test_l1_partition_candidate_rejects_non_exact_owner(tmp_path):
    module = load_module()
    manifest, parameters, artifact, layout_dir, checksums = (
        l1_partition_contract_fixture(
            module,
            tmp_path,
            "matched_l0_informed_reference",
        )
    )
    parameters["balance_mode"] = "l1_graph_prepartition"
    diagnostics = manifest["routing"]["l1_partition_diagnostics"]
    diagnostics["balance_contract"] = module.ORION_L1_UNIT_BALANCE_CONTRACT
    diagnostics["input_scope"] = "production_upper_graph_only"
    diagnostics["new_algorithm_eligible"] = True
    screen_path = Path(manifest["provenance"]["screen_manifest"])
    screen_path.write_text(
        json.dumps(
            {
                "format_version": 1,
                "contract": {
                    "partition_input_scope": "production_upper_graph_only",
                    "partitioner_reads_full_attachments": False,
                    "partitioner_reads_l0_load": False,
                    "partitioner_reads_multi_assignment_state": False,
                    "l0_assignment_after_owner_freeze": True,
                    "l0_repair_or_flow": False,
                    "multi_assignment": {
                        "enabled": True,
                        "min_max_vote": 2,
                        "vote_delta": 0,
                        "max_shards": 0,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    diagnostics["screen_manifest_sha256"] = module.layout_common.sha256_path(
        screen_path
    )

    with pytest.raises(RuntimeError, match="must satisfy exact quota"):
        module.validate_orion_l1_partition_layout(
            manifest,
            parameters,
            artifact,
            layout_dir,
            checksums,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("shard_counts", "do not sum to physical_point_count"),
        ("l1_sizes", "owner counts differ from l1_sizes"),
        ("copy_histogram", "disagrees with logical/physical point counts"),
    ],
)
def test_l1_partition_layout_rejects_inconsistent_counts(
    tmp_path, mutation, message
):
    module = load_module()
    manifest, parameters, artifact, layout_dir, checksums = (
        l1_partition_contract_fixture(
            module,
            tmp_path,
            "l1_graph_prepartition",
        )
    )
    if mutation == "shard_counts":
        manifest["routing"]["shard_counts"] = [5, 4]
    elif mutation == "l1_sizes":
        manifest["routing"]["l1_partition_diagnostics"]["l1_sizes"] = [3, 1]
    else:
        manifest["routing"]["copy_count_histogram"] = {"1": 5, "2": 1}

    with pytest.raises(RuntimeError, match=message):
        module.validate_orion_l1_partition_layout(
            manifest,
            parameters,
            artifact,
            layout_dir,
            checksums,
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
@pytest.mark.parametrize("deferred", [False, True])
def test_routed_prepare_runs_importer_and_matching_installer(
    monkeypatch,
    tmp_path,
    method,
    installer,
    deferred,
):
    module = load_module()
    extra = ["--resume"]
    if deferred:
        extra.append("--defer-artifact-install")
    args = args_for(module, tmp_path, method, *extra)
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
    monkeypatch.setattr(
        module,
        "load_routed_layout",
        lambda *_args, **_kwargs: layout,
    )
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
    manifest = json.loads(manifest_path.read_text())
    assert manifest["method"] == method
    if deferred:
        assert len(commands) == 1
        assert len(manifest["commands"]) == 1
        assert manifest["artifact_installation"]["status"] == "deferred"
        assert manifest["standard_api_smoke"]["status"] == "DEFERRED"
        assert manifest["standard_api_smoke"]["query_issued"] is False
    else:
        installer_command = commands[1][0]
        assert installer in installer_command
        assert installer_command[-2:] == ["--restart", "workers-first"]
        assert len(manifest["commands"]) == 2
        assert (
            manifest["artifact_installation"]["status"]
            == "installed_and_activated"
        )
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
    monkeypatch.setattr(
        module,
        "load_routed_layout",
        lambda *_args, **_kwargs: layout,
    )
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
        "routing": {
            "logical_point_count": 2,
            "physical_point_count": 2,
            "shard_counts": [1, 1],
        },
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
    assert proof["shard_weights"] == [1, 1]

    build_manifest["routing"]["shard_counts"] = [2, 1]
    (layout_dir / module.layout_common.BUILD_MANIFEST_NAME).write_text(
        json.dumps(build_manifest),
        encoding="utf-8",
    )
    (layout_dir / module.layout_common.CHECKSUMS_NAME).unlink()
    module.layout_common.write_checksums(layout_dir)
    with pytest.raises(RuntimeError, match="do not sum to physical_point_count"):
        module.load_routed_layout("simple_kmeans", layout_dir)


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
