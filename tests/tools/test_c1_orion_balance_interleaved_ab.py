from __future__ import annotations

import ast
import dataclasses
import importlib.util
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "experiments/c1/scripts/c1_orion_balance_interleaved_ab.py"


def load_module():
    name = "c1_orion_balance_interleaved_ab_test"
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_paired_order_alternates_odd_a_b_even_b_a():
    module = load_module()
    assert module.paired_order(1) == ("A", "B")
    assert module.paired_order(2) == ("B", "A")
    assert module.paired_order(7) == ("A", "B")
    with pytest.raises(ValueError, match="positive"):
        module.paired_order(0)


def test_adoption_gate_requires_checksum_bound_prerequisite_pass():
    module = load_module()
    common = {
        "stable": True,
        "recall_failures": {},
        "ratio_ci_lower": 1.01,
    }
    assert module.adoption_gate_passes(
        **common, adoption_prerequisites=None
    ) is True
    assert module.adoption_gate_passes(
        **common, adoption_prerequisites={"status": "PASS"}
    ) is True
    assert module.adoption_gate_passes(
        **common, adoption_prerequisites={"status": "FAIL"}
    ) is False
    assert module.adoption_gate_passes(
        **common, adoption_prerequisites={}
    ) is False


def test_independent_saturation_uses_one_shared_counterbalanced_grid():
    module = load_module()
    sweep = [
        {"concurrency": 1, "A": {"qps": 100.0}, "B": {"qps": 101.0}},
        {"concurrency": 2, "A": {"qps": 150.0}, "B": {"qps": 149.0}},
        {"concurrency": 4, "A": {"qps": 151.0}, "B": {"qps": 150.0}},
        {"concurrency": 8, "A": {"qps": 140.0}, "B": {"qps": 139.0}},
    ]
    selected = module.select_independent_saturation(sweep)
    assert selected["selected_concurrency_by_arm"] == {"A": 2, "B": 2}
    assert selected["common_eligible_concurrencies"] == [2, 4]
    assert selected["all_knees_observed"] is True
    assert {
        selected["independent_arm_selections"][label]["selected_concurrency"]
        for label in ("A", "B")
    } == {2}


def test_independent_saturation_allows_different_arm_knees():
    module = load_module()
    sweep = [
        {"concurrency": 1, "A": {"qps": 100.0}, "B": {"qps": 50.0}},
        {"concurrency": 2, "A": {"qps": 50.0}, "B": {"qps": 100.0}},
        {"concurrency": 4, "A": {"qps": 40.0}, "B": {"qps": 80.0}},
    ]
    selected = module.select_independent_saturation(sweep)
    assert selected["selected_concurrency_by_arm"] == {"A": 1, "B": 2}
    assert selected["common_eligible_concurrencies"] == []
    assert selected["all_knees_observed"] is True


def test_formal_pairs_stop_at_five_when_both_qps_cvs_pass():
    module = load_module()
    execution: list[tuple[int, int, str]] = []

    def runner(label, pair_number, position):
        execution.append((pair_number, position, label))
        base = 1_000.0 if label == "A" else 1_100.0
        return {"qps": base + pair_number, "wall_s": 20.0, "query_count": 20_000}

    result = module.run_formal_pairs(runner, {"A": 0.901, "B": 0.904})
    assert result["extended_to_seven"] is False
    assert result["stable"] is True
    assert len(result["pairs"]) == 5
    assert execution == [
        (1, 1, "A"),
        (1, 2, "B"),
        (2, 1, "B"),
        (2, 2, "A"),
        (3, 1, "A"),
        (3, 2, "B"),
        (4, 1, "B"),
        (4, 2, "A"),
        (5, 1, "A"),
        (5, 2, "B"),
    ]
    rows = module.paired_round_csv_rows(result["pairs"])
    assert rows[0]["a_recall_at_10"] == 0.901
    assert rows[0]["b_recall_at_10"] == 0.904
    assert rows[0]["paired_ratio_b_over_a"] == pytest.approx(1101 / 1001)
    assert rows[-1]["a_running_qps_cv"] < 0.05
    assert rows[-1]["b_running_qps_cv"] < 0.05


def test_formal_pairs_extend_directly_to_seven_if_either_five_run_cv_is_high():
    module = load_module()
    a_qps = [100.0, 110.0, 100.0, 110.0, 100.0, 105.0, 105.0]

    def runner(label, pair_number, position):
        del position
        return {
            "qps": a_qps[pair_number - 1] if label == "A" else 200.0,
            "wall_s": 20.0,
        }

    result = module.run_formal_pairs(runner, {"A": 0.9, "B": 0.91})
    assert result["five_pair_stats"]["A"]["cv"] > 0.05
    assert result["extended_to_seven"] is True
    assert len(result["pairs"]) == 7
    assert result["final_stats"]["A"]["cv"] < 0.05
    assert result["stable"] is True
    assert result["pairs"][5]["execution_order"] == ["B", "A"]
    assert result["pairs"][6]["execution_order"] == ["A", "B"]


def write_bound_arm(tmp_path: Path, module, *, multi_assign: bool = True):
    deployment = tmp_path / "deployment.json"
    deployment.write_text('{"run":"synthetic"}\n', encoding="utf-8")
    topology = tmp_path / "topology.json"
    topology.write_text('{"nodes":[]}\n', encoding="utf-8")
    layout = tmp_path / "layout"
    layout.mkdir()
    artifact = layout / "generation-7.json"
    artifact.write_text('{"format_version":1}\n', encoding="utf-8")
    build_manifest = {
        "parameters": {"use_multi_assign": multi_assign},
        "outputs": {"production_artifact": artifact.name},
        "dataset": {"sha256": "d" * 64},
        "routing": {"effective_num_shards": 32},
    }
    build_path = layout / "build-manifest.json"
    build_path.write_text(json.dumps(build_manifest), encoding="utf-8")
    prepare = {
        "method": "orion",
        "collection": "prepared_a",
        "base_url": "http://controller:6333",
        "deployment_manifest": {
            "path": str(deployment),
            "sha256": module.sha256_path(deployment),
        },
        "topology": {
            "path": str(topology),
            "sha256": module.sha256_path(topology),
        },
        "checksums": {
            "routing_artifact_sha256": module.sha256_path(artifact),
            "layout_build_manifest_sha256": module.sha256_path(build_path),
            "topology_sha256": module.sha256_path(topology),
            "deployment_manifest_sha256": module.sha256_path(deployment),
        },
        "layout": {
            "layout_dir": str(layout),
            "build_manifest_path": str(build_path),
            "build_manifest_sha256": module.sha256_path(build_path),
            "artifact_path": str(artifact),
            "artifact_sha256": module.sha256_path(artifact),
        },
        "provenance_metadata": {
            "native_auto_shard_prepare": {
                "schema_version": 2,
                "provenance": {"method": "orion"},
                "provenance_sha256": "e" * 64,
            }
        },
    }
    prepare_path = tmp_path / "preparation_manifest.json"
    prepare_path.write_text(json.dumps(prepare), encoding="utf-8")
    arm = module.ArmSpec(
        label="A",
        name="original",
        collection="prepared_a",
        artifact=artifact,
        layout_dir=layout,
        prepare_manifest=prepare_path,
        allow_balance_layout=False,
        allow_scaling_layout=False,
        allow_l1_partition_layout=False,
        allow_historical_prepare_deployment=False,
    )
    return deployment, topology, arm


def test_runner_accepts_exact_native_cnbr_l1_tool(monkeypatch, tmp_path):
    module = load_module()
    _deployment, _topology, base_arm = write_bound_arm(tmp_path, module)
    arm = dataclasses.replace(base_arm, allow_l1_partition_layout=True)
    build_manifest = {
        "tool": module.native.prepare.ORION_NATIVE_CNBR_LAYOUT_TOOL,
    }
    expected = {
        "layout_dir": str(arm.layout_dir),
        "build_manifest_path": str(arm.layout_dir / "build-manifest.json"),
        "build_manifest_sha256": "a" * 64,
        "import_manifest_path": str(arm.layout_dir / "import.json"),
        "import_manifest_sha256": "b" * 64,
        "artifact_path": str(arm.artifact),
        "l1_partition_layout_proof": {"arm": "C_CNBR"},
        "attachment_search_ef": 100,
        "logical_point_count": 4,
        "physical_point_count": 5,
        "shard_count": 2,
    }
    monkeypatch.setattr(
        module.native.prepare,
        "load_routed_layout",
        lambda *args, **kwargs: expected,
    )

    proof = module.validate_artifact_bundle_for_arm(
        arm,
        {"build_manifest": build_manifest},
        {},
    )

    assert proof["l1_partition_layout_proof"] == {"arm": "C_CNBR"}
    assert proof["attachment_search_ef"] == 100


def test_runner_rejects_unknown_tool_as_l1_layout(tmp_path):
    module = load_module()
    _deployment, _topology, base_arm = write_bound_arm(tmp_path, module)
    arm = dataclasses.replace(base_arm, allow_l1_partition_layout=True)
    with pytest.raises(RuntimeError, match="authorization mismatch"):
        module.validate_artifact_bundle_for_arm(
            arm,
            {"build_manifest": {"tool": "unknown/materializer.py"}},
            {},
        )


def test_prepare_binding_checksum_binds_collection_artifact_layout_and_prepare(tmp_path):
    module = load_module()
    deployment, topology, arm = write_bound_arm(tmp_path, module)
    proof = module.validate_prepare_binding(
        arm,
        base_url="http://controller:6333/",
        deployment_manifest=deployment,
        topology=topology,
    )
    assert proof["status"] == "PASS"
    assert proof["collection"] == "prepared_a"
    assert proof["artifact_sha256"] == module.sha256_path(arm.artifact)
    assert proof["prepare_manifest_sha256"] == module.sha256_path(
        arm.prepare_manifest
    )
    assert proof["parameters"]["use_multi_assign"] is True


def test_prepare_binding_rejects_cancelling_multi_assignment(tmp_path):
    module = load_module()
    deployment, topology, arm = write_bound_arm(
        tmp_path, module, multi_assign=False
    )
    with pytest.raises(RuntimeError, match="disables multi-assignment"):
        module.validate_prepare_binding(
            arm,
            base_url="http://controller:6333",
            deployment_manifest=deployment,
            topology=topology,
        )


def test_prepare_binding_accepts_explicit_single_assignment_layout(tmp_path):
    module = load_module()
    deployment, topology, arm = write_bound_arm(
        tmp_path, module, multi_assign=False
    )
    arm = dataclasses.replace(arm, allow_single_assignment_layout=True)
    proof = module.validate_prepare_binding(
        arm,
        base_url="http://controller:6333",
        deployment_manifest=deployment,
        topology=topology,
    )
    assert proof["status"] == "PASS"
    assert proof["parameters"]["use_multi_assign"] is False


def test_prepare_binding_supports_explicit_historical_deployment_for_baseline(
    tmp_path,
):
    module = load_module()
    deployment, topology, arm = write_bound_arm(tmp_path, module)
    deployment.write_text('{"run":"new-live-deployment"}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="historical"):
        module.validate_prepare_binding(
            arm,
            base_url="http://controller:6333",
            deployment_manifest=deployment,
            topology=topology,
        )
    historical_arm = dataclasses.replace(
        arm, allow_historical_prepare_deployment=True
    )
    proof = module.validate_prepare_binding(
        historical_arm,
        base_url="http://controller:6333",
        deployment_manifest=deployment,
        topology=topology,
    )
    assert proof["prepare_deployment_binding"]["mode"] == "historical"
    assert proof["prepare_deployment_binding"]["explicitly_allowed"] is True
    assert proof["prepare_deployment_binding"]["recorded_digest_reverified"] is False
    assert (
        proof["prepare_deployment_binding"]["historical_digest_authority"]
        == "prepare_manifest_attestation_only"
    )


def test_prepare_binding_supports_explicit_historical_topology_for_baseline(
    tmp_path,
):
    module = load_module()
    deployment, recorded_topology, arm = write_bound_arm(tmp_path, module)
    current_topology = tmp_path / "current-topology.json"
    current_topology.write_text('{"nodes":[{"peer_id":1}]}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="prepare/topology binding is historical"):
        module.validate_prepare_binding(
            arm,
            base_url="http://controller:6333",
            deployment_manifest=deployment,
            topology=current_topology,
        )
    historical_arm = dataclasses.replace(
        arm, allow_historical_prepare_deployment=True
    )
    proof = module.validate_prepare_binding(
        historical_arm,
        base_url="http://controller:6333",
        deployment_manifest=deployment,
        topology=current_topology,
    )
    assert proof["prepare_deployment_binding"]["mode"] == "current"
    assert proof["prepare_topology_binding"] == {
        "mode": "historical",
        "recorded_path": str(recorded_topology.resolve()),
        "recorded_sha256": module.sha256_path(recorded_topology),
        "current_path": str(current_topology.resolve()),
        "current_sha256": module.sha256_path(current_topology),
        "explicitly_allowed": True,
        "recorded_digest_reverified": True,
        "historical_digest_authority": "recorded_file_reverified",
    }


def test_only_arm_a_can_enable_historical_prepare_exception():
    module = load_module()
    required = [
        "--base-url",
        "http://controller:6333",
        "--hdf5-path",
        "dataset.hdf5",
        "--topology",
        "topology.json",
        "--deployment-manifest",
        "deployment.json",
        "--output-dir",
        "output",
        "--collection-a",
        "a",
        "--collection-b",
        "b",
        "--artifact-a",
        "layout-a/generation-a.json",
        "--artifact-b",
        "layout-b/generation-b.json",
        "--layout-dir-a",
        "layout-a",
        "--layout-dir-b",
        "layout-b",
        "--prepare-manifest-a",
        "prepare-a.json",
        "--prepare-manifest-b",
        "prepare-b.json",
    ]
    args = module.parse_args(
        [*required, "--allow-historical-prepare-deployment-a"]
    )
    arms = module.arm_specs(args)
    assert arms["A"].allow_historical_prepare_deployment is True
    assert arms["B"].allow_historical_prepare_deployment is False
    with pytest.raises(SystemExit):
        module.parse_args(
            [*required, "--allow-historical-prepare-deployment-b"]
        )


def test_paired_ratio_confidence_interval_requires_lower_bound_above_one():
    module = load_module()
    passing = module.paired_ratio_confidence_interval(
        [1.10, 1.11, 1.09, 1.12, 1.10]
    )
    assert passing["lower"] > 1.0
    crossing = module.paired_ratio_confidence_interval(
        [0.95, 1.10, 1.05, 0.98, 1.08]
    )
    assert crossing["lower"] < 1.0 < crossing["upper"]


def test_execution_source_bundle_hashes_file_bytes_not_git_status(tmp_path):
    module = load_module()
    source = tmp_path / "untracked_runner.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    before = module.source_bundle_for_paths([source])
    source.write_text("VALUE = 2\n", encoding="utf-8")
    after = module.source_bundle_for_paths([source])
    assert before["sha256"] != after["sha256"]
    assert before["files"][0]["sha256"] != after["files"][0]["sha256"]


def repository_snapshot_fixture(
    module,
    files: dict[str, tuple[str, int]],
    *,
    commit: str = "a" * 40,
    status_sha256: str = "b" * 64,
):
    entries = [
        {"path": path, "sha256": sha256, "size_bytes": size_bytes}
        for path, (sha256, size_bytes) in sorted(files.items())
    ]
    bundle = {
        "algorithm": "sha256",
        "scope": "loaded_repo_local_python_sources",
        "files": entries,
        "sha256": module.canonical_sha256(entries),
    }
    return {
        "commit": commit,
        "status_porcelain_sha256": status_sha256,
        "execution_source_bundle": bundle,
        "execution_source_bundle_sha256": bundle["sha256"],
    }


def test_repository_preflight_transition_allows_newly_loaded_module_only():
    module = load_module()
    common = {
        "experiments/c1/scripts/c1_orion_balance_interleaved_ab.py": (
            "1" * 64,
            100,
        ),
        "tools/orion_native_layout.py": ("2" * 64, 200),
    }
    before = repository_snapshot_fixture(module, common)
    after = repository_snapshot_fixture(
        module,
        {
            **common,
            "experiments/c1/scripts/lazily_loaded_preflight.py": (
                "3" * 64,
                300,
            ),
        },
    )

    proof = module.validate_repository_preflight_transition(before, after)

    assert proof["status"] == "PASS"
    assert proof["added_source_paths"] == [
        "experiments/c1/scripts/lazily_loaded_preflight.py"
    ]
    assert proof["common_source_path_count"] == len(common)


@pytest.mark.parametrize("drift", ["deleted", "hash", "size"])
def test_repository_preflight_transition_rejects_existing_source_drift(drift):
    module = load_module()
    files = {
        "experiments/c1/scripts/common.py": ("1" * 64, 100),
        "experiments/c1/scripts/runner.py": ("2" * 64, 200),
    }
    after_files = dict(files)
    if drift == "deleted":
        del after_files["experiments/c1/scripts/common.py"]
    elif drift == "hash":
        after_files["experiments/c1/scripts/common.py"] = ("3" * 64, 100)
    else:
        after_files["experiments/c1/scripts/common.py"] = ("1" * 64, 101)
    before = repository_snapshot_fixture(module, files)
    after = repository_snapshot_fixture(module, after_files)

    with pytest.raises(RuntimeError):
        module.validate_repository_preflight_transition(before, after)


def test_repository_preflight_transition_rejects_loaded_source_removal():
    module = load_module()
    before = repository_snapshot_fixture(
        module,
        {
            "experiments/c1/scripts/common.py": ("1" * 64, 100),
            "experiments/c1/scripts/runner.py": ("2" * 64, 200),
        },
    )
    after = repository_snapshot_fixture(
        module,
        {"experiments/c1/scripts/runner.py": ("2" * 64, 200)},
    )

    with pytest.raises(RuntimeError):
        module.validate_repository_preflight_transition(before, after)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("commit", "c" * 40),
        ("status_porcelain_sha256", "d" * 64),
    ],
)
def test_repository_preflight_transition_rejects_git_state_drift(
    field,
    replacement,
):
    module = load_module()
    files = {"experiments/c1/scripts/common.py": ("1" * 64, 100)}
    before = repository_snapshot_fixture(module, files)
    after = repository_snapshot_fixture(module, files)
    after[field] = replacement

    with pytest.raises(RuntimeError):
        module.validate_repository_preflight_transition(before, after)


def test_repository_measurement_transition_accepts_exact_source_snapshot():
    module = load_module()
    files = {
        "experiments/c1/scripts/c1_orion_balance_interleaved_ab.py": (
            "1" * 64,
            100,
        ),
        "tools/orion_native_layout.py": ("2" * 64, 200),
    }
    start = repository_snapshot_fixture(module, files)
    end = repository_snapshot_fixture(module, files)

    proof = module.validate_repository_measurement_transition(start, end)

    assert proof["status"] == "PASS"
    assert proof["source_path_count"] == len(files)


@pytest.mark.parametrize("drift", ["added", "deleted", "hash", "size"])
def test_repository_measurement_transition_rejects_source_bundle_drift(drift):
    module = load_module()
    files = {
        "experiments/c1/scripts/common.py": ("1" * 64, 100),
        "experiments/c1/scripts/runner.py": ("2" * 64, 200),
    }
    end_files = dict(files)
    if drift == "added":
        end_files["experiments/c1/scripts/late.py"] = ("3" * 64, 300)
    elif drift == "deleted":
        del end_files["experiments/c1/scripts/common.py"]
    elif drift == "hash":
        end_files["experiments/c1/scripts/common.py"] = ("4" * 64, 100)
    else:
        end_files["experiments/c1/scripts/common.py"] = ("1" * 64, 101)

    start = repository_snapshot_fixture(module, files)
    end = repository_snapshot_fixture(module, end_files)

    with pytest.raises(RuntimeError):
        module.validate_repository_measurement_transition(start, end)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("commit", "c" * 40),
        ("status_porcelain_sha256", "d" * 64),
    ],
)
def test_repository_measurement_transition_rejects_git_state_drift(
    field,
    replacement,
):
    module = load_module()
    files = {"experiments/c1/scripts/common.py": ("1" * 64, 100)}
    start = repository_snapshot_fixture(module, files)
    end = repository_snapshot_fixture(module, files)
    end[field] = replacement

    with pytest.raises(RuntimeError):
        module.validate_repository_measurement_transition(start, end)


@pytest.mark.parametrize(
    "validator_name",
    [
        "validate_repository_preflight_transition",
        "validate_repository_measurement_transition",
    ],
)
@pytest.mark.parametrize(
    "invalid",
    [
        "schema",
        "duplicate_path",
        "noncanonical_path",
        "bundle_digest",
        "summary_digest",
    ],
)
def test_repository_transition_rejects_invalid_source_bundle(
    validator_name,
    invalid,
):
    module = load_module()
    files = {"experiments/c1/scripts/common.py": ("1" * 64, 100)}
    before = repository_snapshot_fixture(module, files)
    after = repository_snapshot_fixture(module, files)
    if invalid == "schema":
        after["execution_source_bundle"]["algorithm"] = "sha1"
    elif invalid == "duplicate_path":
        entries = after["execution_source_bundle"]["files"]
        entries.append(dict(entries[0]))
        digest = module.canonical_sha256(entries)
        after["execution_source_bundle"]["sha256"] = digest
        after["execution_source_bundle_sha256"] = digest
    elif invalid == "noncanonical_path":
        entries = after["execution_source_bundle"]["files"]
        entries[0]["path"] = "experiments/c1/scripts/../common.py"
        digest = module.canonical_sha256(entries)
        after["execution_source_bundle"]["sha256"] = digest
        after["execution_source_bundle_sha256"] = digest
    elif invalid == "bundle_digest":
        after["execution_source_bundle"]["sha256"] = "e" * 64
        after["execution_source_bundle_sha256"] = "e" * 64
    else:
        after["execution_source_bundle_sha256"] = "f" * 64

    validator = getattr(module, validator_name)
    with pytest.raises(RuntimeError):
        validator(before, after)


def test_repo_local_execution_source_bundle_contains_shared_engine():
    module = load_module()
    bundle = module.repo_local_execution_source_bundle()
    paths = {row["path"] for row in bundle["files"]}
    assert "experiments/c1/scripts/c1_orion_balance_interleaved_ab.py" in paths
    assert bundle["scope"] == "loaded_repo_local_python_sources"


def test_four_host_64_cpu_resource_contract_is_validation_only():
    module = load_module()
    topology = {
        "hnsw_graph_build_seed": 20260821,
        "hardware_reporting": True,
        "controller": {
            "ssh_host": "localhost",
            "private_ip": "10.10.1.1",
            "max_search_threads": 16,
        },
        "workers": [
            {
                "ssh_host": f"10.10.1.{index}",
                "private_ip": f"10.10.1.{index}",
                "max_search_threads": 16,
            }
            for index in range(2, 5)
        ],
    }
    identities = [topology["controller"], *topology["workers"]]
    image_id = "sha256:" + "a" * 64
    image_tag = "orion-c1:test"
    deployment = {
        "manifest_sha256": "b" * 64,
        "image": {"id": image_id, "tag": image_tag},
        "nodes": [
            {
                "private_ip": identity["private_ip"],
                "container_name": f"qdrant-{index}",
                "image_id": image_id,
            }
            for index, identity in enumerate(identities)
        ],
        "orion_compact_wire": {"current_version": "2"},
        "peer_premerge": {
            "current_mode": "enabled",
            "current_shards_per_rpc": "all",
        },
    }
    rows = []
    for index, identity in enumerate(identities):
        environment = [
            "QDRANT_HNSW_GRAPH_BUILD_SEED=20260821",
            "QDRANT__SERVICE__HARDWARE_REPORTING=true",
            "QDRANT__STORAGE__PERFORMANCE__MAX_SEARCH_THREADS=16",
        ]
        if index == 0:
            environment.append("QDRANT_ORION_COMPACT_WIRE_VERSION=2")
        rows.append({
            **identity,
            "container": f"qdrant-{index}",
            "container_id": f"container-{index}",
            "image": image_tag,
            "image_id": image_id,
            "cpuset": module.hashall.QDRANT_CPUSET,
            "runtime_affinity": module.hashall.QDRANT_CPUSET,
            "nano_cpus": 16_000_000_000,
            "environment": environment,
        })
    proof = module.validate_64_cpu_resource_contract(rows, topology, deployment)
    assert proof["server_cpus_total"] == 64.0
    assert proof["live_deployment_attestation"]["status"] == "PASS"
    rows[0]["cpuset"] = "0-6,16-23"
    with pytest.raises(RuntimeError, match="64-CPU"):
        module.validate_64_cpu_resource_contract(rows, topology, deployment)
    duplicate_rows = [dict(rows[1]) for _ in range(4)]
    with pytest.raises(RuntimeError, match="attestation"):
        module.validate_64_cpu_resource_contract(
            duplicate_rows, topology, deployment
        )


def test_live_deployment_attestation_rejects_wrong_image_and_environment():
    module = load_module()
    topology = {
        "hnsw_graph_build_seed": 7,
        "hardware_reporting": True,
        "controller": {
            "ssh_host": "localhost",
            "private_ip": "10.10.1.1",
            "max_search_threads": 16,
        },
        "workers": [
            {
                "ssh_host": f"10.10.1.{index}",
                "private_ip": f"10.10.1.{index}",
                "max_search_threads": 16,
            }
            for index in range(2, 5)
        ],
    }
    image_id = "sha256:" + "c" * 64
    nodes = [topology["controller"], *topology["workers"]]
    deployment = {
        "manifest_sha256": "d" * 64,
        "image": {"id": image_id, "tag": "orion:test"},
        "nodes": [
            {
                "private_ip": node["private_ip"],
                "container_name": f"container-{index}",
                "image_id": image_id,
            }
            for index, node in enumerate(nodes)
        ],
        "orion_compact_wire": {"current_version": "2"},
        "peer_premerge": {
            "current_mode": "enabled",
            "current_shards_per_rpc": "all",
        },
    }
    rows = []
    for index, node in enumerate(nodes):
        environment = [
            "QDRANT_HNSW_GRAPH_BUILD_SEED=7",
            "QDRANT__SERVICE__HARDWARE_REPORTING=true",
            "QDRANT__STORAGE__PERFORMANCE__MAX_SEARCH_THREADS=16",
        ]
        if index == 0:
            environment.append("QDRANT_ORION_COMPACT_WIRE_VERSION=2")
        rows.append(
            {
                **node,
                "container": f"container-{index}",
                "container_id": f"id-{index}",
                "image": "orion:test",
                "image_id": image_id,
                "environment": environment,
            }
        )
    rows[2]["image_id"] = "sha256:" + "e" * 64
    with pytest.raises(RuntimeError, match="attestation"):
        module.validate_live_deployment_attestation(rows, topology, deployment)
    rows[2]["image_id"] = image_id
    rows[3]["environment"] = [
        value
        for value in rows[3]["environment"]
        if not value.startswith("QDRANT_HNSW_GRAPH_BUILD_SEED=")
    ]
    with pytest.raises(RuntimeError, match="attestation"):
        module.validate_live_deployment_attestation(rows, topology, deployment)


def test_round_robin_contract_requires_same_p32_map_on_four_hosts():
    module = load_module()
    peer_order = [100, 101, 102, 103]
    placement = {
        str(shard): peer_order[shard % len(peer_order)] for shard in range(32)
    }
    bindings = {
        label: {
            "prepare": {
                "placement_plan": {
                    "strategy": "round_robin",
                    "target_placement": placement,
                },
                "placement_peers": {
                    "mode": "all_peers",
                    "peer_ids": peer_order,
                },
                "final_placement_proof": {"placement_mode": "round_robin"},
            }
        }
        for label in ("A", "B")
    }
    live = {
        label: {"shard_count": 32, "expected_placement": placement}
        for label in ("A", "B")
    }
    cluster_preflight = {
        "controller_peer_id": peer_order[0],
        "worker_peer_ids": peer_order[1:],
    }
    proof = module.validate_round_robin_four_host_contract(
        bindings, live, cluster_preflight
    )
    assert proof["shards_per_host"] == 8
    live["B"] = {
        "shard_count": 32,
        "expected_placement": {
            **placement,
            "31": 100,
        },
    }
    with pytest.raises(RuntimeError, match="round-robin"):
        module.validate_round_robin_four_host_contract(
            bindings, live, cluster_preflight
        )


def test_round_robin_contract_rejects_balanced_but_blocked_placement():
    module = load_module()
    peer_order = [100, 101, 102, 103]
    blocked = {str(shard): peer_order[shard // 8] for shard in range(32)}
    bindings = {
        label: {
            "prepare": {
                "placement_plan": {
                    "strategy": "round_robin",
                    "target_placement": blocked,
                },
                "placement_peers": {
                    "mode": "all_peers",
                    "peer_ids": peer_order,
                },
                "final_placement_proof": {"placement_mode": "round_robin"},
            }
        }
        for label in ("A", "B")
    }
    live = {
        label: {"shard_count": 32, "expected_placement": blocked}
        for label in ("A", "B")
    }
    with pytest.raises(RuntimeError, match="round-robin"):
        module.validate_round_robin_four_host_contract(
            bindings,
            live,
            {"controller_peer_id": 100, "worker_peer_ids": [101, 102, 103]},
        )


def test_synthetic_dataset_uses_disjoint_1000_and_9000_query_splits(tmp_path):
    module = load_module()
    dataset = tmp_path / "synthetic.hdf5"
    with h5py.File(dataset, "w") as handle:
        handle.attrs["distance"] = "angular"
        handle.create_dataset("train", data=np.ones((4, 2), dtype=np.float32))
        tests = np.arange(20_000, dtype=np.float32).reshape(10_000, 2) + 1.0
        handle.create_dataset("test", data=tests)
        truth = np.tile(np.arange(10, dtype=np.int64), (10_000, 1))
        handle.create_dataset("neighbors", data=truth)
    loaded = module.load_protocol_dataset(dataset)
    assert loaded.tuning_queries.shape == (1_000, 2)
    assert loaded.heldout_queries.shape == (9_000, 2)
    assert loaded.tuning_neighbors.shape == (1_000, 10)
    assert loaded.heldout_neighbors.shape == (9_000, 10)
    assert not np.shares_memory(loaded.tuning_queries, loaded.heldout_queries)
    assert loaded.tuning_query_sha256 != loaded.heldout_query_sha256


def test_runner_source_contains_no_cluster_or_resource_mutator_calls():
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    forbidden = {
        "apply_resource_schedule",
        "update_container_resources",
        "delete_collection_if_exists",
        "move_numeric_shards_explicit",
        "install_profile_artifact",
        "run_prepare",
        "prepare_collection",
        "activate_artifact",
    }
    called = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    called.update(
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    )
    assert called.isdisjoint(forbidden)
