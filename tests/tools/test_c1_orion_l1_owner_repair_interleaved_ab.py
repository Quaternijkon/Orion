from __future__ import annotations

import argparse
import ast
import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = (
    REPO_ROOT
    / "experiments/c1/scripts/c1_orion_l1_owner_repair_interleaved_ab.py"
)


def load_module():
    name = "c1_orion_l1_owner_repair_interleaved_ab_test"
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def common_parameters() -> dict[str, object]:
    return {
        "sample_denominator": 32,
        "upper_sample_seed": 100,
        "upper_m": 32,
        "upper_ef_construction": 100,
        "attachment_search_ef": 100,
        "upper_graph_seed": 100,
        "k_overlap": 10,
        "use_multi_assign": True,
        "multi_assign_min_max_vote": 2,
        "multi_assign_vote_delta": 0,
        "multi_assign_max_shards": 0,
        "upper_k": 48,
        "upper_search_ef": 48,
        "dynamic_ef_base": 50,
        "dynamic_ef_factor": 14,
        "initial_num_shards": 32,
        "enable_fission": False,
        "enable_topology_refinement": False,
        "l0_repair": False,
        "balance_mode": "caller-specific",
    }


def diagnostics(
    variant: str,
    *,
    owner_sha256: str,
    phase_a_sha256: str,
    attachments_sha256: str,
    parent_owner_sha256: str | None = None,
    parent_manifest_sha256: str | None = None,
    invocation_overrides: dict[str, int] | None = None,
) -> dict[str, object]:
    counts = {
        "full_attachment_reads": 0,
        "point_to_l1s_reads": 0,
        "multi_assignment_reads": 0,
        "weight_recalibration_calls": 0,
        "fission_calls": 0,
        "topology_refinement_calls": 0,
        "capacity_flow_calls": 0,
        "lower_layout_repair_calls": 0,
    }
    counts.update(invocation_overrides or {})
    result: dict[str, object] = {
        "balance_contract": {"variant": variant},
        "phase_a_manifest_sha256": phase_a_sha256,
        "phase_a_owner_sha256": owner_sha256,
        "owner_sha256": owner_sha256,
        "attachments_sha256": attachments_sha256,
        "forbidden_stage_invocations": counts,
        "multi_assignment_after_owner_freeze": True,
    }
    if parent_owner_sha256 is not None:
        result["parent_n_owner_sha256"] = parent_owner_sha256
    if parent_manifest_sha256 is not None:
        result["parent_n_manifest_sha256"] = parent_manifest_sha256
    return result


def binding(parameters: dict[str, object], proof: dict[str, object]):
    return {
        "parameters": parameters,
        "dataset": {"sha256": "1" * 64},
        "current_deployment_artifact_ledger": {"status": "PASS"},
        "routing": {
            "effective_num_shards": 32,
            "fission_events": [],
            "l1_partition_diagnostics": proof,
        },
    }


def live(physical_points: int) -> dict[str, object]:
    return {
        "upper_navigator_sha256": "2" * 64,
        "upper_graph_semantic_sha256": "3" * 64,
        "vector_schema": {"dimension": 200, "distance": "Cosine"},
        "shard_count": 32,
        "logical_point_count": 1_183_514,
        "physical_point_count": physical_points,
        "hnsw_config": {"m": 32, "ef_construct": 200},
        "optimizer_config": {"indexing_threshold": 20_000},
        "expected_placement": {str(index): index % 4 for index in range(32)},
    }


def arms(module):
    args = argparse.Namespace(
        arm_a_name="N_native",
        arm_b_name="C_CNBR",
        collection_a="native_collection",
        collection_b="cnbr_collection",
        artifact_a=Path("native/generation-1.json"),
        artifact_b=Path("cnbr/generation-2.json"),
        layout_dir_a=Path("native"),
        layout_dir_b=Path("cnbr"),
        prepare_manifest_a=Path("native/preparation_manifest.json"),
        prepare_manifest_b=Path("cnbr/preparation_manifest.json"),
    )
    return module.native_cnbr_arm_specs(args)


def valid_pair(module):
    phase_a = "a" * 64
    native_owner = "b" * 64
    cnbr_owner = "c" * 64
    attachments = "d" * 64
    bindings = {
        "A": binding(
            common_parameters(),
            diagnostics(
                "natural_orion",
                owner_sha256=native_owner,
                phase_a_sha256=phase_a,
                attachments_sha256=attachments,
            ),
        ),
        "B": binding(
            common_parameters(),
            diagnostics(
                "cnbr",
                owner_sha256=cnbr_owner,
                phase_a_sha256=phase_a,
                attachments_sha256=attachments,
                parent_owner_sha256=native_owner,
                parent_manifest_sha256=phase_a,
            ),
        ),
    }
    bindings["A"]["parameters"]["balance_mode"] = "natural_orion"
    bindings["B"]["parameters"]["balance_mode"] = "cnbr"
    live_arms = {"A": live(1_340_000), "B": live(1_330_000)}
    return arms(module), bindings, live_arms


def artifact_ledger_fixture(module, tmp_path):
    collection = "native_collection"
    generation = 17
    artifact_sha256 = "a" * 64
    layout_sha256 = "b" * 64
    peer_ids = [101, 102, 103, 104]
    artifact = tmp_path / f"generation-{generation}.json"
    artifact.write_text("{}\n", encoding="utf-8")
    deployment_path = tmp_path / "deployment.json"
    image = {
        "id": "sha256:image",
        "tag": "orion:test",
        "capabilities": {"orion_compact_wire_max_version": "2"},
    }
    deployment_nodes = [
        {
            "role": role,
            "ssh_host": host,
            "container_name": container,
        }
        for role, host, container in (
            ("controller", "localhost", "controller-container"),
            ("qdrant_shard_1", "host-1", "worker-1"),
            ("qdrant_shard_2", "host-2", "worker-2"),
            ("qdrant_shard_3", "host-3", "worker-3"),
        )
    ]
    expected_placement = {index: peer_ids[index % 4] for index in range(32)}
    local_shards = [
        {"shard_id": index, "points_count": 10, "state": "Active"}
        for index in range(0, 32, 4)
    ]
    remote_shards = [
        {
            "shard_id": index,
            "peer_id": expected_placement[index],
            "state": "Active",
        }
        for index in range(32)
        if index % 4
    ]
    cluster = {
        "peer_id": peer_ids[0],
        "local_shards": local_shards,
        "remote_shards": remote_shards,
        "shard_count": 32,
        "shard_transfers": [],
    }
    policy = {
        "type": "orion",
        "generation": generation,
        "artifact_sha256": artifact_sha256,
    }
    vector_schema = {
        "vector_name": "",
        "dimension": 200,
        "distance": "Cosine",
        "datatype": "float32",
    }
    collection_proof = {
        "status": "green",
        "optimizer_status": "ok",
        "policy_kind": "orion",
        "policy": policy,
        "vector_schema": {**vector_schema, "distance": "cosine"},
        "shard_count": 32,
        "cluster": cluster,
    }
    entry = {
        "policy_kind": "orion",
        "collection": collection,
        "generation": generation,
        "canonical_sha256": artifact_sha256,
        "file_sha256": artifact_sha256,
        "source_path": str(artifact),
        "size_bytes": artifact.stat().st_size,
        "format_version": 1,
        "layout_sha256": layout_sha256,
        "logical_point_count": 1_000,
        "physical_point_count": 1_100,
        "shard_count": 32,
        "vector_schema": vector_schema,
        "nodes": [
            {
                "action": "installed",
                "role": node["role"],
                "ssh_host": node["ssh_host"],
                "sha256": artifact_sha256,
                "destination_path": (
                    f"/storage/collections/{collection}/orion_router/"
                    f"generation-{generation}.json"
                ),
            }
            for node in deployment_nodes
        ],
        "preinstall_collection_proof": copy.deepcopy(collection_proof),
        "activation": {
            "status": "activated_after_restart",
            "restart_order": "workers-first",
            "activated_at": "2026-08-26T00:00:00Z",
            "router_log_proof": [
                {
                    "role": node["role"],
                    "container_name": node["container_name"],
                    "loaded_marker": (
                        f"Loaded Orion routing generation {generation} "
                        f"for collection {collection}"
                    ),
                    "since_epoch": 1,
                }
                for node in deployment_nodes
            ],
            "cluster_snapshot": {
                "status": "ok",
                "result": {
                    "status": "enabled",
                    "message_send_failures": {},
                    "consensus_thread_status": {
                        "consensus_thread_status": "working"
                    },
                    "raft_info": {"pending_operations": 0},
                },
            },
            "collection_proof": copy.deepcopy(collection_proof),
        },
    }
    current = {
        "image": image,
        "nodes": deployment_nodes,
        "orion_artifacts": [entry],
    }
    deployment_path.write_text(
        json.dumps(current, sort_keys=True) + "\n", encoding="utf-8"
    )
    arm = module.base.ArmSpec(
        label="A",
        name="N_native",
        collection=collection,
        artifact=artifact,
        layout_dir=tmp_path,
        prepare_manifest=tmp_path / "prepare.json",
        allow_balance_layout=False,
        allow_scaling_layout=False,
        allow_l1_partition_layout=True,
        allow_historical_prepare_deployment=False,
    )
    binding = {
        "artifact_path": str(artifact),
        "artifact_sha256": artifact_sha256,
        "deployment_manifest_sha256": module.base.sha256_path(deployment_path),
        "prepare": {
            "deployment_manifest": {
                "path": str(deployment_path),
                "sha256": "c" * 64,
                "image": image,
            },
            "layout": {
                "artifact": {
                    "generation": generation,
                    "format_version": 1,
                    "layout_sha256": layout_sha256,
                    "logical_point_count": 1_000,
                    "physical_point_count": 1_100,
                    "shard_count": 32,
                    "vector_schema": vector_schema,
                }
            },
            "final_placement_proof": {
                "expected_placement": {
                    str(shard): peer for shard, peer in expected_placement.items()
                }
            },
        },
    }
    return arm, binding, deployment_path, current


def test_reuses_accepted_measurement_and_statistics_helpers():
    module = load_module()
    assert module.paired_order is module.base.paired_order
    assert module.run_formal_pairs is module.base.run_formal_pairs
    assert (
        module.paired_ratio_confidence_interval
        is module.base.paired_ratio_confidence_interval
    )
    assert module.select_independent_saturation is module.base.select_independent_saturation


def test_execution_source_bundle_includes_wrapper_and_shared_engine():
    module = load_module()
    bundle = module.base.repo_local_execution_source_bundle()
    paths = {row["path"] for row in bundle["files"]}
    assert (
        "experiments/c1/scripts/c1_orion_l1_owner_repair_interleaved_ab.py"
        in paths
    )
    assert "experiments/c1/scripts/c1_orion_balance_interleaved_ab.py" in paths
    assert "experiments/l1_balance/cnbr_construction_cost_gate.py" in paths


def test_arm_specs_authorize_both_l1_layouts_without_historical_exceptions():
    module = load_module()
    specs = arms(module)
    assert set(specs) == {"A", "B"}
    assert {label: arm.name for label, arm in specs.items()} == {
        "A": "N_native",
        "B": "C_CNBR",
    }
    for arm in specs.values():
        assert arm.allow_l1_partition_layout is True
        assert arm.allow_historical_prepare_deployment is False
        assert arm.allow_scaling_layout is False
        assert arm.allow_balance_layout is False


def test_current_deployment_artifact_ledger_accepts_exact_supersession(tmp_path):
    module = load_module()
    arm, binding, deployment_path, _current = artifact_ledger_fixture(
        module, tmp_path
    )
    proof = module.validate_current_deployment_artifact_ledger(
        arm, binding, deployment_path
    )
    assert proof["status"] == "PASS"
    assert proof["whole_manifest_sha256_superseded"] is True
    assert proof["node_count"] == 4
    assert all(proof["checks"].values())


def test_current_deployment_artifact_ledger_accepts_checksum_reused_nodes(tmp_path):
    module = load_module()
    arm, binding, deployment_path, current = artifact_ledger_fixture(
        module, tmp_path
    )
    for node in current["orion_artifacts"][0]["nodes"]:
        node["action"] = "reused"
    deployment_path.write_text(
        json.dumps(current, sort_keys=True) + "\n", encoding="utf-8"
    )
    binding["deployment_manifest_sha256"] = module.base.sha256_path(
        deployment_path
    )
    proof = module.validate_current_deployment_artifact_ledger(
        arm, binding, deployment_path
    )
    assert proof["status"] == "PASS"
    assert proof["checks"]["four_exact_install_nodes"] is True


@pytest.mark.parametrize(
    "drift",
    ["image", "artifact_sha256", "nodes", "activation", "transfers"],
)
def test_current_deployment_artifact_ledger_rejects_drift(
    tmp_path, drift
):
    module = load_module()
    arm, binding, deployment_path, current = artifact_ledger_fixture(
        module, tmp_path
    )
    entry = current["orion_artifacts"][0]
    if drift == "image":
        current["image"] = {**current["image"], "id": "sha256:drift"}
    elif drift == "artifact_sha256":
        entry["canonical_sha256"] = "f" * 64
    elif drift == "nodes":
        entry["nodes"][0]["sha256"] = "f" * 64
    elif drift == "activation":
        entry["activation"]["status"] = "installed_not_activated"
    elif drift == "transfers":
        entry["activation"]["collection_proof"]["cluster"][
            "shard_transfers"
        ] = [{"shard_id": 0}]
    deployment_path.write_text(
        json.dumps(current, sort_keys=True) + "\n", encoding="utf-8"
    )
    binding["deployment_manifest_sha256"] = module.base.sha256_path(
        deployment_path
    )
    with pytest.raises(RuntimeError, match="artifact-ledger supersession failed"):
        module.validate_current_deployment_artifact_ledger(
            arm, binding, deployment_path
        )


def test_native_cnbr_prepare_binding_uses_only_strict_ledger_supersession(
    monkeypatch, tmp_path
):
    module = load_module()
    arm, binding, deployment_path, _current = artifact_ledger_fixture(
        module, tmp_path
    )
    binding = {
        **binding,
        "prepare_deployment_binding": {
            "mode": "historical",
            "explicitly_allowed": True,
        },
        "prepare_topology_binding": {"mode": "current"},
    }

    def fake_generic_binding(observed_arm, **_kwargs):
        assert observed_arm.allow_historical_prepare_deployment is True
        return copy.deepcopy(binding)

    monkeypatch.setattr(
        module.base, "validate_prepare_binding", fake_generic_binding
    )
    proof = module.validate_native_cnbr_prepare_binding(
        arm,
        base_url="http://controller:6333",
        deployment_manifest=deployment_path,
        topology=tmp_path / "topology.json",
    )
    deployment = proof["prepare_deployment_binding"]
    assert deployment["mode"] == "current_artifact_ledger_supersession"
    assert deployment["explicitly_allowed"] is False
    assert deployment["generic_historical_exception_used"] is False
    assert proof["current_deployment_artifact_ledger"]["status"] == "PASS"


def test_native_cnbr_fairness_requires_identical_runtime_and_build_contracts():
    module = load_module()
    _arms, bindings, live_arms = valid_pair(module)
    proof = module.validate_native_cnbr_fairness(bindings, live_arms)
    assert proof["status"] == "PASS"
    assert proof["checks"]["same_upper_navigator"] is True
    assert proof["checks"]["same_hnsw_config"] is True
    assert proof["checks"]["same_numeric_shard_placement"] is True
    assert proof["checks"]["multi_assignment_enabled_in_both"] is True
    assert proof["checks"]["same_frozen_search_budget"] is True
    assert "balance_mode" not in proof["common_build_parameters"]


def test_native_cnbr_fairness_rejects_multi_assignment_drift():
    module = load_module()
    _arms, bindings, live_arms = valid_pair(module)
    bindings["B"]["parameters"] = {
        **bindings["B"]["parameters"],
        "multi_assign_vote_delta": 1,
    }
    with pytest.raises(RuntimeError, match="fairness contract failed"):
        module.validate_native_cnbr_fairness(bindings, live_arms)


def test_native_cnbr_identity_binds_parent_owner_phase_a_and_attachments():
    module = load_module()
    pair_arms, bindings, live_arms = valid_pair(module)
    proof = module.validate_native_cnbr_pair(pair_arms, bindings, live_arms)
    assert proof["status"] == "PASS"
    assert proof["roles"] == {"A": "N_native", "B": "C_CNBR"}
    assert proof["checks"]["cnbr_parent_owner_is_native_owner"] is True
    assert proof["checks"]["same_phase_a_manifest"] is True
    assert proof["checks"]["same_full_attachment_bytes"] is True
    assert proof["checks"]["forbidden_phase_a_invocations_zero"] is True


def test_native_cnbr_identity_rejects_wrong_parent_owner():
    module = load_module()
    pair_arms, bindings, live_arms = valid_pair(module)
    bindings["B"]["routing"]["l1_partition_diagnostics"][
        "parent_n_owner_sha256"
    ] = "e" * 64
    with pytest.raises(RuntimeError, match="identity contract failed"):
        module.validate_native_cnbr_pair(pair_arms, bindings, live_arms)


def test_native_cnbr_identity_rejects_forbidden_phase_a_invocation():
    module = load_module()
    pair_arms, bindings, live_arms = valid_pair(module)
    bindings["B"]["routing"]["l1_partition_diagnostics"][
        "forbidden_stage_invocations"
    ][
        "full_attachment_reads"
    ] = 1
    with pytest.raises(RuntimeError, match="identity contract failed"):
        module.validate_native_cnbr_pair(pair_arms, bindings, live_arms)


@pytest.mark.parametrize(
    ("label", "legacy_variant"),
    [
        ("A", "native_unconstrained_kmeans"),
        ("B", "cnbr_upper_only_boundary_repair"),
    ],
)
def test_native_cnbr_identity_rejects_legacy_variant_aliases(
    label, legacy_variant
):
    module = load_module()
    pair_arms, bindings, live_arms = valid_pair(module)
    bindings[label]["routing"]["l1_partition_diagnostics"]["balance_contract"][
        "variant"
    ] = legacy_variant
    with pytest.raises(RuntimeError, match="non-canonical balance variant"):
        module.validate_native_cnbr_pair(pair_arms, bindings, live_arms)


def test_native_cnbr_identity_requires_exact_phase_a_invocation_schema():
    module = load_module()
    pair_arms, bindings, live_arms = valid_pair(module)
    counters = bindings["A"]["routing"]["l1_partition_diagnostics"][
        "forbidden_stage_invocations"
    ]
    counters.pop("capacity_flow_calls")
    counters["legacy_flow"] = 0
    with pytest.raises(RuntimeError, match="schema drift"):
        module.validate_native_cnbr_pair(pair_arms, bindings, live_arms)


def test_contract_names_c_over_n_direction_and_retains_native_on_failure():
    module = load_module()
    contract = module.native_cnbr_contract()
    assert contract.semantic_ratio_key == "paired_ratio_c_over_n"
    assert contract.adoption_pass_decision == (
        "ADOPT_CNBR_OVER_NATIVE_ON_ONLINE_EVIDENCE"
    )
    assert contract.adoption_fail_decision == "RETAIN_NATIVE_ON_ONLINE_EVIDENCE"
    assert contract.comparison_manifest_key == "native_cnbr_pair"
    assert (
        contract.adoption_prerequisite_validator
        is module.validate_adoption_construction_cost
    )
    assert (
        contract.prepare_binding_validator
        is module.validate_native_cnbr_prepare_binding
    )
    assert "reported separately" in contract.adoption_scope
    assert "requires" not in contract.adoption_scope.split(
        "The mandatory H-versus-C bridge", 1
    )[1]


def test_cli_has_no_historical_or_legacy_layout_escape_hatches():
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
        "native",
        "--collection-b",
        "cnbr",
        "--artifact-a",
        "native/generation.json",
        "--artifact-b",
        "cnbr/generation.json",
        "--layout-dir-a",
        "native",
        "--layout-dir-b",
        "cnbr",
        "--prepare-manifest-a",
        "native/prepare.json",
        "--prepare-manifest-b",
        "cnbr/prepare.json",
        "--construction-cost-audit",
        "construction-cost-audit.json",
    ]
    parsed = module.parse_args(required)
    assert parsed.arm_a_name == "N_native"
    assert parsed.arm_b_name == "C_CNBR"
    parsed_arms = module.native_cnbr_arm_specs(parsed)
    assert all(arm.allow_l1_partition_layout for arm in parsed_arms.values())
    with pytest.raises(SystemExit):
        module.parse_args([*required, "--allow-historical-prepare-deployment-a"])
    with pytest.raises(SystemExit):
        module.parse_args([*required, "--allow-balance-layout-a"])


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("--arm-a-name", "forged-native"),
        ("--arm-b-name", "forged-cnbr"),
        ("--arm-a-name", "N_native"),
        ("--arm-b-name", "C_CNBR"),
    ],
)
def test_cli_exposes_no_arm_display_name_override(option, value):
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
        "native",
        "--collection-b",
        "cnbr",
        "--artifact-a",
        "native/generation.json",
        "--artifact-b",
        "cnbr/generation.json",
        "--layout-dir-a",
        "native",
        "--layout-dir-b",
        "cnbr",
        "--prepare-manifest-a",
        "native/prepare.json",
        "--prepare-manifest-b",
        "cnbr/prepare.json",
        "--construction-cost-audit",
        "construction-cost-audit.json",
    ]
    with pytest.raises(SystemExit):
        module.parse_args([*required, option, value])


@pytest.mark.parametrize(
    ("attribute", "value"),
    [
        ("arm_a_name", "forged-native"),
        ("arm_b_name", "forged-cnbr"),
    ],
)
def test_programmatic_runner_namespace_cannot_relabel_arms(
    tmp_path, attribute, value
):
    module = load_module()
    for name in (
        "dataset.hdf5",
        "topology.json",
        "deployment.json",
        "native-generation.json",
        "cnbr-generation.json",
        "native-prepare.json",
        "cnbr-prepare.json",
    ):
        (tmp_path / name).write_text("{}\n", encoding="utf-8")
    (tmp_path / "native-layout").mkdir()
    (tmp_path / "cnbr-layout").mkdir()
    args = argparse.Namespace(
        base_url="http://controller:6333",
        hdf5_path=tmp_path / "dataset.hdf5",
        topology=tmp_path / "topology.json",
        deployment_manifest=tmp_path / "deployment.json",
        output_dir=str(tmp_path / "output"),
        arm_a_name="N_native",
        arm_b_name="C_CNBR",
        collection_a="native",
        collection_b="cnbr",
        artifact_a=tmp_path / "native-generation.json",
        artifact_b=tmp_path / "cnbr-generation.json",
        layout_dir_a=tmp_path / "native-layout",
        layout_dir_b=tmp_path / "cnbr-layout",
        prepare_manifest_a=tmp_path / "native-prepare.json",
        prepare_manifest_b=tmp_path / "cnbr-prepare.json",
        target_recall=module.TARGET_RECALL,
        concurrency_candidates=[1, 2],
        sweep_seconds=1.0,
        warmup_seconds=1.0,
        construction_cost_audit=tmp_path / "construction-cost-audit.json",
    )
    setattr(args, attribute, value)
    with pytest.raises(ValueError, match="arm names are fixed"):
        module.validate_args(args)


def test_adoption_prerequisite_binds_exact_cost_audit_and_glove_screen(
    monkeypatch, tmp_path
):
    module = load_module()
    _pair_arms, bindings, _live_arms = valid_pair(module)
    audit_path = tmp_path / "construction-cost-audit.json"
    audit_path.write_text("{}\n", encoding="utf-8")
    audit_sha256 = "e" * 64
    glove_screen_sha256 = "f" * 64
    evidence = {
        "status": "PASS",
        "audit": str(audit_path.resolve()),
        "audit_sha256": audit_sha256,
        "all_datasets_pass": True,
        "datasets": {
            "sift1m": {"phase_b_screen_sha256": "1" * 64},
            "glove-200-angular": {
                "phase_b_screen_sha256": glove_screen_sha256
            },
        },
    }
    monkeypatch.setattr(
        module.cost_gate,
        "validate_construction_cost_v4_pass",
        lambda path: evidence,
    )
    diagnostics_b = bindings["B"]["routing"]["l1_partition_diagnostics"]
    diagnostics_b.update(
        {
            "phase_b_manifest_sha256": glove_screen_sha256,
            "construction_cost_v4_gate": "PASS",
            "construction_cost_v4_audit_sha256": audit_sha256,
        }
    )
    bindings["B"]["build_manifest"] = {
        "provenance": {
            "construction_cost_v4_audit": str(audit_path.resolve()),
            "construction_cost_v4_audit_sha256": audit_sha256,
            "construction_cost_v4_gate": {
                "status": "PASS",
                "audit_sha256": audit_sha256,
            },
        }
    }
    proof = module.validate_adoption_construction_cost(
        argparse.Namespace(construction_cost_audit=audit_path),
        bindings,
        {"roles": {"A": "N_native", "B": "C_CNBR"}},
    )
    assert proof["status"] == "PASS"
    assert proof["construction_cost_v4"]["audit_sha256"] == audit_sha256
    assert all(proof["checks"].values())

    diagnostics_b["construction_cost_v4_audit_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="bundle binding failed"):
        module.validate_adoption_construction_cost(
            argparse.Namespace(construction_cost_audit=audit_path),
            bindings,
            {"roles": {"A": "N_native", "B": "C_CNBR"}},
        )


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
