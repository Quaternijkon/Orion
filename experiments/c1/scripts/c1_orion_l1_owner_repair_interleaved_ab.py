#!/usr/bin/env python3
"""Read-only paired A/B for native L1 owners versus CNBR repair.

Arm A is the fixed-P=32 native unconstrained-KMeans owner and arm B is the
CNBR owner derived from that exact frozen owner.  Both collections must already
exist as checksum-bound L1-partition layouts on the same four-host round-robin
deployment.  This runner never prepares, activates, moves, creates, deletes, or
updates collections and never changes container resources.

The measurement protocol is shared with ``c1_orion_balance_interleaved_ab``:
GloVe/Cosine, 1,000 tuning queries, a disjoint 9,000-query held-out split,
Recall@10 in the frozen ``[0.90, 0.93)`` tuning band, independently selected
saturation concurrency, and five counterbalanced 20-second pairs extended to
seven only when either five-run QPS CV exceeds 5%.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.c1.scripts import (  # noqa: E402
    c1_orion_balance_interleaved_ab as base,
)
from experiments.l1_balance import cnbr_construction_cost_gate as cost_gate  # noqa: E402


benchmark_lock = base.benchmark_lock
ARM_LABELS = base.ARM_LABELS
FORMAL_LOGICAL_SHARDS = base.FORMAL_LOGICAL_SHARDS
HISTORICAL_BUDGET = base.HISTORICAL_BUDGET
TARGET_RECALL = base.TARGET_RECALL

# Re-export the accepted measurement/statistics helpers so tests and later
# protocol tooling use one implementation rather than a forked calculation.
paired_order = base.paired_order
paired_ratio_confidence_interval = base.paired_ratio_confidence_interval
run_formal_pairs = base.run_formal_pairs
select_independent_saturation = base.select_independent_saturation


NATIVE_BALANCE_VARIANT = "natural_orion"
CNBR_BALANCE_VARIANT = "cnbr"
NATIVE_ARM_NAME = "N_native"
CNBR_ARM_NAME = "C_CNBR"
FIXED_ARM_NAMES = {"A": NATIVE_ARM_NAME, "B": CNBR_ARM_NAME}
CANONICAL_BALANCE_VARIANTS = {
    NATIVE_BALANCE_VARIANT,
    CNBR_BALANCE_VARIANT,
}
PHASE_A_FORBIDDEN_INVOCATION_KEYS = (
    "full_attachment_reads",
    "point_to_l1s_reads",
    "multi_assignment_reads",
    "weight_recalibration_calls",
    "fission_calls",
    "topology_refinement_calls",
    "capacity_flow_calls",
    "lower_layout_repair_calls",
)
NATIVE_CNBR_COMMON_PARAMETER_KEYS = (
    *base.COMMON_BUILD_PARAMETER_KEYS,
    "initial_num_shards",
    "enable_fission",
    "enable_topology_refinement",
    "l0_repair",
)


def normalize_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise RuntimeError(f"{label} must be a SHA-256 string")
    normalized = value.strip().lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise RuntimeError(f"{label} must be a valid SHA-256 string")
    return normalized


def l1_partition_diagnostics(
    binding: Mapping[str, Any], label: str
) -> dict[str, Any]:
    routing = binding.get("routing")
    if not isinstance(routing, Mapping):
        raise RuntimeError(f"arm {label} build manifest lacks routing diagnostics")
    diagnostics = routing.get("l1_partition_diagnostics")
    if not isinstance(diagnostics, Mapping):
        raise RuntimeError(
            f"arm {label} build manifest lacks l1_partition_diagnostics"
        )
    return dict(diagnostics)


def balance_variant(diagnostics: Mapping[str, Any], label: str) -> str:
    """Return the canonical, object-wrapped L1 owner variant."""
    raw = diagnostics.get("balance_contract")
    if not isinstance(raw, Mapping) or not isinstance(raw.get("variant"), str):
        raise RuntimeError(
            f"arm {label} l1_partition_diagnostics lacks canonical "
            "balance_contract.variant"
        )
    variant = str(raw["variant"])
    if variant not in CANONICAL_BALANCE_VARIANTS:
        raise RuntimeError(
            f"arm {label} uses non-canonical balance variant {variant!r}"
        )
    return variant


def phase_a_forbidden_invocations(
    diagnostics: Mapping[str, Any], label: str
) -> dict[str, int]:
    raw = diagnostics.get("forbidden_stage_invocations")
    if not isinstance(raw, Mapping):
        raise RuntimeError(
            f"arm {label} lacks fail-closed forbidden_stage_invocations"
        )
    observed_keys = set(raw)
    expected_keys = set(PHASE_A_FORBIDDEN_INVOCATION_KEYS)
    if observed_keys != expected_keys:
        raise RuntimeError(
            f"arm {label} forbidden_stage_invocations schema drift: "
            f"missing={sorted(expected_keys - observed_keys)}, "
            f"extra={sorted(observed_keys - expected_keys)}"
        )
    result: dict[str, int] = {}
    for key in PHASE_A_FORBIDDEN_INVOCATION_KEYS:
        value = raw.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise RuntimeError(
                f"arm {label} forbidden_stage_invocations.{key} is invalid"
            )
        result[key] = int(value)
    return result


def diagnostic_sha256(
    diagnostics: Mapping[str, Any], key: str, label: str
) -> str:
    return normalize_sha256(diagnostics.get(key), f"arm {label} {key}")


def cnbr_parent_owner_sha256(diagnostics: Mapping[str, Any]) -> str:
    raw = diagnostics.get("parent_n_owner_sha256")
    if raw is None:
        raw = diagnostics.get("base_owner_sha256")
    return normalize_sha256(raw, "arm B parent N owner SHA-256")


def cnbr_parent_manifest_sha256(diagnostics: Mapping[str, Any]) -> str:
    raw = diagnostics.get("parent_n_manifest_sha256")
    if raw is None:
        raw = diagnostics.get("phase_a_manifest_sha256")
    return normalize_sha256(raw, "arm B parent N manifest SHA-256")


def native_cnbr_arm_specs(args: argparse.Namespace) -> dict[str, base.ArmSpec]:
    def build(label: str, suffix: str) -> base.ArmSpec:
        return base.ArmSpec(
            label=label,
            name=FIXED_ARM_NAMES[label],
            collection=getattr(args, f"collection_{suffix}"),
            artifact=getattr(args, f"artifact_{suffix}").expanduser().resolve(),
            layout_dir=getattr(args, f"layout_dir_{suffix}").expanduser().resolve(),
            prepare_manifest=getattr(
                args, f"prepare_manifest_{suffix}"
            ).expanduser().resolve(),
            allow_balance_layout=False,
            allow_scaling_layout=False,
            allow_l1_partition_layout=True,
            allow_historical_prepare_deployment=False,
        )

    return {"A": build("A", "a"), "B": build("B", "b")}


def _canonical_vector_schema(raw: Any, label: str) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise RuntimeError(f"{label} vector schema is missing")
    dimension = raw.get("dimension")
    if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0:
        raise RuntimeError(f"{label} vector schema dimension is invalid")
    return {
        "vector_name": str(raw.get("vector_name") or ""),
        "dimension": int(dimension),
        "distance": str(raw.get("distance") or "").lower(),
        "datatype": str(raw.get("datatype") or "float32").lower(),
    }


def _artifact_ledger_collection_checks(
    proof: Any,
    *,
    expected_policy: Mapping[str, Any],
    expected_schema: Mapping[str, Any],
    expected_placement: Mapping[int, int],
    expected_shard_count: int,
    label: str,
) -> dict[str, bool]:
    if not isinstance(proof, Mapping):
        raise RuntimeError(f"{label} collection proof is missing")
    cluster = proof.get("cluster")
    if not isinstance(cluster, Mapping):
        raise RuntimeError(f"{label} cluster proof is missing")
    local = cluster.get("local_shards")
    remote = cluster.get("remote_shards")
    transfers = cluster.get("shard_transfers")
    if not isinstance(local, list) or not isinstance(remote, list):
        raise RuntimeError(f"{label} cluster shard proof is malformed")
    if not isinstance(transfers, list):
        raise RuntimeError(f"{label} cluster transfer proof is malformed")
    controller_peer_id = cluster.get("peer_id")
    if (
        isinstance(controller_peer_id, bool)
        or not isinstance(controller_peer_id, int)
    ):
        raise RuntimeError(f"{label} controller peer ID is invalid")
    placement: dict[int, int] = {}
    states_active = True
    for row in local:
        if not isinstance(row, Mapping):
            raise RuntimeError(f"{label} local shard proof is malformed")
        shard_id = row.get("shard_id")
        if isinstance(shard_id, bool) or not isinstance(shard_id, int):
            raise RuntimeError(f"{label} local shard ID is invalid")
        if shard_id in placement:
            raise RuntimeError(f"{label} repeats shard {shard_id}")
        placement[shard_id] = controller_peer_id
        states_active = states_active and row.get("state") == "Active"
    for row in remote:
        if not isinstance(row, Mapping):
            raise RuntimeError(f"{label} remote shard proof is malformed")
        shard_id = row.get("shard_id")
        peer_id = row.get("peer_id")
        if (
            isinstance(shard_id, bool)
            or not isinstance(shard_id, int)
            or isinstance(peer_id, bool)
            or not isinstance(peer_id, int)
        ):
            raise RuntimeError(f"{label} remote shard identity is invalid")
        if shard_id in placement:
            raise RuntimeError(f"{label} repeats shard {shard_id}")
        placement[shard_id] = peer_id
        states_active = states_active and row.get("state") == "Active"
    return {
        "status_green": proof.get("status") == "green",
        "optimizer_ok": proof.get("optimizer_status") == "ok",
        "policy_kind_orion": proof.get("policy_kind") == "orion",
        "policy_exact": proof.get("policy") == expected_policy,
        "vector_schema_exact": _canonical_vector_schema(
            proof.get("vector_schema"), label
        )
        == dict(expected_schema),
        "p32": proof.get("shard_count") == expected_shard_count == 32,
        "cluster_p32": cluster.get("shard_count") == expected_shard_count,
        "all_shards_active": states_active,
        "all_shards_present_once": set(placement)
        == set(range(expected_shard_count)),
        "placement_exact": placement == dict(expected_placement),
        "zero_transfers": transfers == [],
    }


def validate_current_deployment_artifact_ledger(
    arm: base.ArmSpec,
    binding: Mapping[str, Any],
    deployment_manifest: Path,
) -> dict[str, Any]:
    """Prove that the current deployment still carries one exact N/C artifact.

    Installing the second arm updates the deployment manifest's artifact ledger,
    so the first arm's prepare manifest no longer has the current whole-file
    digest.  This validator accepts only that narrow supersession: the current
    manifest must retain a unique, exact, four-node installed-and-activated
    ledger entry for the declared arm.  It is not a generic historical-prepare
    escape hatch.
    """

    if arm.name not in FIXED_ARM_NAMES.values():
        raise RuntimeError(
            f"artifact-ledger supersession is forbidden for arm {arm.name!r}"
        )
    if arm.allow_historical_prepare_deployment:
        raise RuntimeError(
            "native/CNBR artifact-ledger validation requires historical prepare "
            "authorization to remain disabled"
        )
    current_path = deployment_manifest.expanduser().resolve(strict=True)
    current = base.load_json_object(current_path)
    current_sha256 = base.sha256_path(current_path)
    prepare = binding.get("prepare")
    if not isinstance(prepare, Mapping):
        raise RuntimeError("native/CNBR binding lacks its prepare manifest")
    recorded_deployment = prepare.get("deployment_manifest")
    if not isinstance(recorded_deployment, Mapping):
        raise RuntimeError("native/CNBR prepare lacks deployment evidence")
    recorded_path = Path(str(recorded_deployment.get("path") or "")).expanduser().resolve()
    prepare_layout = prepare.get("layout")
    if not isinstance(prepare_layout, Mapping):
        raise RuntimeError("native/CNBR prepare lacks layout evidence")
    artifact_metadata = prepare_layout.get("artifact")
    if not isinstance(artifact_metadata, Mapping):
        raise RuntimeError("native/CNBR prepare lacks artifact metadata")

    generation = artifact_metadata.get("generation")
    shard_count = artifact_metadata.get("shard_count")
    logical_point_count = artifact_metadata.get("logical_point_count")
    physical_point_count = artifact_metadata.get("physical_point_count")
    format_version = artifact_metadata.get("format_version")
    for value, label in (
        (generation, "generation"),
        (shard_count, "shard_count"),
        (logical_point_count, "logical_point_count"),
        (physical_point_count, "physical_point_count"),
        (format_version, "format_version"),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise RuntimeError(f"native/CNBR prepare artifact {label} is invalid")
    expected_schema = _canonical_vector_schema(
        artifact_metadata.get("vector_schema"), "prepare artifact"
    )
    expected_policy = {
        "type": "orion",
        "generation": generation,
        "artifact_sha256": binding["artifact_sha256"],
    }
    expected_placement_raw = (
        (prepare.get("final_placement_proof") or {}).get("expected_placement")
    )
    if not isinstance(expected_placement_raw, Mapping):
        raise RuntimeError("native/CNBR prepare lacks expected placement")
    expected_placement = {
        int(shard): int(peer) for shard, peer in expected_placement_raw.items()
    }

    artifacts = current.get("orion_artifacts")
    if not isinstance(artifacts, list):
        raise RuntimeError("current deployment lacks an Orion artifact ledger")
    matching = [
        row
        for row in artifacts
        if isinstance(row, Mapping)
        and row.get("collection") == arm.collection
        and row.get("generation") == generation
    ]
    if len(matching) != 1:
        raise RuntimeError(
            "native/CNBR current deployment artifact-ledger supersession failed: "
            f"expected one {arm.collection!r}/generation-{generation} entry, "
            f"found {len(matching)}"
        )
    entry = matching[0]
    source_path = Path(str(entry.get("source_path") or "")).expanduser().resolve()
    artifact_path = Path(binding["artifact_path"]).resolve()
    activation = entry.get("activation")
    entry_nodes = entry.get("nodes")
    deployment_nodes = current.get("nodes")
    if not isinstance(activation, Mapping):
        raise RuntimeError("native/CNBR artifact ledger lacks activation proof")
    if not isinstance(entry_nodes, list) or not isinstance(deployment_nodes, list):
        raise RuntimeError("native/CNBR artifact ledger lacks node proof")

    deployment_node_identities = {
        (
            str(row.get("role") or ""),
            str(row.get("ssh_host") or ""),
            str(row.get("container_name") or ""),
        )
        for row in deployment_nodes
        if isinstance(row, Mapping)
    }
    ledger_node_identities = {
        (str(row.get("role") or ""), str(row.get("ssh_host") or ""))
        for row in entry_nodes
        if isinstance(row, Mapping)
    }
    expected_node_identities = {
        (role, ssh_host) for role, ssh_host, _container in deployment_node_identities
    }
    node_rows_valid = len(entry_nodes) == 4 and all(
        isinstance(row, Mapping)
        # A later checksum-verified activation may legitimately reuse the
        # already installed immutable artifact.  The destination checksum is
        # re-probed by the installer before and after the restart, so
        # ``reused`` is as strong as ``installed`` for this ledger proof.
        and row.get("action") in {"installed", "reused"}
        and row.get("sha256") == binding["artifact_sha256"]
        and Path(str(row.get("destination_path") or "")).name
        == f"generation-{generation}.json"
        and f"/collections/{arm.collection}/orion_router/"
        in str(row.get("destination_path") or "")
        for row in entry_nodes
    )

    router_logs = activation.get("router_log_proof")
    router_log_identities: set[tuple[str, str]] = set()
    router_logs_valid = isinstance(router_logs, list) and len(router_logs) == 4
    if router_logs_valid:
        expected_marker = (
            f"Loaded Orion routing generation {generation} for collection "
            f"{arm.collection}"
        )
        for row in router_logs:
            if not isinstance(row, Mapping):
                router_logs_valid = False
                break
            role = str(row.get("role") or "")
            container = str(row.get("container_name") or "")
            since_epoch = row.get("since_epoch")
            router_log_identities.add((role, container))
            router_logs_valid = router_logs_valid and (
                row.get("loaded_marker") == expected_marker
                and not isinstance(since_epoch, bool)
                and isinstance(since_epoch, int)
                and since_epoch > 0
            )
    expected_router_identities = {
        (role, container) for role, _ssh_host, container in deployment_node_identities
    }

    activation_collection_checks = _artifact_ledger_collection_checks(
        activation.get("collection_proof"),
        expected_policy=expected_policy,
        expected_schema=expected_schema,
        expected_placement=expected_placement,
        expected_shard_count=shard_count,
        label="activation",
    )
    preinstall_collection_checks = _artifact_ledger_collection_checks(
        entry.get("preinstall_collection_proof"),
        expected_policy=expected_policy,
        expected_schema=expected_schema,
        expected_placement=expected_placement,
        expected_shard_count=shard_count,
        label="preinstall",
    )
    snapshot = activation.get("cluster_snapshot")
    snapshot_result = snapshot.get("result") if isinstance(snapshot, Mapping) else None
    consensus = (
        snapshot_result.get("consensus_thread_status")
        if isinstance(snapshot_result, Mapping)
        else None
    )
    raft = snapshot_result.get("raft_info") if isinstance(snapshot_result, Mapping) else None
    snapshot_valid = (
        isinstance(snapshot, Mapping)
        and snapshot.get("status") == "ok"
        and isinstance(snapshot_result, Mapping)
        and snapshot_result.get("status") == "enabled"
        and snapshot_result.get("message_send_failures") == {}
        and isinstance(consensus, Mapping)
        and consensus.get("consensus_thread_status") == "working"
        and isinstance(raft, Mapping)
        and raft.get("pending_operations") == 0
    )

    checks = {
        "same_deployment_path": recorded_path == current_path,
        "current_deployment_sha_exact": current_sha256
        == binding.get("deployment_manifest_sha256"),
        "same_image": recorded_deployment.get("image") == current.get("image"),
        "policy_kind_orion": entry.get("policy_kind") == "orion",
        "collection_exact": entry.get("collection") == arm.collection,
        "generation_exact": entry.get("generation") == generation,
        "canonical_sha256_exact": entry.get("canonical_sha256")
        == binding["artifact_sha256"],
        "file_sha256_exact": entry.get("file_sha256")
        == binding["artifact_sha256"],
        "source_path_exact": source_path == artifact_path,
        "source_file_present": source_path.is_file(),
        "source_size_exact": entry.get("size_bytes") == artifact_path.stat().st_size,
        "format_version_exact": entry.get("format_version") == format_version,
        "layout_sha256_exact": entry.get("layout_sha256")
        == artifact_metadata.get("layout_sha256"),
        "logical_point_count_exact": entry.get("logical_point_count")
        == logical_point_count,
        "physical_point_count_exact": entry.get("physical_point_count")
        == physical_point_count,
        "shard_count_exact": entry.get("shard_count") == shard_count == 32,
        "vector_schema_exact": _canonical_vector_schema(
            entry.get("vector_schema"), "artifact ledger"
        )
        == expected_schema,
        "four_exact_install_nodes": node_rows_valid
        and ledger_node_identities == expected_node_identities
        and len(deployment_node_identities) == 4,
        "activated_after_workers_first_restart": (
            activation.get("status") == "activated_after_restart"
            and activation.get("restart_order") == "workers-first"
            and isinstance(activation.get("activated_at"), str)
            and bool(str(activation.get("activated_at")).strip())
        ),
        "four_exact_router_loaded_markers": router_logs_valid
        and router_log_identities == expected_router_identities,
        "activation_cluster_snapshot_healthy": snapshot_valid,
        "activation_collection_proof": all(activation_collection_checks.values()),
        "preinstall_collection_proof": all(preinstall_collection_checks.values()),
    }
    if not all(checks.values()):
        raise RuntimeError(
            "native/CNBR current deployment artifact-ledger supersession failed: "
            + json.dumps(
                {
                    "checks": checks,
                    "activation_collection_checks": activation_collection_checks,
                    "preinstall_collection_checks": preinstall_collection_checks,
                },
                sort_keys=True,
            )
        )
    return {
        "status": "PASS",
        "record_type": "native_cnbr_current_deployment_artifact_ledger",
        "checks": checks,
        "current_deployment_manifest_path": str(current_path),
        "current_deployment_manifest_sha256": current_sha256,
        "recorded_prepare_deployment_sha256": str(
            recorded_deployment.get("sha256") or ""
        ).lower(),
        "whole_manifest_sha256_superseded": str(
            recorded_deployment.get("sha256") or ""
        ).lower()
        != current_sha256,
        "collection": arm.collection,
        "generation": generation,
        "artifact_sha256": binding["artifact_sha256"],
        "layout_sha256": artifact_metadata.get("layout_sha256"),
        "node_count": len(entry_nodes),
        "activation_status": activation.get("status"),
    }


def validate_native_cnbr_prepare_binding(
    arm: base.ArmSpec,
    *,
    base_url: str,
    deployment_manifest: Path,
    topology: Path,
) -> dict[str, Any]:
    """Use the generic binding, then replace only artifact-ledger SHA drift."""

    if arm.allow_historical_prepare_deployment:
        raise RuntimeError(
            "native/CNBR arms must not enable generic historical prepare binding"
        )
    compatibility_arm = replace(
        arm,
        allow_historical_prepare_deployment=True,
    )
    binding = base.validate_prepare_binding(
        compatibility_arm,
        base_url=base_url,
        deployment_manifest=deployment_manifest,
        topology=topology,
    )
    topology_binding = binding.get("prepare_topology_binding")
    if not isinstance(topology_binding, Mapping) or topology_binding.get("mode") != "current":
        raise RuntimeError(
            "native/CNBR artifact-ledger supersession cannot excuse topology drift"
        )
    ledger = validate_current_deployment_artifact_ledger(
        arm, binding, deployment_manifest
    )
    deployment_binding = binding.get("prepare_deployment_binding")
    if not isinstance(deployment_binding, Mapping):
        raise RuntimeError("native/CNBR prepare deployment binding is missing")
    mode = deployment_binding.get("mode")
    if mode not in {"current", "historical"}:
        raise RuntimeError(
            f"native/CNBR prepare deployment mode is invalid: {mode!r}"
        )
    binding = dict(binding)
    binding["prepare_deployment_binding"] = {
        **dict(deployment_binding),
        "mode": (
            "current"
            if mode == "current"
            else "current_artifact_ledger_supersession"
        ),
        "explicitly_allowed": False,
        "generic_historical_exception_used": False,
        "artifact_ledger": ledger,
    }
    binding["current_deployment_artifact_ledger"] = ledger
    return binding


def validate_native_cnbr_fairness(
    bindings: Mapping[str, Mapping[str, Any]],
    live: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    a_binding, b_binding = bindings["A"], bindings["B"]
    a_live, b_live = live["A"], live["B"]
    common_parameters = {
        key: a_binding["parameters"].get(key)
        for key in NATIVE_CNBR_COMMON_PARAMETER_KEYS
    }
    parameter_mismatches = {
        key: {
            "A": a_binding["parameters"].get(key),
            "B": b_binding["parameters"].get(key),
        }
        for key in NATIVE_CNBR_COMMON_PARAMETER_KEYS
        if a_binding["parameters"].get(key)
        != b_binding["parameters"].get(key)
    }
    exact_multi_assignment = all(
        bindings[label]["parameters"].get("use_multi_assign") is True
        and bindings[label]["parameters"].get("multi_assign_min_max_vote") == 2
        and bindings[label]["parameters"].get("multi_assign_vote_delta") == 0
        and bindings[label]["parameters"].get("multi_assign_max_shards") == 0
        for label in ARM_LABELS
    )
    equality_checks = {
        "same_upper_navigator": a_live["upper_navigator_sha256"]
        == b_live["upper_navigator_sha256"],
        "same_upper_graph_semantics": a_live["upper_graph_semantic_sha256"]
        == b_live["upper_graph_semantic_sha256"],
        "same_vector_schema": a_live["vector_schema"] == b_live["vector_schema"],
        "same_dataset_sha256": str(a_binding["dataset"].get("sha256") or "")
        == str(b_binding["dataset"].get("sha256") or ""),
        "same_logical_shard_count": a_live["shard_count"]
        == b_live["shard_count"]
        == FORMAL_LOGICAL_SHARDS,
        "same_logical_point_count": a_live["logical_point_count"]
        == b_live["logical_point_count"],
        "same_hnsw_config": a_live["hnsw_config"] == b_live["hnsw_config"],
        "same_optimizer_config": a_live["optimizer_config"]
        == b_live["optimizer_config"],
        "same_numeric_shard_placement": a_live["expected_placement"]
        == b_live["expected_placement"],
        "same_common_build_parameters": not parameter_mismatches,
        "multi_assignment_enabled_in_both": exact_multi_assignment,
        "same_frozen_search_budget": all(
            all(
                bindings[label]["parameters"].get(key) == expected
                for key, expected in HISTORICAL_BUDGET.items()
            )
            for label in ARM_LABELS
        ),
    }
    if not all(equality_checks.values()):
        raise RuntimeError(
            "native/CNBR A/B fairness contract failed: "
            + json.dumps(
                {
                    "checks": equality_checks,
                    "parameter_mismatches": parameter_mismatches,
                },
                sort_keys=True,
            )
        )
    return {
        "status": "PASS",
        "checks": equality_checks,
        "common_build_parameters": common_parameters,
        "upper_navigator_sha256": a_live["upper_navigator_sha256"],
        "upper_graph_semantic_sha256": a_live["upper_graph_semantic_sha256"],
        "logical_shard_count": FORMAL_LOGICAL_SHARDS,
        "logical_point_count": a_live["logical_point_count"],
        "physical_point_count_by_arm": {
            label: live[label]["physical_point_count"] for label in ARM_LABELS
        },
        "numeric_shard_placement": a_live["expected_placement"],
    }


def validate_native_cnbr_pair(
    arms: Mapping[str, base.ArmSpec],
    bindings: Mapping[str, Mapping[str, Any]],
    live: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    diagnostics = {
        label: l1_partition_diagnostics(bindings[label], label)
        for label in ARM_LABELS
    }
    variants = {
        label: balance_variant(diagnostics[label], label) for label in ARM_LABELS
    }
    phase_a_manifests = {
        label: diagnostic_sha256(
            diagnostics[label], "phase_a_manifest_sha256", label
        )
        for label in ARM_LABELS
    }
    phase_a_owners = {
        label: diagnostic_sha256(
            diagnostics[label], "phase_a_owner_sha256", label
        )
        for label in ARM_LABELS
    }
    owner_sha256 = {
        label: diagnostic_sha256(diagnostics[label], "owner_sha256", label)
        for label in ARM_LABELS
    }
    attachments = {
        label: diagnostic_sha256(diagnostics[label], "attachments_sha256", label)
        for label in ARM_LABELS
    }
    forbidden_invocations = {
        label: phase_a_forbidden_invocations(diagnostics[label], label)
        for label in ARM_LABELS
    }
    cnbr_parent_owner = cnbr_parent_owner_sha256(diagnostics["B"])
    cnbr_parent_manifest = cnbr_parent_manifest_sha256(diagnostics["B"])

    checks = {
        "both_arms_explicit_l1_partition_layouts": all(
            arms[label].allow_l1_partition_layout for label in ARM_LABELS
        ),
        "no_historical_prepare_exception": all(
            not arms[label].allow_historical_prepare_deployment
            for label in ARM_LABELS
        ),
        "arm_a_is_n_native": variants["A"] == NATIVE_BALANCE_VARIANT,
        "arm_b_is_c_cnbr": variants["B"] == CNBR_BALANCE_VARIANT,
        "same_phase_a_manifest": phase_a_manifests["A"]
        == phase_a_manifests["B"],
        "phase_a_owner_matches_final_owner": all(
            phase_a_owners[label] == owner_sha256[label] for label in ARM_LABELS
        ),
        "cnbr_parent_manifest_is_native_phase_a": cnbr_parent_manifest
        == phase_a_manifests["A"],
        "cnbr_parent_owner_is_native_owner": cnbr_parent_owner
        == phase_a_owners["A"],
        "same_full_attachment_bytes": attachments["A"] == attachments["B"],
        "fixed_initial_and_final_p32": all(
            bindings[label]["parameters"].get("initial_num_shards")
            == FORMAL_LOGICAL_SHARDS
            and bindings[label]["routing"].get("effective_num_shards")
            == FORMAL_LOGICAL_SHARDS
            and live[label]["shard_count"] == FORMAL_LOGICAL_SHARDS
            for label in ARM_LABELS
        ),
        "fission_disabled": all(
            bindings[label]["parameters"].get("enable_fission") is False
            and bindings[label]["routing"].get("fission_events") == []
            for label in ARM_LABELS
        ),
        "topology_refinement_disabled": all(
            bindings[label]["parameters"].get("enable_topology_refinement")
            is False
            for label in ARM_LABELS
        ),
        "l0_repair_disabled": all(
            bindings[label]["parameters"].get("l0_repair") is False
            for label in ARM_LABELS
        ),
        "forbidden_phase_a_invocations_zero": all(
            all(
                forbidden_invocations[label][key] == 0
                for key in PHASE_A_FORBIDDEN_INVOCATION_KEYS
            )
            for label in ARM_LABELS
        ),
        "multi_assignment_only_after_owner_freeze": all(
            diagnostics[label].get("multi_assignment_after_owner_freeze") is True
            for label in ARM_LABELS
        ),
        "current_deployment_artifact_ledgers_bound": all(
            (
                bindings[label].get("current_deployment_artifact_ledger") or {}
            ).get("status")
            == "PASS"
            for label in ARM_LABELS
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(
            "native/CNBR identity contract failed: "
            + json.dumps(checks, sort_keys=True)
        )
    return {
        "status": "PASS",
        "roles": {"A": "N_native", "B": "C_CNBR"},
        "checks": checks,
        "balance_variants": variants,
        "phase_a_manifest_sha256": phase_a_manifests["A"],
        "phase_a_owner_sha256_by_arm": phase_a_owners,
        "attachments_sha256": attachments["A"],
        "forbidden_stage_invocations_by_arm": forbidden_invocations,
    }


def validate_candidate_construction_cost_binding(
    args: argparse.Namespace,
    candidate_binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind one materialized C_CNBR bundle to the exact aggregate PASS."""

    evidence = cost_gate.validate_construction_cost_v4_pass(
        args.construction_cost_audit
    )
    diagnostics = l1_partition_diagnostics(candidate_binding, "B")
    build_manifest = candidate_binding.get("build_manifest")
    if not isinstance(build_manifest, Mapping):
        raise RuntimeError("arm B binding lacks its build manifest")
    provenance = build_manifest.get("provenance")
    if not isinstance(provenance, Mapping):
        raise RuntimeError("arm B build manifest lacks provenance")
    diagnostic_audit_sha256 = normalize_sha256(
        diagnostics.get("construction_cost_v4_audit_sha256"),
        "arm B construction-cost-v4 audit SHA-256",
    )
    provenance_audit_sha256 = normalize_sha256(
        provenance.get("construction_cost_v4_audit_sha256"),
        "arm B provenance construction-cost-v4 audit SHA-256",
    )
    provenance_gate = provenance.get("construction_cost_v4_gate")
    if not isinstance(provenance_gate, Mapping):
        raise RuntimeError("arm B provenance lacks construction-cost-v4 gate")
    provenance_audit_path = provenance.get("construction_cost_v4_audit")
    if not isinstance(provenance_audit_path, str):
        raise RuntimeError("arm B provenance lacks construction-cost-v4 audit path")
    glove_screen_sha256 = normalize_sha256(
        evidence["datasets"]["glove-200-angular"]["phase_b_screen_sha256"],
        "construction-cost-v4 GloVe formal Phase-B screen SHA-256",
    )
    candidate_screen_sha256 = diagnostic_sha256(
        diagnostics, "phase_b_manifest_sha256", "B"
    )
    checks = {
        "aggregate_cost_v4_pass": evidence.get("status") == "PASS",
        "both_datasets_pass": evidence.get("all_datasets_pass") is True,
        "candidate_diagnostic_gate_pass": diagnostics.get(
            "construction_cost_v4_gate"
        )
        == "PASS",
        "candidate_diagnostic_binds_exact_audit": diagnostic_audit_sha256
        == evidence["audit_sha256"],
        "candidate_provenance_binds_exact_audit": provenance_audit_sha256
        == evidence["audit_sha256"],
        "candidate_provenance_binds_exact_path": Path(
            provenance_audit_path
        ).expanduser().resolve()
        == Path(evidence["audit"]).resolve(),
        "candidate_provenance_embeds_same_pass": (
            provenance_gate.get("status") == "PASS"
            and provenance_gate.get("audit_sha256") == evidence["audit_sha256"]
        ),
        "candidate_phase_b_is_costed_glove_screen": candidate_screen_sha256
        == glove_screen_sha256,
    }
    if not all(checks.values()):
        raise RuntimeError(
            "C_CNBR construction-cost bundle binding failed: "
            + json.dumps(checks, sort_keys=True)
        )
    return {
        "status": "PASS",
        "checks": checks,
        "construction_cost_v4": evidence,
    }


def validate_adoption_construction_cost(
    args: argparse.Namespace,
    bindings: Mapping[str, Mapping[str, Any]],
    comparison_validation: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind final ADOPT eligibility to the exact dual-dataset cost-v4 PASS."""

    candidate = validate_candidate_construction_cost_binding(args, bindings["B"])
    checks = {
        **candidate["checks"],
        "fixed_roles_reconfirmed": comparison_validation.get("roles")
        == FIXED_ARM_NAMES,
    }
    if not checks["fixed_roles_reconfirmed"]:
        raise RuntimeError(
            "native/CNBR construction-cost adoption binding failed: "
            + json.dumps(checks, sort_keys=True)
        )
    return {
        "status": "PASS",
        "record_type": "native_cnbr_adoption_prerequisites",
        "checks": checks,
        "construction_cost_v4": candidate["construction_cost_v4"],
    }


def native_cnbr_contract() -> base.InterleavedABContract:
    return base.InterleavedABContract(
        arm_specs_factory=native_cnbr_arm_specs,
        fairness_validator=validate_native_cnbr_fairness,
        comparison_validator=validate_native_cnbr_pair,
        comparison_manifest_key="native_cnbr_pair",
        comparison_completion_key="native_cnbr_pair_bound",
        fairness_budget_check_key="same_frozen_search_budget",
        fairness_budget_completion_key="same_frozen_search_budget",
        summary_record_type=(
            "c1_orion_l1_owner_repair_interleaved_ab_summary"
        ),
        run_record_type="c1_orion_l1_owner_repair_interleaved_ab",
        budget_mode="frozen_u48_upper_ef48_base50_factor14_native_cnbr",
        budget_subject="the frozen symmetric Orion search budget",
        adoption_pass_decision="ADOPT_CNBR_OVER_NATIVE_ON_ONLINE_EVIDENCE",
        adoption_fail_decision="RETAIN_NATIVE_ON_ONLINE_EVIDENCE",
        adoption_scope=(
            "This is the CNBR-versus-native causal online QPS gate. Final CNBR "
            "adoption additionally requires the frozen offline topology, build-"
            "cost, and Phase-A identity gates. The mandatory H-versus-C bridge "
            "is reported separately and cannot alter this causal decision."
        ),
        decision_gate_key="online_qps_adoption_gate",
        semantic_ratio_key="paired_ratio_c_over_n",
        completion_error_subject="Orion native/CNBR interleaved A/B",
        adoption_prerequisite_validator=validate_adoption_construction_cost,
        prepare_binding_validator=validate_native_cnbr_prepare_binding,
    )


def validate_args(args: argparse.Namespace) -> None:
    observed_names = {
        "A": getattr(args, "arm_a_name", None),
        "B": getattr(args, "arm_b_name", None),
    }
    if observed_names != FIXED_ARM_NAMES:
        raise ValueError(
            "native/CNBR arm names are fixed to "
            f"A={NATIVE_ARM_NAME} and B={CNBR_ARM_NAME}"
        )
    if getattr(args, "construction_cost_audit", None) in (None, ""):
        raise ValueError("native/CNBR A/B requires --construction-cost-audit")
    base.validate_args(args)


def run(args: argparse.Namespace) -> Path:
    validate_args(args)
    # Reject missing/FAIL/drifted cost evidence before taking the deployment
    # benchmark lock or touching any live experiment state.  The shared engine
    # revalidates and cross-binds it to arm B before measurement and ADOPT.
    cost_gate.validate_construction_cost_v4_pass(args.construction_cost_audit)
    output = Path(args.output_dir).expanduser().resolve()
    output_preexisted = output.exists()
    with benchmark_lock.hold_from_args(
        args,
        args.deployment_manifest,
        owner={
            "kind": "c1_orion_l1_owner_repair_interleaved_ab",
            "collection_a": args.collection_a,
            "collection_b": args.collection_b,
            "output_dir": str(output),
        },
    ) as held_lock:
        try:
            return base._run_locked(
                args,
                held_lock,
                contract=native_cnbr_contract(),
            )
        except BaseException as exc:
            if not output_preexisted and output.is_dir():
                try:
                    base.write_json(
                        output / "execution-failed.json",
                        {
                            "timestamp": base.utc_timestamp(),
                            "record_type": (
                                "c1_orion_l1_owner_repair_interleaved_ab_failure"
                            ),
                            "status": "FAIL",
                            "error": repr(exc),
                            "benchmark_lock": held_lock.evidence(),
                        },
                    )
                except Exception:
                    pass
            raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--hdf5-path", type=Path, required=True)
    parser.add_argument("--topology", type=Path, required=True)
    parser.add_argument("--deployment-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--collection-a", required=True)
    parser.add_argument("--collection-b", required=True)
    parser.add_argument("--artifact-a", type=Path, required=True)
    parser.add_argument("--artifact-b", type=Path, required=True)
    parser.add_argument("--layout-dir-a", type=Path, required=True)
    parser.add_argument("--layout-dir-b", type=Path, required=True)
    parser.add_argument("--prepare-manifest-a", type=Path, required=True)
    parser.add_argument("--prepare-manifest-b", type=Path, required=True)
    parser.add_argument("--construction-cost-audit", type=Path, required=True)
    parser.add_argument("--target-recall", type=float, default=TARGET_RECALL)
    parser.add_argument(
        "--concurrency-candidates", default="1,2,4,8,16,32,64"
    )
    parser.add_argument("--sweep-seconds", type=float, default=8.0)
    parser.add_argument("--warmup-seconds", type=float, default=10.0)
    benchmark_lock.add_cli_arguments(parser)
    args = parser.parse_args(argv)
    args.arm_a_name = NATIVE_ARM_NAME
    args.arm_b_name = CNBR_ARM_NAME
    args.concurrency_candidates = base.parse_int_csv(args.concurrency_candidates)
    return args


def main(argv: Sequence[str] | None = None) -> int:
    output = run(parse_args(argv))
    print(json.dumps({"output_dir": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
