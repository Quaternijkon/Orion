#!/usr/bin/env python3
"""Prepare one native numeric auto-shard collection through existing tools."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import struct
import subprocess
import sys
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import method4_distributed_cluster as cluster_tool  # noqa: E402
from tools import native_auto_shard_benchmark as benchmark  # noqa: E402
from tools import orion_native_layout as layout_common  # noqa: E402
from tools import qdrant_two_level_routing_experiment as experiment  # noqa: E402


ROUTED_METHODS = {"orion", "simple_kmeans"}
PREPARATION_MANIFEST = "preparation_manifest.json"
PROVENANCE_METADATA_KEY = "native_auto_shard_prepare"
PROVENANCE_SCHEMA_VERSION = 2
ORION_LAYOUT_TOOLS = {
    "experiments/l1_balance/materialize_candidate.py",
    "experiments/l1_balance/materialize_native_cnbr_candidate.py",
    "experiments/l1_balance/materialize_sampled_cnbr_candidate.py",
    "experiments/multi_assignment/materialize_budgeted_candidate.py",
    "experiments/multi_assignment/materialize_artifact_owner_bmr10.py",
    "experiments/multi_assignment/materialize_owner_policy_candidate.py",
    "experiments/c1/scripts/c1_orion_bmr10_scaling_materialize.py",
    "tools/orion_native_layout.py",
    "tools/orion_native_runtime_profile.py",
}
ORION_L1_PARTITION_LAYOUT_TOOL = "experiments/l1_balance/materialize_candidate.py"
ORION_NATIVE_CNBR_LAYOUT_TOOL = (
    "experiments/l1_balance/materialize_native_cnbr_candidate.py"
)
ORION_SAMPLED_CNBR_LAYOUT_TOOL = (
    "experiments/l1_balance/materialize_sampled_cnbr_candidate.py"
)
ORION_BUDGETED_MULTI_ASSIGNMENT_LAYOUT_TOOL = (
    "experiments/multi_assignment/materialize_budgeted_candidate.py"
)
ORION_BMR10_SCALING_LAYOUT_TOOL = (
    "experiments/c1/scripts/c1_orion_bmr10_scaling_materialize.py"
)
ORION_ARTIFACT_OWNER_BMR10_LAYOUT_TOOL = (
    "experiments/multi_assignment/materialize_artifact_owner_bmr10.py"
)
ORION_OWNER_POLICY_TOURNAMENT_LAYOUT_TOOL = (
    "experiments/multi_assignment/materialize_owner_policy_candidate.py"
)
ORION_L1_PARTITION_LAYOUT_TOOLS = {
    ORION_L1_PARTITION_LAYOUT_TOOL,
    ORION_NATIVE_CNBR_LAYOUT_TOOL,
    ORION_SAMPLED_CNBR_LAYOUT_TOOL,
    ORION_BUDGETED_MULTI_ASSIGNMENT_LAYOUT_TOOL,
    ORION_BMR10_SCALING_LAYOUT_TOOL,
    ORION_ARTIFACT_OWNER_BMR10_LAYOUT_TOOL,
    ORION_OWNER_POLICY_TOURNAMENT_LAYOUT_TOOL,
}
ORION_L1_PARTITION_MODES = {
    "l1_graph_prepartition",
    "matched_l0_informed_reference",
}
ORION_L1_UNIT_BALANCE_CONTRACT = "unit_exact_quota"
ORION_L1_MASS_BALANCE_CONTRACT = "navigation_mass_capacity"
ORION_L1_REFERENCE_BALANCE_CONTRACT = "matched_l0_informed_reference"
ORION_L1_MASS_MODE = "raw-regularized-v2"
ORION_L1_MASS_SOURCE = (
    "production_upper_navigation_top10_hit_frequency_regularized_v2"
)
ORION_L1_MASS_TRANSFORM = "raw_count_with_unit_l1_prior"
ORION_L1_MASS_ESTIMATOR_VERSION = 2
ORION_L1_MASS_METHOD = "mass-balanced-kmeans"
ORION_L1_RAW_MASS_SOURCE = "production_upper_navigation_top10_raw_hit_frequency"
ORION_L1_RAW_MASS_TRANSFORM = "raw_hit_count"
ORION_L1_PROTOCOL_AMENDMENT_ID = (
    "post-exploratory-raw-regularized-v2-confirmation"
)
ORION_L1_PROTOCOL_PATH = REPO_ROOT / "experiments/l1_balance/PROTOCOL.md"
ORION_L1_PROTOCOL_AMENDMENT_PATH = (
    REPO_ROOT / "experiments/l1_balance/RAW_REGULARIZED_V2_AMENDMENT.json"
)
ORION_L1_V2_CONFIRMATION_BUILDER_PATH = (
    REPO_ROOT / "experiments/l1_balance/build_v2_confirmation.py"
)
ORION_L1_MASS_TOP_K = 10
ORION_L1_MASS_SEARCH_EF = 100
ORION_RUNTIME_PROFILE_TOOL = "tools/orion_native_runtime_profile.py"
ORION_RUNTIME_PARAMETER_KEYS = (
    "generation",
    "upper_k",
    "upper_search_ef",
    "dynamic_ef_base",
    "dynamic_ef_factor",
)
SIMPLE_KMEANS_LAYOUT_TOOLS = {
    "tools/simple_kmeans_native_layout.py",
    "tools/simple_kmeans_native_runtime_profile.py",
}
SIMPLE_KMEANS_RUNTIME_PROFILE_TOOL = (
    "tools/simple_kmeans_native_runtime_profile.py"
)
SIMPLE_KMEANS_RUNTIME_PARAMETER_KEYS = (
    "generation",
    "nprobe",
    "lower_hnsw_ef",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create, place, populate, and activate one native auto-shard collection."
    )
    parser.add_argument(
        "--method",
        choices=("hash_all", "orion", "simple_kmeans"),
        required=True,
    )
    parser.add_argument("--topology", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--collection", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--layout-dir")
    parser.add_argument("--hdf5-path")
    parser.add_argument("--p", "--num-shards", dest="num_shards", type=int)
    parser.add_argument(
        "--vector-distance",
        choices=("cosine", "euclid", "l2"),
        default="cosine",
    )
    parser.add_argument("--vector-name", default="")
    parser.add_argument("--hnsw-m", type=int, default=32)
    parser.add_argument("--ef-construct", type=int, default=100)
    parser.add_argument("--max-indexing-threads", type=int, default=0)
    parser.add_argument("--full-scan-threshold", type=int, default=10)
    parser.add_argument("--indexing-threshold", type=int, default=10)
    parser.add_argument("--max-optimization-threads", type=int)
    parser.add_argument("--max-segment-size-kb", type=int)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--request-timeout-secs", type=int, default=120)
    parser.add_argument("--smoke-limit", type=int, default=10)
    parser.add_argument("--transfer-timeout-secs", type=float, default=3600.0)
    parser.add_argument("--transfer-poll-interval-secs", type=float, default=1.0)
    parser.add_argument(
        "--transfer-method",
        choices=("stream_records", "snapshot"),
        default="stream_records",
        help=(
            "Qdrant shard-transfer method. snapshot preserves a prebuilt HNSW "
            "segment; stream_records rebuilds the destination index."
        ),
    )
    parser.add_argument(
        "--placement-strategy",
        choices=("round_robin", "layout_size_balanced"),
        default="round_robin",
        help=(
            "Physical worker placement for native numeric shards. "
            "layout_size_balanced uses only offline layout shard sizes and preserves "
            "the logical routing/search plan."
        ),
    )
    parser.add_argument(
        "--placement-peers",
        choices=("workers", "all_peers"),
        default="workers",
        help=(
            "Place numeric shards on the three non-coordinator peers (workers) "
            "or on all four Qdrant peers, including the benchmark coordinator."
        ),
    )
    parser.add_argument(
        "--allow-orion-scaling-layout",
        action="store_true",
        help=(
            "Allow initial_num_shards to differ from the canonical 31-shard "
            "Orion configuration for an explicitly labeled shard-scaling experiment. "
            "All other Orion semantic constants remain strict."
        ),
    )
    parser.add_argument(
        "--allow-orion-balance-layout",
        action="store_true",
        help=(
            "Allow an explicitly checksum-bound Orion capacity-balance layout. "
            "The loader still requires satisfied capacity bounds, preserved copy "
            "counts, navigation evidence for every assignment, and shard-count "
            "agreement across the artifact, routing summary, and import bundle."
        ),
    )
    parser.add_argument(
        "--allow-orion-l1-partition-layout",
        action="store_true",
        help=(
            "Allow a checksum-bound Orion L1 graph prepartition bundle (or its "
            "explicitly ineligible matched L0-informed reference). This is a "
            "separate authorization from the legacy CCNB balance-layout path."
        ),
    )
    parser.add_argument(
        "--cargo-runner",
        default=str(REPO_ROOT / "tools/cargo_in_docker.sh"),
    )
    parser.add_argument("--cargo-target-dir")
    parser.add_argument(
        "--importer-binary",
        help=(
            "Run an already-built orion_numeric_shard_import executable directly "
            "instead of invoking Cargo. Useful for isolated parallel prebuilds."
        ),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--preserve-import-checkpoint",
        action="store_true",
        help=(
            "Temporarily archive an existing routed-import checkpoint, perform a "
            "fresh import, archive the new checkpoint as evidence, and restore the "
            "original checkpoint on exit."
        ),
    )
    parser.add_argument(
        "--defer-artifact-install",
        action="store_true",
        help=(
            "For routed methods, finish collection creation, placement, import, and "
            "indexing without copying the routing artifact or restarting the cluster. "
            "This is intended for parallel prebuilds; install and activate artifacts "
            "serially before issuing routed queries."
        ),
    )
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    cluster_tool.validate_run_id(args.run_id)
    cluster_tool.validate_collection_name(args.collection)
    positive = {
        "hnsw-m": args.hnsw_m,
        "ef-construct": args.ef_construct,
        "full-scan-threshold": args.full_scan_threshold,
        "indexing-threshold": args.indexing_threshold,
        "batch-size": args.batch_size,
        "request-timeout-secs": args.request_timeout_secs,
        "smoke-limit": args.smoke_limit,
        "transfer-timeout-secs": args.transfer_timeout_secs,
    }
    for name, value in positive.items():
        if isinstance(value, bool) or value <= 0:
            raise ValueError(f"--{name} must be positive")
    if args.max_indexing_threads < 0:
        raise ValueError("--max-indexing-threads must be non-negative")
    for name, value in (
        ("max-optimization-threads", args.max_optimization_threads),
        ("max-segment-size-kb", args.max_segment_size_kb),
    ):
        if value is not None and value <= 0:
            raise ValueError(f"--{name} must be positive when set")
    if args.transfer_poll_interval_secs < 0:
        raise ValueError("--transfer-poll-interval-secs must be non-negative")
    if args.num_shards is not None and args.num_shards <= 0:
        raise ValueError("--num-shards must be positive")
    if args.method in ROUTED_METHODS:
        if not args.layout_dir:
            raise ValueError(f"--method {args.method} requires --layout-dir")
        if args.hdf5_path:
            raise ValueError("--hdf5-path is only valid for --method hash_all")
    else:
        if not args.hdf5_path:
            raise ValueError("--method hash_all requires --hdf5-path")
        if not args.num_shards:
            raise ValueError("--method hash_all requires --num-shards")
        if args.layout_dir:
            raise ValueError("--layout-dir is only valid for routed methods")
        if args.resume:
            raise ValueError("--resume is only valid for routed import")
        if args.placement_strategy != "round_robin":
            raise ValueError(
                "--placement-strategy layout_size_balanced is only valid for routed methods"
            )
    if args.allow_orion_scaling_layout and args.method != "orion":
        raise ValueError("--allow-orion-scaling-layout is only valid for --method orion")
    if args.allow_orion_balance_layout and args.method != "orion":
        raise ValueError("--allow-orion-balance-layout is only valid for --method orion")
    if args.allow_orion_l1_partition_layout and args.method != "orion":
        raise ValueError(
            "--allow-orion-l1-partition-layout is only valid for --method orion"
        )
    if args.allow_orion_balance_layout and args.allow_orion_l1_partition_layout:
        raise ValueError(
            "--allow-orion-balance-layout and --allow-orion-l1-partition-layout "
            "authorize different layout families and cannot be combined"
        )
    if args.defer_artifact_install and args.method not in ROUTED_METHODS:
        raise ValueError(
            "--defer-artifact-install is only valid for routed methods"
        )
    if args.importer_binary:
        if args.method not in ROUTED_METHODS:
            raise ValueError("--importer-binary is only valid for routed methods")
        importer = Path(args.importer_binary).expanduser().resolve()
        if not importer.is_file() or not os.access(importer, os.X_OK):
            raise ValueError(
                f"--importer-binary must be an executable file: {importer}"
            )
    if args.preserve_import_checkpoint:
        if args.method not in ROUTED_METHODS:
            raise ValueError(
                "--preserve-import-checkpoint is only valid for routed methods"
            )
        if args.resume:
            raise ValueError(
                "--preserve-import-checkpoint cannot be combined with --resume"
            )


def create_output_directory(path: str | Path) -> Path:
    output = Path(path).expanduser().resolve()
    if output == REPO_ROOT or REPO_ROOT in output.parents:
        raise ValueError(f"preparation output must be outside the repository: {output}")
    output.mkdir(parents=True, exist_ok=False)
    return output


def safe_child(root: Path, name: Any, label: str) -> Path:
    if not isinstance(name, str) or not name or Path(name).name != name:
        raise ValueError(f"{label} must be one file in the layout directory")
    path = (root / name).resolve()
    if path.parent != root:
        raise ValueError(f"{label} escapes the layout directory")
    return path


def verify_checksum_listing(layout_dir: Path) -> dict[str, str]:
    checksums_path = layout_dir / layout_common.CHECKSUMS_NAME
    if not checksums_path.is_file():
        raise FileNotFoundError(f"layout checksums not found: {checksums_path}")
    verified: dict[str, str] = {}
    for line_number, line in enumerate(
        checksums_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line:
            continue
        try:
            expected, relative = line.split("  ", 1)
        except ValueError as exc:
            raise ValueError(
                f"invalid layout checksum line {line_number}: {line!r}"
            ) from exc
        path = safe_child(layout_dir, relative, "layout checksum path")
        if not path.is_file():
            raise FileNotFoundError(f"layout checksum target not found: {path}")
        expected = cluster_tool.normalize_sha256(expected)
        actual = layout_common.sha256_path(path)
        if actual != expected:
            raise RuntimeError(
                f"layout checksum mismatch for {relative}: expected={expected}, actual={actual}"
            )
        verified[relative] = actual
    if not verified:
        raise ValueError("layout checksums file is empty")
    return verified


def validate_faithful_orion_build_parameters(
    build_parameters: dict[str, Any],
    artifact_payload: dict[str, Any],
    *,
    allow_scaling_initial_num_shards: bool = False,
    allow_balance_layout: bool = False,
) -> int:
    faithful_constants = {
        "sample_denominator": 32,
        "upper_sample_seed": 100,
        "upper_m": 32,
        "upper_ef_construction": 100,
        "k_overlap": 10,
        "kmeans_iters": 10,
        "kmeans_seed": 1,
        "topology_iters": 50,
        "multi_assign_min_max_vote": 2,
        "multi_assign_vote_delta": 0,
        "multi_assign_max_shards": 0,
        "upper_graph_seed": 100,
        "allow_decoupled_runtime_upper_search": False,
    }
    balance_mode = str(build_parameters.get("balance_mode") or "none")
    if allow_balance_layout:
        if balance_mode not in {
            "capacity_constrained",
            "post_layout_capacity_constrained",
        }:
            raise RuntimeError(
                "--allow-orion-balance-layout requires an explicit capacity balance mode"
            )
        if type(build_parameters.get("use_multi_assign")) is not bool:
            raise RuntimeError("balanced Orion use_multi_assign must be boolean")
        if type(build_parameters.get("enable_fission")) is not bool:
            raise RuntimeError("balanced Orion enable_fission must be boolean")
        if (
            balance_mode == "capacity_constrained"
            and build_parameters.get("enable_fission") is not False
        ):
            raise RuntimeError(
                "fixed-P capacity_constrained Orion layout must disable fission"
            )
    else:
        faithful_constants.update(
            {
                "use_multi_assign": True,
                "enable_fission": True,
            }
        )
    initial_num_shards = build_parameters.get("initial_num_shards")
    if (
        isinstance(initial_num_shards, bool)
        or not isinstance(initial_num_shards, int)
        or initial_num_shards <= 0
    ):
        raise RuntimeError("Orion initial_num_shards must be a positive integer")
    if not allow_scaling_initial_num_shards and not allow_balance_layout:
        faithful_constants["initial_num_shards"] = 31
    semantic_drift = {
        key: {"expected": expected, "actual": build_parameters.get(key)}
        for key, expected in faithful_constants.items()
        if build_parameters.get(key) != expected
        or type(build_parameters.get(key)) is not type(expected)
    }
    if semantic_drift:
        raise RuntimeError(
            "refusing non-faithful Orion layout: main-idea parameter drift: "
            f"{semantic_drift}"
        )
    attachment_search_ef = build_parameters.get("attachment_search_ef")
    if (
        isinstance(attachment_search_ef, bool)
        or not isinstance(attachment_search_ef, int)
        or attachment_search_ef <= 0
    ):
        raise RuntimeError("Orion layout does not prove a positive attachment_search_ef")
    if attachment_search_ef != 100:
        raise RuntimeError(
            "refusing non-faithful Orion layout: attachment_search_ef must be 100"
        )
    runtime_bindings = {
        "upper_k": artifact_payload.get("upper_k"),
        "upper_search_ef": artifact_payload.get("upper_ef_search"),
        "dynamic_ef_base": artifact_payload.get("dynamic_ef_base"),
        "dynamic_ef_factor": artifact_payload.get("dynamic_ef_factor"),
    }
    mismatched_runtime = {
        key: {"manifest": build_parameters.get(key), "artifact": value}
        for key, value in runtime_bindings.items()
        if build_parameters.get(key) != value
    }
    if mismatched_runtime:
        raise RuntimeError(
            "Orion build manifest/runtime artifact mismatch: "
            f"{mismatched_runtime}"
        )
    if build_parameters.get("upper_search_ef") != build_parameters.get("upper_k"):
        raise RuntimeError(
            "refusing non-faithful Orion layout: runtime upper_search_ef "
            "must equal upper_k"
        )
    return attachment_search_ef


def validate_orion_balance_layout(
    build_manifest: dict[str, Any],
    build_parameters: dict[str, Any],
    artifact_payload: dict[str, Any],
) -> dict[str, Any]:
    routing = build_manifest.get("routing")
    if not isinstance(routing, dict):
        raise RuntimeError("balanced Orion layout is missing routing diagnostics")
    diagnostics = routing.get("balance_diagnostics")
    if not isinstance(diagnostics, dict):
        raise RuntimeError("balanced Orion layout is missing balance_diagnostics")
    mode = str(build_parameters.get("balance_mode") or "none")
    if diagnostics.get("mode") != mode:
        raise RuntimeError("balanced Orion mode differs between parameters and diagnostics")
    if diagnostics.get("fixed_num_shards") is not True:
        raise RuntimeError("balanced Orion layout does not prove a fixed final P")
    stage_vote_loss_contract: tuple[int, int] | None = None
    if (
        "balance_l1_max_vote_loss" in build_parameters
        or "balance_l0_max_vote_loss" in build_parameters
    ):
        shared_vote_loss = build_parameters.get("balance_max_vote_loss")
        l1_vote_loss = build_parameters.get("balance_l1_max_vote_loss")
        l0_vote_loss = build_parameters.get("balance_l0_max_vote_loss")
        for name, value in {
            "balance_max_vote_loss": shared_vote_loss,
            "balance_l1_max_vote_loss": l1_vote_loss,
            "balance_l0_max_vote_loss": l0_vote_loss,
        }.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise RuntimeError(
                    f"balanced Orion {name} must be a non-negative integer"
                )
        stage_vote_loss_contract = (int(l1_vote_loss), int(l0_vote_loss))
        if diagnostics.get("l1_max_vote_loss") != stage_vote_loss_contract[0]:
            raise RuntimeError(
                "balanced Orion L1 vote-loss contract differs between parameters "
                "and diagnostics"
            )
        if diagnostics.get("l0_max_vote_loss") != stage_vote_loss_contract[1]:
            raise RuntimeError(
                "balanced Orion L0 vote-loss contract differs between parameters "
                "and diagnostics"
            )
    l1_bounds_satisfied: bool | None = None
    if mode == "capacity_constrained":
        l1 = diagnostics.get("l1_topology")
        if not isinstance(l1, dict):
            raise RuntimeError("balanced Orion layout is missing L1 topology proof")
        l1_required_true = {
            "bounds_satisfied": l1.get("bounds_satisfied"),
        }
        l1_failed = sorted(
            key for key, value in l1_required_true.items() if value is not True
        )
        if l1_failed:
            raise RuntimeError(
                f"balanced Orion L1 topology failed required balance proofs: {l1_failed}"
            )
        l1_over = l1.get("over_upper_shards")
        l1_under = l1.get("under_lower_shards")
        if not isinstance(l1_over, list) or not isinstance(l1_under, list):
            raise RuntimeError(
                "balanced Orion L1 topology proof lacks explicit residual capacity lists"
            )
        if l1_over != [] or l1_under != []:
            raise RuntimeError(
                "balanced Orion L1 topology reports residual capacity violations"
            )
        if (
            stage_vote_loss_contract is not None
            and l1.get("max_vote_loss") != stage_vote_loss_contract[0]
        ):
            raise RuntimeError(
                "balanced Orion L1 topology used a different vote-loss bound"
            )
        l1_bounds_satisfied = True
    l0 = diagnostics.get("l0_physical_copies")
    if not isinstance(l0, dict):
        raise RuntimeError("balanced Orion layout is missing L0 physical-copy proof")
    required_true = {
        "bounds_satisfied": l0.get("bounds_satisfied"),
        "copy_count_preserved": l0.get("copy_count_preserved"),
        "all_assignments_have_navigation_evidence": l0.get(
            "all_assignments_have_navigation_evidence"
        ),
    }
    failed = sorted(key for key, value in required_true.items() if value is not True)
    if failed:
        raise RuntimeError(
            f"balanced Orion layout failed required balance proofs: {failed}"
        )
    if int(l0.get("non_evidence_assignment_count") or 0) != 0:
        raise RuntimeError("balanced Orion layout contains non-evidence assignments")
    if int(l0.get("no_evidence_points") or 0) != 0:
        raise RuntimeError("balanced Orion layout contains points without navigation evidence")
    l0_over = l0.get("over_upper_shards")
    l0_under = l0.get("under_lower_shards")
    if not isinstance(l0_over, list) or not isinstance(l0_under, list):
        raise RuntimeError(
            "balanced Orion L0 physical-copy proof lacks explicit residual capacity lists"
        )
    if l0_over != [] or l0_under != []:
        raise RuntimeError("balanced Orion layout reports residual capacity violations")
    if (
        stage_vote_loss_contract is not None
        and l0.get("configured_max_vote_loss") != stage_vote_loss_contract[1]
    ):
        raise RuntimeError(
            "balanced Orion L0 placement used a different vote-loss bound"
        )

    final = l0.get("final")
    if not isinstance(final, dict) or not isinstance(final.get("loads"), list):
        raise RuntimeError("balanced Orion layout is missing final load histogram")
    final_loads = final["loads"]
    routing_loads = routing.get("shard_counts")
    if final_loads != routing_loads:
        raise RuntimeError("balanced Orion final loads differ from routing shard_counts")
    if len(final_loads) != artifact_payload.get("shard_count"):
        raise RuntimeError("balanced Orion load histogram length differs from shard_count")
    if any(
        isinstance(load, bool) or not isinstance(load, int) or load <= 0
        for load in final_loads
    ):
        raise RuntimeError("balanced Orion final loads must be positive integers")
    if sum(final_loads) != artifact_payload.get("physical_point_count"):
        raise RuntimeError(
            "balanced Orion final loads do not sum to physical_point_count"
        )
    if l0.get("requested_total_copies") != artifact_payload.get(
        "physical_point_count"
    ):
        raise RuntimeError(
            "balanced Orion requested copy count differs from physical_point_count"
        )
    if mode == "capacity_constrained":
        if diagnostics.get("fission_applied") is not False:
            raise RuntimeError("fixed-P balanced Orion unexpectedly applied fission")
        if routing.get("initial_num_shards") != artifact_payload.get("shard_count"):
            raise RuntimeError("fixed-P balanced Orion changed the configured shard count")

    proof = {
        "mode": mode,
        "bounds_satisfied": True,
        "l1_bounds_satisfied": l1_bounds_satisfied,
        "l0_bounds_satisfied": True,
        "copy_count_preserved": True,
        "all_assignments_have_navigation_evidence": True,
        "min_load": min(final_loads),
        "max_load": max(final_loads),
        "physical_point_count": sum(final_loads),
    }
    if stage_vote_loss_contract is not None:
        proof["l1_max_vote_loss"] = stage_vote_loss_contract[0]
        proof["l0_max_vote_loss"] = stage_vote_loss_contract[1]
    return proof


def _canonical_manifest_sha256(
    payload: dict[str, Any],
    field: str,
    label: str,
) -> str:
    value = payload.get(field)
    if not isinstance(value, str):
        raise RuntimeError(f"{label} is missing {field}")
    try:
        normalized = cluster_tool.normalize_sha256(value)
    except ValueError as exc:
        raise RuntimeError(f"{label} {field} is not a valid SHA-256") from exc
    if value != normalized:
        raise RuntimeError(
            f"{label} {field} must be 64 canonical lowercase hexadecimal characters"
        )
    return normalized


def _required_manifest_file(
    payload: dict[str, Any],
    field: str,
    label: str,
) -> Path:
    value = payload.get(field)
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"{label} is missing {field}")
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} {field} file is missing: {path}")
    return path


def _read_manifest_object(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {label} JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} root must be a JSON object")
    return payload


def _validate_orion_l1_v2_protocol_evidence(
    build_manifest: dict[str, Any],
    diagnostics: dict[str, Any],
    screen_manifest: dict[str, Any],
    mass_manifest: dict[str, Any],
    frozen_candidate: dict[str, Any],
    final_candidate: dict[str, Any],
    owner_path: Path,
    owner_sha256: str,
) -> dict[str, Any]:
    """Bind the promoted v2 estimator to its amendment and frozen raw owner."""

    provenance = build_manifest.get("provenance")
    if not isinstance(provenance, dict):
        raise RuntimeError("v2 navigation-mass candidate is missing provenance")
    amendment_path = _required_manifest_file(
        provenance,
        "protocol_amendment",
        "Orion L1 v2 protocol provenance",
    )
    protocol_path = _required_manifest_file(
        provenance,
        "protocol",
        "Orion L1 v2 protocol provenance",
    )
    if amendment_path != ORION_L1_PROTOCOL_AMENDMENT_PATH.resolve():
        raise RuntimeError("v2 candidate does not bind the repository amendment")
    if protocol_path != ORION_L1_PROTOCOL_PATH.resolve():
        raise RuntimeError("v2 candidate does not bind the repository protocol")
    amendment_sha256 = layout_common.sha256_path(amendment_path)
    protocol_sha256 = layout_common.sha256_path(protocol_path)
    if (
        _canonical_manifest_sha256(
            diagnostics,
            "protocol_amendment_sha256",
            "Orion L1 v2 diagnostics",
        )
        != amendment_sha256
    ):
        raise RuntimeError("v2 protocol amendment checksum differs from diagnostics")

    amendment = final_candidate.get("protocol_amendment")
    if not isinstance(amendment, dict):
        raise RuntimeError("v2 candidate lacks its protocol amendment record")
    if frozen_candidate.get("protocol_amendment") != amendment:
        raise RuntimeError("v2 protocol amendment changed after owner freeze")
    if screen_manifest.get("protocol_amendment") not in (None, amendment):
        raise RuntimeError("v2 screen protocol amendment differs from candidate")
    if mass_manifest.get("protocol_amendment") != amendment:
        raise RuntimeError("v2 mass manifest protocol amendment differs from candidate")
    amendment_expected = {
        "format_version": 1,
        "id": ORION_L1_PROTOCOL_AMENDMENT_ID,
        "status": "post_exploratory_promotion_not_preregistered",
        "mass_mode": ORION_L1_MASS_MODE,
        "source": ORION_L1_MASS_SOURCE,
        "transform": ORION_L1_MASS_TRANSFORM,
        "estimator_version": ORION_L1_MASS_ESTIMATOR_VERSION,
        "method": ORION_L1_MASS_METHOD,
        "topology_gates_unchanged": True,
        "old_raw_owner_byte_parity_required": True,
        "online_qps_only_confirmation": True,
        "amendment_path": str(amendment_path),
        "amendment_sha256": amendment_sha256,
        "protocol_path": str(protocol_path),
        "protocol_sha256": protocol_sha256,
    }
    amendment_mismatches = {
        key: {"expected": expected, "actual": amendment.get(key)}
        for key, expected in amendment_expected.items()
        if amendment.get(key) != expected
        or type(amendment.get(key)) is not type(expected)
    }
    if amendment_mismatches:
        raise RuntimeError(
            "v2 protocol amendment record mismatch: "
            f"{amendment_mismatches}"
        )
    amendment_file = _read_manifest_object(
        amendment_path,
        "Orion L1 v2 protocol amendment file",
    )
    file_expected = {
        key: value
        for key, value in amendment_expected.items()
        if key
        not in {
            "amendment_path",
            "amendment_sha256",
            "protocol_path",
            "protocol_sha256",
        }
    }
    amendment_file_mismatches = {
        key: {"expected": expected, "actual": amendment_file.get(key)}
        for key, expected in file_expected.items()
        if amendment_file.get(key) != expected
        or type(amendment_file.get(key)) is not type(expected)
    }
    if amendment_file_mismatches:
        raise RuntimeError(
            "repository v2 amendment contract mismatch: "
            f"{amendment_file_mismatches}"
        )

    parity = final_candidate.get("owner_parity")
    if not isinstance(parity, dict):
        raise RuntimeError("v2 candidate lacks frozen raw-owner parity")
    if frozen_candidate.get("owner_parity") != parity:
        raise RuntimeError("v2 owner parity changed after owner freeze")
    if diagnostics.get("owner_parity") != parity:
        raise RuntimeError("v2 owner parity differs between candidate and diagnostics")
    parity_expected = {
        "reference_mass_mode": "raw",
        "reference_owner_sha256": owner_sha256,
        "owner_bytes_identical": True,
    }
    parity_mismatches = {
        key: {"expected": expected, "actual": parity.get(key)}
        for key, expected in parity_expected.items()
        if parity.get(key) != expected
        or type(parity.get(key)) is not type(expected)
    }
    if parity_mismatches:
        raise RuntimeError(f"v2 old-raw owner parity mismatch: {parity_mismatches}")

    reference_manifest_path = _required_manifest_file(
        parity,
        "reference_candidate_manifest_path",
        "Orion L1 v2 owner parity",
    )
    reference_manifest_sha256 = _canonical_manifest_sha256(
        parity,
        "reference_candidate_manifest_sha256",
        "Orion L1 v2 owner parity",
    )
    if layout_common.sha256_path(reference_manifest_path) != reference_manifest_sha256:
        raise RuntimeError("v2 raw parity-reference manifest checksum mismatch")
    reference_sidecar = reference_manifest_path.with_name(
        reference_manifest_path.name + ".sha256"
    )
    if (
        not reference_sidecar.is_file()
        or reference_sidecar.read_text(encoding="ascii").strip()
        != reference_manifest_sha256
    ):
        raise RuntimeError("v2 raw parity-reference sidecar mismatch")
    reference_manifest = _read_manifest_object(
        reference_manifest_path,
        "Orion L1 v2 raw parity-reference manifest",
    )
    reference_parameters = reference_manifest.get("parameters")
    if (
        reference_manifest.get("format_version") != 1
        or reference_manifest.get("stage") != "upper_only_candidates_frozen"
        or not isinstance(reference_parameters, dict)
        or reference_parameters.get("mass_mode") != "raw"
    ):
        raise RuntimeError("v2 owner parity reference is not the frozen raw ablation")
    raw_candidates = reference_manifest.get("candidates")
    raw_matches = [
        candidate
        for candidate in raw_candidates if isinstance(candidate, dict)
    ] if isinstance(raw_candidates, list) else []
    raw_matches = [
        candidate
        for candidate in raw_matches
        if candidate.get("method") == ORION_L1_MASS_METHOD
    ]
    if len(raw_matches) != 1:
        raise RuntimeError(
            "v2 owner parity reference lacks one mass-balanced-kmeans candidate"
        )
    raw_candidate = raw_matches[0]
    raw_expected = {
        "mass_mode": "raw",
        "navigation_mass_source": ORION_L1_RAW_MASS_SOURCE,
        "navigation_mass_transform": ORION_L1_RAW_MASS_TRANSFORM,
        "source_artifact_sha256": final_candidate.get("source_artifact_sha256"),
        "owner_sha256": owner_sha256,
    }
    raw_mismatches = {
        key: {"expected": expected, "actual": raw_candidate.get(key)}
        for key, expected in raw_expected.items()
        if raw_candidate.get(key) != expected
        or type(raw_candidate.get(key)) is not type(expected)
    }
    if raw_mismatches:
        raise RuntimeError(f"v2 raw parity-reference candidate mismatch: {raw_mismatches}")
    raw_owner_path = _required_manifest_file(
        raw_candidate,
        "owner_path",
        "Orion L1 v2 raw parity-reference candidate",
    )
    if parity.get("reference_owner_path") != str(raw_owner_path):
        raise RuntimeError("v2 owner parity path differs from raw reference")
    if (
        layout_common.sha256_path(raw_owner_path) != owner_sha256
        or layout_common.sha256_path(owner_path) != owner_sha256
    ):
        raise RuntimeError("v2 and frozen raw owner bytes are not identical")

    return {
        "amendment_sha256": amendment_sha256,
        "protocol_sha256": protocol_sha256,
        "raw_reference_manifest_sha256": reference_manifest_sha256,
        "raw_reference_owner_sha256": owner_sha256,
        "owner_bytes_identical": True,
    }


def _validate_orion_l1_cross_dataset_confirmation(
    build_manifest: dict[str, Any],
    diagnostics: dict[str, Any],
    screen_manifest_path: Path,
    final_candidate: dict[str, Any],
    protocol_amendment: dict[str, Any],
) -> dict[str, Any]:
    """Require the checksum-bound SIFT+GloVe offline confirmation gate."""

    provenance = build_manifest.get("provenance")
    if not isinstance(provenance, dict):
        raise RuntimeError("v2 cross-dataset confirmation is missing provenance")
    confirmation_path = _required_manifest_file(
        provenance,
        "cross_dataset_confirmation",
        "Orion L1 v2 cross-dataset confirmation provenance",
    )
    confirmation_sha256 = _canonical_manifest_sha256(
        diagnostics,
        "cross_dataset_confirmation_sha256",
        "Orion L1 v2 diagnostics",
    )
    if layout_common.sha256_path(confirmation_path) != confirmation_sha256:
        raise RuntimeError("v2 cross-dataset confirmation checksum mismatch")
    confirmation_sidecar = confirmation_path.with_name(
        confirmation_path.name + ".sha256"
    )
    if (
        not confirmation_sidecar.is_file()
        or confirmation_sidecar.read_text(encoding="ascii").strip()
        != confirmation_sha256
    ):
        raise RuntimeError("v2 cross-dataset confirmation sidecar mismatch")
    confirmation = _read_manifest_object(
        confirmation_path,
        "Orion L1 v2 cross-dataset confirmation",
    )
    top_expected = {
        "format_version": 1,
        "stage": "raw_regularized_v2_cross_dataset_offline_confirmation",
        "status": "materialization_eligible_online_qps_pending",
        "cross_dataset_identity_all_pass": True,
        "cross_dataset_topology_all_pass": True,
        "owner_parity_all_pass": True,
        "materialization_eligible": True,
        "online_qps_confirmation_required": True,
    }
    top_mismatches = {
        key: {"expected": expected, "actual": confirmation.get(key)}
        for key, expected in top_expected.items()
        if confirmation.get(key) != expected
        or type(confirmation.get(key)) is not type(expected)
    }
    if top_mismatches:
        raise RuntimeError(
            "v2 cross-dataset confirmation verdict mismatch: "
            f"{top_mismatches}"
        )
    method_expected = {
        "mass_mode": ORION_L1_MASS_MODE,
        "source": ORION_L1_MASS_SOURCE,
        "transform": ORION_L1_MASS_TRANSFORM,
        "estimator_version": ORION_L1_MASS_ESTIMATOR_VERSION,
        "method": ORION_L1_MASS_METHOD,
        "upper_only_phase_a": True,
        "l0_evaluation_only_after_owner_freeze": True,
        "multi_assignment_unchanged": True,
    }
    confirmation_contract_expected = {
        "alternate_method_or_parameter_scan": False,
        "old_raw_owner_byte_parity_required": True,
        "online_cluster_entered_by_this_stage": False,
        "online_qps_confirmation_completed": False,
        "online_qps_confirmation_required": True,
        "online_qps_only_confirmation": True,
        "post_exploratory_not_preregistered": True,
        "topology_thresholds_unchanged": True,
        "v1_heldout_failure_retained": True,
    }
    for field, expected in (
        ("method_contract", method_expected),
        ("confirmation_contract", confirmation_contract_expected),
    ):
        if confirmation.get(field) != expected:
            raise RuntimeError(f"v2 cross-dataset {field} mismatch")
    if confirmation.get("protocol_amendment") != protocol_amendment:
        raise RuntimeError("v2 cross-dataset protocol amendment mismatch")
    builder = confirmation.get("builder")
    if not isinstance(builder, dict):
        raise RuntimeError("v2 cross-dataset confirmation lacks builder provenance")
    builder_path = _required_manifest_file(
        builder,
        "path",
        "Orion L1 v2 confirmation builder",
    )
    if builder_path != ORION_L1_V2_CONFIRMATION_BUILDER_PATH.resolve():
        raise RuntimeError("v2 confirmation was not built by the repository builder")
    if layout_common.sha256_path(builder_path) != _canonical_manifest_sha256(
        builder,
        "sha256",
        "Orion L1 v2 confirmation builder",
    ):
        raise RuntimeError("v2 confirmation builder checksum mismatch")

    datasets = confirmation.get("datasets")
    if not isinstance(datasets, dict) or set(datasets) != {"sift", "glove"}:
        raise RuntimeError("v2 confirmation must contain exactly SIFT and GloVe")
    matching_current_dataset = False
    for dataset_name in ("sift", "glove"):
        record = datasets[dataset_name]
        if not isinstance(record, dict):
            raise RuntimeError(f"v2 {dataset_name} confirmation record is invalid")
        record_expected = {
            "dataset": dataset_name,
            "logical_shards": 32,
            "graph_topology_all_pass": True,
            "identity_all_pass": True,
            "topology_all_pass": True,
            "owner_bytes_identical": True,
            "owner_parity_all_pass": True,
            "materialization_eligible": True,
        }
        record_mismatches = {
            key: {"expected": expected, "actual": record.get(key)}
            for key, expected in record_expected.items()
            if record.get(key) != expected
            or type(record.get(key)) is not type(expected)
        }
        if record_mismatches:
            raise RuntimeError(
                f"v2 {dataset_name} confirmation verdict mismatch: "
                f"{record_mismatches}"
            )
        bound_paths: dict[str, Path] = {}
        for prefix in (
            "phase_a_candidate_manifest",
            "phase_b_screen_manifest",
            "raw_reference_candidate_manifest",
            "raw_reference_owner",
            "source_artifact",
            "owner",
        ):
            path = _required_manifest_file(
                record,
                f"{prefix}_path",
                f"Orion L1 v2 {dataset_name} confirmation",
            )
            digest = _canonical_manifest_sha256(
                record,
                f"{prefix}_sha256",
                f"Orion L1 v2 {dataset_name} confirmation",
            )
            if layout_common.sha256_path(path) != digest:
                raise RuntimeError(
                    f"v2 {dataset_name} {prefix} checksum mismatch"
                )
            if prefix in {
                "phase_a_candidate_manifest",
                "phase_b_screen_manifest",
                "raw_reference_candidate_manifest",
            }:
                sidecar = path.with_name(path.name + ".sha256")
                if (
                    not sidecar.is_file()
                    or sidecar.read_text(encoding="ascii").strip() != digest
                ):
                    raise RuntimeError(
                        f"v2 {dataset_name} {prefix} sidecar mismatch"
                    )
            bound_paths[prefix] = path
        phase_a = _read_manifest_object(
            bound_paths["phase_a_candidate_manifest"],
            f"Orion L1 v2 {dataset_name} phase-A manifest",
        )
        phase_b = _read_manifest_object(
            bound_paths["phase_b_screen_manifest"],
            f"Orion L1 v2 {dataset_name} phase-B manifest",
        )
        if (
            phase_a.get("stage") != "upper_only_candidates_frozen"
            or phase_b.get("stage") != "post_freeze_evaluation"
            or phase_b.get("frozen_candidate_manifest")
            != str(bound_paths["phase_a_candidate_manifest"])
            or phase_b.get("frozen_candidate_manifest_sha256")
            != record["phase_a_candidate_manifest_sha256"]
        ):
            raise RuntimeError(f"v2 {dataset_name} phase-A/phase-B chain mismatch")

        def one_method_candidate(payload: Any, label: str) -> dict[str, Any]:
            matches = [
                candidate
                for candidate in payload
                if isinstance(candidate, dict)
                and candidate.get("method") == ORION_L1_MASS_METHOD
            ] if isinstance(payload, list) else []
            if len(matches) != 1:
                raise RuntimeError(
                    f"v2 {dataset_name} {label} lacks one locked candidate"
                )
            return matches[0]

        phase_a_candidate = one_method_candidate(
            phase_a.get("candidates"),
            "phase-A",
        )
        phase_b_candidate = one_method_candidate(
            phase_b.get("candidates"),
            "phase-B",
        )
        phase_a_expected = {
            "graph_topology_all_pass": True,
            "topology_all_pass": False,
            "eligibility_pending_post_freeze_query_gates": True,
            "materialization_eligible": False,
        }
        phase_b_expected = {
            "identity_all_pass": True,
            "topology_gate_inputs_complete": True,
            "topology_all_pass": True,
            "owner_parity_all_pass": True,
            "eligibility_pending_post_freeze_query_gates": False,
            "materialization_eligible": True,
        }
        for candidate, expected, label in (
            (phase_a_candidate, phase_a_expected, "phase-A"),
            (phase_b_candidate, phase_b_expected, "phase-B"),
        ):
            if any(candidate.get(key) is not value for key, value in expected.items()):
                raise RuntimeError(
                    f"v2 {dataset_name} {label} eligibility state mismatch"
                )
        owner_sha256 = record["owner_sha256"]
        if (
            phase_a_candidate.get("owner_sha256") != owner_sha256
            or phase_b_candidate.get("owner_sha256") != owner_sha256
            or phase_b_candidate.get("owner_path") != str(bound_paths["owner"])
            or record.get("raw_reference_owner_sha256") != owner_sha256
        ):
            raise RuntimeError(f"v2 {dataset_name} owner parity binding mismatch")
        if phase_b_candidate.get("protocol_amendment") != protocol_amendment:
            raise RuntimeError(f"v2 {dataset_name} amendment binding mismatch")
        owner_parity = phase_b_candidate.get("owner_parity")
        if not isinstance(owner_parity, dict) or any(
            owner_parity.get(key) != record.get(record_key)
            for key, record_key in (
                ("reference_candidate_manifest_sha256", "raw_reference_candidate_manifest_sha256"),
                ("reference_owner_sha256", "raw_reference_owner_sha256"),
            )
        ):
            raise RuntimeError(f"v2 {dataset_name} raw-owner parity mismatch")
        if (
            owner_parity.get("reference_mass_mode") != "raw"
            or owner_parity.get("owner_bytes_identical") is not True
            or owner_parity.get("reference_candidate_manifest_path")
            != str(bound_paths["raw_reference_candidate_manifest"])
            or owner_parity.get("reference_owner_path")
            != str(bound_paths["raw_reference_owner"])
        ):
            raise RuntimeError(f"v2 {dataset_name} raw-owner parity path mismatch")
        raw_reference = _read_manifest_object(
            bound_paths["raw_reference_candidate_manifest"],
            f"Orion L1 v2 {dataset_name} raw reference",
        )
        raw_candidates = raw_reference.get("candidates")
        raw_matches = [
            candidate
            for candidate in raw_candidates
            if isinstance(candidate, dict)
            and candidate.get("method") == ORION_L1_MASS_METHOD
        ] if isinstance(raw_candidates, list) else []
        if (
            raw_reference.get("stage") != "upper_only_candidates_frozen"
            or (raw_reference.get("parameters") or {}).get("mass_mode") != "raw"
            or len(raw_matches) != 1
            or raw_matches[0].get("mass_mode") != "raw"
            or raw_matches[0].get("navigation_mass_source")
            != ORION_L1_RAW_MASS_SOURCE
            or raw_matches[0].get("navigation_mass_transform")
            != ORION_L1_RAW_MASS_TRANSFORM
            or raw_matches[0].get("owner_sha256") != owner_sha256
            or raw_matches[0].get("owner_path")
            != str(bound_paths["raw_reference_owner"])
        ):
            raise RuntimeError(f"v2 {dataset_name} raw reference contract mismatch")
        topology_gates = phase_b_candidate.get("topology_gates")
        if not isinstance(topology_gates, dict) or not topology_gates:
            raise RuntimeError(f"v2 {dataset_name} topology gates are missing")
        if record.get("topology_gates") != topology_gates:
            raise RuntimeError(f"v2 {dataset_name} topology gates drifted")
        for metric, gate in topology_gates.items():
            if not isinstance(gate, dict) or gate.get("pass") is not True:
                raise RuntimeError(f"v2 {dataset_name} topology gate {metric} failed")
            operator = gate.get("operator")
            observed = gate.get("observed")
            threshold = gate.get("threshold")
            if (
                isinstance(observed, bool)
                or not isinstance(observed, (int, float))
                or isinstance(threshold, bool)
                or not isinstance(threshold, (int, float))
                or operator not in {"<=", ">="}
            ):
                raise RuntimeError(
                    f"v2 {dataset_name} topology gate {metric} is invalid"
                )
            passed = observed <= threshold if operator == "<=" else observed >= threshold
            if not passed:
                raise RuntimeError(
                    f"v2 {dataset_name} topology gate {metric} pass bit is false"
                )
        if bound_paths["phase_b_screen_manifest"] == screen_manifest_path:
            matching_current_dataset = True
            if (
                owner_sha256 != final_candidate.get("owner_sha256")
                or record.get("materialization_parity")
                != final_candidate.get("materialization_parity")
            ):
                raise RuntimeError(
                    "current v2 bundle differs from cross-dataset confirmation"
                )
    if not matching_current_dataset:
        raise RuntimeError("current v2 screen is absent from cross-dataset confirmation")

    v1_failure = confirmation.get("v1_heldout_failure")
    if not isinstance(v1_failure, dict) or any(
        v1_failure.get(key) != expected
        for key, expected in {
            "dataset": "glove",
            "mass_mode": "self-debiased-floor1",
            "method": ORION_L1_MASS_METHOD,
            "topology_all_pass": False,
            "materialization_eligible": False,
        }.items()
    ):
        raise RuntimeError("v2 confirmation did not retain the v1 held-out failure")
    failed_gates = v1_failure.get("failed_gates")
    if not isinstance(failed_gates, dict) or not failed_gates or any(
        not isinstance(gate, dict) or gate.get("pass") is not False
        for gate in failed_gates.values()
    ):
        raise RuntimeError("v2 confirmation v1 failure evidence is invalid")
    v1_screen_path = _required_manifest_file(
        v1_failure,
        "screen_manifest_path",
        "Orion L1 v1 held-out failure",
    )
    if layout_common.sha256_path(v1_screen_path) != _canonical_manifest_sha256(
        v1_failure,
        "screen_manifest_sha256",
        "Orion L1 v1 held-out failure",
    ):
        raise RuntimeError("v1 held-out failure screen checksum mismatch")
    return {
        "manifest_sha256": confirmation_sha256,
        "identity_all_pass": True,
        "topology_all_pass": True,
        "owner_parity_all_pass": True,
        "online_qps_confirmation_required": True,
    }


def _validate_orion_l1_navigation_mass(
    build_manifest: dict[str, Any],
    artifact_payload: dict[str, Any],
    diagnostics: dict[str, Any],
    screen_manifest: dict[str, Any],
    screen_manifest_path: Path,
    owner: Any,
    owner_sizes: list[int],
    shard_count: int,
    upper_point_count: int,
    diagnostic_hashes: dict[str, str],
) -> dict[str, Any]:
    if diagnostics.get("method") != ORION_L1_MASS_METHOD:
        raise RuntimeError(
            "navigation-mass candidate is not the locked v2 confirmation method"
        )
    mass_balance = diagnostics.get("mass_balance")
    if not isinstance(mass_balance, dict):
        raise RuntimeError(
            "navigation-mass candidate is missing mass_balance proof"
        )
    if mass_balance.get("materialization_eligible") is not True:
        raise RuntimeError(
            "navigation-mass candidate is not eligible for materialization"
        )
    fixed_mass_contract = {
        "method": ORION_L1_MASS_METHOD,
        "owner_sha256": diagnostic_hashes["owner_sha256"],
        "mass_mode": ORION_L1_MASS_MODE,
        "navigation_mass_source": ORION_L1_MASS_SOURCE,
        "navigation_mass_transform": ORION_L1_MASS_TRANSFORM,
        "navigation_mass_estimator_version": ORION_L1_MASS_ESTIMATOR_VERSION,
    }
    mass_contract_mismatches = {
        key: {"expected": expected, "actual": mass_balance.get(key)}
        for key, expected in fixed_mass_contract.items()
        if mass_balance.get(key) != expected
        or type(mass_balance.get(key)) is not type(expected)
    }
    if mass_contract_mismatches:
        raise RuntimeError(
            "navigation-mass candidate fixed contract mismatch: "
            f"{mass_contract_mismatches}"
        )
    if mass_balance.get("partition_sizes") != owner_sizes:
        raise RuntimeError(
            "navigation-mass candidate partition_sizes differ from owner counts"
        )
    if min(owner_sizes) <= 0:
        raise RuntimeError(
            "navigation-mass candidate must assign at least one L1 node to every shard"
        )

    mass_hashes = {
        field: _canonical_manifest_sha256(
            mass_balance,
            field,
            "Orion L1 navigation-mass proof",
        )
        for field in (
            "navigation_mass_sha256",
            "navigation_mass_manifest_sha256",
            "upper_navigation_hits_sha256",
            "upper_navigation_manifest_sha256",
        )
    }
    provenance = build_manifest.get("provenance")
    if not isinstance(provenance, dict):
        raise RuntimeError("navigation-mass candidate is missing provenance")
    hits_path = _required_manifest_file(
        provenance,
        "upper_navigation_hits",
        "Orion L1 navigation-mass provenance",
    )
    hits_manifest_path = _required_manifest_file(
        provenance,
        "upper_navigation_manifest",
        "Orion L1 navigation-mass provenance",
    )
    mass_manifest_path = _required_manifest_file(
        provenance,
        "navigation_mass_manifest",
        "Orion L1 navigation-mass provenance",
    )
    for path, digest_field in (
        (hits_path, "upper_navigation_hits_sha256"),
        (hits_manifest_path, "upper_navigation_manifest_sha256"),
        (mass_manifest_path, "navigation_mass_manifest_sha256"),
    ):
        if layout_common.sha256_path(path) != mass_hashes[digest_field]:
            raise RuntimeError(
                f"navigation-mass candidate {digest_field} file mismatch"
            )

    construction_inputs = screen_manifest.get("construction_inputs")
    if not isinstance(construction_inputs, dict):
        raise RuntimeError(
            "navigation-mass screen is missing construction_inputs"
        )
    construction_expected = {
        "artifact_sha256": diagnostic_hashes["source_artifact_sha256"],
        "upper_node_count": upper_point_count,
        "upper_navigation_hits_sha256": mass_hashes[
            "upper_navigation_hits_sha256"
        ],
        "upper_navigation_manifest_sha256": mass_hashes[
            "upper_navigation_manifest_sha256"
        ],
        "upper_navigation_top_k": ORION_L1_MASS_TOP_K,
        "upper_navigation_search_ef": ORION_L1_MASS_SEARCH_EF,
    }
    construction_mismatches = {
        key: {"expected": expected, "actual": construction_inputs.get(key)}
        for key, expected in construction_expected.items()
        if construction_inputs.get(key) != expected
    }
    if construction_mismatches:
        raise RuntimeError(
            "navigation-mass construction input binding mismatch: "
            f"{construction_mismatches}"
        )
    for field in (
        "navigator_sha256",
        "upper_graph_sha256",
        "upper_input_manifest_sha256",
        "ordered_labels_sha256",
        "ordered_vectors_sha256",
        "partitioner_source_sha256",
    ):
        _canonical_manifest_sha256(
            construction_inputs,
            field,
            "Orion L1 navigation-mass construction inputs",
        )
    for path_field, expected_path in (
        ("upper_navigation_hits", hits_path),
        ("upper_navigation_manifest", hits_manifest_path),
    ):
        screen_path = _required_manifest_file(
            construction_inputs,
            path_field,
            "Orion L1 navigation-mass construction inputs",
        )
        if screen_path != expected_path:
            raise RuntimeError(
                f"navigation-mass {path_field} path differs between screen and build"
            )

    upper_nodes = artifact_payload.get("upper_nodes")
    if not isinstance(upper_nodes, list) or len(upper_nodes) != upper_point_count:
        raise RuntimeError(
            "navigation-mass artifact upper_nodes differ from upper_point_count"
        )
    try:
        upper_labels = experiment.np.asarray(
            [int(node["label"]) for node in upper_nodes],
            dtype=experiment.np.int64,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "navigation-mass artifact contains invalid upper labels"
        ) from exc
    logical_point_count = artifact_payload.get("logical_point_count")
    if (
        len(experiment.np.unique(upper_labels)) != upper_point_count
        or experiment.np.any(upper_labels < 0)
        or experiment.np.any(upper_labels >= logical_point_count)
    ):
        raise RuntimeError(
            "navigation-mass artifact upper labels are not unique numeric point IDs"
        )

    hits_manifest = _read_manifest_object(
        hits_manifest_path,
        "Orion L1 self-hit manifest",
    )
    self_hit_expected = {
        "format_version": 1,
        "artifact_sha256": diagnostic_hashes["source_artifact_sha256"],
        "row_count": upper_point_count,
        "top_k": ORION_L1_MASS_TOP_K,
        "search_ef": ORION_L1_MASS_SEARCH_EF,
        "hits_sha256": mass_hashes["upper_navigation_hits_sha256"],
        "hits_size_bytes": upper_point_count * ORION_L1_MASS_TOP_K * 8,
    }
    self_hit_mismatches = {
        key: {"expected": expected, "actual": hits_manifest.get(key)}
        for key, expected in self_hit_expected.items()
        if hits_manifest.get(key) != expected
        or type(hits_manifest.get(key)) is not type(expected)
    }
    if self_hit_mismatches:
        raise RuntimeError(
            "navigation-mass self-hit manifest contract mismatch: "
            f"{self_hit_mismatches}"
        )
    manifest_hits_path = _required_manifest_file(
        hits_manifest,
        "hits_path",
        "Orion L1 self-hit manifest",
    )
    if manifest_hits_path != hits_path:
        raise RuntimeError(
            "navigation-mass self-hit file path differs between manifest and build"
        )
    if hits_path.stat().st_size != self_hit_expected["hits_size_bytes"]:
        raise RuntimeError("navigation-mass self-hit file length mismatch")

    hits = experiment.np.memmap(
        hits_path,
        dtype="<u8",
        mode="r",
        shape=(upper_point_count, ORION_L1_MASS_TOP_K),
    )
    self_hits = experiment.np.asarray(hits) == upper_labels[:, None]
    self_hit_counts = experiment.np.sum(self_hits, axis=1)
    if not experiment.np.all(self_hit_counts == 1):
        raise RuntimeError(
            "navigation-mass self-hit rows must contain their own upper label exactly once"
        )
    self_first_count = int(experiment.np.count_nonzero(self_hits[:, 0]))
    sorted_hits = experiment.np.sort(experiment.np.asarray(hits), axis=1)
    if experiment.np.any(sorted_hits[:, 1:] == sorted_hits[:, :-1]):
        raise RuntimeError("navigation-mass self-hit rows contain duplicate labels")
    maximum_label = max(int(upper_labels.max()), int(experiment.np.max(hits)))
    if maximum_label >= logical_point_count:
        raise RuntimeError("navigation-mass self-hit file contains an invalid label")
    label_to_local = experiment.np.full(maximum_label + 1, -1, dtype="<i4")
    label_to_local[upper_labels] = experiment.np.arange(
        upper_point_count,
        dtype="<i4",
    )
    local_hits = label_to_local[experiment.np.asarray(hits)]
    if experiment.np.any(local_hits < 0):
        raise RuntimeError(
            "navigation-mass self-hit file contains a label outside the upper graph"
        )
    upper_vectors = experiment.np.asarray(
        [node.get("vector") for node in upper_nodes],
        dtype="<f4",
    )
    if upper_vectors.ndim != 2 or len(upper_vectors) != upper_point_count:
        raise RuntimeError("navigation-mass artifact contains invalid upper vectors")
    duplicate_tie_records: list[dict[str, Any]] = []
    for row in range(upper_point_count):
        self_rank = int(experiment.np.flatnonzero(local_hits[row] == row)[0])
        if self_rank == 0:
            continue
        query_bits = experiment.np.ascontiguousarray(upper_vectors[row]).tobytes()
        preceding = local_hits[row, :self_rank]
        if any(
            experiment.np.ascontiguousarray(upper_vectors[int(hit)]).tobytes()
            != query_bits
            for hit in preceding
        ):
            raise RuntimeError(
                "navigation-mass self hit is preceded by a non-duplicate upper vector"
            )
        duplicate_tie_records.append(
            {
                "row": row,
                "query_label": int(upper_labels[row]),
                "self_rank": self_rank,
                "preceding_labels": [
                    int(upper_labels[int(hit)]) for hit in preceding
                ],
                "vector_bits_sha256": hashlib.sha256(query_bits).hexdigest(),
            }
        )
    duplicate_tie_proof_sha256 = canonical_json_sha256(duplicate_tie_records)
    raw_counts = experiment.np.bincount(
        local_hits.reshape(-1),
        minlength=upper_point_count,
    ).astype(experiment.np.int64, copy=False)
    # V2 is the frozen raw count.  Its interpretation is
    # ``non_self_hit_count + unit_l1_prior`` because every validated upper
    # self-query contributes exactly one native L1 unit to its own vertex.
    mass = raw_counts
    total_mass = int(mass.sum())
    mass_digest = hashlib.sha256()
    mass_digest.update(ORION_L1_MASS_SOURCE.encode("ascii"))
    mass_digest.update(b"\0")
    mass_digest.update(ORION_L1_MASS_TRANSFORM.encode("ascii"))
    mass_digest.update(
        struct.pack(
            "<QQ",
            upper_point_count,
            ORION_L1_MASS_TOP_K,
        )
    )
    mass_digest.update(mass.astype("<u8", copy=False).tobytes(order="C"))
    recomputed_mass_sha256 = mass_digest.hexdigest()
    if recomputed_mass_sha256 != mass_hashes["navigation_mass_sha256"]:
        raise RuntimeError("navigation-mass checksum differs from self-hit replay")

    mass_manifest = _read_manifest_object(
        mass_manifest_path,
        "Orion L1 navigation-mass manifest",
    )
    mass_manifest_expected = {
        "format_version": 1,
        "mass_mode": ORION_L1_MASS_MODE,
        "source": ORION_L1_MASS_SOURCE,
        "transform": ORION_L1_MASS_TRANSFORM,
        "estimator_version": ORION_L1_MASS_ESTIMATOR_VERSION,
        "semantic_sha256": recomputed_mass_sha256,
        "total_mass": total_mass,
        "source_artifact_sha256": diagnostic_hashes["source_artifact_sha256"],
        "upper_graph_sha256": construction_inputs["upper_graph_sha256"],
        "upper_navigation_hits_sha256": mass_hashes[
            "upper_navigation_hits_sha256"
        ],
        "upper_navigation_manifest_sha256": mass_hashes[
            "upper_navigation_manifest_sha256"
        ],
        "upper_node_count": upper_point_count,
        "top_k": ORION_L1_MASS_TOP_K,
        "search_ef": ORION_L1_MASS_SEARCH_EF,
        "self_present_count": upper_point_count,
        "self_first_count": self_first_count,
        "self_first_fraction": self_first_count / upper_point_count,
        "duplicate_tie_exception_count": len(duplicate_tie_records),
        "duplicate_tie_proof_sha256": duplicate_tie_proof_sha256,
    }
    mass_manifest_mismatches = {
        key: {"expected": expected, "actual": mass_manifest.get(key)}
        for key, expected in mass_manifest_expected.items()
        if mass_manifest.get(key) != expected
        or type(mass_manifest.get(key)) is not type(expected)
    }
    if mass_manifest_mismatches:
        raise RuntimeError(
            "navigation-mass manifest replay contract mismatch: "
            f"{mass_manifest_mismatches}"
        )
    for manifest_field, construction_field in (
        ("navigator_sha256", "navigator_sha256"),
        ("ordered_labels_sha256", "ordered_labels_sha256"),
        ("ordered_vectors_sha256", "ordered_vectors_sha256"),
        ("partitioner_source_sha256", "partitioner_source_sha256"),
    ):
        manifest_digest = _canonical_manifest_sha256(
            mass_manifest,
            manifest_field,
            "Orion L1 navigation-mass manifest",
        )
        if manifest_digest != construction_inputs[construction_field]:
            raise RuntimeError(
                f"navigation-mass manifest {manifest_field} binding mismatch"
            )
    mass_values_path = _required_manifest_file(
        mass_manifest,
        "values_file",
        "Orion L1 navigation-mass manifest",
    )
    mass_values_sha256 = _canonical_manifest_sha256(
        mass_manifest,
        "values_sha256",
        "Orion L1 navigation-mass manifest",
    )
    if layout_common.sha256_path(mass_values_path) != mass_values_sha256:
        raise RuntimeError("navigation-mass value file checksum mismatch")
    if (
        mass_manifest.get("values_size_bytes") != upper_point_count * 8
        or mass_values_path.stat().st_size != upper_point_count * 8
    ):
        raise RuntimeError("navigation-mass value file length mismatch")
    declared_mass = experiment.np.memmap(
        mass_values_path,
        dtype="<u8",
        mode="r",
        shape=(upper_point_count,),
    )
    if not experiment.np.array_equal(declared_mass, mass):
        raise RuntimeError(
            "navigation-mass value file differs from self-hit replay"
        )

    mass_estimator = screen_manifest.get("mass_estimator")
    if not isinstance(mass_estimator, dict):
        raise RuntimeError("navigation-mass screen is missing mass_estimator")
    estimator_expected = {
        "format_version": 1,
        "source": ORION_L1_MASS_SOURCE,
        "transform": ORION_L1_MASS_TRANSFORM,
        "estimator_version": ORION_L1_MASS_ESTIMATOR_VERSION,
        "mass_mode": ORION_L1_MASS_MODE,
        "sha256": recomputed_mass_sha256,
        "manifest_path": str(mass_manifest_path),
        "manifest_sha256": mass_hashes["navigation_mass_manifest_sha256"],
        "query_count": upper_point_count,
        "top_k": ORION_L1_MASS_TOP_K,
        "total_mass": total_mass,
        "self_present_count": upper_point_count,
        "self_first_count": self_first_count,
        "self_first_fraction": self_first_count / upper_point_count,
        "duplicate_tie_exception_count": len(duplicate_tie_records),
        "duplicate_tie_proof_sha256": duplicate_tie_proof_sha256,
        "rank_weighting": False,
        "eligible_as_finalist": True,
    }
    estimator_mismatches = {
        key: {"expected": expected, "actual": mass_estimator.get(key)}
        for key, expected in estimator_expected.items()
        if mass_estimator.get(key) != expected
        or type(mass_estimator.get(key)) is not type(expected)
    }
    if estimator_mismatches:
        raise RuntimeError(
            "navigation-mass screen estimator mismatch: "
            f"{estimator_mismatches}"
        )
    if total_mass != upper_point_count * ORION_L1_MASS_TOP_K:
        raise RuntimeError("v2 navigation mass total differs from U * top-k")

    expected_mass_target = total_mass / shard_count
    expected_mass_limit = math.ceil(expected_mass_target) + int(mass.max()) - 1
    declared_target = mass_balance.get("estimated_mass_target")
    if (
        isinstance(declared_target, bool)
        or not isinstance(declared_target, (int, float))
        or not math.isfinite(declared_target)
        or float(declared_target) != expected_mass_target
    ):
        raise RuntimeError(
            "navigation-mass estimated_mass_target differs from self-hit replay"
        )
    if mass_balance.get("estimated_mass_limit") != expected_mass_limit:
        raise RuntimeError(
            "navigation-mass estimated_mass_limit differs from fixed bound"
        )
    estimated_loads = mass_balance.get("estimated_partition_masses")
    if not isinstance(estimated_loads, list) or len(estimated_loads) != shard_count:
        raise RuntimeError(
            "navigation-mass estimated_partition_masses length differs from shard_count"
        )
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in estimated_loads
    ):
        raise RuntimeError(
            "navigation-mass estimated_partition_masses must be positive integers"
        )
    recomputed_loads = experiment.np.zeros(shard_count, dtype=experiment.np.int64)
    experiment.np.add.at(recomputed_loads, owner, mass)
    actual_loads = [int(value) for value in recomputed_loads.tolist()]
    if estimated_loads != actual_loads:
        raise RuntimeError(
            "navigation-mass estimated partition loads differ from self-hit replay"
        )
    if sum(actual_loads) != total_mass or max(actual_loads) > expected_mass_limit:
        raise RuntimeError("navigation-mass candidate violates its fixed capacity bound")

    frozen_manifest_path = _required_manifest_file(
        screen_manifest,
        "frozen_candidate_manifest",
        "Orion L1 navigation-mass screen",
    )
    frozen_manifest_sha256 = _canonical_manifest_sha256(
        screen_manifest,
        "frozen_candidate_manifest_sha256",
        "Orion L1 navigation-mass screen",
    )
    if layout_common.sha256_path(frozen_manifest_path) != frozen_manifest_sha256:
        raise RuntimeError(
            "navigation-mass frozen candidate manifest checksum mismatch"
        )
    frozen_manifest = _read_manifest_object(
        frozen_manifest_path,
        "Orion L1 frozen navigation-mass candidate manifest",
    )
    if (
        frozen_manifest.get("format_version") != 1
        or frozen_manifest.get("stage") != "upper_only_candidates_frozen"
    ):
        raise RuntimeError(
            "navigation-mass frozen candidate manifest has an invalid stage"
        )
    frozen_contract = frozen_manifest.get("contract")
    frozen_contract_expected = {
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
    if not isinstance(frozen_contract, dict):
        raise RuntimeError("navigation-mass frozen candidate has no contract")
    frozen_contract_mismatches = {
        key: {"expected": expected, "actual": frozen_contract.get(key)}
        for key, expected in frozen_contract_expected.items()
        if frozen_contract.get(key) != expected
        or type(frozen_contract.get(key)) is not type(expected)
    }
    if frozen_contract_mismatches:
        raise RuntimeError(
            "navigation-mass frozen candidate input-isolation mismatch: "
            f"{frozen_contract_mismatches}"
        )
    if frozen_manifest.get("mass_estimator") != mass_estimator:
        raise RuntimeError(
            "navigation-mass estimator differs between frozen and final screen manifests"
        )
    if frozen_manifest.get("construction_inputs") != construction_inputs:
        raise RuntimeError(
            "navigation-mass construction inputs differ after candidate freeze"
        )

    candidate_common_expected = {
        "method": ORION_L1_MASS_METHOD,
        "balance_contract": ORION_L1_MASS_BALANCE_CONTRACT,
        "source_artifact_sha256": diagnostic_hashes["source_artifact_sha256"],
        "owner_sha256": diagnostic_hashes["owner_sha256"],
        "navigation_mass_manifest_sha256": mass_hashes[
            "navigation_mass_manifest_sha256"
        ],
        "navigation_mass_source": ORION_L1_MASS_SOURCE,
        "navigation_mass_transform": ORION_L1_MASS_TRANSFORM,
        "navigation_mass_sha256": recomputed_mass_sha256,
        "navigation_mass_estimator_version": ORION_L1_MASS_ESTIMATOR_VERSION,
        "mass_mode": ORION_L1_MASS_MODE,
        "partition_sizes": owner_sizes,
        "estimated_partition_masses": actual_loads,
        "estimated_mass_target": expected_mass_target,
        "estimated_mass_limit": expected_mass_limit,
    }

    def unique_candidate(payload: Any, label: str) -> dict[str, Any]:
        if not isinstance(payload, list):
            raise RuntimeError(f"navigation-mass {label} has no candidates list")
        matches = [
            row
            for row in payload
            if isinstance(row, dict)
            and row.get("method") == diagnostics.get("method")
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"navigation-mass {label} does not uniquely identify the method"
            )
        return matches[0]

    frozen_candidate = unique_candidate(
        frozen_manifest.get("candidates"),
        "frozen candidate manifest",
    )
    frozen_candidate_expected = {
        **candidate_common_expected,
        "topology_all_pass": False,
        "eligibility_pending_post_freeze_query_gates": True,
        "materialization_eligible": False,
    }
    frozen_candidate_mismatches = {
        key: {"expected": expected, "actual": frozen_candidate.get(key)}
        for key, expected in frozen_candidate_expected.items()
        if frozen_candidate.get(key) != expected
        or type(frozen_candidate.get(key)) is not type(expected)
    }
    if frozen_candidate_mismatches:
        raise RuntimeError(
            "navigation-mass frozen candidate record mismatch: "
            f"{frozen_candidate_mismatches}"
        )
    frozen_owner_path = _required_manifest_file(
        frozen_candidate,
        "owner_path",
        "Orion L1 frozen navigation-mass candidate",
    )
    if (
        layout_common.sha256_path(frozen_owner_path)
        != diagnostic_hashes["owner_sha256"]
    ):
        raise RuntimeError("navigation-mass frozen owner checksum mismatch")

    final_candidate = unique_candidate(
        screen_manifest.get("candidates"),
        "final screen",
    )
    final_candidate_expected = {
        **candidate_common_expected,
        "attachments_sha256": diagnostic_hashes["attachments_sha256"],
        "upper_navigation_hits_sha256": mass_hashes[
            "upper_navigation_hits_sha256"
        ],
        "upper_navigation_manifest_sha256": mass_hashes[
            "upper_navigation_manifest_sha256"
        ],
        "identity_all_pass": True,
        "topology_gate_inputs_complete": True,
        "topology_all_pass": True,
        "owner_parity_all_pass": True,
        "eligibility_pending_post_freeze_query_gates": False,
        "materialization_eligible": True,
    }
    final_candidate_mismatches = {
        key: {"expected": expected, "actual": final_candidate.get(key)}
        for key, expected in final_candidate_expected.items()
        if final_candidate.get(key) != expected
        or type(final_candidate.get(key)) is not type(expected)
    }
    if final_candidate_mismatches:
        raise RuntimeError(
            "navigation-mass final screen candidate record mismatch: "
            f"{final_candidate_mismatches}"
        )
    routing = build_manifest.get("routing") or {}
    parity = final_candidate.get("materialization_parity")
    parity_expected = {
        "canonical_format": "orion_numeric_import.assignments.jsonl-v1",
        "assignment_bytes_sha256": artifact_payload.get("layout_sha256"),
        "logical_point_count": artifact_payload.get("logical_point_count"),
        "physical_point_count": artifact_payload.get("physical_point_count"),
        "shard_loads": routing.get("shard_counts"),
        "copy_count_histogram": routing.get("copy_count_histogram"),
    }
    if not isinstance(parity, dict):
        raise RuntimeError(
            "navigation-mass final screen lacks materialization parity proof"
        )
    parity_mismatches = {
        key: {"expected": expected, "actual": parity.get(key)}
        for key, expected in parity_expected.items()
        if parity.get(key) != expected
        or type(parity.get(key)) is not type(expected)
    }
    if parity_mismatches:
        raise RuntimeError(
            "navigation-mass materialization parity mismatch: "
            f"{parity_mismatches}"
        )
    v2_protocol_evidence = _validate_orion_l1_v2_protocol_evidence(
        build_manifest,
        diagnostics,
        screen_manifest,
        mass_manifest,
        frozen_candidate,
        final_candidate,
        frozen_owner_path,
        diagnostic_hashes["owner_sha256"],
    )
    cross_dataset_evidence = _validate_orion_l1_cross_dataset_confirmation(
        build_manifest,
        diagnostics,
        screen_manifest_path,
        final_candidate,
        final_candidate["protocol_amendment"],
    )

    return {
        "source": ORION_L1_MASS_SOURCE,
        "transform": ORION_L1_MASS_TRANSFORM,
        "mass_mode": ORION_L1_MASS_MODE,
        "estimator_version": ORION_L1_MASS_ESTIMATOR_VERSION,
        "sha256": recomputed_mass_sha256,
        "query_count": upper_point_count,
        "top_k": ORION_L1_MASS_TOP_K,
        "search_ef": ORION_L1_MASS_SEARCH_EF,
        "total_mass": total_mass,
        "partition_masses": actual_loads,
        "mass_target": expected_mass_target,
        "mass_limit": expected_mass_limit,
        "frozen_candidate_manifest_sha256": frozen_manifest_sha256,
        "navigation_mass_manifest_sha256": mass_hashes[
            "navigation_mass_manifest_sha256"
        ],
        "v2_protocol": v2_protocol_evidence,
        "cross_dataset_confirmation": cross_dataset_evidence,
    }


def _validate_orion_l1_upper_replay(
    build_manifest: dict[str, Any],
    artifact_payload: dict[str, Any],
    diagnostics: dict[str, Any],
    layout_dir: Path,
    checksums: dict[str, str],
    source_artifact_path: Path,
    source_artifact_sha256: str,
) -> dict[str, Any]:
    outputs = build_manifest.get("outputs")
    if not isinstance(outputs, dict):
        raise RuntimeError("Orion L1 partition layout is missing outputs")
    artifact_path = safe_child(
        layout_dir,
        outputs.get("production_artifact"),
        "production_artifact",
    )
    replay_path = safe_child(
        layout_dir,
        outputs.get("upper_replay"),
        "upper_replay",
    )
    replay_manifest_path = safe_child(
        layout_dir,
        outputs.get("upper_replay_manifest"),
        "upper_replay_manifest",
    )
    provenance = build_manifest.get("provenance")
    if not isinstance(provenance, dict):
        raise RuntimeError("Orion L1 upper replay is missing provenance")
    for field, expected_path in (
        ("upper_replay", replay_path),
        ("upper_replay_manifest", replay_manifest_path),
    ):
        if (
            _required_manifest_file(
                provenance,
                field,
                "Orion L1 upper replay provenance",
            )
            != expected_path
        ):
            raise RuntimeError(f"Orion L1 upper replay provenance {field} mismatch")
    verifier_path = _required_manifest_file(
        provenance,
        "upper_replay_verifier",
        "Orion L1 upper replay provenance",
    )
    verifier_sha256 = _canonical_manifest_sha256(
        provenance,
        "upper_replay_verifier_sha256",
        "Orion L1 upper replay provenance",
    )
    if layout_common.sha256_path(verifier_path) != verifier_sha256:
        raise RuntimeError("Orion L1 upper replay verifier checksum mismatch")
    provenance_query_path = _required_manifest_file(
        provenance,
        "upper_replay_queries",
        "Orion L1 upper replay provenance",
    )
    provenance_query_sha256 = _canonical_manifest_sha256(
        provenance,
        "upper_replay_queries_sha256",
        "Orion L1 upper replay provenance",
    )
    if layout_common.sha256_path(provenance_query_path) != provenance_query_sha256:
        raise RuntimeError("Orion L1 upper replay query checksum mismatch")
    for path, label in (
        (artifact_path, "production artifact"),
        (replay_path, "upper replay"),
        (replay_manifest_path, "upper replay manifest"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Orion L1 partition {label} is missing: {path}")
        relative = path.relative_to(layout_dir).as_posix()
        if relative not in checksums:
            raise RuntimeError(
                f"Orion L1 partition {label} is not covered by layout checksums"
            )
        declared_files = outputs.get("files")
        declared = (
            declared_files.get(relative)
            if isinstance(declared_files, dict)
            else None
        )
        if not isinstance(declared, dict):
            raise RuntimeError(
                f"Orion L1 partition outputs/files does not declare {label}"
            )
        if (
            _canonical_manifest_sha256(
                declared,
                "sha256",
                f"Orion L1 partition {label} output",
            )
            != checksums[relative]
            or declared.get("size_bytes") != path.stat().st_size
        ):
            raise RuntimeError(
                f"Orion L1 partition {label} output binding mismatch"
            )

    replay_manifest_sha256 = _canonical_manifest_sha256(
        diagnostics,
        "upper_replay_manifest_sha256",
        "Orion L1 partition diagnostics",
    )
    if (
        checksums[replay_manifest_path.relative_to(layout_dir).as_posix()]
        != replay_manifest_sha256
    ):
        raise RuntimeError("Orion L1 upper replay manifest checksum mismatch")
    replay_manifest = _read_manifest_object(
        replay_manifest_path,
        "Orion L1 upper replay manifest",
    )
    if replay_manifest.get("format_version") != 1:
        raise RuntimeError("unsupported Orion L1 upper replay manifest format")
    if replay_manifest.get("verdict") != "PASS":
        raise RuntimeError("Orion L1 upper replay manifest verdict is not PASS")
    source = replay_manifest.get("source")
    rebound = replay_manifest.get("rebound")
    query = replay_manifest.get("query_corpus")
    replay = replay_manifest.get("replay")
    gates = replay_manifest.get("gates")
    if not all(
        isinstance(value, dict)
        for value in (source, rebound, query, replay, gates)
    ):
        raise RuntimeError("Orion L1 upper replay manifest proof is incomplete")

    actual_artifact_sha256 = layout_common.sha256_path(artifact_path)
    if source.get("file_sha256") != source_artifact_sha256:
        raise RuntimeError("Orion L1 upper replay source artifact checksum mismatch")
    if rebound.get("file_sha256") != actual_artifact_sha256:
        raise RuntimeError("Orion L1 upper replay rebound artifact checksum mismatch")
    for evidence, expected_path, label in (
        (source, source_artifact_path, "source"),
        (rebound, artifact_path, "rebound"),
    ):
        evidence_path = _required_manifest_file(
            evidence,
            "path",
            f"Orion L1 upper replay {label}",
        )
        if evidence_path != expected_path:
            raise RuntimeError(
                f"Orion L1 upper replay {label} artifact path mismatch"
            )
        for field in (
            "file_sha256",
            "canonical_artifact_sha256",
            "layout_sha256",
            "canonical_upper_graph_sha256",
            "ordered_upper_nodes_identity_sha256",
        ):
            _canonical_manifest_sha256(
                evidence,
                field,
                f"Orion L1 upper replay {label}",
            )
    if (
        source.get("canonical_upper_graph_sha256")
        != rebound.get("canonical_upper_graph_sha256")
        or source.get("canonical_upper_graph_size_bytes")
        != rebound.get("canonical_upper_graph_size_bytes")
    ):
        raise RuntimeError("Orion L1 upper replay canonical graph differs")
    if (
        source.get("ordered_upper_nodes_identity_sha256")
        != rebound.get("ordered_upper_nodes_identity_sha256")
    ):
        raise RuntimeError("Orion L1 upper replay ordered upper nodes differ")
    if (
        isinstance(source.get("generation"), bool)
        or not isinstance(source.get("generation"), int)
        or source.get("generation") >= artifact_payload.get("generation")
        or rebound.get("generation") != artifact_payload.get("generation")
    ):
        raise RuntimeError("Orion L1 upper replay generation contract mismatch")
    rebound_expected = {
        "layout_sha256": artifact_payload.get("layout_sha256"),
        "shard_count": artifact_payload.get("shard_count"),
        "physical_point_count": artifact_payload.get("physical_point_count"),
    }
    rebound_mismatches = {
        key: {"expected": expected, "actual": rebound.get(key)}
        for key, expected in rebound_expected.items()
        if rebound.get(key) != expected
        or type(rebound.get(key)) is not type(expected)
    }
    if rebound_mismatches:
        raise RuntimeError(
            "Orion L1 upper replay rebound artifact binding mismatch: "
            f"{rebound_mismatches}"
        )

    query_path = _required_manifest_file(
        query,
        "path",
        "Orion L1 upper replay query corpus",
    )
    query_sha256 = _canonical_manifest_sha256(
        query,
        "sha256",
        "Orion L1 upper replay query corpus",
    )
    dimension = int((artifact_payload.get("vector_schema") or {}).get("dimension") or 0)
    query_rows = query.get("row_count")
    if (
        layout_common.sha256_path(query_path) != query_sha256
        or query_path != provenance_query_path
        or query_sha256 != provenance_query_sha256
        or isinstance(query_rows, bool)
        or not isinstance(query_rows, int)
        or query_rows <= 0
        or query.get("dimension") != dimension
        or query.get("size_bytes") != query_rows * dimension * 4
        or query_path.stat().st_size != query.get("size_bytes")
    ):
        raise RuntimeError("Orion L1 upper replay query corpus binding mismatch")

    replay_evidence_path = _required_manifest_file(
        replay,
        "path",
        "Orion L1 upper replay evidence",
    )
    replay_sha256 = _canonical_manifest_sha256(
        replay,
        "sha256",
        "Orion L1 upper replay evidence",
    )
    ordered_replay_sha256 = _canonical_manifest_sha256(
        replay,
        "ordered_label_and_distance_bits_sha256",
        "Orion L1 upper replay evidence",
    )
    if (
        replay_evidence_path != replay_path
        or layout_common.sha256_path(replay_path) != replay_sha256
        or replay_sha256 != ordered_replay_sha256
        or replay.get("size_bytes") != replay_path.stat().st_size
        or replay.get("upper_k") != artifact_payload.get("upper_k")
        or replay.get("compared_hit_count")
        != query_rows * artifact_payload.get("upper_k")
    ):
        raise RuntimeError("Orion L1 upper replay byte evidence mismatch")
    required_gates = {
        "generation_advanced",
        "immutable_metadata_equal",
        "vector_schema_equal",
        "upper_search_contract_equal",
        "canonical_upper_graph_bytes_equal",
        "ordered_upper_labels_and_vector_bits_equal",
        "ordered_hit_labels_equal",
        "ordered_distance_bits_equal",
        "production_router_replay_complete",
    }
    if set(gates) != required_gates or any(gates[key] is not True for key in gates):
        raise RuntimeError("Orion L1 upper replay did not pass every required gate")

    return {
        "verdict": "PASS",
        "manifest_sha256": replay_manifest_sha256,
        "replay_sha256": replay_sha256,
        "query_sha256": query_sha256,
        "canonical_upper_graph_sha256": rebound[
            "canonical_upper_graph_sha256"
        ],
        "ordered_upper_nodes_identity_sha256": rebound[
            "ordered_upper_nodes_identity_sha256"
        ],
        "compared_hit_count": replay["compared_hit_count"],
    }


def validate_orion_l1_partition_layout(
    build_manifest: dict[str, Any],
    build_parameters: dict[str, Any],
    artifact_payload: dict[str, Any],
    layout_dir: Path,
    checksums: dict[str, str],
) -> dict[str, Any]:
    """Validate the lightweight L1-prepartition family without treating it as CCNB."""
    mode = build_parameters.get("balance_mode")
    if mode not in ORION_L1_PARTITION_MODES:
        raise RuntimeError(
            "Orion L1 partition layout has an unsupported balance_mode: "
            f"{mode!r}"
        )
    is_candidate = mode == "l1_graph_prepartition"

    parameter_contract = {
        "sample_denominator": 32,
        "upper_sample_seed": 100,
        "upper_m": 32,
        "upper_ef_construction": 100,
        "upper_graph_seed": 100,
        "attachment_search_ef": 100,
        "k_overlap": 10,
        "use_multi_assign": True,
        "multi_assign_min_max_vote": 2,
        "multi_assign_vote_delta": 0,
        "multi_assign_max_shards": 0,
        "enable_fission": False,
        "enable_topology_refinement": False,
        "l0_repair": False,
        "allow_decoupled_runtime_upper_search": False,
    }
    parameter_drift = {
        key: {"expected": expected, "actual": build_parameters.get(key)}
        for key, expected in parameter_contract.items()
        if build_parameters.get(key) != expected
        or type(build_parameters.get(key)) is not type(expected)
    }
    if parameter_drift:
        raise RuntimeError(
            "Orion L1 partition layout violates the lightweight/multi-assignment "
            f"contract: {parameter_drift}"
        )

    shard_count = artifact_payload.get("shard_count")
    logical_point_count = artifact_payload.get("logical_point_count")
    physical_point_count = artifact_payload.get("physical_point_count")
    generation = artifact_payload.get("generation")
    for field, value in {
        "shard_count": shard_count,
        "logical_point_count": logical_point_count,
        "physical_point_count": physical_point_count,
        "generation": generation,
    }.items():
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise RuntimeError(f"Orion L1 partition artifact has invalid {field}")
    if physical_point_count < logical_point_count:
        raise RuntimeError(
            "Orion L1 partition physical_point_count is smaller than logical_point_count"
        )
    if build_parameters.get("initial_num_shards") != shard_count:
        raise RuntimeError(
            "Orion L1 partition initial_num_shards differs from artifact shard_count"
        )
    if build_parameters.get("generation") != generation:
        raise RuntimeError(
            "Orion L1 partition generation differs between parameters and artifact"
        )
    runtime_bindings = {
        "upper_k": artifact_payload.get("upper_k"),
        "upper_search_ef": artifact_payload.get("upper_ef_search"),
        "dynamic_ef_base": artifact_payload.get("dynamic_ef_base"),
        "dynamic_ef_factor": artifact_payload.get("dynamic_ef_factor"),
    }
    runtime_mismatches = {
        key: {"expected": value, "actual": build_parameters.get(key)}
        for key, value in runtime_bindings.items()
        if build_parameters.get(key) != value
    }
    if runtime_mismatches:
        raise RuntimeError(
            "Orion L1 partition runtime artifact binding mismatch: "
            f"{runtime_mismatches}"
        )
    if build_parameters.get("upper_search_ef") != build_parameters.get("upper_k"):
        raise RuntimeError(
            "Orion L1 partition runtime upper_search_ef must equal upper_k"
        )

    artifact_binding = build_manifest.get("artifact_binding")
    if not isinstance(artifact_binding, dict):
        raise RuntimeError("Orion L1 partition layout is missing artifact_binding")
    expected_artifact_binding = {
        "generation": generation,
        "layout_sha256": artifact_payload.get("layout_sha256"),
        "logical_point_count": logical_point_count,
        "physical_point_count": physical_point_count,
        "shard_count": shard_count,
    }
    binding_mismatches = {
        key: {"expected": expected, "actual": artifact_binding.get(key)}
        for key, expected in expected_artifact_binding.items()
        if artifact_binding.get(key) != expected
    }
    if binding_mismatches:
        raise RuntimeError(
            "Orion L1 partition artifact_binding mismatch: "
            f"{binding_mismatches}"
        )

    routing = build_manifest.get("routing")
    if not isinstance(routing, dict):
        raise RuntimeError("Orion L1 partition layout is missing routing summary")
    count_contract = {
        "initial_num_shards": shard_count,
        "effective_num_shards": shard_count,
        "logical_point_count": logical_point_count,
        "physical_point_count": physical_point_count,
    }
    count_mismatches = {
        key: {"expected": expected, "actual": routing.get(key)}
        for key, expected in count_contract.items()
        if routing.get(key) != expected
        or type(routing.get(key)) is not type(expected)
    }
    if count_mismatches:
        raise RuntimeError(
            "Orion L1 partition routing count mismatch: "
            f"{count_mismatches}"
        )
    if routing.get("fission_events") != []:
        raise RuntimeError("Orion L1 partition layout must not contain fission events")
    shard_counts = routing.get("shard_counts")
    if not isinstance(shard_counts, list) or len(shard_counts) != shard_count:
        raise RuntimeError(
            "Orion L1 partition shard_counts length differs from shard_count"
        )
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in shard_counts
    ):
        raise RuntimeError(
            "Orion L1 partition shard_counts must contain positive integers"
        )
    if sum(shard_counts) != physical_point_count:
        raise RuntimeError(
            "Orion L1 partition shard_counts do not sum to physical_point_count"
        )

    copy_histogram = routing.get("copy_count_histogram")
    if not isinstance(copy_histogram, dict) or not copy_histogram:
        raise RuntimeError("Orion L1 partition layout is missing copy_count_histogram")
    histogram_rows = 0
    histogram_copies = 0
    for raw_copy_count, frequency in copy_histogram.items():
        try:
            copy_count = int(raw_copy_count)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                "Orion L1 partition copy_count_histogram has a non-integer key"
            ) from exc
        if str(copy_count) != raw_copy_count or copy_count <= 0:
            raise RuntimeError(
                "Orion L1 partition copy_count_histogram keys must be canonical "
                "positive integers"
            )
        if isinstance(frequency, bool) or not isinstance(frequency, int) or frequency <= 0:
            raise RuntimeError(
                "Orion L1 partition copy_count_histogram frequencies must be "
                "positive integers"
            )
        histogram_rows += frequency
        histogram_copies += copy_count * frequency
    if histogram_rows != logical_point_count or histogram_copies != physical_point_count:
        raise RuntimeError(
            "Orion L1 partition copy_count_histogram disagrees with logical/physical "
            "point counts"
        )

    diagnostics = routing.get("l1_partition_diagnostics")
    if not isinstance(diagnostics, dict):
        raise RuntimeError(
            "Orion L1 partition layout is missing l1_partition_diagnostics"
        )
    balance_contract = diagnostics.get("balance_contract")
    if is_candidate:
        if balance_contract == ORION_L1_UNIT_BALANCE_CONTRACT:
            expected_scope = "production_upper_graph_only"
        elif balance_contract == ORION_L1_MASS_BALANCE_CONTRACT:
            expected_scope = (
                "production_upper_graph_and_upper_navigation_mass_only"
            )
        else:
            raise RuntimeError(
                "eligible Orion L1 partition candidate must declare either "
                f"{ORION_L1_UNIT_BALANCE_CONTRACT!r} or "
                f"{ORION_L1_MASS_BALANCE_CONTRACT!r}"
            )
    else:
        if balance_contract != ORION_L1_REFERENCE_BALANCE_CONTRACT:
            raise RuntimeError(
                "matched L0-informed reference must declare its baseline-only "
                "balance_contract"
            )
        expected_scope = "legacy_l0_informed_reference"
    diagnostic_contract = {
        "input_scope": expected_scope,
        "new_algorithm_eligible": is_candidate,
        "l0_repair": False,
        "multi_assignment_after_owner_freeze": True,
    }
    diagnostic_mismatches = {
        key: {"expected": expected, "actual": diagnostics.get(key)}
        for key, expected in diagnostic_contract.items()
        if diagnostics.get(key) != expected
        or type(diagnostics.get(key)) is not type(expected)
    }
    if diagnostic_mismatches:
        raise RuntimeError(
            "Orion L1 partition eligibility/input-scope contract mismatch: "
            f"{diagnostic_mismatches}"
        )
    partitioner = build_parameters.get("l1_partitioner")
    if not isinstance(partitioner, str) or not partitioner:
        raise RuntimeError("Orion L1 partition layout has no l1_partitioner label")
    if diagnostics.get("method") != partitioner:
        raise RuntimeError(
            "Orion L1 partition method differs between parameters and diagnostics"
        )
    diagnostic_hashes = {
        field: _canonical_manifest_sha256(
            diagnostics,
            field,
            "Orion L1 partition diagnostics",
        )
        for field in (
            "owner_sha256",
            "source_artifact_sha256",
            "attachments_sha256",
            "screen_manifest_sha256",
        )
    }
    provenance = build_manifest.get("provenance")
    if not isinstance(provenance, dict):
        raise RuntimeError("Orion L1 partition layout is missing provenance")
    source_provenance_sha256 = _canonical_manifest_sha256(
        provenance,
        "source_artifact_sha256",
        "Orion L1 partition provenance",
    )
    if source_provenance_sha256 != diagnostic_hashes["source_artifact_sha256"]:
        raise RuntimeError(
            "Orion L1 partition source artifact checksum differs between provenance "
            "and diagnostics"
        )
    source_artifact_path = _required_manifest_file(
        provenance,
        "source_artifact",
        "Orion L1 partition provenance",
    )
    if layout_common.sha256_path(source_artifact_path) != source_provenance_sha256:
        raise RuntimeError("Orion L1 partition source artifact checksum mismatch")
    upper_replay_proof = _validate_orion_l1_upper_replay(
        build_manifest,
        artifact_payload,
        diagnostics,
        layout_dir,
        checksums,
        source_artifact_path,
        source_provenance_sha256,
    )
    screen_manifest_path = _required_manifest_file(
        provenance,
        "screen_manifest",
        "Orion L1 partition provenance",
    )
    if (
        layout_common.sha256_path(screen_manifest_path)
        != diagnostic_hashes["screen_manifest_sha256"]
    ):
        raise RuntimeError("Orion L1 partition screen manifest checksum mismatch")
    screen_manifest = _read_manifest_object(
        screen_manifest_path,
        "Orion L1 partition screen manifest",
    )
    if is_candidate:
        screen_contract = screen_manifest.get("contract")
        if not isinstance(screen_contract, dict):
            raise RuntimeError(
                "eligible Orion L1 partition screen is missing its input contract"
            )
        screen_contract_expected = {
            "partition_input_scope": expected_scope,
            "partitioner_reads_full_attachments": False,
            "partitioner_reads_l0_load": False,
            "partitioner_reads_multi_assignment_state": False,
            "l0_assignment_after_owner_freeze": True,
            "l0_repair_or_flow": False,
        }
        if balance_contract == ORION_L1_MASS_BALANCE_CONTRACT:
            screen_contract_expected.update(
                {
                    "construction_process_accepts_l0_inputs": False,
                    "construction_process_accepts_query_ground_truth": False,
                    "separate_evaluator_process_required": True,
                    "evaluator_runs_in_separate_process": True,
                    "owners_changed_after_freeze": False,
                }
            )
        screen_contract_mismatches = {
            key: {"expected": expected, "actual": screen_contract.get(key)}
            for key, expected in screen_contract_expected.items()
            if screen_contract.get(key) != expected
            or type(screen_contract.get(key)) is not type(expected)
        }
        if screen_contract_mismatches:
            raise RuntimeError(
                "eligible Orion L1 partition screen input-isolation contract "
                f"mismatch: {screen_contract_mismatches}"
            )
        if screen_contract.get("multi_assignment") != {
            "enabled": True,
            "min_max_vote": 2,
            "vote_delta": 0,
            "max_shards": 0,
        }:
            raise RuntimeError(
                "eligible Orion L1 partition screen changed multi-assignment"
            )

    upper_point_count = routing.get("upper_point_count")
    if (
        isinstance(upper_point_count, bool)
        or not isinstance(upper_point_count, int)
        or upper_point_count <= 0
    ):
        raise RuntimeError("Orion L1 partition upper_point_count must be positive")
    l1_sizes = diagnostics.get("l1_sizes")
    if not isinstance(l1_sizes, list) or len(l1_sizes) != shard_count:
        raise RuntimeError("Orion L1 partition l1_sizes length differs from shard_count")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in l1_sizes
    ):
        raise RuntimeError("Orion L1 partition l1_sizes must be non-negative integers")
    if sum(l1_sizes) != upper_point_count:
        raise RuntimeError(
            "Orion L1 partition l1_sizes do not sum to upper_point_count"
        )

    outputs = build_manifest.get("outputs")
    if not isinstance(outputs, dict):
        raise RuntimeError("Orion L1 partition layout is missing outputs")
    owner_path = safe_child(layout_dir, outputs.get("owner"), "owner")
    if not owner_path.is_file():
        raise FileNotFoundError(f"Orion L1 partition owner file is missing: {owner_path}")
    owner_relative = owner_path.relative_to(layout_dir).as_posix()
    owner_checksum = checksums.get(owner_relative)
    if owner_checksum is None:
        raise RuntimeError("Orion L1 partition owner is not covered by layout checksums")
    if owner_checksum != diagnostic_hashes["owner_sha256"]:
        raise RuntimeError(
            "Orion L1 partition owner checksum differs from diagnostics"
        )
    declared_files = outputs.get("files")
    declared_owner = (
        declared_files.get(owner_relative)
        if isinstance(declared_files, dict)
        else None
    )
    if not isinstance(declared_owner, dict):
        raise RuntimeError("Orion L1 partition outputs/files does not declare owner")
    declared_owner_sha256 = _canonical_manifest_sha256(
        declared_owner,
        "sha256",
        "Orion L1 partition owner output",
    )
    if declared_owner_sha256 != owner_checksum:
        raise RuntimeError("Orion L1 partition owner output checksum mismatch")
    if declared_owner.get("size_bytes") != owner_path.stat().st_size:
        raise RuntimeError("Orion L1 partition owner output size mismatch")
    if owner_path.stat().st_size != upper_point_count * 4:
        raise RuntimeError(
            "Orion L1 partition owner length differs from upper_point_count"
        )
    owner = experiment.np.fromfile(owner_path, dtype="<i4")
    if len(owner) != upper_point_count:
        raise RuntimeError(
            "Orion L1 partition owner row count differs from upper_point_count"
        )
    if experiment.np.any(owner < 0) or experiment.np.any(owner >= shard_count):
        raise RuntimeError("Orion L1 partition owner contains an out-of-range shard")
    actual_l1_sizes = [
        int(value)
        for value in experiment.np.bincount(owner, minlength=shard_count).tolist()
    ]
    if actual_l1_sizes != l1_sizes:
        raise RuntimeError("Orion L1 partition owner counts differ from l1_sizes")
    actual_exact_quota = max(actual_l1_sizes) - min(actual_l1_sizes) <= 1
    if diagnostics.get("exact_quota") is not actual_exact_quota:
        raise RuntimeError(
            "Orion L1 partition exact_quota declaration differs from owner counts"
        )
    if (
        is_candidate
        and balance_contract == ORION_L1_UNIT_BALANCE_CONTRACT
        and not actual_exact_quota
    ):
        raise RuntimeError(
            "unit-weight Orion L1 graph prepartition candidate must satisfy exact quota"
        )
    mass_proof = None
    if is_candidate and balance_contract == ORION_L1_MASS_BALANCE_CONTRACT:
        mass_proof = _validate_orion_l1_navigation_mass(
            build_manifest,
            artifact_payload,
            diagnostics,
            screen_manifest,
            screen_manifest_path,
            owner,
            actual_l1_sizes,
            shard_count,
            upper_point_count,
            diagnostic_hashes,
        )

    return {
        "mode": mode,
        "balance_contract": balance_contract,
        "input_scope": expected_scope,
        "new_algorithm_eligible": is_candidate,
        "baseline_only": not is_candidate,
        "exact_quota": actual_exact_quota,
        "l0_repair": False,
        "multi_assignment": {
            "enabled": True,
            "min_max_vote": 2,
            "vote_delta": 0,
            "max_shards": 0,
        },
        "fission": False,
        "shard_count": shard_count,
        "l1_sizes": actual_l1_sizes,
        "shard_counts": list(shard_counts),
        "checksums": diagnostic_hashes,
        "navigation_mass": mass_proof,
        "upper_replay": upper_replay_proof,
    }


def _validate_native_cnbr_assignment_golden(
    assignment_path: Path,
    *,
    golden: dict[str, Any],
    shard_count: int,
) -> dict[str, Any]:
    """Stream the production assignment file and replay every frozen parity field."""
    expected_sha256 = _canonical_manifest_sha256(
        golden,
        "assignment_bytes_sha256",
        "native CNBR golden assignment parity",
    )
    if layout_common.sha256_path(assignment_path) != expected_sha256:
        raise RuntimeError("native CNBR assignment bytes differ from frozen golden parity")

    primary_digest = hashlib.sha256()
    primary_loads = [0] * shard_count
    physical_loads = [0] * shard_count
    copy_histogram: dict[str, int] = {}
    physical_point_count = 0
    logical_point_count = 0
    with assignment_path.open("rb") as handle:
        for expected_id, raw_line in enumerate(handle):
            try:
                row = json.loads(raw_line)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"invalid native CNBR assignment JSON at row {expected_id}"
                ) from exc
            if not isinstance(row, dict) or set(row) != {"id", "shards"}:
                raise RuntimeError(
                    f"native CNBR assignment row {expected_id} has an invalid schema"
                )
            point_id = row.get("id")
            shards = row.get("shards")
            if (
                isinstance(point_id, bool)
                or not isinstance(point_id, int)
                or point_id != expected_id
            ):
                raise RuntimeError("native CNBR assignment IDs are not contiguous")
            if not isinstance(shards, list) or not shards:
                raise RuntimeError(
                    f"native CNBR assignment row {expected_id} has no shard"
                )
            if any(
                isinstance(shard, bool)
                or not isinstance(shard, int)
                or shard < 0
                or shard >= shard_count
                for shard in shards
            ):
                raise RuntimeError(
                    f"native CNBR assignment row {expected_id} has an invalid shard"
                )
            if shards != sorted(set(shards)):
                raise RuntimeError(
                    f"native CNBR assignment row {expected_id} is not canonical"
                )
            primary_digest.update(struct.pack("<i", shards[0]))
            primary_loads[shards[0]] += 1
            for shard in shards:
                physical_loads[shard] += 1
            copy_key = str(len(shards))
            copy_histogram[copy_key] = copy_histogram.get(copy_key, 0) + 1
            physical_point_count += len(shards)
            logical_point_count += 1

    observed = {
        "canonical_format": (
            "orion_numeric_import.assignments.jsonl-v1"
        ),
        "assignment_bytes_sha256": expected_sha256,
        "logical_point_count": logical_point_count,
        "physical_point_count": physical_point_count,
        "primary_shards_sha256": primary_digest.hexdigest(),
        "primary_shard_loads": primary_loads,
        "physical_copy_shard_loads": physical_loads,
        "copy_count_histogram": {
            key: copy_histogram[key]
            for key in sorted(copy_histogram, key=int)
        },
    }
    if observed != golden:
        mismatches = {
            key: {"expected": expected, "actual": observed.get(key)}
            for key, expected in golden.items()
            if observed.get(key) != expected
        }
        extras = sorted(set(observed) - set(golden))
        raise RuntimeError(
            "native CNBR assignment/golden parity mismatch: "
            f"mismatches={mismatches}, extras={extras}"
        )
    return observed


def _resolve_native_cnbr_construction_cost_audit(
    provenance: dict[str, Any],
    diagnostics: dict[str, Any],
    arm_name: str,
) -> Path | None:
    fields = (
        "construction_cost_v4_audit",
        "construction_cost_v4_audit_sha256",
        "construction_cost_v4_gate",
    )
    if arm_name == "N_native":
        if any(field in provenance or field in diagnostics for field in fields):
            raise RuntimeError("N_native bundle must not claim a CNBR cost-v4 gate")
        return None
    if arm_name != "C_CNBR":
        raise RuntimeError("native CNBR layout has an unsupported arm")

    path = _required_manifest_file(
        provenance,
        "construction_cost_v4_audit",
        "native CNBR provenance",
    )
    digest = _canonical_manifest_sha256(
        provenance,
        "construction_cost_v4_audit_sha256",
        "native CNBR provenance",
    )
    if layout_common.sha256_path(path) != digest:
        raise RuntimeError("native CNBR construction-cost checksum mismatch")
    if diagnostics.get("construction_cost_v4_gate") != "PASS":
        raise RuntimeError("native CNBR diagnostics do not bind cost-v4 PASS")
    if diagnostics.get("construction_cost_v4_audit_sha256") != digest:
        raise RuntimeError(
            "native CNBR diagnostics bind a different construction-cost audit"
        )
    return path


def validate_orion_native_cnbr_layout(
    build_manifest: dict[str, Any],
    build_parameters: dict[str, Any],
    artifact_payload: dict[str, Any],
    layout_dir: Path,
    checksums: dict[str, str],
) -> dict[str, Any]:
    """Validate the formal N_native/C_CNBR bundle under its own frozen contract."""
    from experiments.l1_balance import (  # Imported lazily for non-finalist layouts.
        materialize_native_cnbr_candidate as native_cnbr,
    )

    arm_name = build_parameters.get("l1_partitioner")
    if arm_name not in native_cnbr.ARM_CONTRACTS:
        raise RuntimeError("native CNBR layout has an unsupported arm")
    arm = native_cnbr.ARM_CONTRACTS[arm_name]
    expected_parameters = {
        "generation": artifact_payload.get("generation"),
        "initial_num_shards": artifact_payload.get("shard_count"),
        "sample_denominator": 32,
        "upper_sample_seed": 100,
        "upper_m": 32,
        "upper_ef_construction": 100,
        "upper_graph_seed": 100,
        "attachment_search_ef": 100,
        "upper_k": artifact_payload.get("upper_k"),
        "upper_search_ef": artifact_payload.get("upper_ef_search"),
        "dynamic_ef_base": artifact_payload.get("dynamic_ef_base"),
        "dynamic_ef_factor": artifact_payload.get("dynamic_ef_factor"),
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
        "balance_mode": arm.balance_variant,
    }
    if build_parameters != expected_parameters:
        mismatches = {
            key: {
                "expected": expected,
                "actual": build_parameters.get(key),
            }
            for key, expected in expected_parameters.items()
            if build_parameters.get(key) != expected
            or type(build_parameters.get(key)) is not type(expected)
        }
        raise RuntimeError(
            "native CNBR lightweight parameter contract drifted: "
            f"mismatches={mismatches}, extra={sorted(set(build_parameters) - set(expected_parameters))}"
        )

    shard_count = artifact_payload.get("shard_count")
    logical_point_count = artifact_payload.get("logical_point_count")
    physical_point_count = artifact_payload.get("physical_point_count")
    generation = artifact_payload.get("generation")
    for field, value in {
        "shard_count": shard_count,
        "logical_point_count": logical_point_count,
        "physical_point_count": physical_point_count,
        "generation": generation,
    }.items():
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise RuntimeError(f"native CNBR artifact has invalid {field}")
    if physical_point_count < logical_point_count:
        raise RuntimeError("native CNBR artifact loses logical points")

    artifact_binding = build_manifest.get("artifact_binding")
    expected_artifact_binding = {
        "generation": generation,
        "layout_sha256": artifact_payload.get("layout_sha256"),
        "logical_point_count": logical_point_count,
        "physical_point_count": physical_point_count,
        "shard_count": shard_count,
    }
    if artifact_binding != expected_artifact_binding:
        raise RuntimeError("native CNBR artifact_binding mismatch")

    routing = build_manifest.get("routing")
    if not isinstance(routing, dict):
        raise RuntimeError("native CNBR routing summary is missing")
    count_contract = {
        "initial_num_shards": shard_count,
        "effective_num_shards": shard_count,
        "logical_point_count": logical_point_count,
        "physical_point_count": physical_point_count,
    }
    if any(routing.get(key) != value for key, value in count_contract.items()):
        raise RuntimeError("native CNBR routing counts differ from the artifact")
    if routing.get("fission_events") != []:
        raise RuntimeError("native CNBR layout must not contain fission events")
    shard_counts = routing.get("shard_counts")
    if (
        not isinstance(shard_counts, list)
        or len(shard_counts) != shard_count
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in shard_counts
        )
        or sum(shard_counts) != physical_point_count
    ):
        raise RuntimeError("native CNBR shard_counts are invalid")

    provenance = build_manifest.get("provenance")
    diagnostics = routing.get("l1_partition_diagnostics")
    outputs = build_manifest.get("outputs")
    if not all(isinstance(value, dict) for value in (provenance, diagnostics, outputs)):
        raise RuntimeError("native CNBR bundle proof is incomplete")

    phase_b_path = _required_manifest_file(
        provenance, "phase_b_screen", "native CNBR provenance"
    )
    phase_b_sha256 = _canonical_manifest_sha256(
        provenance, "phase_b_screen_sha256", "native CNBR provenance"
    )
    if layout_common.sha256_path(phase_b_path) != phase_b_sha256:
        raise RuntimeError("native CNBR Phase-B screen checksum mismatch")
    selection_path: Path | None = None
    construction_cost_path: Path | None = None
    if arm_name == "C_CNBR":
        selection_path = _required_manifest_file(
            provenance, "selection_manifest", "native CNBR provenance"
        )
        selection_sha256 = _canonical_manifest_sha256(
            provenance, "selection_manifest_sha256", "native CNBR provenance"
        )
        if layout_common.sha256_path(selection_path) != selection_sha256:
            raise RuntimeError("native CNBR selection checksum mismatch")
    construction_cost_path = _resolve_native_cnbr_construction_cost_audit(
        provenance, diagnostics, arm_name
    )
    try:
        binding = native_cnbr.validate_phase_b_screen(
            phase_b_path,
            arm_name,
            selection_path,
            construction_cost_path,
        )
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"native CNBR frozen evidence rejected: {exc}") from exc

    if binding.screen_sha256 != phase_b_sha256:
        raise RuntimeError("native CNBR Phase-B binding drifted")
    expected_diagnostics = {
        "arm": arm_name,
        "input_scope": (
            "production_upper_graph_upper_vectors_and_upper_self_navigation_only"
        ),
        "balance_contract": {"variant": arm.balance_variant},
        "phase_a_role": arm.phase_a_role,
        "phase_b_role": arm.phase_b_role,
        "phase_a_manifest_sha256": binding.frozen.manifest_sha256,
        "phase_a_owner_sha256": binding.owner.sha256,
        "phase_a_owner_record_sha256": binding.owner.owner_record_sha256,
        "owner_sha256": binding.owner.sha256,
        "phase_b_manifest_sha256": binding.screen_sha256,
        "phase_b_evaluator_source_code_record_sha256": (
            binding.evaluator_source_code_record_sha256
        ),
        "attachments_sha256": binding.inputs.attachments_sha256,
        "golden_assignment_bytes_sha256": binding.record[
            "materialization_parity"
        ]["assignment_bytes_sha256"],
        "upper_replay_manifest_sha256": diagnostics.get(
            "upper_replay_manifest_sha256"
        ),
        "selected_adoption_candidate": arm.selected_adoption_candidate,
        "identity_all_pass": binding.record["identity_all_pass"],
        "topology_all_pass": binding.record["topology_all_pass"],
        "topology_metrics": binding.record["topology_metrics"],
        "graph_topology_gates": binding.record["graph_topology_gates"],
        "query_topology_gates": binding.record["query_topology_gates"],
        "physical_copy_load_improves_over_reference": binding.record[
            "physical_copy_load_improves_over_reference"
        ],
        "l1_sizes": binding.owner.record["partition_sizes"],
        "multi_assignment_after_owner_freeze": True,
        "forbidden_stage_invocations": binding.forbidden_stage_invocations,
        "invocation_counts": native_cnbr.MATERIALIZER_INVOCATION_COUNTS,
    }
    if arm_name == "C_CNBR":
        expected_diagnostics.update(
            {
                "parent_n_owner_sha256": binding.frozen.owners[
                    "N_native"
                ].sha256,
                "parent_n_manifest_sha256": binding.frozen.manifest_sha256,
                "selection_manifest_sha256": binding.selection_sha256,
                "selection_source_code_record_sha256": (
                    binding.selection_source_code_record_sha256
                ),
                "construction_cost_v4_gate": "PASS",
                "construction_cost_v4_audit_sha256": (
                    binding.construction_cost_sha256
                ),
            }
        )
    optional_legacy_n = {
        "selection_manifest_sha256",
        "selection_source_code_record_sha256",
    } if arm_name == "N_native" else set()
    diagnostic_extras = set(diagnostics) - set(expected_diagnostics)
    if diagnostic_extras - optional_legacy_n:
        raise RuntimeError(
            "native CNBR diagnostics contain unsupported fields: "
            f"{sorted(diagnostic_extras - optional_legacy_n)}"
        )
    diagnostic_mismatches = {
        key: {"expected": expected, "actual": diagnostics.get(key)}
        for key, expected in expected_diagnostics.items()
        if diagnostics.get(key) != expected
    }
    if diagnostic_mismatches:
        raise RuntimeError(
            "native CNBR frozen diagnostics mismatch: "
            f"{diagnostic_mismatches}"
        )
    if any(binding.forbidden_stage_invocations.values()):
        raise RuntimeError("native CNBR Phase-A invoked a forbidden stage")
    if any(native_cnbr.MATERIALIZER_INVOCATION_COUNTS.values()):
        raise RuntimeError("native CNBR materializer invoked a forbidden stage")
    if provenance.get("phase_b_evaluator_source_code_record_sha256") != (
        binding.evaluator_source_code_record_sha256
    ):
        raise RuntimeError("native CNBR evaluator source binding mismatch")
    if arm_name == "C_CNBR" and (
        provenance.get("selection_source_code_record_sha256")
        != binding.selection_source_code_record_sha256
    ):
        raise RuntimeError("native CNBR selection source binding mismatch")
    if arm_name == "C_CNBR":
        if (
            binding.construction_cost_path is None
            or binding.construction_cost_sha256 is None
            or binding.construction_cost_gate is None
        ):
            raise RuntimeError("native CNBR lost its construction-cost binding")
        if _required_manifest_file(
            provenance,
            "construction_cost_v4_audit",
            "native CNBR provenance",
        ) != binding.construction_cost_path.resolve():
            raise RuntimeError("native CNBR construction-cost path mismatch")
        if provenance.get("construction_cost_v4_audit_sha256") != (
            binding.construction_cost_sha256
        ):
            raise RuntimeError("native CNBR construction-cost SHA mismatch")
        if provenance.get("construction_cost_v4_gate") != (
            binding.construction_cost_gate
        ):
            raise RuntimeError("native CNBR construction-cost PASS proof drifted")

    provenance_files = {
        "phase_a_manifest": binding.frozen.manifest_path,
        "phase_a_owner": binding.owner.path,
        "phase_a_owner_record": binding.owner.owner_record_path,
        "source_artifact": binding.source_artifact_path,
        "attachments": binding.inputs.attachments_path,
        "attachments_manifest": binding.inputs.attachments_manifest_path,
        "vectors_source": binding.vectors_path,
        "vectors_source_build_manifest": binding.source_build_manifest_path,
        "dataset_manifest": binding.dataset_manifest_path,
        "upper_replay_queries": binding.replay_queries_path,
    }
    for field, expected_path in provenance_files.items():
        if _required_manifest_file(
            provenance, field, "native CNBR provenance"
        ) != expected_path.resolve():
            raise RuntimeError(f"native CNBR provenance {field} path mismatch")
    provenance_hashes = {
        "phase_a_manifest_sha256": binding.frozen.manifest_sha256,
        "phase_a_owner_sha256": binding.owner.sha256,
        "phase_a_owner_record_sha256": binding.owner.owner_record_sha256,
        "source_artifact_sha256": binding.source_artifact_sha256,
        "vectors_source_sha256": binding.vectors_sha256,
        "vectors_source_build_manifest_sha256": (
            binding.source_build_manifest_sha256
        ),
        "dataset_manifest_sha256": binding.dataset_manifest_sha256,
        "upper_replay_queries_sha256": layout_common.sha256_path(
            binding.replay_queries_path
        ),
    }
    for field, expected_sha256 in provenance_hashes.items():
        if _canonical_manifest_sha256(
            provenance, field, "native CNBR provenance"
        ) != expected_sha256:
            raise RuntimeError(f"native CNBR provenance {field} mismatch")

    owner_path = safe_child(layout_dir, outputs.get("owner"), "owner")
    owner_relative = owner_path.relative_to(layout_dir).as_posix()
    if checksums.get(owner_relative) != binding.owner.sha256:
        raise RuntimeError("native CNBR output owner checksum mismatch")
    owner_values = experiment.np.fromfile(owner_path, dtype="<i4")
    if not experiment.np.array_equal(owner_values, binding.owner.values):
        raise RuntimeError("native CNBR output owner differs from frozen Phase-A")
    upper_point_count = routing.get("upper_point_count")
    if upper_point_count != len(binding.owner.values):
        raise RuntimeError("native CNBR upper point count mismatch")
    actual_l1_sizes = [
        int(value)
        for value in experiment.np.bincount(
            owner_values, minlength=shard_count
        ).tolist()
    ]
    if actual_l1_sizes != diagnostics["l1_sizes"]:
        raise RuntimeError("native CNBR owner loads differ from frozen diagnostics")

    import_manifest_path = safe_child(
        layout_dir, outputs.get("import_manifest"), "import_manifest"
    )
    import_manifest = _read_manifest_object(
        import_manifest_path, "native CNBR import manifest"
    )
    assignment_path = safe_child(
        layout_dir, import_manifest.get("assignments_file"), "assignments_file"
    )
    vector_path = safe_child(
        layout_dir, import_manifest.get("vectors_file"), "vectors_file"
    )
    for path, label in (
        (assignment_path, "assignment"),
        (vector_path, "vector"),
    ):
        relative = path.relative_to(layout_dir).as_posix()
        declared = (outputs.get("files") or {}).get(relative)
        if (
            checksums.get(relative) is None
            or not isinstance(declared, dict)
            or declared.get("sha256") != checksums[relative]
            or declared.get("size_bytes") != path.stat().st_size
        ):
            raise RuntimeError(f"native CNBR {label} output binding mismatch")
    golden = binding.record.get("materialization_parity")
    if not isinstance(golden, dict):
        raise RuntimeError("native CNBR frozen assignment parity is missing")
    parity = _validate_native_cnbr_assignment_golden(
        assignment_path,
        golden=golden,
        shard_count=shard_count,
    )
    if (
        parity["logical_point_count"] != logical_point_count
        or parity["physical_point_count"] != physical_point_count
        or parity["physical_copy_shard_loads"] != shard_counts
        or parity["copy_count_histogram"] != routing.get("copy_count_histogram")
        or parity["assignment_bytes_sha256"]
        != artifact_payload.get("layout_sha256")
        or checksums.get(assignment_path.relative_to(layout_dir).as_posix())
        != parity["assignment_bytes_sha256"]
    ):
        raise RuntimeError("native CNBR production bundle differs from golden parity")

    if provenance.get("vectors_materialization") != "hardlink":
        raise RuntimeError("native CNBR vectors were not materialized as a hardlink")
    if provenance.get("vectors_same_file") is not True or not os.path.samefile(
        vector_path, binding.vectors_path
    ):
        raise RuntimeError("native CNBR vector hardlink identity mismatch")
    if checksums.get(vector_path.relative_to(layout_dir).as_posix()) != (
        binding.vectors_sha256
    ):
        raise RuntimeError("native CNBR vector checksum mismatch")
    source_mode = f"{binding.vectors_path.stat().st_mode & 0o777:o}"
    if provenance.get("source_vector_mode_preserved") != source_mode:
        raise RuntimeError("native CNBR source vector mode was not preserved")

    rebind_path = layout_dir / "rebind-memberships.json"
    rebind_relative = rebind_path.relative_to(layout_dir).as_posix()
    if not rebind_path.is_file() or rebind_relative not in checksums:
        raise RuntimeError("native CNBR rebind proof is missing")
    rebind = _read_manifest_object(rebind_path, "native CNBR rebind proof")
    rebind_expected = {
        "format_version": 1,
        "source_artifact_sha256": binding.source_artifact_sha256,
        "source_generation": binding.frozen.artifact["generation"],
        "generation": generation,
        "layout_sha256": artifact_payload.get("layout_sha256"),
        "physical_point_count": physical_point_count,
        "shard_count": shard_count,
        "shard_memberships": [
            node.get("shard_membership")
            for node in artifact_payload.get("upper_nodes", [])
        ],
    }
    if rebind != rebind_expected:
        raise RuntimeError("native CNBR rebind proof differs from the artifact")

    replay_proof = _validate_orion_l1_upper_replay(
        build_manifest,
        artifact_payload,
        diagnostics,
        layout_dir,
        checksums,
        binding.source_artifact_path,
        binding.source_artifact_sha256,
    )
    proof = {
        "mode": arm.balance_variant,
        "arm": arm_name,
        "input_scope": diagnostics["input_scope"],
        "selected_adoption_candidate": arm.selected_adoption_candidate,
        "selection_bound": arm_name == "C_CNBR",
        "multi_assignment": dict(native_cnbr.MULTI_ASSIGNMENT_CONTRACT),
        "forbidden_stage_invocations": dict(binding.forbidden_stage_invocations),
        "invocation_counts": dict(native_cnbr.MATERIALIZER_INVOCATION_COUNTS),
        "assignment_parity": parity,
        "shard_count": shard_count,
        "l1_sizes": actual_l1_sizes,
        "shard_counts": list(shard_counts),
        "upper_replay": replay_proof,
        "construction_cost_v4_bound": arm_name == "C_CNBR",
    }
    if arm_name == "C_CNBR":
        assert binding.construction_cost_path is not None
        assert binding.construction_cost_sha256 is not None
        proof["construction_cost_v4"] = {
            "status": "PASS",
            "audit": str(binding.construction_cost_path),
            "audit_sha256": binding.construction_cost_sha256,
        }
    return proof


def _validate_budgeted_multi_assignment_bytes(
    assignment_path: Path,
    *,
    candidate: dict[str, Any],
    shard_count: int,
    require_exact_budget: bool = True,
) -> dict[str, Any]:
    """Replay the frozen BMR assignment, copy cap, budget, and semantic hash."""
    expected_sha256 = _canonical_manifest_sha256(
        candidate,
        "assignment_jsonl_sha256",
        "BMR_10 candidate",
    )
    if layout_common.sha256_path(assignment_path) != expected_sha256:
        raise RuntimeError("BMR_10 assignment bytes differ from frozen selection")

    primary_digest = hashlib.sha256()
    semantic_digest = hashlib.sha256()
    primary_loads = [0] * shard_count
    physical_loads = [0] * shard_count
    copy_histogram: dict[str, int] = {}
    physical_point_count = 0
    logical_point_count = 0
    packed_width = (shard_count + 7) // 8
    with assignment_path.open("rb") as handle:
        for expected_id, raw_line in enumerate(handle):
            try:
                row = json.loads(raw_line)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"invalid BMR_10 assignment JSON at row {expected_id}"
                ) from exc
            if not isinstance(row, dict) or set(row) != {"id", "shards"}:
                raise RuntimeError(
                    f"BMR_10 assignment row {expected_id} has an invalid schema"
                )
            point_id = row.get("id")
            shards = row.get("shards")
            if (
                isinstance(point_id, bool)
                or not isinstance(point_id, int)
                or point_id != expected_id
            ):
                raise RuntimeError("BMR_10 assignment IDs are not contiguous")
            if not isinstance(shards, list) or not shards or len(shards) > 2:
                raise RuntimeError(
                    "BMR_10 assignment violates the one-or-two-copy contract"
                )
            if any(
                isinstance(shard, bool)
                or not isinstance(shard, int)
                or shard < 0
                or shard >= shard_count
                for shard in shards
            ):
                raise RuntimeError(
                    f"BMR_10 assignment row {expected_id} has an invalid shard"
                )
            if shards != sorted(set(shards)):
                raise RuntimeError(
                    f"BMR_10 assignment row {expected_id} is not canonical"
                )
            packed = bytearray(packed_width)
            for shard in shards:
                packed[shard // 8] |= 1 << (shard % 8)
                physical_loads[shard] += 1
            semantic_digest.update(packed)
            primary_digest.update(struct.pack("<i", shards[0]))
            primary_loads[shards[0]] += 1
            copy_key = str(len(shards))
            copy_histogram[copy_key] = copy_histogram.get(copy_key, 0) + 1
            physical_point_count += len(shards)
            logical_point_count += 1

    extra_copies = physical_point_count - logical_point_count
    exact_budget = (
        logical_point_count
        * int(candidate.get("extra_copy_budget_numerator", 1))
        // int(candidate.get("extra_copy_budget_denominator", 10))
    )
    # The original P=32 candidate consumes the full budget.  Fixed-P scaling
    # may have fewer than floor(N/10) navigation-supported secondary copies,
    # so the same frozen policy is bounded by, rather than forced to fill, it.
    budget_ok = (
        extra_copies == exact_budget
        if require_exact_budget
        else extra_copies <= exact_budget
    )
    if not budget_ok:
        if require_exact_budget:
            raise RuntimeError(
                "BMR_10 assignment does not consume the exact frozen 10% budget"
            )
        raise RuntimeError("BMR_10 assignment exceeds the frozen 10% copy budget")
    load_array = experiment.np.asarray(physical_loads, dtype=experiment.np.int64)
    load_mean = float(load_array.mean())
    observed_candidate = {
        "policy": "BMR_10",
        "assignment_jsonl_sha256": expected_sha256,
        "membership_semantic_sha256": semantic_digest.hexdigest(),
        "logical_point_count": logical_point_count,
        "physical_point_count": physical_point_count,
        "expansion_ratio": physical_point_count / logical_point_count,
        "extra_copy_fraction": physical_point_count / logical_point_count - 1.0,
        "copy_count_histogram": {
            key: copy_histogram[key]
            for key in sorted(copy_histogram, key=int)
        },
        "physical_copy_load_min": int(load_array.min()),
        "physical_copy_load_max": int(load_array.max()),
        "physical_copy_load_mean": load_mean,
        "physical_copy_load_cv": float(load_array.std() / load_mean),
        "physical_copy_load_max_over_mean": float(load_array.max() / load_mean),
        "physical_copy_load_min_over_mean": float(load_array.min() / load_mean),
        "physical_copy_load_empty_shards": int((load_array == 0).sum()),
    }
    mismatches = {
        key: {"expected": candidate.get(key), "actual": value}
        for key, value in observed_candidate.items()
        if candidate.get(key) != value
    }
    if mismatches:
        raise RuntimeError(
            f"BMR_10 assignment differs from frozen candidate: {mismatches}"
        )
    return {
        **observed_candidate,
        "canonical_format": "orion_numeric_import.assignments.jsonl-v1",
        "primary_shards_sha256": primary_digest.hexdigest(),
        "primary_shard_loads": primary_loads,
        "physical_copy_shard_loads": physical_loads,
        "extra_copy_count": extra_copies,
        "exact_extra_copy_budget": exact_budget,
        "copy_budget_mode": "exact" if require_exact_budget else "upper_bound",
        "maximum_copies_per_point": 2,
    }


def validate_orion_budgeted_multi_assignment_layout(
    build_manifest: dict[str, Any],
    build_parameters: dict[str, Any],
    artifact_payload: dict[str, Any],
    layout_dir: Path,
    checksums: dict[str, str],
) -> dict[str, Any]:
    """Validate BMR_10 while keeping the frozen C_CNBR owner unchanged."""
    from experiments.l1_balance import (  # Imported lazily for other layouts.
        cnbr_construction_cost_gate as cost_gate,
    )
    from experiments.l1_balance import materialize_native_cnbr_candidate as native_cnbr
    from experiments.l1_balance import native_cnbr_core
    from experiments.multi_assignment import budgeted_policy
    from experiments.multi_assignment import (
        materialize_budgeted_candidate as budgeted_materializer,
    )

    expected_parameters = {
        "generation": artifact_payload.get("generation"),
        "initial_num_shards": artifact_payload.get("shard_count"),
        "sample_denominator": 32,
        "upper_sample_seed": 100,
        "upper_m": 32,
        "upper_ef_construction": 100,
        "upper_graph_seed": 100,
        "attachment_search_ef": 100,
        "upper_k": artifact_payload.get("upper_k"),
        "upper_search_ef": artifact_payload.get("upper_ef_search"),
        "dynamic_ef_base": artifact_payload.get("dynamic_ef_base"),
        "dynamic_ef_factor": artifact_payload.get("dynamic_ef_factor"),
        "k_overlap": 10,
        "use_multi_assign": True,
        "multi_assign_min_max_vote": 2,
        "multi_assign_vote_delta": 0,
        "multi_assign_max_shards": 2,
        "multi_assign_extra_copy_budget_numerator": (
            budgeted_policy.EXTRA_COPY_BUDGET_NUMERATOR
        ),
        "multi_assign_extra_copy_budget_denominator": (
            budgeted_policy.EXTRA_COPY_BUDGET_DENOMINATOR
        ),
        "multi_assign_score": budgeted_policy.SCORE,
        "multi_assignment_policy": budgeted_policy.CANDIDATE_ID,
        "multi_assignment_policy_version": budgeted_policy.POLICY_VERSION,
        "enable_fission": False,
        "enable_topology_refinement": False,
        "l0_repair": False,
        "allow_decoupled_runtime_upper_search": False,
        "l1_partitioner": "C_CNBR",
        "balance_mode": "cnbr",
    }
    if build_parameters != expected_parameters:
        mismatches = {
            key: {"expected": expected, "actual": build_parameters.get(key)}
            for key, expected in expected_parameters.items()
            if build_parameters.get(key) != expected
            or type(build_parameters.get(key)) is not type(expected)
        }
        raise RuntimeError(
            "BMR_10 parameter contract drifted: "
            f"mismatches={mismatches}, "
            f"extra={sorted(set(build_parameters) - set(expected_parameters))}"
        )

    shard_count = artifact_payload.get("shard_count")
    logical_point_count = artifact_payload.get("logical_point_count")
    physical_point_count = artifact_payload.get("physical_point_count")
    generation = artifact_payload.get("generation")
    for field, value in {
        "shard_count": shard_count,
        "logical_point_count": logical_point_count,
        "physical_point_count": physical_point_count,
        "generation": generation,
    }.items():
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise RuntimeError(f"BMR_10 artifact has invalid {field}")
    expected_physical = logical_point_count + (
        logical_point_count
        * budgeted_policy.EXTRA_COPY_BUDGET_NUMERATOR
        // budgeted_policy.EXTRA_COPY_BUDGET_DENOMINATOR
    )
    if physical_point_count != expected_physical:
        raise RuntimeError("BMR_10 artifact violates the exact 10% copy budget")
    expected_artifact_binding = {
        "generation": generation,
        "layout_sha256": artifact_payload.get("layout_sha256"),
        "logical_point_count": logical_point_count,
        "physical_point_count": physical_point_count,
        "shard_count": shard_count,
    }
    if build_manifest.get("artifact_binding") != expected_artifact_binding:
        raise RuntimeError("BMR_10 artifact_binding mismatch")

    routing = build_manifest.get("routing")
    provenance = build_manifest.get("provenance")
    outputs = build_manifest.get("outputs")
    if not all(isinstance(value, dict) for value in (routing, provenance, outputs)):
        raise RuntimeError("BMR_10 bundle proof is incomplete")
    diagnostics = routing.get("l1_partition_diagnostics")
    if not isinstance(diagnostics, dict):
        raise RuntimeError("BMR_10 diagnostics are missing")
    if routing.get("fission_events") != []:
        raise RuntimeError("BMR_10 must not contain fission events")
    count_contract = {
        "initial_num_shards": shard_count,
        "effective_num_shards": shard_count,
        "logical_point_count": logical_point_count,
        "physical_point_count": physical_point_count,
    }
    if any(routing.get(key) != value for key, value in count_contract.items()):
        raise RuntimeError("BMR_10 routing counts differ from the artifact")
    shard_counts = routing.get("shard_counts")
    if (
        not isinstance(shard_counts, list)
        or len(shard_counts) != shard_count
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in shard_counts
        )
        or sum(shard_counts) != physical_point_count
    ):
        raise RuntimeError("BMR_10 shard_counts are invalid")

    phase_b_path = _required_manifest_file(
        provenance, "phase_b_screen", "BMR_10 provenance"
    )
    phase_b_sha256 = _canonical_manifest_sha256(
        provenance, "phase_b_screen_sha256", "BMR_10 provenance"
    )
    if layout_common.sha256_path(phase_b_path) != phase_b_sha256:
        raise RuntimeError("BMR_10 Phase-B screen checksum mismatch")
    cnbr_selection_path = _required_manifest_file(
        provenance, "selection_manifest", "BMR_10 provenance"
    )
    cnbr_selection_sha256 = _canonical_manifest_sha256(
        provenance, "selection_manifest_sha256", "BMR_10 provenance"
    )
    if layout_common.sha256_path(cnbr_selection_path) != cnbr_selection_sha256:
        raise RuntimeError("BMR_10 C_CNBR selection checksum mismatch")
    construction_cost_path = _resolve_native_cnbr_construction_cost_audit(
        provenance, diagnostics, "C_CNBR"
    )
    assert construction_cost_path is not None

    frozen_core = phase_b_path.parent / "source" / "native_cnbr_core.py"
    if not frozen_core.is_file():
        raise RuntimeError(f"BMR_10 frozen Phase-B core is missing: {frozen_core}")
    original_core_file = native_cnbr_core.__file__
    original_cost_source_validator = cost_gate._validate_current_source_bindings
    try:
        # The owner/cost evidence predates this additive BMR research.  Replay it
        # against the immutable Phase-B source snapshot instead of requiring the
        # later working tree to equal that historical snapshot byte-for-byte.
        native_cnbr_core.__file__ = str(frozen_core)
        cost_gate._validate_current_source_bindings = lambda _audit: None
        binding = native_cnbr.validate_phase_b_screen(
            phase_b_path,
            "C_CNBR",
            cnbr_selection_path,
            construction_cost_path,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise RuntimeError(f"BMR_10 frozen C_CNBR evidence rejected: {exc}") from exc
    finally:
        native_cnbr_core.__file__ = original_core_file
        cost_gate._validate_current_source_bindings = original_cost_source_validator
    if binding.screen_sha256 != phase_b_sha256:
        raise RuntimeError("BMR_10 Phase-B binding drifted")

    bmr_selection_path = _required_manifest_file(
        provenance, "bmr10_selection_manifest", "BMR_10 provenance"
    )
    bmr_selection_sha256 = _canonical_manifest_sha256(
        provenance, "bmr10_selection_manifest_sha256", "BMR_10 provenance"
    )
    try:
        bmr_selection, validated_bmr_sha256 = (
            budgeted_materializer.load_bmr_selection(
                bmr_selection_path,
                binding,
            )
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise RuntimeError(f"BMR_10 selection evidence rejected: {exc}") from exc
    if validated_bmr_sha256 != bmr_selection_sha256:
        raise RuntimeError("BMR_10 selection binding drifted")
    candidate = bmr_selection["rows"]["glove-200-angular"]["candidate"]

    policy_source = Path(budgeted_policy.__file__).resolve()
    if (
        _required_manifest_file(
            provenance, "bmr10_policy_source", "BMR_10 provenance"
        )
        != policy_source
        or _canonical_manifest_sha256(
            provenance, "bmr10_policy_source_sha256", "BMR_10 provenance"
        )
        != layout_common.sha256_path(policy_source)
    ):
        raise RuntimeError("BMR_10 policy source binding mismatch")

    multi_assignment_contract = {
        "candidate_id": budgeted_policy.CANDIDATE_ID,
        "version": budgeted_policy.POLICY_VERSION,
        "extra_copy_budget_numerator": (
            budgeted_policy.EXTRA_COPY_BUDGET_NUMERATOR
        ),
        "extra_copy_budget_denominator": (
            budgeted_policy.EXTRA_COPY_BUDGET_DENOMINATOR
        ),
        "score": budgeted_policy.SCORE,
        "maximum_copies_per_point": 2,
        "load_balance_owner_unchanged": True,
        "canonical_default_eligible": False,
    }
    expected_diagnostics = {
        "arm": "C_CNBR",
        "input_scope": (
            "frozen_cnbr_owner_upper_proxy_mass_and_normal_l0_attachments"
        ),
        "balance_contract": {"variant": "cnbr"},
        "phase_a_role": native_cnbr.ARM_CONTRACTS["C_CNBR"].phase_a_role,
        "phase_b_role": native_cnbr.ARM_CONTRACTS["C_CNBR"].phase_b_role,
        "phase_a_manifest_sha256": binding.frozen.manifest_sha256,
        "phase_a_owner_sha256": binding.owner.sha256,
        "phase_a_owner_record_sha256": binding.owner.owner_record_sha256,
        "owner_sha256": binding.owner.sha256,
        "phase_b_manifest_sha256": binding.screen_sha256,
        "phase_b_evaluator_source_code_record_sha256": (
            binding.evaluator_source_code_record_sha256
        ),
        "attachments_sha256": binding.inputs.attachments_sha256,
        "golden_assignment_bytes_sha256": candidate["assignment_jsonl_sha256"],
        "upper_replay_manifest_sha256": diagnostics.get(
            "upper_replay_manifest_sha256"
        ),
        "selected_adoption_candidate": True,
        "identity_all_pass": binding.record["identity_all_pass"],
        "topology_all_pass": binding.record["topology_all_pass"],
        "topology_metrics": binding.record["topology_metrics"],
        "graph_topology_gates": binding.record["graph_topology_gates"],
        "query_topology_gates": binding.record["query_topology_gates"],
        "physical_copy_load_improves_over_reference": binding.record[
            "physical_copy_load_improves_over_reference"
        ],
        "l1_sizes": binding.owner.record["partition_sizes"],
        "multi_assignment_after_owner_freeze": True,
        "multi_assignment_contract": multi_assignment_contract,
        "forbidden_stage_invocations": binding.forbidden_stage_invocations,
        "invocation_counts": native_cnbr.MATERIALIZER_INVOCATION_COUNTS,
        "parent_n_owner_sha256": binding.frozen.owners["N_native"].sha256,
        "parent_n_manifest_sha256": binding.frozen.manifest_sha256,
        "selection_manifest_sha256": binding.selection_sha256,
        "selection_source_code_record_sha256": (
            binding.selection_source_code_record_sha256
        ),
        "construction_cost_v4_gate": "PASS",
        "construction_cost_v4_audit_sha256": binding.construction_cost_sha256,
    }
    diagnostic_mismatches = {
        key: {"expected": expected, "actual": diagnostics.get(key)}
        for key, expected in expected_diagnostics.items()
        if diagnostics.get(key) != expected
    }
    if diagnostic_mismatches or set(diagnostics) != set(expected_diagnostics):
        raise RuntimeError(
            "BMR_10 frozen diagnostics mismatch: "
            f"mismatches={diagnostic_mismatches}, "
            f"extra={sorted(set(diagnostics) - set(expected_diagnostics))}"
        )
    if any(binding.forbidden_stage_invocations.values()) or any(
        native_cnbr.MATERIALIZER_INVOCATION_COUNTS.values()
    ):
        raise RuntimeError("BMR_10 owner path invoked a forbidden stage")

    provenance_files = {
        "phase_a_manifest": binding.frozen.manifest_path,
        "phase_a_owner": binding.owner.path,
        "phase_a_owner_record": binding.owner.owner_record_path,
        "source_artifact": binding.source_artifact_path,
        "attachments": binding.inputs.attachments_path,
        "attachments_manifest": binding.inputs.attachments_manifest_path,
        "vectors_source": binding.vectors_path,
        "vectors_source_build_manifest": binding.source_build_manifest_path,
        "dataset_manifest": binding.dataset_manifest_path,
        "upper_replay_queries": binding.replay_queries_path,
    }
    for field, expected_path in provenance_files.items():
        if _required_manifest_file(
            provenance, field, "BMR_10 provenance"
        ) != expected_path.resolve():
            raise RuntimeError(f"BMR_10 provenance {field} path mismatch")
    provenance_hashes = {
        "phase_a_manifest_sha256": binding.frozen.manifest_sha256,
        "phase_a_owner_sha256": binding.owner.sha256,
        "phase_a_owner_record_sha256": binding.owner.owner_record_sha256,
        "source_artifact_sha256": binding.source_artifact_sha256,
        "vectors_source_sha256": binding.vectors_sha256,
        "vectors_source_build_manifest_sha256": (
            binding.source_build_manifest_sha256
        ),
        "dataset_manifest_sha256": binding.dataset_manifest_sha256,
        "upper_replay_queries_sha256": layout_common.sha256_path(
            binding.replay_queries_path
        ),
    }
    for field, expected_sha256 in provenance_hashes.items():
        if _canonical_manifest_sha256(
            provenance, field, "BMR_10 provenance"
        ) != expected_sha256:
            raise RuntimeError(f"BMR_10 provenance {field} mismatch")

    owner_path = safe_child(layout_dir, outputs.get("owner"), "owner")
    owner_relative = owner_path.relative_to(layout_dir).as_posix()
    if checksums.get(owner_relative) != binding.owner.sha256:
        raise RuntimeError("BMR_10 output owner checksum mismatch")
    owner_values = experiment.np.fromfile(owner_path, dtype="<i4")
    if not experiment.np.array_equal(owner_values, binding.owner.values):
        raise RuntimeError("BMR_10 changed the frozen C_CNBR owner")
    if routing.get("upper_point_count") != len(binding.owner.values):
        raise RuntimeError("BMR_10 upper point count mismatch")
    actual_l1_sizes = [
        int(value)
        for value in experiment.np.bincount(
            owner_values, minlength=shard_count
        ).tolist()
    ]
    if actual_l1_sizes != diagnostics["l1_sizes"]:
        raise RuntimeError("BMR_10 owner loads differ from frozen C_CNBR")

    import_manifest_path = safe_child(
        layout_dir, outputs.get("import_manifest"), "import_manifest"
    )
    import_manifest = _read_manifest_object(
        import_manifest_path, "BMR_10 import manifest"
    )
    assignment_path = safe_child(
        layout_dir, import_manifest.get("assignments_file"), "assignments_file"
    )
    vector_path = safe_child(
        layout_dir, import_manifest.get("vectors_file"), "vectors_file"
    )
    for path, label in ((assignment_path, "assignment"), (vector_path, "vector")):
        relative = path.relative_to(layout_dir).as_posix()
        declared = (outputs.get("files") or {}).get(relative)
        if (
            checksums.get(relative) is None
            or not isinstance(declared, dict)
            or declared.get("sha256") != checksums[relative]
            or declared.get("size_bytes") != path.stat().st_size
        ):
            raise RuntimeError(f"BMR_10 {label} output binding mismatch")
    assignment_proof = _validate_budgeted_multi_assignment_bytes(
        assignment_path,
        candidate={
            **candidate,
            "extra_copy_budget_numerator": (
                budgeted_policy.EXTRA_COPY_BUDGET_NUMERATOR
            ),
            "extra_copy_budget_denominator": (
                budgeted_policy.EXTRA_COPY_BUDGET_DENOMINATOR
            ),
        },
        shard_count=shard_count,
    )
    if (
        assignment_proof["logical_point_count"] != logical_point_count
        or assignment_proof["physical_point_count"] != physical_point_count
        or assignment_proof["physical_copy_shard_loads"] != shard_counts
        or assignment_proof["copy_count_histogram"]
        != routing.get("copy_count_histogram")
        or assignment_proof["assignment_jsonl_sha256"]
        != artifact_payload.get("layout_sha256")
        or checksums.get(assignment_path.relative_to(layout_dir).as_posix())
        != assignment_proof["assignment_jsonl_sha256"]
    ):
        raise RuntimeError("BMR_10 production assignment binding mismatch")
    expected_load_summary = {
        "cv": assignment_proof["physical_copy_load_cv"],
        "max": assignment_proof["physical_copy_load_max"],
        "max_over_mean": assignment_proof["physical_copy_load_max_over_mean"],
        "mean": assignment_proof["physical_copy_load_mean"],
        "min": assignment_proof["physical_copy_load_min"],
        "min_over_mean": assignment_proof["physical_copy_load_min_over_mean"],
    }
    if routing.get("load_summary") != expected_load_summary:
        raise RuntimeError("BMR_10 routing load summary mismatch")

    if provenance.get("vectors_materialization") != "hardlink":
        raise RuntimeError("BMR_10 vectors were not materialized as a hardlink")
    if provenance.get("vectors_same_file") is not True or not os.path.samefile(
        vector_path, binding.vectors_path
    ):
        raise RuntimeError("BMR_10 vector hardlink identity mismatch")
    if checksums.get(vector_path.relative_to(layout_dir).as_posix()) != (
        binding.vectors_sha256
    ):
        raise RuntimeError("BMR_10 vector checksum mismatch")
    source_mode = f"{binding.vectors_path.stat().st_mode & 0o777:o}"
    if provenance.get("source_vector_mode_preserved") != source_mode:
        raise RuntimeError("BMR_10 source vector mode was not preserved")

    rebind_path = layout_dir / "rebind-memberships.json"
    rebind_relative = rebind_path.relative_to(layout_dir).as_posix()
    if not rebind_path.is_file() or rebind_relative not in checksums:
        raise RuntimeError("BMR_10 rebind proof is missing")
    rebind = _read_manifest_object(rebind_path, "BMR_10 rebind proof")
    rebind_expected = {
        "format_version": 1,
        "source_artifact_sha256": binding.source_artifact_sha256,
        "source_generation": binding.frozen.artifact["generation"],
        "generation": generation,
        "layout_sha256": artifact_payload.get("layout_sha256"),
        "physical_point_count": physical_point_count,
        "shard_count": shard_count,
        "shard_memberships": [
            node.get("shard_membership")
            for node in artifact_payload.get("upper_nodes", [])
        ],
    }
    if rebind != rebind_expected:
        raise RuntimeError("BMR_10 rebind proof differs from the artifact")

    replay_proof = _validate_orion_l1_upper_replay(
        build_manifest,
        artifact_payload,
        diagnostics,
        layout_dir,
        checksums,
        binding.source_artifact_path,
        binding.source_artifact_sha256,
    )
    return {
        "mode": "cnbr_with_budgeted_multi_assignment",
        "arm": "C_CNBR",
        "candidate": budgeted_policy.CANDIDATE_ID,
        "canonical_default_eligible": False,
        "load_balance_owner_unchanged": True,
        "owner_sha256": binding.owner.sha256,
        "selection_sha256": bmr_selection_sha256,
        "multi_assignment": multi_assignment_contract,
        "assignment": assignment_proof,
        "shard_count": shard_count,
        "l1_sizes": actual_l1_sizes,
        "shard_counts": list(shard_counts),
        "upper_replay": replay_proof,
        "construction_cost_v4_bound": True,
    }


def validate_orion_bmr10_scaling_layout(
    build_manifest: dict[str, Any],
    build_parameters: dict[str, Any],
    artifact_payload: dict[str, Any],
    layout_dir: Path,
    checksums: dict[str, str],
) -> dict[str, Any]:
    """Validate the fixed-P C_CNBR+BMR_10 C1 scaling extension."""
    from experiments.multi_assignment import budgeted_policy

    routing = build_manifest.get("routing")
    provenance = build_manifest.get("provenance")
    outputs = build_manifest.get("outputs")
    if not all(isinstance(value, dict) for value in (routing, provenance, outputs)):
        raise RuntimeError("BMR_10 scaling bundle proof is incomplete")
    diagnostics = routing.get("l1_partition_diagnostics")
    assignment_metrics = (
        diagnostics.get("assignment_metrics")
        if isinstance(diagnostics, dict)
        else None
    )
    if not isinstance(diagnostics, dict) or not isinstance(assignment_metrics, dict):
        raise RuntimeError("BMR_10 scaling diagnostics are missing")
    shard_count = artifact_payload.get("shard_count")
    logical_count = artifact_payload.get("logical_point_count")
    physical_count = artifact_payload.get("physical_point_count")
    if shard_count not in {1, 2, 4, 8, 16, 32}:
        raise RuntimeError("BMR_10 scaling shard count is outside the frozen grid")
    expected_parameters = {
        "l1_partitioner": "C_CNBR",
        "balance_mode": "cnbr",
        "initial_num_shards": shard_count,
        "enable_fission": False,
        "enable_topology_refinement": False,
        "use_multi_assign": True,
        "multi_assign_min_max_vote": 2,
        "multi_assign_vote_delta": 0,
        "multi_assign_max_shards": 2,
        "multi_assign_extra_copy_budget_numerator": 1,
        "multi_assign_extra_copy_budget_denominator": 10,
        "multi_assign_score": budgeted_policy.SCORE,
        "multi_assignment_policy": budgeted_policy.CANDIDATE_ID,
        "multi_assignment_policy_version": budgeted_policy.POLICY_VERSION,
    }
    parameter_mismatches = {
        key: {"expected": expected, "actual": build_parameters.get(key)}
        for key, expected in expected_parameters.items()
        if build_parameters.get(key) != expected
        or type(build_parameters.get(key)) is not type(expected)
    }
    if parameter_mismatches:
        raise RuntimeError(
            f"BMR_10 scaling parameters drifted: {parameter_mismatches}"
        )
    routing_expected = {
        "initial_num_shards": shard_count,
        "effective_num_shards": shard_count,
        "logical_point_count": logical_count,
        "physical_point_count": physical_count,
        "fission_events": [],
    }
    routing_mismatches = {
        key: {"expected": expected, "actual": routing.get(key)}
        for key, expected in routing_expected.items()
        if routing.get(key) != expected
        or type(routing.get(key)) is not type(expected)
    }
    if routing_mismatches:
        raise RuntimeError(f"BMR_10 scaling routing drifted: {routing_mismatches}")

    owner_manifest_path = _required_manifest_file(
        provenance, "scaling_owner_manifest", "BMR_10 scaling provenance"
    )
    owner_manifest_sha256 = _canonical_manifest_sha256(
        provenance,
        "scaling_owner_manifest_sha256",
        "BMR_10 scaling provenance",
    )
    if layout_common.sha256_path(owner_manifest_path) != owner_manifest_sha256:
        raise RuntimeError("BMR_10 scaling owner-manifest checksum mismatch")
    owner_manifest = _read_manifest_object(
        owner_manifest_path, "BMR_10 scaling owner manifest"
    )
    if (
        owner_manifest.get("status") != "PASS"
        or owner_manifest.get("record_type")
        != "c1_orion_bmr10_scaling_owner_manifest"
        or owner_manifest.get("partitions") != shard_count
        or (owner_manifest.get("causal_boundary") or {}).get(
            "owner_frozen_before_full_l0_attachment_access"
        )
        is not True
        or (owner_manifest.get("causal_boundary") or {}).get(
            "full_l0_attachments_opened"
        )
        is not False
    ):
        raise RuntimeError("BMR_10 scaling owner causal/identity contract drifted")
    source_code = owner_manifest.get("source_code")
    expected_sources = {
        "native_cnbr_core": REPO_ROOT
        / "experiments/l1_balance/native_cnbr_core.py",
        "budgeted_policy": Path(budgeted_policy.__file__).resolve(),
        "materializer": REPO_ROOT
        / "experiments/c1/scripts/c1_orion_bmr10_scaling_materialize.py",
    }
    if not isinstance(source_code, dict) or set(source_code) != set(expected_sources):
        raise RuntimeError("BMR_10 scaling source-code binding is incomplete")
    for key, path in expected_sources.items():
        record = source_code[key]
        if (
            not isinstance(record, dict)
            or Path(str(record.get("path") or "")).resolve() != path.resolve()
            or record.get("sha256") != layout_common.sha256_path(path)
            or record.get("size_bytes") != path.stat().st_size
        ):
            raise RuntimeError(f"BMR_10 scaling source binding drifted: {key}")

    policy_path = _required_manifest_file(
        provenance, "bmr10_policy_source", "BMR_10 scaling provenance"
    )
    if (
        policy_path != Path(budgeted_policy.__file__).resolve()
        or layout_common.sha256_path(policy_path)
        != _canonical_manifest_sha256(
            provenance,
            "bmr10_policy_source_sha256",
            "BMR_10 scaling provenance",
        )
    ):
        raise RuntimeError("BMR_10 scaling policy source binding mismatch")

    owner_record = (owner_manifest.get("owners") or {}).get("C_CNBR")
    if not isinstance(owner_record, dict):
        raise RuntimeError("BMR_10 scaling C_CNBR owner binding is missing")
    owner_path = safe_child(layout_dir, outputs.get("owner"), "owner")
    owner_relative = owner_path.relative_to(layout_dir).as_posix()
    owner_sha256 = _canonical_manifest_sha256(
        owner_record, "sha256", "BMR_10 scaling owner"
    )
    if (
        checksums.get(owner_relative) != owner_sha256
        or layout_common.sha256_path(owner_path) != owner_sha256
        or diagnostics.get("owner_sha256") != owner_sha256
        or diagnostics.get("scaling_owner_manifest_sha256")
        != owner_manifest_sha256
    ):
        raise RuntimeError("BMR_10 scaling owner bytes are not bound end-to-end")

    import_manifest_path = safe_child(
        layout_dir, outputs.get("import_manifest"), "import_manifest"
    )
    import_manifest = _read_manifest_object(
        import_manifest_path, "BMR_10 scaling import manifest"
    )
    assignment_path = safe_child(
        layout_dir, import_manifest.get("assignments_file"), "assignments_file"
    )
    assignment_relative = assignment_path.relative_to(layout_dir).as_posix()
    if checksums.get(assignment_relative) != assignment_metrics.get(
        "assignment_jsonl_sha256"
    ):
        raise RuntimeError("BMR_10 scaling assignment checksum binding mismatch")
    assignment_proof = _validate_budgeted_multi_assignment_bytes(
        assignment_path,
        candidate={
            **assignment_metrics,
            "extra_copy_budget_numerator": 1,
            "extra_copy_budget_denominator": 10,
        },
        shard_count=shard_count,
        require_exact_budget=False,
    )
    if (
        assignment_proof["logical_point_count"] != logical_count
        or assignment_proof["physical_point_count"] != physical_count
        or assignment_proof["assignment_jsonl_sha256"]
        != artifact_payload.get("layout_sha256")
        or assignment_proof["physical_copy_shard_loads"]
        != routing.get("shard_counts")
        or assignment_proof["copy_count_histogram"]
        != routing.get("copy_count_histogram")
    ):
        raise RuntimeError("BMR_10 scaling assignment/artifact binding mismatch")
    expected_load_summary = {
        "cv": assignment_proof["physical_copy_load_cv"],
        "max": assignment_proof["physical_copy_load_max"],
        "max_over_mean": assignment_proof["physical_copy_load_max_over_mean"],
        "mean": assignment_proof["physical_copy_load_mean"],
        "min": assignment_proof["physical_copy_load_min"],
        "min_over_mean": assignment_proof["physical_copy_load_min_over_mean"],
    }
    if routing.get("load_summary") != expected_load_summary:
        raise RuntimeError("BMR_10 scaling load summary mismatch")

    p32 = diagnostics.get("p32_reproduction")
    if not isinstance(p32, dict) or p32.get("required") is not (shard_count == 32):
        raise RuntimeError("BMR_10 scaling P=32 reproduction declaration drifted")
    if shard_count == 32 and not all(
        value is True for key, value in p32.items() if key != "required"
    ):
        raise RuntimeError("BMR_10 scaling failed P=32 byte parity")

    source_artifact_record = (
        owner_manifest.get("construction_inputs") or {}
    ).get("artifact")
    if not isinstance(source_artifact_record, dict):
        raise RuntimeError("BMR_10 scaling source artifact record is missing")
    source_artifact_path = _required_manifest_file(
        source_artifact_record, "path", "BMR_10 scaling source artifact"
    )
    source_artifact_sha256 = _canonical_manifest_sha256(
        source_artifact_record,
        "sha256",
        "BMR_10 scaling source artifact",
    )
    if layout_common.sha256_path(source_artifact_path) != source_artifact_sha256:
        raise RuntimeError("BMR_10 scaling source artifact checksum mismatch")
    replay_proof = _validate_orion_l1_upper_replay(
        build_manifest,
        artifact_payload,
        diagnostics,
        layout_dir,
        checksums,
        source_artifact_path,
        source_artifact_sha256,
    )
    return {
        "mode": "fixed_p_cnbr_with_budgeted_multi_assignment",
        "arm": "C_CNBR",
        "candidate": budgeted_policy.CANDIDATE_ID,
        "canonical_default_eligible": False,
        "scaling_extension": "fixed_p_c_cnbr_bmr10_v1",
        "owner_sha256": owner_sha256,
        "owner_manifest_sha256": owner_manifest_sha256,
        "assignment": assignment_proof,
        "shard_count": shard_count,
        "l1_sizes": diagnostics.get("l1_sizes"),
        "shard_counts": routing.get("shard_counts"),
        "upper_replay": replay_proof,
        "p32_reproduction": p32,
    }


def validate_orion_artifact_owner_bmr10_layout(
    build_manifest: dict[str, Any],
    build_parameters: dict[str, Any],
    artifact_payload: dict[str, Any],
    layout_dir: Path,
    checksums: dict[str, str],
) -> dict[str, Any]:
    """Validate a research-only artifact-primary-owner + BMR_10 bundle."""
    from experiments.multi_assignment import budgeted_policy
    from experiments.multi_assignment import (
        materialize_artifact_owner_bmr10 as materializer,
    )

    routing = build_manifest.get("routing")
    provenance = build_manifest.get("provenance")
    outputs = build_manifest.get("outputs")
    if not all(isinstance(value, dict) for value in (routing, provenance, outputs)):
        raise RuntimeError("artifact-owner BMR_10 bundle proof is incomplete")
    diagnostics = routing.get("l1_partition_diagnostics")
    assignment_metrics = (
        diagnostics.get("assignment_metrics")
        if isinstance(diagnostics, dict)
        else None
    )
    if not isinstance(diagnostics, dict) or not isinstance(assignment_metrics, dict):
        raise RuntimeError("artifact-owner BMR_10 diagnostics are missing")
    shard_count = artifact_payload.get("shard_count")
    logical_count = artifact_payload.get("logical_point_count")
    physical_count = artifact_payload.get("physical_point_count")
    if shard_count != 32:
        raise RuntimeError("artifact-owner BMR_10 requires P=32")
    expected_parameters = {
        "generation": artifact_payload.get("generation"),
        "initial_num_shards": 32,
        "sample_denominator": 32,
        "upper_sample_seed": 100,
        "upper_m": 32,
        "upper_ef_construction": 100,
        "upper_graph_seed": 100,
        "attachment_search_ef": 100,
        "upper_k": artifact_payload.get("upper_k"),
        "upper_search_ef": artifact_payload.get("upper_ef_search"),
        "dynamic_ef_base": artifact_payload.get("dynamic_ef_base"),
        "dynamic_ef_factor": artifact_payload.get("dynamic_ef_factor"),
        "k_overlap": 10,
        "use_multi_assign": True,
        "multi_assign_min_max_vote": 2,
        "multi_assign_vote_delta": 0,
        "multi_assign_max_shards": 2,
        "multi_assign_extra_copy_budget_numerator": 1,
        "multi_assign_extra_copy_budget_denominator": 10,
        "multi_assign_score": budgeted_policy.SCORE,
        "multi_assignment_policy": budgeted_policy.CANDIDATE_ID,
        "multi_assignment_policy_version": budgeted_policy.POLICY_VERSION,
        "enable_fission": False,
        "enable_topology_refinement": False,
        "l0_repair": False,
        "allow_decoupled_runtime_upper_search": False,
        "l1_partitioner": "HISTORICAL_PRIMARY",
        "balance_mode": "artifact_first_membership_primary_research",
    }
    if build_parameters != expected_parameters:
        mismatches = {
            key: {"expected": expected, "actual": build_parameters.get(key)}
            for key, expected in expected_parameters.items()
            if build_parameters.get(key) != expected
            or type(build_parameters.get(key)) is not type(expected)
        }
        raise RuntimeError(
            "artifact-owner BMR_10 parameters drifted: "
            f"mismatches={mismatches}, "
            f"extra={sorted(set(build_parameters) - set(expected_parameters))}"
        )
    routing_expected = {
        "initial_num_shards": 32,
        "effective_num_shards": 32,
        "logical_point_count": logical_count,
        "physical_point_count": physical_count,
        "fission_events": [],
    }
    routing_mismatches = {
        key: {"expected": expected, "actual": routing.get(key)}
        for key, expected in routing_expected.items()
        if routing.get(key) != expected
        or type(routing.get(key)) is not type(expected)
    }
    if routing_mismatches:
        raise RuntimeError(
            f"artifact-owner BMR_10 routing drifted: {routing_mismatches}"
        )
    if (
        diagnostics.get("research_candidate")
        != "artifact_first_membership_primary_plus_bmr10"
        or diagnostics.get("historical_full_layout_reproduction_claim") is not False
        or diagnostics.get("topology_all_pass") is not True
        or diagnostics.get("identity_all_pass") is not True
    ):
        raise RuntimeError("artifact-owner BMR_10 research boundary drifted")

    owner_manifest_path = _required_manifest_file(
        provenance, "artifact_owner_manifest", "artifact-owner BMR_10 provenance"
    )
    owner_manifest_sha256 = _canonical_manifest_sha256(
        provenance,
        "artifact_owner_manifest_sha256",
        "artifact-owner BMR_10 provenance",
    )
    if layout_common.sha256_path(owner_manifest_path) != owner_manifest_sha256:
        raise RuntimeError("artifact-owner manifest checksum mismatch")
    owner_manifest = _read_manifest_object(
        owner_manifest_path, "artifact-owner BMR_10 owner manifest"
    )
    boundary = owner_manifest.get("causal_boundary") or {}
    if (
        owner_manifest.get("status") != "PASS"
        or owner_manifest.get("record_type")
        != "artifact_owner_bmr10_owner_manifest"
        or owner_manifest.get("owner_name") != "HISTORICAL_PRIMARY"
        or owner_manifest.get("partitions") != 32
        or boundary.get("owner_frozen_before_full_l0_attachment_access") is not True
        or boundary.get("full_l0_attachments_opened_by_freeze_step") is not False
        or boundary.get("queries_opened_by_freeze_step") is not False
        or boundary.get("ground_truth_opened_by_freeze_step") is not False
        or boundary.get("owner_source_uses_upper_memberships_only") is not True
    ):
        raise RuntimeError("artifact-owner manifest causal/identity contract drifted")
    source_code = owner_manifest.get("source_code")
    expected_sources = {
        "budgeted_policy": Path(budgeted_policy.__file__).resolve(),
        "matrix_screen": REPO_ROOT
        / "experiments/multi_assignment/screen_owner_policy_matrix.py",
        "materializer": Path(materializer.__file__).resolve(),
    }
    if not isinstance(source_code, dict) or set(source_code) != set(expected_sources):
        raise RuntimeError("artifact-owner source-code binding is incomplete")
    for key, path in expected_sources.items():
        record = source_code[key]
        if (
            not isinstance(record, dict)
            or Path(str(record.get("path") or "")).resolve() != path.resolve()
            or record.get("sha256") != layout_common.sha256_path(path)
            or record.get("size_bytes") != path.stat().st_size
        ):
            raise RuntimeError(f"artifact-owner source binding drifted: {key}")

    policy_path = _required_manifest_file(
        provenance, "bmr10_policy_source", "artifact-owner BMR_10 provenance"
    )
    if (
        policy_path != Path(budgeted_policy.__file__).resolve()
        or layout_common.sha256_path(policy_path)
        != _canonical_manifest_sha256(
            provenance,
            "bmr10_policy_source_sha256",
            "artifact-owner BMR_10 provenance",
        )
    ):
        raise RuntimeError("artifact-owner BMR_10 policy source mismatch")
    matrix_path = _required_manifest_file(
        provenance, "owner_policy_matrix", "artifact-owner BMR_10 provenance"
    )
    matrix_sha256 = _canonical_manifest_sha256(
        provenance,
        "owner_policy_matrix_sha256",
        "artifact-owner BMR_10 provenance",
    )
    if (
        layout_common.sha256_path(matrix_path) != matrix_sha256
        or (owner_manifest.get("construction_inputs") or {})
        .get("owner_policy_matrix", {})
        .get("sha256")
        != matrix_sha256
    ):
        raise RuntimeError("artifact-owner matrix binding mismatch")
    selected = owner_manifest.get("selected_matrix_row")
    if (
        not isinstance(selected, dict)
        or selected.get("combination") != "HISTORICAL_PRIMARY+BMR_10"
        or selected.get("all_offline_gates_pass") is not True
        or selected.get("membership_semantic_sha256")
        != assignment_metrics.get("membership_semantic_sha256")
    ):
        raise RuntimeError("artifact-owner selected matrix row drifted")

    owner_record = owner_manifest.get("owner")
    if not isinstance(owner_record, dict):
        raise RuntimeError("artifact-owner binary binding is missing")
    owner_path = safe_child(layout_dir, outputs.get("owner"), "owner")
    owner_relative = owner_path.relative_to(layout_dir).as_posix()
    owner_sha256 = _canonical_manifest_sha256(
        owner_record, "sha256", "artifact-owner BMR_10 owner"
    )
    if (
        checksums.get(owner_relative) != owner_sha256
        or layout_common.sha256_path(owner_path) != owner_sha256
        or diagnostics.get("owner_sha256") != owner_sha256
        or diagnostics.get("owner_manifest_sha256") != owner_manifest_sha256
        or diagnostics.get("owner_semantic_sha256")
        != owner_record.get("semantic_sha256")
    ):
        raise RuntimeError("artifact-owner bytes are not bound end-to-end")

    import_manifest_path = safe_child(
        layout_dir, outputs.get("import_manifest"), "import_manifest"
    )
    import_manifest = _read_manifest_object(
        import_manifest_path, "artifact-owner BMR_10 import manifest"
    )
    assignment_path = safe_child(
        layout_dir, import_manifest.get("assignments_file"), "assignments_file"
    )
    assignment_relative = assignment_path.relative_to(layout_dir).as_posix()
    if checksums.get(assignment_relative) != assignment_metrics.get(
        "assignment_jsonl_sha256"
    ):
        raise RuntimeError("artifact-owner assignment checksum mismatch")
    assignment_proof = _validate_budgeted_multi_assignment_bytes(
        assignment_path,
        candidate={
            **assignment_metrics,
            "extra_copy_budget_numerator": 1,
            "extra_copy_budget_denominator": 10,
        },
        shard_count=32,
        require_exact_budget=True,
    )
    if (
        assignment_proof["logical_point_count"] != logical_count
        or assignment_proof["physical_point_count"] != physical_count
        or assignment_proof["assignment_jsonl_sha256"]
        != artifact_payload.get("layout_sha256")
        or assignment_proof["physical_copy_shard_loads"]
        != routing.get("shard_counts")
        or assignment_proof["copy_count_histogram"]
        != routing.get("copy_count_histogram")
    ):
        raise RuntimeError("artifact-owner assignment/artifact binding mismatch")
    expected_load_summary = {
        "cv": assignment_proof["physical_copy_load_cv"],
        "max": assignment_proof["physical_copy_load_max"],
        "max_over_mean": assignment_proof["physical_copy_load_max_over_mean"],
        "mean": assignment_proof["physical_copy_load_mean"],
        "min": assignment_proof["physical_copy_load_min"],
        "min_over_mean": assignment_proof["physical_copy_load_min_over_mean"],
    }
    if routing.get("load_summary") != expected_load_summary:
        raise RuntimeError("artifact-owner load summary mismatch")

    source_artifact_path = _required_manifest_file(
        provenance, "source_artifact", "artifact-owner BMR_10 provenance"
    )
    source_artifact_sha256 = _canonical_manifest_sha256(
        provenance,
        "source_artifact_sha256",
        "artifact-owner BMR_10 provenance",
    )
    if layout_common.sha256_path(source_artifact_path) != source_artifact_sha256:
        raise RuntimeError("artifact-owner source artifact checksum mismatch")
    replay_proof = _validate_orion_l1_upper_replay(
        build_manifest,
        artifact_payload,
        diagnostics,
        layout_dir,
        checksums,
        source_artifact_path,
        source_artifact_sha256,
    )
    return {
        "mode": "artifact_first_membership_primary_with_budgeted_multi_assignment",
        "arm": "HISTORICAL_PRIMARY",
        "candidate": budgeted_policy.CANDIDATE_ID,
        "canonical_default_eligible": False,
        "historical_full_layout_reproduction_claim": False,
        "owner_sha256": owner_sha256,
        "owner_manifest_sha256": owner_manifest_sha256,
        "matrix_sha256": matrix_sha256,
        "assignment": assignment_proof,
        "shard_count": 32,
        "l1_sizes": diagnostics.get("l1_sizes"),
        "shard_counts": routing.get("shard_counts"),
        "upper_replay": replay_proof,
    }


def _validate_owner_policy_tournament_assignment_bytes(
    assignment_path: Path,
    *,
    candidate: dict[str, Any],
    shard_count: int,
    policy_name: str,
) -> dict[str, Any]:
    """Replay an arbitrary frozen tournament assignment and its exact metrics."""
    expected_sha256 = _canonical_manifest_sha256(
        candidate,
        "assignment_jsonl_sha256",
        "owner-policy tournament candidate",
    )
    if layout_common.sha256_path(assignment_path) != expected_sha256:
        raise RuntimeError("owner-policy assignment bytes differ from the bundle proof")

    primary_digest = hashlib.sha256()
    semantic_digest = hashlib.sha256()
    primary_loads = [0] * shard_count
    physical_loads = [0] * shard_count
    copy_histogram: dict[str, int] = {}
    physical_point_count = 0
    logical_point_count = 0
    maximum_copies = 0
    packed_width = (shard_count + 7) // 8
    with assignment_path.open("rb") as handle:
        for expected_id, raw_line in enumerate(handle):
            try:
                row = json.loads(raw_line)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"invalid owner-policy assignment JSON at row {expected_id}"
                ) from exc
            if not isinstance(row, dict) or set(row) != {"id", "shards"}:
                raise RuntimeError(
                    f"owner-policy assignment row {expected_id} has an invalid schema"
                )
            point_id = row.get("id")
            shards = row.get("shards")
            if (
                isinstance(point_id, bool)
                or not isinstance(point_id, int)
                or point_id != expected_id
            ):
                raise RuntimeError("owner-policy assignment IDs are not contiguous")
            if not isinstance(shards, list) or not shards:
                raise RuntimeError("owner-policy assignment contains an empty row")
            if any(
                isinstance(shard, bool)
                or not isinstance(shard, int)
                or shard < 0
                or shard >= shard_count
                for shard in shards
            ):
                raise RuntimeError(
                    f"owner-policy assignment row {expected_id} has an invalid shard"
                )
            if shards != sorted(set(shards)):
                raise RuntimeError(
                    f"owner-policy assignment row {expected_id} is not canonical"
                )
            packed = bytearray(packed_width)
            for shard in shards:
                packed[shard // 8] |= 1 << (shard % 8)
                physical_loads[shard] += 1
            semantic_digest.update(packed)
            primary_digest.update(struct.pack("<i", shards[0]))
            primary_loads[shards[0]] += 1
            copy_key = str(len(shards))
            copy_histogram[copy_key] = copy_histogram.get(copy_key, 0) + 1
            physical_point_count += len(shards)
            logical_point_count += 1
            maximum_copies = max(maximum_copies, len(shards))

    if policy_name == "single_rank" and maximum_copies != 1:
        raise RuntimeError("single_rank tournament assignment contains replicas")
    if policy_name == "BMR_10":
        extra = physical_point_count - logical_point_count
        if maximum_copies > 2 or extra > logical_point_count // 10:
            raise RuntimeError("BMR_10 tournament assignment exceeds its copy contract")
    load_array = experiment.np.asarray(physical_loads, dtype=experiment.np.int64)
    load_mean = float(load_array.mean())
    observed = {
        "policy": policy_name,
        "assignment_jsonl_sha256": expected_sha256,
        "membership_semantic_sha256": semantic_digest.hexdigest(),
        "logical_point_count": logical_point_count,
        "physical_point_count": physical_point_count,
        "expansion_ratio": physical_point_count / logical_point_count,
        "extra_copy_fraction": physical_point_count / logical_point_count - 1.0,
        "copy_count_histogram": {
            key: copy_histogram[key] for key in sorted(copy_histogram, key=int)
        },
        "physical_copy_load_min": int(load_array.min()),
        "physical_copy_load_max": int(load_array.max()),
        "physical_copy_load_mean": load_mean,
        "physical_copy_load_cv": float(load_array.std() / load_mean),
        "physical_copy_load_max_over_mean": float(load_array.max() / load_mean),
        "physical_copy_load_min_over_mean": float(load_array.min() / load_mean),
        "physical_copy_load_empty_shards": int((load_array == 0).sum()),
    }
    mismatches = {
        key: {"expected": candidate.get(key), "actual": value}
        for key, value in observed.items()
        if candidate.get(key) != value
    }
    if mismatches:
        raise RuntimeError(
            f"owner-policy assignment differs from frozen matrix row: {mismatches}"
        )
    return {
        **observed,
        "canonical_format": "orion_numeric_import.assignments.jsonl-v1",
        "primary_shards_sha256": primary_digest.hexdigest(),
        "primary_shard_loads": primary_loads,
        "physical_copy_shard_loads": physical_loads,
        "extra_copy_count": physical_point_count - logical_point_count,
        "maximum_copies_per_point": maximum_copies,
    }


def validate_orion_owner_policy_tournament_layout(
    build_manifest: dict[str, Any],
    build_parameters: dict[str, Any],
    artifact_payload: dict[str, Any],
    layout_dir: Path,
    checksums: dict[str, str],
) -> dict[str, Any]:
    """Validate one research-only owner x multi-assignment tournament bundle."""
    from experiments.multi_assignment import (
        materialize_owner_policy_candidate as materializer,
    )

    routing = build_manifest.get("routing")
    provenance = build_manifest.get("provenance")
    outputs = build_manifest.get("outputs")
    if not all(isinstance(value, dict) for value in (routing, provenance, outputs)):
        raise RuntimeError("owner-policy tournament bundle proof is incomplete")
    diagnostics = routing.get("l1_partition_diagnostics")
    assignment_metrics = (
        diagnostics.get("assignment_metrics")
        if isinstance(diagnostics, dict)
        else None
    )
    if not isinstance(diagnostics, dict) or not isinstance(assignment_metrics, dict):
        raise RuntimeError("owner-policy tournament diagnostics are missing")
    owner_name = diagnostics.get("owner_name")
    policy_name = diagnostics.get("multi_assignment_policy")
    combination = diagnostics.get("combination")
    if (
        owner_name not in materializer.OWNER_NAMES
        or policy_name not in materializer.POLICY_NAMES
        or combination != f"{owner_name}+{policy_name}"
        or diagnostics.get("research_candidate") != "owner_policy_tournament"
        or diagnostics.get("online_matched_recall_qps_required") is not True
        or diagnostics.get("canonical_default_eligible") is not False
    ):
        raise RuntimeError("owner-policy tournament identity boundary drifted")
    shard_count = artifact_payload.get("shard_count")
    logical_count = artifact_payload.get("logical_point_count")
    physical_count = artifact_payload.get("physical_point_count")
    if shard_count != 32:
        raise RuntimeError("owner-policy tournament requires P=32")

    expected_parameters = {
        "generation": artifact_payload.get("generation"),
        "initial_num_shards": 32,
        "sample_denominator": 32,
        "upper_sample_seed": 100,
        "upper_m": 32,
        "upper_ef_construction": 100,
        "upper_graph_seed": 100,
        "attachment_search_ef": 100,
        "upper_k": artifact_payload.get("upper_k"),
        "upper_search_ef": artifact_payload.get("upper_ef_search"),
        "dynamic_ef_base": artifact_payload.get("dynamic_ef_base"),
        "dynamic_ef_factor": artifact_payload.get("dynamic_ef_factor"),
        "k_overlap": 10,
        "allow_decoupled_runtime_upper_search": False,
        **materializer.policy_parameter_overrides(owner_name, policy_name),
    }
    if build_parameters != expected_parameters:
        mismatches = {
            key: {"expected": expected, "actual": build_parameters.get(key)}
            for key, expected in expected_parameters.items()
            if build_parameters.get(key) != expected
            or type(build_parameters.get(key)) is not type(expected)
        }
        raise RuntimeError(
            "owner-policy tournament parameters drifted: "
            f"mismatches={mismatches}, "
            f"extra={sorted(set(build_parameters) - set(expected_parameters))}"
        )
    routing_expected = {
        "initial_num_shards": 32,
        "effective_num_shards": 32,
        "logical_point_count": logical_count,
        "physical_point_count": physical_count,
        "fission_events": [],
    }
    routing_mismatches = {
        key: {"expected": expected, "actual": routing.get(key)}
        for key, expected in routing_expected.items()
        if routing.get(key) != expected
        or type(routing.get(key)) is not type(expected)
    }
    if routing_mismatches:
        raise RuntimeError(
            f"owner-policy tournament routing drifted: {routing_mismatches}"
        )

    materializer_path = _required_manifest_file(
        provenance,
        "tournament_materializer_source",
        "owner-policy tournament provenance",
    )
    matrix_source_path = _required_manifest_file(
        provenance,
        "matrix_screen_source",
        "owner-policy tournament provenance",
    )
    expected_sources = {
        materializer_path: _canonical_manifest_sha256(
            provenance,
            "tournament_materializer_source_sha256",
            "owner-policy tournament provenance",
        ),
        matrix_source_path: _canonical_manifest_sha256(
            provenance,
            "matrix_screen_source_sha256",
            "owner-policy tournament provenance",
        ),
    }
    if materializer_path != Path(materializer.__file__).resolve():
        raise RuntimeError("owner-policy materializer source path drifted")
    if matrix_source_path != REPO_ROOT / "experiments/multi_assignment/screen_owner_policy_matrix.py":
        raise RuntimeError("owner-policy matrix source path drifted")
    for path, digest in expected_sources.items():
        if layout_common.sha256_path(path) != digest:
            raise RuntimeError(f"owner-policy source checksum drifted: {path}")

    matrix_path = _required_manifest_file(
        provenance, "owner_policy_matrix", "owner-policy tournament provenance"
    )
    matrix_sha256 = _canonical_manifest_sha256(
        provenance,
        "owner_policy_matrix_sha256",
        "owner-policy tournament provenance",
    )
    if layout_common.sha256_path(matrix_path) != matrix_sha256:
        raise RuntimeError("owner-policy matrix checksum drifted")
    matrix_manifest = _read_manifest_object(matrix_path, "owner-policy matrix")
    selected_rows = [
        row
        for row in matrix_manifest.get("rows", [])
        if isinstance(row, dict) and row.get("combination") == combination
    ]
    if len(selected_rows) != 1:
        raise RuntimeError("owner-policy matrix candidate is missing or duplicated")
    selected = selected_rows[0]
    parity_fields = (
        "logical_point_count",
        "physical_point_count",
        "expansion_ratio",
        "extra_copy_fraction",
        "copy_count_histogram",
        "membership_semantic_sha256",
        "physical_copy_load_min",
        "physical_copy_load_max",
        "physical_copy_load_mean",
        "physical_copy_load_cv",
        "physical_copy_load_max_over_mean",
        "physical_copy_load_min_over_mean",
        "physical_copy_load_empty_shards",
        "eligible_secondary_count",
        "kept_secondary_count",
        "extra_copy_budget_cap",
    )
    matrix_drift = {
        key: {"bundle": assignment_metrics.get(key), "matrix": selected.get(key)}
        for key in parity_fields
        if (key in assignment_metrics or key in selected)
        and assignment_metrics.get(key) != selected.get(key)
    }
    if matrix_drift:
        raise RuntimeError(f"owner-policy matrix metric parity drifted: {matrix_drift}")
    if (
        diagnostics.get("matrix_row_all_offline_gates_pass")
        is not selected.get("all_offline_gates_pass")
        or diagnostics.get("owner_semantic_sha256")
        != selected.get("owner_semantic_sha256")
    ):
        raise RuntimeError("owner-policy matrix row metadata drifted")

    owner_manifest_path = _required_manifest_file(
        provenance,
        "owner_evidence_manifest",
        "owner-policy tournament provenance",
    )
    owner_manifest_sha256 = _canonical_manifest_sha256(
        provenance,
        "owner_evidence_manifest_sha256",
        "owner-policy tournament provenance",
    )
    if layout_common.sha256_path(owner_manifest_path) != owner_manifest_sha256:
        raise RuntimeError("owner-policy owner manifest checksum drifted")
    owner_manifest = _read_manifest_object(
        owner_manifest_path, "owner-policy owner manifest"
    )
    boundary = owner_manifest.get("causal_boundary") or {}
    if (
        owner_manifest.get("record_type")
        != "owner_policy_tournament_owner_manifest"
        or owner_manifest.get("status") != "PASS"
        or owner_manifest.get("owner_name") != owner_name
        or owner_manifest.get("partitions") != 32
        or boundary.get("owner_frozen_before_full_l0_attachment_access") is not True
        or boundary.get("full_l0_attachments_opened_by_freeze_step") is not False
        or boundary.get("queries_opened_by_freeze_step") is not False
        or boundary.get("ground_truth_opened_by_freeze_step") is not False
    ):
        raise RuntimeError("owner-policy owner causal contract drifted")
    manifest_matrix = (owner_manifest.get("construction_inputs") or {}).get(
        "owner_policy_matrix"
    )
    if (
        not isinstance(manifest_matrix, dict)
        or manifest_matrix.get("path") != str(matrix_path)
        or manifest_matrix.get("sha256") != matrix_sha256
    ):
        raise RuntimeError("owner-policy owner/matrix binding drifted")
    owner_record = owner_manifest.get("owner")
    if not isinstance(owner_record, dict):
        raise RuntimeError("owner-policy owner binary record is missing")
    owner_path = safe_child(layout_dir, outputs.get("owner"), "owner")
    owner_relative = owner_path.relative_to(layout_dir).as_posix()
    owner_sha256 = _canonical_manifest_sha256(
        owner_record, "sha256", "owner-policy owner"
    )
    if (
        checksums.get(owner_relative) != owner_sha256
        or layout_common.sha256_path(owner_path) != owner_sha256
        or diagnostics.get("owner_sha256") != owner_sha256
        or diagnostics.get("owner_manifest_sha256") != owner_manifest_sha256
        or owner_record.get("semantic_sha256")
        != selected.get("owner_semantic_sha256")
    ):
        raise RuntimeError("owner-policy owner bytes are not bound end-to-end")

    import_manifest_path = safe_child(
        layout_dir, outputs.get("import_manifest"), "import_manifest"
    )
    import_manifest = _read_manifest_object(
        import_manifest_path, "owner-policy import manifest"
    )
    assignment_path = safe_child(
        layout_dir, import_manifest.get("assignments_file"), "assignments_file"
    )
    assignment_relative = assignment_path.relative_to(layout_dir).as_posix()
    if checksums.get(assignment_relative) != assignment_metrics.get(
        "assignment_jsonl_sha256"
    ):
        raise RuntimeError("owner-policy assignment checksum binding drifted")
    assignment_proof = _validate_owner_policy_tournament_assignment_bytes(
        assignment_path,
        candidate=assignment_metrics,
        shard_count=32,
        policy_name=policy_name,
    )
    if (
        assignment_proof["logical_point_count"] != logical_count
        or assignment_proof["physical_point_count"] != physical_count
        or assignment_proof["assignment_jsonl_sha256"]
        != artifact_payload.get("layout_sha256")
        or assignment_proof["physical_copy_shard_loads"]
        != routing.get("shard_counts")
        or assignment_proof["copy_count_histogram"]
        != routing.get("copy_count_histogram")
    ):
        raise RuntimeError("owner-policy assignment/artifact binding drifted")
    expected_load_summary = {
        "cv": assignment_proof["physical_copy_load_cv"],
        "max": assignment_proof["physical_copy_load_max"],
        "max_over_mean": assignment_proof["physical_copy_load_max_over_mean"],
        "mean": assignment_proof["physical_copy_load_mean"],
        "min": assignment_proof["physical_copy_load_min"],
        "min_over_mean": assignment_proof["physical_copy_load_min_over_mean"],
    }
    if routing.get("load_summary") != expected_load_summary:
        raise RuntimeError("owner-policy load summary drifted")

    source_artifact_path = _required_manifest_file(
        provenance, "source_artifact", "owner-policy tournament provenance"
    )
    source_artifact_sha256 = _canonical_manifest_sha256(
        provenance,
        "source_artifact_sha256",
        "owner-policy tournament provenance",
    )
    if layout_common.sha256_path(source_artifact_path) != source_artifact_sha256:
        raise RuntimeError("owner-policy source artifact checksum drifted")
    replay_proof = _validate_orion_l1_upper_replay(
        build_manifest,
        artifact_payload,
        diagnostics,
        layout_dir,
        checksums,
        source_artifact_path,
        source_artifact_sha256,
    )
    return {
        "mode": "owner_policy_tournament",
        "combination": combination,
        "owner": owner_name,
        "policy": policy_name,
        "canonical_default_eligible": False,
        "owner_sha256": owner_sha256,
        "owner_manifest_sha256": owner_manifest_sha256,
        "matrix_sha256": matrix_sha256,
        "matrix_row_all_offline_gates_pass": selected.get(
            "all_offline_gates_pass"
        ),
        "assignment": assignment_proof,
        "shard_count": 32,
        "l1_sizes": diagnostics.get("l1_sizes"),
        "shard_counts": routing.get("shard_counts"),
        "upper_replay": replay_proof,
    }


def validate_orion_sampled_cnbr_layout(
    build_manifest: dict[str, Any],
    build_parameters: dict[str, Any],
    artifact_payload: dict[str, Any],
    layout_dir: Path,
    checksums: dict[str, str],
) -> dict[str, Any]:
    """Validate the fixed sampled-CNBR research bundle.

    This is intentionally a research-only authorization.  It accepts exactly
    the pre-screened GloVe fraction=0.25%, seed=100, 25/75 blend candidate and
    does not classify it as the canonical upper-only CNBR implementation.
    """

    from experiments.l1_balance import (  # Imported lazily for normal layouts.
        materialize_sampled_cnbr_candidate as sampled_cnbr,
    )

    expected_parameters = {
        "generation": artifact_payload.get("generation"),
        "initial_num_shards": artifact_payload.get("shard_count"),
        "sample_denominator": 32,
        "upper_sample_seed": 100,
        "upper_m": 32,
        "upper_ef_construction": 100,
        "upper_graph_seed": 100,
        "attachment_search_ef": 100,
        "upper_k": artifact_payload.get("upper_k"),
        "upper_search_ef": artifact_payload.get("upper_ef_search"),
        "dynamic_ef_base": artifact_payload.get("dynamic_ef_base"),
        "dynamic_ef_factor": artifact_payload.get("dynamic_ef_factor"),
        "k_overlap": 10,
        "use_multi_assign": True,
        "multi_assign_min_max_vote": 2,
        "multi_assign_vote_delta": 0,
        "multi_assign_max_shards": 0,
        "enable_fission": False,
        "enable_topology_refinement": False,
        "l0_repair": False,
        "allow_decoupled_runtime_upper_search": False,
        "l1_partitioner": sampled_cnbr.ARM_NAME,
        "balance_mode": sampled_cnbr.BALANCE_VARIANT,
        "sample_fill_fraction": sampled_cnbr.EXPECTED_FRACTION,
        "sample_fill_seed": sampled_cnbr.EXPECTED_SEED,
        "sample_fill_weight": sampled_cnbr.EXPECTED_SAMPLE_WEIGHT,
        "upper_prior_weight": 1.0 - sampled_cnbr.EXPECTED_SAMPLE_WEIGHT,
    }
    if build_parameters != expected_parameters:
        mismatches = {
            key: {"expected": expected, "actual": build_parameters.get(key)}
            for key, expected in expected_parameters.items()
            if build_parameters.get(key) != expected
            or type(build_parameters.get(key)) is not type(expected)
        }
        raise RuntimeError(
            "sampled-CNBR parameter contract drifted: "
            f"mismatches={mismatches}, "
            f"extra={sorted(set(build_parameters) - set(expected_parameters))}"
        )

    shard_count = artifact_payload.get("shard_count")
    logical_point_count = artifact_payload.get("logical_point_count")
    physical_point_count = artifact_payload.get("physical_point_count")
    generation = artifact_payload.get("generation")
    if (
        shard_count != 32
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in (logical_point_count, physical_point_count, generation)
        )
        or physical_point_count < logical_point_count
    ):
        raise RuntimeError("sampled-CNBR artifact count contract is invalid")
    expected_binding = {
        "generation": generation,
        "layout_sha256": artifact_payload.get("layout_sha256"),
        "logical_point_count": logical_point_count,
        "physical_point_count": physical_point_count,
        "shard_count": shard_count,
    }
    if build_manifest.get("artifact_binding") != expected_binding:
        raise RuntimeError("sampled-CNBR artifact_binding mismatch")

    routing = build_manifest.get("routing")
    provenance = build_manifest.get("provenance")
    outputs = build_manifest.get("outputs")
    if not all(isinstance(value, dict) for value in (routing, provenance, outputs)):
        raise RuntimeError("sampled-CNBR bundle proof is incomplete")
    for key, expected in {
        "initial_num_shards": shard_count,
        "effective_num_shards": shard_count,
        "logical_point_count": logical_point_count,
        "physical_point_count": physical_point_count,
    }.items():
        if routing.get(key) != expected:
            raise RuntimeError(f"sampled-CNBR routing {key} mismatch")
    if routing.get("fission_events") != []:
        raise RuntimeError("sampled-CNBR must keep fission disabled")
    shard_counts = routing.get("shard_counts")
    if (
        not isinstance(shard_counts, list)
        or len(shard_counts) != shard_count
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in shard_counts
        )
        or sum(shard_counts) != physical_point_count
    ):
        raise RuntimeError("sampled-CNBR shard_counts are invalid")

    diagnostics = routing.get("l1_partition_diagnostics")
    if not isinstance(diagnostics, dict):
        raise RuntimeError("sampled-CNBR diagnostics are missing")
    diagnostic_expected = {
        "arm": sampled_cnbr.ARM_NAME,
        "input_scope": (
            "production_upper_graph_upper_self_navigation_and_sampled_l0_attachments"
        ),
        "balance_contract": {"variant": sampled_cnbr.BALANCE_VARIANT},
        "canonical_default_eligible": False,
        "offline_research_only": True,
        "sampled_attachment_fraction": sampled_cnbr.EXPECTED_FRACTION,
        "sampled_attachment_rows_read": 2959,
        "multi_assignment_after_owner_freeze": True,
    }
    mismatches = {
        key: {"expected": expected, "actual": diagnostics.get(key)}
        for key, expected in diagnostic_expected.items()
        if diagnostics.get(key) != expected
        or type(diagnostics.get(key)) is not type(expected)
    }
    if mismatches:
        raise RuntimeError(f"sampled-CNBR diagnostics drifted: {mismatches}")
    if not all(
        all(isinstance(gate, dict) and gate.get("pass") is True for gate in gates.values())
        for gates in (
            diagnostics.get("graph_topology_gates") or {},
            diagnostics.get("query_topology_gates") or {},
        )
    ):
        raise RuntimeError("sampled-CNBR offline topology gates are not all PASS")

    evidence_path = _required_manifest_file(
        provenance, "sampled_owner_evidence", "sampled-CNBR provenance"
    )
    evidence_sha256 = _canonical_manifest_sha256(
        provenance,
        "sampled_owner_evidence_sha256",
        "sampled-CNBR provenance",
    )
    if layout_common.sha256_path(evidence_path) != evidence_sha256:
        raise RuntimeError("sampled-CNBR owner evidence checksum mismatch")
    if diagnostics.get("sampled_owner_evidence_sha256") != evidence_sha256:
        raise RuntimeError("sampled-CNBR diagnostics bind different owner evidence")
    evidence = _read_manifest_object(evidence_path, "sampled-CNBR owner evidence")
    evidence_expected = {
        "format_version": 1,
        "record_type": "sampled_cnbr_research_owner",
        "status": "PASS",
        "canonical_default_eligible": False,
        "dataset": "glove-200-angular",
        "method": sampled_cnbr.ARM_NAME,
        "balance_variant": sampled_cnbr.BALANCE_VARIANT,
    }
    if any(evidence.get(key) != value for key, value in evidence_expected.items()):
        raise RuntimeError("sampled-CNBR owner evidence identity drifted")
    if evidence.get("parameters") != {
        "fraction": sampled_cnbr.EXPECTED_FRACTION,
        "seed": sampled_cnbr.EXPECTED_SEED,
        "sample_weight": sampled_cnbr.EXPECTED_SAMPLE_WEIGHT,
        "upper_prior_weight": 1.0 - sampled_cnbr.EXPECTED_SAMPLE_WEIGHT,
        "cnbr_ratio": [9, 4],
        "num_partitions": 32,
        "multi_assignment": {
            "enabled": True,
            "min_max_vote": 2,
            "vote_delta": 0,
            "max_shards": 0,
        },
    }:
        raise RuntimeError("sampled-CNBR owner evidence parameter drift")
    source_code = evidence.get("source_code")
    if not isinstance(source_code, dict) or not source_code:
        raise RuntimeError("sampled-CNBR source-code evidence is missing")
    for relative, record in source_code.items():
        source_path = (REPO_ROOT / relative).resolve()
        if (
            not isinstance(record, dict)
            or not source_path.is_file()
            or record.get("sha256") != layout_common.sha256_path(source_path)
            or record.get("size_bytes") != source_path.stat().st_size
        ):
            raise RuntimeError(f"sampled-CNBR source-code binding drifted: {relative}")

    owner_path = safe_child(layout_dir, outputs.get("owner"), "owner")
    owner_relative = owner_path.relative_to(layout_dir).as_posix()
    owner_sha256 = _canonical_manifest_sha256(
        evidence, "owner_sha256", "sampled-CNBR owner evidence"
    )
    if (
        checksums.get(owner_relative) != owner_sha256
        or layout_common.sha256_path(owner_path) != owner_sha256
        or diagnostics.get("owner_sha256") != owner_sha256
    ):
        raise RuntimeError("sampled-CNBR owner bytes/checksum mismatch")
    owner = experiment.np.fromfile(owner_path, dtype="<i4")
    if (
        len(owner) != routing.get("upper_point_count")
        or experiment.np.any(owner < 0)
        or experiment.np.any(owner >= shard_count)
    ):
        raise RuntimeError("sampled-CNBR owner shape/range is invalid")
    l1_sizes = [
        int(value)
        for value in experiment.np.bincount(owner, minlength=shard_count).tolist()
    ]
    if l1_sizes != evidence.get("partition_sizes") or l1_sizes != diagnostics.get(
        "l1_sizes"
    ):
        raise RuntimeError("sampled-CNBR owner partition sizes drifted")

    import_manifest_path = safe_child(
        layout_dir, outputs.get("import_manifest"), "import_manifest"
    )
    import_manifest = _read_manifest_object(
        import_manifest_path, "sampled-CNBR import manifest"
    )
    assignment_path = safe_child(
        layout_dir, import_manifest.get("assignments_file"), "assignments_file"
    )
    vector_path = safe_child(
        layout_dir, import_manifest.get("vectors_file"), "vectors_file"
    )
    parity = evidence.get("materialization_parity")
    if not isinstance(parity, dict) or parity != diagnostics.get(
        "materialization_parity"
    ):
        raise RuntimeError("sampled-CNBR assignment parity evidence is missing")
    observed_parity = _validate_native_cnbr_assignment_golden(
        assignment_path,
        golden=parity,
        shard_count=shard_count,
    )
    if (
        observed_parity["logical_point_count"] != logical_point_count
        or observed_parity["physical_point_count"] != physical_point_count
        or observed_parity["physical_copy_shard_loads"] != shard_counts
        or observed_parity["copy_count_histogram"]
        != routing.get("copy_count_histogram")
        or observed_parity["assignment_bytes_sha256"]
        != artifact_payload.get("layout_sha256")
    ):
        raise RuntimeError("sampled-CNBR assignment parity differs from artifact")
    vectors_source = _required_manifest_file(
        provenance, "vectors_source", "sampled-CNBR provenance"
    )
    vectors_sha256 = _canonical_manifest_sha256(
        provenance, "vectors_source_sha256", "sampled-CNBR provenance"
    )
    if (
        provenance.get("vectors_materialization") != "hardlink"
        or provenance.get("vectors_same_file") is not True
        or not os.path.samefile(vector_path, vectors_source)
        or layout_common.sha256_path(vector_path) != vectors_sha256
    ):
        raise RuntimeError("sampled-CNBR vector hardlink/checksum contract failed")

    source_artifact_path = _required_manifest_file(
        provenance, "source_artifact", "sampled-CNBR provenance"
    )
    source_artifact_sha256 = _canonical_manifest_sha256(
        provenance, "source_artifact_sha256", "sampled-CNBR provenance"
    )
    if layout_common.sha256_path(source_artifact_path) != source_artifact_sha256:
        raise RuntimeError("sampled-CNBR source artifact checksum mismatch")
    rebind_path = layout_dir / "rebind-memberships.json"
    rebind_relative = rebind_path.relative_to(layout_dir).as_posix()
    if not rebind_path.is_file() or rebind_relative not in checksums:
        raise RuntimeError("sampled-CNBR rebind proof is missing")
    rebind = _read_manifest_object(rebind_path, "sampled-CNBR rebind proof")
    if rebind != {
        "format_version": 1,
        "source_artifact_sha256": source_artifact_sha256,
        "source_generation": json.loads(
            source_artifact_path.read_text(encoding="utf-8")
        )["generation"],
        "generation": generation,
        "layout_sha256": artifact_payload.get("layout_sha256"),
        "physical_point_count": physical_point_count,
        "shard_count": shard_count,
        "shard_memberships": [
            node.get("shard_membership")
            for node in artifact_payload.get("upper_nodes", [])
        ],
    }:
        raise RuntimeError("sampled-CNBR rebind proof differs from artifact")
    replay = _validate_orion_l1_upper_replay(
        build_manifest,
        artifact_payload,
        diagnostics,
        layout_dir,
        checksums,
        source_artifact_path,
        source_artifact_sha256,
    )
    return {
        "mode": sampled_cnbr.BALANCE_VARIANT,
        "arm": sampled_cnbr.ARM_NAME,
        "research_only": True,
        "canonical_default_eligible": False,
        "sample_fraction": sampled_cnbr.EXPECTED_FRACTION,
        "sample_seed": sampled_cnbr.EXPECTED_SEED,
        "sample_weight": sampled_cnbr.EXPECTED_SAMPLE_WEIGHT,
        "upper_prior_weight": 1.0 - sampled_cnbr.EXPECTED_SAMPLE_WEIGHT,
        "multi_assignment": {
            "enabled": True,
            "min_max_vote": 2,
            "vote_delta": 0,
            "max_shards": 0,
        },
        "assignment_parity": observed_parity,
        "shard_count": shard_count,
        "l1_sizes": l1_sizes,
        "shard_counts": shard_counts,
        "upper_replay": replay,
        "owner_evidence_sha256": evidence_sha256,
    }


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_orion_runtime_profile_derivation(
    build_manifest: dict[str, Any],
    build_parameters: dict[str, Any],
    artifact_payload: dict[str, Any],
    *,
    allow_scaling_initial_num_shards: bool = False,
    allow_balance_layout: bool = False,
    allow_l1_partition_layout: bool = False,
) -> dict[str, Any]:
    derivation = build_manifest.get("derivation")
    if not isinstance(derivation, dict):
        raise RuntimeError("Orion runtime profile is missing derivation metadata")
    if derivation.get("format_version") != 1:
        raise RuntimeError("unsupported Orion runtime-profile derivation format")
    if derivation.get("kind") != "orion_runtime_profile":
        raise RuntimeError("Orion runtime-profile derivation kind mismatch")
    if bool(derivation.get("orion_balance_layout_allowed", False)) != bool(
        allow_balance_layout
    ):
        raise RuntimeError("Orion runtime-profile balance-layout authorization mismatch")
    if bool(derivation.get("orion_l1_partition_layout_allowed", False)) != bool(
        allow_l1_partition_layout
    ):
        raise RuntimeError(
            "Orion runtime-profile L1-partition authorization mismatch"
        )
    if derivation.get("allowed_parameter_changes") != list(
        ORION_RUNTIME_PARAMETER_KEYS
    ):
        raise RuntimeError("Orion runtime-profile allowed parameter set mismatch")
    rebuild = derivation.get("rebuild")
    if not isinstance(rebuild, dict):
        raise RuntimeError("Orion runtime profile is missing rebuild provenance")
    rebuild_expected = {
        "upper_graph_seed": build_parameters.get("upper_graph_seed"),
        "upper_m": build_parameters.get("upper_m"),
        "upper_ef_construction": build_parameters.get("upper_ef_construction"),
    }
    rebuild_mismatches = {
        key: {"expected": expected, "actual": rebuild.get(key)}
        for key, expected in rebuild_expected.items()
        if rebuild.get(key) != expected
    }
    if rebuild_mismatches:
        raise RuntimeError(
            f"Orion runtime-profile upper graph rebuild mismatch: {rebuild_mismatches}"
        )

    source = derivation.get("source")
    if not isinstance(source, dict):
        raise RuntimeError("Orion runtime profile is missing its source binding")
    source_parameters = source.get("parameters")
    if not isinstance(source_parameters, dict):
        raise RuntimeError("Orion runtime-profile source parameters are missing")
    expected_parameters_sha256 = cluster_tool.normalize_sha256(
        str(source.get("parameters_sha256") or "")
    )
    if canonical_json_sha256(source_parameters) != expected_parameters_sha256:
        raise RuntimeError("Orion runtime-profile source parameters checksum mismatch")
    for digest_field in (
        "build_manifest_sha256",
        "graphless_artifact_sha256",
        "production_artifact_sha256",
        "import_manifest_sha256",
        "dataset_sha256",
        "routing_sha256",
        "vectors_sha256",
        "assignments_sha256",
    ):
        cluster_tool.normalize_sha256(str(source.get(digest_field) or ""))

    changed_parameters = {
        key
        for key in set(source_parameters) | set(build_parameters)
        if source_parameters.get(key) != build_parameters.get(key)
    }
    forbidden_changes = changed_parameters - set(ORION_RUNTIME_PARAMETER_KEYS)
    if forbidden_changes:
        raise RuntimeError(
            "Orion runtime profile changes offline/main-idea parameters: "
            f"{sorted(forbidden_changes)}"
        )
    parameter_changes = derivation.get("parameter_changes")
    if not isinstance(parameter_changes, dict) or set(parameter_changes) != set(
        ORION_RUNTIME_PARAMETER_KEYS
    ):
        raise RuntimeError("Orion runtime-profile parameter change proof is incomplete")
    for key in ORION_RUNTIME_PARAMETER_KEYS:
        change = parameter_changes.get(key)
        if not isinstance(change, dict) or change != {
            "source": source_parameters.get(key),
            "derived": build_parameters.get(key),
        }:
            raise RuntimeError(
                f"Orion runtime-profile parameter change proof mismatch for {key}"
            )

    source_generation = source.get("generation")
    if (
        isinstance(source_generation, bool)
        or not isinstance(source_generation, int)
        or source_generation <= 0
    ):
        raise RuntimeError("Orion runtime-profile source generation is invalid")
    if source_parameters.get("generation") != source_generation:
        raise RuntimeError("Orion runtime-profile source generation binding mismatch")
    if build_parameters.get("generation") != artifact_payload.get("generation"):
        raise RuntimeError("Orion runtime-profile derived generation binding mismatch")

    dataset = build_manifest.get("dataset")
    routing = build_manifest.get("routing")
    if not isinstance(dataset, dict) or not isinstance(routing, dict):
        raise RuntimeError("Orion runtime-profile dataset/routing proof is missing")
    if canonical_json_sha256(dataset) != source.get("dataset_sha256"):
        raise RuntimeError("Orion runtime-profile dataset changed from its source")
    if canonical_json_sha256(routing) != source.get("routing_sha256"):
        raise RuntimeError("Orion runtime-profile routing summary changed from its source")

    current_expected = {
        "layout_sha256": artifact_payload.get("layout_sha256"),
        "logical_point_count": artifact_payload.get("logical_point_count"),
        "physical_point_count": artifact_payload.get("physical_point_count"),
        "shard_count": artifact_payload.get("shard_count"),
    }
    source_binding_mismatches = {
        key: {"expected": expected, "actual": source.get(key)}
        for key, expected in current_expected.items()
        if source.get(key) != expected
    }
    if source_binding_mismatches:
        raise RuntimeError(
            "Orion runtime-profile source layout binding mismatch: "
            f"{source_binding_mismatches}"
        )
    if source.get("assignments_sha256") != artifact_payload.get("layout_sha256"):
        raise RuntimeError(
            "Orion runtime-profile source assignments do not match layout_sha256"
        )

    source_artifact_payload = dict(artifact_payload)
    source_artifact_payload.update(
        {
            "generation": source_generation,
            "upper_k": source_parameters.get("upper_k"),
            "upper_ef_search": source_parameters.get("upper_search_ef"),
            "dynamic_ef_base": source_parameters.get("dynamic_ef_base"),
            "dynamic_ef_factor": source_parameters.get("dynamic_ef_factor"),
        }
    )
    result = dict(source)
    if allow_l1_partition_layout:
        source_layout_dir = Path(str(source.get("layout_dir") or "")).expanduser().resolve()
        source_layout = load_routed_layout(
            "orion",
            source_layout_dir,
            allow_orion_l1_partition_layout=True,
        )
        source_mismatches = {
            "build_manifest_sha256": (
                source.get("build_manifest_sha256"),
                source_layout.get("build_manifest_sha256"),
            ),
            "production_artifact_sha256": (
                source.get("production_artifact_sha256"),
                source_layout.get("artifact_sha256"),
            ),
            "import_manifest_sha256": (
                source.get("import_manifest_sha256"),
                source_layout.get("import_manifest_sha256"),
            ),
            "parameters": (
                source_parameters,
                source_layout.get("build_parameters"),
            ),
        }
        source_mismatches = {
            key: {"expected": expected, "actual": actual}
            for key, (expected, actual) in source_mismatches.items()
            if expected != actual
        }
        if source_mismatches:
            raise RuntimeError(
                "Orion runtime-profile L1 source binding mismatch: "
                f"{source_mismatches}"
            )
        proof = source_layout.get("l1_partition_layout_proof")
        if not isinstance(proof, dict):
            raise RuntimeError("Orion runtime-profile L1 source proof is missing")
        result["_l1_partition_layout_proof"] = proof
    else:
        validate_faithful_orion_build_parameters(
            source_parameters,
            source_artifact_payload,
            allow_scaling_initial_num_shards=(
                allow_scaling_initial_num_shards or allow_balance_layout
            ),
            allow_balance_layout=allow_balance_layout,
        )
    return result


def validate_simple_kmeans_runtime_profile_derivation(
    build_manifest: dict[str, Any],
    build_parameters: dict[str, Any],
    artifact_payload: dict[str, Any],
) -> dict[str, Any]:
    derivation = build_manifest.get("derivation")
    if not isinstance(derivation, dict):
        raise RuntimeError(
            "Simple KMeans runtime profile is missing derivation metadata"
        )
    if derivation.get("format_version") != 1:
        raise RuntimeError(
            "unsupported Simple KMeans runtime-profile derivation format"
        )
    if derivation.get("kind") != "simple_kmeans_runtime_profile":
        raise RuntimeError("Simple KMeans runtime-profile derivation kind mismatch")
    if derivation.get("allowed_parameter_changes") != list(
        SIMPLE_KMEANS_RUNTIME_PARAMETER_KEYS
    ):
        raise RuntimeError(
            "Simple KMeans runtime-profile allowed parameter set mismatch"
        )

    source = derivation.get("source")
    if not isinstance(source, dict):
        raise RuntimeError(
            "Simple KMeans runtime profile is missing its source binding"
        )
    source_parameters = source.get("parameters")
    if not isinstance(source_parameters, dict):
        raise RuntimeError(
            "Simple KMeans runtime-profile source parameters are missing"
        )
    expected_parameters_sha256 = cluster_tool.normalize_sha256(
        str(source.get("parameters_sha256") or "")
    )
    if canonical_json_sha256(source_parameters) != expected_parameters_sha256:
        raise RuntimeError(
            "Simple KMeans runtime-profile source parameters checksum mismatch"
        )
    for digest_field in (
        "build_manifest_sha256",
        "graphless_artifact_sha256",
        "production_artifact_sha256",
        "import_manifest_sha256",
        "dataset_sha256",
        "routing_sha256",
        "offline_artifact_sha256",
        "vectors_sha256",
        "assignments_sha256",
    ):
        cluster_tool.normalize_sha256(str(source.get(digest_field) or ""))

    changed_parameters = {
        key
        for key in set(source_parameters) | set(build_parameters)
        if source_parameters.get(key) != build_parameters.get(key)
    }
    forbidden_changes = changed_parameters - set(
        SIMPLE_KMEANS_RUNTIME_PARAMETER_KEYS
    )
    if forbidden_changes:
        raise RuntimeError(
            "Simple KMeans runtime profile changes offline KMeans parameters: "
            f"{sorted(forbidden_changes)}"
        )
    parameter_changes = derivation.get("parameter_changes")
    if not isinstance(parameter_changes, dict) or set(parameter_changes) != set(
        SIMPLE_KMEANS_RUNTIME_PARAMETER_KEYS
    ):
        raise RuntimeError(
            "Simple KMeans runtime-profile parameter change proof is incomplete"
        )
    for key in SIMPLE_KMEANS_RUNTIME_PARAMETER_KEYS:
        change = parameter_changes.get(key)
        if not isinstance(change, dict) or change != {
            "source": source_parameters.get(key),
            "derived": build_parameters.get(key),
        }:
            raise RuntimeError(
                "Simple KMeans runtime-profile parameter change proof mismatch "
                f"for {key}"
            )

    source_generation = source.get("generation")
    if (
        isinstance(source_generation, bool)
        or not isinstance(source_generation, int)
        or source_generation <= 0
    ):
        raise RuntimeError(
            "Simple KMeans runtime-profile source generation is invalid"
        )
    if source_parameters.get("generation") != source_generation:
        raise RuntimeError(
            "Simple KMeans runtime-profile source generation binding mismatch"
        )
    runtime_bindings = {
        "generation": artifact_payload.get("generation"),
        "nprobe": artifact_payload.get("nprobe"),
        "lower_hnsw_ef": artifact_payload.get("lower_hnsw_ef"),
    }
    runtime_mismatches = {
        key: {"manifest": build_parameters.get(key), "artifact": value}
        for key, value in runtime_bindings.items()
        if build_parameters.get(key) != value
    }
    if runtime_mismatches:
        raise RuntimeError(
            "Simple KMeans build manifest/runtime artifact mismatch: "
            f"{runtime_mismatches}"
        )
    source_nprobe = source_parameters.get("nprobe")
    source_lower_ef = source_parameters.get("lower_hnsw_ef")
    shard_count = artifact_payload.get("shard_count")
    if (
        isinstance(source_nprobe, bool)
        or not isinstance(source_nprobe, int)
        or source_nprobe <= 0
        or isinstance(shard_count, bool)
        or not isinstance(shard_count, int)
        or source_nprobe > shard_count
    ):
        raise RuntimeError("Simple KMeans runtime-profile source nprobe is invalid")
    if (
        isinstance(source_lower_ef, bool)
        or not isinstance(source_lower_ef, int)
        or source_lower_ef <= 0
    ):
        raise RuntimeError(
            "Simple KMeans runtime-profile source lower_hnsw_ef is invalid"
        )
    if source_parameters.get("num_shards") != shard_count:
        raise RuntimeError(
            "Simple KMeans runtime-profile source shard-count binding mismatch"
        )

    dataset = build_manifest.get("dataset")
    routing = build_manifest.get("routing")
    if not isinstance(dataset, dict) or not isinstance(routing, dict):
        raise RuntimeError(
            "Simple KMeans runtime-profile dataset/routing proof is missing"
        )
    if canonical_json_sha256(dataset) != source.get("dataset_sha256"):
        raise RuntimeError(
            "Simple KMeans runtime-profile dataset changed from its source"
        )
    if canonical_json_sha256(routing) != source.get("routing_sha256"):
        raise RuntimeError(
            "Simple KMeans runtime-profile routing summary changed from its source"
        )

    current_expected = {
        "layout_sha256": artifact_payload.get("layout_sha256"),
        "logical_point_count": artifact_payload.get("logical_point_count"),
        "physical_point_count": artifact_payload.get("physical_point_count"),
        "shard_count": shard_count,
    }
    source_binding_mismatches = {
        key: {"expected": expected, "actual": source.get(key)}
        for key, expected in current_expected.items()
        if source.get(key) != expected
    }
    if source_binding_mismatches:
        raise RuntimeError(
            "Simple KMeans runtime-profile source layout binding mismatch: "
            f"{source_binding_mismatches}"
        )
    if source.get("assignments_sha256") != artifact_payload.get("layout_sha256"):
        raise RuntimeError(
            "Simple KMeans runtime-profile source assignments do not match "
            "layout_sha256"
        )
    offline_artifact = {
        key: value
        for key, value in artifact_payload.items()
        if key not in SIMPLE_KMEANS_RUNTIME_PARAMETER_KEYS
    }
    if canonical_json_sha256(offline_artifact) != source.get(
        "offline_artifact_sha256"
    ):
        raise RuntimeError(
            "Simple KMeans runtime-profile centroid/offline artifact changed "
            "from its source"
        )
    return source


def load_routed_layout(
    method: str,
    layout_path: str | Path,
    *,
    allow_orion_scaling_layout: bool = False,
    allow_orion_balance_layout: bool = False,
    allow_orion_l1_partition_layout: bool = False,
) -> dict[str, Any]:
    layout_dir = Path(layout_path).expanduser().resolve()
    if not layout_dir.is_dir():
        raise FileNotFoundError(f"layout directory not found: {layout_dir}")
    checksums = verify_checksum_listing(layout_dir)
    build_manifest_path = layout_dir / layout_common.BUILD_MANIFEST_NAME
    if not build_manifest_path.is_file():
        raise FileNotFoundError(f"layout build manifest not found: {build_manifest_path}")
    build_manifest = json.loads(build_manifest_path.read_text(encoding="utf-8"))
    if not isinstance(build_manifest, dict):
        raise ValueError("layout build manifest root must be a JSON object")
    expected_tools = (
        ORION_LAYOUT_TOOLS
        if method == "orion"
        else SIMPLE_KMEANS_LAYOUT_TOOLS
    )
    actual_tool = build_manifest.get("tool")
    if actual_tool not in expected_tools:
        raise RuntimeError(
            f"layout tool mismatch: expected one of={sorted(expected_tools)!r}, "
            f"actual={actual_tool!r}"
        )
    is_orion_l1_partition_layout = actual_tool in ORION_L1_PARTITION_LAYOUT_TOOLS
    is_orion_l1_runtime_profile = (
        actual_tool == ORION_RUNTIME_PROFILE_TOOL
        and bool(
            (build_manifest.get("derivation") or {}).get(
                "orion_l1_partition_layout_allowed", False
            )
        )
    )
    uses_orion_l1_partition_layout = (
        is_orion_l1_partition_layout or is_orion_l1_runtime_profile
    )
    if uses_orion_l1_partition_layout and not allow_orion_l1_partition_layout:
        raise RuntimeError(
            "Orion L1 partition layout requires "
            "--allow-orion-l1-partition-layout"
        )
    if allow_orion_l1_partition_layout and not uses_orion_l1_partition_layout:
        raise RuntimeError(
            "--allow-orion-l1-partition-layout only authorizes build tools "
            f"{sorted(ORION_L1_PARTITION_LAYOUT_TOOLS)}"
        )
    if uses_orion_l1_partition_layout and allow_orion_balance_layout:
        raise RuntimeError(
            "Orion L1 partition layouts are not CCNB balance layouts; use only "
            "--allow-orion-l1-partition-layout"
        )
    if build_manifest.get("mode") != "production_bundle":
        raise RuntimeError("layout must be a completed production_bundle build")
    build_parameters = build_manifest.get("parameters") or {}
    if not isinstance(build_parameters, dict):
        raise ValueError("layout build manifest parameters must be a JSON object")

    outputs = build_manifest.get("outputs") or {}
    artifact_path = safe_child(
        layout_dir,
        outputs.get("production_artifact"),
        "production_artifact",
    )
    import_manifest_path = safe_child(
        layout_dir,
        outputs.get("import_manifest"),
        "import_manifest",
    )
    if not artifact_path.is_file() or not import_manifest_path.is_file():
        raise FileNotFoundError("layout production artifact or import manifest is missing")
    artifact_relative = artifact_path.relative_to(layout_dir).as_posix()
    import_manifest_relative = import_manifest_path.relative_to(layout_dir).as_posix()
    if artifact_relative not in checksums:
        raise RuntimeError("production artifact is not covered by layout checksums")
    if import_manifest_relative not in checksums:
        raise RuntimeError("import manifest is not covered by layout checksums")
    declared_files = outputs.get("files") or {}
    for relative in (artifact_relative, import_manifest_relative):
        declared = declared_files.get(relative)
        if not isinstance(declared, dict):
            raise RuntimeError(f"layout build manifest does not declare {relative}")
        declared_sha256 = cluster_tool.normalize_sha256(
            str(declared.get("sha256") or "")
        )
        if declared_sha256 != checksums[relative]:
            raise RuntimeError(f"layout build manifest checksum mismatch for {relative}")
    artifact_sha256 = checksums[artifact_relative]
    artifact_payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    generation = artifact_payload.get("generation")
    if isinstance(generation, bool) or not isinstance(generation, int) or generation <= 0:
        raise ValueError("routing artifact generation must be positive")
    validator = (
        cluster_tool.validate_local_orion_artifact
        if method == "orion"
        else cluster_tool.validate_local_simple_kmeans_artifact
    )
    artifact = validator(artifact_path, generation, artifact_sha256)
    attachment_search_ef = None
    runtime_profile_source: dict[str, Any] | None = None
    balance_layout_proof: dict[str, Any] | None = None
    l1_partition_layout_proof: dict[str, Any] | None = None
    if method == "orion":
        if is_orion_l1_partition_layout:
            if actual_tool == ORION_L1_PARTITION_LAYOUT_TOOL:
                l1_partition_layout_proof = validate_orion_l1_partition_layout(
                    build_manifest,
                    build_parameters,
                    artifact_payload,
                    layout_dir,
                    checksums,
                )
            elif actual_tool == ORION_NATIVE_CNBR_LAYOUT_TOOL:
                l1_partition_layout_proof = validate_orion_native_cnbr_layout(
                    build_manifest,
                    build_parameters,
                    artifact_payload,
                    layout_dir,
                    checksums,
                )
            elif actual_tool == ORION_SAMPLED_CNBR_LAYOUT_TOOL:
                l1_partition_layout_proof = validate_orion_sampled_cnbr_layout(
                    build_manifest,
                    build_parameters,
                    artifact_payload,
                    layout_dir,
                    checksums,
                )
            elif actual_tool == ORION_BUDGETED_MULTI_ASSIGNMENT_LAYOUT_TOOL:
                l1_partition_layout_proof = (
                    validate_orion_budgeted_multi_assignment_layout(
                        build_manifest,
                        build_parameters,
                        artifact_payload,
                        layout_dir,
                        checksums,
                    )
                )
            elif actual_tool == ORION_BMR10_SCALING_LAYOUT_TOOL:
                l1_partition_layout_proof = validate_orion_bmr10_scaling_layout(
                    build_manifest,
                    build_parameters,
                    artifact_payload,
                    layout_dir,
                    checksums,
                )
            elif actual_tool == ORION_ARTIFACT_OWNER_BMR10_LAYOUT_TOOL:
                l1_partition_layout_proof = (
                    validate_orion_artifact_owner_bmr10_layout(
                        build_manifest,
                        build_parameters,
                        artifact_payload,
                        layout_dir,
                        checksums,
                    )
                )
            elif actual_tool == ORION_OWNER_POLICY_TOURNAMENT_LAYOUT_TOOL:
                l1_partition_layout_proof = (
                    validate_orion_owner_policy_tournament_layout(
                        build_manifest,
                        build_parameters,
                        artifact_payload,
                        layout_dir,
                        checksums,
                    )
                )
            else:  # Defensive: authorization and validation must stay exact-tool bound.
                raise RuntimeError(
                    f"no Orion L1 validator for build tool {actual_tool!r}"
                )
            attachment_search_ef = 100
        elif is_orion_l1_runtime_profile:
            attachment_search_ef = 100
            runtime_profile_source = validate_orion_runtime_profile_derivation(
                build_manifest,
                build_parameters,
                artifact_payload,
                allow_l1_partition_layout=True,
            )
            l1_partition_layout_proof = runtime_profile_source.get(
                "_l1_partition_layout_proof"
            )
        else:
            attachment_search_ef = validate_faithful_orion_build_parameters(
                build_parameters,
                artifact_payload,
                allow_scaling_initial_num_shards=allow_orion_scaling_layout,
                allow_balance_layout=allow_orion_balance_layout,
            )
            if allow_orion_balance_layout:
                balance_layout_proof = validate_orion_balance_layout(
                    build_manifest,
                    build_parameters,
                    artifact_payload,
                )
            if actual_tool == ORION_RUNTIME_PROFILE_TOOL:
                runtime_profile_source = validate_orion_runtime_profile_derivation(
                    build_manifest,
                    build_parameters,
                    artifact_payload,
                    allow_scaling_initial_num_shards=allow_orion_scaling_layout,
                    allow_balance_layout=allow_orion_balance_layout,
                    allow_l1_partition_layout=False,
                )
    elif actual_tool == SIMPLE_KMEANS_RUNTIME_PROFILE_TOOL:
        runtime_profile_source = (
            validate_simple_kmeans_runtime_profile_derivation(
                build_manifest,
                build_parameters,
                artifact_payload,
            )
        )

    import_manifest = json.loads(import_manifest_path.read_text(encoding="utf-8"))
    if not isinstance(import_manifest, dict):
        raise ValueError("import manifest root must be a JSON object")
    if method == "orion":
        if import_manifest.get("format_version") != 1:
            raise RuntimeError("Orion layout requires import manifest format_version=1")
        manifest_generation = import_manifest.get("orion_generation")
        manifest_artifact_file = import_manifest.get("orion_artifact_file")
        manifest_artifact_sha256 = import_manifest.get("orion_artifact_sha256")
    else:
        if import_manifest.get("format_version") != 2:
            raise RuntimeError("Simple KMeans layout requires import manifest format_version=2")
        if import_manifest.get("routing_policy") != "simple_kmeans":
            raise RuntimeError("Simple KMeans import manifest routing_policy mismatch")
        manifest_generation = import_manifest.get("routing_generation")
        manifest_artifact_file = import_manifest.get("routing_artifact_file")
        manifest_artifact_sha256 = import_manifest.get("routing_artifact_sha256")
    if manifest_generation != generation:
        raise RuntimeError("import manifest routing generation mismatch")
    if manifest_artifact_file != artifact_path.name:
        raise RuntimeError("import manifest routing artifact filename mismatch")
    if cluster_tool.normalize_sha256(str(manifest_artifact_sha256 or "")) != artifact_sha256:
        raise RuntimeError("import manifest routing artifact checksum mismatch")
    expected_fields = {
        "dimension": artifact["vector_schema"].get("dimension"),
        "point_count": artifact["logical_point_count"],
        "shard_count": artifact["shard_count"],
        "total_point_copies": artifact["physical_point_count"],
        "vector_name": artifact["vector_schema"].get("vector_name"),
        "assignments_sha256": artifact["layout_sha256"],
    }
    mismatches = {
        field: {"expected": value, "actual": import_manifest.get(field)}
        for field, value in expected_fields.items()
        if import_manifest.get(field) != value
    }
    if mismatches:
        raise RuntimeError(f"import manifest/artifact mismatch: {mismatches}")
    bundle_paths: dict[str, Path] = {}
    for field in ("vectors_file", "assignments_file"):
        file_path = safe_child(layout_dir, import_manifest.get(field), field)
        if not file_path.is_file():
            raise FileNotFoundError(f"import bundle file is missing: {file_path}")
        digest_field = field.replace("_file", "_sha256")
        if layout_common.sha256_path(file_path) != cluster_tool.normalize_sha256(
            str(import_manifest.get(digest_field) or "")
        ):
            raise RuntimeError(f"import bundle {field} checksum mismatch")
        bundle_paths[field] = file_path
    if runtime_profile_source is not None:
        profile_label = "Orion" if method == "orion" else "Simple KMeans"
        source_import_mismatches = {
            field: {
                "expected": runtime_profile_source.get(field),
                "actual": import_manifest.get(field),
            }
            for field in ("vectors_sha256", "assignments_sha256")
            if runtime_profile_source.get(field) != import_manifest.get(field)
        }
        if source_import_mismatches:
            raise RuntimeError(
                f"{profile_label} runtime-profile reused import payload "
                "binding mismatch: "
                f"{source_import_mismatches}"
            )
        reused_payloads = (build_manifest.get("derivation") or {}).get(
            "reused_payloads"
        )
        if not isinstance(reused_payloads, dict):
            raise RuntimeError(
                f"{profile_label} runtime profile lacks reused payload proof"
            )
        for proof_name, manifest_prefix in (
            ("vectors", "vectors"),
            ("assignments", "assignments"),
        ):
            proof = reused_payloads.get(proof_name)
            if not isinstance(proof, dict):
                raise RuntimeError(
                    f"{profile_label} runtime profile lacks {proof_name} reuse proof"
                )
            destination_path = bundle_paths[f"{manifest_prefix}_file"]
            expected_proof = {
                "destination_file": import_manifest.get(f"{manifest_prefix}_file"),
                "sha256": import_manifest.get(f"{manifest_prefix}_sha256"),
                "size_bytes": destination_path.stat().st_size,
            }
            if (
                not isinstance(proof.get("source_file"), str)
                or not proof.get("source_file")
                or Path(proof["source_file"]).name != proof["source_file"]
            ):
                expected_proof["source_file"] = "one source file name"
            proof_mismatches = {
                key: {"expected": expected, "actual": proof.get(key)}
                for key, expected in expected_proof.items()
                if proof.get(key) != expected
            }
            if isinstance(proof.get("size_bytes"), bool) or not isinstance(
                proof.get("size_bytes"), int
            ):
                proof_mismatches["size_bytes"] = {
                    "expected": destination_path.stat().st_size,
                    "actual": proof.get("size_bytes"),
                }
            if proof.get("materialization") not in {"hardlink", "copy"}:
                proof_mismatches["materialization"] = {
                    "expected": "hardlink or copy",
                    "actual": proof.get("materialization"),
                }
            expected_same_inode = proof.get("materialization") == "hardlink"
            if proof.get("same_inode") is not expected_same_inode:
                proof_mismatches["same_inode"] = {
                    "expected": expected_same_inode,
                    "actual": proof.get("same_inode"),
                }
            if proof_mismatches:
                raise RuntimeError(
                    f"{profile_label} runtime-profile {proof_name} reuse proof "
                    "mismatch: "
                    f"{proof_mismatches}"
                )
        formal_evidence_eligible = all(
            reused_payloads[name].get("materialization") == "copy"
            for name in ("vectors", "assignments")
        )
        if (
            (build_manifest.get("derivation") or {}).get(
                "formal_evidence_eligible"
            )
            is not formal_evidence_eligible
        ):
            raise RuntimeError(
                f"{profile_label} runtime-profile formal evidence eligibility "
                "mismatch"
            )
    dimension = int(artifact["vector_schema"].get("dimension") or 0)
    with bundle_paths["vectors_file"].open("rb") as handle:
        first_row = handle.read(dimension * 4)
    if len(first_row) != dimension * 4:
        raise RuntimeError("import bundle vector file does not contain one complete row")
    smoke_vector = experiment.np.frombuffer(first_row, dtype="<f4").astype(
        experiment.np.float32,
        copy=True,
    )
    if not experiment.np.isfinite(smoke_vector).all():
        raise RuntimeError("import bundle first vector contains a non-finite value")
    routing = build_manifest.get("routing")
    shard_weights = None
    if isinstance(routing, dict) and "shard_counts" in routing:
        raw_shard_weights = routing.get("shard_counts")
        if not isinstance(raw_shard_weights, list):
            raise ValueError("layout routing shard_counts must be a list")
        if len(raw_shard_weights) != artifact["shard_count"]:
            raise RuntimeError(
                "layout routing shard_counts length does not match artifact shard_count"
            )
        if any(
            isinstance(weight, bool) or not isinstance(weight, int) or weight <= 0
            for weight in raw_shard_weights
        ):
            raise ValueError("layout routing shard_counts must contain positive integers")
        if sum(raw_shard_weights) != artifact["physical_point_count"]:
            raise RuntimeError(
                "layout routing shard_counts do not sum to physical_point_count"
            )
        shard_weights = list(raw_shard_weights)

    return {
        "layout_dir": str(layout_dir),
        "build_manifest_path": str(build_manifest_path),
        "build_manifest_sha256": layout_common.sha256_path(build_manifest_path),
        "artifact_path": str(artifact_path),
        "artifact_sha256": artifact_sha256,
        "artifact": artifact,
        "import_manifest_path": str(import_manifest_path),
        "import_manifest_sha256": layout_common.sha256_path(import_manifest_path),
        "checksums": checksums,
        "build_parameters": build_parameters,
        "attachment_search_ef": attachment_search_ef,
        "balance_layout_proof": balance_layout_proof,
        "l1_partition_layout_proof": l1_partition_layout_proof,
        "generation": generation,
        "vector_schema": artifact["vector_schema"],
        "shard_count": artifact["shard_count"],
        "logical_point_count": artifact["logical_point_count"],
        "physical_point_count": artifact["physical_point_count"],
        "shard_weights": shard_weights,
        "smoke_vector": smoke_vector.tolist(),
    }


def optional_collection_info(base_url: str, collection: str) -> dict[str, Any] | None:
    try:
        return experiment.collection_info(base_url, collection)
    except RuntimeError as exc:
        if "(HTTP 404)" in str(exc):
            return None
        raise


def optimizer_ok(value: Any) -> bool:
    return value == "ok" or (isinstance(value, dict) and value.get("ok") is True)


def collection_readiness_proof(
    info: dict[str, Any], expected_points: int
) -> dict[str, Any]:
    indexed_vectors_count = int(info.get("indexed_vectors_count") or 0)
    return {
        "status": info.get("status"),
        "optimizer_status": info.get("optimizer_status"),
        "points_count": int(info.get("points_count") or 0),
        "expected_points_count": int(expected_points),
        "indexed_vectors_count": indexed_vectors_count,
        "fully_indexed": indexed_vectors_count >= expected_points,
        "completion_mode": (
            "fully_indexed"
            if indexed_vectors_count >= expected_points
            else "stable_small_segment_full_scan_exception"
        ),
        "segments_count": int(info.get("segments_count") or 0),
    }


def build_provenance_metadata(
    *,
    method: str,
    schema: dict[str, Any],
    shard_count: int,
    logical_point_count: int,
    physical_point_count: int,
    dataset_proof: dict[str, Any] | None,
    layout: dict[str, Any] | None,
) -> dict[str, Any]:
    provenance: dict[str, Any] = {
        "method": method,
        "vector_schema": {
            "vector_name": str(schema["vector_name"]),
            "dimension": int(schema["dimension"]),
            "distance": str(schema["distance"]),
            "datatype": str(schema["datatype"]),
        },
        "shard_count": int(shard_count),
        "logical_point_count": int(logical_point_count),
        "physical_point_count": int(physical_point_count),
    }
    if method == "hash_all":
        if dataset_proof is None:
            raise ValueError("HashAll provenance requires dataset proof")
        provenance["dataset"] = {
            "sha256": cluster_tool.normalize_sha256(
                str(dataset_proof.get("sha256") or "")
            ),
            "train_shape": [int(logical_point_count), int(schema["dimension"])],
            "train_count": int(logical_point_count),
        }
    else:
        if layout is None:
            raise ValueError("routed provenance requires layout proof")
        provenance["routing"] = {
            "layout_sha256": cluster_tool.normalize_sha256(
                str((layout.get("artifact") or {}).get("layout_sha256") or "")
            ),
            "artifact_sha256": cluster_tool.normalize_sha256(
                str(layout.get("artifact_sha256") or "")
            ),
            "generation": int(layout["generation"]),
        }
        if method == "orion":
            attachment_search_ef = layout.get("attachment_search_ef")
            if attachment_search_ef != 100:
                raise ValueError(
                    "routed Orion provenance requires attachment_search_ef=100"
                )
            provenance["routing"]["attachment_search_ef"] = 100
    return {
        PROVENANCE_METADATA_KEY: {
            "schema_version": PROVENANCE_SCHEMA_VERSION,
            "provenance": provenance,
            "provenance_sha256": experiment.canonical_json_sha256(provenance),
        }
    }


def validate_collection_provenance(
    info: dict[str, Any],
    expected_metadata: dict[str, Any],
) -> dict[str, Any]:
    metadata = (info.get("config") or {}).get("metadata")
    if not isinstance(metadata, dict):
        raise RuntimeError("refusing to reuse collection: provenance metadata is missing")
    actual = metadata.get(PROVENANCE_METADATA_KEY)
    expected = expected_metadata[PROVENANCE_METADATA_KEY]
    if actual != expected:
        raise RuntimeError(
            "refusing to reuse collection: provenance metadata mismatch: "
            f"actual={actual!r}, expected={expected!r}"
        )
    return actual


def _result_rows(response: dict[str, Any], api: str) -> list[dict[str, Any]]:
    result = response.get("result")
    if api == "query" and isinstance(result, dict):
        result = result.get("points")
    if not isinstance(result, list) or not result:
        raise RuntimeError(f"standard {api} smoke returned no points")
    if not all(isinstance(point, dict) and "id" in point for point in result):
        raise RuntimeError(f"standard {api} smoke returned malformed points")
    return result


def run_standard_api_smoke(
    base_url: str,
    collection: str,
    vector: list[float],
    *,
    vector_name: str,
    limit: int,
    timeout: float,
) -> dict[str, Any]:
    if not vector or not all(experiment.np.isfinite(value) for value in vector):
        raise ValueError("smoke vector must be non-empty and finite")
    encoded_collection = urllib.parse.quote(collection, safe="")
    search_vector: Any = (
        {"name": vector_name, "vector": vector} if vector_name else vector
    )
    search_body = {
        "vector": search_vector,
        "limit": limit,
        "with_payload": False,
        "with_vector": False,
    }
    query_body: dict[str, Any] = {
        "query": vector,
        "limit": limit,
        "with_payload": False,
        "with_vector": False,
    }
    if vector_name:
        query_body["using"] = vector_name
    calls = [
        (
            "search",
            f"/collections/{encoded_collection}/points/search",
            search_body,
        ),
        (
            "query",
            f"/collections/{encoded_collection}/points/query",
            query_body,
        ),
    ]
    proofs: dict[str, Any] = {}
    for api, path, body in calls:
        response = experiment.request_json(
            base_url,
            "POST",
            path,
            body=body,
            timeout=timeout,
        )
        rows = _result_rows(response, api)
        ids = [point["id"] for point in rows]
        encoded_ids = [
            json.dumps(point_id, sort_keys=True, separators=(",", ":"))
            for point_id in ids
        ]
        if len(encoded_ids) != len(set(encoded_ids)):
            raise RuntimeError(f"standard {api} smoke returned duplicate external IDs")
        proofs[api] = {
            "path": path,
            "request": body,
            "result_count": len(rows),
            "external_ids": ids,
            "external_ids_unique": True,
        }
    vector_bytes = experiment.np.asarray(vector, dtype="<f4").tobytes()
    return {
        "standard_request_contract": True,
        "forbidden_request_fields": [
            "shard_key",
            "shard_id",
            "hnsw_entry_points",
            "hnsw_entry_points_by_shard",
            "hnsw_ef_by_shard",
            "source_id_dedup_block_size",
        ],
        "query_vector_sha256": hashlib.sha256(vector_bytes).hexdigest(),
        "query_vector_dimension": len(vector),
        **proofs,
    }


def validate_safe_numeric_placement(
    cluster_info: dict[str, Any],
    cluster_preflight: dict[str, Any],
    expected_shard_count: int,
) -> dict[str, Any]:
    """Accept a recoverable RF=1 placement before converging it to round-robin."""
    placement = experiment.numeric_shard_placement_from_cluster(
        cluster_info,
        expected_shard_count=expected_shard_count,
    )
    allowed_peers = {
        int(cluster_preflight["controller_peer_id"]),
        *[int(peer_id) for peer_id in cluster_preflight["worker_peer_ids"]],
    }
    unexpected = sorted(set(placement.values()) - allowed_peers)
    if unexpected:
        raise RuntimeError(
            f"existing numeric shards are placed on unknown peers: {unexpected}"
        )
    return {
        "valid": True,
        "placement": placement,
        "allowed_peers": sorted(allowed_peers),
        "needs_round_robin_convergence": True,
    }


def validate_collection_configuration(
    info: dict[str, Any],
    *,
    method: str,
    expected_schema: dict[str, Any],
    expected_shard_count: int,
    expected_policy: dict[str, Any] | None,
    expected_metadata: dict[str, Any],
    hnsw_m: int,
    ef_construct: int,
    max_indexing_threads: int = 0,
    full_scan_threshold: int,
    indexing_threshold: int,
    max_optimization_threads: int | None = None,
    max_segment_size_kb: int | None = None,
    expected_point_count: int,
    allow_empty: bool,
    allow_partial: bool,
) -> dict[str, Any]:
    config = info.get("config") or {}
    params = config.get("params") or {}
    errors: list[str] = []
    if str(params.get("sharding_method") or "auto").lower() != "auto":
        errors.append("sharding_method is not auto")
    expected_params = {
        "shard_number": expected_shard_count,
        "replication_factor": 1,
        "write_consistency_factor": 1,
    }
    for field, expected in expected_params.items():
        if params.get(field) != expected:
            errors.append(f"{field}={params.get(field)!r}, expected={expected!r}")
    live_schema = benchmark.collection_vector_schema(
        info,
        str(expected_schema.get("vector_name") or ""),
    )
    for field in ("vector_name", "dimension", "distance", "datatype"):
        actual = live_schema[field]
        expected = expected_schema[field]
        if field in {"distance", "datatype"}:
            actual = str(actual).lower()
            expected = str(expected).lower()
        if actual != expected:
            errors.append(f"vector_schema.{field}={actual!r}, expected={expected!r}")
    hnsw = config.get("hnsw_config") or {}
    if hnsw.get("m") != hnsw_m:
        errors.append(f"hnsw.m={hnsw.get('m')!r}, expected={hnsw_m}")
    if hnsw.get("ef_construct") != ef_construct:
        errors.append(
            f"hnsw.ef_construct={hnsw.get('ef_construct')!r}, expected={ef_construct}"
        )
    live_max_indexing_threads = hnsw.get("max_indexing_threads")
    max_indexing_matches = live_max_indexing_threads == max_indexing_threads or (
        max_indexing_threads == 0 and live_max_indexing_threads is None
    )
    if not max_indexing_matches:
        errors.append(
            "hnsw.max_indexing_threads="
            f"{live_max_indexing_threads!r}, expected={max_indexing_threads}"
        )
    if hnsw.get("full_scan_threshold") != full_scan_threshold:
        errors.append(
            "hnsw.full_scan_threshold="
            f"{hnsw.get('full_scan_threshold')!r}, expected={full_scan_threshold}"
        )
    optimizer = config.get("optimizer_config") or config.get("optimizers_config") or {}
    if optimizer.get("indexing_threshold") != indexing_threshold:
        errors.append(
            "optimizer.indexing_threshold="
            f"{optimizer.get('indexing_threshold')!r}, expected={indexing_threshold}"
        )
    if (
        max_optimization_threads is not None
        and optimizer.get("max_optimization_threads") != max_optimization_threads
    ):
        errors.append(
            "optimizer.max_optimization_threads="
            f"{optimizer.get('max_optimization_threads')!r}, "
            f"expected={max_optimization_threads}"
        )
    live_max_segment_size_kb = optimizer.get(
        "max_segment_size_kb", optimizer.get("max_segment_size")
    )
    if (
        max_segment_size_kb is not None
        and live_max_segment_size_kb != max_segment_size_kb
    ):
        errors.append(
            "optimizer.max_segment_size_kb="
            f"{live_max_segment_size_kb!r}, expected={max_segment_size_kb}"
        )
    if str(info.get("status") or "").lower() != "green":
        errors.append(f"status={info.get('status')!r}, expected='green'")
    if not optimizer_ok(info.get("optimizer_status")):
        errors.append(f"optimizer_status is not ok: {info.get('optimizer_status')!r}")
    live_policy = benchmark.live_policy_for_method(info, method)
    if live_policy != expected_policy:
        errors.append(f"auto_shard_policy={live_policy!r}, expected={expected_policy!r}")
    try:
        provenance = validate_collection_provenance(info, expected_metadata)
    except RuntimeError as exc:
        errors.append(str(exc))
        provenance = None
    points_count = info.get("points_count")
    allowed_counts = {expected_point_count}
    if allow_empty:
        allowed_counts.add(0)
    if (
        allow_partial
        and isinstance(points_count, int)
        and 0 <= points_count <= expected_point_count
    ):
        allowed_counts.add(points_count)
    if isinstance(points_count, bool) or not isinstance(points_count, int):
        errors.append(f"points_count is invalid: {points_count!r}")
    elif points_count not in allowed_counts:
        errors.append(
            f"points_count={points_count}, expected one of {sorted(allowed_counts)}"
        )
    if errors:
        raise RuntimeError("refusing to reuse collection: " + "; ".join(errors))
    return {
        "schema": live_schema,
        "policy": live_policy,
        "points_count": points_count,
        "status": info.get("status"),
        "optimizer_status": info.get("optimizer_status"),
        "provenance": provenance,
    }


def run_command(command: list[str], *, env: dict[str, str] | None = None) -> dict[str, Any]:
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
        env=env,
    )
    return {
        "command": command,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def importer_command(
    args: argparse.Namespace,
    topology: dict[str, Any],
    manifest: Path,
) -> list[str]:
    importer_arguments = [
        "--manifest",
        str(manifest),
        "--uri",
        cluster_tool.controller_uri(topology),
        "--http-url",
        args.base_url.rstrip("/"),
        "--collection",
        args.collection,
        "--batch-size",
        str(args.batch_size),
        "--request-timeout-secs",
        str(args.request_timeout_secs),
        "--wait",
        "visible",
        "--ordering",
        "medium",
        *(["--resume"] if args.resume else []),
    ]
    if args.importer_binary:
        return [
            str(Path(args.importer_binary).expanduser().resolve()),
            *importer_arguments,
        ]
    return [
        str(Path(args.cargo_runner).expanduser().resolve()),
        "run",
        "--release",
        "--example",
        "orion_numeric_shard_import",
        "--",
        *importer_arguments,
    ]


def installer_command(
    args: argparse.Namespace,
    layout: dict[str, Any],
) -> list[str]:
    subcommand = (
        "install-orion-artifact"
        if args.method == "orion"
        else "install-simple-kmeans-artifact"
    )
    return [
        sys.executable,
        str(REPO_ROOT / "tools/method4_distributed_cluster.py"),
        "--topology",
        str(Path(args.topology).expanduser().resolve()),
        "--run-id",
        args.run_id,
        subcommand,
        "--collection",
        args.collection,
        "--generation",
        str(layout["generation"]),
        "--artifact",
        layout["artifact_path"],
        "--expected-sha256",
        layout["artifact_sha256"],
        "--restart",
        "workers-first",
    ]


def prepare(args: argparse.Namespace) -> Path:
    validate_args(args)
    output_dir = create_output_directory(args.output_dir)
    topology_path = Path(args.topology).expanduser().resolve()
    topology = cluster_tool.load_topology(topology_path)
    run_manifest = cluster_tool.read_manifest(topology, args.run_id)
    if not run_manifest:
        raise RuntimeError(f"deployment manifest does not exist for run {args.run_id!r}")
    cluster_preflight = experiment.validate_cluster_preflight(
        args.base_url,
        experiment.load_cluster_topology(topology_path),
    )
    include_controller_in_placement = args.placement_peers == "all_peers"
    placement_peer_ids = [int(peer_id) for peer_id in cluster_preflight["worker_peer_ids"]]
    if include_controller_in_placement:
        placement_peer_ids.insert(0, int(cluster_preflight["controller_peer_id"]))

    layout: dict[str, Any] | None = None
    train = None
    dataset_proof: dict[str, Any] | None = None
    if args.method in ROUTED_METHODS:
        layout = load_routed_layout(
            args.method,
            args.layout_dir,
            allow_orion_scaling_layout=args.allow_orion_scaling_layout,
            allow_orion_balance_layout=args.allow_orion_balance_layout,
            allow_orion_l1_partition_layout=(
                args.allow_orion_l1_partition_layout
            ),
        )
        schema = dict(layout["vector_schema"])
        shard_count = int(layout["shard_count"])
        logical_count = int(layout["logical_point_count"])
        physical_count = int(layout["physical_point_count"])
        if args.num_shards is not None and args.num_shards != shard_count:
            raise RuntimeError(
                f"--num-shards {args.num_shards} does not match layout {shard_count}"
            )
        if args.vector_name and args.vector_name != schema.get("vector_name"):
            raise RuntimeError("--vector-name does not match routed layout schema")
        policy = {
            "type": args.method,
            "generation": int(layout["generation"]),
            "artifact_sha256": layout["artifact_sha256"],
        }
        smoke_vector = [float(value) for value in layout["smoke_vector"]]
    else:
        distance = experiment.vector_distance_config(args.vector_distance)
        train, dataset_proof = layout_common.load_train_vectors(
            Path(args.hdf5_path).expanduser().resolve(),
            None,
            distance["name"],
        )
        schema = {
            "vector_name": args.vector_name,
            "dimension": int(train.shape[1]),
            "distance": distance["qdrant_distance"],
            "datatype": "float32",
        }
        shard_count = int(args.num_shards)
        logical_count = physical_count = int(len(train))
        policy = None
        smoke_vector = train[0].astype(experiment.np.float32, copy=False).tolist()

    provenance_metadata = build_provenance_metadata(
        method=args.method,
        schema=schema,
        shard_count=shard_count,
        logical_point_count=logical_count,
        physical_point_count=physical_count,
        dataset_proof=dataset_proof,
        layout=layout,
    )

    existing = optional_collection_info(args.base_url, args.collection)
    created = existing is None
    create_response = None
    initial_readiness = None
    if created:
        create_response = experiment.create_numeric_auto_shard_collection(
            args.base_url,
            args.collection,
            dim=int(schema["dimension"]),
            num_shards=shard_count,
            m=args.hnsw_m,
            ef_construct=args.ef_construct,
            vector_distance=str(schema["distance"]),
            auto_shard_policy=policy,
            replication_factor=1,
            write_consistency_factor=1,
            full_scan_threshold=args.full_scan_threshold,
            indexing_threshold=args.indexing_threshold,
            max_indexing_threads=args.max_indexing_threads,
            max_optimization_threads=args.max_optimization_threads,
            max_segment_size_kb=args.max_segment_size_kb,
            metadata=provenance_metadata,
        )
        initial_readiness = collection_readiness_proof(
            experiment.wait_collection_indexed(
                args.base_url,
                args.collection,
                0,
            ),
            0,
        )
        info = experiment.collection_info(args.base_url, args.collection)
    else:
        info = existing

    reuse_proof = validate_collection_configuration(
        info,
        method=args.method,
        expected_schema=schema,
        expected_shard_count=shard_count,
        expected_policy=policy,
        expected_metadata=provenance_metadata,
        hnsw_m=args.hnsw_m,
        ef_construct=args.ef_construct,
        max_indexing_threads=args.max_indexing_threads,
        full_scan_threshold=args.full_scan_threshold,
        indexing_threshold=args.indexing_threshold,
        max_optimization_threads=args.max_optimization_threads,
        max_segment_size_kb=args.max_segment_size_kb,
        expected_point_count=physical_count,
        allow_empty=True,
        allow_partial=bool(args.resume and args.method in ROUTED_METHODS),
    )
    initial_placement_proof = None
    if not created:
        cluster_existing = experiment.collection_cluster_info(args.base_url, args.collection)
        if cluster_existing is None:
            raise RuntimeError("existing collection cluster placement is unavailable")
        initial_placement_proof = validate_safe_numeric_placement(
            cluster_existing,
            cluster_preflight,
            shard_count,
        )

    placement_plan: dict[str, Any]
    if args.placement_strategy == "round_robin":
        target_placement = experiment.round_robin_numeric_shard_targets(
            list(range(shard_count)),
            placement_peer_ids,
        )
        placement_plan = {
            "strategy": "round_robin",
            "weight_source": None,
            "target_placement": target_placement,
        }
        placement_proof = experiment.move_numeric_shards_round_robin(
            args.base_url,
            args.collection,
            placement_peer_ids,
            expected_shard_count=shard_count,
            include_controller=include_controller_in_placement,
            transfer_method=args.transfer_method,
            timeout_sec=args.transfer_timeout_secs,
            poll_interval_sec=args.transfer_poll_interval_secs,
        )
    else:
        if layout is None or layout.get("shard_weights") is None:
            raise RuntimeError(
                "layout_size_balanced placement requires routing.shard_counts in the layout "
                "build manifest"
            )
        current_placement = experiment.discover_numeric_shard_placement(
            args.base_url,
            args.collection,
            expected_shard_count=shard_count,
        )
        target_placement = experiment.size_balanced_numeric_shard_targets(
            layout["shard_weights"],
            placement_peer_ids,
            current_placement=current_placement,
        )
        worker_weights = {
            peer_id: sum(
                layout["shard_weights"][shard_id]
                for shard_id, owner in target_placement.items()
                if owner == peer_id
            )
            for peer_id in placement_peer_ids
        }
        placement_plan = {
            "strategy": "layout_size_balanced",
            "weight_source": "layout_build_manifest.routing.shard_counts",
            "shard_weights": layout["shard_weights"],
            "target_placement": target_placement,
            "target_weight_per_worker": worker_weights,
            "target_weight_max_over_mean": max(worker_weights.values())
            / (sum(worker_weights.values()) / len(worker_weights)),
        }
        placement_proof = experiment.move_numeric_shards_explicit(
            args.base_url,
            args.collection,
            placement_peer_ids,
            target_placement,
            expected_shard_count=shard_count,
            include_controller=include_controller_in_placement,
            transfer_method=args.transfer_method,
            timeout_sec=args.transfer_timeout_secs,
            poll_interval_sec=args.transfer_poll_interval_secs,
        )

    commands: list[dict[str, Any]] = []
    checkpoint_preservation: dict[str, Any] | None = None
    initial_points_count = int(info.get("points_count") or 0)
    if args.method == "hash_all":
        if initial_points_count == 0:
            assert train is not None
            upsert_proof = experiment.upsert_numeric_auto_points(
                args.base_url,
                args.collection,
                train,
                vector_name=str(schema["vector_name"]),
                batch_size=args.batch_size,
                timeout=float(args.request_timeout_secs),
            )
        else:
            upsert_proof = {
                "status": "reused_complete",
                "point_count": initial_points_count,
            }
        commands.append({"kind": "public_hash_all_upsert", "proof": upsert_proof})
    else:
        assert layout is not None
        if initial_points_count != physical_count or args.resume:
            environment = os.environ.copy()
            if args.cargo_target_dir:
                environment["CARGO_TARGET_DIR"] = str(
                    Path(args.cargo_target_dir).expanduser().resolve()
                )
            import_manifest_path = Path(layout["import_manifest_path"])
            checkpoint = Path(str(import_manifest_path) + ".import-state.json")
            checkpoint_before = output_dir / "import-checkpoint-before.json"
            checkpoint_after = output_dir / "import-checkpoint-after.json"
            had_checkpoint = bool(
                args.preserve_import_checkpoint and checkpoint.is_file()
            )
            if had_checkpoint:
                shutil.copy2(checkpoint, checkpoint_before)
                checkpoint.unlink()
            try:
                commands.append(
                    run_command(
                        importer_command(
                            args,
                            topology,
                            import_manifest_path,
                        ),
                        env=environment,
                    )
                )
            finally:
                generated_checkpoint = checkpoint.is_file()
                if generated_checkpoint and args.preserve_import_checkpoint:
                    shutil.copy2(checkpoint, checkpoint_after)
                    checkpoint.unlink()
                if had_checkpoint:
                    shutil.copy2(checkpoint_before, checkpoint)
                checkpoint_preservation = {
                    "enabled": bool(args.preserve_import_checkpoint),
                    "source_path": str(checkpoint),
                    "preexisting_checkpoint_preserved": had_checkpoint,
                    "generated_checkpoint_archived": bool(
                        generated_checkpoint and args.preserve_import_checkpoint
                    ),
                    "before_archive": str(checkpoint_before)
                    if had_checkpoint
                    else None,
                    "after_archive": str(checkpoint_after)
                    if generated_checkpoint and args.preserve_import_checkpoint
                    else None,
                    "original_restored": had_checkpoint,
                }
        else:
            commands.append(
                {"kind": "numeric_import", "status": "reused_complete"}
            )

    populated = experiment.collection_info(args.base_url, args.collection)
    if populated.get("points_count") != physical_count:
        raise RuntimeError(
            f"collection points_count={populated.get('points_count')!r}, "
            f"expected={physical_count}"
        )
    indexing_readiness = collection_readiness_proof(
        experiment.wait_collection_indexed(
            args.base_url,
            args.collection,
            physical_count,
        ),
        physical_count,
    )
    artifact_installation: dict[str, Any]
    if args.method in ROUTED_METHODS and not args.defer_artifact_install:
        assert layout is not None
        install_result = run_command(installer_command(args, layout))
        commands.append(install_result)
        artifact_installation = {
            "status": "installed_and_activated",
            "deferred": False,
            "command_index": len(commands) - 1,
        }
    elif args.method in ROUTED_METHODS:
        artifact_installation = {
            "status": "deferred",
            "deferred": True,
            "reason": "parallel_prebuild_without_cluster_restart",
            "required_before_query": (
                "install the declared routing artifact on every node and restart "
                "the run-owned cluster containers"
            ),
        }
    else:
        artifact_installation = {
            "status": "not_applicable",
            "deferred": False,
        }

    if args.method in ROUTED_METHODS and args.defer_artifact_install:
        smoke_proof = {
            "status": "DEFERRED",
            "reason": "routing artifact is intentionally not installed or activated",
            "standard_request_contract": True,
            "query_issued": False,
        }
    else:
        smoke_proof = run_standard_api_smoke(
            args.base_url,
            args.collection,
            smoke_vector,
            vector_name=str(schema["vector_name"]),
            limit=min(args.smoke_limit, logical_count),
            timeout=float(args.request_timeout_secs),
        )

    final_info = experiment.collection_info(args.base_url, args.collection)
    final_proof = validate_collection_configuration(
        final_info,
        method=args.method,
        expected_schema=schema,
        expected_shard_count=shard_count,
        expected_policy=policy,
        expected_metadata=provenance_metadata,
        hnsw_m=args.hnsw_m,
        ef_construct=args.ef_construct,
        max_indexing_threads=args.max_indexing_threads,
        full_scan_threshold=args.full_scan_threshold,
        indexing_threshold=args.indexing_threshold,
        max_optimization_threads=args.max_optimization_threads,
        max_segment_size_kb=args.max_segment_size_kb,
        expected_point_count=physical_count,
        allow_empty=False,
        allow_partial=False,
    )
    final_cluster = experiment.collection_cluster_info(args.base_url, args.collection)
    if final_cluster is None:
        raise RuntimeError("final collection cluster placement is unavailable")
    if args.placement_strategy == "round_robin":
        final_placement = experiment.validate_numeric_shard_round_robin_placement(
            final_info,
            final_cluster,
            placement_peer_ids,
            shard_count,
            include_controller=include_controller_in_placement,
        )
    else:
        final_placement = experiment.validate_numeric_shard_explicit_placement(
            final_info,
            final_cluster,
            placement_peer_ids,
            shard_count,
            target_placement,
            include_controller=include_controller_in_placement,
        )

    placement_map_path = output_dir / "placement_map.json"
    layout_common.write_json_new(
        placement_map_path,
        {
            "strategy": args.placement_strategy,
            "target_placement": target_placement,
        },
    )

    manifest = {
        "format_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "method": args.method,
        "run_id": args.run_id,
        "collection": args.collection,
        "base_url": args.base_url.rstrip("/"),
        "topology": {
            "path": str(topology_path),
            "sha256": layout_common.sha256_path(topology_path),
        },
        "deployment_manifest": {
            "path": str(cluster_tool.manifest_path(topology, args.run_id)),
            "sha256": layout_common.sha256_path(
                cluster_tool.manifest_path(topology, args.run_id)
            ),
            "image": run_manifest.get("image"),
        },
        "checksums": {
            "topology_sha256": layout_common.sha256_path(topology_path),
            "deployment_manifest_sha256": layout_common.sha256_path(
                cluster_tool.manifest_path(topology, args.run_id)
            ),
            "dataset_sha256": (dataset_proof or {}).get("sha256"),
            "layout_build_manifest_sha256": (
                layout.get("build_manifest_sha256") if layout else None
            ),
            "routing_artifact_sha256": (
                layout.get("artifact_sha256") if layout else None
            ),
            "import_manifest_sha256": (
                layout.get("import_manifest_sha256") if layout else None
            ),
            "layout_files": layout.get("checksums") if layout else None,
        },
        "cluster_preflight": {
            key: value for key, value in cluster_preflight.items() if key != "raw"
        },
        "schema": schema,
        "hnsw": {
            "m": args.hnsw_m,
            "ef_construct": args.ef_construct,
            "max_indexing_threads": args.max_indexing_threads,
            "full_scan_threshold": args.full_scan_threshold,
            "indexing_threshold": args.indexing_threshold,
            "max_optimization_threads": args.max_optimization_threads,
            "max_segment_size_kb": args.max_segment_size_kb,
        },
        "replication_factor": 1,
        "write_consistency_factor": 1,
        "shard_count": shard_count,
        "logical_point_count": logical_count,
        "physical_point_count": physical_count,
        "created_collection": created,
        "create_response": create_response,
        "initial_readiness": initial_readiness,
        "indexing_readiness": indexing_readiness,
        "initial_collection_proof": reuse_proof,
        "initial_placement_proof": initial_placement_proof,
        "layout": layout,
        "dataset": dataset_proof,
        "commands": commands,
        "import_checkpoint_preservation": checkpoint_preservation,
        "artifact_installation": artifact_installation,
        "standard_api_smoke": smoke_proof,
        "provenance_metadata": provenance_metadata,
        "placement_plan": placement_plan,
        "placement_peers": {
            "mode": args.placement_peers,
            "peer_ids": placement_peer_ids,
            "includes_controller": include_controller_in_placement,
        },
        "transfer_method": args.transfer_method,
        "orion_scaling_layout_allowed": bool(args.allow_orion_scaling_layout),
        "orion_balance_layout_allowed": bool(args.allow_orion_balance_layout),
        "orion_l1_partition_layout_allowed": bool(
            args.allow_orion_l1_partition_layout
        ),
        "placement_map": {
            "path": str(placement_map_path),
            "sha256": layout_common.sha256_path(placement_map_path),
        },
        "placement": placement_proof,
        "final_collection_proof": final_proof,
        "final_placement_proof": final_placement,
    }
    manifest_path = output_dir / PREPARATION_MANIFEST
    layout_common.write_json_new(manifest_path, manifest)
    print(json.dumps({"preparation_manifest": str(manifest_path)}, indent=2))
    return manifest_path


def main(argv: list[str] | None = None) -> int:
    try:
        prepare(parse_args(argv))
    except (
        ValueError,
        FileNotFoundError,
        FileExistsError,
        RuntimeError,
        TimeoutError,
        subprocess.CalledProcessError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
