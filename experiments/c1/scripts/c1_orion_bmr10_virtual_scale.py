#!/usr/bin/env python3
"""Measure fixed-P C_CNBR+BMR_10 under the HashAll linear-CPU protocol."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SOURCE = Path(__file__).resolve().with_name("c1_orion_virtual_scale.py")
SPEC = importlib.util.spec_from_file_location("c1_orion_virtual_scale_bmr10_base", SOURCE)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load {SOURCE}")
base = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = base
SPEC.loader.exec_module(base)

from tools import native_auto_shard_prepare as prepare  # noqa: E402


DEFAULT_ROOT = Path(
    "/proj/intelisys-PG0/exp/orion-distributed/"
    "orion-bmr10-virtual-linear-cpu-glove-20260828"
)
DEFAULT_CONFIG = REPO_ROOT / "experiments/c1/bmr10-scaling-profiles-20260828.json"
EXPECTED_TOOL = "experiments/c1/scripts/c1_orion_bmr10_scaling_materialize.py"
RUNTIME_PROFILE_TOOL = "tools/orion_native_runtime_profile.py"


def collection_name(profile: base.LayoutProfile) -> str:
    return f"orion_bmr10_vscale_glove_p{profile.actual_shards}_20260828"


def audit_orion_profile(profile: base.LayoutProfile) -> dict[str, Any]:
    manifest, artifact_path, artifact = base.runtime_artifact(profile)
    proof = prepare.load_routed_layout(
        "orion",
        profile.layout_dir,
        allow_orion_l1_partition_layout=True,
    )
    parameters = manifest.get("parameters") or {}
    routing = manifest.get("routing") or {}
    l1_proof = proof.get("l1_partition_layout_proof") or {}
    tool = manifest.get("tool")
    if tool not in {EXPECTED_TOOL, RUNTIME_PROFILE_TOOL}:
        raise RuntimeError(f"unsupported BMR_10 scaling profile tool: {tool!r}")
    expected = {
        "artifact_shards": profile.actual_shards,
        "routing_shards": profile.actual_shards,
        "proof_shards": profile.actual_shards,
        "l1_partitioner": "C_CNBR",
        "balance_mode": "cnbr",
        "multi_assignment_policy": "BMR_10",
        "multi_assignment_policy_version": 1,
        "enable_fission": False,
        "enable_topology_refinement": False,
    }
    actual = {
        "artifact_shards": artifact.get("shard_count"),
        "routing_shards": routing.get("effective_num_shards"),
        "proof_shards": l1_proof.get("shard_count"),
        "l1_partitioner": parameters.get("l1_partitioner"),
        "balance_mode": parameters.get("balance_mode"),
        "multi_assignment_policy": parameters.get("multi_assignment_policy"),
        "multi_assignment_policy_version": parameters.get(
            "multi_assignment_policy_version"
        ),
        "enable_fission": parameters.get("enable_fission"),
        "enable_topology_refinement": parameters.get("enable_topology_refinement"),
    }
    mismatches = {
        key: {"expected": value, "actual": actual.get(key)}
        for key, value in expected.items()
        if actual.get(key) != value or type(actual.get(key)) is not type(value)
    }
    if mismatches:
        raise RuntimeError(f"BMR_10 scaling profile mismatch: {mismatches}")
    derivation = manifest.get("derivation") or {}
    if tool == RUNTIME_PROFILE_TOOL and (
        derivation.get("kind") != "orion_runtime_profile"
        or derivation.get("formal_evidence_eligible") is not True
        or derivation.get("orion_l1_partition_layout_allowed") is not True
        or (derivation.get("source") or {}).get("assignments_sha256")
        != artifact.get("layout_sha256")
    ):
        raise RuntimeError("BMR_10 runtime profile derivation contract failed")
    if int(parameters.get("upper_search_ef") or 0) != int(
        parameters.get("upper_k") or 0
    ):
        raise RuntimeError("BMR_10 scaling requires upper_search_ef=upper_k")
    assignment = l1_proof.get("assignment") or {}
    logical = int(artifact["logical_point_count"])
    physical = int(artifact["physical_point_count"])
    if (
        logical != 1_183_514
        or physical < logical
        or physical - logical > logical // 10
        or assignment.get("maximum_copies_per_point") != 2
        or assignment.get("copy_budget_mode") != "upper_bound"
    ):
        raise RuntimeError("BMR_10 scaling copy-budget contract failed")
    return {
        "status": "PASS",
        "method": "native_orion_c_cnbr_bmr10",
        "profile_tool": tool,
        "key": profile.key,
        "layout_dir": str(profile.layout_dir),
        "build_manifest": str(profile.layout_dir / "build-manifest.json"),
        "build_manifest_sha256": base.simple.sha256(
            profile.layout_dir / "build-manifest.json"
        ),
        "artifact": str(artifact_path),
        "artifact_sha256": base.simple.sha256(artifact_path),
        "generation": int(artifact["generation"]),
        "import_manifest": str(
            base.simple.import_manifest_path(profile.layout_dir, manifest)
        ),
        "import_manifest_sha256": base.simple.sha256(
            base.simple.import_manifest_path(profile.layout_dir, manifest)
        ),
        "initial_num_shards": profile.actual_shards,
        "actual_shards": profile.actual_shards,
        "upper_k": parameters["upper_k"],
        "upper_search_ef": parameters["upper_search_ef"],
        "dynamic_ef_base": parameters["dynamic_ef_base"],
        "dynamic_ef_factor": parameters["dynamic_ef_factor"],
        "logical_point_count": logical,
        "physical_point_count": physical,
        "expansion_ratio": physical / logical,
        "fission_events": [],
        "l1_partition_layout_proof": l1_proof,
        "fixed_p_exact": True,
        "runtime_profile_derivation": derivation if tool == RUNTIME_PROFILE_TOOL else None,
    }


def preflight(
    args: argparse.Namespace,
    exp: Any,
    config: base.ExperimentConfig,
    audits: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    record = base.preflight_original(args, exp, config, audits)
    existing_names = set(record.get("existing_collection_names") or [])
    live_prebuilt: dict[str, dict[str, Any]] = {}
    for key, profile_audit in audits.items():
        profile = config.layouts[key]
        name = collection_name(profile)
        if name not in existing_names:
            continue
        info = exp.collection_info(args.base_url, name)
        live_config = info.get("config") or {}
        live_params = live_config.get("params") or {}
        live_policy = (
            live_config.get("auto_shard_policy")
            or live_params.get("auto_shard_policy")
            or {}
        )
        hnsw = live_config.get("hnsw_config") or {}
        optimizer = live_config.get("optimizer_config") or {}
        expected_physical = int(profile_audit["physical_point_count"])
        if (
            info.get("status") != "green"
            or int(info.get("points_count") or 0) != expected_physical
            or int(info.get("indexed_vectors_count") or 0) < 1_183_514
            or hnsw.get("m") != 32
            or hnsw.get("ef_construct") != 200
            or optimizer.get("max_segment_size") != base.hashall.MAX_SEGMENT_SIZE_KB
            or live_policy
            != {
                "type": "orion",
                "generation": int(profile_audit["generation"]),
                "artifact_sha256": profile_audit["artifact_sha256"],
            }
        ):
            raise RuntimeError(f"prebuilt collection contract mismatch: {name}")
        placement = base.simple.live_numeric_shard_placement(
            exp, name, profile.actual_shards
        )
        shard_counts = list(placement["shards_per_node"])
        if max(shard_counts) - min(shard_counts) > 1:
            raise RuntimeError(f"prebuilt collection placement is unbalanced: {name}")
        live_prebuilt[key] = {
            "status": "PASS",
            "collection": name,
            "actual_shards": profile.actual_shards,
            "physical_point_count": expected_physical,
            "indexed_vectors_count": int(info["indexed_vectors_count"]),
            "segments_count": int(info["segments_count"]),
            "hnsw_m": hnsw["m"],
            "ef_construct": hnsw["ef_construct"],
            "max_segment_size_kb": optimizer["max_segment_size"],
            "policy": live_policy,
            "placement": placement,
        }
    record.update(
        {
            "record_type": "orion_bmr10_virtual_linear_cpu_preflight",
            "method": "C_CNBR+BMR_10",
            "logical_shard_rule": "exact fixed P = nominal M",
            "fission_policy": "disabled for every fixed-P point",
            "multi_assignment_policy": "BMR_10 with at most floor(N/10) extra copies",
            "live_prebuilt_collections": live_prebuilt,
        }
    )
    return record


def run_benchmark(
    args: argparse.Namespace,
    exp: Any,
    **kwargs: Any,
) -> dict[str, Any]:
    record = base.run_benchmark_original(args, exp, **kwargs)
    protocol = record["protocol"]
    protocol.update(
        {
            "profile_selection": (
                "fixed-P C_CNBR+BMR_10 profile; accepted for formal QPS only "
                "when held-out Recall@10 is in [0.90,0.93)"
            ),
            "fission_enabled": False,
            "logical_shard_rule": "exact fixed P = nominal M",
            "load_balancing_policy": "C_CNBR",
            "multi_assignment_policy": "BMR_10",
        }
    )
    return record


def run_prepare(
    args: argparse.Namespace,
    profile: base.LayoutProfile,
    output_dir: Path,
) -> dict[str, Any]:
    manifest, _artifact_path, _artifact = base.runtime_artifact(profile)
    import_manifest = base.simple.import_manifest_path(profile.layout_dir, manifest)
    checkpoint = Path(str(import_manifest) + ".import-state.json")
    checkpoint_before = output_dir.parent / "import-checkpoint-before.json"
    checkpoint_after = output_dir.parent / "import-checkpoint-after.json"
    checkpoint_before.unlink(missing_ok=True)
    checkpoint_after.unlink(missing_ok=True)
    had_checkpoint = checkpoint.is_file()
    if had_checkpoint:
        shutil.copy2(checkpoint, checkpoint_before)
        checkpoint.unlink()
    command = [
        sys.executable,
        str(REPO_ROOT / "tools/native_auto_shard_prepare.py"),
        "--method",
        "orion",
        "--topology",
        str(args.topology),
        "--run-id",
        args.run_id,
        "--collection",
        collection_name(profile),
        "--base-url",
        args.base_url,
        "--output-dir",
        str(output_dir),
        "--layout-dir",
        str(profile.layout_dir),
        "--allow-orion-l1-partition-layout",
        "--hnsw-m",
        "32",
        "--ef-construct",
        "200",
        "--max-indexing-threads",
        "1",
        "--full-scan-threshold",
        "10",
        "--indexing-threshold",
        "10",
        "--max-optimization-threads",
        "1",
        "--max-segment-size-kb",
        str(base.hashall.MAX_SEGMENT_SIZE_KB),
        "--batch-size",
        str(args.upload_batch_size),
        "--request-timeout-secs",
        "300",
        "--smoke-limit",
        "10",
        "--transfer-timeout-secs",
        str(args.index_timeout),
        "--transfer-poll-interval-secs",
        "1",
        "--placement-strategy",
        "round_robin",
        "--placement-peers",
        "all_peers",
        "--cargo-runner",
        str(args.cargo_runner),
        "--cargo-target-dir",
        str(args.cargo_target_dir),
        "--defer-artifact-install",
    ]
    if args.reuse_existing:
        command.extend(["--transfer-method", "snapshot"])
    if args.importer_binary is not None:
        command.extend(["--importer-binary", str(args.importer_binary)])
    started = time.monotonic()
    try:
        completed = subprocess.run(command, check=False, text=True, capture_output=True)
    finally:
        if checkpoint.is_file():
            shutil.copy2(checkpoint, checkpoint_after)
            checkpoint.unlink()
        if had_checkpoint:
            shutil.copy2(checkpoint_before, checkpoint)
    if completed.returncode != 0:
        failure = {
            "status": "FAIL",
            "returncode": completed.returncode,
            "command": command,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
        base.hashall.write_json(output_dir.parent / "native-prepare-failure.json", failure)
        raise RuntimeError(
            "native BMR_10 scaling prepare failed: "
            + (completed.stderr.strip() or completed.stdout.strip())
        )
    manifest_path = output_dir / "preparation_manifest.json"
    return {
        "status": "PASS",
        "seconds": time.monotonic() - started,
        "command": command,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "import_checkpoint": {
            "source_path": str(checkpoint),
            "preexisting_checkpoint_preserved": had_checkpoint,
            "before_archive": str(checkpoint_before) if had_checkpoint else None,
            "after_archive": str(checkpoint_after),
            "layout_checkpoint_restored": had_checkpoint,
        },
        "preparation_manifest_path": str(manifest_path),
        "preparation_manifest_sha256": base.simple.sha256(manifest_path),
        "preparation_manifest": base.simple.load_json(manifest_path),
    }


def configure_base() -> None:
    base.DEFAULT_ROOT = DEFAULT_ROOT
    base.DEFAULT_CONFIG = DEFAULT_CONFIG
    base.collection_name = collection_name
    base.audit_orion_profile = audit_orion_profile
    base.run_prepare = run_prepare
    if not hasattr(base, "preflight_original"):
        base.preflight_original = base.preflight
    base.preflight = preflight
    if not hasattr(base, "run_benchmark_original"):
        base.run_benchmark_original = base.run_benchmark
    base.run_benchmark = run_benchmark


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    configure_base()
    return base.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    return base.execute(args)


if __name__ == "__main__":
    raise SystemExit(main())
