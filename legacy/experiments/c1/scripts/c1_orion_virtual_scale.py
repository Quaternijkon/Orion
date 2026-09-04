#!/usr/bin/env python3
"""Measure native Orion against the HashAll linear-CPU protocol.

The independent variable is the nominal resource scale M.  Every formal point
receives exactly ``2*M`` Qdrant CPU cores.  Native Orion fission does not always
produce the nominal shard count, so the configuration records exact layouts
and lower/upper boundary points instead of disabling fission.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import importlib.util
import json
import shutil
import statistics
import subprocess
import sys
import time
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np


def load_sibling(name: str):
    source = Path(__file__).resolve().with_name(name)
    module_name = f"{source.stem}_for_orion_vscale"
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


hashall = load_sibling("c1_hashall_virtual_scale.py")
simple = load_sibling("c1_simple_kmeans_virtual_scale.py")
fixed = hashall.fixed


DEFAULT_ROOT = Path(
    "/proj/intelisys-PG0/exp/orion-distributed/"
    "orion-virtual-linear-cpu-glove-20260825"
)
DEFAULT_CONFIG = DEFAULT_ROOT / "profiles.json"
DEFAULT_HDF5 = Path(
    "/users/dry/orion-distributed/datasets/glove-200-angular.hdf5"
)
DEFAULT_TOPOLOGY = Path(__file__).resolve().parents[1] / "topology-amd-4node.json"
DEFAULT_RUN_ID = "c1-20260821-v2"
NOMINAL_M_VALUES = (1, 2, 4, 8, 16, 32)
PHYSICAL_MACHINE_COUNT = 4


@dataclass(frozen=True)
class LayoutProfile:
    key: str
    actual_shards: int
    layout_dir: Path


@dataclass(frozen=True)
class MeasurementPoint:
    point_id: str
    nominal_m: int
    boundary: str
    layout_key: str


@dataclass(frozen=True)
class ExperimentConfig:
    layouts: dict[str, LayoutProfile]
    points: tuple[MeasurementPoint, ...]


def load_config(path: Path) -> ExperimentConfig:
    payload = simple.load_json(path)
    raw_layouts = payload.get("layouts") or {}
    raw_points = payload.get("points") or []
    layouts: dict[str, LayoutProfile] = {}
    for key, value in raw_layouts.items():
        if not isinstance(value, dict):
            raise ValueError(f"invalid layout {key!r}")
        layout_dir = Path(str(value["layout_dir"])).expanduser().resolve()
        layouts[str(key)] = LayoutProfile(
            key=str(key),
            actual_shards=int(value["actual_shards"]),
            layout_dir=layout_dir,
        )
    points = tuple(
        MeasurementPoint(
            point_id=str(value["point_id"]),
            nominal_m=int(value["nominal_m"]),
            boundary=str(value["boundary"]),
            layout_key=str(value["layout_key"]),
        )
        for value in raw_points
    )
    if not layouts or not points:
        raise ValueError("Orion configuration must contain layouts and points")
    if any(point.layout_key not in layouts for point in points):
        raise ValueError("measurement point references an unknown layout")
    if any(point.nominal_m not in NOMINAL_M_VALUES for point in points):
        raise ValueError(f"nominal M must be in {NOMINAL_M_VALUES}")
    point_ids = [point.point_id for point in points]
    if len(point_ids) != len(set(point_ids)):
        raise ValueError("point IDs must be unique")
    return ExperimentConfig(layouts=layouts, points=points)


def runtime_artifact(profile: LayoutProfile) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    manifest_path = profile.layout_dir / "build-manifest.json"
    manifest = simple.load_json(manifest_path)
    artifact_path = simple.runtime_artifact_path(profile.layout_dir, manifest)
    artifact = simple.load_json(artifact_path)
    return manifest, artifact_path, artifact


def audit_orion_profile(profile: LayoutProfile) -> dict[str, Any]:
    manifest, artifact_path, artifact = runtime_artifact(profile)
    parameters = manifest.get("parameters") or {}
    routing = manifest.get("routing") or {}
    import_manifest = simple.import_manifest_path(profile.layout_dir, manifest)
    expected = {
        "shard_count": profile.actual_shards,
        "logical_point_count": 1_183_514,
    }
    mismatches = {
        key: {"expected": expected_value, "actual": artifact.get(key)}
        for key, expected_value in expected.items()
        if artifact.get(key) != expected_value
    }
    if mismatches:
        raise RuntimeError(f"Orion artifact mismatch for {profile.key}: {mismatches}")
    if int(artifact.get("physical_point_count") or 0) < 1_183_514:
        raise RuntimeError("Orion physical point count cannot be below logical count")
    if int(parameters.get("upper_k") or 0) <= 0:
        raise RuntimeError("Orion upper_k must be positive")
    if int(parameters.get("upper_search_ef") or 0) != int(parameters["upper_k"]):
        raise RuntimeError("faithful Orion requires upper_search_ef=upper_k")
    if int(parameters.get("dynamic_ef_base") or 0) <= 0:
        raise RuntimeError("Orion dynamic_ef_base must be positive")
    if int(parameters.get("dynamic_ef_factor") or 0) < 0:
        raise RuntimeError("Orion dynamic_ef_factor cannot be negative")
    if routing.get("effective_num_shards") != profile.actual_shards:
        raise RuntimeError("Orion routing shard count mismatch")
    if parameters.get("use_multi_assign") is not True:
        raise RuntimeError("formal Orion profile must retain multi-assignment")
    if parameters.get("enable_fission") is not True:
        raise RuntimeError("formal Orion profile must retain fission")
    return {
        "status": "PASS",
        "method": "native_orion",
        "key": profile.key,
        "layout_dir": str(profile.layout_dir),
        "build_manifest": str(profile.layout_dir / "build-manifest.json"),
        "build_manifest_sha256": simple.sha256(
            profile.layout_dir / "build-manifest.json"
        ),
        "artifact": str(artifact_path),
        "artifact_sha256": simple.sha256(artifact_path),
        "import_manifest": str(import_manifest),
        "import_manifest_sha256": simple.sha256(import_manifest),
        "initial_num_shards": parameters.get("initial_num_shards"),
        "actual_shards": profile.actual_shards,
        "upper_k": parameters["upper_k"],
        "upper_search_ef": parameters["upper_search_ef"],
        "dynamic_ef_base": parameters["dynamic_ef_base"],
        "dynamic_ef_factor": parameters["dynamic_ef_factor"],
        "logical_point_count": artifact["logical_point_count"],
        "physical_point_count": artifact["physical_point_count"],
        "expansion_ratio": (
            float(artifact["physical_point_count"])
            / float(artifact["logical_point_count"])
        ),
        "fission_events": routing.get("fission_events") or [],
    }


def collection_name(profile: LayoutProfile) -> str:
    return f"orion_vscale_glove_{profile.key}_20260825"


def output_point_dir(root: Path, point: MeasurementPoint, profile: LayoutProfile) -> Path:
    return root / f"m{point.nominal_m}" / f"{point.boundary}-s{profile.actual_shards}"


def expected_collection_names(config: ExperimentConfig) -> set[str]:
    return {collection_name(profile) for profile in config.layouts.values()}


def validate_preflight_collections(
    collections: Sequence[dict[str, Any]],
    config: ExperimentConfig,
    *,
    reuse_existing: bool,
) -> list[str]:
    names = sorted(str(item.get("name") or "") for item in collections)
    if any(not name for name in names):
        raise RuntimeError(f"invalid collection listing: {collections}")
    if not names:
        return names
    if not reuse_existing:
        raise RuntimeError(
            f"Orion experiment requires an empty collection set: {collections}"
        )
    unexpected = set(names) - expected_collection_names(config)
    if unexpected:
        raise RuntimeError(
            "reuse mode found collections outside the selected Orion profiles: "
            f"{sorted(unexpected)}"
        )
    return names


def preflight(
    args: argparse.Namespace,
    exp,
    config: ExperimentConfig,
    audits: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    containers = hashall.inspect_all_containers()
    for snapshot in containers:
        if (
            snapshot["image"] != hashall.EXPECTED_IMAGE
            or snapshot["image_id"] != hashall.EXPECTED_IMAGE_ID
        ):
            raise RuntimeError(f"unexpected image identity: {snapshot}")
    cluster = hashall.wait_cluster_ready(exp, args.base_url)
    collections = exp.request_json(args.base_url, "GET", "/collections")["result"][
        "collections"
    ]
    existing_collection_names = validate_preflight_collections(
        collections,
        config,
        reuse_existing=args.reuse_existing,
    )
    with hashall.h5py.File(args.hdf5_path, "r") as handle:
        shapes = {key: list(handle[key].shape) for key in ("train", "test", "neighbors")}
        distance = str(handle.attrs.get("distance", ""))
    if shapes["train"] != [1_183_514, 200] or shapes["test"][:1] != [10_000]:
        raise RuntimeError(f"unexpected GloVe shapes: {shapes}")
    if distance != "angular":
        raise RuntimeError(f"unexpected HDF5 distance: {distance!r}")
    return {
        "timestamp": hashall.utc_timestamp(),
        "record_type": "orion_virtual_linear_cpu_preflight",
        "status": "PASS",
        "cluster": cluster,
        "collections": collections,
        "existing_collection_names": existing_collection_names,
        "reuse_existing": bool(args.reuse_existing),
        "containers": containers,
        "dataset": {
            "name": "GloVe-200-angular",
            "path": str(args.hdf5_path),
            "shapes": shapes,
            "hdf5_distance": distance,
            "qdrant_distance": "Cosine",
        },
        "nominal_m_values": list(NOMINAL_M_VALUES),
        "resource_rule": "total Qdrant CPU quota = 2 * nominal M",
        "fission_policy": "enabled; non-exact scales are reported as boundaries",
        "profiles": audits,
        "points": [point.__dict__ for point in config.points],
    }


def run_prepare(
    args: argparse.Namespace,
    profile: LayoutProfile,
    output_dir: Path,
) -> dict[str, Any]:
    manifest, _artifact_path, _artifact = runtime_artifact(profile)
    import_manifest = simple.import_manifest_path(profile.layout_dir, manifest)
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
        str(Path(__file__).resolve().parents[3] / "tools/native_auto_shard_prepare.py"),
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
        "--allow-orion-scaling-layout",
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
        str(hashall.MAX_SEGMENT_SIZE_KB),
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
        hashall.write_json(output_dir.parent / "native-prepare-failure.json", failure)
        raise RuntimeError(
            "native Orion prepare failed: "
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
        "preparation_manifest_sha256": simple.sha256(manifest_path),
        "preparation_manifest": simple.load_json(manifest_path),
    }


def artifact_installer_command(
    args: argparse.Namespace, profile: LayoutProfile, *, restart: bool
) -> list[str]:
    _manifest, artifact_path, artifact = runtime_artifact(profile)
    command = [
        sys.executable,
        str(Path(__file__).resolve().parents[3] / "tools/method4_distributed_cluster.py"),
        "--topology",
        str(args.topology),
        "--run-id",
        args.run_id,
        "install-orion-artifact",
        "--collection",
        collection_name(profile),
        "--generation",
        str(artifact["generation"]),
        "--artifact",
        str(artifact_path),
        "--expected-sha256",
        simple.sha256(artifact_path),
    ]
    if restart:
        command.extend(["--restart", "workers-first"])
    return command


def install_profile_artifact(
    args: argparse.Namespace, profile: LayoutProfile, *, restart: bool
) -> dict[str, Any]:
    command = artifact_installer_command(args, profile, restart=restart)
    started = time.monotonic()
    completed = subprocess.run(command, check=False, text=True, capture_output=True)
    record = {
        "timestamp": hashall.utc_timestamp(),
        "record_type": "orion_artifact_activation" if restart else "orion_artifact_install",
        "status": "PASS" if completed.returncode == 0 else "FAIL",
        "profile": profile.key,
        "actual_shards": profile.actual_shards,
        "restart_requested": restart,
        "seconds": time.monotonic() - started,
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }
    if completed.returncode != 0:
        raise RuntimeError(
            f"Orion artifact activation failed for {profile.key}: "
            + (completed.stderr.strip() or completed.stdout.strip())
        )
    return record


def verify_live_policy(exp, args: argparse.Namespace, profile: LayoutProfile) -> dict[str, Any]:
    info = exp.collection_info(args.base_url, collection_name(profile))
    config = info.get("config") or {}
    params = config.get("params") or {}
    policy = config.get("auto_shard_policy") or params.get("auto_shard_policy") or {}
    _manifest, artifact_path, artifact = runtime_artifact(profile)
    expected = {
        "type": "orion",
        "generation": artifact["generation"],
        "artifact_sha256": simple.sha256(artifact_path),
    }
    if policy != expected:
        raise RuntimeError(f"live Orion policy mismatch: {policy} != {expected}")
    if int(params.get("shard_number") or 0) != profile.actual_shards:
        raise RuntimeError("live Orion shard count mismatch")
    hnsw = config.get("hnsw_config") or {}
    if hnsw.get("m") != 32 or hnsw.get("ef_construct") != 200:
        raise RuntimeError(f"live lower HNSW contract mismatch: {hnsw}")
    return {
        "status": "PASS",
        "policy": policy,
        "shard_number": params["shard_number"],
        "points_count": info.get("points_count"),
        "indexed_vectors_count": info.get("indexed_vectors_count"),
        "segments_count": info.get("segments_count"),
        "collection_info": info,
    }


def quota_schedule(nominal_m: int, shards_per_node: Sequence[int]) -> list[float]:
    counts = [int(value) for value in shards_per_node]
    if len(counts) != PHYSICAL_MACHINE_COUNT or any(value < 0 for value in counts):
        raise ValueError(f"invalid Orion placement counts: {counts}")
    actual_shards = sum(counts)
    if actual_shards <= 0:
        raise ValueError("Orion placement has no shards")
    total = 2.0 * nominal_m
    floor = hashall.CONTROL_PLANE_FLOOR_CORES
    distributable = total - floor * PHYSICAL_MACHINE_COUNT
    if distributable <= 0:
        raise ValueError("nominal CPU budget is below the control-plane floor")
    quotas = [floor + distributable * count / actual_shards for count in counts]
    rounded = [round(value, 6) for value in quotas]
    residual = round(total - sum(rounded), 6)
    if residual:
        target = max(range(len(counts)), key=lambda index: counts[index])
        rounded[target] = round(rounded[target] + residual, 6)
    if not np.isclose(sum(rounded), total, rtol=0.0, atol=1e-9):
        raise AssertionError("Orion quota schedule does not preserve CPU=2M")
    if nominal_m == 32 and not all(
        np.isclose(value, 16.0, rtol=0.0, atol=1e-9) for value in rounded
    ):
        raise AssertionError("M=32 must expose all 64 Qdrant CPU cores")
    return rounded


def apply_measurement_resources(
    args: argparse.Namespace,
    exp,
    *,
    point: MeasurementPoint,
    profile: LayoutProfile,
    output: Path,
) -> dict[str, Any]:
    placement = simple.live_numeric_shard_placement(
        exp, collection_name(profile), profile.actual_shards
    )
    quotas = quota_schedule(point.nominal_m, placement["shards_per_node"])
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(
                hashall.update_container_resources,
                node,
                cpuset=hashall.QDRANT_CPUSET,
                cpus=quota,
            )
            for node, quota in zip(hashall.NODES, quotas, strict=True)
        ]
        for future in futures:
            future.result()
    snapshots = hashall.inspect_all_containers()
    for snapshot, expected in zip(snapshots, quotas, strict=True):
        actual = snapshot["nano_cpus"] / 1_000_000_000.0
        if snapshot["cpuset"] != hashall.QDRANT_CPUSET:
            raise RuntimeError(f"cpuset mismatch: {snapshot}")
        if not np.isclose(actual, expected, rtol=0.0, atol=1e-6):
            raise RuntimeError(f"CPU quota mismatch: {actual} != {expected}")
    record = {
        "timestamp": hashall.utc_timestamp(),
        "record_type": "orion_virtual_linear_cpu_resource_schedule",
        "status": "PASS",
        "mode": "measurement_budget",
        "point_id": point.point_id,
        "nominal_m": point.nominal_m,
        "boundary": point.boundary,
        "actual_shards": profile.actual_shards,
        "physical_machines": PHYSICAL_MACHINE_COUNT,
        "qdrant_cpuset": hashall.QDRANT_CPUSET,
        "benchmark_cpuset": hashall.BENCHMARK_CPUSET,
        "control_plane_floor_cores_per_peer": hashall.CONTROL_PLANE_FLOOR_CORES,
        "shards_per_node": placement["shards_per_node"],
        "quota_cores_per_node": quotas,
        "quota_cores_total": round(sum(quotas), 6),
        "expected_total": 2.0 * point.nominal_m,
        "live_placement_proof": placement,
        "containers": snapshots,
    }
    hashall.write_json(output, record)
    return record


def run_benchmark(
    args: argparse.Namespace,
    exp,
    *,
    point: MeasurementPoint,
    profile: LayoutProfile,
    profile_audit: dict[str, Any],
    resource_record: dict[str, Any],
    queries: np.ndarray,
    neighbors: np.ndarray,
) -> dict[str, Any]:
    parsed = urllib.parse.urlparse(args.base_url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 6333
    collection = collection_name(profile)
    endpoint = f"/collections/{urllib.parse.quote(collection, safe='')}/points/search/batch"
    tuning_queries, tuning_neighbors = queries[:1000], neighbors[:1000]
    heldout_queries, heldout_neighbors = queries[1000:10_000], neighbors[1000:10_000]
    tuning = simple.recall_probe(
        host,
        port,
        endpoint,
        tuning_queries,
        tuning_neighbors,
        batch_size=args.batch_size,
        top_k=args.top_k,
    )
    heldout = simple.recall_probe(
        host,
        port,
        endpoint,
        heldout_queries,
        heldout_neighbors,
        batch_size=args.batch_size,
        top_k=args.top_k,
    )
    heldout_recall = float(heldout["recall_at_10"])
    recall_gate = {
        "status": (
            "PASS"
            if args.target_recall <= heldout_recall < args.max_recall
            else "FAIL"
        ),
        "lower_inclusive": args.target_recall,
        "upper_exclusive": args.max_recall,
        "observed": heldout_recall,
    }
    if recall_gate["status"] != "PASS" and not args.recall_only:
        raise RuntimeError(
            f"{point.point_id} held-out recall is outside "
            f"[{args.target_recall}, {args.max_recall}): {heldout}"
        )
    bodies = simple.make_simple_bodies(
        heldout_queries, batch_size=args.batch_size, top_k=args.top_k
    )
    common = {
        "timestamp": hashall.utc_timestamp(),
        "record_type": "native_orion_virtual_linear_cpu_fixed_recall",
        "status": "RECALL_ONLY" if args.recall_only else "PASS",
        "point_id": point.point_id,
        "nominal_m": point.nominal_m,
        "boundary": point.boundary,
        "actual_shards": profile.actual_shards,
        "physical_machines": PHYSICAL_MACHINE_COUNT,
        "base_url": args.base_url,
        "collection": collection,
        "request_contract": {
            "standard_coordinator_request": True,
            "endpoint": endpoint,
            "shard_selector_present": False,
            "client_side_fanout": False,
            "client_hnsw_ef_present": False,
            "server_router": "native_orion",
        },
        "profile_contract": profile_audit,
        "resource_contract": resource_record,
        "protocol": {
            "dataset": "GloVe-200-angular",
            "dataset_key": "glove-200-angular",
            "vector_size": 200,
            "distance": "Cosine",
            "tuning_query_range": [0, 1000],
            "heldout_query_range": [1000, 10000],
            "target_recall_at_10": args.target_recall,
            "profile_selection": (
                "fixed native Orion profile calibrated on the same dataset family; "
                "revalidated on disjoint 9000 held-out queries"
            ),
            "fission_enabled": True,
        },
        "parameters": {
            "upper_k": profile_audit["upper_k"],
            "upper_search_ef": profile_audit["upper_search_ef"],
            "dynamic_ef_base": profile_audit["dynamic_ef_base"],
            "dynamic_ef_factor": profile_audit["dynamic_ef_factor"],
            "top_k": args.top_k,
            "batch_size": args.batch_size,
            "sweep_seconds": args.sweep_seconds,
            "warmup_seconds": args.warmup_seconds,
            "measure_seconds": args.measure_seconds,
        },
        "live_policy_gate": verify_live_policy(exp, args, profile),
        "per_node_collection_cluster": hashall.per_node_collection_cluster(exp, collection),
        "tuning_recall": tuning,
        "heldout_recall": heldout,
        "recall_gate": recall_gate,
    }
    if args.recall_only:
        common["parameters"]["repeat_count"] = 0
        return common
    sweep = [
        hashall.timed_run(
            host,
            port,
            endpoint,
            bodies,
            concurrency=concurrency,
            duration_s=args.sweep_seconds,
            batch_size=args.batch_size,
        )
        for concurrency in args.concurrency_candidates
    ]
    selection = hashall.select_saturation(sweep)
    if not selection["knee_observed"]:
        raise RuntimeError("saturation knee not observed; extend concurrency grid")
    selected_concurrency = int(selection["selected_concurrency"])
    warmup = hashall.timed_run(
        host,
        port,
        endpoint,
        bodies,
        concurrency=selected_concurrency,
        duration_s=args.warmup_seconds,
        batch_size=args.batch_size,
    )
    repeats: list[dict[str, Any]] = []
    while len(repeats) < args.min_repeats:
        repeats.append(
            hashall.timed_run(
                host,
                port,
                endpoint,
                bodies,
                concurrency=selected_concurrency,
                duration_s=args.measure_seconds,
                batch_size=args.batch_size,
            )
        )
    while len(repeats) < args.max_repeats:
        values = [float(row["qps"]) for row in repeats]
        cv = statistics.stdev(values) / statistics.fmean(values)
        if cv <= args.max_cv:
            break
        repeats.append(
            hashall.timed_run(
                host,
                port,
                endpoint,
                bodies,
                concurrency=selected_concurrency,
                duration_s=args.measure_seconds,
                batch_size=args.batch_size,
            )
        )
    values = [float(row["qps"]) for row in repeats]
    qps_mean = statistics.fmean(values)
    qps_stdev = statistics.stdev(values)
    qps_cv = qps_stdev / qps_mean
    if qps_cv > args.max_cv:
        raise RuntimeError(f"formal QPS CV remains too high: {qps_cv:.4f}")
    common["parameters"].update(
        {"selected_concurrency": selected_concurrency, "repeat_count": len(repeats)}
    )
    common.update(
        {
            "concurrency_sweep": sweep,
            "saturation_selection": selection,
            "warmup": warmup,
            "repeats": repeats,
            "qps_mean": qps_mean,
            "qps_stdev": qps_stdev,
            "qps_cv": qps_cv,
        }
    )
    return common


def execute(args: argparse.Namespace) -> int:
    repo_root = Path(__file__).resolve().parents[3]
    exp = hashall.load_experiment_module(repo_root)
    config = load_config(args.config)
    selected_ids = set(args.point_ids)
    selected_points = tuple(
        point for point in config.points if not selected_ids or point.point_id in selected_ids
    )
    if selected_ids - {point.point_id for point in config.points}:
        raise ValueError(f"unknown point IDs: {sorted(selected_ids)}")
    selected_layout_keys = {point.layout_key for point in selected_points}
    selected_layouts = {
        key: profile for key, profile in config.layouts.items() if key in selected_layout_keys
    }
    audits = {key: audit_orion_profile(profile) for key, profile in selected_layouts.items()}
    args.output_root.mkdir(parents=True, exist_ok=True)
    profile_audit_path = args.output_root / "profile-audit.json"
    merged_audits = (
        simple.load_json(profile_audit_path) if profile_audit_path.is_file() else {}
    )
    merged_audits.update(audits)
    hashall.write_json(profile_audit_path, merged_audits)
    phase_id = "__".join(point.point_id for point in selected_points)
    phase_root = args.output_root / "phases" / phase_id
    phase_root.mkdir(parents=True, exist_ok=True)
    original_path = args.output_root / "original-resource-state.json"
    if args.restore_only:
        original = simple.load_json(original_path)["containers"]
        hashall.restore_resource_state(original, args.output_root / "resource-restored.json")
        return 0
    original = hashall.inspect_all_containers()
    if not original_path.exists():
        hashall.write_json(
            original_path,
            {
                "timestamp": hashall.utc_timestamp(),
                "record_type": "orion_virtual_linear_cpu_original_resources",
                "containers": original,
            },
        )
    preflight_record = preflight(
        args,
        exp,
        ExperimentConfig(config.layouts, selected_points),
        audits,
    )
    hashall.write_json(args.output_root / "preflight.json", preflight_record)
    hashall.write_json(phase_root / "preflight.json", preflight_record)
    if args.preflight_only:
        return 0
    queries, neighbors = fixed.load_dataset(args.hdf5_path, args.top_k)
    completed: list[str] = []
    body_error: BaseException | None = None
    try:
        for layout_key, profile in selected_layouts.items():
            layout_root = args.output_root / "layouts" / layout_key
            layout_root.mkdir(parents=True, exist_ok=True)
            hashall.apply_resource_schedule(
                None, output=layout_root / "resources-build.json"
            )
            hashall.wait_cluster_ready(exp, args.base_url)
            prepare = run_prepare(args, profile, layout_root / "native-prepare")
            prepare["profile_contract"] = audits[layout_key]
            prepare["live_policy_gate"] = verify_live_policy(exp, args, profile)
            hashall.write_json(layout_root / "prepare.json", prepare)
            activation = install_profile_artifact(args, profile, restart=True)
            activation["live_policy_gate"] = verify_live_policy(exp, args, profile)
            hashall.write_json(layout_root / "activation.json", activation)

            for point in [item for item in selected_points if item.layout_key == layout_key]:
                point_root = output_point_dir(args.output_root, point, profile)
                point_root.mkdir(parents=True, exist_ok=True)
                resources = apply_measurement_resources(
                    args,
                    exp,
                    point=point,
                    profile=profile,
                    output=point_root / "resources-measure.json",
                )
                hashall.wait_cluster_ready(exp, args.base_url)
                benchmark = run_benchmark(
                    args,
                    exp,
                    point=point,
                    profile=profile,
                    profile_audit=audits[layout_key],
                    resource_record=resources,
                    queries=queries,
                    neighbors=neighbors,
                )
                filename = "recall-probe.json" if args.recall_only else "benchmark.json"
                hashall.write_json(point_root / filename, benchmark)
                completed.append(point.point_id)

            hashall.apply_resource_schedule(
                None, output=layout_root / "resources-cleanup.json"
            )
            if not args.keep_collection:
                exp.delete_collection_if_exists(args.base_url, collection_name(profile))
            remaining = exp.request_json(args.base_url, "GET", "/collections")["result"][
                "collections"
            ]
            remaining_names = sorted(
                str(item.get("name") or "") for item in remaining
            )
            allowed_remaining = expected_collection_names(config)
            expected_selected_present = args.keep_collection
            selected_present = collection_name(profile) in remaining_names
            if (
                selected_present is not expected_selected_present
                or not set(remaining_names).issubset(allowed_remaining)
            ):
                raise RuntimeError(f"collection cleanup failed: {remaining}")
            hashall.write_json(
                layout_root / "cleanup.json",
                {
                    "timestamp": hashall.utc_timestamp(),
                    "record_type": "orion_virtual_linear_cpu_cleanup",
                    "profile": layout_key,
                    "deleted_collection": (
                        None if args.keep_collection else collection_name(profile)
                    ),
                    "preserved_collection": (
                        collection_name(profile) if args.keep_collection else None
                    ),
                    "remaining_collections": remaining,
                    "status": "PASS",
                },
            )
        execution_path = args.output_root / "execution-complete.json"
        previous_completed: list[str] = []
        if execution_path.is_file():
            previous_completed = list(
                simple.load_json(execution_path).get("completed_points") or []
            )
        merged_completed = [
            point.point_id
            for point in config.points
            if point.point_id in set(previous_completed + completed)
        ]
        all_complete = len(merged_completed) == len(config.points)
        execution_record = {
                "timestamp": hashall.utc_timestamp(),
                "record_type": "orion_virtual_linear_cpu_execution",
                "status": (
                    "RECALL_PROBES_COMPLETE"
                    if args.recall_only and all_complete
                    else "RECALL_PROBE_PHASE_COMPLETE"
                    if args.recall_only
                    else "MEASUREMENTS_COMPLETE"
                    if all_complete
                    else "MEASUREMENT_PHASE_COMPLETE"
                ),
                "dataset": "GloVe-200-angular",
                "completed_points": merged_completed,
                "phase_completed_points": completed,
                "physical_machines": PHYSICAL_MACHINE_COUNT,
                "resource_rule": "CPU total = 2 * nominal M",
                "formal_measurement_concurrency": 1,
                "parallel_prebuild_reuse": bool(args.reuse_existing),
            }
        hashall.write_json(execution_path, execution_record)
        hashall.write_json(phase_root / "execution-complete.json", execution_record)
        return 0
    except BaseException as error:
        body_error = error
        failure_record = {
                "timestamp": hashall.utc_timestamp(),
                "record_type": "orion_virtual_linear_cpu_execution",
                "status": "FAILED",
                "completed_points": completed,
                "error": repr(error),
            }
        hashall.write_json(args.output_root / "execution-failed.json", failure_record)
        hashall.write_json(phase_root / "execution-failed.json", failure_record)
        raise
    finally:
        if not args.keep_collection:
            for profile in selected_layouts.values():
                try:
                    exp.delete_collection_if_exists(args.base_url, collection_name(profile))
                except Exception:
                    pass
        saved_original = simple.load_json(original_path)["containers"]
        try:
            hashall.restore_resource_state(
                saved_original, args.output_root / "resource-restored.json"
            )
        except Exception:
            if body_error is None:
                raise
            print("[orion-vscale] resource restoration also failed", flush=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://10.10.1.1:6333")
    parser.add_argument("--hdf5-path", type=Path, default=DEFAULT_HDF5)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--topology", type=Path, default=DEFAULT_TOPOLOGY)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--point-id", dest="point_ids", action="append", default=[])
    parser.add_argument("--upload-batch-size", type=int, default=2000)
    parser.add_argument("--index-timeout", type=float, default=10_800.0)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--target-recall", type=float, default=0.90)
    parser.add_argument("--max-recall", type=float, default=0.93)
    parser.add_argument("--concurrency-candidates", default="1,2,4,8,16,32,64")
    parser.add_argument("--sweep-seconds", type=float, default=8.0)
    parser.add_argument("--warmup-seconds", type=float, default=10.0)
    parser.add_argument("--measure-seconds", type=float, default=20.0)
    parser.add_argument("--min-repeats", type=int, default=5)
    parser.add_argument("--max-repeats", type=int, default=7)
    parser.add_argument("--max-cv", type=float, default=0.05)
    parser.add_argument(
        "--cargo-runner",
        type=Path,
        default=Path(__file__).resolve().parents[3] / "tools/cargo_in_docker.sh",
    )
    parser.add_argument(
        "--cargo-target-dir",
        type=Path,
        default=Path("/proj/intelisys-PG0/exp/orion-distributed/cargo-target-native"),
    )
    parser.add_argument("--importer-binary", type=Path)
    parser.add_argument("--recall-only", action="store_true")
    parser.add_argument(
        "--keep-collection",
        action="store_true",
        help=(
            "Preserve an already prebuilt collection after a recall-only probe so "
            "the same indexed bytes can be reused by the later serial QPS phase."
        ),
    )
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help=(
            "Reuse already populated selected-profile collections. This is intended "
            "for isolated parallel prebuilds; routing activation and all recall/QPS "
            "measurements remain serial."
        ),
    )
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--restore-only", action="store_true")
    args = parser.parse_args(argv)
    args.hdf5_path = args.hdf5_path.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    args.config = args.config.expanduser().resolve()
    args.topology = args.topology.expanduser().resolve()
    args.cargo_runner = args.cargo_runner.expanduser().resolve()
    args.cargo_target_dir = args.cargo_target_dir.expanduser().resolve()
    if args.importer_binary is not None:
        args.importer_binary = args.importer_binary.expanduser().resolve()
    args.concurrency_candidates = hashall.parse_int_csv(args.concurrency_candidates)
    if args.concurrency_candidates != sorted(set(args.concurrency_candidates)):
        raise ValueError("concurrency candidates must be sorted and unique")
    if args.min_repeats < 2 or args.max_repeats < args.min_repeats:
        raise ValueError("invalid repeat bounds")
    if not 0.0 <= args.target_recall < args.max_recall <= 1.0:
        raise ValueError("recall window must satisfy 0 <= target < max <= 1")
    if args.keep_collection and not (args.recall_only and args.reuse_existing):
        raise ValueError(
            "--keep-collection requires both --recall-only and --reuse-existing"
        )
    return args


def main(argv: Sequence[str] | None = None) -> int:
    return execute(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
