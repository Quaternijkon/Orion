#!/usr/bin/env python3
"""Measure the unmodified native Simple KMeans baseline at M=1..32.

The experiment deliberately uses only the plain baseline:

* one nearest-centroid assignment per point (no replication/multi-assignment);
* static nearest-centroid ``nprobe`` routing;
* one fixed ``lower_hnsw_ef`` for every selected shard;
* ordinary Qdrant coordinator Search requests; and
* round-robin physical shard placement.

Serving CPU, query split, batching, saturation search, repetition count, and
resource restoration match ``c1_hashall_virtual_scale.py``.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import importlib.util
import json
import os
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


def load_hashall_module():
    source = Path(__file__).resolve().with_name("c1_hashall_virtual_scale.py")
    spec = importlib.util.spec_from_file_location("c1_hashall_virtual_scale_for_simple", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


hashall = load_hashall_module()
fixed = hashall.fixed


LOGICAL_SHARD_COUNTS = hashall.LOGICAL_SHARD_COUNTS
PHYSICAL_MACHINE_COUNT = hashall.PHYSICAL_MACHINE_COUNT
BENCHMARK_CPUSET = hashall.BENCHMARK_CPUSET
DEFAULT_ROOT = Path(
    "/proj/intelisys-PG0/exp/orion-distributed/"
    "simple-kmeans-virtual-linear-cpu-glove-20260825"
)
DEFAULT_HDF5 = Path(
    "/users/dry/orion-distributed/datasets/glove-200-angular.hdf5"
)
DEFAULT_TOPOLOGY = Path(__file__).resolve().parents[1] / "topology-amd-4node.json"
DEFAULT_RUN_ID = "c1-20260821-v2"


@dataclass(frozen=True)
class RuntimeProfile:
    logical_shards: int
    nprobe: int
    lower_hnsw_ef: int
    layout_dir: Path


PROFILES = {
    1: RuntimeProfile(
        1,
        1,
        384,
        DEFAULT_ROOT / "artifacts/m1/simple-r090-n1-ef384-g101384",
    ),
    2: RuntimeProfile(
        2,
        2,
        256,
        DEFAULT_ROOT / "artifacts/m2/simple-r090-n2-ef256-g102256",
    ),
    4: RuntimeProfile(
        4,
        4,
        272,
        Path(
            "/users/dry/orion-distributed/native-20260818-scale4-32-r080-095-v1/"
            "artifacts/scale4/simple-r090-n4-ef272-g5272"
        ),
    ),
    8: RuntimeProfile(
        8,
        6,
        216,
        Path(
            "/users/dry/orion-distributed/native-20260818-scale4-32-r080-095-v1/"
            "artifacts/scale8/simple-r090-n6-ef216-g86216"
        ),
    ),
    16: RuntimeProfile(
        16,
        10,
        158,
        Path(
            "/users/dry/orion-distributed/native-20260818-scale4-32-r080-095-v1/"
            "artifacts/scale16/simple-r090-n10-ef158-g1610158"
        ),
    ),
    32: RuntimeProfile(
        32,
        16,
        118,
        Path(
            "/users/dry/orion-distributed/native-20260818-scale4-32-r080-095-v1/"
            "artifacts/scale32/simple-r090-n16-ef118-g3216118"
        ),
    ),
}


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def runtime_artifact_path(layout_dir: Path, build_manifest: dict[str, Any]) -> Path:
    outputs = build_manifest.get("outputs") or {}
    relative = outputs.get("production_artifact")
    if not isinstance(relative, str) or Path(relative).name != relative:
        raise RuntimeError(f"invalid production artifact declaration in {layout_dir}")
    path = layout_dir / relative
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def import_manifest_path(layout_dir: Path, build_manifest: dict[str, Any]) -> Path:
    outputs = build_manifest.get("outputs") or {}
    relative = outputs.get("import_manifest")
    if not isinstance(relative, str) or Path(relative).name != relative:
        raise RuntimeError(f"invalid import manifest declaration in {layout_dir}")
    path = layout_dir / relative
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def audit_plain_simple_kmeans(profile: RuntimeProfile) -> dict[str, Any]:
    """Fail if a profile contains any Orion or enhanced-KMeans mechanism."""
    layout_dir = profile.layout_dir.expanduser().resolve()
    manifest_path = layout_dir / "build-manifest.json"
    manifest = load_json(manifest_path)
    artifact_path = runtime_artifact_path(layout_dir, manifest)
    artifact = load_json(artifact_path)
    parameters = manifest.get("parameters") or {}
    routing = manifest.get("routing") or {}

    expected = {
        "shard_count": profile.logical_shards,
        "nprobe": profile.nprobe,
        "lower_hnsw_ef": profile.lower_hnsw_ef,
        "physical_point_count": 1_183_514,
        "logical_point_count": 1_183_514,
        "routing_distance": "squared_l2",
    }
    mismatches = {
        key: {"expected": value, "actual": artifact.get(key)}
        for key, value in expected.items()
        if artifact.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"Simple KMeans artifact mismatch: {mismatches}")
    if "upper_graph" in artifact:
        raise RuntimeError("plain Simple KMeans must not contain an upper graph")
    if routing.get("expansion_ratio") != 1.0:
        raise RuntimeError("plain Simple KMeans must have expansion_ratio=1.0")
    if routing.get("physical_point_count") != routing.get("logical_point_count"):
        raise RuntimeError("plain Simple KMeans must assign every point exactly once")

    fixed_parameters = {
        "num_shards": profile.logical_shards,
        "nprobe": profile.nprobe,
        "lower_hnsw_ef": profile.lower_hnsw_ef,
        "kmeans_train_size": 10_000,
        "kmeans_iters": 10,
        "kmeans_seed": 1,
        "routing_distance": "squared_l2",
    }
    parameter_mismatches = {
        key: {"expected": value, "actual": parameters.get(key)}
        for key, value in fixed_parameters.items()
        if parameters.get(key) != value
    }
    if parameter_mismatches:
        raise RuntimeError(
            f"Simple KMeans build/runtime parameter mismatch: {parameter_mismatches}"
        )
    allowed_runtime_changes = (manifest.get("derivation") or {}).get(
        "allowed_parameter_changes"
    )
    if allowed_runtime_changes is not None and allowed_runtime_changes != [
        "generation",
        "nprobe",
        "lower_hnsw_ef",
    ]:
        raise RuntimeError("runtime profile changes fields beyond generation/nprobe/ef")

    return {
        "status": "PASS",
        "baseline": "native_static_single_assignment_simple_kmeans",
        "layout_dir": str(layout_dir),
        "build_manifest": str(manifest_path),
        "build_manifest_sha256": sha256(manifest_path),
        "artifact": str(artifact_path),
        "artifact_sha256": sha256(artifact_path),
        "layout_sha256": artifact["layout_sha256"],
        "logical_point_count": artifact["logical_point_count"],
        "physical_point_count": artifact["physical_point_count"],
        "expansion_ratio": routing["expansion_ratio"],
        "nprobe": profile.nprobe,
        "lower_hnsw_ef": profile.lower_hnsw_ef,
        "enhancements_disabled": {
            "orion_upper_graph": True,
            "multi_assignment": True,
            "dynamic_ef": True,
            "query_adaptive_nprobe": True,
            "topology_aware_partitioning": True,
            "load_aware_placement": True,
        },
    }


def make_simple_bodies(
    queries: np.ndarray, *, batch_size: int, top_k: int
) -> list[bytes]:
    """Build ordinary requests; nprobe and ef come only from the server artifact."""
    if len(queries) % batch_size:
        raise ValueError("query count must be divisible by batch size")
    bodies: list[bytes] = []
    for start in range(0, len(queries), batch_size):
        searches = [
            {
                "vector": row.tolist(),
                "limit": top_k,
                "with_payload": False,
                "with_vector": False,
            }
            for row in queries[start : start + batch_size]
        ]
        bodies.append(
            json.dumps(
                {"searches": searches}, separators=(",", ":"), allow_nan=False
            ).encode("utf-8")
        )
    return bodies


def recall_probe(
    host: str,
    port: int,
    path: str,
    queries: np.ndarray,
    neighbors: np.ndarray,
    *,
    batch_size: int,
    top_k: int,
) -> dict[str, Any]:
    result_ids: list[list[int]] = []
    for body in make_simple_bodies(queries, batch_size=batch_size, top_k=top_k):
        payload = json.loads(fixed.post_batch(host, port, path, body))
        rows = payload.get("result")
        if not isinstance(rows, list):
            raise RuntimeError("search batch response has no result rows")
        result_ids.extend(
            [[int(point["id"]) for point in row[:top_k]] for row in rows]
        )
    hits = sum(
        len(set(ids).intersection(int(value) for value in truth[:top_k]))
        for ids, truth in zip(result_ids, neighbors, strict=True)
    )
    return {
        "query_count": len(result_ids),
        "hits": hits,
        "recall_at_10": hits / (len(result_ids) * top_k),
    }


def collection_name(logical_shards: int) -> str:
    return f"simple_kmeans_vscale_glove_m{logical_shards}_20260825"


def preflight(args: argparse.Namespace, exp) -> dict[str, Any]:
    client_affinity = hashall.verify_client_affinity()
    containers = hashall.inspect_all_containers()
    for snapshot in containers:
        if (
            snapshot["image"] != hashall.EXPECTED_IMAGE
            or snapshot["image_id"] != hashall.EXPECTED_IMAGE_ID
        ):
            raise RuntimeError(f"unexpected image identity: {snapshot}")
        environment = set(snapshot["environment"])
        required = {
            "QDRANT__STORAGE__PERFORMANCE__MAX_SEARCH_THREADS=16",
            "QDRANT__SERVICE__HARDWARE_REPORTING=true",
            "QDRANT_HNSW_GRAPH_BUILD_SEED=20260821",
        }
        if not required.issubset(environment):
            raise RuntimeError(f"required environment missing: {snapshot}")
    cluster = hashall.wait_cluster_ready(exp, args.base_url)
    collections = exp.request_json(args.base_url, "GET", "/collections")["result"][
        "collections"
    ]
    observed_names = {str(row["name"]) for row in collections}
    if args.prepare_only:
        selected_names = {collection_name(value) for value in args.logical_shards}
        allowed_names = {collection_name(value) for value in LOGICAL_SHARD_COUNTS}
        unexpected = observed_names - allowed_names
        if unexpected:
            raise RuntimeError(
                f"preflight found collections outside parallel prepare scope: {unexpected}"
            )
        collection_gate = "observed_subset_of_simple_kmeans_prebuild_namespace"
    elif args.measure_only:
        allowed_names = {collection_name(value) for value in args.logical_shards}
        if observed_names != allowed_names:
            raise RuntimeError(
                f"measurement requires every prebuilt collection: "
                f"{observed_names} != {allowed_names}"
            )
        collection_gate = "all_measurement_collections_prebuilt"
    else:
        allowed_names = {collection_name(value) for value in args.preprepared}
        if observed_names != allowed_names:
            raise RuntimeError(
                f"preflight collection set mismatch: {observed_names} != {allowed_names}"
            )
        collection_gate = "standard_execution_collection_set"
    with hashall.h5py.File(args.hdf5_path, "r") as handle:
        shapes = {key: list(handle[key].shape) for key in ("train", "test", "neighbors")}
        hdf5_distance = str(handle.attrs.get("distance", ""))
    if shapes["train"] != [1_183_514, 200] or shapes["test"][:1] != [10_000]:
        raise RuntimeError(f"unexpected GloVe dataset shapes: {shapes}")
    if hdf5_distance != "angular":
        raise RuntimeError(f"unexpected dataset distance: {hdf5_distance!r}")
    schedule = {
        str(value): {
            "shards_per_node": hashall.shard_counts(value),
            "quota_cores_per_node": hashall.quota_schedule(value),
            "quota_cores_total": round(sum(hashall.quota_schedule(value)), 6),
        }
        for value in LOGICAL_SHARD_COUNTS
    }
    return {
        "timestamp": hashall.utc_timestamp(),
        "record_type": "simple_kmeans_virtual_linear_cpu_preflight",
        "status": "PASS",
        "client_affinity": client_affinity,
        "containers": containers,
        "cluster": cluster,
        "collections": collections,
        "allowed_preprepared_collections": sorted(allowed_names),
        "selected_collections": sorted(
            selected_names if args.prepare_only else allowed_names
        ),
        "collection_gate": collection_gate,
        "dataset": {
            "key": "glove-200-angular",
            "name": "GloVe-200-angular",
            "path": str(args.hdf5_path),
            "shapes": shapes,
            "hdf5_distance": hdf5_distance,
            "qdrant_distance": "Cosine",
        },
        "resource_schedule": schedule,
    }


def parse_preprepared(values: Sequence[str]) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for value in values:
        left, separator, right = value.partition("=")
        if not separator:
            raise ValueError("--preprepared must use M=/absolute/output/dir")
        logical_shards = int(left)
        if logical_shards not in LOGICAL_SHARD_COUNTS:
            raise ValueError(f"unsupported preprepared M={logical_shards}")
        path = Path(right).expanduser().resolve()
        if not (path / "preparation_manifest.json").is_file():
            raise FileNotFoundError(path / "preparation_manifest.json")
        result[logical_shards] = path
    return result


def run_prepare(
    args: argparse.Namespace,
    *,
    logical_shards: int,
    profile: RuntimeProfile,
    output_dir: Path,
    defer_artifact_install: bool = False,
) -> dict[str, Any]:
    build_manifest = load_json(profile.layout_dir / "build-manifest.json")
    import_manifest = import_manifest_path(profile.layout_dir, build_manifest)
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
        "simple_kmeans",
        "--topology",
        str(args.topology),
        "--run-id",
        args.run_id,
        "--collection",
        collection_name(logical_shards),
        "--base-url",
        args.base_url,
        "--output-dir",
        str(output_dir),
        "--layout-dir",
        str(profile.layout_dir),
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
        *(["--defer-artifact-install"] if defer_artifact_install else []),
    ]
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
            "native Simple KMeans prepare failed: "
            + (completed.stderr.strip() or completed.stdout.strip())
        )
    manifest_path = output_dir / "preparation_manifest.json"
    return {
        "status": "PASS",
        "mode": (
            "fresh_native_parallel_prebuild"
            if defer_artifact_install
            else "fresh_native_prepare"
        ),
        "artifact_install_deferred": defer_artifact_install,
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
        "preparation_manifest_sha256": sha256(manifest_path),
        "preparation_manifest": load_json(manifest_path),
    }


def reuse_prepare(
    *, logical_shards: int, output_dir: Path, expected_collection: str
) -> dict[str, Any]:
    manifest_path = output_dir / "preparation_manifest.json"
    manifest = load_json(manifest_path)
    if manifest.get("method") != "simple_kmeans":
        raise RuntimeError("preprepared collection is not Simple KMeans")
    if manifest.get("collection") != expected_collection:
        raise RuntimeError(
            f"preprepared collection mismatch: {manifest.get('collection')!r}"
        )
    if int(manifest.get("shard_count") or 0) != logical_shards:
        raise RuntimeError("preprepared shard count mismatch")
    return {
        "status": "PASS",
        "mode": "reused_preprepared_native_collection",
        "preparation_manifest_path": str(manifest_path),
        "preparation_manifest_sha256": sha256(manifest_path),
        "preparation_manifest": manifest,
    }


def verify_live_policy(
    exp, args: argparse.Namespace, profile: RuntimeProfile, collection: str
) -> dict[str, Any]:
    info = exp.collection_info(args.base_url, collection)
    config = info.get("config") or {}
    params = config.get("params") or {}
    policy = config.get("auto_shard_policy") or params.get("auto_shard_policy") or {}
    manifest = load_json(profile.layout_dir / "build-manifest.json")
    artifact = load_json(runtime_artifact_path(profile.layout_dir, manifest))
    expected = {
        "type": "simple_kmeans",
        "generation": artifact["generation"],
        "artifact_sha256": sha256(runtime_artifact_path(profile.layout_dir, manifest)),
    }
    if policy != expected:
        raise RuntimeError(f"live Simple KMeans policy mismatch: {policy} != {expected}")
    if int(params.get("shard_number") or 0) != profile.logical_shards:
        raise RuntimeError("live shard count mismatch")
    return {
        "status": "PASS",
        "policy": policy,
        "shard_number": params["shard_number"],
        "points_count": info.get("points_count"),
        "indexed_vectors_count": info.get("indexed_vectors_count"),
        "segments_count": info.get("segments_count"),
        "collection_info": info,
    }


def live_numeric_shard_placement(
    exp,
    collection: str,
    logical_shards: int,
) -> dict[str, Any]:
    per_node = hashall.per_node_collection_cluster(exp, collection)
    shard_ids_per_node: dict[str, list[int]] = {}
    all_shard_ids: list[int] = []
    for node in hashall.NODES:
        rows = per_node[node.private_ip].get("local_shards") or []
        shard_ids = sorted(int(row["shard_id"]) for row in rows)
        shard_ids_per_node[node.private_ip] = shard_ids
        all_shard_ids.extend(shard_ids)
    if sorted(all_shard_ids) != list(range(logical_shards)):
        raise RuntimeError(
            f"live numeric placement for {collection} is incomplete or duplicated: "
            f"{shard_ids_per_node}"
        )
    return {
        "status": "PASS",
        "collection": collection,
        "logical_shards": logical_shards,
        "node_order": [node.private_ip for node in hashall.NODES],
        "shard_ids_per_node": shard_ids_per_node,
        "shards_per_node": [
            len(shard_ids_per_node[node.private_ip]) for node in hashall.NODES
        ],
        "per_node_collection_cluster": per_node,
    }


def apply_measurement_resources(
    args: argparse.Namespace,
    exp,
    *,
    logical_shards: int,
    collection: str,
    output: Path,
) -> dict[str, Any]:
    placement = live_numeric_shard_placement(exp, collection, logical_shards)
    resources = hashall.apply_resource_schedule(
        logical_shards,
        output=output,
        shards_per_node=placement["shards_per_node"],
    )
    resources["live_placement_proof"] = placement
    hashall.write_json(output, resources)
    return resources


def run_benchmark(
    args: argparse.Namespace,
    exp,
    *,
    profile: RuntimeProfile,
    resource_record: dict[str, Any],
) -> dict[str, Any]:
    parsed = urllib.parse.urlparse(args.base_url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 6333
    collection = collection_name(profile.logical_shards)
    endpoint = f"/collections/{urllib.parse.quote(collection, safe='')}/points/search/batch"
    queries, neighbors = fixed.load_dataset(args.hdf5_path, args.top_k)
    tuning_queries, tuning_neighbors = queries[:1000], neighbors[:1000]
    heldout_queries, heldout_neighbors = queries[1000:10_000], neighbors[1000:10_000]

    tuning = recall_probe(
        host,
        port,
        endpoint,
        tuning_queries,
        tuning_neighbors,
        batch_size=args.batch_size,
        top_k=args.top_k,
    )
    heldout = recall_probe(
        host,
        port,
        endpoint,
        heldout_queries,
        heldout_neighbors,
        batch_size=args.batch_size,
        top_k=args.top_k,
    )
    if float(heldout["recall_at_10"]) < args.target_recall:
        raise RuntimeError(
            f"M={profile.logical_shards} profile misses held-out recall: {heldout}"
        )

    bodies = make_simple_bodies(
        heldout_queries, batch_size=args.batch_size, top_k=args.top_k
    )
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
    qps_values = [float(row["qps"]) for row in repeats]
    qps_mean = statistics.fmean(qps_values)
    qps_stdev = statistics.stdev(qps_values)
    qps_cv = qps_stdev / qps_mean
    if qps_cv > args.max_cv:
        raise RuntimeError(f"formal QPS CV remains too high: {qps_cv:.4f}")

    return {
        "timestamp": hashall.utc_timestamp(),
        "record_type": "native_plain_simple_kmeans_virtual_linear_cpu_fixed_recall",
        "status": "PASS",
        "logical_shards": profile.logical_shards,
        "physical_machines": PHYSICAL_MACHINE_COUNT,
        "base_url": args.base_url,
        "collection": collection,
        "request_contract": {
            "standard_coordinator_request": True,
            "endpoint": endpoint,
            "shard_selector_present": False,
            "client_side_fanout": False,
            "client_hnsw_ef_present": False,
            "server_router": "native_simple_kmeans",
            "visited_shards": profile.nprobe,
        },
        "baseline_contract": audit_plain_simple_kmeans(profile),
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
                "bounded native Simple KMeans calibration reused from the same "
                "dataset/layout family; selected profile revalidated on disjoint 9000 queries"
            ),
            "nprobe_upper_bound": max(profile.logical_shards, 1),
            "ef_bounds": [args.top_k, 512],
        },
        "parameters": {
            "nprobe": profile.nprobe,
            "lower_hnsw_ef": profile.lower_hnsw_ef,
            "top_k": args.top_k,
            "batch_size": args.batch_size,
            "selected_concurrency": selected_concurrency,
            "sweep_seconds": args.sweep_seconds,
            "warmup_seconds": args.warmup_seconds,
            "measure_seconds": args.measure_seconds,
            "repeat_count": len(repeats),
        },
        "live_policy_gate": verify_live_policy(exp, args, profile, collection),
        "per_node_collection_cluster": hashall.per_node_collection_cluster(exp, collection),
        "tuning_recall": tuning,
        "heldout_recall": heldout,
        "concurrency_sweep": sweep,
        "saturation_selection": selection,
        "warmup": warmup,
        "repeats": repeats,
        "qps_mean": qps_mean,
        "qps_stdev": qps_stdev,
        "qps_cv": qps_cv,
    }


def artifact_installer_command(
    args: argparse.Namespace,
    profile: RuntimeProfile,
    *,
    restart: bool,
) -> list[str]:
    build_manifest = load_json(profile.layout_dir / "build-manifest.json")
    artifact_path = runtime_artifact_path(profile.layout_dir, build_manifest)
    artifact = load_json(artifact_path)
    command = [
        sys.executable,
        str(Path(__file__).resolve().parents[3] / "tools/method4_distributed_cluster.py"),
        "--topology",
        str(args.topology),
        "--run-id",
        args.run_id,
        "install-simple-kmeans-artifact",
        "--collection",
        collection_name(profile.logical_shards),
        "--generation",
        str(artifact["generation"]),
        "--artifact",
        str(artifact_path),
        "--expected-sha256",
        sha256(artifact_path),
    ]
    if restart:
        command.extend(["--restart", "workers-first"])
    return command


def install_profile_artifact(
    args: argparse.Namespace,
    profile: RuntimeProfile,
    *,
    restart: bool,
) -> dict[str, Any]:
    command = artifact_installer_command(args, profile, restart=restart)
    started = time.monotonic()
    completed = subprocess.run(command, check=False, text=True, capture_output=True)
    record = {
        "timestamp": hashall.utc_timestamp(),
        "record_type": (
            "simple_kmeans_artifact_activation"
            if restart
            else "simple_kmeans_artifact_install"
        ),
        "status": "PASS" if completed.returncode == 0 else "FAIL",
        "logical_shards": profile.logical_shards,
        "restart_requested": restart,
        "seconds": time.monotonic() - started,
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }
    if completed.returncode != 0:
        raise RuntimeError(
            f"Simple KMeans artifact {'activation' if restart else 'install'} failed "
            f"for M={profile.logical_shards}: "
            + (completed.stderr.strip() or completed.stdout.strip())
        )
    return record


def execute_parallel_prepare(
    args: argparse.Namespace,
    exp,
    profiles: dict[int, RuntimeProfile],
    baseline_audits: dict[str, dict[str, Any]],
) -> int:
    build_resources = hashall.apply_resource_schedule(
        None, output=args.output_root / "resources-parallel-build.json"
    )
    hashall.wait_cluster_ready(exp, args.base_url)
    started_at = hashall.utc_timestamp()
    started_monotonic = time.monotonic()
    task_records: dict[int, dict[str, Any]] = {}

    def prepare_one(logical_shards: int, profile: RuntimeProfile) -> dict[str, Any]:
        point_root = args.output_root / f"m{logical_shards}"
        point_root.mkdir(parents=True, exist_ok=True)
        task_started = time.monotonic()
        print(f"[simple-vscale] parallel prebuild start M={logical_shards}", flush=True)
        if logical_shards in args.preprepared:
            prepare_record = reuse_prepare(
                logical_shards=logical_shards,
                output_dir=args.preprepared[logical_shards],
                expected_collection=collection_name(logical_shards),
            )
        else:
            prepare_record = run_prepare(
                args,
                logical_shards=logical_shards,
                profile=profile,
                output_dir=point_root / "native-prepare",
                defer_artifact_install=True,
            )
        prepare_record["build_resource_contract"] = build_resources
        prepare_record["baseline_contract"] = baseline_audits[str(logical_shards)]
        prepare_record["live_policy_gate"] = verify_live_policy(
            exp, args, profile, collection_name(logical_shards)
        )
        prepare_record["parallel_prebuild"] = {
            "requested_workers": args.parallel_prepares,
            "task_started_seconds": task_started - started_monotonic,
            "task_finished_seconds": time.monotonic() - started_monotonic,
        }
        hashall.write_json(point_root / "prepare.json", prepare_record)
        print(f"[simple-vscale] parallel prebuild complete M={logical_shards}", flush=True)
        return prepare_record

    failures: dict[int, str] = {}
    worker_count = min(args.parallel_prepares, len(profiles))
    with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(prepare_one, logical_shards, profile): logical_shards
            for logical_shards, profile in profiles.items()
        }
        for future in concurrent.futures.as_completed(futures):
            logical_shards = futures[future]
            try:
                task_records[logical_shards] = future.result()
            except BaseException as error:
                failures[logical_shards] = repr(error)

    summary = {
        "timestamp": hashall.utc_timestamp(),
        "record_type": "simple_kmeans_parallel_prebuild",
        "status": "PASS" if not failures else "FAIL",
        "started_at": started_at,
        "wall_seconds": time.monotonic() - started_monotonic,
        "requested_parallelism": args.parallel_prepares,
        "effective_parallelism": worker_count,
        "logical_shards": sorted(task_records),
        "failures": {str(key): value for key, value in sorted(failures.items())},
        "artifact_installation": "deferred_until_serial_measurement_phase",
        "formal_measurement_concurrency": 1,
    }
    hashall.write_json(args.output_root / "preparation-complete.json", summary)
    if failures:
        raise RuntimeError(f"parallel Simple KMeans prebuild failures: {failures}")
    return 0


def execute_measure_only(
    args: argparse.Namespace,
    exp,
    profiles: dict[int, RuntimeProfile],
    baseline_audits: dict[str, dict[str, Any]],
) -> int:
    completed = [
        value
        for value in LOGICAL_SHARD_COUNTS
        if (args.output_root / f"m{value}" / "benchmark.json").is_file()
    ]
    (args.output_root / "execution-failed.json").unlink(missing_ok=True)

    # Copy every artifact before the first restart. Otherwise a restart for one
    # collection could encounter another prebuilt routed collection whose
    # declared artifact is still absent.
    for logical_shards, profile in profiles.items():
        point_root = args.output_root / f"m{logical_shards}"
        reuse_prepare(
            logical_shards=logical_shards,
            output_dir=point_root / "native-prepare",
            expected_collection=collection_name(logical_shards),
        )
        install_record = install_profile_artifact(args, profile, restart=False)
        hashall.write_json(point_root / "artifact-installed.json", install_record)

    try:
        ordered = list(profiles.items())
        for index, (logical_shards, profile) in enumerate(ordered):
            point_root = args.output_root / f"m{logical_shards}"
            collection = collection_name(logical_shards)
            print(f"[simple-vscale] activate M={logical_shards}", flush=True)
            activation = install_profile_artifact(args, profile, restart=True)
            activation["live_policy_gate"] = verify_live_policy(
                exp, args, profile, collection
            )
            hashall.write_json(point_root / "activation.json", activation)

            resources = apply_measurement_resources(
                args,
                exp,
                logical_shards=logical_shards,
                collection=collection,
                output=point_root / "resources-measure.json",
            )
            hashall.wait_cluster_ready(exp, args.base_url)
            print(f"[simple-vscale] benchmark M={logical_shards}", flush=True)
            benchmark = run_benchmark(
                args, exp, profile=profile, resource_record=resources
            )
            hashall.write_json(point_root / "benchmark.json", benchmark)
            if logical_shards not in completed:
                completed.append(logical_shards)

            hashall.apply_resource_schedule(
                None, output=point_root / "resources-cleanup.json"
            )
            exp.delete_collection_if_exists(args.base_url, collection)
            remaining = exp.request_json(args.base_url, "GET", "/collections")[
                "result"
            ]["collections"]
            remaining_names = {str(row["name"]) for row in remaining}
            expected_remaining = {
                collection_name(value) for value, _profile in ordered[index + 1 :]
            }
            if remaining_names != expected_remaining:
                raise RuntimeError(
                    f"collection cleanup mismatch after M={logical_shards}: "
                    f"{remaining_names} != {expected_remaining}"
                )
            hashall.write_json(
                point_root / "cleanup.json",
                {
                    "timestamp": hashall.utc_timestamp(),
                    "record_type": "simple_kmeans_virtual_linear_cpu_cleanup",
                    "logical_shards": logical_shards,
                    "deleted_collection": collection,
                    "remaining_collections": remaining,
                    "status": "PASS",
                },
            )
            print(f"[simple-vscale] completed M={logical_shards}", flush=True)

        hashall.write_json(
            args.output_root / "execution-complete.json",
            {
                "timestamp": hashall.utc_timestamp(),
                "record_type": "simple_kmeans_virtual_linear_cpu_execution",
                "status": "MEASUREMENTS_COMPLETE",
                "dataset": "GloVe-200-angular",
                "logical_shards": sorted(completed),
                "physical_machines": PHYSICAL_MACHINE_COUNT,
                "baseline": "plain_native_simple_kmeans",
                "prebuild_mode": "parallel",
                "formal_measurement_concurrency": 1,
            },
        )
        return 0
    except BaseException as error:
        hashall.write_json(
            args.output_root / "execution-failed.json",
            {
                "timestamp": hashall.utc_timestamp(),
                "record_type": "simple_kmeans_virtual_linear_cpu_execution",
                "status": "FAILED",
                "completed_logical_shards": sorted(completed),
                "error": repr(error),
                "unmeasured_prebuilt_collections_preserved": True,
            },
        )
        raise


def execute(args: argparse.Namespace) -> int:
    repo_root = Path(__file__).resolve().parents[3]
    exp = hashall.load_experiment_module(repo_root)
    args.output_root.mkdir(parents=True, exist_ok=True)
    original_path = args.output_root / "original-resource-state.json"
    if args.restore_only:
        original = load_json(original_path)["containers"]
        hashall.restore_resource_state(
            original, args.output_root / "resource-restored.json"
        )
        return 0

    profiles = {logical_shards: PROFILES[logical_shards] for logical_shards in args.logical_shards}
    baseline_audits = {
        str(logical_shards): audit_plain_simple_kmeans(profile)
        for logical_shards, profile in profiles.items()
    }
    hashall.write_json(args.output_root / "baseline-audit.json", baseline_audits)

    original = hashall.inspect_all_containers()
    if not original_path.exists():
        hashall.write_json(
            original_path,
            {
                "timestamp": hashall.utc_timestamp(),
                "record_type": "simple_kmeans_virtual_linear_cpu_original_resources",
                "containers": original,
            },
        )
    preflight_record = preflight(args, exp)
    preflight_record["baseline"] = "plain_native_simple_kmeans"
    preflight_record["baseline_audits"] = baseline_audits
    hashall.write_json(args.output_root / "preflight.json", preflight_record)
    if args.preflight_only:
        return 0

    if args.prepare_only:
        try:
            return execute_parallel_prepare(
                args, exp, profiles, baseline_audits
            )
        finally:
            saved_original = load_json(original_path)["containers"]
            hashall.restore_resource_state(
                saved_original, args.output_root / "resource-restored.json"
            )

    if args.measure_only:
        try:
            return execute_measure_only(args, exp, profiles, baseline_audits)
        finally:
            saved_original = load_json(original_path)["containers"]
            hashall.restore_resource_state(
                saved_original, args.output_root / "resource-restored.json"
            )

    (args.output_root / "execution-failed.json").unlink(missing_ok=True)
    (args.output_root / "execution-complete.json").unlink(missing_ok=True)
    completed: list[int] = []
    body_error: BaseException | None = None
    try:
        for logical_shards, profile in profiles.items():
            point_root = args.output_root / f"m{logical_shards}"
            point_root.mkdir(parents=True, exist_ok=True)
            collection = collection_name(logical_shards)
            print(f"[simple-vscale] prepare M={logical_shards}", flush=True)
            build_resources = hashall.apply_resource_schedule(
                None, output=point_root / "resources-build.json"
            )
            hashall.wait_cluster_ready(exp, args.base_url)
            if logical_shards in args.preprepared:
                prepare_record = reuse_prepare(
                    logical_shards=logical_shards,
                    output_dir=args.preprepared[logical_shards],
                    expected_collection=collection,
                )
            else:
                prepare_record = run_prepare(
                    args,
                    logical_shards=logical_shards,
                    profile=profile,
                    output_dir=point_root / "native-prepare",
                )
            prepare_record["build_resource_contract"] = build_resources
            prepare_record["baseline_contract"] = baseline_audits[str(logical_shards)]
            prepare_record["live_policy_gate"] = verify_live_policy(
                exp, args, profile, collection
            )
            hashall.write_json(point_root / "prepare.json", prepare_record)

            resources = apply_measurement_resources(
                args,
                exp,
                logical_shards=logical_shards,
                collection=collection,
                output=point_root / "resources-measure.json",
            )
            hashall.wait_cluster_ready(exp, args.base_url)
            print(f"[simple-vscale] benchmark M={logical_shards}", flush=True)
            benchmark = run_benchmark(
                args, exp, profile=profile, resource_record=resources
            )
            hashall.write_json(point_root / "benchmark.json", benchmark)
            completed.append(logical_shards)

            hashall.apply_resource_schedule(
                None, output=point_root / "resources-cleanup.json"
            )
            exp.delete_collection_if_exists(args.base_url, collection)
            remaining = exp.request_json(args.base_url, "GET", "/collections")["result"][
                "collections"
            ]
            if remaining:
                raise RuntimeError(f"collection cleanup failed: {remaining}")
            hashall.write_json(
                point_root / "cleanup.json",
                {
                    "timestamp": hashall.utc_timestamp(),
                    "record_type": "simple_kmeans_virtual_linear_cpu_cleanup",
                    "logical_shards": logical_shards,
                    "deleted_collection": collection,
                    "remaining_collections": remaining,
                    "status": "PASS",
                },
            )
            print(f"[simple-vscale] completed M={logical_shards}", flush=True)
        hashall.write_json(
            args.output_root / "execution-complete.json",
            {
                "timestamp": hashall.utc_timestamp(),
                "record_type": "simple_kmeans_virtual_linear_cpu_execution",
                "status": "MEASUREMENTS_COMPLETE",
                "dataset": "GloVe-200-angular",
                "logical_shards": completed,
                "physical_machines": PHYSICAL_MACHINE_COUNT,
                "baseline": "plain_native_simple_kmeans",
            },
        )
        return 0
    except BaseException as error:
        body_error = error
        hashall.write_json(
            args.output_root / "execution-failed.json",
            {
                "timestamp": hashall.utc_timestamp(),
                "record_type": "simple_kmeans_virtual_linear_cpu_execution",
                "status": "FAILED",
                "completed_logical_shards": completed,
                "error": repr(error),
            },
        )
        raise
    finally:
        for logical_shards in args.logical_shards:
            try:
                exp.delete_collection_if_exists(
                    args.base_url, collection_name(logical_shards)
                )
            except Exception:
                pass
        saved_original = load_json(original_path)["containers"]
        try:
            hashall.restore_resource_state(
                saved_original, args.output_root / "resource-restored.json"
            )
        except Exception:
            if body_error is None:
                raise
            print("[simple-vscale] resource restoration also failed", flush=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://10.10.1.1:6333")
    parser.add_argument("--hdf5-path", type=Path, default=DEFAULT_HDF5)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--topology", type=Path, default=DEFAULT_TOPOLOGY)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--logical-shards", default="1,2,4,8,16,32")
    parser.add_argument("--preprepared", action="append", default=[])
    parser.add_argument("--upload-batch-size", type=int, default=2000)
    parser.add_argument("--index-timeout", type=float, default=10_800.0)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--target-recall", type=float, default=0.90)
    parser.add_argument("--tuning-margin", type=float, default=0.002)
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
        default=Path(
            "/proj/intelisys-PG0/exp/orion-distributed/cargo-target-native"
        ),
    )
    phase = parser.add_mutually_exclusive_group()
    phase.add_argument(
        "--prepare-only",
        action="store_true",
        help=(
            "Create, place, import, and index the selected collections concurrently, "
            "but defer artifact installation, restart, queries, and cleanup."
        ),
    )
    phase.add_argument(
        "--measure-only",
        action="store_true",
        help=(
            "Use collections retained by --prepare-only; install every artifact first, "
            "then activate, benchmark, and clean up one M at a time."
        ),
    )
    parser.add_argument(
        "--parallel-prepares",
        type=int,
        default=3,
        help="Maximum concurrent collection prebuilds used by --prepare-only.",
    )
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--restore-only", action="store_true")
    args = parser.parse_args(argv)

    args.dataset_spec = hashall.DATASET_SPECS["glove-200-angular"]
    args.hdf5_path = args.hdf5_path.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    args.topology = args.topology.expanduser().resolve()
    args.cargo_runner = args.cargo_runner.expanduser().resolve()
    args.cargo_target_dir = args.cargo_target_dir.expanduser().resolve()
    args.logical_shards = hashall.parse_int_csv(args.logical_shards)
    if any(value not in LOGICAL_SHARD_COUNTS for value in args.logical_shards):
        raise ValueError(f"logical shards must be a subset of {LOGICAL_SHARD_COUNTS}")
    if args.logical_shards != sorted(set(args.logical_shards)):
        raise ValueError("logical shards must be sorted and unique")
    args.concurrency_candidates = hashall.parse_int_csv(args.concurrency_candidates)
    if args.concurrency_candidates != sorted(set(args.concurrency_candidates)):
        raise ValueError("concurrency candidates must be sorted and unique")
    args.preprepared = parse_preprepared(args.preprepared)
    if set(args.preprepared) - set(args.logical_shards):
        raise ValueError("preprepared points must be included in --logical-shards")
    if args.min_repeats < 2 or args.max_repeats < args.min_repeats:
        raise ValueError("invalid repeat bounds")
    if args.parallel_prepares <= 0:
        raise ValueError("--parallel-prepares must be positive")
    if args.measure_only and args.preprepared:
        raise ValueError("--measure-only discovers native-prepare manifests in output-root")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    return execute(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
