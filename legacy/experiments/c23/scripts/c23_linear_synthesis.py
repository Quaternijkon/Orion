#!/usr/bin/env python3
"""Audit and synthesize the virtualized linear-resource C2/C3 retest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import c23_linear_e3e4 as experiment
import c23_linear_resources as resources


DATASETS = tuple(experiment.DATASETS)
METHODS = tuple(experiment.METHODS)
LOGICAL_SHARDS = (2, 4, 8, 16, 32)
ALL_LOGICAL_SHARDS = (1, *LOGICAL_SHARDS)
LOCAL_RECALL_THRESHOLDS = (0.80, 0.90, 0.95)
BOOTSTRAP_METRICS = {
    "achieved_recall_at_10": "higher_favors_orion",
    "common_ef_mean_local_target_recall": "higher_favors_orion",
    "common_ef_target_weighted_recall": "higher_favors_orion",
    "common_ef_mean_relevant_distance_computations": "lower_favors_orion",
    "P_exact": "lower_favors_orion",
    "P_HNSW": "lower_favors_orion",
    "Delta_P": "lower_favors_orion",
    "W90": "lower_favors_orion",
    "full_fanout_wall_us": "lower_favors_orion",
    "total_distance_computations": "lower_favors_orion",
    "total_graph_nodes_visited": "lower_favors_orion",
    "total_worker_cpu_us": "lower_favors_orion",
    "total_response_bytes": "lower_favors_orion",
}
METHOD_LABELS = {
    "random": "Random",
    "kmeans": "K-Means",
    "orion": "Orion",
    "orion_no_refinement": "Orion-NoRefinement",
    "common_m1": "Common M=1",
}
METHOD_COLORS = {
    "random": "#777777",
    "kmeans": "#d97706",
    "orion": "#2563eb",
    "orion_no_refinement": "#059669",
    "common_m1": "#111827",
}
METHOD_MARKERS = {
    "random": "o",
    "kmeans": "s",
    "orion": "^",
    "orion_no_refinement": "D",
    "common_m1": "x",
}
DATASET_LABELS = {"sift1m": "SIFT1M", "glove-200-angular": "GloVe-200"}
REPO_ROOT = Path(__file__).resolve().parents[3]
RESOURCE_PHASES = ("before_upload", "after_upload", "after_index", "after_e4", "after_e3")


def sha256_path(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().resolve().open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).expanduser().resolve().open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def summarize(values: np.ndarray, prefix: str) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.all(np.isfinite(array)):
        raise ValueError(f"invalid values for summary {prefix}")
    return {
        f"{prefix}_mean": float(np.mean(array)),
        f"{prefix}_median": float(np.median(array)),
        f"{prefix}_p95": float(np.percentile(array, 95)),
        f"{prefix}_std": float(np.std(array)),
    }


def close(left: float, right: float, *, tolerance: float = 1e-9) -> bool:
    return math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=tolerance)


def minimum_shards_many(contributions: np.ndarray) -> np.ndarray:
    values = np.asarray(contributions, dtype=np.int16)
    ordered = np.sort(values, axis=1)[:, ::-1]
    reached = np.cumsum(ordered, axis=1) >= experiment.TARGET_HITS
    first = np.argmax(reached, axis=1) + 1
    return np.where(np.any(reached, axis=1), first, values.shape[1] + 1).astype(
        np.int16
    )


def parse_cpu_stat(raw: str) -> dict[str, int]:
    values: dict[str, int] = {}
    for line in raw.splitlines():
        name, value = line.split()
        values[name] = int(value)
    if "usage_usec" not in values:
        raise ValueError("cgroup cpu.stat is missing usage_usec")
    return values


def validate_phase_snapshots(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    snapshots = manifest.get("resource_phase_snapshots") or {}
    if set(snapshots) != set(RESOURCE_PHASES):
        raise ValueError(f"resource phase snapshots are incomplete: {manifest['run_id']}")
    logical_shards = int(manifest["logical_shards"])
    previous_cpu = np.zeros(logical_shards, dtype=np.int64)
    rows: list[dict[str, Any]] = []
    for phase_index, phase in enumerate(RESOURCE_PHASES):
        snapshot = snapshots[phase]
        if (
            snapshot.get("status") != "PASS"
            or snapshot.get("run_id") != manifest["run_id"]
            or int(snapshot.get("logical_shards") or 0) != logical_shards
            or any(value != "PASS" for value in snapshot["checks"].values())
            or len(snapshot["runtime"]) != logical_shards
        ):
            raise ValueError(f"invalid resource phase snapshot: {manifest['run_id']} {phase}")
        observed: set[int] = set()
        for item in snapshot["runtime"]:
            shard = int(item["placement"]["shard_id"])
            if shard in observed or not 0 <= shard < logical_shards:
                raise ValueError(f"invalid snapshot shard coverage: {manifest['run_id']} {phase}")
            observed.add(shard)
            runtime = item["runtime"]
            cpu = parse_cpu_stat(str(runtime["cgroup_cpu_stat"]))
            usage = int(cpu["usage_usec"])
            if phase_index and usage < int(previous_cpu[shard]):
                raise ValueError(f"cgroup CPU usage decreased: {manifest['run_id']} {phase}")
            memory_current = int(runtime["cgroup_memory_current"])
            memory_max = int(runtime["cgroup_memory_max"])
            if (
                runtime.get("oom_killed")
                or not runtime.get("running")
                or memory_current < 0
                or memory_current > memory_max
                or memory_max != 8 * 1024**3
                or runtime.get("cgroup_memory_swap_max") != "0"
            ):
                raise ValueError(f"invalid cgroup phase counters: {manifest['run_id']} {phase}")
            rows.append(
                {
                    "dataset": manifest["dataset"],
                    "partition_method": manifest["partition_method"],
                    "logical_shards": logical_shards,
                    "run_id": manifest["run_id"],
                    "phase": phase,
                    "phase_index": phase_index,
                    "shard_id": shard,
                    "physical_host_id": int(item["placement"]["node_index"]),
                    "cpuset": item["placement"]["cpuset"],
                    "cpu_usage_usec_cumulative": usage,
                    "cpu_usage_usec_since_previous_snapshot": usage
                    - int(previous_cpu[shard])
                    if phase_index
                    else 0,
                    "user_usec_cumulative": int(cpu.get("user_usec", 0)),
                    "system_usec_cumulative": int(cpu.get("system_usec", 0)),
                    "nr_throttled_cumulative": int(cpu.get("nr_throttled", 0)),
                    "throttled_usec_cumulative": int(cpu.get("throttled_usec", 0)),
                    "memory_current_bytes": memory_current,
                    "memory_max_bytes": memory_max,
                    "oom_killed": False,
                    "snapshot_timestamp": snapshot["timestamp"],
                }
            )
            previous_cpu[shard] = usage
        if observed != set(range(logical_shards)):
            raise ValueError(f"incomplete resource snapshot coverage: {manifest['run_id']} {phase}")
    return rows


def expected_configurations() -> set[tuple[str, str, int]]:
    expected = {(dataset, "common_m1", 1) for dataset in DATASETS}
    expected.update(
        (dataset, method, logical_shards)
        for dataset in DATASETS
        for logical_shards in LOGICAL_SHARDS
        for method in METHODS
    )
    return expected


def validate_resource_contract(manifest: dict[str, Any]) -> None:
    logical_shards = int(manifest["logical_shards"])
    contract = manifest["resource_contract"]
    if any(value != "PASS" for value in contract["checks"].values()):
        raise ValueError(f"resource-contract check failed: {manifest['run_id']}")
    expected = {
        "logical_shards": logical_shards,
        "physical_core_equivalents": logical_shards,
        "memory_bytes_per_shard": 8 * 1024**3,
        "total_memory_capacity_bytes": logical_shards * 8 * 1024**3,
        "physical_hosts_with_shards": min(logical_shards, 4),
    }
    for key, value in expected.items():
        if int(contract[key]) != value:
            raise ValueError(
                f"resource contract {key} mismatch for {manifest['run_id']}: "
                f"{contract[key]} != {value}"
            )
    if not close(float(contract["m32_capacity_fraction"]), logical_shards / 32):
        raise ValueError(f"M/32 capacity fraction mismatch: {manifest['run_id']}")
    placements = contract["placement"]
    if len(placements) != logical_shards:
        raise ValueError(f"placement count mismatch: {manifest['run_id']}")
    observed_shards: set[int] = set()
    per_host_cpus: dict[int, set[int]] = {index: set() for index in range(4)}
    for placement in placements:
        shard = int(placement["shard_id"])
        node = int(placement["node_index"])
        slot = int(placement["host_slot"])
        physical_core = int(placement["physical_core"])
        if shard in observed_shards:
            raise ValueError(f"duplicate placement shard: {manifest['run_id']} {shard}")
        observed_shards.add(shard)
        if node != shard % 4 or slot != shard // 4 or physical_core != slot:
            raise ValueError(f"modulo-four placement mismatch: {manifest['run_id']}")
        expected_cpuset = f"{slot},{slot + 16}"
        if str(placement["cpuset"]) != expected_cpuset:
            raise ValueError(f"cpuset mismatch: {manifest['run_id']} shard={shard}")
        cpus = {slot, slot + 16}
        if per_host_cpus[node].intersection(cpus):
            raise ValueError(f"overlapping shard cpuset: {manifest['run_id']}")
        per_host_cpus[node].update(cpus)
    if observed_shards != set(range(logical_shards)):
        raise ValueError(f"incomplete placement coverage: {manifest['run_id']}")
    expected_per_host = {
        str(node): sum(shard % 4 == node for shard in range(logical_shards))
        for node in range(4)
    }
    if {str(key): int(value) for key, value in contract["shards_per_host"].items()} != expected_per_host:
        raise ValueError(f"per-host placement counts mismatch: {manifest['run_id']}")


def validate_single_hnsw_index_gate(manifest: dict[str, Any], run_id: str) -> None:
    index_gate = manifest["index_gate"]
    logical_shards = int(manifest["logical_shards"])
    if (
        index_gate.get("status") != "PASS"
        or int(index_gate["expected_nonempty_hnsw_segments"]) != logical_shards
        or int(index_gate["expected_total_segments"]) != 2 * logical_shards
        or int(index_gate["observed_total_segments"]) != 2 * logical_shards
    ):
        raise ValueError(f"single-HNSW index gate failed: {run_id}")
    segment_telemetry = index_gate.get("segment_telemetry")
    if segment_telemetry is not None:
        expected_counts = manifest["upload"]["expected_points_per_shard"]
        if len(segment_telemetry) != logical_shards or len(expected_counts) != logical_shards:
            raise ValueError(f"segment telemetry cardinality mismatch: {run_id}")
        for telemetry, expected_count in zip(
            segment_telemetry, expected_counts, strict=True
        ):
            if (
                int(telemetry.get("indexing_threshold_kib") or 0)
                != experiment.INDEXING_THRESHOLD_KIB
                or not experiment.single_hnsw_layout_matches(
                    telemetry, int(expected_count)
                )
            ):
                raise ValueError(f"segment telemetry single-HNSW gate failed: {run_id}")
    tail_compaction = index_gate.get("tail_compaction")
    if tail_compaction is not None:
        if tail_compaction.get("status") not in {"NOT_REQUIRED", "PASS"}:
            raise ValueError(f"tail compaction did not finish: {run_id}")
        if tail_compaction.get("status") == "PASS" and (
            not tail_compaction.get("affected_shards")
            or int(tail_compaction.get("temporary_indexing_threshold_kib") or 0)
            != experiment.TAIL_COMPACTION_THRESHOLD_KIB
            or int(tail_compaction.get("restored_indexing_threshold_kib") or 0)
            != experiment.INDEXING_THRESHOLD_KIB
        ):
            raise ValueError(f"tail compaction audit failed: {run_id}")


def validate_manifest(path: Path, registry_row: dict[str, Any]) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    run_id = str(registry_row["configuration_id"])
    if manifest.get("run_id") != run_id or manifest.get("status") != "PASS":
        raise ValueError(f"invalid completed manifest identity/status: {path}")
    expected = {
        "protocol_version": resources.PROTOCOL_VERSION,
        "dataset": registry_row["dataset"],
        "partition_method": registry_row["partition_method"],
        "logical_shards": int(registry_row["logical_shards"]),
        "ef_search": int(registry_row["ef_search"]),
    }
    mismatches = {
        key: {"expected": value, "actual": manifest.get(key)}
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise ValueError(f"registry/manifest mismatch for {run_id}: {mismatches}")
    checks = manifest.get("checks") or {}
    if not checks or any(value != "PASS" for value in checks.values()):
        raise ValueError(f"configuration checks failed: {run_id}: {checks}")
    if manifest.get("query_range") != [1000, 10000]:
        raise ValueError(f"unexpected held-out query range: {run_id}")
    if tuple(manifest.get("e3_ef_values") or ()) != tuple(experiment.EF_GRID):
        raise ValueError(f"incomplete E3 ef grid: {run_id}")
    if manifest["benchmark_client_affinity"].get("status") != "PASS":
        raise ValueError(f"client affinity failed: {run_id}")
    expected_client = list(range(8, 16)) + list(range(24, 32))
    if manifest["benchmark_client_affinity"].get("actual") != expected_client:
        raise ValueError(f"client affinity CPU list mismatch: {run_id}")
    for key in ("resource_verification_before", "resource_verification_after"):
        if manifest[key].get("status") != "PASS":
            raise ValueError(f"{key} failed: {run_id}")
    validate_single_hnsw_index_gate(manifest, run_id)
    cleanup = manifest["cleanup"]
    if (
        cleanup.get("phase") != "cleanup_complete"
        or cleanup.get("restored", {}).get("status") != "PASS"
        or cleanup.get("restored", {}).get("failures")
    ):
        raise ValueError(f"cleanup/restore failed: {run_id}")
    validate_resource_contract(manifest)
    manifest["_resource_phase_rows"] = validate_phase_snapshots(manifest)
    for path_key, hash_key in (
        ("per_query_path", "per_query_sha256"),
        ("e3_per_query_path", "e3_per_query_sha256"),
    ):
        artifact = Path(manifest[path_key]).expanduser().resolve()
        if not artifact.is_file() or sha256_path(artifact) != manifest[hash_key]:
            raise ValueError(f"raw artifact checksum mismatch: {run_id} {artifact}")
    return manifest


def load_registry(raw_root: Path) -> tuple[dict[str, Any], dict[tuple[str, str, int], dict[str, Any]]]:
    path = raw_root / "matrix/registry.json"
    registry = json.loads(path.read_text(encoding="utf-8"))
    if registry.get("protocol_version") != resources.PROTOCOL_VERSION:
        raise ValueError("matrix registry protocol mismatch")
    rows = registry.get("configurations") or []
    if int(registry.get("completed_count") or 0) != 42 or len(rows) != 42:
        raise ValueError(
            f"matrix is incomplete: completed={registry.get('completed_count')} rows={len(rows)}"
        )
    result: dict[tuple[str, str, int], dict[str, Any]] = {}
    for row in rows:
        key = (str(row["dataset"]), str(row["partition_method"]), int(row["logical_shards"]))
        if key in result:
            raise ValueError(f"duplicate matrix configuration: {key}")
        if row.get("status") != "PASS":
            raise ValueError(f"non-PASS matrix row: {key}")
        manifest_path = Path(row["manifest"]).expanduser().resolve()
        result[key] = validate_manifest(manifest_path, row)
    if set(result) != expected_configurations():
        missing = sorted(expected_configurations() - set(result))
        extra = sorted(set(result) - expected_configurations())
        raise ValueError(f"matrix coverage mismatch: missing={missing} extra={extra}")
    return registry, result


def validate_stage_inputs(registry: dict[str, Any]) -> dict[str, Any]:
    stage_root = (
        REPO_ROOT / "experiments/c23/retests/stage13-linear-resource-virtualized"
    )
    static_files = (
        stage_root / "PROTOCOL.md",
        stage_root / "topology.json",
        stage_root / "preflight.json",
        stage_root / "resource-smoke-m1.json",
        stage_root / "resource-smoke-m32.json",
        REPO_ROOT / "experiments/c23/scripts/c23_linear_resources.py",
        REPO_ROOT / "experiments/c23/scripts/c23_linear_e3e4.py",
        REPO_ROOT / "experiments/c23/scripts/c23_linear_matrix.py",
        REPO_ROOT / "experiments/c23/scripts/c23_linear_synthesis.py",
    )
    for path in static_files:
        if not path.is_file():
            raise ValueError(f"required Stage 13 input is missing: {path}")
    for name in ("preflight.json", "resource-smoke-m1.json", "resource-smoke-m32.json"):
        payload = json.loads((stage_root / name).read_text(encoding="utf-8"))
        if (
            payload.get("protocol_version") != resources.PROTOCOL_VERSION
            or payload.get("status") != "PASS"
            or any(value != "PASS" for value in (payload.get("checks") or {}).values())
        ):
            raise ValueError(f"Stage 13 preflight/smoke failed: {name}")
        if name.startswith("resource-smoke") and payload.get("cleanup", {}).get(
            "restored", {}
        ).get("status") != "PASS":
            raise ValueError(f"Stage 13 smoke cleanup failed: {name}")
    settings = registry["settings"]
    calibrations: dict[str, Any] = {}
    for dataset in DATASETS:
        path = Path(settings["calibrations"][dataset]).expanduser().resolve()
        payload = json.loads(path.read_text(encoding="utf-8"))
        checks = payload.get("checks") or {}
        selected = int(payload.get("selected_ef_search") or 0)
        feasible = [
            int(row["ef_search"])
            for row in payload.get("candidates") or []
            if row.get("meets_recall_target")
        ]
        if (
            payload.get("protocol_version") != resources.PROTOCOL_VERSION
            or payload.get("status") != "PASS"
            or payload.get("dataset") != dataset
            or payload.get("query_range") != [0, 1000]
            or any(value != "PASS" for value in checks.values())
            or selected != int(settings["selected_ef"][dataset])
            or not feasible
            or selected != feasible[0]
        ):
            raise ValueError(f"invalid common-ef calibration: {dataset}")
        candidate_files = []
        for candidate in payload["candidates"]:
            artifact = Path(candidate["per_query_path"]).expanduser().resolve()
            if not artifact.is_file() or sha256_path(artifact) != candidate["per_query_sha256"]:
                raise ValueError(f"calibration candidate checksum mismatch: {artifact}")
            candidate_files.append(
                {
                    "ef_search": int(candidate["ef_search"]),
                    "path": str(artifact),
                    "sha256": candidate["per_query_sha256"],
                }
            )
        calibrations[dataset] = {
            "path": str(path),
            "sha256": sha256_path(path),
            "selected_ef_search": selected,
            "candidate_files": candidate_files,
            "checks": {"selection": "PASS", "raw_checksums": "PASS"},
        }
    return {
        "git_commit": git_commit(),
        "static_files": {
            str(path.relative_to(REPO_ROOT)): sha256_path(path) for path in static_files
        },
        "calibrations": calibrations,
        "checks": {
            "preflight": "PASS",
            "resource_smoke_m1": "PASS",
            "resource_smoke_m32": "PASS",
            "implementation_files_hashed": "PASS",
            "common_ef_calibrations": "PASS",
        },
    }


def load_structural_rows(
    legacy_analysis_root: Path,
) -> tuple[dict[tuple[str, str, int], dict[str, str]], dict[tuple[str, str, int], dict[str, str]]]:
    e1: dict[tuple[str, str, int], dict[str, str]] = {}
    e2: dict[tuple[str, str, int], dict[str, str]] = {}
    for dataset in DATASETS:
        for target, relative in (
            (e1, Path(dataset) / "e1-v1/e1_metrics.csv"),
            (e2, Path(dataset) / "e2-v1/e2_summary.csv"),
        ):
            for row in read_csv(legacy_analysis_root / relative):
                key = (
                    str(row["dataset"]),
                    str(row["partition_method"]),
                    int(row["logical_shards"]),
                )
                if key in target:
                    raise ValueError(f"duplicate structural row: {key}")
                target[key] = row
    expected = {
        (dataset, method, logical_shards)
        for dataset in DATASETS
        for method in METHODS
        for logical_shards in LOGICAL_SHARDS
    }
    if set(e1) != expected or set(e2) != expected:
        raise ValueError("accepted structural evidence does not cover the 40 method/M points")
    return e1, e2


def validate_structural_binding(
    manifests: dict[tuple[str, str, int], dict[str, Any]],
    e1: dict[tuple[str, str, int], dict[str, str]],
    e2: dict[tuple[str, str, int], dict[str, str]],
) -> None:
    for key, row in e1.items():
        manifest = manifests[key]
        observed = str(manifest["partition"]["artifact_sha256"])
        if observed != row["partition_artifact_sha256"] or observed != e2[key]["partition_artifact_sha256"]:
            raise ValueError(f"structural partition checksum binding failed: {key}")
        dataset_sha = manifest["dataset_audit"]["sha256"]
        if dataset_sha != row["dataset_sha256"] or dataset_sha != e2[key]["dataset_sha256"]:
            raise ValueError(f"structural dataset checksum binding failed: {key}")


def validate_e4_arrays(
    manifest: dict[str, Any], arrays: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    logical_shards = int(manifest["logical_shards"])
    query_count = 9000
    one_dimensional = {
        "query_id",
        "p_exact",
        "p_hnsw",
        "delta_p",
        "w90",
        "reached",
        "full_wall_us",
    }
    two_dimensional = {
        "contributions",
        "distance",
        "nodes",
        "worker_cpu",
        "worker_wall",
        "latency_us",
        "response_bytes",
    }
    if set(arrays) != one_dimensional | two_dimensional:
        raise ValueError(f"unexpected E4 arrays: {manifest['run_id']} {sorted(arrays)}")
    for name in one_dimensional:
        if arrays[name].shape != (query_count,):
            raise ValueError(f"invalid E4 shape {name}: {manifest['run_id']}")
    for name in two_dimensional:
        if arrays[name].shape != (query_count, logical_shards):
            raise ValueError(f"invalid E4 shape {name}: {manifest['run_id']}")
    expected_query_ids = np.arange(1000, 10000, dtype=np.int32)
    if not np.array_equal(arrays["query_id"], expected_query_ids):
        raise ValueError(f"E4 query ID sequence mismatch: {manifest['run_id']}")
    contributions = arrays["contributions"]
    if np.any(contributions < 0) or np.any(contributions > experiment.TOP_K):
        raise ValueError(f"invalid E4 recovered contributions: {manifest['run_id']}")
    if np.any(arrays["p_exact"] < 1) or np.any(arrays["p_exact"] > logical_shards):
        raise ValueError(f"invalid exact fan-out range: {manifest['run_id']}")
    if np.any(arrays["p_hnsw"] < 1) or np.any(
        arrays["p_hnsw"] > logical_shards + 1
    ):
        raise ValueError(f"invalid HNSW fan-out range: {manifest['run_id']}")
    if np.any(arrays["delta_p"] < 0):
        raise ValueError(f"negative ANN fan-out penalty: {manifest['run_id']}")
    if np.any(arrays["distance"] <= 0) or np.any(arrays["nodes"] <= 0):
        raise ValueError(f"non-positive E4 graph work: {manifest['run_id']}")
    if np.any(arrays["worker_cpu"] <= 0) or np.any(arrays["worker_wall"] <= 0):
        raise ValueError(f"non-positive E4 hardware time: {manifest['run_id']}")
    if np.any(arrays["latency_us"] < 0) or np.any(arrays["response_bytes"] <= 0):
        raise ValueError(f"invalid E4 transport counters: {manifest['run_id']}")
    return arrays


def validate_e3_arrays(
    manifest: dict[str, Any], arrays: dict[str, np.ndarray], e4: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    logical_shards = int(manifest["logical_shards"])
    ef_values = np.asarray(experiment.EF_GRID, dtype=np.int32)
    shape = (len(ef_values), 9000, logical_shards)
    expected_names = {
        "ef_values",
        "query_id",
        "target_count",
        "recovered",
        "distance",
        "nodes",
        "worker_cpu",
        "latency_us",
    }
    if set(arrays) != expected_names:
        raise ValueError(f"unexpected E3 arrays: {manifest['run_id']} {sorted(arrays)}")
    if not np.array_equal(arrays["ef_values"], ef_values):
        raise ValueError(f"E3 ef values mismatch: {manifest['run_id']}")
    if not np.array_equal(arrays["query_id"], e4["query_id"]):
        raise ValueError(f"E3/E4 query IDs differ: {manifest['run_id']}")
    if arrays["target_count"].shape != (9000, logical_shards):
        raise ValueError(f"E3 target-count shape mismatch: {manifest['run_id']}")
    for name in ("recovered", "distance", "nodes", "worker_cpu", "latency_us"):
        if arrays[name].shape != shape:
            raise ValueError(f"E3 array shape mismatch {name}: {manifest['run_id']}")
    target = arrays["target_count"]
    relevant = target > 0
    if np.any(target < 0) or not np.all(np.sum(target, axis=1) == experiment.TOP_K):
        raise ValueError(f"E3 exact target groups invalid: {manifest['run_id']}")
    for index, _ in enumerate(ef_values):
        recovered = arrays["recovered"][index]
        if np.any(recovered[relevant] < 0) or np.any(recovered[relevant] > target[relevant]):
            raise ValueError(f"E3 recovery counts invalid: {manifest['run_id']}")
        if np.any(recovered[~relevant] != -1):
            raise ValueError(f"E3 irrelevant-pair sentinel invalid: {manifest['run_id']}")
        for name in ("distance", "nodes", "worker_cpu"):
            if np.any(arrays[name][index][relevant] <= 0) or np.any(
                arrays[name][index][~relevant] != 0
            ):
                raise ValueError(f"E3 work counters invalid {name}: {manifest['run_id']}")
        if np.any(arrays["latency_us"][index][relevant] < 0) or np.any(
            arrays["latency_us"][index][~relevant] != 0
        ):
            raise ValueError(f"E3 latency counters invalid: {manifest['run_id']}")
    common = tuple(experiment.EF_GRID).index(int(manifest["ef_search"]))
    if not np.array_equal(arrays["recovered"][common][relevant], e4["contributions"][relevant]):
        raise ValueError(f"common-EF E3/E4 recovery differs: {manifest['run_id']}")
    for e3_name, e4_name in (
        ("distance", "distance"),
        ("nodes", "nodes"),
        ("worker_cpu", "worker_cpu"),
        ("latency_us", "latency_us"),
    ):
        if not np.array_equal(arrays[e3_name][common][relevant], e4[e4_name][relevant]):
            raise ValueError(f"common-EF E3/E4 {e3_name} differs: {manifest['run_id']}")
    exact = minimum_shards_many(target)
    if not np.array_equal(exact, e4["p_exact"]):
        raise ValueError(f"P_exact does not reproduce from E3 target groups: {manifest['run_id']}")
    return arrays


def reproduce_oracle_metrics(e4: dict[str, np.ndarray]) -> None:
    query_count, logical_shards = e4["contributions"].shape
    p_hnsw = np.empty(query_count, dtype=np.int16)
    w90 = np.empty(query_count, dtype=np.uint64)
    reached = np.empty(query_count, dtype=np.bool_)
    for query in range(query_count):
        order = sorted(
            range(logical_shards),
            key=lambda shard: (
                -int(e4["contributions"][query, shard]),
                int(e4["distance"][query, shard]),
                shard,
            ),
        )
        cumulative = 0
        work = 0
        selected = 0
        for shard in order:
            selected += 1
            cumulative += int(e4["contributions"][query, shard])
            work += int(e4["distance"][query, shard])
            if cumulative >= experiment.TARGET_HITS:
                break
        reached[query] = cumulative >= experiment.TARGET_HITS
        p_hnsw[query] = selected if reached[query] else logical_shards + 1
        w90[query] = work
    if not np.array_equal(p_hnsw, e4["p_hnsw"]):
        raise ValueError("P_HNSW does not reproduce")
    if not np.array_equal(w90, e4["w90"]):
        raise ValueError("W90 does not reproduce")
    if not np.array_equal(reached, e4["reached"]):
        raise ValueError("HNSW reach flag does not reproduce")
    if not np.array_equal(e4["p_hnsw"] - e4["p_exact"], e4["delta_p"]):
        raise ValueError("Delta_P does not reproduce")


def partition_seed(manifest: dict[str, Any]) -> Any:
    value = manifest["partition"].get("partition_seed")
    return "" if value is None else int(value)


def structural_fields(
    key: tuple[str, str, int],
    e1: dict[tuple[str, str, int], dict[str, str]],
    e2: dict[tuple[str, str, int], dict[str, str]],
) -> dict[str, Any]:
    if key[2] == 1:
        return {
            "edge_cut_ratio": 0.0,
            "traversal_weighted_edge_cut": 0.0,
            "mean_retained_degree_ratio": 1.0,
            "largest_component_fraction": 1.0,
            "path_shards_mean": 1.0,
            "path_transitions_mean": 0.0,
            "shard_size_std": 0.0,
            "shard_size_min": int(experiment.DATASETS[key[0]]["train_rows"]),
            "shard_size_max": int(experiment.DATASETS[key[0]]["train_rows"]),
            "max_size_over_mean": 1.0,
            "structural_source": "trivial_unpartitioned_control",
        }
    row1, row2 = e1[key], e2[key]
    return {
        "edge_cut_ratio": float(row1["edge_cut_ratio"]),
        "traversal_weighted_edge_cut": float(row2["traversal_weighted_edge_cut"]),
        "mean_retained_degree_ratio": float(row1["mean_retained_degree_ratio"]),
        "largest_component_fraction": float(
            row1["largest_component_fraction_vector_weighted_mean"]
        ),
        "path_shards_mean": float(row2["path_shards_mean"]),
        "path_transitions_mean": float(row2["path_transitions_mean"]),
        "shard_size_std": float(row1["shard_size_std"]),
        "shard_size_min": int(row1["shard_size_min"]),
        "shard_size_max": int(row1["shard_size_max"]),
        "max_size_over_mean": float(row1["max_shard_size_over_mean"]),
        "structural_source": "accepted_metrics_checksum_bound_to_identical_partition",
    }


def analyze_configuration(
    key: tuple[str, str, int],
    manifest: dict[str, Any],
    e1: dict[tuple[str, str, int], dict[str, str]],
    e2: dict[tuple[str, str, int], dict[str, str]],
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, np.ndarray],
]:
    with np.load(manifest["per_query_path"], allow_pickle=False) as archive:
        e4 = validate_e4_arrays(manifest, {name: np.asarray(archive[name]) for name in archive.files})
    with np.load(manifest["e3_per_query_path"], allow_pickle=False) as archive:
        e3 = validate_e3_arrays(
            manifest,
            {name: np.asarray(archive[name]) for name in archive.files},
            e4,
        )
    reproduce_oracle_metrics(e4)
    dataset, method, logical_shards = key
    relevant = e3["target_count"] > 0
    target_values = e3["target_count"][relevant].astype(np.float64)
    e3_rows: list[dict[str, Any]] = []
    recall_cube = np.empty(e3["recovered"].shape, dtype=np.float64)
    recall_cube.fill(np.nan)
    for ef_index, ef_search in enumerate(e3["ef_values"]):
        recovered = e3["recovered"][ef_index][relevant].astype(np.float64)
        recall = recovered / target_values
        recall_cube[ef_index][relevant] = recall
        e3_rows.append(
            {
                "dataset": dataset,
                "partition_method": method,
                "partition_seed": partition_seed(manifest),
                "logical_shards": logical_shards,
                "physical_core_equivalents": logical_shards,
                "m32_capacity_fraction": logical_shards / 32,
                "relevant_query_shard_pairs": int(np.count_nonzero(relevant)),
                "ef_search": int(ef_search),
                **summarize(recall, "local_target_recall"),
                "local_target_recall_target_weighted": float(
                    np.sum(recovered) / np.sum(target_values)
                ),
                **summarize(e3["distance"][ef_index][relevant], "distance_computations"),
                **summarize(e3["nodes"][ef_index][relevant], "graph_nodes_visited"),
                **summarize(e3["worker_cpu"][ef_index][relevant], "worker_cpu_us"),
                **summarize(e3["latency_us"][ef_index][relevant], "local_latency_us"),
            }
        )
    fixed_rows: list[dict[str, Any]] = []
    pair_recall = recall_cube[:, relevant].T
    pair_distance = e3["distance"][:, relevant].T
    pair_nodes = e3["nodes"][:, relevant].T
    ef_values = np.asarray(e3["ef_values"], dtype=np.int64)
    for threshold in LOCAL_RECALL_THRESHOLDS:
        meets = pair_recall >= threshold
        success = np.any(meets, axis=1)
        first = np.argmax(meets, axis=1)
        selected_rows = np.flatnonzero(success)
        selected_columns = first[selected_rows]
        row: dict[str, Any] = {
            "dataset": dataset,
            "partition_method": method,
            "partition_seed": partition_seed(manifest),
            "logical_shards": logical_shards,
            "local_recall_threshold": threshold,
            "query_shard_pair_count": len(pair_recall),
            "success_count": int(np.count_nonzero(success)),
            "success_fraction": float(np.mean(success)),
        }
        if len(selected_rows):
            row.update(summarize(ef_values[selected_columns], "required_ef_search"))
            row.update(
                summarize(
                    pair_distance[selected_rows, selected_columns],
                    "required_distance_computations",
                )
            )
            row.update(
                summarize(
                    pair_nodes[selected_rows, selected_columns],
                    "required_graph_nodes_visited",
                )
            )
        else:
            for prefix in (
                "required_ef_search",
                "required_distance_computations",
                "required_graph_nodes_visited",
            ):
                for suffix in ("mean", "median", "p95", "std"):
                    row[f"{prefix}_{suffix}"] = ""
        fixed_rows.append(row)
    common_index = tuple(map(int, e3["ef_values"])).index(int(manifest["ef_search"]))
    per_query_local_mean = np.asarray(
        [np.mean(recall_cube[common_index, query][relevant[query]]) for query in range(9000)],
        dtype=np.float64,
    )
    per_query_local_weighted = np.sum(e4["contributions"], axis=1) / experiment.TOP_K
    per_query_relevant_work = np.asarray(
        [np.mean(e4["distance"][query][relevant[query]]) for query in range(9000)],
        dtype=np.float64,
    )
    query_metrics = {
        "query_id": e4["query_id"].copy(),
        "achieved_recall_at_10": per_query_local_weighted.astype(np.float64),
        "common_ef_mean_local_target_recall": per_query_local_mean,
        "common_ef_target_weighted_recall": per_query_local_weighted.astype(np.float64),
        "common_ef_mean_relevant_distance_computations": per_query_relevant_work,
        "P_exact": e4["p_exact"].astype(np.float64),
        "P_HNSW": e4["p_hnsw"].astype(np.float64),
        "Delta_P": e4["delta_p"].astype(np.float64),
        "W90": e4["w90"].astype(np.float64),
        "full_fanout_wall_us": e4["full_wall_us"].astype(np.float64),
        "total_distance_computations": np.sum(e4["distance"], axis=1).astype(np.float64),
        "total_graph_nodes_visited": np.sum(e4["nodes"], axis=1).astype(np.float64),
        "total_worker_cpu_us": np.sum(e4["worker_cpu"], axis=1).astype(np.float64),
        "total_response_bytes": np.sum(e4["response_bytes"], axis=1).astype(np.float64),
    }
    query_summary = manifest["query_summary"]
    if not close(np.mean(query_metrics["P_HNSW"]), query_summary["P_HNSW_mean"]):
        raise ValueError(f"manifest P_HNSW summary mismatch: {manifest['run_id']}")
    if not close(np.mean(query_metrics["W90"]), query_summary["W90_mean"]):
        raise ValueError(f"manifest W90 summary mismatch: {manifest['run_id']}")
    counts = np.asarray(manifest["upload"]["expected_points_per_shard"], dtype=np.int64)
    if len(counts) != logical_shards or int(np.sum(counts)) != int(
        experiment.DATASETS[dataset]["train_rows"]
    ):
        raise ValueError(f"partition count coverage mismatch: {manifest['run_id']}")
    placement = {
        int(row["shard_id"]): row for row in manifest["resource_contract"]["placement"]
    }
    balance_rows = [
        {
            "dataset": dataset,
            "partition_method": method,
            "partition_seed": partition_seed(manifest),
            "logical_shards": logical_shards,
            "shard_id": shard,
            "physical_host_id": int(placement[shard]["node_index"]),
            "host_slot": int(placement[shard]["host_slot"]),
            "cpuset": placement[shard]["cpuset"],
            "vector_count": int(counts[shard]),
            "query_participation_count": int(np.count_nonzero(relevant[:, shard])),
            "query_participation_fraction": float(np.mean(relevant[:, shard])),
            "memory_cap_bytes": int(manifest["resource_contract"]["memory_bytes_per_shard"]),
        }
        for shard in range(logical_shards)
    ]
    structural = structural_fields(key, e1, e2)
    common_recall = pair_recall[:, common_index]
    common_distance = pair_distance[:, common_index]
    summary = {
        "dataset": dataset,
        "partition_method": method,
        "partition_seed": partition_seed(manifest),
        "logical_shards": logical_shards,
        "physical_hosts": 4,
        "physical_hosts_with_shards": int(
            manifest["resource_contract"]["physical_hosts_with_shards"]
        ),
        "physical_core_equivalents": logical_shards,
        "m32_capacity_fraction": logical_shards / 32,
        "memory_capacity_gib": logical_shards * 8,
        "common_ef_search": int(manifest["ef_search"]),
        "measurement_query_count": 9000,
        "point_count": int(np.sum(counts)),
        "shard_size_mean": float(np.mean(counts)),
        "shard_size_std": float(np.std(counts)),
        "shard_size_cv": float(np.std(counts) / np.mean(counts)),
        "shard_size_min": int(np.min(counts)),
        "shard_size_max": int(np.max(counts)),
        "max_size_over_mean": float(np.max(counts) / np.mean(counts)),
        **structural,
        **summarize(common_recall, "common_ef_local_target_recall"),
        **summarize(common_distance, "common_ef_local_distance_computations"),
        "achieved_recall_at_10": float(np.mean(per_query_local_weighted)),
        **summarize(e4["p_exact"], "P_exact"),
        **summarize(e4["p_hnsw"], "P_HNSW"),
        **summarize(e4["delta_p"], "Delta_P"),
        **summarize(e4["w90"], "W90"),
        "hnsw_reached_90_fraction": float(np.mean(e4["reached"])),
        **summarize(e4["full_wall_us"], "full_fanout_wall_us"),
        **summarize(np.sum(e4["distance"], axis=1), "total_distance_computations"),
        **summarize(np.sum(e4["nodes"], axis=1), "total_graph_nodes_visited"),
        **summarize(np.sum(e4["worker_cpu"], axis=1), "total_worker_cpu_us"),
        **summarize(np.sum(e4["response_bytes"], axis=1), "total_response_bytes"),
        "closed_loop_qps_secondary": float(query_summary["completed_qps"]),
        "run_id": manifest["run_id"],
        "manifest_path": str(Path(manifest["per_query_path"]).parent / "manifest.json"),
        "manifest_sha256": sha256_path(Path(manifest["per_query_path"]).parent / "manifest.json"),
        "per_query_sha256": manifest["per_query_sha256"],
        "e3_per_query_sha256": manifest["e3_per_query_sha256"],
    }
    return (
        summary,
        e3_rows,
        fixed_rows,
        balance_rows,
        list(manifest["_resource_phase_rows"]),
        query_metrics,
    )


def paired_bootstrap(
    left: np.ndarray,
    right: np.ndarray,
    *,
    seed: int,
    replicates: int,
    batch_size: int = 100,
) -> tuple[float, float, float]:
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    if left_array.shape != right_array.shape:
        raise ValueError("paired bootstrap inputs must have identical shapes")
    difference = left_array - right_array
    if difference.ndim != 1 or not len(difference) or not np.all(np.isfinite(difference)):
        raise ValueError("paired bootstrap requires equal finite one-dimensional arrays")
    rng = np.random.default_rng(seed)
    samples = np.empty(replicates, dtype=np.float64)
    for start in range(0, replicates, batch_size):
        stop = min(start + batch_size, replicates)
        indices = rng.integers(0, len(difference), size=(stop - start, len(difference)))
        samples[start:stop] = np.mean(difference[indices], axis=1)
    return (
        float(np.mean(difference)),
        float(np.percentile(samples, 2.5)),
        float(np.percentile(samples, 97.5)),
    )


def bootstrap_rows(
    per_query: dict[tuple[str, str, int], dict[str, np.ndarray]],
    *,
    seed: int,
    replicates: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset_index, dataset in enumerate(DATASETS):
        for logical_shards in LOGICAL_SHARDS:
            orion = per_query[(dataset, "orion", logical_shards)]
            kmeans = per_query[(dataset, "kmeans", logical_shards)]
            if not np.array_equal(orion["query_id"], kmeans["query_id"]):
                raise ValueError(f"paired query IDs differ: {dataset} M={logical_shards}")
            for metric_index, (metric, direction) in enumerate(BOOTSTRAP_METRICS.items()):
                observed, low, high = paired_bootstrap(
                    orion[metric],
                    kmeans[metric],
                    seed=seed + dataset_index * 100_000 + logical_shards * 100 + metric_index,
                    replicates=replicates,
                )
                if direction == "lower_favors_orion":
                    conclusion = (
                        "ORION_FAVORED"
                        if high < 0
                        else "KMEANS_FAVORED"
                        if low > 0
                        else "INCONCLUSIVE"
                    )
                else:
                    conclusion = (
                        "ORION_FAVORED"
                        if low > 0
                        else "KMEANS_FAVORED"
                        if high < 0
                        else "INCONCLUSIVE"
                    )
                orion_mean = float(np.mean(orion[metric]))
                kmeans_mean = float(np.mean(kmeans[metric]))
                rows.append(
                    {
                        "dataset": dataset,
                        "logical_shards": logical_shards,
                        "metric": metric,
                        "difference_definition": "Orion minus K-Means",
                        "favorable_direction": direction,
                        "orion_mean": orion_mean,
                        "kmeans_mean": kmeans_mean,
                        "mean_difference": observed,
                        "relative_difference_vs_kmeans": observed / kmeans_mean
                        if kmeans_mean != 0
                        else "",
                        "bootstrap_ci95_low": low,
                        "bootstrap_ci95_high": high,
                        "paired_query_count": len(orion[metric]),
                        "bootstrap_replicates": replicates,
                        "bootstrap_seed": seed
                        + dataset_index * 100_000
                        + logical_shards * 100
                        + metric_index,
                        "conclusion": conclusion,
                    }
                )
    return rows


def build_ablation(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    indexed = {
        (row["dataset"], row["partition_method"], int(row["logical_shards"])): row
        for row in summary_rows
    }
    rows = []
    for dataset in DATASETS:
        for logical_shards in LOGICAL_SHARDS:
            full = indexed[(dataset, "orion", logical_shards)]
            no_refinement = indexed[(dataset, "orion_no_refinement", logical_shards)]
            kmeans = indexed[(dataset, "kmeans", logical_shards)]
            rows.append(
                {
                    "dataset": dataset,
                    "logical_shards": logical_shards,
                    "twcut_full_minus_no_refinement": full["traversal_weighted_edge_cut"]
                    - no_refinement["traversal_weighted_edge_cut"],
                    "local_recall_full_minus_no_refinement": full[
                        "common_ef_local_target_recall_mean"
                    ]
                    - no_refinement["common_ef_local_target_recall_mean"],
                    "P_HNSW_full_minus_no_refinement": full["P_HNSW_mean"]
                    - no_refinement["P_HNSW_mean"],
                    "W90_full_minus_no_refinement": full["W90_mean"]
                    - no_refinement["W90_mean"],
                    "twcut_full_minus_kmeans": full["traversal_weighted_edge_cut"]
                    - kmeans["traversal_weighted_edge_cut"],
                    "P_HNSW_full_minus_kmeans": full["P_HNSW_mean"]
                    - kmeans["P_HNSW_mean"],
                    "W90_full_minus_kmeans": full["W90_mean"] - kmeans["W90_mean"],
                }
            )
    return rows


def build_correlations(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = [row for row in summary_rows if int(row["logical_shards"]) > 1]
    relations = (
        (
            "topology_disruption_vs_local_target_recall",
            "traversal_weighted_edge_cut",
            "common_ef_local_target_recall_mean",
            "negative",
        ),
        (
            "topology_disruption_vs_local_work",
            "traversal_weighted_edge_cut",
            "common_ef_local_distance_computations_mean",
            "positive",
        ),
        (
            "local_target_recall_vs_hnsw_fanout",
            "common_ef_local_target_recall_mean",
            "P_HNSW_mean",
            "negative",
        ),
        (
            "topology_disruption_vs_hnsw_fanout",
            "traversal_weighted_edge_cut",
            "P_HNSW_mean",
            "positive",
        ),
        (
            "exact_fanout_vs_hnsw_fanout",
            "P_exact_mean",
            "P_HNSW_mean",
            "positive",
        ),
        (
            "hnsw_fanout_vs_work_to_90",
            "P_HNSW_mean",
            "W90_mean",
            "positive",
        ),
    )
    rows = []
    for scope in ("pooled", *DATASETS):
        scoped = selected if scope == "pooled" else [row for row in selected if row["dataset"] == scope]
        for relation, x_metric, y_metric, expected_direction in relations:
            x = np.asarray([float(row[x_metric]) for row in scoped], dtype=np.float64)
            y = np.asarray([float(row[y_metric]) for row in scoped], dtype=np.float64)
            pearson = pearsonr(x, y)
            spearman = spearmanr(x, y)
            rows.append(
                {
                    "scope": scope,
                    "relation": relation,
                    "x_metric": x_metric,
                    "y_metric": y_metric,
                    "expected_direction_under_claim": expected_direction,
                    "configuration_count": len(scoped),
                    "pearson_r": float(pearson.statistic),
                    "pearson_p_value": float(pearson.pvalue),
                    "spearman_rho": float(spearman.statistic),
                    "spearman_p_value": float(spearman.pvalue),
                    "interpretation": "configuration-level consistency diagnostic; not causal proof",
                }
            )
    return rows


def bootstrap_lookup(rows: list[dict[str, Any]]) -> dict[tuple[str, int, str], dict[str, Any]]:
    return {
        (str(row["dataset"]), int(row["logical_shards"]), str(row["metric"])): row
        for row in rows
    }


def build_verdicts(
    summary_rows: list[dict[str, Any]],
    fixed_rows: list[dict[str, Any]],
    bootstrap: list[dict[str, Any]],
    ablation: list[dict[str, Any]],
) -> dict[str, Any]:
    summary = {
        (str(row["dataset"]), str(row["partition_method"]), int(row["logical_shards"])): row
        for row in summary_rows
    }
    fixed = {
        (
            str(row["dataset"]),
            str(row["partition_method"]),
            int(row["logical_shards"]),
            float(row["local_recall_threshold"]),
        ): row
        for row in fixed_rows
    }
    ci = bootstrap_lookup(bootstrap)
    per_dataset: dict[str, Any] = {}
    c2_structural_contradictions = 0
    c2_structural_support = 0
    c2_by_dataset: dict[str, dict[str, int]] = {}
    c3_support = 0
    c3_contradictions = 0
    c3_points: list[dict[str, Any]] = []
    for dataset in DATASETS:
        points = []
        dataset_twcut_support = 0
        dataset_twcut_contradictions = 0
        for logical_shards in LOGICAL_SHARDS:
            orion = summary[(dataset, "orion", logical_shards)]
            kmeans = summary[(dataset, "kmeans", logical_shards)]
            twcut_difference = float(orion["traversal_weighted_edge_cut"]) - float(
                kmeans["traversal_weighted_edge_cut"]
            )
            if twcut_difference < 0:
                c2_structural_support += 1
                dataset_twcut_support += 1
            elif twcut_difference > 0:
                c2_structural_contradictions += 1
                dataset_twcut_contradictions += 1
            local_ci = ci[(dataset, logical_shards, "common_ef_mean_local_target_recall")]
            orion_fixed = fixed[(dataset, "orion", logical_shards, 0.90)]
            kmeans_fixed = fixed[(dataset, "kmeans", logical_shards, 0.90)]
            fixed_success_difference = float(orion_fixed["success_fraction"]) - float(
                kmeans_fixed["success_fraction"]
            )
            orion_fixed_work = orion_fixed["required_distance_computations_mean"]
            kmeans_fixed_work = kmeans_fixed["required_distance_computations_mean"]
            fixed_work_difference: float | None = None
            if orion_fixed_work != "" and kmeans_fixed_work != "":
                fixed_work_difference = float(orion_fixed_work) - float(kmeans_fixed_work)
            fixed_recall_work_favors_orion = (
                fixed_success_difference >= 0
                and fixed_work_difference is not None
                and fixed_work_difference < 0
            )
            p_hnsw_ci = ci[(dataset, logical_shards, "P_HNSW")]
            w90_ci = ci[(dataset, logical_shards, "W90")]
            imbalance = float(orion["max_size_over_mean"])
            severe_imbalance = imbalance >= 2.0
            w90_relative = float(w90_ci["relative_difference_vs_kmeans"])
            material_w90_regression = (
                float(w90_ci["bootstrap_ci95_low"]) > 0 and w90_relative >= 0.05
            )
            c3_point_supported = (
                p_hnsw_ci["conclusion"] == "ORION_FAVORED"
                and not material_w90_regression
                and not severe_imbalance
            )
            c3_point_contradicted = p_hnsw_ci["conclusion"] == "KMEANS_FAVORED"
            c3_support += int(c3_point_supported)
            c3_contradictions += int(c3_point_contradicted)
            point = {
                "logical_shards": logical_shards,
                "orion_minus_kmeans_twcut": twcut_difference,
                "local_recall_bootstrap_conclusion": local_ci["conclusion"],
                "fixed_recall_090_success_fraction_difference": fixed_success_difference,
                "fixed_recall_090_required_work_difference": fixed_work_difference,
                "fixed_recall_090_favors_orion": fixed_recall_work_favors_orion,
                "P_HNSW_bootstrap_conclusion": p_hnsw_ci["conclusion"],
                "P_HNSW_ci95": [
                    p_hnsw_ci["bootstrap_ci95_low"],
                    p_hnsw_ci["bootstrap_ci95_high"],
                ],
                "W90_relative_difference_vs_kmeans": w90_relative,
                "material_W90_regression": material_w90_regression,
                "orion_max_size_over_mean": imbalance,
                "severe_imbalance_threshold": 2.0,
                "severe_imbalance": severe_imbalance,
                "C3_point_supported": c3_point_supported,
                "C3_point_contradicted": c3_point_contradicted,
            }
            points.append(point)
            c3_points.append({"dataset": dataset, **point})
        c2_by_dataset[dataset] = {
            "orion_lower_twcut_points": dataset_twcut_support,
            "kmeans_lower_twcut_points": dataset_twcut_contradictions,
        }
        per_dataset[dataset] = {
            "C2_structural_counts": c2_by_dataset[dataset],
            "points": points,
        }
    c2_total = len(DATASETS) * len(LOGICAL_SHARDS)
    if c2_structural_support == c2_total:
        local_conclusions = [
            ci[(dataset, logical_shards, "common_ef_mean_local_target_recall")]["conclusion"]
            for dataset in DATASETS
            for logical_shards in LOGICAL_SHARDS
        ]
        fixed_work_conclusions = [
            point["fixed_recall_090_favors_orion"]
            for dataset in DATASETS
            for point in per_dataset[dataset]["points"]
        ]
        c2_status = (
            "SUPPORTED"
            if all(value == "ORION_FAVORED" for value in local_conclusions)
            and all(fixed_work_conclusions)
            else "INCONCLUSIVE"
        )
    elif all(
        c2_by_dataset[dataset]["kmeans_lower_twcut_points"] >= 4
        for dataset in DATASETS
    ):
        c2_status = "CONTRADICTED"
    else:
        c2_status = "INCONCLUSIVE"
    c3_total = len(c3_points)
    if c3_support == c3_total:
        c3_status = "SUPPORTED"
    elif c3_contradictions and not c3_support:
        c3_status = "CONTRADICTED"
    else:
        c3_status = "INCONCLUSIVE"
    refinement_topology_improvements = sum(
        float(row["twcut_full_minus_no_refinement"]) < 0 for row in ablation
    )
    refinement_fanout_improvements = sum(
        float(row["P_HNSW_full_minus_no_refinement"]) < 0 for row in ablation
    )
    return {
        "decision_rules": {
            "C2": (
                "SUPPORTED only when Orion has lower TWCut at every dataset/M point and "
                "paired common-ef local recall plus 0.90 fixed-recall work favor Orion at "
                "every point; CONTRADICTED when K-Means has lower TWCut at at least four "
                "scales in each dataset; otherwise INCONCLUSIVE."
            ),
            "C3": (
                "SUPPORTED only when every dataset/M point has a paired P_HNSW CI favoring "
                "Orion, no >=5% W90 regression with CI above zero, and Orion max/mean shard "
                "size below 2.0; CONTRADICTED when at least one point favors K-Means and no "
                "point satisfies the full support rule; otherwise INCONCLUSIVE."
            ),
        },
        "dataset_evidence": per_dataset,
        "C2": {
            "status": c2_status,
            "orion_lower_twcut_points": c2_structural_support,
            "kmeans_lower_twcut_points": c2_structural_contradictions,
            "total_points": c2_total,
        },
        "C3": {
            "status": c3_status,
            "fully_supporting_points": c3_support,
            "kmeans_favored_P_HNSW_points": c3_contradictions,
            "total_points": c3_total,
        },
        "ablation": {
            "topology_refinement_improves_twcut_points": refinement_topology_improvements,
            "topology_refinement_improves_P_HNSW_points": refinement_fanout_improvements,
            "total_points": len(ablation),
        },
    }


def validate_variance(
    variance_root: Path,
    manifests: dict[tuple[str, str, int], dict[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for dataset in DATASETS:
        root = variance_root / dataset
        if not (root / "manifest.json").is_file():
            root = root / "variance-v1"
        manifest_path = root / "manifest.json"
        metrics_path = root / "variance_metrics.csv"
        summary_path = root / "variance_summary.csv"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        metrics = read_csv(metrics_path)
        summary = read_csv(summary_path)
        if (
            manifest.get("dataset") != dataset
            or int(manifest.get("logical_shards") or 0) != 32
            or int(manifest.get("configuration_count") or 0) != 12
            or int(manifest.get("seed_count_per_method") or 0) != 3
        ):
            raise ValueError(f"variance manifest contract mismatch for {dataset}")
        counts = {
            method: sum(row["method"] == method for row in metrics) for method in METHODS
        }
        if counts != {method: 3 for method in METHODS} or len(summary) != 4:
            raise ValueError(f"variance coverage mismatch for {dataset}: {counts}")
        if any(value != "PASS" for value in (manifest.get("checks") or {}).values()):
            raise ValueError(f"variance audit checks failed for {dataset}")
        if sha256_path(metrics_path) != manifest["files"]["metrics_sha256"]:
            raise ValueError(f"variance metrics checksum mismatch for {dataset}")
        if sha256_path(summary_path) != manifest["files"]["summary_sha256"]:
            raise ValueError(f"variance summary checksum mismatch for {dataset}")
        summary_index = {row["method"]: row for row in summary}
        for method in METHODS:
            selected = [row for row in metrics if row["method"] == method]
            if len({row["seed_label"] for row in selected}) != 3:
                raise ValueError(f"variance seed labels are not unique: {dataset} {method}")
            for row in selected:
                artifact = Path(row["artifact_path"]).expanduser().resolve()
                if (
                    int(row["logical_shards"]) != 32
                    or not artifact.is_file()
                    or sha256_path(artifact) != row["artifact_sha256"]
                ):
                    raise ValueError(
                        f"variance partition artifact failed audit: {dataset} {method}"
                    )
            current_primary_sha = str(
                manifests[(dataset, method, 32)]["partition"]["artifact_sha256"]
            )
            if sum(row["artifact_sha256"] == current_primary_sha for row in selected) != 1:
                raise ValueError(
                    f"variance primary seed does not bind to Stage 13 M=32 partition: "
                    f"{dataset} {method}"
                )
            expected_summary = summary_index[method]
            for metric in (
                "edge_cut_ratio",
                "traversal_weighted_edge_cut",
                "shard_size_cv",
                "max_size_over_mean",
            ):
                values = np.asarray([float(row[metric]) for row in selected])
                reproduced = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "range": float(np.ptp(values)),
                }
                for suffix, value in reproduced.items():
                    if not close(value, float(expected_summary[f"{metric}_{suffix}"])):
                        raise ValueError(
                            f"variance summary does not reproduce: "
                            f"{dataset} {method} {metric}_{suffix}"
                        )
        result[dataset] = {
            "manifest": str(manifest_path),
            "manifest_sha256": sha256_path(manifest_path),
            "metrics": str(metrics_path),
            "metrics_sha256": sha256_path(metrics_path),
            "summary": str(summary_path),
            "summary_sha256": sha256_path(summary_path),
            "seed_count_per_method": 3,
            "checks": {
                "coverage": "PASS",
                "artifact_checksums": "PASS",
                "summary_reproduction": "PASS",
                "primary_seed_stage13_binding": "PASS",
            },
        }
    return result


def plot_lines(
    axis: plt.Axes,
    rows: list[dict[str, Any]],
    dataset: str,
    metric: str,
    *,
    methods: Iterable[str] = METHODS,
) -> None:
    for method in methods:
        selected = sorted(
            [
                row
                for row in rows
                if row["dataset"] == dataset
                and row["partition_method"] == method
                and int(row["logical_shards"]) > 1
            ],
            key=lambda row: int(row["logical_shards"]),
        )
        axis.plot(
            [int(row["logical_shards"]) for row in selected],
            [float(row[metric]) for row in selected],
            color=METHOD_COLORS[method],
            marker=METHOD_MARKERS[method],
            linewidth=1.7,
            markersize=4.5,
            label=METHOD_LABELS[method],
        )
    axis.set_xscale("log", base=2)
    axis.set_xticks(LOGICAL_SHARDS)
    axis.set_xticklabels([str(value) for value in LOGICAL_SHARDS])
    axis.set_xlabel("Logical shards M")
    axis.grid(True, alpha=0.25)


def save_figure(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def generate_figures(
    summary: list[dict[str, Any]],
    e3: list[dict[str, Any]],
    bootstrap: list[dict[str, Any]],
    output: Path,
) -> list[str]:
    output.mkdir(parents=True, exist_ok=True)
    figures: list[str] = []
    specifications = (
        ("fig1_twcut.pdf", "traversal_weighted_edge_cut", "Traversal-weighted edge cut"),
        ("fig2_common_ef_local_recall.pdf", "common_ef_local_target_recall_mean", "Mean local target recall"),
        ("fig3_exact_fanout.pdf", "P_exact_mean", "Mean exact oracle fan-out"),
        ("fig4_hnsw_fanout.pdf", "P_HNSW_mean", "Mean HNSW oracle fan-out"),
        ("fig5_work_to_90.pdf", "W90_mean", "Mean distance computations to 90% recall"),
        ("fig6_closed_loop_qps.pdf", "closed_loop_qps_secondary", "Closed-loop QPS (secondary)"),
        ("fig7_full_fanout_wall.pdf", "full_fanout_wall_us_mean", "Mean full-fan-out wall time (us)"),
        ("fig8_partition_imbalance.pdf", "max_size_over_mean", "Max shard size / mean"),
    )
    for filename, metric, ylabel in specifications:
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
        for axis, dataset in zip(axes, DATASETS, strict=True):
            plot_lines(axis, summary, dataset, metric)
            axis.set_title(DATASET_LABELS[dataset])
            axis.set_ylabel(ylabel)
        axes[0].legend(fontsize=8)
        path = output / filename
        save_figure(fig, path)
        figures.append(filename)
    fig, axes = plt.subplots(2, 5, figsize=(17, 7.3), constrained_layout=True)
    for row_index, dataset in enumerate(DATASETS):
        for column_index, logical_shards in enumerate(LOGICAL_SHARDS):
            axis = axes[row_index, column_index]
            for method in METHODS:
                selected = sorted(
                    [
                        row
                        for row in e3
                        if row["dataset"] == dataset
                        and row["partition_method"] == method
                        and int(row["logical_shards"]) == logical_shards
                    ],
                    key=lambda row: int(row["ef_search"]),
                )
                axis.plot(
                    [float(row["distance_computations_mean"]) for row in selected],
                    [float(row["local_target_recall_mean"]) for row in selected],
                    color=METHOD_COLORS[method],
                    marker=METHOD_MARKERS[method],
                    linewidth=1.2,
                    markersize=3.1,
                    label=METHOD_LABELS[method],
                )
            axis.set_title(f"{DATASET_LABELS[dataset]}, M={logical_shards}", fontsize=9)
            axis.grid(True, alpha=0.25)
            if column_index == 0:
                axis.set_ylabel("Local target recall")
            if row_index == 1:
                axis.set_xlabel("Distance computations")
    axes[0, 0].legend(fontsize=7)
    path = output / "fig9_local_navigability_curves.pdf"
    save_figure(fig, path)
    figures.append(path.name)
    p_hnsw_rows = [row for row in bootstrap if row["metric"] == "P_HNSW"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    for axis, dataset in zip(axes, DATASETS, strict=True):
        rows = sorted(
            [row for row in p_hnsw_rows if row["dataset"] == dataset],
            key=lambda row: int(row["logical_shards"]),
        )
        x = np.asarray([int(row["logical_shards"]) for row in rows])
        y = np.asarray([float(row["mean_difference"]) for row in rows])
        low = np.asarray([float(row["bootstrap_ci95_low"]) for row in rows])
        high = np.asarray([float(row["bootstrap_ci95_high"]) for row in rows])
        axis.errorbar(x, y, yerr=np.vstack([y - low, high - y]), marker="o", capsize=3)
        axis.axhline(0, color="#111827", linewidth=1)
        axis.set_xscale("log", base=2)
        axis.set_xticks(LOGICAL_SHARDS)
        axis.set_xticklabels([str(value) for value in LOGICAL_SHARDS])
        axis.set_xlabel("Logical shards M")
        axis.set_ylabel("Orion - K-Means mean P_HNSW")
        axis.set_title(DATASET_LABELS[dataset])
        axis.grid(True, alpha=0.25)
    path = output / "fig10_p_hnsw_paired_ci.pdf"
    save_figure(fig, path)
    figures.append(path.name)
    return figures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-root",
        default="/proj/intelisys-PG0/exp/orion-distributed/c23-linear-20260824-v1",
    )
    parser.add_argument(
        "--legacy-analysis-root",
        default="/proj/intelisys-PG0/exp/orion-distributed/c23-20260821-v2/analysis",
    )
    parser.add_argument("--variance-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260824)
    args = parser.parse_args()
    if args.bootstrap_replicates != 10_000:
        raise ValueError("formal synthesis requires exactly 10,000 bootstrap replicates")
    raw_root = Path(args.raw_root).expanduser().resolve()
    legacy_analysis_root = Path(args.legacy_analysis_root).expanduser().resolve()
    variance_root = Path(args.variance_root).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite final synthesis: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=str(output.parent))
    )
    try:
        registry, manifests = load_registry(raw_root)
        stage_inputs = validate_stage_inputs(registry)
        e1, e2 = load_structural_rows(legacy_analysis_root)
        validate_structural_binding(manifests, e1, e2)
        variance = validate_variance(variance_root, manifests)
        summary_rows: list[dict[str, Any]] = []
        e3_rows: list[dict[str, Any]] = []
        fixed_rows: list[dict[str, Any]] = []
        balance_rows: list[dict[str, Any]] = []
        resource_phase_rows: list[dict[str, Any]] = []
        per_query: dict[tuple[str, str, int], dict[str, np.ndarray]] = {}
        for index, key in enumerate(sorted(manifests), 1):
            summary, e3, fixed, balance, resource_phases, metrics = analyze_configuration(
                key, manifests[key], e1, e2
            )
            summary_rows.append(summary)
            e3_rows.extend(e3)
            fixed_rows.extend(fixed)
            balance_rows.extend(balance)
            resource_phase_rows.extend(resource_phases)
            per_query[key] = metrics
            print(f"[{index}/42] audited {key}", flush=True)
        bootstrap = bootstrap_rows(
            per_query,
            seed=args.bootstrap_seed,
            replicates=args.bootstrap_replicates,
        )
        ablation = build_ablation(summary_rows)
        correlations = build_correlations(summary_rows)
        verdicts = build_verdicts(summary_rows, fixed_rows, bootstrap, ablation)
        tables = temporary / "tables"
        table_rows = {
            "configuration_summary.csv": summary_rows,
            "e3_local_navigability.csv": e3_rows,
            "fixed_recall_work.csv": fixed_rows,
            "paired_bootstrap_ci.csv": bootstrap,
            "partition_balance.csv": balance_rows,
            "resource_phase_counters.csv": resource_phase_rows,
            "orion_refinement_ablation.csv": ablation,
            "correlation_diagnostics.csv": correlations,
        }
        for name, rows in table_rows.items():
            write_csv(tables / name, rows)
        figure_names = generate_figures(summary_rows, e3_rows, bootstrap, temporary / "figures")
        write_json(temporary / "verdicts.json", verdicts)
        expected_counts = {
            "configuration_summary.csv": 42,
            "e3_local_navigability.csv": 42 * len(experiment.EF_GRID),
            "fixed_recall_work.csv": 42 * len(LOCAL_RECALL_THRESHOLDS),
            "paired_bootstrap_ci.csv": len(DATASETS)
            * len(LOGICAL_SHARDS)
            * len(BOOTSTRAP_METRICS),
            "partition_balance.csv": 498,
            "resource_phase_counters.csv": 498 * len(RESOURCE_PHASES),
            "orion_refinement_ablation.csv": 10,
            "correlation_diagnostics.csv": 18,
        }
        checks = {
            "preflight_smokes_calibrations_and_implementation_files_pass": "PASS",
            "matrix_has_exactly_42_passed_configurations": "PASS",
            "all_configuration_manifests_and_raw_checksums_pass": "PASS",
            "all_resource_contracts_equal_M_physical_cores_and_M_over_32_capacity": "PASS",
            "all_five_phase_cgroup_snapshots_have_monotonic_CPU_no_OOM_and_memory_within_cap": "PASS",
            "all_client_affinity_and_container_cleanup_restore_gates_pass": "PASS",
            "all_configurations_have_one_nonempty_HNSW_per_logical_shard": "PASS",
            "accepted_structural_metrics_bind_to_identical_partition_checksums": "PASS",
            "E3_common_ef_reproduces_E4_per_shard_measurements": "PASS",
            "P_exact_P_HNSW_Delta_P_and_W90_reproduce_from_raw_arrays": "PASS",
            "paired_bootstrap_uses_9000_queries_and_10000_replicates": "PASS"
            if all(
                int(row["paired_query_count"]) == 9000
                and int(row["bootstrap_replicates"]) == 10_000
                for row in bootstrap
            )
            else "FAIL",
            "M32_variance_has_three_seeds_per_method_per_dataset": "PASS",
            "table_row_counts_match_protocol": "PASS"
            if all(len(table_rows[name]) == count for name, count in expected_counts.items())
            else "FAIL",
            "ten_required_PDF_figures_are_valid": "PASS"
            if len(figure_names) == 10
            and all(
                (temporary / "figures" / name).stat().st_size > 1000
                and (temporary / "figures" / name).read_bytes()[:5] == b"%PDF-"
                for name in figure_names
            )
            else "FAIL",
            "C2_and_C3_have_independent_data_driven_verdicts": "PASS"
            if set(verdicts) >= {"C2", "C3"}
            else "FAIL",
        }
        failed = [name for name, status in checks.items() if status != "PASS"]
        if failed:
            raise ValueError(f"final synthesis audit failed: {failed}")
        files = {
            "tables": {
                name: {
                    "path": str(output / "tables" / name),
                    "sha256": sha256_path(tables / name),
                    "rows": len(rows),
                }
                for name, rows in table_rows.items()
            },
            "figures": {
                name: {
                    "path": str(output / "figures" / name),
                    "sha256": sha256_path(temporary / "figures" / name),
                }
                for name in figure_names
            },
            "verdicts": {
                "path": str(output / "verdicts.json"),
                "sha256": sha256_path(temporary / "verdicts.json"),
            },
        }
        structural_inputs = {
            dataset: {
                "e1_metrics": {
                    "path": str(legacy_analysis_root / dataset / "e1-v1/e1_metrics.csv"),
                    "sha256": sha256_path(
                        legacy_analysis_root / dataset / "e1-v1/e1_metrics.csv"
                    ),
                },
                "e2_summary": {
                    "path": str(legacy_analysis_root / dataset / "e2-v1/e2_summary.csv"),
                    "sha256": sha256_path(
                        legacy_analysis_root / dataset / "e2-v1/e2_summary.csv"
                    ),
                },
            }
            for dataset in DATASETS
        }
        audit = {
            "protocol_version": resources.PROTOCOL_VERSION,
            "timestamp": resources.utc_timestamp(),
            "record_type": "c23_linear_resource_final_synthesis",
            "git_commit": stage_inputs["git_commit"],
            "stage_inputs": stage_inputs,
            "raw_root": str(raw_root),
            "matrix_registry": str(raw_root / "matrix/registry.json"),
            "matrix_registry_sha256": sha256_path(raw_root / "matrix/registry.json"),
            "legacy_structural_analysis_root": str(legacy_analysis_root),
            "structural_inputs": structural_inputs,
            "structural_reuse_reason": (
                "The partition assignments and datasets are byte-identical; the Stage 13 "
                "intervention changes execution/resource isolation, so accepted E1/E2 metrics "
                "are reused only after per-configuration checksum binding."
            ),
            "variance": variance,
            "bootstrap_replicates": args.bootstrap_replicates,
            "bootstrap_seed": args.bootstrap_seed,
            "configuration_count": len(summary_rows),
            "checks": checks,
            "verdicts": verdicts,
            "files": files,
        }
        write_json(temporary / "final-audit.json", audit)
        os.replace(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps({"status": "PASS", "output": str(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
