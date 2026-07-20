#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import method4_distributed_cluster as cluster_tool
from tools import qdrant_two_level_routing_experiment as experiment


ROUTED_METHODS = {"orion", "simple_kmeans"}
ORION_ROUTE_TRACE_SOURCE = "exact_offline_production_router_trace"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark Qdrant HashAll, native Orion, or native static Simple KMeans "
            "through ordinary coordinator Search/Query batch requests."
        )
    )
    parser.add_argument("--method", choices=("hash_all", "orion", "simple_kmeans"), required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--collection", required=True)
    parser.add_argument("--hdf5-path", required=True)
    parser.add_argument("--topology", required=True)
    parser.add_argument("--deployment-manifest")
    parser.add_argument("--artifact")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--warmup-query-count", type=int, default=100)
    parser.add_argument("--eval-query-count", type=int, default=10000)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--stability-repeats", type=int, default=3)
    parser.add_argument("--api", choices=("search", "query"), default="search")
    parser.add_argument(
        "--vector-distance",
        choices=("cosine", "euclid", "l2"),
        default="cosine",
    )
    parser.add_argument("--vector-name", default="")
    parser.add_argument("--hnsw-ef", type=int)
    parser.add_argument(
        "--orion-route-trace",
        action="store_true",
        help=(
            "Disabled by default. For Orion only, replay the normalized evaluation "
            "queries through the production OrionRouter outside the timed benchmark."
        ),
    )
    parser.add_argument(
        "--cargo-runner",
        default=str(REPO_ROOT / "tools/cargo_in_docker.sh"),
        help="Cargo-compatible runner used by --orion-route-trace.",
    )
    parser.add_argument(
        "--cargo-target-dir",
        help="Optional external CARGO_TARGET_DIR used by --orion-route-trace.",
    )
    parser.add_argument("--write-per-query-metrics", action="store_true")
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    positive = {
        "eval-query-count": args.eval_query_count,
        "top-k": args.top_k,
        "batch-size": args.batch_size,
        "stability-repeats": args.stability_repeats,
    }
    for name, value in positive.items():
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"--{name} must be a positive integer")
    if args.warmup_query_count < 0:
        raise ValueError("--warmup-query-count must be non-negative")
    if args.hnsw_ef is not None:
        if args.hnsw_ef <= 0:
            raise ValueError("--hnsw-ef must be positive")
        if args.method != "hash_all":
            raise ValueError("--hnsw-ef is only valid for --method hash_all")
    elif args.method == "hash_all":
        raise ValueError("--method hash_all requires an explicit --hnsw-ef")
    if args.method == "hash_all" and args.artifact:
        raise ValueError("--artifact is not valid for --method hash_all")
    if args.orion_route_trace and args.method != "orion":
        raise ValueError("--orion-route-trace is only valid for --method orion")


def create_output_directory(path: str | Path) -> Path:
    output = Path(path).expanduser().resolve()
    if output == REPO_ROOT or REPO_ROOT in output.parents:
        raise ValueError(
            f"benchmark output must be outside the repository to avoid large tracked results: {output}"
        )
    output.mkdir(parents=True, exist_ok=False)
    return output


def write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def load_dataset(
    hdf5_path: str | Path,
    warmup_query_count: int,
    eval_query_count: int,
    top_k: int,
    vector_distance: str,
) -> dict[str, Any]:
    path = Path(hdf5_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"HDF5 dataset not found: {path}")
    with experiment.h5py.File(path, "r") as handle:
        missing = [name for name in ("train", "test", "neighbors") if name not in handle]
        if missing:
            raise ValueError(f"HDF5 dataset is missing arrays: {missing}")
        train_shape = tuple(int(value) for value in handle["train"].shape)
        test_shape = tuple(int(value) for value in handle["test"].shape)
        neighbor_shape = tuple(int(value) for value in handle["neighbors"].shape)
        if len(train_shape) != 2 or len(test_shape) != 2 or len(neighbor_shape) != 2:
            raise ValueError("train, test, and neighbors must all be two-dimensional")
        if train_shape[1] != test_shape[1]:
            raise ValueError("train and test vector dimensions differ")
        if eval_query_count > test_shape[0] or warmup_query_count > test_shape[0]:
            raise ValueError("requested warmup/evaluation queries exceed the dataset test split")
        if top_k > neighbor_shape[1]:
            raise ValueError("--top-k exceeds the dataset ground-truth width")
        eval_queries = handle["test"][:eval_query_count].astype(
            experiment.np.float32, copy=True
        )
        eval_neighbors = handle["neighbors"][:eval_query_count, :top_k].astype(
            experiment.np.int64, copy=True
        )
        warmup_queries = handle["test"][:warmup_query_count].astype(
            experiment.np.float32, copy=True
        )
        warmup_neighbors = handle["neighbors"][:warmup_query_count, :top_k].astype(
            experiment.np.int64, copy=True
        )
    eval_queries = experiment.prepare_vectors_for_distance(eval_queries, vector_distance)
    warmup_queries = experiment.prepare_vectors_for_distance(
        warmup_queries, vector_distance
    )
    return {
        "path": path,
        "train_count": train_shape[0],
        "dimension": train_shape[1],
        "train_shape": list(train_shape),
        "test_shape": list(test_shape),
        "neighbors_shape": list(neighbor_shape),
        "eval_queries": eval_queries,
        "eval_neighbors": eval_neighbors,
        "warmup_queries": warmup_queries,
        "warmup_neighbors": warmup_neighbors,
    }


def collection_vector_schema(
    collection_info: dict[str, Any], vector_name: str
) -> dict[str, Any]:
    params = ((collection_info.get("config") or {}).get("params") or {})
    vectors = params.get("vectors") or {}
    if vector_name:
        vector = vectors.get(vector_name) if isinstance(vectors, dict) else None
    else:
        vector = vectors if isinstance(vectors, dict) and "size" in vectors else None
    if not isinstance(vector, dict):
        raise RuntimeError(
            f"collection does not contain the requested dense vector {vector_name!r}"
        )
    if vector.get("multivector_config") is not None:
        raise RuntimeError("native auto-shard benchmark does not accept multivectors")
    return {
        "vector_name": vector_name,
        "dimension": vector.get("size"),
        "distance": str(vector.get("distance") or ""),
        "datatype": str(vector.get("datatype") or "float32"),
    }


def live_policy_for_method(
    collection_info: dict[str, Any], method: str
) -> dict[str, Any] | None:
    config = collection_info.get("config") or {}
    policy = config.get("auto_shard_policy")
    if method == "hash_all":
        if policy is None:
            return None
        if isinstance(policy, dict) and str(policy.get("type") or "").lower() == "hash_all":
            return None
        raise RuntimeError(f"HashAll benchmark found non-HashAll policy: {policy}")
    if not isinstance(policy, dict):
        raise RuntimeError(f"{method} benchmark requires an explicit auto_shard_policy")
    expected_type = method
    actual_type = str(policy.get("type") or "").lower()
    if actual_type != expected_type:
        raise RuntimeError(
            f"collection policy type mismatch: expected={expected_type}, actual={actual_type}"
        )
    generation = policy.get("generation")
    if isinstance(generation, bool) or not isinstance(generation, int) or generation <= 0:
        raise RuntimeError(f"collection has invalid routing generation: {generation!r}")
    checksum = cluster_tool.normalize_sha256(str(policy.get("artifact_sha256") or ""))
    return {
        "type": expected_type,
        "generation": generation,
        "artifact_sha256": checksum,
    }


def resolve_artifact_path(
    explicit_path: str | None,
    deployment_manifest: dict[str, Any] | None,
    method: str,
    collection: str,
    policy: dict[str, Any] | None,
) -> Path | None:
    if method == "hash_all":
        return None
    if explicit_path:
        return Path(explicit_path).expanduser().resolve()
    assert policy is not None
    manifest_key = "orion_artifacts" if method == "orion" else "simple_kmeans_artifacts"
    for entry in (deployment_manifest or {}).get(manifest_key) or []:
        if not isinstance(entry, dict):
            continue
        if (
            str(entry.get("collection")) == collection
            and entry.get("generation") == policy["generation"]
            and str(entry.get("canonical_sha256") or "").lower()
            == policy["artifact_sha256"]
        ):
            source = Path(str(entry.get("source_path") or "")).expanduser()
            if source.is_file():
                return source.resolve()
    raise RuntimeError(
        f"--method {method} requires --artifact, or a matching deployment-manifest "
        "entry whose source_path is readable"
    )


def validate_artifact(
    method: str,
    artifact_path: Path | None,
    policy: dict[str, Any] | None,
    live_schema: dict[str, Any],
    shard_count: int,
    points_count: int,
    train_count: int,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    if method == "hash_all":
        return None, {
            "status": "not_applicable",
            "method": method,
        }
    assert artifact_path is not None and policy is not None
    if method == "orion":
        metadata = cluster_tool.validate_local_orion_artifact(
            artifact_path,
            policy["generation"],
            policy["artifact_sha256"],
        )
    else:
        metadata = cluster_tool.validate_local_simple_kmeans_artifact(
            artifact_path,
            policy["generation"],
            policy["artifact_sha256"],
        )
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    artifact_schema = metadata["vector_schema"]
    canonical_live_schema = {
        "vector_name": live_schema["vector_name"],
        "dimension": live_schema["dimension"],
        "distance": live_schema["distance"].lower(),
        "datatype": live_schema["datatype"].lower(),
    }
    canonical_artifact_schema = {
        "vector_name": str(artifact_schema.get("vector_name") or ""),
        "dimension": artifact_schema.get("dimension"),
        "distance": str(artifact_schema.get("distance") or "").lower(),
        "datatype": str(artifact_schema.get("datatype") or "float32").lower(),
    }
    mismatches: list[str] = []
    if canonical_artifact_schema != canonical_live_schema:
        mismatches.append(
            f"vector_schema artifact={canonical_artifact_schema} live={canonical_live_schema}"
        )
    if metadata["shard_count"] != shard_count:
        mismatches.append(
            f"shard_count artifact={metadata['shard_count']} live={shard_count}"
        )
    if metadata["logical_point_count"] != train_count:
        mismatches.append(
            f"logical_point_count artifact={metadata['logical_point_count']} dataset={train_count}"
        )
    if metadata["physical_point_count"] != points_count:
        mismatches.append(
            f"physical_point_count artifact={metadata['physical_point_count']} live={points_count}"
        )
    if mismatches:
        raise RuntimeError("artifact/live benchmark mismatch: " + "; ".join(mismatches))
    return payload, {
        "status": "verified",
        "path": str(artifact_path),
        "sha256": metadata["file_sha256"],
        "generation": metadata["generation"],
        "layout_sha256": metadata["layout_sha256"],
        "logical_point_count": metadata["logical_point_count"],
        "physical_point_count": metadata["physical_point_count"],
        "shard_count": metadata["shard_count"],
        "vector_schema": metadata["vector_schema"],
    }


def route_reporting(
    method: str,
    shard_count: int,
    artifact: dict[str, Any] | None,
    hnsw_ef: int | None,
    orion_route_trace: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if method == "hash_all":
        return {
            "visited_shards": shard_count,
            "visited_shards_source": "live_collection_static_all_shards",
            "hnsw_ef": hnsw_ef,
            "ef_sum_per_query": hnsw_ef * shard_count if hnsw_ef is not None else None,
            "ef_sum_source": "cli_scalar_times_shards" if hnsw_ef is not None else "unknown",
        }
    assert artifact is not None
    common = {
        "layout_sha256": artifact["layout_sha256"],
        "logical_point_count": artifact["logical_point_count"],
        "physical_point_count": artifact["physical_point_count"],
        "shard_count": artifact["shard_count"],
    }
    if method == "orion":
        upper_k = artifact.get("upper_k")
        dynamic_base = artifact.get("dynamic_ef_base")
        dynamic_factor = artifact.get("dynamic_ef_factor")
        if (
            isinstance(upper_k, bool)
            or not isinstance(upper_k, int)
            or upper_k <= 0
            or isinstance(dynamic_base, bool)
            or not isinstance(dynamic_base, int)
            or dynamic_base <= 0
            or isinstance(dynamic_factor, bool)
            or not isinstance(dynamic_factor, int)
            or dynamic_factor < 0
        ):
            raise ValueError("Orion artifact has invalid upper_k or Dynamic EF parameters")
        report = {
            **common,
            "upper_k": upper_k,
            "upper_ef_search": artifact.get("upper_ef_search"),
            "dynamic_ef_base": dynamic_base,
            "dynamic_ef_factor": dynamic_factor,
            "visited_shards": None,
            "visited_shards_source": "unknown_without_server_trace",
            "ef_sum_per_query": None,
            "ef_sum_source": "unknown_without_server_trace",
        }
        if orion_route_trace is not None:
            report.update(
                {
                    "visited_shards": orion_route_trace["visited_shards"],
                    "visited_shards_source": ORION_ROUTE_TRACE_SOURCE,
                    "ef_sum_per_query": orion_route_trace["ef_sum_per_query"],
                    "ef_sum_source": ORION_ROUTE_TRACE_SOURCE,
                }
            )
        return report
    nprobe = artifact.get("nprobe")
    lower_hnsw_ef = artifact.get("lower_hnsw_ef")
    return {
        **common,
        "nprobe": nprobe,
        "lower_hnsw_ef": lower_hnsw_ef,
        "visited_shards": nprobe,
        "visited_shards_source": "artifact_derived_static_nprobe",
        "ef_sum_per_query": nprobe * lower_hnsw_ef,
        "ef_sum_source": "artifact_derived_nprobe_times_lower_hnsw_ef",
    }


def effective_cargo_target_dir(args: argparse.Namespace) -> str | None:
    configured = args.cargo_target_dir or os.environ.get("CARGO_TARGET_DIR")
    if not configured:
        return None
    return str(Path(configured).expanduser().resolve())


def orion_route_trace_command(
    args: argparse.Namespace,
    artifact_path: Path,
    queries_path: Path,
    query_count: int,
    dimension: int,
    output_path: Path,
) -> list[str]:
    return [
        str(Path(args.cargo_runner).expanduser().resolve()),
        "run",
        "--release",
        "-p",
        "collection",
        "--example",
        "orion_route_trace",
        "--",
        str(artifact_path),
        str(queries_path),
        str(query_count),
        str(dimension),
        str(output_path),
    ]


def require_trace_object(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError(f"Orion route trace field {field} must be an object")
    return value


def require_trace_positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RuntimeError(
            f"Orion route trace field {field} must be a positive integer; got {value!r}"
        )
    return value


def require_trace_average(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"Orion route trace field {field} must be numeric; got {value!r}")
    average = float(value)
    if not math.isfinite(average) or average <= 0:
        raise RuntimeError(
            f"Orion route trace field {field} must be finite and positive; got {value!r}"
        )
    return average


def validate_orion_route_trace_output(
    trace: Any,
    *,
    artifact_proof: dict[str, Any],
    expected_canonical_sha256: str,
    expected_query_count: int,
    expected_dimension: int,
    expected_query_sha256: str,
) -> dict[str, Any]:
    root = require_trace_object(trace, "root")
    if root.get("format_version") != 1:
        raise RuntimeError(
            "unsupported Orion route trace format_version: "
            f"{root.get('format_version')!r}"
        )
    artifact = require_trace_object(root.get("artifact"), "artifact")
    queries = require_trace_object(root.get("queries"), "queries")
    aggregate = require_trace_object(root.get("aggregate"), "aggregate")

    expected_file_sha256 = cluster_tool.normalize_sha256(
        str(artifact_proof.get("sha256") or "")
    )
    expected_canonical_sha256 = cluster_tool.normalize_sha256(
        expected_canonical_sha256
    )
    actual_file_sha256 = cluster_tool.normalize_sha256(
        str(artifact.get("file_sha256") or "")
    )
    actual_canonical_sha256 = cluster_tool.normalize_sha256(
        str(artifact.get("sha256") or "")
    )
    actual_generation = require_trace_positive_int(
        artifact.get("generation"), "artifact.generation"
    )
    actual_shard_count = require_trace_positive_int(
        artifact.get("shard_count"), "artifact.shard_count"
    )
    actual_layout_sha256 = cluster_tool.normalize_sha256(
        str(artifact.get("layout_sha256") or "")
    )
    expected_layout_sha256 = cluster_tool.normalize_sha256(
        str(artifact_proof.get("layout_sha256") or "")
    )
    artifact_checks = {
        "generation": (actual_generation, artifact_proof.get("generation")),
        "layout_sha256": (actual_layout_sha256, expected_layout_sha256),
        "file_sha256": (actual_file_sha256, expected_file_sha256),
        "canonical_sha256": (
            actual_canonical_sha256,
            expected_canonical_sha256,
        ),
        "shard_count": (actual_shard_count, artifact_proof.get("shard_count")),
    }
    artifact_mismatches = [
        f"{field}: actual={actual!r}, expected={expected!r}"
        for field, (actual, expected) in artifact_checks.items()
        if actual != expected
    ]
    if artifact_mismatches:
        raise RuntimeError(
            "Orion route trace artifact provenance mismatch: "
            + "; ".join(artifact_mismatches)
        )

    actual_query_count = require_trace_positive_int(
        queries.get("query_count"), "queries.query_count"
    )
    actual_dimension = require_trace_positive_int(
        queries.get("dimension"), "queries.dimension"
    )
    actual_aggregate_query_count = require_trace_positive_int(
        aggregate.get("query_count"), "aggregate.query_count"
    )
    query_checks = {
        "query_count": (actual_query_count, expected_query_count),
        "dimension": (actual_dimension, expected_dimension),
        "sha256": (
            cluster_tool.normalize_sha256(str(queries.get("sha256") or "")),
            cluster_tool.normalize_sha256(expected_query_sha256),
        ),
        "aggregate.query_count": (
            actual_aggregate_query_count,
            expected_query_count,
        ),
    }
    query_mismatches = [
        f"{field}: actual={actual!r}, expected={expected!r}"
        for field, (actual, expected) in query_checks.items()
        if actual != expected
    ]
    if query_mismatches:
        raise RuntimeError(
            "Orion route trace query provenance mismatch: "
            + "; ".join(query_mismatches)
        )

    visited_distribution = require_trace_object(
        aggregate.get("visited_shards"), "aggregate.visited_shards"
    )
    ef_distribution = require_trace_object(
        aggregate.get("ef_sum_per_query"), "aggregate.ef_sum_per_query"
    )
    visited_shards = require_trace_average(
        visited_distribution.get("average"), "aggregate.visited_shards.average"
    )
    ef_sum_per_query = require_trace_average(
        ef_distribution.get("average"), "aggregate.ef_sum_per_query.average"
    )
    if visited_shards > actual_shard_count:
        raise RuntimeError(
            "Orion route trace average visited shards exceeds artifact shard_count: "
            f"{visited_shards} > {actual_shard_count}"
        )
    return {
        "visited_shards": visited_shards,
        "ef_sum_per_query": ef_sum_per_query,
        "query_count": expected_query_count,
        "dimension": expected_dimension,
        "query_sha256": expected_query_sha256,
        "artifact_generation": artifact_proof["generation"],
        "artifact_layout_sha256": artifact_proof["layout_sha256"],
        "artifact_file_sha256": expected_file_sha256,
        "artifact_canonical_sha256": expected_canonical_sha256,
    }


def run_orion_route_trace(
    args: argparse.Namespace,
    *,
    artifact_path: Path,
    artifact_proof: dict[str, Any],
    policy: dict[str, Any],
    eval_queries: Any,
    output_dir: Path,
) -> dict[str, Any]:
    queries = experiment.np.asarray(eval_queries, dtype=experiment.np.float32)
    if queries.ndim != 2:
        raise ValueError("Orion route trace queries must be a two-dimensional array")
    if not bool(experiment.np.isfinite(queries).all()):
        raise ValueError("Orion route trace queries contain non-finite values")
    query_count, dimension = (int(value) for value in queries.shape)
    little_endian_queries = experiment.np.ascontiguousarray(
        queries, dtype=experiment.np.dtype("<f4")
    )
    query_bytes = little_endian_queries.tobytes(order="C")
    query_sha256 = hashlib.sha256(query_bytes).hexdigest()

    trace_path = output_dir / "orion_route_trace.json"
    stdout_path = output_dir / "orion_route_trace.stdout.log"
    stderr_path = output_dir / "orion_route_trace.stderr.log"
    query_path: Path | None = None
    proof: dict[str, Any] | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=".orion-route-queries-",
            suffix=".f32le",
            dir=output_dir,
            delete=False,
        ) as handle:
            handle.write(query_bytes)
            query_path = Path(handle.name)
        command = orion_route_trace_command(
            args,
            artifact_path,
            query_path,
            query_count,
            dimension,
            trace_path,
        )
        environment = os.environ.copy()
        cargo_target_dir = effective_cargo_target_dir(args)
        if cargo_target_dir:
            environment["CARGO_TARGET_DIR"] = cargo_target_dir
        try:
            result = subprocess.run(
                command,
                cwd=REPO_ROOT,
                text=True,
                capture_output=True,
                check=False,
                env=environment,
            )
        except OSError as exc:
            stdout_path.write_text("", encoding="utf-8")
            stderr_path.write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
            raise RuntimeError(f"failed to launch Orion route trace: {exc}") from exc
        stdout_path.write_text(result.stdout, encoding="utf-8")
        stderr_path.write_text(result.stderr, encoding="utf-8")
        if result.returncode != 0:
            raise RuntimeError(
                "Orion production-router trace failed with exit code "
                f"{result.returncode}; inspect {stderr_path}"
            )
        if not trace_path.is_file():
            raise RuntimeError(
                f"Orion production-router trace did not create {trace_path}"
            )
        try:
            trace = json.loads(trace_path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"invalid Orion route trace JSON {trace_path}: {exc}") from exc
        metrics = validate_orion_route_trace_output(
            trace,
            artifact_proof=artifact_proof,
            expected_canonical_sha256=policy["artifact_sha256"],
            expected_query_count=query_count,
            expected_dimension=dimension,
            expected_query_sha256=query_sha256,
        )
        proof = {
            "status": "verified",
            "source": ORION_ROUTE_TRACE_SOURCE,
            "included_in_timed_benchmark": False,
            "command": command,
            "cargo_target_dir": cargo_target_dir,
            "trace_path": str(trace_path),
            "stdout_log": str(stdout_path),
            "stderr_log": str(stderr_path),
            "metrics": metrics,
            "aggregate": trace["aggregate"],
        }
    finally:
        if query_path is not None:
            query_path.unlink(missing_ok=True)
    assert proof is not None
    proof["temporary_query_file_removed"] = True
    return proof


def evaluate_once(
    args: argparse.Namespace,
    queries: Any,
    neighbors: Any,
    *,
    include_per_query_metrics: bool,
) -> dict[str, Any]:
    return experiment.evaluate_standard_dense_vector_batches(
        args.base_url,
        args.collection,
        queries,
        neighbors,
        args.top_k,
        args.batch_size,
        api=args.api,
        vector_name=args.vector_name,
        hnsw_ef=args.hnsw_ef,
        include_per_query_metrics=include_per_query_metrics,
    )


def mean_and_stdev(rows: list[dict[str, Any]], field: str) -> tuple[float, float]:
    values = [float(row[field]) for row in rows]
    return statistics.fmean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def validate_repository_binding(
    repository: dict[str, Any], deployment_manifest: dict[str, Any] | None
) -> dict[str, Any]:
    benchmark_commit = repository.get("commit")
    if not benchmark_commit:
        raise RuntimeError("benchmark repository commit provenance is unavailable")
    tracked_dirty = repository.get("tracked_dirty", repository.get("dirty"))
    if tracked_dirty is not False:
        raise RuntimeError(
            "benchmark repository has tracked changes; commit or revert them before timing"
        )
    deployment_repository = (deployment_manifest or {}).get("repository") or {}
    deployment_commit = deployment_repository.get("commit")
    if deployment_manifest is not None and not deployment_commit:
        raise RuntimeError("deployment manifest is missing repository commit provenance")
    if deployment_commit is not None and deployment_commit != benchmark_commit:
        raise RuntimeError(
            "deployment/benchmark commit mismatch: "
            f"deployment={deployment_commit}, benchmark={benchmark_commit}"
        )
    return {
        "deployment_commit": deployment_commit,
        "benchmark_commit": benchmark_commit,
        "tracked_dirty": False,
        "untracked_entry_count": repository.get("untracked_entry_count"),
    }


def run(args: argparse.Namespace) -> Path:
    validate_args(args)
    distance = experiment.vector_distance_config(args.vector_distance)
    topology = experiment.load_cluster_topology(args.topology)
    cluster_preflight = experiment.validate_cluster_preflight(args.base_url, topology)
    deployment_manifest = experiment.load_optional_json(args.deployment_manifest)
    repository = experiment.repository_provenance(REPO_ROOT)
    repository_binding = validate_repository_binding(repository, deployment_manifest)
    dataset = load_dataset(
        args.hdf5_path,
        args.warmup_query_count,
        args.eval_query_count,
        args.top_k,
        distance["name"],
    )

    collection_info = experiment.collection_info(args.base_url, args.collection)
    initial_points_count = collection_info.get("points_count")
    if isinstance(initial_points_count, bool) or not isinstance(initial_points_count, int):
        raise RuntimeError(
            f"collection has invalid points_count: {initial_points_count!r}"
        )
    collection_info = experiment.wait_collection_indexed(
        args.base_url,
        args.collection,
        initial_points_count,
    )
    collection_cluster = experiment.collection_cluster_info(args.base_url, args.collection)
    if collection_cluster is None:
        raise RuntimeError("collection cluster placement is unavailable")
    status = str(collection_info.get("status") or "").lower()
    optimizer_status = collection_info.get("optimizer_status")
    optimizer_ok = optimizer_status == "ok" or (
        isinstance(optimizer_status, dict) and optimizer_status.get("ok") is True
    )
    if status != "green" or not optimizer_ok:
        raise RuntimeError(
            f"collection is not benchmark-ready: status={status!r}, "
            f"optimizer_status={optimizer_status!r}"
        )
    update_queue = collection_info.get("update_queue") or {}
    if int(update_queue.get("length") or 0) != 0:
        raise RuntimeError(
            f"collection update queue is not empty: {update_queue.get('length')!r}"
        )
    params = ((collection_info.get("config") or {}).get("params") or {})
    shard_count = params.get("shard_number")
    points_count = collection_info.get("points_count")
    if isinstance(shard_count, bool) or not isinstance(shard_count, int) or shard_count <= 0:
        raise RuntimeError(f"collection has invalid shard_number: {shard_count!r}")
    if isinstance(points_count, bool) or not isinstance(points_count, int):
        raise RuntimeError(f"collection has invalid points_count: {points_count!r}")
    live_schema = collection_vector_schema(collection_info, args.vector_name)
    expected_distance = distance["qdrant_distance"].lower()
    if live_schema["dimension"] != dataset["dimension"]:
        raise RuntimeError(
            f"dataset/collection dimension mismatch: {dataset['dimension']} != "
            f"{live_schema['dimension']}"
        )
    if live_schema["distance"].lower() != expected_distance:
        raise RuntimeError(
            f"collection distance {live_schema['distance']!r} does not match "
            f"--vector-distance {args.vector_distance!r}"
        )

    policy = live_policy_for_method(collection_info, args.method)
    if args.method == "hash_all" and points_count != dataset["train_count"]:
        raise RuntimeError(
            f"HashAll points_count {points_count} does not match dataset train count "
            f"{dataset['train_count']}"
        )
    artifact_path = resolve_artifact_path(
        args.artifact,
        deployment_manifest,
        args.method,
        args.collection,
        policy,
    )
    artifact, artifact_proof = validate_artifact(
        args.method,
        artifact_path,
        policy,
        live_schema,
        shard_count,
        points_count,
        dataset["train_count"],
    )
    indexed_vectors_count = int(collection_info.get("indexed_vectors_count") or 0)
    indexing_readiness = {
        "points_count": points_count,
        "indexed_vectors_count": indexed_vectors_count,
        "fully_indexed": indexed_vectors_count >= points_count,
        "completion_mode": (
            "fully_indexed"
            if indexed_vectors_count >= points_count
            else "stable_small_segment_full_scan_exception"
        ),
        "status": collection_info.get("status"),
        "optimizer_status": collection_info.get("optimizer_status"),
        "update_queue": collection_info.get("update_queue"),
        "shard_transfers": collection_cluster.get("shard_transfers") or [],
    }
    placement_proof = experiment.validate_numeric_shard_round_robin_placement(
        collection_info,
        collection_cluster,
        cluster_preflight["worker_peer_ids"],
        shard_count,
    )
    route = route_reporting(args.method, shard_count, artifact, args.hnsw_ef)
    output_dir = create_output_directory(args.output_dir)
    route_trace_proof: dict[str, Any] | None = None
    if args.orion_route_trace:
        assert artifact_path is not None and policy is not None
        route_trace_proof = run_orion_route_trace(
            args,
            artifact_path=artifact_path,
            artifact_proof=artifact_proof,
            policy=policy,
            eval_queries=dataset["eval_queries"],
            output_dir=output_dir,
        )
        route = route_reporting(
            args.method,
            shard_count,
            artifact,
            args.hnsw_ef,
            route_trace_proof["metrics"],
        )

    stability_rows: list[dict[str, Any]] = []
    per_query_rows: list[dict[str, Any]] = []
    for repeat in range(1, args.stability_repeats + 1):
        if args.warmup_query_count:
            evaluate_once(
                args,
                dataset["warmup_queries"],
                dataset["warmup_neighbors"],
                include_per_query_metrics=False,
            )
        result = evaluate_once(
            args,
            dataset["eval_queries"],
            dataset["eval_neighbors"],
            include_per_query_metrics=args.write_per_query_metrics,
        )
        row = {
            "run": repeat,
            "method": args.method,
            "api": args.api,
            "query_count": result["query_count"],
            "top_k": args.top_k,
            "batch_size": args.batch_size,
            "recall_at_k": result["recall_at_k"],
            "qps": result["qps"],
            "wall_s": result["wall_s"],
            "latency_p50_ms": result["latency_p50_ms"],
            "latency_p95_ms": result["latency_p95_ms"],
            "latency_p99_ms": result["latency_p99_ms"],
            "visited_shards": route.get("visited_shards"),
            "visited_shards_source": route.get("visited_shards_source"),
            "ef_sum_per_query": route.get("ef_sum_per_query"),
            "ef_sum_source": route.get("ef_sum_source"),
        }
        stability_rows.append(row)
        for per_query in result.get("per_query_rows") or []:
            per_query_rows.append({"run": repeat, **per_query})

    recall_mean, recall_stdev = mean_and_stdev(stability_rows, "recall_at_k")
    qps_mean, qps_stdev = mean_and_stdev(stability_rows, "qps")
    latency_p50_mean, latency_p50_stdev = mean_and_stdev(
        stability_rows, "latency_p50_ms"
    )
    latency_p95_mean, latency_p95_stdev = mean_and_stdev(
        stability_rows, "latency_p95_ms"
    )
    latency_p99_mean, latency_p99_stdev = mean_and_stdev(
        stability_rows, "latency_p99_ms"
    )
    final_metrics = {
        "method": args.method,
        "api": args.api,
        "query_count": args.eval_query_count,
        "top_k": args.top_k,
        "batch_size": args.batch_size,
        "stability_repeats": args.stability_repeats,
        "recall_at_k": recall_mean,
        "recall_stdev": recall_stdev,
        "qps": qps_mean,
        "qps_stdev": qps_stdev,
        "latency_p50_ms": latency_p50_mean,
        "latency_p50_ms_stdev": latency_p50_stdev,
        "latency_p95_ms": latency_p95_mean,
        "latency_p95_ms_stdev": latency_p95_stdev,
        "latency_p99_ms": latency_p99_mean,
        "latency_p99_ms_stdev": latency_p99_stdev,
        "visited_shards": route.get("visited_shards"),
        "visited_shards_source": route.get("visited_shards_source"),
        "ef_sum_per_query": route.get("ef_sum_per_query"),
        "ef_sum_source": route.get("ef_sum_source"),
    }

    experiment.write_csv(output_dir / "stability_runs.csv", stability_rows)
    experiment.write_csv(output_dir / "final_metrics.csv", [final_metrics])
    if args.write_per_query_metrics:
        experiment.write_csv(output_dir / "per_query_metrics.csv", per_query_rows)

    dataset_manifest = {
        "path": str(dataset["path"]),
        "sha256": experiment.sha256_path(dataset["path"]),
        "size_bytes": dataset["path"].stat().st_size,
        "train_shape": dataset["train_shape"],
        "test_shape": dataset["test_shape"],
        "neighbors_shape": dataset["neighbors_shape"],
    }
    deployment_evidence = {
        "path": args.deployment_manifest,
        "image": (deployment_manifest or {}).get("image"),
        "repository": (deployment_manifest or {}).get("repository"),
        "nodes": (deployment_manifest or {}).get("nodes"),
    }
    run_manifest = {
        "schema_version": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "command": [sys.executable, *sys.argv],
        "method": args.method,
        "api": args.api,
        "base_url": args.base_url,
        "collection": args.collection,
        "request_contract": {
            "standard_coordinator_request": True,
            "shard_selector": False,
            "entry_point_hints": False,
            "per_shard_ef": False,
            "source_id_hint": False,
            "scalar_hnsw_ef": args.hnsw_ef,
        },
        "dataset": dataset_manifest,
        "repository": repository,
        "repository_binding": repository_binding,
        "deployment": deployment_evidence,
        "process_affinity": (
            sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
        ),
        "topology": topology,
        "cluster_preflight": cluster_preflight,
        "cluster_snapshot": cluster_preflight["raw"],
        "collection_info": collection_info,
        "collection_cluster": collection_cluster,
        "indexing_readiness": indexing_readiness,
        "placement_proof": placement_proof,
        "live_policy": policy,
        "artifact": artifact_proof,
        "orion_route_trace": route_trace_proof,
        "route_reporting": route,
        "parameters": vars(args),
        "files": {
            "stability_runs": "stability_runs.csv",
            "final_metrics": "final_metrics.csv",
            "per_query_metrics": (
                "per_query_metrics.csv" if args.write_per_query_metrics else None
            ),
            "orion_route_trace": (
                "orion_route_trace.json" if route_trace_proof is not None else None
            ),
            "orion_route_trace_stdout": (
                "orion_route_trace.stdout.log" if route_trace_proof is not None else None
            ),
            "orion_route_trace_stderr": (
                "orion_route_trace.stderr.log" if route_trace_proof is not None else None
            ),
        },
    }
    summary = {
        "method": args.method,
        "collection": args.collection,
        "final_metrics": final_metrics,
        "stability_runs": len(stability_rows),
        "route_reporting": route,
        "artifact_validation": artifact_proof["status"],
        "placement_valid": placement_proof["valid"],
        "limitations": (
            [
                "Orion visited-shard and EF-sum values are unknown without an exact route "
                "trace; artifact parameters are reported without claiming actual routing "
                "measurements."
            ]
            if args.method == "orion" and route_trace_proof is None
            else (
                [
                    "Orion routing-cost values are an exact offline replay through the "
                    "production OrionRouter over the verified artifact. The replay is outside "
                    "the timed benchmark and is not server or network instrumentation."
                ]
                if args.method == "orion"
                else []
            )
        ),
    }
    write_json(output_dir / "run_manifest.json", run_manifest)
    write_json(output_dir / "summary.json", summary)
    return output_dir


def main(argv: list[str] | None = None) -> int:
    output = run(parse_args(argv))
    print(json.dumps({"output_dir": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
