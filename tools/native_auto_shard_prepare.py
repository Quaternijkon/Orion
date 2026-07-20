#!/usr/bin/env python3
"""Prepare one native numeric auto-shard collection through existing tools."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
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
    "tools/orion_native_layout.py",
    "tools/orion_native_runtime_profile.py",
}
ORION_RUNTIME_PROFILE_TOOL = "tools/orion_native_runtime_profile.py"
ORION_RUNTIME_PARAMETER_KEYS = (
    "generation",
    "upper_k",
    "upper_search_ef",
    "dynamic_ef_base",
    "dynamic_ef_factor",
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
    parser.add_argument("--full-scan-threshold", type=int, default=10)
    parser.add_argument("--indexing-threshold", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--request-timeout-secs", type=int, default=120)
    parser.add_argument("--smoke-limit", type=int, default=10)
    parser.add_argument("--transfer-timeout-secs", type=float, default=3600.0)
    parser.add_argument("--transfer-poll-interval-secs", type=float, default=1.0)
    parser.add_argument(
        "--cargo-runner",
        default=str(REPO_ROOT / "tools/cargo_in_docker.sh"),
    )
    parser.add_argument("--cargo-target-dir")
    parser.add_argument("--resume", action="store_true")
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
    build_parameters: dict[str, Any], artifact_payload: dict[str, Any]
) -> int:
    faithful_constants = {
        "initial_num_shards": 31,
        "sample_denominator": 32,
        "upper_sample_seed": 100,
        "upper_m": 32,
        "upper_ef_construction": 100,
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
) -> dict[str, Any]:
    derivation = build_manifest.get("derivation")
    if not isinstance(derivation, dict):
        raise RuntimeError("Orion runtime profile is missing derivation metadata")
    if derivation.get("format_version") != 1:
        raise RuntimeError("unsupported Orion runtime-profile derivation format")
    if derivation.get("kind") != "orion_runtime_profile":
        raise RuntimeError("Orion runtime-profile derivation kind mismatch")
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
    validate_faithful_orion_build_parameters(
        source_parameters,
        source_artifact_payload,
    )
    return source


def load_routed_layout(method: str, layout_path: str | Path) -> dict[str, Any]:
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
        else {"tools/simple_kmeans_native_layout.py"}
    )
    actual_tool = build_manifest.get("tool")
    if actual_tool not in expected_tools:
        raise RuntimeError(
            f"layout tool mismatch: expected one of={sorted(expected_tools)!r}, "
            f"actual={actual_tool!r}"
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
    if method == "orion":
        attachment_search_ef = validate_faithful_orion_build_parameters(
            build_parameters,
            artifact_payload,
        )
        if actual_tool == ORION_RUNTIME_PROFILE_TOOL:
            runtime_profile_source = validate_orion_runtime_profile_derivation(
                build_manifest,
                build_parameters,
                artifact_payload,
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
    if runtime_profile_source is not None:
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
                "Orion runtime-profile reused import payload binding mismatch: "
                f"{source_import_mismatches}"
            )
        reused_payloads = (build_manifest.get("derivation") or {}).get(
            "reused_payloads"
        )
        if not isinstance(reused_payloads, dict):
            raise RuntimeError("Orion runtime profile lacks reused payload proof")
        for proof_name, manifest_prefix in (
            ("vectors", "vectors"),
            ("assignments", "assignments"),
        ):
            proof = reused_payloads.get(proof_name)
            if not isinstance(proof, dict):
                raise RuntimeError(
                    f"Orion runtime profile lacks {proof_name} reuse proof"
                )
            expected_proof = {
                "destination_file": import_manifest.get(f"{manifest_prefix}_file"),
                "sha256": import_manifest.get(f"{manifest_prefix}_sha256"),
            }
            proof_mismatches = {
                key: {"expected": expected, "actual": proof.get(key)}
                for key, expected in expected_proof.items()
                if proof.get(key) != expected
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
                    f"Orion runtime-profile {proof_name} reuse proof mismatch: "
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
                "Orion runtime-profile formal evidence eligibility mismatch"
            )
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
        "generation": generation,
        "vector_schema": artifact["vector_schema"],
        "shard_count": artifact["shard_count"],
        "logical_point_count": artifact["logical_point_count"],
        "physical_point_count": artifact["physical_point_count"],
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
    full_scan_threshold: int,
    indexing_threshold: int,
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
    return [
        str(Path(args.cargo_runner).expanduser().resolve()),
        "run",
        "--release",
        "--example",
        "orion_numeric_shard_import",
        "--",
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

    layout: dict[str, Any] | None = None
    train = None
    dataset_proof: dict[str, Any] | None = None
    if args.method in ROUTED_METHODS:
        layout = load_routed_layout(args.method, args.layout_dir)
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
        full_scan_threshold=args.full_scan_threshold,
        indexing_threshold=args.indexing_threshold,
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

    placement_proof = experiment.move_numeric_shards_round_robin(
        args.base_url,
        args.collection,
        cluster_preflight["worker_peer_ids"],
        expected_shard_count=shard_count,
        timeout_sec=args.transfer_timeout_secs,
        poll_interval_sec=args.transfer_poll_interval_secs,
    )

    commands: list[dict[str, Any]] = []
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
            commands.append(
                run_command(
                    importer_command(
                        args,
                        topology,
                        Path(layout["import_manifest_path"]),
                    ),
                    env=environment,
                )
            )
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
    if args.method in ROUTED_METHODS:
        assert layout is not None
        commands.append(run_command(installer_command(args, layout)))

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
        full_scan_threshold=args.full_scan_threshold,
        indexing_threshold=args.indexing_threshold,
        expected_point_count=physical_count,
        allow_empty=False,
        allow_partial=False,
    )
    final_cluster = experiment.collection_cluster_info(args.base_url, args.collection)
    if final_cluster is None:
        raise RuntimeError("final collection cluster placement is unavailable")
    final_placement = experiment.validate_numeric_shard_round_robin_placement(
        final_info,
        final_cluster,
        cluster_preflight["worker_peer_ids"],
        shard_count,
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
            "full_scan_threshold": args.full_scan_threshold,
            "indexing_threshold": args.indexing_threshold,
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
        "standard_api_smoke": smoke_proof,
        "provenance_metadata": provenance_metadata,
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
