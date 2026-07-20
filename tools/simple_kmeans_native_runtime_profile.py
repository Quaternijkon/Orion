#!/usr/bin/env python3
"""Derive a Simple KMeans runtime profile from one immutable offline layout.

``simple_kmeans_native_layout.py`` trains centroids and assigns every point to
one numeric shard.  ``nprobe`` and ``lower_hnsw_ef`` are runtime controls, so a
fair sweep must not retrain KMeans for every operating point.  This tool copies
the exact centroid table, point assignments, and vector payload from one
checksum-verified production bundle and changes only:

* routing generation;
* ``nprobe``; and
* ``lower_hnsw_ef``.

The production artifact is passed through the Rust typed canonicalizer again.
Existing paths are never overwritten.  Copy mode creates independent payload
files and is eligible for formal evidence; hardlink mode is diagnostic only.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import struct
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import native_auto_shard_prepare as prepare  # noqa: E402
from tools import orion_native_layout as layout_common  # noqa: E402
from tools import simple_kmeans_native_layout as simple_layout  # noqa: E402


TOOL_NAME = "tools/simple_kmeans_native_runtime_profile.py"
DERIVATION_KIND = "simple_kmeans_runtime_profile"
DERIVATION_FORMAT_VERSION = 1
RUNTIME_PARAMETER_KEYS = (
    "generation",
    "nprobe",
    "lower_hnsw_ef",
)
ARTIFACT_RUNTIME_KEYS = set(RUNTIME_PARAMETER_KEYS)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Derive a native Simple KMeans runtime profile while reusing the "
            "exact centroids, assignments, and vector import payload."
        )
    )
    parser.add_argument("--source-layout-dir", required=True)
    parser.add_argument(
        "--output-dir",
        required=True,
        help="New output directory. Existing paths are never overwritten.",
    )
    parser.add_argument("--generation", required=True, type=int)
    parser.add_argument("--nprobe", required=True, type=int)
    parser.add_argument("--lower-hnsw-ef", required=True, type=int)
    parser.add_argument("--cargo", default="cargo")
    parser.add_argument(
        "--cargo-target-dir",
        default=None,
        help="Optional external Cargo target directory used by the canonicalizer.",
    )
    parser.add_argument(
        "--payload-mode",
        choices=("hardlink", "copy"),
        default="copy",
        help=(
            "Copy is the formal default. Hardlink is a storage-saving diagnostic "
            "mode and is not eligible for formal benchmark evidence."
        ),
    )
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    for name, value in (
        ("generation", args.generation),
        ("nprobe", args.nprobe),
        ("lower-hnsw-ef", args.lower_hnsw_ef),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"--{name} must be positive")


def read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON number {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"invalid {label} JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} root must be a JSON object")
    return value


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def float32_bits(value: Any) -> bytes:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"centroid value is not numeric: {value!r}")
    return struct.pack("<f", float(value))


def validate_typed_binding(
    graphless: dict[str, Any], canonical: dict[str, Any]
) -> None:
    """Accept only decimal spelling differences that preserve exact f32 bits."""
    if set(graphless) != set(canonical):
        raise RuntimeError("source graphless/production fields differ")
    for key in graphless:
        if key != "centroids" and graphless.get(key) != canonical.get(key):
            raise RuntimeError(
                f"source graphless field {key!r} differs from production artifact"
            )
    graphless_centroids = graphless.get("centroids")
    canonical_centroids = canonical.get("centroids")
    if not isinstance(graphless_centroids, list) or not isinstance(
        canonical_centroids, list
    ):
        raise RuntimeError("source graphless/production centroids must be arrays")
    if len(graphless_centroids) != len(canonical_centroids):
        raise RuntimeError("source graphless/production centroid count differs")
    for index, (source, typed) in enumerate(
        zip(graphless_centroids, canonical_centroids, strict=True)
    ):
        if not isinstance(source, dict) or not isinstance(typed, dict):
            raise RuntimeError(f"centroid {index} is not a JSON object")
        if source.get("shard_id") != typed.get("shard_id"):
            raise RuntimeError(f"centroid {index} shard_id differs")
        source_vector = source.get("vector")
        typed_vector = typed.get("vector")
        if not isinstance(source_vector, list) or not isinstance(typed_vector, list):
            raise RuntimeError(f"centroid {index} vector is not an array")
        if len(source_vector) != len(typed_vector):
            raise RuntimeError(f"centroid {index} dimension differs")
        if any(
            float32_bits(source_value) != float32_bits(typed_value)
            for source_value, typed_value in zip(
                source_vector, typed_vector, strict=True
            )
        ):
            raise RuntimeError(
                f"centroid {index} differs from production artifact at float32 precision"
            )


def checked_source_file(
    source_dir: Path,
    relative: Any,
    label: str,
    checksums: dict[str, str],
    declared_files: dict[str, Any],
) -> Path:
    path = prepare.safe_child(source_dir, relative, label)
    relative_name = path.relative_to(source_dir).as_posix()
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    if relative_name not in checksums:
        raise RuntimeError(f"{label} is not covered by source checksums")
    declared = declared_files.get(relative_name)
    if not isinstance(declared, dict):
        raise RuntimeError(f"source build manifest does not declare {relative_name}")
    expected_sha256 = prepare.cluster_tool.normalize_sha256(
        str(declared.get("sha256") or "")
    )
    if expected_sha256 != checksums[relative_name]:
        raise RuntimeError(f"source build manifest checksum mismatch for {relative_name}")
    if declared.get("size_bytes") != path.stat().st_size:
        raise RuntimeError(f"source build manifest size mismatch for {relative_name}")
    return path


def validate_source_bundle(source_dir: Path) -> dict[str, Any]:
    layout = prepare.load_routed_layout("simple_kmeans", source_dir)
    build_manifest_path = Path(layout["build_manifest_path"])
    build_manifest = read_json_object(build_manifest_path, "source build manifest")
    if build_manifest.get("tool") != "tools/simple_kmeans_native_layout.py":
        raise RuntimeError(
            "canonical source must be built by tools/simple_kmeans_native_layout.py"
        )
    outputs = build_manifest.get("outputs") or {}
    if not isinstance(outputs, dict):
        raise ValueError("source build manifest outputs must be a JSON object")
    declared_files = outputs.get("files") or {}
    if not isinstance(declared_files, dict):
        raise ValueError("source build manifest outputs/files must be JSON objects")

    graphless_path = checked_source_file(
        source_dir,
        outputs.get("graphless_artifact"),
        "graphless_artifact",
        layout["checksums"],
        declared_files,
    )
    graphless = read_json_object(graphless_path, "source graphless artifact")
    production_path = Path(layout["artifact_path"])
    production = read_json_object(production_path, "source production artifact")
    validate_typed_binding(graphless, production)

    parameters = build_manifest.get("parameters") or {}
    routing = build_manifest.get("routing") or {}
    artifact_binding = build_manifest.get("artifact_binding") or {}
    dataset = build_manifest.get("dataset") or {}
    for label, value in (
        ("parameters", parameters),
        ("routing", routing),
        ("artifact_binding", artifact_binding),
        ("dataset", dataset),
    ):
        if not isinstance(value, dict):
            raise ValueError(f"source build manifest {label} must be a JSON object")

    expected_binding = {
        "generation": int(layout["generation"]),
        "shard_count": int(layout["shard_count"]),
        "layout_sha256": str(layout["artifact"]["layout_sha256"]),
        "logical_point_count": int(layout["logical_point_count"]),
        "physical_point_count": int(layout["physical_point_count"]),
        "nprobe": int(production["nprobe"]),
        "lower_hnsw_ef": int(production["lower_hnsw_ef"]),
    }
    binding_mismatches = {
        field: {"expected": expected, "actual": artifact_binding.get(field)}
        for field, expected in expected_binding.items()
        if artifact_binding.get(field) != expected
    }
    if binding_mismatches:
        raise RuntimeError(
            "source artifact_binding does not match production artifact: "
            f"{binding_mismatches}"
        )
    expected_parameters = {
        "generation": int(layout["generation"]),
        "num_shards": int(layout["shard_count"]),
        "nprobe": int(production["nprobe"]),
        "lower_hnsw_ef": int(production["lower_hnsw_ef"]),
    }
    parameter_mismatches = {
        field: {"expected": expected, "actual": parameters.get(field)}
        for field, expected in expected_parameters.items()
        if parameters.get(field) != expected
    }
    if parameter_mismatches:
        raise RuntimeError(
            f"source build parameters do not match artifact: {parameter_mismatches}"
        )
    expected_routing = {
        "policy": "simple_kmeans",
        "logical_point_count": int(layout["logical_point_count"]),
        "physical_point_count": int(layout["physical_point_count"]),
        "expansion_ratio": 1.0,
    }
    routing_mismatches = {
        field: {"expected": expected, "actual": routing.get(field)}
        for field, expected in expected_routing.items()
        if routing.get(field) != expected
    }
    if routing_mismatches:
        raise RuntimeError(
            f"source routing summary does not match artifact: {routing_mismatches}"
        )

    import_manifest_path = Path(layout["import_manifest_path"])
    import_manifest = read_json_object(import_manifest_path, "source import manifest")
    vectors_path = prepare.safe_child(
        source_dir, import_manifest.get("vectors_file"), "vectors_file"
    )
    assignments_path = prepare.safe_child(
        source_dir, import_manifest.get("assignments_file"), "assignments_file"
    )
    return {
        "layout": layout,
        "build_manifest": build_manifest,
        "build_manifest_path": build_manifest_path,
        "graphless_path": graphless_path,
        # Use Rust's typed f32 representation as the source for derivation.
        "production": production,
        "production_path": production_path,
        "import_manifest": import_manifest,
        "import_manifest_path": import_manifest_path,
        "vectors_path": vectors_path,
        "assignments_path": assignments_path,
        "parameters": parameters,
        "routing": routing,
        "artifact_binding": artifact_binding,
        "dataset": dataset,
    }


def materialize_payload(
    source: Path,
    destination: Path,
    mode: str,
    expected_sha256: str,
) -> dict[str, Any]:
    if destination.exists() or os.path.lexists(destination):
        raise FileExistsError(f"refusing to overwrite payload: {destination}")
    expected_sha256 = prepare.cluster_tool.normalize_sha256(expected_sha256)
    source_size = source.stat().st_size
    copied_sha256: str | None = None
    if mode == "hardlink":
        try:
            os.link(source, destination)
        except OSError as exc:
            raise RuntimeError(
                f"cannot hardlink immutable payload {source} to {destination}: {exc}; "
                "use --payload-mode copy for different filesystems"
            ) from exc
    elif mode == "copy":
        try:
            digest = hashlib.sha256()
            with source.open("rb") as source_handle, destination.open("xb") as target:
                while chunk := source_handle.read(8 * 1024 * 1024):
                    target.write(chunk)
                    digest.update(chunk)
            copied_sha256 = digest.hexdigest()
        except BaseException:
            destination.unlink(missing_ok=True)
            raise
    else:  # pragma: no cover - argparse guarantees this for CLI callers.
        raise ValueError(f"unsupported payload mode: {mode!r}")

    destination_sha256 = copied_sha256 or layout_common.sha256_path(destination)
    if destination_sha256 != expected_sha256 or destination.stat().st_size != source_size:
        destination.unlink(missing_ok=True)
        raise RuntimeError(f"materialized payload differs from source: {source}")
    same_inode = (
        source.stat().st_dev == destination.stat().st_dev
        and source.stat().st_ino == destination.stat().st_ino
    )
    if same_inode is not (mode == "hardlink"):
        destination.unlink(missing_ok=True)
        raise RuntimeError("payload materialization inode proof is inconsistent")
    return {
        "source_file": source.name,
        "destination_file": destination.name,
        "sha256": expected_sha256,
        "size_bytes": source_size,
        "materialization": mode,
        "same_inode": same_inode,
    }


def runtime_parameters(
    source_parameters: dict[str, Any], args: argparse.Namespace
) -> dict[str, Any]:
    derived = copy.deepcopy(source_parameters)
    derived.update(
        {
            "generation": int(args.generation),
            "nprobe": int(args.nprobe),
            "lower_hnsw_ef": int(args.lower_hnsw_ef),
        }
    )
    changed_outside_runtime = {
        key
        for key in set(source_parameters) | set(derived)
        if source_parameters.get(key) != derived.get(key)
        and key not in RUNTIME_PARAMETER_KEYS
    }
    if changed_outside_runtime:
        raise RuntimeError(
            "runtime profile changed non-runtime build parameters: "
            f"{sorted(changed_outside_runtime)}"
        )
    return derived


def verify_finished_bundle(
    output_dir: Path,
    source: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    checksums = prepare.verify_checksum_listing(output_dir)
    build_manifest = read_json_object(
        output_dir / layout_common.BUILD_MANIFEST_NAME,
        "derived build manifest",
    )
    outputs = build_manifest.get("outputs") or {}
    artifact_path = prepare.safe_child(
        output_dir, outputs.get("production_artifact"), "production_artifact"
    )
    artifact_sha256 = checksums.get(artifact_path.name)
    if artifact_sha256 is None:
        raise RuntimeError("derived production artifact is not covered by checksums")
    metadata = prepare.cluster_tool.validate_local_simple_kmeans_artifact(
        artifact_path,
        int(args.generation),
        artifact_sha256,
    )
    artifact = read_json_object(artifact_path, "derived production artifact")
    source_artifact = source["production"]
    offline_source = {
        key: value
        for key, value in source_artifact.items()
        if key not in ARTIFACT_RUNTIME_KEYS
    }
    offline_derived = {
        key: value for key, value in artifact.items() if key not in ARTIFACT_RUNTIME_KEYS
    }
    if offline_derived != offline_source:
        raise RuntimeError("derived artifact changed the offline KMeans routing structure")
    expected_runtime = {
        "generation": int(args.generation),
        "nprobe": int(args.nprobe),
        "lower_hnsw_ef": int(args.lower_hnsw_ef),
    }
    if any(artifact.get(key) != value for key, value in expected_runtime.items()):
        raise RuntimeError("derived artifact runtime parameters mismatch")

    import_manifest_path = prepare.safe_child(
        output_dir, outputs.get("import_manifest"), "import_manifest"
    )
    import_manifest = read_json_object(import_manifest_path, "derived import manifest")
    source_import = source["import_manifest"]
    expected_import = copy.deepcopy(source_import)
    expected_import.update(
        {
            "routing_generation": int(args.generation),
            "routing_artifact_file": artifact_path.name,
            "routing_artifact_sha256": artifact_sha256,
        }
    )
    if import_manifest != expected_import:
        raise RuntimeError("derived import manifest changed immutable import metadata")
    for prefix in ("vectors", "assignments"):
        payload_path = prepare.safe_child(
            output_dir,
            import_manifest.get(f"{prefix}_file"),
            f"{prefix}_file",
        )
        if layout_common.sha256_path(payload_path) != import_manifest.get(
            f"{prefix}_sha256"
        ):
            raise RuntimeError(f"derived {prefix} payload checksum mismatch")
    return {
        "artifact_path": artifact_path,
        "artifact_sha256": metadata["file_sha256"],
        "import_manifest_path": import_manifest_path,
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    validate_args(args)
    source_dir = Path(args.source_layout_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if os.path.lexists(output_dir):
        raise FileExistsError(f"refusing to overwrite existing output path: {output_dir}")
    if output_dir == source_dir or source_dir in output_dir.parents:
        raise ValueError("output directory must not be the source bundle or its descendant")
    if not source_dir.is_dir():
        raise FileNotFoundError(f"source layout directory not found: {source_dir}")

    source = validate_source_bundle(source_dir)
    shard_count = int(source["layout"]["shard_count"])
    if int(args.nprobe) > shard_count:
        raise ValueError(
            f"--nprobe {args.nprobe} exceeds source shard count {shard_count}"
        )
    parameters = runtime_parameters(source["parameters"], args)
    graphless = copy.deepcopy(source["production"])
    graphless.update(
        {
            "generation": int(args.generation),
            "nprobe": int(args.nprobe),
            "lower_hnsw_ef": int(args.lower_hnsw_ef),
        }
    )
    changed_fields = {
        key
        for key in set(source["production"]) | set(graphless)
        if source["production"].get(key) != graphless.get(key)
    }
    if not changed_fields.issubset(ARTIFACT_RUNTIME_KEYS):
        raise RuntimeError(
            "derived artifact changed offline fields: "
            f"{sorted(changed_fields - ARTIFACT_RUNTIME_KEYS)}"
        )

    output_dir.mkdir(parents=True, exist_ok=False)
    graphless_path = output_dir / simple_layout.GRAPHLESS_NAME
    layout_common.write_json_new(graphless_path, graphless)
    production_path = output_dir / f"generation-{int(args.generation)}.json"
    builder_args = argparse.Namespace(
        cargo=str(args.cargo),
        cargo_target_dir=args.cargo_target_dir,
    )
    rust_command = simple_layout.run_rust_builder(
        builder_args, graphless_path, production_path
    )
    checksum_path = Path(f"{production_path}.sha256")
    if not production_path.is_file() or not checksum_path.is_file():
        raise RuntimeError("Rust canonicalizer did not create artifact and checksum files")
    production_sha256 = prepare.cluster_tool.normalize_sha256(
        checksum_path.read_text(encoding="utf-8").strip()
    )
    if layout_common.sha256_path(production_path) != production_sha256:
        raise RuntimeError("derived production artifact checksum mismatch")
    canonical = read_json_object(production_path, "derived production artifact")
    validate_typed_binding(graphless, canonical)
    prepare.cluster_tool.validate_local_simple_kmeans_artifact(
        production_path,
        int(args.generation),
        production_sha256,
    )

    source_import = source["import_manifest"]
    vectors_path = output_dir / str(source_import["vectors_file"])
    assignments_path = output_dir / str(source_import["assignments_file"])
    reused_payloads = {
        "vectors": materialize_payload(
            source["vectors_path"],
            vectors_path,
            str(args.payload_mode),
            str(source_import["vectors_sha256"]),
        ),
        "assignments": materialize_payload(
            source["assignments_path"],
            assignments_path,
            str(args.payload_mode),
            str(source_import["assignments_sha256"]),
        ),
    }
    import_manifest = copy.deepcopy(source_import)
    import_manifest.update(
        {
            "routing_generation": int(args.generation),
            "routing_artifact_file": production_path.name,
            "routing_artifact_sha256": production_sha256,
        }
    )
    import_manifest_path = output_dir / source["import_manifest_path"].name
    layout_common.write_json_new(import_manifest_path, import_manifest)

    artifact_binding = copy.deepcopy(source["artifact_binding"])
    artifact_binding.update(
        {
            "generation": int(args.generation),
            "nprobe": int(args.nprobe),
            "lower_hnsw_ef": int(args.lower_hnsw_ef),
        }
    )
    parameter_changes = {
        key: {
            "source": source["parameters"].get(key),
            "derived": parameters.get(key),
        }
        for key in RUNTIME_PARAMETER_KEYS
    }
    source_binding = {
        "layout_dir": str(source_dir),
        "build_manifest_file": source["build_manifest_path"].name,
        "build_manifest_sha256": layout_common.sha256_path(
            source["build_manifest_path"]
        ),
        "graphless_artifact_file": source["graphless_path"].name,
        "graphless_artifact_sha256": layout_common.sha256_path(
            source["graphless_path"]
        ),
        "production_artifact_file": source["production_path"].name,
        "production_artifact_sha256": source["layout"]["artifact_sha256"],
        "import_manifest_file": source["import_manifest_path"].name,
        "import_manifest_sha256": source["layout"]["import_manifest_sha256"],
        "generation": int(source["layout"]["generation"]),
        "layout_sha256": source["layout"]["artifact"]["layout_sha256"],
        "logical_point_count": int(source["layout"]["logical_point_count"]),
        "physical_point_count": int(source["layout"]["physical_point_count"]),
        "shard_count": shard_count,
        "parameters": copy.deepcopy(source["parameters"]),
        "parameters_sha256": canonical_json_sha256(source["parameters"]),
        "dataset_sha256": canonical_json_sha256(source["dataset"]),
        "routing_sha256": canonical_json_sha256(source["routing"]),
        "offline_artifact_sha256": canonical_json_sha256(
            {
                key: value
                for key, value in source["production"].items()
                if key not in ARTIFACT_RUNTIME_KEYS
            }
        ),
        "vectors_sha256": str(source_import["vectors_sha256"]),
        "assignments_sha256": str(source_import["assignments_sha256"]),
    }
    payload_files = layout_common.relative_file_records(
        output_dir,
        {layout_common.BUILD_MANIFEST_NAME, layout_common.CHECKSUMS_NAME},
    )
    formal_evidence_eligible = str(args.payload_mode) == "copy"
    manifest = {
        "format_version": 1,
        "tool": TOOL_NAME,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "mode": "production_bundle",
        "dataset": copy.deepcopy(source["dataset"]),
        "parameters": parameters,
        "artifact_binding": artifact_binding,
        "routing": copy.deepcopy(source["routing"]),
        "derivation": {
            "format_version": DERIVATION_FORMAT_VERSION,
            "kind": DERIVATION_KIND,
            "allowed_parameter_changes": list(RUNTIME_PARAMETER_KEYS),
            "parameter_changes": parameter_changes,
            "formal_evidence_eligible": formal_evidence_eligible,
            "source": source_binding,
            "reused_payloads": reused_payloads,
        },
        "outputs": {
            "graphless_artifact": graphless_path.name,
            "production_artifact": production_path.name,
            "import_manifest": import_manifest_path.name,
            "rust_builder_command": rust_command,
            "files": payload_files,
        },
    }
    build_manifest_path = output_dir / layout_common.BUILD_MANIFEST_NAME
    layout_common.write_json_new(build_manifest_path, manifest)
    checksums_path = layout_common.write_checksums(output_dir)
    finished = verify_finished_bundle(output_dir, source, args)
    prepared = prepare.load_routed_layout("simple_kmeans", output_dir)
    if Path(prepared["artifact_path"]).resolve() != finished["artifact_path"].resolve():
        raise RuntimeError("production prepare gate resolved a different artifact")
    if prepared["artifact"]["layout_sha256"] != source_binding["layout_sha256"]:
        raise RuntimeError("derived layout checksum changed after final validation")

    summary = {
        "output_dir": str(output_dir),
        "source_layout_dir": str(source_dir),
        "source_generation": int(source["layout"]["generation"]),
        "generation": int(args.generation),
        "nprobe": int(args.nprobe),
        "lower_hnsw_ef": int(args.lower_hnsw_ef),
        "layout_sha256": source_binding["layout_sha256"],
        "production_artifact": str(finished["artifact_path"]),
        "production_artifact_sha256": finished["artifact_sha256"],
        "import_manifest": str(finished["import_manifest_path"]),
        "build_manifest": str(build_manifest_path),
        "checksums": str(checksums_path),
        "payload_mode": str(args.payload_mode),
        "formal_evidence_eligible": formal_evidence_eligible,
        "reused_payloads": reused_payloads,
    }
    print(json.dumps(summary, sort_keys=True, indent=2))
    return summary


def main(argv: list[str] | None = None) -> int:
    try:
        build(parse_args(argv))
    except (
        ValueError,
        FileNotFoundError,
        FileExistsError,
        RuntimeError,
        subprocess.CalledProcessError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
