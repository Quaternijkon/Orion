#!/usr/bin/env python3
"""Derive a native Orion runtime profile without rebuilding its offline layout.

The source must be a checksum-verified, faithful production bundle accepted by
``native_auto_shard_prepare.load_routed_layout``.  This tool preserves the
dataset proof, upper sample, upper-node vectors, shard memberships, fission
result, point assignments, and numeric-shard import payload byte-for-byte.  It
only changes the routing generation and the faithful runtime controls:

* ``upper_k == upper_ef_search``;
* ``dynamic_ef_base``; and
* ``dynamic_ef_factor``.

The portable upper HNSW is rebuilt with the source bundle's deterministic graph
parameters.  Existing output paths are never overwritten.
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


TOOL_NAME = "tools/orion_native_runtime_profile.py"
DERIVATION_KIND = "orion_runtime_profile"
DERIVATION_FORMAT_VERSION = 1
RUNTIME_PARAMETER_KEYS = (
    "generation",
    "upper_k",
    "upper_search_ef",
    "dynamic_ef_base",
    "dynamic_ef_factor",
)
GRAPHLESS_RUNTIME_KEYS = (
    "generation",
    "upper_k",
    "upper_ef_search",
    "dynamic_ef_base",
    "dynamic_ef_factor",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Derive a faithful Orion runtime profile while reusing the exact "
            "offline layout and numeric-shard import assignments."
        )
    )
    parser.add_argument(
        "--source-layout-dir",
        required=True,
        help="Checksum-verified faithful Orion production bundle.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="New output directory. Existing paths are never overwritten.",
    )
    parser.add_argument("--generation", required=True, type=int)
    parser.add_argument(
        "--upper-k",
        required=True,
        type=int,
        help="Runtime upper result count and search EF; both are set to this value.",
    )
    parser.add_argument("--dynamic-ef-base", required=True, type=int)
    parser.add_argument("--dynamic-ef-factor", required=True, type=int)
    parser.add_argument("--cargo", default="cargo")
    parser.add_argument(
        "--cargo-target-dir",
        default=None,
        help="Optional external Cargo target directory used only for the rebuild.",
    )
    parser.add_argument(
        "--payload-mode",
        choices=("hardlink", "copy"),
        default="copy",
        help=(
            "How to materialize the immutable .f32le and assignments payloads. "
            "Copy is the formal default and creates independent bytes. Hardlink "
            "is an explicit storage-saving diagnostic mode, shares mutation risk, "
            "and requires the same filesystem. No silent fallback is performed."
        ),
    )
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    positive = {
        "generation": args.generation,
        "upper-k": args.upper_k,
        "dynamic-ef-base": args.dynamic_ef_base,
    }
    for name, value in positive.items():
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"--{name} must be positive")
    if (
        isinstance(args.dynamic_ef_factor, bool)
        or not isinstance(args.dynamic_ef_factor, int)
        or args.dynamic_ef_factor < 0
    ):
        raise ValueError("--dynamic-ef-factor must be non-negative")


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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


def float32_bits(value: Any) -> bytes:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"upper vector value is not numeric: {value!r}")
    return struct.pack("<f", float(value))


def validate_graphless_typed_binding(
    graphless: dict[str, Any], canonical: dict[str, Any]
) -> None:
    if set(graphless) != set(canonical):
        raise RuntimeError("source graphless/production fields differ")
    for key in graphless:
        if key != "upper_nodes" and graphless.get(key) != canonical.get(key):
            raise RuntimeError(
                f"source graphless field {key!r} differs from production artifact"
            )
    graphless_nodes = graphless.get("upper_nodes")
    canonical_nodes = canonical.get("upper_nodes")
    if not isinstance(graphless_nodes, list) or not isinstance(canonical_nodes, list):
        raise RuntimeError("source graphless/production upper_nodes must be arrays")
    if len(graphless_nodes) != len(canonical_nodes):
        raise RuntimeError("source graphless/production upper_nodes length differs")
    for index, (source_node, canonical_node) in enumerate(
        zip(graphless_nodes, canonical_nodes, strict=True)
    ):
        if not isinstance(source_node, dict) or not isinstance(canonical_node, dict):
            raise RuntimeError(f"upper node {index} is not a JSON object")
        for field in ("label", "shard_membership"):
            if source_node.get(field) != canonical_node.get(field):
                raise RuntimeError(
                    f"source graphless upper node {index} {field} differs from "
                    "production artifact"
                )
        source_vector = source_node.get("vector")
        canonical_vector = canonical_node.get("vector")
        if not isinstance(source_vector, list) or not isinstance(canonical_vector, list):
            raise RuntimeError(f"upper node {index} vector is not an array")
        if len(source_vector) != len(canonical_vector):
            raise RuntimeError(f"upper node {index} vector dimension differs")
        if any(
            float32_bits(source_value) != float32_bits(canonical_value)
            for source_value, canonical_value in zip(
                source_vector, canonical_vector, strict=True
            )
        ):
            raise RuntimeError(
                f"source graphless upper node {index} vector differs from "
                "production artifact at float32 precision"
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
    declared_sha256 = prepare.cluster_tool.normalize_sha256(
        str(declared.get("sha256") or "")
    )
    if declared_sha256 != checksums[relative_name]:
        raise RuntimeError(f"source build manifest checksum mismatch for {relative_name}")
    if declared.get("size_bytes") != path.stat().st_size:
        raise RuntimeError(f"source build manifest size mismatch for {relative_name}")
    return path


def validate_source_bundle(
    source_dir: Path,
) -> dict[str, Any]:
    layout = prepare.load_routed_layout("orion", source_dir)
    build_manifest_path = Path(layout["build_manifest_path"])
    build_manifest = read_json_object(build_manifest_path, "source build manifest")
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
    if "upper_graph" in graphless:
        raise RuntimeError("source graphless artifact unexpectedly contains upper_graph")
    production_path = Path(layout["artifact_path"])
    production = read_json_object(production_path, "source production artifact")
    graphless_from_production = copy.deepcopy(production)
    upper_graph = graphless_from_production.pop("upper_graph", None)
    if not isinstance(upper_graph, dict):
        raise RuntimeError("source production artifact does not contain upper_graph")
    validate_graphless_typed_binding(graphless, graphless_from_production)
    # Use the Rust-typed canonical representation for derived profiles. Python's
    # original writer may spell the same float32 with a longer decimal, while
    # the production artifact emits the shortest round-trippable f32 form.
    graphless = graphless_from_production

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

    artifact_expected = {
        "generation": int(layout["generation"]),
        "layout_sha256": str(layout["artifact"]["layout_sha256"]),
        "logical_point_count": int(layout["logical_point_count"]),
        "physical_point_count": int(layout["physical_point_count"]),
        "shard_count": int(layout["shard_count"]),
    }
    binding_mismatches = {
        field: {"expected": expected, "actual": artifact_binding.get(field)}
        for field, expected in artifact_expected.items()
        if artifact_binding.get(field) != expected
    }
    if binding_mismatches:
        raise RuntimeError(
            f"source artifact_binding does not match production artifact: {binding_mismatches}"
        )
    routing_expected = {
        "effective_num_shards": int(layout["shard_count"]),
        "logical_point_count": int(layout["logical_point_count"]),
        "physical_point_count": int(layout["physical_point_count"]),
    }
    routing_mismatches = {
        field: {"expected": expected, "actual": routing.get(field)}
        for field, expected in routing_expected.items()
        if routing.get(field) != expected
    }
    if routing_mismatches:
        raise RuntimeError(
            f"source routing summary does not match production artifact: {routing_mismatches}"
        )
    if parameters.get("generation") != int(layout["generation"]):
        raise RuntimeError("source build parameters generation mismatch")

    import_manifest_path = Path(layout["import_manifest_path"])
    import_manifest = read_json_object(import_manifest_path, "source import manifest")
    vectors_path = prepare.safe_child(
        source_dir,
        import_manifest.get("vectors_file"),
        "vectors_file",
    )
    assignments_path = prepare.safe_child(
        source_dir,
        import_manifest.get("assignments_file"),
        "assignments_file",
    )
    # load_routed_layout has already verified these payload checksums. Re-resolve
    # and carry the exact files into the derived bundle without parsing them.
    return {
        "layout": layout,
        "build_manifest": build_manifest,
        "build_manifest_path": build_manifest_path,
        "graphless": graphless,
        "graphless_path": graphless_path,
        "production": production,
        "production_path": production_path,
        "upper_graph": upper_graph,
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
            with source.open("rb") as source_handle, destination.open("xb") as target_handle:
                while chunk := source_handle.read(8 * 1024 * 1024):
                    target_handle.write(chunk)
                    digest.update(chunk)
            copied_sha256 = digest.hexdigest()
        except BaseException:
            destination.unlink(missing_ok=True)
            raise
    else:  # pragma: no cover - argparse and callers validate this.
        raise ValueError(f"unsupported payload mode: {mode!r}")

    destination_sha256 = copied_sha256 or layout_common.sha256_path(destination)
    if (
        destination_sha256 != expected_sha256
        or source_size != destination.stat().st_size
    ):
        destination.unlink(missing_ok=True)
        raise RuntimeError(f"materialized payload differs from source: {source}")
    return {
        "source_file": source.name,
        "destination_file": destination.name,
        "sha256": expected_sha256,
        "size_bytes": source_size,
        "materialization": mode,
        "same_inode": (
            source.stat().st_dev == destination.stat().st_dev
            and source.stat().st_ino == destination.stat().st_ino
        ),
    }


def runtime_parameters(
    source_parameters: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    derived = copy.deepcopy(source_parameters)
    derived.update(
        {
            "generation": int(args.generation),
            "upper_k": int(args.upper_k),
            "upper_search_ef": int(args.upper_k),
            "dynamic_ef_base": int(args.dynamic_ef_base),
            "dynamic_ef_factor": int(args.dynamic_ef_factor),
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
    source_generation = int(source["layout"]["generation"])
    upper_nodes = source["graphless"].get("upper_nodes")
    if not isinstance(upper_nodes, list) or not upper_nodes:
        raise RuntimeError("source graphless artifact has no upper_nodes")
    if int(args.upper_k) > len(upper_nodes):
        raise ValueError(
            f"--upper-k {args.upper_k} exceeds source upper tier size {len(upper_nodes)}"
        )

    parameters = runtime_parameters(source["parameters"], args)
    graphless = copy.deepcopy(source["graphless"])
    graphless.update(
        {
            "generation": int(args.generation),
            "upper_k": int(args.upper_k),
            "upper_ef_search": int(args.upper_k),
            "dynamic_ef_base": int(args.dynamic_ef_base),
            "dynamic_ef_factor": int(args.dynamic_ef_factor),
        }
    )
    changed_graphless_fields = {
        key
        for key in set(source["graphless"]) | set(graphless)
        if source["graphless"].get(key) != graphless.get(key)
    }
    if not changed_graphless_fields.issubset(GRAPHLESS_RUNTIME_KEYS):
        raise RuntimeError(
            "derived graphless artifact changed offline fields: "
            f"{sorted(changed_graphless_fields - set(GRAPHLESS_RUNTIME_KEYS))}"
        )

    output_dir.mkdir(parents=True, exist_ok=False)
    graphless_path = output_dir / layout_common.GRAPHLESS_NAME
    layout_common.write_json_new(graphless_path, graphless)
    production_path = output_dir / f"generation-{int(args.generation)}.json"
    builder_args = argparse.Namespace(
        cargo=str(args.cargo),
        cargo_target_dir=args.cargo_target_dir,
        upper_graph_seed=int(parameters["upper_graph_seed"]),
        upper_m=int(parameters["upper_m"]),
        upper_ef_construction=int(parameters["upper_ef_construction"]),
    )
    rust_command = layout_common.run_rust_builder(
        builder_args,
        graphless_path,
        production_path,
    )
    production_sha256 = layout_common.verify_production_artifact(production_path)
    production = read_json_object(production_path, "derived production artifact")
    production_graph = production.get("upper_graph")
    production_without_graph = copy.deepcopy(production)
    production_without_graph.pop("upper_graph", None)
    if production_without_graph != graphless:
        raise RuntimeError("Rust builder changed fields outside upper_graph")
    if production_graph != source["upper_graph"]:
        raise RuntimeError(
            "deterministic upper graph changed despite identical offline upper nodes "
            "and graph build parameters"
        )
    prepare.cluster_tool.validate_local_orion_artifact(
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
            "orion_generation": int(args.generation),
            "orion_artifact_file": production_path.name,
            "orion_artifact_sha256": production_sha256,
        }
    )
    import_manifest_path = output_dir / source["import_manifest_path"].name
    layout_common.write_json_new(import_manifest_path, import_manifest)

    artifact_binding = copy.deepcopy(source["artifact_binding"])
    artifact_binding["generation"] = int(args.generation)
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
        "generation": source_generation,
        "layout_sha256": source["layout"]["artifact"]["layout_sha256"],
        "logical_point_count": int(source["layout"]["logical_point_count"]),
        "physical_point_count": int(source["layout"]["physical_point_count"]),
        "shard_count": int(source["layout"]["shard_count"]),
        "parameters": copy.deepcopy(source["parameters"]),
        "parameters_sha256": canonical_json_sha256(source["parameters"]),
        "dataset_sha256": canonical_json_sha256(source["dataset"]),
        "routing_sha256": canonical_json_sha256(source["routing"]),
        "vectors_sha256": str(source_import["vectors_sha256"]),
        "assignments_sha256": str(source_import["assignments_sha256"]),
    }
    payload_files = layout_common.relative_file_records(
        output_dir,
        {layout_common.BUILD_MANIFEST_NAME, layout_common.CHECKSUMS_NAME},
    )
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
            "rebuild": {
                "upper_graph_seed": int(parameters["upper_graph_seed"]),
                "upper_m": int(parameters["upper_m"]),
                "upper_ef_construction": int(parameters["upper_ef_construction"]),
                "cargo_target_dir": layout_common.effective_cargo_target_dir(
                    builder_args
                ),
            },
            "formal_evidence_eligible": str(args.payload_mode) == "copy",
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

    # Verify the finished bundle through the same production preparation gate.
    derived_layout = prepare.load_routed_layout("orion", output_dir)
    if derived_layout["artifact"]["layout_sha256"] != source_binding["layout_sha256"]:
        raise RuntimeError("derived layout checksum changed after final validation")
    summary = {
        "output_dir": str(output_dir),
        "source_layout_dir": str(source_dir),
        "source_generation": source_generation,
        "generation": int(args.generation),
        "upper_k": int(args.upper_k),
        "upper_search_ef": int(args.upper_k),
        "dynamic_ef_base": int(args.dynamic_ef_base),
        "dynamic_ef_factor": int(args.dynamic_ef_factor),
        "layout_sha256": source_binding["layout_sha256"],
        "production_artifact": str(production_path),
        "production_artifact_sha256": production_sha256,
        "import_manifest": str(import_manifest_path),
        "build_manifest": str(build_manifest_path),
        "checksums": str(checksums_path),
        "payload_mode": str(args.payload_mode),
        "formal_evidence_eligible": str(args.payload_mode) == "copy",
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
