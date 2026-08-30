#!/usr/bin/env python3
"""Project a production Orion artifact into a neutral upper-only snapshot.

The output preserves every upper graph, label, vector, schema, and search
parameter used by Phase A. It also carries only the four safe upper-build
parameters from a checksum-bound source manifest. Historical L0 memberships,
layout metadata, source paths, and balance fields are omitted so the balance
candidate never receives an earlier L0 assignment. Existing outputs are never
overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import struct
from typing import Any


FORMAT_VERSION = 1
RECORD_TYPE = "orion_upper_only_projection"
NEUTRAL_LAYOUT_SHA256 = "0" * 64
NEUTRAL_SHARD_MEMBERSHIP = [0]
UPPER_BUILD_PARAMETER_KEYS = (
    "upper_sample_seed",
    "upper_m",
    "upper_ef_construction",
    "upper_graph_seed",
)
REPLAY_ATTESTATION_MODE = "production_artifact_deterministic_replay_attestation"

ARTIFACT_KEYS = {
    "format_version",
    "generation",
    "vector_schema",
    "shard_count",
    "layout_sha256",
    "logical_point_count",
    "physical_point_count",
    "upper_k",
    "upper_ef_search",
    "dynamic_ef_base",
    "dynamic_ef_factor",
    "upper_nodes",
    "upper_graph",
}
UPPER_NODE_KEYS = {"label", "vector", "shard_membership"}


def projection_contract() -> dict[str, Any]:
    return {
        "preserves_upper_graph": True,
        "preserves_upper_label_order": True,
        "preserves_upper_vector_f32_bits": True,
        "preserves_schema_and_search_parameters": True,
        "historical_memberships_removed": True,
        "historical_layout_metadata_removed": True,
        "neutral_membership_sentinel": NEUTRAL_SHARD_MEMBERSHIP,
        "neutral_layout_sha256": NEUTRAL_LAYOUT_SHA256,
        "output_physical_point_count_equals_logical": True,
        "does_not_read_full_l0_attachments": True,
        "does_not_emit_membership_values_or_distribution": True,
        "source_artifact_path_emitted": False,
        "source_build_manifest_path_emitted": False,
        "upper_build_parameter_whitelist": list(UPPER_BUILD_PARAMETER_KEYS),
        "historical_balance_fields_emitted": False,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-artifact", required=True)
    parser.add_argument("--source-build-manifest", required=True)
    parser.add_argument("--output-artifact", required=True)
    parser.add_argument("--output-manifest", required=True)
    return parser.parse_args(argv)


def sha256_path(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def upper_nodes_identity_sha256(nodes: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    digest.update(struct.pack("<Q", len(nodes)))
    for index, node in enumerate(nodes):
        label_bytes = json.dumps(
            node["label"],
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        digest.update(struct.pack("<Q", len(label_bytes)))
        digest.update(label_bytes)
        vector = node["vector"]
        if not isinstance(vector, list) or not vector:
            raise ValueError(f"upper_nodes[{index}].vector must be a non-empty list")
        digest.update(struct.pack("<Q", len(vector)))
        for column, raw_value in enumerate(vector):
            if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
                raise TypeError(
                    f"upper_nodes[{index}].vector[{column}] is not numeric"
                )
            value = float(raw_value)
            if not math.isfinite(value):
                raise ValueError(
                    f"upper_nodes[{index}].vector[{column}] is not finite"
                )
            digest.update(struct.pack("<f", value))
    return digest.hexdigest()


def _absolute_file(raw: str, label: str) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        raise ValueError(f"{label} must be an absolute path")
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} is not a file: {path}")
    return path


def _absolute_new_file(raw: str, label: str) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        raise ValueError(f"{label} must be an absolute path")
    path = path.resolve()
    sidecar = path.with_name(path.name + ".sha256")
    if path.exists() or sidecar.exists():
        raise FileExistsError(f"refusing to overwrite {label}: {path}")
    if not path.parent.is_dir():
        raise FileNotFoundError(f"{label} parent does not exist: {path.parent}")
    return path


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _write_with_sidecar(path: Path, data: bytes) -> tuple[str, Path]:
    sidecar = path.with_name(path.name + ".sha256")
    with path.open("xb") as handle:
        handle.write(data)
    digest = sha256_path(path)
    with sidecar.open("xb") as handle:
        handle.write((digest + "\n").encode("ascii"))
    os.chmod(path, 0o444)
    os.chmod(sidecar, 0o444)
    return digest, sidecar


def _validated_source(path: Path) -> dict[str, Any]:
    try:
        artifact = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"source artifact is not valid UTF-8 JSON: {path}") from exc
    if not isinstance(artifact, dict) or set(artifact) != ARTIFACT_KEYS:
        raise ValueError("source artifact schema differs from the frozen Orion format")
    if artifact.get("format_version") != 1:
        raise ValueError("source artifact format_version must be 1")
    for field in (
        "generation",
        "shard_count",
        "logical_point_count",
        "physical_point_count",
        "upper_k",
        "upper_ef_search",
        "dynamic_ef_base",
        "dynamic_ef_factor",
    ):
        value = artifact.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"source artifact {field} must be a positive integer")
    if artifact["physical_point_count"] < artifact["logical_point_count"]:
        raise ValueError("source physical_point_count is smaller than logical")
    if not isinstance(artifact.get("vector_schema"), dict):
        raise ValueError("source artifact lacks vector_schema")
    if not isinstance(artifact.get("upper_graph"), dict):
        raise ValueError("source artifact lacks upper_graph")
    nodes = artifact.get("upper_nodes")
    if not isinstance(nodes, list) or not nodes:
        raise ValueError("source artifact lacks upper_nodes")
    for index, node in enumerate(nodes):
        if not isinstance(node, dict) or set(node) != UPPER_NODE_KEYS:
            raise ValueError(f"upper_nodes[{index}] schema drifted")
        memberships = node.get("shard_membership")
        if not isinstance(memberships, list) or not memberships:
            raise ValueError(f"upper_nodes[{index}] has no source membership")
    upper_nodes_identity_sha256(nodes)
    canonical_sha256(artifact["upper_graph"])
    return artifact


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid UTF-8 JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return value


def _safe_upper_build_parameters(source_build: dict[str, Any]) -> dict[str, int]:
    parameters = source_build.get("parameters")
    if not isinstance(parameters, dict):
        raise ValueError("source build manifest lacks parameters")
    safe_parameters: dict[str, int] = {}
    for key in UPPER_BUILD_PARAMETER_KEYS:
        value = parameters.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(
                f"source build parameter {key} must be a positive integer"
            )
        safe_parameters[key] = value
    return safe_parameters


def _output_record(
    build_manifest: dict[str, Any], output_key: str, label: str
) -> tuple[str, dict[str, Any]]:
    outputs = build_manifest.get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError(f"{label} lacks outputs")
    files = outputs.get("files")
    if not isinstance(files, dict):
        raise ValueError(f"{label} lacks output file records")
    output_name = outputs.get(output_key)
    if not isinstance(output_name, str) or not output_name:
        raise ValueError(f"{label} lacks {output_key}")
    record = files.get(output_name)
    if not isinstance(record, dict):
        raise ValueError(f"{label} lacks the {output_key} file record")
    return output_name, record


def _command_positive_int(command: Any, flag: str) -> int:
    if not isinstance(command, list) or any(
        not isinstance(value, str) for value in command
    ):
        raise ValueError("replay builder command is invalid")
    positions = [index for index, value in enumerate(command) if value == flag]
    if len(positions) != 1 or positions[0] + 1 >= len(command):
        raise ValueError(f"replay builder command lacks exactly one {flag}")
    try:
        value = int(command[positions[0] + 1])
    except ValueError as exc:
        raise ValueError(f"replay builder command {flag} is not an integer") from exc
    if value <= 0:
        raise ValueError(f"replay builder command {flag} must be positive")
    return value


def _validated_replay_bridge(
    replay_manifest: dict[str, Any], source_artifact_sha256: str
) -> dict[str, int]:
    provenance = replay_manifest.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("source-build replay lacks provenance")
    if provenance.get("artifact_byte_identity_reproduced") is not True:
        raise ValueError("source-build replay did not reproduce artifact bytes")

    _production_name, production_record = _output_record(
        replay_manifest,
        "production_artifact",
        "source-build replay",
    )
    _replay_name, replay_record = _output_record(
        replay_manifest,
        "replay_artifact",
        "source-build replay",
    )
    _graphless_name, graphless_record = _output_record(
        replay_manifest,
        "source_graphless_artifact",
        "source-build replay",
    )
    if (
        production_record.get("sha256") != source_artifact_sha256
        or replay_record.get("sha256") != source_artifact_sha256
    ):
        raise ValueError(
            "source-build replay artifact checksum does not bind source artifact"
        )

    source_manifest_raw = provenance.get("source_build_manifest_path")
    if not isinstance(source_manifest_raw, str) or not source_manifest_raw:
        raise ValueError("source-build replay lacks source manifest path")
    source_manifest_path = _absolute_file(
        source_manifest_raw, "replay source-build-manifest"
    )
    expected_source_sha = provenance.get("source_build_manifest_sha256")
    if sha256_path(source_manifest_path) != expected_source_sha:
        raise ValueError("source-build replay source manifest checksum drifted")
    original_build = _load_json_object(
        source_manifest_path, "replay source build manifest"
    )
    parameters = _safe_upper_build_parameters(original_build)

    _original_graphless_name, original_graphless_record = _output_record(
        original_build,
        "graphless_artifact",
        "replay source build manifest",
    )
    if original_graphless_record.get("sha256") != graphless_record.get("sha256"):
        raise ValueError("source-build replay graphless artifact binding drifted")

    command = provenance.get("builder_command")
    expected_flags = {
        "--seed": parameters["upper_graph_seed"],
        "--m": parameters["upper_m"],
        "--ef": parameters["upper_ef_construction"],
    }
    for flag, expected in expected_flags.items():
        if _command_positive_int(command, flag) != expected:
            raise ValueError(f"source-build replay {flag} differs from source manifest")
    return parameters


def _validated_upper_build_provenance(
    path: Path, source_artifact_sha256: str
) -> dict[str, Any]:
    source_build = _load_json_object(path, "source build manifest")
    if source_build.get("mode") == REPLAY_ATTESTATION_MODE:
        safe_parameters = _validated_replay_bridge(
            source_build, source_artifact_sha256
        )
    else:
        _production_name, production_record = _output_record(
            source_build,
            "production_artifact",
            "source build manifest",
        )
        if production_record.get("sha256") != source_artifact_sha256:
            raise ValueError(
                "source build production artifact checksum does not bind source artifact"
            )
        safe_parameters = _safe_upper_build_parameters(source_build)
    return {
        "source_manifest_sha256": sha256_path(path),
        "source_manifest_size_bytes": path.stat().st_size,
        "parameters": safe_parameters,
    }


def run(args: argparse.Namespace) -> tuple[Path, str, Path, str]:
    source_path = _absolute_file(args.source_artifact, "source-artifact")
    source_build_path = _absolute_file(
        args.source_build_manifest, "source-build-manifest"
    )
    output_path = _absolute_new_file(args.output_artifact, "output-artifact")
    manifest_path = _absolute_new_file(args.output_manifest, "output-manifest")
    if output_path == manifest_path:
        raise ValueError("output artifact and manifest must be different files")

    source = _validated_source(source_path)
    source_artifact_sha256 = sha256_path(source_path)
    upper_build_provenance = _validated_upper_build_provenance(
        source_build_path, source_artifact_sha256
    )
    source_graph_sha256 = canonical_sha256(source["upper_graph"])
    source_identity_sha256 = upper_nodes_identity_sha256(source["upper_nodes"])

    projected = json.loads(json.dumps(source, allow_nan=False))
    projected["layout_sha256"] = NEUTRAL_LAYOUT_SHA256
    projected["physical_point_count"] = projected["logical_point_count"]
    for node in projected["upper_nodes"]:
        node["shard_membership"] = list(NEUTRAL_SHARD_MEMBERSHIP)

    projected_graph_sha256 = canonical_sha256(projected["upper_graph"])
    projected_identity_sha256 = upper_nodes_identity_sha256(projected["upper_nodes"])
    if projected_graph_sha256 != source_graph_sha256:
        raise AssertionError("projection changed the upper graph")
    if projected_identity_sha256 != source_identity_sha256:
        raise AssertionError("projection changed ordered labels or float32 vector bits")

    artifact_sha256, artifact_sidecar = _write_with_sidecar(
        output_path,
        _json_bytes(projected),
    )
    script_path = Path(__file__).resolve()
    manifest = {
        "format_version": FORMAT_VERSION,
        "record_type": RECORD_TYPE,
        "contract": projection_contract(),
        "source_artifact": {
            "sha256": source_artifact_sha256,
            "size_bytes": source_path.stat().st_size,
            "role": "historical_production_artifact_read_only_by_projection_only",
        },
        "upper_build_provenance": upper_build_provenance,
        "projection_source": {
            "path": str(script_path),
            "sha256": sha256_path(script_path),
            "size_bytes": script_path.stat().st_size,
        },
        "output_artifact": {
            "path": str(output_path),
            "sha256": artifact_sha256,
            "size_bytes": output_path.stat().st_size,
            "sha256_sidecar": str(artifact_sidecar),
        },
        "identity": {
            "generation": int(projected["generation"]),
            "logical_point_count": int(projected["logical_point_count"]),
            "upper_node_count": len(projected["upper_nodes"]),
            "upper_graph_sha256": projected_graph_sha256,
            "ordered_upper_labels_vectors_f32_sha256": projected_identity_sha256,
            "source_and_output_upper_graph_equal": True,
            "source_and_output_ordered_upper_identity_equal": True,
        },
        "redaction_proof": {
            "output_layout_sha256": NEUTRAL_LAYOUT_SHA256,
            "output_physical_point_count": int(projected["physical_point_count"]),
            "every_output_membership_is_exact_neutral_sentinel": True,
            "neutral_membership_sentinel": NEUTRAL_SHARD_MEMBERSHIP,
            "historical_membership_values_emitted": False,
            "historical_membership_distribution_emitted": False,
        },
    }
    manifest_sha256, _manifest_sidecar = _write_with_sidecar(
        manifest_path,
        _json_bytes(manifest),
    )
    return output_path, artifact_sha256, manifest_path, manifest_sha256


def main(argv: list[str] | None = None) -> None:
    artifact, artifact_sha, manifest, manifest_sha = run(parse_args(argv))
    print(f"upper_only_artifact={artifact}")
    print(f"upper_only_artifact_sha256={artifact_sha}")
    print(f"projection_manifest={manifest}")
    print(f"projection_manifest_sha256={manifest_sha}")


if __name__ == "__main__":
    main()
