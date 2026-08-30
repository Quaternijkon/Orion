#!/usr/bin/env python3
"""Freeze upper-only N_native and CNBR owner arrays, then exit before L0.

The only data inputs are a neutral projection of the immutable production
upper artifact, its ordered upper-vector extraction, and frozen production-
upper self-navigation rows.  The projection contains no historical L0 layout.
There is intentionally no attachment, observed-load, multi-assignment, query,
ground-truth, flow, fission, or lower-layout input or code path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import sys
import time
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.l1_balance import native_cnbr_core as core  # noqa: E402
from experiments.l1_balance import (  # noqa: E402
    project_upper_only_artifact as upper_projection,
)


FORMAT_VERSION = 1
NUM_PARTITIONS = 32
UPPER_SUBSET_DENOMINATOR = 32
UPPER_NAVIGATION_SEARCH_EF = 100
FORMAL_CANDIDATE_SET = "formal"
FROZEN_GRID_CANDIDATE_SET = "frozen-grid"
CANDIDATE_SETS = (FORMAL_CANDIDATE_SET, FROZEN_GRID_CANDIDATE_SET)
FORMAL_STAGE = "upper_only_n_native_cnbr_owners_frozen"
GRID_STAGE = "upper_only_cnbr_frozen_family_owners_frozen"
MANIFEST_NAME = "phase-a.manifest.json"
PERFORMANCE_NAME = "phase-a.performance.json"
PROTOCOL_ADDENDUM_JSON = REPO_ROOT / "experiments/l1_balance/CNBR_CORRECTION_ADDENDUM.json"
PROTOCOL_ADDENDUM_MD = REPO_ROOT / "experiments/l1_balance/CNBR_CORRECTION_ADDENDUM.md"
UPPER_ONLY_PROJECTION_SCRIPT = (
    REPO_ROOT / "experiments/l1_balance/project_upper_only_artifact.py"
)
UPPER_ONLY_PROJECTION_RECORD_TYPE = upper_projection.RECORD_TYPE
NEUTRAL_LAYOUT_SHA256 = upper_projection.NEUTRAL_LAYOUT_SHA256
NEUTRAL_SHARD_MEMBERSHIP = upper_projection.NEUTRAL_SHARD_MEMBERSHIP
UPPER_BUILD_PARAMETER_KEYS = upper_projection.UPPER_BUILD_PARAMETER_KEYS

FORBIDDEN_STAGE_INVOCATIONS = (
    "full_attachment_reads",
    "point_to_l1s_reads",
    "multi_assignment_reads",
    "weight_recalibration_calls",
    "fission_calls",
    "topology_refinement_calls",
    "capacity_flow_calls",
    "lower_layout_repair_calls",
)

GRID_OWNER_KEYS = {
    "9/4": "CNBR_9_4",
    "2/1": "CNBR_2_1",
    "7/4": "CNBR_7_4",
    "8/5": "CNBR_8_5",
    "3/2": "CNBR_3_2",
    "7/5": "CNBR_7_5",
}

OWNER_DTYPE = "<i4"
OWNER_ENCODING = "i32le"
MASS_DTYPE = "<u8"
MASS_ENCODING = "u64le"
ESTIMATED_PARTITION_MASS_FIELD = "estimated_partition_proxy_masses"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--upper-only-projection-manifest", required=True)
    parser.add_argument("--upper-input-manifest", required=True)
    parser.add_argument("--upper-navigation-hits", required=True)
    parser.add_argument("--upper-navigation-manifest", required=True)
    parser.add_argument(
        "--candidate-set",
        choices=CANDIDATE_SETS,
        default=FORMAL_CANDIDATE_SET,
        help="Closed formal pair or the immutable six-ratio family; no free ratio.",
    )
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


def sha256_path(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def bytes_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return bytes_sha256(encoded)


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _write_bytes_new(path: Path, data: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(data)


def _write_json_new(path: Path, value: Any) -> None:
    _write_bytes_new(path, _json_bytes(value))


def _absolute_input_path(raw: str, field: str) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        raise ValueError(f"{field} must be an absolute path")
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{field} does not reference a file: {path}")
    return path


def _load_json_object(path: Path, field: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{field} is not valid UTF-8 JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{field} must contain a JSON object")
    return value


def _require_equal(actual: Any, expected: Any, field: str) -> None:
    if actual != expected:
        raise ValueError(f"{field} mismatch: expected {expected!r}, got {actual!r}")


def _relative(path: Path, output_dir: Path) -> str:
    return path.relative_to(output_dir).as_posix()


def _peak_rss_bytes() -> int:
    raw = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return raw * 1024 if sys.platform != "darwin" else raw


def _load_production_upper(
    artifact_path: Path,
) -> tuple[
    dict[str, Any],
    tuple[tuple[int, ...], ...],
    np.ndarray,
    np.ndarray,
    str,
    str,
    int,
]:
    artifact = _load_json_object(artifact_path, "artifact")
    _require_equal(artifact.get("shard_count"), NUM_PARTITIONS, "artifact.shard_count")
    logical_point_count = artifact.get("logical_point_count")
    if isinstance(logical_point_count, bool) or not isinstance(logical_point_count, int):
        raise ValueError("artifact.logical_point_count must be an integer")
    if logical_point_count <= 0:
        raise ValueError("artifact.logical_point_count must be positive")
    _require_equal(
        artifact.get("layout_sha256"),
        NEUTRAL_LAYOUT_SHA256,
        "artifact.layout_sha256",
    )
    _require_equal(
        artifact.get("physical_point_count"),
        logical_point_count,
        "artifact.physical_point_count",
    )

    upper_nodes = artifact.get("upper_nodes")
    if not isinstance(upper_nodes, list) or not upper_nodes:
        raise ValueError("artifact has no upper_nodes")
    if len(upper_nodes) != logical_point_count // UPPER_SUBSET_DENOMINATOR:
        raise ValueError("artifact upper node count is not floor(N/32)")
    for index, node in enumerate(upper_nodes):
        if not isinstance(node, dict):
            raise ValueError(f"artifact.upper_nodes[{index}] is not an object")
        if node.get("shard_membership") != NEUTRAL_SHARD_MEMBERSHIP:
            raise ValueError(
                "artifact contains a non-neutral historical shard membership at "
                f"upper_nodes[{index}]"
            )

    schema = artifact.get("vector_schema")
    if not isinstance(schema, dict):
        raise ValueError("artifact has no vector_schema")
    dimension = schema.get("dimension")
    if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0:
        raise ValueError("artifact vector dimension must be positive")

    try:
        labels = np.asarray([int(node["label"]) for node in upper_nodes], dtype="<u8")
        vectors = np.asarray([node["vector"] for node in upper_nodes], dtype="<f4")
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError("artifact upper_nodes have invalid labels or vectors") from exc
    if vectors.shape != (len(upper_nodes), dimension):
        raise ValueError("artifact upper vectors are not a rectangular U x dimension matrix")
    if not bool(np.isfinite(vectors).all()):
        raise ValueError("artifact upper vectors contain a non-finite value")
    if len(np.unique(labels)) != len(labels):
        raise ValueError("artifact upper labels are not unique")

    graph = artifact.get("upper_graph")
    if not isinstance(graph, dict):
        raise ValueError("artifact has no production upper_graph")
    graph_nodes = graph.get("nodes")
    if not isinstance(graph_nodes, list) or len(graph_nodes) != len(labels):
        raise ValueError("production upper graph node count mismatch")
    label_to_local = {int(label): index for index, label in enumerate(labels.tolist())}
    adjacency: list[tuple[int, ...] | None] = [None] * len(labels)
    seen_graph_nodes: set[int] = set()
    for graph_index, graph_node in enumerate(graph_nodes):
        if not isinstance(graph_node, dict):
            raise ValueError(f"upper_graph.nodes[{graph_index}] is not an object")
        try:
            label = int(graph_node["label"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"upper_graph.nodes[{graph_index}] has invalid label") from exc
        local = label_to_local.get(label)
        if local is None:
            raise ValueError("upper graph contains a label outside upper_nodes")
        if local in seen_graph_nodes:
            raise ValueError("upper graph repeats a node label")
        seen_graph_nodes.add(local)
        levels = graph_node.get("neighbors_by_level")
        if not isinstance(levels, list) or not levels or not isinstance(levels[0], list):
            raise ValueError("upper graph node lacks a level-0 neighbor list")
        neighbors: list[int] = []
        seen_neighbors: set[int] = set()
        for raw_neighbor in levels[0]:
            try:
                neighbor_label = int(raw_neighbor)
            except (TypeError, ValueError) as exc:
                raise ValueError("upper graph contains an invalid neighbor label") from exc
            neighbor = label_to_local.get(neighbor_label)
            if neighbor is None:
                raise ValueError("upper graph contains a neighbor outside upper_nodes")
            if neighbor == local:
                raise ValueError("upper graph contains a level-0 self edge")
            if neighbor in seen_neighbors:
                raise ValueError("upper graph repeats a level-0 neighbor")
            seen_neighbors.add(neighbor)
            neighbors.append(neighbor)
        adjacency[local] = tuple(neighbors)
    if len(seen_graph_nodes) != len(labels) or any(row is None for row in adjacency):
        raise ValueError("upper graph does not cover every upper node exactly once")

    try:
        entry_label = int(graph["entry_point"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("upper graph has an invalid entry point") from exc
    if entry_label not in label_to_local:
        raise ValueError("upper graph entry point is outside upper_nodes")

    navigator = {
        "vector_schema": schema,
        "upper_labels": [int(value) for value in labels.tolist()],
        "upper_vectors": [node["vector"] for node in upper_nodes],
        "upper_graph": graph,
    }
    normalized = core.normalize_undirected_adjacency(
        tuple(row for row in adjacency if row is not None)
    )
    edge_count = sum(len(row) for row in normalized) // 2
    return (
        artifact,
        normalized,
        np.ascontiguousarray(vectors, dtype="<f4"),
        np.ascontiguousarray(labels, dtype="<u8"),
        canonical_sha256(graph),
        canonical_sha256(navigator),
        edge_count,
    )


def _validate_upper_input_manifest(
    *,
    artifact: dict[str, Any],
    artifact_path: Path,
    artifact_sha256: str,
    labels: np.ndarray,
    vectors: np.ndarray,
    manifest_path: Path,
) -> dict[str, Any]:
    manifest = _load_json_object(manifest_path, "upper-input manifest")
    _require_equal(manifest.get("format_version"), 1, "upper-input.format_version")
    vectors_path = _absolute_input_path(str(manifest.get("vectors", "")), "upper-input.vectors")
    labels_path = _absolute_input_path(str(manifest.get("labels", "")), "upper-input.labels")
    vector_bytes = vectors.astype("<f4", copy=False).tobytes(order="C")
    label_bytes = labels.astype("<u8", copy=False).tobytes(order="C")
    expected = {
        "artifact": str(artifact_path),
        "artifact_sha256": artifact_sha256,
        "generation": int(artifact["generation"]),
        "row_count": len(labels),
        "dimension": int(vectors.shape[1]),
        "vectors": str(vectors_path),
        "vectors_sha256": sha256_path(vectors_path),
        "vectors_size_bytes": len(vector_bytes),
        "labels": str(labels_path),
        "labels_sha256": sha256_path(labels_path),
        "labels_size_bytes": len(label_bytes),
    }
    for field, value in expected.items():
        _require_equal(manifest.get(field), value, f"upper-input.{field}")
    if vectors_path.read_bytes() != vector_bytes:
        raise ValueError("upper-input vectors differ from artifact float32 bits")
    if labels_path.read_bytes() != label_bytes:
        raise ValueError("upper-input labels differ from artifact upper-node order")
    return {
        **manifest,
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_path(manifest_path),
        "vectors_path": str(vectors_path),
        "labels_path": str(labels_path),
    }


def _validate_upper_only_projection_manifest(
    *,
    manifest_path: Path,
    artifact_path: Path,
    artifact_sha256: str,
    artifact: dict[str, Any],
    upper_graph_sha256: str,
) -> dict[str, Any]:
    manifest = _load_json_object(manifest_path, "upper-only projection manifest")
    expected_manifest_keys = {
        "format_version",
        "record_type",
        "contract",
        "source_artifact",
        "upper_build_provenance",
        "projection_source",
        "output_artifact",
        "identity",
        "redaction_proof",
    }
    if set(manifest) != expected_manifest_keys:
        raise ValueError("upper-only projection manifest schema drifted")
    _require_equal(manifest.get("format_version"), 1, "projection.format_version")
    _require_equal(
        manifest.get("record_type"),
        UPPER_ONLY_PROJECTION_RECORD_TYPE,
        "projection.record_type",
    )
    contract = manifest.get("contract")
    if not isinstance(contract, dict):
        raise ValueError("upper-only projection manifest lacks contract")
    _require_equal(
        contract,
        upper_projection.projection_contract(),
        "projection.contract",
    )

    source_artifact = manifest.get("source_artifact")
    if not isinstance(source_artifact, dict) or set(source_artifact) != {
        "sha256",
        "size_bytes",
        "role",
    }:
        raise ValueError("projection source artifact provenance schema drifted")
    source_sha = source_artifact.get("sha256")
    if (
        not isinstance(source_sha, str)
        or len(source_sha) != 64
        or any(character not in "0123456789abcdef" for character in source_sha)
    ):
        raise ValueError("projection source artifact checksum is invalid")
    if (
        isinstance(source_artifact.get("size_bytes"), bool)
        or not isinstance(source_artifact.get("size_bytes"), int)
        or source_artifact["size_bytes"] <= 0
        or source_artifact.get("role")
        != "historical_production_artifact_read_only_by_projection_only"
    ):
        raise ValueError("projection source artifact provenance is invalid")

    upper_build_provenance = manifest.get("upper_build_provenance")
    if not isinstance(upper_build_provenance, dict) or set(
        upper_build_provenance
    ) != {
        "source_manifest_sha256",
        "source_manifest_size_bytes",
        "parameters",
    }:
        raise ValueError("projection upper-build provenance schema drifted")
    build_sha = upper_build_provenance.get("source_manifest_sha256")
    if (
        not isinstance(build_sha, str)
        or len(build_sha) != 64
        or any(character not in "0123456789abcdef" for character in build_sha)
    ):
        raise ValueError("projection source build checksum is invalid")
    build_size = upper_build_provenance.get("source_manifest_size_bytes")
    if (
        isinstance(build_size, bool)
        or not isinstance(build_size, int)
        or build_size <= 0
    ):
        raise ValueError("projection source build size is invalid")
    upper_build_parameters = upper_build_provenance.get("parameters")
    if not isinstance(upper_build_parameters, dict) or set(
        upper_build_parameters
    ) != set(UPPER_BUILD_PARAMETER_KEYS):
        raise ValueError("projection upper-build parameter whitelist drifted")
    for key in UPPER_BUILD_PARAMETER_KEYS:
        value = upper_build_parameters[key]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(
                f"projection upper-build parameter {key} must be a positive integer"
            )

    output = manifest.get("output_artifact")
    if not isinstance(output, dict):
        raise ValueError("upper-only projection manifest lacks output_artifact")
    expected_output = {
        "path": str(artifact_path),
        "sha256": artifact_sha256,
        "size_bytes": artifact_path.stat().st_size,
        "sha256_sidecar": str(
            artifact_path.with_name(artifact_path.name + ".sha256")
        ),
    }
    for field, expected in expected_output.items():
        _require_equal(
            output.get(field),
            expected,
            f"projection.output_artifact.{field}",
        )
    artifact_sidecar = Path(expected_output["sha256_sidecar"])
    if artifact_sidecar.read_text(encoding="ascii").strip() != artifact_sha256:
        raise ValueError("upper-only artifact checksum sidecar mismatch")

    projection_source = manifest.get("projection_source")
    if not isinstance(projection_source, dict):
        raise ValueError("upper-only projection manifest lacks projection_source")
    projection_script = UPPER_ONLY_PROJECTION_SCRIPT.resolve()
    expected_source = {
        "path": str(projection_script),
        "sha256": sha256_path(projection_script),
        "size_bytes": projection_script.stat().st_size,
    }
    for field, expected in expected_source.items():
        _require_equal(
            projection_source.get(field),
            expected,
            f"projection.projection_source.{field}",
        )

    identity = manifest.get("identity")
    if not isinstance(identity, dict):
        raise ValueError("upper-only projection manifest lacks identity")
    identity_expected = {
        "generation": int(artifact["generation"]),
        "logical_point_count": int(artifact["logical_point_count"]),
        "upper_node_count": len(artifact["upper_nodes"]),
        "upper_graph_sha256": upper_graph_sha256,
        "source_and_output_upper_graph_equal": True,
        "source_and_output_ordered_upper_identity_equal": True,
    }
    for field, expected in identity_expected.items():
        _require_equal(identity.get(field), expected, f"projection.identity.{field}")
    ordered_identity = identity.get("ordered_upper_labels_vectors_f32_sha256")
    if not isinstance(ordered_identity, str) or len(ordered_identity) != 64:
        raise ValueError("projection ordered upper identity checksum is invalid")

    redaction = manifest.get("redaction_proof")
    if not isinstance(redaction, dict):
        raise ValueError("upper-only projection manifest lacks redaction_proof")
    redaction_expected = {
        "output_layout_sha256": NEUTRAL_LAYOUT_SHA256,
        "output_physical_point_count": int(artifact["logical_point_count"]),
        "every_output_membership_is_exact_neutral_sentinel": True,
        "neutral_membership_sentinel": NEUTRAL_SHARD_MEMBERSHIP,
        "historical_membership_values_emitted": False,
        "historical_membership_distribution_emitted": False,
    }
    for field, expected in redaction_expected.items():
        _require_equal(redaction.get(field), expected, f"projection.redaction_proof.{field}")

    return {
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_path(manifest_path),
        "manifest_size_bytes": manifest_path.stat().st_size,
        "source_artifact": source_artifact,
        "upper_build_provenance": upper_build_provenance,
        "projection_source": projection_source,
        "identity": identity,
        "redaction_proof": redaction,
        "contract": contract,
    }


def _map_hits_to_local(hits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    order = np.argsort(labels, kind="stable")
    sorted_labels = labels[order]
    flat = hits.reshape(-1)
    positions = np.searchsorted(sorted_labels, flat)
    valid = positions < len(sorted_labels)
    if bool(np.any(valid)):
        valid_indices = np.flatnonzero(valid)
        valid[valid_indices] = sorted_labels[positions[valid_indices]] == flat[valid_indices]
    if not bool(np.all(valid)):
        first = int(np.flatnonzero(~valid)[0])
        raise ValueError(
            "upper-navigation hits contain a label outside the production upper graph "
            f"at flat offset {first}"
        )
    mapped = order[positions].astype(np.int32, copy=False)
    return np.ascontiguousarray(mapped.reshape(hits.shape), dtype=np.int32)


def _duplicate_tie_proof(
    prepared_navigation: core.PreparedUpperNavigation,
    labels: np.ndarray,
    vectors: np.ndarray,
) -> tuple[list[dict[str, Any]], str, int]:
    records: list[dict[str, Any]] = []
    rows = prepared_navigation.rows
    self_ranks = prepared_navigation.self_ranks
    self_first_count = int(np.count_nonzero(self_ranks == 0))
    for raw_row in np.flatnonzero(self_ranks != 0):
        row = int(raw_row)
        row_values = rows[row]
        self_rank = int(self_ranks[row])
        query_bits = np.ascontiguousarray(vectors[row], dtype="<f4").tobytes()
        preceding = row_values[:self_rank]
        for hit in preceding:
            candidate_bits = np.ascontiguousarray(
                vectors[int(hit)], dtype="<f4"
            ).tobytes()
            if candidate_bits != query_bits:
                raise ValueError(
                    f"upper self-navigation row {row} has a non-duplicate hit before self"
                )
        records.append(
            {
                "row": row,
                "query_label": int(labels[row]),
                "self_rank": self_rank,
                "preceding_labels": [int(labels[int(hit)]) for hit in preceding],
                "vector_bits_sha256": bytes_sha256(query_bits),
            }
        )
    return records, canonical_sha256(records), self_first_count


def _validate_navigation_manifest(
    *,
    artifact: dict[str, Any],
    artifact_path: Path,
    artifact_sha256: str,
    upper_input: dict[str, Any],
    hits_path: Path,
    manifest_path: Path,
    labels: np.ndarray,
    vectors: np.ndarray,
) -> tuple[
    dict[str, Any],
    core.PreparedUpperNavigation,
    list[dict[str, Any]],
    str,
    int,
]:
    manifest = _load_json_object(manifest_path, "upper-navigation manifest")
    _require_equal(manifest.get("format_version"), 1, "upper-navigation.format_version")
    manifest_hits_path = _absolute_input_path(
        str(manifest.get("hits_path", "")),
        "upper-navigation.hits_path",
    )
    manifest_vectors_path = _absolute_input_path(
        str(manifest.get("vectors_path", "")),
        "upper-navigation.vectors_path",
    )
    _require_equal(manifest_hits_path, hits_path, "upper-navigation.hits_path")
    _require_equal(
        str(manifest_vectors_path),
        upper_input["vectors_path"],
        "upper-navigation.vectors_path",
    )
    expected_size = len(labels) * core.UPPER_NAVIGATION_TOP_K * np.dtype("<u8").itemsize
    expected = {
        "artifact_path": str(artifact_path),
        "artifact_sha256": artifact_sha256,
        "generation": int(artifact["generation"]),
        "upper_graph_present": True,
        "source_upper_k": int(artifact["upper_k"]),
        "source_upper_ef_search": int(artifact["upper_ef_search"]),
        "vectors_sha256": upper_input["vectors_sha256"],
        "row_count": len(labels),
        "dimension": int(vectors.shape[1]),
        "top_k": core.UPPER_NAVIGATION_TOP_K,
        "search_ef": UPPER_NAVIGATION_SEARCH_EF,
        "hits_path": str(hits_path),
        "hits_sha256": sha256_path(hits_path),
        "hits_size_bytes": expected_size,
    }
    for field, value in expected.items():
        _require_equal(manifest.get(field), value, f"upper-navigation.{field}")
    if hits_path.stat().st_size != expected_size:
        raise ValueError("upper-navigation hit file size mismatch")
    hits = np.fromfile(hits_path, dtype="<u8").reshape(
        len(labels), core.UPPER_NAVIGATION_TOP_K
    )
    navigation_local = _map_hits_to_local(hits, labels)
    prepared_navigation = core.prepare_upper_navigation(
        navigation_local,
        len(labels),
    )
    tie_records, tie_sha256, self_first_count = _duplicate_tie_proof(
        prepared_navigation,
        labels,
        vectors,
    )
    return (
        {
            **manifest,
            "manifest_path": str(manifest_path),
            "manifest_sha256": sha256_path(manifest_path),
        },
        prepared_navigation,
        tie_records,
        tie_sha256,
        self_first_count,
    )


def _partition_masses(
    owner: tuple[int, ...],
    mass: core.UpperNavigationMass,
) -> tuple[int, ...]:
    loads = np.zeros(NUM_PARTITIONS, dtype=np.int64)
    np.add.at(
        loads,
        np.asarray(owner, dtype=np.int32),
        np.asarray(mass.values, dtype=np.int64),
    )
    return tuple(int(value) for value in loads.tolist())


def _candidate_specs(candidate_set: str) -> list[tuple[str, str]]:
    if candidate_set == FORMAL_CANDIDATE_SET:
        return [("C_CNBR", core.CNBR_SELECTED_TRIGGER_ID)]
    if candidate_set == FROZEN_GRID_CANDIDATE_SET:
        return [(GRID_OWNER_KEYS[ratio], ratio) for ratio in core.CNBR_FROZEN_TRIGGER_FAMILY]
    raise AssertionError(f"unknown candidate set {candidate_set}")


def phase_a_stage(candidate_set: str) -> str:
    if candidate_set == FORMAL_CANDIDATE_SET:
        return FORMAL_STAGE
    if candidate_set == FROZEN_GRID_CANDIDATE_SET:
        return GRID_STAGE
    raise ValueError("candidate-set is outside the closed formal/grid enum")


def phase_a_owner_roles(candidate_set: str) -> dict[str, str]:
    roles = {"N_native": "balance_free_orion_core"}
    candidate_role = (
        "replacement_candidate"
        if candidate_set == FORMAL_CANDIDATE_SET
        else "frozen_cnbr_family_member"
    )
    roles.update({name: candidate_role for name, _ratio in _candidate_specs(candidate_set)})
    return roles


def phase_a_contract() -> dict[str, Any]:
    """Canonical architecture boundary imported by the Phase-B validator."""

    return {
        "partition_input_scope": (
            "neutral_upper_only_projection_vectors_and_upper_self_navigation_only"
        ),
        "accepts_historical_shard_membership": False,
        "requires_neutral_upper_only_projection": True,
        "accepts_full_l0_attachments": False,
        "accepts_l0_load": False,
        "accepts_multi_assignment": False,
        "accepts_query_gt": False,
        "owners_immutable": True,
        "phase_b_must_verify_before_l0_open": True,
        "construction_process_accepts_l0_inputs": False,
        "construction_process_accepts_query_inputs": False,
        "construction_process_accepts_ground_truth": False,
        "construction_process_accepts_multi_assignment_state": False,
        "owner_checksum_freeze_precedes_l0_access": True,
        "generator_exits_before_l0_materialization": True,
        "multi_assignment_applies_after_owner_freeze": True,
        "multi_assignment_orthogonal_to_balance": True,
        "separate_phase_b_materializer_required": True,
        "identity_manifest_excludes_nondeterministic_timing": True,
        "performance_record_binds_identity_manifest_one_way": True,
    }


def phase_a_forbidden_stage_invocations() -> dict[str, int]:
    return {name: 0 for name in FORBIDDEN_STAGE_INVOCATIONS}


def phase_a_output_keys(candidate_set: str) -> tuple[str, ...]:
    candidate_owner_keys = tuple(
        f"{name}_owner" for name, _ratio in _candidate_specs(candidate_set)
    )
    return (
        "mass_values",
        "N_native_owner",
        "source_core",
        "source_generator",
        *candidate_owner_keys,
        "owners_dir",
        "records_dir",
        "traces_dir",
        "source_dir",
        "performance_record",
    )


def phase_a_fixed_parameters(candidate_set: str) -> dict[str, Any]:
    return {
        "num_partitions": NUM_PARTITIONS,
        "upper_subset_denominator": UPPER_SUBSET_DENOMINATOR,
        "N_native": {
            "algorithm": "unconstrained_kmeans",
            "seed": core.NATIVE_KMEANS_SEED,
            "iterations": core.NATIVE_KMEANS_ITERATIONS,
            "assignment": "plain_argmin",
            "distance": "squared_l2",
            "empty_cluster_during_training": "retain_previous_centroid",
            "empty_partition_after_final_argmin": "fail_closed",
            "tie_break": "lowest_partition_id",
            "weights": False,
            "capacity": False,
            "quota": False,
            "owner_repair": False,
        },
        "proxy_mass": {
            "proxy_mass_contract_id": core.PROXY_MASS_CONTRACT_ID,
            "source": core.MASS_SOURCE,
            "transform": core.MASS_TRANSFORM,
            "estimator_version": core.MASS_ESTIMATOR_VERSION,
            "top_k": core.UPPER_NAVIGATION_TOP_K,
            "search_ef": UPPER_NAVIGATION_SEARCH_EF,
            "includes_self": True,
            "rank_weighting": False,
            "observed_l0_load": False,
        },
        "C_CNBR": {
            "algorithm": "cnbr",
            "candidate_set": candidate_set,
            "trigger_ratio": {
                "numerator": core.CNBR_TRIGGER_NUMERATOR,
                "denominator": core.CNBR_TRIGGER_DENOMINATOR,
            },
            "selected_trigger_id": core.CNBR_SELECTED_TRIGGER_ID,
            "frozen_trigger_family": {
                key: {"numerator": value[0], "denominator": value[1]}
                for key, value in core.CNBR_FROZEN_TRIGGER_FAMILY.items()
            },
            "max_rounds": core.CNBR_MAX_ROUNDS,
            "node_move_limit": core.CNBR_NODE_MOVE_LIMIT,
            "requires_cross_owner_direct_graph_neighbor": (
                core.CNBR_REQUIRES_CROSS_OWNER_DIRECT_GRAPH_NEIGHBOR
            ),
            "self_navigation_only_target_allowed": (
                core.CNBR_SELF_NAVIGATION_ONLY_TARGET_ALLOWED
            ),
            "snapshot_proposals": True,
            "sequential_commit": True,
            "integer_load_predicates": True,
            "integer_cumulative_cut_predicate": (
                "candidate_cut_count*100<=n_native_cut_count*103"
            ),
            "per_move_edge_delta_hard_gate": False,
            "per_move_navigation_loss_hard_gate": False,
            "round_gate_failure": "rollback_entire_round_and_stop",
        },
    }


def phase_a_graph_gate_contract() -> dict[str, Any]:
    return {
        "version": "l1-topology-preregistered-v1",
        "reference": "N_native",
        "edge_cut_max_ratio": core.EDGE_CUT_MAX_RATIO,
        "retained_degree_mean_min_ratio": core.RETAINED_DEGREE_MIN_RATIO,
        "retained_degree_p10_max_drop": core.RETAINED_DEGREE_P10_MAX_DROP,
        "isolated_fraction_max_delta": core.ISOLATED_FRACTION_MAX_DELTA,
        "isolated_fraction_absolute_max": core.ISOLATED_FRACTION_ABSOLUTE_MAX,
        "largest_component_mean_max_drop": core.LARGEST_COMPONENT_MEAN_MAX_DROP,
        "largest_component_min_floor": core.LARGEST_COMPONENT_MIN_FLOOR,
        "all_required": True,
        "all_six_required": True,
        "round_failure_action": "rollback_entire_round_and_stop",
    }


def phase_a_schema(candidate_set: str) -> dict[str, Any]:
    """Single source of truth for Phase-A/Phase-B schema validation."""

    return {
        "format_version": FORMAT_VERSION,
        "stage": phase_a_stage(candidate_set),
        "candidate_set": candidate_set,
        "owner_roles": phase_a_owner_roles(candidate_set),
        "owner_dtype": OWNER_DTYPE,
        "owner_encoding": OWNER_ENCODING,
        "mass_dtype": MASS_DTYPE,
        "mass_encoding": MASS_ENCODING,
        "estimated_partition_mass_field": ESTIMATED_PARTITION_MASS_FIELD,
        "contract": phase_a_contract(),
        "forbidden_stage_invocations": phase_a_forbidden_stage_invocations(),
        "fixed_parameters": phase_a_fixed_parameters(candidate_set),
        "graph_gate_contract": phase_a_graph_gate_contract(),
        "output_keys": list(phase_a_output_keys(candidate_set)),
    }


def _validate_protocol_addendum() -> dict[str, str]:
    addendum = _load_json_object(PROTOCOL_ADDENDUM_JSON, "CNBR correction addendum")
    _require_equal(addendum.get("id"), "cnbr-replacement-correction-v1", "addendum.id")
    proxy = addendum.get("proxy_mass")
    if not isinstance(proxy, dict):
        raise ValueError("CNBR addendum lacks proxy_mass")
    _require_equal(proxy.get("top_k"), core.UPPER_NAVIGATION_TOP_K, "addendum.proxy_mass.top_k")
    _require_equal(proxy.get("ef"), UPPER_NAVIGATION_SEARCH_EF, "addendum.proxy_mass.ef")
    cnbr = addendum.get("cnbr")
    if not isinstance(cnbr, dict):
        raise ValueError("CNBR addendum lacks cnbr")
    ratio = cnbr.get("overload_trigger_ratio")
    if not isinstance(ratio, dict):
        raise ValueError("CNBR addendum lacks selected overload ratio")
    _require_equal(ratio.get("numerator"), core.CNBR_TRIGGER_NUMERATOR, "addendum.cnbr.trigger.numerator")
    _require_equal(ratio.get("denominator"), core.CNBR_TRIGGER_DENOMINATOR, "addendum.cnbr.trigger.denominator")
    _require_equal(cnbr.get("node_move_limit"), core.CNBR_NODE_MOVE_LIMIT, "addendum.cnbr.node_move_limit")
    _require_equal(
        cnbr.get("requires_cross_owner_graph_neighbor"),
        core.CNBR_REQUIRES_CROSS_OWNER_DIRECT_GRAPH_NEIGHBOR,
        "addendum.cnbr.requires_cross_owner_graph_neighbor",
    )
    return {
        "json_path": str(PROTOCOL_ADDENDUM_JSON.resolve()),
        "json_sha256": sha256_path(PROTOCOL_ADDENDUM_JSON),
        "markdown_path": str(PROTOCOL_ADDENDUM_MD.resolve()),
        "markdown_sha256": sha256_path(PROTOCOL_ADDENDUM_MD),
    }


def run(args: argparse.Namespace) -> tuple[Path, str]:
    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        raise ValueError("output-dir must be an absolute path")
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output_dir}")
    if args.candidate_set not in CANDIDATE_SETS:
        raise ValueError("candidate-set is outside the closed formal/grid enum")

    total_wall_start = time.perf_counter()
    total_cpu_start = time.process_time()
    evidence_preflight_wall_start = time.perf_counter()
    evidence_preflight_cpu_start = time.process_time()
    artifact_path = _absolute_input_path(args.artifact, "artifact")
    projection_manifest_path = _absolute_input_path(
        args.upper_only_projection_manifest,
        "upper-only-projection-manifest",
    )
    upper_input_manifest_path = _absolute_input_path(
        args.upper_input_manifest,
        "upper-input-manifest",
    )
    navigation_hits_path = _absolute_input_path(
        args.upper_navigation_hits,
        "upper-navigation-hits",
    )
    navigation_manifest_path = _absolute_input_path(
        args.upper_navigation_manifest,
        "upper-navigation-manifest",
    )
    core_path = Path(core.__file__).resolve()
    generator_path = Path(__file__).resolve()
    source_originals = {
        "core": core_path,
        "generator": generator_path,
        "upper_only_projection": UPPER_ONLY_PROJECTION_SCRIPT.resolve(),
        "protocol_addendum_json": PROTOCOL_ADDENDUM_JSON.resolve(),
        "protocol_addendum_markdown": PROTOCOL_ADDENDUM_MD.resolve(),
    }
    source_snapshot = {
        key: {
            "sha256": sha256_path(path),
            "size_bytes": path.stat().st_size,
        }
        for key, path in source_originals.items()
    }
    protocol_sources = _validate_protocol_addendum()
    evidence_preflight_wall_seconds = (
        time.perf_counter() - evidence_preflight_wall_start
    )
    evidence_preflight_cpu_seconds = time.process_time() - evidence_preflight_cpu_start

    # Loading and normalizing the already-built upper state is shared by
    # N_native and C_CNBR.  In the integrated production path this state is
    # handed over in memory by upper-graph construction, so it is reported
    # separately and is not charged to the incremental balance patch.
    shared_input_wall_start = time.perf_counter()
    shared_input_cpu_start = time.process_time()
    (
        artifact,
        adjacency,
        vectors,
        labels,
        upper_graph_sha256,
        navigator_sha256,
        upper_edge_count,
    ) = _load_production_upper(artifact_path)
    shared_input_wall_seconds = time.perf_counter() - shared_input_wall_start
    shared_input_cpu_seconds = time.process_time() - shared_input_cpu_start

    # This is the complete incremental input-validation boundary used by the
    # 5% construction-cost gate.  It includes the projected-artifact checksum,
    # projection provenance/redaction checks, exact upper vector/label parity,
    # self-navigation validation, and proxy-mass replay.  It deliberately
    # starts after the shared upper state has been materialized.
    required_validation_wall_start = time.perf_counter()
    required_validation_cpu_start = time.process_time()
    artifact_sha256 = sha256_path(artifact_path)
    upper_only_projection = _validate_upper_only_projection_manifest(
        manifest_path=projection_manifest_path,
        artifact_path=artifact_path,
        artifact_sha256=artifact_sha256,
        artifact=artifact,
        upper_graph_sha256=upper_graph_sha256,
    )
    upper_input = _validate_upper_input_manifest(
        artifact=artifact,
        artifact_path=artifact_path,
        artifact_sha256=artifact_sha256,
        labels=labels,
        vectors=vectors,
        manifest_path=upper_input_manifest_path,
    )

    (
        navigation,
        prepared_navigation,
        tie_records,
        tie_proof_sha256,
        self_first_count,
    ) = _validate_navigation_manifest(
        artifact=artifact,
        artifact_path=artifact_path,
        artifact_sha256=artifact_sha256,
        upper_input=upper_input,
        hits_path=navigation_hits_path,
        manifest_path=navigation_manifest_path,
        labels=labels,
        vectors=vectors,
    )
    mass = prepared_navigation.mass
    required_validation_wall_seconds = (
        time.perf_counter() - required_validation_wall_start
    )
    required_validation_cpu_seconds = (
        time.process_time() - required_validation_cpu_start
    )
    if mass.total_mass != len(labels) * core.UPPER_NAVIGATION_TOP_K:
        raise AssertionError("proxy mass total is not exactly 10U")

    n_wall_start = time.perf_counter()
    n_cpu_start = time.process_time()
    n_result = core.build_n_native_owner(vectors, NUM_PARTITIONS)
    n_wall_seconds = time.perf_counter() - n_wall_start
    n_cpu_seconds = time.process_time() - n_cpu_start
    n_topology_wall_start = time.perf_counter()
    n_topology_cpu_start = time.process_time()
    n_metrics = core.topology_metrics(adjacency, n_result.owner, NUM_PARTITIONS)
    n_gates = core.graph_gate_results(n_metrics, n_metrics)
    n_topology_wall_seconds = time.perf_counter() - n_topology_wall_start
    n_topology_cpu_seconds = time.process_time() - n_topology_cpu_start
    if not all(bool(record["pass"]) for record in n_gates.values()):
        raise ValueError("N_native fails the absolute part of its topology reference gates")

    cnbr_results: dict[str, tuple[str, core.CnbrResult]] = {}
    cnbr_performance: dict[str, dict[str, Any]] = {}
    for owner_key, trigger_id in _candidate_specs(args.candidate_set):
        wall_start = time.perf_counter()
        cpu_start = time.process_time()
        result = core.build_frozen_cnbr_family_owner_prepared(
            adjacency,
            prepared_navigation,
            n_result.owner,
            NUM_PARTITIONS,
            trigger_id=trigger_id,
        )
        wall_seconds = time.perf_counter() - wall_start
        cpu_seconds = time.process_time() - cpu_start
        cnbr_results[owner_key] = (trigger_id, result)
        cnbr_performance[owner_key] = {
            "wall_seconds": wall_seconds,
            "process_cpu_seconds": cpu_seconds,
            "peak_rss_bytes_after": _peak_rss_bytes(),
            "upper_rows": len(labels),
            "edges_examined": result.edges_examined,
            "navigation_hits_examined": result.navigation_hits_examined,
            "topology_evaluations": result.topology_evaluations,
            "topology_edges_examined": result.topology_evaluations * upper_edge_count,
            "proposal_count": sum(int(row["proposal_count"]) for row in result.rounds),
            "commit_decision_count": sum(
                len(row["commit_decisions"]) for row in result.rounds
            ),
            "sequential_accept_count_before_round_gates": sum(
                int(row["accepted_move_count"]) for row in result.rounds
            ),
            "committed_move_count": len(result.moved_nodes),
            "commit_count": len(result.moved_nodes),
            "round_count": len(result.rounds),
        }

    source_stability_wall_start = time.perf_counter()
    source_stability_cpu_start = time.process_time()
    for key, original in source_originals.items():
        snapshot = source_snapshot[key]
        if (
            sha256_path(original) != snapshot["sha256"]
            or original.stat().st_size != snapshot["size_bytes"]
        ):
            raise RuntimeError(
                f"source {key} changed during Phase-A owner construction"
            )
    source_stability_wall_seconds = time.perf_counter() - source_stability_wall_start
    source_stability_cpu_seconds = time.process_time() - source_stability_cpu_start

    output_dir.mkdir(parents=True, exist_ok=False)
    owners_dir = output_dir / "owners"
    records_dir = output_dir / "records"
    traces_dir = output_dir / "traces"
    source_dir = output_dir / "source"
    for directory in (owners_dir, records_dir, traces_dir, source_dir):
        directory.mkdir()

    written_files: list[Path] = []
    source_copies: dict[str, dict[str, Any]] = {}
    for key, original in source_originals.items():
        copy_path = source_dir / original.name
        if copy_path.exists():
            copy_path = source_dir / f"{key}-{original.name}"
        shutil.copyfile(original, copy_path)
        written_files.append(copy_path)
        copied_sha = sha256_path(copy_path)
        if (
            copied_sha != source_snapshot[key]["sha256"]
            or copy_path.stat().st_size != source_snapshot[key]["size_bytes"]
        ):
            raise AssertionError("source copy checksum changed during Phase-A freeze")
        source_copies[key] = {
            "path": _relative(copy_path, output_dir),
            "sha256": copied_sha,
            "size_bytes": copy_path.stat().st_size,
        }

    source_record_path = records_dir / "source-code.record.json"
    source_record = {
        "format_version": FORMAT_VERSION,
        "record_type": "phase_a_source_code",
        "files": source_copies,
        "protocol_contract_id": "cnbr-replacement-correction-v1",
        "proxy_mass_contract_id": core.PROXY_MASS_CONTRACT_ID,
    }
    _write_json_new(source_record_path, source_record)
    written_files.append(source_record_path)
    source_record_sha256 = sha256_path(source_record_path)

    mass_values_path = output_dir / "proxy-mass.u64le"
    _write_bytes_new(
        mass_values_path,
        np.asarray(mass.values, dtype="<u8").tobytes(order="C"),
    )
    written_files.append(mass_values_path)
    mass_record_path = records_dir / "proxy-mass.record.json"
    mass_array = np.asarray(mass.values, dtype=np.int64)
    mass_record = {
        "format_version": FORMAT_VERSION,
        "record_type": "upper_self_navigation_proxy_mass",
        "proxy_mass_contract_id": core.PROXY_MASS_CONTRACT_ID,
        "source": mass.source,
        "transform": mass.transform,
        "estimator_version": mass.estimator_version,
        "semantic_sha256": mass.semantic_sha256,
        "values": {
            "path": _relative(mass_values_path, output_dir),
            "sha256": sha256_path(mass_values_path),
            "dtype": MASS_DTYPE,
            "encoding": MASS_ENCODING,
            "row_count": len(mass.values),
            "count": len(mass.values),
            "size_bytes": mass_values_path.stat().st_size,
        },
        "query_count": mass.query_count,
        "top_k": mass.top_k,
        "search_ef": UPPER_NAVIGATION_SEARCH_EF,
        "total_mass": mass.total_mass,
        "vertex_mass_min": int(mass_array.min()),
        "vertex_mass_max": int(mass_array.max()),
        "vertex_mass_mean": float(mass_array.mean()),
        "self_present_count": len(labels),
        "self_first_count": self_first_count,
        "duplicate_tie_exception_count": len(tie_records),
        "duplicate_tie_exceptions": tie_records,
        "duplicate_tie_proof_sha256": tie_proof_sha256,
        "upper_navigation_hits": str(navigation_hits_path),
        "upper_navigation_hits_sha256": navigation["hits_sha256"],
        "upper_navigation_manifest": str(navigation_manifest_path),
        "upper_navigation_manifest_sha256": navigation["manifest_sha256"],
        "source_code_record_sha256": source_record_sha256,
    }
    _write_json_new(mass_record_path, mass_record)
    written_files.append(mass_record_path)
    mass_record_sha256 = sha256_path(mass_record_path)

    n_owner_path = owners_dir / "N_native.owner.i32le"
    _write_bytes_new(
        n_owner_path,
        np.asarray(n_result.owner, dtype="<i4").tobytes(order="C"),
    )
    written_files.append(n_owner_path)
    n_owner_sha256 = sha256_path(n_owner_path)
    n_owner_record_path = records_dir / "N_native.owner-record.json"
    n_owner_record = {
        "format_version": FORMAT_VERSION,
        "record_type": "phase_a_owner",
        "candidate_id": "N_native",
        "role": phase_a_owner_roles(args.candidate_set)["N_native"],
        "owner": {
            "path": _relative(n_owner_path, output_dir),
            "sha256": n_owner_sha256,
            "dtype": OWNER_DTYPE,
            "encoding": OWNER_ENCODING,
            "row_count": len(n_result.owner),
            "count": len(n_result.owner),
            "size_bytes": n_owner_path.stat().st_size,
        },
        "partition_sizes": list(n_result.partition_sizes),
        ESTIMATED_PARTITION_MASS_FIELD: list(_partition_masses(n_result.owner, mass)),
        "topology_metrics": n_metrics,
        "topology_gates": n_gates,
        "topology_all_pass": True,
        "graph_gates_all_pass": True,
        "source_code_record_sha256": source_record_sha256,
        "proxy_mass_record_sha256": mass_record_sha256,
        "algorithm": phase_a_fixed_parameters(args.candidate_set)["N_native"],
    }
    _write_json_new(n_owner_record_path, n_owner_record)
    written_files.append(n_owner_record_path)
    n_owner_record_sha256 = sha256_path(n_owner_record_path)

    owners_manifest: dict[str, Any] = {
        "N_native": {
            "candidate_id": "N_native",
            "role": phase_a_owner_roles(args.candidate_set)["N_native"],
            "owner": n_owner_record["owner"],
            "owner_record": {
                "path": _relative(n_owner_record_path, output_dir),
                "sha256": n_owner_record_sha256,
                "size_bytes": n_owner_record_path.stat().st_size,
            },
            "phase_a_owner_record_sha256": n_owner_record_sha256,
            "partition_sizes": list(n_result.partition_sizes),
            ESTIMATED_PARTITION_MASS_FIELD: list(
                _partition_masses(n_result.owner, mass)
            ),
            "topology_metrics": n_metrics,
            "topology_gates": n_gates,
            "topology_all_pass": True,
            "graph_gates_all_pass": True,
            "construction_operation_counts": {
                "upper_rows": len(labels),
                "centroid_update_iterations": core.NATIVE_KMEANS_ITERATIONS,
                "final_argmin_passes": 1,
            },
        }
    }

    for owner_key, (trigger_id, result) in cnbr_results.items():
        trigger_num, trigger_den = core.CNBR_FROZEN_TRIGGER_FAMILY[trigger_id]
        selected = trigger_id == core.CNBR_SELECTED_TRIGGER_ID
        candidate_id = "C_CNBR" if selected else owner_key
        owner_role = phase_a_owner_roles(args.candidate_set)[owner_key]
        owner_path = owners_dir / f"{owner_key}.owner.i32le"
        _write_bytes_new(
            owner_path,
            np.asarray(result.owner, dtype="<i4").tobytes(order="C"),
        )
        written_files.append(owner_path)
        owner_sha256 = sha256_path(owner_path)
        trace_path = traces_dir / f"{owner_key}.rounds.json"
        trace_record = {
            "format_version": FORMAT_VERSION,
            "record_type": "cnbr_round_trace",
            "candidate_id": candidate_id,
            "family_member_id": owner_key,
            "trigger_ratio": {"numerator": trigger_num, "denominator": trigger_den},
            "base_owner_sha256": n_owner_sha256,
            "rounds": list(result.rounds),
            "final_moved_nodes": list(result.moved_nodes),
            "final_partition_proxy_masses": list(result.partition_masses),
            "final_topology_metrics": result.topology_metrics,
            "final_topology_gates": result.topology_gates,
        }
        _write_json_new(trace_path, trace_record)
        written_files.append(trace_path)
        trace_sha256 = sha256_path(trace_path)
        proposal_count = sum(int(row["proposal_count"]) for row in result.rounds)
        committed_move_count = len(result.moved_nodes)
        rounds_record = {
            "path": _relative(trace_path, output_dir),
            "sha256": trace_sha256,
            "size_bytes": trace_path.stat().st_size,
            "count": len(result.rounds),
            "proposal_count": proposal_count,
            "committed_move_count": committed_move_count,
            "rolled_back_round_count": sum(
                1 for row in result.rounds if bool(row["rolled_back"])
            ),
        }
        owner_record_path = records_dir / f"{owner_key}.owner-record.json"
        owner_record = {
            "format_version": FORMAT_VERSION,
            "record_type": "phase_a_owner",
            "candidate_id": candidate_id,
            "family_member_id": owner_key,
            "role": owner_role,
            "selected_adoption_candidate": selected,
            "trigger_ratio": {"numerator": trigger_num, "denominator": trigger_den},
            "owner": {
                "path": _relative(owner_path, output_dir),
                "sha256": owner_sha256,
                "dtype": OWNER_DTYPE,
                "encoding": OWNER_ENCODING,
                "row_count": len(result.owner),
                "count": len(result.owner),
                "size_bytes": owner_path.stat().st_size,
            },
            "base_owner_role": "N_native",
            "base_owner_sha256": n_owner_sha256,
            "parent_n_owner_record_sha256": n_owner_record_sha256,
            "trigger_mass_numerator": result.trigger_mass_numerator,
            "trigger_mass_denominator": result.trigger_mass_denominator,
            "rounds": list(result.rounds),
            "round_trace": rounds_record,
            "partition_sizes": np.bincount(
                np.asarray(result.owner, dtype=np.int32), minlength=NUM_PARTITIONS
            ).astype(int).tolist(),
            ESTIMATED_PARTITION_MASS_FIELD: list(result.partition_masses),
            "moved_node_count": len(result.moved_nodes),
            "topology_metrics": result.topology_metrics,
            "topology_gates": result.topology_gates,
            "topology_all_pass": all(
                bool(record["pass"]) for record in result.topology_gates.values()
            ),
            "graph_gates_all_pass": all(
                bool(record["pass"]) for record in result.topology_gates.values()
            ),
            "source_code_record_sha256": source_record_sha256,
            "proxy_mass_record_sha256": mass_record_sha256,
        }
        _write_json_new(owner_record_path, owner_record)
        written_files.append(owner_record_path)
        owner_record_sha256 = sha256_path(owner_record_path)
        owners_manifest[owner_key] = {
            **owner_record,
            "owner_record": {
                "path": _relative(owner_record_path, output_dir),
                "sha256": owner_record_sha256,
                "size_bytes": owner_record_path.stat().st_size,
            },
            "phase_a_owner_record_sha256": owner_record_sha256,
            "construction_operation_counts": {
                key: value
                for key, value in cnbr_performance[owner_key].items()
                if key
                not in {
                    "wall_seconds",
                    "process_cpu_seconds",
                    "peak_rss_bytes_after",
                }
            },
        }

    stage = phase_a_stage(args.candidate_set)
    manifest_path = output_dir / MANIFEST_NAME
    performance_path = output_dir / PERFORMANCE_NAME
    manifest = {
        "format_version": FORMAT_VERSION,
        "stage": stage,
        "candidate_set": args.candidate_set,
        "contract": phase_a_contract(),
        "construction_inputs": {
            "upper_only_projection": upper_only_projection,
            "artifact": {
                "path": str(artifact_path),
                "sha256": artifact_sha256,
                "size_bytes": artifact_path.stat().st_size,
                "generation": int(artifact["generation"]),
                "logical_point_count": int(artifact["logical_point_count"]),
                "shard_count": int(artifact["shard_count"]),
            },
            "artifact_sha256": artifact_sha256,
            "generation": int(artifact["generation"]),
            "logical_point_count": int(artifact["logical_point_count"]),
            "upper_node_count": len(labels),
            "dimension": int(vectors.shape[1]),
            "source_upper_k": int(artifact["upper_k"]),
            "source_upper_ef_search": int(artifact["upper_ef_search"]),
            "upper_graph_sha256": upper_graph_sha256,
            "navigator_sha256": navigator_sha256,
            "ordered_labels": {
                "path": upper_input["labels_path"],
                "sha256": upper_input["labels_sha256"],
                "size_bytes": int(upper_input["labels_size_bytes"]),
                "dtype": MASS_DTYPE,
                "row_count": len(labels),
            },
            "ordered_labels_sha256": upper_input["labels_sha256"],
            "ordered_vectors": {
                "path": upper_input["vectors_path"],
                "sha256": upper_input["vectors_sha256"],
                "size_bytes": int(upper_input["vectors_size_bytes"]),
                "dtype": "<f4",
                "row_count": len(labels),
                "dimension": int(vectors.shape[1]),
            },
            "ordered_vectors_sha256": upper_input["vectors_sha256"],
            "upper_input_manifest": {
                "path": str(upper_input_manifest_path),
                "sha256": upper_input["manifest_sha256"],
                "size_bytes": upper_input_manifest_path.stat().st_size,
            },
            "upper_input_manifest_sha256": upper_input["manifest_sha256"],
            "self_navigation": {
                "path": str(navigation_hits_path),
                "sha256": navigation["hits_sha256"],
                "size_bytes": navigation_hits_path.stat().st_size,
                "dtype": MASS_DTYPE,
                "row_count": len(labels),
                "top_k": core.UPPER_NAVIGATION_TOP_K,
                "search_ef": UPPER_NAVIGATION_SEARCH_EF,
                "self_present_count": len(labels),
                "self_first_count": self_first_count,
                "duplicate_tie_exception_count": len(tie_records),
                "duplicate_tie_proof_sha256": tie_proof_sha256,
                "manifest_path": str(navigation_manifest_path),
                "manifest_sha256": navigation["manifest_sha256"],
            },
            "upper_navigation_hits": str(navigation_hits_path),
            "upper_navigation_hits_sha256": navigation["hits_sha256"],
            "upper_navigation_manifest": str(navigation_manifest_path),
            "upper_navigation_manifest_sha256": navigation["manifest_sha256"],
            "upper_navigation_top_k": core.UPPER_NAVIGATION_TOP_K,
            "upper_navigation_search_ef": UPPER_NAVIGATION_SEARCH_EF,
        },
        "source_code": {
            "core": source_copies["core"],
            "generator": source_copies["generator"],
            "record": {
                "path": _relative(source_record_path, output_dir),
                "sha256": source_record_sha256,
                "size_bytes": source_record_path.stat().st_size,
            },
            "supporting_files": {
                key: value
                for key, value in source_copies.items()
                if key not in {"core", "generator"}
            },
            "protocol_addendum_external_sources": protocol_sources,
        },
        "mass": {
            "mode": core.PROXY_MASS_CONTRACT_ID,
            "proxy_mass_contract_id": core.PROXY_MASS_CONTRACT_ID,
            "source": mass.source,
            "transform": mass.transform,
            "estimator_version": mass.estimator_version,
            "semantic_sha256": mass.semantic_sha256,
            "query_count": mass.query_count,
            "top_k": mass.top_k,
            "total_mass": mass.total_mass,
            "min": int(mass_array.min()),
            "max": int(mass_array.max()),
            "mean": float(mass_array.mean()),
            "values": mass_record["values"],
            "record": {
                "path": _relative(mass_record_path, output_dir),
                "sha256": mass_record_sha256,
                "size_bytes": mass_record_path.stat().st_size,
            },
            "duplicate_tie_exception_count": len(tie_records),
            "duplicate_tie_proof_sha256": tie_proof_sha256,
        },
        "fixed_parameters": phase_a_fixed_parameters(args.candidate_set),
        "graph_gate_contract": phase_a_graph_gate_contract(),
        "owners": owners_manifest,
        "forbidden_stage_invocations": phase_a_forbidden_stage_invocations(),
        "outputs": {
            "mass_values": mass_record["values"]["path"],
            "N_native_owner": owners_manifest["N_native"]["owner"]["path"],
            "source_core": source_copies["core"]["path"],
            "source_generator": source_copies["generator"]["path"],
            **{
                f"{name}_owner": owners_manifest[name]["owner"]["path"]
                for name in cnbr_results
            },
            "owners_dir": "owners",
            "records_dir": "records",
            "traces_dir": "traces",
            "source_dir": "source",
            "performance_record": {"path": PERFORMANCE_NAME},
        },
    }
    _write_json_new(manifest_path, manifest)
    written_files.append(manifest_path)
    manifest_sha256 = sha256_path(manifest_path)
    sidecar_path = output_dir / f"{MANIFEST_NAME}.sha256"
    _write_bytes_new(sidecar_path, (manifest_sha256 + "\n").encode("ascii"))
    written_files.append(sidecar_path)

    total_wall_seconds = time.perf_counter() - total_wall_start
    total_cpu_seconds = time.process_time() - total_cpu_start
    performance = {
        "format_version": FORMAT_VERSION,
        "record_type": "phase_a_construction_performance",
        "phase_a_manifest": MANIFEST_NAME,
        "phase_a_manifest_sha256": manifest_sha256,
        "candidate_set": args.candidate_set,
        "source_artifact_sha256": artifact_sha256,
        "timing_contract": {
            "contract_id": "cnbr-integrated-incremental-cost-v1",
            "cost_gate_includes": [
                "external_upper_self_navigation",
                "required_input_validation_and_mass_replay",
                "formal_cnbr",
            ],
            "required_input_validation_start": (
                "after shared upper artifact deserialization and graph normalization"
            ),
            "required_input_validation_includes": [
                "projected_artifact_sha256",
                "upper_only_projection_provenance_and_redaction",
                "upper_input_manifest_and_exact_byte_parity",
                "self_navigation_manifest_and_row_validation",
                "proxy_mass_replay_and_duplicate_tie_proof",
            ],
            "reported_but_excluded_from_incremental_cost_gate": [
                "evidence_preflight",
                "shared_upper_input_materialization",
                "n_native_kmeans",
                "n_native_reference_topology_validation",
                "post_cnbr_source_stability_validation",
                "evidence_serialization_and_freeze",
            ],
        },
        "shared": {
            "source_snapshot_verified_before_freeze": True,
            "source_snapshot": source_snapshot,
            "upper_rows": len(labels),
            "upper_edge_count": upper_edge_count,
            "required_input_validation_and_mass_replay_wall_seconds": (
                required_validation_wall_seconds
            ),
            "required_input_validation_and_mass_replay_process_cpu_seconds": (
                required_validation_cpu_seconds
            ),
            "evidence_preflight_wall_seconds": evidence_preflight_wall_seconds,
            "evidence_preflight_process_cpu_seconds": evidence_preflight_cpu_seconds,
            "shared_upper_input_materialization_wall_seconds": (
                shared_input_wall_seconds
            ),
            "shared_upper_input_materialization_process_cpu_seconds": (
                shared_input_cpu_seconds
            ),
            "n_native_wall_seconds": n_wall_seconds,
            "n_native_process_cpu_seconds": n_cpu_seconds,
            "n_native_reference_topology_validation_wall_seconds": (
                n_topology_wall_seconds
            ),
            "n_native_reference_topology_validation_process_cpu_seconds": (
                n_topology_cpu_seconds
            ),
            "post_cnbr_source_stability_validation_wall_seconds": (
                source_stability_wall_seconds
            ),
            "post_cnbr_source_stability_validation_process_cpu_seconds": (
                source_stability_cpu_seconds
            ),
            "total_phase_a_wall_seconds": total_wall_seconds,
            "total_phase_a_process_cpu_seconds": total_cpu_seconds,
            "peak_rss_bytes": _peak_rss_bytes(),
        },
        "owners": {
            "N_native": {
                "owner_sha256": n_owner_sha256,
                "owner_record_sha256": n_owner_record_sha256,
                "wall_seconds": n_wall_seconds,
                "process_cpu_seconds": n_cpu_seconds,
                "peak_rss_bytes_after": _peak_rss_bytes(),
                "upper_rows": len(labels),
                "edges_examined": 0,
                "proposal_count": 0,
                "commit_count": 0,
                "round_count": 0,
            },
            **{
                key: {
                    "owner_sha256": owners_manifest[key]["owner"]["sha256"],
                    "owner_record_sha256": owners_manifest[key][
                        "phase_a_owner_record_sha256"
                    ],
                    **metrics,
                }
                for key, metrics in cnbr_performance.items()
            },
        },
    }
    _write_json_new(performance_path, performance)
    written_files.append(performance_path)

    for path in written_files:
        os.chmod(path, 0o444)
    return manifest_path, manifest_sha256


def main(argv: list[str] | None = None) -> None:
    manifest_path, manifest_sha256 = run(parse_args(argv))
    print(f"phase_a_manifest={manifest_path}")
    print(f"phase_a_manifest_sha256={manifest_sha256}")


if __name__ == "__main__":
    main()
