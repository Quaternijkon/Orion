#!/usr/bin/env python3
"""Materialize one audited frozen L1 owner into an Orion import bundle.

Eligibility is established by the final post-freeze screen, never by a CLI
boolean.  This tool fail-closes on any drift in the frozen candidate, source
artifact, partitioner, mass proxy, topology gates, attachment export, canonical
vectors, or the evaluator's exact multi-assignment byte stream.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
from typing import Any

import numpy as np


UNIT_BALANCE_CONTRACT = "unit_exact_quota"
MASS_BALANCE_CONTRACT = "navigation_mass_capacity"
REFERENCE_BALANCE_CONTRACT = "matched_l0_informed_reference"
MASS_SOURCE = "production_upper_navigation_top10_hit_frequency_regularized_v2"
MASS_TRANSFORM = "raw_count_with_unit_l1_prior"
MASS_MODE = "raw-regularized-v2"
MASS_ESTIMATOR_VERSION = 2
MASS_METHOD = "mass-balanced-kmeans"
PROTOCOL_AMENDMENT_ID = "post-exploratory-raw-regularized-v2-confirmation"
CROSS_DATASET_STAGE = "raw_regularized_v2_cross_dataset_offline_confirmation"
CROSS_DATASET_STATUS = "materialization_eligible_online_qps_pending"
ATTACHMENT_TOP_K = 10
ATTACHMENT_SEARCH_EF = 100
ASSIGNMENT_FORMAT = "orion_numeric_import.assignments.jsonl-v1"
MULTI_ASSIGNMENT_CONTRACT = {
    "enabled": True,
    "min_max_vote": 2,
    "vote_delta": 0,
    "max_shards": 0,
}
REPLAY_GATES = (
    "generation_advanced",
    "immutable_metadata_equal",
    "vector_schema_equal",
    "upper_search_contract_equal",
    "canonical_upper_graph_bytes_equal",
    "ordered_upper_labels_and_vector_bits_equal",
    "ordered_hit_labels_equal",
    "ordered_distance_bits_equal",
    "production_router_replay_complete",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-artifact", required=True)
    parser.add_argument("--attachments", required=True)
    parser.add_argument("--attachments-manifest", required=True)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--owner-sha256", required=True)
    parser.add_argument("--screen-manifest", required=True)
    parser.add_argument(
        "--cross-dataset-confirmation",
        help=(
            "Required for the navigation-mass finalist. This checksum-bound "
            "manifest must prove SIFT and GloVe phase-A/phase-B topology, "
            "identity, and old-raw owner parity before materialization."
        ),
    )
    parser.add_argument("--method", required=True)
    parser.add_argument(
        "--reference-owner",
        action="store_true",
        help=(
            "Materialize a manifest-declared, explicitly ineligible L0-informed "
            "matched reference. The flag cannot reclassify a candidate."
        ),
    )
    parser.add_argument("--vectors", required=True)
    parser.add_argument(
        "--vectors-build-manifest",
        required=True,
        help="Source build-manifest.json that checksum-binds canonical vector rows.",
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset-sha256", required=True)
    parser.add_argument("--generation", type=int, required=True)
    parser.add_argument("--rebind-binary", required=True)
    parser.add_argument("--replay-verifier", required=True)
    parser.add_argument("--replay-queries", required=True)
    parser.add_argument("--replay-query-row-count", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--chunk-size", type=int, default=100_000)
    return parser.parse_args()


def sha256_path(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_sha256(name: str, value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a SHA-256 string")
    normalized = value.strip().lower()
    if len(normalized) != 64 or any(c not in "0123456789abcdef" for c in normalized):
        raise ValueError(f"{name} must be a lowercase-compatible SHA-256")
    return normalized


def load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {label}: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def resolved_record_path(record: dict[str, Any], key: str, label: str) -> Path:
    value = record.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} is missing path field {key}")
    return Path(value).expanduser().resolve()


def canonical_close(left: Any, right: Any) -> bool:
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(
            canonical_close(left[key], right[key]) for key in left
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            canonical_close(a, b) for a, b in zip(left, right)
        )
    numeric = (int, float)
    if (
        isinstance(left, numeric)
        and not isinstance(left, bool)
        and isinstance(right, numeric)
        and not isinstance(right, bool)
    ):
        return math.isclose(float(left), float(right), rel_tol=1e-12, abs_tol=1e-12)
    return left == right and type(left) is type(right)


def mismatches(
    actual: dict[str, Any], expected: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    return {
        key: {"expected": value, "actual": actual.get(key)}
        for key, value in expected.items()
        if not canonical_close(actual.get(key), value)
    }


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def assignment_bytes(point_id: int, shards: list[int] | np.ndarray) -> bytes:
    return (
        f'{{"id":{point_id},"shards":['
        + ",".join(str(int(shard)) for shard in shards)
        + "]}\n"
    ).encode("ascii")


def compact_membership(
    owner: np.ndarray,
    local_hits: np.ndarray,
    num_partitions: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Replay Orion's unchanged enabled/2/0/0 max-vote rule."""

    hit_owner = owner[local_hits]
    rows = len(hit_owner)
    votes = np.zeros((rows, num_partitions), dtype=np.uint8)
    row_ids = np.arange(rows, dtype=np.int64)
    for column in range(hit_owner.shape[1]):
        np.add.at(votes, (row_ids, hit_owner[:, column]), 1)
    maximum = votes.max(axis=1)
    membership = votes == maximum[:, None]
    fallback = maximum < 2
    membership[fallback] = False
    membership[row_ids[fallback], hit_owner[fallback, 0]] = True
    return membership, membership.sum(axis=1, dtype=np.int16)


def semantic_mass_sha256(
    source: str, transform: str, values: np.ndarray, top_k: int
) -> str:
    digest = hashlib.sha256()
    digest.update(source.encode("ascii"))
    digest.update(b"\0")
    digest.update(transform.encode("ascii"))
    digest.update(struct.pack("<QQ", len(values), top_k))
    digest.update(values.astype("<u8", copy=False).tobytes(order="C"))
    return digest.hexdigest()


def integer_list(
    value: Any, *, length: int, label: str, minimum: int = 0
) -> list[int]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{label} must contain exactly {length} integers")
    if any(
        isinstance(item, bool) or not isinstance(item, int) or item < minimum
        for item in value
    ):
        raise ValueError(f"{label} contains an invalid integer")
    return [int(item) for item in value]


def parse_checksum_file(path: Path) -> dict[str, str]:
    checksums: dict[str, str] = {}
    for line_number, raw in enumerate(path.read_text(encoding="ascii").splitlines(), 1):
        if not raw:
            continue
        parts = raw.split("  ", 1)
        if len(parts) != 2 or Path(parts[1]).name != parts[1]:
            raise ValueError(f"invalid checksums entry at {path}:{line_number}")
        checksums[parts[1]] = normalized_sha256("checksums entry", parts[0])
    return checksums


def validate_vector_source_contract(
    *,
    vectors_path: Path,
    vectors_sha256: str,
    source_build_manifest_path: Path,
    dataset_path: Path,
    dataset_sha256: str,
    artifact: dict[str, Any],
    chunk_rows: int,
) -> dict[str, Any]:
    """Bind canonical import rows to source build evidence and preprocessing."""

    build = load_json_object(source_build_manifest_path, "vector source build manifest")
    checksum_path = source_build_manifest_path.with_name("checksums.sha256")
    if not checksum_path.is_file():
        raise ValueError("vector source build checksums.sha256 is missing")
    checksums = parse_checksum_file(checksum_path)
    if checksums.get(source_build_manifest_path.name) != sha256_path(source_build_manifest_path):
        raise ValueError("vector source build-manifest checksum mismatch")
    if vectors_path.parent != source_build_manifest_path.parent:
        raise ValueError("canonical vectors are not in the source build-manifest directory")
    if checksums.get(vectors_path.name) != vectors_sha256:
        raise ValueError("source checksums do not bind the canonical vector bytes")
    outputs = build.get("outputs")
    declared_files = outputs.get("files") if isinstance(outputs, dict) else None
    vector_record = declared_files.get(vectors_path.name) if isinstance(declared_files, dict) else None
    expected_size = (
        int(artifact["logical_point_count"])
        * int(artifact["vector_schema"]["dimension"])
        * 4
    )
    if not isinstance(vector_record, dict) or mismatches(
        vector_record,
        {"sha256": vectors_sha256, "size_bytes": expected_size},
    ):
        raise ValueError("source build-manifest does not bind canonical vector rows")
    dataset = build.get("dataset")
    if not isinstance(dataset, dict):
        raise ValueError("vector source build-manifest lacks dataset provenance")
    dataset_expected = {
        "path": str(dataset_path),
        "sha256": dataset_sha256,
        "dimension": int(artifact["vector_schema"]["dimension"]),
        "train_rows_used": int(artifact["logical_point_count"]),
    }
    errors = mismatches(dataset, dataset_expected)
    if errors:
        raise ValueError(f"canonical vectors/dataset row-order binding mismatch: {errors}")
    parameters = build.get("parameters")
    if not isinstance(parameters, dict):
        raise ValueError("vector source build-manifest lacks parameters")
    artifact_distance = str(artifact["vector_schema"]["distance"]).lower()
    declared_distance = str(parameters.get("vector_distance") or "").lower()
    if declared_distance == "l2":
        declared_distance = "euclid"
    expected_distance = "euclid" if artifact_distance in {"euclid", "euclidean", "l2"} else artifact_distance
    if declared_distance != expected_distance:
        raise ValueError("canonical vector distance differs from source artifact")

    vectors = np.memmap(
        vectors_path,
        dtype="<f4",
        mode="r",
        shape=(int(artifact["logical_point_count"]), int(artifact["vector_schema"]["dimension"])),
    )
    if expected_distance == "cosine":
        for start in range(0, len(vectors), chunk_rows):
            block = np.asarray(vectors[start : start + chunk_rows], dtype=np.float64)
            if not np.isfinite(block).all():
                raise ValueError("canonical cosine vectors contain a non-finite value")
            norms = np.linalg.norm(block, axis=1)
            nonzero = norms > 1e-12
            if np.any(np.abs(norms[nonzero] - 1.0) > 2e-5):
                raise ValueError("canonical cosine vectors are not row-normalized")
        preprocessing = "cosine_l2_normalize_nonzero_rows_float32"
    else:
        preprocessing = "identity_float32_row_order"
    return {
        "build_manifest": str(source_build_manifest_path),
        "build_manifest_sha256": sha256_path(source_build_manifest_path),
        "checksums": str(checksum_path),
        "checksums_sha256": sha256_path(checksum_path),
        "dataset_sha256": dataset_sha256,
        "row_count": int(artifact["logical_point_count"]),
        "dimension": int(artifact["vector_schema"]["dimension"]),
        "distance": expected_distance,
        "preprocessing_contract": preprocessing,
        "vectors_sha256": vectors_sha256,
    }


def validate_attachment_contract(
    *,
    artifact: dict[str, Any],
    source_artifact_path: Path,
    source_sha256: str,
    attachments_path: Path,
    attachments_manifest_path: Path,
    vectors_path: Path,
    vectors_sha256: str,
) -> tuple[dict[str, Any], np.memmap]:
    rows = int(artifact["logical_point_count"])
    dimension = int(artifact["vector_schema"]["dimension"])
    manifest = load_json_object(attachments_manifest_path, "attachment export manifest")
    expected = {
        "format_version": 1,
        "artifact_path": str(source_artifact_path),
        "artifact_sha256": source_sha256,
        "generation": int(artifact["generation"]),
        "upper_graph_present": True,
        "source_upper_k": int(artifact["upper_k"]),
        "source_upper_ef_search": int(artifact["upper_ef_search"]),
        "vectors_path": str(vectors_path),
        "vectors_sha256": vectors_sha256,
        "row_count": rows,
        "dimension": dimension,
        "top_k": ATTACHMENT_TOP_K,
        "search_ef": ATTACHMENT_SEARCH_EF,
        "hits_path": str(attachments_path),
        "hits_sha256": sha256_path(attachments_path),
        "hits_size_bytes": rows * ATTACHMENT_TOP_K * 8,
    }
    errors = mismatches(manifest, expected)
    if errors:
        raise ValueError(f"attachment exporter contract mismatch: {errors}")
    if vectors_path.stat().st_size != rows * dimension * 4:
        raise ValueError("hardlink vector length differs from artifact logical row order")
    if attachments_path.stat().st_size != rows * ATTACHMENT_TOP_K * 8:
        raise ValueError("attachment file length mismatch")
    return manifest, np.memmap(
        attachments_path,
        dtype="<u8",
        mode="r",
        shape=(rows, ATTACHMENT_TOP_K),
    )


def validate_topology_record(
    candidate: dict[str, Any],
    contract: dict[str, Any],
    search_contract: dict[str, Any],
) -> None:
    if contract.get("all_gates_required") is not True:
        raise ValueError("screen topology contract does not require every gate")
    if contract.get("preregistered_before_mass_candidate_results") is not True:
        raise ValueError("screen topology gates were not preregistered")
    metrics = candidate.get("topology_metrics")
    actual_metrics = candidate.get("actual_metrics")
    gates = candidate.get("topology_gates")
    if (
        not isinstance(metrics, dict)
        or not isinstance(actual_metrics, dict)
        or not isinstance(gates, dict)
        or not gates
    ):
        raise ValueError("candidate lacks topology metrics or gates")
    if search_contract.get("all_gates_required") is not True or search_contract.get(
        "inputs_complete"
    ) is not True:
        raise ValueError("search-induced topology inputs/gates are incomplete")
    if candidate.get("topology_gate_inputs_complete") is not True:
        raise ValueError("candidate topology gate inputs are incomplete")
    combined_metrics = {**metrics, **actual_metrics}
    results = []
    for metric, gate in gates.items():
        if metric not in combined_metrics or not isinstance(gate, dict):
            raise ValueError(f"candidate topology gate {metric} is malformed")
        observed = combined_metrics[metric]
        threshold = gate.get("threshold")
        if (
            isinstance(observed, bool)
            or not isinstance(observed, (int, float))
            or isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
        ):
            raise ValueError(f"candidate topology gate {metric} is not numeric")
        if not canonical_close(observed, gate.get("observed")):
            raise ValueError(f"candidate topology gate {metric} observed value drifted")
        if gate.get("operator") == "<=":
            passed = float(observed) <= float(threshold)
        elif gate.get("operator") == ">=":
            passed = float(observed) >= float(threshold)
        else:
            raise ValueError(f"candidate topology gate {metric} operator is invalid")
        if type(gate.get("pass")) is not bool or gate["pass"] is not passed:
            raise ValueError(f"candidate topology gate {metric} pass bit is inconsistent")
        results.append(passed)
    if type(candidate.get("topology_all_pass")) is not bool:
        raise ValueError("candidate topology_all_pass must be a boolean")
    if candidate["topology_all_pass"] is not all(results):
        raise ValueError("candidate topology_all_pass is inconsistent")
    if not all(results):
        raise ValueError("candidate failed at least one topology gate")
    if candidate.get("graph_topology_all_pass") is not True:
        raise ValueError("candidate graph-topology gates did not all pass")
    if candidate.get("identity_all_pass") is not True:
        raise ValueError("candidate identity gates did not all pass")
    identity_gates = candidate.get("identity_gates")
    if not isinstance(identity_gates, dict) or not identity_gates:
        raise ValueError("candidate identity gates are missing")
    for name, gate in identity_gates.items():
        passed = gate if type(gate) is bool else gate.get("pass") if isinstance(gate, dict) else None
        if passed is not True:
            raise ValueError(f"candidate identity gate {name} did not pass")


def select_screen_record(
    screen: dict[str, Any], method: str, owner_path: Path, owner_sha256: str
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for key in ("candidates", "references"):
        value = screen.get(key, [])
        if not isinstance(value, list):
            raise ValueError(f"screen manifest {key} must be a list")
        records.extend(record for record in value if isinstance(record, dict))
    matches = []
    for record in records:
        if record.get("method") != method or record.get("owner_sha256") != owner_sha256:
            continue
        if resolved_record_path(record, "owner_path", "screen record") == owner_path:
            matches.append(record)
    if len(matches) != 1:
        raise ValueError(
            "screen must contain exactly one record matching method, owner path, and SHA; "
            f"found {len(matches)}"
        )
    return matches[0]


def validate_frozen_chain(
    screen: dict[str, Any], screen_path: Path, candidate: dict[str, Any]
) -> str:
    frozen_path = resolved_record_path(screen, "frozen_candidate_manifest", "screen")
    frozen_sha = normalized_sha256(
        "frozen-candidate-manifest-sha256",
        screen.get("frozen_candidate_manifest_sha256"),
    )
    if sha256_path(frozen_path) != frozen_sha:
        raise ValueError("frozen candidate manifest checksum mismatch")
    sidecar = frozen_path.with_name(frozen_path.name + ".sha256")
    if not sidecar.is_file() or sidecar.read_text(encoding="ascii").strip() != frozen_sha:
        raise ValueError("frozen candidate manifest checksum sidecar mismatch")
    frozen = load_json_object(frozen_path, "frozen candidate manifest")
    if frozen.get("format_version") != 1 or frozen.get("stage") != "upper_only_candidates_frozen":
        raise ValueError("screen does not bind an upper-only frozen candidate manifest")
    for key in ("mass_estimator", "topology_gate_contract", "construction_inputs"):
        if not canonical_close(screen.get(key), frozen.get(key)):
            raise ValueError(f"screen changed frozen {key}")
    frozen_records = frozen.get("candidates")
    if not isinstance(frozen_records, list):
        raise ValueError("frozen candidate manifest lacks candidates")
    matches = [
        record
        for record in frozen_records
        if isinstance(record, dict)
        and record.get("method") == candidate.get("method")
        and record.get("owner_path") == candidate.get("owner_path")
        and record.get("owner_sha256") == candidate.get("owner_sha256")
    ]
    if len(matches) != 1:
        raise ValueError("final candidate is not uniquely bound to its frozen record")
    post_freeze_fields = {
        "eligibility_pending_post_freeze_query_gates",
        "materialization_eligible",
        "topology_all_pass",
    }
    changed = {
        key: {"frozen": value, "screen": candidate.get(key)}
        for key, value in matches[0].items()
        if key not in post_freeze_fields
        and (key not in candidate or not canonical_close(candidate[key], value))
    }
    if changed:
        raise ValueError(f"final screen changed frozen candidate fields: {changed}")
    if matches[0].get("eligibility_pending_post_freeze_query_gates") is not True:
        raise ValueError("frozen candidate was not pending post-freeze query gates")
    if candidate.get("eligibility_pending_post_freeze_query_gates") is not False:
        raise ValueError("final candidate still reports pending post-freeze query gates")
    if screen_path == frozen_path:
        raise ValueError("prepare manifest cannot masquerade as final screen")
    return frozen_sha


def validate_v2_protocol_evidence(
    *,
    screen: dict[str, Any],
    candidate: dict[str, Any],
    mass_manifest: dict[str, Any],
    owner_path: Path,
    owner_sha256: str,
) -> dict[str, Any]:
    amendment = candidate.get("protocol_amendment")
    if not isinstance(amendment, dict):
        amendment = screen.get("protocol_amendment")
    if not isinstance(amendment, dict):
        raise ValueError("v2 candidate lacks its protocol amendment")
    if not canonical_close(mass_manifest.get("protocol_amendment"), amendment):
        raise ValueError("v2 protocol amendment differs between mass and final screen")
    expected = {
        "format_version": 1,
        "id": PROTOCOL_AMENDMENT_ID,
        "status": "post_exploratory_promotion_not_preregistered",
        "mass_mode": MASS_MODE,
        "source": MASS_SOURCE,
        "transform": MASS_TRANSFORM,
        "estimator_version": MASS_ESTIMATOR_VERSION,
        "method": MASS_METHOD,
        "topology_gates_unchanged": True,
        "old_raw_owner_byte_parity_required": True,
        "online_qps_only_confirmation": True,
    }
    errors = mismatches(amendment, expected)
    if errors:
        raise ValueError(f"v2 protocol amendment mismatch: {errors}")
    amendment_path = resolved_record_path(amendment, "amendment_path", "protocol amendment")
    protocol_path = resolved_record_path(amendment, "protocol_path", "protocol amendment")
    amendment_sha = normalized_sha256(
        "protocol amendment SHA", amendment.get("amendment_sha256")
    )
    protocol_sha = normalized_sha256("protocol SHA", amendment.get("protocol_sha256"))
    if sha256_path(amendment_path) != amendment_sha:
        raise ValueError("v2 protocol amendment bytes drifted")
    if sha256_path(protocol_path) != protocol_sha:
        raise ValueError("v2 protocol document bytes drifted")
    amendment_file = load_json_object(amendment_path, "v2 protocol amendment file")
    errors = mismatches(amendment_file, expected)
    if errors:
        raise ValueError(f"v2 amendment file mismatch: {errors}")

    parity = candidate.get("owner_parity")
    if not isinstance(parity, dict):
        raise ValueError("v2 candidate lacks old-raw owner parity")
    errors = mismatches(
        parity,
        {
            "reference_mass_mode": "raw",
            "reference_owner_sha256": owner_sha256,
            "owner_bytes_identical": True,
        },
    )
    if errors:
        raise ValueError(f"v2 owner parity mismatch: {errors}")
    reference_manifest_path = resolved_record_path(
        parity, "reference_candidate_manifest_path", "owner parity"
    )
    reference_manifest_sha = normalized_sha256(
        "raw reference candidate manifest SHA",
        parity.get("reference_candidate_manifest_sha256"),
    )
    if sha256_path(reference_manifest_path) != reference_manifest_sha:
        raise ValueError("raw parity-reference candidate manifest drifted")
    reference_sidecar = reference_manifest_path.with_name(
        reference_manifest_path.name + ".sha256"
    )
    if (
        not reference_sidecar.is_file()
        or reference_sidecar.read_text(encoding="ascii").strip()
        != reference_manifest_sha
    ):
        raise ValueError("raw parity-reference manifest sidecar mismatch")
    reference = load_json_object(reference_manifest_path, "raw parity-reference manifest")
    if (
        reference.get("stage") != "upper_only_candidates_frozen"
        or (reference.get("parameters") or {}).get("mass_mode") != "raw"
    ):
        raise ValueError("owner parity reference is not the frozen raw ablation")
    matches = [
        record
        for record in reference.get("candidates", [])
        if isinstance(record, dict) and record.get("method") == MASS_METHOD
    ]
    if len(matches) != 1:
        raise ValueError("raw parity reference lacks one mass-balanced-kmeans record")
    raw_record = matches[0]
    raw_owner_path = resolved_record_path(raw_record, "owner_path", "raw parity record")
    if parity.get("reference_owner_path") not in {None, str(raw_owner_path)}:
        raise ValueError("owner parity reference path differs from raw manifest")
    if raw_record.get("owner_sha256") != owner_sha256:
        raise ValueError("v2 owner is not byte-identical to frozen raw owner")
    if sha256_path(raw_owner_path) != owner_sha256 or sha256_path(owner_path) != owner_sha256:
        raise ValueError("v2/raw owner bytes drifted after parity freeze")
    return {
        "protocol_amendment": str(amendment_path),
        "protocol_amendment_sha256": amendment_sha,
        "protocol": str(protocol_path),
        "protocol_sha256": protocol_sha,
        "owner_parity": parity,
    }


def validate_checksum_sidecar(path: Path, expected_sha256: str, label: str) -> None:
    sidecar = path.with_name(path.name + ".sha256")
    if not sidecar.is_file():
        raise ValueError(f"{label} checksum sidecar is missing")
    if sidecar.read_text(encoding="ascii").strip() != expected_sha256:
        raise ValueError(f"{label} checksum sidecar mismatch")


def validate_cross_dataset_contract(
    confirmation: dict[str, Any], amendment: dict[str, Any]
) -> None:
    errors = mismatches(
        confirmation,
        {
            "format_version": 1,
            "stage": CROSS_DATASET_STAGE,
            "status": CROSS_DATASET_STATUS,
            "materialization_eligible": True,
            "cross_dataset_identity_all_pass": True,
            "cross_dataset_topology_all_pass": True,
            "owner_parity_all_pass": True,
            "online_qps_confirmation_required": True,
            "method_contract": {
                "estimator_version": MASS_ESTIMATOR_VERSION,
                "l0_evaluation_only_after_owner_freeze": True,
                "mass_mode": MASS_MODE,
                "method": MASS_METHOD,
                "multi_assignment_unchanged": True,
                "source": MASS_SOURCE,
                "transform": MASS_TRANSFORM,
                "upper_only_phase_a": True,
            },
            "confirmation_contract": {
                "alternate_method_or_parameter_scan": False,
                "old_raw_owner_byte_parity_required": True,
                "online_cluster_entered_by_this_stage": False,
                "online_qps_confirmation_completed": False,
                "online_qps_confirmation_required": True,
                "online_qps_only_confirmation": True,
                "post_exploratory_not_preregistered": True,
                "topology_thresholds_unchanged": True,
                "v1_heldout_failure_retained": True,
            },
        },
    )
    if errors:
        raise ValueError(f"cross-dataset confirmation contract mismatch: {errors}")
    if not canonical_close(confirmation.get("protocol_amendment"), amendment):
        raise ValueError("cross-dataset protocol amendment differs from finalist")


def validate_cross_dataset_confirmation(
    *,
    confirmation_path: Path,
    screen_path: Path,
    screen: dict[str, Any],
    candidate: dict[str, Any],
    source_artifact_path: Path,
    source_sha256: str,
    owner_path: Path,
    owner_sha256: str,
    frozen_candidate_manifest_sha256: str,
) -> dict[str, str]:
    """Replay the post-exploratory two-dataset promotion proof.

    The current finalist is not sufficient by itself: the promotion is valid
    only if the independently frozen SIFT and GloVe phase-A/phase-B chains both
    remain byte-bound, pass every topology/identity gate, and retain exact
    parity with their pre-existing raw-count owners.
    """

    confirmation_sha = sha256_path(confirmation_path)
    validate_checksum_sidecar(
        confirmation_path, confirmation_sha, "cross-dataset confirmation"
    )
    confirmation = load_json_object(
        confirmation_path, "cross-dataset confirmation manifest"
    )
    amendment = candidate.get("protocol_amendment")
    if not isinstance(amendment, dict):
        raise ValueError("current finalist lacks the v2 protocol amendment")
    validate_cross_dataset_contract(confirmation, amendment)

    builder = confirmation.get("builder")
    if not isinstance(builder, dict):
        raise ValueError("cross-dataset confirmation lacks builder provenance")
    builder_path = resolved_record_path(builder, "path", "confirmation builder")
    builder_sha = normalized_sha256(
        "confirmation builder SHA", builder.get("sha256")
    )
    if not builder_path.is_file() or sha256_path(builder_path) != builder_sha:
        raise ValueError("cross-dataset confirmation builder checksum drifted")

    datasets = confirmation.get("datasets")
    if not isinstance(datasets, dict) or set(datasets) != {"sift", "glove"}:
        raise ValueError("cross-dataset confirmation must contain exactly SIFT and GloVe")
    current_matches = 0
    for dataset_name in ("sift", "glove"):
        record = datasets.get(dataset_name)
        if not isinstance(record, dict):
            raise ValueError(f"cross-dataset record {dataset_name} is malformed")
        errors = mismatches(
            record,
            {
                "dataset": dataset_name,
                "logical_shards": 32,
                "materialization_eligible": True,
                "graph_topology_all_pass": True,
                "identity_all_pass": True,
                "owner_bytes_identical": True,
                "owner_parity_all_pass": True,
                "topology_all_pass": True,
            },
        )
        if errors:
            raise ValueError(
                f"cross-dataset record {dataset_name} gate mismatch: {errors}"
            )

        phase_a_path = resolved_record_path(
            record, "phase_a_candidate_manifest_path", f"{dataset_name} record"
        )
        phase_a_sha = normalized_sha256(
            f"{dataset_name} phase-A SHA",
            record.get("phase_a_candidate_manifest_sha256"),
        )
        phase_b_path = resolved_record_path(
            record, "phase_b_screen_manifest_path", f"{dataset_name} record"
        )
        phase_b_sha = normalized_sha256(
            f"{dataset_name} phase-B SHA",
            record.get("phase_b_screen_manifest_sha256"),
        )
        raw_manifest_path = resolved_record_path(
            record,
            "raw_reference_candidate_manifest_path",
            f"{dataset_name} record",
        )
        raw_manifest_sha = normalized_sha256(
            f"{dataset_name} raw reference manifest SHA",
            record.get("raw_reference_candidate_manifest_sha256"),
        )
        declared_owner_path = resolved_record_path(
            record, "owner_path", f"{dataset_name} record"
        )
        declared_owner_sha = normalized_sha256(
            f"{dataset_name} owner SHA", record.get("owner_sha256")
        )
        raw_owner_path = resolved_record_path(
            record, "raw_reference_owner_path", f"{dataset_name} record"
        )
        raw_owner_sha = normalized_sha256(
            f"{dataset_name} raw owner SHA",
            record.get("raw_reference_owner_sha256"),
        )
        declared_source_path = resolved_record_path(
            record, "source_artifact_path", f"{dataset_name} record"
        )
        declared_source_sha = normalized_sha256(
            f"{dataset_name} source artifact SHA",
            record.get("source_artifact_sha256"),
        )
        for evidence_path, evidence_sha, label, sidecar_required in (
            (phase_a_path, phase_a_sha, f"{dataset_name} phase-A", True),
            (phase_b_path, phase_b_sha, f"{dataset_name} phase-B", True),
            (
                raw_manifest_path,
                raw_manifest_sha,
                f"{dataset_name} raw reference manifest",
                True,
            ),
            (declared_owner_path, declared_owner_sha, f"{dataset_name} owner", False),
            (raw_owner_path, raw_owner_sha, f"{dataset_name} raw owner", False),
            (
                declared_source_path,
                declared_source_sha,
                f"{dataset_name} source artifact",
                False,
            ),
        ):
            if not evidence_path.is_file() or sha256_path(evidence_path) != evidence_sha:
                raise ValueError(f"{label} checksum drifted")
            if sidecar_required:
                validate_checksum_sidecar(evidence_path, evidence_sha, label)
        if declared_owner_sha != raw_owner_sha:
            raise ValueError(f"{dataset_name} v2/raw owner SHA differs")

        phase_a = load_json_object(phase_a_path, f"{dataset_name} phase-A manifest")
        if (
            phase_a.get("format_version") != 1
            or phase_a.get("stage") != "upper_only_candidates_frozen"
            or not canonical_close(phase_a.get("protocol_amendment"), amendment)
        ):
            raise ValueError(f"{dataset_name} phase-A contract drifted")
        phase_a_record = select_screen_record(
            phase_a, MASS_METHOD, declared_owner_path, declared_owner_sha
        )
        errors = mismatches(
            phase_a_record,
            {
                "eligibility_pending_post_freeze_query_gates": True,
                "materialization_eligible": False,
                "topology_all_pass": False,
                "owner_parity": record.get("owner_parity")
                if "owner_parity" in record
                else phase_a_record.get("owner_parity"),
            },
        )
        if errors:
            raise ValueError(f"{dataset_name} phase-A finalist mismatch: {errors}")

        phase_b = load_json_object(phase_b_path, f"{dataset_name} phase-B manifest")
        if (
            phase_b.get("format_version") != 1
            or phase_b.get("stage") != "post_freeze_evaluation"
            or phase_b.get("identity_all_pass") is not True
            or phase_b.get("owner_parity_all_pass") is not True
            or not canonical_close(phase_b.get("protocol_amendment"), amendment)
        ):
            raise ValueError(f"{dataset_name} phase-B contract drifted")
        if (
            resolved_record_path(phase_b, "frozen_candidate_manifest", "phase-B")
            != phase_a_path
            or phase_b.get("frozen_candidate_manifest_sha256") != phase_a_sha
        ):
            raise ValueError(f"{dataset_name} phase-B does not bind phase-A")
        phase_b_record = select_screen_record(
            phase_b, MASS_METHOD, declared_owner_path, declared_owner_sha
        )
        if (
            phase_b_record.get("materialization_eligible") is not True
            or phase_b_record.get("eligibility_pending_post_freeze_query_gates") is not False
            or phase_b_record.get("owner_parity_all_pass") is not True
        ):
            raise ValueError(f"{dataset_name} phase-B finalist is not eligible")
        topology_contract = phase_b.get("topology_gate_contract")
        search_contract = phase_b.get("search_induced_gate_contract")
        if not isinstance(topology_contract, dict) or not isinstance(search_contract, dict):
            raise ValueError(f"{dataset_name} phase-B lacks topology contracts")
        validate_topology_record(phase_b_record, topology_contract, search_contract)
        logical_rows = record.get("logical_point_count")
        if isinstance(logical_rows, bool) or not isinstance(logical_rows, int) or logical_rows <= 0:
            raise ValueError(f"{dataset_name} logical_point_count is invalid")
        validate_parity_evidence(
            phase_b_record.get("materialization_parity"), rows=logical_rows, partitions=32
        )
        if not canonical_close(
            record.get("materialization_parity"),
            phase_b_record.get("materialization_parity"),
        ):
            raise ValueError(f"{dataset_name} materialization parity drifted")
        if not canonical_close(record.get("topology_gates"), phase_b_record.get("topology_gates")):
            raise ValueError(f"{dataset_name} topology gates drifted")

        source_artifact = load_json_object(
            declared_source_path, f"{dataset_name} source artifact"
        )
        errors = mismatches(
            source_artifact,
            {"logical_point_count": logical_rows, "shard_count": 32},
        )
        if errors:
            raise ValueError(f"{dataset_name} source artifact mismatch: {errors}")
        raw_manifest = load_json_object(
            raw_manifest_path, f"{dataset_name} raw reference manifest"
        )
        if (
            raw_manifest.get("stage") != "upper_only_candidates_frozen"
            or (raw_manifest.get("parameters") or {}).get("mass_mode") != "raw"
        ):
            raise ValueError(f"{dataset_name} raw owner reference is invalid")
        raw_records = [
            item
            for item in raw_manifest.get("candidates", [])
            if isinstance(item, dict) and item.get("method") == MASS_METHOD
        ]
        if len(raw_records) != 1:
            raise ValueError(f"{dataset_name} raw reference lacks one finalist")
        if (
            resolved_record_path(raw_records[0], "owner_path", "raw finalist")
            != raw_owner_path
            or raw_records[0].get("owner_sha256") != raw_owner_sha
        ):
            raise ValueError(f"{dataset_name} raw owner reference drifted")

        materializer_inputs = record.get("materializer_inputs")
        if not isinstance(materializer_inputs, dict):
            raise ValueError(f"{dataset_name} materializer inputs are missing")
        errors = mismatches(
            materializer_inputs,
            {
                "method": MASS_METHOD,
                "owner": str(declared_owner_path),
                "owner_sha256": declared_owner_sha,
                "screen_manifest": str(phase_b_path),
                "screen_manifest_sha256": phase_b_sha,
                "source_artifact": str(declared_source_path),
                "source_artifact_sha256": declared_source_sha,
            },
        )
        if errors:
            raise ValueError(f"{dataset_name} materializer binding mismatch: {errors}")
        attachment_path = resolved_record_path(
            materializer_inputs, "attachments", f"{dataset_name} materializer inputs"
        )
        attachment_sha = normalized_sha256(
            f"{dataset_name} attachments SHA",
            materializer_inputs.get("attachments_sha256"),
        )
        attachment_manifest_path = resolved_record_path(
            materializer_inputs,
            "attachments_manifest",
            f"{dataset_name} materializer inputs",
        )
        attachment_manifest_sha = normalized_sha256(
            f"{dataset_name} attachments manifest SHA",
            materializer_inputs.get("attachments_manifest_sha256"),
        )
        if (
            not attachment_path.is_file()
            or sha256_path(attachment_path) != attachment_sha
            or not attachment_manifest_path.is_file()
            or sha256_path(attachment_manifest_path) != attachment_manifest_sha
        ):
            raise ValueError(f"{dataset_name} attachment evidence drifted")
        attachment_manifest = load_json_object(
            attachment_manifest_path, f"{dataset_name} attachment manifest"
        )
        errors = mismatches(
            attachment_manifest,
            {
                "artifact_sha256": declared_source_sha,
                "row_count": logical_rows,
                "top_k": ATTACHMENT_TOP_K,
                "search_ef": ATTACHMENT_SEARCH_EF,
                "hits_path": str(attachment_path),
                "hits_sha256": attachment_sha,
            },
        )
        if errors:
            raise ValueError(f"{dataset_name} attachment manifest mismatch: {errors}")

        actual = phase_b_record.get("actual_metrics")
        topology = phase_b_record.get("topology_metrics")
        if not isinstance(actual, dict) or not isinstance(topology, dict):
            raise ValueError(f"{dataset_name} finalist metrics are missing")
        errors = mismatches(
            record.get("metrics") if isinstance(record.get("metrics"), dict) else {},
            {
                "expansion_ratio": actual.get("expansion_ratio"),
                "gt_routing_coverage_mean": actual.get("gt_routing_coverage_mean"),
                "l0_load_cv": actual.get("load_cv"),
                "l0_load_max_over_mean": actual.get("load_max_over_mean"),
                "routed_shards_mean": actual.get("routed_shards_mean"),
                "upper_edge_cut_ratio": topology.get("upper_edge_cut_ratio"),
            },
        )
        if errors:
            raise ValueError(f"{dataset_name} confirmation metrics drifted: {errors}")

        if (
            phase_b_path == screen_path
            and declared_source_path == source_artifact_path
            and declared_source_sha == source_sha256
            and declared_owner_path == owner_path
            and declared_owner_sha == owner_sha256
        ):
            current_matches += 1
            if phase_b_sha != sha256_path(screen_path):
                raise ValueError("current screen SHA differs from cross-dataset record")
            if phase_a_sha != frozen_candidate_manifest_sha256:
                raise ValueError("current frozen phase-A SHA differs from cross proof")
            if not canonical_close(phase_b_record, candidate):
                raise ValueError("current finalist record differs from cross proof")

    if current_matches != 1:
        raise ValueError(
            "cross-dataset confirmation must bind the current materialization exactly once"
        )

    v1_failure = confirmation.get("v1_heldout_failure")
    if not isinstance(v1_failure, dict):
        raise ValueError("cross-dataset confirmation dropped the v1 held-out failure")
    errors = mismatches(
        v1_failure,
        {
            "dataset": "glove",
            "mass_mode": "self-debiased-floor1",
            "method": MASS_METHOD,
            "topology_all_pass": False,
            "materialization_eligible": False,
        },
    )
    if errors:
        raise ValueError(f"v1 held-out failure contract mismatch: {errors}")
    v1_screen_path = resolved_record_path(
        v1_failure, "screen_manifest_path", "v1 held-out failure"
    )
    v1_screen_sha = normalized_sha256(
        "v1 held-out screen SHA", v1_failure.get("screen_manifest_sha256")
    )
    if not v1_screen_path.is_file() or sha256_path(v1_screen_path) != v1_screen_sha:
        raise ValueError("v1 held-out failure screen drifted")
    failed_gates = v1_failure.get("failed_gates")
    if not isinstance(failed_gates, dict) or set(failed_gates) != {"routed_shards_mean"}:
        raise ValueError("v1 held-out failure must retain the routed-shards gate")
    routed_gate = failed_gates["routed_shards_mean"]
    if (
        not isinstance(routed_gate, dict)
        or routed_gate.get("operator") != "<="
        or routed_gate.get("pass") is not False
        or not isinstance(routed_gate.get("observed"), (int, float))
        or not isinstance(routed_gate.get("threshold"), (int, float))
        or float(routed_gate["observed"]) <= float(routed_gate["threshold"])
    ):
        raise ValueError("v1 routed-shards failure proof is inconsistent")

    return {
        "cross_dataset_confirmation": str(confirmation_path),
        "cross_dataset_confirmation_sha256": confirmation_sha,
    }


def validate_mass_candidate_allowlist(
    candidate: dict[str, Any],
    mass_info: dict[str, Any],
    construction: dict[str, Any],
) -> None:
    errors = mismatches(
        candidate,
        {
            "method": MASS_METHOD,
            "balance_contract": MASS_BALANCE_CONTRACT,
            "source_artifact_sha256": construction.get("artifact_sha256"),
            "partitioner_source_sha256": construction.get("partitioner_source_sha256"),
            "navigation_mass_manifest_sha256": mass_info.get("manifest_sha256"),
            "navigation_mass_source": MASS_SOURCE,
            "navigation_mass_transform": MASS_TRANSFORM,
            "navigation_mass_estimator_version": MASS_ESTIMATOR_VERSION,
            "navigation_mass_sha256": mass_info.get("sha256"),
            "mass_mode": MASS_MODE,
            "materialization_eligible": True,
        },
    )
    if errors:
        raise ValueError(f"candidate navigation-mass binding mismatch: {errors}")
    errors = mismatches(
        mass_info,
        {
            "format_version": 1,
            "mass_mode": MASS_MODE,
            "source": MASS_SOURCE,
            "transform": MASS_TRANSFORM,
            "estimator_version": MASS_ESTIMATOR_VERSION,
            "eligible_as_finalist": True,
            "rank_weighting": False,
        },
    )
    if errors:
        raise ValueError(f"navigation-mass estimator mismatch: {errors}")


def validate_mass_candidate(
    *,
    screen: dict[str, Any],
    candidate: dict[str, Any],
    owner: np.ndarray,
    owner_path: Path,
    owner_sizes: np.ndarray,
    artifact: dict[str, Any],
) -> dict[str, Any]:
    construction = screen.get("construction_inputs")
    mass_info = screen.get("mass_estimator")
    if not isinstance(construction, dict) or not isinstance(mass_info, dict):
        raise ValueError("navigation-mass screen lacks construction evidence")
    upper_count = len(owner)
    partitions = int(artifact["shard_count"])
    validate_mass_candidate_allowlist(candidate, mass_info, construction)
    errors = mismatches(
        mass_info,
        {
            "query_count": upper_count,
            "top_k": ATTACHMENT_TOP_K,
        },
    )
    if errors:
        raise ValueError(f"navigation-mass estimator mismatch: {errors}")

    partitioner_path = resolved_record_path(construction, "partitioner_source", "construction")
    partitioner_sha = normalized_sha256(
        "partitioner-source-sha256", construction.get("partitioner_source_sha256")
    )
    if not partitioner_path.is_file() or sha256_path(partitioner_path) != partitioner_sha:
        raise ValueError("partitioner source checksum drifted after freeze")
    for path_key, sha_key, label in (
        ("upper_input_manifest", "upper_input_manifest_sha256", "upper-input manifest"),
        ("upper_navigation_hits", "upper_navigation_hits_sha256", "upper-navigation hits"),
        (
            "upper_navigation_manifest",
            "upper_navigation_manifest_sha256",
            "upper-navigation manifest",
        ),
    ):
        evidence_path = resolved_record_path(construction, path_key, "construction")
        expected_sha = normalized_sha256(label, construction.get(sha_key))
        if not evidence_path.is_file() or sha256_path(evidence_path) != expected_sha:
            raise ValueError(f"{label} checksum drifted")

    upper_input_path = resolved_record_path(construction, "upper_input_manifest", "construction")
    upper_input = load_json_object(upper_input_path, "upper-input manifest")
    errors = mismatches(
        upper_input,
        {
            "artifact_sha256": construction.get("artifact_sha256"),
            "row_count": upper_count,
            "dimension": int(artifact["vector_schema"]["dimension"]),
            "labels_sha256": construction.get("ordered_labels_sha256"),
            "vectors_sha256": construction.get("ordered_vectors_sha256"),
        },
    )
    if errors:
        raise ValueError(f"ordered upper-input mismatch: {errors}")
    for path_key, sha_key in (("labels", "labels_sha256"), ("vectors", "vectors_sha256")):
        ordered_path = resolved_record_path(upper_input, path_key, "upper-input manifest")
        if sha256_path(ordered_path) != upper_input[sha_key]:
            raise ValueError(f"ordered upper {path_key} checksum drifted")

    nav_hits = resolved_record_path(construction, "upper_navigation_hits", "construction")
    nav_manifest_path = resolved_record_path(
        construction, "upper_navigation_manifest", "construction"
    )
    nav_manifest = load_json_object(nav_manifest_path, "upper-navigation manifest")
    errors = mismatches(
        nav_manifest,
        {
            "format_version": 1,
            "artifact_sha256": construction.get("artifact_sha256"),
            "upper_graph_present": True,
            "vectors_sha256": construction.get("ordered_vectors_sha256"),
            "row_count": upper_count,
            "dimension": int(artifact["vector_schema"]["dimension"]),
            "top_k": ATTACHMENT_TOP_K,
            "search_ef": ATTACHMENT_SEARCH_EF,
            "hits_path": str(nav_hits),
            "hits_sha256": construction.get("upper_navigation_hits_sha256"),
            "hits_size_bytes": upper_count * ATTACHMENT_TOP_K * 8,
        },
    )
    if errors:
        raise ValueError(f"upper-navigation exporter mismatch: {errors}")

    mass_manifest_path = resolved_record_path(
        candidate, "navigation_mass_manifest_path", "candidate"
    )
    if mass_manifest_path != resolved_record_path(mass_info, "manifest_path", "mass estimator"):
        raise ValueError("candidate mass manifest path differs from screen")
    mass_manifest_sha = normalized_sha256(
        "navigation-mass-manifest-sha256",
        candidate.get("navigation_mass_manifest_sha256"),
    )
    if sha256_path(mass_manifest_path) != mass_manifest_sha:
        raise ValueError("navigation-mass manifest checksum drifted")
    mass_manifest = load_json_object(mass_manifest_path, "navigation-mass manifest")
    errors = mismatches(
        mass_manifest,
        {
            "format_version": 1,
            "mass_mode": MASS_MODE,
            "source": MASS_SOURCE,
            "transform": MASS_TRANSFORM,
            "estimator_version": MASS_ESTIMATOR_VERSION,
            "semantic_sha256": mass_info.get("sha256"),
            "source_artifact_sha256": construction.get("artifact_sha256"),
            "navigator_sha256": construction.get("navigator_sha256"),
            "ordered_labels_sha256": construction.get("ordered_labels_sha256"),
            "ordered_vectors_sha256": construction.get("ordered_vectors_sha256"),
            "upper_navigation_hits": str(nav_hits),
            "upper_navigation_hits_sha256": construction.get("upper_navigation_hits_sha256"),
            "upper_navigation_manifest": str(nav_manifest_path),
            "upper_navigation_manifest_sha256": construction.get(
                "upper_navigation_manifest_sha256"
            ),
            "upper_node_count": upper_count,
            "dimension": int(artifact["vector_schema"]["dimension"]),
            "top_k": ATTACHMENT_TOP_K,
            "search_ef": ATTACHMENT_SEARCH_EF,
            "partitioner_source": str(partitioner_path),
            "partitioner_source_sha256": partitioner_sha,
        },
    )
    if errors:
        raise ValueError(f"navigation-mass manifest mismatch: {errors}")
    v2_evidence = validate_v2_protocol_evidence(
        screen=screen,
        candidate=candidate,
        mass_manifest=mass_manifest,
        owner_path=owner_path,
        owner_sha256=candidate["owner_sha256"],
    )
    mass_values_path = resolved_record_path(mass_manifest, "values_file", "mass manifest")
    values_sha = normalized_sha256("navigation-mass-values-sha256", mass_manifest.get("values_sha256"))
    if sha256_path(mass_values_path) != values_sha or mass_values_path.stat().st_size != upper_count * 8:
        raise ValueError("navigation-mass values checksum/length mismatch")
    mass = np.memmap(mass_values_path, dtype="<u8", mode="r", shape=(upper_count,))
    if np.any(mass < 1):
        raise ValueError("regularized navigation mass must be at least one")
    mass_sha = semantic_mass_sha256(MASS_SOURCE, MASS_TRANSFORM, mass, ATTACHMENT_TOP_K)
    if mass_sha != mass_info.get("sha256"):
        raise ValueError("navigation-mass semantic checksum mismatch")
    total_mass = int(np.sum(mass, dtype=np.uint64))
    if total_mass != mass_info.get("total_mass") or total_mass != mass_manifest.get("total_mass"):
        raise ValueError("navigation-mass total differs across evidence")

    sizes = integer_list(
        candidate.get("partition_sizes"),
        length=partitions,
        label="candidate partition_sizes",
        minimum=1,
    )
    if sizes != [int(value) for value in owner_sizes]:
        raise ValueError("candidate partition sizes differ from owner bytes")
    loads = np.zeros(partitions, dtype=np.int64)
    np.add.at(loads, owner, mass.astype(np.int64, copy=False))
    declared_loads = integer_list(
        candidate.get("estimated_partition_masses"),
        length=partitions,
        label="candidate estimated_partition_masses",
        minimum=1,
    )
    if declared_loads != [int(value) for value in loads]:
        raise ValueError("candidate estimated masses differ from replay")
    target = total_mass / partitions
    limit = math.ceil(target) + int(np.max(mass)) - 1
    if not canonical_close(candidate.get("estimated_mass_target"), target):
        raise ValueError("candidate estimated mass target differs from replay")
    if candidate.get("estimated_mass_limit") != limit or int(np.max(loads)) > limit:
        raise ValueError("candidate violates the fixed navigation-mass bound")
    return {
        "method": candidate["method"],
        "owner_sha256": candidate["owner_sha256"],
        "partition_sizes": sizes,
        "estimated_partition_masses": declared_loads,
        "estimated_mass_target": target,
        "estimated_mass_limit": limit,
        "navigation_mass_source": MASS_SOURCE,
        "navigation_mass_transform": MASS_TRANSFORM,
        "navigation_mass_estimator_version": MASS_ESTIMATOR_VERSION,
        "mass_mode": MASS_MODE,
        "navigation_mass_sha256": mass_sha,
        "navigation_mass_manifest_sha256": mass_manifest_sha,
        "upper_navigation_hits_sha256": construction["upper_navigation_hits_sha256"],
        "upper_navigation_manifest_sha256": construction[
            "upper_navigation_manifest_sha256"
        ],
        "materialization_eligible": True,
        "navigation_mass_manifest": str(mass_manifest_path),
        "upper_navigation_hits": str(nav_hits),
        "upper_navigation_manifest": str(nav_manifest_path),
        "partitioner_source": str(partitioner_path),
        "partitioner_source_sha256": partitioner_sha,
        **v2_evidence,
    }


def validate_parity_evidence(parity: Any, *, rows: int, partitions: int) -> None:
    if not isinstance(parity, dict):
        raise ValueError("screen record lacks golden materialization parity")
    errors = mismatches(
        parity,
        {"canonical_format": ASSIGNMENT_FORMAT, "logical_point_count": rows},
    )
    if errors:
        raise ValueError(f"golden materialization parity mismatch: {errors}")
    normalized_sha256("golden assignment SHA", parity.get("assignment_bytes_sha256"))
    golden_loads = integer_list(
        parity.get("shard_loads"),
        length=partitions,
        label="golden shard_loads",
        minimum=1,
    )
    histogram = parity.get("copy_count_histogram")
    if not isinstance(histogram, dict) or not histogram:
        raise ValueError("golden copy_count_histogram is missing")
    evaluation_path = resolved_record_path(parity, "evaluation_npz_path", "parity")
    evaluation_sha = normalized_sha256(
        "evaluation NPZ SHA", parity.get("evaluation_npz_sha256")
    )
    if sha256_path(evaluation_path) != evaluation_sha:
        raise ValueError("golden evaluation NPZ checksum drifted")
    with np.load(evaluation_path, allow_pickle=False) as evaluation:
        if not np.array_equal(
            np.asarray(evaluation["loads"], dtype=np.int64),
            np.asarray(golden_loads, dtype=np.int64),
        ):
            raise ValueError("golden NPZ loads differ from screen")
        npz_histogram = {
            str(int(value)): int(count)
            for value, count in zip(
                np.asarray(evaluation["copy_count_hist_values"]),
                np.asarray(evaluation["copy_count_hist_counts"]),
            )
        }
        if npz_histogram != histogram:
            raise ValueError("golden NPZ copy histogram differs from screen")


def validate_materialization_inputs(
    *,
    source_artifact_path: Path,
    artifact: dict[str, Any],
    source_sha256: str,
    attachments_path: Path,
    attachments_manifest_path: Path,
    owner_path: Path,
    owner_sha256: str,
    owner: np.ndarray,
    screen_manifest_path: Path,
    cross_dataset_confirmation_path: Path | None,
    method: str,
    reference_owner: bool,
    vectors_path: Path,
    vectors_sha256: str,
) -> dict[str, Any]:
    if not method or Path(method).name != method:
        raise ValueError("method must be a filename-safe label")
    rows = int(artifact["logical_point_count"])
    upper_count = len(artifact["upper_nodes"])
    partitions = int(artifact["shard_count"])
    if len(owner) != upper_count or np.any(owner < 0) or np.any(owner >= partitions):
        raise ValueError("owner shape or partition IDs differ from the source artifact")
    owner_sizes = np.bincount(owner, minlength=partitions)
    if np.any(owner_sizes == 0):
        raise ValueError("L1 owner contains an empty partition")
    exact_quota = int(owner_sizes.max() - owner_sizes.min()) <= 1
    attachment_manifest, hits = validate_attachment_contract(
        artifact=artifact,
        source_artifact_path=source_artifact_path,
        source_sha256=source_sha256,
        attachments_path=attachments_path,
        attachments_manifest_path=attachments_manifest_path,
        vectors_path=vectors_path,
        vectors_sha256=vectors_sha256,
    )

    screen = load_json_object(screen_manifest_path, "final screen manifest")
    if screen.get("format_version") != 1 or screen.get("stage") != "post_freeze_evaluation":
        raise ValueError("materialization requires a final post-freeze screen manifest")
    contract = screen.get("contract")
    construction = screen.get("construction_inputs")
    post_freeze = screen.get("post_freeze_evaluation_inputs")
    if not all(isinstance(value, dict) for value in (contract, construction, post_freeze)):
        raise ValueError("screen lacks architecture/input bindings")
    errors = mismatches(
        contract,
        {
            "partitioner_reads_full_attachments": False,
            "partitioner_reads_l0_load": False,
            "partitioner_reads_multi_assignment_state": False,
            "l0_repair_or_flow": False,
            "evaluator_runs_in_separate_process": True,
            "owners_changed_after_freeze": False,
            "l0_assignment_after_owner_freeze": True,
            "multi_assignment": MULTI_ASSIGNMENT_CONTRACT,
        },
    )
    if errors:
        raise ValueError(f"screen architecture contract mismatch: {errors}")
    errors = mismatches(
        construction,
        {
            "artifact": str(source_artifact_path),
            "artifact_sha256": source_sha256,
            "logical_point_count": rows,
            "upper_node_count": upper_count,
            "dimension": int(artifact["vector_schema"]["dimension"]),
        },
    )
    if errors:
        raise ValueError(f"screen source-artifact binding mismatch: {errors}")
    errors = mismatches(
        post_freeze,
        {
            "attachments": str(attachments_path),
            "attachments_sha256": attachment_manifest["hits_sha256"],
            "attachments_manifest": str(attachments_manifest_path),
            "attachments_manifest_sha256": sha256_path(attachments_manifest_path),
        },
    )
    if errors:
        raise ValueError(f"screen attachment binding mismatch: {errors}")

    candidate = select_screen_record(screen, method, owner_path, owner_sha256)
    balance_contract = candidate.get("balance_contract")
    if balance_contract not in {
        UNIT_BALANCE_CONTRACT,
        MASS_BALANCE_CONTRACT,
        REFERENCE_BALANCE_CONTRACT,
    }:
        raise ValueError("screen record has an unknown balance_contract")
    is_reference = balance_contract == REFERENCE_BALANCE_CONTRACT
    if reference_owner is not is_reference:
        raise ValueError("--reference-owner does not match the manifest classification")
    errors = mismatches(
        candidate,
        {
            "source_artifact_sha256": source_sha256,
            "attachments_sha256": attachment_manifest["hits_sha256"],
            "owner_sha256": owner_sha256,
            "owner_path": str(owner_path),
        },
    )
    if errors:
        raise ValueError(f"screen record binding mismatch: {errors}")
    sizes = integer_list(
        candidate.get("partition_sizes"),
        length=partitions,
        label="screen partition_sizes",
        minimum=1,
    )
    if sizes != [int(value) for value in owner_sizes]:
        raise ValueError("screen partition sizes differ from owner bytes")

    mass_balance = None
    frozen_sha = None
    if is_reference:
        if cross_dataset_confirmation_path is not None:
            raise ValueError(
                "cross-dataset confirmation cannot authorize a matched reference"
            )
        if candidate.get("materialization_eligible") is not False:
            raise ValueError("matched reference must remain explicitly ineligible")
        if candidate.get("new_algorithm_eligible", False) is not False:
            raise ValueError("matched reference cannot masquerade as a candidate")
        input_scope = "legacy_l0_informed_reference"
        eligible = False
    else:
        if candidate.get("materialization_eligible") is not True:
            raise ValueError("candidate is not manifest-eligible for materialization")
        if candidate.get("new_algorithm_eligible", True) is not True:
            raise ValueError("candidate is explicitly ineligible")
        topology_contract = screen.get("topology_gate_contract")
        search_contract = screen.get("search_induced_gate_contract")
        if not isinstance(topology_contract, dict) or not isinstance(
            search_contract, dict
        ):
            raise ValueError("candidate screen lacks topology gate contracts")
        if screen.get("identity_all_pass") is not True:
            raise ValueError("screen identity gates did not all pass")
        validate_topology_record(candidate, topology_contract, search_contract)
        eligible = True
        if balance_contract == MASS_BALANCE_CONTRACT:
            input_scope = "production_upper_graph_and_upper_navigation_mass_only"
            if contract.get("partition_input_scope") != input_scope:
                raise ValueError("mass candidate input scope mismatch")
            if cross_dataset_confirmation_path is None:
                raise ValueError(
                    "mass candidate requires --cross-dataset-confirmation"
                )
            frozen_sha = validate_frozen_chain(screen, screen_manifest_path, candidate)
            mass_balance = validate_mass_candidate(
                screen=screen,
                candidate=candidate,
                owner=owner,
                owner_path=owner_path,
                owner_sizes=owner_sizes,
                artifact=artifact,
            )
            mass_balance.update(
                validate_cross_dataset_confirmation(
                    confirmation_path=cross_dataset_confirmation_path,
                    screen_path=screen_manifest_path,
                    screen=screen,
                    candidate=candidate,
                    source_artifact_path=source_artifact_path,
                    source_sha256=source_sha256,
                    owner_path=owner_path,
                    owner_sha256=owner_sha256,
                    frozen_candidate_manifest_sha256=frozen_sha,
                )
            )
        else:
            if cross_dataset_confirmation_path is not None:
                raise ValueError(
                    "cross-dataset confirmation is valid only for the mass finalist"
                )
            input_scope = "production_upper_graph_only"
            if contract.get("partition_input_scope") != input_scope:
                raise ValueError("unit candidate input scope mismatch")
            if not exact_quota:
                raise ValueError("unit-balance candidate does not satisfy exact quota")
            partitioner_path = resolved_record_path(
                construction, "partitioner_source", "construction"
            )
            partitioner_sha = normalized_sha256(
                "partitioner source SHA", construction.get("partitioner_source_sha256")
            )
            if sha256_path(partitioner_path) != partitioner_sha:
                raise ValueError("unit partitioner source checksum drifted")
            if candidate.get("partitioner_source_sha256") != partitioner_sha:
                raise ValueError("unit candidate partitioner checksum mismatch")
    validate_parity_evidence(candidate.get("materialization_parity"), rows=rows, partitions=partitions)
    return {
        "candidate": candidate,
        "candidate_record_sha256": canonical_json_sha256(candidate),
        "screen_manifest_sha256": sha256_path(screen_manifest_path),
        "balance_contract": balance_contract,
        "input_scope": input_scope,
        "new_algorithm_eligible": eligible,
        "exact_quota": exact_quota,
        "owner_sizes": [int(value) for value in owner_sizes],
        "mass_balance": mass_balance,
        "frozen_candidate_manifest_sha256": frozen_sha,
        "partitioner_source": (
            mass_balance["partitioner_source"]
            if mass_balance
            else construction.get("partitioner_source")
        ),
        "partitioner_source_sha256": (
            mass_balance["partitioner_source_sha256"]
            if mass_balance
            else construction.get("partitioner_source_sha256")
        ),
        "materialization_parity": candidate["materialization_parity"],
        "hits": hits,
    }


def validate_materialization_parity(
    binding: dict[str, Any],
    *,
    layout_sha256: str,
    rows: int,
    physical_count: int,
    shard_counts: np.ndarray,
    copy_histogram: Counter[int],
) -> None:
    parity = binding["materialization_parity"]
    expected = {
        "canonical_format": ASSIGNMENT_FORMAT,
        "assignment_bytes_sha256": layout_sha256,
        "logical_point_count": rows,
        "physical_point_count": physical_count,
        "shard_loads": [int(value) for value in shard_counts],
        "copy_count_histogram": {
            str(key): int(value) for key, value in sorted(copy_histogram.items())
        },
    }
    errors = mismatches(parity, expected)
    if errors:
        raise ValueError(f"golden multi-assignment parity mismatch: {errors}")
    metrics = binding["candidate"].get("actual_metrics")
    if isinstance(metrics, dict):
        mean = physical_count / len(shard_counts)
        errors = mismatches(
            metrics,
            {
                "physical_point_count": physical_count,
                "expansion_ratio": physical_count / rows,
                "load_min": int(shard_counts.min()),
                "load_max": int(shard_counts.max()),
                "load_mean": mean,
                "load_cv": float(np.std(shard_counts) / mean),
                "load_max_over_mean": float(shard_counts.max() / mean),
                "load_min_over_mean": float(shard_counts.min() / mean),
                "empty_shards": int(np.count_nonzero(shard_counts == 0)),
            },
        )
        if errors:
            raise ValueError(f"screen actual metrics differ from replay: {errors}")


def file_record(path: Path) -> dict[str, Any]:
    return {"sha256": sha256_path(path), "size_bytes": path.stat().st_size}


def run_upper_replay_gate(
    *,
    verifier: Path,
    source_artifact: Path,
    rebound_artifact: Path,
    queries: Path,
    query_rows: int,
    dimension: int,
    replay_path: Path,
    manifest_path: Path,
    source_sha256: str,
    rebound_sha256: str,
    source_generation: int,
    rebound_generation: int,
    source_layout_sha256: str,
    rebound_layout_sha256: str,
    shard_count: int,
    rebound_physical_count: int,
) -> dict[str, Any]:
    completed = subprocess.run(
        [
            str(verifier),
            str(source_artifact),
            str(rebound_artifact),
            str(queries),
            str(query_rows),
            str(dimension),
            str(replay_path),
            str(manifest_path),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    (manifest_path.parent / "upper-replay.stdout.log").write_text(
        completed.stdout, encoding="utf-8"
    )
    if completed.stderr:
        (manifest_path.parent / "upper-replay.stderr.log").write_text(
            completed.stderr, encoding="utf-8"
        )
    manifest = load_json_object(manifest_path, "upper replay PASS manifest")
    source = manifest.get("source")
    rebound = manifest.get("rebound")
    query = manifest.get("query_corpus")
    replay = manifest.get("replay")
    gates = manifest.get("gates")
    if not all(isinstance(value, dict) for value in (source, rebound, query, replay, gates)):
        raise ValueError("upper replay manifest is incomplete")
    errors = mismatches(
        manifest,
        {"format_version": 1, "verdict": "PASS"},
    )
    errors.update(
        {
            f"source.{key}": value
            for key, value in mismatches(
                source,
                {
                    "path": str(source_artifact),
                    "file_sha256": source_sha256,
                    "generation": source_generation,
                    "layout_sha256": source_layout_sha256,
                    "shard_count": shard_count,
                },
            ).items()
        }
    )
    errors.update(
        {
            f"rebound.{key}": value
            for key, value in mismatches(
                rebound,
                {
                    "path": str(rebound_artifact),
                    "file_sha256": rebound_sha256,
                    "generation": rebound_generation,
                    "layout_sha256": rebound_layout_sha256,
                    "shard_count": shard_count,
                    "physical_point_count": rebound_physical_count,
                },
            ).items()
        }
    )
    query_sha256 = sha256_path(queries)
    errors.update(
        {
            f"query_corpus.{key}": value
            for key, value in mismatches(
                query,
                {
                    "path": str(queries),
                    "sha256": query_sha256,
                    "row_count": query_rows,
                    "dimension": dimension,
                    "size_bytes": query_rows * dimension * 4,
                },
            ).items()
        }
    )
    replay_sha256 = sha256_path(replay_path)
    errors.update(
        {
            f"replay.{key}": value
            for key, value in mismatches(
                replay,
                {
                    "path": str(replay_path),
                    "sha256": replay_sha256,
                    "ordered_label_and_distance_bits_sha256": replay_sha256,
                    "size_bytes": replay_path.stat().st_size,
                },
            ).items()
        }
    )
    if errors:
        raise ValueError(f"upper replay manifest binding mismatch: {errors}")
    if source.get("canonical_upper_graph_sha256") != rebound.get(
        "canonical_upper_graph_sha256"
    ):
        raise ValueError("upper replay canonical graph checksums differ")
    if source.get("canonical_upper_graph_size_bytes") != rebound.get(
        "canonical_upper_graph_size_bytes"
    ):
        raise ValueError("upper replay canonical graph sizes differ")
    if source.get("ordered_upper_nodes_identity_sha256") != rebound.get(
        "ordered_upper_nodes_identity_sha256"
    ):
        raise ValueError("upper replay ordered upper-node identities differ")
    for field in REPLAY_GATES:
        if gates.get(field) is not True:
            raise ValueError(f"upper replay gate {field} did not pass")
    return {
        "manifest_sha256": sha256_path(manifest_path),
        "replay_sha256": replay_sha256,
        "query_sha256": query_sha256,
        "canonical_upper_graph_sha256": source["canonical_upper_graph_sha256"],
        "ordered_upper_nodes_identity_sha256": source[
            "ordered_upper_nodes_identity_sha256"
        ],
    }


def main() -> None:
    args = parse_args()
    if (
        args.generation <= 0
        or args.chunk_size <= 0
        or args.replay_query_row_count <= 0
    ):
        raise ValueError("generation, chunk-size, and replay-query-row-count must be positive")
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output_dir}")

    source_artifact_path = Path(args.source_artifact).expanduser().resolve()
    attachments_path = Path(args.attachments).expanduser().resolve()
    attachments_manifest_path = Path(args.attachments_manifest).expanduser().resolve()
    owner_path = Path(args.owner).expanduser().resolve()
    screen_manifest_path = Path(args.screen_manifest).expanduser().resolve()
    cross_dataset_confirmation_path = (
        Path(args.cross_dataset_confirmation).expanduser().resolve()
        if args.cross_dataset_confirmation
        else None
    )
    vectors_path = Path(args.vectors).expanduser().resolve()
    vectors_build_manifest_path = Path(args.vectors_build_manifest).expanduser().resolve()
    dataset_path = Path(args.dataset).expanduser().resolve()
    rebind_binary = Path(args.rebind_binary).expanduser().resolve()
    replay_verifier = Path(args.replay_verifier).expanduser().resolve()
    replay_queries = Path(args.replay_queries).expanduser().resolve()
    for path in (
        source_artifact_path,
        attachments_path,
        attachments_manifest_path,
        owner_path,
        screen_manifest_path,
        vectors_path,
        vectors_build_manifest_path,
        dataset_path,
        rebind_binary,
        replay_verifier,
        replay_queries,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if (
        cross_dataset_confirmation_path is not None
        and not cross_dataset_confirmation_path.is_file()
    ):
        raise FileNotFoundError(cross_dataset_confirmation_path)

    source_sha256 = sha256_path(source_artifact_path)
    artifact = load_json_object(source_artifact_path, "source artifact")
    upper_nodes = artifact["upper_nodes"]
    upper_labels = np.asarray([int(node["label"]) for node in upper_nodes], dtype=np.int64)
    upper_count = len(upper_labels)
    row_count = int(artifact["logical_point_count"])
    partitions = int(artifact["shard_count"])
    dimension = int(artifact["vector_schema"]["dimension"])
    source_generation = int(artifact["generation"])
    if args.generation <= source_generation:
        raise ValueError("generation must be greater than source generation")
    if (
        np.any(upper_labels < 0)
        or np.any(upper_labels >= row_count)
        or len(np.unique(upper_labels)) != upper_count
    ):
        raise ValueError("upper labels must be unique numeric logical point IDs")
    if replay_queries.stat().st_size != args.replay_query_row_count * dimension * 4:
        raise ValueError("replay query corpus length differs from row-count/dimension")

    expected_owner_sha256 = normalized_sha256("owner-sha256", args.owner_sha256)
    if sha256_path(owner_path) != expected_owner_sha256:
        raise ValueError("owner checksum mismatch")
    if owner_path.stat().st_size != upper_count * 4:
        raise ValueError("owner length differs from upper-node count")
    owner = np.memmap(owner_path, dtype="<i4", mode="r", shape=(upper_count,))
    dataset_sha256 = normalized_sha256("dataset-sha256", args.dataset_sha256)
    if sha256_path(dataset_path) != dataset_sha256:
        raise ValueError("dataset checksum mismatch")
    vectors_sha256 = sha256_path(vectors_path)
    vector_source = validate_vector_source_contract(
        vectors_path=vectors_path,
        vectors_sha256=vectors_sha256,
        source_build_manifest_path=vectors_build_manifest_path,
        dataset_path=dataset_path,
        dataset_sha256=dataset_sha256,
        artifact=artifact,
        chunk_rows=args.chunk_size,
    )
    binding = validate_materialization_inputs(
        source_artifact_path=source_artifact_path,
        artifact=artifact,
        source_sha256=source_sha256,
        attachments_path=attachments_path,
        attachments_manifest_path=attachments_manifest_path,
        owner_path=owner_path,
        owner_sha256=expected_owner_sha256,
        owner=owner,
        screen_manifest_path=screen_manifest_path,
        cross_dataset_confirmation_path=cross_dataset_confirmation_path,
        method=args.method,
        reference_owner=args.reference_owner,
        vectors_path=vectors_path,
        vectors_sha256=vectors_sha256,
    )
    hits = binding.pop("hits")
    maximum_label = max(int(upper_labels.max()), int(np.max(hits)))
    if maximum_label >= row_count:
        raise ValueError("attachment contains a label outside logical point range")
    label_to_local = np.full(maximum_label + 1, -1, dtype=np.int32)
    label_to_local[upper_labels] = np.arange(upper_count, dtype=np.int32)

    output_dir.mkdir(parents=True, exist_ok=False)
    assignments_path = output_dir / "orion_numeric_import.assignments.jsonl"
    assignments_tmp = output_dir / "orion_numeric_import.assignments.jsonl.tmp"
    upper_memberships: list[list[int] | None] = [None] * upper_count
    upper_position = np.full(row_count, -1, dtype=np.int32)
    upper_position[upper_labels] = np.arange(upper_count, dtype=np.int32)
    assignment_digest = hashlib.sha256()
    shard_counts = np.zeros(partitions, dtype=np.int64)
    copy_histogram: Counter[int] = Counter()
    physical_count = 0
    try:
        with assignments_tmp.open("xb") as handle:
            for start in range(0, row_count, args.chunk_size):
                stop = min(row_count, start + args.chunk_size)
                local = label_to_local[np.asarray(hits[start:stop])]
                if np.any(local < 0):
                    raise ValueError(f"attachment chunk {start}:{stop} has unknown upper label")
                membership, copy_counts = compact_membership(owner, local, partitions)
                shard_counts += membership.sum(axis=0, dtype=np.int64)
                values, counts = np.unique(copy_counts, return_counts=True)
                copy_histogram.update(
                    {int(value): int(count) for value, count in zip(values, counts)}
                )
                physical_count += int(copy_counts.sum())
                for offset, membership_row in enumerate(membership):
                    point_id = start + offset
                    shards = np.flatnonzero(membership_row).astype(int).tolist()
                    encoded = assignment_bytes(point_id, shards)
                    handle.write(encoded)
                    assignment_digest.update(encoded)
                    upper_index = int(upper_position[point_id])
                    if upper_index >= 0:
                        upper_memberships[upper_index] = shards
                print(f"materialized={stop}/{row_count}", flush=True)
        layout_sha256 = assignment_digest.hexdigest()
        validate_materialization_parity(
            binding,
            layout_sha256=layout_sha256,
            rows=row_count,
            physical_count=physical_count,
            shard_counts=shard_counts,
            copy_histogram=copy_histogram,
        )
        os.replace(assignments_tmp, assignments_path)
    except BaseException:
        assignments_tmp.unlink(missing_ok=True)
        raise
    if any(row is None for row in upper_memberships):
        raise RuntimeError("not every upper point received final L0 membership")
    if np.any(shard_counts <= 0) or sha256_path(assignments_path) != layout_sha256:
        raise RuntimeError("published assignments are empty or checksum-inconsistent")

    sidecar_path = output_dir / "rebind-memberships.json"
    sidecar_path.write_text(
        json.dumps(
            {
                "format_version": 1,
                "source_artifact_sha256": source_sha256,
                "source_generation": source_generation,
                "generation": args.generation,
                "layout_sha256": layout_sha256,
                "shard_count": partitions,
                "physical_point_count": physical_count,
                "shard_memberships": upper_memberships,
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    rebound_path = output_dir / f"generation-{args.generation}.json"
    completed = subprocess.run(
        [str(rebind_binary), str(source_artifact_path), str(sidecar_path), str(rebound_path)],
        check=True,
        text=True,
        capture_output=True,
    )
    (output_dir / "rebind.stdout.log").write_text(completed.stdout, encoding="utf-8")
    if completed.stderr:
        (output_dir / "rebind.stderr.log").write_text(completed.stderr, encoding="utf-8")
    rebound_sha256 = sha256_path(rebound_path)
    checksum_sidecar = rebound_path.with_name(rebound_path.name + ".sha256")
    if checksum_sidecar.read_text(encoding="utf-8").strip() != rebound_sha256:
        raise RuntimeError("rebind checksum sidecar differs from rebound artifact")

    replay_path = output_dir / "upper-replay.bin"
    replay_manifest_path = output_dir / "upper-replay-manifest.json"
    replay_gate = run_upper_replay_gate(
        verifier=replay_verifier,
        source_artifact=source_artifact_path,
        rebound_artifact=rebound_path,
        queries=replay_queries,
        query_rows=args.replay_query_row_count,
        dimension=dimension,
        replay_path=replay_path,
        manifest_path=replay_manifest_path,
        source_sha256=source_sha256,
        rebound_sha256=rebound_sha256,
        source_generation=source_generation,
        rebound_generation=args.generation,
        source_layout_sha256=str(artifact["layout_sha256"]),
        rebound_layout_sha256=layout_sha256,
        shard_count=partitions,
        rebound_physical_count=physical_count,
    )

    hardlink_path = output_dir / "orion_numeric_import.f32le"
    os.link(vectors_path, hardlink_path)
    if not os.path.samefile(vectors_path, hardlink_path):
        raise RuntimeError("materialized vectors are not a hardlink to exporter input")
    if sha256_path(hardlink_path) != vectors_sha256:
        raise RuntimeError("hardlinked vector checksum changed")
    import_manifest_path = output_dir / "orion_numeric_import.manifest.json"
    import_manifest_path.write_text(
        json.dumps(
            {
                "format_version": 1,
                "orion_generation": args.generation,
                "orion_artifact_file": rebound_path.name,
                "orion_artifact_sha256": rebound_sha256,
                "dimension": dimension,
                "point_count": row_count,
                "shard_count": partitions,
                "vector_name": str(artifact["vector_schema"]["vector_name"]),
                "vectors_file": hardlink_path.name,
                "vectors_sha256": vectors_sha256,
                "assignments_file": assignments_path.name,
                "assignments_sha256": layout_sha256,
                "total_point_copies": physical_count,
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    owner_copy_path = output_dir / f"l1-owner-{args.method}.i32le"
    shutil.copy2(owner_path, owner_copy_path)

    candidate = binding["candidate"]
    diagnostics: dict[str, Any] = {
        "input_scope": binding["input_scope"],
        "method": args.method,
        "balance_contract": binding["balance_contract"],
        "new_algorithm_eligible": binding["new_algorithm_eligible"],
        "exact_quota": binding["exact_quota"],
        "l1_sizes": binding["owner_sizes"],
        "owner_sha256": expected_owner_sha256,
        "source_artifact_sha256": source_sha256,
        "attachments_sha256": sha256_path(attachments_path),
        "screen_manifest_sha256": binding["screen_manifest_sha256"],
        "candidate_record_sha256": binding["candidate_record_sha256"],
        "partitioner_source_sha256": binding["partitioner_source_sha256"],
        "topology_all_pass": candidate.get("topology_all_pass"),
        "topology_metrics": candidate.get("topology_metrics"),
        "topology_gates": candidate.get("topology_gates"),
        "frozen_candidate_manifest_sha256": binding[
            "frozen_candidate_manifest_sha256"
        ],
        "golden_assignment_bytes_sha256": candidate["materialization_parity"][
            "assignment_bytes_sha256"
        ],
        "upper_replay_manifest_sha256": replay_gate["manifest_sha256"],
        "l0_repair": False,
        "multi_assignment_after_owner_freeze": True,
    }
    if binding["mass_balance"] is not None:
        diagnostics["protocol_amendment_sha256"] = binding["mass_balance"][
            "protocol_amendment_sha256"
        ]
        diagnostics["cross_dataset_confirmation_sha256"] = binding["mass_balance"][
            "cross_dataset_confirmation_sha256"
        ]
        diagnostics["owner_parity"] = binding["mass_balance"]["owner_parity"]
        diagnostics["mass_balance"] = {
            key: value
            for key, value in binding["mass_balance"].items()
            if key
            not in {
                "navigation_mass_manifest",
                "upper_navigation_hits",
                "upper_navigation_manifest",
                "partitioner_source",
                "partitioner_source_sha256",
                "protocol_amendment",
                "protocol_amendment_sha256",
                "protocol",
                "protocol_sha256",
                "owner_parity",
                "cross_dataset_confirmation",
                "cross_dataset_confirmation_sha256",
            }
        }

    created_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    load_mean = physical_count / partitions
    build_manifest_path = output_dir / "build-manifest.json"
    build_manifest: dict[str, Any] = {
        "format_version": 1,
        "tool": "experiments/l1_balance/materialize_candidate.py",
        "mode": "production_bundle",
        "created_at": created_at,
        "dataset": {
            "path": str(dataset_path),
            "sha256": dataset_sha256,
            "size_bytes": dataset_path.stat().st_size,
            "dimension": dimension,
            "train_rows_total": row_count,
            "train_rows_used": row_count,
        },
        "parameters": {
            "balance_mode": (
                "matched_l0_informed_reference"
                if args.reference_owner
                else "l1_graph_prepartition"
            ),
            "l1_partitioner": args.method,
            "initial_num_shards": partitions,
            "enable_fission": False,
            "enable_topology_refinement": False,
            "use_multi_assign": True,
            "multi_assign_min_max_vote": 2,
            "multi_assign_vote_delta": 0,
            "multi_assign_max_shards": 0,
            "sample_denominator": 32,
            "upper_sample_seed": 100,
            "upper_m": 32,
            "upper_ef_construction": 100,
            "upper_graph_seed": 100,
            "attachment_search_ef": ATTACHMENT_SEARCH_EF,
            "k_overlap": ATTACHMENT_TOP_K,
            "upper_k": int(artifact["upper_k"]),
            "upper_search_ef": int(artifact["upper_ef_search"]),
            "dynamic_ef_base": int(artifact["dynamic_ef_base"]),
            "dynamic_ef_factor": int(artifact["dynamic_ef_factor"]),
            "allow_decoupled_runtime_upper_search": False,
            "generation": args.generation,
            "l0_repair": False,
        },
        "artifact_binding": {
            "generation": args.generation,
            "layout_sha256": layout_sha256,
            "logical_point_count": row_count,
            "physical_point_count": physical_count,
            "shard_count": partitions,
        },
        "routing": {
            "initial_num_shards": partitions,
            "effective_num_shards": partitions,
            "logical_point_count": row_count,
            "physical_point_count": physical_count,
            "expansion_ratio": physical_count / row_count,
            "shard_counts": [int(value) for value in shard_counts],
            "upper_point_count": upper_count,
            "fission_events": [],
            "l1_partition_diagnostics": diagnostics,
            "load_summary": {
                "min": int(shard_counts.min()),
                "max": int(shard_counts.max()),
                "mean": load_mean,
                "cv": float(np.std(shard_counts) / load_mean),
                "max_over_mean": float(shard_counts.max() / load_mean),
                "min_over_mean": float(shard_counts.min() / load_mean),
            },
            "copy_count_histogram": {
                str(key): int(value) for key, value in sorted(copy_histogram.items())
            },
        },
        "provenance": {
            "source_artifact": str(source_artifact_path),
            "source_artifact_sha256": source_sha256,
            "attachments": str(attachments_path),
            "attachments_manifest": str(attachments_manifest_path),
            "owner_source": str(owner_path),
            "screen_manifest": str(screen_manifest_path),
            "rebind_binary": str(rebind_binary),
            "rebind_binary_sha256": sha256_path(rebind_binary),
            "vectors_source": str(vectors_path),
            "vectors_source_build_manifest": str(vectors_build_manifest_path),
            "vectors_source_contract": vector_source,
            "attachments_exporter_vectors_sha256": vectors_sha256,
            "point_id_order_contract": (
                "attachment row i = hardlinked vector row i = logical point id i"
            ),
            "vectors_materialization": "hardlink",
            "vectors_same_file": os.path.samefile(vectors_path, hardlink_path),
            "partitioner_source": binding["partitioner_source"],
            "upper_replay_verifier": str(replay_verifier),
            "upper_replay_verifier_sha256": sha256_path(replay_verifier),
            "upper_replay_queries": str(replay_queries),
            "upper_replay_queries_sha256": replay_gate["query_sha256"],
            "upper_replay_manifest": str(replay_manifest_path),
            "upper_replay": str(replay_path),
        },
        "outputs": {
            "production_artifact": rebound_path.name,
            "import_manifest": import_manifest_path.name,
            "owner": owner_copy_path.name,
            "upper_replay_manifest": replay_manifest_path.name,
            "upper_replay": replay_path.name,
        },
    }
    if binding["mass_balance"] is not None:
        build_manifest["provenance"].update(
            {
                "upper_navigation_hits": binding["mass_balance"][
                    "upper_navigation_hits"
                ],
                "upper_navigation_manifest": binding["mass_balance"][
                    "upper_navigation_manifest"
                ],
                "navigation_mass_manifest": binding["mass_balance"][
                    "navigation_mass_manifest"
                ],
                "protocol_amendment": binding["mass_balance"][
                    "protocol_amendment"
                ],
                "protocol": binding["mass_balance"]["protocol"],
                "cross_dataset_confirmation": binding["mass_balance"][
                    "cross_dataset_confirmation"
                ],
            }
        )
    build_manifest_path.write_text(
        json.dumps(build_manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    payload_files = [
        rebound_path,
        checksum_sidecar,
        assignments_path,
        hardlink_path,
        import_manifest_path,
        owner_copy_path,
        sidecar_path,
        replay_path,
        replay_manifest_path,
        output_dir / "rebind.stdout.log",
        output_dir / "upper-replay.stdout.log",
    ]
    for optional in (
        output_dir / "rebind.stderr.log",
        output_dir / "upper-replay.stderr.log",
    ):
        if optional.exists():
            payload_files.append(optional)
    build_manifest["outputs"]["files"] = {
        path.name: file_record(path) for path in payload_files
    }
    build_manifest_path.write_text(
        json.dumps(build_manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    checksums_path = output_dir / "checksums.sha256"
    checksum_files = sorted(payload_files + [build_manifest_path], key=lambda path: path.name)
    with checksums_path.open("x", encoding="ascii", newline="\n") as handle:
        for path in checksum_files:
            handle.write(f"{sha256_path(path)}  {path.name}\n")

    print(f"bundle={output_dir}")
    print(f"artifact={rebound_path}")
    print(f"artifact_sha256={rebound_sha256}")
    print(f"layout_sha256={layout_sha256}")
    print(f"upper_replay_manifest_sha256={replay_gate['manifest_sha256']}")
    print(f"physical_point_count={physical_count}")
    print(f"load_max_over_mean={shard_counts.max() / load_mean:.9f}")


if __name__ == "__main__":
    main()
